# `lbm/lattice.py`, `lbm/d2q9_solver.py` — GPU D2Q9 forward solver

**Work package:** WP 4. **Depends on:** WP 0 (WP 3 for bodies). **Math:** derivations §2. **Decisions:** G5, G6, G7, G18.

## Purpose

Fast, vectorized, differentiable D2Q9 solver with BGK and TRT collision, Bouzidi interpolated bounce-back on a NURBS body, and the report's inlet/outlet/periodic boundaries. Runs on CPU, MPS and CUDA from the same code.

## Lattice parameters (computed by `scripts/lattice_table.py`)

`U_lb = 0.1/√3 = 0.05774` (Mach 0.1), `nu_lb = U_lb N / Re`, `tau = 3 nu_lb + 0.5`, one convective time `= N / U_lb` steps. Domain `20 × 10` reference lengths, body 5 lengths from the inlet, centred vertically (report Fig 4.7).

| Body | Re | Tier | N | Grid `Nx × Ny` | Nodes | `nu_lb` | `tau` | Steps per conv. time |
|---|---|---|---|---|---|---|---|---|
| Cylinder | 20 | smoke | 10 | 200 × 100 | 20k | 0.02887 | 0.5866 | 173 |
| Cylinder | 20 | dev | 20 | 400 × 200 | 80k | 0.05774 | 0.6732 | 346 |
| Cylinder | 20 | full | 40 | 800 × 400 | 320k | 0.11547 | 0.8464 | 693 |
| Cylinder | 40 | smoke | 10 | 200 × 100 | 20k | 0.01443 | 0.5433 | 173 |
| Cylinder | 40 | dev | 20 | 400 × 200 | 80k | 0.02887 | 0.5866 | 346 |
| Cylinder | 40 | full | 40 | 800 × 400 | 320k | 0.05774 | 0.6732 | 693 |
| Cylinder | 100 | smoke | 10 | 200 × 100 | 20k | 0.00577 | 0.5173 | 173 |
| Cylinder | 100 | dev | 20 | 400 × 200 | 80k | 0.01155 | 0.5346 | 346 |
| Cylinder | 100 | full | 40 | 800 × 400 | 320k | 0.02309 | 0.5693 | 693 |
| Airfoil | 5 | dev | 20 | 400 × 200 | 80k | 0.23094 | 1.1928 | 346 |
| Airfoil | 20 | smoke | 10 | 200 × 100 | 20k | 0.02887 | 0.5866 | 173 |
| Airfoil | 20 | dev | 20 | 400 × 200 | 80k | 0.05774 | 0.6732 | 346 |
| Airfoil | 20 | full | 40 | 800 × 400 | 320k | 0.11547 | 0.8464 | 693 |
| Airfoil | 1000 | smoke | 24 | 480 × 240 | 115k | 0.00139 | 0.5042 | 416 |
| Airfoil | 1000 | dev | 48 | 960 × 480 | 461k | 0.00277 | 0.5083 | 831 |
| Airfoil | 1000 | full | 100 | 2000 × 1000 | 2000k | 0.00577 | 0.5173 | 1732 |

`tau < 0.52` is close to the BGK stability limit → TRT is the default for Re ≥ 100 (G7). Re 1000 smoke (`tau = 0.504`) is expected to be marginal; it only has to run without NaN for a few convective times.

## `lbm/lattice.py`

```python
C: Tensor        # [9, 2] long, order per derivations (2.1)
W: Tensor        # [9]
OPP: Tensor      # [9] long
CS2 = 1/3
def lattice_constants(device, dtype) -> tuple[Tensor, Tensor, Tensor]   # cached per (device, dtype)
def make_params(cfg, cond: Condition, N: int | None = None) -> LBMParams
def lbm_to_nondim(rho, ux, uy, u_lb) -> tuple[Tensor, Tensor, Tensor]   # derivations (2.9)
```

## `lbm/d2q9_solver.py`

```python
def moments(f) -> tuple[Tensor, Tensor, Tensor]           # rho, ux, uy  [Ny, Nx]
def equilibrium(rho, ux, uy) -> Tensor                    # [9, Ny, Nx]
def collide(f, params, solid) -> Tensor                   # BGK or TRT; solid nodes untouched
def stream(f_col) -> Tensor                               # torch.roll per direction
def apply_ibb(f_streamed, f_col, links, params) -> Tensor # Bouzidi (2.6), index_put via torch.where-free scatter
def apply_inlet_outlet(f, params) -> Tensor
def step(f, geom: GridGeometry, params) -> tuple[Tensor, Tensor]     # returns (f_next, f_col) — f_col needed for forces
def init_state(params, geom) -> Tensor                    # equilibrium at (rho=1, U_lb, 0) in fluid, rest in solid
def run(f0, geom, params, n_steps=None, mode="steady"|"unsteady", record_every=50,
        callback=None) -> SolveResult
```

## Step algorithm

1. `rho, ux, uy = moments(f)`; in solid nodes force `ux = uy = 0`, `rho = 1` via `torch.where` (keeps solid populations harmless).
2. `feq = equilibrium(rho, ux, uy)`.
3. Collision:
   - BGK: `f_col = f − omega (f − feq)`.
   - TRT: `f_col = f − omega (f⁺ − feq⁺) − omega_minus (f⁻ − feq⁻)` with `f⁺ = (f + f[OPP]) / 2` etc. (derivations 2.5).
4. `f_s = stream(f_col)`: `f_s[i] = torch.roll(f_col[i], shifts=(C[i,1], C[i,0]), dims=(0, 1))`. Implemented as one `torch.stack` of 9 rolls. Top/bottom periodicity comes for free; the left/right wrap is overwritten in step 6.
5. **IBB** at boundary links (derivations 2.6). With flat views `fc = f_col.view(9, −1)`, `fs = f_s.view(9, −1)`:
   - `a = fc[i, node]`, `b = fc[i, node − off_i]` (next fluid node back along `−c_i`), `c = fc[opp_i, node]`.
   - `new = where(q < 0.5, 2q a + (1 − 2q) b, a/(2q) + (2q − 1)/(2q) c)`; `new = where(far_ok, new, a)`.
   - Write: `fs = fs.index_put((opp_i, node), new)` (out-of-place `index_put`, differentiable in `new` → in `q`).
   - Links are grouped by direction once at setup so gathers are contiguous.
6. **Inlet** (column 0): `f_s[:, :, 0] = feq(rho=1, U_lb, 0)`. **Outlet** (column Nx−1), mode from config:
   - `fixed` (report §5.2.5; used whenever the adjoint is run): `feq(rho=1, U_lb, 0)`.
   - `extrapolate` (Re 100 validation, unsteady runs): `feq(rho=1, u from column Nx−2)`.
7. Solid nodes: reset to rest equilibrium every step (prevents drift; they carry no physics).
8. Optional sponge: `omega` field lowered toward the outlet (`nu × nu_factor` ramp over `width_d`). Only for unsteady runs; not used when the hand adjoint is validated.

## `run`

- **Steady**: step until `|CD_n − CD_{n−K}| / |CD_n| < steady_tol` with `K` = one convective time, or `max_convective_times`. Forces evaluated every `record_every` steps (and at every step for the last convective time).
- **Unsteady**: run `transient` convective times (default 30, report: shedding by t ≈ 35 at Re 100), detect the shedding period from upward zero crossings of `CL` (needs at least 3), then average `f` (and forces) over `average_periods` periods → `f_mean`. Return `period_steps`.
- History buffers pre-allocated on device; one `.cpu()` per log interval.
- `callback(step, f)` hook used by visualization to grab snapshots.

## Performance targets (WP 4 measures and records in `runs/perf/lbm_mlups.json`)

| Device | Target throughput (million lattice updates / s) at 400 × 200 |
|---|---|
| M4 Pro MPS | ≥ 50 |
| M4 Pro CPU (12 threads) | ≥ 10 |
| CUDA (T4 / L4 / A100) | ≥ 150 / 300 / 1000 |

If MPS is below target: fuse the 9 rolls into one `torch.cat` of slices (pre-allocated output), and wrap `step` in `torch.no_grad()` when no adjoint/VJP is needed. Kernel-launch overhead dominates small grids; that is expected for smoke.

## Tests (`tests/lbm/test_d2q9_solver.py`)

| Test | Pass |
|---|---|
| Equilibrium moments | `Σ feq = rho`, `Σ c feq = rho u` to 1e-12 (float64) |
| Collision conserves mass and momentum | BGK and TRT, random `f`, to 1e-12 |
| Periodic box mass | 1,000 steps, no body, all periodic → relative mass change < 1e-6 |
| Uniform flow | no body, inlet/outlet on → stays uniform to 1e-6 |
| Poiseuille | channel with straight walls via IBB at `q = 0.5` and `q = 0.3` → profile error < 1% at N = 20 |
| Couette / q-accuracy | wall at `q = 0.25` and `0.75` → second-order convergence slope ≥ 1.8 over N ∈ {10, 20, 40} |
| TRT ≡ BGK when `omega⁻ = omega` | identical outputs |
| Differentiability | `torch.autograd.grad(CD after 20 steps, q)` finite and non-zero |
| Device parity | one step on CPU vs MPS, max abs diff < 1e-6 |

## Pitfalls

- `torch.roll` sign: `f_s[i](x + c_i) = f_col[i](x)` means `shifts = +c_i`. Test with a single non-zero population.
- Indexing `[9, Ny, Nx]`: `c_i = (cx, cy)` maps to shifts `(cy, cx)` on dims `(0, 1)`.
- Never apply IBB to links whose `node − off_i` lies outside the grid (assert body clearance).
- Do the IBB overwrite **after** streaming but read from the **pre-streaming** `f_col`.
