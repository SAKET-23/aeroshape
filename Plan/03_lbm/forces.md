# `lbm/aero_forces.py` — drag and lift by momentum exchange

**Work package:** WP 5. **Depends on:** WP 3, WP 4. **Math:** derivations §3.

## Interface

```python
def momentum_exchange(f_col, f_next, links: LinkSet, Nx) -> Tensor         # [2] force (lattice units)
def coefficients(force, params: LBMParams) -> tuple[Tensor, Tensor]        # CD, CL (0-dim)
def period_average(history: ForceHistory, period_steps) -> tuple[float, float]
def surface_pressure_coefficient(state, geom, params, boundary) -> Tensor   # Cp along the surface, for plots
```

## Algorithm

1. For each link `(x_A, i)`: `ΔP = c_i (f_col[i, x_A] + f_next[opp_i, x_A])` (derivations 3.1). `f_next[opp_i, x_A]` is the IBB value written in this step.
2. `F = Σ ΔP` via `(c[dir] * (a + b)[:, None]).sum(0)`.
3. `CD = 2 F_x / (U_lb² N)`, `CL = 2 F_y / (U_lb² N)` (`rho_inf = 1`).
4. AoA: the body is rotated, flow stays `+x`, so drag is `F_x` and lift `F_y` without rotation.
5. `Cp` for plots: `(rho − 1)/3 / (½ U_lb²)` bilinearly interpolated at surface points (no gradient needed).

Fully differentiable in `f_col`, `f_next` and, through the IBB values, in `q`.

## Tests

| Test | Pass |
|---|---|
| No body | zero force |
| Uniform flow, symmetric body, 1 step | `|F_y| < 1e-10` |
| Steady symmetric body at AoA 0, Re 20 | `|CL| < 1e-3` |
| Galilean sanity | doubling `N` at fixed Re changes CD by < 5% at dev tier (convergence trend) |
| Gradient | `autograd.grad(CD, q)` equals central difference of CD in `q` for one step (float64) |

## Pitfalls

- Using `f` before collision instead of `f_col` gives a wrong force (commonly a 1–5% bias).
- Cylinder reference length is the diameter; airfoil is the chord. Both are `N` cells by construction.
- For unsteady runs report the period mean, and the RMS of `CL` (shedding amplitude) as a diagnostic.
