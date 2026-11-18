# `geometry/ibb_grid_mapper.py` — body onto the lattice, with differentiable `q`

**Work package:** WP 3. **Depends on:** WP 1. **Math:** derivations (2.6), (2.7). **Decisions:** G4, G5.

## Purpose

Given a `Boundary` (non-dimensional) and lattice parameters, produce the solid mask and the list of boundary links with wall fractions `q` that are **differentiable with respect to `alpha`**. This is what makes the adjoint gradient non-zero (report §5.2.6: standard bounce-back kills the gradient).

## Interface

```python
def to_lattice(boundary: Boundary, N: int, origin: tuple[float, float]) -> tuple[Tensor, Tensor]
    # non-dim (x, y) [B, P] -> lattice coords; B must be 1 here
def solid_mask(xl: Tensor, yl: Tensor, Ny: int, Nx: int, chunk_rows: int = 64) -> Tensor   # [Ny, Nx] bool
def find_links(solid: Tensor) -> tuple[Tensor, Tensor, Tensor]        # node [L], dir [L], far_ok [L]
def link_q(node, dir, xl, yl, Nx, q_min=0.01) -> tuple[Tensor, Tensor]   # q [L] (differentiable), seg [L]
def map_to_grid(boundary, params: LBMParams) -> GridGeometry          # all of the above
```

## Algorithm

1. **Lattice coordinates**: `x_l = i0 + N·x`, `y_l = j0 + N·y`, where `(i0, j0)` = body reference placement from the config. Done in the solver dtype; gradients flow.
2. **Solid mask** (no gradient needed, computed under `torch.no_grad()` on detached coordinates):
   - Even-odd rule: for each node, count crossings of a horizontal ray to `+x` with polygon edges.
   - Vectorize over a bounding box of the body only (body occupies a small part of the grid): nodes `[R, C]` × edges `[E]`, chunked by rows to cap memory at about 64 × C × E booleans.
   - Edge test: `(y1 > yn) != (y2 > yn)` and `xn < x1 + (yn − y1)(x2 − x1)/(y2 − y1)`.
   - Post-process: a fluid node with all 4 axis neighbours solid becomes solid (removes pinholes at thin trailing edges).
3. **Links**: for each direction `i = 1..8`, `is_link = fluid & solid shifted by −c_i` (i.e. the neighbour `x_A + c_i` is solid). `nonzero` → `node`, `dir`. `far_ok = fluid at x_A − c_i`. Never use `torch.roll` wrap-around here (top/bottom periodic but the body is far from edges; assert the body bounding box is at least 2 cells from every edge).
4. **`q` by ray–segment intersection** (differentiable):
   - Link ray: `P(t) = x_A + t·c_i`, `t ∈ [0, 1]`.
   - All links × all surface segments `[L, E]` (L ≈ 10³, E ≈ 320 → 3·10⁵ pairs; trivial). Chunk if `L·E > 5·10⁶`.
   - Segment `Q(u) = a + u (b − a)`. Solve the 2×2 system by cross products: `denom = c_i × (b − a)`, `t = ((a − x_A) × (b − a)) / denom`, `u = ((a − x_A) × c_i) / denom`.
   - Valid hit: `|denom| > eps`, `0 ≤ u ≤ 1`, `0 < t ≤ 1 + tol`. Pick the smallest valid `t` per link with `torch.where(valid, t, inf).min(1)`.
   - Because `(a, b)` are surface points `= R·alpha` and `t` is a rational function of them, `q = t` is differentiable; the `argmin` selection is piecewise constant (no gradient needed).
   - Fallback when no valid hit (grazing corner): `q = 0.5`, flagged in a counter that tests assert is 0 for smooth shapes.
   - Clamp to `[q_min, 1]` with a straight-through clamp (gradient passes inside the range, zero outside).
5. Return `GridGeometry(solid, LinkSet(node, dir, q, far_ok, seg), N, origin)`.

## Topology changes

When the surface crosses a lattice node, the link set changes and drag jumps slightly. Consequences and handling:
- The adjoint gradient is the gradient **inside the current topology**. That is the correct quantity for gradient descent with small steps.
- Finite-difference checks use `Δ` small enough that the topology is unchanged (`Δ = 1e-4` chord at 20 cells/chord → 0.002 cells); the test asserts identical `node` and `dir` for `alpha ± Δ`.
- Optimizer steps cross topologies; the backtracking line search absorbs the small jumps.

## Tests (`tests/geometry/test_ibb_grid_mapper.py`)

| Test | Pass |
|---|---|
| Circle mask | solid-cell count within 1% of `pi r²`; mask symmetric about the centreline |
| Analytic `q` on a circle (polyline with 4,096 points, float64) | max error vs. exact ray–circle intersection < 1e-5 |
| Link invariants | every `node` fluid, every `node + c_dir` solid, `q ∈ [q_min, 1]` |
| Link symmetry | symmetric body: mirrored links have equal `q` to 1e-10 |
| `dq/dalpha` | autograd vs. central difference (float64, `Δ = 1e-6`) relative error < 1e-6 |
| No fallbacks | zero no-hit links for circle, NACA0012, RAE2822 at 10, 20, 40 cells per chord |
| Pinholes | no fluid node fully enclosed by solid |
| Device parity | same `node`, `dir`; `q` within 1e-6 on MPS |

## Acceptance

All tests pass; `map_to_grid` on RAE2822 at 100 cells/chord runs in < 0.5 s on MPS.

## Pitfalls

- The plan doc's `torch.cdist` gives the shortest distance to the surface, which is not the distance along `c_i`. Wrong `q` breaks both accuracy and the gradient (G4).
- Diagonal links have length √2; `q` is the fraction of the link (`t`), not a physical distance.
- Mixed dtypes: the mask is computed on detached coordinates; `q` must come from the attached ones.
