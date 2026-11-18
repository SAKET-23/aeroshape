# `geometry/nurbs.py` — differentiable NURBS airfoil

**Work package:** WP 1. **Depends on:** WP 0. **Math:** [derivations §1](../raw/derivations.md). **Decision:** G1, G2, G21.

## Purpose

Map a design vector `alpha` to a closed, smooth, differentiable body outline with outward normals and arc-length weights. Used by the grid mapper (LBM), the PINN (surface points and geometry tokens), the stress integrator, the optimizer and diffusion.

## Parameterization

- `k` control points per surface. x-positions fixed, y-positions are the design variables.
- Control-point x-positions: `x_c = [0, 0, x_2, …, x_{k−2}, 1]` where the interior are cosine-spaced `(1 − cos(pi s))/2`. **The repeated `x = 0` at index 1 makes the leading-edge tangent vertical**, which a round nose (and the circle start) needs. Index 1 is a free y-variable.
- Ends pinned: `y_0 = y_{k−1} = 0` (leading and trailing edge on the chord line).
- **Symmetric mode**: `alpha [B, k]`; lower surface is `−upper`.
- **Cambered mode**: `alpha [B, 2k]` = `[upper (k), lower (k)]`. Lower-surface values are usually negative.
- Free dimensions: `k − 2` (symmetric), `2k − 4` (cambered). The pinned entries are zeroed by a fixed mask buffer, so callers may pass any value there.
- Default `k = 8` symmetric (plan doc). For cambered, WP 2 measures the RAE2822 fit error for `k ∈ {8, 10, 12, 16}` and sets the case config to the smallest `k` with max error < 2e-3 chord.
- Weights: buffer, default all ones (B-spline). A circle-exact mode is not needed because the analytic circle is used for LBM validation (see [validation.md](../03_lbm/validation.md)).

## Interface

```python
class NURBSAirfoil(nn.Module):
    def __init__(self, k: int = 8, degree: int = 3, M: int = 160,
                 mode: Literal["symmetric", "cambered"] = "symmetric",
                 weights: Tensor | None = None, device=None, dtype=None): ...
    n_free: int                                   # k-2 or 2k-4
    x_ctrl: Tensor                                # [k] buffer
    def forward(self, alpha: Tensor) -> Boundary  # alpha [B, k] or [B, 2k]
    def surface_y(self, alpha) -> tuple[Tensor, Tensor]   # (x [M], y_upper [B, M], y_lower [B, M])
    def thickness_at_ctrl(self, alpha) -> Tensor  # [B, k]: y_upper - y_lower evaluated at x_ctrl
    def area(self, alpha) -> Tensor               # [B], shoelace on the loop
    def control_tokens(self, alpha) -> Tensor     # [B, k_tot, 3] = (x_c, y_c, surface_id) for the PINN
    def free_to_full(self, z: Tensor) -> Tensor   # [B, n_free] -> [B, k or 2k], inserting zeros
    def full_to_free(self, alpha) -> Tensor

def rotate(boundary: Boundary, aoa_deg: float, pivot=(0.25, 0.0)) -> Boundary
def basis_matrix(k, degree, s: Tensor, weights) -> tuple[Tensor, Tensor]   # R [M, k], dR/ds [M, k]
```

## Algorithm

1. **Knots**: clamped uniform, length `k + degree + 1`.
2. **Parameter samples**: `s_m = (1 − cos(pi m/(M−1)))/2`, so points cluster at both ends (leading-edge curvature and trailing edge).
3. **Basis**: Cox–de Boor (derivations 1.1) evaluated on CPU in float64 once in `__init__`, vectorized over `s` (loop only over degree `p ≤ 3` and span index, never over points). Rational normalization (1.2). Derivative basis (1.3). Store `R`, `dR` as buffers cast to the model dtype.
4. **Forward**:
   - `a = alpha * mask` (pins ends, no in-place write).
   - `x = R @ x_ctrl` → `[M]`; `y_up = a_up @ R.T` → `[B, M]`; `y_lo` = `−y_up` or `a_lo @ R.T`.
   - Loop: trailing edge → along the **upper** surface to the leading edge → along the lower surface back to the trailing edge → `2M − 1` points, counter-clockwise (positive shoelace area, outward normals). (An earlier draft said lower-first; that order is clockwise.)
   - Tangents from `dR` (analytic), normals `n = (y', −x')/|T|` for CCW (derivations 1.3). At the leading edge the two surfaces share a point; use the average of the two one-sided normals.
   - `ds`: trapezoid weights from segment lengths.
5. **thickness_at_ctrl**: evaluate `y_up − y_lo` at the x-positions of the control points. Since `x(s)` is monotone, invert by a precomputed nearest-`s` table (fine grid, 4,096 samples, linear interpolation). Fixed table → differentiable in `alpha`.

## Device notes

- Basis built in float64 on CPU, then moved; forward is matmul only, so identical on all devices to float32 precision.
- `M = 160` per side gives about 320 boundary points; enough for grids up to 100 cells per chord (≥ 1 point per lattice spacing near the nose thanks to clustering). Raise `M` if a grid has more boundary links than points.

## Tests (`tests/geometry/test_nurbs.py`)

| Test | Pass condition |
|---|---|
| Partition of unity | `R.sum(1) == 1` to 1e-12 (float64) |
| Pinning | `y` at both ends is exactly 0 for random `alpha` |
| Closed CCW loop | first point = last point; shoelace area > 0 |
| Normals | unit length to 1e-6; for an ellipse-like shape, `n · (x − centroid) > 0` everywhere |
| Symmetric mirror | `y_lower == −y_upper` |
| Linearity | `forward(a + b).y == forward(a).y + forward(b).y` (x fixed) |
| gradcheck | `torch.autograd.gradcheck` of `y` and of `thickness_at_ctrl` w.r.t. `alpha`, float64 |
| Circle fit | least-squares `alpha` for `y = sqrt(x(1 − x))`, error measured on `x ≤ 0.95`: max < 2.5e-2 at k = 8 (measured 2.2e-2; the k = 8 circle is only the coarse Test Case 1 start shape), < 3e-3 at k = 41 (measured 2.3e-4). Full-chord max error is reported, not gated: the trailing edge has a finite slope by design, so it cannot follow the circle's vertical tangent there (measured: 4.7e-2 at k = 8, 7.2e-3 at k = 41, all within the last 1% of chord). See G21 |
| Device parity | CPU vs MPS boundary identical to 1e-6 |

## Acceptance

All tests pass on CPU and MPS; `uv run pytest tests/geometry/test_nurbs.py` under 10 s.

## Pitfalls

- Cox–de Boor's last basis function is zero at `s = 1` with half-open spans. Close the last span on the right.
- `x(s)` is not uniform in `s`; never assume `x_m = s_m`.
- The repeated leading-edge x makes `dx/ds = 0` at `s = 0`; normals there come from `dy/ds` only. Guard `|T|` with `+1e-12`.
- Do not normalize thickness by `max` (non-smooth); constraints use thickness at fixed x-stations.
