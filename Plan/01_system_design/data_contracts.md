# Data contracts

Single source of truth for every type that crosses a module boundary. Defined in `src/aeroshape/core/types.py` as frozen dataclasses (tensor fields). Any doc that names an interface must match this file.

## Frames and units

- **Non-dimensional frame**: leading edge at `(0, 0)`, chord along `+x`, chord length 1. For the cylinder, centre at `(0.5, 0)` and diameter 1, so the same window works for both.
- **Angle of attack**: the body is rotated by `−AoA` about the quarter-chord `(0.25, 0)`; inflow is always `+x`. Lift is `+y`.
- **Lattice frame**: node `(j, i)` at row `j` (y), column `i` (x). Body reference point placed at column `body_offset_d · N`, row `Ny/2`. Conversion `x = (i − i0)/N`, `y = (j − j0)/N`.
- Lattice → non-dimensional fields: derivations (2.9).

## Types

| Name | Fields | Notes |
|---|---|---|
| `Condition` | `re: float`, `aoa_deg: float` | `as_tensor() -> [3] = [log Re, sin AoA, cos AoA]` |
| `Boundary` | `x, y, nx, ny, ds: [B, 2M−1]` | closed CCW loop starting at the trailing edge along the upper surface; outward unit normals; `ds` trapezoid arc-length weights; non-dimensional |
| `LinkSet` | `node: [L] long`, `dir: [L] long (1..8)`, `q: [L] float`, `far_ok: [L] bool`, `seg: [L] long` | `node` is the flat index `j·Nx + i` of fluid node `x_A`; `dir` points from `x_A` into the solid; `q` differentiable w.r.t. `alpha`; `seg` = which surface segment was hit (for debugging and plots) |
| `GridGeometry` | `solid: [Ny, Nx] bool`, `links: LinkSet`, `N: int`, `origin: (i0, j0)` | output of `map_to_grid` |
| `LBMParams` | `tau, omega, omega_minus, u_lb, N, Nx, Ny, collision, sponge: [Ny, Nx] or None` | built by `make_params(cfg, condition)` |
| `FlowState` | `f: [9, Ny, Nx]`, `rho, ux, uy: [Ny, Nx]`, `step: int` | `rho, ux, uy` computed on demand |
| `ForceHistory` | `step: [T] long`, `cd, cl: [T] float` | |
| `SolveResult` | `state: FlowState`, `f_mean: [9, Ny, Nx] or None`, `history: ForceHistory`, `cd, cl: 0-dim`, `period_steps: int or None`, `converged: bool` | |
| `AdjointResult` | `F_star: [9, Ny, Nx]`, `dI_dq: [L]`, `dI_dalpha: [D]`, `residual_history: [T]` | |
| `EvalResult` | `cd, cl: 0-dim`, `dcd_dalpha, dcl_dalpha: [D]`, `wall_s: float`, `extras: dict` | `Evaluator.evaluate` returns this |
| `D` | design dimension: `k` symmetric, `2k` cambered | used for every `alpha` shape below |
| `PINNOutput` | `psi, p: [B, N]`, `fx, fy: [B, N] or None`, `attn: [B, heads, N, D+1] or None` | |
| `FlowFields` | `u, v, p: [B, N]`, plus derivatives dict when requested | from `pinn.predict_flow` |

## Files

### Anchor dataset `data/lbm_anchors/<tier>/merged.pt`

```python
{
  "alpha": [S, D] float32,          # D = k (symmetric) or 2k (cambered); pinned entries are zero
  "mode": "symmetric" | "cambered",
  "re": [S] float32, "aoa": [S] float32,
  "xy": [S, P, 2] float32,          # non-dimensional coordinates of saved points (fluid only)
  "edge": [S, P] bool,              # True for points on the window edges
  "u": [S, P], "v": [S, P], "p": [S, P],
  "cd": [S], "cl": [S],
  "split": [S] int8,                # 0 train, 1 holdout
  "kind": [S] int8,                 # 0 steady, 1 time-averaged
  "meta": {"tier", "cells_per_d", "git", "created", "lbm_params": [...]}
}
```

Per-shape files `<shape_id>.pt` have the same keys without the leading `S` axis; `merge_anchors()` stacks them. Points are fluid-only, so `P` may differ per shape; it is fixed by subsampling to exactly `points_per_shape`.

### PINN checkpoint `runs/<name>/ckpt/step_XXXXXX.pt`, `best.pt`

`{"model": state_dict, "config": dict, "norm": {"alpha_mean", "alpha_std", "xy_scale"}, "step": int, "metrics": dict, "torch": str, "git": str, "device": str}`

### Diffusion checkpoint

`{"model": state_dict, "ema": state_dict, "config": dict, "norm": {"mean": [k_free], "std": [k_free]}, "cond_norm": {...}, "step": int}`

### Run outputs

- `runs/<name>/log.csv`: one row per logged step; first columns `step, wall_s` then metrics.
- `runs/<name>/results.json`: final metrics, all floats, plus `device_info` and config hash.
- `runs/<name>/history.pt`: optimizer history `{"alpha": [C, D], "cd": [C], "cl": [C], "viol": [C], "wall_s": [C]}`.

## Invariants (checked by tests)

- `Boundary` loop is closed (first point = last point) and counter-clockwise (shoelace area > 0).
- `LinkSet.q` in `[q_min, 1]`; `node` is fluid; `node + offset(dir)` is solid.
- `alpha[..., 0]` and `alpha[..., -1]` (and the lower-surface ends in cambered mode) are zero after `NURBSAirfoil` masking.
- Anchor `xy` points lie outside the body (ray test) and inside `dataset.window_d`.
