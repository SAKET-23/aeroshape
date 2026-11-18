# `pinn/navier_stokes_loss.py`, `pinn/sampling.py` — hybrid physics + data loss

**Work package:** WP 11. **Depends on:** WP 1. **Math:** derivations (5.1)–(5.4). **Decisions:** G10, G24.

## Loss

For a batch of shapes `b = 1..B` with condition `c_b`:

```
L = w_pde  · mean_colloc (r_x² + r_y²)                                       derivations (5.2)
  + w_wall · mean_surface (u² + v²)                                          no-slip
  + w_edge · mean_edge   (|u − u_LBM|² + |v − v_LBM|²)                       window edges, from anchors
  + w_pin  · (mean_outlet_edge(p) − mean_outlet_edge(p_LBM))²                pressure gauge
  + w_data · mean_anchor (|u − u_LBM|² + |v − v_LBM|² + c_p |p − p_LBM|²)
  + w_force· mean_colloc (fx² + fy²)                                         v2 only
```

### Why window-edge data instead of analytic inlet/outlet

The PINN window (`[-1.5, 4] × [-1.5, 1.5]` chords) is much smaller than the LBM domain (20 × 10). `u = 1` at `x = −1.5` and `p = 0` at `x = 4` are wrong there (upstream blockage effect, wake, and the LBM's pressure reference far downstream). The anchors therefore include extra points on the window edges, and those serve as the far-field condition. For shapes without anchors (during optimization or diffusion) the network extrapolates its learned edge behaviour.

### v1 / v2 staging

| | v1 (gated) | v2 (best effort) |
|---|---|---|
| Data | steady anchors (Re 20, 40) | + time-averaged anchors (Re 1000, AoA 0 and 10) |
| Heads | `psi, p` | `psi, p, fx, fy` |
| `w_pde` | 1 | 1 for steady samples, 0.1 for time-averaged samples |
| `w_data` | 10 | 10 steady, 30 time-averaged |
| `w_force` | — | 1e-2 (keeps `f` small where the flow is steady) |
| Gate | Gate 3 | metrics reported, no gate |

Per-sample weights use the `kind` flag in the anchors.

### Weight balancing (optional, `loss.balance_every > 0`)

Every `balance_every` steps compute `‖∇_θ L_term‖` for each term, set `w_term ← 0.9 w_term + 0.1 · (max_norm / ‖∇_θ L_term‖)` (Wang et al. 2021 learning-rate annealing). Off by default; turned on only if the PDE term dominates or vanishes in WP 15.

## Interface

```python
def ns_residuals(fields: FlowFields, nu: Tensor) -> tuple[Tensor, Tensor]       # r_x, r_y [B, N]
def hybrid_loss(model, nurbs, batch: TrainBatch, cfg, step: int) -> tuple[Tensor, dict]   # total, per-term floats (detached)

@dataclass
class TrainBatch:
    alpha: Tensor [B, D]; cond: Tensor [B, 3]; boundary: Boundary
    xy_colloc: [B, Nc, 2]; xy_surface: [B, Ns, 2]; xy_edge, uv_edge, p_edge
    xy_anchor, u_anchor, v_anchor, p_anchor; kind: [B]

def sample_batch(anchors, nurbs, cfg, generator) -> TrainBatch    # pinn/sampling.py
def inside_body(xy, boundary) -> Tensor                           # [B, N] bool, same even-odd test as the grid mapper
```

## Sampling (`pinn/sampling.py`), per shape per step

| Set | Count (dev) | How |
|---|---|---|
| Collocation, uniform | 2,000 | uniform in the window, reject inside body |
| Collocation, near wall | 2,000 | pick random surface points, offset along the normal by `d = d_min · (d_max/d_min)^U`, `d_min = 0.002`, `d_max = 0.5` (density ∝ 1/d), small tangential jitter, reject inside |
| Surface | 400 | random arc-length positions on the NURBS loop (interpolated, differentiable in alpha) |
| Edge | 400 | random subset of the anchor edge points |
| Anchor | 1,000 | random subset of the anchor interior points |

Resampled every step from a CPU `torch.Generator` (reproducible across devices), then moved. Shapes per step: 8 (dev), mixing training shapes at random.

## Curriculum (with [pinn_training.md](../06_training/pinn_training.md))

- Steps `0 … warmup_data_only`: `w_pde = 0` (data + wall + edge only). Learns the rough field fast.
- Next `pde_ramp` steps: `w_pde` ramps linearly to its target.
- Rationale: starting with a strong PDE term from a random network tends to collapse to the trivial `u ≈ 0` solution that satisfies no-slip and the PDE but not the edges.

## Tests

| Test | Pass |
|---|---|
| Kovasznay residual | analytic Kovasznay fields plugged into `ns_residuals` (via a tiny "model" returning the analytic `psi, p`) → max residual < 1e-4 |
| Sampling | no collocation point inside the body; near-wall distances follow the log-uniform law (KS test p > 0.01) |
| Loss decreases | 50 steps on one shape (smoke size) → loss falls by > 50% |
| No NaN on MPS | 100 steps, all terms finite |
| Per-term dict | all expected keys present, floats |

## Pitfalls

- Residuals use non-dimensional `nu = 1/Re`; mixing in `nu_lb` is a factor of `N/U_lb` error.
- Pressure in anchors is relative to the LBM outlet; in the window it has an offset. The pin term fixes the gauge; never compare raw `p` without it.
- `inside_body` on many points × many segments: chunk as in the grid mapper.
