# `pinn/embeddings.py`, `pinn/mlp_pinn.py` — baseline parametric PINN

**Work package:** WP 12. **Depends on:** WP 11. **Math:** derivations §5.

## Purpose

The plain baseline for the 4-way benchmark (method 2): same inputs, same outputs, same loss as the Transformer PINN, so any difference is due to the architecture.

## `pinn/embeddings.py` (shared with the Transformer)

```python
class FourierFeatures(nn.Module):      # fixed random B ~ N(0, scale²), [2, F]; out [..., 2F] = [sin(2πxB), cos(2πxB)]
class Wavelet(nn.Module):              # w1·sin(z) + w2·cos(z), learnable w1, w2 per channel (init 1, 1)
class InputNorm(nn.Module):            # standardizes alpha and cond with stats stored in the checkpoint
def wall_distance(xy: Tensor, boundary: Boundary) -> Tensor   # [B, N], min distance to surface polyline, detached
def predict_flow(model, xy, alpha, cond, boundary, derivs: int = 0) -> FlowFields
    # sets requires_grad on xy, calls model, returns u, v, p (+ derivatives up to `derivs` order for the loss)
```

`predict_flow` is the only place that turns `psi` into velocities, so both models share the derivative code:
- `g1 = autograd.grad(psi.sum(), xy, create_graph=True)` → `psi_x, psi_y` → `u = psi_y`, `v = −psi_x`.
- Second and third order by further `autograd.grad` calls on sums of the needed components.
- `p_x, p_y` from `autograd.grad(p.sum(), xy)`.

The "sum then grad" trick is valid because each output point depends only on its own `xy` (no cross-point interaction; see transformer rule).

## `MLPPINN`

```python
class MLPPINN(nn.Module):
    def __init__(self, D: int, hidden=128, layers=6, fourier=64, scale=2.0, v2=False): ...
    def forward(self, xy [B, N, 2], alpha [B, D], cond [B, 3], d_wall [B, N]) -> PINNOutput
```

1. `ff = FourierFeatures(xy)` → `[B, N, 128]`.
2. Concatenate with `d_wall`, broadcast `norm(alpha)` `[B, 1, D]` and `norm(cond)` `[B, 1, 3]` along `N`.
3. 6 × (Linear 128 → SiLU). Output Linear → 2 channels (`psi`, `p`) or 4 when `v2` (`fx`, `fy`).
4. Output scaling: `psi` multiplied by a fixed scale (≈ window height) so the initial `u = ∂psi/∂y` is O(1).

Parameters: about 0.1 M.

## Tests

| Test | Pass |
|---|---|
| Shapes | output `[B, N]` per channel for B ∈ {1, 4}, N ∈ {1, 100} |
| Divergence-free | `u_x + v_y` < 1e-5 everywhere (float32) |
| Point independence | outputs for a permuted point set are the permuted outputs; derivative via sum-trick equals per-point loop on 8 points |
| Third derivatives finite | on CPU and MPS |
| Fit Kovasznay | with `alpha` ignored, fits Kovasznay `(u, v, p)` at Re 20 to relative L2 < 1e-2 in 2,000 steps (CPU) — proves the loss and derivative plumbing |

## Pitfalls

- SiLU is C∞; do not swap in ReLU (third derivative is zero almost everywhere).
- Fourier scale too high → noisy third derivatives; keep `scale ≤ 4` for a window of size ~5.
- `d_wall` must be detached and computed from the same `Boundary` passed to the loss, otherwise the surface condition and the input disagree.
