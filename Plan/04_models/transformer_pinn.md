# `pinn/transformer_pinn.py` — Geometry cross-attention PINN

**Work package:** WP 13. **Depends on:** WP 11. **Math:** derivations §5. **Decisions:** G9, G10.

## Purpose

Tier 2 surrogate: `(x, y, alpha, Re, AoA) → (psi, p)` with the shape entering as a set of control-point tokens that each flow point attends to. Gives exact mass conservation (stream function), fast inference, and gradients `∇_alpha CD, CL` through the stress integrator. The attention weights are the interpretability output compared with the adjoint field.

## Architecture

```text
control points (x_c, y_c, surface_id)  [B, D, 3] ──► GeomEmbed: Linear(3,d) → Wavelet → Linear(d,d)
condition [log Re, sin AoA, cos AoA]   [B, 3]    ──► CondEmbed: Linear(3,d) → Wavelet → Linear(d,d)   (1 token)
                         tokens G = concat  [B, D+1, d]
                         ──► 2 × pre-norm self-attention block (MHA + Wavelet FFN, residual)

flow points (x, y, d_wall)             [B, N, 3] ──► QueryEmbed: FourierFeatures(x,y) ⊕ d_wall → Linear → Wavelet → Linear   [B, N, d]
                         ──► 2 × cross-attention block: Q = points, K = V = G  (pre-norm, residual, Wavelet FFN)
                         ──► Head: Linear(d,d) → Wavelet → Linear(d,d) → SiLU → Linear(d, 2 or 4)
outputs psi, p (+ fx, fy in v2)
```

| Hyperparameter | smoke | dev / full |
|---|---|---|
| `d_model` | 64 | 128 |
| heads | 4 | 4 |
| self-attention layers | 1 | 2 |
| cross-attention layers | 1 | 2 |
| FFN width | 2d | 4d |
| Parameters (approx.) | 0.1 M | 0.8 M |

### Hard rules

1. **Points never attend to each other** and there is no BatchNorm or any reduction over `N`. Each point's output depends only on its own inputs plus the shared tokens. This is what makes per-point derivatives correct via the sum trick, and makes inference cost linear in `N`.
2. LayerNorm normalizes over channels only (per point) — allowed.
3. All activations C³ or smoother (Wavelet, SiLU). No ReLU / GELU-approximate tanh is fine but not needed.
4. Geometry tokens are computed once per shape and reused for all `N` points (cache them in `forward`; expose `encode_geometry()` for the optimizer and diffusion, which query many points for one shape).

## Interface

```python
class GeometryCrossAttentionPINN(nn.Module):
    def __init__(self, D: int, d_model=128, heads=4, n_self=2, n_cross=2, v2=False, fourier=64, scale=2.0): ...
    def encode_geometry(self, tokens [B, D, 3], cond [B, 3]) -> Tensor           # G [B, D+1, d]
    def forward(self, xy [B, N, 2], alpha [B, D], cond [B, 3], d_wall [B, N],
                return_attn: bool = False) -> PINNOutput                         # attn [B, heads, N, D+1]
```

`tokens` come from `NURBSAirfoil.control_tokens(alpha)` so the token positions follow the real control-point layout (including the repeated leading-edge `x = 0`).

## Attention implementation

- Default: explicit attention (WP 13: SDPA third derivatives work on MPS but SDPA double backward is not implemented on CPU). `sdpa` remains selectable.
- **MPS check (first task of WP 13)**: compute third derivatives of `psi` w.r.t. `xy` through SDPA on MPS and compare with CPU. If it errors or differs (> 1e-3 relative), switch to the explicit version `softmax(Q Kᵀ / √d_h) V` via a config flag `pinn.attn_impl: sdpa | explicit`. The explicit version is always used when `return_attn=True`.

## Sizes and cost

Per query point the cross-attention is `O((D + 1) · d)`; with D ≤ 33 and d = 128, a batch of 10⁴ points is about 0.1 GFLOP forward. Third derivatives cost roughly 10–20× forward. Training step at dev tier (8 shapes × 5,600 points): target < 0.5 s on M4 Pro MPS, measured in WP 13.

## Interpretability output

`attn` averaged over heads and over the last cross-attention layer gives, for each flow point, a weight per control point. Plot (WP 19): for each control point, the field of attention weight; and the "attention-weighted sensitivity" `Σ_k attn_k · |dCD/dalpha_k|` next to the adjoint `|F*|` magnitude.

## Tests

| Test | Pass |
|---|---|
| Shapes | as MLP; `attn` sums to 1 over the token axis |
| Divergence-free | < 1e-5 |
| Point independence | permutation equivariance; sum-trick derivative = per-point loop |
| Geometry caching | `forward` equals using a precomputed `encode_geometry` |
| MPS higher-order autograd | third derivatives on MPS match CPU within 1e-3 relative (both `sdpa` and `explicit`) |
| Overfit one shape | 1 shape, anchor data only, velocity relative L2 < 2% in 2,000 steps (dev size, MPS) |
| Condition sensitivity | changing `cond` changes outputs (no dead condition path) |

## Pitfalls

- Applying dropout in attention breaks point independence at train time only — keep dropout 0.
- Using `need_weights=True` with `nn.MultiheadAttention` may route to a slower path; prefer the module's own explicit attention for weights.
- Large `fourier` scale + Wavelet activations can make third derivatives explode at init; scale the head's last layer init by 0.1.
