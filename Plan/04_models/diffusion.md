# `diffusion/` — physics-guided conditional diffusion over control points

**Work package:** WP 16 (model), WP 17 (training). **Depends on:** WP 14. **Math:** derivations §6. **Decisions:** G11, G12, G13.

## Purpose

Tier 3, inverse design: given a target `c = [CL_ref, CD_ref, Re, AoA]` and thickness constraints, produce `alpha` in one sampling pass (50 steps), steered at each step by the frozen Transformer PINN's exact gradients.

## Variable

- `z = standardize(nurbs.full_to_free(alpha))`, dimension `n_free` = `k − 2` (symmetric) or `2k − 4` (cambered).
- Standardization stats (mean, std per dimension) from the diffusion training set; stored in the checkpoint.

## `diffusion/schedule.py`

```python
class CosineSchedule:
    def __init__(self, T=1000, s=0.008): ...     # gamma_bar [T+1], derivations (6.1)
    def q_sample(self, z0, t, eps) -> Tensor      # derivations (6.2)
    def ddim_timesteps(self, n_steps) -> Tensor   # evenly spaced, descending
```

## `diffusion/denoiser.py`

```python
class ShapeDenoiser(nn.Module):
    def __init__(self, n_free, x_ctrl_free: Tensor, surface_id: Tensor, d=128, layers=4, heads=4, cond_dim=4): ...
    def forward(self, z_t [B, n_free], t [B], cond [B, 4] | None) -> Tensor    # eps_hat [B, n_free]
```

- One token per free control point: `Linear(1 → d)(value) + pos_embed(x_c, surface_id)` (sinusoidal on `x_c`, learned per surface).
- Timestep: sinusoidal embedding → MLP → added to every token (and to a CLS token).
- Condition `[CL, CD, log Re, AoA_rad]` (standardized) → MLP → one extra token. Classifier-free: replaced by a learned null token with probability `cond_drop` during training.
- 4 pre-norm Transformer encoder layers (self-attention among tokens, here interaction between control points **is** wanted).
- Output: `Linear(d → 1)` per control-point token → `eps_hat`.
- Parameters ≈ 0.8 M.

## `diffusion/guided_shape_diffusion.py`

```python
class GuidedSampler:
    def __init__(self, denoiser, schedule, nurbs, pinn, norm, cfg): ...
    def sample(self, cond: Tensor [B, 4], t_min: Tensor [k], n_steps=50, cfg_scale=2.0,
               guide_scale=1.0, guide_clip=1.0, generator=None, record=False) -> SampleResult
    # SampleResult: alpha [B, D], z_trajectory [n_steps, B, n_free] if record, J_history [n_steps, B]

def project_thickness(nurbs, alpha, t_min, iters=50) -> Tensor   # small projected-gradient fix-up after sampling
```

Per step from `t` to `t'` (derivations 6.3–6.6):

1. `eps_c = denoiser(z_t, t, c)`, `eps_u = denoiser(z_t, t, None)`, `eps_hat = (1 + w) eps_c − w eps_u`.
2. `z0_hat = (z_t − sqrt(1 − gamma_bar_t) eps_hat) / sqrt(gamma_bar_t)`; clip to `[−4, 4]` (standardized units).
3. With `z0_hat` as a grad-enabled leaf: `alpha0_hat = free_to_full(unstandardize(z0_hat))`, `CD, CL = surface_forces(pinn, nurbs, alpha0_hat, cond_pinn)`, `J` per derivations (6.5).
4. `g = autograd.grad(J.sum(), z0_hat)` with `z0_hat` a detached leaf (not `z_t`: the denoiser Jacobian `dz0_hat/dz_t` is not positive definite and made the step climb J, WP 17 round 2); per-sample norm clip to `guide_clip`.
5. `z0_hat ← z0_hat − guide_scale · g`; `z_t' = sqrt(gamma_bar_t') z0_hat + sqrt(1 − gamma_bar_t') eps_hat`.
6. Skip guidance for the first 20% of steps (very noisy `z0_hat` gives meaningless physics) — configurable `guide_start`.

After the last step: `alpha = free_to_full(unstandardize(z_0))`, then `project_thickness`, then report `CD, CL` by the PINN and (in the benchmark) by the LBM.

Cost: 50 steps × (2 denoiser calls + 1 PINN surface evaluation with backward). Target < 1 s for B = 16 on MPS.

## Implementation notes (WP 16)

- `physics_fn(alpha, cond)` takes `cond`; `ddim_timesteps(n)` skips `t = T` (`gamma_bar_T = 0`).
- `project_thickness` is a Gauss–Newton step (thickness is linear in `alpha`); `t_min` must be 0 at control points 0, 1 and k−1 (thickness is identically 0 there).
- Measured: 50-step guided sample, B = 16, untrained smoke Transformer PINN on MPS ≈ 3.7 s (target was < 1 s; the PINN value-and-grad dominates).
- Guidance with a fixed `guide_scale` plus norm clip overshoots when too large (stand-in: 0.3 good, 1.0 worse than unguided) → the WP 17 guidance sweep is required, not optional.

## Tests

| Test | Pass |
|---|---|
| Schedule | `gamma_bar` monotone, `gamma_bar_0 ≈ 1`, `gamma_bar_T ≈ 0`; `q_sample` empirical mean/var match theory for 10⁵ samples |
| DDIM inversion | deterministic DDIM with a perfect oracle `eps` recovers `z0` |
| Overfit | 16 shapes, 2,000 steps → unguided samples within 0.01 of a training shape |
| Validity | 64 unguided samples from the trained model: ≥ 95% pass `is_valid` |
| Guidance helps | with a stand-in PINN (analytic `CD = ‖alpha − a*‖²`), guided `J` < unguided `J` by > 50% |
| Null condition | `cond = None` path works and differs from conditioned output |
| Device | sampling on MPS matches CPU (same generator) within 1e-3 |

## Pitfalls

- Guiding on the noisy `z_t` with a PINN trained on clean shapes is out of distribution (G11) — always guide through `z0_hat`.
- Guidance gradients through the PINN need `create_graph=False`; only first derivatives of `J` w.r.t. `z_t` are needed (the PINN's internal autograd for velocities does need `create_graph=True`).
- Condition values outside the training range (e.g. a CD lower than any training shape) give poor samples; the sampler warns when `c` is outside the 1st–99th percentile of the training labels.
