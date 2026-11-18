# `pinn/stress_integrator.py` — differentiable surface forces from the PINN

**Work package:** WP 14. **Depends on:** WP 12 or WP 13. **Math:** derivations (5.5). **Decision:** G15.

## Purpose

Turn the PINN's fields into `CD`, `CL` and their gradients with respect to `alpha`. This is what the PINN evaluators, the diffusion guidance and the interpretability plots use.

## Interface

```python
def surface_forces(model, nurbs, alpha [B, D], cond [B, 3], aoa_deg, n_pts: int | None = None) -> tuple[Tensor, Tensor]
    # CD [B], CL [B]; differentiable in alpha
def pinn_value_and_grad(model, nurbs, alpha, cond) -> EvalResult        # used by PINNEvaluator
def surface_cp(model, nurbs, alpha, cond) -> Tensor                     # [B, 2M-1], for plots
```

## Algorithm

1. `boundary = rotate(nurbs(alpha), aoa)` — surface points, normals `n` (outward, into the fluid), weights `ds`, all functions of `alpha`.
2. Evaluate on the surface: `fields = predict_flow(model, xy = boundary points, alpha, cond, derivs=1)` → `p`, `∇u = [[u_x, u_y], [v_x, v_y]]`.
3. Traction: `t = −p n + nu (∇u + ∇uᵀ) n`, `nu = 1/Re` (derivations 5.5).
4. Force: `F = Σ_m t_m ds_m`; `CD = 2 F_x`, `CL = 2 F_y`.
5. Gradient: `torch.autograd.grad(CD, alpha)` flows through two paths — the surface positions/normals/weights (geometry) and the network's shape input — both needed.

The pressure gauge: since the loss pins the window-outlet pressure to the LBM's, the integrated pressure force is consistent with the LBM's momentum-exchange force (a constant offset in `p` integrates to zero over a closed loop anyway).

## Implementation notes (WP 14)

- Forces are integrated over polygon **edges** (midpoint evaluation, edge normal `(dy, −dx)/|e|`, weight `|e|`), not nodes: `Σ n_e |e| = 0` exactly on a closed loop, so a pressure offset gives zero force (node normals leaked ~3e-3).
- Measured latency of `pinn_value_and_grad` (one shape, CD + CL + both gradients, MPS, warm): ~74 ms (d = 64, M = 40), ~96 ms (d = 128, M = 160). The original plan's "< 2 ms" is not met; the cost is launch/graph bound (second-order graph inside `predict_flow`, then backward to `alpha`). Report as measured (G17); batching several shapes per call is the main lever.

## Tests

| Test | Pass |
|---|---|
| Constant pressure | `p = const`, `u = 0` → `F = 0` to 1e-6 (closed loop) |
| Analytic pressure on a circle | `p = −cos θ`, `u = 0` on a circle of radius `r = 0.5` (`n = (cos θ, sin θ)`) → `F_x = ∮ cos²θ r dθ = π r = π/2`, `F_y = 0`, both to 1e-4 |
| Pure shear | linear shear flow gives the analytic viscous force |
| Gradient | `autograd` `dCD/dalpha` vs central difference through the same model (CPU float64, k = 8) relative < 1e-5 |
| Batch consistency | B shapes at once = B separate calls |
| Rotation | rotating a symmetric body by +θ and −θ gives `CL(+θ) = −CL(−θ)` for a model with symmetric outputs (tested with an analytic stand-in model) |

## Acceptance at Gate 3

CD from the trained PINN on held-out shapes within 5% of the LBM's (see [pinn_training.md](../06_training/pinn_training.md)).

## Pitfalls

- Normals must point **into the fluid** (outward from the body); the opposite sign flips drag.
- The surface evaluation points must be the same differentiable points used in the loss's no-slip term, or the force sees an unconstrained region of the network.
- At the trailing edge the normal is discontinuous; trapezoid weights halve the end contributions, which is enough.
