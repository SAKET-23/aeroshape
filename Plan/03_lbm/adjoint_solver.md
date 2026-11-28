# `lbm/adjoint_solver.py` — discrete adjoint and shape gradient

**Work packages:** WP 7 (solver), WP 8 (gradient pipeline and Gate 2a). **Depends on:** WP 5. **Math:** derivations §4. **Decisions:** G8, G25.

## Purpose

Compute `dI0/dalpha` for a cost `I0` (drag, or the augmented Lagrangian of drag, lift and thickness) at the cost of roughly one extra flow solve, independent of the number of control points. This is Tier 1 of the benchmark and the ground-truth gradient the PINN gradient is compared against.

## Interface

```python
CostFn = Callable[[Tensor, Tensor, GridGeometry, LBMParams], Tensor]   # (f_col, f_next, geom, params) -> scalar per step

def cost_source(f, geom, params, cost: CostFn) -> tuple[Tensor, Tensor]
    # dg/df [9, Ny, Nx] and dg/dq [L] at one step, by local autograd

def adjoint_step_hand(F_next, f, geom, params) -> Tensor        # transposed step, BGK, derivations (4.4)-(4.8)
def adjoint_step_vjp(F_next, f, geom, params) -> Tensor         # same via torch.autograd.grad on step(); BGK or TRT
def solve_adjoint(f_ref, geom, params, cost, mode="steady"|"time_averaged",
                  horizon_steps=None, tol=1e-8, impl="hand"|"vjp") -> AdjointResult
def shape_gradient(dI_dq: Tensor, q: Tensor, alpha: Tensor) -> Tensor     # autograd.grad(q, alpha, dI_dq)
def lbm_value_and_grad(alpha, nurbs, cond, cfg) -> EvalResult             # forward solve + adjoint + gradient, end to end
def fd_gradient(alpha, nurbs, cond, cfg, delta=1e-4, idx=None) -> Tensor  # central differences, derivations (4.11)
```

## Algorithm

### Forward part (inside `lbm_value_and_grad`)

1. `boundary = nurbs(alpha)` (rotate for AoA) → `geom = map_to_grid(boundary, params)`; keep the autograd graph from `alpha` to `geom.links.q`.
2. `res = run(...)` with `torch.no_grad()` (no graph through time; the adjoint replaces it).
3. Reference state `f_ref`: converged `f` (steady) or `res.f_mean` (time-averaged).

### Adjoint iteration

Linearized around a fixed `f_ref`, so the Jacobian is the same each iteration:

```
F* ← 0
repeat:
    F*_new = A^T(F*) + dg/df          # A^T = adjoint_step_* linearized at f_ref
    F*_new[:, :, 0] = 0; F*_new[:, :, -1] = 0      # inlet/outlet, derivations (4.8)
    F*_new[:, solid] = 0
    until ‖F*_new − F*‖ / ‖F*_new‖ < tol            (steady)
       or iterations == horizon_steps                (time_averaged; report: 20 convective times)
```

### Hand-derived transposed BGK step (`adjoint_step_hand`)

1. **IBB transpose first** (it is the last linear op of the forward step): for each link, take `a* = F*[opp_i, x_A]`, set `F*[opp_i, x_A] ← 0` (that slot was overwritten in the forward step, so the streamed value never reached it), and scatter:
   - `q < 0.5`: `+2q a*` to `G_col[i, x_A]`, `+(1 − 2q) a*` to `G_col[i, x_A − c_i]`
   - `q ≥ 0.5`: `+a*/(2q)` to `G_col[i, x_A]`, `+(2q − 1)/(2q) a*` to `G_col[opp_i, x_A]`
   - `far_ok = False`: `+a*` to `G_col[i, x_A]`
2. **Reverse streaming**: `G[i](x) = F*[i](x + c_i)` → `torch.roll(·, shifts = −c_i)`; add the IBB scatter contributions.
3. **Adjoint collision** at `f_ref` (moment form, derivations 4.6): `F*_i = (1 − omega) G_i + omega F*eq_i(G; u_ref)`.
4. Solid nodes and inlet/outlet columns set to zero.

`adjoint_step_vjp` instead calls `torch.autograd.grad(step(f_ref_requires_grad, …)[0], f_ref, grad_outputs=F_next)`. It needs no derivation and supports TRT and the sponge; it costs about 2–3× a forward step and a little memory (one step's graph).

### Gradient

```
dI/dq_l = Σ_over_iterations [ dg/dq_l + F*_{opp_i}(x_A) · ∂f_{opp_i}/∂q_l ]     (derivations 4.3, 2.7)
dI/dalpha = autograd.grad(q, alpha, grad_outputs=dI/dq)                         (one call)
```

For the steady case the sum over iterations collapses to the converged `F*` times one step's `∂S/∂q` (the system is at a fixed point). The implementation computes `∂S/∂q` with one VJP of `step` with respect to `q` at `f_ref`, which covers BGK and TRT alike.

### Lift and constraints in one solve

The cost passed in is the combined scalar `CD + lambda CL + (rho_al/2)(CL − CL_ref)²`. The quadratic term's derivative is `rho_al (CL − CL_ref) ∂CL`, evaluated at the current `CL` (a constant during the solve). Thickness penalties depend on `alpha` only, so they are added outside the LBM by autograd on `nurbs.thickness_at_ctrl`.

## Validation ladder (all CPU float64, 100 × 50 grid, N = 10, Re 20, BGK, outlet `fixed`)

| Level | Test | Pass |
|---|---|---|
| 1 | `adjoint_step_hand` vs `adjoint_step_vjp` on random `F*` | max abs diff < 1e-10 |
| 2 | Dot-product test: random `v`, `w`: `⟨w, J v⟩ = ⟨Jᵀ w, v⟩` with `J v` by `torch.func.jvp` | relative diff < 1e-10 |
| 3 | `dI/dq` vs FD in `q` (perturb one link's `q`, rerun to steady) | relative error < 1e-4 for 10 random links |
| 4 | `dI/dalpha` vs `fd_gradient` (k = 8) | relative L2 < 1e-3 |
| 5 (Gate 2a) | Circle, **k = 41**, Re 20, N = 20 (dev), float64 CPU | relative L2 < 2e-3 (report Fig 5.1) |
| 6 | Same as 4 on MPS float32 vs CPU float64 adjoint | cosine similarity > 0.999 |

Level 5 runs 82 forward solves; it is marked `slow` and run by `scripts/01_validate_lbm_cylinder.py --adjoint-check`.

## Time-averaged mode caveat

At Re 1000 the mean flow is linearly unstable, so iterating the linearized adjoint around it grows without bound. Use the report's fixed horizon (20 convective times), track `‖F*‖` per convective time in `residual_history`, and stop early if it grows by more than 10× in one convective time (logged warning). The gradient is then an approximate descent direction, not an exact derivative. Checked by:
- sign agreement with central differences of the period-averaged drag on ≥ 70% of control points (dev tier);
- the optimizer's backtracking line search accepting a step in ≥ 80% of cycles.

In this mode `_forward` passes `window_convective_times` so a run with no detectable period returns a long-window mean (never a single snapshot), and `AdjointLBMEvaluator` does not warm-start the adjoint across cycles (the forward still warm-starts).

## Pitfalls

- Overwritten populations: in the forward step the IBB value replaces the streamed one, so in the transpose the adjoint of the streamed-in value must be zero. Forgetting this is the classic bug; level 1 catches it.
- Solid-node handling must match exactly between forward and transpose (both zero).
- `index_add_` with duplicate indices (one node can be `x_A − c_i` for several links) is required; never use `index_put` for the scatter.
- Run validation with outlet `fixed`; with `extrapolate`, hand and VJP differ at the outlet column by design.
