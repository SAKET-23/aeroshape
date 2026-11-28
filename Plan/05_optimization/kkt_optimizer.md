# `optimization/` — evaluators and the constrained shape optimizer

**Work package:** WP 9. **Depends on:** WP 8 (WP 14 for PINN evaluators). **Math:** derivations §7. **Decisions:** G3, G19, G20.

## `optimization/evaluators.py`

```python
class Evaluator(Protocol):
    def evaluate(self, alpha: Tensor, cond: Condition, cost_weights: CostWeights) -> EvalResult

class AdjointLBMEvaluator:     # lbm_value_and_grad: forward solve + one adjoint for the combined cost
class PINNEvaluator:           # pinn_value_and_grad with a loaded MLP or Transformer PINN
class LBMScoreEvaluator:       # forward solve only; no gradient; used to re-score final shapes
def build_evaluator(cfg, name: str) -> Evaluator
```

`CostWeights(lam, rho_al, cl_ref)` lets the adjoint evaluator solve one adjoint for `CD + lam·CL + (rho_al/2)(CL − CL_ref)²` (derivations 7.3). PINN evaluators just autograd the same expression.

Warm start: `AdjointLBMEvaluator` keeps the last converged `f` and starts the next cycle's forward solve from it (shapes change little between cycles) — about 2–3× faster convergence. The grid geometry is rebuilt each cycle.

## `optimization/kkt_optimizer.py`

```python
class AugmentedLagrangianOptimizer:
    def __init__(self, nurbs, evaluator, cond, t_min: Tensor, cl_ref: float | None, cfg): ...
    def step(self) -> CycleRecord
    def run(self, alpha0, max_cycles) -> History
```

### One cycle

1. `res = evaluator.evaluate(alpha, cond, CostWeights(lam, rho_al, cl_ref))`.
2. Thickness penalty and its gradient by autograd: `P = (mu/2) Σ relu(t_min − thickness_at_ctrl(alpha))²`.
3. `g = res.grad_L + ∇P`, restricted to free entries (pinned entries zeroed); symmetric mode keeps symmetry automatically.
4. Step: `alpha_trial = alpha − eta · g / max(1, eta · ‖g‖_∞ / cap)` with `cap = g_cap · cap_scale · backtrack^nb` (nb = backtracks so far), so the largest control-point move is at most `g_cap` (default 0.02 chord) on the first trial and strictly smaller on every retry. (Review H1: dividing by `‖g‖_∞ / g_cap` without `eta` capped moves at `eta · g_cap`. Later fix: with a fixed cap, a large gradient made every halving of `eta` give the same trial point.)
5. **Backtracking**: evaluate `L` at `alpha_trial` (forward only for LBM, cheap for PINN); if it does not decrease, shrink `eta` and the cap by `backtrack` (up to 4 times). `cap_scale` (persistent, starts 1) multiplies by `backtrack^(nb+1)` after a flagged cycle (so the next cycle starts below the last trial) and doubles (max 1) after an accepted one. A non-finite gradient flags the cycle at once with no trial solves. If still no decrease, **do not move**: `alpha` stays, the evaluator warm start is restored to the solve at `alpha`, and the cycle is flagged. After `max_flagged` consecutive flagged cycles, stop with reason `stalled`. Accepted points therefore never have a higher `L`.
6. Multiplier updates (CL taken at the accepted point, or at `alpha` if the step was rejected): `lam ← lam + clip(rho_al (CL − CL_ref), ±lam_step_max)` (config `optimizer.lam_step_max`, default 0.5); `mu ← min(2 mu, mu_max)` only if the max thickness violation > tol **and** it is not already shrinking (`viol > 0.75 ×` the previous cycle's; the first cycle has no previous one, so it doubles).
7. Record: `alpha`, CD, CL, `L`, violation (of the cycle's start point; plus `alpha_new, cd_new, cl_new, viol_new, L_new` of the accepted trial, absent when rejected), KKT residual (derivations 7.4), `eta`, wall-clock.

### Best shape

`best_alpha` (also `history["best_alpha"]`) is the lowest-CD point (over every record's start point and accepted trial point) with violation < 1e-3 and (if `CL_ref` is set) `|CL − CL_ref| < cl_tol`; if none is feasible, the lexicographic minimum of (violation, `|CL − CL_ref|`, CD) (`L` is not comparable across cycles). `results.json` also has `best_feasible`, `best_cycle` (1-based) and `best_point` (`start`|`accepted`). It is derived from the records, so it survives checkpoint and resume. `scripts/06_optimize_case.py` projects the start shape onto the thickness floor (`project_thickness`, pinned ends unchanged), LBM re-scores and reports `best_alpha` (`alpha_final`), and stores the last iterate as `alpha_last`.

### Stopping

Any of: KKT residual < `kkt_tol` and `|CL − CL_ref| < cl_tol` and violation < 1e-3; relative CD change < 1e-4 for 3 cycles; `max_cycles` (report took 17 / 11 / 15 cycles).

### Cycle cost with the LBM evaluator (report §6.2)

Forward 40 convective times (warm-started: often less), adjoint 20 convective times. At dev tier for Test Case 1 (80k nodes, 346 steps per convective time) that is ≈ 21k forward + 7k adjoint steps.

## Test-case configs

| Case | Start | Mode, D | Re | AoA | `t_min` (G20) | `CL_ref` (G19) | Adjoint mode |
|---|---|---|---|---|---|---|---|
| 1 | circle fit | symmetric, k = 8 | 20 (Re 5 extra) | 0° | NACA0012 thickness at control stations | none | steady |
| 2 | circle fit | cambered, k from WP 2 | 1000 | 0° | RAE2822 thickness | LBM CL of RAE2822 at Re 1000, AoA 0 | time-averaged |
| 3 | NACA2412 fit | cambered | 1000 | 10° | RAE2822 thickness | LBM CL of RAE2822 at Re 1000, AoA 10 | time-averaged |

Success criterion per case: CD decreases (monotonically in the accepted steps), constraints met within tolerance at the end, and the final shape's distance to the reference airfoil is reported (`shape_distance`). Exact recovery of NACA0012 / RAE2822 is not required (the report notes the optimum depends on the constraints).

## Tests

| Test | Pass |
|---|---|
| Quadratic with linear equality | analytic evaluator `CD = ‖a − a*‖²`, `CL = wᵀa` → converges to the analytic KKT point within 1e-4 |
| Thickness active | analytic evaluator that rewards thinning → final violation < 1e-3 |
| Backtracking | evaluator with a step that overshoots → `eta` halves and `L` decreases |
| Pinned ends | never move |
| Smoke run | Test Case 1, smoke tier, adjoint LBM, 2 cycles: CD(cycle 2) ≤ CD(cycle 1) |

## Pitfalls

- Mixing the PINN's and LBM's CD in one run (e.g. PINN gradient, LBM line search) is allowed only in the explicit hybrid mode, which is not part of the benchmark.
- Large steps flip many links at once and the time-averaged adjoint direction is approximate — keep `g_cap` small for cases 2 and 3.
