# `scripts/05_run_4way_benchmark.py` — the 4-way benchmark

**Work package:** WP 18. **Depends on:** WP 9, 15, 17. **Decisions:** G16, G17.

## Methods

| # | Method | Evaluator / generator | Iterations |
|---|---|---|---|
| 1 | LBM + discrete adjoint | `AdjointLBMEvaluator` + `AugmentedLagrangianOptimizer` | up to `max_cycles` |
| 2 | MLP-PINN | `PINNEvaluator(mlp)` + same optimizer | up to `max_cycles` × 10 (cheap cycles) |
| 3 | Transformer-PINN | `PINNEvaluator(transformer)` + same optimizer | same as 2 |
| 4 | Guided diffusion | `GuidedSampler.sample` with the case target; best of 8 samples by PINN `J` | 1 pass, 50 steps |

Targets for diffusion: `c = [CL_ref, CD_target, Re, AoA]` with `CD_target` = the CD of method 1's final shape (so it asks for what the classical method achieved), and `t_min` from the case.

## Cases

Test Cases 1–3 from [kkt_optimizer.md](../05_optimization/kkt_optimizer.md). Methods 2–4 use the PINN version that covers the case (v1 for case 1, v2 for cases 2–3).

## Metrics (per method × case)

| Metric | How |
|---|---|
| Wall-clock to final shape | `Timer` with device sync; excludes model training |
| Up-front cost | training time of the models used (from their `results.json`), reported separately |
| Peak device memory | `peak_memory(device)` |
| Iterations | cycles or sampling steps |
| **CD, CL (LBM)** | `LBMScoreEvaluator` on the final shape at the case's tier — the comparable number |
| CD, CL (method's own) | what the method believed; difference shows surrogate error |
| Drag reduction | `(CD_start − CD_final) / CD_start`, LBM numbers |
| Constraint violation | max thickness violation; `|CL − CL_ref|` |
| Distance to reference airfoil | `shape_distance` (chord units) |
| Single-query latency | methods 2–3: one CD + gradient evaluation; reported for the "< 2 ms" claim |

## Outputs (`runs/benchmark/<tier>/`)

- `results.json`, `results.md` (one table per case, plus a summary table).
- `fig/shapes_case{1,2,3}.png`: start, final shapes of all methods, reference airfoil.
- `fig/cd_vs_wallclock_case{n}.png`: LBM-scored CD vs time (log x-axis).
- `fig/morph_case{n}_<method>.gif`: from the optimizer history (see [visualization.md](visualization.md)).
- `histories/<case>_<method>.pt`.

## Command

```bash
uv run python scripts/05_run_4way_benchmark.py --tier dev --device mps --cases 1          # Mac, case 1
uv run python scripts/05_run_4way_benchmark.py --tier dev --device mps --cases 2 3        # Mac overnight
uv run python scripts/05_run_4way_benchmark.py --tier full --device cuda                  # cloud
```

Each method × case is a separate sub-run (`scripts/06_optimize_case.py`) with its own folder, so a crash in one does not lose the others and `--resume` skips finished ones.

## Fairness rules

- Same start shape, constraints, `CL_ref`, tier and device for all methods.
- Same optimizer hyperparameters for methods 1–3, except the cycle cap.
- Warm-up evaluation before timing for every method (MPS shader compile, CUDA context).
- Report numbers as measured; no claim (e.g. "> 1000×") is written into the README unless the table shows it.

## Results: Case 1, dev tier, MPS (2026-11-18)

Start CD is 2.4086 (circle). Every CD is the LBM re-score of the final shape (G16). Outputs are in `runs/benchmark/dev/`.

| Method | CD (LBM) | Drag reduction | Wall (s) | Iterations | Training (s) | Own CD (last) |
|---|---|---|---|---|---|---|
| LBM adjoint | 1.3956 | 42.1% | 1376 | 20 (cap; still descending) | — | 1.450 |
| Guided diffusion (guide 3, cfg 0) | 1.4075 | 41.6% | 3.9 | 50 steps × 8 samples | 13,389 | 1.396 |
| Transformer-PINN (boxed, G37) | 1.4937 | 38.0% | 5.2 | 62 (stalled) | 13,017 | −0.714 |
| MLP-PINN (boxed, G37) | 1.5465 | 35.8% | 2.0 | 70 (cd_stalled) | 2,622 | 0.615 |
| Transformer-PINN (no envelope) | 1.4987 | 37.8% | 11.0 | 118 | 13,017 | −0.701 |
| MLP-PINN (no envelope) | 1.8350 | 23.8% | 6.0 | 200 | 2,622 | 0.073 |

- **Latency.** One CD + CL + gradient query takes 17 ms (MLP) and 57 ms (Transformer) on MPS, so the "< 2 ms" target is not met (G17).
- **Constraints.** No boxed row violates thickness. The unboxed Transformer violated it by 9e-3.
- **Finding.**
  - The PINN optimizers exploit surrogate error even inside the axis-aligned training box: the Transformer's own CD is −0.71 while LBM gives 1.49.
  - The box contains combinations of control points that no training shape has. The PINN is accurate along the adjoint's own path (G34) but not along the path its own gradient takes.
  - Diffusion avoids this because its samples stay on the data manifold.
- **Not yet run.** Cases 2–3 need PINN v2 and a cambered diffusion model.
