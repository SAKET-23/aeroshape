# End-to-end on the Mac mini — `scripts/00_smoke_e2e.py`

**Work package:** WP 20 (grows with each phase). **Goal:** prove the whole pipeline runs and connects on the Mac mini, on CPU and on Metal (MPS), before any long run.

## Machine setup (once)

```bash
cd ~/Desktop/Code/projects/pinns
uv python install 3.12
uv sync --extra dev                  # installs torch with MPS support (macOS arm64 wheel)
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# expected: 2.x True
```

## Stages

Each stage writes to `runs/smoke/<device>/` and records `ok`, wall-clock and key numbers in `summary.json`. A failed stage stops the run with a clear message naming the stage.

| # | Stage | Available after | Checks |
|---|---|---|---|
| 1 | Geometry: circle and NACA0012 NURBS, map to a 200 × 100 grid | Phase 1 | links found, no fallback `q`, mask symmetric |
| 2 | LBM forward: 2,000 steps, Re 20, circle | Phase 1 | CD finite and in [1.5, 3.5] (coarse grid), mass drift < 1e-4 |
| 3 | Adjoint gradient (smoke horizon) + one optimizer step, Test Case 1 | Phase 2 | gradient finite, non-zero; CD after step ≤ CD before (+1e-3 tolerance) |
| 4 | Dataset: 3 shapes at smoke tier | Phase 2 | `merged.pt` matches the contract |
| 5 | PINN: MLP and Transformer, 100 steps each on the 3-shape set | Phase 3 | loss falls ≥ 50%; no NaN |
| 6 | Stress integrator: CD, CL and gradient from each PINN | Phase 3 | finite; gradient shape `[D]` |
| 7 | Diffusion: train 300 steps on 200 PINN-labelled shapes, sample 4 guided shapes | Phase 4 | ≥ 3 of 4 valid; guided J ≤ unguided J |
| 8 | Benchmark harness: Test Case 1, all 4 methods, 2 cycles each, LBM re-score | Phase 4 | `results.md` written with 4 rows |

## Pass criteria

- Exit code 0 on `--device cpu` and `--device mps`.
- Every artefact listed above exists.
- CPU vs MPS: stage 2 CD relative difference < 1e-4; stage 5 final loss relative difference < 5e-2.
- Comparison with `tests/fixtures/smoke_reference.json` within the same tolerances.
- Total wall-clock: target under 10 minutes per device (recorded; a miss is reported, not a failure).

## Commands

```bash
uv run python scripts/00_smoke_e2e.py --device cpu
uv run python scripts/00_smoke_e2e.py --device mps
uv run python scripts/00_smoke_e2e.py --device mps --stages 1-3      # partial run during Phase 2
uv run pytest -m e2e                                                   # same, via pytest
```

## After smoke: dev-tier run order on the Mac mini

1. `01_validate_lbm_cylinder.py --tier dev --device mps` → Gate 1.
2. `01_validate_lbm_cylinder.py --adjoint-check --tier dev --device cpu` → Gate 2a.
3. `06_optimize_case.py --case case1_circle_to_naca_re20 --evaluator adjoint_lbm --tier dev --device mps`.
4. `02_generate_lbm_dataset.py --set v1 --tier dev --device mps` → Gate 2b.
5. `03_train_transformer_pinn.py --model transformer|mlp --tier dev --device mps` → Gate 3.
6. `04_train_diffusion_model.py --tier dev --device mps`.
7. `05_run_4way_benchmark.py --cases 1 --tier dev --device mps`.
8. Overnight: dataset v2, PINN v2, benchmark cases 2–3.
