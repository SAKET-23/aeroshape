# Milestones and gates

| Phase | Packages | Gate | Plan-doc days | Runs on |
|---|---|---|---|---|
| 1. Geometry + LBM | 0–6 | **Gate 1**: cylinder CD within 2% at Re 20, 40, 100 | 1–3 | Mac (dev); cloud only if Re 100 needs full tier |
| 2. Adjoint + dataset | 7–10 | **Gate 2a**: adjoint vs FD relative L2 < 2e-3 (41 control points). **Gate 2b**: v1 anchor dataset valid | 4–6 | Mac (CPU float64 for 2a) |
| 3. PINN surrogates | 11–15 | **Gate 3**: holdout velocity L2 < 2%, CD < 5%, gradient cosine > 0.9 | 7–10 | Mac (dev) |
| 4. Diffusion + benchmark | 16–21 | **Gate 4**: benchmark tables for all 3 cases at dev tier; smoke e2e passes on CPU, MPS and CUDA | 11–14 | Mac (dev, overnight for cases 2–3); cloud for full tier |

The plan-doc day counts are the original plan's; they assume no blocking issues. Phase 4's cases 2–3 depend on dataset v2 and PINN v2, which are the longest compute jobs ([compute_budget.md](../06_training/compute_budget.md)).

## Gate details

| Gate | Spec | Command | Evidence saved |
|---|---|---|---|
| 1 | [validation.md](../03_lbm/validation.md) | `01_validate_lbm_cylinder.py --tier dev --device mps` | `runs/validate_cylinder/dev/results.json`, `table1.md`, figures |
| 2a | [validation.md](../03_lbm/validation.md) | `01_validate_lbm_cylinder.py --adjoint-check --tier dev --device cpu` | `runs/validate_adjoint/results.json`, Fig 5.1 analogue |
| 2b | [dataset_generation.md](../06_training/dataset_generation.md) | `02_generate_lbm_dataset.py --set v1 --tier dev` + checker | `data/lbm_anchors/dev/merged.pt`, check report |
| 3 | [pinn_training.md](../06_training/pinn_training.md) | `03_train_transformer_pinn.py --tier dev` (both models) | `runs/pinn_*_v1_dev/results.json` |
| 4 | [four_way_benchmark.md](../08_benchmark/four_way_benchmark.md), [e2e_mac_mini.md](../07_testing/e2e_mac_mini.md) | `05_run_4way_benchmark.py --tier dev`, `00_smoke_e2e.py`, `cuda_check.sh` | `runs/benchmark/dev/results.md`, smoke summaries per device |

## Rules

- The orchestrator runs every gate itself and pastes the key numbers into `Plan/README.md`'s status table.
- A miss is reported with numbers and the debugging done; the user decides whether to accept it, fix, or move the gate to full tier on cloud.
- The smoke e2e run is extended and rerun at the end of every phase, so the pipeline is always runnable end to end.

## Portfolio deliverables (from the original plan §6)

1. Table 1 reproduction with Re 100 vortex street and force histories (Phase 1).
2. Adjoint sensitivity vs Transformer attention side by side (Phases 2–3, plotted in Phase 4).
3. 4-way speed and accuracy benchmark (Phase 4).
4. Shape-morphing GIFs (Phase 4).
