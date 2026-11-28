# Plan — Differentiable Aerodynamic Shape Optimization

The full specification for building the project: a GPU D2Q9 lattice Boltzmann solver with a discrete adjoint, two physics-informed neural network (PINN) surrogates, and a physics-guided diffusion model, benchmarked against each other on three airfoil optimization cases from the original report.

- Runs first on the Mac mini (Apple M4 Pro, Metal/MPS), and on NVIDIA CUDA (cloud or Colab) from the same code.
- Built by an **Opus 5.5 orchestrator** dispatching **Sonnet implementers**, one work package at a time ([agent_orchestration.md](09_execution/agent_orchestration.md)).
- Sources: the [original project plan](raw/original_project_plan.md) and the 63-page report `docs/SSL (1).pdf` (notes in [report_notes.md](raw/report_notes.md)). Where they disagree or leave gaps, [gaps_and_decisions.md](raw/gaps_and_decisions.md) decides.

## Reading order

1. [raw/gaps_and_decisions.md](raw/gaps_and_decisions.md) — what changed from the original plan and why.
2. [01_system_design/architecture.md](01_system_design/architecture.md) — modules and data flow.
3. [01_system_design/data_contracts.md](01_system_design/data_contracts.md) — every shared type and file format.
4. The module doc for the work package at hand.
5. [09_execution/work_packages.md](09_execution/work_packages.md) — what to build next.

## Contents

| Folder | Files |
|---|---|
| `raw/` | [original_project_plan](raw/original_project_plan.md) · [report_notes](raw/report_notes.md) · [derivations](raw/derivations.md) · [gaps_and_decisions](raw/gaps_and_decisions.md) |
| `01_system_design/` | [architecture](01_system_design/architecture.md) · [repo_layout](01_system_design/repo_layout.md) · [device_and_precision](01_system_design/device_and_precision.md) · [config_and_run_tiers](01_system_design/config_and_run_tiers.md) · [data_contracts](01_system_design/data_contracts.md) |
| `02_geometry/` | [nurbs](02_geometry/nurbs.md) · [reference_shapes](02_geometry/reference_shapes.md) · [grid_mapping_ibb](02_geometry/grid_mapping_ibb.md) |
| `03_lbm/` | [forward_solver](03_lbm/forward_solver.md) · [forces](03_lbm/forces.md) · [adjoint_solver](03_lbm/adjoint_solver.md) · [validation](03_lbm/validation.md) |
| `04_models/` | [mlp_pinn](04_models/mlp_pinn.md) · [transformer_pinn](04_models/transformer_pinn.md) · [ns_loss_and_sampling](04_models/ns_loss_and_sampling.md) · [stress_integrator](04_models/stress_integrator.md) · [diffusion](04_models/diffusion.md) |
| `05_optimization/` | [kkt_optimizer](05_optimization/kkt_optimizer.md) |
| `06_training/` | [dataset_generation](06_training/dataset_generation.md) · [pinn_training](06_training/pinn_training.md) · [diffusion_training](06_training/diffusion_training.md) · [compute_budget](06_training/compute_budget.md) |
| `07_testing/` | [test_strategy](07_testing/test_strategy.md) · [e2e_mac_mini](07_testing/e2e_mac_mini.md) · [cuda_cloud_runbook](07_testing/cuda_cloud_runbook.md) |
| `08_benchmark/` | [four_way_benchmark](08_benchmark/four_way_benchmark.md) · [visualization](08_benchmark/visualization.md) |
| `09_execution/` | [agent_orchestration](09_execution/agent_orchestration.md) · [work_packages](09_execution/work_packages.md) · [milestones_and_gates](09_execution/milestones_and_gates.md) · [risks](09_execution/risks.md) |

## Status

Updated by the orchestrator after each work package and gate.

| Phase | Work packages | Gate | Status |
|---|---|---|---|
| 1. Geometry + LBM | 0–6 | Gate 1 — cylinder CD within 2% | **passed** (WP 6, revised domains/targets G26): Re 20 +0.86%, Re 40 −0.94% vs 1.550, Re 100 +1.39% |
| 2. Adjoint + dataset | 7–10 | Gate 2a — adjoint vs FD < 2e-3; Gate 2b — dataset | WP 7, 9, 10 done. **Gate 2b passed** (set v1, 40 runs; revised pipeline check −0.3%). **Gate 2a passed** (k = 41, CPU float64: rel L2 3.21e-4 on 32 points + 2.99e-4 one-sided on the 7 topology-change points). Review 2 fixed (M1–M9). Phase 2 complete |
| 3. PINN surrogates | 11–15 | Gate 3 — velocity L2 < 2%, CD < 5%, gradient cosine > 0.9 | **passed on the clean 10-shape test set (Transformer):** Transformer v1c (G33/G34): velocity 1.47%, CD 1.33%, free cosine 0.970; Case 1 path fixed (cosine ≥ 0.53, mostly ≥ 0.8; CD within 3.2%). MLP stays v1b: 2.29% / 1.32% / 0.949 (velocity deviation). Weak spot: t ≈ 0.08 shapes |
| 4. Diffusion + benchmark | 16–22 | Gate 4 — benchmark tables; smoke e2e on CPU, MPS, CUDA | WP 16 done; WP 17 done for symmetric Re 20 (G35/G36: sampler guidance-gradient bug fixed; guided −69% J beyond the data range at validity 1.0; in-range, conditioning alone is best; cambered/Re 1000 waits for PINN v2); WP 18 Case 1 done at dev and re-run to convergence on A40 (`dev_c1v2`, G41: LBM 52.9% `cd_stalled` at 51 cycles, diffusion 51.0% in 4.2 s, MLP 35.8%, Transformer 32.0%, LBM re-scored; G37 envelope); Case 3 LBM adjoint done (G42: −20% CD at target CL, best feasible cycle 14), its PINN/diffusion rows pending; Case 2 adjoint gradient unusable on the blob (cosine −0.13, G42), `fd_lbm` run pending; v2b coverage set and retrain done (G40: v2b selected). PINN v2 trained (G39: Transformer holdout velocity 3.6%, MLP 5.1%; CD-gradient cosine vs the time-averaged adjoint 0.26 / 0.36 on 3 runs). WP 20 smoke e2e passes on CPU (8 stages, ~450–745 s; MPS run pending); WP 22 CPU image builds and passes tests in-container (CUDA image unbuilt, cloud only); dataset v2 done (60 Re 1000 runs: 53 period-mean, 7 window-mean, `--check` passes; R14 checked: BGK ≈ TRT); WP 19 figures done (`scripts/07_make_figures.py` → `runs/figures/`) |

## Glossary

| Term | Meaning |
|---|---|
| LBM, D2Q9 | lattice Boltzmann method on a 2D lattice with 9 velocities |
| BGK, TRT | single- and two-relaxation-time collision operators |
| IBB | interpolated bounce-back (Bouzidi): curved-wall boundary using the wall fraction `q` |
| Adjoint | backward-in-time solve giving the gradient of a cost w.r.t. all design variables at once |
| PINN | physics-informed neural network: trained on PDE residuals plus data |
| `alpha` | the shape vector: y-values of the NURBS control points |
| `D` | design dimension: `k` (symmetric) or `2k` (cambered) |
| Anchors | LBM fields saved at fixed points, used as PINN training data |
| MPS | PyTorch's Metal backend for Apple GPUs |
| Tier | run size: smoke (minutes), dev (Mac, hours), full (cloud) |
| Gate | a pass/fail check that must pass before the next phase starts |
| MLUPS | million lattice-node updates per second |
