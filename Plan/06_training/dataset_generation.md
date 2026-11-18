# `scripts/02_generate_lbm_dataset.py` — LBM anchor dataset (Gate 2b)

**Work package:** WP 10. **Depends on:** WP 6 (Gate 1), WP 2. **Contract:** [data_contracts.md](../01_system_design/data_contracts.md) anchor file.

## What it produces

`data/lbm_anchors/<tier>/<shape_id>_<re>_<aoa>.pt` per run, then `merged.pt`. Each holds the shape, condition, steady (or time-averaged) `u, v, p` at fixed non-dimensional points, and `CD, CL`.

## Sets

| Set | Shapes | Conditions | Mode | Runs | Used by |
|---|---|---|---|---|---|
| v1 | 30 LHS shapes, symmetric (k = 8) | Re 20 AoA 0; Re 40 AoA 0 for 10 of them | steady | 40 | PINN v1, Test Case 1, Gate 3 |
| v2 | 30 LHS shapes, cambered (k from WP 2) | Re 1000 at AoA 0 and 10 | time-averaged | 60 | PINN v2, Test Cases 2–3 |

Holdout: last 5 shapes in LHS order of each set (all their conditions).

## Per-run procedure

1. `alpha` → `Boundary` (rotated by AoA) → `map_to_grid`.
2. Solve: steady (Re ≤ 40, BGK, outlet fixed) or unsteady (Re 1000, TRT, outlet extrapolate + sponge, `transient = 30` convective times, average over 4 periods).
3. Convert to non-dimensional (derivations 2.9).
4. Points: all fluid nodes inside `window_d`, subsampled uniformly to `points_per_shape` with 30% of the budget reserved for nodes within 0.2 chord of the wall; plus `edge_points` on the window boundary (bilinear interpolation of the lattice field at exact edge coordinates).
5. Save with `meta` (lattice params, steps, wall-clock, converged flag, period).

## Resumability and parallelism

- Skips runs whose output file exists and has `meta.converged = True`.
- `--shard i/n` processes every n-th run, so two processes (e.g. CPU + MPS on the Mac, or two cloud GPUs) can share the work.
- Each run writes atomically (`tmp` then rename).

## Gate 2b

| Check | Pass |
|---|---|
| `merged.pt` loads, keys and shapes match the contract | yes |
| No NaN / Inf | yes |
| All v1 runs converged | yes |
| Pipeline sanity (revised after WP 10) | NURBS circle fit run through the dataset pipeline (20×10, N = 20, Re 20) has CD within 2% of 2.415 (the 20×10 cylinder value, G26). The original check compared the closest LHS shape, which sits 0.053 RMS from the circle, so it measured shape distance, not the pipeline |
| Holdout covers the space | holdout CD range overlaps ≥ 60% of the training CD range |

### Results (WP 10, set v1, dev tier, MPS)

| Check | Result |
|---|---|
| Loads, keys and shapes | pass: `alpha (40, 8)`, `xy (40, 22000, 2)`, `u, v, p, edge (40, 22000)`, `cd, cl, re, aoa, split, kind (40,)`; `check_anchors` returns no issues |
| No NaN / Inf | pass |
| All runs converged | pass: 40 of 40, 3,650–9,650 steps, 6–35 s each, 14 min total; `merged.pt` 18.5 MB |
| Pipeline sanity | pass: `--check` (fixed in Review 2, M8) gives CD 2.4120 vs 2.415 (−0.12%), CPU float64, converged in 8,150 steps. In float32 on CPU the CD noise (~2e-5) exceeds `steady_tol` 1e-6, so the check uses float64 on CPU |
| Holdout coverage | pass: Re 20 CD train 1.087–2.056, holdout 1.175–2.119 (91% overlap) |

Notes:
- All 5 holdout runs are at Re 20, so Gate 3 is evaluated at Re 20.
- About 6.5k lattice nodes fall in the window at N = 20; the other saved points are bilinear-interpolated.
- Blockage (G27): NACA0012 at Re 20 gives CD 1.1266 at 20×10 and 1.0935 at 40×20 (+3.0%). Optimization and datasets keep 20×10.
- Config overrides use `-o key=val`. Set v2 (Re 1000, TRT Λ = 0.001, perturb 0.1) is unblocked.

## Cost (estimates; replaced by WP 4 measurements)

| Set | Tier | Per run | Total | Where |
|---|---|---|---|---|
| v1 | dev (80k nodes, ~15 conv. times) | ≈ 0.5–1 min | < 1 h | Mac MPS |
| v2 | dev (461k nodes, ~50 conv. times) | ≈ 5–15 min | 5–15 h | Mac overnight, or split with cloud |
| v2 | full (2M nodes) | ≈ 30–60 min | cloud only | CUDA |

## Command

```bash
uv run python scripts/02_generate_lbm_dataset.py --tier dev --set v1 --device mps
uv run python scripts/02_generate_lbm_dataset.py --tier dev --set v2 --device mps --shard 0/2   # + 1/2 elsewhere
uv run python scripts/02_generate_lbm_dataset.py --merge --tier dev
```

## Results (dataset v2, dev tier, 2026-11-18)

- 60 runs (30 cambered shapes × AoA 0/10, Re 1000, MPS float32) in `data/lbm_anchors/dev_v2/merged.pt` (27.9 MB). `--check` passes.
- 53 runs `period_mean`. 7 runs found no period on the first pass and were redone with `--redo-mode last_state`; all 7 came out `window_mean` (20 convective times, 66,480 steps, 857–1,144 s each).
- Of the 7 window-mean runs:
  - The four thick ones (s01, s03, s17, s20 at AoA 0) shed irregularly, with CL about ±0.4 (s20 about ±1.3). CL is positive about half the time, so there is no bistability. Their window CDs (0.47, 0.47, 0.44, 1.05) sit with thickness neighbours. Their window-mean CL carries noise of ±0.05–0.3 because 20 T is short for irregular shedding.
  - The thin ones (s21 and s29 at AoA 0, s29 at AoA 10) are essentially steady, with CL constant to ±0.01.
- `cl_hist` is saved only for the 7 redone runs.
