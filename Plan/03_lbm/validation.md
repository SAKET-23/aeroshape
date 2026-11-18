# LBM validation — Gate 1 and Gate 2a

**Work packages:** WP 6 (Gate 1), WP 8 (Gate 2a). **Script:** `scripts/01_validate_lbm_cylinder.py`. **Report:** ch. 4.4 Table 1, ch. 5.4 Fig 5.1.

## Gate 1: cylinder drag (report Table 1)

> **Revised after WP 6 (decision G26).** The spec's 20D × 10D domain gives CD = 2.415 at Re 20 (+18%) because of blockage (periodic 10D height) and the short run-in; CD falls with domain size (2.415 → 2.170 → 2.101 → 2.058 for 20×10 → 40×20 → 60×30 → 100×50) and is grid-converged (2.433 / 2.415 / 2.411 at N = 10 / 20 / 40, order 2.1). A control-volume momentum balance agrees with momentum exchange to 0.01%. Gate 1 therefore runs in larger domains and compares Re 40 with the Navier–Stokes value 1.550 (Nieuwstadt and Keller, listed in the report), not Tritton's experimental 1.65, which a converged solver does not reproduce.
>
> | Re | Domain, offset | N | Target | WP 6 result | Pass |
> |---|---|---|---|---|---|
> | 20 | 100 × 50, 25 | 20 | 2.04 | 2.0576 (+0.86%) | yes |
> | 40 | 100 × 50, 25 | 20 | 1.550 | 1.5355 (−0.94%) | yes |
> | 100 | 40 × 20, 10 | 32 | 1.365 | 1.3840 (+1.39%), St 0.169, CL rms 0.237 | yes |
>
> The table below is the original spec, kept for reference.

The cylinder uses an **analytic circle** polyline (4,096 points), not the NURBS shape, so this gate tests the solver alone.

| Re | Report CD | Literature | Collision | Outlet | Mode | Pass |
|---|---|---|---|---|---|---|
| 20 | 2.04 | Dennis and Chang 2.045 | BGK | fixed | steady | `|CD − 2.04| / 2.04 < 2%` |
| 40 | 1.65 | Tritton 1.65; Nieuwstadt and Keller 1.550 | BGK | fixed | steady | within 2% of 1.65 |
| 100 | 1.365 (mean) | Hejranfar and Ezzatneshan 1.336 | TRT | extrapolate + sponge | unsteady | mean within 2% of 1.365; `|mean CL| < 0.01`; periodic shedding detected |

Additional diagnostics (reported, not gated):
- Re 100 Strouhal number `St = f D / U` (literature about 0.16–0.17).
- Grid convergence at Re 20: N ∈ {10, 20, 40} → CD vs `1/N` plot; observed order printed.
- Blockage: 10D height gives about 10% blockage, which raises CD a few percent vs. unbounded flow. This is why the report's values sit above some literature values; the gate compares with the report's own numbers.

**Tier policy.** Dev tier (N = 20) is expected to pass at Re 20 and 40. If Re 100 misses at dev tier, Gate 1 may be passed at full tier on cloud CUDA (N = 40); the dev-tier number is still reported. If Re 20 misses at N = 40, stop and debug before Phase 2 — the adjoint and dataset depend on it.

### Outputs (`runs/validate_cylinder/<tier>/`)

- `results.json`: per Re — CD, CL mean, CL RMS, St, steps, wall-clock, device, pass/fail.
- `fig/cd_cl_history_re{20,40,100}.png` (report Figs 4.10–4.12).
- `fig/vorticity_re100_t{30,35,60}.png` (report Fig 4.8).
- `fig/grid_convergence_re20.png`.
- `table1.md`: the report table with this solver's numbers added as a column.

### Command

```bash
uv run python scripts/01_validate_lbm_cylinder.py --tier dev --device mps
uv run python scripts/01_validate_lbm_cylinder.py --tier full --device cuda     # cloud
```

## Gate 2a: adjoint gradient (report Fig 5.1)

| Setting | Value |
|---|---|
| Shape | NURBS circle fit, symmetric, **k = 41** |
| Re, N | 20, 20 (dev) |
| Cost | CD (steady) |
| Precision | CPU float64 |
| FD step | `Δ = 1e-4` chord, topology checked unchanged per control point |
| Metric | `‖g_adj − g_fd‖₂ / ‖g_fd‖₂` over control points with unchanged topology |
| Pass | < 2e-3 |

Outputs: `runs/validate_adjoint/fig/adjoint_vs_fd.png` (bar plot per control point, like Fig 5.1), `fig/adjoint_sensitivity_field.png` (`|F*|` magnitude, like Fig 6.3), `results.json` (error, skipped indices, wall-clock for adjoint vs. 82 FD solves).

Command: `uv run python scripts/01_validate_lbm_cylinder.py --adjoint-check --tier dev --device cpu`.

### Result (WP 8) — **passed**

| Item | Value |
|---|---|
| Relative L2 error | **3.21e-4** (threshold 2e-3) over 32 of 39 free control points |
| Excluded (topology changes at ±Δ) | 9, 12, 13, 20, 21, 28, 29 |
| Base CD | 2.3957 (20 × 10, N = 20, Re 20, BGK, CPU float64, steady tol 1e-13, up to 300 convective times) |
| FD noise estimate | 3.7e-4 (5 largest points re-differenced at Δ/2; includes O(Δ²) truncation, so an upper bound) |
| Wall-clock | adjoint 1,213 s (forward 441 + adjoint 772, CD and CL adjoints) vs 64 FD solves 13,506 s (11×) |

On the 32 included points the error is at the FD noise level for the large-gradient points (small-gradient points show up to ~5% relative error, e.g. point 32, which contributes little to the L2). **Coverage (Review 2, M1/M2):** the 7 excluded points hold about 19% of ‖g_adj‖. They were then checked with a one-sided FD (side −1, Δ = 1e-4, topology unchanged including the q-clamp state, all solves converged): rel L2 **2.99e-4** (points 20/21: 0.121332 vs 0.121328, 0.108030 vs 0.108033). All 39 free points are now verified (`--excluded-onesided`, `excluded_onesided` in results.json). Figures are in `runs/validate_adjoint/dev/fig/`.

## If a gate fails — debugging order

1. Mass and momentum conservation tests (solver).
2. Poiseuille and q-accuracy tests (IBB).
3. Force on a straight channel wall vs. analytic shear stress.
4. Grid convergence trend (if CD is converging to the wrong value, suspect the force formula; if not converging, suspect boundaries).
5. For Gate 2a: adjoint validation ladder levels 1–4 in [adjoint_solver.md](adjoint_solver.md) in order.
