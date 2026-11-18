# Compute budget

**Status: estimates.** WP 4 measures LBM throughput on the Mac mini (MPS and CPU) and WP 13 measures PINN step time; this file is then updated with measured numbers.

## Assumptions

| Item | Value |
|---|---|
| Mac mini | Apple M4 Pro, 12 cores, 24 GB unified memory |
| LBM throughput (MPS, float32, 400 × 200) | 50–150 MLUPS (million lattice-node updates per second) |
| LBM throughput (CUDA) | T4 ~150–300, L4 ~300–600, A100 ~1,000+ MLUPS |
| Adjoint step cost | 1.5× forward (hand), 2–3× forward (VJP) |
| PINN train step (dev, Transformer, MPS) | 0.3–0.8 s |

Steps for a run = `convective_times × N / U_lb`. Work = nodes × steps.

## Estimates

| Job | Size | Mac mini (MPS) | Cloud CUDA (L4) |
|---|---|---|---|
| Gate 1 dev: Re 20 + 40 + 100 | 80k nodes, ~40 + 40 + 100 conv. times | 2–6 min | < 1 min |
| Gate 1 full | 320k nodes | 10–30 min | 2–5 min |
| Gate 2a (82 FD solves + 1 adjoint, CPU float64, 80k) | ~82 × 15 conv. times | 30–90 min (CPU) | n/a (CPU check) |
| Test Case 1, adjoint-LBM, 20 cycles, dev | 80k × 60 conv. times × 20 | 10–30 min | 2–5 min |
| Test Cases 2–3, adjoint-LBM, 15 cycles, dev | 461k × 90 conv. times × 15 | 6–20 h | 1–3 h |
| Dataset v1 (40 runs, dev) | 80k × ~15 conv. times | 15–45 min | < 10 min |
| Dataset v2 (60 runs, dev) | 461k × ~50 conv. times | 5–15 h | 1–3 h |
| Dataset v2 (full) | 2M × ~50 conv. times | not planned | 10–25 h |
| PINN v1 training, dev (5,500 steps) | | 1–2 h | 20–40 min |
| PINN v2 training, dev | | 2–4 h | 0.5–1 h |
| Diffusion training, dev | | 5–20 min | < 5 min |
| Benchmark, dev (3 cases × 4 methods + LBM re-scoring) | dominated by adjoint-LBM on cases 2–3 | 8–24 h | 2–4 h |

## Plan for where things run

- **Mac mini**: everything at smoke and dev tier for Phase 1–3 and Test Case 1 end to end. Cases 2–3 with adjoint-LBM at dev tier run overnight.
- **Cloud CUDA**: Gate 1 at full tier (if dev misses 2% at Re 100), dataset v2 (dev or full), PINN v2 training, the full-tier benchmark.
- Cloud spend control: every cloud job is resumable and writes outputs every few minutes; notebooks download `runs/` and `data/` artefacts at the end.

## Memory

- Largest grid (2000 × 1000): `f` is 9 × 2M × 4 B = 72 MB; solver keeps ~4 copies (f, f_col, feq, streamed) → ~300 MB. Time averaging adds one more copy.
- Adjoint VJP keeps one step's graph: ~10 copies → < 1 GB at full tier.
- PINN dev batch (8 × 5,600 points, third derivatives, d = 128): estimated 2–6 GB. If MPS memory pressure appears, halve shapes per step and accumulate gradients over 2 micro-batches.
- No unsteady history storage anywhere (G8).
