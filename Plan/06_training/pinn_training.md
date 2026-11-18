# `scripts/03_train_transformer_pinn.py` — PINN training (Gate 3)

**Work package:** WP 15. **Depends on:** WP 10, 12, 13, 14. **Loss:** [ns_loss_and_sampling.md](../04_models/ns_loss_and_sampling.md).

## Command

```bash
uv run python scripts/03_train_transformer_pinn.py --model transformer --tier dev --device mps --set pinn.version=v1
uv run python scripts/03_train_transformer_pinn.py --model mlp         --tier dev --device mps --set pinn.version=v1
```

Both models are trained with identical data, sampling, loss, schedule and step counts.

## Schedule

| Stage | Steps (dev) | Optimizer | Notes |
|---|---|---|---|
| Warm-up | 0–500 | AdamW `lr` linear warm-up 0 → 1e-3 over 200 steps | `w_pde = 0` |
| PDE ramp | 500–1500 | AdamW | `w_pde` 0 → 1 linearly |
| Main | 1500–5000 | AdamW, cosine decay to 1e-5 | weight decay 1e-4, grad-clip 1.0 |
| Refine | 500 | L-BFGS (strong Wolfe, history 50, `lr = 1`) | fixed batch: 8 shapes × all sets, resampled every 50 L-BFGS steps |

Smoke: 100 AdamW (warm-up 20, ramp 30) + 5 L-BFGS, 2 shapes, `d_model = 64`.

## Evaluation (every 250 steps and at the end)

On the 5 holdout shapes (v1: Re 20 conditions):
- Velocity relative L2: `‖(u, v) − (u, v)_LBM‖ / ‖(u, v)_LBM‖` over all anchor points.
- Pressure relative L2 (after removing the mean offset).
- CD and CL absolute and relative error vs. LBM.
- Gradient agreement: cosine similarity of `∇_alpha CD` (PINN) vs the adjoint gradient, on 3 holdout shapes. The adjoint gradients are precomputed once by `scripts/03_train_transformer_pinn.py --precompute-adjoint` and cached.
- Residual mean `|r|` on fresh collocation points.

`best.pt` = lowest holdout velocity L2.

## Gate 3 (v1, dev tier)

| Metric | Pass |
|---|---|
| Holdout velocity relative L2 | < 2% (Transformer) |
| Holdout CD error | < 5% |
| `cos(∇CD_PINN, ∇CD_adjoint)` | > 0.9 (mean over 3 holdout shapes) |
| Inference latency, one shape, CD + gradient | measured and reported per device |
| Speed-up vs one forward LBM solve | measured and reported |

The MLP baseline is evaluated with the same metrics; it has no gate (it is the comparison).

If the Transformer misses the 2% target: (1) increase anchor weight and points near the wall; (2) train longer at full tier on cloud; (3) report the achieved number and continue — Phase 4 does not need 2%, it needs a useful gradient (the cosine metric).

## v2 (Re 1000, time-averaged)

Same script with `--set pinn.version=v2` and the v2 anchors. Metrics reported for AoA 0 and 10 separately; no gate. Used by Test Cases 2–3 PINN evaluators and diffusion guidance at Re 1000.

## Logging and outputs

`runs/pinn_<model>_<version>_<tier>/`: `log.csv` (step, wall, lr, each loss term, eval metrics), `ckpt/`, `fig/` (loss curves, holdout field comparison: PINN vs LBM vs error for `u`, `v`, `p`), `results.json`. Optional TensorBoard (`--tb`).

## Resume

`--resume` loads the latest checkpoint including optimizer and generator state and continues the step count; the cosine schedule continues where it stopped.

## Tuning notes from implementation

- WP 12: `fourier_scale = 2.0` (spec default) was too noisy for the PDE term on the Kovasznay test; `0.25` reached < 1% error (with `psi_scale = 1.0`, lr 3e-3 cosine, data weight ×100). The airfoil window is larger (5.5 × 3 chords), so sweep `pinn.fourier_scale ∈ {0.25, 0.5, 1.0, 2.0}` at smoke/dev size before the full dev run.

- WP 13: Transformer (d = 128) on MPS: one `hybrid_loss` step is ~1.3 s at 600 collocation points per shape and ~1.9 s at 1,000 (8 shapes); at the spec's 4,000 it did not finish in 100 s (likely a memory cliff in the third-derivative graph). Dev tier is therefore `n_collocation = 1000`, and the training script must support `train.micro_batch_shapes` (gradient accumulation) so larger point counts can be used without the cliff. Full tier keeps 8,000 points on CUDA. At ~2 s/step, 5,500 steps ≈ 3 h on the Mac.
- `attn_impl` default is `explicit`: SDPA's double backward is not implemented on CPU (works on MPS).

## Pitfalls

- L-BFGS on MPS: run it with full-batch closures; if a line-search op falls back to CPU and is slow, run Stage B on CPU (small model) — it is 500 steps.
- Keep `create_graph=True` only inside the loss; evaluation runs third derivatives only for the residual metric.
- Holdout leakage: the holdout shapes must never appear in `sample_batch` (assert on shape ids).

## Results (WP 15, dev tier, MPS) — Gate 3 passed with a recorded deviation

Final settings (G29–G32): `fourier_scale` 0.25 (sweep: 6.4 / 7.8 / 15.1 / 88% velocity L2 at 0.25 / 0.5 / 1 / 2), near-wall anchor oversampling `near_frac` 0.5, force supervision `w_cd` 1.0, adjoint-gradient supervision `w_grad` 0.3, dataset v1 + v1b (75 train runs, 40 of them thin shapes t = 0.08–0.35), 5,000 Adam + 500 L-BFGS.

| Validation (the 5 'holdout' shapes, Re 20; used for tuning, see note) | MLP | Transformer |
|---|---|---|
| Velocity L2 (target < 2%) | 2.89% (per shape 2.2–3.7%) | 2.008% (1.8–2.2%) |
| Pressure L2 | 19.7% | 17.2% |
| CD error (target < 5%) | 1.38% (max 1.8%) | 1.05% (max 2.4%) |
| ∇CD cosine vs adjoint, free entries (target > 0.9) | 0.948 (min 0.747) | 0.955 (min 0.889) |
| Latency, CD + gradient (MPS) | 16 ms | 65 ms |
| Throughput, 64 shapes | 503 shapes/s | 104 shapes/s |
| Training wall-clock | 44 min | 3.9 h (L-BFGS ≈ 1.5 h of it, for 2.19 → 2.01% velocity) |

What it took (each step is recorded in gaps_and_decisions): plain hybrid loss gave velocity 1.1% but CD ≈ 5% of LBM (flattened wall pressure, G29); force supervision fixed CD but the cosine was 0.26–0.42 (G30); gradient supervision fixed the cosine except on thin shapes, which were sparse in the data (G31); thin-shape data plus the full step budget gave the table above (G32).

**Review 3 caveats.** The 5 holdout shapes were used to choose the Fourier scale, the v1b thickness window, the loss weights and `best.pt`, so they act as a validation set and the table above is optimistic. A clean 10-shape test set (new seed, t = 0.08–0.80) is being generated, and Gate 3 is re-reported on it. The MLP's best-of-trajectory selection is noisy (L-BFGS iterates 2.9–5.4%; Adam-end 3.44%). CD agrees to about 1% while the pressure-field L2 is 17–20%: the claim is "integrated CD and ∇CD match", not "the field is accurate". Case 1 starts from a circle (t = 1.01) outside the training range (t ≤ 0.80), so the PINN's early-cycle gradients are being checked along the adjoint optimization path.

