# `scripts/04_train_diffusion_model.py` — diffusion training

**Work package:** WP 17. **Depends on:** WP 15 (trained PINN), WP 16. **Model:** [diffusion.md](../04_models/diffusion.md). **Decision:** G13.

## Data

1. Shapes: `lhs_shapes(n = diffusion.n_shapes, family = "diffusion")` (2,000 at dev), validity-filtered; one set per mode (symmetric k = 8, cambered).
2. Conditions per shape: Re and AoA sampled from the case grid (symmetric: Re ∈ {20, 40}, AoA 0; cambered: Re 1000, AoA ∈ [0°, 12°]).
3. Labels: `CD, CL` from the frozen Transformer PINN (`surface_forces`, batched 256 shapes per call). Minutes on MPS.
4. **LBM cross-check**: 20 random shapes labelled by `LBMScoreEvaluator` at dev tier; report label error (mean and max). If mean CD error > 10%, the guidance targets will be biased — noted in results, and the benchmark uses LBM re-scoring anyway.
5. Saved as `data/diffusion_shapes/<tier>_<mode>.pt`: `alpha [S, D]`, `cond [S, 4]`, `label_source`, `pinn_ckpt` hash.

## Training

| Setting | dev |
|---|---|
| Steps | 20,000 |
| Batch | 256 |
| Optimizer | AdamW, `lr = 2e-4`, cosine to 1e-6, weight decay 0 |
| EMA | 0.999 (sampling uses EMA weights) |
| Condition dropout | 10% |
| Loss | MSE on `eps`, uniform `t` |

Smoke: 200 shapes, 300 steps. Expected time on MPS (dev): minutes — the model and data are small.

## Evaluation (end of training)

- Unguided samples (64): validity rate, smoothness (max second difference of control points), nearest-training-shape distance (memorization check: median > 0.005 chord).
- Conditioned samples for 8 held-out condition targets: PINN-measured `|CD − CD_ref|`, `|CL − CL_ref|`, with and without guidance.
- Guidance sweep over `guide_scale ∈ {0, 0.3, 1, 3}` and `cfg_scale ∈ {0, 1, 2, 4}`: table of mean `J` and validity → the best pair is written into the case configs for the benchmark.

## Acceptance (WP 17)

- Validity ≥ 95% unguided, ≥ 90% guided.
- Guided mean `J` (including the thickness penalty) lower than unguided by ≥ 30% on held-out targets with an active thickness floor and on targets beyond the data range, at the same cfg (G35).
- Memorization: median sample-to-train distance ≥ 0.5 × median train-to-train nearest distance (G35).
- Outputs: `runs/diffusion_<mode>_<tier>/` with `results.json`, `fig/samples.png`, `fig/guidance_sweep.png`, denoising trajectory GIF for one target.
