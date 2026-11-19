---
title: PINN v2 at Re 1000 fits the fields but its drag gradient barely matches the time-averaged adjoint
author: SAKET-23
date: 2026-11-19
branch: feat/wp-00-scaffold
cost: { minutes: 31, tokens: 117465 }
touches:
  - Plan/09_execution/risks.md
  - Plan/README.md
  - Plan/raw/gaps_and_decisions.md
  - scripts/02_generate_lbm_dataset.py
  - scripts/03_train_transformer_pinn.py
  - src/aeroshape/pinn/navier_stokes_loss.py
  - src/aeroshape/visualization/plots.py
  - tests/scripts/test_benchmark.py
  - tests/scripts/test_train_pinn.py
  - tests/visualization/test_plots.py
---

## Decided
- Keep both v2 PINNs (registered as mlp_v2 / transformer_v2) for Cases 2-3, always behind the training-box envelope and with LBM re-scoring of every final shape. Report the low gradient cosine next to their benchmark rows.
- Fix coverage first (v2b set of round/ellipse <-> RAE/NACA cambered blends at Re 1000), then retrain both v2 models.

## Tried and ruled out
- Gradient supervision on the time-averaged rows: one time-averaged adjoint at N=48 costs 1-1.5 h on CPU, so there is no adjoint cache for training; w_grad applies only to the steady Re 20 rows.

## Watch out
- v2 anchors were first written with kind=0, which silently disabled the per-kind loss weights. Check `kind` on any regenerated v2 data.
- The Case 2 start (circle) lies outside the v2 train range on 10 of 12 free entries, so surrogate optimizers start out of distribution.
- The cosine reference is only 3 runs and is itself the report's time-averaged approximation, so treat 0.26/0.36 as a weak signal, not a verdict.

In the code: scripts/03c_pinn_v2_eval.py (eval + --select), src/aeroshape/pinn/mixing.py, runs/adjoint_cache/holdout_v2_ta.pt. Numbers: Transformer v2 holdout velocity 3.6%, CD err 10.4%/6.0% (AoA 0/10), cosine 0.26; MLP v2 velocity 5.1%, CD 12.6%/11.4%, cosine 0.36. Decision row G39 in Plan/raw/gaps_and_decisions.md.
