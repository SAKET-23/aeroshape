---
title: "PINN surrogates must be trained on the optimizer's start shapes and kept inside their training range"
author: SAKET-23
date: 2026-11-18
branch: feat/wp-00-scaffold
cost: { minutes: 257, tokens: 1522182 }
touches:
  - docs/SSL (1).pdf
  - ".dockerignore"
  - ".python-version"
  - CLAUDE.md
  - Dockerfile
  - Plan/01_system_design/architecture.md
  - Plan/01_system_design/config_and_run_tiers.md
  - Plan/01_system_design/data_contracts.md
  - Plan/01_system_design/device_and_precision.md
  - Plan/01_system_design/repo_layout.md
---

## Decided
- The PINN training set must cover the shapes the optimizer starts from. The circle start sat outside the v1b training range, and the Transformer's gradient cosine along the circle -> NACA0012 path fell to 0.04–0.4, with CD −82% at cycle 19. Adding about 30 circle/ellipse/NACA blends and thick round shapes (v1c) fixed it: cosine 0.80–0.998, CD within 3.2%.
- PINN-driven optimization clamps each step to the per-coordinate training box (G37). Without it, the MLP reached only 23.8% drag reduction instead of 35.8%, and the Transformer violated thickness.
- The test set is never used for any choice. The old "holdout" is validation.

## Tried and ruled out
- Training longer or adding gradient supervision alone did not fix the path gradients; the problem was coverage.

## Watch out
- Even inside the box, the surrogates still exploit their own error: the Transformer reports CD −0.71 where LBM gives 1.49. The axis-aligned box contains control-point combinations no training shape has. Always LBM re-score the final shapes.
- Never train on adjoint-trajectory shapes, or the path check stops being a fair test.

In the code: `scripts/03b_pinn_checks.py` (`gen-v1c`, `path-check`), `train_box()` in `src/aeroshape/optimization/kkt_optimizer.py`, `optimizer.box: pinn_train`, decisions G33, G34 and G37 in `Plan/raw/gaps_and_decisions.md`.
