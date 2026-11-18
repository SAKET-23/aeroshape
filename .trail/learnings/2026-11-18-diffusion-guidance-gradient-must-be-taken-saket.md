---
title: Diffusion guidance gradient must be taken with respect to the clean estimate it is applied to
author: SAKET-23
date: 2026-11-18
branch: feat/wp-00-scaffold
cost: { minutes: 50, tokens: 389759 }
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
- In guided DDIM sampling, the guidance gradient is taken w.r.t. `z0_hat`, made a detached leaf, because `z0_hat` is where the step is applied.

## Tried and ruled out
- The original sampler took the gradient w.r.t. `z_t`, through the denoiser, and applied it to `z0_hat`. The denoiser Jacobian between the two is not positive definite, so guidance often increased J. Raising `guide_scale` made results steadily worse. Tuning `guide_scale` and `cfg` could not fix this.
- Judging guidance on in-range targets: the conditional model already hits them (J about 1e-7), so guidance has nothing to remove. Measure it on targets beyond the data range or with an active thickness floor (G35).

## Watch out
- Choose the best (guide, cfg) pair only among cells with validity of at least 90%.
- Guide scales above about 0.03 at cfg 0 overshoot on in-range targets. The benchmark pair is guide 3, cfg 0, which suits beyond-range targets.

In the code: `GuidedSampler.guidance_grad` in `src/aeroshape/diffusion/guided_shape_diffusion.py`, `tests/diffusion/test_guidance_direction.py`, `accept()` in `scripts/04_train_diffusion_model.py`, decisions G35 and G36.
