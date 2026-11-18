---
title: "Config overrides only accept keys that already exist, so removing a config key breaks any test that overrides it"
author: SAKET-23
date: 2026-11-17
branch: feat/wp-00-scaffold
cost: { minutes: 191, tokens: 840947 }
touches:
  - ".python-version"
  - CLAUDE.md
  - Plan/01_system_design/architecture.md
  - Plan/01_system_design/config_and_run_tiers.md
  - Plan/01_system_design/data_contracts.md
  - Plan/01_system_design/device_and_precision.md
  - Plan/01_system_design/repo_layout.md
  - Plan/02_geometry/grid_mapping_ibb.md
  - Plan/02_geometry/nurbs.md
  - Plan/02_geometry/reference_shapes.md
---

## Decided
Unused config keys are kept, with a comment marking them UNUSED, until the same change also removes every override of them. Remove both together.

## Tried and ruled out
Deleting the unused validation keys on their own broke a smoke test. That test passes an override for one of those keys, and the config loader rejects an override for a key that does not exist.

## Watch out
Before you delete a config key, grep the tests and scripts for overrides of it.

In the code: configs/validate_cylinder.yaml keys validate.perturb and validate.transient; tests/scripts/test_validate_cylinder_smoke.py passes `--set validate.transient=0.5`; src/aeroshape/core/config.py load_config(case, tier, overrides) rejects unknown keys.
