---
title: "Write YAML floats with a dot and a signed exponent, and never wait on pgrep patterns that match the waiter"
author: SAKET-23
date: 2026-11-18
branch: feat/wp-00-scaffold
cost: { minutes: 227, tokens: 1382773 }
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

## Watch out
- PyYAML (YAML 1.1) loads `1.0e4` as the STRING '1.0e4'. Write `1.0e+4`. This crashed the optimizer (`min(2*mu, mu_max)`) the first time a thickness constraint became active.
- Background wait loops such as `while pgrep -f "<pattern>"` can match their own shell command, so they never end. An agent that "cleans up" with pkill on such a pattern also killed other sessions' watchers (exit 144). Put the condition in a file check, or use a pattern that does not appear in the waiting command, and never pkill processes you did not start.

In the code: `optimizer.mu_max` in `configs/base.yaml`. The test in `tests/scripts/test_benchmark.py` asserts that the optimizer config values are numbers.
