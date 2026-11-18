# Agent orchestration — Opus 5.5 orchestrates, Sonnet implements

## Roles

| Role | Model | Does | Never does |
|---|---|---|---|
| Orchestrator | **Opus 5.5** (main session, `/model opus`) | reads `Plan/`, picks the next work package(s), writes the brief, dispatches, reviews diffs, reruns tests, runs gates, updates `Plan/README.md` status, asks the user at decision points | write or edit implementation code or tests |
| Implementer | **Sonnet** (subagent, `model: "sonnet"`, strictly) | writes tests first, implements one work package, runs its tests on CPU and MPS, reports | touch files outside its package's list; change `Plan/`; commit |
| Reviewer | Sonnet (fresh subagent, read-only) | independent review of all code changed since the last review, against `Plan/`: correctness, physics, spec compliance, test quality, device rules, results; ranked findings with file:line. Runs at the end of every phase and on request | fix code, edit any file except its report |

The orchestrator may edit `Plan/` docs (e.g. updating measured numbers or recording a decision change in `gaps_and_decisions.md`), configs it owns, and the status table.

## Dispatch protocol

1. **Select**: the lowest-numbered package whose dependencies are done ([work_packages.md](work_packages.md)); dispatch a whole parallel group in one message when allowed.
2. **Brief** (template below). The brief is the implementer's only context.
3. **Wait** for the result (never predict it).
4. **Verify** independently:
   - `git diff --stat` — only allowed files changed;
   - read the diff against the `Plan/` doc's interface and algorithm;
   - `uv run pytest tests/<module> -q` on CPU and MPS;
   - the package's acceptance command;
   - for packages that extend the smoke harness: `scripts/00_smoke_e2e.py --stages …`.
5. **Fix loop**: send precise failures back to the same agent (SendMessage, keeps its context). Max 2 rounds; then stop and report to the user with the failing output.
6. **Close**: mark the package done in `Plan/README.md` with the date, the acceptance output summary and any measured numbers.
7. **Commit**: only after the user has said commits are wanted; one commit per package on a feature branch `feat/wp-<nn>-<name>`.

## Brief template

```text
You are implementing Work Package <NN>: <name> for the aeroshape project (repo root: <path>).

Read first (in order):
- Plan/<module doc>.md            (the spec: interface, algorithm, tests, pitfalls)
- Plan/01_system_design/data_contracts.md   (types you must use exactly)
- Plan/01_system_design/device_and_precision.md
- Plan/raw/derivations.md §<n>    (math)
- Existing code you build on: <files from finished packages>

You may create or edit ONLY:
- src/aeroshape/<...>.py
- tests/<...>/test_<...>.py
- <configs or fixtures, if any>

Do:
1. Write the tests listed in the spec's Tests table first; run them; confirm they fail.
2. Implement until they pass on CPU, then run them on MPS (AEROSHAPE_DEVICE=mps).
3. Run: <acceptance command>.
4. Keep to the interface names and tensor shapes in the spec. No new dependencies.

Rules: no .item()/.cpu()/.numpy() or Python loops over nodes/points inside solvers or losses; no in-place ops on tensors that need grad; float64 only on CPU.

Report back: files changed; the exact test and acceptance commands you ran with their output (last 30 lines each); any deviation from the spec and why; anything you could not finish.
```

## Gates

A phase gate ([milestones_and_gates.md](milestones_and_gates.md)) is run by the orchestrator itself after the phase's last package. No package of the next phase is dispatched until it passes, or the user explicitly accepts a documented miss.

## Parallelism and cost

- At most 3 implementers at once (parallel groups in work_packages.md are sized to that).
- Each implementer works on disjoint files; shared files (`core/types.py`, configs) are owned by WP 0 and changed only by the orchestrator between packages.
- Long runs (dataset v2, PINN training, benchmark) are started by the orchestrator with `run_in_background`, not by implementers.

## When to stop and ask the user

- A gate fails after the debugging order in its doc.
- A spec decision turns out wrong (e.g. MPS cannot do something) and the fix changes an interface.
- Any cloud spend is about to start.
- Commits/pushes.
