# PINNs: Differentiable Aerodynamic Shape Optimization

A 2D airfoil shape optimizer in GPU PyTorch. A NURBS shape (control points `alpha`) feeds three solver tiers that are benchmarked against each other:

1. **D2Q9 Lattice Boltzmann (LBM) + discrete adjoint**: the classical ground truth. Uses Bouzidi interpolated bounce-back so the gradient sees the curved boundary.
2. **Transformer PINN**: a fast surrogate. Each flow point attends to the control-point tokens, and the network outputs a stream function `psi` and pressure `p`. `u = dpsi/dy` and `v = -dpsi/dx`, so the flow is divergence-free by construction. The baseline is a Fourier-feature MLP PINN.
3. **Physics-guided diffusion**: a DDPM over `alpha` that generates a shape for a target `[CL, CD, Re, AoA]`. Each denoising step is steered by the frozen PINN's gradients plus the thickness (KKT) penalty.

## The spec is [Plan/](Plan/README.md)

- [Plan/README.md](Plan/README.md): index, reading order, **status table** (updated after every work package and gate).
- [Plan/raw/gaps_and_decisions.md](Plan/raw/gaps_and_decisions.md): binding decisions where the original plan and the report disagree. Change a decision there first, then the code.
- [Plan/01_system_design/data_contracts.md](Plan/01_system_design/data_contracts.md): every shared type, tensor shape and file format. Code must match it.
- [Plan/09_execution/work_packages.md](Plan/09_execution/work_packages.md): what to build next, file lists, acceptance commands.
- Sources: [Plan/raw/original_project_plan.md](Plan/raw/original_project_plan.md) and the original 63-page report `docs/SSL (1).pdf` (notes in [Plan/raw/report_notes.md](Plan/raw/report_notes.md)).

## How the build runs

- **Opus 5.5 orchestrates, Sonnet implements (strictly).** The orchestrator never writes implementation code; it dispatches one work package per Sonnet subagent with the brief template in [Plan/09_execution/agent_orchestration.md](Plan/09_execution/agent_orchestration.md), then reviews the diff and reruns the tests itself.
- Tests first for every package. A phase gate must pass before the next phase starts.
- First end-to-end test on the Mac mini (CPU and MPS): `uv run python scripts/00_smoke_e2e.py --device mps`. CUDA is verified on cloud with `bash scripts/cuda_check.sh`.

## Stack and layout

- `uv` with Python 3.12 (the system Python 3.14 is too new to rely on), PyTorch 2.x, NumPy, SciPy, Matplotlib, pytest.
- Package `aeroshape` in `src/aeroshape/{core,geometry,lbm,pinn,diffusion,optimization,visualization}`; numbered `scripts/0N_*.py`; YAML configs in `configs/` with `--tier smoke|dev|full`.
- Device order CUDA → MPS → CPU (`AEROSHAPE_DEVICE` overrides). float32 on GPUs; float64 only on CPU (MPS has none) for gradient checks.

## Rules for this codebase

- **Everything on the path from `alpha` to drag and lift must stay differentiable.** No `.item()`, `.cpu()`, `.numpy()` or Python loops over lattice nodes or points inside solvers and losses. Vectorize with tensor ops.
- No in-place writes on tensors that need gradients (pin ends with a mask).
- Every gradient (adjoint, PINN, NURBS, `q`) gets a float64 finite-difference test on CPU.
- PINN points never attend to each other (keeps per-point derivatives correct).
- Report measured numbers only; no claimed speed-ups without a benchmark row.
- Keep `data/`, `runs/`, trained weights and anchor datasets out of git.

## Tooling

- **graft**: the code graph is in `graft/` (git-ignored, rebuild with `graft build`). Use the `graft_*` MCP tools before you grep or read files.
- **Superpowers**: brainstorm, then plan, then TDD, then verify before you call something done.
- **Ponytail**: make the smallest change that fully solves the task.
- Commit only when asked. Branch off `main` for feature work.
