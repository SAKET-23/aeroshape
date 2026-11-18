# Repository layout

```text
pinns/
├── CLAUDE.md
├── README.md                       results, figures, benchmark table (written in WP 21)
├── pyproject.toml
├── uv.lock
├── .python-version                 3.12
├── configs/
│   ├── base.yaml
│   ├── tiers/{smoke,dev,full}.yaml
│   ├── validate_cylinder.yaml
│   ├── case1_circle_to_naca_re20.yaml
│   ├── case1_re5.yaml
│   ├── case2_circle_to_rae_re1000.yaml
│   └── case3_naca_to_rae_aoa10.yaml
├── data/                           git-ignored
│   ├── lbm_anchors/<tier>/<shape_id>.pt, merged.pt
│   └── diffusion_shapes/<tier>.pt
├── runs/                           git-ignored; one folder per run: config.yaml, log.csv, ckpt/, fig/, results.json
├── src/aeroshape/
│   ├── __init__.py
│   ├── core/        device.py config.py seeding.py timing.py io.py types.py
│   ├── geometry/    nurbs.py reference_shapes.py ibb_grid_mapper.py data/rae2822.csv
│   ├── lbm/         lattice.py d2q9_solver.py aero_forces.py adjoint_solver.py
│   ├── pinn/        embeddings.py mlp_pinn.py transformer_pinn.py navier_stokes_loss.py sampling.py stress_integrator.py
│   ├── diffusion/   schedule.py denoiser.py guided_shape_diffusion.py
│   ├── optimization/evaluators.py kkt_optimizer.py
│   └── visualization/plots.py animate.py
├── scripts/
│   ├── 00_smoke_e2e.py
│   ├── 01_validate_lbm_cylinder.py
│   ├── 02_generate_lbm_dataset.py
│   ├── 03_train_transformer_pinn.py
│   ├── 04_train_diffusion_model.py
│   ├── 05_run_4way_benchmark.py
│   ├── 06_optimize_case.py         one optimizer run, one evaluator (used by 05)
│   ├── lattice_table.py            prints the lattice-parameter table; --fit-report; --benchmark (MLUPS)
│   ├── compare_summary.py          compares a smoke summary.json with tests/fixtures/smoke_reference.json
│   └── cuda_check.sh
├── tests/                          mirrors src/aeroshape/: tests/geometry/test_nurbs.py, …
│   ├── conftest.py                 device fixtures, float64 fixture, tolerance helpers
│   └── test_layering.py
├── notebooks/colab_cuda.ipynb
├── docs/SSL (1).pdf                the original report
└── Plan/                           this specification
```

## `pyproject.toml`

- Build backend: `hatchling`; package discovered at `src/aeroshape`.
- `requires-python = ">=3.10,<3.14"`; `.python-version` pins 3.12 for `uv` (the system Python is 3.14; see G22).
- Runtime deps: `torch>=2.4`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `imageio` (GIFs).
- Optional `viz3d`: `pyvista`. Optional `dev`: `pytest`, `pytest-timeout`, `ruff`, `tensorboard`.
- CUDA install on cloud: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` before `uv sync --no-install-package torch` (exact commands in [cuda_cloud_runbook.md](../07_testing/cuda_cloud_runbook.md)).
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, markers `slow`, `gpu`, `float64`, `e2e`; default `addopts = "-m 'not slow'"`.

## Naming conventions

- Modules and functions `snake_case`; classes `PascalCase`; tensor shape in the docstring's first line, e.g. `"""alpha [B, k] -> Boundary"""`.
- Lattice-unit variables carry a `_lb` suffix (`u_lb`, `nu_lb`); everything else is non-dimensional.
- Scripts start with a two-digit number and are runnable as `uv run python scripts/NN_name.py --tier smoke`.

## `.gitignore` additions

`data/`, `runs/`, `*.pt` outside `tests/fixtures/`, `.venv/`, `__pycache__/`, `.pytest_cache/`.
