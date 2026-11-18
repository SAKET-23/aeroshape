# Test strategy

## Principles

- **Tests first** for every work package (Superpowers TDD): the Sonnet implementer writes the package's tests from its `Plan/` doc, sees them fail, then implements.
- Each module doc's **Tests** table is the minimum test list; implementers may add more, never fewer.
- Physics code is tested against **analytic solutions** (Poiseuille, Kovasznay, ray–circle intersection, analytic forces) and **conservation laws**, not only against itself.
- Every gradient is tested against **central finite differences in float64 on CPU**.
- Every device-dependent function is tested on **CPU and on the available GPU** (MPS on the Mac, CUDA on cloud) and compared.

## Layout

```text
tests/
  conftest.py            fixtures: devices, float64, tolerances, small shapes, tmp run dirs
  test_layering.py       import-direction rules from architecture.md
  core/                  test_device.py test_config.py test_io.py test_timing.py
  geometry/              test_nurbs.py test_reference_shapes.py test_ibb_grid_mapper.py
  lbm/                   test_lattice.py test_d2q9_solver.py test_aero_forces.py test_adjoint_solver.py
  pinn/                  test_embeddings.py test_mlp_pinn.py test_transformer_pinn.py
                         test_navier_stokes_loss.py test_sampling.py test_stress_integrator.py
  diffusion/             test_schedule.py test_denoiser.py test_guided_sampler.py
  optimization/          test_evaluators.py test_kkt_optimizer.py
  scripts/               test_scripts_smoke.py   (each script runs at smoke tier with --set overrides to tiny sizes)
  fixtures/              tiny reference outputs (< 1 MB total), e.g. smoke CD values per device
```

## Fixtures (`conftest.py`)

```python
@pytest.fixture(params=available_devices())   # ["cpu"] + ["mps"] or ["cuda"]
def device(request): ...
@pytest.fixture
def f64(): return torch.float64      # tests using it are marked float64 and forced to CPU
def assert_close(a, b, rtol, atol, msg)    # wraps torch.testing.assert_close with readable diffs
def tiny_lbm(): ...                         # 100 × 50 grid, N = 10, circle, Re 20
```

## Markers

| Marker | Meaning | Default run |
|---|---|---|
| (none) | fast unit test (< 5 s) | yes |
| `gpu` | needs MPS or CUDA; skipped otherwise | yes if a GPU exists |
| `float64` | CPU only | yes |
| `slow` | > 30 s (Gate checks, convergence studies) | no; run with `-m slow` |
| `e2e` | runs `scripts/00_smoke_e2e.py` | no; run with `-m e2e` |

Default `uv run pytest` target: < 2 minutes on the Mac mini.

## Tolerances

Device-parity tolerances are in [device_and_precision.md](../01_system_design/device_and_precision.md). Gradient checks: relative L2 < 1e-5 for pure functions (NURBS, `q`, stress integrator), < 1e-3 for the LBM adjoint at small grids, < 2e-3 for Gate 2a.

## What runs when

| When | Command |
|---|---|
| After every edit by an implementer | `uv run pytest tests/<module>` |
| End of every work package (orchestrator) | `uv run pytest` + the package's acceptance command |
| End of every phase | `uv run pytest -m "slow or not slow"` + `uv run python scripts/00_smoke_e2e.py --device cpu` and `--device mps` |
| On cloud | `bash scripts/cuda_check.sh` |

## Regression fixtures

`tests/fixtures/smoke_reference.json` holds the smoke-tier CD values and PINN loss after 100 steps on CPU. `00_smoke_e2e.py` compares against it (tolerances above). Updated deliberately with `--update-reference`, never silently.
