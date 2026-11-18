# CUDA on cloud / Colab — runbook

**Work package:** WP 20. **Files:** `scripts/cuda_check.sh`, `notebooks/colab_cuda.ipynb`.

The Mac mini is the development machine; CUDA is verified and used for full-tier runs on a rented or Colab GPU. The code has no CUDA-specific branches except optional `torch.compile`, so the check is mostly "the same tests pass on a different backend".

## Getting the code there

Private repo: push to GitHub and clone with a token, or upload a `git bundle` (`git bundle create pinns.bundle --all`) to the notebook. `data/` and `runs/` are uploaded/downloaded as tarballs (Google Drive mount on Colab).

## Setup

```bash
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv python install 3.12
uv sync --frozen --extra dev        # locked torch 2.15.0: the PyPI Linux wheel is built for CUDA 13.2 and needs driver >= 580
# Older driver (CUDA 12.x, e.g. Colab T4)? The cu12x indexes have no 2.15.x, so override torch (WP 22 finding):
#   uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

### Alternative: Docker (WP 22)

On any machine with Docker and the NVIDIA container toolkit (not Colab, which does not run Docker). The default image uses the locked CUDA 13 torch (driver >= 580); for CUDA 12 drivers add `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 --build-arg TORCH_VERSION=2.14.1`:

```bash
docker build --target cuda -t aeroshape:cuda .
docker run --rm --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs aeroshape:cuda bash scripts/cuda_check.sh
```

## `scripts/cuda_check.sh`

```bash
set -euo pipefail
nvidia-smi
# unless SKIP_SYNC=1 (Docker image): uv sync --frozen --extra dev, or the TORCH_INDEX/TORCH_VERSION override
python -c "import torch; assert torch.cuda.is_available()"
pytest -m "not slow" -q
python scripts/00_smoke_e2e.py --device cuda
python scripts/compare_summary.py runs/smoke/cuda/summary.json tests/fixtures/smoke_reference.json
```

Pass: every command exits 0. `compare_summary.py` checks every stage ok, stage-2 CD within rtol 1e-4, and both PINN final losses within 5e-2. Optional, not gated: `python scripts/lattice_table.py --benchmark --device cuda` (MLUPS for the compute budget), and `--compile` timings recorded by hand.

## Full-tier jobs (from the notebook)

| Job | Command | Output to download |
|---|---|---|
| Gate 1 full | `01_validate_lbm_cylinder.py --tier full --device cuda` | `runs/validate_cylinder/full/` |
| Dataset v2 | `02_generate_lbm_dataset.py --set v2 --tier dev|full --device cuda` | `data/lbm_anchors/<tier>/` |
| PINN v2 | `03_train_transformer_pinn.py --set pinn.version=v2 --tier dev --device cuda` | `runs/pinn_transformer_v2_dev/` |
| Benchmark full | `05_run_4way_benchmark.py --tier full --device cuda` | `runs/benchmark/full/` |

Each job is resumable without a flag: the dataset script skips runs already converged on disk, and training writes checkpoints. After a disconnect, the notebook cell re-runs the same command.

## Bringing results back to the Mac

`load_ckpt(path, device="mps")` loads CUDA-trained checkpoints (tensors saved on CPU). A test (`tests/core/test_io.py`) round-trips a checkpoint through CPU → MPS to guard this.

## Colab specifics

- Runtime type GPU; T4 is enough for smoke and dev; L4/A100 for full tier.
- Sessions time out: keep each cell under the session limit or rely on the skip-if-done behaviour.
- Mount Drive at `/content/drive` and point `data/` and `runs/` there with symlinks so outputs survive.
