#!/usr/bin/env bash
# CUDA verification on a cloud/Colab GPU (Plan/07_testing/cuda_cloud_runbook.md). Run from the repo root.
# SKIP_SYNC=1 skips the install (the Docker cuda image already has the environment).
set -euo pipefail
nvidia-smi
if [ -z "${SKIP_SYNC:-}" ]; then
  # uv.lock pins torch 2.15.0, whose Linux wheel is CUDA 13 (host driver >= 580). Older drivers (Colab T4 ~ CUDA 12): set
  # TORCH_INDEX=https://download.pytorch.org/whl/cu126 TORCH_VERSION=2.14.1 (that index has no 2.15; there is no cu124 torch >= 2.15 either).
  uv sync --frozen --extra dev
  if [ -n "${TORCH_INDEX:-}" ]; then
    uv pip install --reinstall-package torch --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION:-2.14.1}"
  fi
  source .venv/bin/activate
fi
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
pytest -m "not slow" -q
python scripts/00_smoke_e2e.py --device cuda
python scripts/compare_summary.py runs/smoke/cuda/summary.json tests/fixtures/smoke_reference.json
