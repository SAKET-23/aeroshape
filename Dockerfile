# syntax=docker/dockerfile:1
# Two targets (docker/README.md):  docker build --target cpu -t aeroshape:cpu .   |   docker build --target cuda -t aeroshape:cuda .
#
# Torch choice: uv.lock pins torch 2.15.0 from PyPI. On Linux that wheel is the CUDA 13.2 build (bundled nvidia-* libraries;
# the base image's own CUDA version does not matter, only the host driver does, >= 580 for CUDA 13).
#   cuda target: `uv sync --frozen` installs exactly that locked wheel set. For an older driver pass
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 --build-arg TORCH_VERSION=2.14.1 (the cu126 index has no 2.15).
#   cpu target:  the locked CUDA wheels are ~3 GB of dead weight, so the locked deps are exported WITHOUT torch/nvidia/cuda/triton
#                and the same torch version is installed from the CPU index.
ARG TORCH_VERSION=2.15.0
ARG UV_VERSION=0.12.2

# ---------------------------------------------------------------- cpu
FROM python:3.12-slim AS cpu-build
ARG TORCH_VERSION
ARG UV_VERSION
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==${UV_VERSION}
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv \
 && uv export --frozen --no-hashes --no-emit-project --extra dev | grep -vE '^(torch|triton|nvidia-|cuda-)' > /tmp/req.txt \
 && uv pip install --python /opt/venv/bin/python -r /tmp/req.txt \
 && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION}
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-deps -e .

FROM python:3.12-slim AS cpu
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 AEROSHAPE_DEVICE=cpu
WORKDIR /app
COPY --from=cpu-build /opt/venv /opt/venv
COPY --from=cpu-build /app/src ./src
COPY pyproject.toml ./
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
CMD ["pytest", "-m", "not slow", "-q"]

# ---------------------------------------------------------------- cuda
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda-build
ARG TORCH_VERSION
ARG UV_VERSION
ARG TORCH_INDEX=""
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python UV_PROJECT_ENVIRONMENT=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv==${UV_VERSION}
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --extra dev --no-install-project
RUN if [ -n "$TORCH_INDEX" ]; then \
      uv pip install --python /opt/venv/bin/python --reinstall-package torch --index-url "$TORCH_INDEX" torch==${TORCH_VERSION}; fi
COPY src ./src
RUN uv sync --frozen --extra dev --inexact --no-install-package torch

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility SKIP_SYNC=1
WORKDIR /app
COPY --from=cuda-build /opt/python /opt/python
COPY --from=cuda-build /opt/venv /opt/venv
COPY --from=cuda-build /app/src ./src
COPY pyproject.toml uv.lock .python-version ./
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
CMD ["bash", "scripts/cuda_check.sh"]
