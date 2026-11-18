Docker on macOS cannot use Metal/MPS: containers there run Linux on CPU only, so Mac GPU runs stay native via `uv` (see Plan/07_testing/e2e_mac_mini.md).

# Docker images (WP 22)

| Target | Base | Torch | Use |
|---|---|---|---|
| `cpu` | `python:3.12-slim` | CPU wheel from `download.pytorch.org/whl/cpu` | tests anywhere, CI, Mac (CPU) |
| `cuda` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | locked PyPI wheel (CUDA 13.2, bundled libraries) | cloud GPU runs |

## Build and run

```bash
docker build --target cpu -t aeroshape:cpu .
docker run --rm aeroshape:cpu                       # pytest -m "not slow" -q
docker run --rm aeroshape:cpu python scripts/00_smoke_e2e.py --device cpu

docker build --target cuda -t aeroshape:cuda .      # needs a host driver >= 580 (CUDA 13 wheels), see below
docker run --rm --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs aeroshape:cuda   # runs scripts/cuda_check.sh
```

`data/`, `runs/`, `docs/` and `*.pt` are not copied into the image (`.dockerignore`); mount `data/` and `runs/` as above. The cuda image needs the NVIDIA container toolkit on the host.

## How torch is chosen

`uv.lock` pins torch 2.15.0 from PyPI, and on Linux that wheel is the CUDA 13.2 build with its `nvidia-*` libraries bundled. So:

- **cuda**: `uv sync --frozen` installs exactly the locked set. The base image's CUDA 12.4 only supplies the container runtime; the wheel brings its own CUDA 13 libraries and needs a host driver >= 580. For an older driver build with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 --build-arg TORCH_VERSION=2.14.1`: the cu126 index has no torch 2.15 (nor does cu124), so the image swaps in 2.14.1 from that index.
- **cpu**: the locked CUDA wheels would add ~3 GB. The locked dependencies are exported with `uv export --frozen` minus `torch`, `triton`, `nvidia-*` and `cuda-*`, and the same torch version (`--build-arg TORCH_VERSION`, default 2.15.0) is installed from the CPU index.

Both stages build in a builder image and copy only the virtual environment into the final one (no uv, pip cache or compilers).
