# Device and precision

Target machines: **Mac mini (Apple M4 Pro, 12 CPU cores, 24 GB unified memory, Metal via PyTorch MPS)** for all first runs, and **NVIDIA CUDA on a cloud or Colab GPU** for full-tier runs. CPU is always available as a reference.

## `core/device.py`

```python
def get_device(pref: str | None = None) -> torch.device
    # order: pref arg > AEROSHAPE_DEVICE env > cuda > mps > cpu
def get_dtype(device: torch.device, purpose: Literal["sim", "train", "check"]) -> torch.dtype
    # "check" -> float64, and raises if device is mps
    # otherwise float32
def device_info(device) -> dict      # name, torch version, memory, logged at run start
```

## Rules

| Rule | Why |
|---|---|
| float32 on MPS and CUDA for simulation and training | MPS has no float64; float32 is enough for LBM at Mach 0.1 |
| float64 only on CPU, only for gradient checks and adjoint validation | FD checks at `Δ ~ 1e-4` need float64 |
| No mixed precision (`autocast`) | PINN takes third derivatives; LBM relaxes small differences |
| Every tensor created with `device=` and `dtype=` explicitly | Avoids silent CPU tensors and float64 creep from numpy |
| No `.item()`, `.cpu()`, `.numpy()`, `print(tensor)` inside step functions or losses | Each forces a device sync; on MPS this can cut throughput several-fold |
| Logging reads values every `log_every` steps only, in one batched `.cpu()` | Same reason |
| `torch.roll` for streaming, `index_select` / `index_add_` for links | Both are supported and fast on MPS and CUDA |
| No in-place ops on tensors that need gradients | Breaks autograd; use masks and `torch.where` |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` set by every script | Unsupported ops fall back to CPU instead of crashing; a warning filter logs which op |
| `torch.compile` only behind `--compile` and only on CUDA | MPS support for compile is not reliable enough to depend on |

## Synchronized timing (`core/timing.py`)

```python
def sync(device): torch.cuda.synchronize() if cuda; torch.mps.synchronize() if mps
class Timer:            # context manager; calls sync() on enter and exit
def peak_memory(device) -> int   # cuda: max_memory_allocated; mps: driver_allocated_memory (high-water tracked by sampling)
def reset_peak(device)
```

All benchmark numbers come from `Timer`. A warm-up call precedes every timed region (first MPS kernels compile shaders).

## Seeding (`core/seeding.py`)

`seed_everything(seed)`: Python, NumPy, torch, CUDA. Random tensors used in tests and smoke runs are drawn on CPU with a `torch.Generator` and then moved, so CPU, MPS and CUDA see identical inputs. Results are then compared at float32 tolerance, not bitwise.

## Checkpoint portability (`core/io.py`)

- `save_ckpt(path, obj)`: moves tensors to CPU before `torch.save`.
- `load_ckpt(path, device)`: `torch.load(path, map_location=device, weights_only=False)` for our own files.
- Every checkpoint stores `torch.__version__`, git hash, device it was trained on, and the full config.

## Device-parity tolerances

| Quantity | CPU vs MPS / CUDA tolerance |
|---|---|
| One LBM step | max abs diff ≤ 1e-6 (relative to `f` values ~0.1) |
| CD after 2,000 steps (smoke) | relative ≤ 1e-4 |
| PINN forward | relative ≤ 1e-5 |
| PINN third derivatives | relative ≤ 1e-3 |
| Training loss after 100 steps | relative ≤ 5e-2 (non-associative reductions accumulate) |

## Known MPS risks to check early

1. Higher-order autograd through `scaled_dot_product_attention` (checked in WP 13; fallback is explicit matmul + softmax attention).
2. `torch.linalg.lstsq` on MPS (used for shape fitting) — run fitting on CPU; it is one-off.
3. Large `index_add_` with duplicate indices — supported, but check determinism tolerance in tests.
4. Memory reporting is coarser than CUDA; peak-memory numbers on MPS are labelled approximate.
