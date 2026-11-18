import time

import torch


def sync(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Context manager; syncs on enter and exit; `.elapsed` in seconds."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.elapsed = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self._t0


_mps_peak = 0


def peak_memory(device) -> int:
    """bytes; cuda exact, mps approximate (high-water of sampled driver allocation), cpu 0"""
    global _mps_peak
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "mps":
        _mps_peak = max(_mps_peak, torch.mps.driver_allocated_memory())
        return _mps_peak
    return 0


def reset_peak(device) -> None:
    global _mps_peak
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        _mps_peak = torch.mps.driver_allocated_memory()
