import os
from typing import Literal

import torch


def get_device(pref: str | None = None) -> torch.device:
    """pref arg > AEROSHAPE_DEVICE env > cuda > mps > cpu"""
    name = pref if pref not in (None, "auto") else os.environ.get("AEROSHAPE_DEVICE")
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device, purpose: Literal["sim", "train", "check"]) -> torch.dtype:
    if purpose == "check":
        if device.type == "mps":
            raise ValueError("float64 'check' precision is not available on mps; use cpu")
        return torch.float64
    return torch.float32


def device_info(device: torch.device) -> dict:
    name, mem = device.type, None
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "mps":
        name = "Apple MPS"
        mem = torch.mps.recommended_max_memory()
    return {"device": str(device), "name": name, "torch": torch.__version__, "memory": mem}
