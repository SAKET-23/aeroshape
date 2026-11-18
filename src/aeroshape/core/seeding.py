import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA / MPS


def rand_on(device, *shape, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    """Standard normal drawn on CPU with a Generator, then moved: identical inputs on every device."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype).to(device)
