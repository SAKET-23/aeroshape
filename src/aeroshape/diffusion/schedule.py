"""Cosine noise schedule (derivations 6.1-6.2); gamma_bar is the noise schedule, alpha is the shape."""
import math

import torch


class CosineSchedule:
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
        self.gamma_bar = f / f[0]  # [T+1], gamma_bar_0 = 1, gamma_bar_T = 0

    def q_sample(self, z0, t, eps):
        gb = self.gamma_bar.to(z0.device, z0.dtype)[t.to(z0.device)][:, None]
        return gb.sqrt() * z0 + (1 - gb).sqrt() * eps

    def ddim_timesteps(self, n_steps: int):
        """[n_steps + 1] descending ints from T-1 (gamma_bar_T = 0 would divide by zero) to 0; step i goes ts[i] -> ts[i+1]."""
        return torch.linspace(self.T - 1, 0, n_steps + 1).round().long()
