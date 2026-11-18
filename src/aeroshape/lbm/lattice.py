"""D2Q9 constants, lattice-parameter construction and unit conversion (derivations 2.1, 2.8, 2.9)."""
import math
from functools import lru_cache

import torch

from aeroshape.core.types import Condition, LBMParams

CS2 = 1 / 3
C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float64)
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])


@lru_cache(maxsize=None)
def lattice_constants(device, dtype):
    device = torch.device(device)
    return C.to(device), W.to(device=device, dtype=dtype), OPP.to(device)


def make_params(cfg, cond: Condition, N: int | None = None, device="cpu", dtype=torch.float32) -> LBMParams:
    c = cfg.lbm
    N = N or c.cells_per_d
    Nx, Ny = int(c.domain_d[0] * N), int(c.domain_d[1] * N)
    u_lb = c.mach * math.sqrt(CS2)
    nu = u_lb * N / cond.re
    tau = 3 * nu + 0.5
    tau_m = 0.5 + c.trt_magic / (tau - 0.5)
    sponge = None
    if c.sponge.enabled:
        width = c.sponge.width_d * N
        s = ((torch.arange(Nx, dtype=torch.float64) - (Nx - width)) / width).clamp(min=0) ** 2
        omega = 1 / (3 * nu * (1 + (c.sponge.nu_factor - 1) * s) + 0.5)
        sponge = omega.expand(Ny, Nx).to(device=device, dtype=dtype).contiguous()
    return LBMParams(tau=tau, omega=1 / tau, omega_minus=1 / tau_m if c.collision == "trt" else 1 / tau,
                     u_lb=u_lb, N=N, Nx=Nx, Ny=Ny, collision=c.collision, sponge=sponge)


def lbm_to_nondim(rho, ux, uy, u_lb):
    return ux / u_lb, uy / u_lb, (rho - 1) * CS2 / u_lb**2
