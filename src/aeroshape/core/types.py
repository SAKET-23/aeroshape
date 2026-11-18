"""Frozen dataclasses for every cross-module type (Plan/01_system_design/data_contracts.md)."""
from dataclasses import dataclass, field
from typing import Optional, Protocol

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Condition:
    re: float
    aoa_deg: float

    def as_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        """-> [3] = [log Re, sin AoA, cos AoA]"""
        a = torch.deg2rad(torch.tensor(self.aoa_deg, dtype=torch.float64))
        re = torch.log(torch.tensor(self.re, dtype=torch.float64))
        return torch.stack([re, torch.sin(a), torch.cos(a)]).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class Boundary:
    """x, y, nx, ny, ds: [B, 2M-1]"""
    x: Tensor
    y: Tensor
    nx: Tensor
    ny: Tensor
    ds: Tensor


@dataclass(frozen=True)
class LinkSet:
    node: Tensor  # [L] long, flat index j*Nx+i
    dir: Tensor  # [L] long, 1..8
    q: Tensor  # [L] float
    far_ok: Tensor  # [L] bool
    seg: Tensor  # [L] long


@dataclass(frozen=True)
class GridGeometry:
    solid: Tensor  # [Ny, Nx] bool
    links: LinkSet
    N: int
    origin: tuple[int, int]  # (i0, j0)


@dataclass(frozen=True)
class LBMParams:
    tau: float
    omega: float
    omega_minus: float
    u_lb: float
    N: int
    Nx: int
    Ny: int
    collision: str
    sponge: Optional[Tensor] = None  # [Ny, Nx]


@dataclass(frozen=True)
class FlowState:
    f: Tensor  # [9, Ny, Nx]
    rho: Tensor
    ux: Tensor
    uy: Tensor
    step: int


@dataclass(frozen=True)
class ForceHistory:
    step: Tensor  # [T] long
    cd: Tensor  # [T]
    cl: Tensor  # [T]


@dataclass(frozen=True)
class SolveResult:
    state: FlowState
    f_mean: Optional[Tensor]  # [9, Ny, Nx]
    history: ForceHistory
    cd: Tensor  # 0-dim
    cl: Tensor
    period_steps: Optional[int]
    converged: bool


@dataclass(frozen=True)
class AdjointResult:
    F_star: Tensor  # [9, Ny, Nx]
    dI_dq: Tensor  # [L]
    dI_dalpha: Tensor  # [D]
    residual_history: Tensor  # [T]


@dataclass(frozen=True)
class EvalResult:
    cd: Tensor  # 0-dim
    cl: Tensor
    dcd_dalpha: Tensor  # [D]
    dcl_dalpha: Tensor
    wall_s: float
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PINNOutput:
    psi: Tensor  # [B, N]
    p: Tensor
    fx: Optional[Tensor] = None
    fy: Optional[Tensor] = None
    attn: Optional[Tensor] = None  # [B, heads, N, D+1]


@dataclass(frozen=True)
class FlowFields:
    u: Tensor  # [B, N]
    v: Tensor
    p: Tensor
    derivs: Optional[dict] = None


@dataclass(frozen=True)
class CostWeights:
    lam: float
    rho_al: float
    cl_ref: float


class Evaluator(Protocol):
    def evaluate(self, alpha: Tensor, cond: Condition, weights: Optional[CostWeights] = None) -> EvalResult: ...

    def value(self, alpha: Tensor, cond: Condition, weights: Optional[CostWeights] = None) -> tuple[Tensor, Tensor]: ...
