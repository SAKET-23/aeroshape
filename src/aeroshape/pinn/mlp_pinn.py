"""Baseline parametric PINN (Plan/04_models/mlp_pinn.md): Fourier features + d_wall + alpha + cond -> MLP."""
import torch
from torch import nn

from aeroshape.core.types import PINNOutput
from aeroshape.pinn.embeddings import FourierFeatures, InputNorm


class MLPPINN(nn.Module):
    def __init__(self, D: int, hidden: int = 128, layers: int = 6, fourier: int = 64, scale: float = 2.0,
                 v2: bool = False, psi_scale: float = 3.0, seed: int = 0):
        super().__init__()
        self.psi_scale = psi_scale  # ~ window height, so the initial u = dpsi/dy is O(1)
        self.ff, self.norm = FourierFeatures(fourier, scale, seed), InputNorm(D)
        dims = [2 * fourier + 1 + D + 3] + [hidden] * layers
        self.body = nn.Sequential(*[m for i in range(layers) for m in (nn.Linear(dims[i], dims[i + 1]), nn.SiLU())])
        self.out = nn.Linear(hidden, 4 if v2 else 2)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, xy, alpha, cond, d_wall) -> PINNOutput:
        N = xy.shape[1]
        a, c = self.norm(alpha, cond)
        z = torch.cat([self.ff(xy), d_wall[..., None], a[:, None].expand(-1, N, -1), c[:, None].expand(-1, N, -1)], -1)
        o = self.out(self.body(z))
        return PINNOutput(psi=self.psi_scale * o[..., 0], p=o[..., 1],
                          fx=o[..., 2] if o.shape[-1] == 4 else None, fy=o[..., 3] if o.shape[-1] == 4 else None)
