"""Transformer eps-predictor over control-point tokens (Plan/04_models/diffusion.md)."""
import math

import torch
from torch import nn


def sincos(v, d):
    """[...] -> [..., d] sinusoidal features."""
    f = torch.exp(-math.log(1000.0) * torch.arange(d // 2, device=v.device) / (d // 2))
    a = v[..., None] * f * 2 * math.pi
    return torch.cat([a.sin(), a.cos()], -1)


class ShapeDenoiser(nn.Module):
    def __init__(self, n_free, x_ctrl_free, surface_id, d=128, layers=4, heads=4, cond_dim=4, T=1000):
        super().__init__()
        self.d, self.T = d, T
        self.register_buffer("x_ctrl", x_ctrl_free.detach().float().clone())
        self.register_buffer("surface_id", surface_id.long().clone())
        self.register_buffer("cond_mean", torch.zeros(cond_dim))  # set from training labels
        self.register_buffer("cond_std", torch.ones(cond_dim))
        self.val = nn.Linear(1, d)
        self.surf = nn.Embedding(2, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, z_t, t, cond, null_mask=None):
        """z_t [B, n_free], t [B] int, cond [B, cond_dim] raw | None, null_mask [B] bool (True -> null token)."""
        B = z_t.shape[0]
        tok = self.val(z_t[..., None]) + sincos(self.x_ctrl, self.d) + self.surf(self.surface_id)
        tok = tok + self.t_mlp(sincos(t.to(z_t.dtype) / self.T, self.d))[:, None]
        if cond is None:
            c = self.null.expand(B, -1)
        else:
            c = self.c_mlp((cond - self.cond_mean) / self.cond_std)
            if null_mask is not None:
                c = torch.where(null_mask[:, None], self.null, c)
        h = self.enc(torch.cat([c[:, None] + self.t_mlp(sincos(t.to(z_t.dtype) / self.T, self.d))[:, None], tok], 1))
        return self.out(self.norm(h[:, 1:]))[..., 0]
