"""Geometry cross-attention PINN (Plan/04_models/transformer_pinn.md): control-point tokens + one condition token,
every flow point cross-attends to them. Points never interact, so per-point derivatives (sum trick) are exact."""
import torch
import torch.nn.functional as F
from torch import nn

from aeroshape.core.types import PINNOutput
from aeroshape.pinn.embeddings import FourierFeatures, InputNorm, Wavelet


def _embed(i, d):
    return nn.Sequential(nn.Linear(i, d), Wavelet(d), nn.Linear(d, d))


class _Block(nn.Module):
    """Pre-norm attention (Q from x, K = V from kv) + Wavelet FFN, residual. No dropout."""

    def __init__(self, d, heads, ffn, impl):
        super().__init__()
        self.h, self.impl = heads, impl
        self.nq, self.nk, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), Wavelet(ffn), nn.Linear(ffn, d))

    def forward(self, x, kv, return_attn=False):
        B, N, d = x.shape
        q = self.q(self.nq(x)).view(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(self.nk(kv)).view(B, kv.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        w = None
        if self.impl == "sdpa" and not return_attn:
            a = F.scaled_dot_product_attention(q, k, v)
        else:
            w = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).softmax(-1)
            a = w @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ffn(self.nf(x)), w


class GeometryCrossAttentionPINN(nn.Module):
    """`nurbs` fixes the token layout (x_ctrl, mask, mode): symmetric -> k upper tokens (D = k), cambered -> 2k.
    attn_impl: 'explicit' (default; SDPA has no CPU double-backward) | 'sdpa' (works on MPS, not for CPU derivs)."""

    def __init__(self, nurbs, d_model: int = 128, heads: int = 4, n_self: int = 2, n_cross: int = 2, v2: bool = False,
                 fourier: int = 64, scale: float = 2.0, ffn_mult: int = 4, psi_scale: float = 3.0,
                 attn_impl: str = "explicit", seed: int = 0):
        super().__init__()
        assert attn_impl in ("sdpa", "explicit")
        self.psi_scale, self.cambered, self.k = psi_scale, nurbs.mode == "cambered", nurbs.k
        self.register_buffer("x_ctrl", nurbs.x_ctrl.detach().clone().float())
        self.register_buffer("mask", nurbs.mask.detach().clone().float())
        D = nurbs.D
        self.ff, self.norm = FourierFeatures(fourier, scale, seed), InputNorm(D)
        d, f = d_model, ffn_mult * d_model
        self.geom, self.cond = _embed(3, d), _embed(3, d)
        self.query = nn.Sequential(nn.Linear(2 * fourier + 1, d), Wavelet(d), nn.Linear(d, d))
        self.self_blocks = nn.ModuleList(_Block(d, heads, f, attn_impl) for _ in range(n_self))
        self.cross_blocks = nn.ModuleList(_Block(d, heads, f, attn_impl) for _ in range(n_cross))
        self.head = nn.Sequential(nn.Linear(d, d), Wavelet(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 4 if v2 else 2))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def tokens(self, alpha):
        """alpha [B, D] (raw) -> [B, D, 3] = (x_c, normalized y_c, surface_id); matches NURBSAirfoil.control_tokens
        (upper tokens only in symmetric mode, upper then lower in cambered)."""
        a, _ = self.norm(alpha * self.mask.to(alpha.dtype), alpha.new_zeros(alpha.shape[0], 3))
        x = self.x_ctrl.to(alpha.dtype).expand(alpha.shape[0], -1)
        if self.cambered:
            x, sid = torch.cat([x, x], 1), torch.cat([torch.zeros_like(x), torch.ones_like(x)], 1)
        else:
            sid = torch.zeros_like(x)
        return torch.stack([x, a, sid], -1)

    def encode_geometry(self, alpha, cond):
        """-> G [B, D+1, d]; compute once per shape and pass to forward(G=...) for many point queries."""
        _, c = self.norm(alpha, cond)
        G = torch.cat([self.geom(self.tokens(alpha)), self.cond(c)[:, None]], 1)
        for blk in self.self_blocks:
            G = blk(G, G)[0]
        return G

    def forward(self, xy, alpha, cond, d_wall, return_attn: bool = False, G=None) -> PINNOutput:
        if G is None:
            G = self.encode_geometry(alpha, cond)
        h, w = self.query(torch.cat([self.ff(xy), d_wall[..., None]], -1)), None
        for blk in self.cross_blocks:
            h, w = blk(h, G, return_attn)
        o = self.head(h)
        return PINNOutput(psi=self.psi_scale * o[..., 0], p=o[..., 1], fx=o[..., 2] if o.shape[-1] == 4 else None,
                          fy=o[..., 3] if o.shape[-1] == 4 else None, attn=w if return_attn else None)
