"""WP17 round 2: the guidance step must be a first-order descent step on J(alpha0_hat) in the space where it is applied (z0_hat)."""
import torch

from aeroshape.diffusion.guided_shape_diffusion import GuidedSampler
from aeroshape.diffusion.schedule import CosineSchedule
from tests.pinn import make_nurbs


class Shrink(torch.nn.Module):
    """eps such that z0_hat = -0.5 z_t exactly: the denoiser Jacobian dz0_hat/dz_t is negative definite, so a gradient taken w.r.t. z_t
    and applied to z0_hat would climb J."""

    def __init__(self, sched):
        super().__init__()
        self.g = sched.gamma_bar

    def forward(self, z_t, t, cond, null_mask=None):
        gb = self.g[t.cpu()].to(z_t)[:, None]
        return (z_t + 0.5 * gb.sqrt() * z_t) / (1 - gb).sqrt()


def make(dtype=torch.float64):
    n, s = make_nurbs(dtype=dtype), CosineSchedule()
    mean, std = torch.zeros(n.n_free, dtype=dtype), torch.full((n.n_free,), 0.7, dtype=dtype)
    phys = lambda a, c: (a.sum(1), torch.zeros_like(a[:, 0]))  # stand-in: CD = sum(alpha), CL = 0 -> J quadratic in alpha
    return n, GuidedSampler(Shrink(s), s, n, None, (mean, std), physics_fn=phys), torch.tensor([[0.0, 5.0, 0.0, 0.0]], dtype=dtype).repeat(3, 1)


def test_tiny_guided_step_lowers_J():
    n, smp, cond = make()
    tm = torch.full((n.k,), -10.0, dtype=torch.float64)
    kw = dict(n_steps=2, guide_start=0.5, cfg_scale=0.0, guide_clip=1e6, t_min=tm, cond=cond)  # step 1 (mid t) is the only guided one
    J = lambda gs: smp.objective(smp.sample(guide_scale=gs, generator=torch.Generator().manual_seed(0), **kw).alpha, cond, tm)
    assert (J(1e-3) < J(0.0)).all()


def test_guidance_gradient_matches_finite_difference():
    n, smp, cond = make()
    tm = torch.full((n.k,), -10.0, dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    z0, d = torch.randn(3, n.n_free, generator=g).double(), torch.randn(3, n.n_free, generator=g).double()
    grad, _ = smp.guidance_grad(z0, cond, tm)
    f = lambda z: smp.objective(smp.to_alpha(z), cond, tm)
    h = 1e-6
    fd = (f(z0 + h * d) - f(z0 - h * d)) / (2 * h)
    assert torch.allclose((grad * d).sum(1), fd, rtol=1e-6, atol=1e-9)
