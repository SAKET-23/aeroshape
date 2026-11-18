import torch

from aeroshape.diffusion.denoiser import ShapeDenoiser
from aeroshape.diffusion.schedule import CosineSchedule
from aeroshape.diffusion.guided_shape_diffusion import diffusion_loss, project_thickness
from tests.pinn import make_nurbs, thin_alpha


def make_denoiser(nurbs, **kw):
    xc = nurbs.x_ctrl[nurbs.free_idx % nurbs.k]
    sid = (nurbs.free_idx // nurbs.k).long()
    return ShapeDenoiser(nurbs.n_free, xc, sid, **kw)


def test_shapes_null_and_cond():
    torch.manual_seed(0)
    n = make_nurbs()
    m = make_denoiser(n, d=32, layers=2).eval()
    z, t, c = torch.randn(5, n.n_free), torch.randint(0, 1000, (5,)), torch.randn(5, 4)
    e_c, e_n = m(z, t, c), m(z, t, None)
    assert e_c.shape == e_n.shape == z.shape
    assert not torch.allclose(e_c, e_n, atol=1e-4)
    assert torch.allclose(m(z, t, c, null_mask=torch.ones(5, dtype=torch.bool)), e_n, atol=1e-6)


def test_diffusion_loss_backprop():
    n = make_nurbs()
    m = make_denoiser(n, d=32, layers=2)
    loss = diffusion_loss(m, CosineSchedule(), torch.randn(8, n.n_free), torch.randn(8, 4), 0.1, torch.Generator().manual_seed(0))
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss) and m.out.weight.grad is not None


def test_project_thickness():
    n = make_nurbs(dtype=torch.float64)
    a = thin_alpha(3).double()
    t_min = torch.full((n.k,), 0.1, dtype=torch.float64)
    t_min[[0, 1, -1]] = 0  # thickness is 0 at the LE controls (x = 0) and TE
    assert (n.thickness_at_ctrl(a) < t_min - 1e-3).any()
    out = project_thickness(n, a, t_min)
    assert (n.thickness_at_ctrl(out) >= t_min - 1e-6).all()
