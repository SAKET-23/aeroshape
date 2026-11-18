import torch

from aeroshape.diffusion.schedule import CosineSchedule


def test_gamma_bar_shape_and_ends():
    s = CosineSchedule(T=1000)
    g = s.gamma_bar
    assert g.shape == (1001,)
    assert abs(g[0].item() - 1) < 1e-9 and abs(g[-1].item()) < 1e-9
    assert (g[1:] < g[:-1]).all()


def test_q_sample_moments():
    s = CosineSchedule(T=1000)
    t = torch.full((100000,), 300)
    z0 = torch.full((100000, 1), 1.5)
    g = torch.Generator().manual_seed(0)
    zt = s.q_sample(z0, t, torch.randn(100000, 1, generator=g))
    gb = s.gamma_bar[300].item()
    assert abs(zt.mean().item() - gb**0.5 * 1.5) < 0.01
    assert abs(zt.var().item() - (1 - gb)) < 0.01


def test_ddim_timesteps():
    ts = CosineSchedule(T=1000).ddim_timesteps(50)
    assert len(ts) == 51 and ts[-1] == 0 and ts[0] < 1000 and (ts[1:] < ts[:-1]).all()
