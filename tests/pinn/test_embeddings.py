import torch

from aeroshape.pinn.embeddings import FourierFeatures, InputNorm, Wavelet


def test_fourier_fixed_and_seeded():
    a, b = FourierFeatures(16, 2.0, seed=3), FourierFeatures(16, 2.0, seed=3)
    xy = torch.rand(2, 5, 2)
    assert a(xy).shape == (2, 5, 32) and torch.equal(a(xy), b(xy))
    assert not any(True for _ in a.parameters()) and "B" in a.state_dict()
    assert not torch.equal(a.B, FourierFeatures(16, 2.0, seed=4).B)


def test_wavelet_learnable():
    w = Wavelet(4)
    z = torch.randn(3, 4)
    assert torch.allclose(w(z), z.sin() + z.cos())
    w(z).sum().backward()
    assert w.w1.grad is not None and w.w2.grad is not None


def test_input_norm_state_dict_roundtrip():
    n = InputNorm(8)
    a, c = torch.randn(50, 8) * 3 + 1, torch.randn(50, 3) * 2 - 1
    n.fit(a, c)
    na, nc = n(a, c)
    assert na.mean().abs() < 1e-5 and (na.std(0) - 1).abs().max() < 1e-4 and (nc.std(0) - 1).abs().max() < 1e-4
    m = InputNorm(8)
    m.load_state_dict(n.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(m(a, c), (na, nc)))
    n.fit(torch.zeros(4, 8), torch.zeros(4, 3))  # constant columns must not divide by zero
    assert torch.isfinite(n(a, c)[0]).all()


def test_input_norm_constant_dim_stays_bounded():  # L7: constant AoA in training, AoA 10 deg at inference
    n = InputNorm(8)
    c = torch.zeros(20, 3)
    c[:, 0] = torch.rand(20)
    n.fit(torch.randn(20, 8), c)
    z = n(torch.zeros(1, 8), torch.tensor([[0.5, 0.17, 0.98]]))[1]
    assert z.abs().max() < 1e3
