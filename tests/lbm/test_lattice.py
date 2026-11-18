import math

import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Condition
from aeroshape.lbm.lattice import OPP, lattice_constants, lbm_to_nondim, make_params


def test_constants():
    C, W, opp = lattice_constants(torch.device("cpu"), torch.float64)
    assert W.sum().item() == 1 and (C[opp] == -C).all() and (opp[opp] == torch.arange(9)).all()
    assert (W[:, None] * C).sum(0).abs().max() < 1e-15
    assert OPP.tolist() == [0, 3, 4, 1, 2, 7, 8, 5, 6]


def test_make_params_table_row():  # Cylinder Re 20 dev: nu 0.05774, tau 0.6732, 400x200
    p = make_params(load_config(None, "dev", []), Condition(20, 0), 20)
    assert (p.Nx, p.Ny) == (400, 200)
    assert abs(p.tau - 0.6732) < 1e-4 and abs(p.u_lb - 0.1 / math.sqrt(3)) < 1e-12
    assert p.sponge is None and p.omega_minus == p.omega


def test_trt_and_sponge():
    cfg = load_config(None, "dev", ["lbm.collision=trt", "lbm.sponge.enabled=true"])
    p = make_params(cfg, Condition(100, 0), 20)
    assert abs((1 / p.omega - 0.5) * (1 / p.omega_minus - 0.5) - 3 / 16) < 1e-12
    assert p.sponge.shape == (200, 400) and p.sponge[0, 0].item() > p.sponge[0, -1].item()


def test_nondim():
    u, v, p = lbm_to_nondim(torch.tensor(1.0), torch.tensor(0.05), torch.tensor(0.0), 0.05)
    assert (u.item(), v.item(), p.item()) == (1.0, 0.0, 0.0)
