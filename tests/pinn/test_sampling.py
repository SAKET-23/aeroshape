import torch
from scipy.stats import kstest

from aeroshape.geometry.ibb_grid_mapper import points_in_polygon
from aeroshape.pinn.embeddings import wall_distance
from aeroshape.pinn.sampling import inside_body, near_wall, sample_batch
from tests.pinn import make_cfg, make_nurbs, synthetic_anchors


def test_shapes_and_no_collocation_inside_body():
    cfg, nurbs = make_cfg(), make_nurbs()
    b = sample_batch(synthetic_anchors(), nurbs, cfg, torch.Generator().manual_seed(0))
    B, Nc = cfg.train.shapes_per_step, cfg.sampling.n_collocation
    assert b.xy_colloc.shape == (B, Nc, 2) and b.xy_surface.shape == (B, cfg.sampling.n_surface, 2)
    assert b.xy_edge.shape == (B, cfg.sampling.n_edge, 2) and b.uv_edge.shape == (B, cfg.sampling.n_edge, 2)
    assert b.xy_anchor.shape == (B, cfg.sampling.n_anchor, 2) and b.kind.shape == (B,)
    assert not inside_body(b.xy_colloc, b.boundary).any()
    assert not points_in_polygon(b.xy_colloc[..., 0], b.xy_colloc[..., 1], b.boundary.x, b.boundary.y).any()
    # surface points lie on the loop
    assert wall_distance(b.xy_surface, b.boundary).max() < 0.05


def test_near_wall_log_uniform():
    nurbs = make_nurbs()
    bd = nurbs(torch.zeros(1, 8).index_fill_(1, torch.arange(1, 7), 0.06))
    xy, d = near_wall(bd, 4000, torch.Generator().manual_seed(0), 0.002, 0.5)
    cdf = lambda x: torch.log(torch.as_tensor(x) / 0.002).div(torch.log(torch.tensor(250.0))).numpy()
    assert kstest(d.flatten().numpy(), cdf).pvalue > 0.01
    # offset along the normal: true wall distance never exceeds d (plus jitter slack)
    assert (wall_distance(xy, bd) <= 1.1 * d + 1e-3).all()


def test_reproducible_across_generators():
    cfg, nurbs, a = make_cfg(), make_nurbs(), synthetic_anchors()
    b1 = sample_batch(a, nurbs, cfg, torch.Generator().manual_seed(3))
    b2 = sample_batch(a, nurbs, cfg, torch.Generator().manual_seed(3))
    assert torch.equal(b1.xy_colloc, b2.xy_colloc)
