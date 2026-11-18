import copy

import torch

from aeroshape.core.io import check_anchors, merge_anchors
from tests.pinn import synthetic_anchors


def _one(a, i):
    return {k: (v[i] if isinstance(v, torch.Tensor) else v) for k, v in a.items()}


def test_synthetic_anchors_pass():
    assert check_anchors(synthetic_anchors()) == []


def test_missing_key_nan_and_shape_are_caught():
    a = synthetic_anchors()
    b = {k: v for k, v in a.items() if k != "cd"}
    assert any("cd" in m for m in check_anchors(b))
    b = copy.deepcopy(a)
    b["u"][0, 0] = float("nan")
    assert any("u" in m and "finite" in m for m in check_anchors(b))
    b = copy.deepcopy(a)
    b["xy"] = b["xy"][..., :1]
    assert any("xy" in m for m in check_anchors(b))
    b = copy.deepcopy(a)
    b["edge"] = b["edge"].float()
    assert any("edge" in m for m in check_anchors(b))


def test_merge_stacks_per_shape_dicts():
    a = synthetic_anchors(S=3)
    m = merge_anchors([_one(a, i) for i in range(3)])
    assert m["xy"].shape == a["xy"].shape and m["alpha"].shape == a["alpha"].shape and m["split"].shape == (3,)
    assert check_anchors(m) == []
