import importlib.util
from pathlib import Path

import pytest
import torch

from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.geometry.reference_shapes import circle_alpha, shape_distance

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts" / "03b_pinn_checks.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

pytestmark = pytest.mark.skipif(not (ROOT / chk.TRAJ).exists(), reason="adjoint trajectory not on disk")


@pytest.fixture(scope="module")
def shapes():
    ref = NURBSAirfoil(k=8, dtype=torch.float64)
    return ref, chk.v1c_shapes(ref)


def test_away_from_trajectory_and_circle(shapes):
    ref, a = shapes
    traj = torch.load(ROOT / chk.TRAJ, weights_only=False)["alpha"].double().reshape(-1, 8)
    assert len(a) == 30
    assert min(float(shape_distance(ref, a[i:i + 1].double().expand(len(traj), -1), traj).min()) for i in range(len(a))) >= 0.005
    d_circ = shape_distance(ref, a.double(), circle_alpha(ref).expand(len(a), -1))
    assert (d_circ[20:] > 0.01).all()


def test_thickness_range(shapes):
    ref, a = shapes
    t = chk.g02.thickness(ref, a.double())
    assert (t[20:] >= 0.8).all() and (t[20:] <= 1.1).all()
    assert t[:20].min() < 0.3 and t[:20].max() > 0.9  # morphs span airfoil to near-circle
