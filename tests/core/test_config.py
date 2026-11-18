import pytest

from aeroshape.core.config import dump_config, load_config


def test_tiers():
    assert load_config(None, "smoke", []).lbm.cells_per_d == 10
    assert load_config(None, "full", []).train.adam_steps == 20000
    assert load_config(None, "dev", []).lbm.cells_per_d_high_re == 48


def test_override_and_parse():
    c = load_config(None, "smoke", ["lbm.cells_per_d=12", "lbm.sponge.enabled=true", "geometry.mode=cambered"])
    assert c.lbm.cells_per_d == 12 and c.lbm.sponge.enabled is True
    assert c.geometry.mode == "cambered"


def test_unknown_override():
    with pytest.raises(KeyError):
        load_config(None, "smoke", ["lbm.nope=1"])


def test_case_precedes_override(tmp_path):
    case = tmp_path / "c.yaml"
    case.write_text("lbm: {collision: trt}\nstart: circle\n")
    c = load_config(str(case), "smoke", ["lbm.cells_per_d=11"])
    assert c.lbm.collision == "trt" and c.start == "circle" and c.lbm.cells_per_d == 11


def test_dump_roundtrip():
    import yaml
    c = load_config(None, "dev", [])
    assert yaml.safe_load(dump_config(c))["pinn"]["d_model"] == 128
