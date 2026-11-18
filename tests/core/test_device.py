import pytest
import torch

from aeroshape.core.device import device_info, get_device, get_dtype


def test_pref_beats_env(monkeypatch):
    monkeypatch.setenv("AEROSHAPE_DEVICE", "mps")
    assert get_device("cpu").type == "cpu"
    assert get_device().type == "mps" if torch.backends.mps.is_available() else True


def test_env(monkeypatch):
    monkeypatch.setenv("AEROSHAPE_DEVICE", "cpu")
    assert get_device().type == "cpu"
    assert get_device("auto").type == "cpu"


def test_dtype():
    cpu = torch.device("cpu")
    assert get_dtype(cpu, "check") == torch.float64
    assert get_dtype(cpu, "sim") == torch.float32
    assert get_dtype(cpu, "train") == torch.float32
    with pytest.raises(ValueError):
        get_dtype(torch.device("mps"), "check")


def test_info():
    info = device_info(torch.device("cpu"))
    assert {"name", "torch", "memory"} <= info.keys()
