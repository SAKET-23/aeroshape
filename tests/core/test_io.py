import torch

from aeroshape.core.io import load_ckpt, save_ckpt


def test_roundtrip(tmp_path, device):
    p = tmp_path / "a" / "c.pt"
    save_ckpt(p, {"w": torch.arange(3.0, device=device), "n": 1}, device=str(device))
    ck = load_ckpt(p, device)
    assert ck["w"].device.type == device.type
    assert ck["n"] == 1 and {"torch", "git", "device"} <= ck.keys()
    assert load_ckpt(p, "cpu")["w"].device.type == "cpu"
