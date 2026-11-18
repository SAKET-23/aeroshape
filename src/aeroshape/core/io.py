import subprocess
from pathlib import Path

import torch


def _to_cpu(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu()
    if isinstance(o, dict):
        return {k: _to_cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_cpu(v) for v in o)
    return o


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_ckpt(path, obj: dict, device: str | None = None) -> None:
    """Tensors -> CPU; adds torch version, git hash, device."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _to_cpu(obj)
    out.update(torch=torch.__version__, git=_git_hash(), device=device or "unknown")
    torch.save(out, path)


def load_ckpt(path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)  # our own files only


_ANCHOR_KEYS = ("alpha", "mode", "re", "aoa", "xy", "edge", "u", "v", "p", "cd", "cl", "split", "kind", "meta")


def check_anchors(d: dict) -> list[str]:
    """Contract problems of a merged anchor dict (data_contracts.md); empty list = OK."""
    bad = [f"missing key: {k}" for k in _ANCHOR_KEYS if k not in d]
    if bad:
        return bad
    S, P = d["xy"].shape[0], d["xy"].shape[1]
    shapes = {"alpha": (S, d["alpha"].shape[-1]), "re": (S,), "aoa": (S,), "xy": (S, P, 2), "edge": (S, P),
              "u": (S, P), "v": (S, P), "p": (S, P), "cd": (S,), "cl": (S,), "split": (S,), "kind": (S,)}
    for k, s in shapes.items():
        if tuple(d[k].shape) != s:
            bad.append(f"{k}: shape {tuple(d[k].shape)} != {s}")
    if d["edge"].dtype != torch.bool:
        bad.append("edge: dtype must be bool")
    if d["mode"] not in ("symmetric", "cambered"):
        bad.append(f"mode: {d['mode']!r}")
    bad += [f"{k}: not finite" for k in ("alpha", "re", "aoa", "xy", "u", "v", "p", "cd", "cl")
            if d[k].is_floating_point() and not bool(torch.isfinite(d[k]).all())]
    return bad


def merge_anchors(items: list[dict]) -> dict:
    """Stack per-shape anchor dicts (no S axis) into the merged contract; `meta` becomes the first item's meta with
    meta['runs'] = every item's meta (in order)."""
    out = {k: torch.stack([it[k] for it in items]) for k in _ANCHOR_KEYS if isinstance(items[0].get(k), torch.Tensor)}
    out["mode"] = items[0]["mode"]
    out["meta"] = {**items[0]["meta"], "runs": [it["meta"] for it in items]}
    return out
