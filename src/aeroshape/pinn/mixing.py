"""v2 training set = v1 steady symmetric runs (Re 20) + time-averaged cambered runs (Re 1000), in the cambered D = 2k layout."""
import torch

Tensor = torch.Tensor
_PER_RUN = ("re", "aoa", "xy", "edge", "u", "v", "p", "cd", "cl", "split")


def to_cambered(alpha: Tensor) -> Tensor:
    """[.., k] symmetric half thickness -> [.., 2k] cambered (upper = alpha, lower = -alpha; NURBSAirfoil._split)."""
    return torch.cat([alpha, -alpha], -1)


def sym_grad(g: Tensor) -> Tensor:
    """[.., 2k] cambered gradient of f(to_cambered(a)) -> [.., k] gradient w.r.t. the symmetric alpha (chain rule)."""
    k = g.shape[-1] // 2
    return g[..., :k] - g[..., k:]


def mix_anchors(steady: dict, tavg: dict) -> dict:
    """Merged anchors with kind 0 (steady, train split only: its holdout is dropped so the holdout stays Re 1000) then kind 1."""
    assert steady["mode"] == "symmetric" and tavg["mode"] == "cambered" and steady["xy"].shape[1:] == tavg["xy"].shape[1:]
    keep = steady["split"] == 0
    out = {k: torch.cat([steady[k][keep], tavg[k]]) for k in _PER_RUN}
    out["alpha"] = torch.cat([to_cambered(steady["alpha"][keep]), tavg["alpha"]])
    out["kind"] = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.int8), torch.ones(len(tavg["re"]), dtype=torch.int8)])
    runs = [r for r, k in zip(steady["meta"]["runs"], keep.tolist()) if k] + list(tavg["meta"]["runs"])
    out.update(mode="cambered", meta={**tavg["meta"], "runs": runs})
    return out
