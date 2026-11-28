"""Phase 5A: does the Re 1000 time-averaged LBM adjoint CD gradient have a usable DIRECTION?
Compares a cold CD-only adjoint (per horizon) with central FD of the time-averaged CD on a few free control points.
uv run python scripts/09_check_re1000_gradient.py --device mps  -> runs/diagnostics/re1000_gradient.json"""
import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import load_config  # noqa: E402
from aeroshape.core.device import device_info, get_device  # noqa: E402
from aeroshape.core.types import Condition, CostWeights  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.lbm.adjoint_solver import fd_gradient, lbm_value_and_grad  # noqa: E402

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("opt06c", ROOT / "scripts" / "06_optimize_case.py")
opt06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt06)

CKPT = ROOT / "runs/benchmark/dev/case2_adjoint_lbm/state.pt"
CASES = {"case3_start": "configs/case3_naca_to_rae_aoa10.yaml", "case2_cycle7": "configs/case2_circle_to_rae_re1000.yaml"}


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def compare(adj, fd):
    adj, fd = adj.detach().cpu().double(), fd.detach().cpu().double()  # MPS has no float64
    cos = float(adj @ fd / (adj.norm() * fd.norm() + 1e-300))
    return dict(cosine=cos, sign_match=float((adj.sign() == fd.sign()).double().mean()), norm_ratio=float(adj.norm() / (fd.norm() + 1e-300)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--tier", default="dev")
    ap.add_argument("--shapes", default="case3_start,case2_cycle7")
    ap.add_argument("--idx", default=None, help="comma list of free control-point indices (default: 4 spread over the free ones)")
    ap.add_argument("--delta", type=float, default=5e-3)
    ap.add_argument("--horizons", default="20,10,5")
    ap.add_argument("--out", default="runs/diagnostics/re1000_gradient.json")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    a = ap.parse_args(argv)

    device = get_device(a.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(out.read_text()) if out.exists() else {}
    except ValueError:
        old = {}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str))  # noqa: E731
    res = {"device": device_info(device), "delta": a.delta, "sets": a.sets, "shapes": {}}
    print(f"{'shape':13s} {'horizon':>7s} {'cos':>7s} {'sign':>5s} {'|adj|/|fd|':>10s} {'cd':>8s} {'cl':>8s} {'t_adj':>7s}")
    for name in a.shapes.split(","):
        cfg = load_config(CASES[name], a.tier, a.sets)
        c, g = cfg.case, cfg.geometry
        cond, N = Condition(c.re, c.aoa), cfg.lbm.cells_per_d_high_re
        nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=dtype)
        if name == "case3_start":
            alpha = opt06.shape(nurbs, c.start)
        elif CKPT.exists():
            alpha = torch.load(CKPT, weights_only=False)["records"][6]["alpha"].to(nurbs.mask)
        else:
            print(f"{name}: skipped, checkpoint {CKPT} missing")
            res["shapes"][name] = {"skipped": f"missing {CKPT}"}
            save()
            continue
        free = nurbs.free_idx
        idx = [int(i) for i in a.idx.split(",")] if a.idx else [int(free[round(i)]) for i in torch.linspace(0, len(free) - 1, min(4, len(free))).tolist()]
        prev = old.get("shapes", {}).get(name, {})
        if prev.get("fd") and prev.get("idx") == idx and old.get("delta") == a.delta and old.get("sets") == a.sets:
            print(f"{name}: reusing FD gradient from {out}")
            fd, r = torch.tensor(prev["fd"]), {"idx": idx, "fd": prev["fd"], "t_fd": prev.get("t_fd"), "horizons": {}}
        else:
            t = time.perf_counter()
            fd = fd_gradient(alpha, nurbs, cond, cfg, a.delta, idx, None, N).detach().reshape(-1)
            sync(device)
            r = {"idx": idx, "fd": fd.detach().cpu().tolist(), "t_fd": time.perf_counter() - t, "horizons": {}}
        res["shapes"][name] = r
        save()
        for h in [float(x) for x in a.horizons.split(",")]:
            cfg = load_config(CASES[name], a.tier, [*a.sets, f"adjoint.horizon_convective_times={h}"])  # Config is frozen
            t = time.perf_counter()
            ev = lbm_value_and_grad(alpha, nurbs, cond, cfg, CostWeights(0.0, 0.0, 0.0), N)
            sync(device)
            adj = ev.extras["dI_dalpha"].detach().reshape(-1)[idx]
            row = dict(compare(adj, fd), adj=adj.cpu().tolist(), cd=float(ev.cd), cl=float(ev.cl), t_adjoint=time.perf_counter() - t)
            r["horizons"][str(h)] = row
            print(f"{name:13s} {h:7g} {row['cosine']:7.3f} {row['sign_match']:5.2f} {row['norm_ratio']:10.3f} {row['cd']:8.5f} {row['cl']:8.5f} {row['t_adjoint']:6.1f}s")
            save()
    save()
    return 0


if __name__ == "__main__":
    main()
