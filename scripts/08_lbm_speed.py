"""L3: forward-LBM step speed, eager vs torch.compile vs CUDA graph (Re 1000 case-2 settings, circle body).
uv run python scripts/08_lbm_speed.py --device cuda --re 1000 --N 48 --steps 200  -> runs/lbm_speed_<device>.json"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Boundary, Condition
from aeroshape.geometry.ibb_grid_mapper import map_to_grid
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.lattice import make_params

OV = ["lbm.collision=trt", "lbm.trt_magic=0.001", "lbm.outlet=extrapolate", "lbm.sponge.enabled=true"]


def circle(device, r=0.5, n=2048):
    th = (torch.arange(n + 1, dtype=torch.float64) + 0.5) * (2 * math.pi / n)
    x, y = (r * torch.cos(th))[None].float().to(device), (r * torch.sin(th))[None].float().to(device)
    z = torch.zeros_like(x)
    return Boundary(x, y, z, z, z)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--re", type=float, default=1000)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tier", default="dev")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cfg = load_config(None, a.tier, OV)
    L = cfg.lbm
    p = make_params(cfg, Condition(a.re, 0), a.N, device=dev, dtype=torch.float32)
    g = map_to_grid(circle(dev), p, body_offset_d=L.body_offset_d)
    f0 = S.init_state(p, g, torch.float32, perturb=0.1)
    variants = {"eager": {}, "compile": {"compile": True}}
    if dev.type == "cuda":
        variants["cuda_graph"] = {"cuda_graph": True, "graph_steps": L.graph_steps}
    out = {"device": a.device, "re": a.re, "N": a.N, "grid": [p.Ny, p.Nx], "steps": a.steps}
    for name, kw in variants.items():
        run = lambda n: S.run(f0, g, p, n_steps=n, outlet=L.outlet, **kw)  # noqa: E731  (no force_fn: pure stepping)
        try:
            warnings.simplefilter("ignore")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run(2)  # probe: did compile fall back to eager?
            if any("torch.compile failed" in str(x.message) for x in w):
                raise RuntimeError("torch.compile unavailable, fell back to eager (timing skipped)")
            run(min(a.steps, max(L.graph_steps if kw.get("cuda_graph") else 2, 2)))  # warm-up (compile / capture)
            sync(dev)
            t = time.perf_counter()
            run(a.steps)
            sync(dev)
            dt = (time.perf_counter() - t) / a.steps
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{name:11s} failed: {out[name]['error']}")
            continue
        out[name] = {"ms_per_step": dt * 1e3, "mlups": p.Ny * p.Nx / dt / 1e6}
        print(f"{name:11s} {dt * 1e3:9.3f} ms/step {out[name]['mlups']:9.2f} MLUPS")
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/lbm_speed_{a.device}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
