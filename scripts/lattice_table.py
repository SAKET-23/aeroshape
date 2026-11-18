"""Prints the lattice-parameter table of Plan/03_lbm/forward_solver.md."""
import argparse
import math

U_LB = 0.1 / math.sqrt(3)
# (body, Re, tier, N)
ROWS = [("Cylinder", re, t, n) for re in (20, 40, 100) for t, n in (("smoke", 10), ("dev", 20), ("full", 40))]
ROWS += [("Airfoil", 5, "dev", 20)]
ROWS += [("Airfoil", 20, t, n) for t, n in (("smoke", 10), ("dev", 20), ("full", 40))]
ROWS += [("Airfoil", 1000, t, n) for t, n in (("smoke", 24), ("dev", 48), ("full", 100))]


def cylinder_geometry(N, device, dtype=None):
    """400x200-style grid, cylinder of diameter N centred at (5N, 5N); links by ray-circle intersection."""
    import torch
    from aeroshape.core.types import GridGeometry, LinkSet
    from aeroshape.lbm.lattice import C

    dtype = dtype or torch.float32
    Nx, Ny, R = 20 * N, 10 * N, N / 2
    jj, ii = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing="ij")
    px, py = ii.double() - 5 * N, jj.double() - Ny / 2
    solid = px**2 + py**2 < R**2
    nodes, dirs, qs, fars = [], [], [], []
    for d in range(1, 9):
        cx, cy = int(C[d, 0]), int(C[d, 1])
        nb = torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1))  # solid at x + c_d
        far = ~torch.roll(solid, shifts=(cy, cx), dims=(0, 1))  # fluid at x - c_d
        m = ~solid & nb
        b = px * cx + py * cy
        t = -b + torch.sqrt((b**2 - (cx**2 + cy**2) * (px**2 + py**2 - R**2)).clamp(min=0))
        t = t / (cx**2 + cy**2)
        nodes.append((jj * Nx + ii)[m]); dirs.append(torch.full((int(m.sum()),), d))
        qs.append(t[m].clamp(1e-3, 1)); fars.append(far[m])
    node, d = torch.cat(nodes), torch.cat(dirs)
    links = LinkSet(node.to(device), d.to(device), torch.cat(qs).to(device=device, dtype=dtype),
                    torch.cat(fars).to(device), torch.zeros_like(node).to(device))
    return GridGeometry(solid.to(device), links, N, (5 * N, Ny // 2))


def benchmark(device_name):
    import json
    import pathlib

    import torch
    from aeroshape.core.config import load_config
    from aeroshape.core.device import device_info, get_device
    from aeroshape.core.timing import Timer
    from aeroshape.core.types import Condition
    from aeroshape.lbm import d2q9_solver as S
    from aeroshape.lbm.lattice import make_params

    dev = get_device(device_name)
    p = make_params(load_config(None, "dev", []), Condition(20, 0), 20, device=dev)
    geom = cylinder_geometry(20, dev)
    f = S.init_state(p, geom)
    n_warm, n = 50, 500
    with torch.no_grad():
        for _ in range(n_warm):
            f, _ = S.step(f, geom, p)
        with Timer(dev) as t:
            for _ in range(n):
                f, _ = S.step(f, geom, p)
    mlups = p.Nx * p.Ny * n / t.elapsed / 1e6
    out = pathlib.Path("runs/perf/lbm_mlups.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[dev.type] = {"mlups": mlups, "grid": [p.Nx, p.Ny], "steps": n, "links": int(geom.links.node.numel()),
                      "finite": bool(torch.isfinite(f).all()), **device_info(dev)}
    out.write_text(json.dumps(data, indent=2))
    print(f"{dev.type}: {mlups:.1f} MLUPS ({p.Nx}x{p.Ny}, {n} steps, {t.elapsed:.2f}s)")


def fit_report():
    import numpy as np
    import torch
    from aeroshape.geometry import reference_shapes as rs
    from aeroshape.geometry.nurbs import NURBSAirfoil

    print("| shape | mode | k | max err | rms err |\n|---|---|---|---|---|")
    chosen = None
    for name, mode in (("naca0012", "symmetric"), ("naca2412", "cambered"), ("rae2822", "cambered")):
        c = rs.reference_coords(name)
        for k in (8, 10, 12, 16):
            n = NURBSAirfoil(k=k, mode=mode, dtype=torch.float64)
            x, yu, yl = n.surface_y(rs.fit_alpha(n, c["x_up"], c["y_up"], c["x_lo"], c["y_lo"]))
            x = x.numpy()
            e = np.concatenate([yu[0].numpy() - np.interp(x, c["x_up"], c["y_up"]),
                                (yl[0].numpy() - np.interp(x, c["x_lo"], c["y_lo"]))])
            mx = np.abs(e).max()
            print(f"| {name} | {mode} | {k} | {mx:.2e} | {np.sqrt((e**2).mean()):.2e} |")
            if name == "rae2822" and chosen is None and mx < 2e-3:
                chosen = k
    print(f"smallest cambered k with RAE2822 max error < 2e-3: {chosen}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--fit-report", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    if a.fit_report:
        fit_report()
    if a.benchmark:
        benchmark(a.device)
    if a.fit_report or a.benchmark:
        return
    print(f"U_lb = {U_LB:.5f}")
    print("| Body | Re | Tier | N | Grid | Nodes | nu_lb | tau | Steps/conv |")
    print("|---|---|---|---|---|---|---|---|---|")
    for body, re, tier, n in ROWS:
        nu = U_LB * n / re
        nodes = 20 * n * 10 * n
        print(f"| {body} | {re} | {tier} | {n} | {20*n} x {10*n} | {nodes/1000:.0f}k | "
              f"{nu:.5f} | {3*nu+0.5:.4f} | {round(n/U_LB)} |")


if __name__ == "__main__":
    main()
