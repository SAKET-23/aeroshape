"""WP 10: LBM anchor dataset + Gate 2b (Plan/06_training/dataset_generation.md).
uv run python scripts/02_generate_lbm_dataset.py --set v1 --tier dev --device mps [--shard 0/2]
uv run python scripts/02_generate_lbm_dataset.py --set v1 --tier dev --merge
uv run python scripts/02_generate_lbm_dataset.py --set v1 --tier dev --check [--blockage]
Config overrides: -o key=value (repeatable); `--set` selects the dataset set (v1 | v2)."""
import argparse
import datetime
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import load_config  # noqa: E402
from aeroshape.core.device import get_device, get_dtype  # noqa: E402
from aeroshape.core.io import _git_hash, check_anchors, merge_anchors, save_ckpt  # noqa: E402
from aeroshape.core.timing import Timer  # noqa: E402
from aeroshape.core.types import Condition  # noqa: E402
from aeroshape.geometry.ibb_grid_mapper import map_to_grid  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil, rotate  # noqa: E402
from aeroshape.geometry.reference_shapes import circle_alpha, lhs_shapes, naca4_alpha  # noqa: E402
from aeroshape.lbm import d2q9_solver as S  # noqa: E402
from aeroshape.lbm.aero_forces import force_fn  # noqa: E402
from aeroshape.lbm.lattice import lbm_to_nondim, make_params  # noqa: E402

K_SYM, NEAR_WALL_D, NEAR_FRACTION, CYLINDER_CD_20x10 = 8, 0.2, 0.3, 2.415  # WP 6 cylinder, 20 x 10, N = 20 (G21)


def run_list(cfg, n_re40=10, dset="v1"):
    """[(shape_idx, re, aoa)] v2: Re 1000 at AoA 0 and 10 per shape. v1: Re 20 for every shape, then Re 40 for the first n_re40; v1b: Re 20 only, ids from V1B_BASE."""
    n = cfg.dataset.n_shapes
    if dset == "v2":  # Re 1000 at AoA 0 and 10 for every cambered shape
        return [(i, 1000, aoa) for i in range(n) for aoa in (0, 10)]
    if dset == "v1b":
        return [(V1B_BASE + i, 20, 0) for i in range(n)]
    return [(i, 20, 0) for i in range(n)] + [(i, 40, 0) for i in range(min(n_re40, n))]


V1B_BASE, V1B_T = 100, (0.08, 0.35)  # G31: extra thin-shape set, run ids s100.., all train


def thickness(nurbs, alpha):
    """[n] maximum thickness (chord 1) = 2 max y of the symmetric body."""
    return 2 * nurbs(alpha).y.amax(1)


def thin_shapes(nurbs, n, seed, lo=V1B_T[0], hi=V1B_T[1], pool=4000):
    """[n, D] valid LHS shapes with max thickness in [lo, hi], spread evenly over the thickness range
    (pool of valid candidates -> keep those in range -> n quantile picks by thickness)."""
    a = lhs_shapes(nurbs, pool, seed, family="dataset")
    a = a[(lambda t: (t >= lo) & (t <= hi))(thickness(nurbs, a))]
    if len(a) < n:
        raise ValueError(f"only {len(a)} of {pool} candidates have thickness in [{lo}, {hi}]; raise `pool`")
    a = a[thickness(nurbs, a).argsort()]
    return a[torch.linspace(0, len(a) - 1, n).round().long()]


def shard(runs, i, n):
    return runs[i::n]


def run_name(i, re, aoa):
    return f"s{i:02d}_re{int(re)}_aoa{int(aoa)}.pt"


V2_CASE = ["lbm.collision=trt", "lbm.trt_magic=0.001", "lbm.perturb=0.1", "lbm.outlet=extrapolate",
           "lbm.sponge.enabled=true"]  # G28 / case-2 Re 1000 settings; user -o overrides come after and win


def solve_unsteady(cfg, p, geom):
    """Cold-start unsteady solve. Period found: period-mean f, CD, CL (mode_used 'period_mean'). No period (irregular or
    very slow shedding): mean over the next lbm.window_convective_times convective times ('window_mean'); never raises."""
    c = cfg.lbm
    f0 = S.init_state(p, geom, p_dtype(geom), perturb=c.perturb)
    r = S.run(f0, geom, p, mode="unsteady", force_fn=force_fn, record_every=c.record_every, outlet=c.outlet,
              max_convective_times=c.max_convective_times, average_periods=c.average_periods, transient=c.transient,
              window_convective_times=c.window_convective_times)
    h, K = r.history, round(p.N / p.u_lb)
    span = r.period_steps * c.average_periods if r.period_steps else c.window_convective_times * K
    m = h.step > h.step[-1] - span
    return dict(f=r.f_mean, cd=r.cd, cl=r.cl, converged=True, steps=r.state.step, window_ct=c.window_convective_times,
                mode_used="period_mean" if r.period_steps else "window_mean",
                period=int(r.period_steps) if r.period_steps else None,
                st=p.N / (r.period_steps * p.u_lb) if r.period_steps else None, cl_rms=h.cl[m].pow(2).mean().sqrt().item(), cl_hist=h.cl.float().cpu().tolist())  # CL every record_every steps (diagnosis)


def p_dtype(geom):
    return geom.links.q.dtype


def solve(cfg, nurbs, alpha, re, aoa, N, device, dtype, f0=None, unsteady=False):
    """Steady (BGK, v1) or unsteady time-averaged (v2) solve of one shape -> dict(cd, cl, f, geom, params, boundary,
    converged, steps, mode_used, period, st, cl_rms, window_ct)."""
    c = cfg.lbm
    p = make_params(cfg, Condition(re, aoa), N, device=device, dtype=dtype)
    bd = rotate(nurbs(alpha.reshape(1, -1).to(device, dtype)), aoa)
    geom = map_to_grid(bd, p, body_offset_d=c.body_offset_d)
    if unsteady:
        u = solve_unsteady(cfg, p, geom)
        u = {**u, "cd": u["cd"].item(), "cl": u["cl"].item()}
        return dict(geom=geom, params=p, boundary=bd, **u)
    if f0 is None or f0.shape[1:] != (p.Ny, p.Nx):
        f0 = S.init_state(p, geom, dtype)
    r = S.run(f0, geom, p, force_fn=force_fn, mode="steady", record_every=c.record_every, outlet=c.outlet,
              steady_tol=c.steady_tol, max_convective_times=c.max_convective_times)
    return dict(cd=r.cd.item(), cl=r.cl.item(), f=r.state.f, geom=geom, params=p, boundary=bd,
                converged=bool(r.converged), steps=r.state.step, mode_used="steady", period=None, st=None, cl_rms=None,
                window_ct=None)


def bilinear(field, xl, yl):
    """field [Ny, Nx] sampled at lattice coords (xl, yl) [n]; also returns validity mask [n] of in-grid points."""
    Ny, Nx = field.shape
    x0, y0 = xl.floor().long().clamp(0, Nx - 2), yl.floor().long().clamp(0, Ny - 2)
    fx, fy = xl - x0, yl - y0
    g = lambda dy, dx: field[y0 + dy, x0 + dx]
    return (g(0, 0) * (1 - fx) * (1 - fy) + g(0, 1) * fx * (1 - fy) + g(1, 0) * (1 - fx) * fy + g(1, 1) * fx * fy)


def sample_points(res, cfg, gen):
    """Fields on interior points (fluid nodes in the window, 30% near the wall) + edge points -> dict of [P] tensors."""
    d, box = cfg.dataset, cfg.dataset.window_d
    p, geom = res["params"], res["geom"]
    N, (i0, j0) = p.N, geom.origin
    _, ux, uy = S.moments(res["f"])
    rho = S.moments(res["f"])[0]
    u, v, pr = (t.float().cpu() for t in lbm_to_nondim(rho, ux, uy, p.u_lb))
    solid = geom.solid.cpu()
    fields = torch.stack([u, v, pr])  # [3, Ny, Nx]
    jj, ii = torch.meshgrid(torch.arange(p.Ny), torch.arange(p.Nx), indexing="ij")
    X, Y = (ii - i0).float() / N, (jj - j0).float() / N
    fluid = ~solid & (X >= box[0]) & (X <= box[1]) & (Y >= box[2]) & (Y <= box[3])
    xy = torch.stack([X[fluid], Y[fluid]], -1)
    val = fields[:, fluid].T  # [n, 3]
    B = d.points_per_shape
    if len(xy) >= B:  # 30% of the budget within 0.2 chord of the wall
        bx, by = res["boundary"].x[0].cpu().float(), res["boundary"].y[0].cpu().float()
        near = torch.cdist(xy, torch.stack([bx, by], -1)).min(1).values < NEAR_WALL_D
        order = torch.rand(len(xy), generator=gen).argsort()
        n_near = min(round(NEAR_FRACTION * B), int(near.sum()))
        pick = torch.cat([order[near[order]][:n_near], order[~near[order]][:B - n_near]])
        pick = torch.cat([pick, order[near[order]][n_near:][:B - len(pick)]])  # too few far nodes: top up
        xy, val = xy[pick], val[pick]
    else:  # shortcut: window has fewer fluid nodes than the budget -> keep all, top up with bilinear points that have
        # four fluid neighbours (values near the wall are then never smeared across solid), upgrade: n/a
        need, ex = B - len(xy), []
        while need > 0:
            q = torch.stack([box[0] + (box[1] - box[0]) * torch.rand(4 * need, generator=gen),
                             box[2] + (box[3] - box[2]) * torch.rand(4 * need, generator=gen)], -1)
            xl, yl = i0 + N * q[:, 0], j0 + N * q[:, 1]
            x0, y0 = xl.floor().long(), yl.floor().long()
            ok = ~(solid[y0, x0] | solid[y0, x0 + 1] | solid[y0 + 1, x0] | solid[y0 + 1, x0 + 1])
            ex.append(q[ok][:need])
            need -= len(ex[-1])
        q = torch.cat(ex)
        xy, val = torch.cat([xy, q]), torch.cat([val, torch.stack([bilinear(f, i0 + N * q[:, 0], j0 + N * q[:, 1])
                                                                   for f in fields], -1)])
    # edge points: uniform in arc length on the window rectangle
    w, h = box[1] - box[0], box[3] - box[2]
    s = torch.rand(d.edge_points, generator=gen) * 2 * (w + h)
    ex_ = torch.where(s < w, box[0] + s, torch.where(s < w + h, box[1], torch.where(s < 2 * w + h, box[1] - (s - w - h), box[0])))
    ey_ = torch.where(s < w, box[2], torch.where(s < w + h, box[2] + (s - w), torch.where(s < 2 * w + h, box[3], box[3] - (s - 2 * w - h))))
    e = torch.stack([ex_, ey_], -1)
    ev = torch.stack([bilinear(f, i0 + N * e[:, 0], j0 + N * e[:, 1]) for f in fields], -1)
    xy, val = torch.cat([xy, e]), torch.cat([val, ev])
    edge = torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(d.edge_points, dtype=torch.bool)])
    return dict(xy=xy, edge=edge, u=val[:, 0], v=val[:, 1], p=val[:, 2])


def lbm_signature(cfg, N):
    """The config values a stored run must match to be reused."""
    c = cfg.lbm
    d = dict(N=N, domain_d=list(c.domain_d), body_offset_d=c.body_offset_d, collision=c.collision, outlet=c.outlet,
             mach=c.mach)
    return {**d, "trt_magic": c.trt_magic} if c.collision == "trt" else d


def converged_on_disk(path, want, redo_mode=None):
    try:
        m = torch.load(path, weights_only=False)["meta"]
        if redo_mode and m.get("mode_used") == redo_mode:
            return False  # --redo-mode: re-run runs that were averaged this way
        return (bool(m["converged"]) or m.get("mode_used") == "last_state") and m["cells_per_d"] == want["N"] and all(m["lbm_params"][k] == v for k, v in want.items())
    except Exception:
        return False


def generate(a, cfg, device, dtype):
    out = Path(a.out)
    nurbs = NURBSAirfoil(k=K_SYM, device=device, dtype=dtype)
    ref = NURBSAirfoil(k=K_SYM, dtype=torch.float64)
    dset = getattr(a, "dset", "v1")
    b, v2 = dset == "v1b", dset == "v2"
    if v2:  # cambered, D = 2k = 16 (G2)
        nurbs = NURBSAirfoil(k=K_SYM, mode="cambered", device=device, dtype=dtype)
        ref = NURBSAirfoil(k=K_SYM, mode="cambered", dtype=torch.float64)
    alphas = (thin_shapes(ref, cfg.dataset.n_shapes, cfg.seed + 777) if b else
              lhs_shapes(ref, cfg.dataset.n_shapes, cfg.seed, family="dataset"))
    base = V1B_BASE if b else 0
    first_holdout = 10 ** 9 if b else cfg.dataset.n_shapes - cfg.dataset.n_holdout  # v1b: every run is train
    N = cfg.lbm.cells_per_d_high_re if v2 else cfg.lbm.cells_per_d
    runs = shard(run_list(cfg, a.n_re40, dset), *a.shard)
    want = lbm_signature(cfg, N)
    prev, t_all = {}, time.perf_counter()
    for n, (i, re, aoa) in enumerate(runs):
        path = out / run_name(i, re, aoa)
        if converged_on_disk(path, want, getattr(a, "redo_mode", None)):
            print(f"[{n + 1}/{len(runs)}] skip {path.name} (converged)", flush=True)
            continue
        with Timer(device) as t:
            res = (solve(cfg, nurbs, alphas[i], re, aoa, N, device, dtype, unsteady=True) if v2 else
                   solve(cfg, nurbs, alphas[i - base], re, aoa, N, device, dtype, prev.get(re)))
        if not v2 and res["converged"] and torch.isfinite(res["f"]).all():
            prev[re] = res["f"]  # v2 starts cold: the 30-convective-time transient is fixed, a warm start saves nothing
        gen = torch.Generator().manual_seed(cfg.seed + 1000 * i + int(re))
        pts = sample_points(res, cfg, gen)
        p, c = res["params"], cfg.lbm
        meta = dict(id=path.stem, tier=a.tier, cells_per_d=N, git=_git_hash(), created=datetime.datetime.now().isoformat(),
                    lbm_params=dict(tau=p.tau, u_lb=p.u_lb, N=p.N, Nx=p.Nx, Ny=p.Ny, collision=p.collision, outlet=c.outlet,
                                    domain_d=list(c.domain_d), body_offset_d=c.body_offset_d, mach=c.mach,
                                    trt_magic=c.trt_magic),
                    steps=res["steps"], wall_s=t.elapsed, converged=res["converged"], period=res["period"], st=res["st"],
                    cl_rms=res["cl_rms"], mode_used=res["mode_used"], window_ct=res["window_ct"], cl_hist=res.get("cl_hist"), device=str(device))
        d = dict(alpha=alphas[i - base].float().cpu(), mode="cambered" if v2 else "symmetric", re=torch.tensor(float(re)), aoa=torch.tensor(float(aoa)),
                 cd=torch.tensor(res["cd"]), cl=torch.tensor(res["cl"]),
                 split=torch.tensor(int(i >= first_holdout), dtype=torch.int8), kind=torch.tensor(0, dtype=torch.int8),
                 meta=meta, **pts)
        tmp = path.with_suffix(".tmp")
        save_ckpt(tmp, d, str(device))
        os.replace(tmp, path)  # atomic
        print(f"[{n + 1}/{len(runs)}] {path.name}: CD={res['cd']:.4f} CL={res['cl']:+.1e} steps={res['steps']} "
              f"converged={res['converged']} mode={res['mode_used']} period={res['period']} {t.elapsed:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - t_all:.0f}s")


def merge(a):
    out = Path(a.out)
    items = [torch.load(f, weights_only=False) for f in sorted(out.glob("s*_re*_aoa*.pt"))]
    if not items:
        raise SystemExit(f"no run files in {out}")
    m = merge_anchors(items)
    m["meta"]["created"] = datetime.datetime.now().isoformat()
    save_ckpt(out / "merged.pt", m)
    print(f"merged {len(items)} runs -> {out / 'merged.pt'} ({(out / 'merged.pt').stat().st_size / 1e6:.1f} MB)")


def circle_row(a, cfg):
    """G26 pipeline sanity: the NURBS circle fit through the dataset's own solve + sample path (Re 20) -> (ok, message)."""
    device = get_device(a.device)
    dtype = torch.float64 if device.type == "cpu" else get_dtype(device, "sim")  # float32 CD noise (~1e-5) > steady_tol on CPU
    nurbs = NURBSAirfoil(k=K_SYM, device=device, dtype=dtype)
    al = circle_alpha(NURBSAirfoil(k=K_SYM, dtype=torch.float64))
    res = solve(cfg, nurbs, al, 20, 0, cfg.lbm.cells_per_d, device, dtype)
    pts = sample_points(res, cfg, torch.Generator().manual_seed(0))  # exercises the sampler too
    err = 100 * (res["cd"] - CYLINDER_CD_20x10) / CYLINDER_CD_20x10
    ok = abs(err) < 2 and res["converged"] and all(torch.isfinite(pts[k]).all() for k in ("xy", "u", "v", "p"))
    return bool(ok), f"CD={res['cd']:.4f} ({err:+.2f}%) converged={res['converged']} steps={res['steps']}"


def check(a, cfg):
    """Gate 2b report; returns True when everything passes."""
    from aeroshape.pinn.sampling import sample_batch

    path = Path(a.out) / "merged.pt"
    d = torch.load(path, weights_only=False)
    rows = []
    bad = check_anchors(d)
    rows.append(("merged loads, contract keys/shapes, finite", not bad, "; ".join(bad) or f"{d['xy'].shape[0]} runs"))
    runs = d["meta"]["runs"]
    if a.dset == "v2":  # Re 1000 time-averaged: contract + run count + how each run was averaged
        want = len(run_list(cfg, dset="v2"))
        modes = [r["mode_used"] for r in runs]
        rows.append(("all runs present", len(runs) == want, f"{len(runs)}/{want}; mode_used: " +
                     ", ".join(f"{m}={modes.count(m)}" for m in sorted(set(modes)))))
        for name, ok, msg in rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        return all(ok for _, ok, _ in rows)
    want = len(run_list(cfg, a.n_re40)) + len(list(Path(a.out).glob(f"s{V1B_BASE // 100}[0-9][0-9]_re*_aoa*.pt")))  # + v1b runs on disk
    rows.append(("all runs present and converged", len(runs) == want and all(r["converged"] for r in runs),
                 f"{sum(r['converged'] for r in runs)}/{len(runs)} converged, expected {want}"))
    ok, msg = circle_row(a, cfg)
    rows.append(("NURBS circle through the dataset pipeline: CD within 2% of cylinder (2.415)", ok, msg))
    re20 = d["re"] == 20
    cd20, sp = d["cd"][re20], d["split"][re20]
    tr, ho = cd20[sp == 0], cd20[sp == 1]
    ov = (min(tr.max(), ho.max()) - max(tr.min(), ho.min())).clamp(min=0) / (tr.max() - tr.min())
    rows.append(("holdout CD range overlaps >= 60% of train (Re 20)", bool(ov >= 0.6),
                 f"train [{tr.min():.3f}, {tr.max():.3f}] holdout [{ho.min():.3f}, {ho.max():.3f}] overlap {100 * ov:.0f}%"))
    try:
        b = sample_batch(d, NURBSAirfoil(k=K_SYM, M=40), cfg, torch.Generator().manual_seed(0))
        rows.append(("sample_batch draws a TrainBatch", True, f"xy_anchor {tuple(b.xy_anchor.shape)}, xy_edge {tuple(b.xy_edge.shape)}"))
    except Exception as e:  # report, don't crash
        rows.append(("sample_batch draws a TrainBatch", False, repr(e)))
    for name, ok, msg in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    return all(ok for _, ok, _ in rows)


def blockage(cfg_over, tier, device, dtype, N=20):
    """G27: NACA0012 (k = 8 fit), Re 20, CD in 20 x 10 (offset 5) vs 40 x 20 (offset 10)."""
    nurbs = NURBSAirfoil(k=K_SYM, device=device, dtype=dtype)
    al = naca4_alpha(NURBSAirfoil(k=K_SYM, dtype=torch.float64), "0012")
    cds = {}
    for dom, off in (([20, 10], 5), ([40, 20], 10)):
        cfg = load_config(None, tier, cfg_over + [f"lbm.domain_d={dom}", f"lbm.body_offset_d={off}"])
        r = solve(cfg, nurbs, al[0], 20, 0, N, device, dtype)
        cds[tuple(dom)] = r["cd"]
        print(f"NACA0012 Re 20 N={N} domain {dom[0]}x{dom[1]} offset {off}: CD={r['cd']:.4f} converged={r['converged']} steps={r['steps']}")
    a, b = cds[(20, 10)], cds[(40, 20)]
    print(f"blockage effect: CD(20x10) vs CD(40x20) differ by {100 * (a - b) / b:+.2f}%")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v1", choices=["v1", "v1b", "v2"], dest="dset")
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th run starting at i")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--blockage", action="store_true", help="G27 NACA0012 domain-size measurement")
    ap.add_argument("--redo-mode", default=None, help="re-run stored runs whose meta mode_used equals this (e.g. last_state)")
    ap.add_argument("--n-re40", type=int, default=10, help="shapes that also get a Re 40 run")
    ap.add_argument("--out", default=None, help="default data/lbm_anchors/<tier>")
    ap.add_argument("-o", action="append", default=[], dest="overrides", help="config override key=value")
    a = ap.parse_args(argv)
    if a.dset == "v2":
        a.overrides = V2_CASE + a.overrides
    a.out = a.out or f"data/lbm_anchors/{a.tier}" + ("_v2" if a.dset == "v2" else "")
    a.shard = tuple(int(x) for x in a.shard.split("/"))
    cfg = load_config(None, a.tier, a.overrides)
    device = get_device(a.device)
    dtype = get_dtype(device, "sim")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.blockage:
        blockage(a.overrides, a.tier, device, dtype)
    elif a.merge:
        merge(a)
    elif a.check:
        if not check(a, cfg):
            raise SystemExit(1)
    else:
        generate(a, cfg, device, dtype)


if __name__ == "__main__":
    main()
