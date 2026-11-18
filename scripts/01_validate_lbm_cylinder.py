"""Gate 1: analytic-circle drag validation (Plan/03_lbm/validation.md).
uv run python scripts/01_validate_lbm_cylinder.py --tier dev --device mps [--re 20 40] [--grid-convergence]
Gate 2a: --adjoint-check --tier dev --device cpu (adjoint vs finite-difference gradient, 41-point NURBS circle)
Domain size / body offset / Mach / collision / outlet are config keys: --set lbm.domain_d=[40,20] lbm.body_offset_d=10"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import load_config  # noqa: E402
from aeroshape.core.device import device_info, get_device, get_dtype  # noqa: E402
from aeroshape.core.seeding import seed_everything  # noqa: E402
from aeroshape.core.timing import Timer  # noqa: E402
from aeroshape.core.types import Boundary, Condition  # noqa: E402
from aeroshape.geometry.ibb_grid_mapper import map_to_grid  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.geometry.reference_shapes import circle_alpha  # noqa: E402
from aeroshape.lbm import d2q9_solver as S  # noqa: E402
from aeroshape.lbm.adjoint_solver import lbm_value, lbm_value_and_grad  # noqa: E402
from aeroshape.lbm.aero_forces import force_fn  # noqa: E402
from aeroshape.lbm.lattice import make_params  # noqa: E402
from aeroshape.visualization import plots  # noqa: E402

# Re -> (report CD, literature note, case defaults as lbm.* overrides, mode)
CASES = {
    20: (2.04, "Dennis and Chang 2.045", {"collision": "bgk", "outlet": "fixed"}, "steady"),
    40: (1.550, "Nieuwstadt and Keller 1.550 (gate target, G26); reference only: report/Tritton 1.65", {"collision": "bgk", "outlet": "fixed"}, "steady"),
    100: (1.365, "Hejranfar and Ezzatneshan 1.336",
          {"collision": "trt", "outlet": "extrapolate", "sponge.enabled": True, "max_convective_times": 100,
           "transient": 70, "perturb": 0.1},
          "unsteady"),  # transient: onset 30-35 convective times, saturated by ~70
}


def circle(r=0.5, n=4096, device="cpu", dtype=torch.float64) -> Boundary:
    th = (torch.arange(n + 1, dtype=torch.float64) + 0.5) * (2 * math.pi / n)
    x, y = (r * torch.cos(th))[None].to(device, dtype), (r * torch.sin(th))[None].to(device, dtype)
    z = torch.zeros_like(x)
    return Boundary(x, y, z, z, z)


def load_case_cfg(tier, re, user_sets):
    """Case defaults first, user --set last (so explicit keys win)."""
    d = [f"lbm.{k}={json.dumps(v)}" for k, v in CASES[re][2].items()]
    return load_config("validate_cylinder.yaml", tier, d + list(user_sets))


def run_case(cfg, re, N, device, dtype, mode, fig=None, tag=""):
    """-> result dict (+ '_hist', '_solid' for plotting). Honours cfg.lbm.{domain_d, body_offset_d, ...}."""
    c, v = cfg.lbm, cfg.validate
    p = make_params(cfg, Condition(re, 0), N, device=device, dtype=dtype)
    geom = map_to_grid(circle(device=device, dtype=dtype), p, body_offset_d=c.body_offset_d)
    K = round(N / p.u_lb)
    snaps, want = {}, sorted(v.snapshot_t) if mode == "unsteady" else []

    def cb(n, f):
        while want and n >= want[0] * K:
            _, ux, uy = S.moments(f)
            w = torch.gradient(uy, dim=1)[0] - torch.gradient(ux, dim=0)[0]
            snaps[want.pop(0)] = (w / p.u_lb * N).cpu()

    f0 = S.init_state(p, geom, dtype, perturb=c.perturb if mode == "unsteady" else 0.0)
    kw = dict(record_every=c.record_every, force_fn=force_fn, outlet=c.outlet, steady_tol=c.steady_tol,
              max_convective_times=c.max_convective_times, average_periods=c.average_periods)
    with Timer(device) as t:
        r = S.run(f0, geom, p, n_steps=v.n_steps, mode=mode, transient=c.transient, callback=cb if want else None, **kw)
    h = r.history
    out = dict(re=re, N=N, cd=r.cd.item(), cl_mean=r.cl.item(), steps=r.state.step, wall_s=t.elapsed,
               device=str(device), converged=r.converged, outlet=c.outlet, collision=c.collision,
               domain_d=list(c.domain_d), body_offset_d=c.body_offset_d, mach=c.mach, cl_rms=None, st=None,
               period_steps=r.period_steps, _hist=h, _snaps=snaps, _solid=geom.solid.cpu())
    if mode == "unsteady" and r.period_steps:
        m = h.step > h.step[-1] - c.average_periods * r.period_steps
        out["cl_rms"] = h.cl[m].pow(2).mean().sqrt().item()
        out["st"] = N / (r.period_steps * p.u_lb)
    return out


def judge(res, report):
    err = 100 * (res["cd"] - report) / report
    ok = math.isfinite(res["cd"]) and abs(err) < 2
    if res["re"] == 100:
        ok = ok and res["st"] is not None and res["cl_rms"] > 0.05 and abs(res["cl_mean"]) < 0.01  # shedding detected
    return err, bool(ok)


Q_MIN = 0.01  # link_q clamp floor


def _adjoint_setup(a, device, dtype):
    v = load_config("validate_cylinder.yaml", a.tier, a.sets).validate
    ov = ["lbm.collision=bgk", "lbm.outlet=fixed", f"lbm.steady_tol={float(v.adjoint_steady_tol):.1e}",
          f"lbm.max_convective_times={v.adjoint_max_convective_times}", "adjoint.mode=steady"]
    cfg = load_config("validate_cylinder.yaml", a.tier, ov + list(a.sets))
    cond, N = Condition(20, 0), v.adjoint_N
    nurbs = NURBSAirfoil(k=v.adjoint_k, device=device, dtype=dtype)
    alpha = circle_alpha(nurbs)
    p = make_params(cfg, cond, N, device=device, dtype=dtype)

    def topo(al):  # solid mask, link nodes/dirs, and the q-clamp state (signs of q - q_min, q - 1) per link
        g = map_to_grid(nurbs(al), p, body_offset_d=cfg.lbm.body_offset_d)
        return g.solid, g.links.node, g.links.dir, torch.sign(g.links.q - Q_MIN), torch.sign(g.links.q - 1.0)

    def same(t, u):
        return all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(t, u))

    return v, cfg, cond, N, nurbs, alpha, p, topo, same


def adjoint_check(a, device, dtype, out):
    """Gate 2a: adjoint dCD/dalpha vs central FD of the same solver at the free control points (alpha[0], alpha[-1] fixed)."""
    v, cfg, cond, N, nurbs, alpha, p, topo, same = _adjoint_setup(a, device, dtype)
    delta = v.adjoint_delta
    n_unconv = [0]

    t0 = time.perf_counter()
    r = lbm_value_and_grad(alpha, nurbs, cond, cfg, N=N)
    wall_adj = time.perf_counter() - t0
    f_base, g_adj, base_conv = r.extras["f"], r.dcd_dalpha.reshape(-1), bool(r.extras["converged"])
    print(f"adjoint: CD={r.cd.item():.6f} wall={wall_adj:.1f}s (forward {r.extras['t_forward']:.1f}s, "
          f"adjoint {r.extras['t_adjoint']:.1f}s)", flush=True)

    def fd(k, d):
        vals = []
        for s in (1, -1):
            b = alpha.clone()
            b[0, k] += s * d
            info = {}
            vals.append(lbm_value(b, nurbs, cond, cfg, N=N, f0=f_base, info=info)[0])
            n_unconv[0] += not info["converged"]
        return ((vals[0] - vals[1]) / (2 * d)).item()

    t0, T0 = time.perf_counter(), topo(alpha)
    free = list(range(1, alpha.shape[1] - 1))
    g_fd, included, excluded = {}, [], []
    for k in free:
        ok = True
        for s in (1, -1):
            b = alpha.clone()
            b[0, k] += s * delta
            ok = ok and same(topo(b), T0)
        if not ok:
            excluded.append(k)
            print(f"  point {k}: topology changes at +-delta, excluded", flush=True)
            continue
        g_fd[k] = fd(k, delta)
        included.append(k)
        print(f"  point {k}: adj={g_adj[k].item():+.6e} fd={g_fd[k]:+.6e}", flush=True)
    wall_fd = time.perf_counter() - t0

    inc = torch.tensor(included)
    gf = torch.tensor([g_fd[k] for k in included], dtype=dtype)
    err = ((g_adj[inc] - gf).norm() / gf.norm()).item()
    # FD-noise estimate: re-difference the largest-|g| points at delta/2 (includes O(delta^2) truncation)
    big = sorted(included, key=lambda k: -abs(g_fd[k]))[:v.noise_points]
    noise = None
    if big:
        h = torch.tensor([fd(k, delta / 2) for k in big], dtype=dtype)
        ref = torch.tensor([g_fd[k] for k in big], dtype=dtype)
        noise = ((h - ref).norm() / ref.norm()).item()
    full_fd = torch.full((alpha.shape[1],), float("nan"), dtype=dtype)
    full_fd[inc] = gf
    g_ex = g_adj[torch.tensor(excluded, dtype=torch.long)].norm() if excluded else g_adj.new_zeros(())
    ex_frac = (g_ex / g_adj[torch.tensor(free)].norm()).item()
    worst = sorted(included, key=lambda k: -abs(g_adj[k].item() - g_fd[k]))[:5]
    res = dict(tier=a.tier, device=str(device), dtype=str(dtype), k=v.adjoint_k, N=N, domain_d=list(cfg.lbm.domain_d),
               body_offset_d=cfg.lbm.body_offset_d, re=20, delta=delta, steady_tol=cfg.lbm.steady_tol,
               max_convective_times=cfg.lbm.max_convective_times, n_free=len(free), included=included, excluded=excluded,
               rel_l2_error=err, threshold=v.adjoint_pass, passed=bool(err < v.adjoint_pass), fd_noise_rel=noise,
               fd_noise_points=big, cd=r.cd.item(), g_adj=[g_adj[k].item() for k in free],
               g_fd=[g_fd.get(k) for k in free], free_indices=free,
               worst5=[dict(index=k, adjoint=g_adj[k].item(), fd=g_fd[k]) for k in worst],
               wall_adjoint_s=wall_adj, wall_fd_s=wall_fd, n_fd_solves=2 * len(included),
               adjoint_matvecs=r.extras["adjoint_matvecs"], excluded_adjoint_norm_fraction=ex_frac,
               base_converged=base_conv, n_unconverged_fd_solves=n_unconv[0])
    plots.plot_gradient_check(g_adj, full_fd, included, out / "fig" / "adjoint_vs_fd.png", err)
    g0 = map_to_grid(nurbs(alpha), p, body_offset_d=cfg.lbm.body_offset_d)
    plots.plot_adjoint_field(r.extras["F_star"], g0.solid, out / "fig" / "adjoint_sensitivity_field.png")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"Gate 2a: rel L2 error {err:.3e} (< {v.adjoint_pass}: {'PASS' if res['passed'] else 'FAIL'}); included "
          f"{len(included)}, excluded {excluded} ({100 * ex_frac:.1f}% of |g_adj|); base converged {base_conv}, "
          f"unconverged FD solves {n_unconv[0]}; FD noise {noise}; adjoint {wall_adj:.0f}s vs FD {wall_fd:.0f}s", flush=True)
    return 0


def excluded_onesided(a, device, dtype, out):
    """Opt-in: one-sided FD at the points the central check excluded (reads out/results.json, adds results only)."""
    v, cfg, cond, N, nurbs, alpha, p, topo, same = _adjoint_setup(a, device, dtype)
    path = out / "results.json"
    res = json.loads(path.read_text())
    g_adj = dict(zip(res["free_indices"], res["g_adj"]))
    T0, info = topo(alpha), {}
    cd0, _, f_base = lbm_value(alpha, nurbs, cond, cfg, N=N, info=info)
    base_conv = info["converged"]
    print(f"base CD={cd0.item():.6f} converged={base_conv} (central run: {res['cd']:.6f})", flush=True)
    rows = []
    for k in res["excluded"]:
        pick = None
        for d in (v.adjoint_delta / m for m in (1, 10, 100)):
            for s in (1, -1):
                b = alpha.clone()
                b[0, k] += s * d
                if same(topo(b), T0):
                    pick = (s, d, b)
                    break
            if pick:
                break
        if pick is None:
            rows.append(dict(index=k, side=None, step=None, adjoint=g_adj[k], fd=None, converged=None))
            print(f"  point {k}: no side/step keeps the topology", flush=True)
            continue
        s, d, b = pick
        info = {}
        cd1 = lbm_value(b, nurbs, cond, cfg, N=N, f0=f_base, info=info)[0]
        fd = ((cd1 - cd0) / (s * d)).item()
        rows.append(dict(index=k, side=s, step=d, adjoint=g_adj[k], fd=fd, converged=info["converged"]))
        print(f"  point {k}: side {s:+d} step {d:.0e} adj={g_adj[k]:+.6e} fd1={fd:+.6e} converged={info['converged']}", flush=True)
    ok = [r for r in rows if r["fd"] is not None]
    err = None
    if ok:
        ga, gf = (torch.tensor([r[key] for r in ok], dtype=torch.float64) for key in ("adjoint", "fd"))
        err = ((ga - gf).norm() / gf.norm()).item()
    res.update(excluded_rel_l2_onesided=err, excluded_onesided=rows, excluded_onesided_unconverged=sum(r["converged"] is False for r in rows),
               excluded_onesided_base_converged=base_conv)
    path.write_text(json.dumps(res, indent=2))
    print(f"excluded one-sided rel L2 error {err} over {len(ok)}/{len(rows)} points (one-sided FD is O(step) accurate)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default=None, choices=["float32", "float64"])
    ap.add_argument("--re", type=float, nargs="*", help="default: validate.re from the config")
    ap.add_argument("--set", action="append", default=[], dest="sets", help="config override key=value")
    ap.add_argument("--grid-convergence", action="store_true")
    ap.add_argument("--adjoint-check", action="store_true", help="Gate 2a instead of Gate 1")
    ap.add_argument("--excluded-onesided", action="store_true", help="with --adjoint-check: one-sided FD at the excluded points, reusing results.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = get_device(a.device)
    dtype = getattr(torch, a.dtype) if a.dtype else get_dtype(device, "sim")
    base = load_config("validate_cylinder.yaml", a.tier, a.sets)
    seed_everything(base.seed)
    out = Path(a.out or f"runs/validate_{'adjoint' if a.adjoint_check else 'cylinder'}/{a.tier}")
    (out / "fig").mkdir(parents=True, exist_ok=True)
    if a.adjoint_check:
        if a.dtype is None:
            dtype = torch.float64
        return (excluded_onesided if a.excluded_onesided else adjoint_check)(a, device, dtype, out)
    res_list = [int(r) for r in (a.re or base.validate.re)]
    print("device:", device_info(device), dtype)

    cases, rows = {}, []
    for re in res_list:
        report, lit, _, mode = CASES[re]
        cfg = load_case_cfg(a.tier, re, a.sets)
        N = cfg.lbm.cells_per_d_high_re if re >= 100 else cfg.lbm.cells_per_d  # tau ~ 0.52 is unstable at N=10, Re 100
        r = run_case(cfg, re, N, device, dtype, mode)
        r["error_pct"], r["pass"] = judge(r, report)
        r["target"] = report
        K = round(N / make_params(cfg, Condition(re, 0)).u_lb)
        plots.plot_force_history(r["_hist"], out / "fig" / f"cd_cl_history_re{re}.png", f"Re {re}", ref=report, x=K)
        for t, w in r["_snaps"].items():
            plots.plot_field(w, out / "fig" / f"vorticity_re{re}_t{t}.png", f"vorticity Re {re}, t={t}", r["_solid"])
        print(f"Re {re}: CD={r['cd']:.4f} (target {report}, {r['error_pct']:+.2f}%) CLmean={r['cl_mean']:.2e} "
              f"CLrms={r['cl_rms']} St={r['st']} steps={r['steps']} wall={r['wall_s']:.1f}s "
              f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)
        cases[str(re)] = {k: v for k, v in r.items() if not k.startswith("_")}
        rows.append(dict(re=re, report=report, literature=lit, cd=r["cd"], error_pct=r["error_pct"], passed=r["pass"]))

    summary = dict(tier=a.tier, device_info=device_info(device), dtype=str(dtype), cases=cases)
    if a.grid_convergence:
        cfg = load_case_cfg(a.tier, 20, a.sets)
        Ns = list(cfg.validate.grid_N)
        cds = [run_case(cfg, 20, n, device, dtype, "steady")["cd"] for n in Ns]
        order = None
        if len(cds) >= 3 and (cds[1] - cds[2]) * (cds[0] - cds[1]) > 0:
            order = math.log((cds[0] - cds[1]) / (cds[1] - cds[2])) / math.log(Ns[1] / Ns[0])
        plots.plot_convergence(Ns, cds, out / "fig" / "grid_convergence_re20.png", ref=CASES[20][0])
        summary["grid_convergence"] = dict(N=Ns, cd=cds, observed_order=order)
        print("grid convergence Re 20:", Ns, [round(c, 4) for c in cds], "observed order:", order)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    plots.write_table1_md(rows, out / "table1.md")
    print("wrote", out / "results.json")
    return 0  # pass/fail lives in results.json and table1.md


if __name__ == "__main__":
    raise SystemExit(main())
