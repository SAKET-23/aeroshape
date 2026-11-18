"""Review-3 checks on the trained PINNs (all CPU).
uv run python scripts/03b_pinn_checks.py gen-test                       # M1: 10 clean test shapes -> data/lbm_anchors/dev_test/
uv run python scripts/03b_pinn_checks.py path-check                     # M3: PINN vs LBM/adjoint along the case-1 optimization path
uv run python scripts/03b_pinn_checks.py cp-check                       # M4: PINN surface Cp vs LBM wall pressure on 2 test shapes
uv run python scripts/03b_pinn_checks.py gen-v1c                       # G33: v1c round/blend train shapes (CPU) -> data/lbm_anchors/dev/, merge, adjoint cache
Gate 3 on the test set itself: 03_train_transformer_pinn.py --eval-only <ckpt> --model <kind> --eval-set test --grad-shapes 10."""
import argparse
import datetime
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aeroshape.core.config import load_config  # noqa: E402
from aeroshape.core.io import _git_hash, check_anchors, load_ckpt, merge_anchors, save_ckpt  # noqa: E402
from aeroshape.core.types import Condition, CostWeights  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.geometry.reference_shapes import circle_alpha, fit_alpha, is_valid, naca4_alpha, perturb, shape_distance  # noqa: E402
from aeroshape.lbm.adjoint_solver import lbm_value_and_grad  # noqa: E402
from aeroshape.optimization.evaluators import PINNEvaluator  # noqa: E402
from aeroshape.pinn.stress_integrator import _rotated, surface_cp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g02, t03 = _load("gen02", "02_generate_lbm_dataset.py"), _load("train03", "03_train_transformer_pinn.py")
TEST_BASE, TEST_DIR, T_RANGE = 200, Path("data/lbm_anchors/dev_test"), (0.08, 0.80)
COND = Condition(20.0, 0.0)


def gen_test():
    cfg = load_config(None, "dev", ["lbm.steady_tol=0.00001"])  # 1e-5 as v1b (G32): float32 CL noise never meets 1e-6 on thin shapes
    dev, dt = torch.device("cpu"), torch.float32
    ref, nurbs = NURBSAirfoil(k=g02.K_SYM, dtype=torch.float64), NURBSAirfoil(k=g02.K_SYM, device=dev, dtype=dt)
    alphas = g02.thin_shapes(ref, 10, cfg.seed + 4242, *T_RANGE)
    print("thickness:", [round(float(t), 3) for t in g02.thickness(ref, alphas.double())])
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    N, want = cfg.lbm.cells_per_d, g02.lbm_signature(cfg, cfg.lbm.cells_per_d)
    for k, al in enumerate(alphas):
        i, path = TEST_BASE + k, TEST_DIR / g02.run_name(TEST_BASE + k, 20, 0)
        if g02.converged_on_disk(path, want):
            continue
        res = g02.solve(cfg, nurbs, al, 20, 0, N, dev, dt)
        pts = g02.sample_points(res, cfg, torch.Generator().manual_seed(cfg.seed + 1000 * i + 20))
        p, c = res["params"], cfg.lbm
        meta = dict(id=path.stem, tier="dev", cells_per_d=N, git=_git_hash(), created=datetime.datetime.now().isoformat(),
                    lbm_params=dict(tau=p.tau, u_lb=p.u_lb, N=p.N, Nx=p.Nx, Ny=p.Ny, collision=p.collision, outlet=c.outlet,
                                    domain_d=list(c.domain_d), body_offset_d=c.body_offset_d, mach=c.mach, trt_magic=c.trt_magic),
                    steps=res["steps"], converged=res["converged"], mode_used="steady", device="cpu", period=None, st=None, cl_rms=None)
        save_ckpt(path, dict(alpha=al.float().cpu(), mode="symmetric", re=torch.tensor(20.0), aoa=torch.tensor(0.0), cd=torch.tensor(res["cd"]),
                             cl=torch.tensor(res["cl"]), split=torch.tensor(1, dtype=torch.int8), kind=torch.tensor(0, dtype=torch.int8),
                             meta=meta, **pts), "cpu")
        print(f"[{k + 1}/10] {path.name}: CD={res['cd']:.4f} converged={res['converged']} steps={res['steps']}", flush=True)
    items = [torch.load(f, weights_only=False) for f in sorted(TEST_DIR.glob("s2*_re20_aoa0.pt"))]
    assert len(items) == 10
    save_ckpt(TEST_DIR / "merged.pt", merge_anchors(items))
    print("merged ->", TEST_DIR / "merged.pt")


V1C_BASE, V1C_DIR, TRAJ = 300, Path("data/lbm_anchors/dev"), "runs/smoke_case1_mps_20cyc/history.pt"  # ids s300.. (test set uses s200..)


def _ellipse(ref, tr):
    x = (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
    return fit_alpha(ref, x, tr * np.sqrt(x * (1 - x)))  # tr = thickness ratio: the circle (tr 1) has y = sqrt(x(1-x)), t = 1


def thickness_(ref, a):
    return g02.thickness(ref, a.double())[0]


def v1c_shapes(ref, seed=7):
    """G33: [30, D] float32 = 20 circle<->{NACA0012, NACA0024, ellipse 0.5, 0.75} blends (w 0.1..0.9, smooth perturbation) + 10 thick round
    shapes (t 0.8-1.1, RMS > 0.01 from the circle). Each is valid and RMS >= 0.005 from every adjoint-trajectory alpha (TRAJ)."""
    circ = circle_alpha(ref)
    traj = torch.load(TRAJ, weights_only=False)["alpha"].double().reshape(-1, circ.shape[1]) if Path(TRAJ).exists() else None
    tgt = [naca4_alpha(ref, "0012"), naca4_alpha(ref, "0024"), _ellipse(ref, 0.5), _ellipse(ref, 0.75)]
    ws, out = [0.1, 0.3, 0.5, 0.7, 0.9], []

    def ok(a, lo, hi, away_circle):
        if not bool(is_valid(ref, a)[0]) or not lo <= float(thickness_(ref, a)) <= hi:
            return False
        if away_circle and float(shape_distance(ref, a, circ)[0]) <= 0.01:
            return False
        return traj is None or float(shape_distance(ref, a.expand(len(traj), -1), traj).min()) >= 0.005

    for ti, t in enumerate(tgt):
        for wi, w in enumerate(ws):
            for r in range(100):  # retry seeds until valid
                a = perturb((1 - w) * circ + w * t, 1, 0.015, 0.3, seed + 100 * ti + 10 * wi + 1000 * r, ref)
                if ok(a, 0.0, 2.0, False):
                    out.append(a)
                    break
            else:
                raise RuntimeError("no valid morph")
    for k, tr in enumerate(np.linspace(0.8, 1.1, 10)):
        for r in range(100):
            a = perturb(_ellipse(ref, tr), 1, 0.02, 0.3, seed + 5000 + k + 1000 * r, ref)
            if ok(a, 0.8, 1.1, True):
                out.append(a)
                break
        else:
            raise RuntimeError("no valid round shape")
    return torch.cat(out).float()


def gen_v1c():
    cfg = load_config(None, "dev", ["lbm.steady_tol=0.00001"])  # as v1b / test (G32)
    dev, dt, t0 = torch.device("cpu"), torch.float32, time.perf_counter()
    ref, nurbs = NURBSAirfoil(k=g02.K_SYM, dtype=torch.float64), NURBSAirfoil(k=g02.K_SYM, device=dev, dtype=dt)
    alphas = v1c_shapes(ref)
    n = len(alphas)
    print("thickness:", [round(float(t), 3) for t in g02.thickness(ref, alphas.double())], flush=True)
    N, want = cfg.lbm.cells_per_d, g02.lbm_signature(cfg, cfg.lbm.cells_per_d)
    for k, al in enumerate(alphas):
        i, path = V1C_BASE + k, V1C_DIR / g02.run_name(V1C_BASE + k, 20, 0)
        if g02.converged_on_disk(path, want):
            continue
        t1 = time.perf_counter()
        res = g02.solve(cfg, nurbs, al, 20, 0, N, dev, dt)
        pts = g02.sample_points(res, cfg, torch.Generator().manual_seed(cfg.seed + 1000 * i + 20))
        p, c = res["params"], cfg.lbm
        meta = dict(id=path.stem, tier="dev", cells_per_d=N, git=_git_hash(), created=datetime.datetime.now().isoformat(),
                    lbm_params=dict(tau=p.tau, u_lb=p.u_lb, N=p.N, Nx=p.Nx, Ny=p.Ny, collision=p.collision, outlet=c.outlet,
                                    domain_d=list(c.domain_d), body_offset_d=c.body_offset_d, mach=c.mach, trt_magic=c.trt_magic),
                    steps=res["steps"], wall_s=time.perf_counter() - t1, converged=res["converged"], mode_used="steady", device="cpu",
                    period=None, st=None, cl_rms=None)
        save_ckpt(path, dict(alpha=al.float().cpu(), mode="symmetric", re=torch.tensor(20.0), aoa=torch.tensor(0.0), cd=torch.tensor(res["cd"]),
                             cl=torch.tensor(res["cl"]), split=torch.tensor(0, dtype=torch.int8), kind=torch.tensor(0, dtype=torch.int8),
                             meta=meta, **pts), "cpu")
        print(f"[{k + 1}/{n}] {path.name}: CD={res['cd']:.4f} converged={res['converged']} steps={res['steps']} {time.perf_counter() - t1:.0f}s", flush=True)
    print(f"LBM wall {time.perf_counter() - t0:.0f}s", flush=True)
    m = V1C_DIR / "merged.pt"
    if not (V1C_DIR / "merged_v1v1b.pt").exists():
        shutil.copy(m, V1C_DIR / "merged_v1v1b.pt")  # back up the v1 + v1b merge once
    old = torch.load(V1C_DIR / "merged_v1v1b.pt", weights_only=False)
    items = [torch.load(f, weights_only=False) for f in sorted(V1C_DIR.glob("s*_re*_aoa*.pt"))]
    new = merge_anchors(items)
    new["meta"]["created"] = datetime.datetime.now().isoformat()
    save_ckpt(m, new)
    ho, ho_old = new["split"] == 1, old["split"] == 1
    ids, ids_old = [r["id"] for r in new["meta"]["runs"]], [r["id"] for r in old["meta"]["runs"]]
    same = ho.sum() == ho_old.sum() and all(torch.equal(new[k][ho], old[k][ho_old]) for k in ("alpha", "xy", "u", "v", "p", "cd", "cl"))
    print(f"merged {len(items)} runs -> {m}; holdout bit-identical: {bool(same)}; check_anchors: {check_anchors(new) or 'ok'}; "
          f"all converged: {all(r['converged'] for r in new['meta']['runs'])}; {len(ids) - len(ids_old)} added", flush=True)
    t0 = time.perf_counter()
    A = load_ckpt(m, "cpu")
    t03.precompute_train_adjoints(A, nurbs, ROOT / t03.TRAIN_ADJ, "dev", "cpu")
    print(f"adjoint wall {time.perf_counter() - t0:.0f}s", flush=True)
    a_tr = A["alpha"][A["split"] == 0]
    print("train alpha max:", a_tr.amax(0).tolist(), "\ncircle alpha:", circle_alpha(ref).float().reshape(-1).tolist())


def _models(sel, A):
    return {k: t03.load_for_eval(ROOT / v["checkpoint"], k, "cpu", A) for k, v in sel.items()}


def path_check():
    """M3: circle start, the adjoint case-1 trajectory and NACA0012 through the LBM + adjoint (CPU) and both PINNs."""
    sel = json.loads(Path("runs/pinn_selected.json").read_text())
    A = load_ckpt(t03.ANCHORS, "cpu")
    tr = A["split"] == 0
    thick = lambda a: float(g02.thickness(NURBSAirfoil(k=g02.K_SYM, dtype=torch.float64), a.double().reshape(1, -1))[0])
    ref, nurbs = NURBSAirfoil(k=g02.K_SYM, dtype=torch.float64), NURBSAirfoil(k=g02.K_SYM, dtype=torch.float32)
    circ = circle_alpha(ref).float().reshape(-1)
    hist = torch.load("runs/smoke_case1_mps_20cyc/history.pt", weights_only=False)["alpha"]
    shapes = {"circle (start)": circ, **{f"cycle {c}": hist[c].float() for c in (5, 10, 15, 19)},
              "NACA0012": naca4_alpha(ref, "0012").float().reshape(-1)}
    cfg_adj, models, free = load_config(t03.CASE, "dev", []), _models(sel, A), nurbs.free_idx
    rows = []
    for name, al in shapes.items():
        r = lbm_value_and_grad(al, nurbs, COND, cfg_adj, CostWeights(0.0, 0.0, 0.0), None)
        g_adj, row = r.extras["dI_dalpha"].cpu(), dict(shape=name, thickness=thick(al), cd_lbm=float(r.cd))
        for k, (m, _) in models.items():
            e = PINNEvaluator(m, nurbs).evaluate(al, COND)
            gp, cs = e.dcd_dalpha.cpu(), torch.nn.functional.cosine_similarity
            row[k] = dict(cd=float(e.cd), cd_err=float(e.cd) / float(r.cd) - 1, cos_free=float(cs(gp[free.cpu()], g_adj[free.cpu()], dim=0)),
                          cos_full=float(cs(gp, g_adj, dim=0)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    a_tr = A["alpha"][tr]
    res = dict(rows=rows, train_alpha_max=a_tr.amax(0).tolist(), train_alpha_min=a_tr.amin(0).tolist(), circle_alpha=circ.tolist(),
               train_thickness_max=max(thick(a) for a in a_tr), circle_exceeds_train_max=(circ > a_tr.amax(0)).tolist())
    Path("runs/pinn_path_check.json").write_text(json.dumps(res, indent=2))


def _wall_p(xy, p, bx, by, r=0.15):
    """LBM wall pressure at loop nodes: weighted linear fit of anchor p within r of each node, evaluated at the node. -> [n] (NaN if < 4 pts)."""
    out = torch.full((len(bx),), float("nan"))
    for j in range(len(bx)):
        d = xy - torch.stack([bx[j], by[j]])
        m = d.norm(dim=1) < r
        if m.sum() < 4:
            continue
        w = (1 - d[m].norm(dim=1) / r).sqrt()
        X = torch.cat([torch.ones(int(m.sum()), 1), d[m]], 1) * w[:, None]
        out[j] = torch.linalg.lstsq(X, (p[m] * w)[:, None]).solution[0, 0]
    return out


def cp_check(shapes=(0, 5)):
    sel = json.loads(Path("runs/pinn_selected.json").read_text())
    T = load_ckpt(t03.TEST_ANCHORS, "cpu")
    nurbs, models, res = NURBSAirfoil(k=g02.K_SYM, dtype=torch.float32), _models(sel, load_ckpt(t03.ANCHORS, "cpu")), []
    for i in shapes:
        keep = ~T["edge"][i]
        al, cond = T["alpha"][i][None], COND.as_tensor("cpu")[None]
        bd = _rotated(nurbs, al, cond, None, None)
        bx, by = bd.x[0], bd.y[0]
        lbm = 2 * _wall_p(T["xy"][i][keep], T["p"][i][keep], bx, by)  # Cp = 2p (rho = U = 1)
        ok = torch.isfinite(lbm)
        row = dict(test_shape=i, thickness=float(2 * by.amax()), nodes_with_lbm=int(ok.sum()), nodes=len(bx),
                   cp_lbm_range=[float(lbm[ok].min()), float(lbm[ok].max())])
        for k, (m, _) in models.items():
            cp = surface_cp(m, nurbs, al, cond)[0].detach()
            e = cp[ok] - lbm[ok]
            row[k] = dict(rms_cp_err=float(e.pow(2).mean().sqrt()), rms_cp_err_mean_removed=float((e - e.mean()).pow(2).mean().sqrt()),
                          cp_range=[float(cp[ok].min()), float(cp[ok].max())])
        res.append(row)
        print(json.dumps(row), flush=True)
    Path("runs/pinn_cp_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen-test", "gen-v1c", "path-check", "cp-check"])
    {"gen-test": gen_test, "gen-v1c": gen_v1c, "path-check": path_check, "cp-check": cp_check}[ap.parse_args().cmd]()
