"""PINN training + Gate 3 (Plan/06_training/pinn_training.md).
uv run python scripts/03_train_transformer_pinn.py --model transformer --tier dev --device mps --out runs/pinn_transformer_v1_dev
uv run python scripts/03_train_transformer_pinn.py --model mlp --sweep --tier dev --device mps --out runs/pinn_sweep   # Fourier-scale sweep
uv run python scripts/03_train_transformer_pinn.py --model mlp --eval-only runs/<name>/ckpt/best.pt --device mps --out runs/<name>
Writes runs/<name>/{config.yaml,log.csv,ckpt/,results.json,gate3.json}. Not implemented: --resume, --tb, fig/."""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import Config, dump_config, load_config  # noqa: E402
from aeroshape.core.device import device_info, get_device  # noqa: E402
from aeroshape.core.io import check_anchors, load_ckpt, save_ckpt  # noqa: E402
from aeroshape.core.seeding import seed_everything  # noqa: E402
from aeroshape.core.timing import Timer, sync  # noqa: E402
from aeroshape.core.types import Boundary, Condition, CostWeights  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.lbm.adjoint_solver import lbm_value_and_grad  # noqa: E402
from aeroshape.optimization.evaluators import PINNEvaluator  # noqa: E402
from aeroshape.pinn.embeddings import predict_flow, wall_distance  # noqa: E402
from aeroshape.pinn.mixing import mix_anchors  # noqa: E402
from aeroshape.pinn.mlp_pinn import MLPPINN  # noqa: E402
from aeroshape.pinn.navier_stokes_loss import TrainBatch, hybrid_loss  # noqa: E402
from aeroshape.pinn.sampling import sample_batch  # noqa: E402
from aeroshape.pinn.stress_integrator import _rotated, surface_forces  # noqa: E402
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN  # noqa: E402

ANCHORS = "data/lbm_anchors/dev/merged.pt"
CASE = "case1_circle_to_naca_re20.yaml"
TRAIN_ADJ = "runs/adjoint_cache/train_dev.pt"
TEST_ANCHORS = "data/lbm_anchors/dev_test/merged.pt"  # M1: clean test set, never used for any choice (all split == 1)
TEST_ADJ = "runs/adjoint_cache/test_dev.pt"


def adj_hash(cfg_adj, tier) -> str:
    """L3: the settings an adjoint cache entry depends on (a different value -> recompute, never reuse)."""
    c = cfg_adj.lbm
    return hashlib.md5(json.dumps([tier, c.cells_per_d, list(c.domain_d), c.body_offset_d, c.collision, c.steady_tol]).encode()).hexdigest()[:10]


def train_range(A) -> dict:
    """L1: (log Re, AoA deg) range of the training runs, stored in every checkpoint."""
    tr = A["split"] == 0
    return {"log_re": [float(A["re"][tr].log().min()), float(A["re"][tr].log().max())],
            "aoa": [float(A["aoa"][tr].min()), float(A["aoa"][tr].max())]}


def load_for_eval(path, kind, device, A=None):
    """-> (model, nurbs). L5: the checkpoint's model kind must match `kind`. L1: sets model.cond_range (from the checkpoint, else
    from the training split of `A` for checkpoints saved before it was stored)."""
    ck = load_ckpt(path, device)
    ck_kind = ck.get("metrics", {}).get("kind")
    if ck_kind is not None and ck_kind != kind:
        raise SystemExit(f"{path} is a '{ck_kind}' checkpoint but --model is '{kind}'")
    cfg = Config(ck["config"])
    g = cfg.geometry
    nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=torch.float32)
    model = build_model(cfg, kind, nurbs).to(device)
    model.load_state_dict(ck["model"])
    model.cond_range = ck.get("train_range") or (train_range(A) if A is not None else None)
    return model.eval(), nurbs


def build_model(cfg, kind, nurbs):
    P, v2 = cfg.pinn, cfg.pinn.version == "v2"
    if kind == "mlp":
        return MLPPINN(nurbs.D, P.mlp_hidden, P.mlp_layers, P.fourier_features, P.fourier_scale, v2)
    return GeometryCrossAttentionPINN(nurbs, P.d_model, P.heads, P.self_attn_layers, P.cross_attn_layers, v2,
                                      P.fourier_features, P.fourier_scale)


def cond_of(re, aoa):
    return torch.stack([re.log(), torch.deg2rad(aoa).sin(), torch.deg2rad(aoa).cos()], -1)


def slice_batch(b: TrainBatch, i: int, j: int) -> TrainBatch:
    return TrainBatch(**{f.name: (Boundary(*(t[i:j] for t in dataclasses.astuple(v))) if isinstance(v, Boundary) else v if v is None else v[i:j])
                         for f in dataclasses.fields(b) for v in [getattr(b, f.name)]})


def accumulate(model, batch: TrainBatch, cfg, step: int, micro: int, log: bool = False, nurbs=None):
    """F4: backward per chunk of `micro` shapes, gradients scaled by the chunk fraction (every loss term is a mean over
    shapes, so the sum equals the full-batch gradient). -> (total tensor, logged floats or {}). Caller zeroes grads."""
    S, total, logs = batch.alpha.shape[0], 0.0, {}
    for i in range(0, S, micro):
        sub = slice_batch(batch, i, i + micro)
        w = sub.alpha.shape[0] / S
        loss, d = hybrid_loss(model, nurbs, sub, cfg, step, log)
        (w * loss).backward()
        total = total + w * loss.detach()
        for k, v in d.items():
            logs[k] = logs.get(k, 0.0) + w * v
    return total, logs


def lr_at(step: int, cfg) -> float:
    """Linear warm-up over min(200, steps/5), constant, then cosine to lr_min over the main stage."""
    T = t = cfg.train
    warm = min(200, t.adam_steps // 5)
    main0 = min(t.warmup_data_only + t.pde_ramp, t.adam_steps - 1)
    if step < warm:
        return T.lr * (step + 1) / warm
    if step < main0:
        return T.lr
    r = (step - main0) / max(t.adam_steps - main0, 1)
    return T.lr_min + 0.5 * (T.lr - T.lr_min) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def _flat_holdout(A, device, dtype):
    idx = (A["split"] == 1).nonzero()[:, 0]
    keep = ~A["edge"][idx]
    n = int(keep.sum(1).min())  # fluid anchor points only; the same count per shape by construction
    sel = lambda t: torch.stack([t[i][keep[k]][:n] for k, i in enumerate(idx)]).to(device, dtype)
    alpha, re, aoa = (A[k][idx].to(device, dtype) for k in ("alpha", "re", "aoa"))
    return alpha, cond_of(re, aoa), sel(A["xy"]), sel(A["u"]), sel(A["v"]), sel(A["p"]), A["cd"][idx].to(device, dtype)


def holdout_eval(model, nurbs, A, device, chunk: int = 4000) -> dict:
    """Relative L2 of (u, v) and of p (mean offset removed) on the holdout anchors."""
    alpha, cond, xy, u, v, p, _ = _flat_holdout(A, device, nurbs.x_ctrl.dtype)
    bd = _rotated(nurbs, alpha, cond, None, None)
    out = []
    with torch.enable_grad():
        for s in range(0, xy.shape[1], chunk):
            f = predict_flow(model, xy[:, s:s + chunk], alpha, cond, bd)
            out.append(torch.stack([f.u.detach(), f.v.detach(), f.p.detach()], -1))
    pred = torch.cat(out, 1)
    uv = lambda a, b: ((a[..., :2] - b[..., :2]) ** 2).sum((1, 2)) / (b[..., :2] ** 2).sum((1, 2))
    ref = torch.stack([u, v, p], -1)
    pc, rc = pred[..., 2] - pred[..., 2].mean(1, keepdim=True), ref[..., 2] - ref[..., 2].mean(1, keepdim=True)
    e_uv, e_p = uv(pred, ref).sqrt(), ((pc - rc) ** 2).sum(1).sqrt() / (rc ** 2).sum(1).sqrt()
    glob = (((pred[..., :2] - ref[..., :2]) ** 2).sum() / (ref[..., :2] ** 2).sum()).sqrt()
    return {"vel_l2": float(glob), "p_l2": float(e_p.mean()), "vel_l2_per_shape": e_uv.tolist(), "p_l2_per_shape": e_p.tolist()}


@torch.no_grad()
def add_near_wall(A, nurbs, dist: float):
    """A['near'] [S, P] bool: anchor points within `dist` of the body (sampling.near_frac of each batch comes from them)."""
    a = A["alpha"].to(nurbs.x_ctrl)
    bd = _rotated(nurbs, a, cond_of(A["re"], A["aoa"]).to(a), None, None)
    A["near"] = torch.cat([wall_distance(A["xy"][i:i + 1].to(a), Boundary(*(t[i:i + 1] for t in dataclasses.astuple(bd)))) < dist
                           for i in range(a.shape[0])]).cpu()
    return A


def train(cfg, kind, out: Path, device, A, nurbs=None, quiet=False) -> tuple:
    """-> (model with the best-holdout weights loaded, nurbs, results dict). Writes log.csv, ckpt/, config.yaml."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(dump_config(cfg))
    seed_everything(cfg.seed)
    g, T = cfg.geometry, cfg.train
    nurbs = nurbs or NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=torch.float32)
    model = build_model(cfg, kind, nurbs).to(device)
    if cfg.sampling.near_frac > 0 and "near" not in A:
        add_near_wall(A, nurbs, cfg.sampling.near_dist)
    if cfg.loss.w_grad > 0 and "gcd" not in A:
        attach_adjoint_grads(A, Path(TRAIN_ADJ))
    tr = A["split"] == 0
    model.norm.fit(A["alpha"][tr].to(device), cond_of(A["re"][tr], A["aoa"][tr]).to(device))
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)
    rows, best, t0 = [], {"vel_l2": float("inf"), "step": -1}, time.perf_counter()
    sample = lambda: sample_batch(A, nurbs, cfg, gen, device=device)

    def snapshot(step, row):
        nonlocal best
        m = holdout_eval(model, nurbs, A, device)
        row.update(vel_l2=m["vel_l2"], p_l2=m["p_l2"])
        norm = {"alpha_mean": model.norm.a_mean, "alpha_std": model.norm.a_std, "xy_scale": 1.0}
        ck = {"model": model.state_dict(), "config": cfg.to_dict(), "norm": norm, "step": step,
              "metrics": {"vel_l2": m["vel_l2"], "p_l2": m["p_l2"], "kind": kind}, "train_range": train_range(A)}
        save_ckpt(out / "ckpt" / f"step_{step:06d}.pt", ck, str(device))
        if math.isfinite(m["vel_l2"]) and m["vel_l2"] < best["vel_l2"]:
            best = {"vel_l2": m["vel_l2"], "step": step}
            save_ckpt(out / "ckpt" / "best.pt", ck, str(device))
        if not quiet:
            print(f"  eval step {step}: vel_l2 {m['vel_l2']:.4f} p_l2 {m['p_l2']:.4f} (best {best['vel_l2']:.4f}@{best['step']})", flush=True)

    def flush():
        keys = ["step", "wall_s"] + sorted({k for r in rows for k in r} - {"step", "wall_s"})
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)

    log_every = max(1, min(50, T.adam_steps // 10))
    model.train()
    for step in range(T.adam_steps):
        lr = lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        do_log = step % log_every == 0 or step == T.adam_steps - 1
        _, logs = accumulate(model, sample(), cfg, step, T.micro_batch_shapes, do_log, nurbs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), T.grad_clip)
        opt.step()
        row = None
        if do_log:
            if not math.isfinite(logs["total"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            sync(device)
            row = {"step": step, "wall_s": time.perf_counter() - t0, "lr": lr, **logs}
            if not quiet:
                print(f"step {step} loss {logs['total']:.4g} pde {logs['pde']:.3g} data {logs['data']:.3g} wall {row['wall_s']:.0f}s", flush=True)
            rows.append(row)
        if (step + 1) % T.eval_every == 0 or step == T.adam_steps - 1:
            if row is None:
                row = {"step": step, "wall_s": time.perf_counter() - t0, "lr": lr}
                rows.append(row)
            snapshot(step, row)
            flush()

    if T.lbfgs_steps > 0:  # fixed batch, resampled every 50 L-BFGS steps; best.pt guards against a bad refine
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=10, history_size=50, line_search_fn="strong_wolfe")
        batch, loss_t = None, None

        def closure():
            nonlocal loss_t
            lb.zero_grad()
            loss_t, _ = accumulate(model, batch, cfg, T.adam_steps, T.micro_batch_shapes, nurbs=nurbs)
            return loss_t

        for k in range(T.lbfgs_steps):
            if k % 50 == 0:
                batch = sample()
            lb.step(closure)
            if (k + 1) % 100 == 0 or k == T.lbfgs_steps - 1:
                sync(device)
                row = {"step": T.adam_steps + k, "wall_s": time.perf_counter() - t0, "total": float(loss_t)}
                rows.append(row)
                snapshot(T.adam_steps + k, row)
                flush()
    sync(device)
    wall = time.perf_counter() - t0
    model.load_state_dict(load_ckpt(out / "ckpt" / "best.pt", device)["model"])
    model.eval()
    final = next((r for r in reversed(rows) if "total" in r and "pde" in r), rows[-1])
    res = {"model": kind, "adam_steps": T.adam_steps, "lbfgs_steps": T.lbfgs_steps, "wall_s": wall, "best_step": best["step"],
           "best_vel_l2": best["vel_l2"], "final_losses": {k: v for k, v in final.items() if k not in ("step", "wall_s", "lr")},
           "fourier_scale": cfg.pinn.fourier_scale, "device_info": device_info(device),
           "config_hash": hashlib.md5(dump_config(cfg).encode()).hexdigest()[:10]}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    return model, nurbs, res


def _adjoint_grads(nurbs, A, cfg_adj, G, tier, cache: Path, device):
    """CD adjoint gradients for the first G holdout shapes (cached: they do not depend on the PINN)."""
    alpha, cond, *_ = _flat_holdout(A, "cpu", torch.float32)
    alpha, cond = alpha[:G], cond[:G]
    h = adj_hash(cfg_adj, tier)
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        if c.get("cfg_hash") == h and c["alpha"].shape == alpha.shape and torch.allclose(c["alpha"], alpha) and c["device"] == str(device):
            return c
    dcd, cd, wall = [], [], []
    for i in range(G):
        re, aoa = float(torch.exp(cond[i, 0])), float(torch.rad2deg(torch.atan2(cond[i, 1], cond[i, 2])))
        r = lbm_value_and_grad(alpha[i].to(device), nurbs, Condition(re, aoa), cfg_adj, CostWeights(0.0, 0.0, 0.0), None)
        dcd.append(r.extras["dI_dalpha"].cpu()), cd.append(float(r.cd)), wall.append(r.wall_s)
        print(f"  adjoint shape {i}: cd {cd[-1]:.4f} wall {wall[-1]:.1f}s", flush=True)
    c = {"tier": tier, "alpha": alpha, "dcd": torch.stack(dcd), "cd": torch.tensor(cd), "wall_s": torch.tensor(wall), "device": str(device), "cfg_hash": h}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c, cache)
    return c


def precompute_train_adjoints(A, nurbs, cache: Path, adj_tier="dev", device="cpu"):
    """G30: adjoint dCD/dalpha and dCL/dalpha (one call, weights=None) for every training run, each at its own Re.
    Resumable: `cache` = {run id: {dcd, dcl, cd, cl, wall_s}}, rewritten after every run."""
    cfg_adj = load_config(CASE, adj_tier, [])
    done, h = (torch.load(cache, weights_only=False) if cache.exists() else {}), adj_hash(cfg_adj, adj_tier)
    if done and done.get("__cfg__", h) != h:
        print(f"adjoint cache {cache}: settings changed, recomputing", flush=True)
        done = {}
    done.setdefault("__cfg__", h)  # shortcut: a cache saved before the hash existed is adopted as-is (it was made with the case-1 dev settings)
    ids = [r["id"] for r in A["meta"]["runs"]]
    for i in (A["split"] == 0).nonzero()[:, 0].tolist():
        if ids[i] in done:
            continue
        r = lbm_value_and_grad(A["alpha"][i].to(device, nurbs.x_ctrl.dtype), nurbs, Condition(float(A["re"][i]), float(A["aoa"][i])),
                               cfg_adj, None, None)
        done[ids[i]] = dict(dcd=r.dcd_dalpha.cpu(), dcl=r.dcl_dalpha.cpu(), cd=float(r.cd), cl=float(r.cl), wall_s=r.wall_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(done, cache)
        print(f"adjoint {ids[i]}: cd {r.cd:.4f} (anchor {A['cd'][i]:.4f}) wall {r.wall_s:.0f}s", flush=True)
    return done


def attach_adjoint_grads(A, cache: Path):
    """A['gcd'], A['gcl'] [S, D]: adjoint gradients of the steady (kind 0) training runs, NaN elsewhere (time-averaged runs have none);
    needs a complete cache. Symmetric-run gradients ([k]) go in the first k columns of a cambered [2k] row (see hybrid_loss)."""
    done = torch.load(cache, weights_only=False)
    S, D = A["alpha"].shape
    A["gcd"], A["gcl"] = torch.full((S, D), float("nan")), torch.full((S, D), float("nan"))
    for i, rid in enumerate(r["id"] for r in A["meta"]["runs"]):
        if rid in done:
            for key, src in (("gcd", "dcd"), ("gcl", "dcl")):
                A[key][i] = 0.0 if done[rid][src].numel() < D else float("nan")
                A[key][i, :done[rid][src].numel()] = done[rid][src]
    need = (A["split"] == 0) & (A["kind"] == 0)
    assert torch.isfinite(A["gcd"][need]).all(), f"{cache} is incomplete: run --precompute-train-adjoint"
    return A


def gate3(model, nurbs, A, device, out: Path, adj_tier="dev", grad_shapes=5, cache=None, fname="gate3.json") -> dict:
    model.eval()
    h = holdout_eval(model, nurbs, A, device)
    alpha, cond, *_, cd_ref = _flat_holdout(A, device, nurbs.x_ctrl.dtype)
    cd, cl = surface_forces(model, nurbs, alpha, cond)
    cd_err = ((cd.detach() - cd_ref).abs() / cd_ref.abs()).cpu()

    cfg_adj = load_config(CASE, adj_tier, [])
    cache = Path(cache or f"runs/adjoint_cache/holdout_{adj_tier}.pt")
    G = min(grad_shapes, alpha.shape[0])
    adj = _adjoint_grads(nurbs, A, cfg_adj, G, adj_tier, cache, device)
    ev, cos, cos_free = PINNEvaluator(model, nurbs), [], []
    free = nurbs.free_idx.cpu()
    for i in range(G):
        r = ev.evaluate(alpha[i], Condition(float(torch.exp(cond[i, 0])), float(torch.rad2deg(torch.atan2(cond[i, 1], cond[i, 2])))))
        g, ga = r.dcd_dalpha.cpu(), adj["dcd"][i]
        cos.append(float(torch.nn.functional.cosine_similarity(g, ga, dim=0)))
        cos_free.append(float(torch.nn.functional.cosine_similarity(g[free], ga[free], dim=0)))  # the design variables

    cnd0 = Condition(20.0, 0.0)
    for _ in range(3):
        ev.evaluate(alpha[0], cnd0)
    lat = []
    for _ in range(20):
        with Timer(device) as t:
            ev.evaluate(alpha[0], cnd0)
        lat.append(t.elapsed)
    B, a = 64, alpha[:1].repeat(64, 1)
    cB = cond[:1].repeat(64, 1)

    def batched():
        x = a.clone().requires_grad_(True)
        torch.autograd.grad(surface_forces(model, nurbs, x, cB)[0].sum(), x)
    batched()
    with Timer(device) as t:
        for _ in range(3):
            batched()
    lbm_wall = float(adj["wall_s"].median())
    res = {"holdout_vel_l2": h["vel_l2"], "holdout_vel_l2_per_shape": h["vel_l2_per_shape"], "holdout_p_l2": h["p_l2"],
           "cd_rel_err_mean": float(cd_err.mean()), "cd_rel_err_per_shape": cd_err.tolist(), "cd_pinn": cd.detach().cpu().tolist(),
           "cd_lbm": cd_ref.cpu().tolist(), "grad_cosine_full_mean": statistics.mean(cos), "grad_cosine_full_per_shape": cos,
           "grad_cosine_free_mean": statistics.mean(cos_free), "grad_cosine_free_per_shape": cos_free,
           "grad_shapes": G, "adjoint_tier": adj_tier, "latency_single_cd_grad_s": statistics.median(lat),
           "throughput_shapes_per_s_batch64": 3 * B / t.elapsed, "adjoint_lbm_wall_s_median": lbm_wall,
           "speedup_vs_adjoint_lbm": lbm_wall / statistics.median(lat), "device": str(device)}
    res["pass"] = {"vel_l2<2%": res["holdout_vel_l2"] < 0.02, "cd<5%": res["cd_rel_err_mean"] < 0.05, "cos_free>0.9": res["grad_cosine_free_mean"] > 0.9}
    (out / fname).write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "transformer"])
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default=None)
    ap.add_argument("-o", "--set", action="append", default=[], dest="sets")
    ap.add_argument("--anchors", default=ANCHORS)
    ap.add_argument("--mix-steady", default=None, help="v2: steady symmetric anchors mixed into --anchors (cambered layout, kind 0)")
    ap.add_argument("--eval-only", default=None, help="checkpoint to run Gate 3 on (no training)")
    ap.add_argument("--sweep", action="store_true", help="Fourier-scale sweep (use -o to cut the budget)")
    ap.add_argument("--scales", default="0.25,0.5,1,2")
    ap.add_argument("--precompute-train-adjoint", action="store_true", help="G30: cache adjoint grads of the training runs")
    ap.add_argument("--train-adjoint-cache", default=TRAIN_ADJ)
    ap.add_argument("--adjoint-tier", default="dev", help="tier of the adjoint-LBM reference for the gradient cosine")
    ap.add_argument("--adjoint-cache", default=None)
    ap.add_argument("--grad-shapes", type=int, default=5)
    ap.add_argument("--eval-set", default="holdout", help="holdout (validation, default) | test (clean test set) | path to a merged anchor file")
    a = ap.parse_args()

    cfg = load_config(None, a.tier, a.sets)
    device = get_device(a.device)
    A = load_ckpt(a.anchors, "cpu")
    assert not check_anchors(A), check_anchors(A)
    if a.mix_steady:
        A = mix_anchors(load_ckpt(a.mix_steady, "cpu"), A)
        assert not check_anchors(A), check_anchors(A)
    out = Path(a.out or f"runs/pinn_{a.model}_{cfg.pinn.version}_{a.tier}")

    if a.precompute_train_adjoint:
        g = cfg.geometry
        precompute_train_adjoints(A, NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=torch.float32),
                                  Path(a.train_adjoint_cache), a.adjoint_tier, device)
        return

    if a.sweep:
        rows = []
        for s in (float(x) for x in a.scales.split(",")):
            c = load_config(None, a.tier, a.sets + [f"pinn.fourier_scale={s}"])
            _, _, r = train(c, "mlp", out / f"mlp_s{s:g}", device, A)
            rows.append({"scale": s, "best_vel_l2": r["best_vel_l2"], "best_step": r["best_step"], "wall_s": r["wall_s"],
                         "adam_steps": r["adam_steps"]})
        res = {"rows": rows, "best_scale": min(rows, key=lambda r: r["best_vel_l2"])["scale"], "overrides": a.sets}
        (out / "sweep.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if a.eval_only:
        model, nurbs = load_for_eval(a.eval_only, a.model, device, A)
    else:
        model, nurbs, res = train(cfg, a.model, out, device, A)
        print("results:", json.dumps(res, indent=2))
        if cfg.pinn.version == "v2":  # gate3 is the Re 20 steady check; v2 is evaluated by scripts/03c_pinn_v2_eval.py
            return
    out.mkdir(parents=True, exist_ok=True)
    fname, cache = "gate3.json", a.adjoint_cache
    if a.eval_set != "holdout":
        path = TEST_ANCHORS if a.eval_set == "test" else a.eval_set
        A = load_ckpt(path, "cpu")
        assert not check_anchors(A), check_anchors(A)
        stem = "test" if a.eval_set == "test" else Path(path).stem
        fname, cache = f"gate3_{stem}.json", cache or (TEST_ADJ if a.eval_set == "test" else f"runs/adjoint_cache/{stem}.pt")
    r = gate3(model, nurbs, A, device, out, a.adjoint_tier, a.grad_shapes, cache, fname)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
