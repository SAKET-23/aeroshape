"""Plot helpers for the validation scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_force_history(hist, path, title="", ref=None, x=None):
    """hist: ForceHistory; x = abscissa scale divisor (steps per convective time) if given."""
    s = hist.step.cpu().numpy() / (x or 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(s, hist.cd.cpu().numpy(), label="CD")
    ax.plot(s, hist.cl.cpu().numpy(), label="CL")
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=0.8, label=f"ref {ref}")
    ax.set(xlabel="convective time" if x else "step", title=title)
    ax.legend()
    _save(fig, path)


def plot_field(a, path, title="", solid=None, cmap="RdBu_r"):
    """a: [Ny, Nx] array/tensor, symmetric colour scale; solid nodes are masked."""
    a = np.array(a.detach().cpu().numpy() if hasattr(a, "numpy") else a, dtype=float)
    if solid is not None:
        a[np.asarray(solid)] = np.nan
    v = np.nanpercentile(np.abs(a), 99) or 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(a, origin="lower", cmap=cmap, vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    _save(fig, path)


def plot_convergence(Ns, cds, path, ref=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(1 / np.array(Ns, dtype=float), cds, "o-")
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=0.8, label=f"report {ref}")
        ax.legend()
    ax.set(xlabel="1/N", ylabel="CD", title="Grid convergence, Re 20")
    _save(fig, path)


def write_table1_md(rows, path):
    """rows: dicts with re, report, literature, cd, error_pct, passed."""
    L = ["| Re | Report CD | Literature | This solver CD | Error vs report (%) | Gate |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['re']} | {r['report']} | {r['literature']} | {r['cd']:.3f} | {r['error_pct']:+.2f} | "
                 f"{'PASS' if r['passed'] else 'FAIL'} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n")


def plot_gradient_check(g_adj, g_fd, included, path, err=None):
    """Bars per control point, adjoint vs FD (report Fig 5.1); excluded points are grey hatched bars at the adjoint value."""
    ga, gf = (np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=float) for x in (g_adj, g_fd))
    i, inc = np.arange(len(ga)), np.isin(np.arange(len(ga)), included)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(i[inc] - 0.2, ga[inc], 0.4, color="C0")
    ax.bar(i[inc] + 0.2, gf[inc], 0.4, color="C1")
    ax.bar(i[~inc], ga[~inc], 0.8, color="0.8", hatch="///", edgecolor="0.4")  # excluded: adjoint value only
    handles = [Patch(color="C0", label="adjoint"), Patch(color="C1", label="finite difference")]
    if (~inc).any():
        handles.append(Patch(facecolor="0.8", edgecolor="0.4", hatch="///", label="excluded (topology change), adjoint"))
    ax.set(xlabel="control point (free: 1..k-2)", ylabel="dCD/dalpha",
           title="Adjoint vs FD" + (f", rel L2 error {err:.2e}" if err is not None else ""))
    ax.legend(handles=handles)
    _save(fig, path)


def plot_adjoint_field(F, solid, path, title="adjoint field |F*| (CD)"):
    """F: [Q, Ny, Nx] adjoint populations; shows the Euclidean magnitude over directions (report Fig 6.3)."""
    m = F.detach().cpu().double().pow(2).sum(0).sqrt().numpy()
    m[np.asarray(solid)] = np.nan
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(m, origin="lower", cmap="magma", vmax=np.nanpercentile(m, 99.5))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    _save(fig, path)


def plot_shapes(nurbs, groups, path, n=8):
    """groups: {name: alpha [S, D]}; one row of outlines (first n shapes) per group."""
    fig, axs = plt.subplots(len(groups), 1, figsize=(8, 2.2 * len(groups)), squeeze=False)
    for ax, (name, a) in zip(axs[:, 0], groups.items()):
        x, yu, yl = (t.detach().cpu().numpy() for t in nurbs.surface_y(a[:n].to(nurbs.x_ctrl)))
        x = np.broadcast_to(x, yu.shape)
        for i in range(yu.shape[0]):
            ax.plot(x[i], yu[i], lw=0.8), ax.plot(x[i], yl[i], lw=0.8, color=ax.lines[-1].get_color())
        ax.set(title=name, aspect="equal")
    _save(fig, path)


def plot_guidance_sweep(rows, path):
    """rows: [{guide_scale, cfg_scale, J}]; mean J vs guide_scale, one line per cfg_scale."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for c in sorted({r["cfg_scale"] for r in rows}):
        rs = sorted((r for r in rows if r["cfg_scale"] == c), key=lambda r: r["guide_scale"])
        ax.plot([r["guide_scale"] for r in rs], [r["J"] for r in rs], "o-", label=f"cfg {c:g}")
    ax.set(xlabel="guide_scale", ylabel="mean J", yscale="log")
    ax.legend()
    _save(fig, path)


def plot_denoise_gif(nurbs, alphas, path, fps=5, label="denoising step"):
    """alphas [T, D]: one frame per denoising step (matplotlib + pillow)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    x, yu, yl = (t.detach().cpu().numpy() for t in nurbs.surface_y(alphas.to(nurbs.x_ctrl)))
    fig, ax = plt.subplots(figsize=(5, 2.5))
    (u,), (l,) = ax.plot(x, yu[0]), ax.plot(x, yl[0])
    ax.set(ylim=(float(yl.min()) - 0.05, float(yu.max()) + 0.05), aspect="equal")

    def frame(i):
        u.set_ydata(yu[i]), l.set_ydata(yl[i]), ax.set_title(f"{label} {i + 1}/{len(yu)}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, frame, frames=len(yu)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _body(ax, bx, by):
    if bx is not None:
        ax.fill(bx, by, color="0.5", zorder=3)


def plot_field_comparison(xy, ref, pred, path, title="", body=None):
    """Scatter-point fields (e.g. |u|): reference (LBM), prediction (PINN), error. xy [P, 2]; body = (x, y) outline."""
    xy, ref, pred = (np.asarray(t.detach().cpu() if hasattr(t, "detach") else t, dtype=float) for t in (xy, ref, pred))
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    lo, hi = float(min(ref.min(), pred.min())), float(max(ref.max(), pred.max()))
    for ax, f, name, cm, kw in ((axs[0], ref, "LBM", "viridis", dict(vmin=lo, vmax=hi)), (axs[1], pred, "PINN", "viridis", dict(vmin=lo, vmax=hi)),
                                (axs[2], pred - ref, "PINN - LBM", "coolwarm", dict(vmin=-np.abs(pred - ref).max(), vmax=np.abs(pred - ref).max()))):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=f, s=2, cmap=cm, **kw)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set(title=f"{title} {name}".strip(), aspect="equal")
        _body(ax, *(body or (None, None)))
    _save(fig, path)
    np.savez(Path(path).with_suffix(".npz"), xy=xy, ref=ref, pred=pred)


def plot_attention(xy, attn, x_ctrl, path, title="", body=None):
    """attn [P, T] head-averaged cross-attention weights of the last block (T = control tokens, optional last = condition token).
    Left: mean attention per token; right: where the most-attended token is attended."""
    xy, attn = np.asarray(xy, dtype=float), np.asarray(attn, dtype=float)
    mean, top = attn.mean(0), int(attn.mean(0).argmax())
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw=dict(width_ratios=[1, 2]))
    a0.bar(np.arange(len(mean)), mean, color="C3")
    a0.set(xlabel="token (control point; last = condition)" if len(mean) > len(x_ctrl) else "control point", ylabel="mean attention")
    sc = a1.scatter(xy[:, 0], xy[:, 1], c=attn[:, top], s=2, cmap="magma")
    fig.colorbar(sc, ax=a1, shrink=0.8)
    a1.set(title=f"{title} attention on token {top}".strip(), aspect="equal")
    _body(a1, *(body or (None, None)))
    _save(fig, path)
    np.savez(Path(path).with_suffix(".npz"), xy=xy, attn=attn, x_ctrl=np.asarray(x_ctrl))


def plot_training_log(logs, path, keys=("total", "data", "pde", "vel_l2")):
    """logs: {name: {key: [values]} with 'step'}; one panel per key (log y), one line per run. NaN/empty entries are skipped."""
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axs[0], keys):
        for name, d in logs.items():
            s, v = np.asarray(d["step"], dtype=float), np.asarray(d.get(k, []), dtype=float)
            if len(v) == len(s) and np.isfinite(v).any():
                ok = np.isfinite(v) & (v > 0)
                ax.plot(s[ok], v[ok], label=name)
        ax.set(xlabel="step", title=k, yscale="log")
    axs[0, 0].legend(fontsize=7)
    _save(fig, path)


def plot_benchmark_bars(rows, path):
    """rows: result dicts (method, cd_lbm, drag_reduction, wall_s); bars of drag reduction and wall-clock per method."""
    names = [r["method"] for r in rows]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.8))
    a0.bar(names, [100 * r["drag_reduction"] for r in rows], color="C0")
    a0.set(ylabel="drag reduction vs start (%)")
    a1.bar(names, [r["wall_s"] for r in rows], color="C1")
    a1.set(ylabel="wall-clock (s)", yscale="log")
    for a in (a0, a1):
        a.tick_params(axis="x", rotation=45, labelsize=7)
        [t.set_ha("right") for t in a.get_xticklabels()]
    _save(fig, path)
