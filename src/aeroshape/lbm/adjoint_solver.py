"""Discrete adjoint of the D2Q9 step and the shape gradient (Plan/03_lbm/adjoint_solver.md, derivations 4.1-4.11)."""
import time
import warnings
from dataclasses import replace

import torch

from aeroshape.core.types import AdjointResult, CostWeights, EvalResult
from aeroshape.geometry.ibb_grid_mapper import map_to_grid
from aeroshape.geometry.nurbs import rotate
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.aero_forces import force_fn
from aeroshape.lbm.lattice import lattice_constants, make_params

Tensor = torch.Tensor


def _mask(F, geom):
    """Zero adjoint on solid nodes and inlet/outlet columns (forward output there is a constant, derivations 4.8)."""
    F = F.masked_fill(geom.solid, 0.0)
    F[:, :, 0] = 0
    F[:, :, -1] = 0
    return F


def cost_source(f, geom, params, cost, outlet="fixed") -> tuple[Tensor, Tensor]:
    """dg/df [9, Ny, Nx] and dg/dq [L] for g(f; q) = cost(step(f; q)), by local autograd (derivations 4.9)."""
    f, q = f.detach().requires_grad_(True), geom.links.q.detach().requires_grad_(True)
    with torch.enable_grad():
        g = replace(geom, links=replace(geom.links, q=q))
        fn, fc = S.step(f, g, params, outlet)
        return torch.autograd.grad(cost(fc, fn, g, params), (f, q))


def adjoint_step_hand(F_next, f, geom, params, outlet="fixed") -> Tensor:
    """Exact transpose of S.step for BGK (+ sponge) with fixed outlet: IBB, reverse streaming, collision (4.4-4.7)."""
    assert params.collision == "bgk" and outlet == "fixed", "hand transpose: BGK + fixed outlet only"
    Ny, Nx = params.Ny, params.Nx
    SZ = Ny * Nx
    Cl, _, opp = lattice_constants(f.device, f.dtype)
    lk = geom.links
    d, node, q = lk.dir, lk.node, lk.q
    back = (((node // Nx - Cl[d, 1]) % Ny) * Nx + (node % Nx - Cl[d, 0]) % Nx)  # x_A - c_i
    # 1. IBB transpose: the overwritten slot (opp_i, x_A) receives no streamed value; its adjoint goes to f_col
    flat = _mask(F_next, geom).reshape(-1)
    slot = opp[d] * SZ + node
    a = flat[slot]
    flat = flat.index_fill(0, slot, 0.0)
    lo = q < 0.5
    ca, cb, cc = torch.where(lo, 2 * q, 1 / (2 * q)), torch.where(lo, 1 - 2 * q, 0 * q), torch.where(lo, 0 * q, (2 * q - 1) / (2 * q))
    ca, cb, cc = torch.where(lk.far_ok, ca, 1 + 0 * q), torch.where(lk.far_ok, cb, 0 * q), torch.where(lk.far_ok, cc, 0 * q)
    # 2. reverse streaming G_i(x) = F_i(x + c_i), then add the IBB scatter (duplicate indices -> index_add_)
    G = torch.stack([torch.roll(flat.view(9, Ny, Nx)[i], shifts=(-S._SHIFTS[i][0], -S._SHIFTS[i][1]), dims=(0, 1)) for i in range(9)])
    G = G.reshape(-1).clone()
    G.index_add_(0, d * SZ + node, ca * a)
    G.index_add_(0, d * SZ + back, cb * a)
    G.index_add_(0, opp[d] * SZ + node, cc * a)
    G = G.view(9, Ny, Nx)
    # 3. adjoint collision in moment form (4.6) at f_ref
    _, ux, uy = S.moments(f)
    cx, cy, w = S._cw(f)
    cu = cx * ux + cy * uy
    wG = w * G
    A, C = wG.sum(0), (wG * cu**2).sum(0)
    Bx, By, Dx, Dy = (wG * cx).sum(0), (wG * cy).sum(0), (wG * cu * cx).sum(0), (wG * cu * cy).sum(0)
    Feq = (A * (1 + 1.5 * (ux**2 + uy**2)) - 4.5 * C + 3 * (cx * Bx + cy * By) + 9 * (cx * Dx + cy * Dy)
           - 3 * (ux * cx + uy * cy) * A)
    om = params.omega if params.sponge is None else params.sponge
    return torch.where(geom.solid, G, (1 - om) * G + om * Feq)  # solid collision is the identity


def adjoint_step_vjp(F_next, f, geom, params, outlet="fixed") -> Tensor:
    f = f.detach().requires_grad_(True)
    with torch.enable_grad():
        fn, _ = S.step(f, geom, params, outlet)
        return torch.autograd.grad(fn, f, F_next)[0]


def _gmres(mv, b, x, tol, max_mv, m=30):
    """Restarted GMRES(m) for mv(x) = b on flat tensors; returns x, relative-residual history, matvec count."""
    nb, hist, n, prev = b.norm(), [], 0, None
    if nb == 0:
        return torch.zeros_like(b), hist, n
    while n < max_mv:
        r = b - mv(x)
        n += 1
        beta = r.norm()
        rel = float(beta / nb)
        hist.append(rel)
        if rel < tol or (prev is not None and rel > 0.999 * prev):  # converged / stagnated (e.g. float32 floor)
            break
        prev = rel
        V, H, e1, k = [r / beta], torch.zeros(m + 1, m, dtype=torch.float64), torch.zeros(m + 1, dtype=torch.float64), 0
        e1[0] = float(beta)
        for j in range(m):
            w = mv(V[j])
            n += 1
            for i in range(j + 1):  # modified Gram-Schmidt
                H[i, j] = float(w @ V[i])
                w = w - H[i, j].item() * V[i]
            H[j + 1, j] = float(w.norm())
            k = j + 1
            y = torch.linalg.lstsq(H[:k + 1, :k], e1[:k + 1, None]).solution
            res = float((H[:k + 1, :k] @ y - e1[:k + 1, None]).norm()) / float(nb)
            hist.append(res)
            if res < tol or H[j + 1, j] < 1e-14 * float(beta) or n >= max_mv:
                break
            V.append(w / H[j + 1, j].item())
        x = x + sum(c * v for c, v in zip(y[:, 0].tolist(), V))
    return x, hist, n


def solve_adjoint(f_ref, geom, params, cost, mode="steady", horizon_steps=None, tol=1e-8, impl="hand",
                  outlet="fixed", solver="krylov", F0=None, stats=None) -> AdjointResult:
    """Steady: solve (I - A^T) F* = dg/df linearized at f_ref by GMRES (solver='krylov') or plain fixed-point iteration
    F* <- A^T F* + dg/df ('fixed_point'); F0 warm-starts; relative residual tol; stats['matvecs'] counts transposed steps.
    time_averaged is always fixed-horizon iteration. dI_dalpha is left empty (see shape_gradient).
    shortcut: time_averaged uses the final F* in the dI/dq formula (not the sum over iterations)."""
    f_ref = f_ref.detach()
    K = max(1, round(params.N / params.u_lb))
    g_f, g_q = cost_source(f_ref, geom, params, cost, outlet)
    g_f = _mask(g_f, geom)
    stepf = adjoint_step_hand if impl == "hand" else adjoint_step_vjp
    ta, cap = mode == "time_averaged", horizon_steps or 100 * K
    F = torch.zeros_like(f_ref) if F0 is None else _mask(F0.detach().clone(), geom)
    hist, prev, n = [], None, 0
    if not ta and solver == "krylov":
        shp = f_ref.shape
        mv = lambda v: (v.view(shp) - _mask(stepf(v.view(shp), f_ref, geom, params, outlet), geom)).reshape(-1)
        F, hist, n = _gmres(mv, g_f.reshape(-1), F.reshape(-1), tol, cap)
        F = F.view(shp)
        if hist[-1] >= tol:
            warnings.warn(f"steady adjoint (krylov) stopped at relative residual {hist[-1]:.2e} > tol after {n} matvecs")
        hist = [torch.tensor(h) for h in hist]
    else:
        for n in range(1, cap + 1):
            Fn = _mask(stepf(F, f_ref, geom, params, outlet) + g_f, geom)
            if ta and n % K == 0:  # |F*| per convective time; abort on >10x growth (unstable mean flow)
                nrm = Fn.norm()
                hist.append(nrm)
                if prev is not None and nrm > 10 * prev:
                    warnings.warn(f"adjoint norm grew {float(nrm / prev):.1f}x in one convective time (n={n}); stopping")
                    F = Fn
                    break
                prev = nrm
            elif not ta and n % 20 == 0:
                r = (Fn - F).norm() / Fn.norm()
                hist.append(r)
                if r < tol:
                    F = Fn
                    break
            F = Fn
        else:
            if not ta:
                warnings.warn("steady adjoint did not reach tol within the iteration cap")
    if stats is not None:
        stats["matvecs"] = n
    q = geom.links.q.detach().requires_grad_(True)
    with torch.enable_grad():
        fn, _ = S.step(f_ref, replace(geom, links=replace(geom.links, q=q)), params, outlet)
        (sq,) = torch.autograd.grad(fn, q, F)
    hist = torch.stack([h.to(f_ref) for h in hist]) if hist else f_ref.new_zeros(0)
    return AdjointResult(F, g_q + sq, f_ref.new_zeros(0), hist)


def shape_gradient(dI_dq: Tensor, q: Tensor, alpha: Tensor) -> Tensor:
    return torch.autograd.grad(q, alpha, dI_dq, retain_graph=True)[0].reshape(-1)


def _setup(a, nurbs, cond, cfg, N):
    p = make_params(cfg, cond, N, device=a.device, dtype=a.dtype)
    return p, map_to_grid(rotate(nurbs(a), cond.aoa_deg), p, body_offset_d=cfg.lbm.body_offset_d)


@torch.no_grad()
def _forward(g, p, cfg, mode="steady", f0=None):
    """-> (g, f_lin, cd, cl, f_warm, converged). Steady: converged state and (cd, cl) of one step at it. Unsteady with a
    period: f_lin = period-mean state (the adjoint's linearization point), (cd, cl) = the period-mean force.
    f_warm = instantaneous final state, the warm start for the next solve (re-kicked there so shedding restarts)."""
    L = cfg.lbm
    g = replace(g, links=replace(g.links, q=g.links.q.detach()))
    dt, pert = g.links.q.dtype, L.perturb if mode != "steady" else 0.0
    transient = L.transient if f0 is None else L.transient_warm  # warm start: wake already developed
    f0 = S.init_state(p, g, dt, perturb=pert) if f0 is None else f0 + (S.init_state(p, g, dt, perturb=pert) - S.init_state(p, g, dt))
    r = S.run(f0, g, p, force_fn=force_fn, mode="steady" if mode == "steady" else "unsteady", record_every=L.record_every,
              outlet=L.outlet, steady_tol=L.steady_tol, max_convective_times=L.max_convective_times,
              average_periods=L.average_periods, transient=transient,
              **({} if mode == "steady" else {"window_convective_times": L.window_convective_times}),
              compile=getattr(L, "compile", False), cuda_graph=getattr(L, "cuda_graph", False),
              graph_steps=getattr(L, "graph_steps", 49))
    if mode != "steady" and r.f_mean is not None:
        return g, r.f_mean, r.cd, r.cl, r.state.f, r.converged
    fn, fc = S.step(r.state.f, g, p, L.outlet)
    return g, r.state.f, *force_fn(fc, fn, g, p), r.state.f, r.converged


def _total(cd, cl, w: CostWeights | None):
    return cd if w is None else cd + w.lam * cl + 0.5 * w.rho_al * (cl - w.cl_ref) ** 2


def _fit(x, p):
    """Warm-start state if it matches the grid, else None."""
    return x if x is not None and x.shape[1:] == (p.Ny, p.Nx) else None


@torch.no_grad()
def lbm_value(alpha, nurbs, cond, cfg, N=None, f0=None, info=None):
    """Forward only (no adjoint): (cd, cl, warm-start f); f0 warm-starts the forward solve. info (dict) gets 'converged'."""
    a = alpha.detach().reshape(1, -1)
    p, g = _setup(a, nurbs, cond, cfg, N)
    _, _, cd, cl, f, conv = _forward(g, p, cfg, cfg.adjoint.mode, _fit(f0, p))
    if info is not None:
        info["converged"] = conv
    return cd, cl, f


def lbm_value_and_grad(alpha, nurbs, cond, cfg, weights: CostWeights | None = None, N=None, f0=None, F0=None) -> EvalResult:
    """Forward solve + adjoint + shape gradient. weights=None: separate adjoints for CD and CL (dcd/dcl_dalpha).
    weights given: ONE adjoint of CD + lam CL + rho/2 (CL - CL_ref)^2 with (CL - CL_ref) frozen; its gradient is
    extras['dI_dalpha'] (and dcd/dcl_dalpha are NaN). f0 / F0 warm-start the forward / adjoint; extras 'f', 'F_star' return
    the converged states (F_star is the CD adjoint when weights=None)."""
    t0 = time.perf_counter()
    a = alpha.detach().reshape(1, -1).requires_grad_(True)
    with torch.enable_grad():
        p, g = _setup(a, nurbs, cond, cfg, N)
    mode = cfg.adjoint.mode
    g_d, f_ref, cd, cl, f_warm, conv = _forward(g, p, cfg, mode, _fit(f0, p))
    t1 = time.perf_counter()
    K = round(p.N / p.u_lb)
    hz = round(cfg.adjoint.horizon_convective_times * K) if mode == "time_averaged" else None
    F0, st = _fit(F0, p), {}

    def adj(cost):
        r = solve_adjoint(f_ref, g_d, p, cost, mode, hz, cfg.adjoint.tol, cfg.adjoint.impl, cfg.lbm.outlet,
                          getattr(cfg.adjoint, "solver", "krylov"), F0, st)
        return shape_gradient(r.dI_dq, g.links.q, a), r

    extras = {}
    if weights is None:
        dcd, rd = adj(lambda *x: force_fn(*x)[0])
        dcl, _ = adj(lambda *x: force_fn(*x)[1])
    else:
        c = weights.lam + weights.rho_al * float(cl - weights.cl_ref)
        dI, rd = adj(lambda *x: (lambda z: z[0] + c * z[1])(force_fn(*x)))
        dcd = dcl = torch.full_like(dI, float("nan"))
        extras.update(I=_total(cd, cl, weights), dI_dalpha=dI)
    t2 = time.perf_counter()
    extras.update(t_forward=t1 - t0, t_adjoint=t2 - t1, residual_history=rd.residual_history, F_star=rd.F_star, dI_dq=rd.dI_dq,
                  f=f_warm, converged=conv, adjoint_matvecs=st["matvecs"])
    return EvalResult(cd, cl, dcd, dcl, t2 - t0, extras)


def fd_gradient(alpha, nurbs, cond, cfg, delta=1e-4, idx=None, weights: CostWeights | None = None, N=None) -> Tensor:
    """Central differences of CD (or the combined cost) for the entries idx (default all), warm-started (4.11)."""
    a = alpha.detach().reshape(1, -1)
    mode = cfg.adjoint.mode
    with torch.no_grad():
        _, g0 = _setup(a, nurbs, cond, cfg, N)
        p = make_params(cfg, cond, N, device=a.device, dtype=a.dtype)
        f0 = _forward(g0, p, cfg, mode)[4]
        out = []
        for k in (range(a.shape[1]) if idx is None else idx):
            v = []
            for s in (1, -1):
                b = a.clone()
                b[0, k] += s * delta
                _, g = _setup(b, nurbs, cond, cfg, N)
                v.append(_total(*_forward(g, p, cfg, mode, f0)[2:4], weights))
            out.append((v[0] - v[1]) / (2 * delta))
    return torch.stack(out)
