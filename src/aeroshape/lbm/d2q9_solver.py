"""Vectorized differentiable D2Q9 solver: BGK/TRT, Bouzidi IBB, inlet/outlet, periodic top/bottom.
Arrays are [9, Ny, Nx]; node (j, i) is row j (y), column i (x)."""
import warnings

import torch

from aeroshape.core.types import FlowState, ForceHistory, SolveResult
from aeroshape.lbm.lattice import C, OPP, lattice_constants

_SHIFTS = [(int(c[1]), int(c[0])) for c in C]  # (c_y, c_x) on dims (0, 1)


def _cw(f):
    Cl, W, _ = lattice_constants(f.device, f.dtype)
    return Cl.to(f.dtype)[:, 0, None, None], Cl.to(f.dtype)[:, 1, None, None], W[:, None, None]


def moments(f):
    cx, cy, _ = _cw(f)
    rho = f.sum(0)
    return rho, (cx * f).sum(0) / rho, (cy * f).sum(0) / rho


def equilibrium(rho, ux, uy):
    cx, cy, w = _cw(rho.unsqueeze(0))
    cu = cx * ux + cy * uy
    return w * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2))


def collide(f, params, solid):
    rho, ux, uy = moments(f)
    rho, ux, uy = torch.where(solid, 1.0, rho), torch.where(solid, 0.0, ux), torch.where(solid, 0.0, uy)
    feq = equilibrium(rho, ux, uy)
    om = params.omega if params.sponge is None else params.sponge
    if params.collision == "bgk":
        fc = f - om * (f - feq)
    else:
        om_m = params.omega_minus
        if params.sponge is not None:  # keep the magic parameter Lambda fixed on the omega field
            lam = (1 / params.omega - 0.5) * (1 / params.omega_minus - 0.5)
            om_m = 1 / (0.5 + lam / (1 / om - 0.5))
        d = f - feq
        dp, dm = (d + d[OPP.to(f.device)]) / 2, (d - d[OPP.to(f.device)]) / 2
        fc = f - om * dp - om_m * dm
    return torch.where(solid, f, fc)


def stream(f_col):
    return torch.stack([torch.roll(f_col[i], shifts=_SHIFTS[i], dims=(0, 1)) for i in range(9)])


def apply_ibb(f_streamed, f_col, links, params):
    S = params.Ny * params.Nx
    Cl, _, opp = lattice_constants(f_col.device, f_col.dtype)
    d, node, q = links.dir, links.node, links.q
    cy, cx = Cl[d, 1], Cl[d, 0]
    back = ((node // params.Nx - cy) % params.Ny) * params.Nx + (node % params.Nx - cx) % params.Nx  # x_A - c_i
    fc = f_col.reshape(-1)
    a, b, c = fc[d * S + node], fc[d * S + back], fc[opp[d] * S + node]
    new = torch.where(q < 0.5, 2 * q * a + (1 - 2 * q) * b, a / (2 * q) + (2 * q - 1) / (2 * q) * c)
    new = torch.where(links.far_ok, new, a)
    return f_streamed.reshape(-1).index_put((opp[d] * S + node,), new).view_as(f_streamed)


def apply_inlet_outlet(f, params, outlet="fixed"):
    one = torch.ones(f.shape[1], 1, device=f.device, dtype=f.dtype)
    ref = equilibrium(one, params.u_lb * one, 0 * one)
    if outlet == "fixed":
        out = ref
    else:
        _, ux, uy = moments(f[:, :, -2:-1])
        out = equilibrium(one, ux, uy)
    return torch.cat([ref, f[:, :, 1:-1], out], 2)


def step(f, geom, params, outlet="fixed", force=0.0, open_x=True):
    """-> (f_next, f_col). `force`: uniform x body force (first-order, used by tests); open_x=False -> periodic in x."""
    f_col = collide(f, params, geom.solid)
    if force:
        cx, _, w = _cw(f)
        f_col = f_col + 3 * w * cx * force
    f_s = apply_ibb(stream(f_col), f_col, geom.links, params)
    if open_x:
        f_s = apply_inlet_outlet(f_s, params, outlet)
    _, _, w = _cw(f)
    return torch.where(geom.solid, w, f_s), f_col


def init_state(params, geom, dtype=torch.float32, perturb=0.0):
    """perturb: transverse kick uy = perturb * u_lb in a N x N box one diameter behind the body origin (breaks the
    up/down symmetry so unsteady shedding starts; 0 = off)."""
    dev, N = geom.solid.device, params.N
    one = torch.ones(params.Ny, params.Nx, device=dev, dtype=dtype)
    uy = torch.zeros_like(one)
    if perturb:
        i0, j0 = geom.origin
        uy[max(j0, 0):j0 + N, i0 + N:i0 + 2 * N] = perturb * params.u_lb
    feq = equilibrium(one, params.u_lb * one, uy)
    return torch.where(geom.solid, lattice_constants(dev, dtype)[1][:, None, None], feq)


def detect_period(cl, record_every, n_consistent=3, tol=0.05, decay=0.5):
    """cl: 1-D CL records (post-transient). Period in steps from upward crossings of the window mean: the last run of
    n_consistent consecutive periods whose lengths agree within tol; None if there is none or the oscillation decays
    (amplitude over the window's last n_consistent periods < decay x amplitude over the first n_consistent periods of the window)."""
    cl = cl - cl.mean()
    up = torch.nonzero((cl[:-1] < 0) & (cl[1:] >= 0)).flatten()
    if len(up) < n_consistent + 1:
        return None
    if cl[up[-n_consistent - 1]:].abs().max() < decay * cl[up[0]:up[n_consistent] + 1].abs().max():
        return None
    a, b = cl[up].double(), cl[up + 1].double()
    d = (up.double() - a / (b - a)).diff()  # fractional crossings -> period lengths in records
    for i in range(len(d) - n_consistent, -1, -1):
        w = d[i:i + n_consistent]
        if w.max() - w.min() <= tol * w.mean():
            return round(w.mean().item() * record_every)
    return None


_COMPILE_FAILED = False  # process-wide: after one failed probe, later run() calls go straight to eager
GRAPH_REPLAYS = 0  # CUDA-graph replays so far (test hook)
_COMPILED = {}  # (id(geom), id(params), outlet) -> (geom, params, fn); holds refs so ids stay valid


def _compiled_step(geom, params, outlet):
    """torch.compile of the per-step fn (default mode: reduce-overhead's CUDA graphs would alias returned buffers; the
    default inductor fusion already removes the ~100 launches). Eager fallback with one warning if compile raises."""
    key = (id(geom), id(params), outlet)
    if key not in _COMPILED:
        fn = torch.compile(lambda f: step(f, geom, params, outlet))
        _COMPILED.clear()  # one entry only: old geometries must be released
        _COMPILED[key] = (geom, params, fn)
    return _COMPILED[key][2]


class _Graph:
    """CUDA graph of m eager steps, static in/out buffers (grid, links, params fixed for the run)."""

    def __init__(self, f, geom, params, outlet, m):
        def body(x):
            for _ in range(m):
                x, _ = step(x, geom, params, outlet)
            return x
        self.m, self.inp = m, f.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body(f.clone())
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = body(self.inp)

    def __call__(self, f):
        self.inp.copy_(f)
        self.g.replay()
        return self.out.clone()


def _state(f, n):
    rho, ux, uy = moments(f)
    return FlowState(f, rho, ux, uy, n)


@torch.no_grad()
def run(f0, geom, params, n_steps=None, mode="steady", record_every=50, callback=None, force_fn=None,
        outlet="fixed", steady_tol=1e-6, max_convective_times=60, transient=30, average_periods=4, require_period=False,
        window_convective_times=0, compile=False, cuda_graph=False, graph_steps=50):
    """force_fn(f_col, f_next, geom, params) -> (cd, cl) 0-dim tensors; None = no force tracking, fixed step count.
    shortcut: forces recorded every record_every steps only (not every step in the last convective time).
    compile / cuda_graph (opt-in, forward only): torch.compile the step; replay CUDA graphs of graph_steps steps between
    record steps (CUDA only, else ignored). The f_sum averaging phase stays per-step (it accumulates every step)."""
    global _COMPILE_FAILED, GRAPH_REPLAYS
    K = round(params.N / params.u_lb)
    cap_steps = max_convective_times * K * (1 + average_periods) + round(window_convective_times * K)
    dev, dt = f0.device, f0.dtype
    hs = torch.zeros(cap_steps // record_every + 2, dtype=torch.long, device=dev)
    hcd, hcl = torch.zeros(hs.shape[0], device=dev, dtype=dt), torch.zeros(hs.shape[0], device=dev, dtype=dt)
    f, n, k = f0, 0, 0
    f_sum, n_sum = None, 0

    stepper = lambda f: step(f, geom, params, outlet)  # noqa: E731
    if compile and not _COMPILE_FAILED:
        try:
            stepper = _compiled_step(geom, params, outlet)
            stepper(f0)  # compile errors surface lazily, on the first call
        except Exception as e:  # noqa: BLE001
            _COMPILE_FAILED = True
            warnings.warn(f"torch.compile failed ({type(e).__name__}: {str(e)[:200]}); running eager", RuntimeWarning)
            stepper = lambda f: step(f, geom, params, outlet)  # noqa: E731
    graph = [None]
    graph_steps = min(graph_steps, record_every - 1)  # a chunk must fit the record-free gap, else the graph never fires
    use_graph = cuda_graph and f0.is_cuda and graph_steps > 0

    def advance(f, n, k):
        f_next, f_col = stepper(f)
        n += 1
        if n % record_every == 0:
            if force_fn is not None:
                hcd[k], hcl[k] = force_fn(f_col, f_next, geom, params)
                hs[k] = n
                k += 1
            if callback is not None:
                callback(n, f_next)
        return f_next, n, k

    def advance_n(f, n, k, cnt):
        """cnt steps; chunks free of record steps are replayed as a CUDA graph."""
        end = n + cnt
        while n < end:
            free = min(end, n + record_every - n % record_every - 1) - n  # steps before the next record step
            if use_graph and free >= graph_steps:
                if graph[0] is None:
                    graph[0] = _Graph(f, geom, params, outlet, graph_steps)
                f, n = graph[0](f), n + graph_steps
                GRAPH_REPLAYS += 1
            else:
                f, n, k = advance(f, n, k)
        return f, n, k

    converged, period = False, None
    if force_fn is None:
        f, n, k = advance_n(f, n, k, n_steps if n_steps is not None else max_convective_times * K)
        converged = n_steps is not None
    elif mode == "steady":
        lag, limit = max(1, round(K / record_every)), n_steps or max_convective_times * K
        while n < limit and not converged:
            f, n, k2 = advance_n(f, n, k, min(record_every - n % record_every, limit - n))
            if k2 > k:
                k = k2
                if k > lag:  # one host sync per log interval
                    a, b, c, d = torch.stack([hcd[k - 1], hcd[k - 1 - lag], hcl[k - 1], hcl[k - 1 - lag]]).cpu().tolist()
                    # CL (often ~0) is judged on the force scale max(|CL|, |CD|), so it is no stricter than CD
                    converged = abs(a - b) < steady_tol * abs(a) and abs(c - d) < steady_tol * max(abs(c), abs(a))
    else:
        limit = n_steps or max_convective_times * K
        n_tr = min(transient * K, limit)
        f, n, k = advance_n(f, n, k, n_tr)
        k_tr = k  # no force-based period detection before the transient ends (start-up wiggles)
        while True:
            period = detect_period(hcl[k_tr:k].cpu(), record_every)
            if period:
                break
            if n >= limit:
                break
            f, n, k = advance_n(f, n, k, K)
        if period:
            k0, f_sum = k, torch.zeros_like(f)
            for _ in range(average_periods * period):
                f, n, k = advance(f, n, k)
                f_sum, n_sum = f_sum + f, n_sum + 1
            hcd_m, hcl_m = hcd[k0:k].mean(), hcl[k0:k].mean()
            converged = True
        elif window_convective_times and not require_period:  # no period: long-window mean over the next W convective times
            k0, f_sum = k, torch.zeros_like(f)
            for _ in range(round(window_convective_times * K)):
                f, n, k = advance(f, n, k)
                f_sum, n_sum = f_sum + f, n_sum + 1
            hcd_m, hcl_m = hcd[k0:k].mean(), hcl[k0:k].mean()
            converged = True
        else:
            msg = "unsteady run found no period (no shedding or budget too short): returning the last record, f_mean=None"
            if require_period:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning)
    if force_fn is None:
        cd = cl = torch.zeros((), device=dev, dtype=dt)
    elif mode == "unsteady" and n_sum:
        cd, cl = hcd_m, hcl_m
    else:
        cd, cl = hcd[max(k - 1, 0)], hcl[max(k - 1, 0)]
    hist = ForceHistory(hs[:k], hcd[:k], hcl[:k])
    return SolveResult(_state(f, n), None if f_sum is None else f_sum / n_sum, hist, cd, cl, period, converged)
