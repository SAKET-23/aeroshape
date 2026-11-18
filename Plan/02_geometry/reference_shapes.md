# `geometry/reference_shapes.py` — start shapes, targets and shape sampling

**Work package:** WP 2. **Depends on:** WP 1. **Decisions:** G2, G19, G20, G21.

## Purpose

Produce `alpha` for named shapes (circle, NACA 4-digit, RAE2822), and sample families of valid shapes for the LBM dataset (30) and the diffusion dataset (2,000+).

## Interface

```python
def fit_alpha(nurbs: NURBSAirfoil, x_up, y_up, x_lo=None, y_lo=None) -> Tensor   # [1, k or 2k], least squares
def circle_alpha(nurbs) -> Tensor                    # y = ±sqrt(x(1-x)), diameter 1
def naca4_alpha(nurbs, code: str) -> Tensor          # e.g. "0012", "2412"
def rae2822_alpha(nurbs) -> Tensor                   # from data/rae2822.csv
def reference_coords(name) -> dict                   # raw coordinates, for plots and distance metrics
def shape_distance(nurbs, alpha_a, alpha_b) -> Tensor   # RMS y-difference over the surface, chord units
def is_valid(nurbs, alpha, t_floor=0.02, x_hi=0.85) -> Tensor   # [B] bool: no self-intersection, thickness ≥ floor on (0.05, x_hi); 0.95 rejected NACA0012 and RAE2822 themselves
def lhs_shapes(nurbs, n, seed, family="dataset") -> Tensor   # [n, k or 2k]
def perturb(alpha, n, scale, smooth_len, seed, nurbs) -> Tensor     # smooth Gaussian-process perturbations of free entries
```

## Details

- **NACA 4-digit**: standard thickness `y_t = 5t(0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1036x⁴)` (closed trailing edge variant: last coefficient −0.1036) and camber line; upper/lower surfaces offset normal to the camber line. Sampled at 400 cosine-spaced x.
- **RAE2822**: published coordinates (about 65 points per surface) shipped as `src/aeroshape/geometry/data/rae2822.csv` (`surface,x,y`). Trailing edge closed by a linear blend over the last 2% chord if the published data have a finite thickness there.
- **Fit**: the surface x-positions from NURBS are fixed, so for each surface interpolate target `y` at the NURBS sample x, then solve `R_free a_free = y − R_pinned·0` by `torch.linalg.lstsq` on CPU float64. Report max and RMS error.
- **Circle (G21)**: y-only control points with fixed x cannot make an exact circle; the fitted shape is the start shape and its own LBM CD is the reference drag. The report's narrative (circle → airfoil) is preserved.
- **Choosing `k` for cambered cases**: script `scripts/lattice_table.py --fit-report` prints fit error vs. `k` for NACA0012, NACA2412, RAE2822. Pick the smallest `k` with max error < 2e-3 chord; write it into the case configs.

## Dataset shapes (30, for LBM anchors)

`lhs_shapes(n=30, family="dataset")`:
1. Latin hypercube in 3D: `(s1, s2, s3)`, mapped to barycentric weights over `{circle, NACA0012, RAE2822}` (normalize to sum 1), plus a thickness scale in `[0.6, 1.4]` (applied to the free entries) and an AoA in `[0°, 10°]` when the condition list includes AoA.
2. Blend `alpha = Σ w_i alpha_i`, scale, then add one smooth perturbation (`scale = 0.01` chord).
3. Reject invalid shapes, redraw with the next LHS point until 30 are accepted.
4. Seeded; holdout is the last 5 by LHS order, so it spans the space.
5. In symmetric mode RAE2822 is replaced by its thickness distribution (symmetrized).

## Diffusion shapes (2,000+)

`lhs_shapes(n, family="diffusion")` mixes: NACA 4-digit with `m ∈ [0, 6]%`, `p ∈ [0.2, 0.6]`, `t ∈ [8, 24]%` (40%); RAE2822 and NACA blends (30%); smooth perturbations of either (30%). All fitted and validity-filtered.

## Tests

| Test | Pass |
|---|---|
| NACA0012 fit | max error < 2e-3 chord at k = 8 symmetric |
| RAE2822 fit | max error < 2e-3 chord at the chosen cambered `k` |
| NACA formula | t/c of NACA0012 is 0.12 ± 1e-3 at x ≈ 0.30 |
| Validity | a self-intersecting shape (lower above upper) is rejected |
| LHS reproducible | same seed → identical tensor |
| All sampled shapes valid | `is_valid` true for every returned shape |

## Pitfalls

- NACA closed-TE coefficient differs from the open one; use the closed version (the LBM needs a closed body).
- Thin trailing edges make links whose second fluid node is solid; the grid mapper handles this (`far_ok`), but very thin TEs (< 1 lattice cell) also create disconnected solid cells. The validity check enforces thickness only on `(0.05, 0.95)`, and the grid mapper fills isolated fluid cells inside the body.
