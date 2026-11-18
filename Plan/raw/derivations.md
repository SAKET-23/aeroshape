# Derivations

The maths the code implements, written out step by step. Equations the report shows only as images are reconstructed here from standard sources and marked **(reconstructed)**. Every module doc refers back to the numbered equations below.

## 0. Notation

| Symbol | Meaning |
|---|---|
| `alpha ∈ R^k` | shape design vector: y-values of the free NURBS control points |
| `c_i, w_i, i = 0..8` | D2Q9 velocities and weights; `ī = opp(i)` is the reverse direction |
| `f_i(x, n)` | population at node `x`, time step `n`; `f^col` is the post-collision value |
| `rho, u` | lattice density and velocity |
| `tau, omega = 1/tau` | BGK relaxation time and rate; `nu = (tau − 1/2)/3` |
| `q` | wall fraction along a link, `q ∈ (0, 1]` |
| `F*_i` | adjoint population (Lagrange multiplier of the LBM step) |
| `psi, p` | PINN stream function and pressure (non-dimensional) |
| `gamma_t, gamma_bar_t` | diffusion noise schedule (named `gamma` so `alpha` stays the shape) |

Units: lattice units inside the LBM (`Δx = Δt = 1`, `c_s² = 1/3`). Everywhere else non-dimensional: reference length `L_ref` (diameter or chord) = 1, `U_inf = 1`, density 1, `nu = 1/Re`.

## 1. NURBS (report ch. 3)

**(1.1)** B-spline basis by Cox–de Boor, knot vector `t_0 ≤ … ≤ t_{k+p}`:

```
N_{i,0}(s) = 1 if t_i ≤ s < t_{i+1} else 0        (last span closed on the right)
N_{i,p}(s) = (s − t_i)/(t_{i+p} − t_i) · N_{i,p−1}(s) + (t_{i+p+1} − s)/(t_{i+p+1} − t_{i+1}) · N_{i+1,p−1}(s)
```
with `0/0 := 0`. Clamped uniform knots: `p+1` zeros, uniform interior, `p+1` ones.

**(1.2)** Rational curve with weights `W_i`:

```
R_i(s) = N_{i,p}(s) W_i / Σ_j N_{j,p}(s) W_j
P(s)   = Σ_i R_i(s) P_i
```

With weights and `s`-samples fixed, `R [M, k]` is a constant matrix, so `y(s) = R · alpha` and `∂y/∂alpha = R`. Exact gradients for free.

**(1.3)** Derivative for normals: `dR_i/ds = (N'_i W_i S − N_i W_i S') / S²`, `S = Σ N_j W_j`, `N'_{i,p} = p/(t_{i+p} − t_i) N_{i,p−1} − p/(t_{i+p+1} − t_{i+1}) N_{i+1,p−1}`. Tangent `T = (x', y')`; outward normal for a counter-clockwise loop is `n = (y', −x') / |T|`.

## 2. D2Q9 lattice Boltzmann (report ch. 4)

**(2.1)** Velocities and weights:

```
c = (0,0) (1,0) (0,1) (−1,0) (0,−1) (1,1) (−1,1) (−1,−1) (1,−1)
w = 4/9   1/9 ×4                     1/36 ×4
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
```

**(2.2)** Moments: `rho = Σ f_i`, `rho u = Σ c_i f_i`.

**(2.3)** Equilibrium: `feq_i = w_i rho [1 + 3 (c_i·u) + 4.5 (c_i·u)² − 1.5 u·u]`.

**(2.4)** BGK collide, then stream:

```
f^col_i(x, n)  = f_i(x, n) − omega (f_i − feq_i)(x, n)
f_i(x + c_i, n+1) = f^col_i(x, n)
```

**(2.5)** TRT collision. Split into symmetric and antisymmetric parts: `f⁺_i = (f_i + f_ī)/2`, `f⁻_i = (f_i − f_ī)/2`, same for `feq`. Then

```
f^col_i = f_i − omega⁺ (f⁺_i − feq⁺_i) − omega⁻ (f⁻_i − feq⁻_i)
omega⁺ = 1/tau,  tau⁻ = 1/2 + Λ/(tau − 1/2),  omega⁻ = 1/tau⁻,  Λ = 3/16
```

`Λ = 3/16` places straight-wall bounce-back exactly halfway and is the standard stable choice. BGK is the special case `omega⁻ = omega⁺`.

**(2.6)** Linear interpolated bounce-back (Bouzidi, Firdaouss, Lallemand 2001). Fluid node `x_A`, direction `c_i` points into the solid, the wall is at fraction `q` along the link. The unknown incoming population after streaming is

```
q < 1/2:  f_ī(x_A, n+1) = 2q f^col_i(x_A) + (1 − 2q) f^col_i(x_A − c_i)
q ≥ 1/2:  f_ī(x_A, n+1) = 1/(2q) f^col_i(x_A) + (2q − 1)/(2q) f^col_ī(x_A)
```

If `x_A − c_i` is solid (thin trailing edge), fall back to halfway bounce-back: `f_ī(x_A, n+1) = f^col_i(x_A)`.

**(2.7)** Derivatives with respect to `q`, needed for the shape gradient:

```
q < 1/2:  ∂f_ī/∂q = 2 f^col_i(x_A) − 2 f^col_i(x_A − c_i)
q ≥ 1/2:  ∂f_ī/∂q = −1/(2q²) f^col_i(x_A) + 1/(2q²) f^col_ī(x_A)
```

Both one-sided limits at `q = 1/2` equal each other only when the populations are equal, so the drag is continuous but not smooth there. Finite-difference checks avoid links with `q` within `Δ` of 1/2.

**(2.8)** Lattice parameters at Mach 0.1: `Ma = U_lb / c_s` with `c_s = 1/√3`, so `U_lb = 0.1/√3 = 0.05774`, `N` cells per reference length, `nu_lb = U_lb N / Re`, `tau = 3 nu_lb + 1/2`. One convective time is `N / U_lb` steps. The table in [forward_solver.md](../03_lbm/forward_solver.md) is computed from these.

**(2.9)** Conversion to non-dimensional fields: `u = u_lb / U_lb`, `p = (rho − 1) c_s² / U_lb²` (pressure relative to the outlet, which is fixed at `rho = 1`).

## 3. Forces (report §2.2)

**(3.1)** Momentum exchange. For each boundary link `l = (x_A, i)`:

```
ΔP_l = c_i [ f^col_i(x_A, n) + f_ī(x_A, n+1) ]
F    = Σ_l ΔP_l
```

**(3.2)** Coefficients: `CD = 2 F_x / (rho_inf U_lb² N)`, `CL = 2 F_y / (rho_inf U_lb² N)`, `rho_inf = 1`.

**(3.3)** Time-averaged coefficients for unsteady flow: mean over an integer number of shedding periods detected from the zero crossings of `CL(t)`.

## 4. Discrete adjoint (report ch. 5)

Write one full step as a map `f^{n+1} = S(f^n; q)` (collision, streaming, IBB, inlet and outlet). Cost `I0 = Σ_n g(f^n; q)` (e.g. drag averaged over the last window).

**(4.1)** Lagrangian `J = I0 + Σ_n ⟨F*^{n+1}, S(f^n; q) − f^{n+1}⟩`.

**(4.2)** Stationarity in `f^n` gives the backward recursion

```
F*^n = (∂S/∂f^n)^T F*^{n+1} + ∂g/∂f^n,    F*^{N} = 0   (terminal condition)
```

**(4.3)** Stationarity in `q` gives the gradient

```
dI0/dq_l = Σ_n [ ∂g/∂q_l (f^n) + ⟨F*^{n+1}, ∂S/∂q_l (f^n)⟩ ]
dI0/dalpha = Σ_l (dI0/dq_l) (dq_l/dalpha)
```

`∂S/∂q_l` is non-zero only in the single population that link `l` overwrites, so the inner product is a scalar per link: `F*_ī(x_A, n+1) · ∂f_ī/∂q` from (2.7). `dq_l/dalpha` comes from autograd through the grid mapper and NURBS.

**(4.4)** Transposed BGK step **(reconstructed; matches report §5.2.4 in form)**. The forward step is `f_j(x + c_j, n+1) = f^col_j(x, n)`, so

```
G_j(x)    = F*_j(x + c_j, n+1)                         (reverse streaming: pull from downstream)
F*_i(x,n) = (1 − omega) G_i(x) + omega Σ_j G_j(x) ∂feq_j/∂f_i (x, n) + ∂g/∂f_i(x, n)
```

**(4.5)** Equilibrium Jacobian. With `∂rho/∂f_i = 1`, `∂u/∂f_i = (c_i − u)/rho`:

```
∂feq_j/∂f_i = w_j [ 1 + 3 c_j·c_i + 9 (c_j·u)(c_j·c_i) − 4.5 (c_j·u)² − 3 u·c_i + 1.5 u·u ]
```

**(4.6)** Adjoint equilibrium in moment form (cheap to vectorize). Define, at each node, from the streamed adjoint `G`:

```
A = Σ_j w_j G_j                B = Σ_j w_j G_j c_j            (vector)
C = Σ_j w_j G_j (c_j·u)²       D = Σ_j w_j G_j (c_j·u) c_j    (vector)
F*eq_i = A (1 + 1.5 u·u) − 4.5 C + 3 c_i·B + 9 c_i·D − 3 (u·c_i) A
```

Then `F*_i = (1 − omega) G_i + omega F*eq_i + ∂g/∂f_i`. This is the report's "analogous to LBM" form with adjoint moments `rho* ~ A`, `j* ~ B`.

**(4.7)** IBB adjoint. The IBB overwrite (2.6) is linear in `f^col`. Its transpose sends the adjoint of each overwritten population `F*_ī(x_A, n+1)` back, with the same coefficients, to `f^col_i(x_A)`, `f^col_i(x_A − c_i)` or `f^col_ī(x_A)`. Implemented with `index_add_` on the link index lists. The population that streaming would have delivered (from inside the solid) receives zero adjoint.

**(4.8)** Inlet and outlet. Both are set to an equilibrium independent of the interior in the report's setup, so their Jacobian rows are zero: `F* = 0` on the inlet and outlet columns (report §5.2.5).

**(4.9)** Cost source term. For drag measured at step `n`: `g = 2 F_x / (U_lb² N)`, so `∂g/∂f^col_i(x_A) = 2 c_{i,x} / (U_lb² N)` on links, plus the contribution through the IBB value `f_ī(n+1)` which chains through (2.6). In code this is obtained with `torch.autograd.grad` of the force function at a single step (cheap, local) rather than by hand.

**(4.10)** Steady adjoint. If `f` is a fixed point `f* = S(f*)`, iterate `F* ← (∂S/∂f)^T F* + ∂g/∂f` until it stops changing. **Time-averaged variant** (report §6.3): replace `f*` by the period-averaged `f̄`. Caveat: at Re 1000 the mean flow is linearly unstable, so this iteration can grow without bound. Run it for a **fixed horizon** (report: 20 convective times), not to convergence, and monitor `‖F*‖`.

**(4.11)** Central difference check (report §5.3): `dI/dalpha_k ≈ [I(alpha + Δ e_k) − I(alpha − Δ e_k)] / (2Δ)`. Error `O(Δ²)`, `2k` forward solves.

## 5. PINN (plan doc §2.3)

**(5.1)** Stream function: `u = ∂psi/∂y`, `v = −∂psi/∂x`, so `∂u/∂x + ∂v/∂y = 0` identically.

**(5.2)** Steady incompressible N-S residuals (non-dimensional, `nu = 1/Re`):

```
r_x = u u_x + v u_y + p_x − nu (u_xx + u_yy) − f_x
r_y = u v_x + v v_y + p_y − nu (v_xx + v_yy) − f_y
```

`f_x = f_y = 0` for steady flow. For time-averaged flow, the averaged equations carry `−∂(u'u')/∂x − ∂(u'v')/∂y` etc., which the network models as `(f_x, f_y)`.

**(5.3)** Third derivatives of `psi` are needed (`u_xx = ∂³psi/∂x²∂y`). Every activation must be at least C³: sine/cosine wavelet and SiLU qualify; ReLU does not.

**(5.4)** Kovasznay flow, used as a test oracle (exact steady N-S solution):

```
lambda = Re/2 − sqrt(Re²/4 + 4 pi²)
psi = y − (1/(2 pi)) e^{lambda x} sin(2 pi y)
u   = 1 − e^{lambda x} cos(2 pi y)
v   = (lambda/(2 pi)) e^{lambda x} sin(2 pi y)
p   = (1 − e^{2 lambda x}) / 2
```

**(5.5)** Surface traction and forces (normal `n` points out of the body into the fluid):

```
t  = −p n + nu (∇u + ∇uᵀ) n
F  = ∮ t ds ≈ Σ_m t_m ds_m
CD = 2 F_x,  CL = 2 F_y        (rho = 1, U = 1, L_ref = 1)
```

For a rotated body (AoA), drag is along the inflow (+x) and lift along +y; the body is rotated, not the flow.

## 6. Guided diffusion (plan doc §2.4, corrected per G11–G12)

**(6.1)** Cosine schedule (Nichol and Dhariwal 2021), `s = 0.008`:

```
gamma_bar_t = cos²( ((t/T + s)/(1 + s)) · pi/2 ) / cos²( (s/(1 + s)) · pi/2 )
```

**(6.2)** Forward noising: `alpha_t = sqrt(gamma_bar_t) alpha_0 + sqrt(1 − gamma_bar_t) eps`, `eps ~ N(0, I)`. Training loss `‖eps − eps_phi(alpha_t, t, c)‖²`.

**(6.3)** Classifier-free guidance: `eps_hat = (1 + w) eps_phi(alpha_t, t, c) − w eps_phi(alpha_t, t, ∅)`.

**(6.4)** Clean estimate: `alpha0_hat = (alpha_t − sqrt(1 − gamma_bar_t) eps_hat) / sqrt(gamma_bar_t)`.

**(6.5)** Physics objective on the clean estimate (after un-standardizing):

```
J(alpha0_hat) = ½ (CD(alpha0_hat) − CD_ref)² + w_L ½ (CL(alpha0_hat) − CL_ref)²
              + mu Σ_k relu(t_min,k − thickness_k(alpha0_hat))²
```

`CD, CL` from the frozen PINN at the requested `(Re, AoA)`.

**(6.6)** Guided DDIM step (eta = 0), from `t` to the previous step `t'`:

```
g      = ∇_{alpha_t} J(alpha0_hat(alpha_t))          (backprop through 6.4 and the denoiser)
g      = g · min(1, g_max / ‖g‖)                      (norm clamp)
alpha0_hat ← alpha0_hat − s_guide · g
alpha_t' = sqrt(gamma_bar_t') alpha0_hat + sqrt(1 − gamma_bar_t') eps_hat
```

Applying the correction to the clean estimate keeps the noise level of `alpha_t'` consistent with the schedule.

## 7. Constrained optimization (report ch. 2)

**(7.1)** Problem: `min CD(alpha)` s.t. `CL(alpha) = CL_ref` (TC2, TC3), `thickness_k(alpha) ≥ t_min,k`.

**(7.2)** Augmented Lagrangian:

```
L(alpha) = CD + lambda (CL − CL_ref) + (rho_al/2)(CL − CL_ref)² + (mu/2) Σ_k relu(t_min,k − thickness_k)²
```

**(7.3)** Updates per outer cycle: `alpha ← alpha − eta ∇L` (backtracking on `eta` so `L` decreases), then `lambda ← lambda + rho_al (CL − CL_ref)`, `mu ← min(mu · 2, mu_max)` while thickness is violated. The gradient of `L` needs only one adjoint solve for the combined scalar, because the adjoint is linear in the cost.

**(7.4)** KKT residual for stopping: `‖∇CD + lambda ∇CL + μ-terms‖` projected onto the free directions, plus `|CL − CL_ref|` and max thickness violation.
