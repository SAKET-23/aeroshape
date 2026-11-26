# Equations

The maths this code implements, from the shape vector to drag, through all three solver tiers. Section numbers match the module layout in `src/aeroshape/`. Where the implementation departs from the textbook form, the decision ID (G-number, see [`Plan/raw/gaps_and_decisions.md`](../Plan/raw/gaps_and_decisions.md)) says why.

**Units.** Lattice units inside the LBM ($\Delta x = \Delta t = 1$, $c_s^2 = 1/3$). Everything else is non-dimensional: reference length (chord or diameter) $= 1$, $U_\infty = 1$, $\rho = 1$, $\nu = 1/\mathrm{Re}$.

| Symbol | Meaning |
|---|---|
| $\alpha \in \mathbb{R}^D$ | shape vector: y-values of the free NURBS control points ($D = 8$ symmetric, $16$ cambered) |
| $f_i(\mathbf{x}, n)$ | D2Q9 population, direction $i = 0..8$; $\bar i$ is the opposite direction; $f^{\mathrm{col}}$ is post-collision |
| $q$ | fraction of a lattice link that lies in the fluid, $q \in (0, 1]$ |
| $F^*_i$ | adjoint population |
| $\psi, p$ | PINN stream function and pressure |
| $\bar\gamma_t$ | diffusion noise schedule |

---

## 1. Geometry: NURBS airfoil (`geometry/nurbs.py`)

Cox–de Boor B-spline basis on a clamped uniform knot vector, degree $p = 3$:

$$
N_{i,0}(s) = \begin{cases} 1 & t_i \le s < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
\qquad
N_{i,p}(s) = \frac{s - t_i}{t_{i+p} - t_i} N_{i,p-1}(s) + \frac{t_{i+p+1} - s}{t_{i+p+1} - t_{i+1}} N_{i+1,p-1}(s)
$$

Rational form with weights $W_i$:

$$
R_i(s) = \frac{N_{i,p}(s)\, W_i}{\sum_j N_{j,p}(s)\, W_j},
\qquad
\mathbf{P}(s) = \sum_i R_i(s)\, \mathbf{P}_i
$$

The control-point x-positions are fixed (cosine-spaced on $[0, 1]$) and only their y-values move. With the sample points fixed, $R \in \mathbb{R}^{M \times k}$ is a constant matrix, so the surface is linear in the design vector:

$$
y(s) = R\,\alpha, \qquad \frac{\partial y}{\partial \alpha} = R
$$

Leading and trailing edges are pinned by a fixed mask, not by in-place writes. The outward normal of a counter-clockwise loop is $\mathbf{n} = (y', -x') / \lVert (x', y') \rVert$. It uses the analytic derivative basis $N'_{i,p}$, not finite differences.

## 2. Lattice Boltzmann, D2Q9 (`lbm/d2q9_solver.py`)

**Moments and equilibrium.**

$$
\rho = \sum_i f_i, \qquad \rho\,\mathbf{u} = \sum_i \mathbf{c}_i f_i,
\qquad
f_i^{\mathrm{eq}} = w_i \rho \left[ 1 + 3\,\mathbf{c}_i\!\cdot\!\mathbf{u} + \tfrac{9}{2} (\mathbf{c}_i\!\cdot\!\mathbf{u})^2 - \tfrac{3}{2}\,\mathbf{u}\!\cdot\!\mathbf{u} \right]
$$

with $w_0 = 4/9$, $w_{1..4} = 1/9$ and $w_{5..8} = 1/36$.

**BGK collision and streaming.**

$$
f_i^{\mathrm{col}}(\mathbf{x}, n) = f_i - \omega \left( f_i - f_i^{\mathrm{eq}} \right),
\qquad
f_i(\mathbf{x} + \mathbf{c}_i, n+1) = f_i^{\mathrm{col}}(\mathbf{x}, n),
\qquad
\nu = \tfrac{1}{3}\left(\tfrac{1}{\omega} - \tfrac{1}{2}\right)
$$

**TRT collision** (Re 1000). Split into symmetric and antisymmetric parts, $f_i^{\pm} = \tfrac12 (f_i \pm f_{\bar i})$:

$$
f_i^{\mathrm{col}} = f_i - \omega^{+} \left( f_i^{+} - f_i^{\mathrm{eq}+} \right) - \omega^{-} \left( f_i^{-} - f_i^{\mathrm{eq}-} \right),
\qquad
\Lambda = \left(\tfrac{1}{\omega^{+}} - \tfrac12\right)\left(\tfrac{1}{\omega^{-}} - \tfrac12\right)
$$

The textbook $\Lambda = 3/16$ is unstable at $\tau \approx 0.504$. The code uses $\Lambda = 0.001$ (**G28**). At Re 1000 TRT and BGK give the same drag on NACA0012 (0.1231 vs 0.1232), so BGK is also acceptable.

**Lattice parameters at Mach 0.1.** With $N$ cells per reference length:

$$
U_{\mathrm{lb}} = \frac{0.1}{\sqrt 3},
\qquad
\nu_{\mathrm{lb}} = \frac{U_{\mathrm{lb}}\, N}{\mathrm{Re}},
\qquad
\tau = 3\nu_{\mathrm{lb}} + \tfrac12,
\qquad
\text{one convective time} = N / U_{\mathrm{lb}} \text{ steps}
$$

**Curved wall: Bouzidi interpolated bounce-back** (`geometry/ibb_grid_mapper.py`). Fluid node $\mathbf{x}_A$, link $\mathbf{c}_i$ into the solid, wall at fraction $q$ along it:

$$
f_{\bar i}(\mathbf{x}_A, n+1) =
\begin{cases}
2q\, f_i^{\mathrm{col}}(\mathbf{x}_A) + (1 - 2q)\, f_i^{\mathrm{col}}(\mathbf{x}_A - \mathbf{c}_i) & q < \tfrac12 \\[4pt]
\dfrac{1}{2q}\, f_i^{\mathrm{col}}(\mathbf{x}_A) + \dfrac{2q - 1}{2q}\, f_{\bar i}^{\mathrm{col}}(\mathbf{x}_A) & q \ge \tfrac12
\end{cases}
$$

$q$ comes from a ray–segment intersection of the link with the NURBS polyline. That makes $q$ a smooth function of $\alpha$, and this is how the gradient "sees" the curved boundary. Thin trailing edges, where $\mathbf{x}_A - \mathbf{c}_i$ is solid, fall back to halfway bounce-back.

**Export to non-dimensional fields.** $\mathbf{u} = \mathbf{u}_{\mathrm{lb}} / U_{\mathrm{lb}}$ and $p = (\rho - 1)\, c_s^2 / U_{\mathrm{lb}}^2$.

## 3. Forces (`lbm/aero_forces.py`)

Momentum exchange over the boundary links $l = (\mathbf{x}_A, i)$:

$$
\mathbf{F} = \sum_{l} \mathbf{c}_i \left[ f_i^{\mathrm{col}}(\mathbf{x}_A, n) + f_{\bar i}(\mathbf{x}_A, n+1) \right],
\qquad
C_D = \frac{2 F_x}{\rho_\infty U_{\mathrm{lb}}^2 N},
\quad
C_L = \frac{2 F_y}{\rho_\infty U_{\mathrm{lb}}^2 N}
$$

**Unsteady flow** (Re 1000). The shedding period is detected from upward zero crossings of $C_L(t)$. Forces and $f$ are averaged over 4 periods. If no consistent period appears, the mean is taken over a 20-convective-time window.

## 4. Discrete adjoint (`lbm/adjoint_solver.py`)

One full time step is a map $f^{n+1} = S(f^n; q)$ (collide, stream, IBB, inlet/outlet), with cost $I_0 = \sum_n g(f^n; q)$. The Lagrangian

$$
\mathcal{J} = I_0 + \sum_n \left\langle F^{*,n+1},\; S(f^n; q) - f^{n+1} \right\rangle
$$

is stationary in $f^n$ when the adjoint runs backwards:

$$
F^{*,n} = \left( \frac{\partial S}{\partial f^n} \right)^{\!\top} F^{*,n+1} + \frac{\partial g}{\partial f^n},
\qquad F^{*,N} = 0
$$

and stationarity in $q$ gives the shape gradient. $\partial S / \partial q_l$ touches only the one population link $l$ overwrites:

$$
\frac{dI_0}{dq_l} = \sum_n \left[ \frac{\partial g}{\partial q_l} + F^{*,n+1}_{\bar i}(\mathbf{x}_A)\, \frac{\partial f_{\bar i}}{\partial q_l} \right],
\qquad
\frac{dI_0}{d\alpha} = \sum_l \frac{dI_0}{dq_l}\, \frac{dq_l}{d\alpha}
$$

$dq_l/d\alpha$ is one `autograd` call through the grid mapper and the NURBS.

**Transposed BGK step.** Reverse streaming pulls from downstream, $G_j(\mathbf{x}) = F^*_j(\mathbf{x} + \mathbf{c}_j, n+1)$. Then

$$
F^*_i = (1 - \omega)\, G_i + \omega \sum_j G_j \frac{\partial f_j^{\mathrm{eq}}}{\partial f_i} + \frac{\partial g}{\partial f_i},
\qquad
\frac{\partial f_j^{\mathrm{eq}}}{\partial f_i} = w_j \left[ 1 + 3\,\mathbf{c}_j\!\cdot\!\mathbf{c}_i + 9 (\mathbf{c}_j\!\cdot\!\mathbf{u})(\mathbf{c}_j\!\cdot\!\mathbf{c}_i) - \tfrac92 (\mathbf{c}_j\!\cdot\!\mathbf{u})^2 - 3\,\mathbf{u}\!\cdot\!\mathbf{c}_i + \tfrac32\,\mathbf{u}\!\cdot\!\mathbf{u} \right]
$$

The IBB map is linear in $f^{\mathrm{col}}$, so its transpose scatters each overwritten adjoint population back with the same coefficients. Inlet and outlet are fixed equilibria, so $F^* = 0$ there.

**Steady and time-averaged modes.**
- **Steady (Re 20):** freeze the converged $f^\star$ and solve the fixed point $F^* = (\partial S/\partial f)^\top F^* + \partial g/\partial f$ by restarted GMRES.
- **Time-averaged (Re 1000):** linearise around the period-mean $\bar f$ and iterate for a fixed horizon of 20 convective times. The mean flow is linearly unstable, so this does not converge.

**Verification.** On CPU in float64:
- the hand-derived step matches `autograd` VJP to $10^{-10}$, and the dot-product test passes;
- against central differences $\left[ I(\alpha + \Delta e_k) - I(\alpha - \Delta e_k) \right] / 2\Delta$ with 41 control points, the relative $L_2$ error is $3.2 \times 10^{-4}$ (the gate is $2 \times 10^{-3}$).

## 5. Physics-informed surrogates (`pinn/`)

**Divergence-free by construction.** The network outputs a stream function $\psi$ and pressure $p$:

$$
u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}
\quad\Longrightarrow\quad
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \equiv 0
$$

The momentum residuals need third derivatives of $\psi$, so every activation is at least $C^3$ (SiLU, sine/cosine wavelets):

$$
r_x = u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy}) - f_x,
\qquad
r_y = u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy}) - f_y
$$

- **Steady flow (v1, Re 20):** $f_x = f_y = 0$.
- **Time-averaged flow (v2, Re 1000):** the mean equations carry Reynolds-stress divergence terms $-\partial_j \overline{u_i' u_j'}$. The network learns these as an extra forcing head $(f_x, f_y)$, penalised so it stays near zero where the flow is steady.

**Transformer architecture.** Each flow point $(x, y, d_{\mathrm{wall}})$ is a query. It cross-attends to $k$ control-point tokens plus one condition token $[\log \mathrm{Re}, \sin \mathrm{AoA}, \cos \mathrm{AoA}]$. Points never attend to each other, so one `autograd.grad` call gives exact per-point derivatives. The baseline is an MLP on random Fourier features of $(x, y)$.

**Training loss** (`pinn/navier_stokes_loss.py`):

$$
\begin{aligned}
\mathcal{L} = {} & w_{\mathrm{pde}} \,\overline{r_x^2 + r_y^2}
+ w_{\mathrm{wall}} \,\overline{u^2 + v^2}\big|_{\mathrm{wall}}
+ w_{\mathrm{edge}} \,\overline{\lVert \mathbf{u} - \mathbf{u}_{\mathrm{LBM}} \rVert^2}\big|_{\mathrm{edge}}
+ w_{\mathrm{data}} \,\overline{\lVert \mathbf{u} - \mathbf{u}_{\mathrm{LBM}} \rVert^2 + c_p (p - p_{\mathrm{LBM}})^2} \\
& + w_f \,\overline{f_x^2 + f_y^2}
+ w_{cd} \left[ (C_D - C_D^{\mathrm{LBM}})^2 + (C_L - C_L^{\mathrm{LBM}})^2 \right]
+ w_{\mathrm{grad}} \left[ 1 - \cos\!\left( \nabla_\alpha C_D,\; \nabla_\alpha C_D^{\mathrm{adj}} \right) \right]
\end{aligned}
$$

- **Data and boundary terms:** only these are active for the first 500 steps, then the PDE weight ramps in over 1000 steps.
- **Force supervision ($w_{cd}$):** this term is **G29**. Field-only training flattened the wall pressure and gave drag 80–97% too low.
- **Gradient supervision ($w_{\mathrm{grad}}$):** this is Sobolev-style training (**G30**) against cached adjoint gradients, used on the steady rows.

**Surface stress and forces** (`pinn/stress_integrator.py`), with $\mathbf{n}$ pointing from the body into the fluid:

$$
\mathbf{t} = -p\,\mathbf{n} + \nu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\top} \right) \mathbf{n},
\qquad
\mathbf{F} = \oint \mathbf{t}\, ds \approx \sum_m \mathbf{t}_m\, \Delta s_m,
\qquad
C_D = 2 F_x, \quad C_L = 2 F_y
$$

$\nabla_\alpha C_D$ flows through both the NURBS surface and the network's shape input in one backward pass. That is what makes the surrogate usable as a gradient evaluator.

## 6. Physics-guided diffusion (`diffusion/`)

**Cosine schedule and forward noising** on the standardised shape $z = (\alpha - \mu)/\sigma$:

$$
\bar\gamma_t = \frac{\cos^2\!\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)}{\cos^2\!\left( \frac{s}{1 + s} \cdot \frac{\pi}{2} \right)},
\qquad
z_t = \sqrt{\bar\gamma_t}\, z_0 + \sqrt{1 - \bar\gamma_t}\, \varepsilon,
\qquad
\mathcal{L}_{\mathrm{diff}} = \left\lVert \varepsilon - \varepsilon_\phi(z_t, t, c) \right\rVert^2
$$

The condition is $c = [C_L, C_D, \log \mathrm{Re}, \mathrm{AoA}]$. It is dropped 10% of the time, which enables classifier-free guidance:

$$
\hat\varepsilon = (1 + w)\, \varepsilon_\phi(z_t, t, c) - w\, \varepsilon_\phi(z_t, t, \varnothing),
\qquad
\hat z_0 = \frac{z_t - \sqrt{1 - \bar\gamma_t}\, \hat\varepsilon}{\sqrt{\bar\gamma_t}}
$$

**Physics objective** on the clean estimate. $C_D$ and $C_L$ come from the frozen PINN, and $t_k$ is the local thickness:

$$
J(\hat z_0) = \tfrac12 \left( C_D - C_D^{\mathrm{ref}} \right)^2 + \tfrac{w_L}{2} \left( C_L - C_L^{\mathrm{ref}} \right)^2 + \mu \sum_k \mathrm{relu}\!\left( t_{\min,k} - t_k \right)^2
$$

**Guided DDIM step** from $t$ to $t'$. The gradient is taken with respect to $\hat z_0$ as a detached leaf (**G36**). An earlier version backpropagated through the denoiser to $z_t$, and that Jacobian is not positive definite, so guidance often *raised* $J$.

$$
g = \nabla_{\hat z_0} J,
\qquad
g \leftarrow g \cdot \min\!\left(1, \frac{g_{\max}}{\lVert g \rVert}\right),
\qquad
\hat z_0 \leftarrow \hat z_0 - s_{\mathrm{guide}}\, g,
\qquad
z_{t'} = \sqrt{\bar\gamma_{t'}}\, \hat z_0 + \sqrt{1 - \bar\gamma_{t'}}\, \hat\varepsilon
$$

Correcting the clean estimate, not $z_t$, keeps the noise level of $z_{t'}$ on schedule.

## 7. Constrained optimisation (`optimization/kkt_optimizer.py`)

$$
\min_\alpha\; C_D(\alpha)
\quad \text{s.t.} \quad
C_L(\alpha) = C_L^{\mathrm{ref}}\ (\text{cases 2–3}),
\qquad
t_k(\alpha) \ge t_{\min,k}
$$

Augmented Lagrangian, one gradient per cycle from whichever evaluator is plugged in (adjoint LBM, MLP-PINN or Transformer-PINN):

$$
\mathcal{L}_A = C_D + \lambda \left( C_L - C_L^{\mathrm{ref}} \right) + \frac{\rho_{al}}{2} \left( C_L - C_L^{\mathrm{ref}} \right)^2 + \frac{\mu}{2} \sum_k \mathrm{relu}\!\left( t_{\min,k} - t_k \right)^2
$$

$$
\alpha \leftarrow \Pi_{\mathcal{B}}\!\left( \alpha - \eta\, \mathrm{clip}(\nabla \mathcal{L}_A) \right),
\qquad
\lambda \leftarrow \lambda + \rho_{al} \left( C_L - C_L^{\mathrm{ref}} \right),
\qquad
\mu \leftarrow \min(2\mu,\ \mu_{\max}) \ \text{while thickness is violated}
$$

- **Step size:** $\eta$ backtracks until $\mathcal{L}_A$ decreases, and the per-cycle move is capped (`g_cap`).
- **Adjoint cost:** the adjoint is linear in the cost, so one adjoint solve on the combined scalar gives $\nabla \mathcal{L}_A$.
- **Training-box projection $\Pi_{\mathcal{B}}$ (G37):** for the PINN evaluators, $\Pi_{\mathcal{B}}$ clamps $\alpha$ to the per-coordinate min/max of the surrogate's training shapes. Without it, the optimizer walks out of the surrogate's validity domain and exploits its error. In Case 1 the unconstrained MLP reported $C_D = 0.07$ while the LBM gave $1.84$.
- **Scoring:** every method's final shape is re-scored by one LBM run (**G16**), so the comparison never trusts a surrogate's own number.
