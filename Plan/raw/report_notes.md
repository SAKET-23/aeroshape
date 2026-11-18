# Report notes

Extracted from `docs/SSL (1).pdf`, "Aerodynamic Shape Optimization Using Lattice Boltzmann Method" (63 pages, endsem project report, 1 June – 30 November). Page numbers are the report's own. Many equations in the PDF are images and did not survive text extraction; where that happened the equation is reconstructed in [derivations.md](derivations.md) and marked as such.

## Chapter 2: problem formulation (pp. 9–12)

- General problem: minimize `f(x)` subject to `h_i(x) = 0`, `g_j(x) ≤ 0`. Optimality by KKT conditions; second-order sufficiency via the Hessian of the Lagrangian projected on the null space of active constraint gradients.
- Shape problem: minimize `J(U, alpha)` subject to the PDE constraint (LBM), a geometric constraint and an aerodynamic constraint.
- `CD = 2 F_x / (rho_inf U_inf² A)`, `CL = 2 F_y / (rho_inf U_inf² A)`. In 2D, `A` is the reference length (diameter or chord) per unit span.

## Chapter 3: shape parameterization (pp. 13–18)

- Covers Bézier, B-spline, NURBS. NURBS chosen because it represents conics (the circle start shape) exactly, with weights.
- Properties relied on: C2 continuity for cubic, convex hull, variation diminishing, all-ones weights reduce to B-spline, knot vector non-decreasing.
- Fig 3.2 shows NURBS with different control-point counts and weights. No numeric settings given.

## Chapter 4: LBM (pp. 19–35)

- D2Q9, BGK collision, collide-then-stream.
- Boundary conditions discussed: periodic, full-way and halfway bounce-back, linear interpolated bounce-back (IBB) with `q = d / |c_i Δt|`, `q = 0.5` equals simple bounce-back.
- **Cylinder validation domain**: `−5D ≤ x ≤ 15D`, `−5D ≤ y ≤ 5D` (20D × 10D, body 5D from inlet, centred vertically).
- **Boundaries**: inlet Dirichlet velocity `Ux = 1, Uy = 0`; outlet Neumann velocity (`∂u/∂x = ∂v/∂x = 0`) and Dirichlet pressure; top and bottom periodic; IBB on the body.
- **Viscosities quoted**: Re 40 → `nu = 0.025`; Re 20 → `nu = 0.05` (with `U = 1`, `D = 1`, these are `nu = U D / Re`).

### Table 1 (p. 28): cylinder drag

| Re | Source | Method | CD |
|---|---|---|---|
| 20 | Tritton | Experiment | 2.2 |
| 20 | Dennis and Chang | N-S | 2.045 |
| 20 | Report | LBM | 2.04 |
| 40 | Tritton | Experiment | 1.65 |
| 40 | Nieuwstadt and Keller | N-S | 1.550 |
| 40 | Report | LBM | 1.65 |
| 100 | Hejranfar and Ezzatneshan | CFD-LBM | 1.336 |
| 100 | Report | LBM | 1.365 |

- Re 100: shedding starts between 30 and 35 convective times; time-averaged CL ≈ 0.
- Re 20 and 40: steady, CL = 0, constant CD.

### NURBS–LBM coupling (pp. 31–35)

- Solid/fluid classification by **ray casting**; boundary (marker) nodes and wall distances computed for IBB.
- Cylinder at Re 1000 from NURBS: vortex shedding shown at t = 10, 15, 50, 99.
- NACA0012 at Re 1000, AoA 0–50°: unsteady pattern from about 8° AoA; **CD ≈ 0.12 at AoA 0**; mean CD rises with AoA; stall is gradual at this Re.

## Chapter 5: adjoint (pp. 36–44)

- Lagrangian `J = I0 + Σ F*·R`; stationarity in `F*` gives the LBM, in `f` gives the adjoint LBM (ALBE), in `alpha` gives the gradient.
- ALBE runs backward in time; it has an adjoint equilibrium `F*,eq` built from adjoint moments `rho*`, `j*` (analogous to LBM moments).
- **Adjoint boundary conditions**: velocity and pressure both fixed at inlet and outlet (valid when the body is far from both). This gives `F* = 0` at inlet and outlet. Terminal condition `F*(x, t_f) = 0`.
- **Gradient**: `dI0/dalpha = ∂I0/∂alpha + F*·∂R/∂alpha`. The first term is zero (cost depends on `alpha` only through the flow). The second is non-zero only on boundary links, through `q`. **Standard bounce-back gives zero gradient**; IBB is required.
- **Gradient check (Fig 5.1)**: 41 control points on a circle, y-coordinates; adjoint vs. central finite differences `(I(alpha+Δ) − I(alpha−Δ)) / 2Δ`; total error about **2 × 10⁻³**.

## Chapter 6: test cases (pp. 45–59)

| | Test Case 1 | Test Case 2 | Test Case 3 |
|---|---|---|---|
| Start shape | circle | circle | arbitrary NACA |
| Result | NACA0012-like | RAE2822-like | RAE2822-like |
| Re | **5** | 1000 | 1000 |
| Mach | 0.1 | 0.1 | 0.1 |
| AoA | 0 | 0 | prescribed (plan doc: 10°) |
| Variables | 41 control points, y only | y only | 41 control points, y only |
| Objective | min CD | min CD | min CD |
| Constraints | t/c ≥ baseline on upper surface; symmetry | t/c, CL | t/c, target CL, AoA |
| Flow treatment | steady | time-averaged-flow steady adjoint | same as 2 |
| Cycles to converge | 17 | 11 | 15 |

- Test Case 1: one cycle = **40 LB convective times + 20 adjoint convective times**. Drag falls through both shape change and reduced surface area. `∂I0/∂f` involves Hermite polynomials (moment projection of the force).
- Test Case 2: Re 1000 lies between the first and second wake bifurcations: periodic 2D shedding, still laminar. Cost defined on one shedding cycle of `N_T` steps; primal averaged over that cycle; steady adjoint solved on the averaged primal; gradient uses only averaged primal + converged steady adjoint.
- Test Case 3: drag falls and lift rises each cycle. The report notes the optimum's uniqueness comes from the constraints; different constraints give different shapes.
- Chord length held fixed throughout all cases.

## What the report does not give (filled in [gaps_and_decisions.md](gaps_and_decisions.md))

- Grid resolution and lattice viscosity for any case.
- NURBS degree, knot vector, weights, or control-point x-distribution.
- Exact `CL_ref`, `t/c` reference values, and AoA for Test Case 3.
- The optimizer step size, relaxation factor and how constraints were enforced numerically.
- Wall-clock times.
