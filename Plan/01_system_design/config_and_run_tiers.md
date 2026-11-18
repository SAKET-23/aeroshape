# Config and run tiers

## Loading (`core/config.py`)

```python
def load_config(case: str | None, tier: str, overrides: list[str]) -> Config
```

Merge order (later wins): `configs/base.yaml` → `configs/tiers/<tier>.yaml` → case file → `--set key.sub=value` overrides. `Config` is a frozen nested dataclass (or `SimpleNamespace` built from the dict) with attribute access; unknown keys raise. The resolved config is written to `runs/<name>/config.yaml` at the start of every run.

Every script accepts: `--tier {smoke,dev,full}`, `--case <yaml>`, `--device {auto,cpu,mps,cuda}`, `--seed`, `--out runs/<name>`, `--set k=v` (repeatable), `--compile` (CUDA only), `--resume`.

## `configs/base.yaml` (full key list)

```yaml
seed: 0
device: auto

geometry:
  k: 8                    # control points per surface (G2)
  degree: 3
  mode: symmetric         # symmetric | cambered
  M: 160                  # surface samples per side
  x_spacing: cosine
  weights: ones

lbm:
  mach: 0.1
  domain_d: [20, 10]      # domain size in reference lengths (x, y)
  body_offset_d: 5        # leading edge (or centre for cylinder) distance from inlet
  cells_per_d: 20
  collision: bgk          # bgk | trt
  trt_magic: 0.1875
  sponge: {enabled: false, width_d: 1.5, nu_factor: 10}
  steady_tol: 1.0e-6
  max_convective_times: 60
  average_periods: 4
  record_every: 50

adjoint:
  mode: steady            # steady | time_averaged
  horizon_convective_times: 20
  tol: 1.0e-8
  impl: hand              # hand | vjp

pinn:
  model: transformer      # transformer | mlp
  d_model: 128
  heads: 4
  self_attn_layers: 2
  cross_attn_layers: 2
  mlp_hidden: 128
  mlp_layers: 6
  fourier_features: 64
  fourier_scale: 2.0
  version: v1             # v1 steady | v2 with forcing
  domain_box: [-1.5, 4.0, -1.5, 1.5]   # x_min, x_max, y_min, y_max (non-dim); equals dataset.window_d

loss:
  w_pde: 1.0
  w_wall: 10.0
  w_edge: 5.0             # window-edge anchor values (replaces analytic inlet/outlet, see ns_loss_and_sampling.md)
  w_pin: 1.0              # pressure gauge: mean p on the outlet edge matches anchors
  w_data: 10.0
  w_force: 1.0e-2
  c_p: 0.25               # pressure weight inside data term
  balance_every: 0        # 0 = off; else gradient-norm balancing period

sampling:
  n_collocation: 4000
  wall_fraction: 0.5
  n_surface: 400
  n_anchor: 1000
  n_edge: 400

train:
  shapes_per_step: 8
  adam_steps: 5000
  lr: 1.0e-3
  lr_min: 1.0e-5
  weight_decay: 1.0e-4
  grad_clip: 1.0
  lbfgs_steps: 500
  warmup_data_only: 500
  pde_ramp: 1000
  eval_every: 250

dataset:
  n_shapes: 30
  n_holdout: 5
  conditions: [{re: 20, aoa: 0}]
  points_per_shape: 20000
  window_d: [-1.5, 4.0, -1.5, 1.5]
  edge_points: 2000       # extra points saved on the window edges

diffusion:
  n_shapes: 2000
  T: 1000
  layers: 4
  d_model: 128
  heads: 4
  cond_drop: 0.1
  steps: 20000
  batch: 256
  lr: 2.0e-4
  ema: 0.999
  sample_steps: 50
  cfg_scale: 2.0
  guide_scale: 1.0
  guide_clip: 1.0
  w_lift: 1.0
  mu_thickness: 40.0

optimizer:
  evaluator: adjoint_lbm  # adjoint_lbm | mlp_pinn | transformer_pinn
  max_cycles: 20
  eta: 0.05
  backtrack: 0.5
  rho_al: 10.0
  mu0: 10.0
  mu_max: 1.0e4
  kkt_tol: 1.0e-4
  cl_tol: 1.0e-2
```

## Tier overrides

| Key | smoke | dev | full |
|---|---|---|---|
| `lbm.cells_per_d` (cylinder / Re ≤ 40) | 10 | 20 | 40 |
| `lbm.cells_per_d` (Re 1000) | 24 | 48 | 100 |
| `lbm.max_convective_times` | 5 | 60 | 100 |
| `adjoint.horizon_convective_times` | 2 | 20 | 20 |
| `pinn.d_model` | 64 | 128 | 128 |
| `sampling.n_collocation` | 500 | 4000 | 8000 |
| `train.shapes_per_step` | 2 | 8 | 16 |
| `train.adam_steps` / `lbfgs_steps` | 100 / 5 | 5000 / 500 | 20000 / 1000 |
| `dataset.n_shapes` | 3 | 30 | 30 |
| `dataset.points_per_shape` | 2000 | 20000 | 50000 |
| `diffusion.n_shapes` / `steps` | 200 / 300 | 2000 / 20000 | 5000 / 50000 |
| `optimizer.max_cycles` | 2 | 20 | 30 |

Tier purpose: **smoke** proves every stage connects (minutes, CPU or MPS). **dev** gives real numbers at reduced resolution on the Mac mini (hours). **full** is report quality on cloud CUDA. Cells per D for Re 1000 apply per chord.

## Case files

| File | Sets |
|---|---|
| `validate_cylinder.yaml` | `geometry: circle_exact` (analytic circle, not NURBS), Re list `[20, 40, 100]`, `collision: bgk` for 20/40, `trt` for 100 |
| `case1_circle_to_naca_re20.yaml` | Re 20, AoA 0, symmetric, k 8, start `circle`, `t_min = naca0012`, no lift constraint, steady adjoint |
| `case1_re5.yaml` | as case 1 with Re 5 (report setting) |
| `case2_circle_to_rae_re1000.yaml` | Re 1000, AoA 0, cambered, k 16, start `circle`, `t_min = rae2822`, `cl_ref = rae2822@lbm`, time-averaged adjoint, TRT, sponge on |
| `case3_naca_to_rae_aoa10.yaml` | as case 2 with AoA 10°, start `naca2412` (the "arbitrary NACA") |

`cl_ref: rae2822@lbm` means: computed once by an LBM run on the fitted RAE2822 at the case's Re and AoA, and cached in `runs/refs/`.
