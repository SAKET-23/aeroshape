# `visualization/plots.py`, `visualization/animate.py` — figures and animations

**Work package:** WP 19 (built alongside WP 6, 8, 15, 18). Matplotlib only; PyVista optional (`viz3d` extra) and not required for any figure below.

## Conventions

- Every plot function takes plain tensors/arrays and an output path; no solver calls inside plotting code.
- Saves PNG at 150 dpi plus the data used as `.npz` next to it, so figures can be restyled without rerunning.
- One colormap per quantity: velocity magnitude `viridis`, vorticity `RdBu_r` (symmetric limits), pressure `coolwarm`, attention/sensitivity `magma`.
- Body drawn as a filled grey polygon from the `Boundary`.

## Figure list

| Figure | Function | Produced by | Report analogue |
|---|---|---|---|
| CD/CL time history | `plot_force_history` | WP 6 | Figs 4.10–4.12 |
| Vorticity / velocity snapshots | `plot_field(kind=…)` | WP 6 | Figs 4.8, 4.15 |
| Table 1 comparison | `write_table1_md` | WP 6 | Table 1 |
| Grid convergence | `plot_convergence` | WP 6 | — |
| Adjoint vs FD gradient bars | `plot_gradient_check` | WP 8 | Fig 5.1 |
| Adjoint sensitivity magnitude `|F*|` | `plot_field(kind="adjoint")` | WP 8, 18 | Figs 6.3, 6.6 |
| Cross-attention heat map + attention-weighted sensitivity | `plot_attention` | WP 15 | — (new) |
| `|F*|` vs attention side by side | `plot_adjoint_vs_attention` | WP 18 | — (portfolio item 2) |
| PINN vs LBM fields with error | `plot_field_comparison` | WP 15 | — |
| Loss curves | `plot_training_log` | WP 15, 17 | — |
| Diffusion samples and guidance sweep | `plot_samples`, `plot_sweep` | WP 17 | — |
| Shapes per method | `plot_shapes` | WP 18 | Figs 6.4, 6.7, 6.11 |
| Cp along the surface | `plot_cp` | WP 18 | Fig 6.8 |
| CD vs wall-clock | `plot_cost_curves` | WP 18 | — |
| Benchmark bars | `plot_benchmark_bars` | WP 18 | — |

## Animation (`animate.py`)

`morph_gif(history, fields_fn, path, fps=4)`: three panels per frame — (left) shape at cycle `c` over the start shape and the reference airfoil; (centre) velocity or vorticity field at cycle `c` (LBM snapshot saved by the optimizer when the evaluator is LBM; PINN field otherwise); (right) CD vs cycle with the current point highlighted. Written with `imageio`. A denoising-trajectory variant shows the diffusion sampler's `alpha0_hat` per step.

## Tests

Each plot function is called once on tiny synthetic data in `tests/visualization/test_plots.py` and must write a non-empty file (headless `Agg` backend).
