# HFLM Gaussian-Curvature Sudoku Experiment

**Question:** Does the Gaussian curvature `K` of the hyperbolic space matter for H-FLM
Sudoku accuracy, holding everything else (prior, rho_max, init, schedule, sampler) fixed?

## The knob

`algo.gaussian_curvature = K < 0` (new; default `-1.0` = the standard unit hyperboloid =
previous behavior, bit-identical code path).

## Architecture (curvature lives inside the geometry APIs)

`GeoUtils._curvature_scale(K)` validates `K != 0` and returns the model radius
`R = 1/sqrt(|K|)`. Every coordinate-conversion method in `geo_bridge.py` takes
`gaussian_curvature` and is natively curvature-aware: the Lorentz model is the
radius-`R` hyperboloid (`<z,z>_L = 1/K`), the Poincare model the radius-`R` ball
(`z = R tanh(rho/2R) u`), and polar `(rho, u)` is intrinsic (chart-free).
`_geodesic_kernel` is also curvature-native: the *sign* of `K` selects hyperbolic
(sinh) vs spherical (sin) interpolation and the interpolation angle is
`theta = d/R` from the difference form divided by `R^2`. The `geodesic()` wrappers
are pure dispatch — no curvature arithmetic outside the conversions/kernel.

Effect of `K` on H-FLM: clean embeddings and the wrapped-normal prior keep their
intrinsic radials, but the network's Poincare chart and the geodesic path both scale
with `R` — larger `|K|` bows interpolants harder toward the origin (intrinsic midpoint
radial for two points at rho=6, orthogonal directions: 2.60 at K=-0.1 -> 0.88 at
K=-1 -> 0.62 at K=-2).

Both training (`algo.py HFLM`) and sampling (`samplers.py HFLMSampler`) use the same
`K`; a train/sample mismatch would be a geometry bug (cf. the EFLM-port
scale-mismatch lesson).

## Constraint

The float64 Lorentz precision bound is `rho / R <= 20` (`_check_lorentz_rho_bound`).
With `rho_max = 12` and prior `E[rho] ~ 11.3`, `K = -2` is safe (~17) but `K = -4`
would raise (~24). Hence the sweep stops at `K = -2`.

## Sweep (12 cells)

Grid: `gaussian_curvature x difficulty`, 20k steps, otherwise the exact
`scripts/train/sudoku/hflm.sh` recipe (prior_cov=0.25, rho_max=12, log-linear,
tiny-hyperbolic-dit, batch 256), 1 GPU per cell (~2-2.5h train + ~10min eval each):

| K      | note                              | difficulties       |
|--------|-----------------------------------|--------------------|
| -0.25  | flattest (most Euclidean-like)    | easy, medium, hard |
| -0.5   |                                   | easy, medium, hard |
| -1.0   | baseline (reproduces prior HFLM)  | easy, medium, hard |
| -2.0   | most curved (safe under rho bound)| easy, medium, hard |

Easy alone separates poorly (baselines saturate ~80%+); medium/hard added to spread
the curvature effect. Eval: `scripts/sample/sudoku/hflm.sh` defaults — `sudoku_eval`,
180 steps, exact velocity, greedy, `top_k_velocity=1`, same `K` as training.

## Outputs

- per cell: `outputs/hflm_curv_sudoku/d-<difficulty>_k<K>/` (checkpoints + `eval/results.json`)
- logs:     `experiments/hflm_curv_sudoku/logs/<tag>_<jobid>.log`

## Launch

```bash
python experiments/hflm_curv_sudoku/sweep.py [--dry-run]
```

`sweep.py` (simple_slurm, orchestration-only) submits one job per cell
(`hcurv_d-<difficulty>_k<K>`), skipping cells whose `eval/results.json` exists or
whose job name is already queued; resubmits auto-resume from `last.ckpt`.
