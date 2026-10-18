# Truncation: `alpha_max` and `tau_max` (EFLM)

One truncation knob per noise schedule. `noise_schedules.get_noise` enforces this:

| noise schedule | truncation knob | the other knob |
|---|---|---|
| `autonomous`, `autonomous-adaptive` | `noise.tau_max` | `noise.alpha_max` must be null |
| any other (e.g. `log-linear-adaptive` + trunc) | `noise.alpha_max` | `noise.tau_max` must be unset / null |
| untruncated | both null | |

Values come from `alpha_star_euclidean(V, embed_norm=R)` in `noise_schedules.py`, with delta = 0.1 and unit noise.
R is the pinned embedding norm (`algo.rho_min = algo.rho_max`), so this applies to the rescaled EFLM only.

- `z = sqrt(2 ln(2 (V - 1) / delta)) / R`
- `alpha_max = alpha* = z / (1 + z)`: the signal level at the clean end (MDLM convention, `invert_time_convention=false`)
- `tau_max = tau* = log(1 + z) = -log(1 - alpha*)`: the same decode point on the autonomous clock (`auto_clock=True`)

| setup | V | R | `alpha_max` | `tau_max` | |
|---|---|---|---|---|---|
| sudoku | 12 | 1 | 0.7666 | 1.4550 | script default |
| sudoku | 12 | 5 | 0.3965 | 0.5049 | |
| sudoku | 12 | 8 | 0.2911 | 0.3440 | |
| sudoku | 12 | 16 | 0.1703 | 0.1867 | |
| TinyStories | 50257 | 0.1 | 0.9813 | 3.9811 | |
| TinyStories | 50257 | 0.5 | 0.9132 | 2.4436 | |
| TinyStories | 50257 | 1 | 0.8402 | 1.8338 | script default |
| TinyStories | 50257 | 8 | 0.3966 | 0.5051 | |

The scripts `scripts/{train,sample}/{sudoku,tinystories}/eflm_rescale_*.sh` default to `RHO=1.0` and set the matching
values literally: sudoku `ALPHA_MAX=0.767` / `TAU_MAX=1.455`, TinyStories `ALPHA_MAX=0.840` / `TAU_MAX=1.834`.
For any other R, pass the values from the table, and pass the same values to the train and the sample script.
