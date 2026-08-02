# vmulan_sudoku — learned (MuLAN-style) noise schedule for EFLM, sudoku hard

> **Round 2 (2026-07-26).** Round 1 (the `vmulan_{scope}_*` cells in
> RESULTS.md) underperformed the fixed baseline everywhere. A from-scratch
> external audit (codex gpt-5.6-sol) + a re-read of MuLAN Suppl. D found
> three real implementation bugs, all fixed for round 2
> (`vmulan_prompt_deg{5,9}_*` cells):
>
> 1. **Oracle/drifting context** — the schedule conditioned on a DLM hidden
>    state of the corrupted TRUE solution (train) but on the evolving
>    integration state, re-read every step (sample). This is exactly MuLAN
>    Suppl. D.2's `c = f(x0)` ablation, which the paper itself shows
>    underperforming the unconditioned baseline. Fix: `var_context=prompt` —
>    the context is an encoding of the PROMPT (sudoku clues:
>    inference-available, MuLAN D.1), identical construction at train and
>    sample, computed once and fixed per trajectory, clue-position-aware.
> 2. **Scalar time conditioning** — per-position sigma was mean-collapsed
>    before the DLM's timestep embedder (with 91 always-clean prompt
>    positions polluting the mean), so the denoiser could not know the
>    learned per-position noise levels. Fix: per-token adaLN conditioning
>    (`sigma [B, L]` end to end), prompt positions pinned at sigma = 0, and
>    the DLM additionally conditioned on the schedule context z (zero-init
>    `W_z`, MuLAN's reverse-on-context).
> 3. **Dropout inside the GammaNet** — the loss weight divides two GammaNet
>    evaluations; train-mode dropout gave them different coefficients,
>    breaking E[w] = 1 and feeding phi gradients of a different random warp.
>    Fix: the GammaNet is dropout-free.
>
> Plus two hardening changes from the same audit: the warp is parametrized
> over u = (t - t_min)/(1 - t_min) (endpoints pinned on the *sampled*
> interval, so the weight's span is constant and E[w|c] = 1 is exact), and a
> warm-up gate anneals the warp in over `var_warmup_steps` = 2k steps so the
> schedule cannot lock onto an untrained denoiser's order. Round 2 sweeps
> degree ∈ {5, 9}: 5 re-baselines the fixed implementation, 9 is the
> requested higher polynomial order (audit verified the degree-9 algebra to
> 1e-6 against autograd).

## What was built

A NN-parametrized noise schedule `gamma_phi(c, t)` for EFLM, following MuLAN
(*Diffusion Models With Learned Adaptive Noise*, Suppl. E.2), able to emit a
**global** (per-sequence) or a **per-position** schedule. The per-position mode
is the point: it gives each position its own alpha_t(t), i.e. a **learned
decoding order**.

| file | role |
|---|---|
| `models/var_noise.py` | `GammaNet`: (c, t) -> monotone time warp (tau, dtau/dt) |
| `noise_schedules.py` | `VariationalAdaptiveSchedule`: composes the warp with the base schedule; `get_noise` gains a `variational` branch |
| `models/sphere_dit.py` | `SphereDiT.forward(..., output_state=True)` returns the hidden state = the context c |
| `algo.py` | `EFLM._nll_variational` (two-pass loss + MuLAN reweighting), `EFLM.dlm_state` |
| `samplers.py` | `EFLMSampler._schedule_alphas`: per-position alpha and per-position Euler step |
| `configs/noise/log-linear-variational.yaml` | the arm's config |
| `scripts/{train,sample}/sudoku/eflm_rescale_variational.sh` | one train / one eval run |
| `tests/test_var_noise_schedule.py` | contract tests on the schedule + the autocast-cache regression |

### Parametrization

`GammaNet` reads the DLM hidden state `c` in [B, L, D] and emits basis
coefficients; the schedule is closed form in `t`, so `d/dt` is analytic.

* **polynomial** (default, MuLAN Suppl. E.2, degree 5): `f'(t) = p(t)^2` with
  `p` of degree `(degree-1)/2`, so `f` is monotone by construction. Degree
  must be odd.
* **rbf**: `f'(t) = sum_k softplus(psi_k) N(t; mu_k, s)` on a fixed grid of
  `degree` centers; monotone because `f' >= 0`.

Both are normalized, `tau(c, t) = f(c, t) / f(c, 1)` in [0, 1] with
`tau(c, 0) = 0`, `tau(c, 1) = 1`, plus a uniform ridge `f' += 1e-3` (the
AdaptiveSchedule `uniform_mix` trick) so the normalizer stays positive and the
endpoints stay exact even if the basis coefficients collapse. Note the rbf
identity init is approximate (a Gaussian comb sags at the interval ends, tau'
ripples ~20%); polynomial init is exactly the identity and is the default.
**Deviation from MuLAN**: MuLAN maps its
polynomial onto `gamma in [gamma_min, gamma_max]` and sets
`alpha^2 = sigmoid(-gamma)` (variance preserving). EFLM's interpolant
`x_t = alpha_t e + (1 - alpha_t) eps` is not variance preserving, so we
instead compose the warp with the base schedule,

    alpha_t(c, t) = alpha_base(tau(c, t)),
    alpha'_t(c, t) = alpha'_base(tau) * dtau/dt.

`tau` is MuLAN's `gamma` up to `gamma = gamma_min + (gamma_max - gamma_min) tau`,
so nothing is lost; the gain is that the **identity init `tau = t` reproduces
the un-warped EFLM baseline exactly**, keeping this arm directly comparable to
`experiments/eflm_rescale_sudoku`.

Options (`configs/noise/log-linear-variational.yaml`): `var_degree` (5),
`var_basis` (polynomial / rbf), `var_scope` (positional / global),
`var_model_type` (dit / mlp), `var_num_layer` (2), `var_embed_dim` (128).

### Where the context c comes from (the cycle, and how it is broken)

`x_t` needs `alpha_t`, which needs `c`, which needs a forward pass. Broken with
a two-pass step, the same structure as the existing self-conditioning path:

1. draw an **independent** `t_ctx ~ U[0,1]` and corrupt with the base
   schedule: `xt_ctx = a_base(t_ctx) e + (1-a_base(t_ctx)) eps`;
2. `c = DLM_hidden(xt_ctx, sigma(t_ctx))`, **no grad**;
3. `alpha_t = alpha_base(tau_phi(c, t))`, corrupt at the loss's `t` with the
   **same** `eps` draw;
4. loss on `DLM(x_t, sigma(alpha_t))`.

`t_ctx` must be independent of `t` (found in external review): the DLM is
sigma-conditioned, so a context taken at the loss's own `t` hands `phi` the
value of `t` through `c`; the MuLAN weight only accounts for the *explicit*
`t`-slot (`dtau/dt` is a partial derivative holding `c` fixed), and the
un-reweighted `c`-pathway re-opens the degenerate collapse. With `t_ctx`
independent, `E_t[w | c] = 1` exactly and the invariance argument holds
conditional on `c` — mirroring MuLAN, whose context (the auxiliary latent z)
is likewise t-independent. At sampling, `c` is read from the current
integration state; its corruption level is one draw of the arbitrary levels
seen in training.

Cost: ~1.5x the train step, 2 NFE per sampling step (`avg_nfe` doubles).
For sudoku (V=12) the context pass's discarded logits are negligible; at LM
vocab sizes a state-only forward path would be worth adding.

### Training objective (why it does not collapse)

EFLM's loss is plain CE. Minimizing it w.r.t. `phi` is degenerate: the warp
would push `alpha ~ 1` over the whole interior and cram all denoising into
`t ~ 1`. The fix is MuLAN's own mechanism — weight the per-token CE by

    w = (-dalpha/dt) / (alpha(0) - alpha(1)),      E_{t~U[0,1]}[w] = 1.

For a **global** schedule the weighted objective is `int CE(alpha) dalpha`,
which is *invariant* to how the warp distributes alpha over t — so there is no
degenerate optimum and no gradient signal either. Non-invariance (hence the
learning signal) appears only through the **per-position coupling**: `CE_i`
depends on every position's alpha through attention, so the objective becomes a
path integral in R^L and `phi` is rewarded for denoising the informative
positions first (MuLAN Sec. 3.5). `w == 1` for the un-warped base schedule, so
the loss reduces **exactly** to today's EFLM CE at init.

## Hypothesis

1. **H1 (order is learnable)**: with `var_scope=positional` the schedule
   separates across positions — `sched/alpha_spread` (std of alpha over
   positions, logged each step) grows away from 0 and stays there.
2. **H2 (order helps)**: sudoku exact-match accuracy for the positional arm
   exceeds both the `global` arm and the fixed-schedule
   `eflm_rescale_sudoku` naive arm at the same R, because sudoku has a real
   constraint-propagation order (high-constraint cells first) to discover.
3. **H3 (R interacts with the warp)**: R sets *when* a token decodes under a
   fixed schedule (`eflm_rescale_sudoku`: t*(R) = 0.60 / 0.71 / 0.83 for
   R = 5 / 8 / 16). A learned warp can compensate, so the accuracy spread
   across R should be *smaller* for the learned schedule than for the fixed
   one.

## Design

- Data: sudoku **hard** (30 clues), 48k train / 2k val, L = 180
  (`[BOS] puzzle(89) [BOS] solution(89)`); the loss and the sampler only touch
  the 89 solution positions.
- Model: `tiny-sphere-dit` (hidden 512, 8 blocks, 8 heads, ~28.6M) + the gamma
  net (`dit`, 2 blocks, width 128, ~0.4M).
- Algo: `eflm`, `invert_time_convention=false`, `rho_min = rho_max = R`,
  bs 256, 20k steps, lr 3e-4 — identical to
  `scripts/train/sudoku/eflm_rescale.sh` except `noise=log-linear-variational`.
- Grid (`sweep.py`): **R x scope x seed** = 3 x 2 x 3 = **18 jobs**.
  - R in {5.0, 8.0, 16.0} (`algo.rho_min = rho_max`)
  - scope in {`positional`, `global`}; `global` is the control for H2
  - seed in {1, 2, 3}
- Eval: `mode=sudoku_eval`, 180 steps, exact velocity, greedy last,
  `top_k_velocity=-1` — the same protocol as `eflm_rescale_sudoku`, so its
  `naive` (fixed log-linear) cells at R = 5.0 / 8 / 16 are the baseline.
  Caveat (repo-wide pattern): the eval scripts rebuild the model from CLI
  overrides with `strict_loading=false`, so `VAR_SCOPE`/`RHO` must match the
  training cell — the sweep guarantees this; hand runs must be careful.

## Success criteria

1. **Runs clean**: `tests/test_var_noise_schedule.py` green; a 30-step train +
   a 16-step sudoku eval complete end to end (smoke).
2. **H1**: `sched/alpha_spread` > 0 and rising for `positional`, identically 0
   for `global`.
3. **H2**: mean accuracy (over seeds) of `positional` > `global` and >
   `eflm_rescale_sudoku` `naive` at the same R.
4. **H3**: `max_R acc - min_R acc` smaller for the learned arm than for the
   fixed-schedule arm.

## Compute

18 x 1-GPU SLURM jobs, partition `thickstun,desa` (exclude `desa-compute-01`),
2 cpus, 16G, 8h walltime (train ~3-5h at 1.5x the baseline step cost, eval
~30min at 2 NFE/step). Outputs in `outputs/vmulan_sudoku/{tag}/`,
`tag = vmulan_{scope}_r-{R}_lr-{lr}_d-hard_rs{seed}`. Checkpoints are kept —
the learned schedule itself (`noise.gamma_net.*`) is the object of study.

## Run

    python experiments/vmulan_sudoku/sweep.py --dry-run   # inspect (18)
    python experiments/vmulan_sudoku/sweep.py             # submit all
    # subsets:
    python experiments/vmulan_sudoku/sweep.py --scopes positional --seeds 1

Post-hoc (compute node): load `checkpoints/last.ckpt`, evaluate
`noise.alpha_t(t, c)` on validation puzzles over a t grid, and plot the
per-position alpha curves — the decoding order — against the sudoku grid.
