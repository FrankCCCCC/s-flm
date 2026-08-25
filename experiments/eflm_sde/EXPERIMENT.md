# eflm_sde — GScheduler / eta search for the EFLM SDE sampler

Goal (`setup.md`): design a `GScheduler` (diffusion scale g(t) of the
marginal-preserving SDE, `experiments/eflm_sde/derivation.md`) and find the
`eta` range that lets EFLM trace a competitive GenPPL-vs-entropy frontier
against DUO / MDLM / FLM (measured in
`experiments/naive_ar_tinystories_s256/frontier_cells.csv`) and S-FLM.

## Why this experiment exists (the entropy trap)

Under the recorded eval protocol (exact velocity, `top_k_velocity=1`,
`noise_removal=greedy`) both the sampling temperature and the last-step decode
are argmax-invariant, so EFLM's ODE sampler has **no diversity knob**: its only
stochasticity is the prior draw. Recorded operating points (180 NFE):

| method | GenPPL | entropy |
|---|---|---|
| EFLM r=0.5 lr 1e-3 m=1.0 (3 seeds) | 11.44 +- 0.54 | 3.82 |
| S-FLM trunc+ada SC (3 seeds)       | 11.22 +- 0.47 | 3.97 |
| MDLM frontier minimum (T=0.5, NFE 256) | 8.44 | 4.06 |

The discrete baselines' frontier lives at H in [4.06, 4.88]; EFLM sits *left*
of the entire measured range. The SDE sampler (`sampler.eta > 0`) injects
sqrt(eta) g(t) dW during integration while preserving the marginals in the
exact/continuous limit, so at finite NFE it is a principled diversity knob.
g(t)'s shape decides *when* noise is injected (early = global structure,
late = token-level), which is the design question.

## Design

**Checkpoints** (frozen, from `experiments/eflm_rescale_auto_trunc_tinystories_256`):
the recommended config `eflmratr_lr-1e-3_r-0.5_m-1.0{,_s2,_s3}` — R=0.5,
lr 1e-3, TAU_MAX=2.4436, ALPHA_MAX=null, self-cond off, 30k steps, global
batch 512, seq 256. 3 training seeds, matching the 3-seed protocol of the
baseline frontier.

**Eval protocol** (aligned with the parent sweep + the baseline frontier):
`scripts/sample/tinystories/eflm_rescale_auto_truncation.sh`, exact velocity,
`top_k_velocity=1`, `noise_removal=greedy`, T=1.0 (inert), 512 samples/cell
(EVAL_BS 16 x 32 batches), GenPPL = gpt2-large retokenized, per-sample unigram
entropy. `RUN_PPL_EVAL=false` (valid PPL is sampler-independent and already
recorded). Eval seed fixed (isolates training-seed variance, same convention
as the baseline frontier).

**GScheduler candidates** — the power family g(t) = (1-t)^p on the
flow-matching time t (t=0 noise, t=1 data; on this schedule 1-t = b, the noise
fraction, b in [0.087, 1]):

| gt_method | p | character |
|---|---|---|
| const  | 0   | uniform injection, full strength up to the decode point |
| sqrt   | 0.5 | the §9 "regular family" a(t) = eta(1-t) |
| linear | 1   | a(t) = eta b^2, vanishes at decode (already implemented) |
| quad   | 2   | early-only injection |

**Swept axes**

- Phase 1 (pilot, seed 1 only): NFE {16, 64} x gt {const, sqrt, linear, quad}
  x eta {0.25, 0.5, 1, 2, 4, 8, 16, 32}, plus eta=0 ODE references at
  NFE {16, 64, 180} (the 180 one ties back to the recorded 11.53).
  67 cells. Verdict rule: per (NFE, gt), the eta-curve in (entropy, GenPPL);
  best gt = lowest GenPPL at matched entropy over H in [4.0, 4.5];
  cells with entropy < 3.0 or GenPPL NaN are collapse, excluded.
- Phase 2 (frontier, 3 seeds): NFE {1, 4, 8, 16, 32, 64, 128, 256} x winning
  gt (top 1-2) x refined eta grid (~8 points spanning the usable entropy
  range), matching the baseline frontier's NFE grid. Plus eta=0 references.

  **Phase-1 verdict (2026-08-16, seed 1, 67 cells):** H1 confirmed — eta
  traces a clean frontier for every g(t). Beyond it, a *free-lunch region*:
  at NFE 64, sqrt eta in [2, 8] and const eta in [0.5, 2] IMPROVE GenPPL
  below the ODE point (9.46-9.9 vs 11.53) while raising entropy — the SDE
  corrects sampler error. H2 partially confirmed: const collapses first
  (eta >= 2-4, GenPPL in the thousands) and quad barely moves entropy
  (dominated), but the winner is NFE-dependent — sqrt dominates at NFE 64
  (11.40 @ H=4.10 vs MDLM 9.63, DUO 10.14, FLM unreachable; 13.75 @ 4.20),
  linear edges sqrt at NFE 16 (16.20 vs 17.64 @ 4.10). H3 already met at
  NFE 64 for FLM (~3x better) and within 13-18% of MDLM/DUO.
  Phase 2 therefore runs gt {sqrt, linear} with NFE-banded eta grids (the
  entropy bought by a fixed eta falls as NFE grows, so the useful window
  shifts up): NFE {4,8} x eta {0.25..32}, NFE {16,32} x {0.5..64},
  NFE {64,128,256} x {1..128}, all log-2 spaced, 3 seeds, + eta=0 refs.
  NFE 1 is excluded (the single step is the decode step; eta never acts) —
  the ODE point stands in for it. 51 jobs / 360 cells, submitted.

  **Phase-2 status (2026-08-16): COMPLETE** — 398 EFLM + 24 S-FLM cells, zero
  failures; results in `RESULTS.md`. A refinement arm gt=p75 (g(t) =
  (1-t)^0.75, 3 seeds x NFE {16, 64, 128} x eta {1..64}, 63 cells) tested
  whether it dominates the sqrt/linear crossover band — it does not
  (interpolates within ~5%); see RESULTS.md §4. EXPERIMENT COMPLETE.
- Phase 2b (S-FLM reference): `outputs/seed_errbar_tinystories_256/sfm_seed{1,2,3}`
  via `sfm_truncated_adaptive.sh` at NFE {1,...,256}, default decode. One point
  per NFE (temperature is inert at k=1/greedy), 3 seeds; 24 cheap cells.

**Hypotheses**

- H1: eta > 0 raises entropy monotonically at fixed NFE; GenPPL rises with it
  (a frontier, not a free lunch).
- H2: g-shape matters at fixed eta-induced entropy: late-noise schedules
  (const) buy entropy cheaply at the token level but should hurt GenPPL less
  than early-noise ones only if the model can still denoise the injection —
  the decode-point invariant (parent RESULTS §4.5) suggests injections beyond
  ratio ~10 at the endpoint collapse decodability, so const at large eta
  should fail first, quad last.
- H3: with the best (gt, eta), EFLM's frontier at NFE >= 32 reaches the
  H >= 4.1 region at GenPPL below FLM's curve (the weakest baseline,
  ~13.4-49 over the range) — i.e. EFLM stops being off-the-chart and becomes
  comparable; parity with MDLM (8.4-17.8) is the stretch goal.

**Compute** — Unicorn, partitions `thickstun,desa` (exclude desa-compute-01),
1 GPU/job. Cells are bundled one job per (seed, NFE, gt) walking the eta grid.
Estimated: pilot ~11 GPU-h (9 jobs), phase 2 ~40-60 GPU-h (24-48 jobs),
phase 2b ~4 GPU-h (3 jobs). Idempotent: a cell is skipped iff its
`samples_genppl.json` exists or its job name is in squeue.

**Outputs** — `outputs/eflm_sde/sd-{seed}/frontier/nfe-{nfe}_gt-{gt}_eta-{eta}/`
(and `.../sfm_sd-{seed}/frontier/nfe-{nfe}/` for 2b);
figure + tables via `experiments/eflm_sde/analyze.py`, which overlays the
EFLM eta-curves on the baseline T-curves from
`experiments/naive_ar_tinystories_s256/frontier_cells.csv`. Results in
`experiments/eflm_sde/RESULTS.md`.

## Code changes required (phase 0)

1. `noise_schedules.py`: `PowerGScheduler(p)`; `GT_Method` += CONST/SQRT/QUAD;
   `get_gscheduler` maps const/sqrt/quad -> p=0/0.5/2 (linear keeps the
   existing `LinearGScheduler`).
2. `configs/sampler/eflm.yaml`: `gt_method: linear`.
3. `samplers.py`: `EFLMState.gt_method` read from `config.sampler.gt_method`
   in `init_state`, passed to `_sde_update` in `step` (previously hardcoded
   to linear).
4. `scripts/sample/tinystories/eflm_rescale_auto_truncation.sh`: env knobs
   ETA, GT_METHOD, TEMPERATURE, NUM_SAMPLE_BATCHES, RUN_PPL_EVAL (defaults
   preserve current behavior).
5. `scripts/sample/tinystories/sfm_truncated_adaptive.sh`: env knobs
   NUM_SAMPLE_BATCHES, RUN_PPL_EVAL.

Verify: smoke sample_eval on a compute node (each gt at eta>0, small batch) —
runs clean, entropy responds to eta, no NaN.

---

## Phase 3 — the temperature frontier line (setup.md "Frontier Line Evaluation")

`setup.md` adds **{Rescale + Auto + Trunc + EFLM}** to the {MDLM, DUO, FLM}
Gen. PPL / entropy frontier of `experiments/naive_ar_tinystories_s256`, swept
over sampling temperature instead of `eta`. Sweep:
`experiments/eflm_sde/tfrontier_sweep.py`.

**Why `top_k_velocity = -1` is the right (and paper-faithful) setting.**
Temperature enters as `logits / T` in `trainer_base.Diffusion.forward`. Under
`top_k_velocity = 1` the velocity is `E[argmax] - x` and the last step is
`argmax`, both invariant to a positive rescaling of the logits, so T cannot
move the sample. Measured (seed 1, NFE 16, 32 samples):

| arm | T=0.50 | T=1.00 | T=1.20 |
|---|---|---|---|
| `top_k_v = 1` | 13.54 @ H 3.803 | 13.54 @ H 3.803 | 13.87 @ H 3.802 |
| `top_k_v = -1` | 14.72 @ H 4.137 | 28.78 @ H 4.469 | 50.30 @ H 4.587 |

(T=0.5 is bit-identical to T=1.0 — 0.5 is a power of two so `logits/T` is
exact; T=1.2 differs only by float rounding flipping near-ties.) With
`top_k_v = -1` the velocity is `softmax(logits/T) @ E - x`, so T reshapes the
target at every Euler step and traces a curve. This is exactly the paper's
"S-FLM exact velocity" arm; App. C.8 states that the k=1 velocity variant
"does not depend on the temperature, so it appears as a single marker rather
than a curve" — that marker is the eta=0 ODE cell of phase 1/2.

**Protocol** — `scripts/sample/tinystories/eflm_rescale_auto_truncation.sh`
on the same frozen 3-seed checkpoints `eflmratr_lr-1e-3_r-0.5_m-1.0{,_s2,_s3}`
(R=0.5, TAU_MAX=2.4436, ALPHA_MAX=null, SNR_CE=false, self-cond off, seq 256):
`VELOCITY=exact TOPK_VELOCITY=-1 ETA=0.0 GT_METHOD=linear`,
`noise_removal=greedy`, 512 samples/cell (EVAL_BS 16 x 32), `RUN_PPL_EVAL=false`,
eval seed fixed at 1 so the prior draw is shared across T.

**Grid** — 3 seeds x NFE {1, 4, 8, 16, 32, 64, 128, 256} x T {0.50, 0.55, ...,
1.20} = 360 cells, plus the 3 missing NFE=1 eta=0 cells that complete the
k=1 marker arm across all eight panels.

**Alignment with the paper (App. C.8)** — matched: 15 evenly spaced
T in [0.50, 1.20]; N = 512 samples/cell; Gen. PPL from a pretrained GPT-2-large
with tokens at or after the first end-of-text masked; per-sample unigram
entropy in nats (Eq. 48) averaged over the cell; one (H, PPL) point per cell;
exact / k=1 velocity variants drawn as curve / marker. Deliberate deviations:

1. NFE {1, ..., 256} rather than {32, ..., 1024} — this is TinyStories at
   seq 256, not OpenWebText at seq 1024, and the repo's baseline frontier
   already uses this grid.
2. The paper's visible window (4.5 <= H <= 6.0, PPL <= 500) is OWT-specific;
   TinyStories seq-256 entropies sit near 3.7-4.7, so the plot shows the full
   measured range instead.
3. `metrics.py` accumulates Gen. PPL corpus-level, `exp(sum nll / sum tokens)`,
   rather than the paper's mean over per-sample perplexities (Eq. 47). This is
   the MDLM-codebase convention and is applied identically to every method
   here, so the curves are mutually comparable; only absolute values shift.

**Hypotheses**

- H4: T is a genuine diversity knob under exact velocity — entropy rises
  monotonically with T and Gen. PPL with it, giving EFLM a real frontier line
  (confirmed at NFE 16 by the smoke above; the sweep tests all NFE).
- H5: the exact-velocity T-curve is *worse* than the eta-curve of phase 2 at
  matched entropy, because full-vocab velocity blurs the target toward the
  embedding mean whereas the SDE keeps a top-1 target and adds isotropic
  noise. If so, the SDE remains the recommended diversity mechanism and the
  T-curve is the paper-protocol reference point.
- H6: the k=1 marker lies below (better than) the T-curve at its own entropy,
  reproducing the paper's finding that top-1 velocity decoding dominates
  unrestricted decoding at matched budget.

**Compute** — measured ~0.8 s per batch-step (32 batches of 16) for the
float64 full-vocab velocity einsum, ~6x the top-1 cost; a cell costs
~40 s + 25.6 s x NFE. 45 SLURM jobs (one per seed x NFE, split into 5-T chunks
for NFE >= 64 so the ~110 min NFE-256 cells run in parallel), ~167 GPU-h,
~10 h wall on `thickstun,desa`.

**Outputs** — `outputs/eflm_sde/sd-{seed}/tfrontier/nfe-{nfe}_t-{temp}/`.
Figure and tables via
`visualization/genppl_entropy_frontier_line.py --eflm-dir outputs/eflm_sde`,
written to `experiments/naive_ar_tinystories_s256/` per setup.md.

**Phase-3 verdict (360 cells + 3 markers, 0 failures): COMPLETE.**
H4 **confirmed** — entropy is strictly monotone in T for all 24 (NFE, seed) curves.
H5 **refuted** — the temperature knob beats the SDE knob by 7-92% throughout
H >= 4.2 (NFE 16-64); the SDE wins only below H ~= 4.1, which temperature
cannot reach. The two knobs cover disjoint bands and neither traces EFLM's
full frontier alone. H6 **partially confirmed** — the k=1 marker has lower
Gen. PPL than the exact-velocity curve at every NFE (11.33 vs 12.44 at
NFE 256), but at ~0.3 nats lower entropy, so neither Pareto-dominates; the
SDE Pareto-dominates the k=1 marker outright at NFE >= 16. Full analysis:
`experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md` §7.
