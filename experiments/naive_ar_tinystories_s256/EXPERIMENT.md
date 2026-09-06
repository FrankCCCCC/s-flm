# naive_ar_tinystories_s256 — Experiment Design

**Slides:** `slides/jun25_2026/slides.md` — "Naive AR Baseline", **Max Seq Len = 256**.
**Spec:** `setup.md` (methods / grid / eval protocol).

## Hypothesis
Standard non-geometry small-DiT baselines trained on TinyStories at seq-256 are the
reference points against which the geometry flows (S/E/H-FLM) and advanced tricks are
compared. Four methods, each swept over LR and seed:
- **`ar`**   — causal autoregressive. Valid PPL is a *true* AR PPL.
- **`mdlm`** — masked (absorbing) discrete diffusion. Valid PPL is a denoising-ELBO bound.
- **`duo`**  — uniform-state discrete diffusion. Valid PPL is a denoising-ELBO bound.
- **`flm`**  — base flow language model. Valid PPL is an unweighted denoising CE.
- **`sfmta`** — S-FLM + truncation (`alpha_max` 0.121) + adaptive schedule, the best
  "advanced geometry" recipe from `experiments/adv_geo_tinystories_s256`. Valid PPL is
  an unweighted denoising CE on the sphere.

## Design
- **45 cells** = 5 methods × LR {3e-4, 1e-3, 5e-3} × seed {1, 2, 3}.
- Identical small DiT (width 768, depth 12, heads 12): `model=small` for ar/mdlm/duo,
  `model=small-flm` for flm, `model=small-sphere-dit` (`init=ngpt`) for sfmta — all
  768/12/12.
- 30k steps, global batch 512, **seq 256**, bf16, EMA 0.9999, AdamW (wd 0,
  betas (0.9,0.999), eps 1e-8, grad-clip 1.0), constant schedule w/ 2500-step warmup.
- **Checkpoints every 5k steps, all retained** (`SAVE_TOPK=-1`).
- Eval: 180 sampling steps for mdlm/duo/flm (`setup.md`); greedy last step where the
  sampler exposes it (duo `noise_removal=greedy`; flm's Euler sampler always argmaxes the
  last step). AR ignores `sampler.steps` and does one forward per token.
- Only the **training** seed is swept — the sample scripts expose no SEED knob, so eval
  noise is common across seeds and the error bars isolate training-seed variance
  (same convention as `experiments/seed_errbar_tinystories_256`).

## Known comparability caveats
- **Valid PPL mixes three estimands** (exact NLL / ELBO bound / denoising CE) and must not
  be ranked in one column. Only `mdlm` ↔ `duo` are directly comparable.
- **AR is decoded greedily** (`scripts/sample/tinystories/ar.sh` defaults `GREEDY=true`), so
  its 64 samples collapse to one string and its GenPPL is a mode decode. Re-run that cell
  with `GREEDY=false` for a number rankable against the stochastic rows.
- **AR carries `algo.adaLN: False`** (162.2M params vs 169.6M for the other three), so
  "parameter-matched" is approximate.
- **`sfmta` trains on 1 GPU** (`PER_GPU_BS=32`, accum 16) where the four flat baselines
  use 4 (accum 4). Global batch is 512 either way; the choice makes the three sfmta
  seeds identical to the `adv_geo_tinystories_s256/sfm_ada_trunc_lr1e-3` checkpoint that
  the seed-1 cell reuses, so the seed error bar carries no accumulation confound.
- **`sfmta` seed-1 reuses `adv_geo_tinystories_s256/sfm_ada_trunc_lr1e-3`** — a config
  diff of the two `.hydra/config.yaml` shows an identical training recipe (seed 1,
  lr 1e-3, 30k steps, seq 256, global batch 512, `alpha_max` 0.121, EMA 0.9999, AdamW
  wd 0 / clip 1.0, 2500-step warmup); only the run name, wandb group and DDP width
  differ. `last.ckpt` was copied (not linked) into the cell, and the re-run eval
  reproduces it: val PPL 12.177 vs 12.1765 recorded.

## GPU allocation
- 1 job per cell, `gpu:4` on `thickstun,desa` (exclude desa-compute-01). `PER_GPU_BS=32`
  (accum = 512/(4×32) = 4). Train→eval in one SLURM job; idempotent/resumable.

## Expected wall-clock
- ~16.4 GPU-hr per cell on RTX 6000 Ada at bs=32 → ≈ 4–5 hr train + ~0.5 hr eval per cell
  on 4 GPUs. 36 cells ≈ **720 GPU-hr**; at ~7 concurrent 4-GPU jobs on a free cluster that
  is ~6 waves ≈ **1.5–2 days**, longer under contention.

## Outputs
`outputs/naive_ar_tinystories_s256/m-{method}_lr-{lr}_sd-{seed}/` → checkpoints/,
eval/ppl.json, eval/samples_genppl.json.
Report: `experiments/naive_ar_tinystories_s256/RESULTS.md` (via `report.py`).

---

## Phase 2 — Gen. PPL / entropy frontier (`frontier_sweep.py`)

`setup.md` "GenPPL & Entropy Frontier Evaluation". Consumes the phase-1 checkpoints; no
retraining.

- **1440 cells** = 4 methods {mdlm, duo, flm, sfmta} × seed {1,2,3}
  × NFE {1,4,8,16,32,64,128,256} × T {0.50 … 1.20, 15 values}, **512 samples each**.
- **LR fixed at 1e-3** — `setup.md` sweeps only the seed here, and §3.4 of RESULTS.md
  selects 1e-3 as the shared LR. AR is excluded (its sampler has no NFE budget).
- Same decoders as the phase-1 eval: `ancestral` for mdlm, `ancestral` +
  `noise_removal=greedy` for duo, `flm_euler` for flm, `sfm` + `noise_removal=greedy`
  for sfmta. Only `sampler.steps` and `sampler.temperature` move — except for the one
  sfmta deviation below.

### sfmta: the frontier needs the full-vocab velocity
`setup.md`'s recorded geometry protocol is exact velocity with **`top_k_velocity=1`**.
`SFMSampler._select_topk` re-`log_softmax`es the retained logits, so at k = 1 the
velocity weights are a point mass and `v = log_x(e_argmax)` — the trajectory is a pure
argmax walk, the greedy last step is an argmax too, and **the temperature cancels
completely**. Measured on the seed-1 checkpoint at NFE 32 (16 samples): H = 3.975 at
T = 0.50 vs 3.967 at T = 1.20, a pure RNG jitter. All 15 T-cells would collapse to one
point and there would be no frontier to draw. This is the same argmax-inertness that
made `experiments/eflm_sde` sweep the SDE `eta` instead of T.

So sfmta's frontier is run at **`top_k_velocity=-1`** (the exact full-vocab velocity),
which keeps the tempered `p` inside `v = Σ_k p_k log_x(e_k)`. Measured at NFE 32:
H 4.23 → 4.59 and GenPPL 15.7 → 49.2 over T 0.50 → 1.20 — a frontier of the same shape
and range as the flat baselines'. The cost is memory: several dense (B, L, V) float64
tensors are live at once, hence `EVAL_BS=8` for sfmta (16 for flm, 32 for mdlm/duo).

**Read the sfmta curve as "S-FLM with the exact velocity", not as the top-1 protocol
row of RESULTS.md.** The top-1 protocol point is still recorded, once, by the phase-1
eval (180 steps: GenPPL 12.20, H 3.933, seed 1).
- Deliverable: Gen. PPL (y, log) vs per-sample unigram entropy (x), one frontier line per
  NFE, mean ± sd over the 3 training seeds (S-FLM paper Fig. 10 / App. C.8).

**Expected invariance (not a bug):** duo and flm argmax on the final sampling step, so at
**NFE = 1** the entire sample is a temperature-independent argmax — their 15 T-cells collapse
to one point. MDLM's last step is stochastic, so it keeps a curve at NFE = 1.

- GPU allocation: 96 SLURM jobs (one per method × seed × NFE), `gpu:1` on `thickstun,desa`
  (exclude desa-compute-01); each job walks its 15 temperatures in sequence.
  `EVAL_BS` 32 for mdlm/duo, 16 for flm, 8 for sfmta (dense (B, L, V) float64 sampler
  state); all measured to fit the 24 GB A5000.
- Expected wall-clock: ~11 GPU-hr each for mdlm/duo, ~22 for flm, ~18 for sfmta
  ⇒ **≈ 63 GPU-hr**; longest single job (flm, NFE 256) ≈ 3.5–6 hr.
- Outputs: `outputs/.../m-{method}_lr-1e-3_sd-{seed}/frontier/nfe-{nfe}_t-{T}/samples_genppl.json`.
  Figure + tables: `visualization/genppl_entropy_frontier_line.py`.
  Report: `experiments/naive_ar_tinystories_s256/FRONTIER_RESULTS.md`.
- Staging: `frontier_sweep.py --methods … --seeds …` restricts submission, and a cell
  whose `checkpoints/last.ckpt` does not exist yet is skipped, so sfmta's seeds can be
  submitted as their training lands.

**Legacy dirs:** `ar/` and `mdlm/` predate this naming and are the lr-3e-4 / seed-1 cells.
Rename them to `m-ar_lr-3e-4_sd-1` / `m-mdlm_lr-3e-4_sd-1` to reuse their checkpoints;
otherwise those two cells retrain from scratch. Their *eval* outputs are stale regardless
(mdlm was evaluated at 256 sampling steps, now 180).
