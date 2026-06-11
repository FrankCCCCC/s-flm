# EXPERIMENT: LangFlow — faithful reimplementation + fair comparison to S-FLM (TinyStories)

**Type:** faithful-reimplementation + validation. The algorithm is *fixed* by the paper
(`/tmp/langflow_spec.md`, authoritative; `/tmp/langflow_paper.txt`, detail). This experiment
defines how we *validate that the implementation is correct* and how we *compare it to S-FLM*
on TinyStories. We are NOT proposing novel research; "success" = correct, stable, paper-faithful
LangFlow whose central claims reproduce in miniature on a dev cluster.

**Strategy (decided, do not re-litigate):** Strategy 1, full faithful.
`LangFlow(trainer_base.Diffusion)` owns the γ = logNSR path. `UnifInfoSchedule` in
`noise_schedules.py` becomes a self-contained learnable **Gumbel** `NoiseSchedule` (params
`P_µ`, `P_β`, `H_∞`) owning `sample_gamma` / `alpha_sigma_from_gamma` / `scheduler_loss`, wired
into `get_noise`, with a config option for a **uniform fixed** scheduler. Reuse the `sphere-dit`
backbone (same as S-FLM, for fair comparison) with flag-gated additions: self-conditioning input
projection (zero-init `W_in`, `W_SC`), Plaid-style logit bias (r ramped 0→1 over 5k steps).
Self-conditioning `p_SC=0.25`, always-on at sampling.

---

## 1. Hypotheses (falsifiable, validation-style)

- **H1 (correctness / stability).** LangFlow trains stably on TinyStories with the
  `small-sphere-dit` backbone: training CE loss decreases monotonically (in trend) over the first
  few thousand steps, no NaN/Inf anywhere (model output, γ, α/σ, scheduler loss), and the
  geometric/schedule invariants hold throughout:
  - sphere embeddings stay L2-normalized (× √D) — reuse S-FLM's `sphere_normalize`;
  - VP identity `α²_γ + σ²_γ = 1` holds to fp tolerance (`σ²=sigmoid(γ)`, `α²=sigmoid(−γ)`);
  - γ stays inside the configured clip `[a, b]` (1e-5 / 1−1e-5 Gumbel quantiles);
  - `P_β > 0` and `H_∞ > 0` are preserved by the softplus/exp parameterization every step.
  **Refuted if** any NaN appears in a smoke run, or CE loss fails to drop below its step-0 value
  by step 1k, or any invariant is violated.

- **H2 (trainable Gumbel scheduler — paper's central claim, Note 2).** The trainable Gumbel
  scheduler reduces generative perplexity dramatically vs the uniform-fixed scheduler, holding
  everything else equal. The paper shows Gen-PPL ≈ 1000 → 154 from this single change. Mechanism
  diagnostics: `L_Scheduler` decreases over training, and the learned surrogate entropy
  `H_γ = H_∞·exp(−exp(−(γ−P_µ)/P_β))` tracks the empirical `ℓ_CE(γ)` profile (i.e. `H_γ ≈
  stopgrad(L_CE)` in expectation across the γ range; residual `(L_CE − H_γ)²` shrinks).
  **Refuted if** trainable Gen-PPL ≥ uniform-fixed Gen-PPL at the comparison horizon, or if
  `L_Scheduler` does not decrease, or if `P_µ/P_β/H_∞` diverge / collapse.

- **H3 (self-conditioning, paper Table 1).** Enabling self-conditioning (`p_SC=0.25` train,
  always-on sample) improves **both** Gen-PPL and validation PPL vs self-cond-off, all else equal.
  **Refuted if** self-cond-on is worse than self-cond-off on **both** metrics (a single-metric
  regression is a `TBD` flag, not a refutation, given short horizons).

- **H4 (fair comparison to S-FLM).** With the **same** `small-sphere-dit` backbone and the same
  TinyStories data, LangFlow is competitive with S-FLM under two hyperparameter regimes:
  (a) LangFlow's own hyperparameters; (b) S-FLM's hyperparameters (mirror `sfm.sh`). This is a
  characterization, not a "LangFlow must win" claim — the deliverable is a *fair, apples-to-apples*
  comparison table. **Reported, not refuted**: we report Gen-PPL / val-PPL / entropy / NFE for
  LangFlow(a), LangFlow(b), S-FLM on one shared axis. We flag a *correctness* problem only if
  LangFlow is wildly worse than S-FLM (e.g. >5× Gen-PPL) under regime (b), which would suggest a
  reimplementation bug rather than a method difference.

---

## 2. Success criteria (concrete thresholds)

| ID | Criterion | Where measured |
|----|-----------|----------------|
| H1 | 0 NaN/Inf over the full **smoke** run (200 steps); CE loss(step 1k) < CE loss(step 0); all four invariants asserted in-code pass every logged step | smoke run + 1k of short run |
| H2 | `Gen-PPL(trainable Gumbel) < Gen-PPL(uniform-fixed)` at the comparison horizon, by a clear margin (target ≥1.5× lower, mirroring the paper's ~6.5× but allowing for the tiny horizon); `L_Scheduler` strictly decreasing in trend; final `(L_CE−H_γ)²` < its value at warmup end | short run, both scheduler variants |
| H3 | self-cond-on Gen-PPL ≤ self-cond-off Gen-PPL **and** self-cond-on val-PPL ≤ self-cond-off val-PPL (ties acceptable at short horizon) | short run, sc on/off |
| H4 | LangFlow(b) Gen-PPL within 5× of S-FLM Gen-PPL (sanity, rules out gross bugs); full comparison table produced for all three configs | short run, 3 configs |

**Seeds.** Smoke / correctness: 1 seed (`seed=1`). Comparison claims (H2–H4): **3 seeds**
(`seed=1,2,3`) for the primary metric (Gen-PPL) on the must-run cells only; report mean ± std.
Optional cells: 1 seed. (3 seeds is affordable here — see budget.)

**Statistical note.** With 3 seeds we report mean ± std and a sign test / non-overlapping ±1σ
bands rather than a formal p-value; n=3 is too small for a meaningful t-test. We accept H2/H3 on a
**consistent directional effect across all 3 seeds**.

---

## 3. Baselines

- **S-FLM (primary baseline).** Existing `algo=sfm`, `model=small-sphere-dit model.init=ngpt`,
  `noise=log-linear`, runnable today via `scripts/train/tinystories/sfm.sh`
  (GLOBAL_BS=512, batch_size=32, max_steps=30_000, `wandb.project=tinystories-flm`,
  `wandb.group=geometry-vs-tricks`, `+wandb.offline=true`). Eval via
  `scripts/sample/tinystories/eval.sh` with `MODEL_TYPE=sfm`.
  - `TBD:` exact W&B run ID of the most recent S-FLM TinyStories run (commit a9eeb77 mentions OWT
    runs; the latest TinyStories S-FLM run ID should be filled in by whoever has W&B access). The
    architect can also just *re-run* `sfm.sh` for a same-cluster, same-data baseline rather than
    rely on a historical ID — preferred for a clean comparison.
- **LangFlow uniform-fixed scheduler** is the *internal* baseline for H2.
- **LangFlow self-cond-off** is the *internal* baseline for H3.

---

## 4. Variables (changed vs held fixed)

**Held fixed across all LangFlow + S-FLM runs (for fairness):**
- Backbone: `model=small-sphere-dit` (hidden 768, 12 layers, 12 heads, cond_dim 128), `model.init=ngpt`.
- Data: `data=tinystories` (gpt2 tokenizer, ctx 1024, wrap=True, EOS inserted).
- Precision bf16; gradient_clip_val 1.0; EMA 0.9999.
- Eval protocol: same `eval.sh` harness, same `sampler.steps`, same GPT-2-large Gen-PPL scorer.

**Changed (the independent variables):**
- `algo`: `sfm` (baseline) vs `langflow`.
- Scheduler: trainable Gumbel `UnifInfoSchedule` vs uniform-fixed.
- Self-conditioning: on (`p_SC=0.25`) vs off.
- Logit bias: on (r: 0→1 over 5k) vs off.
- Hyperparameter regime for LangFlow: (a) LangFlow paper vs (b) S-FLM (`sfm.sh`).

---

## 5. Ablations

Factorial: `{trainable Gumbel, uniform-fixed} × {self-cond on/off} × {logit-bias on/off}` = 8 cells.
**Do not run the full 8×3-seed grid.** Use one-factor-at-a-time around the reference config.

**Reference config (R):** trainable Gumbel + self-cond on + logit-bias on, LangFlow hyperparameters.

| # | Config | Purpose | Must-run? | Seeds |
|---|--------|---------|-----------|-------|
| R | trainable + sc-on + bias-on | reference LangFlow | **yes** | 3 |
| A1 | uniform-fixed + sc-on + bias-on | H2 (scheduler) | **yes** | 3 |
| A2 | trainable + sc-off + bias-on | H3 (self-cond) | **yes** | 3 |
| A3 | trainable + sc-on + bias-off | logit-bias effect | optional | 1 |
| A4 | R but S-FLM hyperparameters | H4(b) | **yes** | 3 |
| B0 | S-FLM (`algo=sfm`) | H4 baseline | **yes** | 3 |

Must-run set = {R, A1, A2, A4, B0} → 5 cells × 3 seeds = 15 short runs, + A3 (1 run) optional.

---

## 6. Metrics (W&B keys)

**Primary**
- `val/gen_ppl` — generative perplexity (GPT-2-large scorer, retokenized, first-chunk-only), the
  decisive metric for H2/H3/H4. Logged at val by `_generate_samples`/Metrics; also emitted to
  `samples_genppl.json` as `gen_ppl_first_chunk_retok` by `eval.sh`.

**Secondary**
- `val/ppl`, `val/nll` — validation perplexity / NLL bound from `_eval_ppl` (`trainer.validate`).
  - **DECIDED (user, 2026-06-09): the Theorem-3.1 ODE-NLL bound is DEFERRED.** Report the existing
    `val/ppl` the repo already computes for sphere models (comparable to S-FLM); Gen-PPL is the
    headline metric for H2/H3/H4. The tight ODE bound is a later pass.
- `val/sample_entropy` (and `entropy` in `samples_genppl.json`) — guards against mode collapse /
  degenerate repetition.
- `avg_nfe` (sampler NFE per batch, in `samples_genppl.json`) — sampling cost; held fixed via
  `sampler.steps` for fair comparison.

**Diagnostic (LangFlow-specific, new keys to add — see §9 feasibility)**
- `sched/P_mu`, `sched/P_beta`, `sched/H_inf` — Gumbel parameter trajectories (must stay finite,
  `P_beta>0`, `H_inf>0`).
- `sched/loss` — `L_Scheduler = (stopgrad(L_CE) − H_γ)²`; expected to decrease (H2 mechanism).
- `sched/H_gamma_mean`, `sched/ce_vs_gamma_residual` — surrogate entropy vs empirical `ℓ_CE(γ)`
  alignment (H2 mechanism).
- `train/ce_loss` — the CE term alone (so the scheduler term doesn't contaminate the loss curve).
- `diag/gamma_min`, `diag/gamma_max` — confirm γ stays in clip range (H1 invariant).
- `diag/logit_bias_r` — current ramp value of r (should be 0→1 over first 5k steps).

---

## 7. Compute budget

**Hardware:** single node, 1–4 GPUs. Cluster has many `nvidia_rtx_a6000` (48 GB) and
`nvidia_rtx_6000_ada` (48 GB) nodes; memory note pins `thickstun-compute-01` (8× RTX 6000 Ada).
Follow the cluster single-GPU workaround for sampling/eval (`strategy=single-device`,
`SLURM_JOB_NAME=bash`, `NCCL_P2P_DISABLE=1`) to avoid NCCL crashes; multi-GPU DDP only for training.

**Smoke (correctness, H1):** `max_steps=200`, `tiny-sphere-dit` model, `global_batch_size=32`,
1 GPU. Wall time **minutes** (<10 min). Run once per code change.

**Short comparison (H2–H4):** mirror `sfm.sh` — `small-sphere-dit`, `global_batch_size=512`,
`batch_size=32`, `max_steps=30_000`, 4 GPUs. S-FLM's existing 30k-step TinyStories run is the
calibration point for wall time.
- `TBD:` measured wall time for `sfm.sh` at 30k steps on 4× A6000/Ada — fill from the existing
  S-FLM run logs. *Estimate*: small DiT (~130M) at bs 512, 30k steps is on the order of a few hours
  on 4 GPUs. If that proves too long for 3-seed × 5-cell, the cheap fallback is **max_steps=10_000**
  for the ablations and 30k only for R, A1, B0 (the H2 cells). Decide after the first timed run.

**Run accounting (must-run set, short config):**
- 5 cells × 3 seeds = **15 training runs** + 1 optional (A3). Plus 15 eval passes (`eval.sh`).
- GPU-hours ≈ 15 × (4 GPUs × wall_hours). With wall ≈ 3 h ⇒ ~180 GPU-h. With the 10k-step
  fallback for ablations ⇒ roughly half that. **Feasible on this dev cluster.** If budget is tight,
  drop ablation seeds to 1 (R/A1/B0 keep 3 seeds for the H2 headline) → ~9 multi-seed-equivalent runs.

---

## 8. Failure modes to watch for (invalidators)

- **Gradients crossing the stopgrad boundary.** γ must be `stopgrad` in the CE path and `L_CE`
  must be `stopgrad` in the scheduler path. If the scheduler params receive CE gradients (or vice
  versa) the comparison to "uniform-fixed" is meaningless. *Check:* with uniform-fixed, P_µ/P_β/H_∞
  must not move; with trainable, the backbone must train identically whether or not `L_Scheduler` is
  added (CE-path gradients unchanged). Add an assertion / unit check.
- **Scheduler param blow-up.** `P_β` or `H_∞` going ≤0 or →∞ (parameterize via softplus/exp;
  clamp/log them). Watch `sched/*` keys.
- **Self-conditioning leakage.** The first (no-grad) self-cond pass must be `torch.no_grad` +
  `stopgrad`; if it leaks grads, training cost doubles and gradients are wrong. *Check:* step time
  with `p_SC>0` should be ~1× (one grad pass), not 2×.
- **Logit-bias double-counting / NaN at γ→±∞.** The Plaid bias `r·(α_γ/σ²_γ)·eₓᵀz_γ` blows up as
  σ²_γ→0 (γ→−∞, clean). The γ clip `[a,b]` must bound this; assert finite logits.
- **Eval contamination / mismatch.** Gen-PPL scorer must be the **same** GPT-2-large for all
  configs; `sampler.steps`, retokenize, first-chunk-only must match across S-FLM and LangFlow
  (use the single `eval.sh` harness). TinyStories train/valid both = `tinystories` split — confirm
  no train/valid overlap leaks into PPL (existing repo behavior; just don't change it).
- **EMA scope.** `_get_parameters()` chains `backbone.parameters() + noise.parameters()`, so EMA
  will now cover the Gumbel params. Confirm that's intended (paper EMAs the model; EMA over a 3-dim
  scheduler is harmless but verify it doesn't freeze the schedule). Flag for architect.
- **Sphere-norm × √D vs S-FLM normalization.** Paper uses sphere-normalized embeddings × √D;
  the repo's S-FLM normalizes to the unit sphere. For *fair comparison* both LangFlow and S-FLM
  must use the **identical** embedding normalization. `TBD:` confirm the √D scaling is applied
  consistently (or dropped consistently) for both — otherwise the H4 comparison is confounded.

---

## 9. Feasibility flags / minimal changes needed

- **Scheduler params are auto-optimized.** `trainer_base._get_parameters()` already chains
  `self.noise.parameters()` into AdamW, so registering `P_µ/P_β/H_∞` as `nn.Parameter` in the
  rewritten `UnifInfoSchedule` is sufficient — **no optimizer change needed.**
- **Folding `L_Scheduler` into the loss.** `training_step` returns `losses.loss`. The scheduler
  term must be added to the training loss (`L_CE + L_Scheduler`). Minimal change: have
  `LangFlow.nll`/`_loss` return the combined loss, OR add the scheduler loss inside `training_step`.
  Architect's call; the stopgrad structure (§8) must be preserved either way.
- **`record_time_loss_pair` hook is gated on `config.noise.adaptive`** (trainer_base ~L464). The
  Gumbel scheduler needs the per-(γ, L_CE) pairs too. Minimal change: generalize the gate (e.g.
  `config.noise.adaptive or config.noise.type == 'gumbel'`) or have LangFlow feed the scheduler
  directly in its own `nll`. Prefer LangFlow feeding it directly to keep trainer_base untouched.
- **New W&B diagnostic keys (§6).** `sched/*`, `train/ce_loss`, `diag/*` are not currently logged.
  Add `self.log(...)` calls in `LangFlow.training_step`/`nll`. Cheap, ~6 new keys.
- **`main.py` algo dispatch.** `langflow` must be added to the `if config.algo.name == ...` chain
  (~L630). One-line addition + a `configs/algo/langflow.yaml`.
- **New configs needed:** `configs/algo/langflow.yaml`, `configs/noise/gumbel.yaml` (trainable) and
  `configs/noise/gumbel-uniform.yaml` (or a `noise.trainable: bool` flag on one file),
  `configs/sampler/langflow.yaml` (Euler-on-γ, Algorithm 2). The `small-sphere-dit` /
  `tiny-sphere-dit` model configs are reused as-is.
- **Theorem-3.1 ODE-NLL bound:** DEFERRED (user-confirmed) — report existing `val/ppl`. Out of scope.
- **Self-conditioning is a user-facing toggle (user-confirmed):** `algo.self_conditioning: bool`
  (default true) plus `algo.p_self_cond` (default 0.25). Must be cleanly enable/disable-able.

---

## 10. Required training scripts (exact hydra overrides)

Place under `scripts/train/tinystories/`. Both train **LangFlow**; they differ only in the
optimizer/schedule/horizon regime. Mirror `sfm.sh` structure (REPO_ROOT, CACHE_DIR, DEVICES,
GLOBAL_BS, `+wandb.offline=true`, `hydra.run.dir`).

### (i) `langflow.sh` — LangFlow paper hyperparameters
```
data=tinystories
data.cache_dir="${CACHE_DIR}"
model=small-sphere-dit
model.init=ngpt
algo=langflow
algo.invert_time_convention=false
noise=gumbel                       # trainable Gumbel UnifInfoSchedule
noise.trainable=true               # P_mu, P_beta, H_inf are nn.Parameters
algo.self_conditioning=true
algo.p_self_cond=0.25
algo.logit_bias=true
algo.logit_bias_warmup_steps=5000  # r ramps 0 -> 1
lr_scheduler=constant_warmup       # 2500-step warmup (default num_warmup_steps=2500)
optim.lr=3e-4
training.ema=0.9999
loader.global_batch_size=512
loader.batch_size=32
loader.eval_batch_size=32
loader.num_workers=8
trainer.max_steps=30_000           # paper is 1M; this is the dev-cluster horizon
trainer.devices=4
trainer.val_check_interval=60_000  # mirror sfm.sh (eval via eval.sh post-hoc)
trainer.limit_val_batches=0
trainer.num_sanity_val_steps=0
callbacks.checkpoint_every_n_steps.every_n_train_steps=2_500
eval.generate_samples=False
wandb.project=tinystories-flm
wandb.group=geometry-vs-tricks
+wandb.name=tinystories_langflow_paperhp
+wandb.offline=true
```

### (ii) `langflow_sfmhp.sh` — LangFlow under S-FLM hyperparameters (mirror `sfm.sh`)
Identical to (i) **except** the optimizer/schedule/horizon block is taken verbatim from `sfm.sh`
(same `optim.lr`, `lr_scheduler`, warmup, `max_steps=30_000`, `global_batch_size=512`,
`batch_size=32`, `gradient_clip_val`, EMA). Since `sfm.sh` already uses lr 3e-4 / EMA 0.9999 /
`constant_warmup` / 30k steps, the *numerical* HP block is the same; the script exists to (a) make
the "regime (b)" run explicit and reproducible, and (b) absorb any future divergence between the
two regimes. Change only:
```
+wandb.name=tinystories_langflow_sfmhp
```
`TBD:` if LangFlow's paper warmup (2500) ever differs from `sfm.sh`'s, keep (ii) pinned to whatever
`sfm.sh` uses at run time so the comparison stays honest.

### (iii) smoke config (CI-speed, correctness for H1) — add as `langflow_smoke.sh` or env-flag on (i)
```
model=tiny-sphere-dit
trainer.max_steps=200
loader.global_batch_size=32
loader.batch_size=32
trainer.devices=1
strategy=single-device
trainer.val_check_interval=200
trainer.num_sanity_val_steps=0
algo=langflow noise=gumbel noise.trainable=true
algo.self_conditioning=true algo.logit_bias=true
+wandb.name=tinystories_langflow_smoke
+wandb.offline=true
```
Run env per cluster workaround: `SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1`.

**Ablation toggles** (reuse (i), override one factor):
- A1 uniform-fixed: `noise.trainable=false` (or `noise=gumbel-uniform`).
- A2 self-cond off: `algo.self_conditioning=false`.
- A3 logit-bias off: `algo.logit_bias=false`.
- Seeds: `seed=1 / 2 / 3`.

**Eval** (all configs): `scripts/sample/tinystories/eval.sh` — add a `MODEL_TYPE=langflow` case
mapping to `MARGS=(model=small-sphere-dit model.init=ngpt algo=langflow noise=gumbel
sampler=langflow ...)` mirroring the existing `sfm` case. Produces `val/ppl` + `ppl.json` and
`gen_ppl_first_chunk_retok` + `entropy` + `avg_nfe` in `samples_genppl.json`.

---

## 11. W&B project / run naming convention (exact strings)

- **project:** `tinystories-flm`  (matches existing S-FLM / HFLM TinyStories runs)
- **group:** `geometry-vs-tricks` (matches `sfm.sh`)
- **offline:** `+wandb.offline=true` (cluster default; sync later)
- **tags:** default `${noise.type}, ${data.train}, ${data.valid}, ${algo.name}` plus add
  `langflow-validation` for this experiment's runs (so they filter cleanly from prior S-FLM runs).
- **run names** (`+wandb.name=...`):
  - reference: `tinystories_langflow_paperhp[_sN]`
  - S-FLM-HP regime: `tinystories_langflow_sfmhp[_sN]`
  - ablations: `tinystories_langflow_uniform[_sN]`, `tinystories_langflow_noselfcond[_sN]`,
    `tinystories_langflow_nobias[_sN]`
  - smoke: `tinystories_langflow_smoke`
  - eval runs (from `eval.sh`): `langflow_<STEP_TAG>_ppl` and the sample-eval run (mirrors the
    existing `${MODEL_TYPE}_${STEP_TAG}_ppl` pattern)
  - baseline (re-run): `tinystories_sfm_naive[_sN]` (matches the existing `sfm.sh` name)
  where `_sN` = `_s1/_s2/_s3` for the seed.
