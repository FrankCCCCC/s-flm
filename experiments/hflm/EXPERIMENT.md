# EXPERIMENT: HFLM (Hyperbolic Flow Language Model) on Sudoku-easy

**Owner:** deuterium1729@gmail.com · **Date:** 2026-06-05 · **Branch:** `hflm`
**One-line:** Swap the sphere for `H^d` in the working S-FLM and check it still solves Sudoku-easy.

This is a *single comparison run* (HFLM vs S-FLM) plus a tiny 2–3 config sweep to clear a known
numerical bound. The implementation is the bulk of the work; this doc only fixes what success means
and what to measure. Geometry is provided by `geo_bridge.py` (`HyperbolicHeatKernel`, `GeoUtils`) — do
not re-derive it.

---

## 1. Hypothesis

An HFLM that (a) embeds tokens on `H^d` with the **embedding length as the radial coordinate**
(`ρ=‖e_v‖`, direction `θ=e_v/‖e_v‖`, embeddings trainable / unnormalized), (b) draws the prior from an
origin-centred **wrapped normal** (`GeoUtils.wrapped_normal`), (c) noises via constant-speed
**hyperbolic geodesic** interpolation (`HyperbolicHeatKernel.geodesic`), and (d) trains the denoiser
`p^θ_{1|t}` with the **same cross-entropy loss** as S-FLM (gradient into the embedding table through the
geodesic endpoint `z₁`, not through the loss head) —

**trains stably and reaches Sudoku-easy exact-match accuracy comparable to S-FLM**, validating
hyperbolic flow as a drop-in alternative geometry to the hypersphere.

Falsifiable form: with the same backbone (`tiny` DiT: 8 layers, dim 512), data, batch size, and step
budget, HFLM's Sudoku-easy exact-match accuracy lands within a fixed margin of the S-FLM baseline and
above the trivial floor, without NaNs or ρ-bound crashes.

---

## 2. Primary metric + success criteria

**Primary metric:** Sudoku-easy **exact-match accuracy (%)** — fraction of generated solutions equal to
ground truth across all 81 cells. Computed today at `main.py:551` as
`correct = (generated == gt).all(dim=1)` over the 2k held-out puzzles (easy = 40/81 givens), sampled at
**180 steps**. This is the paper's Table 1 metric.

**Reference anchors (paper Table 1, easy):** S-FLM (linear) = 81.5; S-FLM + truncation = 94.0;
S-FLM + truncation + adaptive = 94.8; FLM = 94.2; AR-greedy = 14.6. AR-greedy ≈ 14.6 is the "did it
learn anything global" floor.

Let `A_H` = HFLM accuracy and `A_S` = the **S-FLM baseline run in this experiment** (Run B, same
budget — do NOT compare against the paper number, compare against our own re-run).

| Verdict | Condition |
|---|---|
| **Confirmed** | `A_H ≥ A_S − 5` percentage points (absolute) **and** `A_H ≥ 60%` (clears AR-greedy ~14.6 and the CANDI-pure-Gaussian ~63.9 region by a wide margin, i.e. genuinely solving). |
| **Partial** | `A_H` clears the floor (`> 30%`, learns global structure) but `A_S − 5 > A_H`. Mechanism works; geometry/schedule needs tuning. Report as "trains but underperforms." |
| **Inconclusive** | HFLM trains (CE decreases, no crash) but `A_H ≤ 30%`; or the ρ-bound sweep never finds a crash-free config within the 3 allotted configs. |
| **Refuted** | NaNs in CE/`log_x_theta`, training diverges, or **every** sweep config crashes on the `_LORENTZ_RHO_MAX` guard (`geo_bridge.py:81`). |

Rationale for the 5-point margin and single seed: the paper trains one model per config (cost), so we
mirror that — **1 seed per run** (see §8 risk note). The 5-point band is a pragmatic "same ballpark"
threshold, not a p-value; with 1 seed we cannot do a significance test, so we report point estimates and
flag this explicitly.

---

## 3. Runs (single comparison + minimal ρ-bound sweep)

**The ρ-bound problem (hard design constraint).** `HyperbolicHeatKernel.geodesic` lifts polar→Lorentz and
**rejects any `ρ > _LORENTZ_RHO_MAX = 20`** (`geo_bridge.py:48`, raised at `:81`). A unit-covariance
wrapped normal in `d = 512` (Sudoku width) has `ρ = ‖v‖ ≈ √d ≈ 22.6` — **over the bound** — so the
default prior crashes. Two knobs must be set so both the prior radius and the trainable embedding radius
stay well under 20:
- **`algo.prior_cov = s²`** scales the wrapped-normal tangent variance; `E[ρ] ≈ s·√d`. We want
  `s·√d ≪ 20`, target `E[ρ] ≈ 8` with tail `< 20`.
- **soft radial clamp** on embeddings: `ρ_eff = ρ_max · tanh(ρ / ρ_max)` with `algo.rho_max` (e.g. 12–15),
  plus a small init scale, so trainable `‖e_v‖` cannot drift past the bound during training.

Recommended defaults (see §7 for derivation): `prior_cov = 0.25` (⇒ `s = 0.5`, `E[ρ] ≈ 0.5·22.6 ≈ 11.3`,
99.9% tail `≈ 12.5` — under 20), `rho_max = 12`.

| ID | Name | Config delta | Purpose |
|---|---|---|---|
| **A** | `hflm-easy-cov0.25` | `algo=hflm model=tiny-hyperbolic-dit`, `prior_cov=0.25`, `rho_max=12`, linear schedule | Primary HFLM run |
| **B** | `sfm-easy-baseline` | `scripts/train/sudoku/sfm.sh` verbatim (linear) | Baseline `A_S` (re-run at our budget) |
| **C** | `hflm-easy-cov0.0625` | as A but `prior_cov=0.0625` (`s=0.25`, `E[ρ]≈5.6`) | Sweep point: smaller prior if A is ρ-unstable or under-noised |

**Conditional 4th run (only if A confirms and time allows):**

| ID | Name | Config delta | Purpose |
|---|---|---|---|
| **D** | `hflm-easy-trunc` | as A + `noise.alpha_max=0.093` (matches `sfm_truncated.sh`) | Truncation ablation — paper notes truncation is *critical for Sudoku* (§3.2, Table 1). |

Keep the sweep to **A + C** (and B baseline). Add **D** only if the primary run confirms; do not expand
into a grid. Total committed runs: **3** (A, B, C); optional **4th** (D).

---

## 4. Training recipe (exact CLI)

Baseline **B** is `scripts/train/sudoku/sfm.sh` unchanged. The HFLM runs mirror it with three swaps:
`model=tiny-hyperbolic-dit`, `algo=hflm`, and the two ρ-bound knobs. Run **A**:

```bash
python -u -m main \
    data=sudoku data.difficulty=easy data.cache_dir="${CACHE_DIR}" \
    model=tiny-hyperbolic-dit \
    algo=hflm \
    algo.invert_time_convention=false \
    algo.prior_cov=0.25 \
    algo.rho_max=12 \
    noise=log-linear \
    loader.global_batch_size=256 loader.batch_size=256 loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.devices=1 trainer.num_nodes=1 \
    trainer.val_check_interval=20_000 trainer.limit_val_batches=0 \
    trainer.max_steps=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    sampler=hflm \
    wandb.project=hflm-sudoku wandb.group=hflm-easy wandb.name=hflm-easy-cov0.25 \
    hydra.run.dir="${OUTPUT_DIR}"
```

Run **C**: identical with `algo.prior_cov=0.0625 wandb.name=hflm-easy-cov0.0625`.
Run **B**: `bash scripts/train/sudoku/sfm.sh` with `wandb.project=hflm-sudoku wandb.group=sfm-easy
wandb.name=sfm-easy-baseline` appended.

**Held fixed across all runs** (clean HFLM-vs-S-FLM comparison): backbone size (`tiny`: 8 blocks, dim 512,
8 heads, dropout 0.1, ngpt init), data + difficulty (sudoku/easy, 48k train / 2k val), global batch 256,
`max_steps=20_000`, `noise=log-linear` (for A/B/C; truncation only in optional D), `eps`, optimizer/LR
(inherited from config), `invert_time_convention=false`, 180 sampling steps.

**Variables (HFLM vs S-FLM):** geometry (`H^d` geodesic + wrapped-normal prior vs sphere SLERP + uniform
prior); embeddings (unnormalized flexible-length, **no** `renormalize_weights`, **no** sphere-calibrated
LM output) vs sphere-normalized. Within the HFLM sweep: `prior_cov` (A vs C), and truncation (optional D).

**Sampling config** — new `configs/sampler/hflm.yaml`, mirroring `configs/sampler/sfm.yaml`:
```yaml
predictor: hflm
steps: 180            # paper §4.1 Sudoku
noise_removal: greedy # argmax decode at the final step (exact-match task)
use_float64: true
velocity: exact       # marginalized exact velocity, hyperbolic analog of eq. 15
top_k_velocity: 1     # top-1 velocity (paper's strongest Sudoku/GSM8K setting)
p_nucleus: 1.0
top_k: -1
temperature: 1.0      # T=1
num_sample_batches: 2
num_sample_log: 2
```
(Architect: if the marginalized hyperbolic velocity is not ready, fall back to `velocity: exact`,
`top_k_velocity: -1`; record which was used in W&B config.)

---

## 5. Compute budget

- **Hardware:** single GPU (cluster `gpu`/`gpu-interactive` partition; `sinfo` shows idle nodes incl.
  `sablab-gpu-[01-06]`, `dgx2-compute-*`). 1× modern GPU (≥24 GB) is sufficient — d=512, batch 256,
  L=180, 8-layer DiT.
- **Per run:** 20k steps. S-FLM at this size/batch runs ~3–5 it/s on one GPU ⇒ **~1.5–2.5 h** training.
  HFLM adds per-step cost from `wrapped_normal` + `geodesic` (CPU/GPU radial sampling); budget a **1.3–1.7×**
  factor ⇒ **~2.5–4 h** per HFLM run. Sudoku eval (2k puzzles × 180 steps) adds ~15–30 min.
- **Total:** A + B + C = **~8–12 GPU-hours**; optional D adds ~3–4 h. Well within a single-GPU day.
  `TBD: confirm HFLM step time — if the H^d radial sampler dominates (it builds a 2000-pt grid per call,
  geo_bridge.py:862), cache or vectorize before scaling beyond Sudoku.`

---

## 6. W&B project / run naming convention

Default config has `wandb.project=debug` (`configs/config.yaml:93`). Override for this experiment:

- **project:** `hflm-sudoku`
- **group:** `hflm-easy` (runs A, C, D) / `sfm-easy` (run B)
- **name (exact strings):** `hflm-easy-cov0.25` (A), `sfm-easy-baseline` (B), `hflm-easy-cov0.0625` (C),
  `hflm-easy-trunc` (D)
- **tags:** auto-templated as `[${noise.type}, ${data.train}, ${data.valid}, ${algo.name}]`
  (`configs/config.yaml:98`); add `sudoku`, `easy`, and `cov<value>` manually.

**Metrics to log (W&B keys):**
- *Primary:* `sudoku/exact_match_acc` (%) — currently only printed (`main.py:582`) and written to
  `results.json` (`:587`). **Action required:** also log the `accuracy` scalar to W&B at eval time
  (see §7). Without this the primary metric is not in W&B.
- *Secondary:* `trainer/loss` / `train/ce_loss` (CE, eq. 14 — should match S-FLM's loss path exactly),
  `val/ce_loss`.
- *Diagnostic (the ρ-bound watch — HFLM-specific, must add):*
  - `hflm/rho_prior_mean`, `hflm/rho_prior_max` — wrapped-normal radius per batch (watch vs 20).
  - `hflm/rho_embed_mean`, `hflm/rho_embed_max` — trainable embedding radius `‖e_v‖` (watch the clamp).
  - `hflm/rho_zt_max` — radius of the interpolated latent `z_t` fed to `geodesic`.
  - `hflm/nan_count` or rely on `utils.print_nans` (`algo.py:307/428`) — any NaN ⇒ refuted.

---

## 7. TBDs and required minimal changes

**Required code changes for feasibility (flagged for architect — NOT implemented here):**
1. **Dispatch wiring (missing today):** `model.type=hyperbolic-dit` is absent from the backbone dispatch
   `trainer_base.py:69-85`; `algo.name=hflm` is absent from `main.py:623-637`; `predictor=hflm` is absent
   from `samplers.get_sampler` (`samplers.py:1162-1197`). All three must be added, plus a
   `configs/model/tiny-hyperbolic-dit.yaml` and `configs/algo/hflm.yaml` (clones of the sphere configs
   with `type: hyperbolic-dit`, `backbone: hyperbolic-dit`, `renormalize_weights: False`, new keys
   `prior_cov`, `rho_max`), and `configs/sampler/hflm.yaml` (§4).
2. **`algo.HFLM` is broken** (`algo.py:317-436`): undefined `e_clean`, `wrapped_normal` not imported,
   `_hyeprbolic_geodesic` references undefined `clean`/`out` and has an unfinished `geodesic(...)` call.
   Must be completed against `geo_bridge.HyperbolicHeatKernel.geodesic` / `GeoUtils.wrapped_normal`.
3. **`HFLMSampler` is a verbatim sphere clone** (`samplers.py:727-851`): `init_state` still
   `sphere_normalize`s the prior and `step` pulls `sphere_embed` / `sphere_normalize`d `E`. Must use the
   wrapped-normal prior and unnormalized hyperbolic embeddings, with the hyperbolic marginalized velocity.
4. **Primary metric not in W&B:** the Sudoku `accuracy` scalar (`main.py:587`) is only written to JSON.
   Add a one-line `wandb_logger.log_metrics({'sudoku/exact_match_acc': accuracy*100})` (or equivalent) so
   the success criterion is queryable. Minimal change, but load-bearing for this experiment.

**Resolved-with-recommendation TBDs:**
- `TBD: exact prior_cov default.` **Recommend `prior_cov = 0.25`** (`s=0.5`). With `d=512`,
  `E[ρ] = s·E[χ_d] ≈ s·√d ≈ 11.3`; a wrapped-normal radius is tightly concentrated (`σ_ρ ≈ s/√2 ≈ 0.35`),
  so the max over a 256×180 batch sits near `11.3 + ~5σ ≈ 13`, comfortably `< 20`. `prior_cov=0.0625`
  (run C) gives `E[ρ] ≈ 5.6` as a safety fallback. **Do NOT use the `wrapped_normal` default `cov=1.0`** —
  it crashes the geodesic at d=512.
- `TBD: rho_max for the soft clamp.` **Recommend `rho_max = 12`** (`ρ_eff = 12·tanh(ρ/12)`); caps the
  trainable embedding radius below 20 with headroom while leaving range for length-as-radial structure.
- `TBD: embedding init scale.` **Recommend** init `‖e_v‖ ≈ E[ρ_prior]` (i.e. ngpt-style `std=1/√d` gives
  `‖e_v‖≈1`; scale up to ~`s·√d`) so clean and prior radii are comparable at `t≈0`. Architect to confirm
  against the geodesic's behavior; flag if embeddings collapse to the origin (CE should prevent this,
  paper §3 "Training with Cross-Entropy").
- `TBD: hyperbolic marginalized velocity availability.` If the exact hyperbolic velocity (analog of
  eq. 15 via `log_{z_t}` on `H^d`) is not yet implemented in the sampler, the run still proceeds with the
  geodesic-based integrator; record the actual velocity variant in W&B config.
- `TBD: confirm S-FLM-easy baseline number at 20k steps` (paper trains longer / reports linear=81.5 at
  its own budget). Run B establishes the in-experiment `A_S`; treat the paper's 81.5/94.x as context only.

---

## 8. Failure modes to watch for

- **ρ-bound crash** (`ValueError: Lorentz-Cartesian output requires rho <= 20`, `geo_bridge.py:81`): the
  central risk. Mitigated by `prior_cov`/`rho_max` (§3/§7). If A and C both crash → refuted; report the
  observed `hflm/rho_*` maxima.
- **NaNs** in `log_x_theta` (`utils.print_nans`, `algo.py:307/428`) from `acosh`/`sinh` at large ρ or
  near-antipodal geodesics → refuted; reduce `prior_cov`.
- **Embedding collapse** (all `e_v` → origin / same direction): CE should prevent it (paper §3); watch
  `hflm/rho_embed_mean` not → 0 and accuracy not stuck at chance.
- **Eval contamination / leakage:** prompt (puzzle) is kept clean and only the solution region is scored
  (`main.py:549-551`, `gt = batch[:, prompt_len:]`); train/val are the disjoint 48k/2k split
  (`sudoku_generator.py`). No extra leakage risk introduced — but verify HFLM's `valid_tokens` masking in
  `q_xt` keeps givens clean (it currently has a bug: `e_clean` is undefined in the masked branch,
  `algo.py:360-362`).
- **Single seed:** like the paper, we run 1 seed/config (cost). Results are point estimates; **no
  significance test** — the §2 margin is a heuristic band, stated as a limitation. If A lands within ±2pts
  of the Partial/Confirmed boundary, run 1 extra seed of A before concluding.
- **Backbone drift:** any change to `tiny-hyperbolic-dit` beyond the three essential deltas (unnormalized
  flexible-length embeddings, no `renormalize_weights`, no sphere-calibrated LM output) invalidates the
  clean HFLM-vs-S-FLM comparison. Keep the DiT body byte-identical to `SphereDiT`.

---

### Artifacts
- This file: `/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm/experiments/hflm/EXPERIMENT.md`
- Code anchors: `algo.py:317-436` (HFLM, broken), `samplers.py:727-851` (HFLMSampler, sphere clone),
  `models/hyperbolic_dit.py:180` (`get_hyperbolic_polar_embeddings`), `geo_bridge.py:48,81,733,843,906`
  (ρ-bound, geodesic, `HyperbolicHeatKernel`, `sample_radial`), `main.py:551,587,623-637`,
  `trainer_base.py:69-85`. Reference (working): `SFM` (`algo.py:211-315`), `SFMSampler`
  (`samplers.py:597-721`), `configs/algo/sfm.yaml`, `configs/model/tiny-sphere-dit.yaml`,
  `scripts/train/sudoku/sfm.sh`.
