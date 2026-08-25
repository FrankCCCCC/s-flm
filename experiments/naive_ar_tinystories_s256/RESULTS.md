# naive_ar_tinystories_s256 — Results

**36 / 36 cells complete, zero failures.** 4 methods × LR {3e-4, 1e-3, 5e-3} × seed {1,2,3},
TinyStories seq-256, 30k steps, global batch 512, bf16, EMA 0.9999, AdamW (wd 0, betas
(0.9,0.999), eps 1e-8, grad-clip 1.0), constant schedule + 2500-step warmup. Eval: 180 sampling
steps for mdlm/duo/flm, 64 samples/cell, gpt2-large retokenized GenPPL.

> **NOTE — this file was written by hand, not by `experiments/report.py`.** Re-running
> `python experiments/report.py naive_ar_tinystories_s256` will overwrite it with a table that
> ranks the `Valid PPL` column across all four methods. That ranking is invalid (see
> §"Metric validity") and would declare FLM the winner of the experiment. Fix `report.py:73`
> before regenerating.

---

## 1. Aggregate results (mean ± seed sd, n=3 per cell)

| method | lr | val/ppl | ±sd | GenPPL | ±sd | entropy | uniq/64 |
|---|---|---|---|---|---|---|---|
| ar   | 3e-4 | 3.3556 | 0.0009 | 7.28 | 0.48 | 4.277 | **1** |
| ar   | **1e-3** | 3.2869 | 0.0036 | 7.23 | 0.30 | 4.256 | **1** |
| ar   | 5e-3 | 3.2764 | 0.0107 | 7.88 | 0.63 | 4.237 | **1** |
| mdlm | 3e-4 | 4.9748 | 0.0089 | 19.02 | 0.78 | 4.408 | 64 |
| mdlm | **1e-3** | **4.8652** | 0.0288 | 18.76 | 0.38 | 4.392 | 64 |
| mdlm | 5e-3 | 16.8962 | 16.5202 | 107.24 | 124.11 | 4.446 | 64 |
| duo  | 3e-4 | 5.2678 | 0.0082 | 18.50 | 0.69 | 4.322 | 64 |
| duo  | **1e-3** | 5.1275 | 0.0071 | 18.46 | 0.36 | 4.326 | 64 |
| duo  | 5e-3 | 5.3993 | 0.0407 | **17.41** | 0.18 | 4.343 | 64 |
| flm  | 3e-4 | 1.5302 | 0.0045 | 60.28 | 2.01 | 4.588 | 64 |
| flm  | **1e-3** | 1.5168 | 0.0070 | 51.55 | 3.28 | 4.562 | 64 |
| flm  | 5e-3 | 1.6252 | 0.1017 | 49.20 | 4.25 | 4.548 | 64 |

`uniq/64` = distinct strings among the 64 generated samples.

---

## 2. Metric validity — READ BEFORE RANKING ANYTHING

**`val/ppl` mixes three different estimands and MUST NOT be ranked across methods.**

| method | what `val/ppl` actually is | rankable against |
|---|---|---|
| ar   | **exact** autoregressive NLL (`configs/algo/ar.yaml`) | nothing else here |
| mdlm | denoising-**ELBO upper bound** (`loss_type: elbo`) | duo only |
| duo  | denoising-**ELBO upper bound** (`loss_type: elbo`) | mdlm only |
| flm  | **unweighted denoising CE** (`algo.py:1166`) — neither a likelihood nor a bound | nothing |

FLM's 1.52 is the lowest number in the table and means nothing: a PPL of 1.5 on TinyStories is
unachievable by any language model. Near-clean interpolants contribute CE ≈ 0 and drag the mean
toward zero. **Only mdlm ↔ duo is a legitimate likelihood comparison.**

**`entropy` does not detect AR's collapse.** `metrics.py` computes unigram entropy *per sample*
then averages, so it is blind to all 64 samples being identical. AR scores 4.24–4.28 and passes
the `entropy >= 3.0` gate in `report.py:71` while having **one effective sample**. `uniq/N` is the
column that catches this and is not currently recorded in `samples_genppl.json`.

---

## 3. Findings

### 3.1 AR's GenPPL is a single greedy decode and is not comparable to the other three
`scripts/sample/tinystories/ar.sh:13` defaults `GREEDY=true`, overriding `configs/sampler/ar.yaml:7`
(`greedy: False`). `ARSampler` has no other randomness, so **all 64 samples are byte-identical
(`uniq=1`) in all 9 AR cells**. Consequences:
- AR's GenPPL is a mode decode, which minimises evaluator surprisal by construction.
- With n_eff = 1 it is extremely noisy: seed sd 0.30–0.63 (CV 4–8%) versus 0.027% for its val/ppl
  — **~243× noisier in relative terms**. It cannot resolve even the LR difference.
- A prior non-greedy run at p=0.9 (`outputs/.../ar/eval_fixed_nucleus0.9`) gives **7.367 with 64/64
  unique**, inside the greedy cells' noise band — so AR's GenPPL is genuinely ≈7.3, but the
  untruncated (p=1.0) value has never been validly measured.

**Action:** re-run AR eval with `GREEDY=false P_NUCLEUS=1.0` (~1 GPU-hr, existing checkpoints).

### 3.2 MDLM beats DUO — the one fully-controlled cross-method result
Same estimand, same 180 NFE, same LR, same seeds, n=3 each:

| lr | MDLM | DUO | MDLM advantage |
|---|---|---|---|
| 3e-4 | 4.9748 | 5.2678 | **5.9%** |
| 1e-3 | 4.8652 | 5.1275 | **5.4%** |

Seed distributions do not overlap at either LR. Masking beats uniform-state on TinyStories at this
budget. Caveat: DUO carries `sampler.noise_removal=greedy` (per `setup.md`) while MDLM inherits
`ancestral` — a small decode advantage **to DUO**, which still loses.

### 3.3 Generation quality: DUO ≈ MDLM ≫ FLM
At matched NFE=180, full sampling diversity, comparable entropy:

| method | GenPPL (best LR) | entropy |
|---|---|---|
| duo  | **17.41** (5e-3) | 4.343 |
| mdlm | 18.76 (1e-3) | 4.392 |
| flm  | 49.20 (5e-3) | 4.548 |

FLM is **2.6–2.8× worse** than both discrete baselines. This is not a diversity artifact — FLM has
the *highest* entropy of all four methods, so the gap would widen at matched entropy. Its samples
are locally fluent but semantically incoherent.

### 3.4 lr 1e-3 is the correct shared learning rate for all four methods
`setup.md`'s 3e-4 leaves ~2% on the table for every method:

| method | 3e-4 → 1e-3, val/ppl | 3e-4 → 1e-3, GenPPL |
|---|---|---|
| ar   | −2.05% | noise |
| mdlm | −2.20% | −1.4% |
| duo  | −2.66% | −0.2% |
| flm  | −0.9% | **−14.5%** |

Consistency across three architecturally distinct families indicates a schedule effect, not a
method-specific one: at 30k steps with a **constant LR and no decay phase**, 3e-4 is too
conservative. FLM is the exception that proves the metric point — its likelihood proxy barely
registers a 14.5% generative improvement (§2).

### 3.5 MDLM has a stability cliff at 5e-3; DUO and FLM do not
| method @ 5e-3 | diverged | surviving val/ppl |
|---|---|---|
| **mdlm** | **1 / 3** (40.2592, GenPPL 282.8) | 5.203, 5.227 |
| duo  | 0 / 3 | 5.343–5.437 |
| flm  | 0 / 3 | 1.551–1.769 |

MDLM's seed sd also grows with LR (0.0089 → 0.0288 → bimodal) while DUO's stays flat
(0.0082 → 0.0071 → 0.0407). Mechanism: MDLM's ELBO weight is exactly −1/t over t ∈ [1e-3, 1]
(`algo.py:128`, `noise_schedules.py:61-65`) — a **1000× dynamic range** whose relative variance is
4× larger at L=256 than at the papers' L=1024. DUO's uniform-state loss has no such spike.

Even discarding the diverged seed, 5e-3 is worse than 1e-3 for MDLM (5.215 vs 4.865), so it is
dominated regardless of stability.

### 3.6 GenPPL is anti-correlated with model quality within a method
Across the 15 S-FLM cells in `adv_geo_tinystories_s256`, **Spearman(val/ppl, GenPPL) = −0.604**:
worse density models score better GenPPL. Visible here too — DUO's best GenPPL (17.41) comes at
its *worst* val/ppl (5.3993). GenPPL at a single fixed temperature measures decode sharpness as
much as model quality, which is why the S-FLM paper compares temperature-swept **frontiers** at
matched entropy, never single points.

---

## 4. Conclusions

1. **AR ≫ diffusion on this budget**, but the margin is not measurable from these runs because AR's
   decode is greedy. In nats, the AR↔MDLM gap is 0.389 — close to MDLM's published converged OWT
   gap of 0.280 at 1/133 the compute, so the diffusion models are **not** underperforming.
2. **MDLM > DUO** by 5.4–5.9% on the likelihood bound, and they are tied on GenPPL (17.4–19.0).
   DUO's published GenPPL advantage over MDLM (paper ratio 0.741) is **absent here (0.98)** —
   consistent with `algo=duo-base` omitting the curriculum learning of the DUO paper's Sec. 4.1
   (`grep -ri curriculum` → 0 hits; no `configs/algo/duo.yaml`).
3. **FLM is the weakest generator by 2.6×** despite the most favourable entropy.
4. **Use lr 1e-3** for all future runs on this setup.
5. **More steps will not close the AR↔diffusion gap** — the MDLM−AR training-loss gap is flat at
   ~1.02 nats from 5k→30k (5–10k: 1.0523, 25–30k: 1.0218), matching Nie et al. (arXiv:2410.18514):
   masked diffusion needs ~16× AR's compute, a roughly constant offset.

## 5. Recommended next steps

| priority | action | cost |
|---|---|---|
| 1 | Re-eval AR with `GREEDY=false P_NUCLEUS=1.0` — makes AR's column rankable for the first time | ~1 GPU-hr |
| 2 | Fix `report.py:73` (drop/scope "Lowest Valid PPL") and record `uniq/N` in `samples_genppl.json` | ~10 lines |
| 3 | Switch to `lr_scheduler=cosine_decay_warmup` — currently no decay phase at all | 10 hr/cell |
| 4 | Port DUO curriculum learning (paper Sec. 4.1) — the only intervention with a published, budget-matched effect size (3.2–3.9× gradient-variance cut at 10–20k steps) | 1–2 days + 11 hr |

## 6. Per-cell results

| method | lr | seed | val/ppl | val/nll | GenPPL | entropy | uniq/64 | NFE |
|---|---|---|---|---|---|---|---|---|
| ar | 3e-4 | 1 | 3.3566 | 1.2109 | 6.62 | 4.221 | 1 | 255 |
| ar | 3e-4 | 2 | 3.3557 | 1.2107 | 7.52 | 4.286 | 1 | 255 |
| ar | 3e-4 | 3 | 3.3544 | 1.2103 | 7.72 | 4.325 | 1 | 255 |
| ar | 1e-3 | 1 | 3.2914 | 1.1913 | 6.88 | 4.233 | 1 | 255 |
| ar | 1e-3 | 2 | 3.2826 | 1.1886 | 7.62 | 4.343 | 1 | 255 |
| ar | 1e-3 | 3 | 3.2866 | 1.1899 | 7.20 | 4.191 | 1 | 255 |
| ar | 5e-3 | 1 | 3.2892 | 1.1907 | 7.84 | 4.266 | 1 | 255 |
| ar | 5e-3 | 2 | 3.2630 | 1.1826 | 7.14 | 4.243 | 1 | 255 |
| ar | 5e-3 | 3 | 3.2769 | 1.1869 | 8.68 | 4.202 | 1 | 255 |
| mdlm | 3e-4 | 1 | 4.9622 | 1.6018 | 19.93 | 4.413 | 64 | 180 |
| mdlm | 3e-4 | 2 | 4.9808 | 1.6056 | 18.02 | 4.395 | 64 | 180 |
| mdlm | 3e-4 | 3 | 4.9814 | 1.6057 | 19.10 | 4.415 | 64 | 180 |
| mdlm | 1e-3 | 1 | 4.8279 | 1.5744 | 18.96 | 4.383 | 64 | 180 |
| mdlm | 1e-3 | 2 | 4.8981 | 1.5889 | 19.09 | 4.392 | 64 | 180 |
| mdlm | 1e-3 | 3 | 4.8696 | 1.5830 | 18.23 | 4.400 | 64 | 180 |
| mdlm | 5e-3 | 1 | **40.2592** | 3.6953 | **282.77** | 4.527 | 64 | 180 |
| mdlm | 5e-3 | 2 | 5.2270 | 1.6538 | 19.74 | 4.411 | 64 | 180 |
| mdlm | 5e-3 | 3 | 5.2024 | 1.6491 | 19.22 | 4.399 | 64 | 180 |
| duo | 3e-4 | 1 | 5.2793 | 1.6638 | 19.45 | 4.315 | 64 | 180 |
| duo | 3e-4 | 2 | 5.2611 | 1.6603 | 17.87 | 4.329 | 64 | 180 |
| duo | 3e-4 | 3 | 5.2631 | 1.6607 | 18.16 | 4.322 | 64 | 180 |
| duo | 1e-3 | 1 | 5.1373 | 1.6365 | 18.42 | 4.336 | 64 | 180 |
| duo | 1e-3 | 2 | 5.1240 | 1.6339 | 18.92 | 4.315 | 64 | 180 |
| duo | 1e-3 | 3 | 5.1211 | 1.6334 | 18.03 | 4.326 | 64 | 180 |
| duo | 5e-3 | 1 | 5.4185 | 1.6898 | 17.42 | 4.330 | 64 | 180 |
| duo | 5e-3 | 2 | 5.4367 | 1.6932 | 17.19 | 4.355 | 64 | 180 |
| duo | 5e-3 | 3 | 5.3427 | 1.6757 | 17.63 | 4.343 | 64 | 180 |
| flm | 3e-4 | 1 | 1.5298 | 0.4252 | 62.80 | 4.589 | 64 | 180 |
| flm | 3e-4 | 2 | 1.5249 | 0.4219 | 60.15 | 4.590 | 64 | 180 |
| flm | 3e-4 | 3 | 1.5359 | 0.4291 | 57.89 | 4.586 | 64 | 180 |
| flm | 1e-3 | 1 | 1.5121 | 0.4135 | 56.15 | 4.582 | 64 | 180 |
| flm | 1e-3 | 2 | 1.5267 | 0.4231 | 49.76 | 4.554 | 64 | 180 |
| flm | 1e-3 | 3 | 1.5115 | 0.4131 | 48.75 | 4.550 | 64 | 180 |
| flm | 5e-3 | 1 | 1.7689 | 0.5704 | 43.39 | 4.517 | 64 | 180 |
| flm | 5e-3 | 2 | 1.5557 | 0.4419 | 50.80 | 4.553 | 64 | 180 |
| flm | 5e-3 | 3 | 1.5509 | 0.4388 | 53.42 | 4.573 | 64 | 180 |
