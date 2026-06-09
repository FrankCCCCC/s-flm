# HFLM vs S-FLM on OpenWebText — Gen-PPL / Entropy Frontier (IN PROGRESS)

**Status: running.** This is a live document; numbers update as checkpoints accrue. OWT runs are
multi-day at our scale, so absolute Gen-PPL will NOT match the paper's 1M-step Fig-4 numbers — the
goal here is the **matched-checkpoint HFLM-vs-S-FLM comparison** (does hyperbolic geometry help vs
spherical, all else equal).

## Setup
- **Data:** OpenWebText (`jdeschena/openwebtext`), GPT-2 tokenizer, context length 1024. Cache built
  locally (train 8.73M / valid 110k 1024-blocks).
- **Model (both):** standard DiT, hidden 768, 12 layers, 12 heads (§4.3). batch 512, AdamW 3e-4,
  β(0.9,0.999), no weight decay, EMA 0.9999, bf16 (geodesic float64). 100k-step cap (reduced from
  the paper's 1M — infeasible here), checkpoints every 5k.
- **Three runs:**
  - `S-FLM (paper)` — sphere DiT, init ngpt, **noise=log-linear-adaptive + truncation α=0.121** (the
    repo/paper OWT recipe). 4×RTX-6000-Ada (thickstun).
  - `HFLM (plain)` — hyperbolic DiT, prior_cov=0.25, rho_max=12, **noise=log-linear** (NO truncation;
    sphere truncation collapses HFLM — see Sudoku). 4×A6000 (desa).
  - `S-FLM (plain)` — sphere DiT, **noise=log-linear** (matched-schedule control vs HFLM, to isolate
    *geometry* from *schedule*). 4×RTX-6000-Ada (thickstun). [running]
- **Gen-PPL / entropy frontier (Suppl. C.8):** sample at NFE∈{32,1024}, sweep T∈{0.70…1.20}, compute
  Gen-PPL (gpt2-large, retokenize, first-chunk) + avg unigram entropy; plot Gen-PPL(↓) vs entropy(↑).
  Each point = 64 samples (first pass; will increase).

## Trajectory — NFE=32, GenPPL / entropy (PRELIMINARY, undertrained)

| Model | step | T=0.8 | T=1.0 | T=1.2 |
|---|---|---|---|---|
| **S-FLM (paper)** | 5k  | 27.4 / 1.73 | 26.8 / 1.80 | 27.2 / 1.75 |
| **S-FLM (paper)** | 10k | 19.6 / 1.26 | 18.4 / 1.41 | (…) |
| **HFLM (plain)**  | 5k  | 47.5 / 3.91 | 45.6 / 3.92 | 46.8 / 3.92 |
| **HFLM (plain)**  | 10k | 32.9 / 3.89 | 33.0 / 3.91 | 32.7 / 3.88 |

**Emerging picture (5k→10k), grounded in the actual generations (NOT just the entropy scalar):**
- **HFLM(plain)**: entropy ≈ 3.9 (stable), Gen-PPL drops **46 → 33**. The samples are **varied
  real-word English fragments** (names, words, sentence pieces) — *but still incoherent / disfluent*
  at this budget (word salad). So: **non-degenerate and improving, NOT "good text" yet.** (Earlier I
  called this "healthy diversity" — overstated; corrected here.)
- **S-FLM(paper)**: entropy *falls* 1.75 → ~1.3; samples literally degenerate to a **soup of commas
  and "the"** (e.g. `the the,,,,, the,, of,, the,,,`). Its low Gen-PPL (~19) is the repetition
  artifact the paper warns of (§4.3), not quality.

### ★ Geometry vs schedule — matched 5k (both plain log-linear, only geometry differs)
The above confounds geometry with schedule (S-FLM=truncated, HFLM=plain). Controlling for it:

| 5k, T=1.0 | Gen-PPL | entropy | ~%unique | top-token | sample (first ~120 chars) |
|---|---|---|---|---|---|
| **HFLM (plain)** | 45.6 | **3.92** | 47% | "the" 9% | `it's not do. Yes, I said… a one's happening here. The Kickstartering… the game and t…` |
| **S-FLM (plain)** | 30.4 | **2.15** | 35% | "the" **28%** | `, the the,,, now. the year of the. in the… The the the the, the…` |
| S-FLM (paper, trunc) | 26.8 | 1.80 | 40% | "," 20% | `, the the,,, the,, of,, the crown, and and the…` |

**Clean decomposition (matched plain schedule, entropy at T=1.0, NFE=32):**

| step | HFLM (plain, hyperbolic) | S-FLM (plain, sphere) | S-FLM (paper, sphere+trunc) |
|---|---|---|---|
| 5k  | **3.92** | 2.15 | 1.80 |
| 10k | **3.91** | 2.19 | 1.41 |
| 20k | (≈3.76) | (pending) | 0.95 |

Two separable effects:
- **GEOMETRY (the answer to your question):** at matched plain schedule, **hyperbolic gives ≈1.8× the
  sample entropy of spherical** (3.9 vs 2.15) — substantially less repetition, consistently across 5k/10k.
- **SCHEDULE:** S-FLM(plain) is *stable* at ~2.15 — it does **not** collapse. The catastrophic comma-soup
  collapse (→0.95) is the **paper's truncated+adaptive schedule**, which is built for 1M steps and is
  degenerate at our ≪1M budget — *not* the sphere geometry per se.

**Caveat (unchanged):** 5–20k steps — "less repetitive" ≠ "more coherent." HFLM's varied text is still
disfluent word-salad. Whether the geometry advantage converts to better *coherent* generation (Gen-PPL
at matched entropy ≈5–5.5) needs ~1M steps, which is infeasible at our compute.

**Read carefully — this is NOT "S-FLM wins."** The two
sit in *different entropy regimes*:
- **S-FLM (paper) entropy ≈ 1.75** → near-degenerate, **repetitive** generations. Repetition
  artificially *lowers* Gen-PPL (the paper explicitly warns Gen-PPL rewards repetition, §4.3), so its
  "27" is a low-diversity artifact, not quality. Temperature barely moves it (collapsed).
- **HFLM (plain) entropy ≈ 3.92** → **~2× more diverse** text already at 5k, at a higher (worse-looking)
  Gen-PPL of ~46.

The paper's frontier compares Gen-PPL **at matched entropy** (their curves span entropy ≈ 5.0–5.8).
Neither model is there yet at 5k, and they're not at the same entropy, so a frontier verdict is
premature. The one robust early signal: **HFLM reaches much higher sample diversity far sooner**.

### Confound flagged
The 5k gap conflates **geometry** (hyperbolic vs sphere) with **schedule** (HFLM plain vs S-FLM
adaptive+truncation). Truncation concentrates training on high-noise levels and is designed for long
runs; at 5k it may be *why* S-FLM looks collapsed. The **`S-FLM (plain)` control** (matched plain
log-linear, now running) isolates geometry: HFLM(plain) vs S-FLM(plain) is the clean test.

## 20k steps, NFE=32 (temperature sweep) — the trend sharpens

| step | HFLM(plain) entropy | S-FLM(paper) entropy |
|---|---|---|
| 5k  | 3.91 | 1.75 |
| 10k | 3.89 | 1.30 |
| 20k | **3.76** (stable) | **0.77** (collapsed) |

At 20k, across **T∈{0.7…1.2}**: HFLM(plain) sits at PPL ≈30 / H ≈3.76 (flat in T); S-FLM(paper) at
PPL ≈16 / H ≈0.8 (flat in T). Two observations:
1. **S-FLM(paper) mode-collapses, and it worsens with training** (H 1.75→1.3→0.77). At 20k its
   generations are *almost pure commas*: `,,,,,, B,,,,,,, —,,,,, double,,,,,, Alan,,,,,,…`. The low
   Gen-PPL (~16) is 100% a repetition artifact.
2. **Temperature has ~no effect on either** at this budget — both are "locked" (S-FLM into repetition,
   HFLM into varied-but-incoherent). So the Gen-PPL/entropy *frontier* is **two separated clusters,
   not the paper's smooth curves** — those curves only appear once models are coherent (≫ our steps).

**What this robustly shows (and what it doesn't):**
- ✅ **Hyperbolic geometry strongly resists the repetition / mode-collapse failure** that afflicts
  spherical S-FLM at matched budget — even the matched-schedule control (S-FLM-plain, H≈2.1 at 5k)
  is far more repetitive than HFLM-plain (H≈3.9). The paper's truncated schedule makes S-FLM collapse
  *harder/faster*.
- ❌ **We CANNOT yet conclude HFLM generates better *language*.** Both are far from coherent text
  (H≈3.8 vs natural ≈5.5; HFLM is varied word-salad). Reaching the paper's coherent-text frontier
  needs ~1M steps (≈50× our budget) — infeasible at our compute. So the honest verdict is about
  **robustness-to-collapse**, not final quality.

**NFE=1024 (full sampling quality) at 20k confirms the same picture:**

| 20k, NFE=1024 | T=0.8 | T=1.0 | T=1.2 |
|---|---|---|---|
| **HFLM (plain)** | PPL 25 / H 3.90 | 25 / 3.91 | 25 / 3.90 |
| **S-FLM (paper)** | PPL 13 / H 1.10 | 13 / 1.16 | 13 / 1.14 |

More sampling steps (1024 vs 32) lower both models' Gen-PPL (HFLM 30→25, S-FLM 16→13) but do **not**
change the collapse: HFLM stays diverse (H≈3.9), S-FLM stays degenerate (H≈1.1). S-FLM's lower PPL is
still the repetition artifact. Figure: `outputs/owt_frontier_20k.png` (entropy-vs-step + 20k scatter).

## Plan (updating this doc as it runs)
1. Repeat the frontier at **25k / 50k / 100k** checkpoints (entropy should climb toward the paper's
   5–5.8 range, making the frontier comparison meaningful).
2. Add **NFE=1024** and the **full T∈{0.70…1.20}** grid at the higher checkpoints.
3. Add the **S-FLM(plain)** curve once it trains, for the geometry-isolating comparison.
4. Increase samples/point (64 → 256+) for stable Gen-PPL once the headline checkpoint is chosen.

## Caveats
- 5k steps ≪ paper's 1M; 64 samples/point; single seed. Preliminary.
- HFLM sampler = top-1 geodesic step; S-FLM = top-1 velocity (`top_k_velocity=1`), matched.
- Absolute Gen-PPL not comparable to the paper (200× fewer steps). Relative/at-matched-entropy is the valid read.

## Reproduction
- Train: `sbatch --export=ALL,MAX_STEPS=100000,DEVICES=4,RUN=hflm scripts/owt_train_hflm.sbatch`
  (HFLM); `scripts/owt_train_sfm.sbatch` with `NOISE_MODE=paper|plain` (S-FLM).
- Eval point: `sbatch --export=ALL,MODEL_TYPE=hflm,CKPT=<ckpt>,OUTDIR=<dir>,STEP_TAG=5k scripts/owt_frontier_eval.sbatch`
  (sweeps T∈{0.8,1.0,1.2}, NFE=32 by default; set `TEMPS`/`NFES` to extend).
- Infra: `SLURM_JOB_NAME=bash` (native multi-GPU launcher) + `NCCL_P2P_DISABLE=1`; A6000 (desa) / 6000-Ada (thickstun).
