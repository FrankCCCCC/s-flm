# geo_curv_sc_tinystories_256 — Experiment Design

TinyStories (seq 256) study of two axes from `setup.md`:
**(A) geometry × self-conditioning** for S-FLM / E-FLM, and
**(B) Gaussian curvature × embedding-init × prior-noise** for H-FLM.

## Hypotheses

- **H1 (truncation, Part A).** Restricting the flow's signal range to the paper's
  Eq. 17 bound (`alpha_star_*`) removes the uninformative high-noise regime and
  improves the flow-bound PPL / GenPPL over the naive full-range schedule; the
  adaptive schedule concentrates sampling where `|dL/dt|` is largest and stacks
  further with truncation.
- **H2 (self-conditioning, Part A).** LangFlow-style self-conditioning helps the
  Euclidean (E-FLM) flow more than the naive baselines; measured On vs Off.
- **H3 (curvature, Part B).** For H-FLM there is an interior-optimal Gaussian
  curvature `K<0`: too flat (`K→0⁻`) ≈ Euclidean, too curved over-compresses the
  radial coordinate. The optimum co-varies with the embedding-init / prior-noise
  scale (radial-coordinate match, cf. `hflm-ngpt-init-mismatch`).

## Grid — 58 cells

**Part A (13).** small-sphere-dit 768/12/12, init=ngpt. Variant → script + `ALPHA_MAX`:

| geom | variant | train/sample script | ALPHA_MAX | self-cond |
|---|---|---|---|---|
| S-FLM | naive | `sfm.sh` | – | On, **Off** |
| S-FLM | ada | `sfm_truncated_adaptive.sh` | null | On |
| S-FLM | trunc | `sfm_truncated.sh` | **0.121** | On |
| S-FLM | ada+trunc | `sfm_truncated_adaptive.sh` | **0.121** | On |
| E-FLM | naive | `eflm.sh` | – | On, Off |
| E-FLM | ada | `eflm_truncated_adaptive.sh` | null | On, Off |
| E-FLM | trunc | `eflm_truncated.sh` | **0.840** | On, Off |
| E-FLM | ada+trunc | `eflm_truncated_adaptive.sh` | **0.840** | On, Off |

`ALPHA_MAX` = paper Eq. 17 truncation bound, computed from `noise_schedules.py`
(V=50257, dim=768, δ=0.1): `alpha_star_sphere → 0.12151` (S-FLM),
`alpha_star_euclidean → 0.84019` (E-FLM). Passed identically to train + eval.
(S-FLM: all 4 variants SC-On, plus the naive baseline also run SC-Off — setup.md L10.)

**Part B (45).** small-hyperbolic-dit 768/12/12, naive `hlfm.sh`, SC-On, rho_max=12:

- init: `random` (std 0.02) · `custom 0.01` · `custom 0.04`  (3)
- prior_cov (init diffusion noise): 0.5 · 0.8 · 1.0  (3)
- Gaussian curvature K: −0.01 · −0.1 · −0.25 · −0.5 · −0.75  (5)

> **RISK.** Part B inits (‖e‖≈0.3–1.1) are smaller than the prior-noise radius
> (s≈0.7–1.0) → the radial coordinate may collapse (cf. `hflm-ngpt-init-mismatch`).
> Running as specified; a collapse is reported as a finding, not silently fixed.

## Training / eval recipe (all cells)

30k steps · global batch 512 (1 GPU × `PER_GPU_BS=32`, accum auto=16) · seq 256 ·
bf16 · EMA 0.9999 · AdamW lr 3e-4, wd 0, betas (0.9, 0.999), grad-clip 1.0 · CE loss.
Checkpoint every 5k (save_top_k=1 + last), **deleted after eval** (quota).
Eval: `ppl_eval` → `eval/ppl.json` (denoising-CE flow bound: val/nll, ppl, bpd) and
`sample_eval` → `eval/samples_genppl.json` (gpt2-large GenPPL + sample entropy).
Sampler: exact velocity, top_k_velocity=1, 180 steps, greedy last step.

## GPU allocation (nice=200 everywhere)

| Part | Cells | Site | Queues | Why |
|---|---|---|---|---|
| A | 13 | **Unicorn** (sc3379) | `thickstun,desa` (excl. desa-compute-01) | data local, direct submit |
| B | 27 | **ARC TinkerCliffs** (shengyenc) | `{h200,a100}_{preemptable,normal}_q` | A100/H200 pool, K∈{−0.01,−0.1,−0.5} |
| B | 18 | **ARC Falcon** (shengyenc) | `{a30,l40s}_{preemptable,normal}_q` | L40S/A30 pool, K∈{−0.25,−0.75} |

TC + Falcon share ARC `/home` → their K sets are **disjoint** (no double-owned cell).
Preemptable queues + `--requeue` + ckpt-5k auto-resume ⇒ preemption cost ≤ one refit
window. `sweep.py` is idempotent (skips a cell whose `eval/ppl.json` exists or whose
job is queued) and resumable; rsync ARC→unicorn results before `report.py`.

## Expected wall-clock

~3.9B train tokens/run. At ~66k tok/s (bs32) ≈ **12–16 h/run** on Ada/A100, longer on
A30/L40S; eval ≈ +0.5–1 h. Part A (13) is gated by Unicorn queue drain (currently
saturated by other jobs at nice 0). Part B (45) on ARC's ~420-GPU pool with ~20 cells
concurrent ⇒ ~2–3 waves ≈ **1.5–3 days** wall-clock.

## Deliverable

`experiments/report.py geo_curv_sc_tinystories_256` → `RESULTS.md` with a table of
val/ppl, GenPPL, entropy per cell + insights (truncation gain, self-cond On/Off delta,
curvature optimum and any radial collapse).
