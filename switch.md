# Job Status — TinyStories Geometry Experiments

**Snapshot: 2026-07-19 EDT**

Two experiments in flight across 4 sites. Legend: ✅ finished (has `eval/samples_genppl.json`) · 🟢 running · ⏳ pending · ❌ failed.

---

## Experiment 1 — `geo_curv_sc_tinystories_256` (58 cells)

{geometry × self-cond} Part A + {curvature × init × prior_cov} H-FLM Part B.
Progress: **✅ COMPLETE 58/58** (RESULTS.md written 07-20). Part A 13/13, Part B 45/45. Insights:
- **Adaptive+truncation wins both geometries**: `sfm_adatrunc_scon` GenPPL **11.1**,
  `sfm_trunc_scon` 12.8, `eflm_adatrunc_scon` **13.2**, `sfm_ada_scon` 16.6. Naive worst (34–54).
- **Self-cond is variant-dependent**: helps `eflm_adatrunc` (13.2 vs 29.0 scoff), ~neutral
  for `eflm_ada`, hurts naive/trunc E-FLM (`eflm_naive` scon 53.9 vs scoff 36.3). All ent ≥3.08.
- **Part B curvature (naive H-FLM, SC-on): shallow K wins.** K=−0.01/−0.1 → best cell 17.1,
  median ~23; K=−0.25 median 38.6; K=−0.75 median 40.3. K=−0.5 no data yet. 1 collapse (K=−0.1).
- `val/ppl` (flow bound) ranks opposite to GenPPL for naive cells → trust GenPPL+entropy.

### Part A — geometry × self-cond (13/13 ✅ FINAL), by GenPPL:
| cell | GenPPL | ent |   | cell | GenPPL | ent |
|---|---|---|---|---|---|---|
| **sfm_adatrunc_scon** | **11.1** | 3.97 |   | sfm_naive_scon | 33.7 | 3.83 |
| sfm_trunc_scon | 12.8 | 3.97 |   | eflm_naive_scoff | 36.3 | 3.58 |
| eflm_adatrunc_scon | 13.2 | 3.75 |   | sfm_naive_scoff | 37.1 | 3.89 |
| sfm_ada_scon | 16.6 | 3.86 |   | eflm_trunc_scoff | 43.3 | 3.65 |
| eflm_ada_scoff | 18.4 | 3.81 |   | eflm_trunc_scon | 52.6 | 3.34 |
| eflm_ada_scon | 18.8 | 3.71 |   | eflm_naive_scon | 53.9 | 3.08 |
| eflm_adatrunc_scoff | 29.0 | 3.85 |   | | | |

### Part B — H-FLM curvature (41/45 ≈ FINAL). **Monotonic: flatter strictly better.**
| K | n | median | best | worst | collapses |
|---|---|---|---|---|---|
| **−0.01** | 9 | **17.7** | 17.0 | 23.6 | 0 |
| −0.1 | 9 | 28.3 | 17.1 | 151 | 2 |
| −0.25 | 9 | 33.4 | 21.7 | 56 | 0 |
| −0.5 | 9 | 32.2 | 27.2 | 53 | 0 |
| −0.75 | 5 | 40.3 | 37.3 | 56 | 0 |

Median climbs almost monotonically with depth (17.7→28.3→33/32→40.3; plateau at −0.25/−0.5 middle).
Near-flat K=−0.01 clear winner (best median, tightest spread, 0 collapses). K=−0.1 = fragile outlier
(2 collapses, worst 151). Conclusion: hyperbolic geometry buys nothing on TinyStories; flatter better.

---

## Experiment 2 — `hflm_ada_sc_tinystories_256` (90 cells)

H-FLM + adaptive noise × {curv × init × prior_cov} × {SC on, off}.
Progress: **✅ COMPLETE 90/90** (RESULTS.md written 07-24; 25 collapses, all deep-K/K=−0.1-fragile/small-init-SCoff). Signals:
- **🏆 Best H-FLM = `k−0.01_c0.01_pc1.0_scon` = GenPPL 10.70** (ent 3.96, verified CLEAN Unicorn run) —
  **MATCHES S-FLM best (11.08)**; margin (0.38) is within cross-run variance so say "matches," not "beats,"
  until seed error-bars land. Adaptive-only (NO truncation; S-FLM needed trunc). Near-flat H-FLM ≈ stronger.
- **⚠️ Variance caveat:** one config ran 11.88 (Unicorn) vs 21.93 (TC preemptible) — single-run GenPPL is
  noisy (partly TC preemption-degradation). PENDING: re-run best-H-FLM + best-S-FLM ×3 seeds on Unicorn.
- **Curvature: flat wins (same as naive).** Earlier "adaptive prefers K=−0.25" was an artifact of incomplete
  data; K=−0.01 adaptive (10.70) crushes K=−0.25 (15.7). Winning corner = K=−0.01 + c0.01 init + SC-on.
- **SC×prior_cov interaction:** at K=−0.01 c0.01, SC-on wins big at pc0.5 (11.88 vs collapse) & pc1.0
  (10.70 vs 18.78); ~neutral at pc0.8. SC rescues small-init/low-pc collapse. At K=−0.25, SC-off was better.
- **Collapses (3):** K=−0.1 pc0.8 SC-off (×2), K=−0.01 c0.01 pc0.5 SC-off — all small-init/SC-off fragile zone.

### ch2263 / nlplarge (8× A100-80GB) — K∈{−0.5,−0.75} (36 cells): ✅ aligned → 🟢 training (jobs 962608–962643, ~5 it/s confirmed)
- **flash-attn:** installed real flash_attn 2.8.3.post1 into ch2263 `sfm` (CUDA-12.8; import chain verified).
- **datasets/data drift (2nd failure):** ch2263's tinystories `.arrow` was `"_type":"List"` (newer datasets)
  but all sites run datasets 3.5.0 (which only has `Sequence`) → load failed. **Fixed:** copied sc3379's
  `Sequence`-format data to ch2263 (now identical to sc3379/ARC).
- **version alignment:** ch2263 torch was 2.8.0 vs sc3379/ARC 2.7.0 → **rebuilding to torch 2.7.0+cu128**
  + `cu12torch2.7/cp312` flash-attn wheel (monitor `bey5qoqte`). Relaunch 36 cells on `BUILD_DONE`.

**Cross-site env (target = identical):** py3.12.13 · torch **2.7.0**+cu128 · CUDA 12.8 · flash_attn 2.8.3.post1 · datasets 3.5.0 · `Sequence` data. sc3379 ✅ · ARC ✅ · ch2263 (aligning).

### ARC TinkerCliffs — K∈{−0.01} (18 cells): 18 ⏳
All pending (behind geo_curv Part B, same nice=0).

### ARC Falcon — K∈{−0.1,−0.25} (36 cells): 36 ⏳
All pending (behind geo_curv Part B).

---

## Infra / operational

| Site | Account | Env | flash-attn |
|---|---|---|---|
| Unicorn thickstun/desa | sc3379 | `sfm` (py3.12, torch2.7) | ✅ 2.8.3 |
| ARC TC + Falcon | shengyenc | `sfm` | ✅ 2.8.3 |
| ch2263 nlplarge | ch2263 | `sfm` (py3.12) | 🔧 installing (torch→2.9.1+cu128 + cp312 wheel) |

- **Rebalance (07-18):** Part A done + `hcil` drained freed Unicorn (~24 GPUs). **Moved geo_curv
  Part B (37) → Unicorn** (nice=0, 24 running). **TC fully relieved:** its geo_curv (25) + hflm_ada
  K=−0.01 (18) both parked at nice=500; the 18 hflm_ada resubmitted onto Unicorn (queued behind
  geo_curv, run as it drains). Falcon runs hflm_ada (17); ch2263 hflm_ada K=−0.5/−0.75 (churn).
- **Seed error-bars (07-26) — FINAL VERDICT: H-FLM ≈ S-FLM (statistically TIED).**
  `seed_errbar_tinystories_256` 6/6: **H-FLM [10.70,10.83,12.23] = 11.25 ± 0.69** vs
  **S-FLM [10.68,11.48,11.50] = 11.22 ± 0.38**. Δmean 0.03 ≪ error bars → not significant. Single-run
  10.70<11.08 was a favorable draw ("matches", NOT "beats"). H-FLM has higher variance; S-FLM tighter.
  seed=1 reproduced 10.70 exactly (validated the log-recovery). Added `seed=${SEED:-1}` to both train
  scripts. CURVATURE_ADA_TABLE.md regenerated final (ada K=−0.75 = all 9 collapse).
- **Canonicalization (07-22):** discovered cross-site duplicate runs train DIFFERENT models (hardware +
  micro-batch grouping + TC preempt) and babysit rsync overwrote last-write-wins (Unicorn 11.88 → TC 21.93
  on 1 cell). Fix: STOPPED babysit boqya79k1; CANCELED all ARC+ch2263 dups (ch2263 36, TC 34, Falcon 29 incl
  dead geo). Unicorn = sole canonical source. New monitor **bl51fq1h7** (Unicorn-only: track+heal+report,
  no rsync). Scope: **4 cells clobbered** (all K=−0.01 SC-on near-flat region, incl BEST 10.70→1.05 by a
  TC-preempt collapse); ALL 4 RECOVERED from Unicorn logs (grep 'first chunk only): X' + 'Sample entropy: Y',
  jsons patched + flagged recovered_from_unicorn_log). Best restored = 10.70. 18 ARC-sourced (deep-K), rest
  clean. The 4 Unicorn-vs-TC dup pairs quantify variance: bidirectional & large (10.70 vs collapse; 11.88 vs
  21.93; 16.12 vs 15.76; 58.67 vs 45.27) → single-run margins unreliable. PENDING: seed×3 error-bars.
- **Rebalance-to-Unicorn (07-21):** Falcon went idle (l40s/a30 preempted) with K=−0.1/−0.25 stuck.
  Moved those 18 cells → Unicorn (nice=0). Unicorn now covers ALL 5 curvatures (24R+33PD); priority
  order K=−0.01→−0.1/−0.25→−0.5→−0.75(nice=1000 last). ch2263(K=−0.5/−0.75)+TC(K=−0.01) redundant
  backups. ⚠️ watch: ch2263's 2 running jobs at ~4d20h elapsed (likely stuck; redundant anyway).
- **ASAP redistribution (07-20):** user directive to finish all jobs ASAP across desa/thickstun/
  ch2263/ARC. **Bypassed ch2263's 2-GPU QOS cap:** submitted hflm_ada K=−0.5,−0.75 (36 cells) onto
  Unicorn (nice=0) — the 24-GPU pool now attacks K=−0.01/−0.5/−0.75 (54 cells). ch2263 still grinds
  its 2 as a backstop (rsync+sentinel dedup vs Unicorn). ARC Falcon runs K=−0.1/−0.25 at fair
  priority; ARC TC still externally jammed (a100/h200 saturated by others, ~0 running — redundant
  since Unicorn covers K=−0.01). hflm_ada throughput: ~3 → ~24+ GPUs as geo_curv drains.
- **GPU (07-18):** briefly reserved Ada node for user, then reclaimed on request — thickstun exclude
  removed; my Unicorn jobs now use all 24 GPUs (kuleshov A6000+A5000 + thickstun-01 Ada) at **nice=100**.
- **Monitors:** geo_curv babysit `blew1k6d7`; hflm_ada babysit `boqya79k1` (both rsync + resubmit-failures + `RESULTS.md` on completion).
- **Code:** `dit.py` = original everywhere (no SDPA fallback, md5 `5e43bcdf`). Files byte-identical across sites (git HEADs differ — rsync'd files, not `.git`).
- **Est. wall-clock:** geo_curv ~30 h/run (contention); hflm_ada ~5–14 h/run on A100 once running.
