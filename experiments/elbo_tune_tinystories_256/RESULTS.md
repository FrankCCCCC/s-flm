# elbo_tune_tinystories_256 — Results

Can EFLM and HBFM reach MDLM's likelihood bound on TinyStories under a fair protocol? Design, grids
and the fairness rules: `EXPERIMENT.md`. NELBO in nats per token on the full validation set
(`mode=ppl_eval`, one (t, noise) draw per sequence per pass); regenerate the tables with
`python sweep.py --pull && python report.py`.

**Status (2026-10-12 19:20): stage 1 running** — EFLM: all cells in except E4 (τ 4, due ~20:45);
the (H³)³ HBFM cells land between 2026-10-13 (CE control ~03:00, naive readout ~18:00) and
2026-10-15 (the three horosphere cells on the A5000/A6000 nodes); the incumbent ~2026-10-14 00:00.

<!-- report:begin -->
8 evaluated cells; MDLM at 30k steps (3 seeds) = 1.582 nats/token.

### Every cell

| cell | steps | seed | NELBO (nats / token) | passes | CE objective | vs MDLM 30k |
|---|---:|---:|---:|---:|---:|---:|
| eflm_loss-ce_r-0.5_tau-2.4436_noise-autonomous_lr-1e-3_seed-1 | 10000 | 1 | 2.0892 | 1 | 3.3467 | +0.507 |
| eflm_loss-elbo_r-0.5_tau-0_noise-log-linear_lr-1e-3_seed-1 | 10000 | 1 | 136.4147 | 1 |  | +134.833 |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous-adaptive_lr-1e-3_seed-1 | 10000 | 1 | 2.5772 | 1 |  | +0.995 |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous_lr-1e-3_seed-1 | 10000 | 1 | 1.9140 | 1 |  | +0.332 |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous_lr-3e-4_seed-1 | 10000 | 1 | 1.9588 | 1 |  | +0.377 |
| eflm_loss-elbo_r-2_tau-1.2889_noise-autonomous_lr-1e-3_seed-1 | 10000 | 1 | 1.8806 | 1 |  | +0.299 |
| eflm_loss-elbo_r-8_tau-0.5051_noise-autonomous_lr-1e-3_seed-1 | 10000 | 1 | 1.8886 | 1 |  | +0.307 |
| mdlm_lr-1e-3_seed-1 | 10000 | 1 | 1.7604 | 1 |  | +0.178 |

### Per configuration (mean ± std over seeds)

| configuration | steps | seeds | NELBO (nats / token) |
|---|---:|---|---:|
| eflm_loss-ce_r-0.5_tau-2.4436_noise-autonomous_lr-1e-3 | 10000 | 1 | **2.0892** (n=1) |
| eflm_loss-elbo_r-0.5_tau-0_noise-log-linear_lr-1e-3 | 10000 | 1 | **136.4147** (n=1) |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous-adaptive_lr-1e-3 | 10000 | 1 | **2.5772** (n=1) |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous_lr-1e-3 | 10000 | 1 | **1.9140** (n=1) |
| eflm_loss-elbo_r-0.5_tau-2.4436_noise-autonomous_lr-3e-4 | 10000 | 1 | **1.9588** (n=1) |
| eflm_loss-elbo_r-2_tau-1.2889_noise-autonomous_lr-1e-3 | 10000 | 1 | **1.8806** (n=1) |
| eflm_loss-elbo_r-8_tau-0.5051_noise-autonomous_lr-1e-3 | 10000 | 1 | **1.8886** (n=1) |
| mdlm_lr-1e-3 | 10000 | 1 | **1.7604** (n=1) |
<!-- report:end -->

## 1. Reference points

| model | steps | NELBO (nats / token) | source |
|---|---:|---:|---|
| MDLM, LR 1e-3 | 30k | **1.582 ± 0.007** (3 seeds) | `naive_ar_tinystories_s256`, re-scored under this code |
| MDLM, LR 1e-3 | 10k | **1.760** | M0, this study (the stage-1 yardstick) |
| EFLM CE-trained, R 0.5, τ* 2.44, LR 1e-3 | 30k | 1.893 ± 0.004 (3 seeds) | `eflm_rescale_auto_trunc_tinystories_256`, scored under the bound |
| EFLM CE-trained, same configuration | 10k | 2.089 (CE objective 3.347) | E7, this study |
| HBFM CE-trained, (H³)¹⁶, trunc_exp ub 1, LR 3e-4 | 30k | 2.104 (seed 1) | `hbfm_rlf_ada_curv_tinystories_256`, scored under the bound |

## 2. Stage 1 — EFLM (seed 1, 10k steps)

Small sphere-DiT 768 / 12 / 12, R = embedding norm, autonomous clock truncated at τ_max, ELBO-trained
unless noted; NELBO on the full validation set, one pass. MDLM at the same budget: **1.760**.

| cell | R | τ_max | LR | NELBO | vs MDLM 10k | note |
|---|---|---|---|---:|---:|---|
| E7 CE control | 0.5 | τ*(0.5) = 2.44 | 1e-3 | 2.089 | +0.329 | CE objective 3.347; the same configuration at 30k: 1.893 |
| E0 incumbent | 0.5 | 2.44 | 1e-3 | 1.914 | +0.154 | ELBO training buys 0.175 at equal budget |
| E1 | 0.5 | 2.44 | 3e-4 | 1.959 | +0.199 | the lower LR loses |
| **E2** | **2** | **τ*(2) = 1.29** | **1e-3** | **1.881** | **+0.121** | best; the R axis plateaus from R 2 up |
| E3 | 8 | τ*(8) = 0.51 | 1e-3 | 1.889 | +0.129 | the Sudoku setting |
| E4 | 0.5 | 4 | 1e-3 | pending | | cleaner clean end (floor e⁻⁴) |
| E5 | 0.5 | — (log-linear, untruncated) | 1e-3 | 136 | | closed: the weight ∝ 1/b³ diverges at b = 10⁻³ |
| E6 | 0.5 | 2.44 (autonomous-adaptive) | 1e-3 | 2.577 | | closed: biased training objective (0.66 trained vs 2.55 scored) |

Reading: training on the bound is the first-order effect (2.089 → 1.914 at the incumbent, and the
10k-step ELBO cells already beat the earlier study's 30k-step CE model, 1.893); the embedding norm
is the second (R 2 / R 8: 1.881 / 1.889, i.e. +0.03 over R 0.5), the LR prefers 1e-3; the clock has
to be the truncated autonomous one. The remaining gap to MDLM at this budget is 0.12 nats, which
the §6 profile of the earlier study located in the mid-noise band — the cost of a 768-dimensional
Gaussian channel — rather than at the endpoints.

## 3. Stage 1 — HBFM on (H³)³ (seed 1, 10k steps)

| cell | NELBO | note |
|---|---:|---|
| G5 CE-trained control (horosphere readout) | **2.258** | CE objective 3.166 |

Learned curvature scale (`curvature_probe.py`, 2026-10-14 01:03): without a prompt every position's
t = 0 state is the origin, so λ is one global number per model, a learned clock speed rather than a
per-position curvature: 1.746 for G5 (CE), 2.437 for G0, 2.628 for G2. Unlike the (H³)¹⁶ study,
where λ sat at the lower guard 1/4, on (H³)³ it rises and stretches the trained heat-time range from
[0, 2.30] to [0, 4.0–6.1], about the sampler horizon T_MAX 6.
| G2 ELBO-trained, naive readout | 2.621 | worse than the CE control |
| G0 ELBO-trained, horosphere readout (incumbent) | 2.418 | worse than the CE control by 0.16 |
| G1 LR 3e-4 / G3 truncated proposal / G4 fixed curvature | pending | A5000 / A6000 nodes, ~Oct 15 |

Radial vs angular part of the rate (`rate_split.py`, job 258690; 1024 validation sequences, one
(t, noise) draw each, the training-time proposal and weights; user question of 2026-10-14). The rate
per factor is (d−1)²κ²/2 · ‖Σ_v μ_v w_v‖² and splits exactly into the radial part (Σ μ_v A_v)² plus the
angular part ‖Σ μ_v C_v‖² (C_v ⊥ θ):

| checkpoint | radial | angular | reconstruction | NELBO | corr(radial, angular) over sequences |
|---|---:|---:|---:|---:|---:|
| G5 CE-trained | 0.768 (34 %) | 1.503 (66 %) | 0.014 | 2.285 ± 0.026 | ≈ 0.93 |
| G2 ELBO-trained, naive | 0.898 (34 %) | 1.747 (66 %) | 0.007 | 2.653 ± 0.024 | ≈ 0.83 |

The radial part is a fixed third of the rate for both models and moves in lockstep with the angular
part (per-sequence coefficient of variation 0.36 for the full rate, 0.34 without the radial part).
Dropping it is therefore not an option for the bound (it is no longer a bound: it would report
2.285 − 0.768 ≈ 1.52 for the CE control, "below" MDLM for the wrong reason) and not a promising
training objective either (it removes a third of the constraints on the posterior, 3 of 9 drift
coordinates per position, without making the estimator less noisy in relative terms). The ELBO-trained
model is worse than the CE-trained one by the same ~16 % in both parts, so its deficit is not
radial-specific. The reconstruction at the horizon is negligible (0.3–0.6 %): T_MAX 6 resolves the
tokens on (H³)³, unlike the CE-trained (H³)¹⁶ model at ub 1 (0.43 nats, §1).

## 4. Stage 2 — the picks at 30k steps, three seeds

pending

## 5. Conclusion

pending

## 6. Incidents

- 2026-10-12 00:40: HBFM-ELBO and MDLM smoke cells out of memory on the 11 GB cards at the first
  micro-batches; per-(arm, objective) micro-batches per site.
- 2026-10-12 01:00: the explicit form of the path-KL rate needs (B, L, V, d) tensors, 1.2 GB per
  factor per 4 sequences at V = 50257; rewritten into the memory-light form (same function to
  1e-10, `tests/test_elbo_loss.py`).
- 2026-10-12 06:12: the exact rate on (H³)¹⁶ measured at 2.5 s per micro-batch of 16 on an A100
  (≈ 80 s per optimizer step). A Monte-Carlo estimate over sampled words was implemented, then
  rejected by the user and removed; the (H³)¹⁶ cells that had started on it (220013–220019) were
  cancelled with the user's approval; HBFM moved to (H³)³ with the exact rate.
- 2026-10-12 08:49: E0 (8× 2080 Ti) died of an NCCL all-reduce timeout after 7.5 h, the
  dataloader workers' listener sockets on the NFS temp directory having gone missing; local sites'
  TMPDIR moved to the node's /tmp; the cell resumed from its step-4000 checkpoint.
- 2026-10-12 10:11 / 10:43: E5 (untruncated log-linear clock) and E6 (adaptive clock) excluded as
  a divergent bound (136) and a biased training objective (0.66 trained vs 2.55 scored) — see
  `EXPERIMENT.md`.
- 2026-10-13: two more 8-GPU failures on desa-compute-01 (an NCCL timeout and a CUDA illegal memory
  access, both in the first 15 min); the EFLM / MDLM train scripts had ignored `NUM_WORKERS`
  (8 workers per rank on a 10-CPU allocation) and now honour it; the node is no longer used for
  multi-GPU cells.
- 2026-10-13: a VDM-style (variance-minimising) adaptive clock was implemented and started on the
  EFLM pick, then cancelled and rolled back at the user's request before it produced a result
  (`EXPERIMENT.md`).
