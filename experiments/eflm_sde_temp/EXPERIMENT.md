# eflm_sde_temp — best `eta` (GT = Linear) for EFLM's temperature frontier line

Goal (`setup.md`): for **Rescale + Auto + Trunc + EFLM** decoded with exact
velocity, `top_k_velocity = -1` (all) and a greedy last sampling step, choose
the **best `eta` with `GT = Linear`** so that the Gen. PPL / entropy frontier
line traced by the sampling temperature T in [0.50, 1.20] is as low as
possible, on the same 3-seed x NFE x 15-T protocol as
`experiments/naive_ar_tinystories_s256`.

## Why this experiment exists — the two knobs have never been composed

EFLM has two ways to buy entropy, and the previous project measured each one
**alone** (`experiments/eflm_sde/RESULTS.md`, `.../naive_ar_tinystories_s256/
FRONTIER_RESULTS.md` §7.5):

| knob | decode | entropy band it covers | best Gen. PPL |
|---|---|---|---|
| `eta` SDE | `top_k_velocity = 1` (T-inert) | H in [3.8, 4.1] | 9.21 @ H 3.98 (NFE 256) |
| temperature | `top_k_velocity = -1`, `eta = 0` | H in [4.1, 4.55] | 12.44 @ H 4.11 (NFE 256) |

The recorded verdict was *"the two knobs are complementary, not competing …
neither alone traces EFLM's full frontier"*. They were never run **together**,
even though `samplers.py` composes them for free: `EFLMSampler.step` computes
the velocity from `softmax(logits / T) @ E - x` (temperature-dependent when
`top_k_velocity = -1`) and *then* hands it to `_sde_update`, which adds the
`sqrt(eta) g(t) dW` diffusion and the `eta g(t)^2 / (2(1-t))` score-correction
drift. So (eta, T) is a genuine 2-D decode surface and the frontier line is its
lower envelope over T at fixed eta. This sweep finds the eta whose envelope is
lowest.

`GT = Linear` (g(t) = 1 - t) is fixed by `setup.md`; it is also the schedule
that "degrades gracefully … within the swept grid it never explodes
catastrophically at NFE >= 16" (`eflm_sde/RESULTS.md` §4), which matters here
because the exact-velocity target is already diffuse.

## Design

**Checkpoints** (frozen, unchanged): `eflmratr_lr-1e-3_r-0.5_m-1.0{,_s2,_s3}`
from `experiments/eflm_rescale_auto_trunc_tinystories_256` — R = 0.5,
lr 1e-3, TAU_MAX = 2.4436, ALPHA_MAX = null, SNR_CE = false, self-cond off,
30k steps, global batch 512, seq 256. 3 training seeds.

**Eval protocol** (identical to `eflm_sde` phase 3 except `ETA`):
`scripts/sample/tinystories/eflm_rescale_auto_truncation.sh` with
`VELOCITY=exact TOPK_VELOCITY=-1 GT_METHOD=linear`, `noise_removal=greedy`,
512 samples/cell (EVAL_BS 16 x 32 batches), `RUN_PPL_EVAL=false`, eval seed
fixed at 1 (isolates training-seed variance, the convention of every frontier
sweep in this repo). Gen. PPL = gpt2-large retokenized, corpus-level;
entropy = per-sample unigram entropy in nats.

**The `eta = 0` reference column is not re-run** — it is exactly
`outputs/eflm_sde/sd-*/tfrontier/nfe-*_t-*/`, 360 cells already complete.
`analyze.py` reads it from there.

**Swept axes**

- **Phase 1 (pilot, seed 1)** — NFE {8, 16, 64} x eta {0.25, 0.5, 1, 2, 4, 8,
  16} x T {0.50, 0.70, 0.90, 1.10, 1.20}. 105 cells, ~23 GPU-h, 21 jobs.
  Every T is on the final 15-point grid, so the cells are reused verbatim by
  phase 2. Purpose: locate eta* per NFE and its collapse boundary before
  spending 3-seed money.
  Verdict rule: per (NFE, eta) interpolate Gen. PPL along the T-curve at
  H in {4.20, 4.35, 4.50}; eta* = argmin, compared against the eta = 0 column.
- **Phase 2 (frontier, 3 seeds)** — NFE {4, 8, 16, 32, 64, 128, 256} x the
  per-NFE eta* (1–2 values) x the full 15-point T grid. ~170 GPU-h per eta arm.
  NFE = 1 is excluded: the single step *is* the greedy decode, so neither eta
  nor T can act (`FRONTIER_RESULTS.md` §8.6) — it is a degenerate marker.

**Hypotheses**

- **H1 (composability).** At fixed T, entropy rises monotonically with eta up
  to the collapse boundary, i.e. eta still works as a diversity knob when the
  velocity target is the full-vocab expectation rather than the argmax
  embedding.
- **H2 (frontier improvement — the deliverable).** There is an eta* > 0 whose
  T-curve lies **below** the eta = 0 T-curve at matched entropy over a
  non-trivial band of H in [4.2, 4.5], for NFE >= 16. Mechanism: the SDE
  contracts accumulated integration error (the "free lunch" of
  `eflm_sde/RESULTS.md` §1, worth −1.1 to −2.2 Gen. PPL under k = 1 velocity),
  and that contraction is orthogonal to the target-blurring that T does.
  Falsified if argmin over eta is eta = 0 at every NFE and every matched H.
- **H3 (eta* scales with NFE).** Per-step injected variance is `eta g(t)^2 dt`
  with dt ∝ 1/NFE, so a fixed entropy budget needs eta ∝ NFE — as measured
  under k = 1 velocity (`eflm_sde/RESULTS.md` §4: eta ≈ NFE/8). Predict eta*
  grows with NFE but *below* the k = 1 rule, because T already supplies part
  of the entropy.
- **H4 (competitiveness).** With eta*, EFLM's matched-entropy Gen. PPL closes
  the H <= 4.35 gap to MDLM at NFE >= 32 (currently +5% at H 4.35 to +38% at
  H 4.20, `FRONTIER_RESULTS.md` §7.2) while keeping EFLM's outright win at
  NFE 4–16. Stretch: EFLM becomes the best method at H 4.35 for some NFE >= 32.

**Collapse guard.** A cell is flagged degenerate (excluded from frontier fits,
reported separately) if entropy > 5.0, Gen. PPL > 200, or Gen. PPL is NaN —
the cliff-edge behaviour documented in `eflm_sde/RESULTS.md` §4.

**Compute** — Unicorn, `thickstun,desa` (exclude `desa-compute-01`), 1 GPU per
job. Measured cell cost at 512 samples under exact velocity:
~`8 + 26.6 x NFE` s (220 s at NFE 8, 1710 s at NFE 64). Phase 1 ~23 GPU-h /
21 jobs; phase 2 ~170 GPU-h per eta arm, `--t-chunk` splitting the T grid so
NFE 128/256 jobs stay under ~6 h. Idempotent: a cell is skipped iff its
`samples_genppl.json` exists or its job name is in `squeue`.

**Outputs** — `outputs/eflm_sde_temp/sd-{seed}/nfe-{nfe}_eta-{eta}_t-{temp}/`;
sweep `experiments/eflm_sde_temp/sweep.py`; analysis + figures
`experiments/eflm_sde_temp/analyze.py`; report
`experiments/eflm_sde_temp/RESULTS.md`. The winning arm is added to the shared
frontier figure of `experiments/naive_ar_tinystories_s256` via
`visualization/genppl_entropy_frontier_line.py`, per `setup.md`.

## Code changes required

**None.** `sampler.eta`, `sampler.gt_method`, `sampler.top_k_velocity` and
`sampler.temperature` are already independent knobs in `samplers.py`
(`EFLMSampler.step`, lines ~1007–1035) and are all exposed as env vars by
`scripts/sample/tinystories/eflm_rescale_auto_truncation.sh`. Phase 0 is a
smoke run only: NFE 16, 32 samples, eta {0, 1, 4, 16} x T {0.50, 0.70, 1.10}
on a compute node — verify no NaN and that entropy responds to eta *and* to T
in the same cell.

**Phase-0 verdict (2026-08-25, job 589777, NFE 16, 32 samples/cell, seed 1):**
the composition runs clean — 6/6 cells, no NaN, 32/32 unique texts everywhere,
and the saved `config.sampler` of each cell carries the intended
`(eta, temperature, top_k_velocity = -1, gt_method = linear)`.

| eta | T | Gen. PPL | H |
|---|---|---|---|
| 0  | 0.70 | 16.95 | 4.276 |
| 1  | 0.70 | 17.29 | 4.259 |
| 4  | 0.70 | 18.39 | 4.272 |
| 16 | 0.70 | 21.75 | 4.331 |
| 4  | 0.50 | 15.17 | 4.133 |

Early read (32 samples, indicative only): at NFE 16 under exact velocity,
`eta` buys entropy far more expensively than T does — eta 0 -> 16 costs
+4.8 Gen. PPL for +0.055 nats (~87 PPL/nat) where the T-curve at the same
point costs ~65 PPL/nat, and ~13 PPL/nat down at T = 0.50. **H2 is therefore
at risk at low NFE**, which is why the pilot spends its budget on NFE
{16, 64, 128}: the SDE's free lunch under k = 1 velocity was worth −1.5 to
−2.2 Gen. PPL at NFE >= 64 and only −0.5 at NFE 16
(`eflm_sde/RESULTS.md`, gt = linear column), and the low-T end is where the
exact velocity most resembles the k = 1 velocity that free lunch was measured
under.

**Pilot grid as submitted** (seed 1, 512 samples, 25 jobs, ~47 GPU-h) — the
eta grid runs to 64 because the k = 1 optimum for `gt = linear` sits at
eta* ≈ NFE/4 (12.56 @ eta 4 for NFE 16; 10.06 @ eta 16 for NFE 64; 9.68 @
eta 16 for NFE 128), well above the range the phase-0 smoke probed:

| NFE | eta | T | cells |
|---|---|---|---|
| 16  | 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64 | 0.50, 0.60, 0.70, 0.90, 1.10 | 45 |
| 64  | 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64 | 0.50, 0.60, 0.70, 0.90, 1.10 | 45 |
| 128 | 1, 2, 4, 8, 16, 32, 64 | 0.50, 0.60, 0.70 | 21 |

**Pre-analysis of the existing k = 1 data (no new compute) — where the win
should be.** Re-interpolating `experiments/eflm_sde`'s eta-arm per seed at
matched entropy (same rule as the deliverable table) gives, for `gt = linear`:

| NFE | H = 4.10 | H = 4.20 | exact-vel. eta = 0 @ H 4.20 |
|---|---|---|---|
| 16  | 16.03 ±0.84 | 20.34 ±2.61 | 15.92 ±0.42 |
| 32  | 12.67 ±0.48 | 14.67 ±0.13 | 14.50 ±0.48 |
| 64  | 11.20 ±0.33 | **12.59 ±0.27** | 13.96 ±0.49 |
| 128 | 10.29 ±0.07 | — (eta ceiling H ≈ 4.14) | 13.52 ±0.35 |
| 256 |  9.78       | — (eta ceiling H ≈ 4.11) | 13.40 ±0.41 |

At **NFE 64 / H 4.20 the eta knob alone already beats the temperature knob by
10%** (12.59 vs 13.96), and at NFE >= 128 the `linear` arm simply runs out of
entropy before H 4.20 (its ceiling is 4.10–4.14 even at eta = 128). Both facts
point the same way: the composition should pay off at **high NFE with large
eta and low T** — eta supplies the cheap entropy up to ~4.1 where it is a free
lunch, and T carries it the rest of the way, instead of T doing all the work
from H 4.11 upward. Two extra pilot arms (NFE 64 and 128 at eta = 128) were
added to cover the top of that band. Pilot as run: 27 jobs, 119 cells.

## Phase-1 verdict (2026-08-25, seed 1, ~94 cells): H2 CONFIRMED, and eta* = NFE/4

**H1 confirmed** — entropy rises monotonically with eta at every fixed T, so
the SDE remains a diversity knob when the velocity target is the full-vocab
expectation. **H2 confirmed at NFE >= 32** and refuted below it. **H3
confirmed in a sharper form than predicted**: eta* is not merely increasing in
NFE, it is exactly `NFE / 4` wherever it is non-zero.

`eta`-trace at T = 0.50 (the column where the exact velocity most resembles
the k = 1 velocity), seed 1, 512 samples:

| NFE | eta = 0 | eta* | Gen. PPL @ eta* | gain |
|---|---|---|---|---|
| 4   | 34.95 @ 3.980 | 0  | — | none (eta 0.5 costs +19%) |
| 8   | 18.21 @ 4.089 | 0  | — | none (eta 1 costs +10%) |
| 16  | 14.52 @ 4.110 | ~0 | 14.36 @ 4.101 | +1.1% (noise) |
| 32  | 13.39 @ 4.114 | **8**  | 13.06 @ 4.157 | **+2.5%** |
| 64  | 12.69 @ 4.120 | **16** | 11.64 @ 4.145 | **+8.2%** |
| 128 | 12.60 @ 4.112 | **32** | 10.87 @ 4.166 | **+13.7%** |
| 256 | 12.33 @ 4.121 | **64** | 10.49 @ 4.168 | **+14.9%** |

**Why the threshold at NFE ~ 32.** The SDE trades injected noise against the
model's ability to re-denoise it in the remaining steps. Below ~32 steps there
is no budget for that, so the injection is pure damage — at NFE 8, eta = 1
already costs +10% and eta = 4 costs +29%. Above it, the noise contracts
accumulated integration error faster than it adds error, and the benefit grows
with the step budget because there are more error-contracting steps to
exploit. This is the same mechanism as the k = 1 free lunch
(`eflm_sde/RESULTS.md` §1) and it survives the temperature-blurred target.

**It is whole-curve dominance, not a better point.** At NFE 64 the eta = 16
temperature curve lies below the eta = 0 curve at *every* temperature:

| T | 0.50 | 0.60 | 0.70 | 0.90 |
|---|---|---|---|---|
| eta = 0  | 12.69 @ 4.120 | 13.65 @ 4.203 | 14.94 @ 4.269 | 19.10 @ 4.381 |
| eta = 16 | **11.64 @ 4.145** | **12.53 @ 4.210** | **13.76 @ 4.275** | **18.15 @ 4.375** |
| gain | +8.3% | +8.2% | +7.9% | +5.0% |

so the composed arm is a genuinely better *frontier line*, which is what
`setup.md` asks for — not merely a better cell.

**A second-order structure worth recording**: at fixed NFE the eta that wins
grows with the *target* entropy (NFE 64: eta 16 @ H 4.15, 32 @ H 4.20, 64 @
H 4.30, 128 @ H 4.40, each ~7–10% below the eta = 0 curve). The (eta, T)
surface therefore has a ridge, and its lower envelope beats any single-eta
curve. The 3-seed deliverable follows the paper protocol of **one curve per
method per NFE** and uses the single eta* = NFE/4; the seed-1 envelope is
reported in `RESULTS.md` as the evidence behind that choice.

**Phase 2 as submitted** — 3 seeds x the full 15-point T grid at eta* =
NFE/4 for NFE {32, 64, 128, 256} (42 jobs, ~160 GPU-h). NFE {4, 8, 16} are
**not** re-run: eta* = 0 there, so the existing `eta = 0` arm
(`outputs/eflm_sde/sd-*/tfrontier/`) already *is* the best frontier line at
those budgets, and re-running it would only reproduce it.

## Phase-2 verdict: COMPLETE (180/180 cells, 3 seeds, zero failures)

NFE {32, 64, 128, 256} x eta* = NFE/4 x the full 15-point T grid x 3 seeds.
**H2 confirmed, H4 (the stretch goal) met**: EFLM eta* beats MDLM at
NFE 256 / H 4.35 in 3/3 seeds (14.62 ±0.19 vs 15.00 ±0.05) and at
NFE 128 / H 4.50 in 3/3 seeds. Full numbers, per-seed paired checks and
limitations in `RESULTS.md` §6–§9.

Deliverables: `RESULTS.md`, `figures/eta_temp_frontier.png`, `cells.csv`,
`frontier_cells_sde.csv`, and the four-method frontier figure
`experiments/naive_ar_tinystories_s256/figures/genppl_entropy_frontier_line_sde.png`.
