# simpflm_auto_trunc_tinystories_256 — RESULTS

Design, hypotheses and protocol: `EXPERIMENT.md`. Arm:
`scripts/{train,sample}/tinystories/simpflm_auto_truncation.sh`.

**Status: COMPLETE for what was run.** LR x R grid at m = 1.0: 3 seeds x
2 LR x 4 R = 24 runs (plus 4 single-seed 5e-3 runs, not replicated -- see
below). GenPPL/entropy frontier: 360 SimpFLM cells (3 seeds x 8 NFE x 15 T,
512 samples each), integrated into the 4-method 1440-cell frontier. **Not run:**
the truncation multiplier m (every cell is m = 1.0) and the adaptive arm.

SimpFLM = E-FLM with the word-embedding matrix replaced by the diagonal R·I_V.
All cells: `small-flm` / flm-dit 768x12x12, 30k steps, global batch 512, seq
256, bf16, EMA 0.9999, plain CE, 1 seed. Every cell truncates at its OWN
closed-form decode point (m = 1.0, `TAU_MAX = tau*(R) = log(1 + C/R)`), so R is
measured without confounding it with a mis-set horizon. Eval: 180 steps, exact
velocity, top_k_v = 1, greedy last, 64 samples.

| LR | R | TAU_MAX | GenPPL | entropy | val/ppl | admissible |
|---|---|---|---|---|---|---|
| 3e-4 | 0.5 | 2.4436 | 20.42 | 3.743 | 40.40 | yes |
| 3e-4 | 1.0 | 1.8338 | 19.66 | 3.749 | 28.60 | yes |
| 3e-4 | 2.0 | 1.2889 | **18.87** | 3.774 | 19.95 | yes |
| 3e-4 | 5.0 | 0.7186 | 20.18 | 3.795 | 13.91 | yes |
| 1e-3 | 0.5 | 2.4436 | 20.72 | 3.784 | 40.01 | yes |
| 1e-3 | 1.0 | 1.8338 | 20.40 | 3.790 | 27.59 | yes |
| 1e-3 | 2.0 | 1.2889 | 21.00 | 3.789 | 19.91 | yes |
| 1e-3 | 5.0 | 0.7186 | 20.81 | 3.790 | 13.75 | yes |
| 5e-3 | 0.5 | 2.4436 | ~~1.09~~ | **0.000** | NaN | **NO** |
| 5e-3 | 1.0 | 1.8338 | ~~18.00~~ | **0.668** | inf | **NO** |
| 5e-3 | 2.0 | 1.2889 | ~~1.09~~ | **0.000** | NaN | **NO** |
| 5e-3 | 5.0 | 0.7186 | ~~1.20~~ | **0.000** | inf | **NO** |

## Findings

**F1. LR 5e-3 diverges at every R — it is an optimizer failure, not a geometry
one.** Three of four cells collapse to a single repeated token (entropy exactly
0.000), with NaN or infinite validation bounds:

> `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!` (R = 0.5)
> ` journals journals journals journals journals journals …`  (R = 5.0)

Varying the simplex radius over a 10x range does not rescue it, which answers
the question the R sweep was extended to 5e-3 to settle: the collapse is
driven by the optimizer diverging, not by the ambient geometry. **5e-3 is
simply above the stability edge for this model at this batch size.**

**F1b. The degenerate cells have the BEST GenPPL in the entire grid (1.09).**
A constant token stream is trivially predictable for gpt2-large, so GenPPL
rewards it. Any ranking on GenPPL alone selects a model that emits `!!!!!!!!`.
The entropy >= 3.0 gate is not a formality here — it is the only thing
separating the best-scoring cells from the usable ones.

**F2. Among admissible cells, GenPPL is FLAT in both R and LR — now measured
over 3 seeds, not inferred from a borrowed noise floor.** Final grid
(GenPPL mean +/- sd over seeds {1,2,3}, 24 runs):

| LR | R=0.5 | R=1.0 | R=2.0 | R=5.0 |
|---|---|---|---|---|
| 3e-4 | 20.79 ±0.86 | 20.52 ±0.87 | 19.84 ±0.87 | 20.44 ±0.23 |
| 1e-3 | 20.34 ±1.15 | **19.67 ±0.76** | 19.81 ±1.03 | 19.77 ±0.90 |

All eight cells span **19.67-20.79** (spread 1.12) against a typical per-cell
sd of **0.83**. Best vs worst is **1.34 sigma** — i.e. the entire 2x4 grid is a
statistical tie. Entropy is equally flat (3.755-3.802). **There is no optimum
in R or LR here; the response surface is genuinely flat**, and H1/H2/H3 as
applied to this grid are all unsupported.

**F2a. The single-seed grid picked a winner that does not survive
replication.** Seed 1 alone selected (3e-4, R=2.0) at 18.87 as best; its 3-seed
mean is **19.84 +/- 0.87**, so that draw sat >1 sigma below the truth, and the
actual 3-seed minimum is a different cell (1e-3, R=1.0) that is itself within
noise of every other. Tuning on the single-seed grid would have been chasing
noise. This is the concrete cost of the resolution floor this project
pre-registered, and it is why the replication was worth the GPU time.

The measured sd (0.76-1.15, mean 0.83) closely matches the 0.5-0.85 figure
borrowed from `experiments/seed_errbar_tinystories_256`, so that floor was
well calibrated for this protocol.

Contrast with E-FLM, where R mattered a great deal (11.81 at R=0.5 rising
monotonically to 18.19 at R=28). Fixing the geometry appears to have removed R
as a meaningful knob: every cell truncates at its own tau*(R), only the ratio
noise/R enters the bound, and the flows are near-equivalent by construction.

**F3. val/ppl moves 3x with R while GenPPL does not move at all.** 40.40 ->
28.60 -> 19.95 -> 13.91 as R goes 0.5 -> 5, monotone and far outside any noise
band, while GenPPL stays at ~19-21 throughout.

**Do not read this as R=5 being a better model.** `val/ppl` here is the
denoising-CE flow bound, and it is evaluated under each cell's own schedule:
larger R means smaller tau*(R), a shorter flow, less noise at the endpoint, and
therefore a mechanically lower CE. It is measuring the truncation level, not
generation quality. This grid is a clean demonstration that **val/ppl is not
comparable across schedules** and that GenPPL + entropy is the deliverable.

**F4. Placed against the right peer set, SimpFLM is competitive with discrete
diffusion and ~2x better than the existing one-hot flow.** Best admissible cell
per method, same protocol (30k steps, seq 256, batch 512, 180 sampling steps,
64 samples, gpt2-large retokenised GenPPL):

| method | GenPPL | entropy | family |
|---|---|---|---|
| AR (`naive_ar`) | 6.62 | 4.22 | autoregressive |
| E-FLM, log-linear + trunc + adaptive, R=1 | 10.36 | 3.58 | Euclidean flow, learned embeddings |
| S-FLM / H-FLM (`seed_errbar`) | 10.68 | 3.98 | sphere / hyperbolic flow |
| E-FLM, autonomous + trunc, R=0.5 | 11.81 | 3.82 | Euclidean flow, learned embeddings |
| DUO | 17.19 | 4.36 | discrete diffusion |
| MDLM | 18.02 | 4.40 | discrete diffusion |
| **SimpFLM (this work), 3e-4, R=2.0** | **18.87** | **3.77** | **flow on the one-hot simplex** |
| FLM (the repo's existing one-hot/simplex flow) | 35.06 | 4.46 | flow on the one-hot simplex |

Two corrections to an earlier version of this file, which compared only against
E-FLM and concluded "SimpFLM underperforms":

* **Against its true peer — FLM, the other Gaussian flow on the one-hot space —
  SimpFLM is nearly 2x better (18.87 vs 35.06).** Replacing FLM's LUT-based
  gamma schedule with E-FLM's machinery (autonomous clock + the closed-form
  decode point + a pinned simplex radius) is a large gain within the simplex
  family. That is the comparison this project was set up to make.
* **Against DUO (17.19) and MDLM (18.02) it is roughly at parity on GenPPL**,
  not far behind. The gap to E-FLM (11.81) is real, but E-FLM is a different
  family: it gets to *learn* its ambient geometry, which the diagonal by
  construction cannot.

On the GenPPL-entropy frontier, DUO and MDLM still weakly dominate SimpFLM —
they have both lower GenPPL and higher entropy. SimpFLM is *not* dominated by
FLM (much better GenPPL, lower entropy), so that is a genuine trade rather than
a strict loss.

Entropy patterns by family, not by quality: the flow arms all sit at 3.6-4.0
(SimpFLM 3.77, E-FLM 3.82, S-FLM/H-FLM 3.95-3.98) while discrete diffusion sits
at 4.3-4.4. SimpFLM's entropy is normal *for a flow*.

## Verification of the metrics (F5)

The numbers above were challenged and checked rather than taken on trust:

* **Entropy recomputed independently** from the stored token arrays
  (`np_tokens_b64`) with a fresh implementation: exact agreement to 4 decimals
  on every cell tested (3.7741, 3.7490, 3.7905, 0.0000). Shapes are (64, 256)
  as expected.
* **GenPPL and entropy are shared, model-agnostic code** (`metrics.py`)
  operating on decoded text and token ids. There is no SimpFLM-specific path,
  so a metric bug would have to affect every arm equally.
* **Cross-validated on the collapse signature.** SimpFLM's diverged 5e-3 cells
  report GenPPL 1.09 / entropy 0.000 — the *same* signature E-FLM's own
  diverged 5e-3 cells produce (1.09 / 0.000, 1.11 / 0.000) and FLM's (1.14 /
  0.000). Identical sample count (64) and NFE (180) throughout.
* **The model is healthy; the sampler is not losing the quality.** Feeding real
  validation `x0` through each arm's own `q_xt` and forwarding once (no sampler
  involved), top-1 denoising accuracy is:

  | alpha (signal) | SimpFLM R=2.0 | E-FLM R=0.5 |
  |---|---|---|
  | 0.9500 | 90.4% | 99.8% |
  | 0.9132 (E-FLM's alpha*) | 92.8% | 97.5% |
  | 0.9000 | 93.2% | 95.8% |
  | 0.8000 | 92.4% | 66.2% |
  | 0.7244 (SimpFLM's alpha*) | 91.3% | 33.0% |
  | 0.6000 | 82.2% | 15.3% |
  | 0.4000 | 27.0% | 9.0% |

  SimpFLM denoises at 91-93% — it is not broken. Two things follow. (a) At its
  own decode point it is 91.3% vs E-FLM's 97.5% at *its* decode point, i.e. a
  ~3.5x higher token error rate at the moment decoding happens, which is a
  sufficient explanation for the GenPPL gap without invoking any bug.
  (b) **SimpFLM is far more robust at high noise** — 82.2% vs 15.3% at
  alpha = 0.6 — a real property of the fixed diagonal, just not the one GenPPL
  rewards. (Each arm is out-of-distribution above its own truncation, which is
  why SimpFLM dips slightly at 0.95: it never trained above alpha = 0.7247.)
* **Generated text was read, not just scored.** SimpFLM and E-FLM samples are
  qualitatively similar TinyStories prose with comparable local breakdowns;
  SimpFLM's are somewhat more broken mid-sentence, consistent with (a).

## F6. The frontier: DUO dominates SimpFLM at matched entropy

The single-point comparison in F4 flattered SimpFLM, because it is scored at a
far more mode-seeking operating point (entropy 3.77) than DUO/MDLM (4.32-4.41).
Re-evaluating the SAME checkpoint (3e-4, R=2.0) across decode rules -- no
retraining -- traces its GenPPL-entropy curve:

| sampler setting | GenPPL | entropy |
|---|---|---|
| greedy, T=1.0, top_k_v=1 | 19.99 | 3.737 |
| ancestral, T=1.0, top_k_v=1 | 21.04 | 3.791 |
| ancestral, T=1.0, top_k_v=all | 51.15 | 4.481 |
| ancestral, T=1.1, top_k_v=all | 83.60 | 4.582 |
| **DUO 3e-4 (3-seed mean)** | **18.50** | **4.322** |

**At matched entropy SimpFLM is ~2.8x worse than DUO** (51.15 @ 4.48 vs
18.50 @ 4.32). The apparent parity in F4 was an artifact of comparing a
mode-seeking flow decode against a stochastic ancestral decode. DUO dominates
SimpFLM on the frontier, and the earlier "roughly at parity" reading is
withdrawn.

**Mechanism: `top_k_velocity` is the whole story, and it is specific to the
one-hot geometry.** Going from top-1 to the full-vocab velocity costs 30 GenPPL
points (21.04 -> 51.15). The exact velocity aims at `x_hat = R * p`, the
probability-weighted mean of the vertices. In E-FLM that mean is an average of
*learned* embeddings and lands near the embedding manifold, so it is a sensible
target. On the simplex, `R * p` for a diffuse `p` is an interior point that is
maximally far from *every* vertex, so the flow spends its trajectory heading
somewhere it must then abandon. SimpFLM only works when the velocity snaps to a
single vertex (top-1), which is precisely what forces its low entropy.

This is the structural reason the diffusion-duality intuition ("DUO is a
Gaussian diffusion on the one-hot space, so it should match SimpFLM") does not
transfer: the duality relates the *marginals*, but DUO's sampler never leaves
the vertex set -- every intermediate state is a token -- whereas SimpFLM's must
traverse the simplex interior, where its own velocity target is ill-posed.

**Caveat on single-eval noise.** Re-running the identical greedy/top-1 setting
gave 19.99 vs the sweep's 18.87 (different sampling RNG, same 64 samples), so a
single GenPPL evaluation carries ~1 point of noise on top of the seed-to-seed
spread. Differences below ~2 points should not be read as real from one eval.

## F7. GenPPL / entropy frontier — SimpFLM is a FEW-STEP specialist

Full protocol from `setup.md` (S-FLM Fig. 10): 4 methods x 3 seeds x 8 NFE x
15 temperatures, 512 samples/cell = **1440 cells, all complete, zero failures**.
Figure: `experiments/naive_ar_tinystories_s256/figures/genppl_entropy_frontier_line.png`;
per-cell data: `.../frontier_cells.csv`.

Best Gen. PPL per (method, NFE), minimised over T, mean +/- sd over 3 seeds:

| NFE | MDLM | DUO | FLM | **SimpFLM** |
|---|---|---|---|---|
| 1 | 87.69 ±0.91 | 79.30 ±10.08 | 49.53 ±0.48 | **21.16 ±0.00** |
| 4 | 49.08 ±0.31 | 65.68 ±1.08 | 50.60 ±43.65 | **39.56 ±1.19** |
| 8 | 22.24 ±0.09 | **18.42 ±0.61** | 69.12 ±3.75 | 25.63 ±0.64 |
| 16 | 13.19 ±0.12 | **12.38 ±0.13** | 56.26 ±1.62 | 21.80 ±0.77 |
| 32 | **10.47 ±0.03** | 10.60 ±0.14 | 44.99 ±0.94 | 20.42 ±0.63 |
| 64 | **9.27 ±0.01** | 10.03 ±0.10 | 36.40 ±0.65 | 19.85 ±0.60 |
| 128 | **8.64 ±0.09** | 9.76 ±0.02 | 30.46 ±0.33 | 19.47 ±0.59 |
| 256 | **8.44 ±0.02** | 9.66 ±0.08 | 26.84 ±0.34 | 19.33 ±0.60 |

**F7a. SimpFLM is the best method in the few-step regime, by a wide margin.**
At NFE = 1 it scores 21.16 against 49.53 / 79.30 / 87.69 — a 2.3x to 4.1x
advantage, and its seed sd is 0.00 (identical across all three seeds, since one
step from the Gaussian prior with a top-1 velocity is nearly deterministic). At
NFE = 4 it still leads (39.56 vs 49.08 / 50.60 / 65.68). These margins are
enormous relative to the 0.6-1.2 seed sd.

**F7b. The crossover is at NFE = 8**, where DUO (18.42) overtakes SimpFLM
(25.63). From there the discrete-diffusion arms pull away monotonically, ending
at 8.44 (MDLM) and 9.66 (DUO) vs SimpFLM's 19.33 at NFE = 256.

**F7c. SimpFLM saturates; the others keep buying quality with compute.** From
NFE 8 -> 256 (32x the sampling compute) SimpFLM improves 25.63 -> 19.33, just
6.3 points, while MDLM improves 22.24 -> 8.44 (13.8 points) and DUO 18.42 ->
9.66. The same property that makes SimpFLM excellent at 1-4 steps caps it later:
the top-1 velocity snaps straight to a vertex, so the trajectory is essentially
resolved within a few steps and extra NFE has almost nothing left to do.

**F7d. SimpFLM beats FLM — its true peer — at every single NFE**, often by
2-3x (21.80 vs 56.26 at NFE 16; 19.33 vs 26.84 at NFE 256). Both are Gaussian
flows on the one-hot space; the difference is E-FLM's machinery (autonomous
clock, closed-form decode point, pinned simplex radius) replacing FLM's
LUT-based gamma schedule. That is the comparison this project set out to make,
and it is unambiguous.

**F7e. Temperature is inert, confirmed on all three seeds.** Entropy spans
0.0014-0.0053 nats over T = 0.50->1.20 at every NFE and every seed (exactly
0.0000 at NFE = 1), versus 0.29-0.65 nats for the other methods. SimpFLM
therefore contributes a *point* per NFE, not a curve — visible in the figure as
a single marker per panel. Mechanism (F6): temperature divides the logits,
top-1 velocity and greedy decode both take argmax, and argmax is invariant to
positive scaling. Consequence: **SimpFLM cannot be moved along the
entropy axis under this protocol at all**, and its entropy sits at ~3.79 where
MDLM/DUO cannot go below ~4.04. The matched-entropy column of the generated
tables therefore compares each method at the edge of its reachable range, not
at a genuinely shared operating point.

**Reading.** SimpFLM is not a worse MDLM/DUO; it occupies a different regime.
It is the strongest arm tested when sampling budget is 1-4 steps and the lowest
in reachable entropy, but it neither scales with NFE nor trades quality for
diversity. For few-step generation that is a genuine advantage; for the
high-NFE, high-entropy regime the discrete arms are better.

## What this grid does NOT establish

- **Nothing about the truncation multiplier.** Every cell ran at m = 1.0.
  H1 (best m in [0.7, 1.0]) and H2 (sharper optimum than E-FLM) are untested.
- **Nothing about the adaptive remap** (H4) — no `trunc_ada` cell was run.
- **Seed replication is partial by design.** The LR x R grid is being extended
  to seeds 2 and 3 for **LR 3e-4 and 1e-3 only** (16 cells). The 5e-3 row was
  deliberately NOT replicated -- all four of its seed-1 cells collapsed
  (entropy 0.000-0.668, NaN/inf bounds), and confirming a collapse at ~200-350
  GPU-h was judged not worth the pool time. **Consequence: F1 ("5e-3 diverges
  at every R") rests on a single seed and should be read as such** -- it is
  consistent with E-FLM's and FLM's own 5e-3 collapses at the same protocol,
  which is corroboration but not replication.
- Until the seed-2/3 cells land, F2's tie is a tie *by the resolution floor*,
  not a measured equality. F4's margin is large enough to survive it; F2's is
  not.

## Suggested next step

F2 and F4 together argue against more single-seed grid points: the response
surface in R is flat, and the gap to E-FLM is ~8 points, far larger than any
knob in this grid moved anything. The two informative options:

1. **Seed-replicate** the best admissible cell and the R=1 baseline (2 extra
   seeds each, 4 cells) — turns F2/F4 into measured statements.
2. **Test m** at (3e-4, R=2.0): `--lrs 3e-4 --rhos 2.0 --mults 0.7 0.85 1.25 1.5`.
   Worth doing for completeness of the pre-registered design, but F2 suggests
   the schedule endpoint is not where the loss is.

## Reproducing

    python .../sweep.py --lrs 3e-4 1e-3 5e-3 --rhos 0.5 1.0 2.0 5.0 --mults 1.0   # this grid
    python .../sweep.py --lrs 3e-4 --rhos 2.0 --mults 0.7 0.85 1.25 1.5           # m sweep
    python .../sweep.py --lrs 3e-4 --rhos 2.0 --mults 1.0 --seeds 1 2 3           # replication

**Operational note.** The cluster sets `TMPDIR=/home/sc3379/tmp`, a filesystem
with ~300 MB free. Lightning's `_atomic_save` stages the whole checkpoint
through `tempfile.mkstemp()`, so every 2.6 GB save died with ENOSPC at the first
checkpoint (jobs 387971-3, 451109). `sweep.py` now sets
`TMPDIR=/share/desa/nfs02/sc3379/tmp/$SLURM_JOB_ID` — short (a longer per-cell
path hit the 108-byte `sockaddr_un` limit, "AF_UNIX path too long", jobs
452554-6) and on the checkpoint filesystem so the move is a rename. Running the
train scripts by hand outside the sweep still inherits the bad `TMPDIR`.
