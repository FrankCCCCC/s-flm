# eflm_rescale_tinystories_256 — Results

Rescaled EFLM (norm pinned to R) × schedule arm on TinyStories seq 256, 30k
steps, 1 seed (see EXPERIMENT.md). Eval: valid PPL (flow bound), GenPPL
(gpt2-large retokenized, lower=better), entropy (≥3.0 = non-degenerate),
180 steps exact velocity. Coverage: 9/9 cells (one cell initially lost to a
dataset-cache build race at first launch; re-run cleanly).

Truncation (trunc / trunc_ada arms): ALPHA_MAX = α*(R) =
`alpha_star_euclidean(V=50257, embed_norm=R)` → R=1: **0.840**, R=8: **0.397**,
R=28: **0.158** (flow-time t* = 1−α*: 0.16 / 0.60 / 0.84); ada arm:
alpha_max=null.

| arm | R | valid PPL† | GenPPL | entropy |
|---|---|---|---|---|
| ada | 1 | 24.11 | 12.57 | 3.80 |
| ada | 8 | 4.99 | 17.87 | 3.92 |
| ada | 28 | 2.48 | 20.35 | 3.89 |
| trunc | 1 | 58.15 | 12.78 | 3.75 |
| trunc | 8 | 10.70 | 16.72 | 3.88 |
| trunc | 28 | 7.06 | 18.39 | 3.90 |
| **trunc_ada** | **1** | 36.70 | **10.99** | 3.69 |
| trunc_ada | 8 | 13.87 | 15.11 | 3.90 |
| trunc_ada | 28 | 10.87 | 15.86 | 3.93 |
| *baseline: naive_geo eflm (raw norms)* | — | *1.10* | *34.58* | *3.67* |

† valid PPL is the denoising-CE flow bound **under each run's own schedule** —
truncation/adaptation change the bound's integration range, so it is NOT
comparable across arms or to the untruncated baseline; GenPPL+entropy is the
cross-arm quality metric.

## Headline

**Every rescale+schedule cell beats the raw-norm naive EFLM baseline on
generation quality by 1.7–3.1×** (GenPPL 11.0–20.4 vs 34.58), with healthy
entropy (3.69–3.93, all above the 3.0 collapse bar — mostly *higher* than the
baseline, so this is not repetition collapse). Best: **trunc_ada @ R=1,
GenPPL 10.99** — 3.1× better than baseline.

## Patterns

1. **Schedule ordering: trunc_ada < ada ≈ trunc at every R** (GenPPL
   11.0/15.1/15.9 vs 12.6/17.9/20.4 vs 12.8/16.7/18.4). Combining the static
   Eq.-17 cut with adaptive refitting is uniformly best — on sudoku the combo
   was not tested; here it wins at all three R.
2. **Small R wins on language: GenPPL is monotone in R within every arm**
   (R=1 < 8 < 28). With the schedule matched to the transition, a *small*
   pinned norm — early, smooth decode transition (t*(1)≈0.16 for V=50k) —
   generates better text than noise-scale norms. This is the opposite end
   from sudoku accuracy (where mid/large R + matched schedule won), consistent
   with generation quality favoring many usable denoising steps (smooth ramp)
   over a late sharp transition.
3. **The raw-norm baseline's handicap is now explained**: unrescaled EFLM has
   a 20:1 norm split (type-median 6.4 vs token-median 138.9) — its rare and
   frequent words transition at wildly different times and no single schedule
   fits both; pinning R + matching the schedule removes that mismatch.

## Degeneracy check (passed)

Per-cell over all 64 samples: distinct-4-gram ratio ≈ 0.99 everywhere (no
phrase looping; worst repeated 4-gram appears 2–4×, normal for story text),
64/64 distinct openings (no cross-sample mode collapse), entropy 3.69–3.93.
Eyeball: samples are story-shaped TinyStories prose with correct dialogue
structure and consistent characters; the winner (trunc_ada R=1) reads
noticeably cleaner than large-R cells, whose texts show token-level blending
artifacts ("noise noise", "seesays", "Timmymy") — consistent with the GenPPL
ordering and with late-transition decoding being harder at large R.

## Caveats

- 1 seed, GenPPL from 4×16 sampled sequences — ordering between adjacent cells
  (e.g. ada vs trunc) is suggestive, not significant; the arm×R trends and the
  ~3× baseline gap are far larger than typical seed noise from round-1 sudoku.
- No rescale-naive arm (fixed schedule, no trunc) — the baseline differs in
  BOTH norm treatment and schedule. Add `eflm_rescale.sh`-style cells if the
  attribution matters.
- valid PPL column not cross-comparable (see †).

## Follow-ups

- Loss-geometry L(t) on the kept checkpoints: predicted uniform transition at
  t*(R), no rare/frequent split (the codebook-figure signature).
- Seeds 2–3 on the winners (trunc_ada R=1, R=8) to firm the ordering.
- Rescale-naive control arm for clean attribution.

Data: `outputs/eflm_rescale_tinystories_256/eflmrs256_{arm}_r-{R}_rs1/eval/`.
