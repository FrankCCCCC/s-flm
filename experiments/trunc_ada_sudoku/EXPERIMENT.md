# trunc_ada_sudoku — EFLM/HFLM truncated+adaptive implementation check

**Owner:** sc3379@cornell.edu · **Date:** 2026-07-09 · **Branch:** `claude/ada_sched`
**One-line:** Sanity-check the new geometry-specific truncation bounds + fixed adaptive
noise scheduler (branch `claude/ada_sched`) by training E-FLM and H-FLM with
truncation+adaptive on Sudoku easy/medium/hard at the best hyperparameters from
`experiments/hflm_curv_init_lr_sudoku`, and comparing against that sweep's naive anchors.

## Hypothesis / correctness bar

This is an implementation check, not a hypothesis test. The new code is **correct** if:

1. **No collapse.** The historical failure mode — the sphere bound α⋆=0.093 applied to
   HFLM collapsed it to 12.25%/0.00% (`experiments/hflm/RESULTS.md`) — must not recur
   with the geometry-corrected bounds. Any arm falling far below its naive anchor
   (e.g. <50% of it) signals a broken schedule.
2. **Ballpark or better vs naive.** Trunc+adaptive helped S-FLM (+7pt easy, +14pt med,
   +23pt hard over naive S-FLM in the `bl_*` baselines); we expect the same direction —
   or at minimum parity — for E-FLM/H-FLM. Single seed, so read with the sweep's noise
   bars (hard slice per-run std ≈ 6.6pt).
3. **Adaptive engaged.** Each run's `last.ckpt` must show `noise.has_schedule=True` and
   `refit_count > 0` (the pre-fix code never refit; see `tests/test_adaptive_schedule.py`).

## Hyperparameter choice (from hflm_curv_init_lr_sudoku RESULTS.md)

The sweep's defensible operating region: moderate curvature K∈[−0.7,−0.3] (aggregate
+5-6pt over K=−1, p<1e-5), LR 3e-4 (best or tied at moderate K on both slices), init
c0.01 (nominal top, within noise). Chosen best cell: **K=−0.5, init=custom std 0.01,
LR=3e-4** (hard global peak; medium near-peak). No single cell beats baseline beyond
seed noise, so the exact cell choice is a convention, not a claim.

## Arms (× difficulty easy/medium/hard × seed 1)

| arm | config | ALPHA_MAX | bound source |
|---|---|---|---|
| `eflm_ta` | E-FLM, ngpt init, lr 3e-4 (defaults) | 0.767 | `alpha_star_euclidean(12)` |
| `hflm_ta_best` | H-FLM, K=−0.5, init=custom 0.01, lr 3e-4 | 0.907 | `alpha_star_numeric.py` (exact hyperboloid; the shipped K=−1 tree bound is invalid at c·ρ₁≈0.16) |
| `hflm_ta_k1` | H-FLM, K=−1.0, init=hyperbolic 0.3, lr 3e-4 (script defaults) | 0.624 | `alpha_star_hyperbolic(12, 512)` |

`hflm_ta_k1` is included because it is the arm with the widest (most meaningful)
truncation window and the exact config whose bound was smoke-verified — the strongest
regression test against the historical collapse. `hflm_ta_best` mainly exercises
adaptive + a mild truncation (its tiny init radius pushes the Voronoi-collapse point
to α≈0.91). Note the numeric-exact bound also says the shipped 0.624 tree value
overestimates its own geometry's bound by ~0.10 (numeric 0.527) — 0.624 is kept since
truncating *less* is the safe direction and it is what the shipped scripts default to.

Everything else fixed, identical to the curvature sweep: tiny DiT (512/8/8, ~28.6M),
20k steps, batch 256, seq 180, bf16, EMA 0.9999, AdamW wd 0, grad clip 1.0,
prior_cov 0.25, rho_max 12; adaptive knobs refit_every=50, buffer=50×256, ema=0.9,
uniform_mix=1e-3, warmup=1000 (config default).

**Eval** (identical to the curvature sweep): `sudoku_eval`, 180 steps, exact velocity,
greedy last, `top_k_velocity=-1`, 2000 puzzles. The eval scripts
(`scripts/sample/sudoku/{eflm,hflm}_truncated_adaptive.sh`) mirror the training noise
config so the adapted schedule loads from the checkpoint and drives sampling.

## Anchors (same eval protocol, from hflm_curv_init_lr_sudoku)

| anchor | easy | medium | hard |
|---|---|---|---|
| naive E-FLM (`bl_*`, 3-seed mean) | 88.2 | 62.2 | 19.2 |
| naive S-FLM (`bl_*`, 3-seed mean) | — see RESULTS of baseline sweep — | | |
| S-FLM trunc+ada (`bl_*`, 3-seed mean) | 95.0 | 76.7 | 42.2 |
| naive H-FLM K=−0.5 c0.01@3e-4 (3-seed) | n/a | 81.1 | 46.2 |
| naive H-FLM K=−1.0 c0.01@3e-4 (3-seed) | n/a | 72.9 | 40.4 |

(The curvature sweep ran medium/hard only; easy has no per-cell H-FLM anchor. The
K=−1 anchor uses init c0.01, not 0.3 — closest available.)

## GPU allocation & wall clock

9 cells × 1 GPU (unicorn `thickstun,desa`, excl. desa-compute-01; 2 CPU, 16G, 6h
limit). ~2.5–3h train + ~10 min eval per cell ⇒ ~27 GPU·h; wall clock depends on free
GPUs (cluster was ~5/8 busy at submission). Each job dumps the adaptive-schedule state
(`refit_count`, `has_schedule`, adapted alpha range) from `last.ckpt` to
`eval/noise_state.json` (correctness bar #3), then deletes the ~1.8G checkpoints;
`eval/results.json` + `eval/noise_state.json` are the deliverables.

## Launch

```bash
python experiments/trunc_ada_sudoku/sweep.py            # all 9 cells, seed 1
python experiments/trunc_ada_sudoku/sweep.py --seeds 2 3  # optional: more seeds
```

Outputs: `outputs/trunc_ada_sudoku/tas_{arm}_d-{diff}_rs{seed}/` (+ `eval/results.json`),
logs in `experiments/trunc_ada_sudoku/logs/`.

## Round 2 (2026-07-11): tuning HFLM+trunc+ada after the round-1 medium regression

Round 1 found HFLM trunc+ada ≤ naive (medium 68.3 vs 81.1 at the sweep-best cell).
Mechanism hypothesis: on H^d the geodesic's clean-endpoint weight is
sinh(αD)/sinh(D) ≈ e^{−(1−α)D}, so the loss-vs-t profile is a sharp sigmoid — flat
almost everywhere. (i) Truncation has little waste to cut (α⋆=0.907 cuts 9%, vs 91%
on the sphere), and (ii) the adaptive sampler (∝|dL/dt| + uniform_mix) over-concentrates
on the narrow transition band and starves the high-noise region — which on Sudoku is
where solve-from-clues is learned (prompt cells stay clean). S-FLM avoids (ii) because
its aggressive truncation removes the flat region before adaptation.

Arms (medium + hard, seed 1; all K=−0.5, c0.01, lr 3e-4 unless noted):

| arm | what it isolates | ALPHA_MAX | UNIFORM_MIX |
|---|---|---|---|
| `hflm_to_best` | truncation alone (is adaptive the culprit?) | 0.907 | — |
| `hflm_ao_best` | adaptive alone (is truncation the culprit?) | null | 1e-3 |
| `hflm_ta_umix03` | keep 30% uniform mass → high-noise coverage | 0.907 | 0.3 |
| `hflm_ta_k25` | flatter K=−0.25 widens the band (width ~1/(cD)); α⋆ numeric = 0.894; naive anchor 71.1/34.6 | 0.894 | 1e-3 |

Plus `hflm_ta_best` medium seeds 2–3 to confirm the round-1 regression isn't seed noise.
New scripts: `scripts/sample/sudoku/hflm_truncated.sh`; `UNIFORM_MIX` knob added to the
hflm truncated_adaptive train+sample scripts. 10 jobs ≈ 30 GPU·h.

## Round 3 (2026-07-11): EMPIRICAL truncation bound from the measured loss geometry

The measured L(t) profile for this exact config (dev1
`experiments/loss_geometry_vis/sudoku_hard/hflm_K0.5` = K=−0.5, c0.01, 3e-4, hard)
shows the loss leaves ~0 only at t ≳ 0.66 — i.e. the informative band is
**α ∈ [0, ~0.34] at 5K steps**, shrinking to α ≲ 0.2 by 20K. The transformer +
clean prompt cells collapse the posterior far earlier than the single-token
nearest-neighbor model behind `alpha_star_*` predicts (0.907): the geometric bounds
are per-token and context-free, so they are *upper* bounds that are far too loose for
conditional Sudoku. The historical α_max=0.093 collapse is also explained: it cut
*into* the band, not above it. Data-driven arms (K=−0.5, c0.01, 3e-4):

| arm | ALPHA_MAX | seeds | rationale |
|---|---|---|---|
| `hflm_ta_e35` | 0.35 | 1–3 | covers the band across all training phases; 3 seeds because the naive cell carries ±13pt on hard |
| `hflm_to_e35` | 0.35 | 1 | truncation-only at the empirical bound |
| `hflm_ta_e20` | 0.20 | 1 | aggressive late-band-only probe |

Expectation: with the flat region removed the same way S-FLM's truncation removes it,
adaptive should compose correctly and the S-FLM-style gains should appear. Caveat:
the bound is derived from the *hard* profile and reused for medium (bands are similar
in the K-overlay figures); a medium-specific profile would refine it.
