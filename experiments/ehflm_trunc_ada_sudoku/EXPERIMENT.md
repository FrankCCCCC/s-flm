# ehflm_trunc_ada_sudoku — E/H-FLM x {trunc, trunc+ada} x curvature x init x LR on Sudoku

Full-grid follow-up to `experiments/trunc_ada_sudoku` (which validated the
trunc+ada implementation and tuned it at a few cells around the naive optimum).
This project reruns the naive `hflm_curv_init_lr_sudoku` axes under **truncated**
(`to`) and **truncated+adaptive** (`ta`) noise schedules, so every cell has a
naive anchor at the identical eval protocol. Setup spec: `setup.md`.

## Hypotheses

1. **E-FLM**: trunc+ada's gains (+16.7 med / +25.8 hard over naive at lr 3e-4,
   trunc_ada_sudoku round 1) are robust across LR. The trunc-only arm decomposes
   the driver on flat space — never measured (on H^d truncation alone *costs*
   ~6–10 pts; Euclidean space may behave like the sphere, where the alpha* bound
   is decode-consistent).
2. **H-FLM curvature shift**: trunc+ada moved the curvature optimum flatter
   (−0.25 vs naive −0.5 at c0.01/3e-4). Test whether this replicates across
   init and LR.
3. **Variance regularization**: trunc+ada halved seed-std at its best cells;
   test across the grid (3 seeds per cell).
4. **Init-std interaction**: larger embedding std tightens the bound
   (alpha* at std 0.04: 0.77–0.83). Prior evidence says tighter truncation hurts
   H^d monotonically — expect `to` arms at c0.04 to degrade vs naive anchors
   unless adaptation rescues them.

## Grid (576 cells; `sweep.py`)

| axis | values | n |
|---|---|---|
| method | to (truncated), ta (truncated+adaptive) | 2 |
| geometry | E-FLM: init ngpt, flat. H-FLM: K {−0.25,−0.3,−0.5,−0.7,−1.0} × init {random(std .02), custom .01, custom .04} | 1 + 15 |
| lr | 3e-4, 5e-4, 1e-3 | 3 |
| difficulty | medium (35 clues), hard (30 clues) | 2 |
| seed | 1, 2, 3 (report mean ± std) | 3 |

Fixed: tiny DiT (512/8/8, ~28.6M), 20k steps, batch 256, seq 180, bf16,
EMA 0.9999, AdamW wd=0 betas=(0.9,0.999) eps=1e-8 clip=1.0; H-FLM
prior_cov=0.25, rho_max=12; noise log-linear{,-adaptive} (adaptive:
refit_every=50, buffer 50×256, ema 0.9, uniform_mix 1e-3).
Data: Sudoku 48k train / 2k val per difficulty (data seed 42).

**Eval** (identical to hflm_curv_init_lr_sudoku): sudoku_eval on the 2k val
puzzles, 180 steps, exact velocity, top_k_velocity=-1, greedy last step.
Single-shot headline protocol only.

## ALPHA_MAX per geometry

Per `trunc_ada_sudoku/RESULTS.md`: use each geometry's own alpha* bound
(the winning recipe; anything tighter is sampler-fatal on H^d).

- E-FLM: **0.767** = `alpha_star_euclidean(12)` (noise_schedules.py).
- H-FLM: `alpha_star_hyperbolic_numeric(12, 512, embed_std, K)`
  (`../trunc_ada_sudoku/alpha_star_numeric.py`; prior_cov 0.25, rho_max 12;
  INIT=random ⇒ embed_std 0.02):

| K \ init | c0.01 | random (0.02) | c0.04 |
|---|---|---|---|
| −0.25 | 0.8940 | 0.8389 | 0.7695 |
| −0.3  | 0.8973 | 0.8452 | 0.7796 |
| −0.5  | 0.9067 | 0.8624 | 0.8053 |
| −0.7  | 0.9128 | 0.8729 | 0.8202 |
| −1.0  | 0.9192 | 0.8834 | 0.8340 |

## Anchors

- H-FLM naive: `../hflm_curv_init_lr_sudoku/all_results.csv` (every K/init/lr/
  difficulty cell of this grid exists there; same eval protocol).
- E-FLM naive: 88.2 / 62.2 / 19.2 easy/med/hard at lr 3e-4 (trunc_ada_sudoku).

## GPU allocation & wall clock

576 cells × 1 GPU (unicorn `thickstun,desa`, excl. `desa-compute-01`;
4 CPU, 16G, 6h limit, `--requeue`). ~2.5–3h train + ~10 min eval per cell
⇒ ~1.6k GPU·h; queue shares the partitions with the hcil easy sweep
(~453 jobs pending at submission). Checkpoints are deleted after a
successful eval (~1.8G/cell otherwise).

Priority (nice, applied post-submit via `scontrol update`): rs1 **0**,
rs2 **2000**, rs3 **4000** — seed 1 delivers the full grid first; tiers
dominate the age factor (PriorityWeightAge=1000).

## Deliverables

- `outputs/ehflm_trunc_ada_sudoku/{tag}/eval/results.json` per cell,
  tag = `{model}-{method}[_k{K}_i-{init}]_lr{lr}_d-{difficulty}_rs{seed}`.
- `collect.py` → `all_results.csv` + per-config mean±std table.
- `RESULTS.md`: headline tables (config × difficulty, to/ta vs naive anchor),
  hypothesis verdicts, best-recipe recommendation.
