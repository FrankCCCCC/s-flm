# HFLM curvature loss-geometry — poll state

**Goal:** For HFLM (Sudoku, hard), draw loss geometry (L vs flow-time t) — one
curve per Gaussian curvature — using the **best (init, lr)** config of each
curvature. `prior_cov` is **fixed at 0.25** across the sweep (only
`gaussian_curvature = -k` varies), so the free dims are init & lr.

Deliver: per-curvature figures + one overlay (curve per curvature, linear + log)
in `experiments/loss_geometry_vis/hflm_curv/`, and a RESULT.md section.

## Best config per curvature (by sudoku accuracy, from results.json)

Selection metric = `eval/results.json.accuracy` (num_correct / 2000). All 168
hard sweep runs are already evaluated; ranking below is fixed.

| K (=−curvature) | best init | best lr | seed | acc |
|---|---|---|---|---|
| 0.25 | c0.04 | 5e-4 | rs1 | 0.4585 |
| 0.3  | c0.01 | 3e-4 | rs2 | 0.5015 |
| 0.5  | c0.01 | 3e-4 | rs2 | 0.5815 |
| 0.7  | c0.01 | 3e-4 | rs3 | 0.4670 |
| 1.0  | c0.01 | 3e-4 | rs2 | 0.4610 |
| 1.5  | c0.01 | 3e-4 | rs2 | 0.4510 |

## Why the poll exists (checkpoint blocker)

The best config for 5/6 curvatures is seed rs2/rs3, trained under the **ch2263**
account with checkpoints on `/scratch/ch2263` — **not mounted** on this cluster
(confirmed: 0 rs2/rs3 ckpt files anywhere under `/share` or `/home`, which is a
symlink to the same `/share` tree). K=1.5 has no accessible checkpoint at all.
**The user is re-training these best configs.** This poll waits for their
checkpoints, then draws.

## K=0.5 seed=1 checkpoint refresh — DONE (2026-07-09, cron 6962d472 deleted)

User re-trained `d-hard_k-0.5_i-c0.01_lr3e-4_rs1` from scratch → full
5/10/15/20K. K0.5 redrawn with all four steps (L(1) 0.289→0.251→0.245→0.244),
overlay rebuilt, "15/20K only" caveat dropped in RESULT.md + RESULTS.md. All 6
curvatures now have full 4-step figures.

## COMPLETE (2026-07-09)

- **All 6 curvatures drawn (seed rs1), overlay built, RESULT.md finalized, poll
  cron 81ebad32 DELETED.** Final steps: K=0.25/0.5/0.7/1.0/1.5 @ 20K, K=0.3 @ 15K
  (its rs1 run stopped there; no 20K exists). Finding: L(1)≈0.25–0.26 for
  K≤1.0, jumps to 0.436 at K=1.5 (rose from 0.306@5K → unstable at high curvature).
- **Only possible upgrade:** the higher-scoring rs2/rs3 seeds (on ch2263:/scratch)
  — needs user-authorized retrieval. Recipe below still applies if pursued.

## (historical) Progress

- **ALL 6 curvatures drawn (seed rs1)** as of 2026-07-09: K=0.25/0.3/0.5/0.7/1.0
  + K=1.5. Overlay rebuilt with all 6. The user re-ran the rs1 sweep (accessible),
  so figures are best-config @ rs1 (NOT the best-scoring rs2/rs3 seed, gone).

## (historical) earlier progress

- **K=0.25, 0.3, 0.5, 0.7, 1.0 — interim rs1 previews DONE** (drawn 2026-07-08,
  `hflm_curv/K<k>.{png,_log.png,.json}` + `overlay{,_log}.png`). These use the
  best `(init,lr)` config at **seed rs1** (NOT the best-scoring seed rs2/rs3,
  whose ckpts are unreachable), and at **whatever steps `save_top_k=1` retained**
  (inconsistent: K0.5=15/20K, K0.7=5/10K, etc. — no common step, so the overlay
  is a rough preview at each curve's final-available step).
- **K=1.5 — still MISSING** (no accessible checkpoint for its best config at any
  seed). This is the one true gap.
- Tool + sbatch staged: `s-flm/visualization/loss_geometry_curv.py` and
  `s-flm-dev1/.../visualization/loss_geometry_curv.sbatch` (gpu/gpu-high, cd's to
  non-dev1).

## What the poll should do now

1. **K=1.5 (priority):** when its best config `(c0.01, 3e-4)` gains an accessible
   checkpoint (any seed), draw it → `hflm_curv/K1.5`.
2. **Upgrade the previews:** when the user's re-trained best-config checkpoints
   (ideally the best-scoring seed with a full consistent 5/10/15/20K set) become
   accessible for K=0.25/0.3/0.5/0.7/1.0, REDRAW that curvature from the best
   available seed (overwrite the rs1 preview), then rebuild `overlay{,_log}.png`
   at a COMMON step (now possible).
3. When all 6 have figures from consistent checkpoints and the overlay is rebuilt,
   update RESULT.md and **CronDelete** job 81ebad32.

To find candidates, for each curvature's best `(init, lr)` look for an
**accessible run dir with checkpoints** (step ckpts `*-5000/10000/15000/20000.ckpt` and/or
`last.ckpt`). Search under:
- `/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku/d-hard_k-<K>_i-<init>_lr<lr>_rs*/checkpoints/`
- and any new project dir the user may create under `.../outputs/` (glob for
  `*k-<K>*i-<init>*lr<lr>*/checkpoints`).

Ready = the target `(K, init, lr)` has ≥1 accessible checkpoint (prefer the full
5/10/15/20K set; else use `last.ckpt`). Report ready vs pending; only draw the
ones that are ready.

## Draw recipe (when ready)

⚠️ **HFLM needs the non-dev1 `claude/curv` code.** The `gaussian_curvature` knob
exists ONLY in `/share/thickstun/sychou/workspace/research/s-flm/algo.py`
(branch `claude/curv`). dev1's `algo.py` (main) lacks it and would silently build
a K=−1 hyperboloid → **wrong curves**. So run the tool from the **non-dev1 tree**:

1. Copy the corrected dev1 tool into non-dev1 as a NEW file (don't overwrite
   tracked files): `cp .../s-flm-dev1/s-flm/visualization/loss_geometry.py \
   /share/thickstun/sychou/workspace/research/s-flm/visualization/loss_geometry_curv.py`
   → its `REPO` resolves to non-dev1, so `import algo` picks up curvature-aware HFLM.
2. sbatch that `cd`s to non-dev1 and runs `python visualization/loss_geometry_curv.py`,
   on the **gpu partition, gpu-high** (a6000/6000ada, 48 GB, sm_86+):
   ```
   #SBATCH --partition=gpu --constraint=gpu-high --gres=gpu:1 --mem=48gb -t 1:00:00
   export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
   cd /share/thickstun/sychou/workspace/research/s-flm
   python visualization/loss_geometry_curv.py --mode steps \
     --project /share/thickstun/sychou/workspace/research/s-flm/outputs/<proj> \
     --run <run_name> --steps 5000 10000 15000 20000 \
     --out /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/loss_geometry_vis/hflm_curv/K<k>
   ```
   (If only `last.ckpt` exists, add a `--steps`-free path or use the run's final step.)
3. The tool caches each curve to `hflm_curv/K<k>.json`. After all curvatures are
   drawn, write a small local combine script (no GPU) that overlays the 6 JSONs
   into `hflm_curv/overlay.png` + `overlay_log.png` (one curve per curvature, at
   the final checkpoint), labelled `K=-0.25 … K=-1.5`.

## Verify + deliver

- Sudoku is CONDITIONAL (`[BOS] puzzle(89) [BOS] solution(89)`, loss only on
  solution cells). So NO unigram ceiling; L(1) = solve-from-clues loss and should
  fall with training. Read log-Y figures.
- Deliver figures; add an HFLM-curvature section to
  `experiments/loss_geometry_vis/RESULT.md`.
- Then **CronDelete** the poll job.

## Constraints (Agent.md / user)

- Do NOT submit training jobs (user handles training). Do NOT cancel any job;
  only `nice`. Use the shared `gpu` partition to get GPUs fast; never the login
  node for GPU work.

## Cron job id

`81ebad32` — every 2h at :17, session-scoped (dies if the Claude session ends;
re-create if so). Delete with CronDelete once all 6 curvatures are delivered.
