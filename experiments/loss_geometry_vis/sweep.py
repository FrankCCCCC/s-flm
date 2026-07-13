#!/usr/bin/env python
"""Reproduce every figure under experiments/loss_geometry_vis/.

One SLURM job per config (via simple_slurm). Each job runs the loss-geometry
tool over the 3 x-axes (`t`, `xt_norm`, `riem_dist`), producing the linear + log
figures (with the target-direction arrow) and the cached `.json`, written to
`experiments/loss_geometry_vis/{dataset}/{run_folder}/`. HFLM configs add a
second y-metric, `word_loss_std` (across-vocab distribution of each word's mean
denoising loss, the per-token NLL bucketed by clean target x_0: mean line + std
band + min/max whiskers; `_wordstd` figure suffix; the metric is HFLM-only). An
HFLM cache without per-word stats (no `word_stats` key) is deleted at job start
so the curves are recomputed once.

The heavy work (pin flow-time t on a grid, then evaluate the algo's own `_loss`
on the val split at each t) lives in `visualization/loss_geometry.py`; this file
only builds the grid of configs and submits. The first `--x-axis t` invocation
computes the curves + |x_t| norms + Riemannian distances d(x_t, x_0) from the
checkpoints and caches them; the `--x-axis xt_norm` / `--x-axis riem_dist`
invocations reuse that cache. So a run against a clean output dir reproduces the
results from the checkpoints.

Orchestration only -- it never inlines `python -m main`. Idempotent: a config
whose final figure (`<out>_xtnorm_log.png` + `<out>.json`) already exists, or
whose job name is already in `squeue`, is skipped unless --force.

Every config uses the dev1 `loss_geometry.py`: dev1's `main` now carries the
`gaussian_curvature` knob (its HFLM class + `geo_bridge.py` are byte-identical to
the `claude/curv` worktree), so it loads the HFLM-curvature checkpoints exactly
and computes their hyperbolic d(x_t, x_0) at each run's config curvature. The
`curv` tool path below is retained for provenance but no config selects it.

Usage:
  python experiments/loss_geometry_vis/sweep.py             # submit missing configs
  python experiments/loss_geometry_vis/sweep.py --force     # recompute all (clear caches)
  python experiments/loss_geometry_vis/sweep.py --dry-run   # print, do not submit
  python experiments/loss_geometry_vis/sweep.py --only hflm_K0.5 eflm
  python experiments/loss_geometry_vis/sweep.py --local     # run job bodies here (needs a GPU)
"""
import argparse
import os
import subprocess

from simple_slurm import Slurm

DEV1 = '/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm'   # dev1 (main) tool
SFLM = '/share/thickstun/sychou/workspace/research/s-flm'              # all checkpoints live here
# HFLM-curvature runs need the claude/curv code (gaussian_curvature; no cartesian_model
# requirement). The main SFLM tree has since moved to claude/dflm, so we pin a detached
# git worktree at the figure-drawing commit f3949a3:
#   git -C s-flm worktree add --detach ../s-flm-curv f3949a3
#   cp visualization/loss_geometry.py ../s-flm-curv/visualization/loss_geometry_curv.py
SFLM_CURV = '/share/thickstun/sychou/workspace/research/s-flm-curv'
OUTD = f'{DEV1}/experiments/loss_geometry_vis'
LOGS = f'{OUTD}/logs'
OUT_TS = f'{SFLM}/outputs'                          # tinystories project dirs
SUD = f'{SFLM}/outputs/hflm_curv_init_lr_sudoku'    # sudoku baselines + curvature

ENV_BIN = '/home/sc3379/anaconda3/envs/sfm/bin'     # prepend to PATH (robust: no `source`/`conda init`)
PARTITION, CONSTRAINT = 'gpu', 'gpu-high'           # 48GB, sm_86+ (safe for the cu128 build)

TS_STEPS = '5000 20000 30000'          # tinystories: max_steps 30k
SUD_STEPS = '5000 10000 15000 20000'   # sudoku: max_steps 20k

# name | dataset | out (folder/stem, under OUTD/dataset) | project | run | steps | tool
#   tool 'main' -> DEV1/visualization/loss_geometry.py
#   tool 'curv' -> SFLM/visualization/loss_geometry_curv.py  (curvature-aware HFLM)
CONFIGS = [
  # ---- TinyStories (small-*, gpt2 tokenizer, steps 5/20/30K) ----
  ('eflm_naive_geo',       'tinystories', 'eflm_naive_geo/eflm_naive_geo',             f'{OUT_TS}/naive_geo_tinystories_s256', 'eflm',                 TS_STEPS, 'main'),
  ('hflm_std0.04_pc1.0',   'tinystories', 'hflm_std0.04_pc1.0/hflm_std0.04_pc1.0',     f'{OUT_TS}/hflm_sweep_tinystories_s256', 'std0.04_pc1.0',       TS_STEPS, 'main'),
  ('lf_ada_lr1e-3',        'tinystories', 'lf_ada_lr1e-3/lf_ada_lr1e-3',               f'{OUT_TS}/adv_geo_tinystories_s256', 'lf_ada_lr1e-3',          TS_STEPS, 'main'),
  ('lf_ada_sc_lr1e-3',     'tinystories', 'lf_ada_sc_lr1e-3/lf_ada_sc_lr1e-3',         f'{OUT_TS}/adv_geo_tinystories_s256', 'lf_ada_sc_lr1e-3',       TS_STEPS, 'main'),
  ('sfm_ada_lr1e-3',       'tinystories', 'sfm_ada_lr1e-3/sfm_ada_lr1e-3',             f'{OUT_TS}/adv_geo_tinystories_s256', 'sfm_ada_lr1e-3',         TS_STEPS, 'main'),
  ('sfm_ada_trunc_lr1e-3', 'tinystories', 'sfm_ada_trunc_lr1e-3/sfm_ada_trunc_lr1e-3', f'{OUT_TS}/adv_geo_tinystories_s256', 'sfm_ada_trunc_lr1e-3',   TS_STEPS, 'main'),
  ('sfm_trunc_lr1e-3',     'tinystories', 'sfm_trunc_lr1e-3/sfm_trunc_lr1e-3',         f'{OUT_TS}/adv_geo_tinystories_s256', 'sfm_trunc_lr1e-3',       TS_STEPS, 'main'),
  # ---- Sudoku-hard baselines (tiny-sphere-dit, seed 1, steps 5/10/15/20K) ----
  ('sfm',           'sudoku_hard', 'sfm/sfm',                     SUD, 'bl_d-hard_a-sfm_rs1',           SUD_STEPS, 'main'),
  ('sfm_trunc',     'sudoku_hard', 'sfm_trunc/sfm_trunc',         SUD, 'bl_d-hard_a-sfm_trunc_rs1',     SUD_STEPS, 'main'),
  ('sfm_trunc_ada', 'sudoku_hard', 'sfm_trunc_ada/sfm_trunc_ada', SUD, 'bl_d-hard_a-sfm_trunc_ada_rs1', SUD_STEPS, 'main'),
  ('eflm',          'sudoku_hard', 'eflm/eflm',                   SUD, 'bl_d-hard_a-eflm_rs1',          SUD_STEPS, 'main'),
  ('langflow_ada',  'sudoku_hard', 'langflow_ada/langflow_ada',   SUD, 'bl_d-hard_a-langflow_ada_rs1',  SUD_STEPS, 'main'),
  ('langflow_full', 'sudoku_hard', 'langflow_full/langflow_full', SUD, 'bl_d-hard_a-langflow_full_rs1', SUD_STEPS, 'main'),
  # ---- Sudoku-hard HFLM curvature (tiny-hyperbolic-dit; best init/lr per K; curv tool) ----
  ('hflm_K0.25', 'sudoku_hard', 'hflm_K0.25/K0.25', SUD, 'd-hard_k-0.25_i-c0.04_lr5e-4_rs1', SUD_STEPS, 'main'),
  ('hflm_K0.3',  'sudoku_hard', 'hflm_K0.3/K0.3',   SUD, 'd-hard_k-0.3_i-c0.01_lr3e-4_rs1',  SUD_STEPS, 'main'),
  ('hflm_K0.5',  'sudoku_hard', 'hflm_K0.5/K0.5',   SUD, 'd-hard_k-0.5_i-c0.01_lr3e-4_rs1',  SUD_STEPS, 'main'),
  ('hflm_K0.7',  'sudoku_hard', 'hflm_K0.7/K0.7',   SUD, 'd-hard_k-0.7_i-c0.01_lr3e-4_rs1',  SUD_STEPS, 'main'),
  ('hflm_K1.0',  'sudoku_hard', 'hflm_K1.0/K1.0',   SUD, 'd-hard_k-1.0_i-c0.01_lr3e-4_rs1',  SUD_STEPS, 'main'),
  ('hflm_K1.5',  'sudoku_hard', 'hflm_K1.5/K1.5',   SUD, 'd-hard_k-1.5_i-c0.01_lr3e-4_rs1',  SUD_STEPS, 'main'),
]


def out_prefix(dataset, out):
  return f'{OUTD}/{dataset}/{out}'


def job_body(name, dataset, out, proj, run, steps, tool):
  repo = DEV1 if tool == 'main' else SFLM_CURV
  script = ('visualization/loss_geometry.py' if tool == 'main'
            else 'visualization/loss_geometry_curv.py')
  out_abs = out_prefix(dataset, out)
  wordstd = name.startswith('hflm')  # per-word loss y-metric: HFLM only
  # HFLM: a cache without per-word stats lacks 'word_stats' -> delete it so the
  # first (x-axis t, y-metric loss) call recomputes the curves once (the
  # per-word stats come free).
  stale = (f'''python -c "import json, os; p = '{out_abs}.json'; os.path.exists(p) and 'word_stats' not in json.load(open(p)) and os.remove(p)"
''' if wordstd else '')
  return f'''export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export PATH={ENV_BIN}:$PATH
cd {repo}
{stale}for XAXIS in t xt_norm riem_dist; do
  for YMETRIC in {'loss word_loss_std' if wordstd else 'loss'}; do
    python {script} --mode steps --project {proj} --run {run} \\
      --steps {steps} --x-axis $XAXIS --out {out_abs} --y-metric $YMETRIC
  done
done
echo LOSS_GEOMETRY_DONE'''


def is_done(name, dataset, out):
  p = out_prefix(dataset, out)
  done = (os.path.exists(f'{p}_riemdist_log.png')
          and os.path.exists(f'{p}.json'))
  if name.startswith('hflm'):  # word_loss_std figures required for HFLM only
    done = done and os.path.exists(f'{p}_wordstd_riemdist_log.png')
  return done


def is_queued(job_name):
  q = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-n', job_name,
                      '-h', '-o', '%i'], capture_output=True, text=True)
  return bool(q.stdout.strip())


def clear(dataset, out):
  p = out_prefix(dataset, out)
  for suf in ('.json', '.png', '_log.png', '_xtnorm.png', '_xtnorm_log.png',
              '_riemdist.png', '_riemdist_log.png',
              '_wordstd.png', '_wordstd_log.png', '_wordstd_xtnorm.png',
              '_wordstd_xtnorm_log.png', '_wordstd_riemdist.png',
              '_wordstd_riemdist_log.png'):
    try:
      os.remove(f'{p}{suf}')
    except FileNotFoundError:
      pass


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--force', action='store_true',
                  help='clear cached json + figures and recompute all selected configs')
  ap.add_argument('--dry-run', action='store_true', help='print jobs, do not submit')
  ap.add_argument('--only', nargs='+', default=None, help='subset of config names')
  ap.add_argument('--local', action='store_true',
                  help='run each job body sequentially on this machine '
                       '(compute node with a GPU) instead of sbatch')
  args = ap.parse_args()

  os.makedirs(LOGS, exist_ok=True)
  cfgs = [c for c in CONFIGS if args.only is None or c[0] in args.only]

  submitted = skipped = 0
  for name, dataset, out, proj, run, steps, tool in cfgs:
    job_name = f'lossgeo_{dataset}_{name}'
    body = job_body(name, dataset, out, proj, run, steps, tool)
    if args.dry_run:  # print only -- never touch files or the queue
      print(f'\n=== {job_name} ===\n{body}'); submitted += 1; continue
    if args.force:
      clear(dataset, out)
    elif is_done(name, dataset, out):
      print(f'skip (done):   {name}'); skipped += 1; continue
    if args.local:  # same body sbatch would run, on this machine's GPU
      log = f'{LOGS}/{dataset}_{name}_local.log'
      print(f'running locally: {name} (log: {log})', flush=True)
      with open(log, 'w') as f:
        rc = subprocess.run(['bash', '-c', body], stdout=f,
                            stderr=subprocess.STDOUT).returncode
      print('done:' if rc == 0 else f'FAILED (rc={rc}):', name)
      submitted += 1; continue
    if is_queued(job_name):
      print(f'skip (queued): {name}'); skipped += 1; continue
    slurm = Slurm(job_name=job_name, partition=PARTITION, constraint=CONSTRAINT,
                  gres='gpu:1', nodes=1, ntasks=1, cpus_per_task=4, mem='48G',
                  time='01:00:00', output=f'{LOGS}/{dataset}_{name}_%j.log')
    jid = slurm.sbatch(body)
    print(f'submitted {name} -> job {jid}'); submitted += 1

  print(f'\n{submitted} '
        f'{"planned" if args.dry_run else "ran" if args.local else "submitted"}, '
        f'{skipped} skipped')


if __name__ == '__main__':
  main()
