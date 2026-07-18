#!/usr/bin/env python
"""ehflm_trunc_ada_sudoku — {E,H}-FLM x {trunc, trunc+ada} x curvature x init x LR on Sudoku.

Full-grid follow-up to experiments/trunc_ada_sudoku (which tuned trunc+ada at a
few cells): rerun the naive hflm_curv_init_lr_sudoku axes under truncated and
truncated+adaptive noise schedules (setup: setup.md; design: EXPERIMENT.md).

Grid (576 cells):
  method       : to (truncated), ta (truncated + adaptive)                   (2)
  geometry     : E-FLM init=ngpt (tiny-sphere-dit default), no curvature     (1)
                 H-FLM K {-0.25,-0.3,-0.5,-0.7,-1.0}
                       x init {random(std .02), custom .01, custom .04}     (15)
  lr           : {3e-4, 5e-4, 1e-3}                                          (3)
  difficulty   : {medium, hard}                                              (2)
  seed         : {1, 2, 3}  (report mean +- std)                             (3)
Fixed: tiny DiT (512/8/8), 20k steps, batch 256, seq 180, bf16, EMA 0.9999,
AdamW wd=0; H-FLM prior_cov=0.25, rho_max=12.  Eval: sudoku_eval, 180 steps,
exact velocity, greedy last step, top_k_velocity=-1 — identical protocol to
hflm_curv_init_lr_sudoku, whose all_results.csv provides the naive anchors.

ALPHA_MAX = each geometry's own truncation bound (per trunc_ada_sudoku/RESULTS.md:
per-geometry alpha* + adaptive was the winning recipe, e.g. 0.894 at K=-0.25/c0.01;
fixed tight bounds like 0.35 are sampler-fatal on H^d, so never truncate tighter
than the geometry's alpha*):
  E-FLM : 0.767 = alpha_star_euclidean(12)          (noise_schedules.py)
  H-FLM : alpha_star_hyperbolic_numeric(12, 512, embed_std, K)
          (experiments/trunc_ada_sudoku/alpha_star_numeric.py; embed_std for
           INIT=random is 0.02)

ORCHESTRATION ONLY — each cell calls the single-run scripts
  scripts/train/sudoku/{eflm,hflm}_truncated{,_adaptive}.sh
  scripts/sample/sudoku/{eflm,hflm}_truncated{,_adaptive}.sh
Idempotent + resumable: skips a cell whose eval/results.json exists or whose job
name is already in squeue; resubmitting the same OUTPUT_DIR auto-resumes from
last.ckpt.  No nice here — submit, then tier priorities with
  scontrol update jobid=<ids> nice=<n>     (rs1 0 / rs2 2000 / rs3 4000)

Usage:  python experiments/ehflm_trunc_ada_sudoku/sweep.py [--dry-run]
          [--models eflm hflm] [--methods to ta]
          [--difficulties medium hard] [--seeds 1 2 3]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

METHODS = {'to': 'truncated', 'ta': 'truncated_adaptive'}

# (K, init_tag) -> alpha_star_hyperbolic_numeric(12, 512, embed_std, K)
# with prior_cov=0.25, rho_max=12, delta=0.1 (anchors match trunc_ada_sudoku:
# 0.894 @ -0.25/c0.01, 0.897 @ -0.3/c0.01, 0.907 @ -0.5/c0.01).
ALPHA_HFLM = {
    ('-0.25', 'c0.01'): '0.8940', ('-0.25', 'random'): '0.8389', ('-0.25', 'c0.04'): '0.7695',
    ('-0.3',  'c0.01'): '0.8973', ('-0.3',  'random'): '0.8452', ('-0.3',  'c0.04'): '0.7796',
    ('-0.5',  'c0.01'): '0.9067', ('-0.5',  'random'): '0.8624', ('-0.5',  'c0.04'): '0.8053',
    ('-0.7',  'c0.01'): '0.9128', ('-0.7',  'random'): '0.8729', ('-0.7',  'c0.04'): '0.8202',
    ('-1.0',  'c0.01'): '0.9192', ('-1.0',  'random'): '0.8834', ('-1.0',  'c0.04'): '0.8340',
}
ALPHA_EFLM = '0.767'  # alpha_star_euclidean(12); ngpt init ||e|| ~= 1

# (init_tag, INIT, INIT_STD) — INIT_STD='null' for non-custom modes
INITS = [('random', 'random', 'null'),
         ('c0.01', 'custom', '0.01'), ('c0.04', 'custom', '0.04')]
KS = ['-0.25', '-0.3', '-0.5', '-0.7', '-1.0']
LRS = ['3e-4', '5e-4', '1e-3']
DIFFICULTIES = ['medium', 'hard']
SEEDS = ['1', '2', '3']

# (model, k, init_tag, INIT, INIT_STD, alpha_max)
GEOMETRIES = [('eflm', None, None, None, None, ALPHA_EFLM)] + [
    ('hflm', k, itag, init, istd, ALPHA_HFLM[(k, itag)])
    for k, (itag, init, istd) in itertools.product(KS, INITS)]


def tag_of(model, method, k, itag, lr, diff, seed):
    geo = '' if model == 'eflm' else f'_k{k}_i-{itag}'
    return f'{model}-{method}{geo}_lr{lr}_d-{diff}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(model, method, k, init, istd, lr, alpha, tdir, diff, seed):
    stem = f'{model}_{METHODS[method]}'
    geo = '' if model == 'eflm' else f'GAUSS_CURV={k} INIT={init} INIT_STD={istd} '
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN on $(hostname)"
        {geo}LR={lr} ALPHA_MAX={alpha} DIFFICULTY={diff} SEED={seed} \\
            OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{stem}.sh
        echo "[$(date)] EVAL"
        {geo}ALPHA_MAX={alpha} DIFFICULTY={diff} SEED={seed} TOPK_VELOCITY=-1 \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{stem}.sh
        # checkpoints are transient bulk (~1.8G/cell): once the eval deliverable
        # exists, drop them to keep /share within quota
        if [ -f {tdir}/eval/results.json ]; then
            rm -rf {tdir}/checkpoints && echo "[$(date)] checkpoints cleaned"
        fi
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--models', nargs='+', default=['eflm', 'hflm'],
                    choices=['eflm', 'hflm'])
    ap.add_argument('--methods', nargs='+', default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument('--difficulties', nargs='+', default=DIFFICULTIES,
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/ehflm_trunc_ada_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/ehflm_trunc_ada_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = [(model, method, k, itag, init, istd, alpha, lr, diff, seed)
             for (model, k, itag, init, istd, alpha), method, lr, diff, seed
             in itertools.product(GEOMETRIES, args.methods, LRS,
                                  args.difficulties, args.seeds)
             if model in args.models]

    n_geo = len(set((c[0], c[2], c[3]) for c in cells))
    print(f'ehflm_trunc_ada_sudoku: {len(cells)} cells '
          f'({n_geo} geometry x {len(args.methods)} method x {len(LRS)} lr x '
          f'{len(args.difficulties)} difficulty x {len(args.seeds)} seed)')
    if args.dry_run:
        for model, method, k, itag, init, istd, alpha, lr, diff, seed in cells:
            print(f'  eh_{tag_of(model, method, k, itag, lr, diff, seed)}'
                  f'  alpha_max={alpha}')
        model, method, k, itag, init, istd, alpha, lr, diff, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(model, method, k, init, istd, lr, alpha,
                         f'{out}/{tag_of(model, method, k, itag, lr, diff, seed)}',
                         diff, seed))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for model, method, k, itag, init, istd, alpha, lr, diff, seed in cells:
        tag = tag_of(model, method, k, itag, lr, diff, seed)
        jobname = f'eh_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=4, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(model, method, k, init, istd, lr, alpha,
                                    f'{out}/{tag}', diff, seed),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
