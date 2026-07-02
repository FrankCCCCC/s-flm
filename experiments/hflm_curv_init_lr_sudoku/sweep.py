#!/usr/bin/env python
"""hflm_curv_init_lr_sudoku — H-FLM curvature x init x LR x difficulty x seed grid on Sudoku.

Grid (1008 cells; spec slides/jul02_2026):
  gaussian_curvature : {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}                (6)
  init               : ngpt, random, custom std {0.01,0.02,0.04,0.06,0.08}  (7)
  lr                 : {1e-4, 3e-4, 5e-4, 1e-3}                             (4)
  difficulty         : {medium, hard}                                       (2)
  seed               : {1, 2, 3}  (report the average)                      (3)
Fixed: tiny-hyperbolic-dit (512/8/8), 20k steps, batch 256, seq 180, bf16, EMA 0.9999,
AdamW wd=0 betas=(0.9,0.999) eps=1e-8 clip=1.0, prior_cov=0.25, rho_max=12,
noise=log-linear.  Eval: sudoku_eval, 180 steps, exact velocity, greedy,
top_k_velocity=-1 (avg across vocab; NOTE: differs from hflm_curv_sudoku's top-1).

All jobs submitted with identical priority (sbatch --nice=0).
Re-tier later with:  scontrol update jobid=<id> nice=<n>

Three sites, static DISJOINT split on the curvature axis (tc and falcon share the
ARC /home filesystem — same repo + outputs — so their K sets must not overlap or
in-flight cells get duplicated; results-based skipping works across them):
  --site unicorn : K in {-0.25, -0.5}  (336 cells; partition thickstun,desa)
  --site tc      : K in {-0.3, -0.7}   (336 cells; TinkerCliffs a100+h200 queues)
  --site falcon  : K in {-1.0, -1.5}   (336 cells; Falcon l40s+a30 queues)
Run this script ON the submitting cluster (sc3379@unicorn / shengyenc@tinkercliffs
or falcon). Rsync finished results between unicorn and ARC before (re)submitting so
idempotency sees cross-site completions.

ORCHESTRATION ONLY — each cell calls the single-run shared scripts:
  scripts/train/sudoku/hflm.sh   (DIFFICULTY/GAUSS_CURV/INIT/INIT_STD/LR/SEED knobs)
  scripts/sample/sudoku/hflm.sh  (same + CKPT_PATH/TOPK_VELOCITY knobs)
Idempotent + resumable (skips done/queued cells; resubmits auto-resume from last.ckpt).

Usage:  python sweep.py --site {unicorn,tc,falcon} [--difficulties medium hard]
                        [--seeds 1 2 3] [--dry-run]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

SITES = {
    'unicorn': dict(
        repo='/share/thickstun/sychou/workspace/research/s-flm',
        envbin='/home/sc3379/anaconda3/envs/sfm/bin',
        # 3-site disjoint 2/2/2 K split (rebalanced 2026-07-02 from live queue check:
        # unicorn CPU/mem-saturated by other users; ARC queues empty-pending with much
        # larger pools). If a queue starves, widen a list — idempotency skips
        # rsync-synced done cells.
        curvatures=['-0.25', '-0.5'],
        # 4 CPU / 16G (vs 8/32): fits the CPU-saturated kuleshov nodes and the
        # mem-squeezed thickstun node; training is GPU-bound so this costs ~nothing
        partitions=['thickstun,desa'],  # same-cluster comma-list is fine on unicorn
        slurm=dict(exclude='desa-compute-01',
                   gres='gpu:1', ntasks=1, cpus_per_task=4, mem='16G',
                   time='06:00:00'),
        wandb_offline=False,
    ),
    'tc': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        curvatures=['-0.3', '-0.7'],
        # ARC rejects multi-partition submissions (per-partition QOS), so cells
        # round-robin across these single queues (fast-starting first; preemption
        # costs <=~40 min of redone work: ckpt-5k + --requeue + auto-resume).
        partitions=['h200_preemptable_q', 'a100_preemptable_q',
                    'h200_normal_q', 'a100_normal_q'],
        slurm=dict(account='swan_research_dlm',
                   gres='gpu:1', ntasks=1, cpus_per_task=4, mem='32G',
                   time='06:00:00'),
        wandb_offline=True,
    ),
    'falcon': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        curvatures=['-1.0', '-1.5'],
        # Falcon has NO a100 queue: L40S-48GB (l40s_*) and A30-24GB (a30_*) are the
        # bf16+flash-attn-capable options (V100/T4 are not) — see Agent.md.
        # Round-robin single queues (ARC rejects multi-partition; per-partition QOS),
        # fast-starting first; preemption cost <=~40 min (ckpt-5k + --requeue).
        partitions=['a30_preemptable_q', 'l40s_preemptable_q',
                    'a30_normal_q', 'l40s_normal_q'],
        slurm=dict(account='swan_research_dlm',
                   gres='gpu:1', ntasks=1, cpus_per_task=4, mem='32G',
                   time='06:00:00'),
        wandb_offline=True,
    ),
}

# (tag, INIT, INIT_STD) — INIT_STD='null' for non-custom modes
INITS = [('ngpt', 'ngpt', 'null'), ('random', 'random', 'null')] + [
    (f'c{s}', 'custom', s) for s in ['0.01', '0.02', '0.04', '0.06', '0.08']]
LRS = ['1e-4', '3e-4', '5e-4', '1e-3']
SEEDS = ['1', '2', '3']


def tag_of(k, init_tag, lr, difficulty='medium', seed=1):
    # run-name format: d-{difficulty}_k-{K}_i-{init}_lr{lr}_rs{seed}
    # (all pre-seed-axis outputs renamed to this on 2026-07-02 via rename_runs.py)
    return f'd-{difficulty}_k{k}_i-{init_tag}_lr{lr}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(site, k, init, init_std, lr, tdir, difficulty='medium', seed='1'):
    offline = 'export WANDB_MODE=offline\n' if site['wandb_offline'] else ''
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={site['envbin']}:$PATH
        {offline}cd {site['repo']}
        echo "[$(date)] TRAIN on $(hostname)"
        DIFFICULTY={difficulty} GAUSS_CURV={k} INIT={init} INIT_STD={init_std} LR={lr} \\
            SEED={seed} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/hflm.sh
        echo "[$(date)] EVAL"
        DIFFICULTY={difficulty} GAUSS_CURV={k} INIT={init} INIT_STD={init_std} \\
            SEED={seed} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval \\
            DEVICES=1 TOPK_VELOCITY=-1 \\
            bash scripts/sample/sudoku/hflm.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=SITES)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--difficulties', nargs='+', default=['medium', 'hard'],
                    choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    ap.add_argument('--curvatures', nargs='+', default=None,
                    help="override the site's K list (for live rebalancing; "
                         'keep site K sets disjoint across tc/falcon)')
    args = ap.parse_args()
    site = SITES[args.site]
    if args.curvatures:
        site = dict(site, curvatures=args.curvatures)
    exp = f"{site['repo']}/experiments/hflm_curv_init_lr_sudoku"
    logs = f'{exp}/logs'
    out = f"{site['repo']}/outputs/hflm_curv_init_lr_sudoku"
    nice = 0
    if not args.dry_run:  # site repo path is only writable on the submitting cluster
        os.makedirs(logs, exist_ok=True)

    # (k, init_tag, INIT, INIT_STD, lr, difficulty, seed)
    cells = [(k, itag, init, istd, lr, diff, seed)
             for k, (itag, init, istd), lr, diff, seed
             in itertools.product(site['curvatures'], INITS, LRS,
                                  args.difficulties, args.seeds)]

    print(f'hflm_curv_init_lr_sudoku [{args.site}]: {len(cells)} cells '
          f'({len(site["curvatures"])} K x {len(set(c[1] for c in cells))} init x '
          f'{len(set(c[4] for c in cells))} lr x {len(args.difficulties)} difficulty x '
          f'{len(args.seeds)} seed)')
    if args.dry_run:
        for k, itag, init, istd, lr, diff, seed in cells:
            
            print(f'  hcil_{tag_of(k, itag, lr, diff, seed)}  nice={nice}')
        k, itag, init, istd, lr, diff, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(site, k, init, istd, lr,
                         f'{out}/{tag_of(k, itag, lr, diff, seed)}', diff, seed))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for k, itag, init, istd, lr, diff, seed in cells:
        tag = tag_of(k, itag, lr, diff, seed)
        jobname = f'hcil_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        # --nice on the command line: simple_slurm's '#SBATCH --nice N' directive
        # is rejected by sbatch (--nice takes an optional arg, needs '=' form)
        part = site['partitions'][n_sub % len(site['partitions'])]
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition=part, **site['slurm'])
        jid = slurm.sbatch(job_body(site, k, init, istd, lr, f'{out}/{tag}', diff, seed),
                           sbatch_cmd=f'sbatch --nice={nice} --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
