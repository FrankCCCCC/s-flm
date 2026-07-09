#!/usr/bin/env python
"""retrain_ckpts.py — recover the seed=1 / hard-difficulty CHECKPOINTS.

The sweeps auto-deleted every checkpoint after eval (disk quota), so the weights
are gone; this retrains the seed=1 / hard cells with retention ON (train only, no
eval, no cleanup) so the model weights land back on disk.

Scope (175 cells):
  baseline family (7): ar, sfm, sfm_trunc, sfm_trunc_ada, eflm, langflow_ada, langflow_full
  hflm family (168):   K(6) x init(7) x LR(4)   [gaussian-curvature grid]
All at difficulty=hard, seed=1.

seed=1 is the config DEFAULT (configs/config.yaml), so the bare baseline train
scripts reproduce it directly. H-FLM needs per-cell curvature/init/LR, injected via
the canonical scripts/train/sudoku/hflm.sh env-var knobs (GAUSS_CURV/INIT/INIT_STD/
LR/SEED) per Agent.md's one-script-per-method convention.

Orchestration only (Agent.md): each cell CALLS its method's train script. Checkpoints
are restored into the original run dirs, outputs/hflm_curv_init_lr_sudoku/{tag}/
checkpoints/last.ckpt (same layout Agent.md prescribes), alongside the surviving
eval/results.json. Final home = unicorn /share; Falcon-trained H-FLM checkpoints are
rsync-gathered to unicorn afterwards.

Idempotent: skips a cell whose checkpoints/last.ckpt already exists or whose job is
queued; resubmitting resumes training from last.ckpt.

Usage: python retrain_ckpts.py --site {unicorn,falcon} --family {baseline,hflm}
                               [--algos ...] [--curvatures ...] [--dry-run]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

CKPT_SUBDIR = 'outputs/hflm_curv_init_lr_sudoku'  # restore into the original run dirs

# baseline algo -> (train script stem, extra env)   [reverted bare scripts; seed=1 default]
BASELINES = {
    'ar':            ('ar', {}),
    'sfm':           ('sfm', {}),
    'sfm_trunc':     ('sfm_truncated', {}),
    'sfm_trunc_ada': ('sfm_truncated_adaptive', {}),
    'eflm':          ('eflm', {}),
    'langflow_ada':  ('langflow', {'VARIANT': 'ada_sched'}),
    'langflow_full': ('langflow', {'VARIANT': 'full'}),
}

# H-FLM curvature grid (matches the original hflm_curv_init_lr_sudoku sweep)
KS = ['-0.25', '-0.3', '-0.5', '-0.7', '-1.0', '-1.5']
INITS = [('ngpt', 'ngpt', 'null'), ('random', 'random', 'null')] + [
    (f'c{s}', 'custom', s) for s in ['0.01', '0.02', '0.04', '0.06', '0.08']]
LRS = ['1e-4', '3e-4', '5e-4', '1e-3']

SITES = {
    'unicorn': dict(
        repo='/share/thickstun/sychou/workspace/research/s-flm',
        envbin='/home/sc3379/anaconda3/envs/sfm/bin',
        partitions=['thickstun,desa'],
        slurm=dict(exclude='desa-compute-01', gres='gpu:1', ntasks=1,
                   cpus_per_task=2, mem='16G', time='06:00:00'),
        wandb_offline=False,
    ),
    'falcon': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        partitions=['a30_normal_q', 'l40s_normal_q'],
        slurm=dict(account='swan_research_dlm', gres='gpu:1', ntasks=1,
                   cpus_per_task=4, mem='32G', time='06:00:00'),
        wandb_offline=True,
    ),
}


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(site, tag, script, env, cdir):
    ev = ''.join(f'{k}={v} ' for k, v in env.items())
    offline = 'export WANDB_MODE=offline\n' if site['wandb_offline'] else ''
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={site['envbin']}:$PATH
        {offline}cd {site['repo']}
        if [ -f {cdir}/checkpoints/last.ckpt ]; then
            echo "[$(date)] checkpoint already present -> no-op"; exit 0
        fi
        echo "[$(date)] RETRAIN {tag} on $(hostname)  (train only, keep checkpoints)"
        {ev}DIFFICULTY=hard SEED=1 OUTPUT_DIR={cdir} DEVICES=1 \\
            bash scripts/train/sudoku/{script}.sh
        echo "[$(date)] DONE  (checkpoints retained at {cdir}/checkpoints)"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=SITES)
    ap.add_argument('--family', required=True, choices=['baseline', 'hflm'])
    ap.add_argument('--algos', nargs='+', default=None, choices=list(BASELINES))
    ap.add_argument('--curvatures', nargs='+', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    site = SITES[args.site]
    exp = f"{site['repo']}/experiments/hflm_curv_init_lr_sudoku"
    logs = f'{exp}/logs_retrain'
    out = f"{site['repo']}/{CKPT_SUBDIR}"
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    # build (tag, script, env) cells
    cells = []
    if args.family == 'baseline':
        for a in (args.algos or list(BASELINES)):
            script, env = BASELINES[a]
            cells.append((f'bl_d-hard_a-{a}_rs1', script, dict(env)))
    else:  # hflm
        ks = args.curvatures or KS
        for k, (itag, init, istd), lr in itertools.product(ks, INITS, LRS):
            tag = f'd-hard_k{k}_i-{itag}_lr{lr}_rs1'
            env = dict(GAUSS_CURV=k, INIT=init, INIT_STD=istd, LR=lr)
            cells.append((tag, 'hflm', env))

    print(f'retrain_ckpts [{args.site}/{args.family}]: {len(cells)} cells')
    if args.dry_run:
        for tag, script, env in cells[:6]:
            print(f'  {tag}  <- {script}.sh  {env}')
        print('  ...' if len(cells) > 6 else '')
        tag, script, env = cells[0]
        print('\n--- example body ---\n'
              + job_body(site, tag, script, env, f'{out}/{tag}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for tag, script, env in cells:
        cdir = f'{out}/{tag}'
        jobname = f'rc_{tag}'
        if os.path.exists(f'{cdir}/checkpoints/last.ckpt') or jobname in active:
            n_skip += 1
            continue
        part = site['partitions'][n_sub % len(site['partitions'])]
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition=part, **site['slurm'])
        jid = slurm.sbatch(job_body(site, tag, script, env, cdir),
                           sbatch_cmd='sbatch --nice=0 --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
