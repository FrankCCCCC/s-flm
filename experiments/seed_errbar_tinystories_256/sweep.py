#!/usr/bin/env python
"""seed_errbar_tinystories_256 — 3-seed error bars for the two BEST configs, to settle
whether near-flat adaptive H-FLM 'matches' or 'beats' the best S-FLM (single-run margin
10.70 vs 11.08 is within the cross-run variance the geo/hflm sweeps exposed).

6 cells = {best-H-FLM, best-S-FLM} x seed{1,2,3}, all on Unicorn (same hardware pool),
SEQ_LEN=256, PER_GPU_BS=32 (global batch 512 via accumulation), SC-on. SEED knob added to
the two train scripts (seed=${SEED:-1}); eval seed stays default (isolates train-seed var).

  best-H-FLM: hlfm_adaptive.sh + hflm_adaptive.sh
              GAUSS_CURV=-0.01 INIT=custom INIT_STD=0.01 PRIOR_COV=1.0 RHO_MAX=12 SELF_COND=true
  best-S-FLM: sfm_truncated_adaptive.sh (train+sample)  SELF_COND=true ALPHA_MAX=0.121

Idempotent: skip a cell whose eval/samples_genppl.json exists or whose job is queued.
Usage: python sweep.py [--nice 0] [--dry-run]
"""
import argparse, getpass, os, subprocess, textwrap
from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
SEQ_LEN, PER_GPU_BS, CKPT_EVERY, SAVE_TOPK = 256, 32, 5000, 1
SEEDS = [1, 2, 3]

CONFIGS = [
    ('hflm', 'hlfm_adaptive.sh', 'hflm_adaptive.sh',
     'INIT=custom INIT_STD=0.01 GAUSS_CURV=-0.01 PRIOR_COV=1.0 RHO_MAX=12 SELF_COND=true'),
    ('sfm', 'sfm_truncated_adaptive.sh', 'sfm_truncated_adaptive.sh',
     'SELF_COND=true ALPHA_MAX=0.121'),
]

_HEAD = '\n'.join([
    'set -e',
    'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
    'export SLURM_JOB_NAME=bash',
    'export NCCL_P2P_DISABLE=1',
    'export NCCL_IB_DISABLE=1',
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
    'export WANDB_MODE=offline',
])


def body(tag, train_s, sample_s, env, seed, tdir):
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        {_HEAD}
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {edir}/samples_genppl.json ]; then echo "[$(date)] {tag} done -> no-op"; exit 0; fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {env} SEED={seed} OUTPUT_DIR={tdir} RUN_NAME=seb_{tag}_s256 DEVICES=1 PER_GPU_BS={PER_GPU_BS} \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/{train_s}
        echo "[$(date)] EVAL {tag}"
        {env} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/{sample_s}
        if [ -f {edir}/samples_genppl.json ]; then rm -rf {tdir}/checkpoints && echo "[$(date)] ckpts cleaned"; fi
        echo "[$(date)] DONE {tag}"
        ''')


def active():
    try:
        return set(subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                                  capture_output=True, text=True).stdout.split())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    exp = f'{REPO}/experiments/seed_errbar_tinystories_256'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/seed_errbar_tinystories_256'
    cells = [(f'{name}_seed{s}', tr, sa, env, s)
             for (name, tr, sa, env) in CONFIGS for s in SEEDS]
    print(f'seed_errbar: {len(cells)} cells (2 configs x seeds {SEEDS})')
    if args.dry_run:
        for c in cells:
            print('  seb_' + c[0])
        print('\n--- example body ---\n' + body(*cells[0], f'{out}/{cells[0][0]}'))
        return
    os.makedirs(logs, exist_ok=True)
    act = active()
    nsub = nskip = 0
    for tag, tr, sa, env, s in cells:
        jn = f'seb_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/samples_genppl.json') or jn in act:
            nskip += 1
            continue
        sl = Slurm(job_name=jn, output=f'{logs}/{tag}_%j.log',
                   partition='thickstun,desa', exclude='desa-compute-01',
                   gres='gpu:1', ntasks=1, cpus_per_task=8, mem='48G', time='3-00:00:00')
        jid = sl.sbatch(body(tag, tr, sa, env, s, f'{out}/{tag}'),
                        sbatch_cmd=f'sbatch --nice={args.nice} --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        nsub += 1
    print(f'submitted {nsub}, skipped {nskip}')


if __name__ == '__main__':
    main()
