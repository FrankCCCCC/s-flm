#!/usr/bin/env python
"""eflm_rescale_tinystories_256 — rescaled EFLM {ada, trunc, trunc+ada} on
TinyStories seq 256, 1 seed.

Sudoku rounds 1-2 (experiments/eflm_rescale_sudoku): embedding norm R sets the
decode time t*(R); a fixed schedule collapses at large R, adaptive rescues it,
truncation at alpha*(R) recovers most of the rescue statically. This sweep
tests the same three schedule arms on TinyStories seq 256 with the norm pinned
to R:

  ada        : eflm_rescale_truncated_adaptive.sh, ALPHA_MAX=null (no trunc)
  trunc      : eflm_rescale_truncated.sh,          ALPHA_MAX=alpha*(R)
  trunc_ada  : eflm_rescale_truncated_adaptive.sh, ALPHA_MAX=alpha*(R)

alpha*(R) = alpha_star_euclidean(V=50257, embed_norm=R) (noise_schedules.py):
R=1 -> 0.840, R=8 -> 0.397, R=28 (~sqrt(d)) -> 0.158.  External baseline:
experiments/naive_geo_tinystories_s256 'eflm' (raw norms, no trunc/ada,
GenPPL 34.58 / entropy 3.67 / valid PPL 1.10).

Fixed (mirrors naive_geo_tinystories_s256): small-sphere-dit, ngpt init, 30k
steps, global batch 512 (DEVICES=4 x PER_GPU_BS=32 x accum 4), seq 256, lr
3e-4. Eval: ppl_eval + sample_eval (GenPPL), exact velocity, top_k_v=1, 180
steps. Checkpoints KEPT (loss-geometry follow-up).

Usage:  python experiments/eflm_rescale_tinystories_256/sweep.py
            [--arms ada trunc trunc_ada] [--rhos 1 8 28] [--seeds 1]
            [--dry-run]
Idempotent: skips cells whose eval/samples_genppl.json exists or whose job is
queued; resubmit auto-resumes from last.ckpt.
"""
import argparse
import getpass
import itertools
import os
import subprocess
import sys
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

V = 50257
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32
CKPT_EVERY = 5000
RHOS = ['1', '8', '28']
# arm -> (script stem, truncated?)  stem is shared by train and sample
ARMS = {
    'ada': ('eflm_rescale_truncated_adaptive', False),
    'trunc': ('eflm_rescale_truncated', True),
    'trunc_ada': ('eflm_rescale_truncated_adaptive', True),
}


def alpha_max_of(arm, rho):
    if not ARMS[arm][1]:
        return 'null'
    return f'{alpha_star_euclidean(V, embed_norm=float(rho)):.3f}'


def tag_of(arm, rho, seed):
    return f'eflmrs256_{arm}_r-{rho}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(arm, rho, seed, tdir):
    stem = ARMS[arm][0]
    am = alpha_max_of(arm, rho)
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export WANDB_MODE=offline
        export PATH={ENVBIN}:$PATH
        # temp + caches on node-local /tmp (shared /home has run full before)
        export TMPDIR=/tmp
        export MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
        export WANDB_CACHE_DIR=/tmp/wandb-cache-$SLURM_JOB_ID
        mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR"
        cd {REPO}
        if [ -f {edir}/samples_genppl.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {arm} r={rho} alpha_max={am} on $(hostname)"
        RHO={rho} ALPHA_MAX={am} SEED={seed} OUTPUT_DIR={tdir} \\
            RUN_NAME={tag_of(arm, rho, seed)} DEVICES={DEVICES} \\
            PER_GPU_BS={PER_GPU_BS} SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} \\
            bash scripts/train/tinystories/{stem}.sh
        echo "[$(date)] EVAL"
        RHO={rho} ALPHA_MAX={am} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/{stem}.sh
        # checkpoints kept for the loss-geometry follow-up.
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--rhos', nargs='+', default=RHOS, choices=RHOS)
    ap.add_argument('--seeds', nargs='+', default=['1'])
    args = ap.parse_args()

    exp = f'{REPO}/experiments/eflm_rescale_tinystories_256'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/eflm_rescale_tinystories_256'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.arms, args.rhos, args.seeds))
    print(f'eflm_rescale_tinystories_256: {len(cells)} cells '
          f'({len(args.arms)} arm x {len(args.rhos)} R x {len(args.seeds)} seed)')
    if args.dry_run:
        for arm, rho, seed in cells:
            print(f'  {tag_of(arm, rho, seed)}  (alpha_max={alpha_max_of(arm, rho)})')
        arm, rho, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(arm, rho, seed, f'{out}/{tag_of(arm, rho, seed)}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for arm, rho, seed in cells:
        tag = tag_of(arm, rho, seed)
        if os.path.exists(f'{out}/{tag}/eval/samples_genppl.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=16,
                      mem='64G', time='4-00:00:00')
        jid = slurm.sbatch(job_body(arm, rho, seed, f'{out}/{tag}'),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
