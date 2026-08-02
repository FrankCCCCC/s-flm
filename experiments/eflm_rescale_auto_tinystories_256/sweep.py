#!/usr/bin/env python
"""eflm_rescale_auto_tinystories_256 — rescaled EFLM on the AUTONOMOUS clock,
{w/, w/o Ada} x {w/, w/o Trunc} x R, TinyStories seq 256, 1 seed.

The autonomous clock (slides/jul09_2026) makes the bridge drift time-invariant
by letting the noise fraction decay exponentially, b_t = exp(-tau) with
tau = tau_max (1 - t). On that clock **tau_max IS the truncation**:

    tau*(R) = -log b*(R),  b*(R) = 1 - alpha_star_euclidean(V=50257, R)

reproduces exactly the b-range of the log-linear `trunc` arm of
experiments/eflm_rescale_tinystories_256 (verified: same b_min/b_max at all six
R), so w/ Trunc is a matched-endpoint comparison against that arm and isolates
the *traversal* — the VDM 5.1 schedule-invariance question. w/o Trunc uses
tau_max = -log(1e-3) = 6.908, the log-linear schedule's own training floor.

  arms: auto | auto_ada | auto_trunc | auto_trunc_ada

Sudoku (experiments/eflm_rescale_auto_sudoku): the clock is worth +11..+21 pts
on the fixed-schedule arm, the optimum tracks tau*(R), and clock vs adaptive
scheduler are substitutes. Prior TinyStories (log-linear): trunc_ada @ R=1 is
best (GenPPL 10.99), GenPPL monotone in R, raw-norm baseline 34.58.

Fixed (mirrors eflm_rescale_tinystories_256): small-sphere-dit, ngpt init, 30k
steps, global batch 512 (DEVICES=4 x PER_GPU_BS=32 x accum 4), seq 256, lr
3e-4. Eval: ppl_eval + sample_eval (GenPPL), exact velocity, top_k_v=1, 180
steps, greedy last. Checkpoints KEPT.

Usage:  python experiments/eflm_rescale_auto_tinystories_256/sweep.py
            [--arms ...] [--rhos 0.5 1 5 8 16 28] [--dry-run]
Idempotent: skips cells whose eval/samples_genppl.json exists or whose job is
queued; resubmit auto-resumes from last.ckpt.
"""
import argparse
import getpass
import itertools
import math
import os
import subprocess
import sys
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

V = 50257
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32
CKPT_EVERY = 5000
MAX_STEPS = 30000
LR = '3e-4'
TAU_UNTRUNCATED = 6.9078  # -log(1e-3): the log-linear schedule's own floor
RHOS = ['0.5', '1', '5', '8', '16', '28']
# arm -> (script stem, truncated?, snr_weighted_ce?)  stem shared by train+sample
ARMS = {
    'auto': ('eflm_rescale_auto', False, False),
    'auto_ada': ('eflm_rescale_auto_adaptive', False, False),
    'auto_trunc': ('eflm_rescale_auto', True, False),
    'auto_trunc_ada': ('eflm_rescale_auto_adaptive', True, False),
    # VDM Eq. 16 weighting on the untruncated horizon. Its dynamic range there
    # is ~1e9 and it up-weights the post-decode region the untruncated arm is
    # already wasting, so this is expected to be worse than `auto`, not better;
    # run to confirm the mechanism rather than to win.
    'auto_snr': ('eflm_rescale_auto', False, True),
    # Eq. 16 weighting on the TRUNCATED horizon. Here the weight's dynamic
    # range is set by tau*(R) and shrinks with R (4e2 at R=28 .. 1e5 at R=0.5),
    # so large R lands in the <=1e3 regime where sudoku found the weighting
    # HELPS. Expect this arm to win at large R and lose at small R.
    'auto_trunc_snr': ('eflm_rescale_auto', True, True),
    # LOG-LINEAR controls at the R values the prior sweep never ran. Needed to
    # attribute this sweep's best cell (10.74 @ R=0.5) to the clock vs to the
    # smaller norm: the prior grid was R in {1, 8, 28} only.
    'll_trunc': ('eflm_rescale_truncated', True, False),
    'll_trunc_ada': ('eflm_rescale_truncated_adaptive', True, False),
}


def tau_max_of(arm, rho):
    """Autonomous horizon: tau*(R) when truncated, the full range otherwise.

    tau* = -log(1 - alpha*) = log(1 + C/R), i.e. the Eq.-17 bound expressed on
    the clock that runs on the noise fraction (auto_clock=True); -log(alpha*)
    would mirror R and is NOT the horizon.
    """
    if not ARMS[arm][1]:
        return f'{TAU_UNTRUNCATED:.4f}'
    tau_star = alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=True)
    assert abs(tau_star + math.log(
        1 - alpha_star_euclidean(V, embed_norm=float(rho)))) < 1e-9
    return f'{tau_star:.4f}'


def tag_of(arm, rho):
    return f'eflmrat_{arm}_r-{rho}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(arm, rho, tdir):
    stem, _, snr = ARMS[arm]
    tau = tau_max_of(arm, rho)
    snr_ce = 'true' if snr else 'false'
    edir = f'{tdir}/eval'
    if arm.startswith('ll_'):   # log-linear clock: truncation is ALPHA_MAX
        sched = f'ALPHA_MAX={alpha_star_euclidean(V, embed_norm=float(rho)):.4f}'
    else:
        sched = f'TAU_MAX={tau}'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {edir}/samples_genppl.json ]; then
            echo "[$(date)] cell already completed -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {arm} R={rho} {sched} snr_ce={snr_ce} on $(hostname)"
        RHO={rho} {sched} SNR_CE={snr_ce} LR={LR} SEQ_LEN={SEQ_LEN} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag_of(arm, rho)} WANDB_GROUP=eflm_rescale_auto \\
            DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} \\
            CKPT_EVERY={CKPT_EVERY} SAVE_TOPK=1 \\
            bash scripts/train/tinystories/{stem}.sh
        echo "[$(date)] EVAL"
        RHO={rho} {sched} SNR_CE={snr_ce} SEQ_LEN={SEQ_LEN} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{stem}.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    args = ap.parse_args()

    exp = f'{REPO}/experiments/eflm_rescale_auto_tinystories_256'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/eflm_rescale_auto_tinystories_256'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.arms, args.rhos))
    print(f'eflm_rescale_auto_tinystories_256: {len(cells)} cells '
          f'({len(args.arms)} arm x {len(args.rhos)} R), '
          f'{DEVICES} GPU each')
    if args.dry_run:
        for arm, rho in cells:
            print(f'  {tag_of(arm, rho):32} tau_max={tau_max_of(arm, rho)}')
        arm, rho = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(arm, rho, f'{out}/{tag_of(arm, rho)}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for arm, rho in cells:
        tag = tag_of(arm, rho)
        if (os.path.exists(f'{out}/{tag}/eval/samples_genppl.json')
                or tag in active):
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=8,
                      mem='64G', time='2-00:00:00')
        jid = slurm.sbatch(job_body(arm, rho, f'{out}/{tag}'),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag} (tau_max={tau_max_of(arm, rho)}): job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
