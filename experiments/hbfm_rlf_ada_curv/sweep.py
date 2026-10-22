#!/usr/bin/env python
"""hbfm_rlf_ada_curv -- adaptive per-position curvature for HBFM on hard Sudoku.

Spec: `setup.md`. Math: `math.md`. Design and hypotheses: `EXPERIMENT.md`.

Every cell is the setup's HBFM run (tiny DiT, (H^3)^3 at base curvature K, 20k steps, batch 256,
EMA 0.9999, 180-step exact-velocity eval with a greedy last step) with one more knob, the adaptive
curvature bound `cb` (`algo.curvature_bound`): cb = 1 is the fixed-curvature baseline, cb > 1 lets
every position run at curvature lambda_i * K, lambda_i in [1 / cb, cb].

ORCHESTRATION ONLY -- every cell calls scripts/train/sudoku/hbfm.sh + scripts/sample/sudoku/hbfm.sh.
Idempotent + resumable: a cell with eval/results.json or a queued job of the same name is skipped;
resubmitting the same OUTPUT_DIR resumes from checkpoints/last.ckpt.

Usage:  python sweep.py --stage {smoke,1,2} [--seeds 1,2,3] [--only SUBSTR] [--dry-run]
        python report.py   ->  the table in RESULTS.md
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/hbfm_rlf_ada_curv'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/hbfm_rlf_ada_curv'

MAX_STEPS = 20000
CKPT_EVERY = 5000
GLOBAL_BATCH = 256
# setup.md: desa + thickstun, 2080 Ti for the large-scale runs (desa-compute-01, 8 GPUs, 11 GB).
PARTITION = 'desa,thickstun'
GRES = 'gpu:nvidia_geforce_rtx_2080_ti:1'
# 11 GB: 128 sequences per micro-batch x 2 accumulation = the optimizer batch of 256, for both
# arms (the harness of claude_test_hbfm_sudoku round 3). The adaptive model's curvature pass runs
# without gradients, so it adds no activation memory.
PER_GPU_BS = 128
MODE_TAG = {'exp': 'exp', 'trunc_exp': 'texp', 'unif': 'unif'}
FWD_TAG = {'naive': 'naive', 'horosphere': 'horo'}

# The best fixed-curvature cell of experiments/claude_test_hbfm_sudoku (63.8 +- 3.9 on this harness).
ANCHOR = dict(cb=1.0, mode='exp', rate=3.0, k=-0.5, ub=None, tmax=6.0, fwd='horosphere', lr='3e-4')


def cell(**kw):
    c = dict(ANCHOR)
    c.update(kw)
    return c


STAGES = {
    # 300 steps + the 2000-board eval: does it run, how fast, how much memory.
    'smoke': [cell(), cell(cb=4.0)],
    # Stage 1 (seed 1). A: the adaptive curvature at the best fixed-curvature cell. B-E: at
    # proposals the fixed-curvature study found mis-set (claude_test_hbfm_sudoku/RESULTS.md) --
    # the curvature can re-scale every position's clock by up to cb, so it should repair them.
    '1': [
        cell(), cell(cb=4.0),                                  # A: anchor (unit rate 6)
        cell(cb=4.0, rate=0.5), cell(rate=0.5),                # B: unit rate 1 (fixed: 31.3 +- 3.5)
        cell(cb=4.0, mode='unif', ub=2.3, tmax=2.3),           # C: flat proposal (fixed: 35.6 +- 6.3)
        cell(cb=4.0, fwd='naive'),                             # D: plain logits (fixed: 55.9, n=1)
        cell(cb=4.0, rate=6.0),                                # E: unit rate 12 (fixed r6.9: 62.6, n=1)
        cell(cb=2.0),                                          # A: a tighter bound
    ],
    # Stage 2 (--seeds 2,3): the anchor block needs three seeds per arm.
    '2': [cell(), cell(cb=4.0), cell(cb=2.0)],
}


def tag_of(c, seed):
    ub = 'na' if c['ub'] is None else f'{c["ub"]:g}'
    return (f'cb-{c["cb"]:g}_mode-{MODE_TAG[c["mode"]]}_rate-{c["rate"]:g}_k{c["k"]:g}_ub-{ub}'
            f'_tm-{c["tmax"]:g}_fwd-{FWD_TAG[c["fwd"]]}_lr-{c["lr"]}_seed-{seed}')


def geometry(c):
    k = c['k']
    return (f"EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[{k:g},{k:g},{k:g}]' "
            f"GAUSS_CURV={k:g} CURV_BOUND={c['cb']:g} PROPOSAL_RATE={c['rate']:g} T_MAX={c['tmax']:g} "
            f"TIME_CONVERSION_MODE={c['mode']} "
            f"TIME_RANGE_UPPER_BOUND={1.0 if c['ub'] is None else c['ub']:g} "
            f"FORWARD_TYPE={c['fwd']}")


def active_jobnames():
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                         capture_output=True, text=True).stdout
    return set(out.split())


def job_body(c, seed, out, max_steps):
    tag = tag_of(c, seed)
    tdir, geom = f'{out}/{tag}', geometry(c)
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfm_ada_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        nvidia-smi --query-gpu=name --format=csv,noheader
        LR={c['lr']} SEED={seed} {geom} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS={PER_GPU_BS} GLOBAL_BATCH={GLOBAL_BATCH} \\
            NUM_WORKERS=2 MAX_STEPS={max_steps} CKPT_EVERY={min(CKPT_EVERY, max_steps)} \\
            bash scripts/train/sudoku/hbfm.sh
        echo "[$(date)] EVAL {tag}"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader
        SEED={seed} {geom} GLOBAL_BATCH={GLOBAL_BATCH} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/hbfm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=sorted(STAGES))
    ap.add_argument('--seeds', default='1', help='comma list of seeds')
    ap.add_argument('--only', default='', help='substring filter on the cell tag')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    smoke = args.stage == 'smoke'
    out = f'{OUT}/smoke' if smoke else OUT
    max_steps = 300 if smoke else MAX_STEPS
    cells = [(c, int(s)) for c in STAGES[args.stage] for s in args.seeds.split(',')]
    cells = [(c, s) for c, s in cells if args.only in tag_of(c, s)]
    active = set() if args.dry_run else active_jobnames()
    n_sub = n_skip = 0
    for c, seed in cells:
        tag = tag_of(c, seed)
        jobname = f'hbfm_ada_{"smoke_" if smoke else ""}{tag}'
        if os.path.exists(f'{out}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        body = job_body(c, seed, out, max_steps)
        n_sub += 1
        if args.dry_run:
            print(f'  [dry] {jobname}')
            continue
        slurm = Slurm(job_name=jobname, partition=PARTITION, gres=GRES,
                      ntasks=1, cpus_per_task=1, mem='14G',
                      time='1:00:00' if smoke else '14:00:00',
                      output=f'{LOGS}/{jobname}_%j.log')
        print(f'  submitted {jobname}: job {slurm.sbatch(body)}')
    print(f'stage {args.stage}: submitted {n_sub}, skipped {n_skip}')
    if args.dry_run and cells:
        print('\n--- example body ---\n' + job_body(*cells[-1], out, max_steps))


if __name__ == '__main__':
    main()
