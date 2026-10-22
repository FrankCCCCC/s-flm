#!/usr/bin/env python
"""NELBO of the best pre-trained TinyStories (seq 256) models of each method, on the validation set.

The checkpoints are the headline cells of the earlier TinyStories studies (nothing is trained here):
  hbfm  hbfm_rlf_ada_curv_tinystories_256: cb-4, trunc_exp rate 3, ub = t_max = 1, LR 3e-4 -- the
        study's best GenPPL (11.9); seed 1 is the only checkpoint on this file system (seeds 2-3 sit
        on the A100 node's scratch). 16 x H^3 at K = -0.5, naive readout, adaptive curvature b = 4.
  eflm  eflm_rescale_auto_trunc_tinystories_256: R = 0.5, LR 1e-3, tau_max = tau*(0.5) = 2.4436, the
        study's recommendation (GenPPL 11.44 +- 0.54), seeds 1-3.
  mdlm  naive_ar_tinystories_s256: LR 1e-3 (its best val/ppl 4.87), seeds 1-3.
Each job scores one checkpoint with `mode=ppl_eval` on the full validation set under the NELBO
(`algo.loss_type=elbo`; for MDLM its own) and, for HBFM / EFLM, under the CE objective it was trained
on -> outputs/elfm_hbfm_ada_curv_elbo_sudoku/tinystories/<cell>/ppl_<loss>_s1.json.

ORCHESTRATION ONLY -- calls scripts/sample/tinystories/{hbfm,eflm_rescale_auto_truncation,mdlm}.sh.
Idempotent: a cell whose ppl_elbo_s1.json exists or whose job name is queued is skipped.

Usage:  python tinystories_elbo.py [--dry-run] ; python tinystories_elbo.py --report
"""
import argparse
import glob
import json
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
PROJECT = 'elfm_hbfm_ada_curv_elbo_sudoku'
LOGS = f'{REPO}/experiments/{PROJECT}/logs'
OUT = f'{REPO}/outputs/{PROJECT}/tinystories'

CELLS = {
    'hbfm_texp-ub1_lr-3e-4_seed-1': dict(
        arm='hbfm', stem='hbfm',
        ckpt=f'{REPO}/outputs/hbfm_rlf_ada_curv_tinystories_256/'
             'cb-4_mode-texp_rate-3_ub-1_tm-1_lr-3e-4_seed-1/checkpoints/last.ckpt',
        knobs=('MODEL=small-hyperbolic-dit SEQ_LEN=256 FORWARD_TYPE=naive EMBED_DIM=48 FACTOR_DIM=3 '
               'GAUSS_CURV=-0.5 CURV_BOUND=4 TIME_CONVERSION_MODE=trunc_exp PROPOSAL_RATE=3 '
               'TIME_RANGE_UPPER_BOUND=1 T_MAX=1 READOUT_PRECISION=float32 GLOBAL_BATCH=512 EVAL_BS=16'),
        gres='gpu:nvidia_rtx_a6000:1', hours=8),   # the path-KL at V = 50k: (B, L, V, 3) float64 per factor
}
for s in (1, 2, 3):
    CELLS[f'eflm_r-0.5_lr-1e-3_seed-{s}'] = dict(
        arm='eflm', stem='eflm_rescale_auto_truncation',
        ckpt=f'{REPO}/outputs/eflm_rescale_auto_trunc_tinystories_256/'
             f'eflmratr_lr-1e-3_r-0.5_m-1.0{"" if s == 1 else f"_s{s}"}/checkpoints/last.ckpt',
        knobs='SEQ_LEN=256 RHO=0.5 TAU_MAX=2.4436 ALPHA_MAX=null SNR_CE=false EVAL_BS=16',
        gres='gpu:1', hours=3)
    CELLS[f'mdlm_lr-1e-3_seed-{s}'] = dict(
        arm='mdlm', stem='mdlm',
        ckpt=f'{REPO}/outputs/naive_ar_tinystories_s256/m-mdlm_lr-1e-3_sd-{s}/checkpoints/last.ckpt',
        knobs='SEQ_LEN=256 EVAL_BS=16', gres='gpu:1', hours=2)


def active_jobnames():
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j %R'],
                         capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if 'JobHeld' not in line}


def job_body(name, c):
    d = f'{OUT}/{name}'
    passes = [('elbo', 'elbo')] if c['arm'] == 'mdlm' else [('elbo', 'elbo'), ('ce', 'ce')]
    runs = ''.join(f'''\
        echo "[$(date)] {name}: {loss} pass on $(hostname)"
        {c["knobs"]} CKPT_PATH={c["ckpt"]} OUTPUT_DIR={d} DEVICES=1 SEED=1 \\
            LOSS_TYPE={loss} PPL_JSON={d}/ppl_{tag}_s1.json RUN_SAMPLE_EVAL=false \\
            bash scripts/sample/tinystories/{c["stem"]}.sh
        ''' for loss, tag in passes)
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/elbo_ts_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        ''') + textwrap.dedent(runs) + f'echo "[$(date)] DONE {name}"\n'


def report():
    rows = []
    for d in sorted(glob.glob(f'{OUT}/*')):
        name = os.path.basename(d)
        vals = {}
        for loss in ('elbo', 'ce'):
            files = sorted(glob.glob(f'{d}/ppl_{loss}_s*.json'))
            if files:
                vals[loss] = sum(json.load(open(f))['val/nll'] for f in files) / len(files)
        rows.append((name, vals))
    print('| checkpoint | NELBO (nats / token) | NELBO as PPL | CE objective (nats / token) |')
    print('|---|---:|---:|---:|')
    for name, v in rows:
        e = v.get('elbo')
        print(f'| {name} | {e:.4f} | {2.718281828 ** e:.3f} | ' if e is not None else f'| {name} | pending | — | ',
              end='')
        print(f'{v["ce"]:.4f} |' if 'ce' in v else '— |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.report:
        return report()
    os.makedirs(LOGS, exist_ok=True)
    active = set() if args.dry_run else active_jobnames()
    for name, c in CELLS.items():
        if args.only not in name:
            continue
        jobname = f'{PROJECT}_ts_{name}'
        if os.path.exists(f'{OUT}/{name}/ppl_elbo_s1.json') or jobname in active:
            print(f'  skip {name}')
            continue
        if not os.path.exists(c['ckpt']):
            print(f'  MISSING checkpoint for {name}: {c["ckpt"]}')
            continue
        body = job_body(name, c)
        if args.dry_run:
            print(f'  [dry] {jobname}')
            continue
        slurm = Slurm(job_name=jobname, partition='desa,thickstun', gres=c['gres'],
                      exclude='desa-compute-01', ntasks=1, cpus_per_task=4, mem='24G',
                      time=f'{c["hours"]}:00:00', output=f'{LOGS}/{jobname}_%j.log')
        print(f'  submitted {jobname}: job {slurm.sbatch(body)}')
    if args.dry_run:
        first = next(iter(CELLS))
        print('\n--- example body ---\n' + job_body(first, CELLS[first]))


if __name__ == '__main__':
    main()
