#!/usr/bin/env python
"""naive_ar_tinystories_s256 — Gen. PPL / entropy frontier evaluation.

`setup.md` "GenPPL & Entropy Frontier Evaluation": for each method
{mdlm, duo, flm} and each pretrained seed {1, 2, 3}, sweep
NFE x sampling temperature and draw 512 samples per cell. The frontier line
(Gen. PPL vs per-sample unigram entropy, one panel per NFE) is drawn by
`visualization/genppl_entropy_frontier_line.py` — S-FLM paper Fig. 10 / App. C.8.

LR is FIXED at 1e-3 — `setup.md` sweeps only the seed for this evaluation, and
RESULTS.md Sec. 3.4 selects 1e-3 as the shared LR for this setup.

ORCHESTRATION ONLY — calls the single-run shared scripts
  scripts/sample/tinystories/{mdlm,duo,flm}.sh
once per (T, NFE) cell via their STEPS / TEMPERATURE / NUM_SAMPLE_BATCHES knobs,
with RUN_PPL_EVAL=false (valid PPL does not depend on the sampler and is already
recorded by `sweep.py`). One SLURM job per (method, seed, NFE) = 3 x 3 x 8 = 72
jobs, each walking the 15 temperatures sequentially.
Idempotent + resumable: skips cells whose samples_genppl.json exists and jobs
already in squeue; a resubmitted job re-runs only the missing cells.

Usage:  python frontier_sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/naive_ar_tinystories_s256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/naive_ar_tinystories_s256'
SEQ_LEN = 256
NUM_SAMPLES = 512

# Searched axes (setup.md). LR is fixed, see module docstring.
METHODS = ['mdlm', 'duo', 'flm']
LR = '1e-3'
SEEDS = [1, 2, 3]
NFES = [1, 4, 8, 16, 32, 64, 128, 256]
TEMPS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
         '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20']
# flm carries a dense (B, L, V) float64 sampler state, so it needs a smaller batch.
EVAL_BS = {'mdlm': 32, 'duo': 32, 'flm': 16}


def jobs():
    return [(f'm-{m}_sd-{sd}_nfe-{nfe}', m, sd, nfe)
            for m in METHODS for sd in SEEDS for nfe in NFES]


def run_dir(method, seed):
    return f'{OUT}/m-{method}_lr-{LR}_sd-{seed}'


def cell_dir(method, seed, nfe, temp):
    return f'{run_dir(method, seed)}/frontier/nfe-{nfe}_t-{temp}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag, method, seed, nfe):
    ckpt = f'{run_dir(method, seed)}/checkpoints/last.ckpt'
    bs = EVAL_BS[method]
    calls = '\n'.join(textwrap.dedent(f'''\
        if [ -f {cell_dir(method, seed, nfe, t)}/samples_genppl.json ]; then
            echo "[$(date)] SKIP {tag}_t-{t} (done)"
        else
            echo "[$(date)] CELL {tag}_t-{t}"
            CKPT_PATH={ckpt} OUTPUT_DIR={cell_dir(method, seed, nfe, t)} \\
                DEVICES=1 SEQ_LEN={SEQ_LEN} STEPS={nfe} TEMPERATURE={t} \\
                EVAL_BS={bs} NUM_SAMPLE_BATCHES={NUM_SAMPLES // bs} \\
                RUN_PPL_EVAL=false \\
                bash scripts/sample/tinystories/{method}.sh
        fi''') for t in TEMPS)
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] FRONTIER {tag} on $(hostname)"
        ''') + calls + f'\necho "[$(date)] DONE {tag}"\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    grid = jobs()
    print(f'naive_ar_tinystories_s256 frontier: {len(grid)} jobs '
          f'({len(METHODS)} methods x {len(SEEDS)} seeds x {len(NFES)} NFE) '
          f'x {len(TEMPS)} temperatures = {len(grid) * len(TEMPS)} cells, '
          f'{NUM_SAMPLES} samples each')
    if args.dry_run:
        for tag, *_ in grid:
            print(f'  nar256fr_{tag}')
        print('\n--- example body ---\n' + job_body(*grid[0]))
        return
    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for tag, method, seed, nfe in grid:
        jobname = f'nar256fr_{tag}'
        done = all(os.path.exists(
            f'{cell_dir(method, seed, nfe, t)}/samples_genppl.json')
            for t in TEMPS)
        if done or jobname in active:
            print(f'skip {tag}: already evaluated or queued')
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='2-00:00:00',
                      exclude='desa-compute-01',
                      output=f'{LOGS}/frontier_{tag}_%j.log')
        jid = slurm.sbatch(job_body(tag, method, seed, nfe))
        print(f'submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
