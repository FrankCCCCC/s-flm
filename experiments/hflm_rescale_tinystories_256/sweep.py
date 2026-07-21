#!/usr/bin/env python
"""hflm_rescale_tinystories_256 — single H-FLM run testing the √d radial rescale.

Trains ONE H-FLM cell (init custom std 0.04, prior_cov 1.0, K=-1.0, lr 3e-4,
seq 256) on the `var_scale` branch, where HFLM._rho_clamp divides rho by
√hidden_size (see EXPERIMENT.md). Then evals valid-PPL + GenPPL.

RUNS FROM THIS CHECKOUT (branch var_scale) — the rescale is only here, so REPO
points at s-flm-dev/s-flm, NOT the shared /share/.../s-flm tree.

ORCHESTRATION ONLY — calls the single-run shared scripts:
  scripts/train/tinystories/hlfm.sh   (INIT/INIT_STD/PRIOR_COV/RHO_MAX/GAUSS_CURV/LR knobs)
  scripts/sample/tinystories/hflm.sh  (PRIOR_COV/RHO_MAX/GAUSS_CURV knobs)
Idempotent + resumable: skip if eval/ppl.json exists or the job is queued;
resubmitting the same OUTPUT_DIR auto-resumes from last.ckpt.

Usage:  python experiments/hflm_rescale_tinystories_256/sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/hflm_rescale_tinystories_256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/hflm_rescale_tinystories_256'

# single cell — the config from the goal
# init_std grid — under the √d rescale, rescaled clean radius ≈ init_std, so the
# small std tuned for the un-rescaled geometry (0.04) collapses to the origin
# (see RESULTS.md). Sweep O(1) stds to place clean data at a usable radius.
STDS = ['0.04', '0.3', '0.5', '1.0']   # 0.04 already done (collapse) -> skipped
FIXED = dict(prior_cov='1.0', gauss_curv='-1.0', lr='3e-4', rho_max='12')
SEQ_LEN = 256
PER_GPU_BS = 32          # global 512 -> accum 16
MAX_STEPS = 30000
CKPT_EVERY = 5000


def tag_of(std):
    return f'std{std}_pc1.0_K-1.0_lr3e-4'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(std):
    tag = tag_of(std)
    tdir = f'{OUT}/{tag}'
    edir = f'{tdir}/eval'
    f = FIXED
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {edir}/ppl.json ]; then
            echo "[$(date)] cell already done -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        INIT=custom INIT_STD={std} PRIOR_COV={f['prior_cov']} \\
            RHO_MAX={f['rho_max']} GAUSS_CURV={f['gauss_curv']} LR={f['lr']} \\
            OUTPUT_DIR={tdir} RUN_NAME=hflm_rescale_{tag} WANDB_GROUP=hflm_rescale \\
            DEVICES=1 PER_GPU_BS={PER_GPU_BS} SEQ_LEN={SEQ_LEN} \\
            MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
            bash scripts/train/tinystories/hlfm.sh
        echo "[$(date)] EVAL {tag}"
        PRIOR_COV={f['prior_cov']} RHO_MAX={f['rho_max']} GAUSS_CURV={f['gauss_curv']} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} \\
            DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/hflm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        print(f'hflm_rescale_tinystories_256: {len(STDS)} cells '
              f'({", ".join(tag_of(s) for s in STDS)})')
        print('\n--- example job body (first cell) ---\n' + job_body(STDS[0]))
        return
    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for std in STDS:
        tag = tag_of(std)
        jobname = f'hfresc_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            print(f'skip {tag} (done or queued)')
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='32G', time='2-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(std), sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
