#!/usr/bin/env python
"""adv_geo_tinystories (slides jun25_2026) — "advanced" geometry baselines, LR sweep.

5 variants x 5 LRs = 25 cells:
  S-FLM   : ada-sched, truncation, ada-sched+truncation   (init=ngpt)
  LangFlow: ada-sched, ada-sched+self-cond                 (init=unit_var, see note)
  LR      : {5e-5, 1e-4, 3e-4, 1e-3, 5e-3}

Recipe: small DiT 768/12/12, 30k steps, effective batch 512 (1 GPU x PER_GPU_BS=8,
accum auto = 64), seq 1024, bf16, EMA 0.9999, AdamW wd 0 / grad-clip 1.0.
Eval: exact-velocity, top_k_v=1, 180 steps, greedy last (langflow: steps=180, top_k=1).

ORCHESTRATION ONLY — each cell calls the single-run shared scripts:
  scripts/train/tinystories/{sfm_truncated_adaptive,sfm_truncated,langflow}.sh
  scripts/sample/tinystories/{...same...}.sh
The variant is selected by which script + a fixed env toggle (ALPHA_MAX / SELF_COND).
Idempotent + resumable (skips done/queued cells; resubmits auto-resume from last.ckpt).

NOTE: the slide says init=ngpt for adv_geo, but LangFlow's N(0,I) VP prior is scale-matched
to unit_var embeddings (ngpt -> noise >> signal; cf. EFLM). So LangFlow keeps init=unit_var
(baked into langflow.sh). The three S-FLM variants do use init=ngpt as specified.

Usage:  python sweep.py [--dry-run]
"""
import argparse
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/adv_geo_tinystories'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/adv_geo_tinystories'

LRS = ['5e-5', '1e-4', '3e-4', '1e-3', '5e-3']
# tag -> (train_script, sample_script, fixed_env)   fixed_env carries the variant toggle
# and must match between train and eval (alpha_max / self_cond affect the loaded model/metric).
VARIANTS = {
    'sfm_ada':       ('sfm_truncated_adaptive.sh', 'sfm_truncated_adaptive.sh', 'ALPHA_MAX=null'),
    'sfm_trunc':     ('sfm_truncated.sh',          'sfm_truncated.sh',          'ALPHA_MAX=0.121'),
    'sfm_ada_trunc': ('sfm_truncated_adaptive.sh', 'sfm_truncated_adaptive.sh', 'ALPHA_MAX=0.121'),
    'lf_ada':        ('langflow.sh',               'langflow.sh',               'SELF_COND=false'),
    'lf_ada_sc':     ('langflow.sh',               'langflow.sh',               'SELF_COND=true'),
}


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag, train_script, sample_script, fixed_env, lr):
    tdir = f'{OUT}/{tag}'
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {fixed_env} LR={lr} OUTPUT_DIR={tdir} RUN_NAME=adv_{tag} DEVICES=1 PER_GPU_BS=8 \\
            bash scripts/train/tinystories/{train_script}
        echo "[$(date)] EVAL {tag}"
        {fixed_env} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{sample_script}
        echo "[$(date)] DONE {tag}"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    cells = list(itertools.product(VARIANTS.keys(), LRS))
    print(f'adv_geo_tinystories: {len(cells)} cells ({len(VARIANTS)} variants x {len(LRS)} lr)')
    if args.dry_run:
        for v, lr in cells:
            print(f'  adv_{v}_lr{lr}')
        v0, lr0 = cells[0]
        print('\n--- example body ---\n' + job_body(f'{v0}_lr{lr0}', *VARIANTS[v0], lr0))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for v, lr in cells:
        tag = f'{v}_lr{lr}'
        jobname = f'adv_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/ppl.json') or jobname in active:
            n_skip += 1
            continue
        train_script, sample_script, fixed_env = VARIANTS[v]
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='10-00:00:00',
                      exclude='desa-compute-01', output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(tag, train_script, sample_script, fixed_env, lr))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
