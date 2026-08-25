#!/usr/bin/env python
"""eflm_rescale_auto_trunc_tinystories_256 — hyperparameter search for the
TRUNCATED autonomous-clock E-FLM on TinyStories seq 256: LR x R x truncation.

Follow-up to experiments/eflm_rescale_auto_tinystories_256, which fixed
lr = 3e-4 and pinned the truncation at the closed-form decode point
tau*(R) = -log(1 - alpha_star_euclidean(V=50257, R)) = log(1 + C/R), C = 5.2575.
That sweep's `auto_trunc` arm is this sweep's (lr 3e-4, m = 1.0) slice; those
six cells are symlinked in rather than re-run.

Three swept knobs, one arm (`scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh`):

  LR   in {3e-4, 1e-3, 5e-3}
  R    in {0.05, 0.1, 0.5, 1, 5, 8, 16, 28}     (algo.rho_min = rho_max = R)
  m    truncation multiplier: TAU_MAX = m * tau*(R)

On the autonomous clock 1 - alpha_t = exp(-tau), tau = TAU_MAX (1 - t), so
TAU_MAX *is* the truncation (noise-fraction floor exp(-TAU_MAX)); m = 1 stops
exactly at the decode point, m < 1 stops before it, m > 1 runs past it toward
the untruncated horizon 6.9078 = -log(1e-3).

`setup.md` specifies **plain CE only** (`{w/o SNR}`), so `--weights` defaults to
`ce` and the swept axes are LR, R and the truncation. The `snr` value exists
only so the parent sweep's already-run Eq.-16 cells can be pulled in for
reference (`--weights snr`); tags carry `_w-snr` there, plain CE has no suffix.

Fixed (mirrors the parent sweep so GenPPL is comparable): small-sphere-dit,
ngpt init, 30k steps, global batch 512 (DEVICES=4 x PER_GPU_BS=32 x accum 4),
seq 256, bf16, EMA 0.9999, 1 seed. Eval: ppl_eval + sample_eval (GenPPL),
exact velocity, top_k_v=1, 180 steps, greedy last.

Usage:
  # stage 1 (LR x R at the closed-form truncation)
  python experiments/eflm_rescale_auto_trunc_tinystories_256/sweep.py
  # stage 2 (truncation around tau*, at the winning R/LR)
  python .../sweep.py --lrs 1e-3 --rhos 0.5 1 --mults 0.5 0.7 0.85 1.25
  python .../sweep.py --dry-run

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

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

EXP = 'eflm_rescale_auto_trunc_tinystories_256'
PARENT = 'eflm_rescale_auto_tinystories_256'   # supplies the (3e-4, m=1.0) slice
STEM = 'eflm_rescale_auto_truncation'
V = 50257
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32
CKPT_EVERY = 5000
MAX_STEPS = 30000
LRS = ['3e-4', '1e-3', '5e-3']
RHOS = ['0.05', '0.1', '0.5', '1', '5', '8', '16', '28']
MULTS = ['1.0']
WEIGHTS = ['ce', 'snr']
SEEDS = ['1']        # seed>1 gets an _s{seed} tag suffix; seed 1 keeps the bare name
# (lr, rho, mult, w) already run by the parent sweep's auto_trunc{,_snr} arms
INHERITED = {('3e-4', r, '1.0', w): f'eflmrat_auto_trunc{sfx}_r-{r}'
             for r in ['0.5', '1', '5', '8', '16', '28']
             for w, sfx in (('ce', ''), ('snr', '_snr'))}


def tau_max_of(rho, mult):
    """m * tau*(R); tau*(R) = -log b*(R) is the decode point on this clock."""
    tau_star = alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=True)
    return f'{float(mult) * tau_star:.4f}'


def tag_of(lr, rho, mult, w, seed='1'):
    sfx = '_w-snr' if w == 'snr' else ''
    sfx += '' if str(seed) == '1' else f'_s{seed}'
    return f'eflmratr_lr-{lr}_r-{rho}_m-{mult}{sfx}'


def link_inherited(out, weights, dry):
    """Point the (3e-4, m=1.0) cells at the parent sweep's runs (no re-run)."""
    for key, src_tag in INHERITED.items():
        if key[3] not in weights:
            continue
        dst = f'{out}/{tag_of(*key)}'
        src = f'{REPO}/outputs/{PARENT}/{src_tag}'
        if os.path.lexists(dst) or not os.path.isdir(src):
            continue
        print(f'  link {os.path.basename(dst)} -> {PARENT}/{src_tag}')
        if not dry:
            os.symlink(src, dst)


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(lr, rho, mult, w, seed, tdir):
    tau = tau_max_of(rho, mult)
    snr = 'true' if w == 'snr' else 'false'
    edir = f'{tdir}/eval'
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
        echo "[$(date)] TRAIN lr={lr} R={rho} m={mult} w={w} seed={seed} TAU_MAX={tau} on $(hostname)"
        RHO={rho} TAU_MAX={tau} ALPHA_MAX=null SNR_CE={snr} LR={lr} SEED={seed} SEQ_LEN={SEQ_LEN} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag_of(lr, rho, mult, w, seed)} WANDB_GROUP={EXP} \\
            DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} \\
            CKPT_EVERY={CKPT_EVERY} SAVE_TOPK=1 \\
            bash scripts/train/tinystories/{STEM}.sh
        echo "[$(date)] EVAL"
        RHO={rho} TAU_MAX={tau} ALPHA_MAX=null SNR_CE={snr} SEQ_LEN={SEQ_LEN} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{STEM}.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    ap.add_argument('--mults', nargs='+', default=MULTS)
    ap.add_argument('--weights', nargs='+', default=['ce'], choices=WEIGHTS)
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    logs = f'{REPO}/experiments/{EXP}/logs'
    out = f'{REPO}/outputs/{EXP}'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)
        os.makedirs(out, exist_ok=True)
    link_inherited(out, args.weights, args.dry_run)

    cells = list(itertools.product(args.lrs, args.rhos, args.mults,
                                   args.weights, args.seeds))
    print(f'{EXP}: {len(cells)} cells '
          f'({len(args.lrs)} lr x {len(args.rhos)} R x {len(args.mults)} m '
          f'x {len(args.weights)} w x {len(args.seeds)} seed), {DEVICES} GPU each')
    if args.dry_run:
        for c in cells:
            note = ' [inherited]' if c[:4] in INHERITED else ''
            print(f'  {tag_of(*c):40} tau_max={tau_max_of(c[1], c[2])}{note}')
        print('\n--- example body (first cell) ---\n'
              + job_body(*cells[0], f'{out}/{tag_of(*cells[0])}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for lr, rho, mult, w, seed in cells:
        tag = tag_of(lr, rho, mult, w, seed)
        if (os.path.exists(f'{out}/{tag}/eval/samples_genppl.json')
                or tag in active):
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=8,
                      mem='64G', time='2-00:00:00')
        jid = slurm.sbatch(job_body(lr, rho, mult, w, seed, f'{out}/{tag}'),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag} (tau_max={tau_max_of(rho, mult)}): job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
