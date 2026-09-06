#!/usr/bin/env python
"""eflm_rescale_auto_trunc_tinystories_256_history — LR x R x truncation for the
TRUNCATED autonomous-clock E-FLM on TinyStories seq 256, with the FULL
checkpoint history retained (every 1k steps, save_top_k=-1).

Same arm and protocol as eflm_rescale_auto_trunc_tinystories_256
(`scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh`, plain CE),
narrowed by `setup.md` to the region that sweep found admissible:

  LR   in {3e-4, 1e-3}                          (5e-3 collapses at R <= 1)
  R    in {0.05, 0.1, 0.5, 1.0}                 (algo.rho_min = rho_max = R)
  m    truncation multiplier: TAU_MAX = m * tau*(R)
  seed in {1, 2, 3}

On the autonomous clock 1 - alpha_t = exp(-tau), tau = TAU_MAX (1 - t), so
TAU_MAX *is* the truncation (noise-fraction floor exp(-TAU_MAX)); m = 1 stops
exactly at the closed-form decode point

  tau*(R) = -log b*(R) = log(1 + C/R),  C = sqrt(2 log(2(V-1)/delta)) = 5.2575
          = alpha_star_euclidean(V=50257, embed_norm=R, auto_clock=True)

giving 4.6649 / 3.9811 / 2.4436 / 1.8338 for the four R above (setup.md).
ALPHA_MAX is pinned to null: TruncatedScheduleWrapper would rescale alpha
affinely and destroy the autonomous clock (see the parent RESULTS.md, section 7).

The parent runs cannot be inherited here — they kept only one periodic
checkpoint (CKPT_EVERY=5000, save_top_k=1), so every cell is trained fresh.

Staged use (the grid is large; ~84 GB and ~8.5 h on 4 GPUs per cell):
  # stage 1: LR x R at the closed-form truncation, seed 1        (8 cells)
  python experiments/eflm_rescale_auto_trunc_tinystories_256_history/sweep.py
  # stage 2: truncation around tau* at the stage-1 winner        (3 cells)
  python .../sweep.py --lrs 1e-3 --rhos 0.5 --mults 0.85 1.15 1.25
  # stage 3: seed replication of the finalists                   (4 cells)
  python .../sweep.py --lrs 1e-3 --rhos 0.5 --mults 1.0 --seeds 2 3
  python .../sweep.py --dry-run
  # or let the babysitter walk the stages: resubmits dead cells, and advances
  # to the next stage once the current one is complete
  python .../sweep.py --auto

Idempotent: skips cells whose eval/samples_genppl.json exists or whose job name
is already in squeue; resubmitting the same OUTPUT_DIR resumes from last.ckpt.
"""
import argparse
import getpass
import glob
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

EXP = 'eflm_rescale_auto_trunc_tinystories_256_history'
STEM = 'eflm_rescale_auto_truncation'
V = 50257
C = math.sqrt(2 * math.log(2 * (V - 1) / 0.1))   # 5.2575
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32
MAX_STEPS = 30000
CKPT_EVERY = 1000     # setup.md: checkpoint every 1k steps ...
SAVE_TOPK = -1        # ... and keep every one of them
LRS = ['3e-4', '1e-3']
RHOS = ['0.05', '0.1', '0.5', '1']
MULTS = ['1.0']
SEEDS = ['1']         # seed > 1 gets an _s{seed} tag suffix; seed 1 keeps the bare name
STAGE2_MULTS = ['0.85', '1.15', '1.25']   # truncation axis, run at the stage-1 winner
STAGE3_SEEDS = ['2', '3']                 # replication of the two stage-1 finalists
ENT_MIN = 3.0         # anti-collapse gate: a degenerate cell has the LOWEST GenPPL
MIN_FREE_GB = 800     # refuse to open a new stage without room for it (~84 GB/cell)
# Pin every cell to ONE node. Training is bit-deterministic given (seed, node):
# this sweep's lr 3e-4, R=1 cell reproduced the parent's to 16 significant
# digits (val/nll 2.896428286072464) because both ran on kuleshov-compute-03,
# while the three cells whose parent runs used a different GPU model differed by
# 0.79-2.94 GenPPL. That is the same magnitude as the effects being measured, so
# letting SLURM pick the node would confound the R and LR axes with hardware.
NODE = 'kuleshov-compute-03'


def tau_max_of(rho, mult):
    """m * tau*(R); tau*(R) = log(1 + C/R) is the decode point on this clock."""
    return f'{float(mult) * math.log1p(C / float(rho)):.4f}'


def tag_of(lr, rho, mult, seed='1'):
    sfx = '' if str(seed) == '1' else f'_s{seed}'
    return f'eflmrath_lr-{lr}_r-{rho}_m-{mult}{sfx}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(lr, rho, mult, seed, tdir):
    tau = tau_max_of(rho, mult)
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
        echo "[$(date)] TRAIN lr={lr} R={rho} m={mult} seed={seed} TAU_MAX={tau} on $(hostname)"
        RHO={rho} TAU_MAX={tau} ALPHA_MAX=null SNR_CE=false LR={lr} SEED={seed} SEQ_LEN={SEQ_LEN} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag_of(lr, rho, mult, seed)} WANDB_GROUP={EXP} \\
            DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} \\
            CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/{STEM}.sh
        echo "[$(date)] EVAL"
        RHO={rho} TAU_MAX={tau} ALPHA_MAX=null SNR_CE=false SEQ_LEN={SEQ_LEN} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{STEM}.sh
        echo "[$(date)] DONE"
        ''')


def done_cells(out):
    """Completed cells as {(lr, R, m, seed): GenPPL or None if collapsed}."""
    res = {}
    for f in glob.glob(f'{out}/*/eval/samples_genppl.json'):
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        m = re.fullmatch(r'eflmrath_lr-([\w.-]+)_r-([\d.]+)_m-([\d.]+)(?:_s(\d+))?',
                         tag)
        if not m:
            continue
        try:
            g = json.load(open(f))
        except Exception:
            continue
        gen, ent = g.get('gen_ppl_first_chunk_retok'), g.get('entropy')
        key = (m[1], m[2], m[3], m[4] or '1')
        # entropy < ENT_MIN => degenerate; such a cell must never win a ranking.
        res[key] = gen if (gen is not None and ent is not None
                           and ent >= ENT_MIN) else None
    return res


def next_stage(out):
    """The cells the next incomplete stage needs, or [] if there is nothing to do.

    Stage 1 (LR x R at m=1.0) -> stage 2 (truncation at the winner) ->
    stage 3 (seeds 2,3 at the two finalists). A stage opens only once the
    previous one has a GenPPL for every one of its cells.
    """
    done = done_cells(out)
    stage1 = list(itertools.product(LRS, RHOS, ['1.0'], ['1']))
    if any(c not in done for c in stage1):
        return stage1
    ranked = sorted((c for c in stage1 if done[c] is not None),
                    key=lambda c: done[c])
    if not ranked:                      # every cell collapsed: nothing to select on
        return []
    lr, R, _, _ = ranked[0]
    stage2 = [(lr, R, m, '1') for m in STAGE2_MULTS]
    if any(c not in done for c in stage2):
        return stage2
    stage3 = [(c[0], c[1], '1.0', s) for c in ranked[:2] for s in STAGE3_SEEDS]
    return [c for c in stage3 if c not in done]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    ap.add_argument('--mults', nargs='+', default=MULTS)
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    ap.add_argument('--auto', action='store_true',
                    help='submit whatever the next incomplete stage needs')
    args = ap.parse_args()

    logs = f'{REPO}/experiments/{EXP}/logs'
    out = f'{REPO}/outputs/{EXP}'
    if args.auto:
        cells = next_stage(out)
        if not cells:
            print(f'{EXP}: all stages complete')
            return
        free_gb = shutil.disk_usage(REPO).free // 2**30
        if free_gb < MIN_FREE_GB:
            print(f'{EXP}: only {free_gb} GB free (< {MIN_FREE_GB}), not opening '
                  f'a stage of {len(cells)} cells')
            return
    else:
        cells = list(itertools.product(args.lrs, args.rhos, args.mults, args.seeds))
    shape = ('next incomplete stage' if args.auto else
             f'{len(args.lrs)} lr x {len(args.rhos)} R x {len(args.mults)} m '
             f'x {len(args.seeds)} seed')
    print(f'{EXP}: {len(cells)} cells ({shape}), {DEVICES} GPU each, '
          f'~{31 * 2.72 * len(cells):.0f} GB of checkpoints')
    if args.dry_run:
        for c in cells:
            print(f'  {tag_of(*c):40} tau_max={tau_max_of(c[1], c[2])}')
        print('\n--- example body (first cell) ---\n'
              + job_body(*cells[0], f'{out}/{tag_of(*cells[0])}'))
        return

    os.makedirs(logs, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for lr, rho, mult, seed in cells:
        tag = tag_of(lr, rho, mult, seed)
        if (os.path.exists(f'{out}/{tag}/eval/samples_genppl.json')
                or tag in active):
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='desa', nodelist=NODE,
                      gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=8,
                      mem='64G', time='2-00:00:00')
        jid = slurm.sbatch(job_body(lr, rho, mult, seed, f'{out}/{tag}'),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag} (tau_max={tau_max_of(rho, mult)}): job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
