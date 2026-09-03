#!/usr/bin/env python
"""simpflm_auto_trunc_tinystories_256 — SimpFLM on the TRUNCATED AUTONOMOUS
clock, TinyStories seq 256: LR x truncation x {plain, adaptive}.

SimpFLM is E-FLM with the word-embedding matrix replaced by the DIAGONAL
R * I_V (`algo.SimpFLM`): every "embedding" is a one-hot simplex vertex, so the
Gaussian Euclidean flow runs in the one-hot (logit) space R^V. R is the radius of the
simplex sphere: `algo.rho_min = rho_max = RHO` (they must be equal) pins every
vertex to norm R.
With R = 1 (setup.md) and V = 50257, delta = 0.1:

    alpha*(1) = alpha_star_euclidean(V=50257, embed_norm=1) = 0.8402
    t* = 1 - alpha* = 0.1598,   tau*(1) = -log(t*) = log(1 + C) = 1.8338

with C = sqrt(2 log(2(V-1)/delta)) = 5.2575. On the autonomous clock
1 - alpha_t = exp(-tau), tau = TAU_MAX (1 - t), so TAU_MAX *is* the truncation
(the noise-fraction floor is exp(-TAU_MAX)) and the swept knob is the RELATIVE
multiplier m: TAU_MAX = m * tau*(1). m = 1 stops exactly at the decode point,
m < 1 before it, m > 1 past it toward the untruncated horizon 6.9078 = -log 1e-3
(m = 3.77). Scaling TAU_MAX is the right parameterization because
`TruncatedScheduleWrapper` rescales alpha AFFINELY, which would turn the noise
fraction into const + scale*exp(-tau) and break the autonomy of the clock.

  LR   in {3e-4, 1e-3, 5e-3}                    (setup.md)
  m    in {0.7, 0.85, 1.0, 1.25, 1.5}           "vary truncation a little bit"
  arm  in {trunc, trunc_ada}                    +/- the adaptive time remap

Fixed: small-flm (flm-dit, 768 wide / 12 blocks / 12 heads), 30k steps, global
batch 512 (DEVICES=4 x PER_GPU_BS=32 x accum 4), seq 256, bf16, EMA 0.9999,
AdamW wd 0, betas (0.9, 0.999), eps 1e-8, clip 1.0, plain CE. Eval: ppl_eval +
sample_eval (GenPPL), exact velocity, top_k_v = 1, 180 steps, greedy last.

Usage:
  # stage 1 (LR ladder at the closed-form truncation)
  python experiments/simpflm_auto_trunc_tinystories_256/sweep.py
  # stage 2 (truncation around tau*, at the winning LR)
  python .../sweep.py --lrs 1e-3 --mults 0.7 0.85 1.25 1.5
  # stage 3 (adaptive remap + seed replication at the winner)
  python .../sweep.py --lrs 1e-3 --mults 1.0 --arms trunc trunc_ada --seeds 1 2 3
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

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
# Short, on the same filesystem as the checkpoints -- see job_body's TMPDIR note.
TMP_BASE = '/share/desa/nfs02/sc3379/tmp'
sys.path.insert(0, REPO)
from noise_schedules import alpha_star_euclidean  # noqa: E402

EXP = 'simpflm_auto_trunc_tinystories_256'
STEMS = {'trunc': 'simpflm_auto_truncation',
         'trunc_ada': 'simpflm_auto_truncation_adaptive'}
V = 50257
RHOS = ['1.0']       # R: the radius of the simplex sphere (rho_min = rho_max).
                     # Only the ratio noise/R enters the bound, so R moves the
                     # decode point: tau*(R) = log(1 + C/R). Swept with --rhos.
SEQ_LEN = 256
DEVICES = 4
PER_GPU_BS = 32      # 13.4 GiB peak at seq 256 (measured, RTX 6000 Ada)
CKPT_EVERY = 5000
MAX_STEPS = 30000
LRS = ['3e-4', '1e-3', '5e-3']
MULTS = ['1.0']
ARMS = ['trunc']
SEEDS = ['1']        # seed>1 gets an _s{seed} tag suffix; seed 1 keeps the bare name


def tau_max_of(rho, mult):
    """m * tau*(R); tau*(R) = -log b*(R) = log(1 + C/R) is the decode point.

    R and the truncation are coupled through tau*(R), which is exactly why the
    truncation is swept as the RELATIVE multiplier m -- it holds the
    R-dependence of the endpoint fixed while R varies.
    """
    tau_star = alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=True)
    return f'{float(mult) * tau_star:.4f}'


def tag_of(lr, rho, mult, arm, seed='1'):
    # R = 1 keeps the bare name so the stage-1 cells stay recognised as done.
    sfx = '' if str(rho) == '1.0' else f'_r-{rho}'
    sfx += '_ada' if arm == 'trunc_ada' else ''
    sfx += '' if str(seed) == '1' else f'_s{seed}'
    return f'simpflm_lr-{lr}_m-{mult}{sfx}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(lr, rho, mult, arm, seed, tdir):
    tau = tau_max_of(rho, mult)
    tmp_base = TMP_BASE
    stem = STEMS[arm]
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        # Lightning's _atomic_save opens inside an fsspec transaction, which
        # stages the WHOLE checkpoint through tempfile.mkstemp() before
        # shutil.move()ing it to the real target. mkstemp honours $TMPDIR, and
        # the cluster environment sets TMPDIR=/home/sc3379/tmp -- a filesystem
        # with ~300 MB free -- so every 2.7 GB save died with ENOSPC at the
        # first checkpoint (jobs 387971-3 and 451109, step 5000, every node).
        # Staging on the target filesystem also makes the move a rename.
        #
        # This path must stay SHORT: multiprocessing/DataLoader put their Unix
        # sockets under $TMPDIR ($TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX) and
        # sockaddr_un caps the path at 108 bytes. Pointing it at the per-cell
        # output dir (~127 chars) died instantly with "AF_UNIX path too long"
        # (jobs 452554-6), so keep the base short and key it on the job id.
        export TMPDIR={tmp_base}/$SLURM_JOB_ID
        mkdir -p "$TMPDIR"
        trap 'rm -rf "$TMPDIR"' EXIT
        cd {REPO}
        if [ -f {edir}/samples_genppl.json ]; then
            echo "[$(date)] cell already completed -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN lr={lr} R={rho} m={mult} arm={arm} seed={seed} TAU_MAX={tau} on $(hostname)"
        RHO={rho} TAU_MAX={tau} LR={lr} SEED={seed} SEQ_LEN={SEQ_LEN} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag_of(lr, rho, mult, arm, seed)} WANDB_GROUP={EXP} \\
            DEVICES={DEVICES} PER_GPU_BS={PER_GPU_BS} MAX_STEPS={MAX_STEPS} \\
            CKPT_EVERY={CKPT_EVERY} SAVE_TOPK=1 \\
            bash scripts/train/tinystories/{stem}.sh
        echo "[$(date)] EVAL"
        RHO={rho} TAU_MAX={tau} SEQ_LEN={SEQ_LEN} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 \\
            bash scripts/sample/tinystories/{stem}.sh
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--lrs', nargs='+', default=LRS)
    ap.add_argument('--rhos', nargs='+', default=RHOS)
    ap.add_argument('--mults', nargs='+', default=MULTS)
    ap.add_argument('--arms', nargs='+', default=ARMS, choices=list(STEMS))
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    args = ap.parse_args()

    logs = f'{REPO}/experiments/{EXP}/logs'
    out = f'{REPO}/outputs/{EXP}'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)
        os.makedirs(out, exist_ok=True)

    cells = list(itertools.product(args.lrs, args.rhos, args.mults,
                                   args.arms, args.seeds))
    print(f'{EXP}: {len(cells)} cells '
          f'({len(args.lrs)} lr x {len(args.rhos)} R x {len(args.mults)} m x '
          f'{len(args.arms)} arm x {len(args.seeds)} seed), {DEVICES} GPU each')
    if args.dry_run:
        for c in cells:
            print(f'  {tag_of(*c):40} R={c[1]:<5} tau_max={tau_max_of(c[1], c[2])}')
        print('\n--- example body (first cell) ---\n'
              + job_body(*cells[0], f'{out}/{tag_of(*cells[0])}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for lr, rho, mult, arm, seed in cells:
        tag = tag_of(lr, rho, mult, arm, seed)
        if (os.path.exists(f'{out}/{tag}/eval/samples_genppl.json')
                or tag in active):
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=8,
                      mem='64G', time='2-00:00:00')
        jid = slurm.sbatch(job_body(lr, rho, mult, arm, seed, f'{out}/{tag}'),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag} (R={rho}, tau_max={tau_max_of(rho, mult)}): '
              f'job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
