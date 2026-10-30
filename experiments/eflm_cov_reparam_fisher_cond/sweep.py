#!/usr/bin/env python
"""eflm_cov_reparam_fisher_cond GOAL2 (setup.md) — the six EFLM checkpoints of the angular-Fisher table retrained at
embedding dim {16, 48, 768} x seed {1, 2, 3}, everything else as the original run.

  arm    naive = eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-{R}_m-1.0
                 (FORWARD_TYPE=naive, COV_WEIGHT=false)
         gauss = eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-{R}_lr-3e-4
                 (FORWARD_TYPE=gaussian, COV_WEIGHT=true: CE x |alpha_t'|)
  R      {0.1, 0.5, 1.0}  (rho_min = rho_max = R), TAU_MAX = tau*(R) on the autonomous clock
  dim    model.embed_dim {16, 48, 768}; 768 = hidden_size (the original model), 16 / 48 lift the state into the
         768-wide DiT with a linear in_proj (models/sphere_dit.py)
  seed   {1, 2, 3}
Shared: small-sphere-dit, ngpt init, lr 3e-4, seq 256, 30k steps, global batch 512, bf16, EMA 0.9999.
The (dim 768, seed 1) cells are the original runs, symlinked in, not retrained.
Eval (1 GPU): ppl_eval -> eval/ppl.json, sample_eval -> eval/samples_genppl.json (exact velocity, top_k_v 1,
180 steps, greedy last), as the original sweeps.

ORCHESTRATION ONLY: one SLURM job per cell runs scripts/{train,sample}/tinystories/eflm_rescale_auto_truncation.sh.
Idempotent + resumable: skips a cell whose eval/samples_genppl.json exists or whose job is queued (on either
cluster); resubmitting resumes from checkpoints/last.ckpt.
  --cluster unicorn  4 GPUs x 32 per GPU x accumulation 4 (the original layout), partitions thickstun,desa
  --cluster g3       1 B300 x 128 x accumulation 4 (same global batch), qos g3, --requeue, checkpoint every 1k steps.
                     TRAIN ONLY, from G3_REPO (G3 has no /share/desa), outputs + logs under it, TRAIN_DONE marker at
                     the end. No eval on G3: the env's NVRTC (CUDA 12.8) cannot JIT for sm_103 (torch.special.entr).
  --eval-g3          (run on Unicorn) for each G3 cell with TRAIN_DONE: link OUT/<run>/checkpoints to the G3 run and
                     submit a 1-GPU Unicorn eval job writing OUT/<run>/eval, so every cell is evaluated on Unicorn.

Usage:  python experiments/eflm_cov_reparam_fisher_cond/sweep.py [--dry-run] [--cluster unicorn|g3] [--eval-g3]
            [--arms naive gauss] [--rhos 0.1 0.5 1.0] [--dims 16 48 768] [--seeds 1 2 3]
"""
import argparse
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

PROJECT = 'eflm_cov_reparam_fisher_cond'
OUT = f'{REPO}/outputs/{PROJECT}'
STEM = 'eflm_rescale_auto_truncation'
V = 50257
# arm -> (FORWARD_TYPE, COV_WEIGHT, original run of (R, dim 768, seed 1))
ARMS = {
  'naive': ('naive', 'false', lambda r: f'eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-{r:g}_m-1.0'),
  'gauss': ('gaussian', 'true', lambda r: f'eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-{r}_lr-3e-4'),
}
RHOS = ['0.1', '0.5', '1.0']
DIMS = ['16', '48', '768']
SEEDS = ['1', '2', '3']
HIDDEN = '768'
LR, SEQ_LEN, MAX_STEPS = '3e-4', 256, 30000
CLUSTERS = {  # name -> (DEVICES, PER_GPU_BS, CKPT_EVERY, simple_slurm kwargs)
  'unicorn': (4, 32, 5000, dict(partition='thickstun,desa', exclude='desa-compute-01', gres='gpu:4',
                                cpus_per_task=8, mem='64G', time='3-00:00:00')),
  'g3': (1, 128, 1000, dict(clusters='g3', partition='g3', qos='g3', account='thickstun',
                            gres='gpu:nvidia_b300_sxm6_ac:1', cpus_per_task=16, mem='128G', time='1-00:00:00')),
}
# G3 does not mount /share/desa: it runs from a checkout + env on /share/thickstun (same code, data copied over).
G3_REPO = '/share/thickstun/sychou/workspace_g3/research/s-flm-dev1/s-flm'
SITES = {  # cluster -> (repo, env bin)
  'unicorn': (REPO, ENVBIN),
  'g3': (G3_REPO, '/share/thickstun/sychou/anaconda3/envs/sfm/bin'),
}
EVAL_SLURM = dict(partition='thickstun,desa', exclude='desa-compute-01', gres='gpu:1', cpus_per_task=4, mem='32G',
                  time='04:00:00')


def out_of(cluster):
  return f'{SITES[cluster][0]}/outputs/{PROJECT}'


def is_done(tag):  # every cell is evaluated on Unicorn, into OUT
  return os.path.exists(f'{OUT}/{tag}/eval/samples_genppl.json')


def g3_trained(tag):
  return os.path.exists(f'{out_of("g3")}/{tag}/TRAIN_DONE')


def tag_of(arm, rho, dim, seed):
  return f'fwd-{arm}_r-{rho}_ed-{dim}_seed-{seed}'


def tau_of(rho):
  return f'{alpha_star_euclidean(V, embed_norm=float(rho), auto_clock=True):.4f}'


def link_originals(dry):
  """(dim 768, seed 1) = the original runs: symlink them in instead of retraining."""
  for arm, rho in itertools.product(ARMS, RHOS):
    dst = f'{OUT}/{tag_of(arm, rho, HIDDEN, "1")}'
    src = os.path.realpath(f'{REPO}/outputs/{ARMS[arm][2](float(rho))}')
    if os.path.lexists(dst) or not os.path.isdir(src):
      continue
    print(f'  link {os.path.basename(dst)} -> {src}')
    if not dry:
      os.symlink(src, dst)


def active_jobnames():
  names = set()
  for cmd in (['squeue', '-h', '-u', 'sc3379', '-o', '%j'], ['squeue', '-M', 'g3', '-h', '-u', 'sc3379', '-o', '%j']):
    try:
      names |= set(subprocess.run(cmd, capture_output=True, text=True).stdout.split())
    except Exception:
      pass
  return names


def job_body(arm, rho, dim, seed, cluster):
  fwd, cov, _ = ARMS[arm]
  devices, bs, ckpt_every, _ = CLUSTERS[cluster]
  repo, envbin = SITES[cluster]
  tag = tag_of(arm, rho, dim, seed)
  tdir = f'{out_of(cluster)}/{tag}'
  model = model_vars(rho, fwd, dim)
  done = f'{tdir}/TRAIN_DONE' if cluster == 'g3' else f'{tdir}/eval/samples_genppl.json'
  if cluster == 'g3':
    tail = f'touch {tdir}/TRAIN_DONE\necho "[$(date)] TRAINED (eval runs on Unicorn: sweep.py --eval-g3)"'
  else:
    tail = eval_cmd(model, f'{tdir}/checkpoints/last.ckpt', f'{tdir}/eval')
  return env_head(envbin, repo) + textwrap.dedent(f'''\
    if [ -f {done} ]; then
        echo "[$(date)] cell already completed -> no-op"; exit 0
    fi
    echo "[$(date)] TRAIN {tag} on $(hostname)"
    {model} COV_WEIGHT={cov} LR={LR} SEED={seed} \\
        OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={PROJECT} \\
        DEVICES={devices} PER_GPU_BS={bs} MAX_STEPS={MAX_STEPS} CKPT_EVERY={ckpt_every} SAVE_TOPK=1 \\
        bash scripts/train/tinystories/{STEM}.sh
    ''') + tail + '\n'


def model_vars(rho, fwd, dim):
  return f'RHO={rho} TAU_MAX={tau_of(rho)} ALPHA_MAX=null FORWARD_TYPE={fwd} EMBED_DIM={dim} SEQ_LEN={SEQ_LEN}'


def eval_cmd(model, ckpt, out):
  return textwrap.dedent(f'''\
    echo "[$(date)] EVAL"
    {model} \\
        CKPT_PATH={ckpt} OUTPUT_DIR={out} DEVICES=1 \\
        bash scripts/sample/tinystories/{STEM}.sh
    echo "[$(date)] DONE"''')


def eval_g3_body(arm, rho, dim, seed):
  """Unicorn 1-GPU eval of a G3-trained cell (checkpoint read from /share/thickstun)."""
  tag = tag_of(arm, rho, dim, seed)
  return env_head(ENVBIN, REPO) + eval_cmd(model_vars(rho, ARMS[arm][0], dim),
                                           f'{out_of("g3")}/{tag}/checkpoints/last.ckpt', f'{OUT}/{tag}/eval') + '\n'


def env_head(envbin, repo):
  return textwrap.dedent(f'''\
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export SLURM_JOB_NAME=bash
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCHDYNAMO_DISABLE=1
    export WANDB_MODE=offline
    export TMPDIR=/tmp/{PROJECT}_$SLURM_JOB_ID MPLCONFIGDIR=/tmp/{PROJECT}_$SLURM_JOB_ID/mpl WANDB_CACHE_DIR=/tmp/{PROJECT}_$SLURM_JOB_ID/wandb
    mkdir -p $TMPDIR $MPLCONFIGDIR $WANDB_CACHE_DIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={envbin}:$PATH
    cd {repo}
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument('--cluster', default='unicorn', choices=list(CLUSTERS))
  ap.add_argument('--eval-g3', action='store_true', help='submit Unicorn evals of G3-trained cells')
  ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
  ap.add_argument('--rhos', nargs='+', default=RHOS)
  ap.add_argument('--dims', nargs='+', default=DIMS)
  ap.add_argument('--seeds', nargs='+', default=SEEDS)
  args = ap.parse_args()

  if args.eval_g3:
    args.cluster = 'unicorn'
  if not args.dry_run:
    logs = f'{SITES[args.cluster][0]}/experiments/{PROJECT}/logs'
    os.makedirs(logs, exist_ok=True)
    os.makedirs(out_of(args.cluster), exist_ok=True)
  link_originals(args.dry_run)
  # submission order: seed, then dim, so every (arm, R, dim) gets its first seed before any gets a second
  cells = [(a, r, d, s) for s, d, a, r in itertools.product(args.seeds, args.dims, args.arms, args.rhos)]
  print(f'{PROJECT}: {len(cells)} cells ({len(args.arms)} arm x {len(args.rhos)} R x {len(args.dims)} dim x '
        f'{len(args.seeds)} seed) on {args.cluster}')
  if args.dry_run:
    for c in cells:
      state = '  [done]' if is_done(tag_of(*c)) else '  [trained on G3]' if g3_trained(tag_of(*c)) else ''
      print(f'  {PROJECT}_{tag_of(*c):34} TAU_MAX={tau_of(c[1])}{state}')
    body = eval_g3_body(*cells[0]) if args.eval_g3 else job_body(*cells[0], args.cluster)
    print('\n--- example body (first cell) ---\n' + body)
    return

  active = active_jobnames()
  n_sub = n_skip = 0
  for c in cells:
    tag = tag_of(*c)
    if args.eval_g3:
      jobname = f'{PROJECT}_eval_{tag}'
      if is_done(tag) or not g3_trained(tag) or jobname in active:
        n_skip += 1
        continue
      os.makedirs(f'{OUT}/{tag}', exist_ok=True)
      if not os.path.lexists(f'{OUT}/{tag}/checkpoints'):
        os.symlink(f'{out_of("g3")}/{tag}/checkpoints', f'{OUT}/{tag}/checkpoints')
      slurm = Slurm(job_name=jobname, output=f'{logs}/eval_{tag}_%j.log', chdir=REPO, ntasks=1, **EVAL_SLURM)
      jid = slurm.sbatch(eval_g3_body(*c), sbatch_cmd='sbatch --requeue', verbose=False)
      print(f'  submitted eval {tag}: job {jid}')
      n_sub += 1
      continue
    jobname = f'{PROJECT}_{tag}'
    if is_done(tag) or g3_trained(tag) or jobname in active:
      n_skip += 1
      continue
    slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log', chdir=SITES[args.cluster][0], ntasks=1,
                  **CLUSTERS[args.cluster][3])
    jid = slurm.sbatch(job_body(*c, args.cluster), sbatch_cmd='sbatch --requeue', verbose=False)
    print(f'  submitted {tag}: job {jid}')
    n_sub += 1
  print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
  main()
