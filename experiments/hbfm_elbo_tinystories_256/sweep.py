#!/usr/bin/env python
"""hbfm_elbo_tinystories_256 -- HBFM on TinyStories (seq 256) trained on the NELBO, with the path-KL
rate estimated by Monte Carlo over sampled words (`algo.elbo_num_words`, `Loss.bridge_loss_elbo_mc`).

Spec: `setup.md`. Shared by both arms: small DiT 768/12/12, naive readout in float32, (H^3)^16 =
embed_dim 48 at K = -0.5, trunc_exp proposal of rate 3, log-linear noise, LR 3e-4, 30k steps, global
batch 512, bf16, EMA 0.9999, loss_type elbo with S = NUM_WORDS sampled words per token.

Arms (setup.md "What differs"):
  ada   curvature_bound 4 (lambda = 4^tanh(h)), time_range_upper_bound 1, sampler t_max 1,
        + lambda read per keep-{step}.ckpt after the eval
  fix   curvature_bound 1, time_range_upper_bound 0.5, sampler t_max 0.5

Eval per cell (setup.md's commented protocol): validation PPL (the CE objective) + 64 samples at
180 steps, TOPK_VELOCITY=1, greedy last step; then the exact NELBO (elbo_num_words unset) over the
full validation set, LOSS_TYPE=elbo SEED=1 EVAL_BS=16 -> eval/nelbo.json.

Sites:
  a100   ch2263@unicorn, nlplarge (A100 80GB): 1 GPU x micro-batch 32 per cell; code copy at R_REPO
         (`--sync-code`), results back with `--pull`; checkpoints on the node's /scratch (experiment.py).
  a6000  sc3379, desa / thickstun without the 2080 Ti node: 4 GPUs x micro-batch 8 per cell.
The micro-batches are half the CE runs' (the reconstruction term is a second forward pass).

ORCHESTRATION ONLY. Idempotent: a cell with eval/nelbo.json (local, after `--pull`) or a queued job
of the same name is skipped; a resubmitted cell resumes from checkpoints/last.ckpt.

Usage:  python sweep.py --sync-code
        python sweep.py --site a100 [--arms ada,fix] [--seeds 1] [--dry-run]
        python sweep.py --pull
"""
import argparse
import os
import subprocess
import textwrap

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
NAME = 'hbfm_elbo_tinystories_256'
EXP = f'{REPO}/experiments/{NAME}'
OUT = f'{REPO}/outputs/{NAME}'
JOB = 'hbfm_elbo_ts'
KEY = '/home/sc3379/.ssh/unicorn_internal'
REMOTE = 'ch2263@unicorn-login-02.coecis.cornell.edu'
R_REPO = '/home/ch2263/syc_workspace/s-flm-hbfm-elbo'
CODE = ['*.py', 'configs', 'models', 'numeric', 'callbacks', 'scripts', 'data', 'tests', 'experiments/hbfm_rlf_ada_curv',
        'experiments/hbfm_rlf_ada_curv_tinystories_256']   # curvature_ts.py and its load_cell

NUM_WORDS = 16   # words drawn from p_theta per token for the path-KL rate

SITES = {
  'a100': dict(remote=True, repo=R_REPO, envbin='/home/ch2263/miniconda3/envs/sfm/bin',
               cache='/home/ch2263/syc_workspace/s-flm_data_home/data_cache', tmp='/home/ch2263/tmp',
               devices=1, per_gpu_bs=32, workers=4,
               sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G'),
  'a6000': dict(remote=False, repo=REPO, envbin='/home/sc3379/anaconda3/envs/sfm/bin',
                cache=f'{REPO}/data_cache', tmp='/home/sc3379/tmp',
                devices=4, per_gpu_bs=8, workers=4,
                sbatch='--partition=desa,thickstun --exclude=desa-compute-01 --gres=gpu:4 --cpus-per-task=16 --mem=128G'),
}

ARMS = {
  'ada': dict(cb=4, ub=1.0, t_max=1.0),
  'fix': dict(cb=1, ub=0.5, t_max=0.5),
}


def tag_of(arm, seed):
  return f'curv-{arm}_seed-{seed}'


def knobs(arm):
  a = ARMS[arm]
  return (f'FORWARD_TYPE=naive READOUT_PRECISION=float32 EMBED_DIM=48 FACTOR_DIM=3 GAUSS_CURV=-0.5 '
          f'CURV_BOUND={a["cb"]} TIME_CONVERSION_MODE=trunc_exp PROPOSAL_RATE=3 '
          f'TIME_RANGE_UPPER_BOUND={a["ub"]:g} T_MAX={a["t_max"]:g} GLOBAL_BATCH=512 SEQ_LEN=256')


def body(s, arm, seed):
  tag = tag_of(arm, seed)
  tdir = f'{s["repo"]}/outputs/{NAME}/{tag}'
  ckpt = f'CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 CACHE_DIR={s["cache"]}'
  text = textwrap.dedent(f'''\
    #!/bin/bash
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
    export TMPDIR={s['tmp']}/{JOB}_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={s['envbin']}:$PATH
    cd {s['repo']}
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo "[$(date)] TRAIN {tag}"
    python experiment.py {tdir} {NAME} {tag}
    LR=3e-4 SEED={seed} {knobs(arm)} LOSS_TYPE=elbo ELBO_NUM_WORDS={NUM_WORDS} CACHE_DIR={s['cache']} \\
      OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={NAME} DEVICES={s['devices']} PER_GPU_BS={s['per_gpu_bs']} \\
      NUM_WORKERS={s['workers']} MAX_STEPS=30000 CKPT_EVERY=1000 KEEP_EVERY=5000 bash scripts/train/tinystories/hbfm.sh
    echo "[$(date)] EVAL {tag}"
    {knobs(arm)} TOPK_VELOCITY=1 STEPS=180 {ckpt} bash scripts/sample/tinystories/hbfm.sh
    echo "[$(date)] NELBO {tag}"
    {knobs(arm)} LOSS_TYPE=elbo SEED=1 EVAL_BS=16 RUN_SAMPLE_EVAL=false PPL_JSON={tdir}/eval/nelbo.json {ckpt} \\
      bash scripts/sample/tinystories/hbfm.sh
    ''')
  if ARMS[arm]['cb'] > 1:   # lambda per keep-{step}.ckpt -> <cell>/curvature.json
    text += f'python experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py {tdir}\n'
  return text + f'echo "[$(date)] DONE {tag}"\n'


def run(s, cmd, stdin=None):
  """A shell command at the site (over ssh with a login shell for the a100 site)."""
  if s['remote']:
    cmd = ['ssh', '-o', 'BatchMode=yes', '-i', KEY, REMOTE, "bash -lc '" + cmd.replace("'", "'\\''") + "'"]
  else:
    cmd = ['bash', '-c', cmd]
  return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--site', choices=sorted(SITES))
  ap.add_argument('--arms', default='ada,fix')
  ap.add_argument('--seeds', default='1')
  ap.add_argument('--sync-code', action='store_true')
  ap.add_argument('--pull', action='store_true')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  ssh = f'ssh -o BatchMode=yes -i {KEY}'

  if args.sync_code:   # the code only; data, outputs and checkpoints stay where they are
    inc = ' '.join(f"--include='/{p}'" if '*' in p else f"--include='/{p}/***'" for p in CODE)
    os.system(f"rsync -a --exclude='__pycache__/' --include='/experiments/' {inc} --exclude='/*' "
              f"--exclude='/experiments/*' -e '{ssh}' {REPO}/ {REMOTE}:{R_REPO}/")
    return
  if args.pull:        # a100 results (eval JSONs, hydra configs, logs; no checkpoints) -> local
    os.system(f"rsync -a --exclude='checkpoints' --exclude='wandb/' -e '{ssh}' "
              f"{REMOTE}:{R_REPO}/outputs/{NAME}/ {OUT}/")
    os.system(f"rsync -a -e '{ssh}' {REMOTE}:{R_REPO}/experiments/{NAME}/logs/ {EXP}/logs/a100/")
    return

  s = SITES[args.site]
  queued = set()   # a cell queued at either site counts
  if not args.dry_run:
    for site in SITES.values():
      queued |= set(run(site, f'squeue -h -u {"ch2263" if site["remote"] else "sc3379"} -o %j').stdout.split())
  logs = f'{s["repo"]}/experiments/{NAME}/logs'
  for arm in args.arms.split(','):
    for seed in args.seeds.split(','):
      tag, jobname = tag_of(arm, seed), f'{JOB}_{tag_of(arm, seed)}'
      if os.path.exists(f'{OUT}/{tag}/eval/nelbo.json') or jobname in queued:
        print(f'  skip {tag}')
        continue
      if args.dry_run:
        print(f'--- {jobname} ({s["sbatch"]})\n{body(s, arm, seed)}')
        continue
      r = run(s, f'mkdir -p {logs} && sbatch --job-name={jobname} {s["sbatch"]} --time=7-00:00:00 --requeue '
                 f'--output={logs}/{jobname}_%j.log', stdin=body(s, arm, seed))
      print(f'  {jobname}: {(r.stdout + r.stderr).strip()}')


if __name__ == '__main__':
  main()
