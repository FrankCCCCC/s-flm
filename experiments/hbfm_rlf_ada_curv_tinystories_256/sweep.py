#!/usr/bin/env python
"""hbfm_rlf_ada_curv_tinystories_256 -- HBFM with the adaptive curvature on TinyStories (seq 256).

Spec: `setup.md`. Method: `experiments/hbfm_rlf_ada_curv/math.md`. Design: `EXPERIMENT.md`.

Every cell is the spec's HBFM run (small DiT 768/12/12, (H^3)^16 = embed_dim 48 at base curvature
K = -0.5, naive readout, log-linear noise, 30k steps, global batch 512, bf16, EMA 0.9999, AdamW wd 0,
clip 1.0, checkpoints every 1k + keep-{step}.ckpt every 5k) with the adaptive-curvature bound `cb`
(`algo.curvature_bound`; 1 = fixed curvature) and the knobs the task varies: time_conversion_mode,
time_exp_rate, time_range_upper_bound, LR in {3e-4, 5e-4, 1e-3}. Eval as
claude_test_hbfm_tinystories_256: valid PPL + 64 samples, 180 steps, exact velocity,
top_k_velocity 1, greedy last step; sampler horizon t_max = 3 (unit 1.5) for exp and = ub for the
bounded modes (t_max > ub is rejected at construction); exp cells are also read at t_max 0.5.

Three sites; a cell runs at one of them:
  a100   ch2263@unicorn, nlplarge-compute-01 (8x A100 80GB): 1 GPU x micro-batch 64 per cell. The
         code is a copy at R_REPO (`--sync-code`); results come back with `--pull`. Checkpoints live
         on the node's /scratch, linked from the run directory (`experiment.py`).
  a6000  sc3379, desa / thickstun: 4 GPUs (A6000 / A5000 / Ada) x micro-batch 16 per cell.
  2080ti sc3379, desa-compute-01 (8x 2080 Ti 11GB), when idle: 8 GPUs x micro-batch 16 per cell.

ORCHESTRATION ONLY. Idempotent: a cell with eval/samples_genppl.json (local, after `--pull`) or a
queued job of the same name is skipped; a resubmitted cell resumes from checkpoints/last.ckpt.

Usage:  python sweep.py --sync-code
        python sweep.py --stage 1 --site a100 [--only SUBSTR] [--seeds 1] [--dry-run]
        python sweep.py --eval-step N [--t-max T] --site a100 [--only SUBSTR]   (read keep-N.ckpt)
        python sweep.py --pull ;  python report.py
"""
import argparse
import os
import subprocess
import textwrap

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm'
EXP = f'{REPO}/experiments/hbfm_rlf_ada_curv_tinystories_256'
OUT = f'{REPO}/outputs/hbfm_rlf_ada_curv_tinystories_256'
NAME = 'hbfm_rlf_ada_curv_tinystories_256'
JOB = 'hbfm_adats'
KEY = '/home/sc3379/.ssh/unicorn_internal'
REMOTE = 'ch2263@unicorn-login-02.coecis.cornell.edu'
R_REPO = '/home/ch2263/syc_workspace/s-flm-adacurv'
CODE = ['*.py', 'configs', 'models', 'numeric', 'callbacks', 'scripts', 'data', 'tests']

SITES = {
  'a100': dict(remote=True, repo=R_REPO, envbin='/home/ch2263/miniconda3/envs/sfm/bin',
               cache='/home/ch2263/syc_workspace/s-flm_data_home/data_cache', tmp='/home/ch2263/tmp',
               devices=1, per_gpu_bs=64, workers=4,
               sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G',
               eval_sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=48G'),
  'a6000': dict(remote=False, repo=REPO, envbin='/home/sc3379/anaconda3/envs/sfm/bin',
                cache=f'{REPO}/data_cache', tmp='/home/sc3379/tmp',
                devices=4, per_gpu_bs=16, workers=4,
                sbatch='--partition=desa,thickstun --exclude=desa-compute-01 --gres=gpu:4 --cpus-per-task=16 --mem=128G',
                eval_sbatch='--partition=desa,thickstun --exclude=desa-compute-01 --gres=gpu:1 --cpus-per-task=4 --mem=48G'),
  '2080ti': dict(remote=False, repo=REPO, envbin='/home/sc3379/anaconda3/envs/sfm/bin',
                 cache=f'{REPO}/data_cache', tmp='/home/sc3379/tmp',
                 devices=8, per_gpu_bs=16, workers=1,
                 sbatch=('--partition=desa --nodelist=desa-compute-01 '
                         '--gres=gpu:nvidia_geforce_rtx_2080_ti:8 --cpus-per-task=10 --mem=120G'),
                 eval_sbatch=('--partition=desa --nodelist=desa-compute-01 '
                              '--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=32G')),
}

MODE_TAG = {'exp': 'exp', 'trunc_exp': 'texp', 'unif': 'unif'}
SPEC = dict(cb=4.0, mode='exp', rate=3.0, ub=None, lr='3e-4')   # setup.md's proposal, clock free


def cell(**kw):
  c = dict(SPEC)
  c.update(kw)
  return c


STAGES = {
  # Seed 1. Without a prompt lambda is one learned global clock (EXPERIMENT.md): a cell trains the
  # proposal rate / lambda on the clock range [0, lambda * ub]. References are the fixed-curvature
  # cells of claude_test_hbfm_tinystories_256 (GenPPL at the own horizon / at t_max 0.5; older code,
  # Ada / A6000 / A5000).
  '1': [
    cell(),                                    # A spec, exp rate 3 (unit 6): 15.58 +- 0.67 / 13.25 +- 0.56
    cell(lr='1e-3'),                           # L the spec at the largest allowed LR
    cell(mode='trunc_exp', ub=0.5),            # B the fixed optimum: 12.64 +- 0.43
    cell(mode='trunc_exp', ub=1.0),            # F wider ranges the clock may shrink back (spec read at t_max 1: 13.34)
    cell(mode='trunc_exp', ub=2.0),            # G
    cell(mode='unif', ub=2.0),                 # C mis-set, flat on [0, 2]: 17.18 / 15.80
    cell(rate=20.0),                           # D mis-set, exp rate 20 (unit 40): 19.01 / 16.36
  ],
  # Seeds 2-3 (--seeds 2,3) of the two best stage-1 cells (seed 1: F 11.91, B 12.56) and of the spec
  # point A (its seed 2 runs on the 2080 Ti node). LR 1e-3 lost to 3e-4 at A, so no LR axis.
  '2': [
    cell(mode='trunc_exp', ub=1.0),            # F
    cell(mode='trunc_exp', ub=0.5),            # B
    cell(),                                    # A
  ],
  # The trunc_exp range around the best cell F (ub 1, clock range lambda * ub ~ 0.28): ub 0.75 and 1.5
  # land the learned clock range near 0.2 and 0.5 (seed 1, 4x A6000 / A5000).
  '3': [
    cell(mode='trunc_exp', ub=0.75),
    cell(mode='trunc_exp', ub=1.5),
  ],
}


def t_max(c):
  return 3.0 if c['mode'] == 'exp' else c['ub']


def tag_of(c, seed):
  ub = 'na' if c['ub'] is None else f'{c["ub"]:g}'
  return (f'cb-{c["cb"]:g}_mode-{MODE_TAG[c["mode"]]}_rate-{c["rate"]:g}_ub-{ub}_tm-{t_max(c):g}'
          f'_lr-{c["lr"]}_seed-{seed}')


def knobs(c, tm):
  ub = 1.0 if c['ub'] is None else c['ub']
  return (f'FORWARD_TYPE=naive EMBED_DIM=48 FACTOR_DIM=3 GAUSS_CURV=-0.5 CURV_BOUND={c["cb"]:g} '
          f'TIME_CONVERSION_MODE={c["mode"]} PROPOSAL_RATE={c["rate"]:g} TIME_RANGE_UPPER_BOUND={ub:g} '
          f'T_MAX={tm:g} READOUT_PRECISION=float32 GLOBAL_BATCH=512 SEQ_LEN=256')


def preamble(s):
  return textwrap.dedent(f'''\
    #!/bin/bash
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
    export TMPDIR={s['tmp']}/{JOB}_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={s['envbin']}:$PATH
    cd {s['repo']}
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    md5sum algo.py models/hyperbolic_dit.py numeric/geo_bridge.py samplers.py configs/algo/hbfm.yaml \\
      scripts/train/tinystories/hbfm.sh scripts/sample/tinystories/hbfm.sh
    ''')


def eval_cmd(s, c, ckpt, out_dir, tm):
  return (f'{knobs(c, tm)} CACHE_DIR={s["cache"]} TOPK_VELOCITY=1 STEPS=180 \\\n'
          f'  CKPT_PATH={ckpt} OUTPUT_DIR={out_dir} DEVICES=1 bash scripts/sample/tinystories/hbfm.sh\n')


def train_body(s, c, seed):
  tag = tag_of(c, seed)
  tdir = f'{s["repo"]}/outputs/{NAME}/{tag}'
  body = preamble(s) + textwrap.dedent(f'''\
    echo "[$(date)] TRAIN {tag}"
    python experiment.py {tdir} {NAME} {tag}; ls -ld {tdir}/checkpoints 2>/dev/null || true
    LR={c['lr']} SEED={seed} {knobs(c, t_max(c))} CACHE_DIR={s['cache']} \\
      OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={NAME} DEVICES={s['devices']} PER_GPU_BS={s['per_gpu_bs']} \\
      NUM_WORKERS={s['workers']} MAX_STEPS=30000 CKPT_EVERY=1000 KEEP_EVERY=5000 bash scripts/train/tinystories/hbfm.sh
    echo "[$(date)] EVAL {tag}"
    ''') + eval_cmd(s, c, f'{tdir}/checkpoints/last.ckpt', f'{tdir}/eval', t_max(c))
  if c['mode'] == 'exp':   # the horizon the bounded cells sample at (claude_test_hbfm_tinystories_256)
    body += eval_cmd(s, c, f'{tdir}/checkpoints/last.ckpt', f'{tdir}/eval_tmax0.5', 0.5)
  return body + f'echo "[$(date)] DONE {tag}"\n'


def eval_body(s, c, seed, step, tm):
  tag = tag_of(c, seed)
  tdir = f'{s["repo"]}/outputs/{NAME}/{tag}'
  sub = f'eval_step{step}' + ('' if tm is None else f'_tmax{tm:g}')
  return (preamble(s) + f'echo "[$(date)] EVAL {tag} keep-{step}"\n'
          + eval_cmd(s, c, f'{tdir}/checkpoints/keep-{step}.ckpt', f'{tdir}/{sub}', t_max(c) if tm is None else tm))


def run(s, cmd, stdin=None):
  """A shell command at the site (over ssh for the a100 site, with a login shell for slurm)."""
  if s['remote']:
    cmd = ['ssh', '-o', 'BatchMode=yes', '-i', KEY, REMOTE, f'bash -lc {sh_quote(cmd)}']
  else:
    cmd = ['bash', '-c', cmd]
  return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)


def sh_quote(x):
  return "'" + x.replace("'", "'\\''") + "'"


def submit(s, jobname, body, opts, dry):
  logs = f'{s["repo"]}/experiments/{NAME}/logs'
  if dry:
    print(f'  [dry] {jobname}')
    return
  r = run(s, f'mkdir -p {logs} && sbatch --job-name={jobname} {opts} --time=7-00:00:00 --requeue '
             f'--output={logs}/{jobname}_%j.log', stdin=body)
  print(f'  {jobname}: {(r.stdout + r.stderr).strip()}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=sorted(STAGES))
  ap.add_argument('--site', choices=sorted(SITES))
  ap.add_argument('--seeds', default='1')
  ap.add_argument('--only', default='')
  ap.add_argument('--eval-step', type=int, default=None)
  ap.add_argument('--t-max', type=float, default=None)
  ap.add_argument('--sync-code', action='store_true')
  ap.add_argument('--pull', action='store_true')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  ssh = f'ssh -o BatchMode=yes -i {KEY}'

  if args.sync_code:   # the code only; data, outputs and checkpoints stay where they are
    inc = ' '.join(f"--include='/{p}'" if '*' in p else f"--include='/{p}/***'" for p in CODE)
    os.system(f"rsync -a --exclude='__pycache__/' {inc} --exclude='/*' -e '{ssh}' {REPO}/ {REMOTE}:{R_REPO}/")
    os.system(f"{ssh} {REMOTE} 'cd {R_REPO} && md5sum algo.py models/hyperbolic_dit.py samplers.py'")
    return
  if args.pull:        # a100 results (eval JSONs, hydra configs, logs; no checkpoints) -> local
    os.system(f"rsync -a --exclude='checkpoints' --exclude='wandb/' -e '{ssh}' "
              f"{REMOTE}:{R_REPO}/outputs/{NAME}/ {OUT}/")
    os.system(f"rsync -a -e '{ssh}' {REMOTE}:{R_REPO}/experiments/{NAME}/logs/ {EXP}/logs/a100/")
    return

  s = SITES[args.site]
  queued = set()   # a cell queued at either site counts
  for site in SITES.values():
    queued |= set(run(site, f'squeue -h -u {"ch2263" if site["remote"] else "sc3379"} -o %j').stdout.split())
  stage = args.stage or '1'
  cells = [(c, int(seed)) for c in STAGES[stage] for seed in args.seeds.split(',')]
  cells = [(c, seed) for c, seed in cells if args.only in tag_of(c, seed)]
  for c, seed in cells:
    tag = tag_of(c, seed)
    if args.eval_step:
      sub = f'eval_step{args.eval_step}' + ('' if args.t_max is None else f'_tmax{args.t_max:g}')
      jobname = f'{JOB}_{tag}_{sub}'
      if os.path.exists(f'{OUT}/{tag}/{sub}/samples_genppl.json') or jobname in queued:
        continue
      if c['mode'] != 'exp' and args.t_max is not None and args.t_max > c['ub']:
        continue
      submit(s, jobname, eval_body(s, c, seed, args.eval_step, args.t_max), s['eval_sbatch'], args.dry_run)
    else:
      jobname = f'{JOB}_{tag}'
      if os.path.exists(f'{OUT}/{tag}/eval/samples_genppl.json') or jobname in queued:
        print(f'  skip {tag}')
        continue
      submit(s, jobname, train_body(s, c, seed), s['sbatch'], args.dry_run)
  if args.dry_run and cells:
    c, seed = cells[0]
    print('\n--- example body ---\n' + train_body(s, c, seed))


if __name__ == '__main__':
  main()
