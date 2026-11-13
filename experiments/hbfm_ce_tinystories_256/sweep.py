#!/usr/bin/env python
"""hbfm_ce_tinystories_256 -- HBFM on TinyStories (seq 256) trained on the denoising CE (LOSS_TYPE=ce),
ranked by the exact validation NELBO, scored as in hbfm_elbo_tinystories_256 (LOSS_TYPE=elbo, words summed
exactly, SEED=1, EVAL_BS=16). Target: NELBO < 1.70. Spec: setup.md. Design: EXPERIMENT.md.

Shared by all cells: small DiT 768/12/12, naive readout in float32, (H^3)^16 = embed_dim 48, log-linear
noise, 30k steps, global batch 512, bf16, EMA 0.9999.

One job per cell, which early-stops itself: train to 6k, score the exact NELBO on STAGE_VAL of the validation
set (eval/nelbo_6000.json), stop if gate.py says so (bars.json); the same at 10k; then 30k and the full evals.
A requeued job resumes from last.ckpt and skips the stages already scored.

Sites (host reached over ssh unless it is this account; `--sync-code` copies the code):
  a100   ch2263@unicorn, nlplarge (A100 80 GB): 1 GPU x micro-batch 64. Checkpoints on /scratch (experiment.py).
         W&B offline, replayed by `--wandb-sync` from sc3379: the API key stays on the sc3379 account.
  a6000  sc3379@unicorn, desa without the 2080 Ti node (A6000 / A5000): 4 GPUs x micro-batch 16. W&B live.
  desa2  the same with 2 GPUs x micro-batch 32: the 2 GPUs per node that 4-GPU cells leave free.

ORCHESTRATION ONLY. Idempotent: a cell with eval/nelbo.json or a queued job of the same name is skipped.

Usage:  python sweep.py --site {a100,a6000} --sync-code
        python sweep.py --site {a100,a6000} --cells c1,c2 [--seeds 1] [--smoke] [--dry-run]
        python sweep.py --site a100 --pull
        python sweep.py --wandb-sync
"""
import argparse
import getpass
import glob
import json
import math
import os
import subprocess
import textwrap

from gate import NAME, run_id, upload

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
JOB = 'hbfm_ce_ts'
KEY = '/home/sc3379/.ssh/unicorn_internal'
SFM_HOME = '/share/desa/nfs02/sc3379/sfm_home'   # sc3379 env, data and HF cache (/share/thickstun is not readable)
CODE = ['*.py', 'configs', 'models', 'numeric', 'callbacks', 'scripts', 'data', 'tests', 'experiments/hbfm_rlf_ada_curv',
        'experiments/hbfm_rlf_ada_curv_tinystories_256', f'experiments/{NAME}']

SITES = {
  'a100': dict(host='ch2263@unicorn-login-02.coecis.cornell.edu', repo='/home/ch2263/syc_workspace/s-flm-dev/s-flm',
               out='/home/ch2263/syc_workspace/s-flm-dev/s-flm/outputs',
               envbin='/home/ch2263/miniconda3/envs/sfm/bin', env='export WANDB_MODE=offline',
               cache='/home/ch2263/syc_workspace/s-flm_data_home/data_cache', offline='true',
               devices=1, per_gpu_bs=64, workers=4,
               sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G'),
  'a6000': dict(host='sc3379@unicorn-login-05.coecis.cornell.edu',
                repo='/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm',
                out=f'{SFM_HOME}/outputs',   # the repo's outputs/ links into /share/thickstun
                envbin=f'{SFM_HOME}/envs/sfm/bin',
                env=f'export HF_HOME={SFM_HOME}/hf_cache XDG_CACHE_HOME={SFM_HOME}/cache '
                    f'MPLCONFIGDIR={SFM_HOME}/cache/matplotlib WANDB_CACHE_DIR={SFM_HOME}/cache/wandb '
                    f'WANDB_CONFIG_DIR={SFM_HOME}/cache/wandb\n'
                    'set -a; . ./.wandb.env; set +a',
                cache=f'{SFM_HOME}/data_cache', offline='false',
                devices=4, per_gpu_bs=16, workers=4,
                sbatch='--partition=desa --exclude=desa-compute-01 --gres=gpu:4 --cpus-per-task=16 --mem=128G'),
}
SITES['desa2'] = dict(SITES['a6000'], devices=2, per_gpu_bs=32,
                      sbatch='--partition=desa --exclude=desa-compute-01 --gres=gpu:2 --cpus-per-task=8 --mem=64G')

# Setup.md's grid: its literal training recipe, the exp proposal, LR 3e-4.
GRID = dict(rate=3, k=-0.5, mode='exp', ub=None, eps=1e-3, curv=1, lr=3e-4, sched='constant', hold=3500, dropout=0.1,
            embed='gpt2-xl')
# The tuning base: the best recipe of hbfm_elbo_tinystories_256 (1.685-1.683), which every cell below varies by one knob.
BASE = dict(GRID, mode='trunc_exp', ub=0.6, lr=2e-3, sched='wsd', hold=11500, dropout=0)
CELLS = {
  # the grid of setup.md: ada on all four, fix on the anchor
  'grid_r3_K0.5_fix': GRID,
  'grid_r3_K0.5_ada': dict(GRID, curv=4),
  'grid_r1.5_K0.5_ada': dict(GRID, rate=1.5, curv=4),
  'grid_r5_K0.5_ada': dict(GRID, rate=5, curv=4),
  'grid_r6_K1_ada': dict(GRID, rate=6, k=-1.0, curv=4),
  'base': BASE,
  # the horizon: finite (trunc_exp, ub) against the exp proposal, whose horizon ln(1/eps)/rate sampling_eps sets.
  # exp_eps1e-3 always runs to 30k: the finite-vs-infinite question needs its final (finish=True skips the gate)
  'ub0.4': dict(BASE, ub=0.4),
  'ub0.8': dict(BASE, ub=0.8),
  'ub1': dict(BASE, ub=1.0),
  'ub1.5': dict(BASE, ub=1.5),
  'exp_eps1e-3': dict(BASE, mode='exp', ub=None, finish=True),   # T 2.30
  'exp_eps1e-2': dict(BASE, mode='exp', ub=None, eps=1e-2),      # T 1.54
  'exp_eps1e-5': dict(BASE, mode='exp', ub=None, eps=1e-5),      # T 3.84
  # the proposal inside the horizon: under CE it only changes the variance of the same integral
  'rate1.5': dict(BASE, rate=1.5),
  'rate6': dict(BASE, rate=6),
  'unif': dict(BASE, mode='unif'),
  # curvature: K at the same unit proposal (rate / |K|, |K| ub), and the learned clock
  'K0.25': dict(BASE, rate=1.5, k=-0.25, ub=1.2),
  'K1': dict(BASE, rate=6, k=-1.0, ub=0.3),
  'ada': dict(BASE, curv=4),
  # optimizer, regularizer, word embedding
  'lr1e-3': dict(BASE, lr=1e-3),
  'lr3e-3': dict(BASE, lr=3e-3),
  'do0.1': dict(BASE, dropout=0.1),
  'bert': dict(BASE, embed='google-bert/bert-large-uncased'),
  # wave 2 (6k: ub 0.4 2.150, 0.6 2.185, 0.8 2.185). The shorter horizons; and the CE without its importance weight, averaged
  # under the proposal (CE_TIME_WEIGHTED=false): the path-KL rate falls 5.6x over [0, 0.5], the weighted CE weighs t evenly
  'ub0.3': dict(BASE, ub=0.3),
  'ub0.2': dict(BASE, ub=0.2),
  'now': dict(BASE, ce_time_weighted=False),
  'now_r6': dict(BASE, rate=6, ce_time_weighted=False),
  # wave 3 (10k: ub 0.4 2.041, the best). Weighted-CE cells whose largest importance weight exceeds ~5 spiked or diverged at
  # LR 2e-3 (exp, ub 1, rate 6); ub 0.4's is 0.77, so it can take the levers that won at 30k under the ELBO: a higher peak,
  # a longer hold
  'ub0.4_lr3e-3': dict(BASE, ub=0.4, lr=3e-3),
  'ub0.4_wsd20k': dict(BASE, ub=0.4, hold=17500),
  # wave 4: BERT-large scored 2.162 at 6k vs base's 2.185 (-1.1%; under the ELBO the two tied), on the best horizon
  'ub0.4_bert': dict(BASE, ub=0.4, embed='google-bert/bert-large-uncased'),
  # wave 5 (the user, 2026-11-07 18:10: is "short horizon better" robust? tune the infinite horizon, try longer ones). At 30k
  # exp T 2.30 scored 1.936 vs ub 0.4's 1.845, but exp trained with importance weights up to 333 at peak LR 2e-3 and spiked.
  # Its suspects: the LR, and the weights (CE_TIME_WEIGHTED=false removes them); with the matched finite cells
  'exp_lr1e-3': dict(BASE, mode='exp', ub=None, lr=1e-3),
  'exp_lr6e-4': dict(BASE, mode='exp', ub=None, lr=6e-4),
  'exp_lr3e-4': dict(BASE, mode='exp', ub=None, lr=3e-4),
  'exp_now': dict(BASE, mode='exp', ub=None, ce_time_weighted=False),
  'exp_eps1e-5_now': dict(BASE, mode='exp', ub=None, eps=1e-5, ce_time_weighted=False),   # T 3.84
  'exp_eps1e-7_now': dict(BASE, mode='exp', ub=None, eps=1e-7, ce_time_weighted=False),   # T 5.37
  'ub0.4_now': dict(BASE, ub=0.4, ce_time_weighted=False),
  'ub0.4_lr1e-3': dict(BASE, ub=0.4, lr=1e-3),
  # wave 6: ub0.4_bert scored 2.144 at 6k (the best 6k) and diverged to 7.05 by 10k at peak LR 2e-3; peak LR 1e-3 tied 2e-3
  # at 30k (1.856 vs 1.858 at ub 0.6), so the same cell at the stabler LR
  'ub0.4_bert_lr1e-3': dict(BASE, ub=0.4, embed='google-bert/bert-large-uncased', lr=1e-3),
}
STAGES = [6000, 10000]   # 1/5, 1/3
MAX_STEPS = 30000
STAGE_VAL = 0.1   # fraction of the validation set behind a stage's NELBO: the same sequences and t draws for every cell


def knobs(c):
  if c['mode'] == 'exp':   # the horizon T = ln(1/eps)/rate
    time = f'TIME_CONVERSION_MODE=exp T_MAX={math.log(1 / c["eps"]) / c["rate"]:.4f}'
  else:                    # the proposal and the horizon end at ub
    time = f'TIME_CONVERSION_MODE={c["mode"]} TIME_RANGE_UPPER_BOUND={c["ub"]:g} T_MAX={c["ub"]:g}'
  return (f'FORWARD_TYPE=naive READOUT_PRECISION=float32 EMBED_DIM=48 FACTOR_DIM=3 GAUSS_CURV={c["k"]:g} '
          f'CURV_BOUND={c["curv"]} PROPOSAL_RATE={c["rate"]:g} SAMPLING_EPS={c["eps"]:g} {time} GLOBAL_BATCH=512 SEQ_LEN=256')


def header(s):
  return textwrap.dedent('''\
    #!/bin/bash
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TMPDIR=/dev/shm/{job}_$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER
    mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={envbin}:$PATH
    cd {repo}
    {env}
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    ''').format(job=JOB, envbin=s['envbin'], repo=s['repo'], env=s['env'])


def body(s, name, c, cell, seed, project, stages, max_steps, ckpt_every, keep_every, stage_val):
  tag = f'{cell}_seed-{seed}'
  tdir = f'{s["out"]}/{project}/{tag}'
  ev = f'CACHE_DIR={s["cache"]} DEVICES=1 CKPT_PATH={tdir}/checkpoints/last.ckpt'

  def train(steps):
    return (f'LR={c["lr"]:g} LR_SCHED={c["sched"]} LR_DECAY_STEPS={max_steps} LR_HOLD_STEPS={c["hold"]} '
            f'DROPOUT={c["dropout"]:g} SEED={seed} CE_TIME_WEIGHTED={str(c.get("ce_time_weighted", True)).lower()} '
            f'{knobs(c)} LOSS_TYPE=ce PRETRAINED_EMBED={c["embed"]} '
            f'CACHE_DIR={s["cache"]} OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={name} WANDB_PROJECT={project} '
            f'WANDB_ID={run_id(tag)} WANDB_OFFLINE={s["offline"]} DEVICES={s["devices"]} PER_GPU_BS={s["per_gpu_bs"]} '
            f'NUM_WORKERS={s["workers"]} MAX_STEPS={steps} CKPT_EVERY={ckpt_every} KEEP_EVERY={keep_every} '
            'bash scripts/train/tinystories/hbfm.sh')

  text = header(s) + f'export WANDB_PROJECT={project}\npython experiment.py {tdir} {project} {tag}\n'
  gate = f'experiments/{NAME}/gate.py {tdir}'
  for steps in stages:
    text += textwrap.dedent(f'''\
      if [ ! -f {tdir}/eval/nelbo_{steps}.json ]; then
        echo "[$(date)] TRAIN {tag} to step {steps}"
        {train(steps)}
        {knobs(c)} LOSS_TYPE=elbo SEED=1 EVAL_BS=16 RUN_SAMPLE_EVAL=false LIMIT_VAL_BATCHES={stage_val} PPL_JSON={tdir}/eval/nelbo_{steps}.json {ev} OUTPUT_DIR={tdir}/eval/step{steps} bash scripts/sample/tinystories/hbfm.sh
      fi
      python {gate} {steps} {"|| true" if c.get("finish") else "|| exit 0"}
      ''')
  limit = f'LIMIT_VAL_BATCHES={stage_val} ' if stage_val < 0.1 else ''   # the smoke test
  text += textwrap.dedent(f'''\
    echo "[$(date)] TRAIN {tag} to step {max_steps}"
    {train(max_steps)}
    echo "[$(date)] EVAL {tag}"
    {limit}{knobs(c)} TOPK_VELOCITY=1 STEPS=180 {ev} OUTPUT_DIR={tdir}/eval bash scripts/sample/tinystories/hbfm.sh
    echo "[$(date)] NELBO {tag}"
    {limit}{knobs(c)} LOSS_TYPE=elbo SEED=1 EVAL_BS=16 RUN_SAMPLE_EVAL=false PPL_JSON={tdir}/eval/nelbo.json {ev} OUTPUT_DIR={tdir}/eval bash scripts/sample/tinystories/hbfm.sh
    python {gate} final
    echo "[$(date)] DONE {tag}"
    ''')
  return text


def local(s):
  return s['host'].split('@')[0] == getpass.getuser()


def run(s, cmd, stdin=None):
  """A shell command at the site's login node (over ssh unless it is this account; a login shell for slurm)."""
  cmd = ['bash', '-lc', cmd] if local(s) else [
    'ssh', '-o', 'BatchMode=yes', '-i', KEY, s['host'], "bash -lc '" + cmd.replace("'", "'\\''") + "'"]
  return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)


def offline_rows(path):
  """The config and the history rows of an offline wandb run file (<cell>/wandb/offline-run-*/*.wandb)."""
  from wandb.proto import wandb_internal_pb2
  from wandb.sdk.internal import datastore
  config, rows = {}, []
  if os.path.getsize(path) == 0:   # a restart preempted before its first write
    return config, rows
  ds = datastore.DataStore()
  ds.open_for_scan(path)
  while True:
    try:
      data = ds.scan_data()
    except AssertionError:   # the tail of a file still being written
      break
    if data is None:
      break
    r = wandb_internal_pb2.Record()
    r.ParseFromString(data)
    if r.HasField('config'):
      config.update((i.key, json.loads(i.value_json)) for i in r.config.update if not i.key.startswith('_'))
    elif r.HasField('history'):
      row = ((i.key or '/'.join(i.nested_key), json.loads(i.value_json)) for i in r.history.item)
      rows.append({k: v for k, v in row if not k.startswith('_')})
  return config, rows


def wandb_sync(out, project):
  """The offline cells under out/ (the a100 site's, pulled) into their W&B runs: the history rows of all their
  offline runs, in launch order, plus their scores. Only rows the W&B run does not have yet are uploaded, so a cell
  still training can be synced again. (`wandb sync --append` of a restart overwrites rows: both count _step from 0.)"""
  import wandb
  os.environ.update(line.strip().split('=', 1) for line in open(f'{REPO}/.wandb.env') if '=' in line)
  os.environ.setdefault('WANDB_CONFIG_DIR', f'{SFM_HOME}/cache/wandb')
  os.environ.setdefault('WANDB_CACHE_DIR', f'{SFM_HOME}/cache/wandb')
  for d in sorted(glob.glob(f'{out}/*/wandb')):
    tdir = os.path.dirname(d)
    tag = os.path.basename(tdir)
    files = sorted(glob.glob(f'{d}/offline-run-*/*.wandb'))
    if not files:   # a live cell logs for itself
      continue
    config, rows = {}, []
    for f in files:
      c, r = offline_rows(f)
      config, rows = config or c, rows + r
    run_ = wandb.init(project=project, id=run_id(tag), name=tag, group=NAME, resume='allow', config=config,
                      dir=f'{SFM_HOME}/cache/wandb', settings=wandb.Settings(silent=True, console='off'))
    run_.define_metric('trainer/global_step')
    run_.define_metric('*', step_metric='trainer/global_step', step_sync=True)
    done = run_.summary.get('synced_rows', 0)
    for row in rows[done:]:
      run_.log(row)
    run_.summary['synced_rows'] = len(rows)
    run_.finish()
    upload(tdir, project=project)
    print(f'  {tag}: {len(rows) - done} new rows ({len(rows)} in all)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--site', choices=sorted(SITES))
  ap.add_argument('--cells', default=None, help='comma list')
  ap.add_argument('--seeds', default='1')
  ap.add_argument('--sync-code', action='store_true')
  ap.add_argument('--pull', action='store_true')
  ap.add_argument('--wandb-sync', action='store_true')
  ap.add_argument('--smoke', action='store_true', help='60 steps through every stage, into project <NAME>_smoke')
  ap.add_argument('--finish', action='store_true', help='skip the gates: train to MAX_STEPS whatever the stage scores')
  ap.add_argument('--sbatch', default='', help='extra sbatch flags, e.g. a --nodelist for the smoke test')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  project = f'{NAME}_smoke' if args.smoke else NAME
  here = f'{SITES["a6000"]["out"]}/{project}'
  ssh = f'ssh -o BatchMode=yes -i {KEY}'

  if args.wandb_sync:
    wandb_sync(here, project)
    return
  s = SITES[args.site]
  remote = f"{s['host']}:{s['repo']}"
  if args.sync_code and local(s):   # this account runs from this checkout
    return
  if args.sync_code:   # the code only; .wandb.env, data, outputs, logs and checkpoints stay where they are
    inc = ' '.join(f"--include='/{p}'" if '*' in p else f"--include='/{p}/***'" for p in CODE)
    os.system(f"rsync -a --exclude='__pycache__/' --exclude='logs/' --include='/experiments/' {inc} --exclude='/*' "
              f"--exclude='/experiments/*' -e '{ssh}' {REPO}/ {remote}/")
    return
  if args.pull:        # results and offline W&B runs (no checkpoints) -> here
    os.system(f"rsync -a --exclude='checkpoints' -e '{ssh}' {s['host']}:{s['out']}/{project}/ {here}/")
    os.system(f"rsync -a -e '{ssh}' {remote}/experiments/{NAME}/logs/ {REPO}/experiments/{NAME}/logs/{args.site}/")
    return

  queued = set()   # a cell queued at either site counts
  if not args.dry_run:
    for site in SITES.values():
      queued |= set(run(site, f'squeue -h -u {site["host"].split("@")[0]} -o %j').stdout.split())
  logs = f'{s["repo"]}/experiments/{NAME}/logs'
  plan = dict(stages=[20, 40], max_steps=60, ckpt_every=20, keep_every=0, stage_val=0.002) if args.smoke else dict(
    stages=STAGES, max_steps=MAX_STEPS, ckpt_every=1000, keep_every=2000, stage_val=STAGE_VAL)
  for cell in args.cells.split(','):
    for seed in args.seeds.split(','):
      tag = f'{cell}_seed-{seed}'
      jobname = f'{JOB}{"_smoke" * args.smoke}_{tag}'
      if os.path.exists(f'{here}/{tag}/eval/nelbo.json') or jobname in queued:
        print(f'  skip {tag}')
        continue
      text = body(s, NAME, dict(CELLS[cell], finish=CELLS[cell].get('finish') or args.finish), cell, seed, project, **plan)
      if args.dry_run:
        print(f'--- {jobname} ({s["sbatch"]})\n{text}')
        continue
      r = run(s, f'mkdir -p {logs} && sbatch --job-name={jobname} {s["sbatch"]} {args.sbatch} --time=7-00:00:00 --requeue '
                 f'--output={logs}/{jobname}_%j.log', stdin=text)
      print(f'  {jobname}: {(r.stdout + r.stderr).strip()}')


if __name__ == '__main__':
  main()
