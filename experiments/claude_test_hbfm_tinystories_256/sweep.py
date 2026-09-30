#!/usr/bin/env python
"""claude_test_hbfm_tinystories_256 — HBFM (hyperbolic bridge DLM) on TinyStories seq 256:
time_conversion_mode x time_exp_rate x curvature x time_range_upper_bound (x seeds) for GenPPL / entropy.

Spec: experiments/claude_test_hbfm_tinystories_256/setup.md (forward_type naive, (H^3_K)^16 = embed_dim 48,
log-linear noise, LR 3e-4, 30k steps, global batch 512, bf16, EMA 0.9999, seeds 1-3). Design: EXPERIMENT.md.
Train: scripts/train/tinystories/hbfm.sh (checkpoints every 1k rolling + keep-{step}.ckpt every 5k).
Eval: scripts/sample/tinystories/hbfm.sh (ppl_eval = importance-weighted denoising-CE bound;
sample_eval = 64 samples, 180 steps, exact velocity, top_k_velocity 1, greedy last step,
gpt2-large GenPPL + entropy).

A cell is (mode, physical time_exp_rate, K, time_range_upper_bound, seed); the tag records all of
them. `rate` is a no-op for mode unif and `ub` for mode exp (both stay at the yaml defaults, 3 / 1.0).
Sampler horizon (physical): exp -> UNIT_T_MAX / |K|; trunc_exp / unif -> min(ub, UNIT_T_MAX / |K|)
(the SDE may not walk past the trained range; algo._validate_configuration enforces it at model
construction, so T_MAX is passed to training too).

Cells (`cells()`): STAGE1 (seed 1) + CONFIRM (hand-picked follow-ups / seeds 2-3 after the reads).

ORCHESTRATION ONLY — one SLURM job per cell (train then eval, `set -e`); idempotent + resumable
(skips cells with eval/samples_genppl.json or a queued job; resubmission resumes from
checkpoints/last.ckpt). `--eval-step N` instead submits 1-GPU evals of keep-N.ckpt for every cell that
has it (-> eval_stepN/); `--eval-step N --t-max T` evaluates at a different physical horizon
(-> eval_stepN_tmaxT/; exp cells only in practice, unif / trunc_exp cells with T > ub are skipped).

Usage:  python sweep.py [--dry-run] [--only SUBSTR] [--eval-step N [--t-max T]]    then  python report.py
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/claude_test_hbfm_tinystories_256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/claude_test_hbfm_tinystories_256'
JOB = 'hbfm_ts256'

SPEC = dict(mode='exp', rate='3', k='-0.5', ub='1.0')   # setup.md defaults
EMBED_DIM = 48        # setup.md: prod_factor_dim = 16 x 3 -> 16 factors of H^3 (the scripts default to 96 = 32 factors)
UNIT_T_MAX = 1.5      # exp-mode sampling horizon in unit time |K| t: ~2x the tau (0.8) where the context-free posterior on (H^3)^16 is one-hot; 180 steps -> dtau 0.0083 (EXPERIMENT.md)
DEVICES = 4
MAX_STEPS = 30000
CKPT_EVERY = 1000
KEEP_EVERY = 5000

# (mode, rate, K, ub, seed). Stage 1 = seed 1, one axis at a time around the spec point.
STAGE1 = (
  [('exp', r, '-0.5', '1.0', '1') for r in ['3', '5', '10', '20']]       # rate ladder (unit rate 6 / 10 / 20 / 40)
  + [('exp', '3', k, '1.0', '1') for k in ['-0.25', '-1.0']]             # curvature at the spec rate (unit rate 12 / 3)
  + [('trunc_exp', '3', '-0.5', ub, '1') for ub in ['0.5']]              # exp(3) truncated to [0, ub]: drops 22% of the draws / 19% of the CE mass; horizon = ub
  + [('unif', '3', '-0.5', ub, '1') for ub in ['0.5', '1.0', '2.0']]     # uniform heat time on [0, ub] (rate unused); unit 0.25 / 0.5 / 1.0; horizon = ub
)
# Follow-ups added after the reads (seeds 2-3 of the best cells, extra points on a winning axis).
# Candidates: ('trunc_exp', '3', '-0.5', '1.0', '1') -- a 5%-of-draws truncation of the spec cell, only worth
# training if trunc_exp ub 0.5 wins; the horizon question it would answer is covered by `--eval-step N --t-max 1.0`
# on the spec cell. ('exp', '40', '-0.5', '1.0', '1') if rate 20 wins; ('unif', ..., '0.25') if unif 0.5 wins.
CONFIRM = [
  # 2026-09-18 01:15, after the 5k rate ladder (rate 3 / 5 / 10 / 20 -> GenPPL 25.1 / 29.4 / 31.7 / 38.3, monotone, the
  # spec rate 3 = unit 6 best): one point BELOW the spec rate, unit 3 (same unit rate as the K -1 cell -> equivalence test),
  # and seeds 2-3 of the spec cell (needed for the final mean whatever wins).
  ('exp', '1.5', '-0.5', '1.0', '1'),
  ('exp', '3', '-0.5', '1.0', '2'),
  ('exp', '3', '-0.5', '1.0', '3'),
  # 2026-09-18 17:30: trunc_exp ub 0.5 finished 30k at GenPPL 12.59 @ 3.92 vs the spec's 12.90 at the same horizon
  # (-2..4% at every checkpoint from 15k on) -> the mode-axis winner on one seed; seeds 2-3 to settle it vs the spec.
  ('trunc_exp', '3', '-0.5', '0.5', '2'),
  ('trunc_exp', '3', '-0.5', '0.5', '3'),
  # 2026-09-20 11:25, stage 3. The 15-cell sweep ended with trunc_exp ub 0.5 (12.64 +- 0.43, 3 seeds) ahead of unif
  # ub 0.5 (13.18, ONE seed) by 4% -- inside the seed spread, so rank 1 vs 2 is undecided. The other open question,
  # whether the bounded ladders continue below ub 0.5, was answered eval-only instead of by training: sampling the
  # trunc_exp winner at t_max 0.35 / 0.25 (< its ub) gives 12.77 / 13.15 vs 12.57 at 0.5 on the same node, worse in
  # 3/3 seeds, so ub 0.5 is a real optimum and ub 0.25 cells are not worth GPU-days. These two seeds are the only
  # remaining cells that can change the answer.
  ('unif', '3', '-0.5', '0.5', '2'),
  ('unif', '3', '-0.5', '0.5', '3'),
]

# Stage 4 (2026-09-20): the user added `forward_type: {naive, horosphere}` to setup.md and asked to tune for the best
# config, so the grid gains a 6th field. Cells are (mode, rate, k, ub, seed, forward_type); a missing 6th field means
# 'naive', which is what every cell above is. The repo has NO head-to-head horosphere-vs-naive run: the paused round 1
# was horosphere but on 32 factors, at 5k-11k of 30k steps, sampled at physical t_max 2.0 and with no
# time_conversion_mode knob, so none of its numbers transfer. A horosphere cell costs ~2.5x a naive one
# (~16.5 h on the Ada, ~27.5 h on the A6000, ~38.5 h on the A5000 for 30k), hence a 3-cell probe first, seed 1 only:
#   1. the spec point, the cleanest head-to-head against naive 15.58 +- 0.67 (own horizon) / 13.25 +- 0.56 (t_max 0.5);
#   2. the naive winner's config (trunc_exp ub 0.5, naive 12.64 +- 0.43) -- can horosphere beat the current answer;
#   3. rate 10, where naive falls off hard (17.82): the horosphere readout IS the Bayes posterior of the bridge
#      direction, so if carrying the geometry explicitly makes the model less sensitive to the proposal, the rate
#      optimum should flatten or move up (round 1 at 32 factors preferred unit 5-20). Either outcome is informative.
# Everything beyond these three is decided from their 5k/10k reads, as in stages 1-2.
STAGE4 = [
  ('exp', '3', '-0.5', '1.0', '1', 'horosphere'),
  ('trunc_exp', '3', '-0.5', '0.5', '1', 'horosphere'),
  ('exp', '10', '-0.5', '1.0', '1', 'horosphere'),
]


FWD_TAG = {'naive': 'naive', 'horosphere': 'horo'}   # same convention as claude_test_hbfm_sudoku/sweep_mode.py


def tag_of(mode, rate, k, ub, seed, fwd='naive'):
  # The naive tag is left exactly as it was: the 15 finished cells keep their directories, their eval JSONs and
  # their rows in RESULTS.md. Only a non-naive forward type adds a component, so it cannot collide with them.
  f = '' if fwd == 'naive' else f'_fwd-{FWD_TAG[fwd]}'
  return f'mode-{mode}_rate-{rate}_k{k}_ub-{ub}{f}_seed-{seed}'


def cells():
  # Cells are (mode, rate, k, ub, seed) or (mode, rate, k, ub, seed, forward_type); normalise to 6-tuples so every
  # helper can take them positionally as *c.
  cs = [tuple(c) for c in STAGE1] + [tuple(c) for c in CONFIRM] + [tuple(c) for c in STAGE4]
  return list(dict.fromkeys(c if len(c) == 6 else c + ('naive',) for c in cs))


def t_max(mode, k, ub):
  horizon = UNIT_T_MAX / abs(float(k))
  return f'{horizon if mode == "exp" else min(float(ub), horizon):g}'


def active_jobnames():
  try:
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                         capture_output=True, text=True).stdout
    return set(out.split())
  except Exception:
    return set()


def preamble():
  return textwrap.dedent(f'''\
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export SLURM_JOB_NAME=bash
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export WANDB_MODE=offline
    export TMPDIR=/home/sc3379/tmp/hbfmts_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
    trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={ENVBIN}:$PATH
    cd {REPO}
    md5sum scripts/train/tinystories/hbfm.sh scripts/sample/tinystories/hbfm.sh algo.py samplers.py configs/algo/hbfm.yaml
    ''')


def knobs(mode, rate, k, ub, t_max_override=None, fwd='naive'):
  # T_MAX goes to training too: algo._validate_configuration checks sampler.t_max against the trained range.
  # FORWARD_TYPE must be passed to the SAMPLER as well: samplers.py calls model.forward, so a horosphere
  # checkpoint evaluated without it is silently scored with the naive readout instead of erroring.
  tm = t_max(mode, k, ub) if t_max_override is None else f'{t_max_override:g}'
  return (f"FORWARD_TYPE={fwd} EMBED_DIM={EMBED_DIM} FACTOR_DIM=3 TIME_CONVERSION_MODE={mode} PROPOSAL_RATE={rate} "
          f"GAUSS_CURV={k} TIME_RANGE_UPPER_BOUND={ub} T_MAX={tm} READOUT_PRECISION=float32 GLOBAL_BATCH=512 SEQ_LEN=256")


def eval_cmd(mode, rate, k, ub, ckpt, out_dir, t_max_override=None, fwd='naive'):
  return (f'{knobs(mode, rate, k, ub, t_max_override, fwd)} TOPK_VELOCITY=1 STEPS=180 \\\n'
          f'    CKPT_PATH={ckpt} OUTPUT_DIR={out_dir} DEVICES=1 \\\n'
          f'    bash scripts/sample/tinystories/hbfm.sh')


def job_body(mode, rate, k, ub, seed, fwd='naive'):
  tag = tag_of(mode, rate, k, ub, seed, fwd)
  tdir = f'{OUT}/{tag}'
  return preamble() + textwrap.dedent(f'''\
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    PER_GPU_BS=$([ "$GPU_MEM" -gt 40000 ] && echo 16 || echo 8)   # 48 GB Ada / A6000 vs 24 GB A5000; global batch fixed at 512
    echo "[$(date)] TRAIN {tag} on $(hostname) (GPU ${{GPU_MEM}} MiB -> PER_GPU_BS ${{PER_GPU_BS}})"
    LR=3e-4 SEED={seed} {knobs(mode, rate, k, ub, None, fwd)} \\
        OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP=claude_test_hbfm_tinystories_256 \\
        DEVICES={DEVICES} PER_GPU_BS=$PER_GPU_BS MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} KEEP_EVERY={KEEP_EVERY} \\
        bash scripts/train/tinystories/hbfm.sh
    echo "[$(date)] EVAL {tag}"
    {eval_cmd(mode, rate, k, ub, f'{tdir}/checkpoints/last.ckpt', f'{tdir}/eval', None, fwd)}
    echo "[$(date)] DONE {tag}"
    ''')


def eval_dir(step, t_max_override=None):
  return f'eval_step{step}' + ('' if t_max_override is None else f'_tmax{t_max_override:g}')


def eval_body(mode, rate, k, ub, seed, fwd, step, t_max_override=None):
  tag = tag_of(mode, rate, k, ub, seed, fwd)
  tdir = f'{OUT}/{tag}'
  return preamble() + textwrap.dedent(f'''\
    echo "[$(date)] EVAL {tag} @ step {step} on $(hostname)"
    {eval_cmd(mode, rate, k, ub, f'{tdir}/checkpoints/keep-{step}.ckpt', f'{tdir}/{eval_dir(step, t_max_override)}', t_max_override, fwd)}
    echo "[$(date)] DONE {tag} @ step {step}"
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument('--only', default=None, help='submit only cells whose tag contains this substring')
  ap.add_argument('--eval-step', type=int, default=None, help='submit 1-GPU evals of keep-<N>.ckpt instead of training')
  ap.add_argument('--t-max', type=float, default=None, help='with --eval-step: physical sampler horizon override (-> eval_stepN_tmaxT/)')
  args = ap.parse_args()
  if args.t_max is not None and args.eval_step is None:
    ap.error('--t-max requires --eval-step')
  os.makedirs(LOGS, exist_ok=True)
  grid = [c for c in cells() if not args.only or args.only in tag_of(*c)]
  if args.t_max is not None:   # the SDE may not walk past a bounded training range
    grid = [c for c in grid if c[0] == 'exp' or args.t_max <= float(c[3])]
  mode = f'eval @ step {args.eval_step}' + (f' t_max {args.t_max:g}' if args.t_max is not None else '') if args.eval_step else 'train + eval'
  print(f'claude_test_hbfm_tinystories_256 ({mode}): {len(grid)} cells')
  if args.dry_run:
    for c in grid:
      print('  ' + tag_of(*c))
    if grid:
      print('\n--- example body ---\n' + (eval_body(*grid[0], args.eval_step, args.t_max) if args.eval_step else job_body(*grid[0])))
    return
  active = active_jobnames()
  n_sub = n_skip = 0
  for c in grid:
    tag = tag_of(*c)
    tdir = f'{OUT}/{tag}'
    if args.eval_step:
      ed = eval_dir(args.eval_step, args.t_max)
      jobname = f'{JOB}_{tag}_{ed[len("eval_step"):]}'
      done = os.path.exists(f'{tdir}/{ed}/samples_genppl.json')
      ready = os.path.exists(f'{tdir}/checkpoints/keep-{args.eval_step}.ckpt')
      if done or not ready or jobname in active:
        n_skip += 1
        continue
      # EVAL_NODELIST pins reads to one node: re-reads of the same checkpoint are bit-identical on the same
      # node but differ by up to 0.46 GenPPL across GPU types, so a horizon curve must be read on one node.
      # It needs EVAL_PARTITION set to a partition that contains the node (slurm rejects a nodelist that is
      # not in every requested partition).
      pin = {'nodelist': os.environ['EVAL_NODELIST']} if os.environ.get('EVAL_NODELIST') else {}
      slurm = Slurm(job_name=jobname, partition=os.environ.get('EVAL_PARTITION', 'thickstun,desa'),
                    gres='gpu:1', ntasks=1,
                    cpus_per_task=8, mem='48G', time='06:00:00', exclude='desa-compute-01', **pin,
                    output=f'{LOGS}/{tag}_{ed[len("eval_step"):]}_%j.log')
      jid = slurm.sbatch(eval_body(*c, args.eval_step, args.t_max))
    else:
      jobname = f'{JOB}_{tag}'
      if os.path.exists(f'{tdir}/eval/samples_genppl.json') or jobname in active:
        n_skip += 1
        continue
      slurm = Slurm(job_name=jobname, partition='thickstun,desa', exclude='desa-compute-01',
                    gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=16, mem='128G', time='7-00:00:00',
                    output=f'{LOGS}/{tag}_%j.log')
      jid = slurm.sbatch(job_body(*c))
    print(f'  submitted {tag}: job {jid}')
    n_sub += 1
  print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
  main()
