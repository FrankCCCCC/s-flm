#!/usr/bin/env python
"""hbfm_rlf_ada_curv_tinystories_256 -- HBFM on the Gen. PPL / entropy frontier of
experiments/naive_ar_tinystories_s256 (FRONTIER_RESULTS.md): NFE {1, 4, ..., 256}, 512 samples
per cell, training seeds {1, 2, 3}, exact velocity with the greedy last step.

Models (both trunc_exp rate 3, K = -0.5, naive readout, LR 3e-4):
  ada  adaptive curvature (bound 4), ub 1 -- this experiment's best cell
  fix  fixed curvature, ub 0.5 -- the best cell of claude_test_hbfm_tinystories_256 (its
       checkpoints are copied to the A100 site with --copy-fixed)
Velocity arms (FRONTIER_RESULTS.md 7.1 / 10.1):
  k1     top_k_velocity = 1, the spec sampler: the temperature cancels, one marker per NFE
  exact  top_k_velocity = -1, the full-vocabulary velocity: T 0.50 .. 1.20 traces a curve
Truncation: the sampler horizon t_max (<= ub, the HBFM validator) is picked per model on the
k1 ladder of seed 1 (--pick): ada t_max in {0.25, 0.5, 0.75, 1}, fix in {0.25, 0.35, 0.5}.

ORCHESTRATION ONLY: one A100 job per (model, seed, t_max, arm) walks the NFE (x T) cells
through scripts/sample/tinystories/hbfm.sh with RUN_PPL_EVAL=false; a cell with
samples_genppl.json is skipped, so a resubmitted job runs only what is missing. Results land in
outputs/<NAME>/frontier/<model>_sd-<seed>/tmax-<t_max>_<arm>/nfe-<NFE>_t-<T>/ (`sweep.py --pull`).

Usage:  python frontier_sweep.py --copy-fixed
        python frontier_sweep.py --pick [--dry-run]
        python frontier_sweep.py --tmax ada=0.75,fix=0.5 --arm k1 --seeds 1,2,3 [--dry-run]
        python frontier_sweep.py --tmax ada=0.75,fix=0.5 --arm exact --seeds 1,2,3 [--dry-run]
"""
import argparse
import os
import textwrap

from sweep import KEY, NAME, REMOTE, SITES, cell, knobs, preamble, run, submit

FIX_SRC = '/share/desa/nfs02/sc3379/workspace/research/s-flm/outputs/claude_test_hbfm_tinystories_256'
FIX_TAG = 'mode-trunc_exp_rate-3_k-0.5_ub-0.5_seed-{seed}'
MODELS = {'ada': cell(mode='trunc_exp', ub=1.0), 'fix': cell(cb=1.0, mode='trunc_exp', ub=0.5)}
PICK = {'ada': [0.25, 0.5, 0.75, 1.0], 'fix': [0.25, 0.35, 0.5]}
NFES = [1, 4, 8, 16, 32, 64, 128, 256]
TEMPS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
         '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20']
NUM_SAMPLES, EVAL_BS = 512, 16
S = SITES['a100']


def ckpt(model, seed):
  if model == 'fix':
    return f'{S["repo"]}/outputs/claude_test_hbfm_tinystories_256/{FIX_TAG.format(seed=seed)}/checkpoints/last.ckpt'
  return f'{S["repo"]}/outputs/{NAME}/cb-4_mode-texp_rate-3_ub-1_tm-1_lr-3e-4_seed-{seed}/checkpoints/last.ckpt'


def body(model, seed, tm, arm):
  out = f'{S["repo"]}/outputs/{NAME}/frontier/{model}_sd-{seed}/tmax-{tm:g}_{arm}'
  temps = TEMPS if arm == 'exact' else ['1.00']
  # last.ckpt is a mid-training checkpoint until the training job's own eval exists.
  done = os.path.dirname(os.path.dirname(ckpt(model, seed))) + '/eval/samples_genppl.json'
  guard = f'[ -f {done} ] || {{ echo "training of {model} seed {seed} not finished"; exit 1; }}\n'
  return preamble(S) + (guard if model == 'ada' else '') + textwrap.dedent(f'''\
    for nfe in {' '.join(map(str, NFES))}; do
      for t in {' '.join(temps)}; do
        d={out}/nfe-${{nfe}}_t-${{t}}
        [ -f $d/samples_genppl.json ] && continue
        echo "[$(date)] {model} seed {seed} t_max {tm:g} {arm}: NFE $nfe T $t"
        {knobs(MODELS[model], tm)} CACHE_DIR={S["cache"]} \\
          TOPK_VELOCITY={-1 if arm == 'exact' else 1} STEPS=$nfe TEMPERATURE=$t RUN_PPL_EVAL=false \\
          EVAL_BS={EVAL_BS} NUM_SAMPLE_BATCHES={NUM_SAMPLES // EVAL_BS} CKPT_PATH={ckpt(model, seed)} \\
          OUTPUT_DIR=$d DEVICES=1 bash scripts/sample/tinystories/hbfm.sh
      done
    done
    echo "[$(date)] DONE"
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--copy-fixed', action='store_true')
  ap.add_argument('--pick', action='store_true')
  ap.add_argument('--tmax', default='', help='per-model t_max, e.g. ada=0.75,fix=0.5')
  ap.add_argument('--arm', choices=['k1', 'exact'], default='k1')
  ap.add_argument('--seeds', default='1,2,3')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()

  if args.copy_fixed:   # the fixed-curvature checkpoints (sc3379 NFS) -> the A100 site's code copy
    for seed in (1, 2, 3):
      dst = os.path.dirname(ckpt('fix', seed))
      os.system(f"ssh -o BatchMode=yes -i {KEY} {REMOTE} 'mkdir -p {dst}' && rsync -a --info=progress2 "
                f"-e 'ssh -o BatchMode=yes -i {KEY}' {FIX_SRC}/{FIX_TAG.format(seed=seed)}/checkpoints/last.ckpt "
                f"{REMOTE}:{dst}/")
    return

  if args.pick:
    jobs = [(m, 1, tm, 'k1') for m, tms in PICK.items() for tm in tms]
  else:
    tmax = dict(kv.split('=') for kv in args.tmax.split(','))
    jobs = [(m, int(seed), float(tm), args.arm) for m, tm in tmax.items() for seed in args.seeds.split(',')]
  queued = set(run(S, 'squeue -h -u ch2263 -o %j').stdout.split())
  for model, seed, tm, arm in jobs:
    jobname = f'hbfm_adats_fr_{model}_sd-{seed}_tm-{tm:g}_{arm}'
    if jobname in queued:
      print(f'  skip {jobname} (queued)')
      continue
    submit(S, jobname, body(model, seed, tm, arm), S['eval_sbatch'], args.dry_run)
  if args.dry_run and jobs:
    print('\n--- example body ---\n' + body(*jobs[0]))


if __name__ == '__main__':
  main()
