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
Step type (--step-type, samplers.HBFMSampler): em (default, the bridge SDE), ode (its
probability-flow ODE) or sde (the marginal-preserving family with the diffusion scaled by
g(t / t_max), --gt-method); the arm directory and job name carry a -ode / -sde-<gt> suffix.

ORCHESTRATION ONLY: one job per (model, seed, t_max, arm) walks the NFE (x T) cells through
scripts/sample/tinystories/hbfm.sh with RUN_PPL_EVAL=false -- or, with --split nfe / cell, one
job per NFE / per (NFE, T) cell (--nfes / --temps restrict the grid). A cell with
samples_genppl.json (at the site, or locally after `sweep.py --pull`) is skipped. --site a6000
runs on sc3379's desa / thickstun GPUs (48 GB: --eval-bs 8 on the 24 GB A5000s); the ada
checkpoints must then sit at outputs/<NAME>/<cell>/checkpoints/last.ckpt on the sc3379 side.
Results land in outputs/<NAME>/frontier/<model>_sd-<seed>/tmax-<t_max>_<arm>/nfe-<NFE>_t-<T>/.

Usage:  python frontier_sweep.py --copy-fixed
        python frontier_sweep.py --pick [--models ada --step-type sde --gt-method sqrt --site a6000] [--dry-run]
        python frontier_sweep.py --tmax ada=0.75,fix=0.5 --arm k1 --seeds 1,2,3 [--dry-run]
        python frontier_sweep.py --tmax ada=0.75,fix=0.5 --arm exact --seeds 1,2,3 [--dry-run]
        python frontier_sweep.py --tmax ada=1 --arm k1 --seeds 1,2,3 --step-type ode
        python frontier_sweep.py --tmax ada=1 --arm k1 --seeds 1,2,3 --step-type sde --gt-method linear
        python frontier_sweep.py --tmax ada=1 --arm exact --seeds 2,3 --step-type ode --site a6000 \
            --nfes 64,128,256 --split cell
"""
import argparse
import os
import textwrap

from sweep import KEY, NAME, REMOTE, REPO, SITES, cell, knobs, preamble, run, submit

FIX_SRC = '/share/desa/nfs02/sc3379/workspace/research/s-flm/outputs/claude_test_hbfm_tinystories_256'
FIX_TAG = 'mode-trunc_exp_rate-3_k-0.5_ub-0.5_seed-{seed}'
MODELS = {'ada': cell(mode='trunc_exp', ub=1.0), 'fix': cell(cb=1.0, mode='trunc_exp', ub=0.5)}
PICK = {'ada': [0.25, 0.5, 0.75, 1.0], 'fix': [0.25, 0.35, 0.5]}
NFES = [1, 4, 8, 16, 32, 64, 128, 256]
TEMPS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
         '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20']
NUM_SAMPLES = 512
OUT = f'{REPO}/outputs/{NAME}/frontier'


def ckpt(s, model, seed):
  if model == 'fix':
    root = f'{s["repo"]}/outputs/claude_test_hbfm_tinystories_256' if s['remote'] else FIX_SRC
    return f'{root}/{FIX_TAG.format(seed=seed)}/checkpoints/last.ckpt'
  return f'{s["repo"]}/outputs/{NAME}/cb-4_mode-texp_rate-3_ub-1_tm-1_lr-3e-4_seed-{seed}/checkpoints/last.ckpt'


def arm_tag(arm, step_type, gt):
  return arm + {'em': '', 'ode': '-ode', 'sde': f'-sde-{gt}'}[step_type]


def cell_dir(root, model, seed, tm, tag, nfe, t):
  return f'{root}/{model}_sd-{seed}/tmax-{tm:g}_{tag}/nfe-{nfe}_t-{t}'


def body(s, model, seed, tm, arm, step_type, gt, nfes, temps, eval_bs):
  tag = arm_tag(arm, step_type, gt)
  out = f'{s["repo"]}/outputs/{NAME}/frontier'
  # last.ckpt is a mid-training checkpoint until the training job's own eval exists.
  done = os.path.dirname(os.path.dirname(ckpt(s, model, seed))) + '/eval/samples_genppl.json'
  guard = f'[ -f {done} ] || {{ echo "training of {model} seed {seed} not finished"; exit 1; }}\n'
  return preamble(s) + (guard if model == 'ada' else '') + textwrap.dedent(f'''\
    for nfe in {' '.join(map(str, nfes))}; do
      for t in {' '.join(temps)}; do
        d={cell_dir(out, model, seed, tm, tag, '${nfe}', '${t}')}
        [ -f $d/samples_genppl.json ] && continue
        echo "[$(date)] {model} seed {seed} t_max {tm:g} {tag}: NFE $nfe T $t"
        {knobs(MODELS[model], tm)} CACHE_DIR={s["cache"]} STEP_TYPE={step_type} GT_METHOD={gt} \\
          TOPK_VELOCITY={-1 if arm == 'exact' else 1} STEPS=$nfe TEMPERATURE=$t RUN_PPL_EVAL=false \\
          EVAL_BS={eval_bs} NUM_SAMPLE_BATCHES={NUM_SAMPLES // eval_bs} CKPT_PATH={ckpt(s, model, seed)} \\
          OUTPUT_DIR=$d DEVICES=1 bash scripts/sample/tinystories/hbfm.sh
      done
    done
    echo "[$(date)] DONE"
    ''')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--copy-fixed', action='store_true')
  ap.add_argument('--pick', action='store_true', help='the seed-1 k1 ladder over PICK t_max of --models')
  ap.add_argument('--models', default='ada,fix')
  ap.add_argument('--tmax', default='', help='per-model t_max, e.g. ada=0.75,fix=0.5')
  ap.add_argument('--arm', choices=['k1', 'exact'], default='k1')
  ap.add_argument('--seeds', default='1,2,3')
  ap.add_argument('--step-type', choices=['em', 'ode', 'sde'], default='em')
  ap.add_argument('--gt-method', default='linear', help='g(s) of --step-type sde (noise_schedules.GT_Method)')
  ap.add_argument('--site', choices=['a100', 'a6000'], default='a100')
  ap.add_argument('--nfes', default='', help='subset of the NFE grid, e.g. 64,128,256')
  ap.add_argument('--temps', default='', help='subset of the T grid (exact arm), e.g. 0.50,0.55')
  ap.add_argument('--split', choices=['none', 'nfe', 'cell'], default='none',
                  help='one job per (model, seed, t_max, arm) / per NFE / per (NFE, T) cell')
  ap.add_argument('--eval-bs', type=int, default=16, help='sampling batch (8 on 24 GB GPUs)')
  ap.add_argument('--exclude', default='', help='extra --exclude nodes for --site a6000')
  ap.add_argument('--suffix', default='', help='job-name suffix, to re-submit cells queued at the other site')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  S = SITES[args.site]

  if args.copy_fixed:   # the fixed-curvature checkpoints (sc3379 NFS) -> the A100 site's code copy
    for seed in (1, 2, 3):
      dst = os.path.dirname(ckpt(SITES['a100'], 'fix', seed))
      os.system(f"ssh -o BatchMode=yes -i {KEY} {REMOTE} 'mkdir -p {dst}' && rsync -a --info=progress2 "
                f"-e 'ssh -o BatchMode=yes -i {KEY}' {FIX_SRC}/{FIX_TAG.format(seed=seed)}/checkpoints/last.ckpt "
                f"{REMOTE}:{dst}/")
    return

  if args.pick:
    jobs = [(m, 1, tm, 'k1') for m in args.models.split(',') for tm in PICK[m]]
  else:
    tmax = dict(kv.split('=') for kv in args.tmax.split(','))
    jobs = [(m, int(seed), float(tm), args.arm) for m, tm in tmax.items() for seed in args.seeds.split(',')]
  nfes = [int(n) for n in args.nfes.split(',')] if args.nfes else NFES
  temps = args.temps.split(',') if args.temps else (TEMPS if args.arm == 'exact' else ['1.00'])
  tag = arm_tag(args.arm, args.step_type, args.gt_method)
  queued = set()   # a job queued at either site counts
  for site in SITES.values():
    queued |= set(run(site, f'squeue -h -u {"ch2263" if site["remote"] else "sc3379"} -o %j').stdout.split())
  opts = S['eval_sbatch'] + (f',{args.exclude}' if args.exclude else '')
  n_sub = 0
  for model, seed, tm, arm in jobs:
    if args.split == 'none':
      chunks = [(nfes, temps, '')]
    elif args.split == 'nfe':
      chunks = [([nfe], temps, f'_nfe-{nfe}') for nfe in nfes]
    else:
      chunks = [([nfe], [t], f'_nfe-{nfe}_t-{t}') for nfe in nfes for t in temps]
    for c_nfes, c_temps, suffix in chunks:
      jobname = f'hbfm_adats_fr_{model}_sd-{seed}_tm-{tm:g}_{tag}{suffix}{args.suffix}'
      if jobname in queued:
        print(f'  skip {jobname} (queued)')
        continue
      if all(os.path.exists(cell_dir(OUT, model, seed, tm, tag, nfe, t) + '/samples_genppl.json')
             for nfe in c_nfes for t in c_temps):
        continue
      submit(S, jobname, body(S, model, seed, tm, arm, args.step_type, args.gt_method, c_nfes, c_temps,
                              args.eval_bs), opts, args.dry_run)
      n_sub += 1
  print(f'{n_sub} jobs submitted to {args.site}')
  if args.dry_run and jobs:
    m, seed, tm, arm = jobs[0]
    print('\n--- example body ---\n' + body(S, m, seed, tm, arm, args.step_type, args.gt_method, nfes[:1],
                                               temps[:1], args.eval_bs))


if __name__ == '__main__':
  main()
