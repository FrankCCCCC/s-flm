#!/usr/bin/env python
"""naive_ar_tinystories_s256 — Gen. PPL / entropy frontier evaluation.

`setup.md` "GenPPL & Entropy Frontier Evaluation": for each method
{mdlm, duo, flm, sfmta} and each pretrained seed {1, 2, 3}, sweep
NFE x sampling temperature and draw 512 samples per cell. The frontier line
(Gen. PPL vs per-sample unigram entropy, one panel per NFE) is drawn by
`visualization/genppl_entropy_frontier_line.py` — S-FLM paper Fig. 10 / App. C.8.

LR is FIXED at 1e-3 — `setup.md` sweeps only the seed for this evaluation, and
RESULTS.md Sec. 3.4 selects 1e-3 as the shared LR for this setup.

ORCHESTRATION ONLY — calls the single-run shared scripts
  scripts/sample/tinystories/{mdlm,duo,flm,sfm_truncated_adaptive}.sh
once per (T, NFE) cell via their STEPS / TEMPERATURE / NUM_SAMPLE_BATCHES knobs,
with RUN_PPL_EVAL=false (valid PPL does not depend on the sampler and is already
recorded by `sweep.py`). One SLURM job per (method, seed, NFE) = 4 x 3 x 8 = 96
jobs, each walking the 15 temperatures sequentially.
Idempotent + resumable: skips cells whose samples_genppl.json exists and jobs
already in squeue; a resubmitted job re-runs only the missing cells.

Usage:  python frontier_sweep.py [--dry-run]
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/naive_ar_tinystories_s256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/naive_ar_tinystories_s256'
SEQ_LEN = 256
NUM_SAMPLES = 512

# Searched axes (setup.md). LR is fixed, see module docstring.
METHODS = ['mdlm', 'duo', 'flm', 'sfmta']
LR = '1e-3'
SEEDS = [1, 2, 3]
NFES = [1, 4, 8, 16, 32, 64, 128, 256]
TEMPS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
         '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20']
# flm/sfmta carry a dense (B, L, V) float64 sampler state, so they need a smaller batch;
# sfmta's full-vocab velocity (below) holds several such tensors at once.
EVAL_BS = {'mdlm': 32, 'duo': 32, 'flm': 16, 'sfmta': 8}
# Most methods have a same-named sample script; sfmta's is the S-FLM trunc+adaptive one.
SCRIPT = {'mdlm': 'mdlm', 'duo': 'duo', 'flm': 'flm',
          'sfmta': 'sfm_truncated_adaptive'}
# Per-method env the shared sample script needs.
#   sfmta: ALPHA_MAX is the truncation and must match the value the checkpoint was
#     trained with (sweep.py OVERRIDES). TOPK_VELOCITY=-1 is a DELIBERATE DEVIATION from
#     setup.md's default S-FLM protocol (top_k_velocity=1): top-1 renormalizes the
#     velocity weights to a point mass, so the trajectory is a pure argmax walk and the
#     temperature cancels — the whole T grid would collapse to one RNG-jittered point
#     (measured: H 3.975 at T=0.50 vs 3.967 at T=1.20). The exact full-vocab velocity
#     keeps the tempered p in v = sum_k p_k log_x(e_k), so T traces the frontier the way
#     it does for the flat baselines (measured at NFE 32: H 4.23 -> 4.59, GenPPL
#     15.7 -> 49.2 over T 0.50 -> 1.20). Same argmax-inertness that made
#     `experiments/eflm_sde` sweep eta instead of T.
EXTRA_ENV = {'sfmta': 'ALPHA_MAX=0.121 TOPK_VELOCITY=-1 '}
# Wall-clock per job. The baselines' NFE-256 job walks its 15 temperatures in ~2 h;
# sfmta's full-vocab velocity is ~20x slower per step and measured 2.9 h PER CELL
# under node contention, i.e. ~44 h for the NFE-256 job -- over the 2-day default.
TIME = {'sfmta': '4-00:00:00'}


def jobs(methods=None, seeds=None):
    return [(f'm-{m}_sd-{sd}_nfe-{nfe}', m, sd, nfe)
            for m in methods or METHODS
            for sd in seeds or SEEDS for nfe in NFES]


def run_dir(method, seed):
    return f'{OUT}/m-{method}_lr-{LR}_sd-{seed}'


def cell_dir(method, seed, nfe, temp):
    return f'{run_dir(method, seed)}/frontier/nfe-{nfe}_t-{temp}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(tag, method, seed, nfe):
    ckpt = f'{run_dir(method, seed)}/checkpoints/last.ckpt'
    bs = EVAL_BS[method]
    calls = '\n'.join(textwrap.dedent(f'''\
        if [ -f {cell_dir(method, seed, nfe, t)}/samples_genppl.json ]; then
            echo "[$(date)] SKIP {tag}_t-{t} (done)"
        else
            echo "[$(date)] CELL {tag}_t-{t}"
            {EXTRA_ENV.get(method, '')}CKPT_PATH={ckpt} OUTPUT_DIR={cell_dir(method, seed, nfe, t)} \\
                DEVICES=1 SEQ_LEN={SEQ_LEN} STEPS={nfe} TEMPERATURE={t} \\
                EVAL_BS={bs} NUM_SAMPLE_BATCHES={NUM_SAMPLES // bs} \\
                RUN_PPL_EVAL=false \\
                bash scripts/sample/tinystories/{SCRIPT[method]}.sh
        fi''') for t in TEMPS)
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] FRONTIER {tag} on $(hostname)"
        ''') + calls + f'\necho "[$(date)] DONE {tag}"\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--methods', nargs='+', default=None, choices=METHODS,
                    help='restrict submission (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='restrict submission; use to stage seeds as their '
                         'pretrained checkpoints land')
    args = ap.parse_args()
    grid = jobs(args.methods, args.seeds)
    print(f'naive_ar_tinystories_s256 frontier: {len(grid)} jobs '
          f'({len(args.methods or METHODS)} methods x '
          f'{len(args.seeds or SEEDS)} seeds x {len(NFES)} NFE) '
          f'x {len(TEMPS)} temperatures = {len(grid) * len(TEMPS)} cells, '
          f'{NUM_SAMPLES} samples each')
    if args.dry_run:
        for tag, *_ in grid:
            print(f'  nar256fr_{tag}')
        print('\n--- example body ---\n' + job_body(*grid[0]))
        return
    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for tag, method, seed, nfe in grid:
        jobname = f'nar256fr_{tag}'
        # Gate on the phase-1 eval, NOT on checkpoints/last.ckpt: `save_last=True` with
        # `every_n_train_steps=5000` rewrites last.ckpt at every 5k-step checkpoint, so
        # it exists ~5 h into a 30k-step run. Gating on it would sample a half-trained
        # model and then mark those cells done forever. eval/ppl.json is written only
        # after training finishes (sweep.py runs train then eval in one job).
        if not os.path.exists(f'{run_dir(method, seed)}/eval/ppl.json'):
            print(f'skip {tag}: training not finished (no eval/ppl.json)')
            n_skip += 1
            continue
        done = all(os.path.exists(
            f'{cell_dir(method, seed, nfe, t)}/samples_genppl.json')
            for t in TEMPS)
        if done or jobname in active:
            print(f'skip {tag}: already evaluated or queued')
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition='thickstun,desa', gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G',
                      time=TIME.get(method, '2-00:00:00'),
                      exclude='desa-compute-01',
                      output=f'{LOGS}/frontier_{tag}_%j.log')
        jid = slurm.sbatch(job_body(tag, method, seed, nfe))
        print(f'submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
