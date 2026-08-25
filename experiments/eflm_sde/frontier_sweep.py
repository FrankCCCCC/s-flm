#!/usr/bin/env python
"""eflm_sde — GenPPL / entropy frontier of the EFLM SDE sampler.

`setup.md`: on the frozen best checkpoints of
`experiments/eflm_rescale_auto_trunc_tinystories_256` (recommended config
eflmratr_lr-1e-3_r-0.5_m-1.0, 3 training seeds), sweep the SDE knobs
eta x gt_method x NFE and draw 512 samples per cell. Under the recorded
protocol (exact velocity, top_k_velocity=1, greedy last) temperature is
argmax-inert, so eta traces EFLM's frontier the way T traces the baselines'
(`experiments/naive_ar_tinystories_s256`). Also submits the S-FLM reference
points (seed_errbar checkpoints, one T-inert point per NFE).

ORCHESTRATION ONLY — calls the shared single-run scripts
  scripts/sample/tinystories/eflm_rescale_auto_truncation.sh   (EFLM cells)
  scripts/sample/tinystories/sfm_truncated_adaptive.sh          (S-FLM refs)
One SLURM job per (seed, NFE, gt) walking the eta grid; one 'ode' (eta=0)
job per seed walking the NFE grid; one sfm job per seed walking the NFE grid.
Idempotent + resumable: cells with samples_genppl.json and queued job names
are skipped; a resubmitted job re-runs only missing cells.

Usage:
  # phase 1 pilot (defaults: seed 1, NFE {16, 64}, 4 gt x 8 eta + ode refs)
  python experiments/eflm_sde/frontier_sweep.py [--dry-run]
  # phase 2 frontier (after the pilot picks the gt / eta grid)
  python experiments/eflm_sde/frontier_sweep.py --seeds 1 2 3 \
      --nfes 1 4 8 16 32 64 128 256 --gts <winner> --etas <refined grid>
  # S-FLM reference points
  python experiments/eflm_sde/frontier_sweep.py --sfm --seeds 1 2 3 \
      --nfes 1 4 8 16 32 64 128 256
"""
import argparse
import getpass
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/eflm_sde'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/eflm_sde'
EFLM_CKPTS = f'{REPO}/outputs/eflm_rescale_auto_trunc_tinystories_256'
SFM_CKPTS = f'{REPO}/outputs/seed_errbar_tinystories_256'

SEQ_LEN = 256
EVAL_BS = 16
NUM_SAMPLES = 512
# Training config of the frozen checkpoints (must match; see parent sweep).
RHO = '0.5'
TAU_MAX = '2.4436'

# Pilot defaults (EXPERIMENT.md phase 1).
SEEDS = [1]
NFES = [16, 64]
GTS = ['const', 'sqrt', 'linear', 'quad']
ETAS = ['0.25', '0.5', '1', '2', '4', '8', '16', '32']
ODE_NFES_EXTRA = [180]   # eta=0 at 180 steps ties back to the recorded 11.53

_HEAD = textwrap.dedent(f'''\
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export TORCHDYNAMO_DISABLE=1
    export SLURM_JOB_NAME=bash
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PATH={ENVBIN}:$PATH
    cd {REPO}
    ''')


def eflm_ckpt(seed):
    sfx = '' if seed == 1 else f'_s{seed}'
    return f'{EFLM_CKPTS}/eflmratr_lr-1e-3_r-0.5_m-1.0{sfx}/checkpoints/last.ckpt'


def cell_dir(seed, nfe, gt, eta):
    return f'{OUT}/sd-{seed}/frontier/nfe-{nfe}_gt-{gt}_eta-{eta}'


def sfm_cell_dir(seed, nfe):
    return f'{OUT}/sfm_sd-{seed}/frontier/nfe-{nfe}'


def eflm_call(seed, nfe, gt, eta):
    d = cell_dir(seed, nfe, gt, eta)
    return textwrap.dedent(f'''\
        if [ -f {d}/samples_genppl.json ]; then
            echo "[$(date)] SKIP nfe-{nfe}_gt-{gt}_eta-{eta} (done)"
        else
            echo "[$(date)] CELL sd-{seed}_nfe-{nfe}_gt-{gt}_eta-{eta}"
            CKPT_PATH={eflm_ckpt(seed)} OUTPUT_DIR={d} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
                RHO={RHO} TAU_MAX={TAU_MAX} ALPHA_MAX=null SNR_CE=false \\
                STEPS={nfe} ETA={0 if gt == 'ode' else eta} GT_METHOD={'linear' if gt == 'ode' else gt} \\
                EVAL_BS={EVAL_BS} NUM_SAMPLE_BATCHES={NUM_SAMPLES // EVAL_BS} \\
                RUN_PPL_EVAL=false \\
                bash scripts/sample/tinystories/eflm_rescale_auto_truncation.sh
        fi''')


def sfm_call(seed, nfe):
    d = sfm_cell_dir(seed, nfe)
    return textwrap.dedent(f'''\
        if [ -f {d}/samples_genppl.json ]; then
            echo "[$(date)] SKIP sfm nfe-{nfe} (done)"
        else
            echo "[$(date)] CELL sfm_sd-{seed}_nfe-{nfe}"
            CKPT_PATH={SFM_CKPTS}/sfm_seed{seed}/checkpoints/last.ckpt \\
                OUTPUT_DIR={d} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
                SELF_COND=true ALPHA_MAX=0.121 \\
                STEPS={nfe} EVAL_BS={EVAL_BS} NUM_SAMPLE_BATCHES={NUM_SAMPLES // EVAL_BS} \\
                RUN_PPL_EVAL=false \\
                bash scripts/sample/tinystories/sfm_truncated_adaptive.sh
        fi''')


def jobs(seeds, nfes, gts, etas, sfm, ode_extra=ODE_NFES_EXTRA):
    """-> [(jobname, [call, ...], [donefile, ...])]"""
    out = []
    if sfm:
        for sd in seeds:
            out.append((f'eflmsde_sfm_sd-{sd}',
                        [sfm_call(sd, n) for n in nfes],
                        [f'{sfm_cell_dir(sd, n)}/samples_genppl.json'
                         for n in nfes]))
        return out
    for sd in seeds:
        # eta=0 ODE references: one job per seed walking the NFE grid. The
        # job name carries the NFE range so banded submissions (different
        # --nfes / --etas per NFE band) don't collide in squeue.
        ode_nfes = sorted(set(nfes) | set(ode_extra))
        out.append((f'eflmsde_sd-{sd}_ode_n{min(ode_nfes)}-{max(ode_nfes)}',
                    [eflm_call(sd, n, 'ode', '0') for n in ode_nfes],
                    [f'{cell_dir(sd, n, "ode", "0")}/samples_genppl.json'
                     for n in ode_nfes]))
        for nfe in nfes:
            if nfe == 1:
                continue   # single step = decode step; the SDE never acts
            for gt in gts:
                out.append((f'eflmsde_sd-{sd}_nfe-{nfe}_gt-{gt}',
                            [eflm_call(sd, nfe, gt, e) for e in etas],
                            [f'{cell_dir(sd, nfe, gt, e)}/samples_genppl.json'
                             for e in etas]))
    return out


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--nfes', nargs='+', type=int, default=NFES)
    ap.add_argument('--gts', nargs='+', default=GTS,
                    choices=['const', 'sqrt', 'p75', 'linear', 'quad'])
    ap.add_argument('--etas', nargs='+', default=ETAS)
    ap.add_argument('--sfm', action='store_true',
                    help='submit the S-FLM reference jobs instead')
    ap.add_argument('--ode-extra', nargs='*', type=int, default=ODE_NFES_EXTRA,
                    help='extra NFEs for the eta=0 reference job; pass with '
                         'no values to disable (banded submissions)')
    args = ap.parse_args()

    grid = jobs(args.seeds, args.nfes, args.gts, args.etas, args.sfm,
                args.ode_extra)
    n_cells = sum(len(c) for _, c, _ in grid)
    print(f'eflm_sde frontier: {len(grid)} jobs, {n_cells} cells, '
          f'{NUM_SAMPLES} samples each')
    if args.dry_run:
        for name, calls, _ in grid:
            print(f'  {name} ({len(calls)} cells)')
        print('\n--- example body ---\n' + _HEAD + '\n'.join(grid[0][1][:2]))
        return

    os.makedirs(LOGS, exist_ok=True)
    active = active_jobnames()
    n_sub = n_skip = 0
    for name, calls, donefiles in grid:
        if name in active or all(os.path.exists(f) for f in donefiles):
            print(f'skip {name}: done or queued')
            n_skip += 1
            continue
        body = (_HEAD + f'echo "[$(date)] {name} on $(hostname)"\n'
                + '\n'.join(calls) + f'\necho "[$(date)] DONE {name}"\n')
        slurm = Slurm(job_name=name, output=f'{LOGS}/{name}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=8,
                      mem='64G', time='2-00:00:00')
        jid = slurm.sbatch(body, sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'submitted {name}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
