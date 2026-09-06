#!/usr/bin/env python
"""eflm_sde — EFLM temperature arm of the Gen. PPL / entropy frontier line.

`setup.md` "Frontier Line Evaluation": add {Rescale + Auto + Trunc + EFLM} to
the {MDLM, DUO, FLM} frontier of `experiments/naive_ar_tinystories_s256`, on
the same 3 pretrained seeds x NFE x 15-temperature grid, 512 samples per cell.

Protocol (setup.md): exact velocity, top_k_velocity = -1 (all), greedy last
step, eta = 0, gt_method = linear. This is the paper's "S-FLM exact velocity"
decoding (App. C.8): the full-vocab expected velocity v = p @ E - x makes p =
softmax(logits / T) temperature-dependent, so T traces a curve. With
top_k_velocity = 1 the velocity is the argmax embedding and T is inert -- the
paper draws that variant as a single marker, and those cells already exist as
the eta=0 ODE points of `frontier_sweep.py`.

ORCHESTRATION ONLY — calls the shared single-run script
  scripts/sample/tinystories/eflm_rescale_auto_truncation.sh
One SLURM job per (seed, NFE, temperature chunk); `--t-chunk` splits the
temperature grid so the expensive high-NFE cells run in parallel (the
full-vocab velocity is a float64 [B,L,V] x [V,d] einsum, ~6x the top-1 cost).
Idempotent + resumable: cells with samples_genppl.json and queued job names
are skipped.

Usage:
  python experiments/eflm_sde/tfrontier_sweep.py --nfes 1 4 8 16 32 [--dry-run]
  python experiments/eflm_sde/tfrontier_sweep.py --nfes 64 128 256 --t-chunk 5
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

SEQ_LEN = 256
EVAL_BS = 16
NUM_SAMPLES = 512
# Training config of the frozen checkpoints (must match; see parent sweep).
RHO = '0.5'
TAU_MAX = '2.4436'

SEEDS = [1, 2, 3]
NFES = [1, 4, 8, 16, 32, 64, 128, 256]
TEMPS = ['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85',
         '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20']

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


def cell_dir(seed, nfe, temp):
    return f'{OUT}/sd-{seed}/tfrontier/nfe-{nfe}_t-{temp}'


def eflm_call(seed, nfe, temp):
    d = cell_dir(seed, nfe, temp)
    return textwrap.dedent(f'''\
        if [ -f {d}/samples_genppl.json ]; then
            echo "[$(date)] SKIP nfe-{nfe}_t-{temp} (done)"
        else
            echo "[$(date)] CELL sd-{seed}_nfe-{nfe}_t-{temp}"
            CKPT_PATH={eflm_ckpt(seed)} OUTPUT_DIR={d} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
                RHO={RHO} TAU_MAX={TAU_MAX} ALPHA_MAX=null SNR_CE=false \\
                STEPS={nfe} ETA=0.0 GT_METHOD=linear \\
                VELOCITY=exact TOPK_VELOCITY=-1 TEMPERATURE={temp} \\
                EVAL_BS={EVAL_BS} NUM_SAMPLE_BATCHES={NUM_SAMPLES // EVAL_BS} \\
                RUN_PPL_EVAL=false \\
                bash scripts/sample/tinystories/eflm_rescale_auto_truncation.sh
        fi''')


def jobs(seeds, nfes, temps, t_chunk):
    """-> [(jobname, [call, ...], [donefile, ...])]"""
    out = []
    for sd in seeds:
        for nfe in nfes:
            chunks = [temps[i:i + t_chunk]
                      for i in range(0, len(temps), t_chunk)]
            for k, ts in enumerate(chunks):
                name = f'eflmsdet_sd-{sd}_nfe-{nfe}'
                if len(chunks) > 1:
                    name += f'_tc-{k}'
                out.append((name, [eflm_call(sd, nfe, t) for t in ts],
                            [f'{cell_dir(sd, nfe, t)}/samples_genppl.json'
                             for t in ts]))
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
    ap.add_argument('--temps', nargs='+', default=TEMPS)
    ap.add_argument('--t-chunk', type=int, default=len(TEMPS),
                    help='temperatures per SLURM job (default: all 15)')
    args = ap.parse_args()

    grid = jobs(args.seeds, args.nfes, args.temps, args.t_chunk)
    n_cells = sum(len(c) for _, c, _ in grid)
    print(f'eflm_sde T-frontier: {len(grid)} jobs, {n_cells} cells, '
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
