#!/usr/bin/env python
"""ex_geo_init_wes — geometry x embedding-init x word-embedding-dim sweep on Sudoku.

Grid (27 cells): geometry {sfm, eflm, hflm}  x  model.init {ngpt, random, unit_var}
                 x  word-embedding dim = model.hidden_size {512, 256, 128}.

Each cell = train 20k steps (identical recipe) + sudoku_eval @ 180 steps (exact velocity,
greedy, top_k_velocity=-1). One SLURM job per cell, submitted via `simple_slurm`.

Difficulty is held fixed (DIFFICULTIES, default ['medium'] — the most discriminative single
difficulty); set it to ['easy','medium','hard'] for the full 81-run sweep.

Usage:
    python experiments/ex_geo_init_wes/sweep.py --dry-run   # print grid + one command
    python experiments/ex_geo_init_wes/sweep.py             # submit all cells
"""
import argparse
import itertools
import os
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm'
PY = '/home/sc3379/anaconda3/envs/sfm/bin/python'
EXP = f'{REPO}/experiments/ex_geo_init_wes'
LOGS = f'{EXP}/logs'

# ---- grid axes ----------------------------------------------------------------
GEOS = ['sfm', 'eflm', 'hflm']
INITS = ['ngpt', 'random', 'unit_var', 'hyperbolic']  # hyperbolic (std 0.3) = hflm's native init
WES = [512, 256, 128]               # word-embedding dim == model.hidden_size
LRS = ['3e-4', '1e-4', '5e-4', '1e-3', '8e-5', '5e-5']  # full LR axis (3e-4 = default); lr-tagged
DIFFICULTIES = ['easy']             # hard first, then easy (set ['easy'] and re-run); medium already done

# ---- per-geometry knobs (everything else is identical across cells) -----------
GEO = {
    'sfm':  dict(model='tiny-sphere-dit',     algo='sfm',  sampler='sfm',  extra=''),
    'eflm': dict(model='tiny-sphere-dit',     algo='eflm', sampler='eflm', extra=''),
    'hflm': dict(model='tiny-hyperbolic-dit', algo='hflm', sampler='hflm',
                 extra='algo.prior_cov=0.25 algo.rho_max=12'),
}

# Common recipe (paper Table 1): 20k steps, effective batch 256, bf16, EMA 0.9999, grad-clip 1.0,
# AdamW lr 3e-4 — the latter four are config defaults; log-linear noise, invert=false, CE loss.
# Per-device batch 64 x grad-accum 4 (auto from global/batch) == effective 256; no batchnorm, so
# this is mathematically identical to batch 256 and fits 11GB GPUs (desa 2080ti) too.
COMMON = ('algo.invert_time_convention=false noise=log-linear '
          'loader.global_batch_size=256 loader.batch_size=64 '
          'loader.eval_batch_size=64 loader.num_workers=8')


def job_body(geo, init, wes, diff, lr=None):
    g = GEO[geo]
    lrtag = '' if lr is None else f'_lr{lr}'
    tag = f'{geo}_{init}_d{wes}{lrtag}_{diff}'
    train_out = f'{REPO}/outputs/sudoku/exgiw/{tag}'
    eval_out = f'{REPO}/eval_runs/sudoku/exgiw/{tag}'
    model_args = f'model={g["model"]} model.init={init} model.hidden_size={wes}'
    lr_arg = '' if lr is None else f'optim.lr={lr}'
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {PY} -u -m main \\
            data=sudoku data.cache_dir={REPO}/data_cache data.difficulty={diff} \\
            {model_args} \\
            algo={g["algo"]} {g["extra"]} sampler={g["sampler"]} {COMMON} {lr_arg} \\
            eval.generate_samples=False \\
            trainer.num_nodes=1 trainer.devices=1 \\
            trainer.val_check_interval=20_000 trainer.limit_val_batches=0 \\
            trainer.max_steps=20_000 \\
            callbacks.checkpoint_every_n_steps.every_n_train_steps=20_000 \\
            hydra.run.dir={train_out}
        echo "[$(date)] EVAL {tag}"
        {PY} -u -m main \\
            mode=sudoku_eval \\
            eval.checkpoint_path={train_out}/checkpoints/last.ckpt \\
            eval.strict_loading=false \\
            data=sudoku data.cache_dir={REPO}/data_cache data.difficulty={diff} \\
            {model_args} \\
            algo={g["algo"]} {g["extra"]} algo.invert_time_convention=false \\
            noise=log-linear sampler={g["sampler"]} \\
            sampler.noise_removal=greedy sampler.velocity=exact \\
            sampler.top_k_velocity=-1 sampler.steps=180 \\
            sudoku.batch_size=64 loader.eval_batch_size=64 loader.num_workers=4 \\
            trainer.num_nodes=1 trainer.devices=1 \\
            sudoku.output_dir={eval_out} +wandb.offline=True hydra.run.dir={eval_out}
        echo "[$(date)] DONE {tag}: $(grep -o '\"accuracy\":[0-9.]*' {eval_out}/results.json 2>/dev/null)"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='print the grid and one example job body without submitting')
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    cells = list(itertools.product(GEOS, INITS, WES, LRS, DIFFICULTIES))
    print(f'ex_geo_init_wes (LR axis): {len(cells)} cells '
          f'({len(GEOS)} geo x {len(INITS)} init x {len(WES)} dim x {len(LRS)} lr x {len(DIFFICULTIES)} diff)')

    if args.dry_run:
        for geo, init, wes, lr, diff in cells:
            print(f'  giw_{geo}_{init}_d{wes}_lr{lr}_{diff}')
        print('\n--- example job body (first cell) ---')
        g0 = cells[0]
        print(job_body(g0[0], g0[1], g0[2], g0[4], lr=g0[3]))
        return

    # Idempotent: skip cells that already produced results.json, so re-running after
    # extending the grid (e.g. a new init) submits only the missing cells.
    manifest = open(f'{EXP}/jobs.txt', 'a')
    n_sub = n_skip = 0
    for geo, init, wes, lr, diff in cells:
        tag = f'{geo}_{init}_d{wes}_lr{lr}_{diff}'
        if os.path.exists(f'{REPO}/eval_runs/sudoku/exgiw/{tag}/results.json'):
            n_skip += 1
            continue
        slurm = Slurm(
            job_name=f'giw_{tag}',
            partition='thickstun,desa',
            gres='gpu:1',
            ntasks=1,
            cpus_per_task=8,
            mem='64G',
            time='10:00:00',
            output=f'{LOGS}/{tag}_%j.log',
        )
        jid = slurm.sbatch(job_body(geo, init, wes, diff, lr=lr))
        print(f'  submitted {tag}: job {jid}')
        manifest.write(f'{jid}\t{tag}\n')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip} (already have results.json)')
    manifest.close()
    print(f'manifest -> {EXP}/jobs.txt')


if __name__ == '__main__':
    main()
