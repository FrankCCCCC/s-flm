#!/usr/bin/env python
"""elfm_hbfm_ada_curv_elbo_sudoku -- ELBO vs CE training of EFLM and HBFM (adaptive curvature) on
hard Sudoku, against MDLM.

Spec: `setup.md`. Math: `math.md`. Design: `EXPERIMENT.md`. Table: `report.py` -> `RESULTS.md`.

Arms (a cell = one 20k-step training run, batch 256, EMA 0.9999, + its evals):
  hbfm  the anchor of experiments/hbfm_rlf_ada_curv ((H^3)^3 at K = -0.5, exp proposal of rate 3,
        horosphere readout, adaptive curvature bound 4), trained with LOSS in {ce, elbo}
  eflm  the best CE cell of experiments/eflm_rescale_auto_sudoku (R = 8, autonomous clock,
        tau_max 0.5), trained with LOSS in {ce, elbo}
  mdlm  the masked-diffusion baseline (its own NELBO)
Every cell is evaluated for board accuracy (sudoku_eval: 180 steps, exact velocity, greedy last
step, 2000 boards) and for the likelihood bound (ppl_eval, PPL_SEEDS passes, nats per solution
token); HBFM / EFLM cells also report the CE objective. CE controls that already exist with the same
training code (hbfm_rlf_ada_curv cb-4 at LR 3e-4, eflm_rescale_auto_sudoku R=8 / tau 0.5 / LR 5e-4,
seeds 1-3) are linked in (checkpoints + results.json) and only get the ppl passes.

ORCHESTRATION ONLY -- every cell calls scripts/train/sudoku/<stem>.sh then
scripts/sample/sudoku/<stem>.sh. Idempotent + resumable: a cell whose last artifact exists or whose
job name is in squeue is skipped; resubmitting the same OUTPUT_DIR resumes from checkpoints/last.ckpt.

Usage:  python sweep.py --stage smoke                       # 300 steps + evals, speed / memory
        python sweep.py --stage 1                           # LR sweep (seed 1) + reused controls
        python sweep.py --stage 2 --lr hbfm-elbo=5e-4 eflm-elbo=1e-3 mdlm-elbo=3e-4 [--seeds 2,3]
        python report.py                                    # -> RESULTS.md
With --pilot N the stage's cells are packed into ONE job of N GPUs run by pilot.py (one cell per
GPU slot at a time) instead of one job per cell: a single-GPU job of this account does not get
scheduled while another sweep of the user holds ~100 pending jobs, a multi-GPU one does
(RESULTS.md, Incidents). --extra adds arbitrary scripts (e.g. the test suite) to the pilot queue.
"""
import argparse
import glob
import os
import shutil
import subprocess
import textwrap
import time

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
PROJECT = 'elfm_hbfm_ada_curv_elbo_sudoku'
EXP = f'{REPO}/experiments/{PROJECT}'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/{PROJECT}'

MAX_STEPS = 20000
CKPT_EVERY = 5000
GLOBAL_BATCH = 256
LRS = ['3e-4', '5e-4', '1e-3']
PPL_SEEDS = '1 2 3 4'
# setup.md: the 2080 Ti (desa-compute-01, 8 GPUs, 11 GB) for the large-scale runs.
PARTITION = 'desa'
GRES = 'gpu:nvidia_geforce_rtx_2080_ti:1'
# Another allocation holds 195 GB of the node's 251 GB in SLURM's books (160 GB are actually
# available), so a cell asks for 9 GB (measured peak 7.5 GB): six pilot slots fit the remaining
# 56 GB.
MEM = '9G'

STEM = {'hbfm': 'hbfm', 'eflm': 'eflm_rescale_auto', 'mdlm': 'mdlm'}
# Fixed knobs of each arm (the searched ones -- loss, LR, seed -- are in the tag).
KNOBS = {
    'hbfm': ("EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[-0.5,-0.5,-0.5]' GAUSS_CURV=-0.5 "
             "CURV_BOUND=4 PROPOSAL_RATE=3 T_MAX=6 TIME_CONVERSION_MODE=exp TIME_RANGE_UPPER_BOUND=1.0 "
             "FORWARD_TYPE=horosphere DIFFICULTY=hard"),
    'eflm': 'RHO=8 TAU_MAX=0.5 SNR_CE=false DIFFICULTY=hard',
    'mdlm': 'DIFFICULTY=hard',
}
# 11 GB: the ELBO's second forward pass doubles the activation memory, so the ELBO arms run
# 64 x 4 micro-batches = the optimizer batch of 256 (the CE arms ran 128 x 2; the gradient is
# the same). MDLM's DiT has no second pass.
PER_GPU_BS = {('hbfm', 'elbo'): 64, ('hbfm', 'ce'): 128, ('eflm', 'elbo'): 64, ('eflm', 'ce'): 128,
              ('mdlm', 'elbo'): 128}
# Measured in stage 1 (2080 Ti, incl. the evals): HBFM-ELBO 18.4 h, EFLM-ELBO 16.2 h, MDLM ~8 h;
# a pilot's limit is ceil(sum / slots) + 2 h, so the budgets carry the margin.
HOURS = {'hbfm': 36, 'eflm': 22, 'mdlm': 12}
# Same-code CE controls, trained on this harness already (see EXPERIMENT.md).
REUSE = {
    ('hbfm', 'ce', '3e-4'): f'{REPO}/outputs/hbfm_rlf_ada_curv/'
                            'cb-4_mode-exp_rate-3_k-0.5_ub-na_tm-6_fwd-horo_lr-3e-4_seed-{seed}',
    ('eflm', 'ce', '5e-4'): f'{REPO}/outputs/eflm_rescale_auto_sudoku/'
                            'eflmra_auto-naive_r-8_tau-0.5_w-ce_lr-5e-4_d-hard_rs{seed}',
}


def cell(arm, loss, lr, seed):
    return dict(arm=arm, loss=loss, lr=lr, seed=int(seed))


def tag_of(c):
    return f'{c["arm"]}_loss-{c["loss"]}_lr-{c["lr"]}_seed-{c["seed"]}'


def reuse_dir(c):
    src = REUSE.get((c['arm'], c['loss'], c['lr']))
    return None if src is None else src.format(seed=c['seed'])


def active_jobnames():
    """Names of the user's queued jobs, held ones excluded (they cannot start)."""
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j %R'],
                         capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if 'JobHeld' not in line}


def link_reused(c, tdir):
    """Checkpoints of an existing control -> symlink; its board accuracy -> copied results.json."""
    src = reuse_dir(c)
    os.makedirs(f'{tdir}/eval', exist_ok=True)
    if not os.path.exists(f'{tdir}/checkpoints'):
        os.symlink(f'{src}/checkpoints', f'{tdir}/checkpoints')
    if not os.path.exists(f'{tdir}/eval/results.json'):
        shutil.copy(f'{src}/eval/results.json', f'{tdir}/eval/results.json')


def job_body(c, tdir, max_steps, ppl_seeds, reused, sudoku_eval=True):
    # TMPDIR is per (job, GPU slot) and SHORT: the dataloader workers bind a Unix socket under
    # it, whose path must stay below 108 bytes (a per-cell name overflowed it).
    arm, stem, knobs = c['arm'], STEM[c['arm']], KNOBS[c['arm']]
    ckpt = f'CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1'
    parts = [f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/elbo_${{SLURM_JOB_ID}}_g${{CUDA_VISIBLE_DEVICES:-0}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        ''']
    if not reused:
        parts.append(f'''\
        echo "[$(date)] TRAIN {tag_of(c)} on $(hostname)"
        nvidia-smi --query-gpu=name --format=csv,noheader
        {knobs} LOSS_TYPE={c["loss"]} LR={c["lr"]} SEED={c["seed"]} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS={PER_GPU_BS[(arm, c["loss"])]} GLOBAL_BATCH={GLOBAL_BATCH} \\
            NUM_WORKERS=2 MAX_STEPS={max_steps} CKPT_EVERY={min(CKPT_EVERY, max_steps)} \\
            bash scripts/train/sudoku/{stem}.sh
        ''')
    # One ppl_eval pass per call of the eval script (seeds `ppl_seeds`, averaged by report.py),
    # then the board accuracy at the cell's seed; the CE objective for the continuous arms.
    def ppl_passes(loss):
        return f'''\
        for s in {ppl_seeds}; do
            {knobs} SEED=$s {ckpt} LOSS_TYPE={loss} RUN_PPL_EVAL=true RUN_SAMPLE_EVAL=false \\
                PPL_JSON={tdir}/eval/ppl_{loss}_s$s.json bash scripts/sample/sudoku/{stem}.sh
        done
        '''
    parts.append(f'echo "[$(date)] EVAL {tag_of(c)}: NELBO + board accuracy"\n')
    parts.append(ppl_passes('elbo'))
    if sudoku_eval:
        parts.append(f'''\
        {knobs} SEED={c["seed"]} {ckpt} LOSS_TYPE=elbo RUN_PPL_EVAL=false RUN_SAMPLE_EVAL=true \\
            bash scripts/sample/sudoku/{stem}.sh
        ''')
    if arm != 'mdlm':
        parts.append(f'echo "[$(date)] EVAL {tag_of(c)}: CE objective"\n')
        parts.append(ppl_passes('ce'))
    parts.append(f'echo "[$(date)] DONE {tag_of(c)}"\n')
    return ''.join(textwrap.dedent(p) for p in parts)


def stage_cells(stage, seeds, lr_pick):
    if stage == 'smoke':
        return [cell('hbfm', 'elbo', '3e-4', 1), cell('eflm', 'elbo', '3e-4', 1),
                cell('mdlm', 'elbo', '3e-4', 1), cell('hbfm', 'ce', '3e-4', 1)]
    if stage == '1':
        return ([cell('hbfm', 'ce', '3e-4', s) for s in (1, 2, 3)]      # reused controls (eval only)
                + [cell('eflm', 'ce', '5e-4', s) for s in (1, 2, 3)]
                + [cell('hbfm', 'elbo', lr, 1) for lr in LRS]
                + [cell('hbfm', 'ce', lr, 1) for lr in LRS if lr != '3e-4']
                + [cell('eflm', 'elbo', lr, 1) for lr in LRS]
                + [cell('mdlm', 'elbo', lr, 1) for lr in LRS])
    if stage == '2':
        cells = []
        for key, lr in lr_pick.items():
            arm, loss = key.split('-')
            cells += [cell(arm, loss, lr, s) for s in seeds]
        return cells
    raise ValueError(stage)


def pilot_tags(active):
    """Cells carried by pilot jobs that are still in the queue."""
    tags = set()
    for d in glob.glob(f'{LOGS}/pilots/*'):
        if os.path.basename(d) in active and os.path.exists(f'{d}/cells.txt'):
            tags |= set(open(f'{d}/cells.txt').read().split())
    return tags


def submit_pilot(args, cells, hours):
    n = args.pilot
    name = f'{PROJECT}_pilot_{args.stage}_{time.strftime("%m%d-%H%M%S")}'
    d = f'{LOGS}/pilots/{name}'
    scripts = list(args.extra)
    if not args.dry_run:
        os.makedirs(d)
        with open(f'{d}/cells.txt', 'w') as f:
            f.write('\n'.join(f'{args.stage}:{tag}' for tag, _ in cells) + '\n')
    for tag, body in cells:
        scripts.append(f'{d}/{tag}.sh')
        if not args.dry_run:
            with open(scripts[-1], 'w') as f:
                f.write(body)
    hours = max(2, -(-hours // n) + 2)       # the slots share the cells: ceil(total / n) + margin
    body = textwrap.dedent(f'''\
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        python experiments/{PROJECT}/pilot.py --gpus {n} {' '.join(scripts)}
        ''')
    print(f'  pilot {name}: {len(cells)} cells + {len(args.extra)} extra on {n} GPUs, {hours} h')
    if args.dry_run:
        return
    slurm = Slurm(job_name=name, partition=PARTITION, gres=GRES.rsplit(':', 1)[0] + f':{n}',
                  ntasks=1, cpus_per_task=4 * n, mem=f'{9 * n}G', time=f'{hours}:00:00',
                  output=f'{d}/slurm_%j.log')
    print(f'  submitted {name}: job {slurm.sbatch(body)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', '1', '2'])
    ap.add_argument('--seeds', default='2,3', help='stage 2: comma list of seeds')
    ap.add_argument('--lr', nargs='*', default=[], metavar='ARM-LOSS=LR',
                    help='stage 2: the LR of each arm, e.g. hbfm-elbo=5e-4')
    ap.add_argument('--only', default='', help='substring filter on the cell tag')
    ap.add_argument('--pilot', type=int, default=0, metavar='N',
                    help='pack the cells into one N-GPU job run by pilot.py (0 = one job per cell)')
    ap.add_argument('--extra', nargs='*', default=[], metavar='SCRIPT',
                    help='--pilot only: more scripts to run first (e.g. the test suite)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    smoke = args.stage == 'smoke'
    out = f'{OUT}/smoke' if smoke else OUT
    max_steps = 300 if smoke else MAX_STEPS
    ppl_seeds = '1' if smoke else PPL_SEEDS
    lr_pick = dict(kv.split('=') for kv in args.lr)
    cells = [c for c in stage_cells(args.stage, args.seeds.split(','), lr_pick)
             if args.only in tag_of(c)]
    active = set() if args.dry_run else active_jobnames()
    n_sub = n_skip = 0
    pilot_cells, pilot_hours = [], 0
    for c in cells:
        tag = tag_of(c)
        tdir = f'{out}/{tag}'
        # A reused control is only evaluated: the smoke stage re-runs its board eval (sampler
        # check), the main stages copy the recorded one and add the ppl passes.
        reused = reuse_dir(c) is not None and (not smoke or c['loss'] == 'ce')
        last = f'{tdir}/eval/ppl_elbo_s{ppl_seeds.split()[-1]}.json'
        jobname = f'{PROJECT}_{"smoke_" if smoke else ""}{tag}'
        if os.path.exists(last) or jobname in active:
            n_skip += 1
            continue
        if reused and not args.dry_run:
            if smoke:
                os.makedirs(f'{tdir}/eval', exist_ok=True)
                if not os.path.exists(f'{tdir}/checkpoints'):
                    os.symlink(f'{reuse_dir(c)}/checkpoints', f'{tdir}/checkpoints')
            else:
                link_reused(c, tdir)
        body = job_body(c, tdir, max_steps, ppl_seeds, reused, sudoku_eval=smoke or not reused)
        n_sub += 1
        hours = 2 if smoke else (1 if reused else HOURS[c['arm']])
        if args.pilot:
            if f'{args.stage}:{tag}' in pilot_tags(active):
                n_sub -= 1
                n_skip += 1
                continue
            pilot_cells.append((tag, body))
            pilot_hours += hours
            continue
        if args.dry_run:
            print(f'  [dry] {jobname}' + (' (reused checkpoint)' if reused else ''))
            continue
        slurm = Slurm(job_name=jobname, partition=PARTITION, gres=GRES,
                      ntasks=1, cpus_per_task=2, mem=MEM, time=f'{hours}:00:00',
                      output=f'{LOGS}/{jobname}_%j.log')
        print(f'  submitted {jobname}: job {slurm.sbatch(body)}')
    if args.pilot and (pilot_cells or args.extra):
        submit_pilot(args, pilot_cells, pilot_hours)
    print(f'stage {args.stage}: submitted {n_sub}, skipped {n_skip}')
    if args.dry_run and cells:
        c = cells[0]
        print('\n--- example body ---\n' + job_body(c, f'{out}/{tag_of(c)}', max_steps, ppl_seeds, False))


if __name__ == '__main__':
    main()
