#!/usr/bin/env python
"""claude_test_hbfm_sudoku, round 3 — the heat-time PROPOSAL of HBFM on hard Sudoku.

Spec: `setup.md` ("vary forward_type, time_exp_rate, curvature, time_conversion_mode and
time_range_upper_bound to find the best accuracy"; LR only from {3e-4, 5e-4, 1e-3}; 2080 Ti only).

Round 2 (`sweep.py`, `sweep_tune.py`) searched the exp proposal alone and left the anchor
  mode=exp, physical rate 3 (unit rate 6), K=-0.5, log-linear, LR 3e-4, forward_type=horosphere,
  sampler t_max 6 (unit 3), 180 steps  ->  61.1 +- 1.8 % boards (seeds 1/2/3 = 60.4/59.8/63.2).
This round adds the two proposals introduced in 2255661 (`trunc_exp`, `unif`) and reads the whole
grid as RANGE x SHAPE of the training heat-time distribution:

  * with `noise=log-linear` (eps 1e-3) the schedule draws u in [1e-3, 0.999], so mode=exp only ever
    trains heat times t <= ln(1e3)/rate = 6.9/rate -- the "unbounded" proposal is in fact bounded by
    the eps floor (rate 3 -> t <= 2.3), and its weight 1/(rate*u) spans 4 decades;
  * mode=trunc_exp / unif set that range EXACTLY (time_range_upper_bound) and keep the weight O(ub),
    which matters because trainer.gradient_clip_val is pinned at 1.0 and round 2 showed the seed
    variance blows up when the loss scale moves (RESULTS.md, "Clip control").
  * the geometry only sees unit time |K|*t, so (rate, K) -> (c*rate, c*K) is an equivalence; K is
    varied at FIXED unit rate 6 to isolate what is left (the DiT input scale R ~ 1/sqrt(|K|)).

Cells (one GPU each, 20k steps, batch 256 = PER_GPU_BS x accumulation, then 2000-board eval):
  RANGE x SHAPE (K -0.5, horosphere, LR 3e-4): unif / trunc_exp at ub in {1, 2.3, 6}, exp at the
  rate that gives range 1 (6.9); trunc_exp at the spec's rate 0.01 truncated to 2.3.
  AXIS (one knob off the anchor): forward_type=naive; LR 5e-4 / 1e-3; K -0.25 / -1 at unit rate 6.
Sampler horizon: t_max = ub for the bounded modes (the HBFM validator rejects t_max > ub), 6 for exp.
`--anchor-tmax` re-evaluates the finished round-2 anchor checkpoints at t_max 1.0 / 2.3 so the
horizon change of the bounded cells is not confounded with their proposal change.

ORCHESTRATION ONLY — every cell calls scripts/train/sudoku/hbfm.sh + scripts/sample/sudoku/hbfm.sh.
Idempotent + resumable: a cell with eval/results.json or a queued job of the same name is skipped;
resubmitting the same OUTPUT_DIR resumes from checkpoints/last.ckpt.

Usage:  python sweep_mode.py [--dry-run] [--only SUBSTR] [--confirm] [--anchor-tmax]
        python report_mode.py   ->  the table in RESULTS.md
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/claude_test_hbfm_sudoku'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/claude_test_hbfm_sudoku'

MAX_STEPS = 20000
CKPT_EVERY = 5000
PER_GPU_BS = 128          # 11 GB 2080 Ti: batch 256 OOMs, 128 x 2 accumulation keeps the spec's 256
GLOBAL_BATCH = 256
# setup.md says 2080 Ti only; desa-compute-01 is the only 2080 Ti node in desa/thickstun (8 GPUs),
# so the `gpu` partition's 2080 Ti nodes were added (user-approved 2026-09-18) with the same gres
# pin -- still no contention with the TinyStories sweep, which holds the A5000/A6000/6000-Ada nodes.
PARTITION = 'desa,thickstun,gpu'
GRES = 'gpu:nvidia_geforce_rtx_2080_ti:1'
# snavely-compute-01 has a flaky GPU: it silently killed 5 cells on 2026-09-18 (one direct
# `GPU is lost`, four `CANCELLED by 0` when an admin fenced it, one more after it was returned to
# service). SLURM records all of them as `COMPLETED 0:0`, so they only show up as a missing
# eval/results.json. Excluded until it has been stable for a while.
EXCLUDE = 'snavely-compute-01'
MODE_TAG = {'exp': 'exp', 'trunc_exp': 'texp', 'unif': 'unif'}
FWD_TAG = {'naive': 'naive', 'horosphere': 'horo'}

ANCHOR = dict(mode='exp', rate=3.0, k=-0.5, ub=None, tmax=6.0, fwd='horosphere', lr='3e-4')


def cell(**kw):
    c = dict(ANCHOR)
    c.update(kw)
    return c


# --- stage 1: one seed per cell -------------------------------------------------------------
STAGE1 = [
    # RANGE x SHAPE at the anchor geometry (K -0.5, horosphere, LR 3e-4)
    cell(mode='unif', ub=1.0, tmax=1.0),
    cell(mode='unif', ub=2.3, tmax=2.3),
    cell(mode='unif', ub=6.0, tmax=6.0),
    cell(mode='trunc_exp', ub=1.0, tmax=1.0),
    cell(mode='trunc_exp', ub=2.3, tmax=2.3),
    cell(mode='trunc_exp', ub=6.0, tmax=6.0),
    cell(mode='exp', rate=6.9),                      # exp whose eps-floor range is 1.0
    # the ub = 6 family is the ONLY horizon-free comparison (t_max 6 = the anchor's), so the shape
    # ladder is run there: mean trained t = 0.33 (texp 3 = the anchor) -> 0.95 -> 1.6 -> 3.0 (unif)
    cell(mode='trunc_exp', rate=1.0, ub=6.0, tmax=6.0),
    cell(mode='trunc_exp', rate=0.5, ub=6.0, tmax=6.0),
    cell(mode='exp', rate=4.0),                      # unit rate 8: the gap between 61.1 (unit 6) and 57.9 (unit 10)
    # the two `unif` horizons `eflm_rescale_auto_sudoku` says cannot both win: its autonomous clock is
    # exactly `unif`, its optimum sat at tau_max = tau* (the context-free decode time, measured 0.833
    # unit = 1.67 physical here), while HBFM's own rate curve says the winner allocates mass like
    # exp rate 3, which is unif at unit 0.22 = 0.44 physical
    cell(mode='unif', ub=1.67, tmax=1.67),
    cell(mode='unif', ub=0.44, tmax=0.44),
    cell(mode='trunc_exp', rate=0.01, ub=2.3, tmax=2.3),   # the spec's rate, truncated
    # AXIS: exactly one knob off the anchor
    cell(fwd='naive'),
    cell(lr='5e-4'),
    cell(lr='1e-3'),
    cell(k=-0.25, rate=1.5, tmax=12.0),              # unit rate 6, |K| halved  (t_max = unit 3 / |K|)
    cell(k=-1.0, rate=6.0, tmax=3.0),                # unit rate 6, |K| doubled (t_max = unit 3 / |K|)
]

# --- stage 2: seeds 2 and 3 --------------------------------------------------------------------
# Chosen for two jobs at once. (a) the three cells at/above the baseline, to find out whether any of
# them is really better than the reproduction cell or whether stage 1's ordering is noise; (b) the
# `trunc_exp(0.01, ub 2.3)` / `unif(2.3)` pair, which are near-identical by construction yet scored
# 33.7 vs 39.7 at seed 1 -- replicating them MEASURES the noise floor instead of assuming it, and
# every other conclusion in the round is quoted against that floor.
CONFIRM = [
    cell(k=-0.25, rate=1.5, tmax=12.0),                    # 68.2 - best of stage 1
    cell(lr='5e-4'),                                       # 67.1
    cell(mode='trunc_exp', ub=6.0, tmax=6.0),              # 65.0 - the baseline / reproduction cell
    cell(k=-1.0, rate=6.0, tmax=3.0),                      # 65.0 - the other end of the K axis
    cell(mode='trunc_exp', rate=0.01, ub=2.3, tmax=2.3),   # 33.7 \ near-identical pair:
    cell(mode='unif', ub=2.3, tmax=2.3),                   # 39.7 / their spread IS the noise floor
]


def tag_of(c, seed):
    ub = 'na' if c['ub'] is None else f'{c["ub"]:g}'
    return (f'mode-{MODE_TAG[c["mode"]]}_rate-{c["rate"]:g}_k{c["k"]:g}_ub-{ub}'
            f'_tm-{c["tmax"]:g}_fwd-{FWD_TAG[c["fwd"]]}_lr-{c["lr"]}_seed-{seed}')


def geometry(c, tmax=None):
    k = c['k']
    return (f"EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[{k:g},{k:g},{k:g}]' "
            f"GAUSS_CURV={k:g} PROPOSAL_RATE={c['rate']:g} T_MAX={tmax or c['tmax']:g} "
            f"TIME_CONVERSION_MODE={c['mode']} "
            f"TIME_RANGE_UPPER_BOUND={1.0 if c['ub'] is None else c['ub']:g} "
            f"FORWARD_TYPE={c['fwd']}")


def active_jobnames():
    out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                         capture_output=True, text=True).stdout
    return set(out.split())


def job_body(c, seed):
    tag = tag_of(c, seed)
    tdir, geom = f'{OUT}/{tag}', geometry(c)
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfm3_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        md5sum scripts/train/sudoku/hbfm.sh scripts/sample/sudoku/hbfm.sh algo.py
        LR={c['lr']} SEED={seed} {geom} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS={PER_GPU_BS} GLOBAL_BATCH={GLOBAL_BATCH} \\
            NUM_WORKERS=4 MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
            bash scripts/train/sudoku/hbfm.sh
        echo "[$(date)] EVAL {tag}"
        SEED={seed} {geom} GLOBAL_BATCH={GLOBAL_BATCH} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/hbfm.sh
        echo "[$(date)] DONE {tag}"
        ''')


def anchor_tmax_body(tmaxes):
    """Eval-only horizon control: the finished round-2 anchor cells re-sampled at t_max 1.0 / 2.3."""
    geom = geometry(cell())
    lines = []
    for seed in (1, 2, 3):
        src = f'{OUT}/lr-3e-4_ada-0_k-0.5_seed-{seed}_rate-3'
        for tmax in tmaxes:
            lines.append(
                f'echo "[$(date)] EVAL anchor seed {seed} t_max {tmax}"\n'
                f'SEED={seed} {geom.replace("T_MAX=6", f"T_MAX={tmax:g}")} GLOBAL_BATCH={GLOBAL_BATCH} \\\n'
                f'    CKPT_PATH={src}/checkpoints/last.ckpt OUTPUT_DIR={src}/eval_tm-{tmax:g} DEVICES=1 \\\n'
                f'    bash scripts/sample/sudoku/hbfm.sh')
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfm3tm_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        ''') + '\n'.join(lines) + '\necho "[$(date)] DONE anchor-tmax"\n'


def cell_tmax_body(cells_tmaxes):
    """Eval-only: re-sample finished round-3 cells at extra horizons, into `<cell>/eval_tm-<t>/`.
    The anchor's horizon control showed t_max 6 -> 12 is worth +1.4 +- 0.35 points at no training cost,
    so a winner should also be reported at its own best horizon -- but a bounded cell may not go past
    its own `ub` (`_validate_configuration`), which is why comparisons BETWEEN cells stay at matched
    t_max."""
    head = textwrap.dedent(f"""\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfm3ct_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        """)
    lines = []
    for c, seed, tmax in cells_tmaxes:
        tag = tag_of(c, seed)
        tdir = f'{OUT}/{tag}'
        lines.append(
            f'echo "[$(date)] EVAL {tag} t_max {tmax:g}"\n'
            f'SEED={seed} {geometry(c, tmax)} GLOBAL_BATCH={GLOBAL_BATCH} \\\n'
            f'    CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval_tm-{tmax:g} DEVICES=1 \\\n'
            f'    bash scripts/sample/sudoku/hbfm.sh')
    return head + '\n'.join(lines) + '\necho "[$(date)] DONE cell-tmax"\n'


def submit(jobname, body, dry, hours=24):
    if dry:
        print(f'  [dry] {jobname}')
        return
    slurm = Slurm(job_name=jobname, partition=PARTITION,
                  gres=GRES,
                  exclude=EXCLUDE,
                  ntasks=1, cpus_per_task=4, mem='28G', time=f'{hours}:00:00',
                  output=f'{LOGS}/{jobname}_%j.log')
    print(f'  submitted {jobname}: job {slurm.sbatch(body)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default='', help='substring filter on the cell tag')
    ap.add_argument('--confirm', action='store_true', help='submit seeds 2,3 of the CONFIRM cells')
    ap.add_argument('--cell-tmax', default=None,
                    help='eval-only: re-sample finished round-3 cells (filter with --only) at the '
                         'comma-listed physical t_max; a bounded cell skips any t_max > its ub')
    ap.add_argument('--anchor-tmax', nargs='?', const='0.5,1.0,2.3', default=None,
                    help='eval-only sampler-horizon control on the round-2 anchor checkpoints: '
                         'a comma list of physical t_max (default 0.5,1.0,2.3; use 4,9,12 for the long arm)')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    if args.cell_tmax:
        tmaxes = [float(t) for t in args.cell_tmax.split(',')]
        todo, skipped = [], []
        for c in STAGE1:
            for seed in (1, 2, 3):
                tag = tag_of(c, seed)
                if args.only and args.only not in tag:
                    continue
                # require the cell's OWN eval to exist: `last.ckpt` is rewritten every CKPT_EVERY
                # steps, so keying on it would silently evaluate a half-trained checkpoint
                if not os.path.exists(f'{OUT}/{tag}/eval/results.json'):
                    continue
                for t in tmaxes:
                    if c['ub'] is not None and t > c['ub']:
                        skipped.append(f'{tag} @ {t:g} (> ub {c["ub"]:g})')
                    elif os.path.exists(f'{OUT}/{tag}/eval_tm-{t:g}/results.json'):
                        skipped.append(f'{tag} @ {t:g} (done)')
                    else:
                        todo.append((c, seed, t))
        print(f'cell-tmax: {len(todo)} evals to run, {len(skipped)} skipped')
        for x in skipped[:8]:
            print('  skip', x)
        if not todo:
            return
        submit('hbfm_sud3_cell-tmax', cell_tmax_body(todo), args.dry_run,
               hours=max(2, 1 + len(todo) // 4))
        return

    if args.anchor_tmax:
        tmaxes = [float(t) for t in args.anchor_tmax.split(',')]
        name = 'hbfm_sud3_anchor-tmax-' + '-'.join(f'{t:g}' for t in tmaxes)
        submit(name, anchor_tmax_body(tmaxes), args.dry_run, hours=6)
        if args.dry_run:
            print('\n--- body ---\n' + anchor_tmax_body(tmaxes))
        return

    cells = [(c, 1) for c in STAGE1]
    if args.confirm:
        cells = [(c, s) for c in CONFIRM for s in (2, 3)]
    cells = [(c, s) for c, s in cells if args.only in tag_of(c, s)]
    print(f'round 3: {len(cells)} cells')
    active = set() if args.dry_run else active_jobnames()
    n_sub = n_skip = 0
    for c, seed in cells:
        tag = tag_of(c, seed)
        jobname = f'hbfm_sud3_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        submit(jobname, job_body(c, seed), args.dry_run)
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')
    if args.dry_run and cells:
        print('\n--- example body ---\n' + job_body(*cells[0]))


if __name__ == '__main__':
    main()
