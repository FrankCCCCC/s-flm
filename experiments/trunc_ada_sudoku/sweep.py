#!/usr/bin/env python
"""sweep.py — EFLM/HFLM truncated+adaptive implementation check on Sudoku.

Trains the new *_truncated_adaptive.sh scripts (branch claude/ada_sched:
geometry-specific truncation bounds + fixed adaptive refit trigger) at the
best hyperparameters from experiments/hflm_curv_init_lr_sudoku, on all three
difficulties, and evals with the same protocol as that sweep (sudoku_eval,
180 steps, exact velocity, greedy last, top_k_velocity=-1) so the naive
anchors in its all_results.csv / bl_* runs are directly comparable.

Grid (arms x difficulty x seed):
  eflm_ta      : E-FLM trunc+ada, ngpt init, lr 3e-4, ALPHA_MAX=0.767
                 (= alpha_star_euclidean(12))
  hflm_ta_best : H-FLM trunc+ada at the sweep-best cell K=-0.5, init=custom
                 std 0.01, lr 3e-4; ALPHA_MAX=0.907
                 (= alpha_star_numeric for that geometry, see
                 alpha_star_numeric.py — the shipped K=-1 bound does not
                 apply at c*rho1 ~ 0.16)
  hflm_ta_k1   : H-FLM trunc+ada at the script defaults K=-1.0,
                 init=hyperbolic std 0.3, lr 3e-4; ALPHA_MAX=0.624
                 (= alpha_star_hyperbolic(12, 512) — the arm where the
                 truncation window is widest/most meaningful)

Correctness bar (see EXPERIMENT.md): no arm collapses (the old sphere-bound
0.093 collapsed HFLM to ~12%/0%), and each arm lands in the ballpark of (or
above) its naive anchor.

Usage:  python experiments/trunc_ada_sudoku/sweep.py
            [--difficulties easy medium hard] [--seeds 1] [--arms ...]
            [--dry-run]
Idempotent: skips a cell whose eval/results.json exists or whose job name is
already in squeue; resubmitting auto-resumes from last.ckpt.
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'

# arm -> (train script stem, sample script stem, extra env for BOTH train+eval)
ARMS = {
    'eflm_ta': ('eflm_truncated_adaptive', 'eflm_truncated_adaptive',
                {'ALPHA_MAX': '0.767'}),
    'hflm_ta_best': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                     {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                      'INIT_STD': '0.01', 'LR': '3e-4',
                      'ALPHA_MAX': '0.907', 'TOPK_VELOCITY': '-1'}),
    'hflm_ta_k1': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                   {'GAUSS_CURV': '-1.0', 'INIT': 'hyperbolic',
                    'LR': '3e-4', 'ALPHA_MAX': '0.624',
                    'TOPK_VELOCITY': '-1'}),
    # ── round 2: tuning arms (see EXPERIMENT.md "Round 2") ──────────
    # decompose the hflm_ta_best medium regression: truncation alone
    'hflm_to_best': ('hflm_truncated', 'hflm_truncated',
                     {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                      'INIT_STD': '0.01', 'LR': '3e-4',
                      'ALPHA_MAX': '0.907', 'TOPK_VELOCITY': '-1'}),
    # ... adaptive alone (no truncation)
    'hflm_ao_best': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                     {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                      'INIT_STD': '0.01', 'LR': '3e-4',
                      'ALPHA_MAX': 'null', 'TOPK_VELOCITY': '-1'}),
    # ... adaptive over-concentration fix: raise the uniform floor so the
    # high-noise (solve-from-clues) region keeps training mass
    'hflm_ta_umix03': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                       {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                        'INIT_STD': '0.01', 'LR': '3e-4',
                        'ALPHA_MAX': '0.907', 'UNIFORM_MIX': '0.3',
                        'TOPK_VELOCITY': '-1'}),
    # ... flatter curvature widens the informative band (transition width
    # ~ 1/(cD)); alpha* recomputed numerically for K=-0.25, c0.01
    'hflm_ta_k25': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                    {'GAUSS_CURV': '-0.25', 'INIT': 'custom',
                     'INIT_STD': '0.01', 'LR': '3e-4',
                     'ALPHA_MAX': '0.894', 'TOPK_VELOCITY': '-1'}),
    # ── round 3: EMPIRICAL truncation bound (loss-geometry-derived) ──
    # The measured L(t) profile for this exact config (loss_geometry_vis
    # sudoku_hard K0.5: c0.01@3e-4) shows loss ~= 0 for alpha > ~0.34 at
    # 5K steps (band shrinks to ~0.2 by 20K): the transformer + clean
    # prompt cells collapse the posterior far earlier than the
    # single-token NN model behind alpha_star_* predicts (0.907).
    # ALPHA_MAX=0.35 covers the band across all training phases.
    'hflm_ta_e35': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                    {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                     'INIT_STD': '0.01', 'LR': '3e-4',
                     'ALPHA_MAX': '0.35', 'TOPK_VELOCITY': '-1'}),
    'hflm_to_e35': ('hflm_truncated', 'hflm_truncated',
                    {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                     'INIT_STD': '0.01', 'LR': '3e-4',
                     'ALPHA_MAX': '0.35', 'TOPK_VELOCITY': '-1'}),
    # aggressive: late-training band only (risks the early-training band)
    'hflm_ta_e20': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                    {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                     'INIT_STD': '0.01', 'LR': '3e-4',
                     'ALPHA_MAX': '0.20', 'TOPK_VELOCITY': '-1'}),
    # ── round 4: flatten further — K=-0.25 flipped trunc+ada to +15 on
    # hard; probe K=-0.1 bracketing toward EFLM (K→0, +26 hard). No naive
    # anchor exists at K=-0.1 (sweep grid ended at -0.25); compare against
    # ta_k25 and eflm_ta directly. alpha* numeric = 0.878.
    'hflm_ta_k10': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                    {'GAUSS_CURV': '-0.1', 'INIT': 'custom',
                     'INIT_STD': '0.01', 'LR': '3e-4',
                     'ALPHA_MAX': '0.878', 'TOPK_VELOCITY': '-1'}),
    # ── round 5: compose the working pieces to surpass the global naive
    # best (81.1/46.2). K=-0.3 is the naive medium PEAK (83.2 best cell)
    # and sits between the naive optimum (-0.5) and the trunc+ada
    # optimum (-0.25). PLOT_PROFILE on the lr3 arm dumps the fitted
    # profile + adapted schedule per refit (first direct look at what
    # adaptive does on H^d).
    'hflm_ta_k30_lr3': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                        {'GAUSS_CURV': '-0.3', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '3e-4',
                         'ALPHA_MAX': '0.897', 'TOPK_VELOCITY': '-1',
                         'PLOT_PROFILE': 'true'}),
    'hflm_ta_k30_lr5': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                        {'GAUSS_CURV': '-0.3', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '5e-4',
                         'ALPHA_MAX': '0.897', 'TOPK_VELOCITY': '-1'}),
    # umix composed with the winning curvature
    'hflm_ta_k25_umix03': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                           {'GAUSS_CURV': '-0.25', 'INIT': 'custom',
                            'INIT_STD': '0.01', 'LR': '3e-4',
                            'ALPHA_MAX': '0.894', 'UNIFORM_MIX': '0.3',
                            'TOPK_VELOCITY': '-1'}),
    # ── round 6: mean-vs-max — can a schedule lift the MEAN toward the
    # lucky-seed level (~58 hard)? |dL/dt|-ada provably compresses the
    # seed distribution toward the mean (K=-0.3 hard: naive 31-50 ->
    # ta 40.6-44.3), so test the two levers that act on basin selection
    # instead: the base schedule SHAPE (cosine^2, never tried on HFLM;
    # oversamples both the near-clean end sampling needs and the
    # solve-from-clues end) and LATE adaptation (naive dynamics for the
    # first 10k steps, concentration after).
    'hflm_cos2_k5': ('hflm_truncated', 'hflm_truncated',
                     {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                      'INIT_STD': '0.01', 'LR': '3e-4',
                      'NOISE': 'cosine-squared', 'ALPHA_MAX': 'null',
                      'TOPK_VELOCITY': '-1'}),
    'hflm_cos2_ta_k5': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                        {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '3e-4',
                         'NOISE': 'cosine-squared-adaptive',
                         'ALPHA_MAX': '0.907', 'TOPK_VELOCITY': '-1'}),
    'hflm_cos2_ta_k25': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                         {'GAUSS_CURV': '-0.25', 'INIT': 'custom',
                          'INIT_STD': '0.01', 'LR': '3e-4',
                          'NOISE': 'cosine-squared-adaptive',
                          'ALPHA_MAX': '0.894', 'TOPK_VELOCITY': '-1'}),
    'hflm_ta_k5_warm10k': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                           {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                            'INIT_STD': '0.01', 'LR': '3e-4',
                            'ALPHA_MAX': '0.907', 'ADAPTIVE_WARMUP': '10000',
                            'TOPK_VELOCITY': '-1'}),
    # ── round 7 (goal: hard ~60, low std) ──────────────────────────
    # (a) PRIOR_COV: the untouched geometry knob (0.25 everywhere so
    # far). Smaller noise radius rho0 shortens the geodesic D and
    # widens the transition band ~1/(cD) — the K=-0.25 mechanism, but
    # stronger (pc=0.05: D 9.1->5.3; pc=0.01: D->2.5). Ada-only (no
    # truncation: model-derived bounds are sampler-unsafe on H^d).
    'hflm_pc05_naive': ('hflm_truncated', 'hflm_truncated',
                        {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '3e-4',
                         'PRIOR_COV': '0.05', 'ALPHA_MAX': 'null',
                         'TOPK_VELOCITY': '-1'}),
    'hflm_pc05_ada': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                      {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                       'INIT_STD': '0.01', 'LR': '3e-4',
                       'PRIOR_COV': '0.05', 'ALPHA_MAX': 'null',
                       'TOPK_VELOCITY': '-1'}),
    'hflm_pc01_naive': ('hflm_truncated', 'hflm_truncated',
                        {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '3e-4',
                         'PRIOR_COV': '0.01', 'ALPHA_MAX': 'null',
                         'TOPK_VELOCITY': '-1'}),
    'hflm_pc01_ada': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                      {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                       'INIT_STD': '0.01', 'LR': '3e-4',
                       'PRIOR_COV': '0.01', 'ALPHA_MAX': 'null',
                       'TOPK_VELOCITY': '-1'}),
    # (b) log-domain importance: ada that sees the exponential ramp
    # (|d log L/dt| ~ const) instead of only the linear-scale band edge
    'hflm_logada_k5': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                       {'GAUSS_CURV': '-0.5', 'INIT': 'custom',
                        'INIT_STD': '0.01', 'LR': '3e-4',
                        'ALPHA_MAX': 'null', 'LOG_IMPORTANCE': 'true',
                        'TOPK_VELOCITY': '-1'}),
    # (c) 2x training on the best ta arm: loss-geometry curves were
    # still dropping at 20k everywhere
    'hflm_ta_k25_40k': ('hflm_truncated_adaptive', 'hflm_truncated_adaptive',
                        {'GAUSS_CURV': '-0.25', 'INIT': 'custom',
                         'INIT_STD': '0.01', 'LR': '3e-4',
                         'ALPHA_MAX': '0.894', 'MAX_STEPS': '40000',
                         'TOPK_VELOCITY': '-1'}),
}

DIFFICULTIES = ['easy', 'medium', 'hard']


def tag_of(arm, difficulty, seed):
    return f'tas_{arm}_d-{difficulty}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(arm, tdir, difficulty, seed):
    train, sample, extra = ARMS[arm]
    ev = ''.join(f'{k}={v} ' for k, v in extra.items())
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {arm} on $(hostname)"
        {ev}DIFFICULTY={difficulty} SEED={seed} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{train}.sh
        echo "[$(date)] EVAL"
        {ev}DIFFICULTY={difficulty} SEED={seed} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{sample}.sh
        # correctness bar #3: persist the adaptive-schedule state (refit_count /
        # has_schedule / adapted alpha range) before the checkpoint is deleted
        python - <<'PYEOF'
        import json, torch
        sd = torch.load('{tdir}/checkpoints/last.ckpt',
                        map_location='cpu', weights_only=False)['state_dict']
        av = sd.get('noise.alpha_vals')  # absent for non-adaptive (trunc-only) runs
        json.dump({{'has_schedule': (bool(sd['noise.has_schedule'])
                                     if 'noise.has_schedule' in sd else None),
                    'refit_count': (int(sd['noise.refit_count'])
                                    if 'noise.refit_count' in sd else None),
                    'alpha_vals_min': (float(av.min()) if av is not None else None),
                    'alpha_vals_max': (float(av.max()) if av is not None else None)}},
                  open('{tdir}/eval/noise_state.json', 'w'), indent=1)
        PYEOF
        # checkpoints are transient bulk (~1.8G/cell): once the eval deliverable
        # exists, drop them to keep /share within quota
        if [ -f {tdir}/eval/results.json ] && [ -f {tdir}/eval/noise_state.json ]; then
            rm -rf {tdir}/checkpoints && echo "[$(date)] checkpoints cleaned"
        fi
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--difficulties', nargs='+', default=DIFFICULTIES,
                    choices=DIFFICULTIES)
    ap.add_argument('--seeds', nargs='+', default=['1'])
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    exp = f'{REPO}/experiments/trunc_ada_sudoku'
    logs = f'{exp}/logs'
    out = f'{REPO}/outputs/trunc_ada_sudoku'
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(args.arms, args.difficulties, args.seeds))
    print(f'trunc_ada_sudoku: {len(cells)} cells '
          f'({len(args.arms)} arm x {len(args.difficulties)} difficulty x '
          f'{len(args.seeds)} seed)')
    if args.dry_run:
        for arm, diff, seed in cells:
            print(f'  {tag_of(arm, diff, seed)}')
        arm, diff, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(arm, f'{out}/{tag_of(arm, diff, seed)}', diff, seed))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for arm, diff, seed in cells:
        tag = tag_of(arm, diff, seed)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition='thickstun,desa', exclude='desa-compute-01',
                      gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                      time='06:00:00')
        jid = slurm.sbatch(job_body(arm, f'{out}/{tag}', diff, seed),
                           sbatch_cmd='sbatch --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
