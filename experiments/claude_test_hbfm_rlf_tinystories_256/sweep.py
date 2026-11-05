#!/usr/bin/env python
"""claude_test_hbfm_rlf_tinystories_256 — HBFM with random Laplacian features (HyLa, "RLF") of the input state vs the
plain Poincaré-coordinate input, on TinyStories seq 256.

Spec: setup.md ("copy the settings, pick a good config, re-train with RLF and see whether RLF is consistently better").
Design and decision rules: EXPERIMENT.md. The anchor is claude_test_hbfm_tinystories_256's spec cell -- mode exp,
physical time_exp_rate 3 (unit 6), K -0.5, (H^3_K)^16 = model.embed_dim 48, ub 1.0, log-linear noise, LR 3e-4, 30k
steps, global batch 512 on 4 GPUs, bf16, EMA 0.9999 -- whose naive seeds 1-3 and horosphere seed 1 already exist there
(outputs/claude_test_hbfm_tinystories_256/mode-exp_rate-3_k-0.5_ub-1.0[_fwd-horo]_seed-N) and are REUSED as the
`none` arm; this project trains the missing horosphere seeds 2-3 and the RLF arm:
  cap5 -- HyLa, 1536 features (96 per factor of H^3), lambda ~ N(0, 1), evaluated at min(s, 5) of the dimensionless
          radial (models/hyperbolic_dit.py HyLaFeatures; the cap never binds on the trained states, s < ~2, and keeps
          the features readable for tokens that drift past s ~ 6 during sampling at the default horizon)
Cells (arm, forward_type, seed): cap5 x {naive, horo} x seeds 1-3 (6) + none x horo x seeds 2-3 (2).

Each cell trains, then evaluates the 30k checkpoint at the default horizon (T_MAX 3.0 = unit 1.5 -> eval/) and at the
matched horizon that won the previous study (T_MAX 0.5 -> eval_tmax0.5/): ppl_eval (importance-weighted denoising-CE
bound) + sample_eval (64 samples, 180 steps, exact velocity, top_k_velocity 1, greedy last step, gpt2-large GenPPL).
`--eval-step N [--t-max T]` submits 1-GPU reads of keep-N.ckpt (-> eval_stepN[_tmaxT]/) for early comparisons with the
previous study's per-step table. `--paired-eval [--t-max T]` re-reads the FINAL checkpoints of every cell of both arms
(including the reused baselines) on ONE node (EVAL_NODELIST / EVAL_PARTITION env) into paired/<tag>/eval_tmax<T>/, since
GenPPL re-reads differ by up to 0.46 across GPU types.

ORCHESTRATION ONLY -- one SLURM job per cell (train then eval, `set -e`); idempotent + resumable (skips cells with
eval_tmax0.5/samples_genppl.json or a queued job; resubmission resumes from checkpoints/last.ckpt).

Usage:  python sweep.py [--dry-run] [--only SUBSTR] [--eval-step N [--t-max T]] [--paired-eval [--t-max T]]
        then  python report.py  ->  RESULTS.md
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/claude_test_hbfm_rlf_tinystories_256'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/claude_test_hbfm_rlf_tinystories_256'
OLD_OUT = f'{REPO}/outputs/claude_test_hbfm_tinystories_256'
JOB = 'hbfm_rlf_ts'

EMBED_DIM = 48            # (H^3)^16
HYLA_DIM = 1536           # 96 features per factor
T_MAX_DEFAULT = 3.0       # unit 1.5 at K -0.5, the previous study's default horizon (eval/)
T_MAX_MATCHED = 0.5       # the previous study's best horizon (eval_tmax0.5/)
DEVICES = 4
MAX_STEPS = 30000
CKPT_EVERY = 1000
KEEP_EVERY = 5000
GLOBAL_BATCH = 512
LIMIT_VAL_BATCHES = 1.0   # < 1.0 only for smoke tests
NUM_SAMPLE_BATCHES = 4    # x EVAL_BS 16 = 64 samples
FWD_TAG = {'horosphere': 'horo', 'naive': 'naive'}
ARMS = {'none': 'HYLA_DIM=null', 'cap5': f'HYLA_DIM={HYLA_DIM} HYLA_SCALE=1 HYLA_RADIUS_CAP=5.0'}
# best: the sudoku stage-2 winner (scale 0.25, unnormalized = reference-code features; 62.8 ± 1.6 vs 61.7 ± 0.4 there),
# trained after the 2026-10-02 pin-gradient fix (irrelevant here: TinyStories has no pinned prompts).
ARMS['best'] = f'HYLA_DIM={HYLA_DIM} HYLA_SCALE=0.25 HYLA_RADIUS_CAP=5.0 HYLA_NORMALIZE=false'
# sphere (stage 4, 2026-10-03): the hyperbolic-harmonic basis replaces HyLa (models/hyperbolic_dit.py `HarmonicFeatures`,
# model.sphere_degree = 4: per H^3 factor 25 features f_l(s) Y_lm(u) + 6 radial channels tanh(s/c) -> 16 x 31 = 496 inputs).
# Sudoku stage 4 (claude_test_hbfm_rlf_sudoku) is the gate: seed 1 gave +6 pt under horosphere at every L in {2, 4, 6}.
ARMS['sphere-L4'] = 'HYLA_DIM=null SPHERE_DEGREE=4'
# Stage 5 tuning (2026-10-04, user: "hyperparameter tuning for the sphere features, try to beat the coordinates"): the
# degree as a hyperbolic positional encoding (L = 8 -> 87 features / factor = 1392 inputs; L = 12 -> 175 / factor = 2800)
# and the trunk LR (5e-4 at L = 4; the 10k reads of L = 4 suggested slower early progress with 10x more inputs).
# Seed-1 screens at 30k; a config that beats seed 1 of `none` (15.56 / 12.90) by > 1 GenPPL gets seeds 2-3.
ARMS['sphere-L8'] = 'HYLA_DIM=null SPHERE_DEGREE=8'
ARMS['sphere-L12'] = 'HYLA_DIM=null SPHERE_DEGREE=12'
ARMS['sphere-L4-lr5e-4'] = 'HYLA_DIM=null SPHERE_DEGREE=4'
# Stage 6 (2026-10-05): stage 5's only lever was the LR (sphere-L4 at 5e-4: 15.70 / 13.30, bound 1.710 on seed 1 vs
# 16.86 / 13.47, 1.717 at 3e-4 and `none` 15.56 / 12.90, 1.715). The control that makes it a feature claim: the
# coordinates at LR 5e-4 (`none-lr5e-4`); and the lever pushed: sphere-L4 at 1e-3, sphere-L2 at 5e-4.
ARMS['none-lr5e-4'] = 'HYLA_DIM=null'
ARMS['sphere-L4-lr1e-3'] = 'HYLA_DIM=null SPHERE_DEGREE=4'
ARMS['sphere-L2-lr5e-4'] = 'HYLA_DIM=null SPHERE_DEGREE=2'
ARMS['none-lr1e-3'] = 'HYLA_DIM=null'      # stage 6 control for sphere-L4-lr1e-3 (bound 1.702)
ARM_LR = {'sphere-L4-lr5e-4': '5e-4', 'none-lr5e-4': '5e-4', 'sphere-L4-lr1e-3': '1e-3',
          'sphere-L2-lr5e-4': '5e-4', 'none-lr1e-3': '1e-3'}       # training LR per arm; default 3e-4 (the anchor)
# --screen: 5k-step runs of the HyLa hyperparameters the reference code tunes (github.com/ydtydr/HyLa:
# lambda_scale 0.01 / 0.1 / 0.5 with the angle Lambdas/2 * log PsK = our hyla_scale 0.005 / 0.05 / 0.25, and no
# 1/sqrt(D) on the features = HYLA_NORMALIZE=false), naive readout, seed 1, read at both horizons; compared with
# the baseline's and the scale-1 arm's 5k reads (RESULTS.md). Tags screen5k_sc-{scale}_nm-{0|1}_seed-1.
SCREEN_STEPS = 5000
SCREEN = [(sc, nm) for sc in ('0.005', '0.05', '0.25') for nm in ('1', '0')]

# The reused baselines of claude_test_hbfm_tinystories_256 (same recipe, trained 2026-09-18..21).
OLD_BASELINES = {('none', 'naive', s): f'{OLD_OUT}/mode-exp_rate-3_k-0.5_ub-1.0_seed-{s}' for s in (1, 2, 3)}
OLD_BASELINES[('none', 'horosphere', 1)] = f'{OLD_OUT}/mode-exp_rate-3_k-0.5_ub-1.0_fwd-horo_seed-1'


def cells():
    return ([('cap5', 'naive', s) for s in (1, 2, 3)] + [('cap5', 'horosphere', s) for s in (1, 2, 3)]
            + [('none', 'horosphere', s) for s in (2, 3)]
            + [('best', 'naive', s) for s in (1, 2, 3)]         # stage 3 (2026-10-02 03:30)
            + [('sphere-L4', fwd, s) for s in (1, 2, 3) for fwd in ('naive', 'horosphere')]    # stage 4
            + [(arm, 'naive', 1) for arm in ('sphere-L8', 'sphere-L12', 'sphere-L4-lr5e-4')]    # stage 5 screens
            + [(arm, 'naive', 1) for arm in ('none-lr5e-4', 'sphere-L4-lr1e-3', 'sphere-L2-lr5e-4', 'none-lr1e-3')])   # stage 6


def tag_of(arm, fwd, seed):
    prefix = arm if arm.startswith('sphere') else f'hyla-{arm}'
    return f'{prefix}_fwd-{FWD_TAG[fwd]}_seed-{seed}'


def cell_dir(arm, fwd, seed):
    return OLD_BASELINES.get((arm, fwd, seed), f'{OUT}/{tag_of(arm, fwd, seed)}')


def run(script):
    """`bash -c "$(cat script)" script`: the script's TEXT is read once at invocation. Plain `bash script` reads the
    file incrementally, so a live edit of the shared script during the hours-long python call makes bash resume at a
    stale byte offset and execute a fragment (seen in the smoke test: `PU_BS}: command not found`, exit 127). $0 stays
    the path, so the script's REPO_ROOT (= $(dirname "$0")/../../..) resolves as usual."""
    return f'bash -c "$(cat {script})" {script}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', 'sc3379', '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def preamble(tmp='hbfmrlfts'):
    return textwrap.dedent(f'''\
        set -e
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/{tmp}_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        md5sum scripts/train/tinystories/hbfm.sh scripts/sample/tinystories/hbfm.sh algo.py models/hyperbolic_dit.py samplers.py
        ''')


def knobs(arm, fwd, t_max):
    # The anchor (spec cell of claude_test_hbfm_tinystories_256) plus the arm's HyLa knobs. T_MAX goes to training too
    # (algo._validate_configuration checks it); FORWARD_TYPE and the HYLA_* knobs go to the sampler too (architecture
    # + readout must match the checkpoint).
    return (f"FORWARD_TYPE={fwd} EMBED_DIM={EMBED_DIM} FACTOR_DIM=3 TIME_CONVERSION_MODE=exp PROPOSAL_RATE=3 "
            f"GAUSS_CURV=-0.5 TIME_RANGE_UPPER_BOUND=1.0 T_MAX={t_max:g} READOUT_PRECISION=float32 "
            f"GLOBAL_BATCH={GLOBAL_BATCH} SEQ_LEN=256 {ARMS[arm]}")


def eval_cmd(arm, fwd, t_max, ckpt, out_dir):
    return (f'{knobs(arm, fwd, t_max)} TOPK_VELOCITY=1 STEPS=180 LIMIT_VAL_BATCHES={LIMIT_VAL_BATCHES} '
            f'NUM_SAMPLE_BATCHES={NUM_SAMPLE_BATCHES} \\\n'
            f'    CKPT_PATH={ckpt} OUTPUT_DIR={out_dir} DEVICES=1 \\\n'
            f'    {run("scripts/sample/tinystories/hbfm.sh")}')


def job_body(arm, fwd, seed):
    tag = tag_of(arm, fwd, seed)
    tdir = f'{OUT}/{tag}'
    ckpt = f'{tdir}/checkpoints/last.ckpt'
    return preamble() + textwrap.dedent(f'''\
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        PER_GPU_BS=$([ "$GPU_MEM" -gt 40000 ] && echo 16 || echo 8)   # 48 GB Ada / A6000 vs 24 GB A5000; global batch fixed
        echo "[$(date)] TRAIN {tag} on $(hostname) (GPU ${{GPU_MEM}} MiB -> PER_GPU_BS ${{PER_GPU_BS}})"
        LR={ARM_LR.get(arm, '3e-4')} SEED={seed} {knobs(arm, fwd, T_MAX_DEFAULT)} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP=claude_test_hbfm_rlf_tinystories_256 \\
            DEVICES={DEVICES} PER_GPU_BS=$PER_GPU_BS MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} KEEP_EVERY={KEEP_EVERY} \\
            {run('scripts/train/tinystories/hbfm.sh')}
        echo "[$(date)] EVAL {tag} t_max {T_MAX_DEFAULT:g}"
        {eval_cmd(arm, fwd, T_MAX_DEFAULT, ckpt, f'{tdir}/eval')}
        echo "[$(date)] EVAL {tag} t_max {T_MAX_MATCHED:g}"
        {eval_cmd(arm, fwd, T_MAX_MATCHED, ckpt, f'{tdir}/eval_tmax{T_MAX_MATCHED:g}')}
        echo "[$(date)] DONE {tag}"
        ''')


def screen_tag(sc, nm, seed=1):
    return f'screen{SCREEN_STEPS // 1000}k_sc-{sc}_nm-{nm}_seed-{seed}'


def screen_body(sc, nm, seed=1):
    tag = screen_tag(sc, nm, seed)
    tdir = f'{OUT}/{tag}'
    ckpt = f'{tdir}/checkpoints/last.ckpt'
    arm_knobs = f'HYLA_DIM={HYLA_DIM} HYLA_SCALE={sc} HYLA_RADIUS_CAP=5.0 HYLA_NORMALIZE={"true" if nm == "1" else "false"}'

    def k(tm):
        return knobs('none', 'naive', tm).replace(ARMS['none'], arm_knobs)

    def ev(tm, out):
        return (f'{k(tm)} TOPK_VELOCITY=1 STEPS=180 LIMIT_VAL_BATCHES={LIMIT_VAL_BATCHES} '
                f'NUM_SAMPLE_BATCHES={NUM_SAMPLE_BATCHES} \\\n    CKPT_PATH={ckpt} OUTPUT_DIR={out} DEVICES=1 \\\n'
                f'    {run("scripts/sample/tinystories/hbfm.sh")}')

    return preamble('hbfmrlfsc') + textwrap.dedent(f'''\
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        PER_GPU_BS=$([ "$GPU_MEM" -gt 40000 ] && echo 16 || echo 8)
        echo "[$(date)] TRAIN {tag} on $(hostname) (GPU ${{GPU_MEM}} MiB -> PER_GPU_BS ${{PER_GPU_BS}})"
        LR=3e-4 SEED={seed} {k(T_MAX_DEFAULT)} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP=claude_test_hbfm_rlf_tinystories_256 \\
            DEVICES={DEVICES} PER_GPU_BS=$PER_GPU_BS MAX_STEPS={SCREEN_STEPS} CKPT_EVERY=1000 KEEP_EVERY=0 \\
            {run('scripts/train/tinystories/hbfm.sh')}
        echo "[$(date)] EVAL {tag} t_max {T_MAX_DEFAULT:g}"
        {ev(T_MAX_DEFAULT, f'{tdir}/eval')}
        echo "[$(date)] EVAL {tag} t_max {T_MAX_MATCHED:g}"
        {ev(T_MAX_MATCHED, f'{tdir}/eval_tmax{T_MAX_MATCHED:g}')}
        echo "[$(date)] DONE {tag}"
        ''')



def eval_body(arm, fwd, seed, step, t_max, out_dir):
    tag = tag_of(arm, fwd, seed)
    ckpt = f'{cell_dir(arm, fwd, seed)}/checkpoints/' + ('last.ckpt' if step is None else f'keep-{step}.ckpt')
    return preamble('hbfmrlfev') + textwrap.dedent(f'''\
        echo "[$(date)] EVAL {tag} step {step or "final"} t_max {t_max:g} on $(hostname)"
        {eval_cmd(arm, fwd, t_max, ckpt, out_dir)}
        echo "[$(date)] DONE {tag}"
        ''')


def submit_eval(jobname, body, log):
    pin = {'nodelist': os.environ['EVAL_NODELIST']} if os.environ.get('EVAL_NODELIST') else {}
    slurm = Slurm(job_name=jobname, partition=os.environ.get('EVAL_PARTITION', 'thickstun,desa'), gres='gpu:1',
                  ntasks=1, cpus_per_task=8, mem='48G', time='06:00:00', exclude='desa-compute-01', **pin,
                  output=log)
    return slurm.sbatch(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default='', help='substring filter on the cell tag')
    ap.add_argument('--eval-step', type=int, default=None, help='submit 1-GPU evals of keep-<N>.ckpt instead of training')
    ap.add_argument('--paired-eval', action='store_true',
                    help='1-GPU re-reads of every FINAL checkpoint (both arms, reused baselines included) into paired/')
    ap.add_argument('--t-max', type=float, default=None,
                    help='physical horizon of --eval-step / --paired-eval (default: the 3.0 default horizon)')
    ap.add_argument('--screen', action='store_true', help=f'submit the {SCREEN_STEPS}-step HyLa hyperparameter screen')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    active = set() if args.dry_run else active_jobnames()
    n_sub = n_skip = 0

    if args.screen:
        for sc, nm in SCREEN:
            tag = screen_tag(sc, nm)
            if args.only not in tag:
                continue
            jobname = f'{JOB}_{tag}'
            if os.path.exists(f'{OUT}/{tag}/eval_tmax{T_MAX_MATCHED:g}/samples_genppl.json') or jobname in active:
                n_skip += 1
                continue
            if args.dry_run:
                print(f'  [dry] {jobname}' + (f'\n{screen_body(sc, nm)}' if n_sub == 0 else ''))
            else:
                slurm = Slurm(job_name=jobname, partition='thickstun,desa', exclude='desa-compute-01',
                              gres=f'gpu:{DEVICES}', ntasks=1, cpus_per_task=16, mem='128G', time='1-00:00:00',
                              output=f'{LOGS}/{tag}_%j.log')
                print(f'  submitted {tag}: job {slurm.sbatch(screen_body(sc, nm))}')
            n_sub += 1
        print(f'submitted {n_sub}, skipped {n_skip}')
        return

    if args.eval_step or args.paired_eval:
        t_max = T_MAX_DEFAULT if args.t_max is None else args.t_max
        todo = list(cells()) + (list(OLD_BASELINES) if args.paired_eval else [])
        for arm, fwd, seed in todo:
            tag = tag_of(arm, fwd, seed)
            if args.only not in tag:
                continue
            cdir = cell_dir(arm, fwd, seed)
            if args.paired_eval:
                ready = os.path.exists(f'{cdir}/eval/samples_genppl.json')     # the cell's own final eval exists
                out_dir = f'{OUT}/paired/{tag}/eval_tmax{t_max:g}'
                jobname = f'{JOB}_{tag}_paired{t_max:g}'
            else:
                ready = os.path.exists(f'{cdir}/checkpoints/keep-{args.eval_step}.ckpt')
                out_dir = f'{cdir}/eval_step{args.eval_step}' + ('' if args.t_max is None else f'_tmax{t_max:g}')
                jobname = f'{JOB}_{tag}_{args.eval_step}' + ('' if args.t_max is None else f'_tmax{t_max:g}')
            if not ready or os.path.exists(f'{out_dir}/samples_genppl.json') or jobname in active:
                n_skip += 1
                continue
            body = eval_body(arm, fwd, seed, None if args.paired_eval else args.eval_step, t_max, out_dir)
            if args.dry_run:
                print(f'  [dry] {jobname}\n{body}' if n_sub == 0 else f'  [dry] {jobname}')
            else:
                print(f'  submitted {jobname}: job {submit_eval(jobname, body, f"{LOGS}/{jobname}_%j.log")}')
            n_sub += 1
        print(f'submitted {n_sub}, skipped {n_skip}')
        return

    grid = [c for c in cells() if args.only in tag_of(*c)]
    print(f'claude_test_hbfm_rlf_tinystories_256: {len(grid)} cells')
    if args.dry_run:
        for c in grid:
            print('  ' + tag_of(*c))
        if grid:
            print('\n--- example body ---\n' + job_body(*grid[0]))
        return
    for c in grid:
        tag = tag_of(*c)
        jobname = f'{JOB}_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval_tmax{T_MAX_MATCHED:g}/samples_genppl.json') or jobname in active:
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
