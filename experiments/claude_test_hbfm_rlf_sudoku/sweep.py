#!/usr/bin/env python
"""claude_test_hbfm_rlf_sudoku — HBFM with random Laplacian features (HyLa, "RLF") of the input state vs the plain
Poincaré-coordinate input, on hard Sudoku.

Spec: setup.md ("copy claude_test_hbfm_sudoku/setup.md, pick a good config, re-train with RLF and see whether RLF is
consistently better than without"). Design and decision rules: EXPERIMENT.md. The anchor is that study's incumbent:
mode exp, physical time_exp_rate 3 (unit rate 6), K -0.5, (H^3_K)^3 = model.embed_dim 9 lifted by in_proj, log-linear
noise, LR 3e-4, 20k steps, batch 256, bf16, EMA 0.9999, sampler t_max 6 (unit 3), 180 steps, exact velocity,
top_k_velocity -1, greedy last step -- for BOTH forward types (setup.md lists naive and horosphere).
The RLF knobs (models/hyperbolic_dit.py `HyLaFeatures`; scripts' HYLA_*) are the arms:
  none  -- the incumbent (Poincaré coordinates -> in_proj)
  pure  -- HyLa, 768 features (256 per factor), lambda ~ N(0, 1), no cap: the paper's map verbatim. CONTROL: the
           sudoku prompt is pinned at the boundary (dimensionless radial s ~ 35), where every HyLa feature is ~0, so
           the trunk cannot see the clues; expected to fail (one seed)
  cap5  -- HyLa evaluated at min(s, 5): a finite pin radius (never binds on the trained states, s < ~2)
  cat   -- HyLa (no cap) concatenated to the coordinates: in_proj sees 9 + 768 channels
Cells (arm, forward_type, LR, scale, seed): main grid {none, cap5, cat} x {horo, naive} x LR 3e-4 x scale 1 x seeds
1-3 (18 cells); controls at seed 1 / horo: pure (1), LR 5e-4 and 1e-3 for cap5 and cat (4), scale 0.5 / 2 for cap5 (2).

ORCHESTRATION ONLY -- each cell calls scripts/train/sudoku/hbfm.sh then scripts/sample/sudoku/hbfm.sh (2000-board
sudoku_eval). Idempotent + resumable: a cell with eval/results.json or a queued job of the same name is skipped;
resubmitting the same OUTPUT_DIR resumes from checkpoints/last.ckpt.

Usage:  python sweep.py [--dry-run] [--only SUBSTR]      then  python report.py  ->  RESULTS.md
"""
import argparse
import os
import subprocess
import textwrap

from simple_slurm import Slurm

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm'
ENVBIN = '/home/sc3379/anaconda3/envs/sfm/bin'
EXP = f'{REPO}/experiments/claude_test_hbfm_rlf_sudoku'
LOGS = f'{EXP}/logs'
OUT = f'{REPO}/outputs/claude_test_hbfm_rlf_sudoku'
JOB = 'hbfm_rlf_sud'

MAX_STEPS = 20000
CKPT_EVERY = 5000
PER_GPU_BS = 256
GLOBAL_BATCH = 256
PARTITION = 'thickstun,desa'
EXCLUDE = 'desa-compute-01'      # 2080 Ti (11 GB)
HYLA_DIM = 768                   # 256 features per factor of H^3
FWD_TAG = {'horosphere': 'horo', 'naive': 'naive'}

# arm -> the HYLA_* knobs of scripts/{train,sample}/sudoku/hbfm.sh (HYLA_SCALE is a cell field)
ARMS = {
    'none': 'HYLA_DIM=null',
    'pure': f'HYLA_DIM={HYLA_DIM}',
    'cap5': f'HYLA_DIM={HYLA_DIM} HYLA_RADIUS_CAP=5.0',
    'cat': f'HYLA_DIM={HYLA_DIM} HYLA_CONCAT_STATE=true',
    # cap5f: the SAME knobs as cap5, trained after the 2026-10-02 02:00 fix in models/hyperbolic_dit.py (stop-gradient
    # on the direction of states past the cap, i.e. the pinned prompts; see RESULTS.md "Stage 3"). The arm name only
    # marks the code version so the two sets of cells never mix in the tables.
    'cap5f': f'HYLA_DIM={HYLA_DIM} HYLA_RADIUS_CAP=5.0',
    # sphere (stage 4, 2026-10-02 19:30): the hyperbolic-harmonic basis replaces HyLa (models/hyperbolic_dit.py
    # `HarmonicFeatures`, model.sphere_degree = L): per H^3 factor the (L+1)^2 features f_l(s) Y_lm(u), bounded and
    # -> Y_lm(u) at the boundary (the pinned prompt is seen through its direction, no cap), plus the 6 radial channels
    # tanh(s/c), c = 1..32. No HyLa knobs; the cell field `L` fills the degree.
    'sphere': 'HYLA_DIM=null SPHERE_DEGREE={L}',
}


def cell(arm, fwd='horosphere', lr='3e-4', scale='1', seed=1, nm='1', dim=HYLA_DIM, L=4, rc=None, mc=None):
    # nm: HYLA_NORMALIZE (1 = features / sqrt(D), the paper's phi; 0 = unnormalized, the reference code)
    # dim: HYLA_DIM (total features; stage 3 varies it around the stage-2 winner)
    # L: SPHERE_DEGREE (sphere arm only); rc: SPHERE_RADIAL_SCALES as a hydra list string, None = the default ladder
    # mc: SPHERE_MC_SAMPLES (sphere arm only), the Monte-Carlo Poisson-kernel features; None = the closed form
    return dict(arm=arm, fwd=fwd, lr=lr, scale=scale, seed=seed, nm=nm, dim=dim, L=L, rc=rc, mc=mc)


def cells():
    main = [cell(arm, fwd, seed=s) for fwd in ('horosphere', 'naive')
            for arm in ('none', 'cap5', 'cat') for s in (1, 2, 3)]
    controls = ([cell('pure')]
                + [cell(arm, lr=lr) for arm in ('cap5', 'cat') for lr in ('5e-4', '1e-3')]
                + [cell('cap5', scale=sc) for sc in ('0.5', '2')])
    # Follow-ups under EXPERIMENT.md's screen rule (seeds 2-3 when a seed-1 screen beats its LR-3e-4 counterpart by
    # > 8 pt): 2026-10-01 03:20, cap5/horo LR 5e-4 scored 63.8 vs 55.0 at LR 3e-4 (seed 1).
    confirm = [cell('cap5', lr='5e-4', seed=s) for s in (2, 3)]
    # Stage 2 (2026-10-01 21:45): HyLa hyperparameters per the reference code (github.com/ydtydr/HyLa,
    # hype/models.py): its angle is Lambdas/2 * log(PoissonKernel) with tuned lambda_scale 0.01 / 0.1 / 0.5, i.e.
    # our hyla_scale 0.005 / 0.05 / 0.25 (20-200x smaller than the 1.0 of stage 1), and its features carry no
    # 1/sqrt(D). Grid: scale x normalize, horosphere, LR 3e-4, 2 seeds each (RLF cells are seed-fragile).
    screen = [cell('cap5', scale=sc, nm=nm, seed=sd)
              for sc in ('0.005', '0.05', '0.25') for nm in ('1', '0') for sd in (1, 2)]
    # Stage 3 (2026-10-02 02:00): the pin-gradient fix, re-running (a) the unstable LR-5e-4 config, (b) the stage-1
    # main config, (c) the stage-2 best (scale 0.25, normalized: 68.0 on seed 1), 3 seeds each, horosphere.
    fix = ([cell('cap5f', lr='5e-4', seed=sd) for sd in (1, 2, 3)]
           + [cell('cap5f', seed=sd) for sd in (1, 2, 3)]
           + [cell('cap5f', scale='0.25', seed=sd) for sd in (1, 2, 3)]
           # 03:20: stage 2 complete, scale 0.25 unnormalized = 62.8 ± 1.6 (both seeds >= the baseline mean) -> post-fix
           # seeds of that config under both readouts
           + [cell('cap5f', scale='0.25', nm='0', seed=sd) for sd in (1, 2, 3)]
           + [cell('cap5f', fwd='naive', scale='0.25', nm='0', seed=sd) for sd in (1, 2, 3)])
    # Stage 4 (2026-10-02 19:30): the sphere map, L in {2, 4, 6} x both readouts x 3 seeds at the anchor LR (the design
    # note's plan; the baseline `none` and the no-cap `pure` control are the stage-1 cells).
    sphere = [cell('sphere', fwd=fwd, L=L, seed=sd)           # seed-major: an early read across L and readouts
              for sd in (1, 2, 3) for L in (2, 4, 6) for fwd in ('horosphere', 'naive')]
    # Stage 5 (2026-10-04 07:30, user: "hyperparameter tuning for the sphere features, try to beat the coordinates"):
    # the sphere knobs sudoku can judge (stability / optimization; L itself was flat at 2-6 on 12 words): higher degree
    # L = 8 (both readouts) and 12 (horosphere), LR 5e-4 at L = 4 (both readouts), and a single radial channel tanh(s/2)
    # (= the coordinates' |z|/R) instead of the 6-channel ladder at L = 4 (horosphere). 3 seeds each.
    tune = ([cell('sphere', fwd=fwd, L=8, seed=sd) for sd in (1, 2, 3) for fwd in ('horosphere', 'naive')]
            + [cell('sphere', L=12, seed=sd) for sd in (1, 2, 3)]
            + [cell('sphere', fwd=fwd, L=4, lr='5e-4', seed=sd) for sd in (1, 2, 3) for fwd in ('horosphere', 'naive')]
            + [cell('sphere', L=4, rc='[2]', seed=sd) for sd in (1, 2, 3)])
    # Stage 6 (2026-10-05 11:15): stage 5 found one lever, the trunk LR (sphere L = 4, naive, LR 5e-4: 64.6 ± 2.1 vs
    # `none` 60.1 ± 2.4 at LR 3e-4), while higher degree and a single radial channel hurt. (a) the missing control: the
    # coordinates at the same LRs (naive was never LR-screened; horosphere LR 1e-3 was, in claude_test_hbfm_sudoku);
    # (b) push the lever: sphere L = 4 at LR 1e-3, and L = 2 (best horosphere degree) at LR 5e-4. 3 seeds each.
    lr = ([cell('none', fwd=fwd, lr='5e-4', seed=sd) for sd in (1, 2, 3) for fwd in ('naive', 'horosphere')]
          + [cell('none', fwd='naive', lr='1e-3', seed=sd) for sd in (1, 2, 3)]
          + [cell('sphere', fwd=fwd, L=4, lr='1e-3', seed=sd) for sd in (1, 2, 3) for fwd in ('naive', 'horosphere')]
          + [cell('sphere', fwd=fwd, L=2, lr='5e-4', seed=sd) for sd in (1, 2, 3) for fwd in ('naive', 'horosphere')])
    # Stage 7 (2026-10-18, user: "Train the IS spherical harmonic on sudoku hard"): model.sphere_mc_samples = M, the
    # harmonics' value replaced by the mean of Y_lm(w) over M fresh boundary angles w ~ P(z, .) (the Poisson kernel as
    # its own proposal, unit weights; closed-form gradient), in training AND sampling. Anchor = the stage-5/6 winner
    # (sphere L = 4, naive, LR 5e-4: 64.6 ± 2.1); M = 4 and 16, 3 seeds each. Each cell is also sampled with the exact
    # closed form (eval_exact/) to separate the training effect from the sampling noise.
    mc = [cell('sphere', fwd='naive', L=4, lr='5e-4', mc=M, seed=sd) for sd in (1, 2, 3) for M in (4, 16)]
    return main + controls + confirm + screen + fix + sphere + tune + lr + mc


def tag_of(c):
    if c['arm'] == 'sphere':
        rc = '' if c.get('rc') is None else f"_rc-{c['rc'].strip('[]')}"
        mc = '' if c.get('mc') is None else f"_mc-{c['mc']}"
        return f"sphere-L{c['L']}_fwd-{FWD_TAG[c['fwd']]}_lr-{c['lr']}{rc}{mc}_seed-{c['seed']}"
    nm = '' if c.get('nm', '1') == '1' else '_nm-0'   # stage-1 tags unchanged
    dim = '' if c.get('dim', HYLA_DIM) == HYLA_DIM else f"_d{c['dim']}"
    return f"hyla-{c['arm']}_fwd-{FWD_TAG[c['fwd']]}_lr-{c['lr']}_sc-{c['scale']}{nm}{dim}_seed-{c['seed']}"


def knobs(c):
    # The anchor geometry / proposal / horizon of claude_test_hbfm_sudoku's incumbent, plus the arm's HyLa knobs.
    # Every knob goes to the SAMPLER too: the architecture (HYLA_*), the readout (FORWARD_TYPE) and t_max.
    arm_knobs = ARMS[c['arm']].replace(f'HYLA_DIM={HYLA_DIM}', f"HYLA_DIM={c.get('dim', HYLA_DIM)}").format(L=c.get('L'))
    if c.get('rc') is not None:
        arm_knobs += f" SPHERE_RADIAL_SCALES='{c['rc']}'"
    if c.get('mc') is not None:
        arm_knobs += f" SPHERE_MC_SAMPLES={c['mc']}"
    return ("EMBED_DIM=9 PROD_FACTOR_DIM='[3,3,3]' PROD_FACTOR_CURV='[-0.5,-0.5,-0.5]' GAUSS_CURV=-0.5 "
            "PROPOSAL_RATE=3 T_MAX=6 TIME_CONVERSION_MODE=exp TIME_RANGE_UPPER_BOUND=1.0 "
            f"FORWARD_TYPE={c['fwd']} {arm_knobs} HYLA_SCALE={c['scale']} "
            f"HYLA_NORMALIZE={'true' if c.get('nm', '1') == '1' else 'false'}")


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


def job_body(c):
    tag = tag_of(c)
    tdir = f'{OUT}/{tag}'
    return textwrap.dedent(f'''\
        set -e
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline
        export TMPDIR=/home/sc3379/tmp/hbfmrlf_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={ENVBIN}:$PATH
        cd {REPO}
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        md5sum scripts/train/sudoku/hbfm.sh scripts/sample/sudoku/hbfm.sh algo.py models/hyperbolic_dit.py
        LR={c['lr']} SEED={c['seed']} {knobs(c)} \\
            OUTPUT_DIR={tdir} DEVICES=1 PER_GPU_BS={PER_GPU_BS} GLOBAL_BATCH={GLOBAL_BATCH} \\
            NUM_WORKERS=4 MAX_STEPS={MAX_STEPS} CKPT_EVERY={CKPT_EVERY} \\
            {run('scripts/train/sudoku/hbfm.sh')}
        echo "[$(date)] EVAL {tag}"
        SEED={c['seed']} {knobs(c)} GLOBAL_BATCH={GLOBAL_BATCH} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            {run('scripts/sample/sudoku/hbfm.sh')}
        ''') + (textwrap.dedent(f'''\
        echo "[$(date)] EVAL_EXACT {tag}"
        SEED={c['seed']} {knobs(c)} SPHERE_MC_SAMPLES=null GLOBAL_BATCH={GLOBAL_BATCH} \\
            CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval_exact DEVICES=1 \\
            {run('scripts/sample/sudoku/hbfm.sh')}
        ''') if c.get('mc') is not None else '') + f'echo "[$(date)] DONE {tag}"\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default='', help='substring filter on the cell tag')
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    grid = [c for c in cells() if args.only in tag_of(c)]
    print(f'claude_test_hbfm_rlf_sudoku: {len(grid)} cells')
    if args.dry_run:
        for c in grid:
            print('  ' + tag_of(c))
        if grid:
            print('\n--- example body ---\n' + job_body(grid[0]))
        return
    active = active_jobnames()
    n_sub = n_skip = 0
    for c in grid:
        tag = tag_of(c)
        jobname = f'{JOB}_{tag}'
        if os.path.exists(f'{OUT}/{tag}/eval/results.json') or jobname in active:
            n_skip += 1
            continue
        slurm = Slurm(job_name=jobname, partition=PARTITION, exclude=EXCLUDE, gres='gpu:1',
                      ntasks=1, cpus_per_task=8, mem='64G', time='12:00:00',
                      output=f'{LOGS}/{tag}_%j.log')
        jid = slurm.sbatch(job_body(c))
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
