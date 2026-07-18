#!/usr/bin/env python
"""geo_curv_sc_tinystories_256 — {geometry x self-cond} (Part A) + H-FLM {curvature x
init x prior_cov} (Part B) on TinyStories, seq 256.  Spec: setup.md.

58 cells total:

Part A — geometry x self-conditioning (13), small-sphere-dit, init=ngpt:
  S-FLM {naive, ada, trunc, ada+trunc} x SC-On + {naive} x SC-Off  (5)
  E-FLM {naive, ada, trunc, ada+trunc} x SC-{On,Off}    (8)
  Truncation alpha_max (noise_schedules.py, V=50257, dim=768, delta=0.1):
    S-FLM alpha_star_sphere    = 0.1215
    E-FLM alpha_star_euclidean = 0.8402
    ada variants use alpha_max=null (adaptive schedule, no truncation).

Part B — H-FLM curvature study (45), small-hyperbolic-dit, SC-On, rho_max=12:
  init {random(std0.02), c0.01, c0.04} x prior_cov {0.5,0.8,1.0} x
       K {-0.01,-0.1,-0.25,-0.5,-0.75}                  (3 x 3 x 5)

Recipe (both parts): 768/12/12 DiT, 30k steps, global batch 512 (1 GPU x PER_GPU_BS=32,
accum auto=16), seq 256, bf16, EMA 0.9999, AdamW lr 3e-4 wd 0 betas (0.9,0.999) clip 1.0.
Checkpoints every 5k, save_top_k=1 + last; deleted after eval (quota). Eval: exact
velocity, top_k_velocity=1, 180 steps, greedy last -> eval/ppl.json (flow-bound PPL) +
eval/samples_genppl.json (GenPPL + entropy).

ORCHESTRATION ONLY — each cell calls the single-run shared scripts (SEQ_LEN=256 knob):
  Part A: scripts/{train,sample}/tinystories/{sfm,sfm_truncated,sfm_truncated_adaptive,
          eflm,eflm_truncated,eflm_truncated_adaptive}.sh  (SELF_COND / ALPHA_MAX env)
  Part B: scripts/{train,sample}/tinystories/{hlfm,hflm}.sh (GAUSS_CURV/INIT/INIT_STD/
          PRIOR_COV/RHO_MAX/SELF_COND env)
Idempotent + resumable: skip a cell whose eval/samples_genppl.json exists (the LAST eval
artifact, written after ppl.json — so a cell preempted mid-GenPPL is re-run, not skipped)
or whose job name is in squeue; resubmitting the same OUTPUT_DIR auto-resumes from
checkpoints/last.ckpt. Checkpoints are deleted only after samples_genppl.json exists.

Sites (run this script ON the submitting cluster):
  --site unicorn : Part A (13)  [sc3379@unicorn; partition thickstun,desa]
  --site tc      : Part B, K in {-0.01,-0.1,-0.5}  (27) [shengyenc@TinkerCliffs a100+h200]
  --site falcon  : Part B, K in {-0.25,-0.75}      (18) [shengyenc@Falcon l40s+a30]
tc and falcon share the ARC /home filesystem (same repo+outputs), so their K sets are
DISJOINT. rsync ARC->unicorn results before report.py so idempotency sees them.

Usage: python sweep.py --site {unicorn,tc,falcon} [--nice 200] [--dry-run]
                       [--curvatures ...] [--sfm-self-cond {on,both}]
"""
import argparse
import getpass
import os
import subprocess
import textwrap

from simple_slurm import Slurm

SEQ_LEN = 256
CKPT_EVERY = 5000
SAVE_TOPK = 1
PER_GPU_BS = 32
ALPHA_SFM = '0.121'      # alpha_star_sphere(V=50257, dim=768, delta=0.1) ~= 0.1215
ALPHA_EFLM = '0.840'     # alpha_star_euclidean(V=50257, delta=0.1) ~= 0.8402

SITES = {
    'unicorn': dict(
        repo='/share/thickstun/sychou/workspace/research/s-flm',
        envbin='/home/sc3379/anaconda3/envs/sfm/bin',
        study='A',
        partitions=['thickstun,desa'],  # comma-list is one submission on unicorn
        slurm=dict(exclude='desa-compute-01',  # 2080 Ti 11G OOMs at seq256
                   gres='gpu:1', ntasks=1, cpus_per_task=8, mem='48G',
                   time='10-00:00:00'),
    ),
    'tc': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        study='B', curvatures=['-0.01', '-0.1', '-0.5'],
        # ARC rejects multi-partition submissions (per-partition QOS): round-robin
        # single queues, fast-starting (preemptable) first; --requeue + ckpt-5k
        # auto-resume make preemption cost <= ~1 refit window.
        partitions=['h200_preemptable_q', 'a100_preemptable_q',
                    'h200_normal_q', 'a100_normal_q'],
        slurm=dict(account='swan_research_dlm', gres='gpu:1', ntasks=1,
                   cpus_per_task=8, mem='48G', time='7-00:00:00'),
    ),
    'falcon': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        study='B', curvatures=['-0.25', '-0.75'],
        # Falcon has no a100: L40S-48G / A30-24G are the bf16+flash-attn options.
        partitions=['a30_preemptable_q', 'l40s_preemptable_q',
                    'a30_normal_q', 'l40s_normal_q'],
        slurm=dict(account='swan_research_dlm', gres='gpu:1', ntasks=1,
                   cpus_per_task=8, mem='48G', time='7-00:00:00'),
    ),
}

# --- Part A: (variant_tag, script_basename, alpha_max_or_None) ------------------
SFM_VARIANTS = [
    ('naive',    'sfm.sh',                    None),
    ('ada',      'sfm_truncated_adaptive.sh', 'null'),
    ('trunc',    'sfm_truncated.sh',          ALPHA_SFM),
    ('adatrunc', 'sfm_truncated_adaptive.sh', ALPHA_SFM),
]
EFLM_VARIANTS = [
    ('naive',    'eflm.sh',                    None),
    ('ada',      'eflm_truncated_adaptive.sh', 'null'),
    ('trunc',    'eflm_truncated.sh',          ALPHA_EFLM),
    ('adatrunc', 'eflm_truncated_adaptive.sh', ALPHA_EFLM),
]

# --- Part B: (init_tag, INIT, INIT_STD) -----------------------------------------
INITS = [('random', 'random', 'null'),
         ('c0.01', 'custom', '0.01'),
         ('c0.04', 'custom', '0.04')]
PRIOR_COVS = ['0.5', '0.8', '1.0']
RHO_MAX = '12'


def part_a_cells():
    """-> [(tag, train_script, sample_script, fixed_env)].

    setup.md L10-11: S-FLM {naive,ada,trunc,adatrunc} x SC-On (4) + {naive} x SC-Off (1);
    E-FLM {naive,ada,trunc,adatrunc} x {SC-On, SC-Off} (8).  = 13 cells.
    """
    cells = []
    for vtag, script, alpha in SFM_VARIANTS:            # S-FLM x SC-On
        env = 'SELF_COND=true' + (f' ALPHA_MAX={alpha}' if alpha is not None else '')
        cells.append((f'sfm_{vtag}_scon', script, script, env))
    cells.append(('sfm_naive_scoff', 'sfm.sh', 'sfm.sh', 'SELF_COND=false'))  # + {naive} x SC-Off
    for vtag, script, alpha in EFLM_VARIANTS:           # E-FLM x {SC-On, SC-Off}
        for sctag, scval in [('scon', 'true'), ('scoff', 'false')]:
            env = f'SELF_COND={scval}' + (f' ALPHA_MAX={alpha}' if alpha is not None else '')
            cells.append((f'eflm_{vtag}_{sctag}', script, script, env))
    return cells


def part_b_cells(curvatures):
    """-> [(tag, K, INIT, INIT_STD, prior_cov)]."""
    cells = []
    for k in curvatures:
        for itag, init, istd in INITS:
            for pc in PRIOR_COVS:
                cells.append((f'k{k}_i-{itag}_pc{pc}', k, init, istd, pc))
    return cells


_HEAD = '\n'.join([
    'set -e',  # abort the body if train/eval fails -> never clean checkpoints on failure
    'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
    'export SLURM_JOB_NAME=bash',
    'export NCCL_P2P_DISABLE=1',
    'export NCCL_IB_DISABLE=1',
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
    'export WANDB_MODE=offline',
])


def body_a(site, tag, train_s, sample_s, fixed_env, tdir):
    edir = f'{tdir}/eval'
    return textwrap.dedent(f'''\
        {_HEAD}
        export PATH={site['envbin']}:$PATH
        cd {site['repo']}
        if [ -f {edir}/samples_genppl.json ]; then echo "[$(date)] {tag} done elsewhere -> no-op"; exit 0; fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {fixed_env} OUTPUT_DIR={tdir} RUN_NAME=gcsc_{tag}_s256 DEVICES=1 PER_GPU_BS={PER_GPU_BS} \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/{train_s}
        echo "[$(date)] EVAL {tag}"
        {fixed_env} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/{sample_s}
        if [ -f {edir}/samples_genppl.json ]; then rm -rf {tdir}/checkpoints && echo "[$(date)] ckpts cleaned"; fi
        echo "[$(date)] DONE {tag}"
        ''')


def body_b(site, tag, k, init, istd, pc, tdir):
    edir = f'{tdir}/eval'
    init_env = f'INIT={init}' + (f' INIT_STD={istd}' if init == 'custom' else '')
    geo = f'GAUSS_CURV={k} PRIOR_COV={pc} RHO_MAX={RHO_MAX} SELF_COND=true'
    return textwrap.dedent(f'''\
        {_HEAD}
        export PATH={site['envbin']}:$PATH
        cd {site['repo']}
        if [ -f {edir}/samples_genppl.json ]; then echo "[$(date)] {tag} done elsewhere -> no-op"; exit 0; fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {init_env} {geo} OUTPUT_DIR={tdir} RUN_NAME=gcsc_{tag}_s256 DEVICES=1 PER_GPU_BS={PER_GPU_BS} \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/hlfm.sh
        echo "[$(date)] EVAL {tag}"
        {geo} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/hflm.sh
        if [ -f {edir}/samples_genppl.json ]; then rm -rf {tdir}/checkpoints && echo "[$(date)] ckpts cleaned"; fi
        echo "[$(date)] DONE {tag}"
        ''')


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=SITES)
    ap.add_argument('--nice', type=int, default=200)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--curvatures', nargs='+', default=None,
                    help="override the site's Part B K list (keep tc/falcon disjoint)")
    args = ap.parse_args()
    site = SITES[args.site]
    exp = f"{site['repo']}/experiments/geo_curv_sc_tinystories_256"
    logs, out = f'{exp}/logs', f"{site['repo']}/outputs/geo_curv_sc_tinystories_256"
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    if site['study'] == 'A':
        cells = [('A', *c) for c in part_a_cells()]
        label = f'Part A: {len(cells)} geometry x self-cond cells'
    else:
        ks = args.curvatures or site['curvatures']
        cells = [('B', *c) for c in part_b_cells(ks)]
        label = f'Part B: {len(cells)} hflm cells (K={ks})'
    print(f'geo_curv_sc_tinystories_256 [{args.site}] nice={args.nice} — {label}')

    if args.dry_run:
        for c in cells:
            print(f'  gcsc_{c[1]}')
        study, first = cells[0][0], cells[0]
        body = (body_a(site, *first[1:], f'{out}/{first[1]}') if study == 'A'
                else body_b(site, first[1], *first[2:], f'{out}/{first[1]}'))
        print('\n--- example body (first cell) ---\n' + body)
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for c in cells:
        study, tag = c[0], c[1]
        jobname = f'gcsc_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/samples_genppl.json') or jobname in active:
            n_skip += 1
            continue
        tdir = f'{out}/{tag}'
        body = (body_a(site, tag, c[2], c[3], c[4], tdir) if study == 'A'
                else body_b(site, tag, c[2], c[3], c[4], c[5], tdir))
        part = site['partitions'][n_sub % len(site['partitions'])]
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition=part, **site['slurm'])
        jid = slurm.sbatch(body, sbatch_cmd=f'sbatch --nice={args.nice} --requeue',
                           verbose=False)
        print(f'  submitted {tag}: job {jid} ({part})')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
