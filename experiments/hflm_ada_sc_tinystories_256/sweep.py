#!/usr/bin/env python
"""hflm_ada_sc_tinystories_256 — H-FLM + adaptive noise, {curvature x init x prior_cov}
x self-cond, on TinyStories seq 256.

Grid (90 cells) = the geo_curv Part B "H-FLM Sweep" params x {SC on, off}:
  init       : random(std0.02), c0.01(std0.01), c0.04(std0.04)   (3)
  prior_cov  : 0.5, 0.8, 1.0                                      (3)
  K (curv)   : -0.01, -0.1, -0.25, -0.5, -0.75                    (5)
  self_cond  : on, off                                            (2)     -> 3x3x5x2 = 90
Fixed: small-hyperbolic-dit 768/12/12, ADAPTIVE noise (log-linear-adaptive, no
truncation), rho_max=12, 30k steps, global batch 512, seq 256, bf16, EMA 0.9999,
AdamW lr 3e-4 wd 0 clip 1.0. Eval: exact velocity, top_k_v=1, 180 steps, greedy last.
Inherits Part B's collapse caveat (small init x large prior_cov); reported as a finding.

ORCHESTRATION ONLY — each cell calls the single-run shared scripts (SEQ_LEN=256 knob):
  scripts/train/tinystories/hlfm_adaptive.sh   (INIT/INIT_STD/PRIOR_COV/RHO_MAX/GAUSS_CURV/SELF_COND)
  scripts/sample/tinystories/hflm_adaptive.sh  (matching adaptive eval schedule)
Idempotent + resumable: skip a cell whose eval/samples_genppl.json exists (the LAST eval
artifact) or whose job is queued; resubmit auto-resumes from checkpoints/last.ckpt.
Checkpoints deleted after samples_genppl.json (quota). set -e so a failed eval never cleans.

Load-balanced across 3 sites (disjoint by K; ch2263 is its own cluster, tc/falcon share
ARC /home so their K sets must stay disjoint). nice=0 (prioritized — user wants ASAP):
  --site ch2263 : K in {-0.5, -0.75}  (36) [ch2263@unicorn nlplarge-claire-highpri, 8xA100]
  --site tc     : K in {-0.01}        (18) [shengyenc@TinkerCliffs a100+h200; saturated]
  --site falcon : K in {-0.1, -0.25}  (36) [shengyenc@Falcon l40s+a30]
Run this script ON the submitting cluster. rsync ARC->unicorn results before report.py.

Usage: python sweep.py --site {ch2263,tc,falcon} [--nice 0] [--dry-run] [--curvatures ...]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

SEQ_LEN = 256
CKPT_EVERY = 5000
SAVE_TOPK = 1
RHO_MAX = '12'

SITES = {
    'ch2263': dict(
        repo='/home/ch2263/syc_workspace/s-flm',
        envbin='/home/ch2263/miniconda3/envs/sfm_fa/bin',  # py3.12+torch2.9+flash-attn
        curvatures=['-0.5', '-0.75'],
        per_gpu_bs=64,                        # A100-80GB
        partitions=['nlplarge-claire-highpri', 'nlplarge'],
        slurm=dict(gres='gpu:1', ntasks=1, cpus_per_task=8, mem='64G',
                   time='10-00:00:00'),
    ),
    'tc': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        curvatures=['-0.01'],                 # smallest share: TC a100/h200 saturated
        per_gpu_bs=64,
        partitions=['h200_preemptable_q', 'a100_preemptable_q',
                    'h200_normal_q', 'a100_normal_q'],
        slurm=dict(account='swan_research_dlm', gres='gpu:1', ntasks=1,
                   cpus_per_task=8, mem='64G', time='7-00:00:00'),
    ),
    'falcon': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        curvatures=['-0.1', '-0.25'],
        per_gpu_bs=32,                        # A30-24GB safe
        partitions=['a30_preemptable_q', 'l40s_preemptable_q',
                    'a30_normal_q', 'l40s_normal_q'],
        slurm=dict(account='swan_research_dlm', gres='gpu:1', ntasks=1,
                   cpus_per_task=8, mem='64G', time='7-00:00:00'),
    ),
}

INITS = [('random', 'random', 'null'),
         ('c0.01', 'custom', '0.01'),
         ('c0.04', 'custom', '0.04')]
PRIOR_COVS = ['0.5', '0.8', '1.0']
SCS = [('scon', 'true'), ('scoff', 'false')]

_HEAD = '\n'.join([
    'set -e',
    'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
    'export SLURM_JOB_NAME=bash',
    'export NCCL_P2P_DISABLE=1',
    'export NCCL_IB_DISABLE=1',
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
    'export WANDB_MODE=offline',
])


def cells_for(curvatures):
    out = []
    for k in curvatures:
        for itag, init, istd in INITS:
            for pc in PRIOR_COVS:
                for sctag, scval in SCS:
                    tag = f'k{k}_i-{itag}_pc{pc}_{sctag}'
                    out.append((tag, k, init, istd, pc, scval))
    return out


def body(site, tag, k, init, istd, pc, scval, tdir):
    edir = f'{tdir}/eval'
    init_env = f'INIT={init}' + (f' INIT_STD={istd}' if init == 'custom' else '')
    geo = f'GAUSS_CURV={k} PRIOR_COV={pc} RHO_MAX={RHO_MAX} SELF_COND={scval}'
    bs = site['per_gpu_bs']
    return textwrap.dedent(f'''\
        {_HEAD}
        export PATH={site['envbin']}:$PATH
        cd {site['repo']}
        if [ -f {edir}/samples_genppl.json ]; then echo "[$(date)] {tag} done elsewhere -> no-op"; exit 0; fi
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        {init_env} {geo} OUTPUT_DIR={tdir} RUN_NAME=hasc_{tag}_s256 DEVICES=1 PER_GPU_BS={bs} \\
            SEQ_LEN={SEQ_LEN} CKPT_EVERY={CKPT_EVERY} SAVE_TOPK={SAVE_TOPK} \\
            bash scripts/train/tinystories/hlfm_adaptive.sh
        echo "[$(date)] EVAL {tag}"
        {geo} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={edir} DEVICES=1 SEQ_LEN={SEQ_LEN} \\
            bash scripts/sample/tinystories/hflm_adaptive.sh
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
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--curvatures', nargs='+', default=None,
                    help="override the site's K list (for live rebalancing; keep "
                         'tc/falcon disjoint)')
    args = ap.parse_args()
    site = SITES[args.site]
    ks = args.curvatures or site['curvatures']
    exp = f"{site['repo']}/experiments/hflm_ada_sc_tinystories_256"
    logs = f'{exp}/logs'
    out = f"{site['repo']}/outputs/hflm_ada_sc_tinystories_256"
    if not args.dry_run:
        os.makedirs(logs, exist_ok=True)

    cells = cells_for(ks)
    print(f'hflm_ada_sc_tinystories_256 [{args.site}] nice={args.nice} — '
          f'{len(cells)} cells (K={ks} x 3 init x 3 prior_cov x 2 SC)')
    if args.dry_run:
        for c in cells:
            print(f'  hasc_{c[0]}')
        print('\n--- example body ---\n' + body(site, *cells[0], f'{out}/{cells[0][0]}'))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for c in cells:
        tag = c[0]
        jobname = f'hasc_{tag}'
        if os.path.exists(f'{out}/{tag}/eval/samples_genppl.json') or jobname in active:
            n_skip += 1
            continue
        part = site['partitions'][n_sub % len(site['partitions'])]
        slurm = Slurm(job_name=jobname, output=f'{logs}/{tag}_%j.log',
                      partition=part, **site['slurm'])
        jid = slurm.sbatch(body(site, *c, f'{out}/{tag}'),
                           sbatch_cmd=f'sbatch --nice={args.nice} --requeue',
                           verbose=False)
        print(f'  submitted {tag}: job {jid} ({part})')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
