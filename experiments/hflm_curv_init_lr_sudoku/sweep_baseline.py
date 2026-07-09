#!/usr/bin/env python
"""sweep_baseline.py — fair 3-seed baseline sweep on Sudoku (spec slides/jul09_2026).

Companion to sweep.py (the H-FLM curvature grid). Runs the NON-hyperbolic baselines
so the jul09 deck's "Recall the Former Results" table can be replaced by faithful,
seed-averaged numbers under the exact same eval protocol as the H-FLM curvature runs.

Grid (63 cells):
  algo       : {ar, sfm, sfm_trunc, sfm_trunc_ada, eflm, langflow_ada, langflow_full}  (7)
  difficulty : {easy, medium, hard}                                                     (3)
  seed       : {1, 2, 3}   (report the average)                                         (3)
Fixed (all algos, from the slide): tiny DiT (512/8/8), 20k steps, batch 256, seq 180,
bf16, EMA 0.9999, AdamW lr=3e-4 wd=0 betas=(0.9,0.999) eps=1e-8 clip=1.0, CE loss.
LR is the config default (3e-4) — the slide fixes LR={3e-4}, so no LR knob is swept.

Eval (per the slide, exact same protocol as the curvature sweep): sudoku_eval, 180
steps, greedy last step. For the FLM family (sfm*/eflm) the sample scripts already
default to velocity=exact, top_k_velocity=-1 (avg across vocab) — matched here with no
override. AR has no velocity/top-k concept (autoregressive greedy decode). LangFlow
uses its own analog knob sampler.top_k; it is left at the canonical fair-comparison
value top_k=1 (the value behind the deck's "Recall the Former Results" LangFlow row),
overridable via LANGFLOW_TOPK.

Each algo maps to its single-run train + sample scripts (SEED knob added to the six
baseline train scripts on 2026-07-07 to enable seeding, mirroring hflm.sh):
  scripts/train/sudoku/{ar,sfm,sfm_truncated,sfm_truncated_adaptive,eflm,langflow}.sh
  scripts/sample/sudoku/{...}.sh
LangFlow's two swept variants are its VARIANT knob: ada_sched, full (=ada_sched+SC).

Three sites, static DISJOINT split on the ALGO axis. tc + falcon SHARE the ARC /home
filesystem (same repo + outputs) so their algo sets must not overlap; unicorn is a
separate filesystem. Split reflects live availability (checked 2026-07-07): falcon is
wide open -> most work; unicorn is CPU-saturated but reliable -> backstop; tc is busy
with the OWT experiment -> light share.
  --site unicorn : {ar, sfm}                              (18 cells; thickstun,desa)
  --site tc      : {eflm, langflow_ada}                   (18 cells; TC a100+h200 queues)
  --site falcon  : {sfm_trunc, sfm_trunc_ada, langflow_full} (27 cells; Falcon l40s+a30)
Run this ON the submitting cluster. --algos overrides a site's set for live rebalancing
(keep tc/falcon disjoint). Idempotent + resumable: skips a cell whose eval/results.json
exists or whose job name is already queued; resubmit auto-resumes from last.ckpt.

Usage:  python sweep_baseline.py --site {unicorn,tc,falcon}
                                 [--difficulties easy medium hard] [--seeds 1 2 3]
                                 [--algos ar sfm ...] [--dry-run]
"""
import argparse
import getpass
import itertools
import os
import subprocess
import textwrap

from simple_slurm import Slurm

# algo_tag -> (train script stem, sample script stem, extra env for BOTH train+eval)
ALGOS = {
    'ar':            ('ar', 'ar', {}),
    'sfm':           ('sfm', 'sfm', {}),
    'sfm_trunc':     ('sfm_truncated', 'sfm_truncated', {}),
    'sfm_trunc_ada': ('sfm_truncated_adaptive', 'sfm_truncated_adaptive', {}),
    'eflm':          ('eflm', 'eflm', {}),
    'langflow_ada':  ('langflow', 'langflow', {'VARIANT': 'ada_sched'}),
    'langflow_full': ('langflow', 'langflow', {'VARIANT': 'full'}),
}

SITES = {
    'unicorn': dict(
        repo='/share/thickstun/sychou/workspace/research/s-flm',
        envbin='/home/sc3379/anaconda3/envs/sfm/bin',
        algos=['ar', 'sfm'],
        partitions=['thickstun,desa'],  # same-cluster comma-list is fine on unicorn
        # cpus_per_task=2 (was 4): kuleshov nodes are CPU-saturated by other users so
        # idle GPUs are only reachable with a slim CPU ask; tiny sudoku training is
        # GPU-bound so 2 cores suffice (dataloader is trivial for 81-token seqs).
        slurm=dict(exclude='desa-compute-01',
                   gres='gpu:1', ntasks=1, cpus_per_task=2, mem='16G',
                   time='06:00:00'),
        wandb_offline=False,
    ),
    'tc': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        algos=['eflm', 'langflow_ada'],
        # ARC rejects multi-partition submissions (per-partition QOS) -> round-robin
        # single queues (fast-starting first; preemption costs <=~40 min: ckpt-5k +
        # --requeue + auto-resume).
        partitions=['h200_preemptable_q', 'a100_preemptable_q',
                    'h200_normal_q', 'a100_normal_q'],
        slurm=dict(account='swan_research_dlm',
                   gres='gpu:1', ntasks=1, cpus_per_task=4, mem='32G',
                   time='06:00:00'),
        wandb_offline=True,
    ),
    'falcon': dict(
        repo='/home/shengyenc/workspace/research/s-flm',
        envbin='/home/shengyenc/anaconda3/envs/sfm/bin',
        algos=['sfm_trunc', 'sfm_trunc_ada', 'langflow_full'],
        # Falcon has NO a100 queue: L40S-48GB / A30-24GB are the bf16+flash-attn
        # options (V100/T4 are not). Round-robin single queues, fast-starting first.
        partitions=['a30_preemptable_q', 'l40s_preemptable_q',
                    'a30_normal_q', 'l40s_normal_q'],
        slurm=dict(account='swan_research_dlm',
                   gres='gpu:1', ntasks=1, cpus_per_task=4, mem='32G',
                   time='06:00:00'),
        wandb_offline=True,
    ),
}

DIFFICULTIES = ['easy', 'medium', 'hard']
SEEDS = ['1', '2', '3']


def tag_of(algo, difficulty, seed):
    # run-name: bl_d-{difficulty}_a-{algo}_rs{seed}  (bl_ prefix keeps these distinct
    # from the curvature runs d-{diff}_k... and out of analyze.py's K-regex).
    return f'bl_d-{difficulty}_a-{algo}_rs{seed}'


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def job_body(site, algo, tdir, difficulty, seed, langflow_topk):
    train, sample, extra = ALGOS[algo]
    env = dict(extra)  # e.g. VARIANT for langflow
    ev = ''.join(f'{k}={v} ' for k, v in env.items())
    # LangFlow eval takes TOPK (its top-k velocity analog); leave others alone.
    topk = f'TOPK={langflow_topk} ' if train == 'langflow' else ''
    offline = 'export WANDB_MODE=offline\n' if site['wandb_offline'] else ''
    return textwrap.dedent(f'''\
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export SLURM_JOB_NAME=bash
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCHDYNAMO_DISABLE=1
        export PATH={site['envbin']}:$PATH
        {offline}cd {site['repo']}
        if [ -f {tdir}/eval/results.json ]; then
            echo "[$(date)] cell already completed elsewhere -> no-op"; exit 0
        fi
        echo "[$(date)] TRAIN {algo} on $(hostname)"
        {ev}DIFFICULTY={difficulty} SEED={seed} OUTPUT_DIR={tdir} DEVICES=1 \\
            bash scripts/train/sudoku/{train}.sh
        echo "[$(date)] EVAL"
        {ev}{topk}DIFFICULTY={difficulty} CKPT_PATH={tdir}/checkpoints/last.ckpt \\
            OUTPUT_DIR={tdir}/eval DEVICES=1 \\
            bash scripts/sample/sudoku/{sample}.sh
        # checkpoints are transient bulk (~1.8G/cell): once the eval deliverable
        # exists, drop them to keep /home (ARC) and /share (unicorn) within quota
        if [ -f {tdir}/eval/results.json ]; then
            rm -rf {tdir}/checkpoints && echo "[$(date)] checkpoints cleaned"
        fi
        echo "[$(date)] DONE"
        ''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=SITES)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--difficulties', nargs='+', default=DIFFICULTIES,
                    choices=DIFFICULTIES)
    ap.add_argument('--seeds', nargs='+', default=SEEDS)
    ap.add_argument('--algos', nargs='+', default=None, choices=list(ALGOS),
                    help="override the site's algo set (live rebalancing; keep "
                         'tc/falcon disjoint since they share ARC /home)')
    ap.add_argument('--partitions', nargs='+', default=None,
                    help="override the site's partition round-robin (e.g. force "
                         'non-preemptable normal queues to avoid preemption churn)')
    ap.add_argument('--langflow-topk', default='1',
                    help='LangFlow sampler.top_k for eval (1=canonical fair-comparison '
                         'argmax; -1=full-vocab expectation, the literal top_k_v=-1 analog)')
    args = ap.parse_args()
    site = SITES[args.site]
    algos = args.algos if args.algos else site['algos']
    partitions = args.partitions if args.partitions else site['partitions']
    exp = f"{site['repo']}/experiments/hflm_curv_init_lr_sudoku"
    logs = f'{exp}/logs'
    out = f"{site['repo']}/outputs/hflm_curv_init_lr_sudoku"
    nice = 0
    if not args.dry_run:  # site repo path is only writable on the submitting cluster
        os.makedirs(logs, exist_ok=True)

    cells = list(itertools.product(algos, args.difficulties, args.seeds))
    print(f'baseline sudoku [{args.site}]: {len(cells)} cells '
          f'({len(algos)} algo x {len(args.difficulties)} difficulty x '
          f'{len(args.seeds)} seed) | algos={algos}')
    if args.dry_run:
        for algo, diff, seed in cells:
            print(f'  {tag_of(algo, diff, seed)}  nice={nice}')
        algo, diff, seed = cells[0]
        print('\n--- example body (first cell) ---\n'
              + job_body(site, algo, f'{out}/{tag_of(algo, diff, seed)}',
                         diff, seed, args.langflow_topk))
        return

    active = active_jobnames()
    n_sub = n_skip = 0
    for algo, diff, seed in cells:
        tag = tag_of(algo, diff, seed)
        if os.path.exists(f'{out}/{tag}/eval/results.json') or tag in active:
            n_skip += 1
            continue
        # --nice on the command line: simple_slurm's '#SBATCH --nice N' directive is
        # rejected by sbatch (--nice takes an optional arg, needs '=' form)
        part = partitions[n_sub % len(partitions)]
        slurm = Slurm(job_name=tag, output=f'{logs}/{tag}_%j.log',
                      partition=part, **site['slurm'])
        jid = slurm.sbatch(
            job_body(site, algo, f'{out}/{tag}', diff, seed, args.langflow_topk),
            sbatch_cmd=f'sbatch --nice={nice} --requeue', verbose=False)
        print(f'  submitted {tag}: job {jid}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
