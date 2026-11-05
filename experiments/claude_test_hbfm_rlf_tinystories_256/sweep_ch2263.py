#!/usr/bin/env python
"""claude_test_hbfm_rlf_tinystories_256 on the borrowed ch2263 account (8x A100-80GB, partitions
nlplarge-claire-highpri / nlplarge; Agent.md "ch2263@Unicorn"). Same cells, knobs and scripts as sweep.py; this file only
changes WHERE they run: the repo copy /home/ch2263/syc_workspace/s-flm (code + data_cache + HF caches rsynced there,
since ch2263 cannot read /share/desa/nfs02/sc3379), the `sfm` conda env there, checkpoints on /scratch/ch2263 behind a
symlink made in-job by experiment.link_scratch_checkpoints, SLURM through `ssh ... bash -lc`, and results pulled back
into this project's outputs/ tree so report.py sees them.

Cells (2026-10-02 11:00 EDT, user: "migrate part of pending jobs to ch2263"): the `best` arm under the horosphere
readout (seeds 1-3, new) and A100 replicates of its naive seeds 2-3 (the A5000 copies finish ~04:30 Oct 3; the
replicate's reads are pulled into eval_a100/ and eval_a100_tmax0.5/ so the two never mix in the tables).
2 GPUs per cell (7 of 8 A100s were taken by other users at submission), micro-batch 32 x accumulation 8 = 512.

Usage:  python sweep_ch2263.py [--dry-run] [--only SUBSTR] | --smoke | --status | --pull
"""
import argparse
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402  (knobs, eval_cmd, run, tag_of, T_MAX_*)

CH = 'ch2263@unicorn-login-02.coecis.cornell.edu'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-i', '/home/sc3379/.ssh/unicorn_internal', CH]
CH_REPO = '/home/ch2263/syc_workspace/s-flm'
# Agent.md names the `hodlr` env, but on 2026-10-02 its flash_attn wheel no longer matched its torch (undefined symbol at
# import); `sfm` there (py3.12, torch 2.7.0+cu128, flash_attn 2.8.3, lightning 2.5.1) imports the repo cleanly.
CH_ENVBIN = '/home/ch2263/miniconda3/envs/sfm/bin'
PROJECT = 'claude_test_hbfm_rlf_tinystories_256'
CH_OUT = f'{CH_REPO}/outputs/{PROJECT}'
CH_LOGS = f'{CH_REPO}/experiments/{PROJECT}/logs'
# The repo copy's data_cache/ and outputs/ are symlinks into /scratch/ch2263 (the user's setup), which only compute
# nodes mount: the data is staged in the home dir and copied to scratch by the job itself, and every cell copies its
# eval dirs back to a home dir at the end, which is what --status / --pull read.
CH_DATA_HOME = '/home/ch2263/syc_workspace/s-flm_data_home/data_cache'
CH_RESULTS = '/home/ch2263/syc_workspace/s-flm_results'
JOB = 'hbfm_rlf_a100'
PARTITION = 'nlplarge-claire-highpri,nlplarge'
DEVICES, PER_GPU_BS = 2, 32

CELLS = [('best', 'horosphere', 1), ('best', 'horosphere', 2), ('best', 'horosphere', 3),
         ('best', 'naive', 2), ('best', 'naive', 3)]
REPLICATES = {('best', 'naive', 2), ('best', 'naive', 3)}   # pulled into eval_a100*/


def ssh(cmd, stdin=None):
    return subprocess.run(SSH + [f'bash -lc {cmd!r}'], input=stdin, capture_output=True, text=True)


def header(tag, gpus, hours, cpus=16, mem='128G'):
    return textwrap.dedent(f'''\
        #!/bin/bash
        #SBATCH -J {JOB}_{tag}
        #SBATCH -p {PARTITION}
        #SBATCH --gres=gpu:{gpus}
        #SBATCH -n 1 -c {cpus} --mem={mem} -t {hours}:00:00
        #SBATCH -o {CH_LOGS}/{tag}_%j.log
        set -e
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export TMPDIR=/tmp/ch2263_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR   # node-local: the dataloader workers' AF_UNIX sockets vanished from the NFS home after ~20 min (job 886611)
        trap 'rm -rf "$TMPDIR"' EXIT
        export PATH={CH_ENVBIN}:$PATH
        cd {CH_REPO}
        hostname; nvidia-smi -L | head -{gpus}
        md5sum scripts/train/tinystories/hbfm.sh scripts/sample/tinystories/hbfm.sh algo.py models/hyperbolic_dit.py
        mkdir -p data_cache/ && rsync -a {CH_DATA_HOME}/tinystories_train_bs256_wrapped.dat {CH_DATA_HOME}/tinystories_valid_bs256_wrapped.dat data_cache/
        ls -la data_cache/ | grep tinystories
        ''')


def job_script(arm, fwd, seed, max_steps=sweep.MAX_STEPS, gpus=DEVICES, per_gpu=PER_GPU_BS, out=None, smoke=False):
    tag = sweep.tag_of(arm, fwd, seed)
    tdir = out or f'{CH_OUT}/{tag}'
    ckpt = f'{tdir}/checkpoints/last.ckpt'
    lim = 'LIMIT_VAL_BATCHES=0.002 NUM_SAMPLE_BATCHES=1 ' if smoke else ''
    body = header(tag if not smoke else 'smoke', gpus, 2 if smoke else 47) + textwrap.dedent(f'''\
        python -c "import experiment; experiment.link_scratch_checkpoints('{tdir}', '{PROJECT}', '{os.path.basename(tdir)}', 'ch2263')"
        echo "[$(date)] TRAIN {tag} on $(hostname)"
        LR=3e-4 SEED={seed} {sweep.knobs(arm, fwd, sweep.T_MAX_DEFAULT)} \\
            OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={PROJECT} \\
            DEVICES={gpus} PER_GPU_BS={per_gpu} MAX_STEPS={max_steps} CKPT_EVERY={2 if smoke else 1000} KEEP_EVERY=5000 \\
            {sweep.run('scripts/train/tinystories/hbfm.sh')}
        echo "[$(date)] EVAL {tag} t_max {sweep.T_MAX_DEFAULT:g}"
        {sweep.eval_cmd(arm, fwd, sweep.T_MAX_DEFAULT, ckpt, f'{tdir}/eval').replace('CKPT_PATH=', lim + 'CKPT_PATH=', 1)}
        echo "[$(date)] EVAL {tag} t_max {sweep.T_MAX_MATCHED:g}"
        {sweep.eval_cmd(arm, fwd, sweep.T_MAX_MATCHED, ckpt, f'{tdir}/eval_tmax{sweep.T_MAX_MATCHED:g}').replace('CKPT_PATH=', lim + 'CKPT_PATH=', 1)}
        mkdir -p {CH_RESULTS}/{os.path.basename(tdir)} && rsync -a {tdir}/eval {tdir}/eval_tmax{sweep.T_MAX_MATCHED:g} {CH_RESULTS}/{os.path.basename(tdir)}/
        echo "[$(date)] DONE {tag}"
        ''')
    return body


def remote_jobs():
    r = ssh(f'squeue -u ch2263 -h -o "%j %t %M"')
    return {line.split()[0]: line.split()[1:] for line in r.stdout.split('\n') if line.startswith(JOB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--smoke', action='store_true', help='4-step, 1-GPU, tiny-eval run of the first cell into _smoke/')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--pull', action='store_true', help='rsync finished evals + logs back into this project')
    args = ap.parse_args()
    ssh(f'mkdir -p {CH_LOGS} {CH_OUT} /home/ch2263/tmp')
    if args.status:
        print(ssh(f'squeue -u ch2263 -o "%.8i %.40j %.3t %.10M %b %R"').stdout)
        for arm, fwd, seed in CELLS:
            tag = sweep.tag_of(arm, fwd, seed)
            r = ssh(f'ls {CH_RESULTS}/{tag}/eval/samples_genppl.json {CH_RESULTS}/{tag}/eval_tmax0.5/samples_genppl.json 2>/dev/null | wc -l; '
                    f'tail -c 20000 {CH_LOGS}/{tag}_*.log 2>/dev/null | tr "\\r" "\\n" | grep -E "it/s|EVAL|DONE|Error" | tail -1 | cut -c1-80')
            print(f'  {tag:32s} finals {r.stdout.strip().replace(chr(10), " | ")}')
        return
    if args.pull:
        for arm, fwd, seed in CELLS:
            tag = sweep.tag_of(arm, fwd, seed)
            local = f'{sweep.OUT}/{tag}'
            os.makedirs(local, exist_ok=True)
            for ed in ('eval', f'eval_tmax{sweep.T_MAX_MATCHED:g}'):
                dest = ed.replace('eval', 'eval_a100', 1) if (arm, fwd, seed) in REPLICATES else ed
                subprocess.run(['rsync', '-a', '-e', ' '.join(SSH[:-1]), f'{CH}:{CH_RESULTS}/{tag}/{ed}/', f'{local}/{dest}/'],
                               capture_output=True)
            subprocess.run(['rsync', '-a', '-e', ' '.join(SSH[:-1]), f'{CH}:{CH_LOGS}/', f'{sweep.LOGS}/a100/'], capture_output=True)
            print(f'  pulled {tag}: ' + ', '.join(d for d in os.listdir(local) if d.startswith('eval')))
        return
    if args.smoke:
        arm, fwd, seed = CELLS[0]
        body = job_script(arm, fwd, seed, max_steps=4, gpus=1, per_gpu=8, out=f'{CH_OUT}/_smoke', smoke=True)
        body = body.replace('GLOBAL_BATCH=512', 'GLOBAL_BATCH=64')
        if args.dry_run:
            print(body); return
        r = ssh('sbatch', stdin=body); print(r.stdout.strip() or r.stderr.strip()); return
    active = {} if args.dry_run else remote_jobs()
    n_sub = n_skip = 0
    for arm, fwd, seed in CELLS:
        tag = sweep.tag_of(arm, fwd, seed)
        if args.only not in tag:
            continue
        jobname = f'{JOB}_{tag}'
        done = not args.dry_run and ssh(f'ls {CH_RESULTS}/{tag}/eval_tmax{sweep.T_MAX_MATCHED:g}/samples_genppl.json 2>/dev/null').stdout.strip()
        if done or jobname in active:
            n_skip += 1
            continue
        body = job_script(arm, fwd, seed)
        if args.dry_run:
            print(f'  [dry] {jobname}' + (f'\n{body}' if n_sub == 0 else ''))
        else:
            r = ssh('sbatch', stdin=body)
            print(f'  submitted {tag}: {r.stdout.strip() or r.stderr.strip()}')
        n_sub += 1
    print(f'submitted {n_sub}, skipped {n_skip}')


if __name__ == '__main__':
    main()
