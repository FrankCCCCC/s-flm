#!/usr/bin/env python
"""Rename hflm_curv_init_lr_sudoku runs to d-{difficulty}_k-{K}_i-{init}_lr{lr}_rs{seed}.

Legacy names:  k-<K>_i-<init>_lr<lr>            (medium, difficulty implicit)
               d-hard_k-<K>_i-<init>_lr<lr>     (hard)
New names:     d-<difficulty>_k-<K>_i-<init>_lr<lr>_rs<seed>

Idempotent and safe to re-run: only renames a dir when (a) it matches a legacy
pattern, (b) its eval/results.json exists (run fully finished), and (c) no active
SLURM job carries its legacy jobname (in-flight jobs have the legacy OUTPUT_DIR
baked into their sbatch body, so they must finish before being renamed).
The seed is read from the run's .hydra/config.yaml.

Usage:  python rename_runs.py [--dry-run]
"""
import argparse
import getpass
import os
import re
import subprocess

OUT = '/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku'
LEGACY = re.compile(r'^(?:d-(?P<diff>[a-z]+)_)?k(?P<k>-[\d.]+)_i-(?P<init>[^_]+)_lr(?P<lr>[\de.-]+)$')


def seed_of(run_dir):
    try:
        with open(os.path.join(run_dir, '.hydra', 'config.yaml')) as f:
            for line in f:
                m = re.match(r'^seed:\s*(\d+)', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def active_jobnames():
    try:
        out = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                             capture_output=True, text=True).stdout
        return set(out.split())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    active = active_jobnames()
    n_ren = n_skip = 0
    for name in sorted(os.listdir(OUT)):
        m = LEGACY.match(name)
        if not m:  # already new-style (has _rs suffix) or unrelated
            continue
        src = os.path.join(OUT, name)
        done = os.path.exists(os.path.join(src, 'eval', 'results.json'))
        in_flight = f'hcil_{name}' in active
        seed = seed_of(src)
        if not done or in_flight or seed is None:
            print(f'  skip {name} (done={done}, in_flight={in_flight}, seed={seed})')
            n_skip += 1
            continue
        diff = m.group('diff') or 'medium'
        new = f'd-{diff}_k{m.group("k")}_i-{m.group("init")}_lr{m.group("lr")}_rs{seed}'
        dst = os.path.join(OUT, new)
        if os.path.exists(dst):
            print(f'  skip {name} (target exists: {new})')
            n_skip += 1
            continue
        print(f'  {name}  ->  {new}')
        if not args.dry_run:
            os.rename(src, dst)
        n_ren += 1
    print(f'{"would rename" if args.dry_run else "renamed"} {n_ren}, skipped {n_skip}')


if __name__ == '__main__':
    main()
