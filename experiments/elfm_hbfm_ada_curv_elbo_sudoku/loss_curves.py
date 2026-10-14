#!/usr/bin/env python
"""Training-loss curves of the cells, from their offline W&B run files (no model is loaded).

Reads outputs/elfm_hbfm_ada_curv_elbo_sudoku/<cell>/wandb/offline-run-*/run-*.wandb, keeps the
`trainer/loss` history (logged every 100 optimizer steps), writes
experiments/elfm_hbfm_ada_curv_elbo_sudoku/loss_curves.csv (cell, step, loss) and, with --plot,
figures/loss_curves.png: one panel per (arm, loss), the LR cells as lines (an objective's value is
only comparable within its own panel -- the ELBO and the CE are different quantities).

Usage:  python loss_curves.py [--cells SUBSTR] [--plot]
"""
import argparse
import csv
import glob
import json
import os

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
OUT = f'{REPO}/outputs/elfm_hbfm_ada_curv_elbo_sudoku'
CSV = f'{REPO}/experiments/elfm_hbfm_ada_curv_elbo_sudoku/loss_curves.csv'


def history(run_file):
    """List of {key: value} rows of the run's logged history."""
    ds = datastore.DataStore()
    ds.open_for_scan(run_file)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof('record_type') == 'history':
            rows.append({(it.key or '/'.join(it.nested_key)): json.loads(it.value_json)
                         for it in rec.history.item})
    return rows


def loss_curve(cell_dir):
    """(step, loss) pairs of a cell, over all its runs (a resumed cell has several)."""
    points = {}
    for run_file in sorted(glob.glob(f'{cell_dir}/wandb/offline-run-*/run-*.wandb')):
        for row in history(run_file):
            if 'trainer/loss' in row and 'trainer/global_step' in row:
                points[int(row['trainer/global_step'])] = float(row['trainer/loss'])
    return sorted(points.items())


# Categorical slots of the validated default palette (dataviz skill), fixed order: one hue per LR.
PALETTE = {'3e-4': '#2a78d6', '5e-4': '#eb6834', '1e-3': '#1baf7a'}
PANELS = [('hbfm', 'elbo', 'HBFM, ELBO loss'), ('hbfm', 'ce', 'HBFM, CE loss'),
          ('eflm', 'elbo', 'EFLM, ELBO loss'), ('mdlm', 'elbo', 'MDLM (its NELBO)')]


def plot(curves, path):
    """curves: {cell: [(step, loss), ...]} -> a 2 x 2 figure of training loss vs step."""
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (arm, loss, title) in zip(axes.flat, PANELS):
        for cell, pts in sorted(curves.items()):
            m = re.match(rf'{arm}_loss-{loss}_lr-([\de.-]+)_seed-(\d+)$', cell)
            if not m or not pts:
                continue
            lr, seed = m.groups()
            steps, losses = zip(*pts)
            # Every 25 epochs one logged value is ~0 (the epoch's last, partial accumulation step);
            # a 3-point running median drops these single points and nothing else. The CSV is raw.
            losses = [sorted(losses[max(0, i - 1):i + 2])[len(losses[max(0, i - 1):i + 2]) // 2]
                      for i in range(len(losses))]
            ax.plot(steps, losses, lw=2, color=PALETTE[lr], alpha=1.0 if seed == '1' else 0.45,
                    label=f'LR {lr}' + ('' if seed == '1' else f' (seed {seed})'))
        ax.set_yscale('log')
        ax.set_title(title, fontsize=11, loc='left', color='#0b0b0b')
        ax.grid(True, color='#e6e5e1', lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors='#52514e', labelsize=9)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('optimizer step', color='#52514e')
    for ax in axes[:, 0]:
        ax.set_ylabel('training loss (nats / token, log)', color='#52514e')
    fig.suptitle('Training loss, hard Sudoku, batch 256 (logged every 100 steps, 3-point running median)',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='', help='substring filter on the cell name')
    ap.add_argument('--plot', action='store_true', help='also write figures/loss_curves.png')
    args = ap.parse_args()
    curves = {}
    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'step', 'loss'])
        for cell_dir in sorted(glob.glob(f'{OUT}/*_seed-*')):
            cell = os.path.basename(cell_dir)
            if args.cells not in cell or os.path.islink(f'{cell_dir}/checkpoints'):
                continue   # reused controls were trained elsewhere
            curve = loss_curve(cell_dir)
            curves[cell] = curve
            for step, loss in curve:
                w.writerow([cell, step, f'{loss:.6g}'])
            print(f'{cell}: {len(curve)} points' + (f', last {curve[-1]}' if curve else ''))
    print(f'wrote {CSV}')
    if args.plot:
        plot(curves, f'{REPO}/experiments/elfm_hbfm_ada_curv_elbo_sudoku/figures/loss_curves.png')


if __name__ == '__main__':
    main()
