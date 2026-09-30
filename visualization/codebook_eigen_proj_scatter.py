#!/usr/bin/env python
"""Train-set word frequency vs projection on the top eigenvector of E^T E.

For one of the normalized codebook variants E (--mode; defined in
codebook_eigen_dist.py, EMA weights) take v_1, the eigenvector of the Gram
matrix E^T E with the largest eigenvalue lambda_1, and scatter every vocab
word i at

  x = <e_i, v_1>   inner product of its embedding row with v_1
  y = count_i      its token count over the whole training set; symlog y, so
                   words that never occur in training sit on the y = 0 line

An eigenvector's sign is arbitrary: v_1 is oriented so the count-weighted mean
of x is >= 0.

Output: <out>_{mode}.png

Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to load the Lightning checkpoint;
CPU-only, but run it on a compute node.

Example:
  python visualization/codebook_eigen_proj_scatter.py --mode normalized \
    --ckpt outputs/naive_ar_tinystories_s256/m-sfmta_lr-1e-3_sd-1/checkpoints/last.ckpt
"""
import argparse, os, sys

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codebook_eigen_dist import embedding_matrices  # noqa: E402
import dataloader  # noqa: E402  (repo root via loss_geometry)

MODES = ('normalized', 'normalized-mean-shift', 'mean-shift-normalized')


def train_counts(cfg, V: int) -> np.ndarray:
  """Per-word token count over the whole tokenized training set.

  Bincounts the cached Arrow table directly; a DataLoader pass over ~1.9M
  blocks would take minutes.
  """
  assert cfg.data.wrap  # wrapped blocks hold no pad tokens to mask out
  train_set = dataloader.get_dataset(
    cfg, dataloader.get_tokenizer(cfg), mode='train')
  cnt = np.zeros(V, dtype=np.int64)
  for chunk in train_set.data.column('input_ids').chunks:
    cnt += np.bincount(chunk.flatten().to_numpy(), minlength=V)
  return cnt


def plot_scatter(x: np.ndarray, cnt: np.ndarray, lam1: float, run: str,
                 mode: str, out: str):
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.scatter(x, cnt, s=3, alpha=0.3, lw=0, rasterized=True)
  ax.set_yscale('symlog', linthresh=1)
  ax.set(xlabel=r'$\langle e_i, v_1 \rangle$,  $v_1$ = top eigenvector of '
                r'$E^\top E$',
         ylabel='train-set frequency (token count, symlog)')
  ax.set_title(f'{run} [{mode}]\n$\\lambda_1$={lam1:.4g}, V={x.size}, '
               f'{cnt.sum():,} train tokens, {(cnt == 0).sum()} words '
               f'never occur (y=0)', fontsize=10)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  path = f'{out}_{mode}.png'
  fig.savefig(path, dpi=150); plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--ckpt', required=True, help='path to a run checkpoint')
  p.add_argument('--mode', choices=MODES, default='normalized',
                 help='codebook variant whose top eigenvector is projected on')
  p.add_argument('--out', default=None,
                 help='output prefix (no extension); default '
                      'experiments/{project}/codebook_eigen_proj_scatter_{run}')
  p.add_argument('--cache-dir', default=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_cache'))
  args = p.parse_args()
  args.batch_size = 16  # required by the shared _load_config; no batches here

  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
  run = os.path.basename(run_dir)
  if args.out is None:
    project = os.path.basename(os.path.dirname(run_dir))
    args.out = os.path.join('experiments', project,
                            f'codebook_eigen_proj_scatter_{run}')
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

  mats, cfg = embedding_matrices(args.ckpt, args)
  W = mats[args.mode]
  cnt = train_counts(cfg, W.shape[0])
  lam, U = torch.linalg.eigh(W.T @ W)  # ascending, so v_1 = U[:, -1]
  x = (W @ U[:, -1]).numpy()
  if cnt @ x < 0:  # orient v_1 toward the token mass
    x = -x
  print(f'{run} [{args.mode}] E{tuple(W.shape)}  lambda_1={lam[-1]:.4g}  '
        f'train tokens={cnt.sum():,}  zero-count words={(cnt == 0).sum()}',
        flush=True)
  plot_scatter(x, cnt, lam[-1].item(), run, args.mode, args.out)


if __name__ == '__main__':
  main()
