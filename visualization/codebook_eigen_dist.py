#!/usr/bin/env python
"""Spectrum of a run's word-embedding matrix E = backbone.sphere_embed.weight.

E is [V, d] and rectangular, so it has no eigenvalues; the two well-defined
spectral quantities, both plotted here, are

  sigma_i   the d singular values of E
  lambda_i  the d eigenvalues of the Gram matrix E^T E, = sigma_i^2

Codebook variants per run:
  raw       E exactly as trained (the nn.Embedding parameter)
  rescaled  rescale_radius(E, algo.rho_min, algo.rho_max) -- the codebook the
            flow actually embeds into (cf. EFLM.q_xt / _sc_embed_table); for
            rho_min == rho_max == R every row is pinned to norm R, i.e. R times
            the row-normalized E. Identity when neither rho is set. Only the
            sphere-dit backbone has it; skipped for plain DiT / FLM-DiT.
  normalized             every row projected onto the unit sphere
  normalized-mean-shift  normalize, then subtract the mean direction
  mean-shift-normalized  subtract the mean row, then normalize

Weights are the EMA ones, as trainer.validate() sees them.

Outputs, per variant:
  <out>_{raw,rescaled}[_logx].png   2x2 fine histograms; rows = (sigma,
                             lambda), columns = (count, portion), log y
                             throughout (portion = count / d, so the columns
                             differ only by that constant factor). Dashed line
                             = median. The spectrum is heavy-tailed, so the
                             `_logx` pass repeats it with log-spaced bins --
                             read that one to see the bulk.

Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to load the Lightning checkpoint;
CPU-only, but run it on a compute node.

Example:
  python visualization/codebook_eigen_dist.py \
    --ckpt outputs/eflm_rescale_tinystories_256/eflmrs256_trunc_ada_r-1_rs1/checkpoints/last.ckpt
"""
import argparse, os, sys

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import ALGO_BY_NAME, _load_config  # noqa: E402
import dataloader, main as main_mod  # noqa: E402  (repo root via loss_geometry)

XLABELS = {'sigma': r'singular value $\sigma_i(E)$',
           'lambda': r'eigenvalue $\lambda_i(E^\top E) = \sigma_i^2$'}


@torch.no_grad()
def embedding_matrices(ckpt: str, args):
  """{'raw','rescaled'} -> [V, d] embedding matrix (EMA weights), + the config."""
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
  cfg = _load_config(run_dir, ckpt, args)
  tokenizer = dataloader.get_tokenizer(cfg)
  model = main_mod._load_from_checkpoint(
    ALGO_BY_NAME[cfg.algo.name], cfg, tokenizer).eval()
  model._eval_mode()  # swap in EMA weights, as trainer.validate does
  bb = model.backbone
  # sphere/hyperbolic backbones name the codebook `sphere_embed`; the plain DiT
  # and FLM-DiT ones keep it as the `vocab_embed` parameter.
  W = (bb.sphere_embed.weight if hasattr(bb, 'sphere_embed')
       else bb.vocab_embed.embedding).detach().double()
  normalized = torch.nn.functional.normalize(W, p=2.0, dim=-1)
  normalized_mean_shift = normalized - normalized.mean(dim=0)
  mean_shift_normalized = torch.nn.functional.normalize(W - W.mean(dim=0), p=2.0, dim=-1)
  mats = {'raw': W}
  if hasattr(bb, 'rescale_radius'):  # sphere-dit only
    mats['rescaled'] = bb.rescale_radius(
      W, cfg.algo.get('rho_min'), cfg.algo.get('rho_max'))
  mats.update({'normalized': normalized,
               'normalized-mean-shift': normalized_mean_shift,
               'mean-shift-normalized': mean_shift_normalized})
  return mats, cfg


def spectrum(W: torch.Tensor) -> dict:
  """The d singular values of E and the matching Gram eigenvalues sigma^2."""
  sigma = torch.linalg.svdvals(W).numpy()
  return {'sigma': sigma, 'lambda': sigma ** 2}


def plot_histograms(spec: dict, run: str, variant: str, shape, out: str,
                    bins: int, log_x: bool):
  """<out>_{variant}[_logx].png: rows = (sigma, lambda), cols = (count, portion).

  The spectrum is heavy-tailed (a handful of sigma_i are ~20x the bulk), so
  linearly spaced bins put ~99% of the d values in the leftmost few; the log_x
  pass re-bins geometrically to resolve that bulk.
  """
  V, d = shape
  fig, axes = plt.subplots(2, 2, figsize=(11, 8))
  for row, (kind, vals) in enumerate(spec.items()):
    edges = (np.geomspace(vals.min() * 0.98, vals.max() * 1.02, bins + 1)
             if log_x else np.linspace(0.0, vals.max() * 1.02, bins + 1))
    for col, (ylabel, w) in enumerate(
        [('count', None),
         ('portion of spectrum', np.full(vals.size, 1.0 / vals.size))]):
      ax = axes[row, col]
      ax.hist(vals, bins=edges, weights=w, color='C0')
      ax.axvline(np.median(vals), color='r', ls='--', lw=1.2,
                 label=f'median={np.median(vals):.4g}')
      ax.set_yscale('log')
      if log_x:
        ax.set_xscale('log')
      ax.set(xlabel=XLABELS[kind], ylabel=f'log {ylabel}',
             title=f'{kind}: max={vals.max():.4g}, min={vals.min():.4g}')
      ax.grid(alpha=0.3); ax.legend(fontsize=8)
  fig.suptitle(f'{run} [{variant}] word-embedding spectrum   '
               f'(V={V}, d={d}, {bins} '
               f'{"log-spaced" if log_x else "linear"} bins)', fontsize=12)
  fig.tight_layout()
  path = f'{out}_{variant}{"_logx" if log_x else ""}.png'
  fig.savefig(path, dpi=150); plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--ckpt', required=True, help='path to a run checkpoint')
  p.add_argument('--out', default=None,
                 help='output prefix (no extension); default '
                      'experiments/{project}/codebook_eigen_dist_{run}')
  p.add_argument('--bins', type=int, default=200)
  p.add_argument('--variants', default=None,
                 help='comma-separated subset of the variants to plot; '
                      'default: all of them')
  p.add_argument('--cache-dir', default=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_cache'))
  args = p.parse_args()
  args.batch_size = 16  # required by the shared _load_config; no batches here

  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
  run = os.path.basename(run_dir)
  if args.out is None:
    project = os.path.basename(os.path.dirname(run_dir))
    args.out = os.path.join('experiments', project,
                            f'codebook_eigen_dist_{run}')
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

  mats, cfg = embedding_matrices(args.ckpt, args)
  if args.variants:
    mats = {k: mats[k] for k in args.variants.split(',')}
  print(f'{run}: algo={cfg.algo.name} rho_min={cfg.algo.get("rho_min")} '
        f'rho_max={cfg.algo.get("rho_max")}', flush=True)
  for variant, W in mats.items():
    spec = spectrum(W)
    lam = spec['lambda']
    print(f'[{variant}] E{tuple(W.shape)}  row-norm median='
          f'{W.norm(dim=1).median():.4g}  sigma: max={spec["sigma"].max():.4g} '
          f'min={spec["sigma"].min():.4g}  '
          f'stable rank ||E||_F^2/sigma_max^2={lam.sum() / lam.max():.2f}',
          flush=True)
    for log_x in (False, True):
      plot_histograms(spec, run, variant, tuple(W.shape), args.out, args.bins,
                      log_x)


if __name__ == '__main__':
  main()
