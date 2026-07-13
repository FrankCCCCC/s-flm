#!/usr/bin/env python
"""Histograms of per-word embedding lengths for an HFLM run (the figures in
slides/jun25_2026 'Distribution over the Length & Value of Word Embeddings').

For every vocab word v the backbone's polar decomposition gives
rho_v = ||e_v||_2 -- the Euclidean length; the Riemannian (hyperbolic) radius
actually used by the sampler is the soft clamp rho_max * tanh(rho_v / rho_max).
rho_max and gaussian_curvature K are loaded from the run's own config
(.hydra/config.yaml), never hard-coded. Writes 4 figures from one checkpoint
load per step: {_eucl, _hyp} x-axis x {linear, _log} y-scale, one histogram
outline per checkpoint step (dashed line = that step's median; dotted green
line = rho_max). Also writes <out>_rank_{step}.json per step: every vocab word
ranked by length (descending), each entry {token_id, word, eucl_len, riem_len}
(the tanh clamp is monotone, so both lengths give the same ranking).

Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to load the Lightning checkpoints;
run on a compute node.

Example:
  python visualization/embedding_length_dist.py \
    --project outputs/hflm_sweep_tinystories_s256 --run std0.04_pc1.0 \
    --steps 5000 20000 30000 --out <dir>/std0.04_pc1.0
"""
import argparse, glob, json, os, sys

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import (ALGO_BY_NAME, _load_config,  # noqa: E402
                           checkpoint_training_step2tag)
import dataloader, main as main_mod  # noqa: E402  (repo root via loss_geometry)


@torch.no_grad()
def _word_rhos(run_dir: str, step: int, args):
  """rho_v = ||e_v|| for all V words (EMA weights), + (rho_max, K) from config."""
  ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', f'*-{step}.ckpt'))
  assert ckpts, f'no *-{step}.ckpt under {run_dir}/checkpoints'
  cfg = _load_config(run_dir, sorted(ckpts)[0], args)
  assert cfg.algo.name == 'hflm', 'embedding-length histograms expect an HFLM run'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = dataloader.get_tokenizer(cfg)
  model = main_mod._load_from_checkpoint(
    ALGO_BY_NAME[cfg.algo.name], cfg, tokenizer).to(device).eval()
  if model.ema:
    model.ema.move_shadow_params_to_device(device)
  model._eval_mode()  # swap in EMA weights, as eval sees them
  ids = torch.arange(model.vocab_size, device=device).unsqueeze(0)  # [1, V]
  rhos, _ = model.backbone.get_hyperbolic_polar_embeddings(ids)     # [1, V, 1]
  return (rhos.reshape(-1).float().cpu().numpy(),
          float(model.rho_max), float(model.gaussian_curvature), tokenizer)


def _word_str(tokenizer, i: int) -> str:
  try:
    return tokenizer.decode([i])
  except Exception:  # ids past the base vocab (e.g. an added mask token)
    return f'<id_{i}>'


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--project', required=True)
  p.add_argument('--run', required=True)
  p.add_argument('--steps', type=int, nargs='+', required=True)
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--bins', type=int, default=80)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()
  args.batch_size = 16  # required by the shared _load_config; no dataloader here

  run_dir = os.path.join(args.project, args.run)
  tags = [checkpoint_training_step2tag(s) for s in args.steps]
  eucl, rho_max, K, tokenizer = {}, None, None, None
  for step, tag in zip(args.steps, tags):
    eucl[tag], rho_max, K, tokenizer = _word_rhos(run_dir, step, args)
    print(f'[{tag}] V={eucl[tag].size}  median ||e||={np.median(eucl[tag]):.3f}'
          f'  max={eucl[tag].max():.2f}  rho_max={rho_max:g}  K={K:g}',
          flush=True)

  values = {'eucl': eucl,
            'hyp': {t: rho_max * np.tanh(r / rho_max) for t, r in eucl.items()}}
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  for tag in tags:  # all V words ranked by length (descending), one file per step
    e, h = eucl[tag], values['hyp'][tag]
    rank = [dict(token_id=int(i), word=_word_str(tokenizer, int(i)),
                 eucl_len=round(float(e[i]), 6), riem_len=round(float(h[i]), 6))
            for i in np.argsort(-e)]
    with open(f'{args.out}_rank_{tag}.json', 'w') as f:
      json.dump(rank, f, indent=1, ensure_ascii=False)
    print(f'wrote {args.out}_rank_{tag}.json')
  xlabels = {'eucl': r'$\|e_v\|_2$ (Euclidean length $= \rho_v$)',
             'hyp': r'hyperbolic radius $\rho_{max}\tanh(\rho_v/\rho_{max})$'}
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  for kind in ('eucl', 'hyp'):
    vals = values[kind]
    hi = max(max(v.max() for v in vals.values()), rho_max) * 1.02
    bins = np.linspace(0.0, hi, args.bins + 1)
    for log_y in (False, True):
      fig, ax = plt.subplots(figsize=(6, 4.5))
      for tag, v in vals.items():
        _, _, patches = ax.hist(v, bins=bins, histtype='step', lw=1.5, label=tag)
        ax.axvline(np.median(v), color=patches[0].get_edgecolor(),
                   ls='--', lw=1.0)
      ax.axvline(rho_max, color='green', ls=':', lw=1.5,
                 label=f'rho_max={rho_max:g}')
      ax.set(xlabel=xlabels[kind], ylabel='# words',
             title=f'{args.run} word-embedding lengths '
                   f'(rho_max={rho_max:g}, K={K:g})')
      if log_y:
        ax.set_yscale('log')
      ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
      out_png = f'{args.out}_{kind}{"_log" if log_y else ""}.png'
      fig.savefig(out_png, dpi=150); plt.close(fig)
      print(f'wrote {out_png}')


if __name__ == '__main__':
  main()
