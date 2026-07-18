#!/usr/bin/env python
"""Per-word TV_lin and t10 vs Riemannian length and training frequency (HFLM).

For EVERY word occurring in the eval batches (count >= 1, no frequency filter),
compute its loss-geometry curve L_v(t) (per-token NLL bucketed by clean target
x_0; same hook/seed/batches as loss_geometry.py, one checkpoint load, cached
in <out>_perword_{step}.npz) and two statistics of the ceiling-normalized
curve g_v = L_v / L_v(1):

    TV_v  = total variation of g_v'(t) = integral of |g_v''|
            (0 = perfectly linear ramp, ~2/dt = one-step cliff)
    t10_v / t90_v = first t with g_v >= 0.1 / 0.9  (onset / completion)
    bandwidth_v = t90_v - t10_v      (width of the top transition)
    loss_v = L_v(t=1)                (per-word mean NLL at pure noise, the
                                      same bucketed quantity loss_geometry.py
                                      computes per word)

Draws one 5x2 panel figure <out>_t10_tv_vs_len_freq.png:
    (TV        vs Riemannian length)   (TV        vs train frequency, log x)
    (t10       vs Riemannian length)   (t10       vs train frequency, log x)
    (bandwidth vs Riemannian length)   (bandwidth vs train frequency, log x)
    (t90       vs Riemannian length)   (t90       vs train frequency, log x)
    (loss(t=1) vs Riemannian length)   (loss(t=1) vs train frequency, log x)
Riemannian length = rho_max * tanh(||e_v|| / rho_max), the sampler's clamped
radius (rho_max read from the run's config). Points are colored by
log10(count in the eval batches): low-count words have noisy few-sample
curves, which inflates TV and quantizes t10 -- the color makes that visible
instead of filtering it away. Words absent from the sampled training batches
(freq = 0) cannot sit on a log axis and are dropped from the two frequency
panels only (count printed).

Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; run on a compute node when the
per-word cache has to be (re)computed.

Example:
  python visualization/t10_tv_freq_len.py \
    --project outputs/hflm_sweep_tinystories_s256 --run std0.04_pc1.0 \
    --step 30000 --out experiments/curv_loss_geo/std0.04_pc1.0
"""
import argparse, glob, itertools, os, sys

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import (ALGO_BY_NAME, _install_word_loss_hook,  # noqa: E402
                           _load_config, _pin_time,
                           checkpoint_training_step2tag)
from embedding_length_dist import RunEmbeddings, _train_freq  # noqa: E402
import dataloader, main as main_mod  # noqa: E402


@torch.no_grad()
def compute_perword(run_dir: str, step: int, args):
  """[33, V] per-word loss sums/counts at pinned t, + lengths and train freqs."""
  ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', f'*-{step}.ckpt'))
  assert ckpts, f'no *-{step}.ckpt under {run_dir}/checkpoints'
  cfg = _load_config(run_dir, sorted(ckpts)[0], args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = dataloader.get_tokenizer(cfg)
  model = main_mod._load_from_checkpoint(
    ALGO_BY_NAME[cfg.algo.name], cfg, tokenizer).to(device).eval()
  if model.ema:
    model.ema.move_shadow_params_to_device(device)
  model._eval_mode()
  _, valid_dl = dataloader.get_dataloaders(
    cfg, tokenizer, skip_train=True, valid_seed=cfg.seed)
  batches = [{k: v.to(device) for k, v in b.items()}
             for b in itertools.islice(iter(valid_dl), args.num_batches)]
  _install_word_loss_hook(model)
  model._word_sum = torch.zeros(model.vocab_size, device=device)
  model._word_cnt = torch.zeros(model.vocab_size, device=device)
  if cfg.algo.name == 'hflm':
    ids = torch.arange(model.vocab_size, device=device).unsqueeze(0)
    rhos, _ = model.backbone.get_hyperbolic_polar_embeddings(ids)
    eucl = rhos.detach().reshape(-1).float().cpu().numpy()
  else:  # embedding-length x-axis is HFLM-only; the freq figure needs only sums/cnts
    eucl = np.zeros(model.vocab_size)
  t_grid = np.linspace(0.001, 1.0, 33)
  sums, cnts = [], []
  for t in t_grid:
    _pin_time(model, cfg, t, step)
    torch.manual_seed(cfg.seed)  # same prior-noise draws as loss_geometry.py
    model._word_sum.zero_(); model._word_cnt.zero_()
    model._wi_sum = model._wi_sq = 0.0
    model._wi_min, model._wi_max = float('inf'), float('-inf')
    for b in batches:
      model._loss(b['input_ids'], b['attention_mask'])
    sums.append(model._word_sum.cpu().numpy())
    cnts.append(model._word_cnt.cpu().numpy())
    print(f'  t={t:.3f} done', flush=True)
  emb = RunEmbeddings(os.path.basename(run_dir), ['x'], {'x': eucl},
                      float(getattr(model, 'rho_max', float('nan'))),
                      float(getattr(model, 'gaussian_curvature', float('nan'))),
                      cfg, tokenizer)
  freq = _train_freq(emb, args.freq_batches)
  return t_grid, np.array(sums), np.array(cnts), eucl, freq


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--project', required=True)
  p.add_argument('--run', required=True)
  p.add_argument('--step', type=int, required=True)
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--num-batches', type=int, default=8)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--freq-batches', type=int, default=8192)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()

  tag = checkpoint_training_step2tag(args.step)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  cache = f'{args.out}_perword_{tag}.npz'
  if os.path.exists(cache):
    d = np.load(cache)
    t, sums, cnts, eucl, freq = d['t'], d['sums'], d['cnts'], d['eucl'], d['freq']
  else:
    t, sums, cnts, eucl, freq = compute_perword(
      os.path.join(args.project, args.run), args.step, args)
    np.savez_compressed(cache, t=t, sums=sums, cnts=cnts, eucl=eucl, freq=freq)
    print(f'wrote {cache}')

  cfg = OmegaConf.load(os.path.join(args.project, args.run,
                                    '.hydra', 'config.yaml'))
  rho_max = float(cfg.algo.get('rho_max', 12.0))
  riem = rho_max * np.tanh(eucl / rho_max)

  cnt = cnts[0]
  sel = np.where(cnt >= 1)[0]                       # EVERY occurring word
  L = sums[:, sel] / cnts[:, sel]
  ok = L[-1] > 0
  sel, L = sel[ok], L[:, ok]
  g = L / L[-1]
  dt = t[1] - t[0]
  gp = np.diff(g, axis=0) / dt
  tv = np.abs(np.diff(gp, axis=0)).sum(0)           # per-word TV_lin

  def crossing(col, q):
    i = np.argmax(col >= q)
    if col[i] < q: return 1.0
    if i == 0: return t[0]
    return t[i-1] + (q - col[i-1]) / max(col[i] - col[i-1], 1e-9) * (t[i] - t[i-1])
  t10 = np.array([crossing(g[:, j], 0.1) for j in range(g.shape[1])])
  t90 = np.array([crossing(g[:, j], 0.9) for j in range(g.shape[1])])
  width = t90 - t10                                 # per-word bandwidth

  x_len, x_freq, color = riem[sel], freq[sel], np.log10(cnt[sel])
  fpos = x_freq > 0
  print(f'occurring words: {len(sel)}; dropped from freq panels (freq=0): '
        f'{int((~fpos).sum())}')

  xlab_len = r'Riemannian length $\rho_{max}\tanh(\|e_v\|/\rho_{max})$'
  panels = [
    (tv, x_len, np.ones_like(fpos), False, 'log', 'TV_v', xlab_len),
    (tv, x_freq, fpos, True, 'log', 'TV_v', 'train frequency (token ratio)'),
    (t10, x_len, np.ones_like(fpos), False, None, 't10_v', xlab_len),
    (t10, x_freq, fpos, True, None, 't10_v', 'train frequency (token ratio)'),
    (width, x_len, np.ones_like(fpos), False, None, 'bandwidth (t90-t10)',
     xlab_len),
    (width, x_freq, fpos, True, None, 'bandwidth (t90-t10)',
     'train frequency (token ratio)'),
    (t90, x_len, np.ones_like(fpos), False, None, 't90_v', xlab_len),
    (t90, x_freq, fpos, True, None, 't90_v', 'train frequency (token ratio)'),
    (L[-1], x_len, np.ones_like(fpos), False, None, 'per-word loss at t=1',
     xlab_len),
    (L[-1], x_freq, fpos, True, None, 'per-word loss at t=1',
     'train frequency (token ratio)'),
  ]
  fig, axes = plt.subplots(5, 2, figsize=(11, 20),
                           gridspec_kw=dict(hspace=0.4, wspace=0.25))
  for ax, (y, x, m, logx, logy, yname, xname) in zip(axes.reshape(-1), panels):
    r = spearmanr(x[m], y[m])
    print(f'Spearman({yname}, {xname}) = {r.statistic:+.3f} '
          f'(p={r.pvalue:.1e}, n={int(m.sum())})')
    sc = ax.scatter(x[m], y[m], c=color[m], s=6, alpha=0.5,
                    edgecolors='none', cmap='viridis')
    if logx:
      ax.set_xscale('log')
    if logy == 'log':
      ax.set_yscale('log')
    elif logy == 'exp':  # stretch the top: y -> exp(k*y)
      k = 5.0
      ax.set_yscale('function',
                    functions=(lambda y, k=k: np.exp(k * y),
                               lambda y, k=k: np.log(np.maximum(y, 1e-12)) / k))
      ax.set_yticks(np.arange(0.0, 0.71, 0.1))  # function scale breaks autoticks
    ax.set(xlabel=xname, ylabel=yname,
           title=f'{yname}  (Spearman {r.statistic:+.2f})')
    ax.grid(alpha=0.3)
  fig.colorbar(sc, ax=axes, label='log10 count in eval batches', shrink=0.85)
  fig.suptitle(f'{args.run} {tag}: per-word TV / t10 / bandwidth / t90 / '
               f'loss(t=1) vs Riemannian length / frequency', fontsize=11)
  out_png = f'{args.out}_t10_tv_vs_len_freq.png'
  fig.savefig(out_png, dpi=150); plt.close(fig)
  print(f'wrote {out_png}')


if __name__ == '__main__':
  main()
