#!/usr/bin/env python
"""Per-word denoising loss vs word frequency and vs embedding length.

For one HFLM run, computes each vocab word's mean denoising loss at a single
pinned flow-time t (default 1.0 = pure noise) exactly as loss_geometry.py does
(same nll-bucketing hook, val batches, and seed), plus each word's embedding
length ||e_v||_2 from the same checkpoint load, and its training-set frequency
(embedding_length_dist._train_freq). Writes, under the --out prefix:

  <out>_loss_vs_freq.png     scatter Y: per-word mean loss, X: train token
                             ratio (log x), one cloud per checkpoint step
  <out>_loss_vs_len.png      scatter Y: per-word mean loss, X: ||e_v||_2
  <out>_loss_vs_riemlen.png  scatter Y: per-word mean loss, X: the Riemannian
                             (clamped hyperbolic) radius rho_max*tanh(.)
  <out>_words_{step}.json    occurring words sorted by loss (descending), each
                             {token_id, word, loss, count, train_freq, eucl_len}

Only words that occur as targets in the evaluated val batches get a loss (and
appear in the outputs); the frequency scatter additionally drops words absent
from the sampled training batches (log axis). Needs
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; run on a compute node.

Example:
  python visualization/loss2lenfreq.py \
    --project outputs/hflm_sweep_tinystories_s256 --run std0.04_pc1.0 \
    --steps 5000 20000 30000 --out experiments/loss2lenfreq/std0.04_pc1.0
"""
import argparse, glob, itertools, json, os, sys

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import (ALGO_BY_NAME, _install_word_loss_hook,  # noqa: E402
                           _load_config, _pin_time,
                           checkpoint_training_step2tag)
from embedding_length_dist import (RunEmbeddings, _save,  # noqa: E402
                                   _train_freq, _word_str)
import dataloader, main as main_mod  # noqa: E402


@torch.no_grad()
def _word_losses_and_rhos(run_dir: str, step: int, t: float, args):
  """One checkpoint -> (per-word mean loss at pinned t [V] (nan = word never a
  target), per-word target count [V], ||e_v|| [V], rho_max, K, cfg, tokenizer).
  Mirrors loss_geometry._loss_curve at a single t."""
  ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', f'*-{step}.ckpt'))
  assert ckpts, f'no *-{step}.ckpt under {run_dir}/checkpoints'
  cfg = _load_config(run_dir, sorted(ckpts)[0], args)
  assert cfg.algo.name == 'hflm', 'loss2lenfreq expects an HFLM run'
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
  _pin_time(model, cfg, t, step)
  torch.manual_seed(cfg.seed)  # same prior-noise draws as loss_geometry.py
  for b in batches:
    model._loss(b['input_ids'], b['attention_mask'])
  loss = torch.full_like(model._word_sum, float('nan'))
  occ = model._word_cnt > 0
  loss[occ] = model._word_sum[occ] / model._word_cnt[occ]
  ids = torch.arange(model.vocab_size, device=device).unsqueeze(0)
  rhos, _ = model.backbone.get_hyperbolic_polar_embeddings(ids)
  return (loss.cpu().numpy(), model._word_cnt.cpu().numpy(),
          rhos.reshape(-1).float().cpu().numpy(),
          float(model.rho_max), float(model.gaussian_curvature), cfg, tokenizer)


def write_words_json(emb: RunEmbeddings, losses, counts, freq, out: str):
  """<out>_words_{tag}.json: occurring words sorted by loss, descending."""
  for tag in emb.tags:
    L, cnt, e = losses[tag], counts[tag], emb.eucl[tag]
    occ = np.nonzero(~np.isnan(L))[0]
    rows = [dict(token_id=int(i), word=_word_str(emb.tokenizer, int(i)),
                 loss=round(float(L[i]), 6), count=int(cnt[i]),
                 train_freq=float(freq[i]), eucl_len=round(float(e[i]), 6))
            for i in occ[np.argsort(-L[occ])]]
    path = f'{out}_words_{tag}.json'
    with open(path, 'w') as f:
      json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'wrote {path}')


def plot_scatters(emb: RunEmbeddings, losses, freq, t: float, out: str):
  """Loss-vs-frequency (log x) and loss-vs-{Euclidean,Riemannian}-length."""
  hyp = emb.hyp
  for xkind, xlabel, suffix in (
      ('freq', 'training-set frequency (token ratio)', 'loss_vs_freq'),
      ('len', r'$\|e_v\|_2$ (embedding length)', 'loss_vs_len'),
      ('riem', r'hyperbolic radius $\rho_{max}\tanh(\rho_v/\rho_{max})$',
       'loss_vs_riemlen')):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag in emb.tags:
      L = losses[tag]
      x = (freq if xkind == 'freq' else
           emb.eucl[tag] if xkind == 'len' else hyp[tag])
      m = ~np.isnan(L) & ((x > 0) if xkind == 'freq' else True)
      ax.scatter(x[m], L[m], s=4, alpha=0.35, edgecolors='none', label=tag)
    if xkind == 'freq':
      ax.set_xscale('log')
    ax.set(xlabel=xlabel, ylabel=f'per-word mean loss at t={t:g}',
           title=f'{emb.run_name} loss vs {xkind} at t={t:g} '
                 f'({emb.const_label})')
    _save(fig, ax, f'{out}_{suffix}.png')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--project', required=True)
  p.add_argument('--run', required=True)
  p.add_argument('--steps', type=int, nargs='+', required=True)
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--t', type=float, default=1.0, help='pinned flow-time')
  p.add_argument('--num-batches', type=int, default=8)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--freq-batches', type=int, default=512)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()

  run_dir = os.path.join(args.project, args.run)
  tags = [checkpoint_training_step2tag(s) for s in args.steps]
  losses, counts, eucl = {}, {}, {}
  rho_max = K = cfg = tokenizer = None
  for step, tag in zip(args.steps, tags):
    (losses[tag], counts[tag], eucl[tag],
     rho_max, K, cfg, tokenizer) = _word_losses_and_rhos(
       run_dir, step, args.t, args)
    print(f'[{tag}] t={args.t:g}  occurring words='
          f'{int((~np.isnan(losses[tag])).sum())}  '
          f'loss mean={np.nanmean(losses[tag]):.3f}  range='
          f'[{np.nanmin(losses[tag]):.3f}, {np.nanmax(losses[tag]):.3f}]',
          flush=True)

  emb = RunEmbeddings(args.run, tags, eucl, rho_max, K, cfg, tokenizer)
  freq = _train_freq(emb, args.freq_batches)
  print(f'frequency over {args.freq_batches} train batches: '
        f'{int((freq > 0).sum())} words with nonzero count', flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  write_words_json(emb, losses, counts, freq, args.out)
  plot_scatters(emb, losses, freq, args.t, args.out)


if __name__ == '__main__':
  main()
