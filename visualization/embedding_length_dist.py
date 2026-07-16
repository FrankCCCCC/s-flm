#!/usr/bin/env python
"""Word-embedding length distributions for an HFLM run (the figures in
slides/jun25_2026 'Distribution over the Length & Value of Word Embeddings').

For every vocab word v the backbone's polar decomposition gives
rho_v = ||e_v||_2 -- the Euclidean length; the Riemannian (hyperbolic) radius
actually used by the sampler is the soft clamp rho_max * tanh(rho_v / rho_max).
rho_max and gaussian_curvature K are loaded from the run's own config
(.hydra/config.yaml), never hard-coded.

Outputs (one checkpoint load per step):
  <out>_{allvocab,trainonly}_{eucl,hyp}[_log].png
                               length histograms, one outline per step (dashed
                               line = that step's median; dotted green =
                               rho_max). Two coverage modes tagged in the file
                               name: `allvocab` = every row of the embedding
                               matrix; `trainonly` = only words that appeared in
                               the sampled training batches (freq > 0).
  <out>_rank_{step}.json       all words ranked by length (descending), each
                               entry {token_id, word, eucl_len, riem_len}; the
                               tanh clamp is monotone so both lengths rank alike
  <out>_freq_{eucl,hyp}.png    scatter of length vs training-set frequency
                               (token ratio, log x; estimated over
                               --freq-batches training batches; words absent
                               from those batches are dropped)

Needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to load the Lightning checkpoints;
run on a compute node.

Example:
  python visualization/embedding_length_dist.py \
    --project outputs/hflm_sweep_tinystories_s256 --run std0.04_pc1.0 \
    --steps 5000 20000 30000 --out experiments/embed_len_dist/std0.04_pc1.0
"""
import argparse, glob, itertools, json, os, sys
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss_geometry import (ALGO_BY_NAME, _load_config,  # noqa: E402
                           checkpoint_training_step2tag)
import dataloader, main as main_mod  # noqa: E402  (repo root via loss_geometry)

XLABELS = {'eucl': r'$\|e_v\|_2$ (Euclidean length $= \rho_v$)',
           'hyp': r'hyperbolic radius $\rho_{max}\tanh(\rho_v/\rho_{max})$'}


@torch.no_grad()
def _checkpoint_rhos(run_dir: str, step: int, args):
  """rho_v = ||e_v|| for all V words (EMA weights), + run constants."""
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
          float(model.rho_max), float(model.gaussian_curvature), cfg, tokenizer)


@dataclass
class RunEmbeddings:
  """||e_v|| of every vocab word per checkpoint step, for one HFLM run."""
  run_name: str
  tags: List[str]
  eucl: Dict[str, np.ndarray]  # tag -> [V] raw lengths (= polar rho_v)
  rho_max: float
  K: float
  cfg: Any
  tokenizer: Any

  @classmethod
  def load(cls, project: str, run: str, steps: List[int], args):
    run_dir = os.path.join(project, run)
    tags = [checkpoint_training_step2tag(s) for s in steps]
    eucl = {}
    for step, tag in zip(steps, tags):
      eucl[tag], rho_max, K, cfg, tokenizer = _checkpoint_rhos(run_dir, step, args)
      print(f'[{tag}] V={eucl[tag].size}  median ||e||={np.median(eucl[tag]):.3f}'
            f'  max={eucl[tag].max():.2f}  rho_max={rho_max:g}  K={K:g}',
            flush=True)
    return cls(run, tags, eucl, rho_max, K, cfg, tokenizer)

  @property
  def hyp(self) -> Dict[str, np.ndarray]:
    """The sampler's soft-clamped hyperbolic radius, per step."""
    return {t: self.rho_max * np.tanh(e / self.rho_max)
            for t, e in self.eucl.items()}

  def lengths(self, kind: str) -> Dict[str, np.ndarray]:
    return self.eucl if kind == 'eucl' else self.hyp

  @property
  def vocab_size(self) -> int:
    return self.eucl[self.tags[0]].size

  @property
  def const_label(self) -> str:
    return f'rho_max={self.rho_max:g}, K={self.K:g}'


def _word_str(tokenizer, i: int) -> str:
  try:
    return tokenizer.decode([i])
  except Exception:  # ids past the base vocab (e.g. an added mask token)
    return f'<id_{i}>'


@torch.no_grad()
def _train_freq(emb: RunEmbeddings, num_batches: int) -> np.ndarray:
  """Empirical per-word token ratio over `num_batches` training batches.

  Builds the train loader straight from the dataset instead of
  dataloader.get_dataloaders, whose DDP global-batch assert divides by
  torch.cuda.device_count() and so needs a live GPU; a frequency count is
  single-process and runs fine on a CPU node.
  """
  train_set = dataloader.get_dataset(emb.cfg, emb.tokenizer, mode='train')
  train_dl = torch.utils.data.DataLoader(
    train_set, batch_size=emb.cfg.loader.batch_size,
    num_workers=emb.cfg.loader.num_workers,
    shuffle=not emb.cfg.data.streaming)
  cnt = torch.zeros(emb.vocab_size)
  for b in itertools.islice(iter(train_dl), num_batches):
    ids = b['input_ids'].reshape(-1)
    m = b['attention_mask'].reshape(-1).bool()
    cnt += torch.bincount(ids[m], minlength=emb.vocab_size).float()
  return (cnt / cnt.sum().clamp(min=1)).numpy()


def _save(fig, ax, path: str):
  ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
  fig.savefig(path, dpi=150); plt.close(fig)
  print(f'wrote {path}')


def write_rankings(emb: RunEmbeddings, out: str):
  """<out>_rank_{tag}.json: all words sorted by length, descending."""
  hyp = emb.hyp
  for tag in emb.tags:
    e, h = emb.eucl[tag], hyp[tag]
    rank = [dict(token_id=int(i), word=_word_str(emb.tokenizer, int(i)),
                 eucl_len=round(float(e[i]), 6), riem_len=round(float(h[i]), 6))
            for i in np.argsort(-e)]
    path = f'{out}_rank_{tag}.json'
    with open(path, 'w') as f:
      json.dump(rank, f, indent=1, ensure_ascii=False)
    print(f'wrote {path}')


def plot_histograms(emb: RunEmbeddings, out: str, bins: int, freq: np.ndarray):
  """<out>_{allvocab,trainonly}_{eucl,hyp}[_log].png: per-step length histograms.

  Two coverage modes (tagged in the file name): `allvocab` = every row of the
  embedding matrix; `trainonly` = only words with freq > 0 (appeared in the
  sampled training batches). The bin range is shared across modes so the two
  figures are directly comparable.
  """
  covers = {'allvocab': np.ones(emb.vocab_size, dtype=bool),
            'trainonly': freq > 0}
  for kind in ('eucl', 'hyp'):
    vals = emb.lengths(kind)
    hi = max(max(v.max() for v in vals.values()), emb.rho_max) * 1.02
    edges = np.linspace(0.0, hi, bins + 1)
    for cov, mask in covers.items():
      for log_y in (False, True):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for tag, v in vals.items():
          vm = v[mask]
          _, _, patches = ax.hist(vm, bins=edges, histtype='step', lw=1.5,
                                  label=tag)
          ax.axvline(np.median(vm), color=patches[0].get_edgecolor(),
                     ls='--', lw=1.0)
        ax.axvline(emb.rho_max, color='green', ls=':', lw=1.5,
                   label=f'rho_max={emb.rho_max:g}')
        ax.set(xlabel=XLABELS[kind], ylabel='# words',
               title=f'{emb.run_name} word-embedding lengths '
                     f'[{cov}: n={int(mask.sum())}] ({emb.const_label})')
        if log_y:
          ax.set_yscale('log')
        _save(fig, ax, f'{out}_{cov}_{kind}{"_log" if log_y else ""}.png')


def plot_freq_scatter(emb: RunEmbeddings, out: str, freq: np.ndarray):
  """<out>_freq_{eucl,hyp}.png: length vs train-set frequency (log x)."""
  pos = freq > 0
  for kind in ('eucl', 'hyp'):
    vals = emb.lengths(kind)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag in emb.tags:
      ax.scatter(freq[pos], vals[tag][pos], s=4, alpha=0.35,
                 edgecolors='none', label=tag)
    ax.axhline(emb.rho_max, color='green', ls=':', lw=1.5,
               label=f'rho_max={emb.rho_max:g}')
    ax.set_xscale('log')
    ax.set(xlabel='training-set frequency (token ratio)', ylabel=XLABELS[kind],
           title=f'{emb.run_name} length vs train frequency ({emb.const_label})')
    _save(fig, ax, f'{out}_freq_{kind}.png')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--project', required=True)
  p.add_argument('--run', required=True)
  p.add_argument('--steps', type=int, nargs='+', required=True)
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--bins', type=int, default=80)
  p.add_argument('--freq-batches', type=int, default=512,
                 help='training batches used to estimate word frequencies '
                      'for the length-vs-frequency scatter')
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()
  args.batch_size = 16  # required by the shared _load_config; no eval batches here

  emb = RunEmbeddings.load(args.project, args.run, args.steps, args)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  freq = _train_freq(emb, args.freq_batches)
  print(f'frequency over {args.freq_batches} train batches: '
        f'{int((freq > 0).sum())} words appeared in training', flush=True)
  write_rankings(emb, args.out)
  plot_histograms(emb, args.out, args.bins, freq)
  plot_freq_scatter(emb, args.out, freq)


if __name__ == '__main__':
  main()
