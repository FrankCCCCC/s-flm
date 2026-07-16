#!/usr/bin/env python
"""Loss-geometry curves (LangFlow Fig. 2, Left): loss vs flow time t, one curve
per checkpoint. Reuses the standard eval pipeline (main._load_from_checkpoint,
dataloader, the algo's own training `_loss`); only t is pinned to a grid
instead of sampled. Convention (invert_time_convention=false): t=0 clean,
t=1 pure noise, so curves read left(clean) -> right(noise), like the paper.

  --mode steps         checkpoints of one run by training step
  --mode hflm_init_pc  HFLM sweep cells by (step, init_std, prior_cov);
                       dirs are std{init}_pc{pc} / ngpt_pc{pc}, literal strings

Writes <out>.json (cached curves), <out>.png (linear y), <out>_log.png (log y).
--y-metric word_loss_std plots the across-vocab distribution of each word's mean
denoising loss (the per-token NLL bucketed by clean target x_0, then averaged
per word): solid line = mean over the occurring words, shaded band = +-std,
whiskers = min/max; suffix _wordstd.
--word-idx N restricts both y-metrics to the single word embedding N: the curve
is N's per-token NLL over the positions where N is the clean target -- 'loss'
draws its mean, 'word_loss_std' adds the +-std band and min/max whiskers over
those positions; outputs get a _w{N} suffix (own cache).
GPU forward passes -> run on a compute node (visualization/loss_geometry.sbatch).
"""
import argparse, glob, itertools, json, math, os, sys, types
from typing import List

import matplotlib, numpy as np, torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # repo-root imports below need this bootstrap first
import algo, dataloader, main as main_mod  # main registers omegaconf resolvers
from geo_bridge import GeoUtils  # curvature-aware manifold coordinate converters

# Dispatch on config.algo.name (same mapping main.main() uses).
ALGO_BY_NAME = {'ar': algo.AR, 'mdlm': algo.MDLM, 'duo_base': algo.DUO_BASE,
                'sfm': algo.SFM, 'eflm': algo.EFLM, 'langflow': algo.LangFlow,
                'hflm': algo.HFLM, 'flm': algo.FLM, 'candi': algo.CANDI}

B: int = 1_000_000_000
M: int = 1_000_000
K: int = 1_000
def checkpoint_training_step2tag(step: int) -> str:
  if step >= B:
    return f'{step / B:g}B'
  elif step >= M:
    return f'{step / M:g}M'
  elif step >= K:
    return f'{step / K:g}K'
  return str(step)


def _load_config(run_dir: str, ckpt: str, args):
  """Each run's exact training config (.hydra/config.yaml) + eval overrides.

  Loading the real config is what makes strict checkpoint loading work for the
  adaptive-spline / truncated / Gumbel-scheduler runs (their noise state lives
  in the checkpoint and must have a matching module to load into).
  """
  cfg = OmegaConf.load(os.path.join(run_dir, '.hydra', 'config.yaml'))
  OmegaConf.set_struct(cfg, False)
  cfg.eval.checkpoint_path = ckpt
  cfg.eval.strict_loading = True
  cfg.data.cache_dir = args.cache_dir
  cfg.trainer.devices, cfg.trainer.num_nodes = 1, 1
  cfg.trainer.accumulate_grad_batches = 1
  cfg.loader.num_workers = 4
  for k in ('global_batch_size', 'eval_global_batch_size',
            'batch_size', 'eval_batch_size'):
    cfg.loader[k] = args.batch_size
  if cfg.algo.name == 'hflm' and cfg.algo.get('gaussian_curvature') is None:
    cfg.algo.gaussian_curvature = -1.0  # pre-knob HFLM configs = unit hyperboloid
  if cfg.algo.get('self_conditioning') is None:
    cfg.algo.self_conditioning = False  # pre-knob configs: no self-conditioning
  if cfg.algo.get('p_self_cond') is None:
    cfg.algo.p_self_cond = 0.0
  return cfg


def _pin_time(model, cfg, t, step):
  """Make the algo evaluate at a single fixed flow-time t (its own schedule).

  SFM/EFLM/HFLM: pin `_sample_t`; the restored noise schedule maps t->alpha_t.
  LangFlow: it draws gamma=logNSR (not t), so pin `sample_gamma` to the run's
  own Gumbel quantile gamma(t)=P_mu - P_beta*log(-log t) (in-distribution).
  """
  if cfg.algo.name == 'langflow':
    ns = model.noise
    tt = min(max(float(t), ns.q_clip), 1 - ns.q_clip)
    gamma = (ns.P_mu - ns.P_beta * math.log(-math.log(tt))).item()
    model.noise.sample_gamma = (
      lambda n, device, antithetic=True, _g=gamma:
        torch.full((n,), _g, device=device))
    # Plaid logit-bias ramp uses global_step (0 without a trainer) -> pin to the
    # checkpoint's real step so r matches training (all our steps >= warmup).
    w = float(cfg.algo.get('logit_bias_warmup_steps', 0) or 0)
    r = 1.0 if w <= 0 else min(1.0, step / w)
    model._logit_bias_r = types.MethodType(lambda self, _r=r: _r, model)
  else:
    model._sample_t = types.MethodType(
      lambda self, n, accum, _t=float(t):
        torch.full((n,), _t, device=self.device), model)


def _riemannian_dist(model, algo_name, a0, xt):
  """Per-token geodesic distance from the corrupted sample x_t to its clean
  target x_0 on the algo's own manifold, as [B, L]. `a0` is q_xt's first
  positional arg: the token ids for SFM/EFLM/HFLM (x_0 = the clean embedding),
  or the clean embedding z itself for LangFlow. HFLM uses the hyperbolic
  distance at the config's Gaussian curvature K; SFM the unit-sphere arc length;
  EFLM/LangFlow the Euclidean norm (their x_t is a Euclidean lerp toward x_0)."""
  if algo_name == 'langflow':                       # Euclidean VP latent; a0 = z
    return (xt - a0).norm(dim=-1)
  if algo_name == 'eflm':                            # Euclidean raw-embedding lerp
    x0 = model.backbone.get_raw_embeddings(a0)
    return (xt - x0).norm(dim=-1)
  if algo_name == 'sfm':                             # unit-sphere geodesic
    x0 = model.backbone.get_sphere_embeddings(a0)
    cos = (xt * x0).sum(-1) / (xt.norm(dim=-1) * x0.norm(dim=-1)).clamp_min(1e-12)
    return torch.arccos(cos.clamp(-1.0, 1.0))
  if algo_name == 'hflm':                            # hyperbolic distance at K
    K = model.gaussian_curvature  # loaded from the run's config (< 0)
    R = 1.0 / math.sqrt(abs(K))   # model radius = 1/sqrt(|K|)
    rhos, thetas = model.backbone.get_hyperbolic_polar_embeddings(a0)
    rhos_c = model._rho_clamp(rhos).squeeze(-1)      # same soft radial clamp as q_xt
    X0 = GeoUtils.hyperbolic_polar_to_lorentz_cartesian(rhos_c, thetas, K).double()
    Xt = GeoUtils.poincare_cartesian_to_lorentz_cartesian(xt.double(), K)  # x_t is Poincaré
    diff = Xt - X0  # Minkowski differential form d = R*arccosh(-<X_t,X_0>_L/R^2)
    inner = -diff[..., 0] * diff[..., 0] + (diff[..., 1:] * diff[..., 1:]).sum(-1)
    return R * torch.arccosh((1.0 + inner / (2.0 * R * R)).clamp_min(1.0))
  raise ValueError(f'riem_dist x-axis unsupported for algo {algo_name!r}')


def _install_xt_hook(model, cfg):
  """Wrap q_xt to accumulate, over valid/corrupted tokens, both the per-token L2
  norm of the corrupted sample x_t and the Riemannian distance from x_t to the
  clean target x_0 (see `_riemannian_dist`). Reset _xt_sum/_dist_sum/_xt_cnt
  before each t; the means are _xt_sum/_xt_cnt and _dist_sum/_xt_cnt (one shared
  count -- both reduce over the same valid-token mask)."""
  orig = model.q_xt
  name = cfg.algo.name
  def wrapped(*a, **k):
    xt = orig(*a, **k)
    vt = k.get('valid_tokens')
    if vt is None and len(a) >= 4:  # LangFlow passes valid_tokens positionally
      vt = a[3]
    norm = xt.float().norm(dim=-1)                   # [B, L] over the embedding dim
    dist = _riemannian_dist(model, name, a[0], xt)   # [B, L] on the algo manifold
    if vt is not None:
      m = vt.bool()
      model._xt_sum += (norm * m).sum().item()
      model._dist_sum += (dist * m).sum().item()
      model._xt_cnt += int(m.sum())
    else:
      model._xt_sum += norm.sum().item()
      model._dist_sum += dist.sum().item()
      model._xt_cnt += norm.numel()
    return xt
  model.q_xt = wrapped


def _install_word_loss_hook(model, word_idx=None):
  """Wrap `nll` to bucket the per-token denoising NLL by clean target word x_0.

  For each valid/corrupted token it adds the per-token NLL into a per-vocabulary
  sum/count ([V] each). The clean target token id per position is `output_tokens`
  (AR) or, when that is None, `x0` (the diffusion/FLM algos delete `output_tokens`
  and treat x_0 as the target). Reset _word_sum/_word_cnt before each t; the
  per-word mean loss is _word_sum/_word_cnt over words that occur; y-metric
  'word_loss_std' plots mean/std/min/max of those per-word means across V.
  With `word_idx`, additionally accumulates the sum/sumsq/min/max of the
  per-token NLL at the positions where `word_idx` is the target (_wi_*), so the
  single-word figures can show mean/std/min/max over its positions."""
  orig = model.nll
  def wrapped(*a, **k):
    per_token_nll, t = orig(*a, **k)
    tgt = a[1] if (len(a) > 1 and a[1] is not None) else a[0]  # target word ids [B,L]
    vt = k.get('valid_tokens')
    if vt is None and len(a) >= 6:  # valid_tokens can arrive positionally
      vt = a[5]
    m = vt.bool() if vt is not None else torch.ones_like(tgt, dtype=torch.bool)
    ids = tgt[m].long().reshape(-1)                        # [N] over valid tokens
    vals = per_token_nll.detach().float()[m].reshape(-1)   # [N] matching per-token NLL
    model._word_sum.scatter_add_(0, ids, vals)
    model._word_cnt.scatter_add_(0, ids, torch.ones_like(vals))
    if word_idx is not None:
      col = vals[ids == word_idx]  # the word's NLL at its target positions
      if col.numel():
        model._wi_sum += col.sum().item()
        model._wi_sq += col.square().sum().item()
        model._wi_min = min(model._wi_min, col.min().item())
        model._wi_max = max(model._wi_max, col.max().item())
    return per_token_nll, t
  model.nll = wrapped


@torch.no_grad()
def _loss_curve(run_dir: str, step: int, t_grid: np.ndarray, args):
  """L(t) = token-mean denoising CE on val batches, at each fixed t."""
  ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', f'*-{step}.ckpt'))
  assert ckpts, f'no *-{step}.ckpt under {run_dir}/checkpoints'
  cfg = _load_config(run_dir, sorted(ckpts)[0], args)
  print(f'[{cfg.algo.name} step={step}] {ckpts[0]}', flush=True)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = dataloader.get_tokenizer(cfg)
  model = main_mod._load_from_checkpoint(
    ALGO_BY_NAME[cfg.algo.name], cfg, tokenizer).to(device).eval()
  if model.ema:
    model.ema.move_shadow_params_to_device(device)
  model._eval_mode()  # swap in EMA weights, as trainer.validate does
  _, valid_dl = dataloader.get_dataloaders(
    cfg, tokenizer, skip_train=True, valid_seed=cfg.seed)
  batches = [{k: v.to(device) for k, v in b.items()}
             for b in itertools.islice(iter(valid_dl), args.num_batches)]
  _install_xt_hook(model, cfg)  # records mean per-token |x_t| and d(x_t, x_0) per t
  wi = args.word_idx
  if wi is not None:
    assert 0 <= wi < model.vocab_size, \
      f'--word-idx {wi} outside vocab [0, {model.vocab_size})'
  _install_word_loss_hook(model, wi)  # buckets per-token NLL by clean target x_0
  model._word_sum = torch.zeros(model.vocab_size, device=device)
  model._word_cnt = torch.zeros(model.vocab_size, device=device)
  curve, norms, dists = [], [], []
  wstats = {k: [] for k in ('mean', 'std', 'min', 'max')}  # over V (or positions)
  for t in t_grid:
    _pin_time(model, cfg, t, step)  # fixed t; rest is the algo's own _loss
    torch.manual_seed(cfg.seed)  # same prior-noise draws at every t/ckpt
    model._xt_sum, model._dist_sum, model._xt_cnt = 0.0, 0.0, 0
    model._word_sum.zero_(); model._word_cnt.zero_()
    model._wi_sum = model._wi_sq = 0.0
    model._wi_min, model._wi_max = math.inf, -math.inf
    out = [model._loss(b['input_ids'], b['attention_mask']) for b in batches]
    curve.append(sum(o.nlls.item() for o in out)
                 / sum(o.num_tokens.item() for o in out))
    norms.append(model._xt_sum / max(model._xt_cnt, 1))
    dists.append(model._dist_sum / max(model._xt_cnt, 1))
    if wi is None:  # distribution of per-word mean loss across occurring words
      occ = model._word_cnt > 0  # vocab words that occur as a target at this t
      wm = model._word_sum[occ] / model._word_cnt[occ]  # per-word mean [n_occ]
      stats = (wm.mean().item() if wm.numel() else 0.0,
               wm.std().item() if wm.numel() > 1 else 0.0,
               wm.min().item() if wm.numel() else 0.0,
               wm.max().item() if wm.numel() else 0.0)
    else:  # distribution of word wi's NLL across its target positions
      n = int(model._word_cnt[wi].item())
      mu = model._wi_sum / max(n, 1)
      stats = (mu, math.sqrt(max(model._wi_sq / max(n, 1) - mu * mu, 0.0)),
               model._wi_min if n else 0.0, model._wi_max if n else 0.0)
    for key, val in zip(('mean', 'std', 'min', 'max'), stats):
      wstats[key].append(val)
    print(f'  t={t:.4f}  loss={curve[-1]:.4f}  |x_t|={norms[-1]:.4f}'
          f'  d={dists[-1]:.4f}  word={wstats["mean"][-1]:.4f}'
          f'+-{wstats["std"][-1]:.4f} [{wstats["min"][-1]:.4f},'
          f' {wstats["max"][-1]:.4f}]', flush=True)
  return curve, norms, dists, wstats


def _add_arrows(ax, x, y, color, log_y=False):
  """Single arrowhead pointing to the generative target (t=0, clean): sampling
  runs noise (t=1) -> clean (t=0), so the arrow points at the smallest-t point.
  Skips non-positive y on a log axis."""
  x = np.asarray(x, float); y = np.asarray(y, float)
  ok = np.arange(len(x)) if not log_y else np.where(y > 0)[0]
  if len(ok) < 2:
    return
  a, b = ok[1], ok[0]  # head at the smallest-t (clean) point = target destination
  ax.annotate('', xy=(x[b], y[b]), xytext=(x[a], y[a]),
              arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2,
                              mutation_scale=18))


def plot(paths: List[str], loaded_steps: List[int], tags: List[str],
         title: str, log_y_axis: bool = False, args=None):
  """paths: run-level dirs; one curve per (path, loaded_step), labelled by tag.

  Curves are computed once via the eval pipeline and cached in <out>.json;
  a second call (e.g. for the log-y figure) redraws from the cache.
  """
  x_axis = getattr(args, 'x_axis', 't')  # 't', 'xt_norm' (|x_t|), or 'riem_dist'
  y_metric = getattr(args, 'y_metric', 'loss')  # 'loss' or 'word_loss_std'
  wi = getattr(args, 'word_idx', None)  # single-word mode (own _w{wi} cache)
  cache = args.out + '.json'
  if os.path.isfile(cache):
    saved = json.load(open(cache))
    t_grid, curves = np.array(saved['t']), saved['curves']
    norms = saved.get('norms', {})
    dists = saved.get('dists', {})
    word_stats = saved.get('word_stats', {})
  else:
    t_grid = np.linspace(args.t_min, 1.0, args.t_points)
    curves, norms, dists, word_stats = {}, {}, {}, {}
    for path in paths:
      for step, tag in zip(loaded_steps, tags):
        label = tag if len(paths) == 1 else f'{os.path.basename(path)}, {tag}'
        (curves[label], norms[label], dists[label],
         word_stats[label]) = _loss_curve(path, step, t_grid, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(cache, 'w') as f:
      json.dump(dict(t=t_grid.tolist(), curves=curves, norms=norms, dists=dists,
                     word_stats=word_stats, word_idx=wi,
                     title=title, paths=paths, loaded_steps=loaded_steps,
                     num_batches=args.num_batches,
                     batch_size=args.batch_size), f, indent=2)
    print(f'wrote {cache}')

  if x_axis == 'xt_norm' and not norms:
    raise SystemExit(f'{cache} predates the xt_norm feature (no cached norms); '
                     'delete it and rerun to recompute.')
  if x_axis == 'riem_dist' and not dists:
    raise SystemExit(f'{cache} predates the riem_dist feature (no cached dists); '
                     'delete it and rerun to recompute.')
  needs_ws = y_metric == 'word_loss_std' or wi is not None
  if needs_ws and not all(word_stats.get(label) for label in curves):
    raise SystemExit(f'{cache} lacks per-word loss stats for some curves (it '
                     'predates the current target-bucketed word_stats) -- '
                     'delete it and rerun to recompute.')
  xlabels = {'t': 't', 'xt_norm': r'$\|x_t\|_2$ (mean per corrupted token)',
             'riem_dist': r'$d(x_t,\,x_0)$ (mean per corrupted token)'}
  if wi is None:
    ylabels = {'loss': 'Loss',
               'word_loss_std': 'Per-word mean loss (mean $\\pm$ std; min/max)'}
  else:
    ylabels = {'loss': f'Mean loss of word {wi}',
               'word_loss_std': (f'Loss of word {wi} (mean $\\pm$ std; '
                                 'min/max over its positions)')}
  fig, ax = plt.subplots(figsize=(6, 4.5))
  for label in curves:
    ws = word_stats[label] if needs_ws else None
    y = np.array(curves[label] if ws is None else ws['mean'])
    if x_axis == 't':
      x = t_grid
    elif x_axis == 'xt_norm':
      x = np.array(norms[label])
    else:
      x = np.array(dists[label])
    line, = ax.plot(x, y, marker='o', markersize=3, label=label)
    if ws is not None and y_metric == 'word_loss_std':  # +- std band, min/max
      s, mn, mx = (np.array(ws[k]) for k in ('std', 'min', 'max'))
      lo = np.maximum(y - s, y * 1e-3) if log_y_axis else y - s  # log-drawable
      ax.fill_between(x, lo, y + s, color=line.get_color(), alpha=0.2, lw=0)
      ax.errorbar(x, y, yerr=(y - mn, mx - y), fmt='none',
                  ecolor=line.get_color(), elinewidth=0.8, capsize=2, alpha=0.5)
    _add_arrows(ax, x, y, line.get_color(), log_y=log_y_axis)  # increasing-t
  ax.set(xlabel=xlabels[x_axis], ylabel=ylabels[y_metric], title=title)
  if log_y_axis:
    ax.set_yscale('log')
  ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
  y_suffix = {'loss': '', 'word_loss_std': '_wordstd'}[y_metric]
  axis_suffix = {'t': '', 'xt_norm': '_xtnorm', 'riem_dist': '_riemdist'}[x_axis]
  suffix = y_suffix + axis_suffix + ('_log' if log_y_axis else '')
  out_png = args.out + suffix + '.png'
  fig.savefig(out_png, dpi=150); plt.close(fig)
  print(f'wrote {out_png}')


def load_run_group_steps(
    project: str,
    run: str,
    loaded_steps: List[int],
    log_y_axis: bool = False
):
  """Checkpoints of one run, tagged by training step (mode 'steps')."""
  paths: List[str] = [os.path.join(project, run)]
  tags: List[str] = [checkpoint_training_step2tag(step=s) for s in loaded_steps]
  return {
    'paths': paths,
    'loaded_steps': loaded_steps,
    'tags': tags,
    'log_y_axis': log_y_axis,
    'title': 'Loss Geometry on Training Steps',
  }


def load_run_group_hflm_steps_init_prior_cov(
    project: str,
    run: str,
    loaded_steps: List[int],
    loaded_inits: List[str],
    loaded_prior_covs: List[str],
    log_y_axis: bool = False
):
  """HFLM sweep cells by (step, init, prior_cov) (mode 'hflm_init_pc').

  `run` is the dir-name template, default 'std{init}_pc{pc}'. init/pc are the
  LITERAL strings from the dir names ('1.0', not '1'); init 'ngpt' -> ngpt_pc*.
  The (init, pc) model overrides are recovered from the dir name in plot().
  """
  paths: List[str] = []
  for loaded_init in loaded_inits:
    for loaded_prior_cov in loaded_prior_covs:
      name = ('ngpt_pc' + loaded_prior_cov if loaded_init == 'ngpt'
              else run.format(init=loaded_init, pc=loaded_prior_cov))
      paths.append(os.path.join(project, name))
  tags: List[str] = [checkpoint_training_step2tag(step=s) for s in loaded_steps]
  return {
    'paths': paths,
    'loaded_steps': loaded_steps,
    'tags': tags,
    'log_y_axis': log_y_axis,
    'title': 'Loss Geometry on Training Steps',
  }


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=['steps', 'hflm_init_pc'], required=True)
  p.add_argument('--project', required=True)
  p.add_argument('--run', default='std{init}_pc{pc}',
                 help='run dir name (steps) or dir template (hflm_init_pc)')
  p.add_argument('--steps', type=int, nargs='+', required=True)
  p.add_argument('--inits', nargs='+', default=['0.04'])
  p.add_argument('--prior-covs', nargs='+', default=['1.0'])
  p.add_argument('--out', required=True, help='output prefix (no extension)')
  p.add_argument('--x-axis', choices=['t', 'xt_norm', 'riem_dist'], default='t',
                 help="x-axis: flow-time t, |x_t| (xt_norm), or the Riemannian "
                      "distance d(x_t, x_0) on the algo's manifold (riem_dist)")
  p.add_argument('--y-metric', choices=['loss', 'word_loss_std'], default='loss',
                 help="y-axis: token-mean denoising loss (loss), or the across-vocab "
                      "distribution of each word's mean loss (per-token NLL bucketed "
                      "by clean target x_0), as mean line + std band + min/max "
                      "whiskers (word_loss_std). Both are computed and cached in "
                      "one pass; this only selects what is plotted.")
  p.add_argument('--word-idx', type=int, default=None,
                 help="restrict the curves to one vocab word id: both y-metrics "
                      "then plot the word's per-token NLL at positions where it is "
                      "the clean target -- its mean (loss), plus a std band and "
                      "min/max whiskers over those positions (word_loss_std). "
                      "Outputs get a _w{idx} suffix.")
  p.add_argument('--t-min', type=float, default=0.001)
  p.add_argument('--t-points', type=int, default=33)
  p.add_argument('--num-batches', type=int, default=8)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--length', type=int, default=256)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()
  if args.word_idx is not None:
    args.out += f'_w{args.word_idx}'  # single-word runs get their own cache/figures

  if args.mode == 'steps':
    data = load_run_group_steps(
      project=args.project, run=args.run, loaded_steps=args.steps)
  elif args.mode == 'hflm_init_pc':
    data = load_run_group_hflm_steps_init_prior_cov(
      project=args.project, run=args.run, loaded_steps=args.steps,
      loaded_inits=args.inits, loaded_prior_covs=args.prior_covs)
  else:
    raise ValueError(f'mode, {args.mode}, is not supported.')

  plot(**{**data, 'log_y_axis': False}, args=args)  # computes + caches curves
  plot(**{**data, 'log_y_axis': True}, args=args)   # redraws from the cache


if __name__ == '__main__':
  main()
