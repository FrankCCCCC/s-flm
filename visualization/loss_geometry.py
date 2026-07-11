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


def _install_xt_norm_hook(model):
  """Wrap q_xt to accumulate the per-token L2 norm of the corrupted sample x_t
  (over valid/corrupted tokens) into model._xt_sum / _xt_cnt. Reset both before
  each t; the mean per-token norm is _xt_sum / _xt_cnt. Note: for SFM the slerp
  keeps x_t on the sphere, so this norm is ~constant."""
  orig = model.q_xt
  def wrapped(*a, **k):
    xt = orig(*a, **k)
    vt = k.get('valid_tokens')
    if vt is None and len(a) >= 4:  # LangFlow passes valid_tokens positionally
      vt = a[3]
    norm = xt.float().norm(dim=-1)  # [B, L] L2 norm over the embedding dim
    if vt is not None:
      m = vt.bool()
      model._xt_sum += (norm * m).sum().item(); model._xt_cnt += int(m.sum())
    else:
      model._xt_sum += norm.sum().item(); model._xt_cnt += norm.numel()
    return xt
  model.q_xt = wrapped


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
  _install_xt_norm_hook(model)  # records mean per-token |x_t| per t
  curve, norms = [], []
  for t in t_grid:
    _pin_time(model, cfg, t, step)  # fixed t; rest is the algo's own _loss
    torch.manual_seed(cfg.seed)  # same prior-noise draws at every t/ckpt
    model._xt_sum, model._xt_cnt = 0.0, 0
    out = [model._loss(b['input_ids'], b['attention_mask']) for b in batches]
    curve.append(sum(o.nlls.item() for o in out)
                 / sum(o.num_tokens.item() for o in out))
    norms.append(model._xt_sum / max(model._xt_cnt, 1))
    print(f'  t={t:.4f}  loss={curve[-1]:.4f}  |x_t|={norms[-1]:.4f}', flush=True)
  return curve, norms


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
  x_axis = getattr(args, 'x_axis', 't')  # 't' or 'xt_norm' (|x_t| on the x-axis)
  cache = args.out + '.json'
  if os.path.isfile(cache):
    saved = json.load(open(cache))
    t_grid, curves = np.array(saved['t']), saved['curves']
    norms = saved.get('norms', {})
  else:
    t_grid = np.linspace(args.t_min, 1.0, args.t_points)
    curves, norms = {}, {}
    for path in paths:
      for step, tag in zip(loaded_steps, tags):
        label = tag if len(paths) == 1 else f'{os.path.basename(path)}, {tag}'
        curves[label], norms[label] = _loss_curve(path, step, t_grid, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(cache, 'w') as f:
      json.dump(dict(t=t_grid.tolist(), curves=curves, norms=norms, title=title,
                     paths=paths, loaded_steps=loaded_steps,
                     num_batches=args.num_batches,
                     batch_size=args.batch_size), f, indent=2)
    print(f'wrote {cache}')

  if x_axis == 'xt_norm' and not norms:
    raise SystemExit(f'{cache} predates the xt_norm feature (no cached norms); '
                     'delete it and rerun to recompute.')
  fig, ax = plt.subplots(figsize=(6, 4.5))
  for label, c in curves.items():
    x = t_grid if x_axis == 't' else np.array(norms[label])
    line, = ax.plot(x, c, marker='o', markersize=3, label=label)
    _add_arrows(ax, x, c, line.get_color(), log_y=log_y_axis)  # increasing-t
  ax.set(xlabel=('t' if x_axis == 't' else r'$\|x_t\|_2$ (mean per corrupted token)'),
         ylabel='Loss', title=title)
  if log_y_axis:
    ax.set_yscale('log')
  ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
  suffix = ('_xtnorm' if x_axis == 'xt_norm' else '') + ('_log' if log_y_axis else '')
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
  p.add_argument('--x-axis', choices=['t', 'xt_norm'], default='t',
                 help="x-axis: flow-time t, or L2 norm of the corrupted sample x_t")
  p.add_argument('--t-min', type=float, default=0.001)
  p.add_argument('--t-points', type=int, default=33)
  p.add_argument('--num-batches', type=int, default=8)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--length', type=int, default=256)
  p.add_argument('--cache-dir',
                 default='/share/thickstun/sychou/workspace/research/s-flm/data_cache')
  args = p.parse_args()

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
