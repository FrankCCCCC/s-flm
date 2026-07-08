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
import argparse, glob, itertools, json, os, re, sys, types
from typing import List

import hydra, matplotlib, numpy as np, torch
import matplotlib.pyplot as plt
import algo, dataloader, main as main_mod  # main registers omegaconf resolvers

matplotlib.use('Agg')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ALGO_CLS = {'eflm': algo.EFLM, 'sfm': algo.SFM, 'hflm': algo.HFLM}
MODEL_ARGS = {  # mirrors scripts/sample/tinystories/eval.sh MARGS
  'eflm': ['model=small-sphere-dit', 'model.init=ngpt', 'algo=eflm'],
  'sfm':  ['model=small-sphere-dit', 'model.init=ngpt', 'algo=sfm'],
  'hflm': ['model=small-hyperbolic-dit', 'algo=hflm', 'algo.rho_max=12'],
}
# hflm sweep run dirs encode (init, prior_cov): std{init}_pc{pc} or ngpt_pc{pc}
SWEEP_RE = re.compile(r'^(?:std(?P<init>[\d.]+)|(?P<ngpt>ngpt))_pc(?P<pc>[\d.]+)$')

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


def _run_overrides(run_name: str):
  """Model type + hydra overrides, derived from the run dir name."""
  m = SWEEP_RE.match(run_name)
  if m:
    init_args = (['model.init=ngpt'] if m['ngpt'] else
                 ['model.init=custom', f'model.init_std={m["init"]}'])
    return 'hflm', MODEL_ARGS['hflm'] + init_args + [f'algo.prior_cov={m["pc"]}']
  if run_name in MODEL_ARGS:
    return run_name, MODEL_ARGS[run_name]
  raise ValueError(f'cannot infer model config from run dir name: {run_name}')


@torch.no_grad()
def _loss_curve(run_dir: str, step: int, t_grid: np.ndarray, args):
  """L(t) = token-mean denoising CE on val batches, at each fixed t."""
  ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', f'*-{step}.ckpt'))
  assert ckpts, f'no *-{step}.ckpt under {run_dir}/checkpoints'
  model_type, run_args = _run_overrides(os.path.basename(run_dir))
  print(f'[{model_type} step={step}] {ckpts[0]}', flush=True)
  with hydra.initialize_config_dir(config_dir=os.path.join(REPO, 'configs'),
                                   version_base=None):
    config = hydra.compose('config', overrides=run_args + [
      'data=tinystories', f'data.cache_dir={args.cache_dir}',
      f'model.length={args.length}', 'noise=log-linear',
      'algo.invert_time_convention=false', 'algo.renormalize_weights=False',
      'trainer.devices=1', 'trainer.accumulate_grad_batches=1',
      f'loader.global_batch_size={args.batch_size}',
      f'loader.eval_global_batch_size={args.batch_size}',
      f'loader.batch_size={args.batch_size}',
      f'loader.eval_batch_size={args.batch_size}', 'loader.num_workers=4',
      f'eval.checkpoint_path={sorted(ckpts)[0]}'])
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = dataloader.get_tokenizer(config)
  model = main_mod._load_from_checkpoint(
    ALGO_CLS[model_type], config, tokenizer).to(device).eval()
  if model.ema:
    model.ema.move_shadow_params_to_device(device)
  model._eval_mode()  # swap in EMA weights, as trainer.validate does
  _, valid_dl = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  batches = [{k: v.to(device) for k, v in b.items()}
             for b in itertools.islice(iter(valid_dl), args.num_batches)]
  curve = []
  for t in t_grid:
    # Pin t: _loss -> nll -> self._sample_t; everything else is the training loss.
    model._sample_t = types.MethodType(
      lambda self, n, accum, _t=float(t):
        torch.full((n,), _t, device=self.device), model)
    torch.manual_seed(config.seed)  # same prior-noise draws at every t/ckpt
    out = [model._loss(b['input_ids'], b['attention_mask']) for b in batches]
    curve.append(sum(o.nlls.item() for o in out)
                 / sum(o.num_tokens.item() for o in out))
    print(f'  t={t:.4f}  loss={curve[-1]:.4f}', flush=True)
  return curve


def plot(paths: List[str], loaded_steps: List[int], tags: List[str],
         title: str, log_y_axis: bool = False, args=None):
  """paths: run-level dirs; one curve per (path, loaded_step), labelled by tag.

  Curves are computed once via the eval pipeline and cached in <out>.json;
  a second call (e.g. for the log-y figure) redraws from the cache.
  """
  cache = args.out + '.json'
  if os.path.isfile(cache):
    saved = json.load(open(cache))
    t_grid, curves = np.array(saved['t']), saved['curves']
  else:
    t_grid = np.linspace(args.t_min, 1.0, args.t_points)
    curves = {}
    for path in paths:
      for step, tag in zip(loaded_steps, tags):
        label = tag if len(paths) == 1 else f'{os.path.basename(path)}, {tag}'
        curves[label] = _loss_curve(path, step, t_grid, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(cache, 'w') as f:
      json.dump(dict(t=t_grid.tolist(), curves=curves, title=title,
                     paths=paths, loaded_steps=loaded_steps,
                     num_batches=args.num_batches,
                     batch_size=args.batch_size), f, indent=2)
    print(f'wrote {cache}')

  fig, ax = plt.subplots(figsize=(6, 4.5))
  for label, c in curves.items():
    ax.plot(t_grid, c, marker='o', markersize=3, label=label)
  ax.set(xlabel='t', ylabel='Loss', title=title)
  if log_y_axis:
    ax.set_yscale('log')
  ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
  out_png = args.out + ('_log.png' if log_y_axis else '.png')
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
