#!/usr/bin/env python
"""Early stop of one cell, run by its job after each stage: exit 1 if the stage NELBO
(<cell dir>/eval/nelbo_{steps}.json) is at or above the bar in bars.json, which can be tightened while
jobs run. With WANDB_API_KEY set it also uploads the cell's evals to its W&B run. No torch.

Usage:  python gate.py <cell dir> <steps | final>
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'hbfm_ce_tinystories_256'


def run_id(tag):
  return tag.replace('.', 'p')


def stage_scores(tdir):
  """{steps: the eval/nelbo_{steps}.json} of a cell dir."""
  return {int(re.search(r'nelbo_(\d+)', f).group(1)): json.load(open(f))
          for f in glob.glob(f'{tdir}/eval/nelbo_*.json')}


def summary(tdir):
  """The stage NELBOs and the final evals of a cell dir, as W&B summary entries."""
  out = {}
  for k, v in stage_scores(tdir).items():
    out.update({f'stage/nelbo@{k}': v['val/nll'], f'stage/nelbo_std@{k}': v.get('val/loss_std')})
  for name, key, f in [('nelbo', 'val/nll', 'nelbo.json'), ('nelbo_std', 'val/loss_std', 'nelbo.json'),
                       ('val_ppl', 'val/ppl', 'ppl.json'),
                       ('gen_ppl', 'gen_ppl_first_chunk_retok', 'samples_genppl.json'),
                       ('entropy', 'entropy', 'samples_genppl.json')]:
    if os.path.exists(f'{tdir}/eval/{f}'):
      out[f'final/{name}'] = json.load(open(f'{tdir}/eval/{f}')).get(key)
  return out


def upload(tdir, group=NAME, project=NAME):
  """The cell's scores into its W&B run: val/nelbo against trainer/global_step, plus the summary."""
  import wandb
  tag = os.path.basename(tdir.rstrip('/'))
  run = wandb.init(project=project, id=run_id(tag), name=tag, group=group, resume='allow', dir=tdir,
                   settings=wandb.Settings(silent=True, console='off'))
  run.define_metric('trainer/global_step')
  run.define_metric('*', step_metric='trainer/global_step', step_sync=True)
  for k, v in sorted(stage_scores(tdir).items()):
    if run.summary.get(f'stage/nelbo@{k}') is None:
      run.log({'val/nelbo': v['val/nll'], 'val/nelbo_std': v.get('val/loss_std'), 'trainer/global_step': k})
  run.summary.update(summary(tdir))
  run.finish()


if __name__ == '__main__':
  tdir, steps = sys.argv[1:3]
  score = None if steps == 'final' else json.load(open(f'{tdir}/eval/nelbo_{steps}.json'))['val/nll']
  if os.environ.get('WANDB_API_KEY'):
    try:
      upload(tdir, project=os.environ.get('WANDB_PROJECT', NAME))
    except Exception as e:   # a W&B outage must not read as a failed gate
      print(f'W&B upload failed: {e}')
  if score is not None:
    bar = json.load(open(f'{HERE}/bars.json'))[steps]
    print(f'[gate] {os.path.basename(tdir)} step {steps}: NELBO {score:.4f}, bar {bar:g} -> '
          f'{"continue" if score < bar else "STOP"}')
    sys.exit(score >= bar)
