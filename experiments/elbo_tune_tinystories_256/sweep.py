#!/usr/bin/env python
"""elbo_tune_tinystories_256 -- tune EFLM and HBFM for the likelihood bound on TinyStories (seq 256).

Question (user, 2026-10-12): can HBFM / EFLM reach MDLM's NELBO on TinyStories under a fair protocol,
or are the algorithms worse than MDLM? Design, grids and fairness: `EXPERIMENT.md`. The losses are
`algo.loss_type=elbo` (experiments/elfm_hbfm_ada_curv_elbo_sudoku/math.md).

Every cell: small DiT 768/12/12, seq 256, global batch 512, bf16, EMA 0.9999, AdamW wd 0, clip 1.0,
ngpt init -- the TinyStories protocol of naive_ar_tinystories_s256 (MDLM, the reference: 30k steps,
LR 1e-3, NELBO 1.582 +- 0.007 over 3 seeds). Stage 1 ranks configurations at 10k steps (seed 1);
stage 2 trains the pick of each arm for the full 30k steps x 3 seeds. The metric is the NELBO in
nats per token on the full validation set (mode=ppl_eval, LOSS_TYPE=elbo, one (t, noise) draw per
sequence per pass; passes 1..PPL_PASSES averaged by report.py).

Sites (a cell runs at one of them; a job of the same name queued anywhere counts as running):
  a100    ch2263@unicorn, nlplarge (8x A100 80GB): 1 GPU per cell. Code copy at R_REPO
          (`--sync-code`), results come back with `--pull`; checkpoints on the node's /scratch,
          linked from the run directory (experiment.py).
  a6000   sc3379, desa / thickstun without the 2080 Ti node: 4 GPUs per cell.
  2080ti  sc3379, desa-compute-01 (8x 2080 Ti 11GB): 8 GPUs per cell.

ORCHESTRATION ONLY: calls scripts/train/tinystories/{hbfm,eflm_rescale_auto_truncation,mdlm}.sh and
scripts/sample/tinystories/{...}.sh. Idempotent: a cell with eval/nelbo_s1.json (local, after
`--pull`) or a queued job of the same name is skipped; a resubmitted cell resumes from last.ckpt.

Usage:  python sweep.py --sync-code
        python sweep.py --stage smoke|1|2 --site a100|a6000|2080ti [--only SUBSTR] [--seeds 1] [--dry-run]
        python sweep.py --pull ;  python report.py
"""
import argparse
import os
import subprocess
import textwrap

REPO = '/share/desa/nfs02/sc3379/workspace/research/s-flm-dev2/s-flm'
NAME = 'elbo_tune_tinystories_256'
EXP = f'{REPO}/experiments/{NAME}'
OUT = f'{REPO}/outputs/{NAME}'
JOB = 'elbo_ts'
KEY = '/home/sc3379/.ssh/unicorn_internal'
REMOTE = 'ch2263@unicorn-login-02.coecis.cornell.edu'
R_REPO = '/home/ch2263/syc_workspace/s-flm-dev2-elbo'
CODE = ['*.py', 'configs', 'models', 'numeric', 'callbacks', 'scripts', 'data', 'tests']
PPL_PASSES = {'smoke': 1, '1': 1, '2': 2}
MAX_STEPS = {'smoke': 10, '1': 10000, '2': 30000}

# Per-GPU micro-batch by (arm, objective): the ELBO arms run a second forward pass with gradients,
# and HBFM's bound materialises (B, L, V) readouts in float32 (smoke on the 11 GB cards: EFLM-ELBO
# fits at 8, HBFM-ELBO and MDLM do not at 8 / 16). The eval pass uses the same micro-batch.
SITES = {
  'a100': dict(remote=True, repo=R_REPO, envbin='/home/ch2263/miniconda3/envs/sfm/bin',
               cache='/home/ch2263/syc_workspace/s-flm_data_home/data_cache', tmp='/home/ch2263/tmp',
               devices=1, workers=4,
               bs={('eflm', 'elbo'): 32, ('hbfm', 'elbo'): 16, ('eflm', 'ce'): 64, ('hbfm', 'ce'): 64, ('mdlm', 'elbo'): 64},
               sbatch='--partition=nlplarge-claire-highpri,nlplarge --gres=gpu:1 --cpus-per-task=4 --mem=64G --requeue'),
  # Local sites use the node's /tmp: the dataloader workers' listener sockets live under TMPDIR,
  # and on the NFS home they went missing mid-run (E0, job 216381: a rank hung on them and NCCL
  # timed out after 7.5 h).
  'a6000': dict(remote=False, repo=REPO, envbin='/home/sc3379/anaconda3/envs/sfm/bin',
                cache=f'{REPO}/data_cache', tmp='/tmp',
                devices=4, workers=4,
                bs={('eflm', 'elbo'): 8, ('hbfm', 'elbo'): 8, ('eflm', 'ce'): 16, ('hbfm', 'ce'): 16, ('mdlm', 'elbo'): 16},
                sbatch='--partition=desa,thickstun --exclude=desa-compute-01 --gres=gpu:4 --cpus-per-task=16 --mem=128G'),
  '2080ti': dict(remote=False, repo=REPO, envbin='/home/sc3379/anaconda3/envs/sfm/bin',
                 cache=f'{REPO}/data_cache', tmp='/tmp',
                 devices=8, workers=1,
                 bs={('eflm', 'elbo'): 8, ('hbfm', 'elbo'): 4, ('eflm', 'ce'): 16, ('hbfm', 'ce'): 16, ('mdlm', 'elbo'): 8},
                 sbatch=('--partition=desa --nodelist=desa-compute-01 '
                         '--gres=gpu:nvidia_geforce_rtx_2080_ti:8 --cpus-per-task=10 --mem=120G')),
}
TIME = {'smoke': '6:00:00', '1': '3-00:00:00', '2': '6-00:00:00'}

# tau*(R) = log(1 + z), z = sqrt(2 log(2 (V - 1) / 0.1)) / R, V = 50257 (noise_schedules.alpha_star_euclidean)
TAU_STAR = {0.5: 2.4436, 2.0: 1.2889, 8.0: 0.5051}


def eflm(loss='elbo', r=0.5, tau=None, noise='autonomous', lr='1e-3'):
  if not noise.startswith('autonomous'):
    tau = 0.0                      # no horizon on the other clocks (TAU_MAX is ignored there)
  return dict(arm='eflm', loss=loss, r=r, tau=TAU_STAR[r] if tau is None else tau, noise=noise, lr=lr)


def hbfm(loss='elbo', cb=4.0, mode='exp', ub=6.0, tmax=6.0, fwd='horosphere', ed=9, lr='1e-3'):
  """(H^3)^(ed/3) at K = -0.5. The default is the (H^3)^3 geometry of the Sudoku ELBO study (user,
  2026-10-12: the exact rate's cost is linear in the number of factors; (H^3)^16 was 5x the price):
  the exp proposal of rate 3 with the sampler / reconstruction horizon T_MAX 6 (configs/algo/hbfm.yaml:
  (H^3)^3 resolves a word by unit time ~3/|K|), horosphere readout, float64 readout."""
  return dict(arm='hbfm', loss=loss, cb=cb, mode=mode, ub=ub, tmax=tmax, fwd=fwd, ed=ed, lr=lr)


def mdlm(lr='1e-3'):
  return dict(arm='mdlm', loss='elbo', lr=lr)


STAGES = {
  'smoke': [eflm(), hbfm(), hbfm(fwd='naive'), mdlm()],
  # Seed 1, 10k steps. One axis at a time around each arm's incumbent (EXPERIMENT.md).
  '1': [
    eflm(),                                   # E0 incumbent (R 0.5, tau* 2.44, autonomous clock, LR 1e-3), ELBO-trained
    eflm(lr='3e-4'),                          # E1 LR
    eflm(r=2.0),                              # E2 embedding norm, horizon tau*(2) = 1.29
    eflm(r=8.0),                              # E3 embedding norm, horizon tau*(8) = 0.51 (the Sudoku setting)
    eflm(tau=4.0),                            # E4 a cleaner clean end: noise-fraction floor e^-4 instead of e^-2.44
    eflm(noise='log-linear', tau=None),       # E5 MDLM's schedule (alpha_t = 1 - t), no truncation
    eflm(noise='autonomous-adaptive'),        # E6 the learned (variance-minimising) reparametrisation of the clock
    eflm(loss='ce'),                          # E7 the CE-trained control at the same budget
    # HBFM on (H^3)^3 (user, 2026-10-12), exact rate. The (H^3)^16 cells of the first submission
    # (trained on a Monte-Carlo estimate of the rate, since removed) are discarded.
    hbfm(),                                   # G0 incumbent: cb 4, exp rate 3, T_MAX 6, horosphere, LR 1e-3, ELBO-trained
    hbfm(lr='3e-4'),                          # G1 LR
    hbfm(fwd='naive'),                        # G2 plain-logit readout (the TinyStories GenPPL studies' choice)
    hbfm(mode='trunc_exp', ub=6.0),           # G3 the proposal truncated at the horizon (no mass beyond T_MAX)
    hbfm(cb=1.0),                             # G4 fixed curvature
    hbfm(loss='ce'),                          # G5 the CE-trained control on the same geometry
    mdlm(),                                   # M0 the reference at the same budget (LR 1e-3, its 30k pick)
  ],
  # The picks of stage 1 at MDLM's budget (30k steps, `--seeds 1,2,3`); '_30k' in the tag keeps them
  # apart from the 10k-step stage-1 cell of the same configuration.
  # EFLM (2026-10-13): E2 = R 2, tau*(2), LR 1e-3, 1.881 at 10k (E3 R 8: 1.889, within 0.01 -> the
  # tie goes to the cell nearer the incumbent R 0.5). HBFM: after its stage 1.
  '2': [dict(eflm(r=2.0), long=True)],
}


def tag_of(c, seed):
  if c['arm'] == 'eflm':
    base = f"eflm_loss-{c['loss']}_r-{c['r']:g}_tau-{c['tau']:g}_noise-{c['noise']}_lr-{c['lr']}"
  elif c['arm'] == 'hbfm':
    fwd = 'horo' if c['fwd'] == 'horosphere' else 'naive'
    base = (f"hbfm_loss-{c['loss']}_cb-{c['cb']:g}_mode-{c['mode']}_ub-{c['ub']:g}_tm-{c['tmax']:g}"
            f"_fwd-{fwd}_ed-{c['ed']}_lr-{c['lr']}")
  else:
    base = f"mdlm_lr-{c['lr']}"
  if c.get('long'):
    base += '_30k'
  return f'{base}_seed-{seed}'


def knobs(c):
  """The env knobs of the train and sample scripts (the geometry must match between the two)."""
  if c['arm'] == 'eflm':
    return (f"RHO={c['r']:g} TAU_MAX={c['tau']:g} NOISE={c['noise']} ALPHA_MAX=null SNR_CE=false "
            f"SELF_COND=false SEQ_LEN=256 LOSS_TYPE={c['loss']}")
  if c['arm'] == 'hbfm':
    # The sampler / reconstruction horizon T_MAX must not exceed the range of the bounded modes.
    assert c['mode'] == 'exp' or c['tmax'] <= c['ub'], c
    return (f"FORWARD_TYPE={c['fwd']} EMBED_DIM={c['ed']} FACTOR_DIM=3 GAUSS_CURV=-0.5 CURV_BOUND={c['cb']:g} "
            f"TIME_CONVERSION_MODE={c['mode']} PROPOSAL_RATE=3 TIME_RANGE_UPPER_BOUND={c['ub']:g} "
            f"T_MAX={c['tmax']:g} READOUT_PRECISION=float64 GLOBAL_BATCH=512 SEQ_LEN=256 LOSS_TYPE={c['loss']}")
  return 'SEQ_LEN=256'


SCRIPT = {'eflm': 'eflm_rescale_auto_truncation', 'hbfm': 'hbfm', 'mdlm': 'mdlm'}


def preamble(s):
  return textwrap.dedent(f'''\
    #!/bin/bash
    set -e
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
    export TMPDIR={s['tmp']}/{JOB}_${{SLURM_JOB_ID}}; mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
    export PATH={s['envbin']}:$PATH
    cd {s['repo']}
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    md5sum algo.py numeric/loss.py scripts/train/tinystories/{{hbfm,eflm_rescale_auto_truncation,mdlm}}.sh
    ''')


def body(s, c, seed, stage):
  tag = tag_of(c, seed)
  tdir = f'{s["repo"]}/outputs/{NAME}/{tag}'
  bs = s['bs'][(c['arm'], c['loss'])]
  ckpt_every = 1000 if stage != 'smoke' else 5
  out = preamble(s) + textwrap.dedent(f'''\
    echo "[$(date)] TRAIN {tag}"
    python experiment.py {tdir} {NAME} {tag}; ls -ld {tdir}/checkpoints 2>/dev/null || true
    {knobs(c)} LR={c['lr']} SEED={seed} CACHE_DIR={s['cache']} \\
      OUTPUT_DIR={tdir} RUN_NAME={tag} WANDB_GROUP={NAME} DEVICES={s['devices']} PER_GPU_BS={bs} \\
      NUM_WORKERS={s['workers']} MAX_STEPS={MAX_STEPS[stage]} CKPT_EVERY={ckpt_every} KEEP_EVERY=0 \\
      bash scripts/train/tinystories/{SCRIPT[c['arm']]}.sh
    echo "[$(date)] EVAL {tag}: NELBO on the validation set"
    ''')
  evals = [('elbo', 'nelbo', p) for p in range(1, PPL_PASSES[stage] + 1)]
  if c['loss'] == 'ce':
    evals.append(('ce', 'ce', 1))     # the CE objective of a CE-trained control, for the record
  for loss, prefix, p in evals:
    geometry = knobs(c).replace(f" LOSS_TYPE={c['loss']}", '')
    out += (f'{geometry} LOSS_TYPE={loss} CACHE_DIR={s["cache"]} RUN_SAMPLE_EVAL=false SEED={p} EVAL_BS={bs} \\\n'
            f'  CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval PPL_JSON={tdir}/eval/{prefix}_s{p}.json \\\n'
            f'  DEVICES=1 bash scripts/sample/tinystories/{SCRIPT[c["arm"]]}.sh\n')
  return out + f'echo "[$(date)] DONE {tag}"\n'


def run(s, cmd, stdin=None):
  """A shell command at the site (over ssh with a login shell for the a100 site)."""
  if s['remote']:
    cmd = ['ssh', '-o', 'BatchMode=yes', '-i', KEY, REMOTE, f'bash -lc {sh_quote(cmd)}']
  else:
    cmd = ['bash', '-c', cmd]
  return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)


def sh_quote(x):
  return "'" + x.replace("'", "'\\''") + "'"


def submit(s, jobname, text, opts, time, dry):
  logs = f'{s["repo"]}/experiments/{NAME}/logs'
  if dry:
    print(f'  [dry] {jobname}')
    return
  r = run(s, f'mkdir -p {logs} && sbatch --job-name={jobname} {opts} --time={time} '
             f'--output={logs}/{jobname}_%j.log', stdin=text)
  print(f'  {jobname}: {(r.stdout + r.stderr).strip()}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=sorted(STAGES))
  ap.add_argument('--site', choices=sorted(SITES))
  ap.add_argument('--seeds', default='1')
  ap.add_argument('--only', default='')
  ap.add_argument('--sync-code', action='store_true')
  ap.add_argument('--pull', action='store_true')
  ap.add_argument('--dry-run', action='store_true')
  args = ap.parse_args()
  ssh = f'ssh -o BatchMode=yes -i {KEY}'

  if args.sync_code:   # the code only; data, outputs and checkpoints stay where they are
    inc = ' '.join(f"--include='/{p}'" if '*' in p else f"--include='/{p}/***'" for p in CODE)
    os.system(f"rsync -a --exclude='__pycache__/' {inc} --exclude='/*' -e '{ssh}' {REPO}/ {REMOTE}:{R_REPO}/")
    os.system(f"{ssh} {REMOTE} 'cd {R_REPO} && md5sum algo.py numeric/loss.py'")
    return
  if args.pull:        # a100 results (eval JSONs, hydra configs, logs; no checkpoints) -> local
    os.system(f"rsync -a --exclude='checkpoints' --exclude='wandb/' -e '{ssh}' "
              f"{REMOTE}:{R_REPO}/outputs/{NAME}/ {OUT}/")
    os.system(f"rsync -a -e '{ssh}' {REMOTE}:{R_REPO}/experiments/{NAME}/logs/ {EXP}/logs/a100/")
    return

  if not (args.stage and args.site):
    ap.error('--stage and --site are required to submit')
  s = SITES[args.site]
  queued = set()
  for site in SITES.values():
    if site['remote'] and args.dry_run:
      continue
    queued |= set(run(site, f'squeue -h -u {"ch2263" if site["remote"] else "sc3379"} -o %j').stdout.split())
  out = f'{OUT}/smoke' if args.stage == 'smoke' else OUT
  n_sub = 0
  for c in STAGES[args.stage]:
    for seed in args.seeds.split(','):
      tag = tag_of(c, seed)
      if args.only not in tag:
        continue
      jobname = f'{JOB}_{args.stage}_{tag}'
      if os.path.exists(f'{out}/{tag}/eval/nelbo_s1.json') or jobname in queued:
        print(f'  skip {tag}')
        continue
      text = body(s, c, seed, args.stage)
      if args.stage == 'smoke':
        text = text.replace(f'outputs/{NAME}/', f'outputs/{NAME}/smoke/')
      opts = s['sbatch']
      if args.stage == 'smoke' and args.site == '2080ti':   # three smoke cells share the node
        opts = opts.replace('rtx_2080_ti:8', 'rtx_2080_ti:2').replace('cpus-per-task=10', 'cpus-per-task=4').replace('mem=120G', 'mem=30G')
        text = text.replace('DEVICES=8', 'DEVICES=2')
      submit(s, jobname, text, opts, TIME[args.stage], args.dry_run)
      n_sub += 1
  print(f'stage {args.stage} at {args.site}: submitted {n_sub}')
  if args.dry_run and STAGES[args.stage]:
    print('\n--- example body ---\n' + body(s, STAGES[args.stage][0], 1, args.stage))


if __name__ == '__main__':
  main()
