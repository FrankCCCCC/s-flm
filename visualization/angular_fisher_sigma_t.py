#!/usr/bin/env python
"""The slides' angular-Fisher posterior covariance Sigma_t along t for EFLM checkpoints (setup.md, slides/sep25_2026).

Slides: F(rho_t, w_t) = (d-1)^2 sinh^2(kappa rho_t) Sigma_t, Sigma_t = Cov_{q(v|z_t)}[P_w v / D_kappa(v, rho_t, w_t)],
Sigma = Cov_{q_data}[v], v in S^{d-1}. In EFLM every |e_v| = R, so at fixed |x_t| its likelihood in w = x_t / |x_t| is a
vMF kernel, N(x_t; a_t e_v, b_t^2 I) ∝ exp(kappa_eff w^T v), kappa_eff = a_t |x_t| R / b_t^2, v = e_v / R. That is the
slides' Poisson kernel D^{-(d-1)} to first order in kappa rho_t ((d-1) sinh(kappa rho_t) <-> kappa_eff, D <-> 1), so
v~_t = P_w v (no 1/D reweighting). It is EFLM's exact angular Fisher: grad_w log q(v | x_t) = |x_t| beta_t P_w (e_v - mu_t),
F_ang = |x_t|^2 beta_t^2 R^2 Sigma_t (experiments/eflm_cov_reparam_fisher_cond/check_angular_fisher.py verifies it
against finite-difference KL).
Spectra of Sigma_t are taken on the tangent space w^perp (d - 1 dims; the trivial zero along w is dropped).
Posterior q(v | z_t) ∝ q(z_t | v) q(v) with q(z_t | v) = N(x_t; a_t e_v, b_t^2 I) and the true word v* ~ q(v):
  data_freq   q(v) = train unigram frequency (the slides' q_data)
  uniform     q(v) = uniform over the words seen in training (control: codebook geometry only)
  model_pred  q(v | z_t) = p_theta(x^i | x_t), the model on validation text (with context), full vocabulary: a plug-in
              for the contextual posterior q(v | x_t^{-i}) N(x_t^i; a_t e_v, b_t^2 I) (no second Bayes step)
Sigma (all d dims) is the near-origin limit: Cov_{q_data}[v] for data_freq / model_pred, Cov_uniform[v] for uniform.
Figures (as figures/fisher_cond_state_{t,snr}.png): per (prior, metric) the median over the N_STATE states at each t
and their 25-75% quartile band; dashed = naive, solid = gaussian, one color per R; star = Sigma at the noise end.

Usage (compute node, 1 GPU; writes data/<label>.json, then redraws figures/ from every data/*.json):
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python visualization/angular_fisher_sigma_t.py \
    --ckpt outputs/.../checkpoints/last.ckpt --label gauss_r-0.5 [--nt 51]
  python visualization/angular_fisher_sigma_t.py --plot   # figures only
"""
import argparse, glob, json, os, types

import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{REPO}/experiments/eflm_cov_reparam_fisher_cond'
DATA, FIG = f'{EXP}/data', f'{EXP}/figures'
SINGULAR = 1e-12  # lambda_min / lambda_max below this is fp64 eigensolver noise (~d * 2.2e-16): cond censored at 1e12
N_STATE = 128     # states per (prior, t)
N_SEQ = 32        # validation sequences for model_pred
PRIORS = ('data_freq', 'uniform', 'model_pred')
METRICS = (('top1', 'lambda_1 / sum lambda_i', False), ('k90', 'k90 (dirs. holding 90% of tr)', True),
           ('k99', 'k99 (dirs. holding 99% of tr)', True), ('pr', 'eff rank (PR) tr^2 / tr Sigma_t^2', True),
           ('log10_cond', 'log10 cond(Sigma_t) (censored at 12)', False))


def spectrum_metrics(lam):
  """lam: eigenvalues, descending, >= 0."""
  tr = lam.sum()
  c = lam.cumsum(0) / tr
  return dict(top1=(lam[0] / tr).item(), k90=int((c < 0.90).sum()) + 1, k99=int((c < 0.99).sum()) + 1,
              pr=(tr ** 2 / (lam ** 2).sum()).item(),
              log10_cond=(lam[0] / lam[-1].clamp(min=SINGULAR * lam[0])).log10().item())


def cov(U, q):
  """Cov_q[u] (float64) over the support of q, centred at the posterior mode so a peaked q loses no precision."""
  U, q = U[q > 0], q[q > 0]
  Uc = U - U[q.argmax()]
  Uc = Uc - q @ Uc
  return (Uc * q[:, None]).T @ Uc


def tangent_eigs(S, w):
  """Eigenvalues (descending) of S on w^perp: H = I - 2 u u^T / |u|^2 with u = w + sign(w_0) e_0 maps w to
  -sign(w_0) e_0, so (H S H)[1:, 1:] is S in an orthonormal basis of the tangent space."""
  u = w.clone()
  u[0] += 1.0 if w[0] >= 0 else -1.0
  HS = S - 2 * torch.outer(u, u @ S) / (u @ u)
  HSH = HS - 2 * torch.outer(HS @ u, u) / (u @ u)
  return torch.linalg.eigvalsh(HSH[1:, 1:]).flip(0).clamp(min=0)


@torch.no_grad()
def compute(args):
  from loss_geometry import ALGO_BY_NAME, _load_config
  from codebook_eigen_proj_scatter import train_counts
  import algo, dataloader, main as main_mod
  torch.manual_seed(0)  # q_xt draws its noise from the global CUDA RNG
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
  cfg = _load_config(run_dir, args.ckpt, types.SimpleNamespace(cache_dir=f'{REPO}/data_cache', batch_size=8))
  assert cfg.algo.name == 'eflm' and not cfg.algo.invert_time_convention  # a_t = alpha_t, b_t = 1 - alpha_t
  tok = dataloader.get_tokenizer(cfg)
  m = main_mod._load_from_checkpoint(ALGO_BY_NAME[cfg.algo.name], cfg, tok).cuda().eval()
  m._eval_mode()  # EMA weights, as at sampling time
  E = m._sc_embed_table().double()  # [V, d], every row rescaled to norm R (rho_min = rho_max = R)
  V, d = E.shape
  norms = E.norm(dim=-1)
  R = norms.mean().item()
  Vn = E / norms[:, None]  # v in S^{d-1}
  sq = norms ** 2
  cnt = torch.as_tensor(train_counts(cfg, V), device='cuda', dtype=torch.float64)
  seen = cnt > 0
  prior = {'data_freq': cnt / cnt.sum(), 'uniform': seen.double() / seen.sum()}
  out = dict(label=args.label, ckpt=os.path.abspath(args.ckpt), forward_type=cfg.algo.forward_type, R=R, V=V, d=d,
             n_seen=int(seen.sum()), norm_min=norms.min().item(), norm_max=norms.max().item(), n_state=N_STATE,
             sigma={k: spectrum_metrics(torch.linalg.eigvalsh(cov(Vn, p)).flip(0).clamp(min=0))
                    for k, p in prior.items()}, rows=[])
  print(f'{args.label}: forward_type={cfg.algo.forward_type} R={R:.3f} (norms {out["norm_min"]:.4f}-'
        f'{out["norm_max"]:.4f}) V={V} d={d} seen={out["n_seen"]} | Sigma: ' +
        ' | '.join(f'{k} ' + ' '.join(f'{kk}={vv:.3g}' for kk, vv in s.items()) for k, s in out['sigma'].items()),
        flush=True)
  g = torch.Generator(device='cuda').manual_seed(0)
  ds = dataloader.get_dataset(cfg, tok, mode='valid')
  x0 = torch.stack([torch.as_tensor(ds[i]['input_ids']) for i in range(0, 100 * N_SEQ, 100)]).cuda()
  L = x0.shape[1]
  for t in np.linspace(0.0, 1.0, args.nt):
    _, alpha = m.noise(torch.full((1, 1), float(t), device='cuda'))
    a = alpha.double().flatten()[0]
    b = 1 - a
    row = dict(t=float(t), alpha=a.item(), log10_snr=2 * np.log10(a.item() * R / b.item()))
    states = {}
    for k, p in prior.items():  # exact Bayes posterior, v* ~ q(v); log p = -inf off the support
      v = torch.multinomial(p, N_STATE, replacement=True, generator=g)
      x = a * E[v] + b * torch.randn(N_STATE, d, device='cuda', dtype=torch.float64, generator=g)
      states[k] = ((p.log() + (a / b ** 2) * x @ E.T - (a ** 2 / (2 * b ** 2)) * sq).log_softmax(-1), x)
    logqs, xs = [], []  # model_pred: inner positions (BOS / EOS excluded) of N_SEQ sequences, 8 per forward pass
    for s in range(0, N_SEQ, 8):
      al = alpha.expand(8, 1)
      xt = m.q_xt(x0[s:s + 8], al, use_pure_noise=False)
      with torch.autocast('cuda', dtype=torch.bfloat16):
        logp = m.forward(xt=xt, sigma=m._sigma_from_alphat(al), context=algo.trainer_base.TrainingContext(alpha_t=al))
      pick = torch.randperm(8 * (L - 2), device='cuda', generator=g)[:N_STATE * 8 // N_SEQ]
      logqs.append(logp.float()[:, 1:L - 1].reshape(-1, V)[pick].double())
      xs.append(xt[:, 1:L - 1].reshape(-1, d)[pick].double())
    states['model_pred'] = (torch.cat(logqs), torch.cat(xs))
    for k, (logq, x) in states.items():
      w = x / x.norm(dim=-1, keepdim=True)
      ms = [spectrum_metrics(tangent_eigs(cov(Vn, lq.exp()), wj)) for lq, wj in zip(logq, w)]
      row[k] = {key: np.quantile([mm[key] for mm in ms], [0.25, 0.5, 0.75]).tolist() for key in ms[0]}
    out['rows'].append(row)
    print(f't={t:.2f} alpha={a.item():.4f} log10snr={row["log10_snr"]:6.2f} | ' + ' | '.join(
      f'{k}: top1 {row[k]["top1"][1]:.3f} k90 {row[k]["k90"][1]:4.0f} k99 {row[k]["k99"][1]:4.0f} '
      f'pr {row[k]["pr"][1]:6.1f} cond {row[k]["log10_cond"][1]:5.2f}' for k in PRIORS), flush=True)
  os.makedirs(DATA, exist_ok=True)
  with open(f'{DATA}/{args.label}.json.tmp', 'w') as f:
    json.dump(out, f, indent=1)
  os.replace(f'{DATA}/{args.label}.json.tmp', f'{DATA}/{args.label}.json')  # atomic: concurrent jobs read data/*.json
  print(f'wrote {DATA}/{args.label}.json', flush=True)


def plot():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  runs = [json.load(open(f)) for f in sorted(glob.glob(f'{DATA}/*.json'))]
  runs.sort(key=lambda r: (r['forward_type'] == 'gaussian', r['R']))  # legend: naive R up, then gauss R up
  color = {0.1: 'C0', 0.5: 'C1', 1.0: 'C2'}
  os.makedirs(FIG, exist_ok=True)
  for axis, xlab in (('t', 't (0 = clean end at tau*, 1 = noise)'), ('snr', 'log10 SNR_eff = 2 log10(a R / b)')):
    fig, axes = plt.subplots(len(PRIORS), len(METRICS), figsize=(4.2 * len(METRICS), 11), sharex=True)
    for run in runs:
      R, gauss = round(run['R'], 1), run['forward_type'] == 'gaussian'
      xs = [row['t'] if axis == 't' else row['log10_snr'] for row in run['rows']]  # rows[-1]: t = 1, noise end
      for i, prior in enumerate(PRIORS):
        sigma = run['sigma']['uniform' if prior == 'uniform' else 'data_freq']
        for k, (m, _, _) in enumerate(METRICS):
          v = [row[prior][m] for row in run['rows']]
          axes[i, k].plot(xs, [u[1] for u in v], '-' if gauss else '--', color=color[R],
                          label=f'{"gauss" if gauss else "naive"} R={R}')
          axes[i, k].fill_between(xs, [u[0] for u in v], [u[2] for u in v], color=color[R], alpha=0.1)
          axes[i, k].scatter(xs[-1], sigma[m], marker='*', s=70, zorder=3, color=color[R] if gauss else 'none',
                             edgecolors=color[R])
    for i, prior in enumerate(PRIORS):
      for k, (m, lab, ylog) in enumerate(METRICS):
        axes[i, k].set_title(f'{prior}: {lab}', fontsize=10)
        axes[i, k].grid(alpha=0.3)
        if ylog:
          axes[i, k].set_yscale('log')
      axes[i, -1].axhline(12, color='k', lw=0.7, ls=':')
    for ax in axes[-1]:
      ax.set_xlabel(xlab)
    if axis == 'snr':
      axes[0, 0].invert_xaxis()  # shared x: invert once, clean end on the left as in the t figure
    axes[0, 0].legend(fontsize=7)
    n = ' / '.join(str(k) for k in sorted({r['n_state'] for r in runs}))
    fig.suptitle(f'Angular-Fisher Sigma_t = Cov_q[P_w v] on the tangent space (median, quartile band over {n} states; '
                 'star = Sigma, near-origin limit)')
    fig.tight_layout()
    path = f'{FIG}/angular_fisher_sigma_t_{axis}.png'
    tmp = f'{path}.{os.getpid()}.tmp.png'
    fig.savefig(tmp, dpi=120)
    os.replace(tmp, path)  # atomic: concurrent jobs redraw the same figure
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt')
  p.add_argument('--label')
  p.add_argument('--nt', type=int, default=51, help='timestep samples: np.linspace(0, 1, nt), t = 0, 0.02, ..., 1')
  p.add_argument('--plot', action='store_true', help='only redraw the figures from data/*.json')
  args = p.parse_args()
  if not args.plot:
    compute(args)
  plot()
