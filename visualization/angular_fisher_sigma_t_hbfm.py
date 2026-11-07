#!/usr/bin/env python
"""angular_fisher_sigma_t.py for HBFM checkpoints: the slides' Sigma_t along t (setup.md, slides/sep25_2026).

HBFM runs a Brownian bridge from the origin of a product of M Poincare balls H^d_K toward the word's boundary point
(u_{v,1}, ..., u_{v,M}), u_{v,m} = the m-th d-block of the word's embedding, normalized. The bridge's direction law is
the Poisson kernel, so the Bayes posterior is the slides' one, factor by factor:
  q(v | z_t) ∝ q(v) prod_m D_m(v)^{-(d-1)},  D_m(v) = cosh s_m - sinh s_m w_m^T u_{v,m},  s_m = kappa rho_m
(numeric/horosphere.py compute_horosphere is this log-likelihood; forward_type=horosphere adds it to the logits).
Sigma_t = Cov_q[v~_t], v~_t = (P_{w_m} u_{v,m} / D_m(v))_m, on the tangent space of the M spheres (M (d-1) dims).
Per unit angle dw_m the score is (d-1) sinh(s_m) (v~_m - E v~_m); in the manifold's own metric (length
sinh(s_m) |dw_m| / kappa) it is (d-1) kappa (v~_m - E v~_m), so with one shared d and K, Sigma_t is the Riemannian
angular Fisher up to the constant (d-1)^2 kappa^2 (check_angular_fisher_hbfm.py verifies the score against
finite-difference KL).
Adaptive curvature (algo.curvature_bound > 1) runs position i on the clock lambda_i tau with the base-K readout, so the
posterior given z_t is the same kernel; states are drawn by the model's own q_xt at heat time lambda_i tau(t).
Priors as in angular_fisher_sigma_t.py: data_freq / uniform (exact Bayes, v* ~ q(v), state at a random position's
lambda) and model_pred (p_theta(x^i | z_t) on validation text, a plug-in for the contextual posterior).
Sigma (all M d dims) = Cov_{q_data}[v] (uniform: Cov_uniform[v]), the near-origin limit.
Figures: per (prior, metric) the median over the N_STATE states at each t and their 25-75% quartile band;
dashed = naive, solid = horosphere, one color per curvature config; star = Sigma at the noise end.

Usage (compute node, 1 GPU; SFLM_REPO = a checkout whose HBFM has algo.curvature_bound, e.g. s-flm-dev2;
writes data/hbfm/<label>.json, then redraws figures/angular_fisher_sigma_t_hbfm_{t,heat}.png from every data/hbfm/*.json):
  SFLM_REPO=<checkout> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python visualization/angular_fisher_sigma_t_hbfm.py \
    --ckpt outputs/.../checkpoints/last.ckpt --label naive_ada_ub-1 [--nt 51]
  python visualization/angular_fisher_sigma_t_hbfm.py --plot   # figures only
"""
import argparse, glob, json, os, types

import numpy as np, torch

from angular_fisher_sigma_t import METRICS, N_SEQ, N_STATE, PRIORS, cov, spectrum_metrics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = f'{REPO}/experiments/eflm_cov_reparam_fisher_cond'
DATA, FIG = f'{EXP}/data/hbfm', f'{EXP}/figures'
CHUNK = 16  # states per posterior pass: compute_horosphere holds [CHUNK, V, M, d] float64


def tangent_basis(w):
  """[..., d] unit w -> [..., d-1, d] orthonormal rows spanning w^perp: rows 1: of the Householder reflection
  H = I - 2 u u^T / |u|^2, u = w + sign(w_0) e_0, which maps w to -sign(w_0) e_0."""
  u = w.clone()
  u[..., 0] += torch.where(w[..., 0] >= 0, 1.0, -1.0)
  eye = torch.eye(w.shape[-1], dtype=w.dtype, device=w.device)
  return (eye - 2 * u[..., :, None] * u[..., None, :] / (u * u).sum(-1)[..., None, None])[..., 1:, :]


def tilde_v(Ub, s, w):
  """v~_t of every word at one state in tangent coordinates: Ub [V, M, d] unit blocks, s [M] = kappa rho, w [M, d]
  unit -> [V, M (d-1)], B_m u_{v,m} / D_m(v) (B_m w_m = 0, so B_m P_w u = B_m u). D in the half-angle form
  e^s sin^2(a/2) + e^-s cos^2(a/2), cancellation-free as in compute_horosphere."""
  D = (s.exp() * (Ub - w).square().sum(-1) + (-s).exp() * (Ub + w).square().sum(-1)) / 4  # [V, M]
  return (torch.einsum('mkd,vmd->vmk', tangent_basis(w), Ub) / D[..., None]).flatten(1)


@torch.no_grad()
def compute(args):
  from loss_geometry import ALGO_BY_NAME, REPO as CODE, _load_config
  from codebook_eigen_proj_scatter import train_counts
  import algo, dataloader, main as main_mod
  from numeric.geo_bridge import GeoUtils
  from numeric.horosphere import HorosphereGeometry
  torch.manual_seed(0)  # q_xt draws its bridge from the global CUDA RNG
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
  cfg = _load_config(run_dir, args.ckpt, types.SimpleNamespace(cache_dir=f'{REPO}/data_cache', batch_size=8))
  assert cfg.algo.name == 'hbfm' and not cfg.algo.invert_time_convention  # t = 1 is the origin
  tok = dataloader.get_tokenizer(cfg)
  m = main_mod._load_from_checkpoint(ALGO_BY_NAME[cfg.algo.name], cfg, tok).cuda().eval()
  m._eval_mode()  # EMA weights, as at sampling time
  W = m.word_embedding.detach().double()  # [V, M d]
  dims, curvs = m.prod_factor_dim, m.prod_factor_gaussian_curvature
  assert len(set(dims)) == 1 and len(set(curvs)) == 1  # one shared factor: Sigma_t is the Riemannian Fisher
  V, M, dm = W.shape[0], len(dims), dims[0]
  kappa = 1.0 / GeoUtils._curvature_scale(curvs[0])
  Ub = W.unflatten(-1, (M, dm))
  Ub = Ub / Ub.norm(dim=-1, keepdim=True)  # u_{v,m} in S^{d-1}
  L = cfg.model.length
  ada = cfg.algo.curvature_bound > 1
  # lambda_i is read at the t = 0 state (every bridge at the origin; TinyStories has no prompt)
  lam = (m.curvature_scale(torch.zeros(1, L, M * dm, device='cuda', dtype=torch.float64))[0] if ada
         else torch.ones(L, device='cuda', dtype=torch.float64))
  cnt = torch.as_tensor(train_counts(cfg, V), device='cuda', dtype=torch.float64)
  seen = cnt > 0
  prior = {'data_freq': cnt / cnt.sum(), 'uniform': seen.double() / seen.sum()}
  out = dict(label=args.label, ckpt=os.path.abspath(args.ckpt), code=CODE, forward_type=cfg.algo.forward_type,
             loss_type=cfg.algo.loss_type, curvature_bound=cfg.algo.curvature_bound,
             mode=cfg.algo.time_conversion_mode, ub=cfg.algo.time_range_upper_bound, M=M, d=dm, K=curvs[0], V=V,
             n_seen=int(seen.sum()), n_state=N_STATE, lam=[lam.min().item(), lam.median().item(), lam.max().item()],
             sigma={k: spectrum_metrics(torch.linalg.eigvalsh(cov(Ub.flatten(1), p)).flip(0).clamp(min=0))
                    for k, p in prior.items()}, rows=[])
  print(f'{args.label}: forward_type={cfg.algo.forward_type} loss={cfg.algo.loss_type} curvature_bound='
        f'{cfg.algo.curvature_bound} {out["mode"]} ub={out["ub"]} (H^{dm}_{curvs[0]})^{M} V={V} seen={out["n_seen"]} '
        f'lambda min/median/max={out["lam"]} | Sigma: ' +
        ' | '.join(f'{k} ' + ' '.join(f'{kk}={vv:.3g}' for kk, vv in s.items()) for k, s in out['sigma'].items()),
        flush=True)

  def polar(xt):  # Poincare-Cartesian [N, M d] -> s = kappa rho [N, M], w [N, M, d]
    rho, th = GeoUtils.poincare_cartesian_to_hyperbolic_polar_prod(xt, dims, curvs)
    return rho * kappa, th.unflatten(-1, (M, dm))

  def posterior(logp, s, w):  # exact Bayes: log q(v) + Poisson-kernel log-likelihood, CHUNK states at a time
    return torch.cat([(logp + HorosphereGeometry.compute_horosphere(
      theta=w[i:i + CHUNK].flatten(1)[:, None], radius=s[i:i + CHUNK, None] / kappa, word_embedding=W,
      prod_factor_dim=dims, prod_factor_gaussian_curvature=curvs, readout_dtype=torch.float64,
      forward_chunked=False)[:, 0]).log_softmax(-1) for i in range(0, s.shape[0], CHUNK)])

  g = torch.Generator(device='cuda').manual_seed(0)
  ds = dataloader.get_dataset(cfg, tok, mode='valid')
  x0 = torch.stack([torch.as_tensor(ds[i]['input_ids']) for i in range(0, 100 * N_SEQ, 100)]).cuda()
  for t in np.linspace(0.0, 1.0, args.nt):
    _, alpha = m.noise(torch.full((1, 1), float(t), device='cuda'))
    tau, _ = m.time_conversion(alpha, invert_time_convention=False, exp_rate=cfg.algo.time_exp_rate,
                               mode=cfg.algo.time_conversion_mode, range_upper_bound=cfg.algo.time_range_upper_bound)
    row = dict(t=float(t), alpha=alpha.item(), heat=tau.item() * lam.median().item())
    states = {}
    for k, p in prior.items():  # v* ~ q(v), drawn by the model's forward process at a random inner position's clock
      v = torch.multinomial(p, N_STATE, replacement=True, generator=g)
      pos = torch.randint(1, L - 1, (N_STATE,), device='cuda', generator=g)
      xt, _ = m.q_xt(v[:, None], alpha.expand(N_STATE, 1), use_pure_noise=False,
                     curvature=lam[pos][:, None] if ada else None)
      s, w = polar(xt[:, 0])
      states[k] = (posterior(p.log(), s, w), s, w)
    logqs, xs = [], []  # model_pred: inner positions (BOS / EOS excluded) of N_SEQ sequences, 8 per forward pass
    for b in range(0, N_SEQ, 8):
      al = alpha.expand(8, 1)
      xt, _ = m.q_xt(x0[b:b + 8], al, use_pure_noise=False, curvature=lam.expand(8, L) if ada else None)
      with torch.autocast('cuda', dtype=torch.bfloat16):
        logp = m.forward(xt=xt, sigma=m._sigma_from_alphat(al), context=algo.trainer_base.TrainingContext())
      pick = torch.randperm(8 * (L - 2), device='cuda', generator=g)[:N_STATE * 8 // N_SEQ]
      logqs.append(logp.float()[:, 1:L - 1].reshape(-1, V)[pick].double())
      xs.append(xt[:, 1:L - 1].reshape(-1, M * dm)[pick].double())
    states['model_pred'] = (torch.cat(logqs), *polar(torch.cat(xs)))
    row['s_median'] = states['data_freq'][1].median().item()  # kappa rho_t, the slides' near-origin variable
    for k, (logq, s, w) in states.items():
      ms = [spectrum_metrics(torch.linalg.eigvalsh(cov(tilde_v(Ub, sj, wj), lq.exp())).flip(0).clamp(min=0))
            for lq, sj, wj in zip(logq, s, w)]
      row[k] = {key: np.quantile([mm[key] for mm in ms], [0.25, 0.5, 0.75]).tolist() for key in ms[0]}
    out['rows'].append(row)
    print(f't={t:.2f} heat={row["heat"]:.3e} median kappa rho={row["s_median"]:.3f} | ' + ' | '.join(
      f'{k}: top1 {row[k]["top1"][1]:.3f} k90 {row[k]["k90"][1]:4.0f} k99 {row[k]["k99"][1]:4.0f} '
      f'pr {row[k]["pr"][1]:6.1f} cond {row[k]["log10_cond"][1]:5.2f}' for k in PRIORS), flush=True)
  os.makedirs(DATA, exist_ok=True)
  with open(f'{DATA}/{args.label}.json.tmp', 'w') as f:
    json.dump(out, f, indent=1)
  os.replace(f'{DATA}/{args.label}.json.tmp', f'{DATA}/{args.label}.json')  # atomic: concurrent jobs read data/hbfm
  print(f'wrote {DATA}/{args.label}.json', flush=True)


def plot():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  runs = [json.load(open(f)) for f in sorted(glob.glob(f'{DATA}/*.json'))]
  runs.sort(key=lambda r: (r['forward_type'] == 'horosphere', -r['curvature_bound']))
  os.makedirs(FIG, exist_ok=True)
  for axis, xlab in (('t', 't (0 = clean end at heat time ub, 1 = origin)'),
                     ('heat', 'effective heat time lambda tau(t) (log)')):
    fig, axes = plt.subplots(len(PRIORS), len(METRICS), figsize=(4.2 * len(METRICS), 11), sharex=True)
    for run in runs:
      ada, horo = run['curvature_bound'] > 1, run['forward_type'] == 'horosphere'
      color = 'C0' if ada else 'C1'
      label = (f'{"horo" if horo else "naive"} {"adaptive" if ada else "fixed"} curv. ub {run["ub"]:g}'
               + (f' ({run["loss_type"]})' if run['loss_type'] != 'ce' else ''))
      xs = [row['t'] if axis == 't' else row['heat'] for row in run['rows']]  # rows[-1]: t = 1, the origin
      for i, prior in enumerate(PRIORS):
        sigma = run['sigma']['uniform' if prior == 'uniform' else 'data_freq']
        for k, (mt, _, _) in enumerate(METRICS):
          v = [row[prior][mt] for row in run['rows']]
          axes[i, k].plot(xs, [u[1] for u in v], '-' if horo else '--', color=color, label=label)
          axes[i, k].fill_between(xs, [u[0] for u in v], [u[2] for u in v], color=color, alpha=0.1)
          axes[i, k].scatter(xs[-1], sigma[mt], marker='*', s=70, zorder=3, color=color if horo else 'none',
                             edgecolors=color)
    for i, prior in enumerate(PRIORS):
      for k, (mt, lab, _) in enumerate(METRICS):
        axes[i, k].set_title(f'{prior}: {lab}', fontsize=10)
        axes[i, k].grid(alpha=0.3)
      axes[i, -1].axhline(12, color='k', lw=0.7, ls=':')
    for ax in axes[-1]:
      ax.set_xlabel(xlab)
    if axis == 'heat':
      axes[0, 0].set_xscale('log')
      axes[0, 0].invert_xaxis()  # shared x: invert once, clean end on the left as in the t figure
    axes[0, 0].legend(fontsize=7)
    n = ' / '.join(str(k) for k in sorted({r['n_state'] for r in runs}))
    fig.suptitle('HBFM angular-Fisher Sigma_t = Cov_q[(P_w u_m / D_m)_m] on the tangent space of (S^2)^M '
                 f'(median, quartile band over {n} states; star = Sigma, near-origin limit)')
    fig.tight_layout()
    path = f'{FIG}/angular_fisher_sigma_t_hbfm_{axis}.png'
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
  p.add_argument('--plot', action='store_true', help='only redraw the figures from data/hbfm/*.json')
  args = p.parse_args()
  if not args.plot:
    compute(args)
  plot()
