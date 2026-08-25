#!/usr/bin/env python
"""eflm_sde — EFLM's two diversity knobs head to head: temperature vs SDE eta.

Both knobs trade Gen. PPL for entropy on the same frozen 3-seed checkpoints, but
through different samplers:

  T-knob    exact velocity over the FULL vocab (top_k_velocity = -1), eta = 0.
            T reshapes p = softmax(logits/T), so the Euler target moves.
            Cells: outputs/eflm_sde/sd-*/tfrontier/nfe-*_t-*/
  eta-knob  top-1 velocity (top_k_velocity = 1) + the marginal-preserving SDE,
            g(t) = (1-t)^p. eta scales the injected Wiener noise.
            Cells: outputs/eflm_sde/sd-*/frontier/nfe-*_gt-*_eta-*/

The velocity setting differs between the arms because it has to: temperature is
argmax-invariant under top-1 velocity (RESULTS.md sec. 6), so "T-knob" and
"eta-knob" name whole sampler configurations, not one isolated variable.

Writes figures/knob_comparison_t_vs_eta.png and prints the comparison tables.

Example:
  python experiments/eflm_sde/compare_knobs.py
"""
import argparse, collections, glob, json, os, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T_RE = re.compile(r'/sd-(?P<seed>\d+)/tfrontier/nfe-(?P<nfe>\d+)_t-(?P<t>[\d.]+)/')
E_RE = re.compile(r'/sd-(?P<seed>\d+)/frontier/nfe-(?P<nfe>\d+)_gt-(?P<gt>\w+)'
                  r'_eta-(?P<eta>[\d.]+)/')
SEEDS = (1, 2, 3)
ETA_GTS = ('sqrt', 'linear', 'p75')   # const/quad were rejected (RESULTS.md sec. 4)
STYLE = {'T': ('temperature (exact vel.)', '#9467bd', 'o-'),
         'eta/sqrt': ('eta, g=sqrt', '#2ca02c', 's-'),
         'eta/linear': ('eta, g=linear', '#d62728', '^-'),
         'eta/p75': ('eta, g=p75', '#17becf', 'v-')}


def load(dir_):
    """-> curves[(arm, nfe, seed)] = [(H, PPL)], marker[nfe] = [(H, PPL)]"""
    curves, marker = collections.defaultdict(list), collections.defaultdict(list)
    for path in glob.glob(f'{dir_}/sd-*/tfrontier/nfe-*_t-*/samples_genppl.json'):
        m = T_RE.search(path)
        d = json.load(open(path))
        curves[('T', int(m['nfe']), int(m['seed']))].append(
            (d['entropy'], d['gen_ppl_first_chunk_retok']))
    for path in glob.glob(f'{dir_}/sd-*/frontier/nfe-*_gt-*_eta-*/samples_genppl.json'):
        m = E_RE.search(path)
        d = json.load(open(path))
        v = (d['entropy'], d['gen_ppl_first_chunk_retok'])
        if m['gt'] == 'ode':
            marker[int(m['nfe'])].append(v)
        elif m['gt'] in ETA_GTS:
            curves[(f"eta/{m['gt']}", int(m['nfe']), int(m['seed']))].append(v)
    return curves, marker


def band(curves, arm, nfe):
    """3-seed intersection of the reachable entropy range, or None."""
    lo, hi = [], []
    for sd in SEEDS:
        p = sorted(curves.get((arm, nfe, sd), []))
        if not p:
            return None
        lo.append(p[0][0])
        hi.append(p[-1][0])
    return max(lo), min(hi)


def at(curves, arm, nfe, h):
    """Gen. PPL at entropy h: interpolate per seed, then mean/sd. None if any
    seed does not bracket h."""
    vals = []
    for sd in SEEDS:
        p = sorted(curves.get((arm, nfe, sd), []))
        if not p:
            return None
        e = np.array([a for a, _ in p])
        g = np.array([b for _, b in p])
        if h < e.min() or h > e.max():
            return None
        vals.append(float(np.interp(h, e, g)))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def best_eta(curves, nfe, h):
    """-> (gt, mean, sd) of the best eta schedule at entropy h."""
    cand = [(g, at(curves, f'eta/{g}', nfe, h)) for g in ETA_GTS]
    cand = [(g, v) for g, v in cand if v]
    if not cand:
        return None
    g, v = min(cand, key=lambda x: x[1][0])
    return g, v[0], v[1]


def mean_curve(curves, arm, nfe):
    """Seed-mean (H, PPL) polyline, ordered along the knob (cells are aligned
    across seeds because every seed walks the same knob grid)."""
    rows = [sorted(curves.get((arm, nfe, sd), [])) for sd in SEEDS]
    if not all(rows):
        return None
    n = min(len(r) for r in rows)
    e = np.array([np.mean([r[i][0] for r in rows]) for i in range(n)])
    g = np.array([np.mean([r[i][1] for r in rows]) for i in range(n)])
    s = np.array([np.std([r[i][1] for r in rows], ddof=1) for i in range(n)])
    return e, g, s


def plot(curves, marker, nfes, out):
    ncol = min(4, len(nfes))
    nrow = int(np.ceil(len(nfes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow),
                             squeeze=False)
    for i, nfe in enumerate(nfes):
        ax = axes[i // ncol][i % ncol]
        drawn = []
        floor = np.inf
        for arm, (label, color, fmt) in STYLE.items():
            c = mean_curve(curves, arm, nfe)
            if c is None:
                continue
            e, g, s = c
            floor = min(floor, g.min())
            drawn.append((e, g))
            ax.plot(e, g, fmt, color=color, ms=3.5, lw=1.5, label=label)
            ax.fill_between(e, g - s, g + s, color=color, alpha=0.15, lw=0)
        if nfe in marker:
            e = np.mean([a for a, _ in marker[nfe]])
            g = np.mean([b for _, b in marker[nfe]])
            floor = min(floor, g)
            drawn.append((np.array([e]), np.array([g])))
            ax.plot(e, g, '*', color='k', ms=14, label='eta=0 ODE (k=1 vel.)')
        ax.set_yscale('log')
        top = 12 * floor
        ax.set_ylim(0.85 * floor, top)
        # x-window = where at least one curve is still on-scale, so a collapsed
        # eta tail (Gen. PPL in the thousands at H -> 5.3) cannot squash the band
        # where the knobs actually compete.
        vis = np.concatenate([e[g <= top] for e, g in drawn if (g <= top).any()])
        pad = 0.03 * (vis.max() - vis.min() + 1e-9)
        ax.set_xlim(vis.min() - pad, vis.max() + pad)
        ax.set(title=f'NFE = {nfe}', xlabel='per-sample unigram entropy (nats)',
               ylabel='Gen. PPL (gpt2-large)')
        ax.grid(alpha=0.3, which='both')
    for j in range(len(nfes), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    by_label = {}
    for row in axes:
        for ax in row:
            by_label.update(dict(zip(*ax.get_legend_handles_labels()[::-1])))
    axes[0][0].legend([by_label[k] for k in by_label], list(by_label), fontsize=8)
    fig.suptitle('EFLM diversity knobs: sampling temperature vs SDE eta '
                 '(mean ± sd over 3 training seeds, 512 samples/cell)\n'
                 'y-axis clipped at 12x each panel\'s minimum; collapsed cells '
                 'run off-scale', y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def tables(curves, marker, nfes):
    print('\n### A. Reachable entropy window per knob (3-seed intersection)\n')
    print('| NFE | temperature | eta (best-covering schedule) | eta=0 ODE marker |')
    print('|---|---|---|---|')
    for nfe in nfes:
        bt = band(curves, 'T', nfe)
        be = [band(curves, f'eta/{g}', nfe) for g in ETA_GTS]
        be = [b for b in be if b]
        lo = min(b[0] for b in be) if be else None
        hi = max(b[1] for b in be) if be else None
        mk = (f'{np.mean([a for a, _ in marker[nfe]]):.3f}'
              if nfe in marker else '—')
        print(f'| {nfe} | ' + (f'{bt[0]:.3f} – {bt[1]:.3f}' if bt else '—') +
              ' | ' + (f'{lo:.3f} – {hi:.3f}' if be else '—') + f' | {mk} |')

    print('\n### B. Head to head at matched entropy (overlap band)\n')
    print('| NFE | H | temperature | best eta | winner | margin |')
    print('|---|---|---|---|---|---|')
    for nfe in nfes:
        for h in (4.15, 4.20, 4.25, 4.30, 4.35, 4.45):
            t = at(curves, 'T', nfe, h)
            e = best_eta(curves, nfe, h)
            if t is None and e is None:
                continue
            ts = f'{t[0]:.2f} ±{t[1]:.2f}' if t else '—'
            es = f'{e[1]:.2f} ±{e[2]:.2f} ({e[0]})' if e else '—'
            if t and e:
                win = 'T' if t[0] < e[1] else 'eta'
                marg = f'{abs(1 - min(t[0], e[1]) / max(t[0], e[1])) * 100:.0f}%'
            else:
                win = marg = '—'
            print(f'| {nfe} | {h:.2f} | {ts} | {es} | {win} | {marg} |')

    print('\n### C. EFLM Pareto envelope: which knob owns which entropy\n')
    print('| NFE | ' + ' | '.join(f'H={h:.2f}' for h in
                                  (3.85, 3.95, 4.05, 4.15, 4.25, 4.35, 4.45)) + ' |')
    print('|---' * 8 + '|')
    for nfe in nfes:
        row = []
        for h in (3.85, 3.95, 4.05, 4.15, 4.25, 4.35, 4.45):
            t = at(curves, 'T', nfe, h)
            e = best_eta(curves, nfe, h)
            opts = []
            if t:
                opts.append(('T', t[0]))
            if e:
                opts.append((f'eta/{e[0]}', e[1]))
            if not opts:
                row.append('—')
                continue
            k, v = min(opts, key=lambda x: x[1])
            row.append(f'{v:.1f} ({k})')
        print(f'| {nfe} | ' + ' | '.join(row) + ' |')

    print('\n### D. Price of entropy: Gen. PPL multiplier per +0.1 nats\n')
    print('| NFE | temperature | best eta schedule |')
    print('|---|---|---|')
    for nfe in nfes:
        out = []
        for arm in ('T', 'eta'):
            bt = band(curves, 'T', nfe)
            if bt is None:
                out.append('—')
                continue
            lo, hi = max(bt[0], 4.15), min(bt[1], 4.35)
            if hi - lo < 0.05:
                out.append('—')
                continue
            a = at(curves, 'T', nfe, lo) if arm == 'T' else best_eta(curves, nfe, lo)
            b = at(curves, 'T', nfe, hi) if arm == 'T' else best_eta(curves, nfe, hi)
            if a is None or b is None:
                out.append('—')
                continue
            av = a[0] if arm == 'T' else a[1]
            bv = b[0] if arm == 'T' else b[1]
            out.append(f'{(bv / av) ** (0.1 / (hi - lo)):.2f}x')
        print(f'| {nfe} | ' + ' | '.join(out) + ' |')
    print('\n(measured over the shared H in [4.15, 4.35] window; lower is better)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', default=f'{REPO}/outputs/eflm_sde')
    ap.add_argument('--out', default=f'{REPO}/experiments/eflm_sde/figures/'
                                     'knob_comparison_t_vs_eta.png')
    args = ap.parse_args()
    curves, marker = load(args.dir)
    nfes = sorted({k[1] for k in curves if k[0] == 'T' and k[1] > 1})
    plot(curves, marker, nfes, args.out)
    tables(curves, marker, nfes)


if __name__ == '__main__':
    main()
