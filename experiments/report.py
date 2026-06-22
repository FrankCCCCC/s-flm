#!/usr/bin/env python
"""Generate a PPL + GenPPL report for a TinyStories experiment (slides jun25_2026).

Scans outputs/<exp>/<cell>/eval/{ppl.json, samples_genppl.json}, extracts validation
perplexity and generative perplexity, and writes experiments/<exp>/RESULTS.md.

  ppl.json            -> Lightning validate() metrics; PPL under a 'val/ppl'-like key
  samples_genppl.json -> gen_ppl_first_chunk_retok  (GenPPL)

Usage:  python report.py <exp_name>      e.g.  python report.py hflm_sweep_tinystories
"""
import glob
import json
import os
import sys

REPO = '/share/thickstun/sychou/workspace/research/s-flm'


def find_ppl(d):
    for k in ('val/ppl', 'val_ppl', 'test/ppl'):
        if k in d:
            return d[k]
    for k, v in d.items():
        if 'ppl' in k.lower() and isinstance(v, (int, float)):
            return v
    return None


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    exp = sys.argv[1]
    out = f'{REPO}/outputs/{exp}'
    rows = []
    for ed in sorted(glob.glob(f'{out}/*/eval')):
        tag = os.path.basename(os.path.dirname(ed))
        pj, gj = load(f'{ed}/ppl.json'), load(f'{ed}/samples_genppl.json')
        ppl = find_ppl(pj) if pj else None
        gen = gj.get('gen_ppl_first_chunk_retok') if gj else None
        ent = gj.get('entropy') if gj else None
        if ppl is None and gen is None:
            continue
        rows.append((tag, ppl, gen, ent))

    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (int, float)) else 'n/a'

    # A low GenPPL paired with low entropy means repetitive/degenerate generation, NOT quality.
    def flag(gen, ent):
        return ' ⚠collapse?' if isinstance(ent, (int, float)) and ent < 3.0 else ''

    lines = [f'# {exp} — Results', '',
             f'{len(rows)} cell(s) with eval metrics. Valid PPL (held-out likelihood bound; '
             'note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). '
             'GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it '
             'WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.',
             '', '| Cell | Valid PPL | GenPPL | Entropy | Note |', '|---|---|---|---|---|']
    for tag, ppl, gen, ent in sorted(rows, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0)):
        lines.append(f'| {tag} | {fmt(ppl)} | {fmt(gen)} | {fmt(ent)} |{flag(gen, ent)} |')
    if rows:
        best_ppl = min((r for r in rows if r[1] is not None), key=lambda r: r[1], default=None)
        # Best GenPPL only among non-collapsed cells (entropy >= 3.0), else GenPPL is gamed by repetition.
        valid_gen = [r for r in rows if r[2] is not None and isinstance(r[3], (int, float)) and r[3] >= 3.0]
        best_gen = min(valid_gen, key=lambda r: r[2], default=None)
        lines += ['', '## Best',
                  f'- Lowest Valid PPL: **{best_ppl[0]}** = {fmt(best_ppl[1])}' if best_ppl else '',
                  (f'- Lowest GenPPL (non-collapsed, entropy≥3.0): **{best_gen[0]}** = {fmt(best_gen[2])}'
                   if best_gen else '- Lowest GenPPL: (all evaluated cells look collapsed)')]

    dest = f'{REPO}/experiments/{exp}/RESULTS.md'
    with open(dest, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{exp}: wrote {dest} ({len(rows)} cells)')


if __name__ == '__main__':
    main()
