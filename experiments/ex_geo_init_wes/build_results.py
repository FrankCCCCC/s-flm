#!/usr/bin/env python
"""Regenerate the per-difficulty LR-sweep section(s) of RESULTS.md from results.json.

Splices in a fresh '# RESULTS - <diff> LR sweep' block per difficulty (replacing any
existing one), so it's safe to re-run as more cells finish. Keeps everything above the
first generated marker untouched.
"""
import json
import os

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXP))
BASE = f'{REPO}/eval_runs/sudoku/exgiw'
RESULTS = f'{EXP}/RESULTS.md'
MARKER = '<!-- LR-SWEEP-SECTIONS (auto-generated below; do not edit by hand) -->'

GEOS = ['sfm', 'eflm', 'hflm']
INITS = ['ngpt', 'random', 'unit_var', 'hyperbolic']
DIMS = [512, 256, 128]
LRS = ['5e-5', '8e-5', '1e-4', '3e-4', '5e-4', '1e-3']
DIFFS = ['medium', 'hard', 'easy']
INITTAG = {'ngpt': 'ng', 'random': 'rd', 'unit_var': 'uv', 'hyperbolic': 'hy'}
GEOLAB = {'sfm': 'S-FLM', 'eflm': 'E-FLM', 'hflm': 'H-FLM'}


def acc(geo, init, d, lr, diff):
    # lr=3e-4 is the untagged baseline for medium, but lr-tagged for hard/easy; try both.
    if lr == '3e-4':
        tags = [f'{geo}_{init}_d{d}_{diff}', f'{geo}_{init}_d{d}_lr3e-4_{diff}']
    else:
        tags = [f'{geo}_{init}_d{d}_lr{lr}_{diff}']
    for tag in tags:
        f = f'{BASE}/{tag}/results.json'
        if os.path.exists(f):
            try:
                return json.load(open(f))['accuracy'] * 100
            except Exception:
                pass
    return None


def section(diff):
    out = []
    cells = [(g, i, d, lr) for g in GEOS for i in INITS for d in DIMS for lr in LRS]
    n = sum(acc(g, i, d, lr, diff) is not None for g, i, d, lr in cells)
    out.append(f'\n# RESULTS — {diff.capitalize()} LR sweep (geometry × init × dim × LR)\n')
    out.append(f'Sudoku-{diff} exact-match accuracy (%), 20k steps, eval @180 exact/greedy/'
               f'tkv=-1, effective batch 256. **{n}/216 cells** done (— = still running).\n')
    out.append('**Per-geometry LR means** (avg over available init×dim cells):\n')
    out.append('| geometry | ' + ' | '.join(LRS) + ' | best LR |')
    out.append('|' + '---|' * (len(LRS) + 2))
    for geo in GEOS:
        ms = []
        for lr in LRS:
            v = [acc(geo, i, d, lr, diff) for i in INITS for d in DIMS]
            v = [x for x in v if x is not None]
            ms.append(sum(v) / len(v) if v else None)
        best = max((lr for lr, m in zip(LRS, ms) if m is not None),
                   key=lambda lr: ms[LRS.index(lr)], default='—')
        out.append('| ' + geo.upper() + ' | '
                   + ' | '.join('—' if m is None else f'{m:.1f}' for m in ms)
                   + f' | **{best}** |')
    # Per-geometry LR Best: max over the 12 init×dim cells at each LR, tagged by the
    # best cell's init; the row's overall max is bolded.
    out.append('\n**Per-geometry LR Best** (Best over the 12 init×dim cells):\n')
    out.append('Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · '
               '<uv>unit_var</uv> · <hy>hyperbolic</hy>\n')
    out.append('| Geo | ' + ' | '.join(LRS) + ' | best LR |')
    out.append('|' + '---|' * (len(LRS) + 2))
    for geo in GEOS:
        per = []  # (lr, value, init) of the best init×dim cell at each LR
        for lr in LRS:
            cand = [(acc(geo, i, d, lr, diff), i)
                    for i in INITS for d in DIMS if acc(geo, i, d, lr, diff) is not None]
            per.append((lr, *max(cand, key=lambda x: x[0])) if cand else (lr, None, None))
        valid = [(lr, v, i) for lr, v, i in per if v is not None]
        rmlr = max(valid, key=lambda x: x[1])[0] if valid else '—'
        row = []
        for lr, v, i in per:
            if v is None:
                row.append('—')
            else:
                s = f'**{v:.1f}**' if lr == rmlr else f'{v:.1f}'
                row.append(f'<{INITTAG[i]}>{s}</{INITTAG[i]}>')
        out.append('| ' + GEOLAB[geo] + ' | ' + ' | '.join(row) + f' | {rmlr} |')
    for geo in GEOS:
        out.append(f'\n### {geo.upper()}\n')
        out.append('| init | dim | ' + ' | '.join(LRS) + ' |')
        out.append('|' + '---|' * (len(LRS) + 2))
        for init in INITS:
            for d in DIMS:
                row = ' | '.join('—' if acc(geo, init, d, lr, diff) is None
                                 else f'{acc(geo, init, d, lr, diff):.1f}' for lr in LRS)
                out.append(f'| {init} | {d} | {row} |')
    return '\n'.join(out) + '\n'


def main():
    text = open(RESULTS).read()
    # Cut off any previously-generated LR-sweep block: at the marker (later runs) or at
    # the first generated "# RESULTS — ... LR sweep" header (first run; the doc title
    # uses "# RESULTS:" with a colon, so the em-dash form only matches generated blocks).
    cut = len(text)
    if MARKER in text:
        cut = min(cut, text.index(MARKER))
    em = '# RESULTS —'
    if em in text:
        cut = min(cut, text.index(em))
    head = text[:cut].rstrip()
    body = MARKER + '\n'
    for diff in DIFFS:
        if any(acc(g, i, d, lr, diff) is not None
               for g in GEOS for i in INITS for d in DIMS for lr in LRS):
            body += section(diff)
    open(RESULTS, 'w').write(head + '\n\n' + body)
    print('rebuilt RESULTS.md')


if __name__ == '__main__':
    main()
