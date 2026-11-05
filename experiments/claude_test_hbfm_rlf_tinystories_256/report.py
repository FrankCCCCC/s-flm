#!/usr/bin/env python
"""Aggregate claude_test_hbfm_rlf_tinystories_256: valid PPL (importance-weighted denoising-CE bound, NOT a
likelihood), GenPPL (gpt2-large retokenized), sample entropy and distinct samples per (HyLa arm, forward_type, seed,
step, horizon), from this project's outputs/<tag>/eval*/ and paired/<tag>/eval_tmax<T>/, plus the reused baselines of
claude_test_hbfm_tinystories_256 (sweep.OLD_BASELINES: eval/ = 30k at t_max 3.0, eval_step<N>[_tmax<T>]/). Prints the
tables and rewrites the block between the markers in RESULTS.md (created if missing). Read GenPPL WITH entropy and
distinct/64: low GenPPL + low entropy / few distinct samples = degenerate, not good.

Usage:  python report.py
"""
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402

RESULTS = f'{sweep.EXP}/RESULTS.md'
EVAL_DIR = re.compile(r'eval(?:_step(\d+))?(?:_tmax([\d.]+))?$')
ARM_ORDER = {'none': 0, 'cap5': 1}
FWD_ORDER = {'naive': 0, 'horosphere': 1}
HEADER = ('# claude_test_hbfm_rlf_tinystories_256 — Results\n\nHBFM on TinyStories seq 256 with random Laplacian '
          'features (HyLa) of the input state vs the plain coordinates (see `EXPERIMENT.md`). Regenerate with '
          '`python report.py`. `±` = sample std over seeds (ddof=1). val/ppl is the denoising-CE bound, not a '
          'likelihood.\n\n<!-- report:begin -->\n<!-- report:end -->\n')


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_eval_dir(ed):
    pj, gj = load(f'{ed}/ppl.json'), load(f'{ed}/samples_genppl.json')
    if pj is None and gj is None:
        return None
    text = gj.get('text', []) if gj else []
    return dict(ppl=pj.get('val/ppl') if pj else None, gen=gj.get('gen_ppl_first_chunk_retok') if gj else None,
                ent=gj.get('entropy') if gj else None, uniq=f'{len(set(text))}/{len(text)}' if text else 'n/a')


def main():
    rows = {}
    all_cells = list(sweep.cells()) + list(sweep.OLD_BASELINES)
    for arm, fwd, seed in all_cells:
        cdir = sweep.cell_dir(arm, fwd, seed)
        for ed in glob.glob(f'{cdir}/eval*'):
            me = EVAL_DIR.match(os.path.basename(ed))
            r = read_eval_dir(ed) if me else None
            if r is None:
                continue
            step = int(me.group(1)) if me.group(1) else sweep.MAX_STEPS
            tm = float(me.group(2)) if me.group(2) else sweep.T_MAX_DEFAULT
            rows[(arm, fwd, step, tm, 'own', seed)] = r
        for ed in glob.glob(f'{sweep.OUT}/paired/{sweep.tag_of(arm, fwd, seed)}/eval_tmax*'):
            me = EVAL_DIR.match(os.path.basename(ed))
            r = read_eval_dir(ed) if me else None
            if r is None:
                continue
            rows[(arm, fwd, sweep.MAX_STEPS, float(me.group(2)), 'paired', seed)] = r

    # stage-2 screen cells (sweep.py --screen): 5k-step runs of the reference code's HyLa hyperparameters
    for sc, nm in getattr(sweep, 'SCREEN', []):
        cdir = f'{sweep.OUT}/{sweep.screen_tag(sc, nm)}'
        for ed in glob.glob(f'{cdir}/eval*'):
            me = EVAL_DIR.match(os.path.basename(ed))
            r = read_eval_dir(ed) if me else None
            if r is None:
                continue
            tm = float(me.group(2)) if me.group(2) else sweep.T_MAX_DEFAULT
            rows[(f'cap5 sc{sc}' + (' unnorm' if nm == '0' else ''), 'naive', sweep.SCREEN_STEPS, tm, 'own', 1)] = r

    def fmt(x, p=2):
        return f'{x:.{p}f}' if isinstance(x, (int, float)) else 'n/a'

    def order(key):
        arm, fwd, step, tm, kind = key[:5]
        return (kind, FWD_ORDER[fwd], tm, -step, ARM_ORDER.get(arm, 2), arm) + tuple(key[5:])

    lines = ['| read | fwd | t_max | step | arm | seed | val/ppl | GenPPL | entropy | distinct |',
             '|---|---|---|---|---|---|---:|---:|---:|---:|']
    groups = {}
    for key in sorted(rows, key=order):
        arm, fwd, step, tm, kind, seed = key
        r = rows[key]
        lines.append(f'| {kind} | {fwd} | {tm:g} | {step} | {arm} | {seed} | {fmt(r["ppl"], 3)} | {fmt(r["gen"])} | '
                     f'{fmt(r["ent"], 3)} | {r["uniq"]} |')
        groups.setdefault(key[:5], []).append(r)

    def ms(vals, p):
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return 'n/a'
        return f'{statistics.mean(vals):.{p}f} ± {statistics.stdev(vals):.{p}f}' if len(vals) > 1 else f'{vals[0]:.{p}f}'

    agg = ['| read | fwd | t_max | step | arm | n | val/ppl | GenPPL | entropy |',
           '|---|---|---|---|---|---|---:|---:|---:|']
    for key, rs in sorted(groups.items(), key=lambda kv: order(kv[0])):
        arm, fwd, step, tm, kind = key
        agg.append(f'| {kind} | {fwd} | {tm:g} | {step} | {arm} | {len(rs)} | {ms([r["ppl"] for r in rs], 3)} | '
                   f'{ms([r["gen"] for r in rs], 2)} | {ms([r["ent"] for r in rs], 3)} |')
    table = (f'{len(rows)} evaluated (cell, step, horizon) triples.\n\n### Per read\n\n' + '\n'.join(lines)
             + '\n\n### Mean ± std over seeds\n\n' + '\n'.join(agg))
    print(table)
    if not os.path.exists(RESULTS):
        with open(RESULTS, 'w') as f:
            f.write(HEADER)
    with open(RESULTS) as f:
        doc = f.read()
    block = f'<!-- report:begin -->\n{table}\n<!-- report:end -->'
    doc = re.sub(r'<!-- report:begin -->.*?<!-- report:end -->', lambda _: block, doc, flags=re.S)
    with open(RESULTS, 'w') as f:
        f.write(doc)


if __name__ == '__main__':
    main()
