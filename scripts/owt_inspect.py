"""Summarize OWT frontier-eval JSONs with sample-grounded diversity stats.

Usage: python scripts/owt_inspect.py eval_runs/owt/<dir1> [<dir2> ...]
Prints per (model,step,NFE,T): GenPPL, entropy, %unique words, top-token share, + a snippet.
"""
import collections
import glob
import json
import os
import re
import sys


def stats(text):
    words = re.findall(r"\S+", text)
    if not words:
        return 0, 0.0, "", 0.0
    c = collections.Counter(words)
    top, topn = c.most_common(1)[0]
    return len(words), len(c) / len(words) * 100, top, topn / len(words) * 100


for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        o = json.load(open(f))
        s = o["config"]["sampler"]
        nw, uniq, top, topshare = stats(o["text"][0])
        # avg %unique across all samples
        uavg = sum(stats(t)[1] for t in o["text"]) / len(o["text"])
        print(f"{os.path.basename(f):42s} GenPPL={o['gen_ppl_first_chunk_retok']:6.1f} "
              f"H={o['entropy']:.2f} uniq~{uavg:.0f}% top={top!r}x{topshare:.0f}% "
              f"(n={len(o['text'])}, NFE={s['steps']}, T={s['temperature']})")
        print(f"    {o['text'][0][:160]!r}")
