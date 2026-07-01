#!/usr/bin/env python
"""Collect the TinyStories decode-policy sweep results into a tidy table + plot.

Scans outputs/tinystories/eval/<model>_30k_s<steps>_<nr>_<kv>/samples_genppl.json
(produced by the sweep array) plus the per-model held-out val/ppl from
outputs/tinystories/eval/<model>_30k/ppl.json (or the earlier hflm dirs), and writes:
  - experiments/tinystories/sweep_results.csv
  - experiments/tinystories/sweep_genppl_vs_nfe.png
  - a printed markdown table

Within each (steps, noise_removal, top_k_velocity) cell the decode policy is identical
across models; only the geometry differs. top_k_velocity=1 = aligned (all 3 models);
top_k_velocity=-1 = sphere-native reference (sfm / sfm_adaptive only).
"""
import json, os, re, csv, glob
from collections import defaultdict

ROOT = "/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm"
EVAL = os.path.join(ROOT, "outputs/tinystories/eval")

CELL_RE = re.compile(r"^(?P<model>sfm_adaptive|sfm|hflm)_30k_s(?P<steps>\d+)_(?P<nr>anc|grd)_(?P<kv>kv1|kvm1)$")
NR = {"anc": "ancestral", "grd": "greedy"}
KV = {"kv1": 1, "kvm1": -1}


def load_genppl():
    rows = []
    for d in sorted(glob.glob(os.path.join(EVAL, "*_30k_s*"))):
        m = CELL_RE.match(os.path.basename(d))
        if not m:
            continue
        jp = os.path.join(d, "samples_genppl.json")
        if not os.path.isfile(jp):
            print(f"  [missing] {os.path.basename(d)}/samples_genppl.json")
            continue
        with open(jp) as f:
            j = json.load(f)
        rows.append(dict(
            model=m["model"], steps=int(m["steps"]),
            noise_removal=NR[m["nr"]], top_k_velocity=KV[m["kv"]],
            genppl=j.get("gen_ppl_first_chunk_retok"),
            entropy=j.get("entropy"), avg_nfe=j.get("avg_nfe"),
        ))
    return rows


def load_valppl():
    out = {}
    for model in ("sfm", "sfm_adaptive", "hflm"):
        for cand in (f"{model}_30k", f"{model}_30k_ppl"):
            jp = os.path.join(EVAL, cand, "ppl.json")
            if os.path.isfile(jp):
                with open(jp) as f:
                    j = json.load(f)
                # ppl.json schema may vary; try common keys
                v = j.get("val/ppl") or j.get("ppl") or j.get("val_ppl")
                if isinstance(j, dict) and v is None:
                    for k, val in j.items():
                        if "ppl" in k.lower():
                            v = val; break
                out[model] = v
                break
    return out


def main():
    rows = load_genppl()
    valppl = load_valppl()
    rows.sort(key=lambda r: (r["model"], r["top_k_velocity"], r["noise_removal"], r["steps"]))

    csv_path = os.path.join(ROOT, "experiments/tinystories/sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "top_k_velocity", "noise_removal", "steps", "genppl", "entropy", "avg_nfe"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path} ({len(rows)} cells)\n")

    # GenPPL is THE cross-geometry-fair metric (external gpt2-large judge on decoded tokens).
    # Markdown: GenPPL (+entropy guard) by steps, one block per (top_k_velocity, noise_removal).
    by = defaultdict(dict)  # (kv,nr,steps) -> {model: row}
    steps_set, models = set(), set()
    for r in rows:
        by[(r["top_k_velocity"], r["noise_removal"], r["steps"])][r["model"]] = r
        steps_set.add(r["steps"]); models.add(r["model"])
    steps_sorted = sorted(steps_set)
    models_sorted = [m for m in ("sfm", "sfm_adaptive", "hflm") if m in models]

    print("## GenPPL — the fair cross-geometry comparison (gpt2-large; lower=better)")
    print("Each cell `GenPPL (H=entropy)`. Watch entropy: a low GenPPL with collapsed entropy")
    print("is degenerate/repetitive text, not quality.\n")
    for kv in (1, -1):
        for nr in ("ancestral", "greedy"):
            cells = {s: by.get((kv, nr, s), {}) for s in steps_sorted}
            if not any(cells.values()):
                continue
            tag = "aligned (top_k_velocity=1)" if kv == 1 else "sphere-native (top_k_velocity=-1)"
            print(f"### {tag}, noise_removal={nr}")
            print("| steps (NFE) | " + " | ".join(models_sorted) + " |")
            print("|---|" + "---|" * len(models_sorted))
            for s in steps_sorted:
                def fmt(m):
                    c = cells[s].get(m)
                    if not c or not isinstance(c.get("genppl"), (int, float)):
                        return "—"
                    h = c.get("entropy")
                    hs = f" (H={h:.2f})" if isinstance(h, (int, float)) else ""
                    return f"{c['genppl']:.2f}{hs}"
                print(f"| {s} | " + " | ".join(fmt(m) for m in models_sorted) + " |")
            print()

    print("## Held-out denoising CE (reported as `val/ppl`) — DIAGNOSTIC ONLY, not a perplexity")
    print("For SFM/HFLM `nll_per_token` returns the UNWEIGHTED denoising cross-entropy")
    print("(algo.py:261,394 delete `dalpha_t`; cf. MDLM algo.py:128 which keeps the ELBO weight),")
    print("averaged over the model's OWN noise schedule (trainer_base.py:551). So this is NOT a")
    print("token-NLL bound: comparable only across checkpoints of the SAME model — never across")
    print("geometries or across the two sphere noise schedules. Use GenPPL above to rank models.\n")
    for m, v in valppl.items():
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        print(f"- {m} (30k): exp(mean denoising CE) = {vs}")
    print()

    # Plot (best-effort; skip if matplotlib unavailable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, kv in zip(axes, (1, -1)):
            for model in models_sorted:
                for nr, ls in (("ancestral", "-"), ("greedy", "--")):
                    pts = sorted((r["steps"], r["genppl"]) for r in rows
                                 if r["model"] == model and r["top_k_velocity"] == kv
                                 and r["noise_removal"] == nr and r["genppl"] is not None)
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, ls, marker="o", label=f"{model}/{nr}")
            ax.set_xscale("log", base=2); ax.set_xlabel("sampling steps (NFE)")
            ax.set_title(f"top_k_velocity={kv}" + (" (aligned)" if kv == 1 else " (sphere-native)"))
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("GenPPL (gpt2-large, first-chunk retok)")
        fig.suptitle("TinyStories 30k: GenPPL vs NFE — geometry comparison under identical decode")
        fig.tight_layout()
        png = os.path.join(ROOT, "experiments/tinystories/sweep_genppl_vs_nfe.png")
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
