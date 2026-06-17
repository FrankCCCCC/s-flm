#!/usr/bin/env python
"""Load pre-trained TinyStories HFLM word embeddings and plot the histogram of their
lengths. For HFLM the embedding `backbone.sphere_embed.weight` is in hyperbolic space and
its "length" is the radial coordinate rho = row-wise L2 norm (hyperbolic_dit.py:183)."""
import os, sys, math
import torch
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

CKPT = "/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/outputs/tinystories/hflm/checkpoints/last.ckpt"
OUT  = "/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/hflm_embed_norms.png"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top-level keys:", list(ckpt.keys()))

# find every tensor whose key ends in sphere_embed.weight (model + any EMA shadow copy)
def find_embed(d, prefix=""):
    hits = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            if torch.is_tensor(v) and kk.endswith("sphere_embed.weight"):
                hits[kk] = v
            elif isinstance(v, (dict, list)):
                hits.update(find_embed(v, kk))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            hits.update(find_embed(v, f"{prefix}[{i}]"))
    return hits

# EMA in Lightning is often a list of flat tensors in callbacks; also scan state_dict.
sd = ckpt.get("state_dict", ckpt)
embeds = find_embed(sd, "state_dict")
embeds.update(find_embed({k: v for k, v in ckpt.items() if k != "state_dict"}))
print("embedding tensors found:")
for k, v in embeds.items():
    print(f"  {k}: shape={tuple(v.shape)}")

# Prefer the raw model embedding (state_dict). EMA may be stored as a flat param list.
key = next((k for k in embeds if k.endswith("backbone.sphere_embed.weight")), None) or next(iter(embeds))
W = embeds[key].float()
print(f"\nUSING: {key}  shape={tuple(W.shape)}")

# EMA shadow: Lightning EMA callback stores a flat list under callbacks; try to recover it
ema_norms = None
cbs = ckpt.get("callbacks", {})
for cbname, cbstate in (cbs.items() if isinstance(cbs, dict) else []):
    if isinstance(cbstate, dict):
        for kk, vv in cbstate.items():
            if isinstance(vv, list) and vv and all(torch.is_tensor(t) for t in vv):
                # match the embedding by shape
                for t in vv:
                    if tuple(t.shape) == tuple(W.shape):
                        ema_norms = t.float().norm(p=2, dim=-1)
                        print(f"found EMA shadow embedding in callbacks[{cbname}][{kk}]")
                        break

rho = W.norm(p=2, dim=-1)  # [vocab] hyperbolic radial length
import numpy as np
r = rho.numpy()
print(f"\nrho (embedding length) over vocab={len(r)}:")
print(f"  min={r.min():.3f}  p1={np.percentile(r,1):.3f}  median={np.median(r):.3f}  "
      f"mean={r.mean():.3f}  p99={np.percentile(r,99):.3f}  max={r.max():.3f}  std={r.std():.3f}")
if ema_norms is not None:
    e = ema_norms.numpy()
    print(f"  [EMA] median={np.median(e):.3f} mean={e.mean():.3f} min={e.min():.3f} max={e.max():.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(r, bins=80, color="#4477aa", edgecolor="white", alpha=0.9, label=f"raw weights (n={len(r)})")
if ema_norms is not None:
    ax.hist(ema_norms.numpy(), bins=80, color="#cc6677", edgecolor="white", alpha=0.55, label="EMA weights")
ax.axvline(np.median(r), color="k", ls="--", lw=1, label=f"median={np.median(r):.2f}")
ax.axvline(12, color="green", ls=":", lw=1.5, label="rho_max=12 (clamp)")
ax.set_xlabel("word-embedding length  ‖e_v‖ = rho (hyperbolic radial coord)")
ax.set_ylabel("# tokens")
ax.set_title("TinyStories HFLM (30k) — distribution of word-embedding lengths")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"\nwrote {OUT}")
