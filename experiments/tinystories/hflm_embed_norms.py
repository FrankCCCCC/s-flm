#!/usr/bin/env python
"""Plot the word-embedding length (L2-norm) distribution of ONE model — linear-Y and log-Y.

Give it a model source + (optionally) the embedding key, and it draws two figures:
  <out>.png        histogram of ||e_v||  (linear Y)
  <out>_log.png    same, log Y           (exposes the heavy tail)

SOURCE is auto-detected:
  *.ckpt/*.pt/*.pth   Lightning/torch checkpoint -> embedding tensor from the state_dict
                      (pick it with --key, e.g. backbone.sphere_embed.weight; else auto)
  *.npy/*.npz         raw [vocab, dim] matrix (npz: --npz-key, else first array)
  anything else       HuggingFace model id, e.g. 'gpt2' / 'openai-community/gpt2'

For HFLM the embedding ||e_v|| IS the hyperbolic radial coord rho (hyperbolic_dit.py:183);
pass --rho-max 12 to also show the sampler's clamped radius 12*tanh(||e||/12).

  python hflm_embed_norms.py outputs/tinystories/hflm/checkpoints/last.ckpt \
      --key backbone.sphere_embed.weight --rho-max 12 \
      --out experiments/tinystories/hflm_embed_norms --label "HFLM TinyStories (30k)"

mmap=True keeps the big checkpoint off RAM — but still run on a compute node (desa/thickstun).
"""
import os, argparse
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EMBED_KEYS = ["sphere_embed.weight", "wte.weight", "embed_tokens.weight",
              "word_embeddings.weight", "tok_embeddings.weight", "embeddings.weight"]


def _find_2d(d, suffixes, prefix=""):
    hits = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            if torch.is_tensor(v) and v.ndim == 2 and any(kk.endswith(s) for s in suffixes):
                hits[kk] = v
            elif isinstance(v, (dict, list)):
                hits.update(_find_2d(v, suffixes, kk))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            hits.update(_find_2d(v, suffixes, f"{prefix}[{i}]"))
    return hits


def load_embedding(spec, key=None, npz_key=None):
    """Return a [vocab, dim] float tensor for a ckpt path / raw matrix / HF model id."""
    if os.path.exists(spec):
        if spec.endswith((".ckpt", ".pt", ".pth")):
            ck = torch.load(spec, map_location="cpu", weights_only=False, mmap=True)
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            if torch.is_tensor(sd):
                return sd.float()
            hits = _find_2d(sd, [key] if key else EMBED_KEYS)
            if not hits:
                hits = {k: v for k, v in _find_2d(sd, ["weight"]).items() if "embed" in k.lower()}
            if not hits:
                raise SystemExit(f"no embedding tensor found in {spec} (try --key)")
            k = max(hits, key=lambda kk: hits[kk].shape[0])
            print(f"  using {k} {tuple(hits[k].shape)}")
            return hits[k].float()
        if spec.endswith(".npy"):
            return torch.from_numpy(np.load(spec)).float()
        if spec.endswith(".npz"):
            z = np.load(spec); return torch.from_numpy(z[npz_key or z.files[0]]).float()
        raise SystemExit(f"unknown file type: {spec}")
    from transformers import AutoModel
    print(f"  loading HF model {spec} …")
    W = AutoModel.from_pretrained(spec).get_input_embeddings().weight.detach().float()
    print(f"  token embedding {tuple(W.shape)}")
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="ckpt path / .npy/.npz/.pt / HF model id")
    ap.add_argument("--out", required=True, help="output base path -> <out>.png and <out>_log.png")
    ap.add_argument("--key", default=None, help="embedding key suffix in the state_dict")
    ap.add_argument("--npz-key", default=None)
    ap.add_argument("--label", default=None, help="title label (default: the source)")
    ap.add_argument("--rho-max", type=float, default=None, help="also show clamped radius rho_max*tanh(||e||/rho_max)")
    ap.add_argument("--bins", type=int, default=80)
    args = ap.parse_args()
    label = args.label or os.path.basename(args.source.rstrip("/"))

    rho = load_embedding(args.source, key=args.key, npz_key=args.npz_key).norm(p=2, dim=-1)
    r = rho.numpy()
    print(f"||e_v||: vocab={len(r)} min={r.min():.3f} p1={np.percentile(r,1):.3f} "
          f"median={np.median(r):.3f} mean={r.mean():.3f} p99={np.percentile(r,99):.3f} "
          f"max={r.max():.3f} std={r.std():.3f}")
    r_clamp = None
    if args.rho_max is not None:
        r_clamp = (args.rho_max * torch.tanh(rho / args.rho_max)).numpy()
    np.savez(args.out + ".npz", raw=r, **({"clamp": r_clamp} if r_clamp is not None else {}))

    def make_fig(logy):
        ncol = 2 if r_clamp is not None else 1
        fig, axes = plt.subplots(1, ncol, figsize=(13 if ncol == 2 else 8, 5), squeeze=False)
        ax = axes[0][0]
        ax.hist(r, bins=args.bins, color="#4477aa", edgecolor="white")
        ax.axvline(np.median(r), color="k", ls="--", lw=1, label=f"median={np.median(r):.2f}")
        if args.rho_max is not None:
            ax.axvline(args.rho_max, color="green", ls=":", lw=1.5, label=f"rho_max={args.rho_max:g}")
        ax.set_xlabel("word-embedding length  ‖e_v‖ (L2 norm)")
        ax.set_ylabel("# tokens" + (" (log)" if logy else "")); ax.legend(fontsize=9)
        ax.set_title("raw embedding length ‖e_v‖")
        if r_clamp is not None:
            ax2 = axes[0][1]
            ax2.hist(r_clamp, bins=args.bins, color="#aa3377", edgecolor="white")
            ax2.set_xlabel(f"clamped radius  {args.rho_max:g}·tanh(‖e‖/{args.rho_max:g})")
            ax2.set_ylabel("# tokens" + (" (log)" if logy else ""))
            ax2.set_title("effective hyperbolic radius")
            if logy: ax2.set_yscale("log")
        if logy: ax.set_yscale("log")
        fig.suptitle(f"{label} — word-embedding length distribution" + (" (log Y)" if logy else ""))
        fig.tight_layout()
        out = args.out + ("_log.png" if logy else ".png")
        fig.savefig(out, dpi=130); print(f"wrote {out}")

    make_fig(False)
    make_fig(True)


if __name__ == "__main__":
    main()
