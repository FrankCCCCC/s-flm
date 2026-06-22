# adv_geo_tinystories — Results

8 cell(s) with eval metrics. Valid PPL (held-out likelihood bound; note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.

| Cell | Valid PPL | GenPPL | Entropy | Note |
|---|---|---|---|---|
| sfm_ada_trunc_lr3e-4 | 1.1386 | 28.6765 | 4.1424 | |
| lf_ada_lr3e-4 | 8.5476 | 26.4779 | 3.8724 | |
| lf_ada_lr1e-4 | 9.7384 | 52.8563 | 3.7258 | |
| lf_ada_lr1e-3 | 33.0262 | 24.4206 | 3.6247 | |
| lf_ada_lr5e-3 | 33.5263 | 14.3692 | 4.3605 | |
| lf_ada_lr5e-5 | 49.3976 | 62.0961 | 3.2006 | |
| sfm_ada_trunc_lr1e-3 | 317.2150 | 24.4328 | 4.1836 | |
| sfm_ada_trunc_lr5e-3 | 317.3729 | 19.1786 | 4.2153 | |

## Best
- Lowest Valid PPL: **sfm_ada_trunc_lr3e-4** = 1.1386
- Lowest GenPPL (non-collapsed, entropy≥3.0): **lf_ada_lr5e-3** = 14.3692
