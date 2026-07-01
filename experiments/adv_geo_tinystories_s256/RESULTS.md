# adv_geo_tinystories_s256 — Results

25 cell(s) with eval metrics. Valid PPL (held-out likelihood bound; note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.

| Cell | Valid PPL | GenPPL | Entropy | Note |
|---|---|---|---|---|
| sfm_ada_lr1e-3 | 1.4585 | 20.2123 | 3.9161 | |
| sfm_ada_lr1e-4 | 1.7471 | 29.9498 | 3.8588 | |
| sfm_ada_lr3e-4 | 2.0322 | 21.8692 | 3.8726 | |
| sfm_ada_lr5e-5 | 2.8715 | 35.6356 | 3.8018 | |
| sfm_ada_lr5e-3 | 4.0095 | 347.0291 | 4.1825 | |
| sfm_trunc_lr1e-3 | 5.4418 | 15.2825 | 3.9587 | |
| sfm_trunc_lr3e-4 | 5.5155 | 16.4428 | 3.9096 | |
| sfm_trunc_lr5e-3 | 5.6589 | 12.9010 | 3.9909 | |
| sfm_trunc_lr1e-4 | 5.7301 | 18.5992 | 3.9206 | |
| sfm_trunc_lr5e-5 | 6.0315 | 22.8395 | 3.8745 | |
| lf_ada_lr1e-4 | 10.5632 | 31.2326 | 3.5253 | |
| sfm_ada_trunc_lr5e-3 | 10.9473 | 11.0234 | 3.9018 | |
| lf_ada_sc_lr5e-5 | 10.9522 | 43.0180 | 3.3748 | |
| sfm_ada_trunc_lr5e-5 | 11.3980 | 20.4754 | 3.8950 | |
| lf_ada_sc_lr1e-4 | 11.8170 | 43.9808 | 3.3556 | |
| sfm_ada_trunc_lr3e-4 | 11.9875 | 14.3908 | 3.9424 | |
| lf_ada_lr5e-5 | 12.0566 | 42.5957 | 3.4353 | |
| sfm_ada_trunc_lr1e-4 | 12.1671 | 16.2626 | 3.9030 | |
| lf_ada_lr3e-4 | 12.5644 | 34.9537 | 3.4421 | |
| lf_ada_lr1e-3 | 35.0311 | 20.7260 | 3.6798 | |
| lf_ada_lr5e-3 | nan | 1.0928 | 0.0000 | ⚠collapse? |
| sfm_ada_trunc_lr1e-3 | 12.1765 | 12.2881 | 3.9414 | |
| lf_ada_sc_lr3e-4 | 13.1161 | 17.5680 | 3.3316 | |
| lf_ada_sc_lr1e-3 | 37.9687 | 18.3619 | 3.3969 | |
| lf_ada_sc_lr5e-3 | 470.5442 | 765.8282 | 3.5496 | |

## Best
- Lowest Valid PPL: **sfm_ada_lr1e-3** = 1.4585
- Lowest GenPPL (non-collapsed, entropy≥3.0): **sfm_ada_trunc_lr5e-3** = 11.0234
