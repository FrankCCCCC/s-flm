# naive_geo_tinystories_s256 — Results

4 cell(s) with eval metrics. Valid PPL (held-out likelihood bound; note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.

| Cell | Valid PPL | GenPPL | Entropy | Note |
|---|---|---|---|---|
| eflm | 1.1014 | 34.5822 | 3.6701 | |
| sfm | 1.2605 | 35.9224 | 3.8569 | |
| hflm_hyperbolic | 6.5251 | 39.2740 | 4.3432 | |
| hflm | 683.4089 | 57.7535 | 2.4503 | ⚠collapse? |

## Best
- Lowest Valid PPL: **eflm** = 1.1014
- Lowest GenPPL (non-collapsed, entropy≥3.0): **eflm** = 34.5822
