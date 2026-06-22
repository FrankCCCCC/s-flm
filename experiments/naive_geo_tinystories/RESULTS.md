# naive_geo_tinystories — Results

3 cell(s) with eval metrics. Valid PPL (held-out likelihood bound; note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.

| Cell | Valid PPL | GenPPL | Entropy | Note |
|---|---|---|---|---|
| eflm | 1.0858 | 29.1471 | 4.1251 | |
| sfm | 1.2423 | 51.0837 | 4.3205 | |
| hflm | 14212.1648 | 6.7183 | 2.2997 | ⚠collapse? |

## Best
- Lowest Valid PPL: **eflm** = 1.0858
- Lowest GenPPL (non-collapsed, entropy≥3.0): **eflm** = 29.1471
