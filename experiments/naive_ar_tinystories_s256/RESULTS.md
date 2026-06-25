# naive_ar_tinystories_s256 — Results

1 cell(s) with eval metrics. Valid PPL (held-out likelihood bound; note: for geometry models this is a diffusion/flow bound, NOT comparable to AR PPL). GenPPL = gpt2-large retokenized generative perplexity (lower=better) — but read it WITH entropy: low GenPPL + low entropy (<3.0) flags repetitive/degenerate collapse.

| Cell | Valid PPL | GenPPL | Entropy | Note |
|---|---|---|---|---|
| ar | 3.3556 | 27.2700 | 0.0256 | ⚠collapse? |

## Best
- Lowest Valid PPL: **ar** = 3.3556
- Lowest GenPPL: (all evaluated cells look collapsed)
