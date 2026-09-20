## HBFM, Sudoku Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

---

## HBFM, Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - time_conversion_mode: {exp}
  - time_range_upper_bound: {1.0}
  - time_exp_rate: {0.5, 1.0, 3.0, 5.0}
  - noise scheduler: {log-linear}
  - prod_factor_dim: [3, 3, 3], prod_factor_gaussian_curvature: [x, x, x]
    - curvature ``x``: {-0.1, -0.5, -1.0, -3.0, -5.0}
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge

---

## HBFM, Sudoku Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step