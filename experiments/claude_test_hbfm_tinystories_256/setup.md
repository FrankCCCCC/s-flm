# HBFM, Tinystories Exp: Setting Up

---

## HBFM, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,
- forward_type: {naive, horosphere}
---

## HBFM, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e-4}, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss * {w/o SNR}
  - time_conversion_mode: {exp}
  - time_range_upper_bound: {1.0}
  - time_exp_rate: {3.0}
  - noise scheduler: {log-linear}
  - prod_factor_dim: 16 * 3dim, prod_factor_gaussian_curvature: [x] * 16
    - curvature ``x``: {-0.5}
  - Seed: {1, 2, 3}
  - Save Checkpoint every 1K steps; keep checkpoint every 5K training step persistently

---

## HBFM, Tinystories Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

## YOUR TASK

Vary forward_type, time_exp_rate, global curvature (prod_factor_gaussian_curvature), time_conversion_mode, and time_range_upper_bound to find best GenPPL and Entropy, refer to the results in experiments/naive_ar_tinystories_s256 and experiments/claude_test_hbfm_sudoku. Don't change other parameters except for forward_type, time_exp_rate, curvaturem, time_conversion_mode, and time_range_upper_bound

Use both desa and thickstun partitions, avoid 2080ti