# Goal

Read Agent.md and implement ELBO loss for both EFLM and HBFM. After math derivation, implement and verify it. Finally, conduct sudoku experiment. Pick the best config for each to train the model from experiments/hbfm_rlf_ada_curv. Present the ELBO and acc, incomparison to MDLM, the acc should be at least tie with CE loss version. Make sure it's a fair comparison

Implement with minimal line of code change, make sure the codebase is simple, intuitive, easy-understand. Refer to loss.py to make sure the numerical stability

Note that in the unigram experiment, extreme curvature might cause high hyperbolic ELBO variance. Setup a base curvature value and guard the boundary of the curvature is important to make the adaptive curvature work. Refer to /share/desa/nfs02/sc3379/workspace/research/unigram2 or login sc3379@unicorn-login-02: /home/sc3379/workspace/research/unigram-dev/unigram to check the latest report.

Use both desa and thickstun partitions, only use 2080ti for large scale exp, if you want to get results faster, use ch2263 `nlplarge` or `nlplarge-claire-highpri` or occupy available gpus in desa an thickstun partitions.

T_MAX must >= time_range_upper_bound to pass the validation of HBFM class

## HBFM, Sudoku Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)
- forward_type: {horosphere}
---

## HBFM, Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - noise scheduler: {log-linear}
  - prod_factor_dim: [3, 3, 3], prod_factor_gaussian_curvature: [-0.5, -0.5, -0.5]
  - loss_type: elbo
  - 3 radom seeds: {1, 2, 3}, take averge
  - hyla (RLF): Off
  - Ada curvature: on

---

## HBFM, Sudoku Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step