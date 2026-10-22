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
  - Seed: {1, 2, 3}
  - Save Checkpoint every 1K steps; keep checkpoint every 5K training step persistently

---

## HBFM, Tinystories Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step


## YOUR TASK

Read Agent.md and refer to papers/Variational Diffusion Models.pdf and papers/Diffusion Models With Learned Adaptive Noise.pdf and slides/, Implement adaptive curvature for each position, consider HBFM, the hyperbolic dit should output not only logits but also curvatures for each position, for example, for a tinystories 256 word leength HBFM, the model should output 256 curvature predicttion corresponding to each word. Then, modify the CE loss function to multiply the CE loss at each position by a weight to compensate the change of the density based on curvature, make sure the CE loss sum is invariant to the change of curature. Additionally, modfify the HBFMSampler to diffuse on the predicted curvature. Derive the math first in math.md. Follow the notation in slides/ and derive the loss function and make sure adaptive curvature doesn't affect the mean of loss, but the variance of loss. Also, derive the corresponding sampler. After math derivation, implement and verify it. Finally, conduct tinystories 256 experiment.

Implement with minimal line of code change, make sure the codebase is simple, intuitive, easy-understand.

Enable adatpive curvature for every position. Vary time_exp_rate, time_conversion_mode, and time_range_upper_bound to find best GenPPL and entropy (low GenPPL and high entropy), refer to the results in experiments/eflm_rescale_auto_trunc_tinystories_256 and experiments/claude_test_hbfm_tinystories_256 and experiments/hbfm_rlf_ada_curv. Don't change other parameters except for time_exp_rate, time_conversion_mode, and time_range_upper_bound, for LR, you can only choose from {3e‑4, 5e‑4, 1e‑3}

T_MAX must >= time_range_upper_bound to pass the validation of HBFM class

Use both desa and thickstun partitions, use 4 a6000 or a5000, if you want to get results faster, use ch2263 `nlplarge` or `nlplarge-claire-highpri` or occupy available gpus in desa an thickstun partitions.

Note that in the unigram experiment, extreme curvature might cause high hyperbolic ELBO variance. Setup a base curvature value and guard the boundary of the curvature is important to make the adaptive curvature work. Refer to /share/desa/nfs02/sc3379/workspace/research/unigram2 or login sc3379@unicorn-login-02: /home/sc3379/workspace/research/unigram-dev/unigram to check the latest report.