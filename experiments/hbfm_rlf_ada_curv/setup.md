# Goal

Read Agent.md and refer to papers/Variational Diffusion Models.pdf and papers/Diffusion Models With Learned Adaptive Noise.pdf and slides/, Implement adaptive curvature for each position, consider HBFM, the hyperbolic dit should output not only logits but also curvatures for each position, for example, for a tinystories 256 word leength HBFM, the model should output 256 curvature predicttion corresponding to each word. Then, modify the CE loss function to multiply the CE loss at each position by a weight to compensate the change of the density based on curvature, make sure the CE loss sum is invariant to the change of curature. Additionally, modfify the HBFMSampler to diffuse on the predicted curvature. Derive the math first in math.md. Follow the notation in slides/ and derive the loss function and make sure adaptive curvature doesn't affect the mean of loss, but the variance of loss. Also, derive the corresponding sampler. After math derivation, implement and verify it. Finally, conduct sudoku experiment.

Implement with minimal line of code change, make sure the codebase is simple, intuitive, easy-understand.

Note that in the unigram experiment, extreme curvature might cause high hyperbolic ELBO variance. Setup a base curvature value and guard the boundary of the curvature is important to make the adaptive curvature work. Refer to /share/desa/nfs02/sc3379/workspace/research/unigram2 or login sc3379@unicorn-login-02: /home/sc3379/workspace/research/unigram-dev/unigram to check the latest report.

Use both desa and thickstun partitions, only use 2080ti for large scale exp, if you want to get results faster, use ch2263 `nlplarge` or `nlplarge-claire-highpri` or occupy available gpus in desa an thickstun partitions.

## HBFM, Sudoku Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)
- forward_type: {naive, horosphere}
---

## HBFM, Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - time_exp_rate: 0.01
  - time_conversion_mode: {exp}
  - time_range_upper_bound: {1.0}
  - noise scheduler: {log-linear}
  - prod_factor_dim: [3, 3, 3], prod_factor_gaussian_curvature: [-0.5, -0.5, -0.5]
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge
  - hyla (RLF): Off

---

## HBFM, Sudoku Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

## YOUR TASK

Vary forward_type, time_exp_rate, time_conversion_mode, and time_range_upper_bound to find best acc, refer to the results in experiments/eflm_rescale_auto_sudoku and experiments/claude_test_hbfm_sudoku. Don't change other parameters except for time_exp_rate, curvaturem, time_conversion_mode, and time_range_upper_bound, for LR, you can only choose from {3e‑4, 5e‑4, 1e‑3}

T_MAX must >= time_range_upper_bound to pass the validation of HBFM class