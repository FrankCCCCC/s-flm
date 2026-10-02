## Rescale + EFLM, Sudoku Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (= custom std 0.0441)
- ``rho``: ``rho_min`` == ``rho_max``
  - {5.0, 8, 16}
  - noise norm E‖ε‖≈√d≈22.6 (d=512); t*(R) = 1−α*: 0.60 / 0.71 / 0.83 for R = 5 / 8 / 16 (α* 0.396 / 0.291 / 0.170, τ* 0.505 / 0.344 / 0.187, V=12)
- ``forward_type``: {``gaussian``}
- ``noise``: {``autonomous``, ``autonomous-adaptive``, ``log-linear-adaptive+trunc``}
  - ``tau_max``: only for ``autonomous`` and ``autonomous-adaptive``, use `alpha_star_euclidean` to decide 
  - ``alpha_max``: only for ``log-linear-adaptive+trunc``, use `alpha_star_euclidean` to decide 

---

## Rescale + EFLM, Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss * {w/o SNR}
  - ``noise_sched_cov_weight``: 
    - for ``autonomous`` noise: {True, False}
    - for ``autonomous-adaptive`` and log-linear-adaptive+trunc noise: {False} (snr_weighted_ce, noise_sched_cov_weight and an adaptive schedule are mutually exclusive)
  - 3 radom seeds: {1, 2, 3}, take averge

---

## Rescale + EFLM, Sudoku Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step