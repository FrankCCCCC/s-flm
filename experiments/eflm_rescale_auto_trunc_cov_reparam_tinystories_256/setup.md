# Rescale + EFLM, Tinystories Exp: Setting Up

---

## Rescale + Autonomous EFLM, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ 
- ``rho``: ``rho_min`` == ``rho_max``
  - {0.1, 0.5, 1.0, 8.0}
  - noise norm E‖ε‖≈√d≈27.7 (d=768); t*(R) = 1−α*: 0.019 / 0.087 / 0.160 / 0.603 for R = 0.1 / 0.5 / 1 / 8 (α* 0.981 / 0.913 / 0.840 / 0.397, τ* 3.981 / 2.444 / 1.834 / 0.505)
  - Refer `alpha_star_euclidean(V=50257, embed_norm=R)` → R=1: **0.840**, R=8: **0.397**,
R=28: **0.158** (flow-time t* = 1−α*: 0.16 / 0.60 / 0.84) to derive the truncation, remember the autonomous clock conversion, refer to the results in experiments/eflm_rescale_tinystories_256/RESULTS.md
  - ``forward_type``: {``gaussian``}
  - ``noise``: {``autonomous``, ``autonomous-adaptive``, ``log-linear-adaptive+trunc``}
    - ``tau_max``: only for ``autonomous`` and ``autonomous-adaptive``, use `alpha_star_euclidean` to decide 
    - ``alpha_max``: only for ``log-linear-adaptive+trunc``, use `alpha_star_euclidean` to decide 

---

## Rescale + Autonomous EFLM, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
    - ``autonomous-adaptive`` and ``log-linear-adaptive+trunc``: run **4 GPUs × 32 per GPU** (gradient accumulation 4, as in the earlier adaptive sweeps). The adaptive refit fires once per micro-batch, so more accumulation (e.g. the script default 1 GPU × 8 = 64) makes the schedule oscillate
  - Optimizer: AdamW
    - LR: {3e-4, 1e-3, 5e-3}, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - ``noise_sched_cov_weight``:
    - for ``autonomous`` noise: {True}
    - for ``autonomous-adaptive`` and log-linear-adaptive+trunc noise: {False} (snr_weighted_ce, noise_sched_cov_weight and an adaptive schedule are mutually exclusive)
  - All use cross entropy loss * {w/o SNR}

---

## Rescale + Autonomous EFLM, Tinystories Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step
