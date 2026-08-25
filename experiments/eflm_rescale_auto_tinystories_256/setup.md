
# Rescale + Autonomous EFLM * {w/, w/o Ada} * {w/, w/o Trunc}, Tinystories Exp: Setting Up

---

## Rescale + Autonomous EFLM * {w/, w/o Ada} * {w/, w/o Trunc}, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{\sqrt{d}})$ 
- ``rho``: ``rho_min`` == ``rho_max``
  - {0.5, 1.0, 5.0, 8.0, 16. 28.0}
  - noise norm E‖ε‖≈√d≈22.6; grid brackets it so t*(R) spans ~0.03→0.91
  - Refer `alpha_star_euclidean(V=50257, embed_norm=R)` → R=1: **0.840**, R=8: **0.397**,
R=28: **0.158** (flow-time t* = 1−α*: 0.16 / 0.60 / 0.84) to derive the truncation, remember the autonomous clock conversion, refer to the results in experiments/eflm_rescale_tinystories_256/RESULTS.md
---

## Rescale + Autonomous EFLM * {w/, w/o Ada} * {w/, w/o Trunc}, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss * {w/ SNR, w/o SNR}

---

## Rescale + Autonomous EFLM * {w/, w/o Ada} * {w/, w/o Trunc}, Tinystories Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

# Rescale + Autonomous EFLM * {w/ Trunc}, Tinystories Exp: Setting Up

---

## Rescale + Autonomous EFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{\sqrt{d}})$ 
- ``rho``: ``rho_min`` == ``rho_max``
  - {0.05, 0.1, 0.5, 1.0, 5.0, 8.0, 16. 28.0}
  - noise norm E‖ε‖≈√d≈22.6; grid brackets it so t*(R) spans ~0.03→0.91
  - Refer `alpha_star_euclidean(V=50257, embed_norm=R)` → R=1: **0.840**, R=8: **0.397**,
R=28: **0.158** (flow-time t* = 1−α*: 0.16 / 0.60 / 0.84) to derive the truncation, remember the autonomous clock conversion, refer to the results in experiments/eflm_rescale_tinystories_256/RESULTS.md
    - Vary truncation a little bit to find the optimal value
---

## Rescale + Autonomous EFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e-4, 1e-3, 5e-3}, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss * {w/ SNR, w/o SNR}

---

## Rescale + Autonomous EFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---