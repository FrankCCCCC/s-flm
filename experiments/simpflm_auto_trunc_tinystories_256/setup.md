# Autonomous SimpFLM * {w/ Trunc}, Tinystories Exp: Setting Up

---

# Autonomous SimpFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,
  - ``rho``: ``rho_min`` == ``rho_max``
    - {0.05, 0.1, 0.5, 1.0, 5.0, 8.0, 16. 28.0}

- Refer `alpha_star_euclidean(V=50257, embed_norm=R)` → R=1: **0.840** (flow-time t* = 1−α*: 0.16) to derive the truncation, remember the autonomous clock conversion
    - Vary truncation a little bit to find the optimal value
---

# Autonomous SimpFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e-4, 1e-3, 5e-3}, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss
  - Vary seed: {1, 2, 3}

---

# Autonomous SimpFLM * {w/ Trunc}, Tinystories Exp: Setting Up

- Naive Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

# Frontier Line Evaluation for Variying Temperature T

- GenPPL & Entropy Frontier Evaluation
  - Refer to ``/home/sc3379/workspace/research/s-flm-dev/s-flm/papers/Language Modeling with Hyperspherical Flows.pdf``, Figure 10, draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE 
  - Cells:
    - For each method: {MDLM, DUO, FLM, Auto + Trunc + SimpFLM}, evaluate on the seed-varying = {1, 2, 3} pretrained models
      - NFE: {1, 4, 8, 16, 32, 64, 128, 256}
      - T: {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20}
  - Draw 512 samples for each cell
  - Draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE, output in ``experiments/naive_ar_tinystories_s256``
  - Report the mean and std across 3 seeds as solid line and the shadow of the frontier
  - Exact-velocity, top_k_v = -1 (all), sampling steps
  - Greedy decoding for last sampling step

---

# SimpFLM SDE Sampler + GenPPL - Entropy Frontier Line Evaluation

- Pick and load best checkpoints from ``experiments/simpflm_auto_trunc_tinystories_256`` to evaluate the frontier line with various ``eta`` and ``GScheduler``
- Search for best ``GScheduler`` and ``eta`` that can maintain the same GenPPL with highest entropy, make SimpFLM become competive to DUO, FLM, MDLM, and S-FLM. Design the best ``GScheduler``
- Make sure the settings are aligned with the existing baselines in ``experiments/simpflm_auto_trunc_tinystories_256``

---

# SimpFLM + Variying Temperature T + GenPPL - Entropy Frontier Line Evaluation

- GenPPL & Entropy Frontier Evaluation
  - Refer to ``/home/sc3379/workspace/research/s-flm-dev/s-flm/papers/Language Modeling with Hyperspherical Flows.pdf``, Figure 10, draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE 
  - Cells:
    - For each method: {MDLM, DUO, FLM, Auto + Trunc + SimpFLM}, evaluate on the seed-varying = {1, 2, 3} pretrained models
      - NFE: {1, 4, 8, 16, 32, 64, 128, 256}
      - T: {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20}
  - Draw 512 samples for each cell
  - Draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE, output in ``experiments/simpflm_auto_trunc_tinystories_256``
  - Report the mean and std across 3 seeds as solid line and the shadow of the frontier
  - Exact-velocity, top_k_v = -1 (all)
  - Greedy decoding for last sampling step
  - $\eta = 0$, ``GT=Linear``