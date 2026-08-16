## Naive Baselines, Tinystories Exp: Setting Up

- Data: TinyStories, **475M train / 5M val**
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**,

- Methods
  - AR
  - MDLM
  - DUO
  - FLM
- Seed: {1, 2, 3}
---

## Naive Baselines, Tinystories Exp: Setting Up

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e-4, 1e-3, 5e-3}, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

---

## Naive Baselines, Tinystories Exp: Setting Up

- Default Evaluation
  - DUO and FLM
    - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
    - Greedy decoding for last sampling step (if available)
  - MLDM
    - 180 sampling steps

---

## Naive Baselines, Tinystories Exp: Setting Up

- GenPPL & Entropy Frontier Evaluation
  - Refer to ``/home/sc3379/workspace/research/s-flm-dev/s-flm/papers/Language Modeling with Hyperspherical Flows.pdf``, Figure 10, draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE 
  - Cells:
    - For each method: {MDLM, DUO, FLM}, evaluate on the seed-varying = {1, 2, 3} pretrained models
      - NFE: {1, 4, 8, 16, 32, 64, 128, 256}
      - T: {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20}
  - Draw 512 samples for each cell
  - Draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE, output in ``experiments/naive_ar_tinystories_s256``
  - Report the mean and std across 3 seeds as solid line and the shadow of the frontier
