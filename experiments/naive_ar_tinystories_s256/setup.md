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

- Evaluation
  - DUO and FLM
    - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
    - Greedy decoding for last sampling step (if available)
  - MLDM
    - 180 sampling steps
