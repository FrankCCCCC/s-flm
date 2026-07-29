## Rescale + Autonomous EFLM * {w/ Ada}, Sudoku Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{\sqrt{d}})$ (= custom 0.0441
- ``rho``: ``rho_min`` == ``rho_max``
  - {5.0, 8, 16}
  - noise norm E‖ε‖≈√d≈22.6; grid brackets it so t*(R) spans ~0.03→0.91
---

## Rescale + Autonomous EFLM * {w/ Ada}, Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge

---

## Rescale + Autonomous EFLM * {w/ Ada}, Sudoku Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step