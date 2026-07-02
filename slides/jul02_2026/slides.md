# H-FLM Curvature + Init on Sudoku

---

## Curvature * Init * LR Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$, ``random``: $\mathcal{N}(0, 4e-4)$
  - ``custom``: std: {0.01, 0.02, 0.04, 0.06, 0.08}

- Geometry Curvature: {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}

---

## Curvature * Init * LR Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {1e‑4, 3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

---

## Curvature * Init * LR Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

## Environment

- Exp name: ``hflm_curv_init_lr_sudoku``
- Use both ``desa`` and ``thickstun`` partition on unicorn and any available GPUs on Tinkercliffs. Read Agent.md for more server details. For Tinkercliffs, use ``/home/shengyenc/workspace/research/s-flm`` as working directory.
- Refer to ``experiments/hflm_curv_sudoku/sweep.py`` for the ``sweep.py`` format
- Prioritize Init {``random``} * LR {3e‑4, 1e-3} * Curvature {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}