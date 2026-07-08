

---
marp: true
theme: default
paginate: true
# _class: invert
# color: white
size: 4:3
class: lead
style: |
  section.lead h1 {
    text-align: center;
  },
  section.lead h2 {
    text-align: center;
  },
  section.lead h3 {
    text-align: center;
  },
  h1 {
    color: #3d3d3d;
  },
  h2 {
    color: #3d3d3d;
  },
  h3 {
    color: #3d3d3d;
  },
  r {
      color: red;
  },
  y {
      color: yellow;
  },
  b {
      color: blue;
  },
  .g {
      color: green;
  }
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
ng { color: #0072B2; }
rd { color: #D55E00; }
uv { color: #008060; }
hy { color: #7B3FA0; }
</style>

# Hyperbolic DLM

#### July 9, 2026



---

# H-FLM Curvature + Init on Sudoku (3 Seeds Avg)

---

## Curvature * Init * LR Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed {1, 2, 3}) · clues: easy 40 / med 35 / hard 30
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

## Baseline: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed {1, 2, 3}) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)
- Algo: {S-FLM naive, S-FLM + trunc, S-FLM + trunc + adaptive, E-FLM Naive, LangFlow + ada sched, LangFlow + ada sched + SC}

---

## Baseline: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

---

## Baseline: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

<!-- ---

## Environment

- Exp name: ``hflm_curv_init_lr_sudoku``
- Use both ``desa`` and ``thickstun`` partition on unicorn and any available GPUs on Tinkercliffs. Read Agent.md for more server details. For Tinkercliffs, use ``/home/shengyenc/workspace/research/s-flm`` as working directory.
- Refer to ``experiments/hflm_curv_sudoku/sweep.py`` for the ``sweep.py`` format
- Prioritize Init {``random``} * LR {3e‑4, 1e-3} * Curvature {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5} -->

---

### Recall the Former Results

| Model | easy | medium | hard |
|---|---|---|---|
| **S-FLM (naive)** | 78.4 | 39.0 | 13.9 |
| S-FLM + trunc | 95.05 | 77.9 | 48.3 |
| S-FLM + trunc + adaptive | 94.15 | 78.2 | 45.95 |
| **E-FLM (naive)** | 90.35 | 62.45 | 17.55 |
| LangFlow + SC | 7.05 | 0.20 | 0.00 |
| LangFlow + ada sched | 81.60 | 59.85 | 23.45 |
| LangFlow + ada sched + SC | 94.90 | 83.30 | 57.00 |
| **H-FLM (naive, our)** | 89.5 | 75.50 | 31.7 |
| **H-FLM (tuned, our)** | 96.5 | 84.50 | 37.1 |

---

## Brief Summary - Sudoku Medium
| K     | LR 3e-4 | LR 1e-3 |
|-------|---------|---------|
| -0.25 | 73.8%   | 70.5%   |
| -0.3  | 77.4%   | 74.2%   |
| -0.5  | 79.3%   | **84.3%** |
| -0.7  | 80.7%   | 82.9%   |
| -1.0  | 66.1%   | 64.4%   |
| -1.5  | 69.6%   | 70.3%   |

---

## Brief Summary - Sudoku Hard

| K     | LR 3e-4 | LR 1e-3 |
|-------|---------|---------|
| -0.25 | 23.9%   | 16.7%   |
| -0.3  | 38.4%   | 28.1%   |
| -0.5  | **43.5%** | 23.7%   |
| -0.7  | 39.7%   | **28.2%** |
| -1.0  | 27.3%   | 20.3%   |
| -1.5  | 34.5%   | 20.8%   |

- K = -0.5@LR = 3e-4 is close with S-FLM + trunc + adaptive

---

# Loss Geometry of Euclidean and Hyperbolic FLM

---

### Loss Geometry of E-FLM on Tinystories, Seq Len: 256

![width:700px](image-4.png)

---

### Loss Geometry of H-FLM on Tinystories, Seq Len: 256

- Curvature = -1.0

![width:700px](image-5.png)

---