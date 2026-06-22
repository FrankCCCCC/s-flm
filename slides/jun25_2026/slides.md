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

#### June 25, 2026

---

## Naive AR Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**
- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256, 1024}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

- Evaluation
  - Greedy decoding

---

## Naive Geometry Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM
  - E-FLM
  - H-FLM

---

## Naive Geometry Baseline - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256, 1024}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM (ada sched, truncation, ada sched + truncation)
  - LangFlow (ada sched, ada sched + Self Cond)

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256, 1024}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

## H-FLM Sweep - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**
  - Word Embdedding Init: 
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
    - ``custom``: std: {0.001, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0}
- Init noise for diffusion process: {0.001, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0}
- ``rho_max``: {12}

---

## H-FLM Sweep - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256, 1024}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

---

![alt text](image-2.png)

---

## H-FLM pre-trained on TinyStories

### Log-scled Y axis

![alt text](image.png)

---

## H-FLM pre-trained on TinyStories

### Linear-scled Y axis

![alt text](image-1.png)