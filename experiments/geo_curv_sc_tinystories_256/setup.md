# Tinystories + {Geometry * Curvature * Self Condition} Experiment

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM {Naive, ada sched, truncation, ada sched + truncation} * {Self Cond: On}
  - E-FLM {Naive, ada sched, truncation, ada sched + truncation} * {Self Cond: On, Off}

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
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
    - ``random``: $\mathcal{N}(0, 4e-4)$ (= custom 0.02)
    - ``custom``: std: {0.01, 0.04}
- Init noise for diffusion process: {0.5, 0.8, 1.0}
- ``rho_max``: {12}
- Self Cond: On
- Gaussian Curvature: {-0.01, -0.1, -0.25, -0.5, -0.75}

---

## Definition of ``init`` / ``prior_cov`` of H-FLM

Given a target word embedding $z_{T} \in \mathbb{R}^{d}$, a Gaussian noise $\epsilon \in \mathbb{R}^d$ and a timestep $t \in [0, 1], t \in \mathbb{R}$

- $\epsilon \sim \mathcal{N}(0, \text{prior\_cov})$
- $z_{t} = \text{geodesic}(z_{T}, \epsilon, t)$

The target word embedding is initialized by $\mathcal{N}(0, \text{init}^2)$

---

## H-FLM Sweep - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step