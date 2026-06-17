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

## Environment

- Use ``desa`` or ``thickstun`` partitions, use as many as GPUs as possible to accelerate the experiment
- refer to https://it.coecis.cornell.edu/researchit/using-the-unicorn-cluster/ for detailed guideline
- For each experiment, create a separate folder in ``experiments`` and ``outputs`` with identical ``name``. Put the checkpoints and sampling results under the experimental subfolder in ``outputs`` folder.
- Write sweep scripts for each experiment and put under the experimental subfolder in ``experiments``. For each sceript, use the pre-defined sampling and training script in ``scripts/sample/tinystories`` and ``scripts/train/tinystories``. You should only add / modify only when it's necessary. Follow the scripts architecutre and scope carefully, you can add new options to the training and sampling script to provide flexibility only if altering the arguement is necessary, single script under train / sampling should only conduct single run instead of grip sweep. The sweep script of each experiment should call the training and sampling scripts under ``scripts/train/tinystories`` and ``scripts/sample/tinystories``.
- Evaluate the valid PPL and GenPPL and provide a report under the experimental subfolder in ``experiments``.
- Follow the principles in ``CLAUDE.md`` and ``Agent.md``
- Use ``simple_slurm`` to write submission sweep script in python
- Show me the trianing / sampling scripts in scripts/train/tinystories and scripts/sample/tinystories before job submission

---

## Naive AR Baseline - TinyStories Exp Setup

- Name: ``naive_ar_tinystories``
- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**
- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **1024**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

- Evaluation
  - Greedy decoding

---

## Naive Geometry Baseline - TinyStories Exp Setup

- Name: ``naive_geo_tinystories``
- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM
  - E-FLM
  - H-FLM

---

## Naive Geometry Baseline - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **1024**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Name: ``adv_geo_tinystories``
- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM (ada sched, truncation, ada sched + truncation)
  - LangFlow (ada sched, ada sched + Self Cond)

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Training
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **1024**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {5e-5, 1e-4, 3e-4, 1e-3, 5e-3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

## H-FLM Sweep - TinyStories Exp Setup

- Name: ``hflm_sweep_tinystories``
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
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **1024**, bf16, EMA 0.9999
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