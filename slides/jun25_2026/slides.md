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
  - Training Steps: **30K**, Batch Size: **512**, Max Seq Len: **{256}**, bf16, EMA 0.9999
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
  <!-- - H-FLM -->

---

## Naive Geometry Baseline - TinyStories Exp Setup

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

## Adv Geometry Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM (ada sched, truncation, ada sched + truncation)
  - LangFlow (ada sched, ada sched + Self Cond)

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
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
    - ``custom``: std: {0.001, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0}
- Init noise for diffusion process: {0.001, 0.01, 0.02, 0.04, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0}
- ``rho_max``: {12}

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

---

# TinyStories — Numerical Results

## Metric

- GenPPL
  - Gold standard: pretrained gpt2-large (↓ better)</small>

---

## Naive Baselines — Generation Quality

| Model | Valid PPL <small>†</small> | GenPPL ↓ | Entropy |
|:--|--:|--:|--:|
| **AR** (greedy) | 3.36 | **6.8** | 4.25 |
| E-FLM | 1.10 | 34.6 | 3.67 |
| S-FLM | 1.26 | 35.9 | 3.86 |

- Low GenPPL + low entropy = degenerate collapse (⚠).
- Valid PPL is **not comparable** across AR (true PPL) vs. flows (a denoising-CE bound).</small>

---

## Adavanced Baselines (LangFlow, S-FLM)

**Adaptive schedule + truncation makes S-FLM the best flow; truncation stabilizes aggressive LR.**

<style scoped>table { font-size: 0.62em; margin: 0 auto; }</style>

GenPPL ↓ &nbsp;<small>(rows = variant, cols = LR)</small>

| variant | 5e-5 | 1e-4 | 3e-4 | 1e-3 | 5e-3 |
|:--|--:|--:|--:|--:|--:|
| sfm_ada | 35.6 | 29.9 | 21.9 | 20.2 | 347 |
| sfm_trunc | 22.8 | 18.6 | 16.4 | 15.3 | 12.9 |
| sfm_ada_trunc | 20.5 | 16.3 | 14.4 | 12.3 | <rd>**11.0**</rd> |
| lf_ada | 42.6 | 31.2 | 35.0 | 20.7 | <ng>1.1\*</ng> |
| lf_ada_sc | 43.0 | 44.0 | 17.6 | 18.4 | 766 |

- <rd>**sfm_ada_trunc**</rd> wins (GenPPL 11.0); self-cond helps LangFlow (20.7 → 17.6)
- Without truncation, LR 5e-3 breaks: <ng>collapse\*</ng> / divergence (347, 766)

---

## Adavanced Baselines - Valid PPL

<style scoped>table { font-size: 0.62em; margin: 0 auto; }</style>

| variant | 5e-5 | 1e-4 | 3e-4 | 1e-3 | 5e-3 |
|:--|--:|--:|--:|--:|--:|
| sfm_ada | 2.9 | 1.7 | 2.0 | <b>1.5</b> | 4.0 |
| sfm_trunc | 6.0 | 5.7 | 5.5 | 5.4 | 5.7 |
| sfm_ada_trunc | 11.4 | 12.2 | 12.0 | 12.2 | 10.9 |
| lf_ada | 12.1 | 10.6 | 12.6 | 35.0 | nan |
| lf_ada_sc | 11.0 | 11.8 | 13.1 | 38.0 | 470 |

- Likelihood ≠ generation: `sfm_ada` best Valid PPL (1.5) yet weak GenPPL; `sfm_ada_trunc` worst PPL but best GenPPL

---

## H-FLM Sweep Summary, ``init`` / ``prior_cov``

<style scoped>table { font-size: 0.8em; margin: 0 auto; }</style>
| Selected for | init / prior_cov | Valid PPL | GenPPL ↓ | Entropy |
|:--|:--|--:|--:|--:|
| Best generation | std0.04 / 1.0 | 9.9 | **17.7** | 4.03 |
| Best balanced | std0.02 / 0.04 | 2.0 | 19.5 | 4.29 |
| Best Valid PPL | std0.001 / 0.001 | **1.03** | 73 | 3.82 |

- 132 / 132 cells complete (full sweep). 
- Tiny `prior_cov` → tight bound but **poor** text; the **mid `prior_cov` (0.04–0.5)** band is the generation sweet spot.

---

## H-FLM Sweep - Valid PPL, (init × prior_cov)

**tightest Valid PPL at small `prior_cov` (left), degrading rightward.**

<style scoped>table { font-size: 0.56em; margin: 0 auto; }</style>

| init \\ pc | 0.001 | 0.01 | 0.02 | 0.04 | 0.1 | 0.3 | 0.5 | 0.8 | 1.0 | 1.5 | 2.0 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `ngpt` | 1.04 | 1.34 | 2.17 | 4.33 | 91.5 | 6.67 | 8.67 | 7.98 | 8.48 | 8.81 | 8.77 |
| 0.001 | <rd>**1.03**</rd> | 1.34 | 2.25 | 8.53 | 3.15 | 13.3 | 7.58 | 4921 | 7.49 | 6.60 | 14.2 |
| 0.01 | 1.04 | 1.33 | 1.58 | 2.09 | 3.55 | 7.42 | 10.7 | 8.48 | 460 | 8.86 | 8.58 |
| 0.02 | 1.04 | 1.70 | 1.97 | 2.03 | 3.71 | 6.62 | 7.94 | 11.7 | 7.68 | 7.53 | 16.0 |
| 0.04 | 1.04 | 1.35 | 2.42 | 13.1 | 9.67 | 6.76 | 13.1 | 8.01 | 9.86 | 9.07 | 8.55 |
| 0.1 | 1.04 | 1.39 | 1.64 | 3.75 | 4.58 | 11.2 | 7.38 | 12.8 | 9.97 | 9.25 | 12.6 |
| 0.3 | 1.04 | 1.45 | 1.64 | 2.51 | 4.04 | 10.3 | 8.07 | 7.36 | 7.93 | 8.63 | 7.74 |
| 0.5 | 1.04 | 1.55 | 2.12 | 2.33 | 9.56 | 11.8 | 11.3 | 13.9 | 15.4 | 15.4 | 8.27 |
| 0.8 | 1.04 | 1.55 | 2.22 | 3.40 | 6.04 | 9.16 | 9.20 | 14.0 | 9.85 | 10.3 | 9.13 |
| 1.0 | 1.04 | 1.55 | 2.22 | 3.41 | 6.10 | 10.2 | 12.0 | 12.9 | 13.2 | 13.5 | 13.7 |
| 1.5 | 1.04 | 1.55 | 2.23 | 3.41 | 9.41 | 10.2 | 12.0 | 12.7 | 13.0 | 13.3 | 13.4 |
| 2.0 | 1.04 | 1.55 | 2.23 | 3.42 | 6.05 | 10.2 | 11.8 | 12.8 | 13.2 | 13.5 | 13.4 |

- `init` scale barely matters; `prior_cov` dominates Valid PPL.

---

## H-FLM Sweep - GenPPL, (init × prior_cov)

**Generation is best in a mid-`prior_cov` band (≈0.02–0.5); the ≈1 cells (\*) are collapses, not wins.**

<style scoped>table { font-size: 0.56em; margin: 0 auto; }</style>

| init \\ pc | 0.001 | 0.01 | 0.02 | 0.04 | 0.1 | 0.3 | 0.5 | 0.8 | 1.0 | 1.5 | 2.0 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `ngpt` | 109 | 23.8 | 159 | 40.3 | <ng>1.0\*</ng> | 27.3 | 20.4 | 32.7 | 44.4 | 31.8 | 35.9 |
| 0.001 | 73.1 | 24.6 | 38.7 | 120 | 22.6 | 438 | 41.5 | 108 | 39.2 | 53.8 | 46.5 |
| 0.01 | 109 | 23.8 | 27.5 | 20.1 | 24.1 | 21.1 | 17.9 | 32.5 | <ng>5.5\*</ng> | 31.6 | 45.8 |
| 0.02 | 112 | <ng>14.4\*</ng> | 21.1 | 19.5 | 19.0 | 23.7 | 29.9 | 22.7 | 37.3 | 70.0 | 54.6 |
| 0.04 | 102 | 33.7 | 31.7 | <ng>1.0\*</ng> | <ng>1.2\*</ng> | 29.2 | 25.0 | 41.2 | <rd>**17.7**</rd> | 33.9 | 32.7 |
| 0.1 | 103 | 29.1 | 23.3 | 48.5 | 49.5 | 58.3 | 54.6 | 33.1 | <ng>1.0\*</ng> | 41.1 | 24.0 |
| 0.3 | 121 | 56.3 | 33.7 | 27.1 | 42.7 | 41.2 | 52.7 | 49.3 | 48.1 | 60.7 | 57.6 |
| 0.5 | 125 | 75.5 | 60.9 | 36.4 | <ng>43.9\*</ng> | 54.7 | 47.9 | 50.2 | 46.4 | 37.1 | 83.2 |
| 0.8 | 110 | 69.9 | 60.4 | 51.8 | 48.7 | 51.1 | 56.5 | 89.6 | 52.4 | 37.6 | 71.0 |
| 1.0 | 128 | 73.8 | 62.3 | 51.4 | 52.5 | 53.0 | 55.4 | 56.5 | 56.1 | 57.6 | 60.5 |
| 1.5 | 128 | 76.5 | 75.9 | 60.5 | 55.1 | 52.8 | 55.5 | 56.6 | 57.6 | 58.6 | 58.1 |
| 2.0 | 124 | 71.8 | 67.8 | 53.7 | 58.5 | 54.7 | 56.3 | 59.6 | 58.8 | 58.9 | 59.7 |

- <rd>**Bold**</rd> = best non-collapsed (17.7)
- <ng>**\***</ng> = entropy < 3 → degenerate

---

## H-FLM Sweep - Entropy, (init × prior_cov)

<style scoped>table { font-size: 0.56em; margin: 0 auto; }</style>

| init \\ pc | 0.001 | 0.01 | 0.02 | 0.04 | 0.1 | 0.3 | 0.5 | 0.8 | 1.0 | 1.5 | 2.0 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `ngpt` | 3.9 | 3.9 | 3.9 | 3.9 | <ng>0.0\*</ng> | 3.6 | 4.1 | 3.6 | 3.7 | 3.6 | 3.7 |
| 0.001 | 3.8 | 4.1 | 4.0 | 3.0 | 4.2 | 4.7 | 4.1 | 3.6 | 4.2 | 4.0 | 4.1 |
| 0.01 | 3.9 | 3.8 | 4.0 | 4.0 | 4.0 | 4.1 | 4.1 | 3.5 | <ng>0.0\*</ng> | 3.8 | 3.5 |
| 0.02 | 3.9 | <ng>1.8\*</ng> | 4.3 | 4.3 | 4.1 | 3.9 | 3.8 | 4.0 | 3.4 | 4.0 | 3.0 |
| 0.04 | 3.9 | 3.7 | 4.1 | <ng>0.0\*</ng> | <ng>0.1\*</ng> | 4.0 | 4.1 | 3.8 | <rd>**4.0**</rd> | 3.4 | 3.8 |
| 0.1 | 3.8 | 3.9 | 4.2 | 3.9 | 4.2 | 3.4 | 4.4 | 3.9 | <ng>0.0\*</ng> | 3.6 | 4.1 |
| 0.3 | 3.7 | 3.9 | 4.1 | 4.2 | 4.3 | 4.1 | 4.3 | 3.8 | 3.9 | 4.2 | 3.7 |
| 0.5 | 3.7 | 3.5 | 3.8 | 3.7 | <ng>2.0\*</ng> | 3.7 | 4.0 | 3.8 | 3.9 | 4.0 | 4.3 |
| 0.8 | 3.7 | 3.5 | 3.6 | 3.7 | 3.8 | 3.9 | 3.9 | 3.0 | 3.9 | 4.0 | 4.2 |
| 1.0 | 3.8 | 3.5 | 3.6 | 3.7 | 3.8 | 3.9 | 4.0 | 4.0 | 3.9 | 4.0 | 3.9 |
| 1.5 | 3.8 | 3.4 | 3.5 | 3.6 | 3.8 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 |
| 2.0 | 3.8 | 3.5 | 3.5 | 3.7 | 3.7 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 |

- Healthy generation ≈ 3.5–4.4
- <rd>**Bold**</rd> = Best non-degenerated
- <ng>\*</ng> = entropy < 3 (collapse)

---

## Conclusion

- Hyperbolic is the bets geometry but requires tuning ``init`` and ``prior_cov``
- Hyperbolic surpass Euclidean and Sphere, but still far from AR

---

## Next Step

- How to set up a proper ``init`` and ``prior_cov`` for trainable word embedding?
- I found the distribution of the length of word embeddings differ across H-FLM and other models.

---

---

# Distribution over the Length & Value of Word Embeddings

![alt text](image-2.png)

- The length of word embeddings of all AR and S-FLM follow normal distribution (no long tail)

---

## H-FLM Pre-trained on TinyStories

### Distribution over Length, Log-scled Y axis

![alt text](image.png)

---

## H-FLM Pre-trained on TinyStories

### Distribution over Length, Linear-scled Y axis

![alt text](image-1.png)

---

## S-FLM Naive Pre-trained on TinyStories

### Distribution over Length, Log-scled Y axis

![alt text](image-9.png)

---

## S-FLM Naive Pre-trained on TinyStories

### Distribution over Length, Linear-scled Y axis

![alt text](image-10.png)

---

## GPT2 Small Pre-trained on OWT

### Distribution over Length, Log-scled Y axis

![alt text](image-3.png)

---

## GPT2 Small Pre-trained on OWT

### Distribution over Length, Linear-scled Y axis

![alt text](image-4.png)

---

## GPT2 XL Pre-trained on OWT

### Distribution over Length, Log-scled Y axis

![alt text](image-5.png)

---

## GPT2 XL Pre-trained on OWT

### Distribution over Length, Linear-scled Y axis

![alt text](image-6.png)

---

### BERT Base Pre-trained on BooksCorpus + Eng Wiki

### Distribution over Length, Log-scled Y axis

![alt text](image-7.png)

---

### BERT Base Pre-trained on BooksCorpus + Eng Wiki

### Distribution over Length, Linear-scled Y axis

![alt text](image-8.png)

---

# Distribution over the Value of Word Embeddings

---

## H-FLM Pre-trained on TinyStories

### Distribution over Values, Log-scled Y axis

![alt text](image-11.png)

---

## H-FLM Pre-trained on TinyStories

### Distribution over Values, Linear-scled Y axis

![alt text](image-12.png)

---

## S-FLM Naive Pre-trained on TinyStories

### Distribution over Values, Log-scled Y axis

![alt text](image-13.png)

---

## S-FLM Naive Pre-trained on TinyStories

### Distribution over Values, Linear-scled Y axis

![alt text](image-14.png)

---

## GPT2 Small Pre-trained on OWT

### Distribution over Values, Log-scled Y axis

![alt text](image-15.png)

---

## GPT2 Small Pre-trained on OWT

### Distribution over Values, Linear-scled Y axis

![alt text](image-16.png)

---

## GPT2 XL Pre-trained on OWT

### Distribution over Values, Log-scled Y axis

![alt text](image-19.png)

---

## GPT2 XL Pre-trained on OWT

### Distribution over Values, Linear-scled Y axis

![alt text](image-20.png)

---

### BERT Base Pre-trained on BooksCorpus + Eng Wiki

### Distribution over Values, Log-scled Y axis

![alt text](image-17.png)

---

### BERT Base Pre-trained on BooksCorpus + Eng Wiki

### Distribution over Values, Linear-scled Y axis

![alt text](image-18.png)