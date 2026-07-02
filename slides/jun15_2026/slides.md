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

#### June 15, 2026

---

# Sudoku Exp: Setting Up

**Same recipe for all systems — reproduces *Hyperspherical Flows* (arXiv:2605.11125, Tbl 1)**

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M) · 81-cell grid (seq-len 180)

---

# Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0

---

# Sudoku Exp: Setting Up

| Model | Space | Sampler @ eval (180 steps) |
|---|---|---|
| AR | discrete, causal | greedy autoregressive |
| MDLM · Duo | discrete, masked | ancestral |
| CANDI | discrete | CANDI |
| FLM (one-hot) | simplex | Euler ODE |

---

# Sudoku Exp: Setting Up

| Model | Space | Sampler @ eval (180 steps) |
|---|---|---|
| S-FLM (naive · trunc · +adaptive) | sphere | exact-velocity, greedy, top_k_v = -1, 1 |
| **H-FLM (ours)** | hyperbolic | exact-velocity, greedy, top_k_v = 1 |

- top_k_v
  - 1: top-1
  - -1: average across the whole vocab

<!-- - S-FLM noise: log-linear · +<b>α⋆=0.093</b> trunc · +adaptive (refit 50) · H-FLM prior: cov 0.25, ρ_max 12 -->

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

All values are reported by the paper

| Model | Easy | Med | Hard |
|---|---|---|---|
| AR (greedy) | 14.6 | 5.1 | 1.0 |
| MDLM | 92.0 | 77.1 | 30.2 |
| Duo | 96.3 | 84.7 | 58.4 |
| CANDI | 79.3 | 45.9 | 16.7 |
| FLM (one-hot) | 94.2 | 82.7 | 44.5 |

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

S-FLMs: Reproduced, Use top_k_v = -1, velocity = average across the whole vocab

| Model | Easy | Med | Hard |
|---|---|---|---|
| S-FLM (naive) | 77.6 | 32.6 | 13.9 |
| S-FLM + α⋆(0.093) trunc | 95.1 | 77.9 | 48.3 |
| S-FLM + α⋆(0.093) + adaptive | 94.2 | 79.2 | 46.0 |
| **H-FLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

---

# Sudoku Exp Results

**Exact-match accuracy (%)**

S-FLMs: Reproduced, Use top_k_v = 1, velocity = only choose top-1 token

| Model | Easy | Med | Hard |
|---|---|---|---|
| S-FLM (naive) | 76.6 | 33.2 | 15.2 |
| S-FLM + α⋆(0.093) trunc | 94.8 | 78.1 | 47.6 |
| S-FLM + α⋆(0.093) + adaptive | 93.8 | 78.9 | 44.9 |
| **H-FLM (naive, ours)** | **93.75** | **67.40** | **24.10** |

---

# Sudoku Exp Results: H-FLM

**Geometry alone lifts the *naive* model: H-FLM (naive) ≫ S-FLM (naive, reproduced)**

| naive model (ours, local) | easy | med | hard |
|---|---|---|---|
| S-FLM (naive), top_k_v = -1 | 77.6 | 32.6 | 13.9 |
| S-FLM (naive), top_k_v = 1 | 76.6 | 33.2 | 15.2 |
| **H-FLM (naive)** | **93.8** | **67.4** | **24.1** |

- Hyperbolic geometry *alone* perform better than naive S-FLM
- H-FLM still trails the *tricked* S-FLM

---

# Sudoku Exp Results: H-FLM samples

**Failures are near-misses — H-FLM learned the rules; errors are local digit swaps**

✓ easy — generated grid, fully correct (all 81 cells):

```
9 2 1 6 4 7 3 8 5
7 6 4 8 3 5 2 9 1
5 8 3 9 1 2 7 6 4
4 5 2 3 6 9 1 7 8
1 7 9 4 5 8 6 2 3
8 3 6 7 2 1 5 4 9
2 1 7 5 8 4 9 3 6
6 9 8 1 7 3 4 5 2
3 4 5 2 9 6 8 1 7
```

---

# TinyStories Exp: Setting Up

**Same recipe for all 3 — only the geometry/noise differs (geometry vs. tricks)**

- Data: TinyStories (gpt2 tok, ~472M train tokens) · ~33 epochs
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12** (~169M)

---

# TinyStories Exp: Setting Up

**Same recipe for all 3 — only the geometry/noise differs (geometry vs. tricks)**

- Training
  - Training Steps: **30k** · Batch Size: **512** · Max Seq Len: **1024**
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0 · EMA: 0.9999

---

# TinyStories Exp: Setting Up

**The 3 models** (everything above held fixed)

| Model | Geometry | Noise schedule |
|---|---|---|
| S-FLM (naive) | sphere | log-linear |
| S-FLM (trunc + adaptive) | sphere | log-linear, <b>α⋆=0.121</b> + adaptive |
| **H-FLM (ours)** | hyperbolic | log-linear (prior cov 0.25, ρ_max 12) |

---

# TinyStories Exp: Setting Up

- Evaluation
  - Sampling Steps (NFE): **1024**
  - Sampler: geometry-matched (sphere / hyperbolic geodesic)
  - noise_removal: greedy (H-FLM) / ancestral (S-FLM)
  - top_k_velocity: top-1 (H-FLM, 1) / top-1 (S-FLM, 1)
  - Metrics: 
    - <b>GenPPL</b> (gpt2-large on samples) = the comparison
    - <r>"val PPL" = held-out denoising CE, a diagnostic — *not* a true perplexity</r>

---

# TinyStories Exp Results

**H-FLM converges early; generation quality peaks mid-training, not at the end**

| H-FLM ckpt | val CE (`exp`) | GenPPL ↓ | entropy |
|---|---|---|---|
| 10k | 6.00 | 45.0 | 4.88 |
| 20k | 6.14 | **41.6** | 4.89 |
| 30k | 6.02 | 49.2 | 4.90 |


---

# TinyStories Exp Results

**Samples: coherent locally, drifts globally** (H-FLM, 20k)

> *Once upon a time there was a little girl named Sarah. She was only three years old … she was at the beach with her mom. … It was a big bottle of sweet water and began coll[ecting] …*

---

# TinyStories Exp Results

> *Once upon a time, there was a little boy named Timmy. Timmy was excited because he liked to play with his wagon outside … Timmy was curious and liked to play with it.*

- ✅ Names, story openings, simple arcs, child-level vocabulary
- ⚠️ Local slips — repetition (*"and and"*, *"soon soon"*), occasional ungrammatical spans

---

# Geometry Comaprison for Continuous DLM

---

## Experiment Setup

### 3 kinds of geometries

Reproduced

- Euclidean: LangFlow (w/o adaptive scheduler and self conditioning), E-FLM (niave)
- Sphere: S-FLM (w/o adaptive scheduler and trucation)
- Hyperbolic: H-FLM (w/o tuning LR and word embedding init)

---

## Experiment Setup

### 3 kinds of geometries

All of them follows the Riemannian Flow Matching + VFM

- Uniform sampled timestep
- Geodesic Path
- Cross Entropy Loss

---

## Experiment Setup

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M), Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)

---

# Sudoku Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = 1 (top-1), 180 sampling steps
  - Greedy decoding for last sampling step

---

| Model | easy | medium | hard |
|---|---|---|---|
| **S-FLM (naive)** | 78.4 | 39.0 | 13.9 |
| S-FLM + trunc | 95.05 | 77.9 | 48.3 |
| S-FLM + trunc + adaptive | 94.15 | 78.2 | 45.95 |
| **E-FLM (naive)** | 90.35 | 62.45 | 17.55 |
| **LangFlow (naive)** | 3.20 | 0.20 | 0.00 |
| LangFlow + SC | 7.05 | 0.20 | 0.00 |
| LangFlow + ada sched | 81.60 | 59.85 | 23.45 |
| LangFlow + ada sched + SC | 94.90 | 83.30 | 57.00 |
| **H-FLM (naive, our)** | 89.5 | 75.50 | 31.7 |
| **H-FLM (tuned, our)** | 96.5 | 84.50 | 37.1 |
<!-- | **H-FLM (naive, our)** | 93.75 | 67.4 | 24.1 | -->

---

## Conclusion

Hyperbolic is the most suitable geometry for diffusing on word embedding

---

# Hyperparameter Sensiticity of Geometries

---

# Sub-Exp 1: Init * Dim on Sudoku Medium

---

## Init * Dim Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$, ``random``: $\mathcal{N}(0, 4e-4)$, ``unit_var``: $\mathcal{N}(0, 1.0^2)$

---

## Init * Dim Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: 3e-4, Weight Decay: 0.0
    - Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

## Init * Dim Exp: S-FLM Results

**S-FLM (sphere)** — normalizes embeddings, so init *scale* is erased:

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | 46.75 | 55.40 | 38.20 | 46.8 |
| random | 46.25 | 52.35 | 50.65 | 49.8 |
| unit_var | 49.55 | 53.20 | 38.30 | 47.0 |
| **dim mean** | **47.5** | **53.6** | **42.4** | **47.9** |

---

## Init * Dim Exp: E-FLM Results

**E-FLM (Euclidean)** — raw embeddings, init *scale* matters most:

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | 51.70 | 48.05 | 50.85 | 50.2 |
| random | 61.00 | 61.40 | 59.05 | **60.5** |
| unit_var | 46.95 | 40.60 | 39.20 | 42.3 |
| **dim mean** | 53.2 | 50.0 | 49.7 | **51.0** |

---

## Init * Dim Exp: H-FLM Results

**H-FLM (hyperbolic)** — ‖e‖ is the radial coordinate (clamped at `rho_max=12`):

| init \ dim | 512 | 256 | 128 | init mean |
|---|---|---|---|---|
| ngpt | **75.50** | 73.05 | 56.40 | 68.3 |
| random | 74.55 | 72.50 | 50.75 | 65.9 |
| unit_var | 65.20 | 68.20 | 42.75 | 58.7 |
| **dim mean** | **71.8** | 71.3 | 50.0 | **64.3** |

---

## Init * Dim Exp: Conclusion

- Initialization Sensitivity: E-FLM > H-FLM >> S-FLM
- H-FLM almost beats E-FLM and S-FLM in every settings, except for 1 case: 128 dim, random init
- H-FLM at least outperform E-FLM and S-FLM by 10% acc for 512 and 256 dim

---

# Sub-Exp 2: Init * Dim * LR on Sudoku Medium & Hard

---

## Init * Dim * LR Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42) · clues: easy 40 / med 35 / hard 30
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$, ``random``: $\mathcal{N}(0, 4e-4)$, ``unit_var``: $\mathcal{N}(0, 1.0^2)$

---

## Init * Dim * LR Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {5e‑5, 8e‑5, 1e‑4, 3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss

---

## Init * Dim * LR Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

# RESULTS — Medium LR sweep (geometry × init × dim × LR)

**Per-geometry LR means** (avg over the 12 init×dim cells):

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | $\eta^*$ |
|---|---|---|---|---|---|---|---|
| S-FLM | 18.2 | 30.8 | 30.9 | 47.8 | 48.7 | 46.5 | **5e-4** |
| E-FLM | 35.0 | 44.6 | 45.2 | 51.0 | 50.1 | 41.6 | **3e-4** |
| H-FLM | 28.4 | 37.8 | 47.7 | 63.1 | 62.0 | 68.9 | **1e-3** |

- $\eta^*$ represents **best LR**

---

# RESULTS — Medium LR sweep (geometry × init × dim × LR)

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | $\eta^*$ |
|---|---|---|---|---|---|---|---|
| S-FLM | <hy>40.2</hy> | <rd>50.1</rd> | <ng>50.8</ng> | <ng>55.4</ng> | <uv>58.5</uv> | <rd>**61.7**</rd> | 1e-3 |
| E-FLM | <rd>64.6</rd> | <rd>67.0</rd> | <rd>**69.9**</rd> | <rd>61.4</rd> | <rd>60.8</rd> | <rd>57.2</rd> | 1e-4 |
| H-FLM | <ng>54.4</ng> | <rd>62.2</rd> | <rd>75.6</rd> | <ng>75.5</ng> | <ng>70.2</ng> | <rd>**84.5**</rd> | 1e-3 |

- $\eta^*$ represents **best LR**

---

# RESULTS — Hard LR sweep (geometry × init × dim × LR)

**Per-geometry LR means** (avg over available init×dim cells):

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |  $\eta^*$ |
|---|---|---|---|---|---|---|---|
| SFM | 2.9 | 5.8 | 7.2 | 13.8 | 14.2 | 15.4 | **1e-3** |
| EFLM | 9.6 | 11.1 | 14.6 | 16.3 | 14.2 | 7.5 | **3e-4** |
| HFLM | 8.6 | 12.7 | 12.6 | 20.3 | 19.0 | 25.1 | **1e-3** |

- $\eta^*$ represents **best LR**

---

# RESULTS — Hard LR sweep (geometry × init × dim × LR)

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | $\eta^*$ |
|---|---|---|---|---|---|---|---|
| S-FLM | <hy>8.0</hy> | <rd>12.0</rd> | <ng>15.9</ng> | <rd>20.6</rd> | <hy>**22.5**</hy> | <hy>20.8</hy> | 5e-4 |
| E-FLM | <rd>27.1</rd> | <rd>24.4</rd> | <ng>25.2</ng> | <rd>**29.1**</rd> | <rd>21.1</rd> | <ng>15.5</ng> | 3e-4 |
| H-FLM | <rd>34.4</rd> | <rd>**37.1**</rd> | <rd>31.2</rd> | <ng>31.7</ng> | <ng>32.2</ng> | <rd>34.6</rd> | 8e-5 |

- $\eta^*$ represents **best LR**

---

# RESULTS — Easy LR sweep (geometry × init × dim × LR)

**Per-geometry LR means** (avg over available init×dim cells):

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | $\eta^*$ |
|---|---|---|---|---|---|---|---|
| SFM | 43.2 | 65.9 | 66.7 | 77.6 | 78.8 | 79.9 | **1e-3** |
| EFLM | 68.8 | 79.1 | 79.6 | 85.7 | 82.6 | 79.5 | **3e-4** |
| HFLM | 66.3 | 77.8 | 81.3 | 87.3 | 90.3 | 89.4 | **5e-4** |

- $\eta^*$ represents **best LR**

---

# RESULTS — Easy LR sweep (geometry × init × dim × LR)

**Per-geometry LR Best** (Best over the 12 init×dim cells):

Acc color = init of the best cell: <ng>ngpt</ng> · <rd>random</rd> · <uv>unit_var</uv> · <hy>hyperbolic</hy>

| Geo | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 | $\eta^*$ |
|---|---|---|---|---|---|---|---|
| S-FLM | <ng>66.6</ng> | <hy>78.1</hy> | <rd>77.1</rd> | <uv>84.0</uv> | <rd>83.9</rd> | <rd>**90.4**</rd> | 1e-3 |
| E-FLM | <ng>**93.3**</ng> | <rd>93.3</rd> | <rd>92.7</rd> | <rd>91.0</rd> | <rd>91.6</rd> | <rd>88.4</rd> | 5e-5 |
| H-FLM | <rd>92.7</rd> | <rd>94.7</rd> | <rd>92.0</rd> | <ng>91.5</ng> | <uv>**96.5**</uv> | <hy>95.5</hy> | 5e-4 |

- $\eta^*$ represents **best LR**

---

### S-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 20.5 | 40.9 | 10.8 | 46.8 | 46.9 | 42.0 |
| ngpt | 256 | 32.5 | 37.0 | 50.8 | 55.4 | 54.2 | 60.1 |
| ngpt | 128 | 1.7 | 17.0 | 14.8 | 38.2 | 51.8 | 42.8 |
| random | 512 | 10.0 | 50.1 | 46.7 | 46.2 | 43.9 | 40.6 |
| random | 256 | 22.0 | 38.2 | 32.4 | 52.3 | 49.8 | 50.5 |
| random | 128 | 3.1 | 17.6 | 21.4 | 50.6 | 40.8 | 61.7 |

---

### S-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 29.6 | 33.6 | 42.3 | 49.5 | 41.8 | 32.1 |
| unit_var | 256 | 29.4 | 40.1 | 42.4 | 53.2 | 50.0 | 46.4 |
| unit_var | 128 | 2.1 | 20.8 | 31.1 | 38.3 | 58.5 | 48.2 |
| hyperbolic | 512 | 40.2 | 16.0 | 31.2 | — | 38.6 | 42.6 |
| hyperbolic | 256 | 22.1 | 36.5 | 40.6 | — | 57.5 | 50.3 |
| hyperbolic | 128 | 5.5 | 21.3 | 6.5 | — | 50.7 | 40.1 |

---

### E-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 51.3 | 59.1 | 65.8 | 51.7 | 55.2 | 40.1 |
| ngpt | 256 | 48.0 | 49.9 | 47.8 | 48.0 | 46.6 | 40.0 |
| ngpt | 128 | 15.0 | 22.8 | 27.3 | 50.8 | 53.8 | 52.3 |
| random | 512 | 64.6 | 67.0 | 69.9 | 61.0 | 59.8 | 39.1 |
| random | 256 | 59.0 | 45.1 | 40.3 | 61.4 | 60.8 | 50.6 |
| random | 128 | 31.5 | 46.7 | 41.3 | 59.1 | 50.3 | 57.2 |

---

### E-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 32.5 | 47.4 | 52.7 | 46.9 | 50.6 | 21.8 |
| unit_var | 256 | 28.7 | 48.0 | 41.7 | 40.6 | 39.9 | 32.6 |
| unit_var | 128 | 4.4 | 14.5 | 18.8 | 39.2 | 36.9 | 38.0 |
| hyperbolic | 512 | 47.5 | 62.0 | 60.5 | — | 43.1 | 27.2 |
| hyperbolic | 256 | 31.9 | 44.6 | 45.2 | — | 52.9 | 52.0 |
| hyperbolic | 128 | 5.0 | 27.4 | 30.8 | — | 51.6 | 48.1 |

---

### H-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 54.4 | 59.5 | 68.8 | 75.5 | 69.2 | 72.6 |
| ngpt | 256 | 20.5 | 33.1 | 53.3 | 73.0 | 70.2 | 79.3 |
| ngpt | 128 | 12.1 | 27.6 | 30.7 | 56.4 | 63.0 | 63.7 |
| random | 512 | 47.9 | 62.2 | 75.6 | 74.6 | 66.4 | 84.5 |
| random | 256 | 28.8 | 50.4 | 52.1 | 72.5 | 65.3 | 70.5 |
| random | 128 | 13.8 | 6.0 | 25.1 | 50.7 | 55.5 | 61.3 |

---

### H-FLM - Medium

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 38.5 | 58.1 | 43.1 | 65.2 | 58.5 | 67.2 |
| unit_var | 256 | 41.5 | 49.9 | 64.6 | 68.2 | 65.5 | 73.3 |
| unit_var | 128 | 19.2 | 11.5 | 36.3 | 42.8 | 48.3 | 56.3 |
| hyperbolic | 512 | 50.1 | 54.3 | 57.2 | 66.7 | 69.5 | 69.8 |
| hyperbolic | 256 | 4.7 | 16.8 | 43.1 | 61.4 | 63.8 | 74.2 |
| hyperbolic | 128 | 9.4 | 24.2 | 21.9 | 49.6 | 49.3 | 54.5 |

---

### S-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 3.8 | 10.9 | 15.9 | 7.8 | 13.8 | 15.2 |
| ngpt | 256 | 1.1 | 4.3 | 9.0 | 11.6 | 12.9 | 17.0 |
| ngpt | 128 | 0.8 | 1.5 | 0.7 | 10.2 | 10.5 | 14.6 |
| random | 512 | 5.6 | 12.0 | 10.5 | 11.5 | 14.6 | 12.4 |
| random | 256 | 2.6 | 4.3 | 10.7 | 20.6 | 13.1 | 20.0 |
| random | 128 | 0.1 | 1.8 | 1.7 | 10.8 | 10.8 | 11.9 |

---

### S-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 6.7 | 11.3 | 10.9 | 12.6 | 17.1 | 15.0 |
| unit_var | 256 | 1.4 | 10.1 | 5.7 | 15.2 | 15.6 | 11.2 |
| unit_var | 128 | 0.1 | 0.5 | 2.1 | 15.8 | 17.2 | 19.6 |
| hyperbolic | 512 | 8.0 | 11.7 | 14.2 | 14.2 | 8.6 | 7.1 |
| hyperbolic | 256 | 4.0 | 1.1 | 3.0 | 15.4 | 22.5 | 20.8 |
| hyperbolic | 128 | 0.1 | 0.4 | 1.9 | 19.8 | 13.2 | 20.2 |

---

### E-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 23.9 | 17.3 | 25.2 | 20.2 | 15.8 | 6.9 |
| ngpt | 256 | 8.1 | 12.7 | 12.5 | 12.2 | 14.3 | 8.4 |
| ngpt | 128 | 1.5 | 3.4 | 9.8 | 12.7 | 12.4 | 15.5 |
| random | 512 | 27.1 | 24.4 | 23.7 | 29.1 | 21.1 | 11.3 |
| random | 256 | 16.5 | 10.4 | 24.9 | 22.9 | 15.7 | 11.7 |
| random | 128 | 7.3 | 13.4 | 11.8 | 15.4 | 19.4 | 10.2 |

---

### E-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 6.9 | 12.4 | 16.9 | 10.9 | 12.1 | 3.8 |
| unit_var | 256 | 5.1 | 6.1 | 10.3 | 18.6 | 11.9 | 2.7 |
| unit_var | 128 | 0.1 | 2.1 | 3.2 | 10.2 | 4.1 | 2.9 |
| hyperbolic | 512 | 11.7 | 16.9 | 19.4 | 14.2 | 10.2 | 1.4 |
| hyperbolic | 256 | 5.7 | 10.5 | 12.3 | 17.6 | 19.8 | 11.2 |
| hyperbolic | 128 | 0.9 | 4.1 | 5.0 | 11.5 | 13.7 | 4.1 |

---

### H-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| ngpt | 512 | 24.4 | 26.4 | 25.2 | 31.7 | 32.2 | 31.9 |
| ngpt | 256 | 2.9 | 0.7 | 5.5 | 23.2 | 27.1 | 31.1 |
| ngpt | 128 | 0.4 | 1.7 | 4.0 | 12.6 | 14.0 | 19.2 |
| random | 512 | 34.4 | 37.1 | 31.2 | 20.9 | 22.8 | 34.6 |
| random | 256 | 11.5 | 15.8 | 23.4 | 21.6 | 14.5 | 26.8 |
| random | 128 | 1.7 | 7.5 | 2.4 | 18.0 | 15.0 | 18.6 |

---

### H-FLM - Hard

| init | dim | 5e-5 | 8e-5 | 1e-4 | 3e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| unit_var | 512 | 5.9 | 15.2 | 4.0 | 17.5 | 21.1 | 22.4 |
| unit_var | 256 | 0.5 | 19.6 | 17.2 | 27.5 | 14.4 | 30.2 |
| unit_var | 128 | 0.3 | 4.1 | 8.5 | 8.8 | 12.3 | 19.8 |
| hyperbolic | 512 | 19.4 | 14.3 | 23.2 | 27.5 | 25.9 | 28.9 |
| hyperbolic | 256 | 1.4 | 7.1 | 2.9 | 24.4 | 18.2 | 28.8 |
| hyperbolic | 128 | 0.8 | 3.2 | 3.5 | 10.1 | 10.6 | 8.6 |

---

## Conclusion

**Hyperbolic is the best geometry — and the only one that *rewards* aggressive LR**

- Best-LR vs best-LR: **H-FLM 68.9** (@1e‑3) ≫ E-FLM 51.0 (@3e‑4) ≈ S-FLM 48.7 (@5e‑4) — single best cell **84.5** (H-FLM, random/512 @1e‑3)
- H-FLM leads at **every LR ≥ 1e‑4** and is *still climbing* at 1e‑3 (the largest tested)
- Only at tiny LR (≤ 8e‑5) does Euclidean lead

---

## Conclusion

- <r>Caveat:</r> 
  - single seed, the Acc is noisy
  - Should try larger LR or more training epoch

---

# Idea: Hyperbolic GPT

---

- The paper nGPT has shown that noramlize embeddings, weight rows, hidden states to the unit hypersphere with additional learnable relaxation scalars can brings 4–20x fewer training steps for the same loss
- I think the relaxation scalars should be interpreted as radial in hyperbolic geometry.
- The paper only conduct experiment on wide network (1024 and 1280), which shouldn't work in narrow network.

---

## nGPT: Normalized Transformer with Representation Learning on the Hypersphere

---

## Motivation

- NVIDIA, ICLR 2025
- **Idea**: constrain every vector — embeddings, weight rows, hidden states — to the **unit hypersphere**
- **Result**: 4–20x fewer training steps for the same loss

---

## Idea

- Remove all normalization layers such as RMSNorm or LayerNorm.
- Normalize everything to sphere, including weight matrices, hidden states, and word embeddings
- Add relaxation scalar on output-side word embedding, Self-Attention, and MLP block

---

## What nGPT Normalizes - Word Embedding

To improve the accuracy of similarity estimation, we propose to
normalize the embedding vectors stored in $\mathbf{E}_{\text{input}}$ and $\mathbf{E}_{\text{output}}$ after each step of the training algorithm.

---

## What nGPT Normalizes - Word Embedding

![alt text](image-2.png)

![alt text](image-3.png)

---

## What nGPT Normalizes - Hidden States

![alt text](image-4.png)

---

## What nGPT Normalizes - Hidden States

Since the paper found SLERP doesn't outperform LERP for residual block, they use LERP instead of SLERP.

![alt text](image-10.png)

I think the reason is wide network + small LR cause the LERP is close to SLERP.

---

## What nGPT Normalizes - Self-Attention

![alt text](image-5.png)

![alt text](image-6.png)

Also normalize $\mathbf{W}_{q}$, $\mathbf{W}_{k}$, $\mathbf{W}_{v}$, and $\mathbf{W}_{o}$ along their embedding dimension.

---

## What nGPT Normalizes - MLP

![alt text](image-7.png)

![alt text](image-8.png)

---

## What nGPT Normalizes - MLP

![alt text](image-9.png)

Also normalize $\mathbf{W}_{u}$ and $\mathbf{W}_{\nu}$ along their embedding dimension.

---

## Experiments Setup - Model Arch

![alt text](image-11.png)

---

## Experiments Setup - Training

![alt text](image-12.png)

---

## Experiments Setup - Optimizer

![alt text](image-13.png)

---

## Experiments Setup - Initialization

![alt text](image-14.png)

---

## Experiments: 10x Fewer Iterations 

1B models, 4k context, OpenWebText — nGPT at **20k** iters matches GPT at **200k**

![w:520 center](image-1.png)

---

## Experiments: Speedup Grows with Context

Same final loss with ~**4x** (1k ctx), **10x** (4k), **20x** (8k) fewer tokens

![w:560 center](image.png)