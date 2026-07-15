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

#### July 2, 2026

---



---

![alt text](image.png)

---

Define MMSE

![alt text](image-1.png)

Define $\gamma(u)$

![alt text](image-2.png)

---

Define the information profile $\rho^*(u)$

![alt text](<Screenshot 2026-06-29 at 9.36.42 PM.png>)


InfoNoise samples thetimestep $u$ from the information profile $\rho^*(u)$

---

![alt text](image-3.png)

The paper also visualizes the information profile along with the Bayes posterior uncertainty. 

---

It claims that sampling the timestep $u$ from the information profile $\rho^*(u)$ can (1) Reduce Variance (2) Reduce training iteration (3) Without additional model training for timestep allocation

However, it doesn't provide optimal guarantee.

---

---

# H-FLM Curvature * Init * LR on Sudoku

---

## Curvature * Init * LR Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {med 35 / hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$(= custom 0.0442), ``random``: $\mathcal{N}(0, 4e-4)$ (= custom 0.02)
  - ``custom``: std: {0.01, 0.04, 0.06, 0.08}

- Geometry Curvature: {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}

---

## Curvature * Init * LR Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {1e‑4, 3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge

---

## Curvature * Init * LR Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

## Environment

- Exp name: ``hflm_curv_init_lr_sudoku``
- Use both ``desa`` and ``thickstun`` partition on unicorn and any available GPUs on Tinkercliffs and Falcon. Read Agent.md for more server details. For Tinkercliffs and Falcon, use ``/home/shengyenc/workspace/research/s-flm`` as working directory.
- Refer to ``experiments/hflm_curv_sudoku/sweep.py`` for the ``sweep.py`` format
- Prioritize Init {``random``} * LR {3e‑4, 1e-3} * Curvature {-0.25, -0.3, -0.5, -0.7, -1.0, -1.5}

---


## Recall Baselines

<div style="font-size: 0.7em;">

| Model | easy | medium | hard |
|---|---|---|---|
| AR | 14.7 ± 3.5 | 3.4 ± 0.3 | 0.5 ± 0.3 |
| S-FLM (naive) | 78.8 ± 1.1 | 43.8 ± 3.2 | 11.1 ± 1.7 |
| S-FLM + trunc | 94.4 ± 0.4 | 79.8 ± 1.7 | 42.4 ± 3.4 |
| S-FLM + trunc + ada | 95.0 ± 0.8 | 76.7 ± 7.3 | 42.2 ± 2.8 |
| E-FLM (naive) | 88.2 ± 1.2 | 62.2 ± 2.3 | 19.2 ± 3.3 |
| LangFlow + ada | 81.2 ± 0.9 | 52.4 ± 2.7 | 18.2 ± 2.1 |
| LangFlow + ada + SC | 97.0 ± 0.5 | 87.2 ± 1.9 | 50.4 ± 4.6 |
| HFLM (tuned) | - | 83.23 ± 5.46 | 46.22 ± 13.13 |

</div>

HFLM (tuned) only grid search on word embedding initialization, LR, and global curvature.

---

## LangFlow Relies on Self Conditioning Heavily

![alt text](image-9.png)

- LangFlowPaper claims their biggest contribution is ``Information-uniform Scheduler``, but it helps limited.

---

## Best Over {Init * LR} per Curvature

### Medium

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 1e-3 | 80.78 ± 2.82 | 3 |
| **−0.30** | c0.01 @ 5e-4 | 83.23 ± 5.46 | 3 |
| −0.50 | random @ 1e-3 | 82.68 ± 1.75 | 3 |
| −0.70 | c0.02 @ 5e-4 | 81.20 ± 1.08 | 3 |
| −1.00 (baseline) | c0.01 @ 5e-4 | 80.88 ± 0.63 | 3 |
| −1.50 | c0.02 @ 1e-3 | 75.87 ± 1.90 | 3 |

---

## Best Over {Init * LR} per Curvature

### Hard

| K | best init@lr | acc % ± seed-std | n |
|---|---|---|---|
| −0.25 | c0.01 @ 5e-4 | 39.87 ± 1.18 | 3 |
| −0.30 | c0.01 @ 3e-4 | 42.07 ± 9.92 | 3 |
| **−0.50** | c0.01 @ 3e-4 | 46.22 ± 13.13 | 3 |
| −0.70 | c0.04 @ 1e-3 | 40.43 ± 4.93 | 3 |
| −1.00 (baseline) | c0.01 @ 3e-4 | 40.37 ± 5.28 | 3 |
| −1.50 | c0.01 @ 3e-4 | 34.98 ± 9.02 | 3 |

---

## Conclusion

### For HFLM

- Curvature does help the accuracy
- HFLM has relatively higher variance than baselines in accuracy, especially for {K: −0.50, c0.01@3e-4, hard}, accuracy std is 13.33
- The single best run of HFLM in hard {K: −0.50, c0.01@3e-4} is **58%** which beats all baselines
- Can we stabalize the training loss for HFLM? 

---

## Conclusion

### For LangFlow

- Self-conditioning brings more imrpovement than ``Information-uniform Scheduler``, but it's a wel-known trick applicable to any DLMs
- What if we apply SC on EFLM and S-FLM and trunc+ada on EFLM?

---

## Conclusion: Next Step

- Visualize the loss geometry (Loss vs timestep)
- Try trunc + ada sched on EFLM, HFLM, and SFLM

---

# Loss Geometry of FLMs

---



## Sudoku Hard

---

### Loss Geometry of S-FLM(naive) on Sudoku Hard, Acc: 11.1%

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/sfm/sfm.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/sfm/sfm_log.png" width="400"/>
</div>

---

### Loss Geometry of S-FLM(+trunc) on Sudoku Hard, Acc: 42.2%

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/sfm_trunc/sfm_trunc.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/sfm_trunc/sfm_trunc_log.png" width="400"/>
</div>

- Truncation mitigates [0.0, 0.8] and the tail, makes it smoother

---

### Loss Geometry of S-FLM(+trunc+ada) on Sudoku Hard, Acc: 42.2%

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/sfm_trunc_ada/sfm_trunc_ada.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/sfm_trunc_ada/sfm_trunc_ada_log.png" width="400"/>
</div>

- Trunc+Ada is the smoothest loss geometry across all DLMs

---

### Loss Geometry of E-FLM on Sudoku Hard, Acc: 19.2%

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/eflm/eflm.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/eflm/eflm_log.png" width="400"/>
</div>

---

### Loss Geometry of E-FLM on Sudoku Hard, Vary Curvature

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/hflm_curv_overlay.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/hflm_curv_overlay_log.png" width="400"/>
</div>

---

### Loss Geometry of H-FLM on Sudoku Hard, Acc: 46.2%

- Curvature = -0.5

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/hflm_K0.5/K0.5.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/hflm_K0.5/K0.5_log.png" width="400"/>
</div>

---

### Loss Geometry of LangFlow(+ada+SC) on Sudoku Hard, Acc: 50.4%

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/sudoku_hard/langflow_full/langflow_full.png" width="400"/>
  <img src="loss_geometry_vis/sudoku_hard/langflow_full/langflow_full_log.png" width="400"/>
</div>

---

# Conclusion

- Curvature helps acc because it shapes the cliff of loss geometry of EFLM and SFLM (naive) to expontential decading through timestep.
- SFLM (trunc + ada) has truncated the tail, which has the most smooth slope across all FLMs.
- What if HFLM + Adaptive sched?
- Does linear decading sched perform better than exponential decsding?

---

## Appendix: Tinystories

---

### Loss Geometry of S-FLM(+ada) on Tinystories, Seq Len: 256, GenPPL: 11.0

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/sfm_ada_lr1e-3/sfm_ada_lr1e-3.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/sfm_ada_lr1e-3/sfm_ada_lr1e-3_log.png" width="400"/>
</div>

---

### Loss Geometry of S-FLM(+trunc) on Tinystories, Seq Len: 256, GenPPL: 11.0

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/sfm_trunc_lr1e-3/sfm_trunc_lr1e-3.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/sfm_trunc_lr1e-3/sfm_trunc_lr1e-3_log.png" width="400"/>
</div>

---

### Loss Geometry of S-FLM(+trunc+ada) on Tinystories, Seq Len: 256, GenPPL: 11.0

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/sfm_ada_trunc_lr1e-3/sfm_ada_trunc_lr1e-3.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/sfm_ada_trunc_lr1e-3/sfm_ada_trunc_lr1e-3_log.png" width="400"/>
</div>

---

### Loss Geometry of E-FLM on Tinystories, Seq Len: 256, GenPPL: 34.6

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/eflm_naive_geo/eflm_naive_geo.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/eflm_naive_geo/eflm_naive_geo_log.png" width="400"/>
</div>

---

### Loss Geometry of H-FLM on Tinystories, Seq Len: 256, GenPPL: 17.7

- Curvature = -1.0

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/hflm_std0.04_pc1.0/hflm_std0.04_pc1.0.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/hflm_std0.04_pc1.0/hflm_std0.04_pc1.0_log.png" width="400"/>
</div>

---

### Loss Geometry of LangFlow(+ada+SC) on Tinystories, Seq Len: 256, GenPPL: 17.6

<div style="display:flex; justify-content:center; align-items:center; gap:20px;">
  <img src="loss_geometry_vis/tinystories/lf_ada_sc_lr1e-3/lf_ada_sc_lr1e-3.png" width="400"/>
  <img src="loss_geometry_vis/tinystories/lf_ada_sc_lr1e-3/lf_ada_sc_lr1e-3_log.png" width="400"/>
</div>

---

---

## EFLM, HFLM + trunc + Ada Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {med 35 / hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{\sqrt{d}})$ (= custom 0.0441)
  - HFLM:
    - ``random``: $\mathcal{N}(0, 4e-4)$ (= custom 0.02)
    - ``custom``: std: {0.01, 0.04}

- Geometry Curvature: 
  - EFLM: No
  - HFLM:
    - {-0.3, -0.5}

---

## Curvature * Init * LR Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge

---

## Curvature * Init * LR Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step

---

## SFLM, EFLM, HFLM + Self Cond Exp: Setting Up

- Data: Sudoku, **48k train / 2k val** per difficulty (seed 42)
  - Difficulties: {med 35 / hard 30}
- Model (DiT, *tiny*): Width **512**, Depth **8**, Heads **8** (~28.6M)

- Model Initialization Choice ($\mathcal{N}(mean, var)$)
  - SFLM, EFLM:
    - ``ngpt``: $\mathcal{N}(0, \frac{1}{\sqrt{d}})$ (= custom 0.0441)
  - HFLM:
    - ``custom``: std: {0.01}

- Geometry Curvature: 
  - EFLM: No
  - HFLM:
    - -0.5

- Self Conditioning
  - {on, off}

---

## Curvature * Init * LR Exp: Setting Up

- Training
  - Training Steps: **20k**, Batch Size: **256**, Max Seq Len: **180**, bf16, EMA 0.9999
  - Optimizer: AdamW
    - LR: {3e‑4, 5e‑4, 1e‑3}
    - Weight Decay: 0.0, Betas: (0.9, 0.999), eps: 1e-8, Gradient Clip: 1.0
  - All use cross entropy loss
  - 3 radom seeds: {1, 2, 3}, take averge

---

## Curvature * Init * LR Exp: Setting Up

- Evaluation
  - Exact-velocity, top_k_v = -1 (avg across vocab), 180 sampling steps
  - Greedy decoding for last sampling step