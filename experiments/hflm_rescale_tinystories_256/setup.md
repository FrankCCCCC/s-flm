# Re-scaled HFLM Sweep

---

## Numerical Example for Length = 12, 27 and, 50

- $\rho_{x} = \rho_{max} \operatorname{tanh}(|| x ||_2 / \rho_{max})$
- $\gamma_{x} = \operatorname{tanh}(\sqrt{|K|} \rho_{x} / 2) / \sqrt{|K|}$, let $K = -1.0$
- For $|| x ||_2 = 12$, $|| y ||_2 = 27$ and, $|| z ||_2 = 50$
  - $\rho_{x} = 9.139$, $\rho_{y} = 11.7363$ and, $\rho_{z} = 11.994$
  - $\gamma_{x} = 0.9997$, $\gamma_{y} = 0.9999$ and, $\gamma_{z} = 0.9999$
- A half word embedding concetrates on boundary

---

## Numerical Example for Length = 12, 27 and, 50, with Re-scaled

- $\rho_{x} = \rho_{max} \operatorname{tanh}(|| x ||_2 / \rho_{max} / \sqrt{d})$, let $d = 768$
- $\gamma_{x} = \operatorname{tanh}(\sqrt{|K|} \rho_{x} / 2) / \sqrt{|K|}$, let $K = -1.0$
- For $|| x ||_2 = 12$, $|| y ||_2 = 27$ and, $|| z ||_2 = 50$, 
  - $\rho_{x} = 0.4328$, $\rho_{y} = 0.9721$ and, $\rho_{z} = 1.7907$
  - $\gamma_{x} = 0.2131$, $\gamma_{y} = 0.4511$ and, $\gamma_{z} = 0.7140$


---

## H-FLM Sweep - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**
  - Word Embdedding Init: 
    - ``random``: $\mathcal{N}(0, 4e-4)$ (= custom 0.02)
    - ``custom``: std: {0.01, 0.04}
- Init noise for diffusion process (``prior_cov``): {2.0, 4.0, 8.0, 10.0, 16.0}
- ``rho_max``: {12}
- Self Cond: off
- Gaussian Curvature: {-1.0}
- Noise Sched: {``log-linear``}

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