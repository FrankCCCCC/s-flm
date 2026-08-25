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

# Angular Movement across Time for Each Geometry

---

## Same Directional Signal, Two Y Scales

Given a diffusion state $z_t \in \mathbb{R}^d$ at timestep $t$ and $k$-th token embedding $X_k \in \mathbb{R}^d$

$$
c(t,k)=\frac{z_t^\top X_k}{\lVert z_t\rVert\,\lVert X_k\rVert},
\qquad
\theta(t,k)=\frac{180}{\pi}\arccos c(t,k).
$$
- Follow S-FLM, but extend to Euclidean and Hyperbolic

---

## Angular Movement across Time for Geometry

![width:750px](image-2.png)



---

#### Fixed Radial
- Sphere: Constant speed

#### Flexible Radial
- Euclidean: Exponentially faster
  - Larger initial noise or longer word embedding $\rightarrow$ smoother slope
- Hyperbolic: Slow at begin and end, fast at the middle (S curve)

---

## Angular Movement across Time with Gaussian Curvature

![width:900px](image-3.png)

---

#### Hyperbolic
- Flatter Gaussian curvature $\rightarrow$ smoother cliff in the middle (slower angular change)

This explain why HFLM with flat curvature with adaptive noise scheduler gets better GenPPL.

---

# Does Flat Curvature Help HFLM?

---

## HFLM on Tinystories (GenPPL)

Average over various ``init`` (std of word embedding init) and ``prior_cov``: var of init Gaussian

<div style="font-size: 0.7em;">

| K (Gaussian curvature) | w/o ada — mean / median | w/ ada — mean / median|
|---|---|---|
| -0.01 | 19.2 / 17.7 (9/9, 0c) | 21.9 / 15.1 (9/9, 0c) |
| -0.1 | 48.1 / 28.3 (9/9, 2c) | 225.3 / 80.0 (8/9, 3c) |
| -0.25 | 33.6 / 33.4 (9/9, 0c) | 21.9 / 19.8 (5/9, 0c) |
| -0.5 | 34.9 / 32.2 (9/9, 0c) | 58.4 / 60.6 (9/9, 1c) |
| -0.75 | 39.0 / 38.6 (9/9, 0c) | — (4/9, 4c) |
</div>

- Flatter, better

---

## HFLM on Tinystories (GenPPL, Single Best Run)

- HFLM + ada + SC (K: 0.01, ``init``: 0.01, ``prior_cov``: 1.0): 10.7
- S-FLM + trunc + Ada + SC: 11.08

---

## Conclusion

- The only advantage of diffusing on the radius is decoding from high to low freq, but not sure if it helps the generative quality
- Hypothesis
  - Diffusion path doesn't matter, but normalizing the word embedding matters
  - Diffusing on the radius might not be helpful (decoding from high to low freq might be a bad idea)

---

## Motivation

- Diffuse with time-invariant, angular-only velocity field with normalized word embdedding
  - Spherical Cauchy Diffusion
  - Diffuse with time-invariant Euclidean bridge, nromalizing all word embedding to a constant sphere with radius $R$
- Achieve learnable decoding order by a learnable space-time dependent noise scheduler, MuLAN?

---

# Euclidean DM with Time Invariance

### Turning a finite-time bridge into an autonomous flow

---

## The Problem: A Non-Autonomous Drift

A Euclidean Brownian bridge conditioned on $X_T = y$ (with $\|y\| = R$) obeys

$$
dX_t = \frac{y - X_t}{T - t}\,dt + dW_t
$$

- The $\dfrac{1}{T-t}$ factor comes from **conditioning on arrival at a finite time** $T$
- The drift is <rd>time-dependent</rd> and <rd>blows up</rd> as $t \to T$

---

## The Idea: An Infinite-Time Clock

Re-parameterize time so the terminal instant $t=T$ maps to $\tau = \infty$:

$$
\tau = -\log\frac{T-t}{T},
\qquad
t = T\left(1 - e^{-\tau}\right)
$$

- This stretches the finite horizon $[0, T)$ onto $[0, \infty)$
- Key relation for the change of variables: $\;dt = (T-t)\,d\tau$

$$
\frac{d \tau}{d t} 
= - \frac{1}{\frac{T-t}{T}} \frac{1}{dt} \frac{T-t}{T}
= \frac{1}{\frac{T-t}{T}} \frac{1}{T}
= \frac{1}{T-t}
$$

---

## The Result: An Autonomous Flow

Substituting the time change, the bridge becomes

$$
\boxed{\,dX_\tau = (y - X_\tau)\,d\tau + \sqrt{T}\,e^{-\tau/2}\,dB_\tau\,}
$$

The velocity field is now <ng>time-invariant</ng>:

$$
\boxed{\,v(X) = y - X\,}
$$

---

## The Price: Vanishing Noise

Autonomy is not free — the noise level decays exponentially:

$$
\sigma(\tau) = \sqrt{T}\,e^{-\tau/2}
$$

- The target $y$ is reached **only as $\tau \to \infty$**
- **Takeaway:** 
  - a Euclidean analogue of moving the terminal target to <ng>infinite algorithmic time</ng>
  - Smooth loss geometry without spherical RFM

---

## Summary: Finite-Time Bridge vs. Infinite-Time Flow

| | <rd>Finite-time bridge</rd> | <ng>Infinite-time flow</ng> |
|---|---|---|
| **Time** | $t \in [0, T)$ | $\tau \in [0, \infty)$ |
| **Drift** | $\dfrac{y - X_t}{T-t}$ (singular) | $y - X_\tau$ (autonomous) |
| **Noise** | constant, $dW_t$ | $\sqrt{T}\,e^{-\tau/2}$ (vanishing) |
| **Target** | reached at finite $T$ | reached as $\tau \to \infty$ |

---

# Spherical Cauchy Diffusion

---

## Poincare disk Brownian Bridge

Recall the Poincare disk brownian bridge

$$
\begin{aligned}
dz_t & = \left( \frac{d-1}{2} \frac{(1-\|z_t\|^2)^2}{\|x - z_t\|^2} (x - z_t) - \frac{d}{4} (1 - \|z_t\|^2) z_t \right) dt + \frac{1-\|z_t\|^2}{2} d\bar{W}_t \\
\end{aligned}
$$

$x$ is a target on the boundary and $z_t$ is an intermediate state at timestep $t$

---

## Poincare disk Brownian Bridge in Polar

Denote the $z_t = r_{z_t} \theta_{z_t}$ in polar coordinate $(r_{z_t}, \theta_{z_t})$

$$
\begin{aligned}
z
&=
r_z\theta_z,
&
r_z
&:=
\|z\|
=
\tanh\left(\frac{\rho_z}{2}\right),
&
\theta_z
&:=
\frac{z}{\|z\|}
\in
\mathbb S^{d-1}.
\end{aligned}
$$

For the polar SDE, define
$$
\alpha_t=\langle x,\theta_{z_t}\rangle,
\qquad
D_t
=
\cosh(\rho_{z_t})
-
\sinh(\rho_{z_t})\alpha_t,
$$
and
$$
A_t
:=
\frac{
\cosh(\rho_{z_t})\alpha_t
-
\sinh(\rho_{z_t})
}{
D_t
}.
$$

---

## Poincare disk Brownian Bridge in Polar

The disk Brownian bridge can be expressed as the dyanmic over radial $d \rho_{z_t}$ and angle $d \theta_{z_t}$

$$
\boxed{
\begin{aligned}
d\rho_{z_t}
&=
dB_t^\rho
+
(d-1)
\left[
\frac12\coth(\rho_{z_t})
+
A_t
\right]dt,
\\
d\theta_{z_t}
&=
\frac{1}{\sinh(\rho_{z_t})}
(I_d-\theta_{z_t}\theta_{z_t}^\top)\,dW_t^{\mathbb S}
\\
&\quad+
\left[
\frac{d-1}{\sinh(\rho_{z_t})D_t}
(x-\alpha_t\theta_{z_t})
-
\frac{d-1}{2\sinh^2(\rho_{z_t})}\theta_{z_t}
\right]dt.
\end{aligned}
}
$$

---

## Radius-as-time spherical Cauchy Posterior

Treats $R$ as an artificial time variable and requires

$$
\theta_{z_R}\sim q_R(\cdot\mid x),
\qquad
z_R=r_{z_R}\theta_{z_R},
\qquad
r_{z_R}=\tanh\left(\frac R2\right).
$$

The Posterior at radial $\rho = R$

$$
\boxed{
\begin{aligned}
q_R(\theta_z\mid x) 
& :=
q(\theta_{z_t}=\theta_z\mid \rho_{z_t}=R,Y_\infty=x)
\\
& =
\left(
\frac{1-r_{z_R}^2}
{1+r_{z_R}^2-2r_{z_R}\langle x,\theta_z\rangle}
\right)^{d-1} \\
\end{aligned}
}
$$

---

## Radius-as-time spherical Cauchy Probability Flow

The time-homogeneous probability-flow ODE is
$$
\boxed{
\frac{d\theta_{z_R}}{dR}
=
(I_d-\theta_{z_R}\theta_{z_R}^\top)x,
\qquad
\theta_{z_0}\sim\operatorname{Uniform}(\mathbb S^{d-1}).
}
$$

---

## Radius-as-time spherical Cauchy Bridge

To construct a continuous-time diffusion with the same marginals, choose $g(R)>0$ and define
$$
\begin{aligned}
s_x(\theta_z,R)
&:=
\nabla_{\mathbb S^{d-1}}\log q_R(\theta_z\mid x)
\\
&=
(d-1)
\frac{\sinh R}{D_x(R,\theta_z)}
(I_d-\theta_z\theta_z^\top)x,
\\
b_x(\theta_z,R)
&:=
(I_d-\theta_z\theta_z^\top)x
+
\frac{g(R)^2}{2}s_x(\theta_z,R).
\end{aligned}
$$
The intrinsic SDE
$$
\boxed{
d\theta_{z_R}
=
b_x(\theta_{z_R},R)\,dR
+
g(R)\,dW_R^{\mathbb S}
}
$$

---

## Radius-as-time Spherical Cauchy Bridge

In ambient Ito coordinates, this is
$$
\boxed{
d\theta_{z_R}
=
\left[
b_x(\theta_{z_R},R)
-
\frac{d-1}{2}g(R)^2\theta_{z_R}
\right]dR
+
g(R)(I_d-\theta_{z_R}\theta_{z_R}^\top)dW_R.
}
$$

---

## Radius-as-time Spherical Cauchy ELBO

Continuous-time path-KL term is
$$
\boxed{
\mathcal L_{\mathrm{SC}}(\theta)
=
\frac12
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
\int_0^{R_{\max}}
\mathbb E_{\theta_{z_R}\sim q_R(\cdot\mid x)}
\left[
\frac{
\|b_x(\theta_{z_R},R)-b_\theta(\theta_{z_R},R)\|^2
}{
g(R)^2
}
\right]dR
\right].
}
$$

---

# Tinystories + {Geometry * Curvature * Self Condition} Experiment

---

## Adv Geometry Baseline - TinyStories Exp Setup

- Data: TinyStories, **475M train / 5M val** (seed 42)
- Model (DiT, *small*): Width **768**, Depth **12**, Heads **12**, Init ``ngpt``: $\mathcal{N}(0, \frac{1}{d})$ (variance)
- Geometries
  - S-FLM {Naive, ada sched, truncation, ada sched + truncation} * {Self Cond: On} + {Naive} * {Self Cond: Off}
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
- Gaussian Curvature: {0.0, -0.01, -0.1, -0.25, -0.5, -0.75}
- Noise Sched: {``log-linear``, ``log-linear-adaptive``}

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



---

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