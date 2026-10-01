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
  }
  section.lead h2 {
    text-align: center;
  }
  section.lead h3 {
    text-align: center;
  }
  section.lead h4 {
    text-align: center;
  }
  h1 {
    color: #3d3d3d;
  }
  h2 {
    color: #3d3d3d;
  }
  h3 {
    color: #3d3d3d;
  }
  r {
      color: red;
  }
  y {
      color: yellow;
  }
  b {
      color: blue;
  }
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
table {
  font-size: 0.58em;
  margin: 0.2em auto;
  max-width: 100%;
}
th, td {
  padding: 1px 4px;
  line-height: 1.05;
}
</style>

# Hyperbolic DLM

#### Sep 25, 2026

---

# Concentrated Word Embedding Angles

---

## S-FLM, LR 1e-3, Spectrum of Normalized Embeddings

![center height:430px Normalized S-FLM embedding spectrum](imgs/naive_ar_tinystories_256/figures/codebook_eigen_dist_m-sfmta_lr-1e-3_sd-1_normalized_logx.png)

Eigenvalue ratio: **322** (6762 / 20.98).

---

## S-FLM, LR 1e-3, Spectrum of Normalized-then-Centered Embeddings

![center height:430px Normalize-then-center S-FLM embedding spectrum](imgs/naive_ar_tinystories_256/figures/codebook_eigen_dist_m-sfmta_lr-1e-3_sd-1_normalized-mean-shift_logx.png)

After centering: eigenvalue ratio **174** (3659 / 20.98).

---

## S-FLM, LR 1e-3, Spectrum of Centered-then-Normalized Embeddings

![center height:430px Mean-shift-normalized S-FLM embedding spectrum](imgs/naive_ar_tinystories_256/figures/codebook_eigen_dist_m-sfmta_lr-1e-3_sd-1_mean-shift-normalized_logx.png)

Eigenvalue ratio: **635** (9729 / 15.33).

---

## E-FLM, LR 3e-4, Spectrum of Normalized Embeddings

![center height:430px Normalized E-FLM embedding spectrum](imgs/eflm_rescale_auto_tinystories_256/codebook_eigen_dist_eflmratr_lr-3e-4_r-1_m-1.0_normalized_logx.png)

Eigenvalue ratio: **354** (8823 / 24.95); S-FLM: 322.

---

## E-FLM, LR 3e-4, Spectrum of Normalized-then-Centered Embeddings

![center height:430px Normalize-then-center E-FLM embedding spectrum](imgs/eflm_rescale_auto_tinystories_256/codebook_eigen_dist_eflmratr_lr-3e-4_r-1_m-1.0_normalized-mean-shift_logx.png)

Eigenvalue ratio: **186** (4634 / 24.95); S-FLM: 174.

---

## E-FLM, LR 3e-4, Spectrum of Centered-then-Normalized Embeddings

![center height:430px Mean-shift-normalized E-FLM embedding spectrum](imgs/eflm_rescale_auto_tinystories_256/codebook_eigen_dist_eflmratr_lr-3e-4_r-1_m-1.0_mean-shift-normalized_logx.png)

Eigenvalue ratio: **285** (6758 / 23.73); S-FLM: 635.

---

<!-- ## Random Decoding by Softmax

Let $E\in\mathbb R^{V\times d}$ have unit rows $e_v^\top$.
A generated direction $x\in\mathbb S^{d-1}$ has density $q(x)$.

For a softmax decoder with temperature $\tau>0$ and no bias:

$$
p(v\mid x)=\operatorname{softmax}(Ex/\tau)_v.
$$

Sampling $v\sim p(\cdot\mid x)$ gives the word marginal

$$
\boxed{p_{\mathrm{gen}}(v)=\int_{\mathbb S^{d-1}}p(v\mid x)q(x)\,dS(x).}
$$

Training aims for $p_{\mathrm{gen}}(v)\approx q_{\mathrm{data}}(v)$, the data frequency.

--- -->

## Hard Decoding by Argmax

With nearest-embedding decoding, each word owns a spherical Voronoi cell:

$$
v(x) = \arg\max_j e_j^\top x,\qquad
C_v = \{x:e_v^\top x\ge e_j^\top x\ \forall j\}.
$$

$$
\boxed{
p_{\mathrm{gen}}(v)=\int_{C_v}q(x)\,dS(x) 
= A_v\bar q_v.
}
$$

$$
A_v:=\operatorname{Area}(C_v),\qquad
\bar q_v:=\frac1{A_v}\int_{C_v}q(x)\,dS(x).
$$

Uniform latent density gives $p_{\mathrm{gen}}(v)=A_v/|\mathbb S^{d-1}|$.

---

## Word Frequencies and Decoding Geometry

To match a skewed data distribution, $p_{\mathrm{gen}}(v)=A_v\bar q_v\approx q_{\mathrm{data}}(v)$:

1. With **uniform latent density** $q(x)$, rare words need smaller decoding cells than frequent words.
2. With **nonuniform latent density** $q(x)$, frequencies depend on both cell area and average density inside the cell.

---

## Word Frequencies v.s. Proj on Major Component

**Hypothesis:** rare-word embeddings cluster, giving them small cells.

---

### S-FLM, LR 1e-3, Word Freq v.s. Proj on Major Component, Normalized Embedding

![width:700px](imgs/naive_ar_tinystories_256/figures/codebook_eigen_proj_scatter_m-sfmta_lr-1e-3_sd-1_normalized.png)

---

### S-FLM, LR 1e-3, Word Freq v.s. Proj on Major Component, Normalized-Mean-Shift Embedding

![width:700px](imgs/naive_ar_tinystories_256/figures/codebook_eigen_proj_scatter_m-sfmta_lr-1e-3_sd-1_normalized-mean-shift.png)

---

### E-FLM, LR 3e-4, Word Freq v.s. Proj on Major Component, Normalized Embedding

![width:700px](imgs/eflm_rescale_auto_tinystories_256/codebook_eigen_proj_scatter_eflmratr_lr-3e-4_r-1_m-1.0_normalized.png)

---

### E-FLM, LR 3e-4, Word Freq v.s. Proj on Major Component, Normalized-Mean-Shift Embedding

![width:700px](imgs/eflm_rescale_auto_tinystories_256/codebook_eigen_proj_scatter_eflmratr_lr-3e-4_r-1_m-1.0_normalized-mean-shift.png)

---

## Frequency Distortion

The word embedding encodes the word frequency information, causing rare words cluster. It distorts the semantic angle with frequency . 

This is also known as frequency distortion.

---

## Hypotheses for the Skewed Spectrum

1. **LayerNorm hypothesis:** bias induced by LayerNorm.

2. **Geometric hypothesis:** skewed token frequencies and large centroid distance induce an ill-conditioned Fisher information matrix (FIM) of the KL terms in the ELBO, also causing high ELBO variance.

We will provide the proof of 2.

<!--
LayerNorm definition: Ba, Kiros and Hinton (2016), https://arxiv.org/abs/1607.06450 . The connection to this particular codebook is a hypothesis; the architecture and embedding tying must be checked before attributing its spectrum to a hidden-state normalization layer.
Related evidence, not a diagnosis of this model: Yu et al. (2022), "Rare Tokens Degenerate All Tokens", https://aclanthology.org/2022.acl-long.3/ .
Anisotropy is not universally forced by Transformer architecture: Machina and Mercer (2024), "Anisotropy is Not Inherent to Transformers", https://aclanthology.org/2024.naacl-long.274/ .
-->

---

## Discrete-Time Bridge NELBO

Now $x\sim q_{\mathrm{data}}(x)$ is a word; for $s<t$, Bayes and Markov give

$$
\boxed{
q(z_t\mid z_s,x)
=q(z_t\mid z_s)\frac{q(x\mid z_t)}{q(x\mid z_s)}.
}
$$

$q(x\mid z_t)$: the **word posterior**, a probability over the $V$ words.

With target bridge $Q_{t:s}:=q(z_t \mid z_s,x)$ and learned transition $P_{t:s}:=p_\theta(z_t \mid z_s)$, the **NELBO** over $T$ steps is

$$
\mathcal L
=\sum_{i=0}^{T-1}\mathbb E_{x\sim q_{\mathrm{data}}(x),\,z_i\sim q(\cdot\mid x)}
\left[\mathrm{KL}(Q_{i+1:i} \| P_{i+1:i})\right].
$$

---

## Polar Decomposition of the Bridge

The hyperbolic bridge starts from the origin $o$, so its free transition is rotation-invariant:

$$
q(z_t \mid z_s = o) = q(\rho_t)\, q(\omega_t), \qquad q(\omega_t) = \frac{1}{| \mathbb{S}^{d-1}|},
$$

where $q(\rho_t)$ is the radial law of the free hyperbolic heat kernel.

With a uniform boundary prior, $q(x \mid z_s=o) = \frac{1}{| \mathbb{S}^{d-1}|}$, so the bridge from the origin factorizes in polar coordinates:

$$
q(z_t \mid z_s=o, x)
= q(\rho_t)\, q(x \mid \rho_t, \omega_t).
$$

---

## Polar Decomposition of the Transition KL

Write $z_t=\tanh(\kappa\rho_t/2)\omega_t$, with $\kappa=\sqrt{|K|}$. At fixed $(z_s,x)$, factor each transition into a radial law and a word posterior:

$$
\begin{aligned}
Q_t^{\parallel}(\rho) &= q(\rho_t=\rho), \quad Q_t^{\perp}(x \mid \rho, \omega) = q(x \mid \rho_t=\rho, \omega_t=\omega), \\
P_t^{\parallel}(\rho) &= q(\rho_t=\rho), \quad P_t^{\perp}(x \mid \rho, \omega) = p_{\theta}(x \mid \rho_t=\rho, \omega_t=\omega).
\end{aligned}
$$

The transition KL then splits as

$$
\boxed{
\begin{aligned}
\mathrm{KL}(Q_t\|P_t)
&=\underbrace{\mathrm{KL}(Q_t^{\parallel} \| P_t^{\parallel})}_{\ell_{\mathrm{rad}}}\\
&\quad+\underbrace{\mathbb E_{(\rho,\omega)\sim Q_t}
\left[\mathrm{KL} \left(Q_t^{\perp}(\cdot\mid\rho,\omega)
\|P_t^{\perp}(\cdot\mid\rho,\omega)\right)\right]}_{\ell_{\mathrm{ang}}}.
\end{aligned}
}
$$

Averaging gives $\mathcal L=\mathcal L_{\mathrm{rad}}+\mathcal L_{\mathrm{ang}}$.

---

## Matching Radial Laws Removes the Radial KL

$$
Q_t^{\parallel}=P_t^{\parallel}
\quad\Longrightarrow\quad
\ell_{\mathrm{rad}}=0,\qquad
\mathrm{KL}(Q_t\|P_t)=\ell_{\mathrm{ang}}.
$$

The following Fisher calculation studies the word-posterior KL

$$
\mathrm{KL}\!\left(q(\cdot\mid z_t)\|p_\theta(\cdot\mid z_t)\right)
$$

under a small angular shift of $z_t$.

---

## Data Weighted Posterior

Let $v \in \mathbb S^{d-1}$ be word embeddings and $q_{\mathrm{data}}(v)$ the data distribution.

$$
z_t = r_t \omega_t,\qquad
r_t=\tanh\!\left(\frac{\kappa\rho_t}{2}\right),\qquad
P_{\omega_t}=I-\omega_t\omega_t^\top.
$$

The Poisson kernel is $D_\kappa^{-(d-1)}$, where 

$$
\begin{aligned}
D_\kappa(x,\rho_t,\omega_t)
&:=\cosh(\kappa\rho_t)-\sinh(\kappa\rho_t)\omega_t^\top x,\\
\end{aligned}
$$

and the posterior $q(x = \bar{v} \mid z_t)$ on word $\bar{v}$ is defined as

$$
\begin{aligned}
q(x = \bar{v} \mid z_t)
&:=\frac{q_{\mathrm{data}}(\bar{v})D_\kappa(\bar{v},\rho_t,\omega_t)^{-(d-1)}}
{\sum_v q_{\mathrm{data}}(v)D_\kappa(v,\rho_t,\omega_t)^{-(d-1)}}.
\end{aligned}
$$

---

## Spherical Score

Differentiate $\log D_\kappa$ on the sphere:

$$
w_\kappa(x):=\nabla_{\omega_t}^{\mathbb{S}}\log D_\kappa(x)
=-\frac{\sinh(\kappa\rho_t)}{D_\kappa(x)}P_{\omega_t}x.
$$

The posterior score subtracts its own posterior mean:

$$
\boxed{
s_t(x) := \nabla_{\omega_t}^{\mathbb S}\log q(x\mid z_t)
=-(d-1)\left(w_\kappa(x)-\mathbb E_{q(v\mid z_t)}[w_\kappa(v)]\right).
}
$$

---

## The Angular Fisher Matrix

Locally model the learned posterior as the true posterior at a shifted angle $\operatorname{Exp}_{\omega_t}d\omega$, with $d\omega\perp\omega_t$ and fixed radius:

$$
\mathrm{KL}\!\left(q(\cdot\mid\rho_t,\omega_t)\,\|\,
q(\cdot\mid\rho_t,\operatorname{Exp}_{\omega_t}d\omega)\right)
=\tfrac12d\omega^\top F(\rho_t,\omega_t)d\omega+o(\|d\omega\|^2).
$$

$$
F(\rho_t,\omega_t)
=\mathbb E_{q(x\mid z_t)}[s_t(x)s_t(x)^\top]
=(d-1)^2\operatorname{Cov}_{q(x\mid z_t)}[w_\kappa(x)].
$$

---

## Spherical Score Near the Origin

At $z_t=o$, $D_\kappa=1$ and $q(x=\bar{v} \mid z_t=o)=q_{\mathrm{data}}(\bar{v})$. 

For $\kappa\rho_t\ll1$, let $\mu=\mathbb E_{q_{\mathrm{data}}(v)}[v]$ and $r_t = \operatorname{tanh}(\frac{\kappa \rho_t}{2}) \approx \frac{\kappa \rho_t}{2}$. Take the Euclidean derivative first, then project onto the sphere:

$$
\begin{aligned}
s_t(x)
&= \nabla_{\omega_t}^{\mathbb S}\log q(x\mid z_t)
 = r_t\,P_{\omega_t}\nabla_{z_t}\log q(x\mid z_t) \\
&\simeq r_t\,P_{\omega_t}\cdot 2(d-1)(x-\mu)
 \simeq (d-1)\kappa\rho_t\,P_{\omega_t}(x-\mu),
\end{aligned}
$$

<!-- Since $r_t = 0$ at origin, there is no displacement if taking $\nabla_{\omega_t}^{\mathbb{S}}$, we take Euclidean gradient at origin

$$
\begin{aligned}
s_t(x) = \nabla_{z_t}\log q(x\mid z_t) = (d-1)(x-\mu)
\end{aligned}
$$ -->

---

## The Angular Fisher Matrix Near the Origin

With $\Sigma=\operatorname{Cov}_{q_{\mathrm{data}}(v)}[v]$, near the origin:

$$
\boxed{
F(\rho_t, \omega_t)\simeq
(d-1)^2 \kappa^2 \rho_t^2 P_{\omega_t}\Sigma P_{\omega_t}
\simeq4(d-1)^2 r_t^2 P_{\omega_t}\Sigma P_{\omega_t}.
}
$$

When $q_{\mathrm{data}}(v)$ is long-tailed and $\omega_t$ lies along a principal direction, $P_{\omega_t}\Sigma P_{\omega_t}$ is ill-conditioned, which amplifies the KL cost along the leading directions.

---

## The Lower Bound of the Condition Number of Data Distribution Covariance $\Sigma$

Let $v_1$ be the most frequent word in the vocabulary. Then

$$
\operatorname{cond}(\Sigma) \geq \frac{(d-1)\, q_{\mathrm{data}}(v_1)\, \| v_1 - \mu \|^2}{1 - q_{\mathrm{data}}(v_1)}.
$$

Therefore, if $q_{\mathrm{data}}(v)$ is long-tailed and the most frequent word $v_1$ is far from the embedding mean $\mu$, then $\Sigma$ is ill-conditioned.

---

## The Lower Bound of the Condition Number of Posterior Covariance $\Sigma_{t}$

<style scoped>section p, section > mjx-container { font-size: 25px; }</style>

At any state $z_t$ the angular Fisher is exactly a posterior covariance:

$$
\begin{aligned}
F(\rho_t,\omega_t)&=(d-1)^2\sinh^2(\kappa\rho_t)\,\Sigma_t,\qquad
\tilde v_t:=P_{\omega_t}v/D_\kappa(v,\rho_t,\omega_t),\\
\Sigma_t&:=\operatorname{Cov}_{q(v\mid z_t)}[\tilde v_t],\qquad
\mu_t:=\mathbb E_{q(v\mid z_t)}[\tilde v_t].
\end{aligned}
$$

For the posterior mode $v_1^{(t)}$ and $d\ge3$, on the tangent space:

$$
\boxed{
\operatorname{cond}(\Sigma_t)
\ \ge\ \frac{(d-2)\,q(v_1^{(t)}\mid z_t)\,\|\tilde v_1^{(t)}-\mu_t\|^2}
{\operatorname{tr}\Sigma_t-q(v_1^{(t)}\mid z_t)\,\|\tilde v_1^{(t)}-\mu_t\|^2}.
}
$$

**Posterior skew and centroid distance control the conditioning.**

Prior frequencies enter only near the origin; far away the posterior collapses onto one word, $F\to0$, and the bound says nothing.

---

## Hyperbolic Geometry Hypothesis for Word Freq and Semantics

1. ELBO Var <-> Curvature & Rotation <-> Word Frequency
2. ELBO Expectation <-> Angle <-> Word Semantic

---

## Oval Geometry Can Whiten the Local Fisher

Near the origin $s_t(x)\simeq\rho_t P_{\omega_t}s_0(x)$ with $s_0(x)=(d-1)\kappa(x-\mu)$ and $F_0=(d-1)^2|K|\Sigma$. Model a local bridge deformation by replacing $\kappa I$ with $A$:

$$
s_A(v)=(d-1)A(v-\mu),\qquad F_A=(d-1)^2A\Sigma A^\top.
$$

For $\Sigma\succ0$, preserve $\operatorname{tr}F_A=\operatorname{tr}F_0$ and choose

$$
\boxed{
A^\star=\sqrt{|K|\frac{\operatorname{tr}\Sigma}{d}}\,\Sigma^{-1/2},
\qquad
F_{A^\star}=cI,\quad
c=(d-1)^2|K|\frac{\operatorname{tr}\Sigma}{d}.
}
$$

Thus the leading angular Fisher becomes $F_{\mathrm{ang},A^\star}\simeq c\rho_t^2P_{\omega_t}$:
**equal KL cost in every tangent direction.**

If $\Sigma$ is singular, whiten its support or use regularization.

---

## Next Step

1. Try $A^\star=\sqrt{|K|\frac{\operatorname{tr}\Sigma}{d}}\,\Sigma^{-1/2}$ to see if it can reduce the ELBO variance
2. Design a method to let model learn the oval geometry, it requires both scale and rotation