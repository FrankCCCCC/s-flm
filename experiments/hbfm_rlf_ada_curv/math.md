# HBFM with Adaptive (Per-Position) Curvature

Notation follows `slides/aug26_2026` and `slides/sep25_2026`: curvature $K<0$,
$\kappa=\sqrt{-K}$, radius $R$, direction $\theta$, dimensionless radius $u=\kappa R$ and
time $\tau=\kappa^2 t$, $D_x(u,\theta)=\cosh u-\sinh u\,\langle x,\theta\rangle$.
References: VDM (Kingma et al., 2021) §5.1 and App. I.2; MuLAN (Sahoo et al., 2024) §3.

**Result in one line.** Giving position $i$ the curvature $K_i=\lambda_iK$ is the same as
running the base bridge on the position's own clock $\lambda_i t$. Weighting its CE by
$\lambda_i$ keeps the loss mean unchanged, so $\lambda_i$ only moves the variance, and the
sampler advances position $i$ by $\lambda_i\,dt$.

---

## 0. Setting

- Sequence $y=(y_1,\dots,y_L)$, word $y_i$ with boundary point $x_i=e_{y_i}$,
  $e_v=(e_{v,m})_{m\le M}$, $e_{v,m}\in\mathbb S^{d_m-1}$.
- Every position lives on the base product manifold
  $\mathcal M=\prod_{m=1}^M\mathbb H^{d_m}_{K_m}$, $K_m=-\kappa_m^2$.
- Forward process given $y$: every position runs an independent Brownian bridge from the
  origin to $x_i$. $z_t=(z_t^1,\dots,z_t^L)$ is the sequence state at physical heat time $t$.
- Model: $p^i_{\theta,v}(z_t):=p_\theta(Y^i_\infty=e_v\mid z_t)$, the posterior of position
  $i$ given the **whole** state.
- HBFM objective (`algo.HyperbolicBoundaryFM`): the heat-time integral of the denoising CE,
  estimated with one $t\sim q(t)$ per sequence and the weight $1/q(t)$,

$$
\mathcal L=\mathbb E_y\int_0^T\mathbb E_{z_t\sim q_t(\cdot\mid y)}
\Big[\sum_{i=1}^L\mathrm{CE}_i(z_t)\Big]dt,
\qquad \mathrm{CE}_i(z_t):=-\log p^i_{\theta,y_i}(z_t).
$$

**Adaptive curvature.** Position $i$ gets its own curvature for all of its factors,

$$
K_{i,m}=\lambda_iK_m,\qquad \kappa_{i,m}^2=\lambda_i\kappa_m^2,\qquad \lambda_i>0 ,
$$

with $\lambda_i$ the *curvature scale* ($\lambda_i\equiv1$ is the base model).

---

## 1. Curvature is a clock

From the slides ("When Is the Angular Path-KL Curvature-Invariant?"), with $u=\kappa R$,
$\tau=\kappa^2t$:

$$
p_t^{\mathrm{rad},\kappa}(R)\,dR=\widetilde p_\tau^{\mathrm{rad}}(u)\,du,\qquad
q_R^{(\kappa)}(\theta\mid x)=q_u(\theta\mid x).
$$

The law of the dimensionless state $(u,\theta)$ depends on $(\kappa,t)$ only through
$\tau=\kappa^2t$. For position $i$, $\tau_{i,m}=\lambda_i\kappa_m^2t=\kappa_m^2(\lambda_it)$:

$$
\boxed{\;
\text{state at curvature }\lambda_iK\text{ and time }t
\;\overset{d}{=}\;
\text{state at base curvature }K\text{ and time } s_i:=\lambda_i t
\quad\text{(dimensionless coordinates).}\;}
$$

So every position stays in the base ball and adaptive curvature is a per-position clock
$s_i=\lambda_it$: flat positions ($\lambda_i<1$) diffuse slowly, sharp ones fast.

**The target does not change.** The bridge marginal is (free radial law) $\times$
(Poisson kernel), and the free part does not depend on the word, so

$$
p(y\mid z)\;\propto\;p_{\mathrm{data}}(y)\prod_{i,m}D_{e_{y_i,m}}(u_{i,m},\theta_{i,m})^{-(d_m-1)}
$$

contains neither $t$ nor $\lambda$. The Bayes posterior is a function of the dimensionless
state alone: one predictor serves every curvature (the slides' scale-equivariant predictor),
and the horosphere readout is untouched.

---

## 2. Loss: weight the CE of position $i$ by $\lambda_i$

The KL rate of one factor carries one factor $\kappa^2$ (slides, "Full Product KL Rate Adds
Across Factors"). Positions are independent factors of one large product manifold, so

$$
\ell(z_t,y)=\frac12\sum_{i}\sum_m(d_m-1)^2\,\lambda_i\kappa_m^2
\big\|w_{x_{i,m}}-\overline w^{\,i}_{\theta,m}\big\|^2
=\sum_i\lambda_i\,\ell^i(z_t),
$$

with $\ell^i$ the **base-curvature** rate of position $i$. HBFM trains on the denoising CE in
place of $\ell^i$ (the slides' CE bound; `algo.py` uses the plain CE), which gives the
adaptive-curvature objective

$$
\boxed{\;
\mathcal L(\lambda)=\mathbb E_y\int_0^T
\mathbb E_{z_t^i\sim q_{\lambda_it}(\cdot\mid x_i)}\Big[\sum_i\lambda_i\,\mathrm{CE}_i(z_t)\Big]dt,
\qquad
\widehat{\mathcal L}=\frac1{q(t)}\sum_i\lambda_i\,\mathrm{CE}_i(z_t),\quad t\sim q .\;}
$$

**The weight compensates the change of density.** Under $t\sim q$ the clock of position
$i$, $s_i=\lambda_it$, has density $q_i(s)=q(s/\lambda_i)/\lambda_i$. The weight of position
$i$ is exactly the importance weight of its own clock:

$$
\frac{\lambda_i}{q(t)}=\frac1{q_i(s_i)} .
$$

---

## 3. The mean does not depend on the curvature

Let $F_i(s_1,\dots,s_L):=\mathbb E\,[\mathrm{CE}_i(z)]$ with $z^j$ the base bridge at time
$s_j$. Then $\mathcal L$ is a line integral along the ray $\gamma_\lambda(t)=t\lambda$ in
clock space (MuLAN §3.5):

$$
\mathcal L(\lambda)=\int_0^T\sum_i\lambda_iF_i(\lambda t)\,dt=\int_{\gamma_\lambda}F\cdot ds .
$$

**(a) Per position.** If $F_i$ depends on its own clock only (context-free posterior),

$$
\int_0^T\lambda_iF_i(\lambda_it)\,dt=\int_0^{\lambda_iT}F_i(s)\,ds
\;\xrightarrow{T\to\infty}\;\int_0^\infty F_i(s)\,ds ,
$$

independent of $\lambda_i$. Without the weight the CE sum would scale as $1/\lambda_i$.

**(b) Global scale.** $\lambda\to c\lambda$ re-parametrizes the same ray, so
$\mathcal L(c\lambda)=\mathcal L(\lambda)$ for any posterior (the slides' "Curvature Cancels
in the Path-KL if Horizon Scales Properly").

**(c) General case.** Integrating by parts along the ray ($tF_i\to0$),

$$
\frac{\partial\mathcal L}{\partial\lambda_i}
=\int_0^\infty t\sum_j\lambda_j\big(\partial_iF_j-\partial_jF_i\big)(\lambda t)\,dt .
$$

The mean is stationary in $\lambda$ whenever $F$ is curl-free. What is left is the
cross-position coupling of an imperfect model — MuLAN's path dependence.

**(d) Full path-KL.** Let $Q^y_\lambda$ be the bridge path law and
$P_\lambda=\sum_yp_{\mathrm{data}}(y)Q^y_\lambda$ the learned bridge with the exact
posterior. For any model $P_{\theta,\lambda}$ with the posterior-mixture drift,

$$
\mathbb E_y\,\mathrm{KL}(Q_\lambda^y\|P_{\theta,\lambda})
=\underbrace{I_\lambda\big(Y;z_{[0,T]}\big)}_{\to\,H(Y)\ \text{for every }\lambda}
+\underbrace{\mathrm{KL}(P_\lambda\|P_{\theta,\lambda})}_{\text{model error}} .
$$

At the optimum the ELBO is $H(Y)$ for every curvature; $\lambda$ enters the mean only
through how the model error is weighted along the path.

**Finite horizon.** With a bounded proposal the limits are $\lambda_iT$ (slides: "Fixed
physical horizons: not invariant"): the clock of position $i$ is trained on $[0,\lambda_iT]$.

---

## 4. The variance does

In the per-position view, with $G_i(s):=\mathbb E[\mathrm{CE}_i^2\mid s_i=s]$,

$$
\mathbb E\Big[\Big(\tfrac{\lambda_i\mathrm{CE}_i}{q(t)}\Big)^2\Big]
=\int_0^\infty\frac{G_i(s)}{q_i(s)}\,ds,\qquad q_i(s)=\frac{q(s/\lambda_i)}{\lambda_i} .
$$

The curvature picks the proposal of position $i$ out of the scale family of $q$; the best
one is $q_i\propto\mathrm{CE}_i$. For $q(t)=re^{-rt}$ and a profile
$\mathrm{CE}_i(s)=c\,e^{-as}$:

$$
M_i(\lambda_i)=\frac{c^2\lambda_i^2}{r\,(2a\lambda_i-r)}
\quad\Big(\lambda_i>\frac r{2a}\Big),\qquad
\lambda_i^\star=\frac ra,\qquad M_i(\lambda_i^\star)=\Big(\frac ca\Big)^2=\text{mean}^2 .
$$

- Too flat, $\lambda_i\le r/2a$: infinite variance (the *cliff* of the unigram report).
- Too sharp: $M_i\simeq c^2\lambda_i/2ar$, linear in $|K|$ (also the unigram report).

Measured on the real model (`RESULTS.md` §1): variance 0.31 at $\lambda=1/2$, rising to 0.73,
1.70, 3.71 at $\lambda=1,2,4$ and to 0.60 at $\lambda=1/4$, with the mean fixed at 0.727.

Extreme curvature in either direction is a variance blow-up, hence a **base curvature and a
guarded range**:

$$
\boxed{\;K_{i,m}=\lambda_iK_m,\qquad \lambda_i=b^{\tanh h_i}\in[1/b,\;b],\;}
$$

$h_i$ the model output, $b\ge1$ the bound (`algo.curvature_bound`; $b=1$ is the base
model). A zero-initialised $h$ starts training exactly at the base curvature.

---

## 5. Where $\lambda$ comes from

The forward bridge and the generative SDE must use the same $\lambda_i$ on the whole path:
they need the same diffusion coefficient or the path-KL is infinite. So $\lambda_i$ may
depend only on what both know from $t=0$ on — the context $c$ (the prompt; MuLAN §3.3.1) —
not on $y$ (MuLAN Suppl. D.2: no continuous-time limit) and not on $z_t$ (the forward
marginal would no longer be closed-form).

The hyperbolic DiT gets a second readout next to its $V$ logits: one curvature output $h_i$
per position. It is read at the $t=0$ state

$$
z_0(c)=\big(\text{origin at every free position, the prompt pinned at its boundary point}\big),
\qquad \lambda(c)=b^{\tanh h(z_0(c))}\in[1/b,b]^L .
$$

Training and sampling read $\lambda$ from the same state, so they use the same curvature.
(Without a prompt every position sees the same input and $\lambda$ is one learned global
scale; a per-position unconditional $\lambda$ needs MuLAN's auxiliary latent.)

---

## 6. Learning $\lambda$: minimise the variance (VDM App. I.2)

$\mathbb E[\widehat{\mathcal L}]$ carries no signal for $\lambda$ (§3), so $\lambda$ is
trained on the variance, as VDM trains its noise schedule:

$$
\nabla_\lambda\mathrm{Var}[\widehat{\mathcal L}]
=\nabla_\lambda\mathbb E[\widehat{\mathcal L}^2]
=\mathbb E\big[2\,\widehat{\mathcal L}\,\nabla_\lambda\widehat{\mathcal L}\big].
$$

$\nabla_\lambda\widehat{\mathcal L}$ is a pathwise derivative: $\lambda_i$ enters through the
weight and through the bridge state, which is a deterministic function of
$(\lambda_it,\text{noise})$. For an $\mathbb H^3$ factor at unit curvature

$$
R=\big\|\sqrt s\,g+s\,e_1\big\|,\quad g\sim\mathcal N(0,I_3)
\qquad\Longrightarrow\qquad
p(R)\propto R\,\sinh R\;e^{-R^2/2s},
$$

the norm of a 3-d Brownian motion with unit drift, i.e. a noncentral $\chi_3$ (curvature $K$:
$R\to R/\kappa$, $s\to\kappa^2s$). The direction is the Lorentz boost of a uniform direction,
reflected to $x_i$ — already deterministic in $R$. Factors of another dimension have no such
closed form and fall back to the quadrature grid of the fixed-curvature model.

One backward pass does both jobs: the gradient that reaches $\lambda$ is multiplied by
$2\widehat{\mathcal L}_b$ (the sequence's own loss), everything else receives
$\nabla\widehat{\mathcal L}$.

**Two sets of parameters, as in VDM.** The variance gradient is a fourth-moment quantity of a
$1/q(t)$-weighted loss: heavy-tailed, with a sign that flips from batch to batch near the
optimum. Two guards keep it from disturbing the denoiser (both measured, `RESULTS.md` §2):

- The curvature readout sees the trunk's features **detached**. The trunk and the logits are
  trained on the loss only, the curvature readout on its variance only.
- $h_i$ is the readout divided by $\sqrt D$ ($D$ hidden units), the scale of one unit, and
  the gradient into the readout carries the same $1/\sqrt D$. A plain sum moves
  $\log\lambda$ by $O(\text{lr}\cdot D)$ per optimizer step and follows the sign of every
  batch; the gradient scale keeps the variance gradient below the gradient-clip threshold of
  the rest of the model (Adam is invariant to it).

**Check (score form).** A global scale $\lambda$ under $q(t)=re^{-rt}$ can also be
differentiated by re-weighting instead of re-sampling:

$$
\frac{\partial\,\mathbb E[\widehat{\mathcal L}^2]}{\partial\log\lambda}
=\mathbb E\big[\widehat{\mathcal L}^2\,(1+t\,\partial_t\log q(t))\big]
=\mathbb E\big[\widehat{\mathcal L}^2\,(1-rt)\big].
$$

The optimum puts the squared-loss-weighted mean heat time at the proposal's mean $1/r$.
The two estimators must agree (tested).

---

## 7. Sampler

The bridge SDE of the slides, written for position $i$, factor $m$ in unit-ball coordinates
$w=z/R_m$ and with the learned posterior mixture:

$$
dw=\lambda_i\kappa_m^2\Big[\tfrac{d_m-1}2(1-\|w\|^2)^2\sum_vp^i_{\theta,v}(z_t)
\frac{e_{v,m}-w}{\|e_{v,m}-w\|^2}-\tfrac{d_m}4(1-\|w\|^2)\,w\Big]dt
+\sqrt{\lambda_i}\,\kappa_m\frac{1-\|w\|^2}2\,dW_t .
$$

Drift $\times\lambda_i$, noise $\times\sqrt{\lambda_i}$: the base SDE with $dt\to\lambda_i\,dt$.

$$
\boxed{\;\text{Euler–Maruyama: position }i\text{ takes the base step of size }
\lambda_i\,\Delta t,\qquad \lambda=\lambda(z_0)\text{ read once at }t=0 .\;}
$$

**Exact for every $\lambda$.** $h(z)=\sum_yp_{\mathrm{data}}(y)\prod_{i,m}P(z^{i,m},e_{y_i,m})$
is harmonic for $\sum_i\lambda_i\sum_m\kappa_m^2\Delta_{i,m}$ whatever the $\lambda_i$,
because every Poisson kernel is harmonic in its own factor. With the exact posterior the
decoded law is $p_{\mathrm{data}}$ for any per-position curvature (tested with a Bayes
oracle); $\lambda$ only changes the order and speed at which positions commit.

Position $i$ reaches the base heat time $\lambda_it_{\max}\ge t_{\max}/b$ with steps
$\lambda_i\Delta t\le b\,\Delta t$.

---

## 8. Implementation map

| math | code |
|---|---|
| $h_i$, curvature readout on the detached trunk | `models/hyperbolic_dit.py`: `curvature_layer`, `HyperbolicDiT.curvature` |
| $\lambda=b^{\tanh h(z_0)}$, $h$ = readout $/\sqrt D$ | `HyperbolicBoundaryFM.curvature_scale` |
| $z_t^i\sim q_{\lambda_it}(\cdot\mid x_i)$ | `q_xt(..., curvature=λ)`: one heat time per position |
| $R=\|\sqrt s\,g+se_1\|$ | `HyperbolicHeatKernel._sample_radial_exact_d3` |
| weight $\lambda_i/q(t)$ | `q_xt` returns `weight * curvature` |
| $2\widehat{\mathcal L}_b\nabla_\lambda\widehat{\mathcal L}_b$ (times $1/\sqrt D$) | gradient hook on `curvature` in `nll` |
| step $\lambda_i\Delta t$ | `HBFMSampler.step`: `dt * state.curvature` |

## 9. Verification

`tests/test_hbfm_ada_curv.py` checks every boxed statement; the numbers are in `RESULTS.md` §1.

| section | check | result |
|---|---|---|
| 1 | `q_xt` at curvature $\lambda$ vs the fixed-curvature bridge at heat time $\lambda t$ | same law (KS on radius and angle) |
| 3 | mean of $\widehat{\mathcal L}$ at $\lambda\in\{1/4,1/2,1,2,4,\text{mixed}\}$ | 0.7225–0.7310, all within 1.3 s.e. of the fixed-curvature 0.7269 |
| 4 | variance of $\widehat{\mathcal L}$ over the same $\lambda$ | 0.31–3.71, U-shaped |
| 6 | pathwise gradient through the hook vs the score form | 0.972 vs 0.964 at $\lambda=1$; 2.010 vs 2.023 at $\lambda=2$ |
| 6 | closed-form $\mathbb H^3$ radial sampler | matches the closed-form CDF and the grid sampler (KS) |
| 7 | SDE with steps $\lambda_i\Delta t$ vs the exact bridge at $\lambda_it$ | same law (KS) |
| 7 | Bayes oracle, $\lambda=(1/2,2)$ | both positions decode the prior ($\chi^2$) |
