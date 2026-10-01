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

#### Aug 26, 2026

---

- Normalized word embeddings concentrate on a pole

![alt text](image-4.png)

---

## GenPPL - Entropy Frontier Lines

- Same settings and temperature config as S-FLM paper

![alt text](image-5.png)

---

$$
\mathbb{E}_{x \sim q} \left[ \nabla_{x}\log q_t(z | x) \right]
$$

---

### IS on Sphere (Angles)

Let $\mathbf{\bar{E}} \in \mathbb{R}^{V} \times \mathbb{S}^{d-1}$ is a normalized word embedding matrix with $V$ word and normalized $d$ dimension. $\mathbf{e}_x \in \mathbb{S}^{d-1}$ is the $x$-th row of the $\mathbf{\bar{E}}$.

$z_t \in \mathbb{R}^d$ is an intermediate state at timestep $t$, the forward process is $q_t(z | x) := \mathcal{N}(z; t \mathbf{e}_x, (1 - t)^2 \mathbf{I})$. 

The EFLM loss is

$$
\mathbb{E}_{z_t \sim q_t(\cdot | x)}\left[ \text{CE}(\mathbf{Y}_x  \| \hat{X}_{\Theta}(z_t, t)) \right]
$$

$\mathbf{Y}_x$ is the one-hot encoding of word $x$.

---

### IS on Sphere (Angles)

To do IS on angular space, we can do 

$$
\mathbb{E}_{z_t \sim \bar{q}_t(\cdot | x)}\left[ \frac{q_t(z_t | x)}{\bar{q}_t(z_t | x)} \text{CE}(\mathbf{Y}_x  \| \hat{X}_{\Theta}(z_t, t)) \right]
$$

where $\bar{q}_t(z | x) := \mathcal{N}(z; t \mathbf{e}_x, (1 - t)^2 \frac{\mathbf{\bar{E}}^{\top} \mathbf{\bar{E}}}{Z})$, where $Z$ is a hyperparameter.

---

### Experiment

See if the variance reduces as we use proposal $\bar{q}_t(z | x)$

---

### Pre-conditioned (Adaptive) Angular Steps



---

# Spherical Cauchy Diffusion

---

## Poincare disk Brownian Bridge

Consider sectional curvature $K<0$, let $\kappa:=\sqrt{-K}$, and use $g_K=\kappa^{-2}g_{-1}$ on the unit ball. Recall the Poincare disk brownian bridge

$$
\begin{aligned}
dz_t & = \kappa^2 \left( \frac{d-1}{2} \frac{(1-\|z_t\|^2)^2}{\|x - z_t\|^2} (x - z_t) - \frac{d}{4} (1 - \|z_t\|^2) z_t \right) dt + \kappa\frac{1-\|z_t\|^2}{2} d\bar{W}_t \\
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
\tanh\left(\frac{\kappa\rho_z}{2}\right),
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
\cosh(\kappa\rho_{z_t})
-
\sinh(\kappa\rho_{z_t})\alpha_t,
$$
and
$$
A_t
:=
\frac{
\cosh(\kappa\rho_{z_t})\alpha_t
-
\sinh(\kappa\rho_{z_t})
}{
D_t
}.
$$

---

## Poincare disk Brownian Bridge in Polar

The disk Brownian bridge can be expressed as the dynamic over radial $d \rho_{z_t}$ and angle $d \theta_{z_t}$

$$
\boxed{
\begin{aligned}
d\rho_{z_t}
&=
dB_t^\rho
+
(d-1)\kappa
\left[
\frac12\coth(\kappa\rho_{z_t})
+
A_t
\right]dt,
\\
d\theta_{z_t}
&=
\frac{\kappa}{\sinh(\kappa\rho_{z_t})}
(I_d-\theta_{z_t}\theta_{z_t}^\top)\,dW_t^{\mathbb S}
\\
&\quad+
\left[
\frac{(d-1)\kappa^2}{\sinh(\kappa\rho_{z_t})D_t}
(x-\alpha_t\theta_{z_t})
-
\frac{(d-1)\kappa^2}{2\sinh^2(\kappa\rho_{z_t})}\theta_{z_t}
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
r_{z_R}=\tanh\left(\frac{\kappa R}{2}\right).
$$

The posterior at radial $\rho = R$ is a hyperbolic Poisson kernel

$$
\boxed{
\begin{aligned}
q_R(\theta_z\mid x) 
& :=
q(\theta_{z_t}=\theta_z\mid \rho_{z_t}=R,Y_\infty=x)
\\
& =
\left(
\frac{1 - \| z_R \|^2}
{\| x - z_{R} \|^2}
\right)^{d-1} \\
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
v_x(\theta_{z_R})
:=
\kappa(I_d-\theta_{z_R}\theta_{z_R}^\top)x,
\qquad
\theta_{z_0}\sim\operatorname{Uniform}(\mathbb S^{d-1}).
}
$$

---

## Radius-as-time spherical Cauchy Score

The score and score-corrected drift are
$$
\begin{aligned}
D_x^{(\kappa)}(R,\theta_z)
&:=
\cosh(\kappa R)-\sinh(\kappa R) \langle x,\theta_z \rangle, \\
% &= \frac{e^{\kappa R} + e^{-\kappa R}}{2} - \frac{e^{\kappa R} - e^{-\kappa R}}{2} \langle x,\theta_z \rangle \\
% &= \frac{e^{\kappa R}}{2} (1 - \langle x,\theta_z \rangle) + \frac{e^{- \kappa R}}{2} (1 + \langle x,\theta_z \rangle) \\
\\
s_x(\theta_z,R)
&:=
\nabla_{\mathbb S^{d-1}}\log q_R(\theta_z\mid x)
\\
&=
(d-1)
\frac{\sinh(\kappa R)}{D_x^{(\kappa)}(R,\theta_z)}
(I_d-\theta_z\theta_z^\top)x,
\\
b_x(\theta_z,R)
&:=
v_x(\theta_z)
+
\frac{g(R)^2}{2}s_x(\theta_z,R).
\end{aligned}
$$

The Fokker–Planck equation reduces to the continuity equation:
$$
-\operatorname{div}_{\mathbb S}(q_R b_x)
+\frac{g(R)^2}{2}\Delta_{\mathbb S}q_R
=
-\operatorname{div}_{\mathbb S}(q_R v_x).
$$

---

## Sharp Lower Bound for $D_x^{(\kappa)}(R,\theta_z)$ at Fixed $R$

Because $x,\theta_z\in\mathbb S^{d-1}$, Cauchy–Schwarz gives
$\langle x,\theta_z\rangle\le 1$. For $R\ge0$,

$$
\begin{aligned}
D_x^{(\kappa)}(R,\theta_z)
&=
\cosh(\kappa R)
-\sinh(\kappa R)\langle x,\theta_z\rangle
\\
&=
\frac{e^{\kappa R} + e^{-\kappa R}}{2}
- \frac{e^{\kappa R} - e^{-\kappa R}}{2} \langle x,\theta_z\rangle
\\
&=
e^{-\kappa R}
+\sinh(\kappa R)
\bigl(1-\langle x,\theta_z\rangle\bigr)
\\
&\ge e^{-\kappa R}.
\end{aligned}
$$

Therefore the largest uniform lower bound is
$$
\boxed{
\inf_{\theta_z\in\mathbb S^{d-1}}
D_x^{(\kappa)}(R,\theta_z)
=e^{-\kappa R}.
}
$$

---

## Radius-as-time Spherical Cauchy SDE in Ambient Coordinates

In ambient Itô coordinates, this is
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

## Angular Component of the Poincare Disk Bridge

Assume the true and learned bridges share the initial law and the noise. The angular diffusion $g_{\mathrm{PD}}(R)$, target angular drift $b_x^{\mathrm{PD}}$, and shared-denominator learned angular drift $b_\theta^{\mathrm{PD}}$ are

$$
g_{\mathrm{PD}}(R) = \frac{\kappa}{\sinh(\kappa R)},
\qquad
b_x^{\mathrm{PD}}(\theta_{z_R},R)
=
\frac{(d-1) \kappa^2}{\sinh(\kappa R) D_x^{(\kappa)}(R,\theta_{z_R})}
(I_d - \theta_{z_R} \theta_{z_R}^\top) x
$$

$$
b_{\theta}^{\mathrm{PD}}(\theta_{z_R},R)
=
\frac{(d-1) \kappa^2}{\sinh(\kappa R) D_x^{(\kappa)}(R,\theta_{z_R})}
(I_d - \theta_{z_R} \theta_{z_R}^\top) \widehat x_\theta(\theta_{z_R}, \kappa R),
\qquad
\widehat x_\theta(\theta_{z_R}, \kappa R) \approx x.
$$

---

## Shared-$D_x$ Angular Path-KL

Conditioning the physical-time path-KL on the radius $R_t=R$ gives
$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{PD,ang}}(\theta)
&=
\frac12
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
\int_0^T\int_0^{R_{\max}}
p_t^{\mathrm{rad},\kappa}(R)
\mathbb E_{\theta_{z_R}\sim q_R^{(\kappa)}(\cdot\mid x)}
\left[
\frac{
\|b_x^{\mathrm{PD}}(\theta_{z_R},R)-b_\theta^{\mathrm{PD}}(\theta_{z_R},R)\|^2
}{
g_{\mathrm{PD}}(R)^2
}
\right]dR\,dt
\right].
\end{aligned}
}
$$

- In physical coordinates, $p_t^{\mathrm{rad},\kappa}$, $q_R^{(\kappa)}$, and the normalized drift error all depend on $\kappa$.
- Taking $R_{\max}=\infty$ and $T=\infty$ gives the full **angular** path-KL for this shared-$D_x$ parameterization.
- This is not yet the full hyperbolic path-KL: the radial drift error is still missing.

---

## A Full Hyperbolic ELBO Needs Every Endpoint Drift

Let $\{e_v\}_{v=1}^V\subset\mathbb S^{d-1}$ be the vocabulary boundary endpoints and let the target be $x=e_y$. For every possible endpoint $e_v$, define
$$
\begin{aligned}
D_{e_v}^{(\kappa)}(R,\theta_z)
&:=\cosh(\kappa R)-\sinh(\kappa R)\langle e_v,\theta_z\rangle,\\
A_{e_v}^{(\kappa)}(R,\theta_z)
&:=\frac{\cosh(\kappa R)\langle e_v,\theta_z\rangle-\sinh(\kappa R)}
{D_{e_v}^{(\kappa)}(R,\theta_z)},\\
C_{e_v}^{(\kappa)}(R,\theta_z)
&:=\frac{P_{\theta_z}e_v}{D_{e_v}^{(\kappa)}(R,\theta_z)},
\qquad P_{\theta_z}:=I_d-\theta_z\theta_z^\top.
\end{aligned}
$$

$A_{e_v}^{(\kappa)}$ is the endpoint-dependent radial component and $C_{e_v}^{(\kappa)}$ is the endpoint-dependent angular component.

---

## The Learned Full Drift Is a Posterior Mixture

Let
$$
p_{\theta,v}(z_t,t):=p_\theta(Y_\infty=e_v\mid z_t,t),
\qquad \sum_{v=1}^V p_{\theta,v}(z_t,t)=1,
$$
and define the posterior-mixture components
$$
\overline A_\theta(z_t,t)
:=\sum_{v=1}^V p_{\theta,v}(z_t,t)A_{e_v}^{(\kappa)}(R,\theta_z),
\qquad
\overline C_\theta(z_t,t)
:=\sum_{v=1}^V p_{\theta,v}(z_t,t)C_{e_v}^{(\kappa)}(R,\theta_z).
$$

The common radial baseline and angular Itô correction cancel between the target and learned bridges. Therefore
$$
\begin{aligned}
b_x^\rho-b_\theta^\rho
&=(d-1)\kappa\left(A_x^{(\kappa)}-\overline A_\theta\right),\\
b_x^{\mathbb S}-b_\theta^{\mathbb S}
&=\frac{(d-1)\kappa^2}{\sinh(\kappa R)}
\left(C_x^{(\kappa)}-\overline C_\theta\right).
\end{aligned}
$$

---

## Full Conditional KL Rate = Radial + Angular

The radial and spherical noises are orthogonal, with
$g_\rho(R)=1$ and $g_{\mathrm{PD}}(R)=\kappa/\sinh(\kappa R)$. Hence
$$
\boxed{
\begin{aligned}
\ell_{\mathrm{PD,full}}(z_t,x,t)
&=\frac12\left[
\frac{|b_x^\rho-b_\theta^\rho|^2}{g_\rho(R)^2}
+\frac{\|b_x^{\mathbb S}-b_\theta^{\mathbb S}\|^2}{g_{\mathrm{PD}}(R)^2}
\right]\\
&=\frac{(d-1)^2\kappa^2}{2}
\left[
\left(A_x^{(\kappa)}-\overline A_\theta\right)^2
+\left\|C_x^{(\kappa)}-\overline C_\theta\right\|^2
\right].
\end{aligned}
}
$$

Equivalently, with $w_{e_v}^{(\kappa)}:=A_{e_v}^{(\kappa)}\theta_z+C_{e_v}^{(\kappa)}$,
$$
\ell_{\mathrm{PD,full}}
=\frac{(d-1)^2\kappa^2}{2}
\left\|w_x^{(\kappa)}-\sum_{v=1}^Vp_{\theta,v}w_{e_v}^{(\kappa)}\right\|^2.
$$

---

## Complete Physical-Time Hyperbolic ELBO

Conditioning $z_t=(R,\theta_{z_R})$ on its radius gives the complete path-KL term
$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{PD,full}}(\theta)
&=
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
\int_0^T
\mathbb E_{z_t\sim q_t(\cdot\mid x)}
\left[\ell_{\mathrm{PD,full}}(z_t,x,t)\right]dt
\right]\\
&=
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
\int_0^T\!\int_0^{R_{\max}}
p_t^{\mathrm{rad},\kappa}(R)
\mathbb E_{\theta_{z_R}\sim q_R^{(\kappa)}(\cdot\mid x)}
\left[\ell_{\mathrm{PD,full}}(R,\theta_{z_R},x,t)\right]
dR\,dt
\right].
\end{aligned}
}
$$

Taking $T=R_{\max}=\infty$ gives the complete infinite-horizon hyperbolic path-KL, including both drift components.

---

## The Shared-$D_x$ Objective Is Only an Angular Surrogate

The earlier learned angular drift imposes the target denominator on the posterior mean
$$
\widehat x_\theta=\sum_{v=1}^Vp_{\theta,v}e_v,
\qquad
\ell_{\mathrm{PD,ang}}^{\mathrm{shared}\text{-}D_x}
=\frac{(d-1)^2\kappa^2}{2D_x^{(\kappa)}(R,\theta_z)^2}
\left\|P_{\theta_z}(x-\widehat x_\theta)\right\|^2.
$$

The full posterior-mixture angular term instead uses
$$
C_x^{(\kappa)}-\overline C_\theta
=
\frac{P_{\theta_z}x}{D_x^{(\kappa)}}
-\sum_{v=1}^Vp_{\theta,v}
\frac{P_{\theta_z}e_v}{D_{e_v}^{(\kappa)}}.
$$

Therefore the two objectives agree only in special cases; the shared-$D_x$ step is not an algebraic identity.

---

## Curvature Enters the Shared-$D_x$ Angular Rate

Writing $P_{\theta_{z_R}} = I_d - \theta_{z_R} \theta_{z_R}^\top$,
$$
\begin{aligned}
\frac{\| b_x^{\mathrm{PD}}(\theta_{z_R},R) - b_\theta^{\mathrm{PD}}(\theta_{z_R},R) \|^2}
{g_{\mathrm{PD}}(R)^2}
& =
\frac{(d-1)^2\kappa^4}{\sinh(\kappa R)^2D_x^{(\kappa)}(R,\theta_{z_R})^2g_{\mathrm{PD}}(R)^2}
\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},\kappa R)\right)\right\|^2 \\
& =
\frac{(d-1)^2\kappa^2}{D_x^{(\kappa)}(R,\theta_{z_R})^2}
\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},\kappa R)\right)\right\|^2 .
\end{aligned}
$$

---

# Condition the Angular Objective on Radius and Angle

$$
\begin{aligned}
\mathcal L_{\mathrm{PD,ang}}(\theta)
&=
\frac12
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^T\int_0^{R_{\max}}
    p_t^{\mathrm{rad},\kappa}(R)
    \mathbb E_{\theta_{z_R}\sim q_R^{(\kappa)}(\cdot\mid x)}
    \left[
        \frac{
        \|b_x^{\mathrm{PD}}(\theta_{z_R},R) - b_\theta^{\mathrm{PD}}(\theta_{z_R},R)\|^2
        }{
        g_{\mathrm{PD}}(R)^2
        }
    \right]dR\,dt
\right] \\
&=
\frac{(d-1)^2 \kappa^4}{2}
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^T\int_0^{R_{\max}}
    \frac{p_t^{\mathrm{rad},\kappa}(R)}{\sinh(\kappa R)^2g_{\mathrm{PD}}(R)^2}
    \mathbb E_{\theta_{z_R}\sim q_R^{(\kappa)}(\cdot\mid x)}
    \left[
        \frac{\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},\kappa R)\right)\right\|^2}
        {D_x^{(\kappa)}(R,\theta_{z_R})^2}
    \right]dR\,dt
\right] \\
&=
\frac{(d-1)^2 \kappa^4}{2}
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^T \int_0^{R_{\max}} \int_{\mathbb S^{d-1}}
    \frac{p_t^{\mathrm{rad},\kappa}(R) q_R^{(\kappa)}(\theta_{z_R} \mid x)}{\sinh(\kappa R)^2D_x^{(\kappa)}(R,\theta_{z_R})^2g_{\mathrm{PD}}(R)^2}
    \left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},\kappa R)\right)\right\|^2
    d\sigma(\theta_{z_R})\, dR\, dt
\right]
\end{aligned}
$$

($\sigma$: uniform probability measure on $\mathbb S^{d-1}$, w.r.t. which $q_R^{(\kappa)}$ is a density; for $d=2$, $d\sigma=d\phi/2\pi$.)

---

## When Is the Angular Path-KL Curvature-Invariant?

Set the dimensionless radius and time to
$$
u=\kappa R,
\qquad
\tau=\kappa^2t.
$$

Metric scaling gives
$$
\begin{aligned}
D_x^{(\kappa)}(R,\theta)&=D_x(u,\theta),
&
q_R^{(\kappa)}(\theta\mid x)&=q_u(\theta\mid x),
\\
p_t^{\mathrm{rad},\kappa}(R)
&= \kappa\, \widetilde p_\tau^{\mathrm{rad}}(u)
&
dt&=\frac{d\tau}{\kappa^2}.
\end{aligned}
$$

---

## Change of Variable on Radius Integral

Since $p_t^{\mathrm{rad},\kappa}$ is a probability density with respect to $dR$
$$
\widetilde p_\tau^{\mathrm{rad}}(u)
:=
\frac{1}{\kappa}
p_{\tau/\kappa^2}^{\mathrm{rad},\kappa}\!\left(\frac{u}{\kappa}\right).
$$

Equivalently,

$$
\begin{aligned}
\int_0^{R_{\max}} p_t^{\mathrm{rad},\kappa}(R) dR
= \int_0^{\kappa R_{\max}} p_{\tau / \kappa^2}^{\mathrm{rad}, \kappa}(\frac{u}{\kappa}) \frac{du}{\kappa}
= \int_0^{\kappa R_{\max}} \widetilde p_\tau^{\mathrm{rad}}(u) du
\end{aligned}
$$

Thus the Jacobian $dR=du/\kappa$ is already absorbed into the transformed density. 

---

## When Is the Angular Path-KL Curvature-Invariant?

With $g_{\mathrm{PD}}(R) = \frac{\kappa}{\sinh(\kappa R)}$ and $p_t^{\mathrm{rad},\kappa}(R)=\kappa\,\widetilde p_\tau^{\mathrm{rad}}(u)$,

$$
\begin{aligned}
\frac{p_t^{\mathrm{rad},\kappa}(R) q_R^{(\kappa)}(\theta_{z_R} \mid x)}{\sinh(\kappa R)^2D_x^{(\kappa)}(R,\theta_{z_R})^2g_{\mathrm{PD}}(R)^2}
\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2
&= \frac{\kappa \widetilde p_\tau^{\mathrm{rad}}(u) q_u(\theta_{z_R} \mid x) \sinh(u)^2}{\sinh(u)^2 \kappa^2}
\frac{\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2}{D_x(u,\theta_{z_R})^2} \\
&= \frac{\widetilde p_\tau^{\mathrm{rad}}(u) q_u(\theta_{z_R} \mid x)}{\kappa}
\frac{\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2}{D_x(u,\theta_{z_R})^2} \\
\end{aligned}
$$

The learned endpoint depends on the dimensionless radius:
$$
\widehat x_\theta(\theta_{z_R},\kappa R)
=
\widehat x_\theta(\theta_{z_R},u).
$$

---

## Scale Equivariance Leaves One $\kappa^2$ Factor

$$
\begin{aligned}
\mathcal L_{\mathrm{PD,ang}}(\theta)
&=
\frac12
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^T \int_0^{R_{\max}}
    p_t^{\mathrm{rad},\kappa}(R)
    \mathbb E_{\theta_{z_R}\sim q_R^{(\kappa)}(\cdot\mid x)}
    \left[
        \frac{
        \|b_x^{\mathrm{PD}}(\theta_{z_R},R) - b_\theta^{\mathrm{PD}}(\theta_{z_R},R)\|^2
        }{
        g_{\mathrm{PD}}(R)^2
        }
    \right]dR\,dt
\right] \\
&=
\frac{(d-1)^2 \kappa^4}{2}
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^{\kappa^2 T} \int_0^{\kappa R_{\max}} \int_{\mathbb S^{d-1}}
    \frac{\widetilde p_\tau^{\mathrm{rad}}(u) q_u(\theta_{z_R} \mid x)}{\kappa D_x(u,\theta_{z_R})^2}
    \left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2
    d\sigma(\theta_{z_R}) \frac{du}{\kappa} \frac{d\tau}{\kappa^2}
\right] \\
&=
\frac{(d-1)^2}{2}
\mathbb E_{x\sim p_{\mathrm{data}}}
\left[
    \int_0^{\kappa^2 T} \int_0^{\kappa R_{\max}} \int_{\mathbb S^{d-1}}
    \widetilde p_\tau^{\mathrm{rad}}(u) q_u(\theta_{z_R} \mid x)
    \frac{\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2}{D_x(u,\theta_{z_R})^2}
    d\sigma(\theta_{z_R})\, du\, d\tau
\right]
\end{aligned}
$$

The conditional KL rate carries exactly one remaining factor $\kappa^2$:
$$
\frac{\|b_x^{\mathrm{PD}}-b_\theta^{\mathrm{PD}}\|^2}
{g_{\mathrm{PD}}^2}
=
\kappa^2(d-1)^2
\frac{\left\|P_{\theta_{z_R}}\!\left(x-\widehat x_\theta(\theta_{z_R},u)\right)\right\|^2}{D_x(u,\theta_{z_R})^2}.
$$

---

## Curvature Cancels in the Path-KL if Horizon Scales Properly

The $\kappa^2$ KL rate cancels with $dt=d\tau/\kappa^2$, giving
$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{PD,ang}}^{(\kappa)}(T,R_{\max})
&=
\frac{(d-1)^2}{2}\mathbb E_x
\left[
\int_0^{\kappa^2T}\!\int_0^{\kappa R_{\max}}
\widetilde p_\tau^{\mathrm{rad}}(u)
\mathbb E_{\theta_u\sim q_u(\cdot\mid x)}
\left[
\frac{\left\|P_{\theta_u}\!\left(x-\widehat x_\theta(\theta_u,u)\right)\right\|^2}{D_x(u,\theta_u)^2}
\right]du\,d\tau
\right].
\end{aligned}
}
$$

Therefore:

- **Fixed physical horizons:** not invariant, because the limits are $\kappa R_{\max}$ and $\kappa^2T$.
- **Infinite-horizon path-KL:** invariant when $T=R_{\max}=\infty$ and the learned predictor is scale-equivariant.

---

# What Curvature Controls in the Angular Objective

- Curvature controls the weighting, but the angular objective is invariant to curvature over infinite time and radius intervals
- Curvature therefore controls decoding speed and which noise levels receive training mass.

---

The proposal distribution $p_t^{\mathrm{rad},\kappa}(R)$ over $t\in[0,5]$

![alt text](image.png)

---

Histogram of $p_t^{\mathrm{rad},\kappa}(R)/D_x^{(\kappa)}(R,\theta)$ over $t\in[0,5]$ for $K\in\{-0.01,-0.1,-1,-5,-10\}$, assuming $\langle x,\theta_z\rangle=1$

![alt text](image-3.png)

---

## Unit Embeddings Give a Weighted CE Bound

Let $\mathbf E=[e_1^\top;\ldots;e_V^\top]$ with $\|e_v\|_2=1$. For token $y$
$$
x=e_y=\mathbf E^\top\delta_y,
\qquad
p_\theta=p_\theta(\cdot\mid\theta_u,u),
\qquad
\widehat x_\theta=\mathbf E^\top p_\theta.
$$

Since $P_{\theta_u}$ is an orthogonal projector and $\|\mathbf E^\top a\|_2\leq\|a\|_1$,
$$
\begin{aligned}
\|P_{\theta_u}(x-\widehat x_\theta)\|_2^2
&\leq \|\mathbf E^\top(\delta_y-p_\theta)\|_2^2 \\
&\leq \|\delta_y-p_\theta\|_1^2 \\
&\leq 2\,\mathrm{KL}(\delta_y\|p_\theta)
=2\,\mathrm{CE}(y,p_\theta),
\end{aligned}
$$

---

## Weighted CE Upper-Bounds the Hyperbolic Path-KL

Substituting the embedding bound into the path-KL term gives
$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{PD,ang}}(\theta)
&\leq
(d-1)^2\mathbb E_y
\left[
\int_0^{\kappa^2T}\!\int_0^{\kappa R_{\max}}
\widetilde p_\tau^{\mathrm{rad}}(u)
\mathbb E_{\theta_u\sim q_u(\cdot\mid e_y)}
\left[
\frac{\mathrm{CE}(y,p_\theta)}{D_{e_y}(u,\theta_u)^2}
\right]du\,d\tau
\right]
\\
&=: \mathcal L_{\mathrm{CE}}^{\mathrm{hyp}}(\theta).
\end{aligned}
}
$$

---

# Conclusion

- Curvature controls the variance of the angular path-KL estimator
- Can we make the geometry adaptive to the implied tree structure?

---

## Hyperbolic Product Manifold

Let
$$
\mathcal M
=
\prod_{m=1}^M \mathbb H_{K_m}^{d_m},
\qquad
K_m=-\kappa_m^2<0.
$$

Assume independent noises and a factorized bridge. For factor $m$, let
$$
D_{x_m} (u_m, \theta_{u_m})
=
\cosh(u_m)
-\sinh(u_m) \langle x_m,\theta_{u_m} \rangle,
\qquad
P_{\theta_{u_m}}=I_{d_m}-\theta_{u_m} \theta_{u_m}^\top
\qquad
g_m(R_m) = \frac{\kappa_m}{\sinh(\kappa_m R_m)}.
$$

Using the shared denominator in the true and learned drifts,
$$
\frac{\| b_{x_m, m}^{\mathrm{PD}} - b_{\theta_{u_m}, m}^{\mathrm{PD}} \|^2}{g_m(R_m)^2}
=
(d_m-1)^2\kappa_m^2
\frac{
\|P_{\theta_{u_m}}(x_m-\widehat x_{\theta,m}(\theta_{u_m}, u_m))\|^2
}{D_{x_m} (u_m, \theta_{u_m})^2}.
$$

---

### Shared-$D_{x_m}$ Angular Product Objective

Set $u_m=\kappa_mR_m$ and $\tau_m=\kappa_m^2t$. Since

$$
\kappa_m^2p_{m,t}^{\mathrm{rad},\kappa_m}(R_m)dR_mdt
=\widetilde p_{\tau_m}^{\mathrm{rad}}(u_m)du_m d\tau_m
$$

the infinite-horizon shared-$D_{x_m}$ angular objective is

$$
\begin{aligned}
\mathcal L_{\mathrm{Prod,ang}}
&=
\sum_{m=1}^{M} \frac{(d_m-1)^2}{2}\mathbb E_x
\left[
\int_0^{\infty}\!\int_0^{\infty}
\widetilde p_{\tau_m}^{\mathrm{rad}}(u_m)
\mathbb E_{\theta_{u_m} \sim q_{u_m}(\cdot \mid x_m)}
\left[
\frac{\left\|P_{\theta_{u_m}}\!\left(x_m - \widehat x_{\theta,m}(\theta_{u_m}, u_m)\right)\right\|^2}{D_{x_m}(u_m, \theta_{u_m})^2}
\right] d u_m d \tau_m
\right].
\end{aligned}
$$

Thus, assuming factorwise scale-equivariant predictors and convergent integrals, this angular product objective is invariant to $\boldsymbol\kappa$.

---

## Full Product Drift Uses the Joint Posterior

Write the endpoint of word $v$ in factor $m$ as $e_{v,m}\in\mathbb S^{d_m-1}$ and $x_m=e_{y,m}$. Define
$$
\begin{aligned}
D_{e_{v,m}}(u_m,\theta_{u_m})
&:=\cosh(u_m)-\sinh(u_m)\langle e_{v,m},\theta_{u_m}\rangle,\\
A_{e_{v,m}}(u_m,\theta_{u_m})
&:=\frac{\cosh(u_m)\langle e_{v,m},\theta_{u_m}\rangle-\sinh(u_m)}
{D_{e_{v,m}}(u_m,\theta_{u_m})},\\
C_{e_{v,m}}(u_m,\theta_{u_m})
&:=\frac{P_{\theta_{u_m}}e_{v,m}}
{D_{e_{v,m}}(u_m,\theta_{u_m})},\\
w_{e_{v,m}}&:=A_{e_{v,m}}\theta_{u_m}+C_{e_{v,m}}.
\end{aligned}
$$

The same joint posterior over words is used in every factor:
$$
p_{\theta,v}(\mathbf z_t,t)
:=p_\theta(Y_\infty=e_v\mid \mathbf z_t,t),
\qquad
\overline w_{\theta,m}:=\sum_{v=1}^Vp_{\theta,v}(\mathbf z_t,t)w_{e_{v,m}}.
$$

---

## Full Product KL Rate Adds Across Factors

Independent factor noises make the full conditional KL rates add:
$$
\boxed{
\ell_{\mathrm{Prod,full}}(\mathbf z_t,y,t)
=\frac12\sum_{m=1}^M(d_m-1)^2\kappa_m^2
\left\|w_{x_m}-\overline w_{\theta,m}\right\|^2.
}
$$

The summand contains both the radial error
$|A_{x_m}-\sum_vp_{\theta,v}A_{e_{v,m}}|^2$ and the angular error
$\|C_{x_m}-\sum_vp_{\theta,v}C_{e_{v,m}}\|^2$ because these components are orthogonal.

---

## Complete Product-Manifold Hyperbolic ELBO

Therefore the complete physical-time path-KL is
$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{Prod,full}}(\theta)
&=\mathbb E_{y\sim p_{\mathrm{data}}}
\left[
\int_0^T
\mathbb E_{\mathbf z_t\sim q_t(\cdot\mid y)}
\left[\ell_{\mathrm{Prod,full}}(\mathbf z_t,y,t)\right]dt
\right]\\
&=\mathbb E_y\left[
\int_0^T\!\int_{\mathbb R_+^M}
\prod_{m=1}^M p_{m,t}^{\mathrm{rad},\kappa_m}(R_m)
\mathbb E_{\boldsymbol\theta\sim\prod_mq_{u_m}(\cdot\mid x_m)}
\left[\ell_{\mathrm{Prod,full}}\right]
d\mathbf R\,dt
\right].
\end{aligned}
}
$$

Because $p_{\theta,v}(\mathbf z_t,t)$ can couple all factors, this full objective does not generally split into independent $\tau_m=\kappa_m^2t$ integrals.

Invariance to $\boldsymbol\kappa$ therefore requires an additional factorization/equivariance assumption.

---

## Relation to the Implemented Losses

For one factor with $K=-1$, the complete local rate is
$$
\ell_{\mathrm{PD,full}}
=\frac{(d-1)^2}{2}
\left\|w_x-\sum_{v=1}^Vp_{\theta,v}w_{e_v}\right\|^2,
$$
which is the quantity evaluated by `bridge_loss_poincare_disk_polar` before time-proposal weighting.

The refactored loss evaluates instead
$$
\ell_{\mathrm{PD,ang}}^{\mathrm{shared}\text{-}D_x}
=\frac{(d-1)^2\kappa^2}{2D_x^2}
\left\|P_{\theta_z}(x-\widehat x_\theta)\right\|^2,
$$
factor by factor. It omits the radial error and replaces the per-endpoint $D_{e_v}$ with the target denominator $D_x$.

---

# Conclusion

- We can build adaptive noise scheduler, not only temporally, but also spatially
- Incorporate with product manifold, we can achieve adaptive curvature for various directions
- Intuively, adaptive curvature means allocate proper step size and time budget on various depth of trees, which can solve the ill condition problem on word embedding for DLM.

---
