# Flow Matching SDE Sampler: Derivation from the Bridge Perspective

## 1. Linear Flow-Matching Bridge

Consider the linear interpolation

$$
x_t = t x_1 + (1-t)\epsilon,
\qquad
\epsilon \sim \mathcal N(0,I),
\qquad
t\in[0,1],
$$

where

- $x_1 \sim p_{\mathrm{data}}$ is a data sample,
- $\epsilon \sim \mathcal N(0,I)$ is the base-noise sample,
- $x_0=\epsilon$,
- $x_1$ is reached at $t=1$.

Conditioned on $x_1$, the distribution of $x_t$ is Gaussian:

$$
\boxed{
p_t(x_t\mid x_1)
=
\mathcal N\!\left(
t x_1,\,
(1-t)^2 I
\right).
}
$$

---

## 2. Conditional Flow-Matching Velocity

Differentiate the interpolation with respect to $t$ while keeping the endpoints $(x_1,\epsilon)$ fixed:

$$
\frac{d x_t}{dt}
=
x_1-\epsilon.
$$

From

$$
x_t=t x_1+(1-t)\epsilon,
$$

we solve for $\epsilon$:

$$
\epsilon
=
\frac{x_t-tx_1}{1-t}.
$$

Therefore,

$$
\begin{aligned}
\frac{d x_t}{dt}
&=
x_1-\frac{x_t-tx_1}{1-t}\\
&=
\frac{(1-t)x_1-x_t+t x_1}{1-t}\\
&=
\frac{x_1-x_t}{1-t}.
\end{aligned}
$$

Hence the conditional velocity field is

$$
\boxed{
u_t(x_t\mid x_1)
=
\frac{x_1-x_t}{1-t}.
}
$$

The corresponding conditional ODE is

$$
\boxed{
d x_t
=
\frac{x_1-x_t}{1-t}\,dt.
}
$$

---

## 3. Marginal Flow-Matching Velocity

At sampling time, $x_1$ is unknown. Therefore, the generative vector field cannot depend directly on the endpoint $x_1$.

The marginal Flow Matching velocity is obtained by averaging the conditional velocity over the posterior distribution of $x_1$ given $x_t=x$:

$$
\boxed{
v_t(x)
=
\mathbb E\left[
u_t(x\mid x_1)
\mid x_t=x
\right].
}
$$

Substituting the conditional velocity,

$$
\boxed{
v_t(x)
=
\mathbb E\left[
\frac{x_1-x}{1-t}
\;\middle|\;
x_t=x
\right].
}
$$

Since $x$ is fixed inside the conditional expectation,

$$
v_t(x)
=
\frac{
\mathbb E[x_1\mid x_t=x]-x
}{1-t}.
$$

Therefore,

$$
\boxed{
\mathbb E[x_1\mid x_t=x]
=
x+(1-t)v_t(x).
}
$$

The marginal density $p_t(x)$ satisfies the continuity equation

$$
\boxed{
\partial_t p_t(x)
=
-\nabla_x\cdot
\left(
p_t(x)v_t(x)
\right).
}
$$

Hence the ordinary Flow Matching sampler is the ODE

$$
\boxed{
dX_t
=
v_t(X_t)\,dt.
}
$$

In practice, a neural network $v_\theta(x,t)$ is trained to approximate $v_t(x)$:

$$
\boxed{
dX_t
=
v_\theta(X_t,t)\,dt,
\qquad
X_0\sim \mathcal N(0,I).
}
$$

---

## 4. Conditional Score of the Bridge

From the conditional Gaussian distribution

$$
p_t(x\mid x_1)
=
\mathcal N\!\left(
t x_1,\,
(1-t)^2I
\right),
$$

its log density is, up to an additive constant,

$$
\log p_t(x\mid x_1)
=
-\frac{1}{2(1-t)^2}
\|x-tx_1\|^2
+
C(t).
$$

Taking the gradient with respect to $x$,

$$
\boxed{
\nabla_x\log p_t(x\mid x_1)
=
-\frac{x-tx_1}{(1-t)^2}.
}
$$

This is the **conditional score**.

It is not directly available at inference time because it depends on the unknown data endpoint $x_1$.

---

## 5. Marginal Score from the Flow-Matching Velocity

Define the marginal score

$$
s_t(x)
:=
\nabla_x\log p_t(x).
$$

Using Fisher's identity,

$$
\nabla_x\log p_t(x)
=
\mathbb E\left[
\nabla_x\log p_t(x\mid x_1)
\mid x_t=x
\right].
$$

Therefore,

$$
\begin{aligned}
s_t(x)
&=
\mathbb E\left[
-\frac{x-tx_1}{(1-t)^2}
\;\middle|\;
x_t=x
\right]\\
&=
-\frac{
x-t\,\mathbb E[x_1\mid x_t=x]
}{(1-t)^2}.
\end{aligned}
$$

From the marginal velocity relation,

$$
\mathbb E[x_1\mid x_t=x]
=
x+(1-t)v_t(x).
$$

Substituting,

$$
\begin{aligned}
s_t(x)
&=
-\frac{
x-t\left[x+(1-t)v_t(x)\right]
}{(1-t)^2}\\
&=
-\frac{
x-tx-t(1-t)v_t(x)
}{(1-t)^2}\\
&=
-\frac{
(1-t)x-t(1-t)v_t(x)
}{(1-t)^2}\\
&=
-\frac{
x-t v_t(x)
}{1-t}.
\end{aligned}
$$

Hence

$$
\boxed{
s_t(x)
=
\nabla_x\log p_t(x)
=
\frac{
t\,v_t(x)-x
}{1-t}.
}
$$

Thus, for this linear Gaussian Flow Matching path, the marginal score can be reconstructed directly from the marginal velocity field.

With a trained Flow Matching model,

$$
\boxed{
s_\theta(x,t)
=
\frac{
t\,v_\theta(x,t)-x
}{1-t}.
}
$$

No separately trained score network is required.

---

## 6. Constructing an SDE with the Same Marginals as the Flow ODE

The Flow Matching ODE

$$
dX_t=v_t(X_t)\,dt
$$

induces the continuity equation

$$
\partial_t p_t
=
-\nabla\cdot(p_t v_t).
$$

Now consider an Itô SDE

$$
dX_t
=
b_t(X_t)\,dt
+
\sqrt{a(t)}\,dW_t,
$$

where $a(t)\ge 0$ is a scalar diffusion schedule.

Its Fokker--Planck equation is

$$
\partial_t p_t
=
-\nabla\cdot(p_t b_t)
+
\frac{a(t)}{2}\Delta p_t.
$$

We want this SDE to have the same one-time marginals $p_t$ as the Flow Matching ODE.

Choose the SDE drift to be

$$
\boxed{
b_t(x)
=
v_t(x)
+
\frac{a(t)}{2}s_t(x),
}
$$

where

$$
s_t(x)=\nabla\log p_t(x).
$$

Then

$$
\begin{aligned}
\partial_t p_t
&=
-\nabla\cdot
\left[
p_t
\left(
v_t+\frac{a(t)}{2}s_t
\right)
\right]
+
\frac{a(t)}{2}\Delta p_t\\
&=
-\nabla\cdot(p_t v_t)
-\frac{a(t)}{2}
\nabla\cdot(p_t s_t)
+
\frac{a(t)}{2}\Delta p_t.
\end{aligned}
$$

Since

$$
p_t s_t
=
p_t\nabla\log p_t
=
\nabla p_t,
$$

we have

$$
\nabla\cdot(p_t s_t)
=
\Delta p_t.
$$

Therefore, the two diffusion-related terms cancel:

$$
-\frac{a(t)}{2}\Delta p_t
+
\frac{a(t)}{2}\Delta p_t
=
0.
$$

Thus,

$$
\boxed{
\partial_t p_t
=
-\nabla\cdot(p_t v_t),
}
$$

which is exactly the same continuity equation as the Flow Matching ODE.

Hence the SDE

$$
\boxed{
dX_t
=
\left[
v_t(X_t)
+
\frac{a(t)}{2}s_t(X_t)
\right]dt
+
\sqrt{a(t)}\,dW_t
}
$$

has the same one-time marginals $p_t$ as the Flow Matching ODE.

Importantly, the ODE and SDE generally do **not** have the same path distribution. They only share the same marginal distribution at each time $t$.

---

## 7. Parameterization with $\eta$ and $g(t)$

Let

$$
a(t)
=
\eta g^2(t),
$$

where $\eta\ge 0$ controls the overall amount of stochasticity.

Then

$$
\boxed{
dX_t
=
\left[
v_t(X_t)
+
\frac{\eta g^2(t)}{2}s_t(X_t)
\right]dt
+
\sqrt{\eta}\,g(t)\,dW_t.
}
$$

Using

$$
s_t(x)
=
\frac{t v_t(x)-x}{1-t},
$$

we obtain

$$
\boxed{
dX_t
=
\left[
v_t(X_t)
+
\frac{\eta g^2(t)}{2}
\frac{
t v_t(X_t)-X_t
}{1-t}
\right]dt
+
\sqrt{\eta}\,g(t)\,dW_t.
}
$$

For a trained Flow Matching model,

$$
\boxed{
dX_t
=
\left[
v_\theta(X_t,t)
+
\frac{\eta g^2(t)}{2(1-t)}
\left(
t\,v_\theta(X_t,t)-X_t
\right)
\right]dt
+
\sqrt{\eta}\,g(t)\,dW_t.
}
$$

This is an SDE sampler that uses only the trained Flow Matching velocity model.

---

## 8. Euler--Maruyama Sampler

Let

$$
0=t_0<t_1<\cdots<t_N<1,
$$

with

$$
\Delta t_k=t_{k+1}-t_k.
$$

Initialize

$$
X_0\sim\mathcal N(0,I).
$$

At each step,

$$
v_k
=
v_\theta(X_k,t_k),
$$

and reconstruct the score

$$
s_k
=
\frac{
t_k v_k-X_k
}{1-t_k}.
$$

Then Euler--Maruyama gives

$$
\boxed{
X_{k+1}
=
X_k
+
\left[
v_k
+
\frac{\eta g^2(t_k)}{2}s_k
\right]\Delta t_k
+
\sqrt{
\eta g^2(t_k)\Delta t_k
}\,\xi_k,
}
$$

where

$$
\xi_k\sim\mathcal N(0,I).
$$

Equivalently,

$$
\boxed{
X_{k+1}
=
X_k
+
\left[
v_\theta(X_k,t_k)
+
\frac{\eta g^2(t_k)}
{2(1-t_k)}
\left(
t_kv_\theta(X_k,t_k)-X_k
\right)
\right]\Delta t_k
+
\sqrt{
\eta g^2(t_k)\Delta t_k
}\,\xi_k.
}
$$

When $\eta=0$, this reduces to the deterministic Flow Matching ODE sampler:

$$
X_{k+1}
=
X_k+
v_\theta(X_k,t_k)\Delta t_k.
$$

---

## 9. Choosing a Regular Diffusion Schedule

The score representation

$$
s_t(x)
=
\frac{t v_t(x)-x}{1-t}
$$

contains an explicit factor $1/(1-t)$, so a constant $g(t)$ can lead to a singular-looking drift as $t\to1$.

A convenient choice is

$$
\eta g^2(t)
=
2\lambda(1-t),
$$

where $\lambda\ge 0$ is a constant or a bounded nonnegative function of time.

Then

$$
\frac{\eta g^2(t)}{2}s_t(x)
=
\lambda
\left(
t v_t(x)-x
\right).
$$

Therefore,

$$
\boxed{
dX_t
=
\left[
(1+\lambda t)v_t(X_t)
-
\lambda X_t
\right]dt
+
\sqrt{
2\lambda(1-t)
}\,dW_t.
}
$$

With a trained model,

$$
\boxed{
dX_t
=
\left[
(1+\lambda t)v_\theta(X_t,t)
-
\lambda X_t
\right]dt
+
\sqrt{
2\lambda(1-t)
}\,dW_t.
}
$$

The corresponding Euler--Maruyama step is

$$
\boxed{
X_{k+1}
=
X_k
+
\left[
(1+\lambda t_k)v_\theta(X_k,t_k)
-
\lambda X_k
\right]\Delta t_k
+
\sqrt{
2\lambda(1-t_k)\Delta t_k
}\,\xi_k.
}
$$

This parameterization has two useful properties:

1. the injected noise vanishes as $t\to1$;
2. the explicit $1/(1-t)$ factor disappears from the drift.

---

## 10. Summary

Starting from the linear Flow Matching bridge

$$
\boxed{
x_t=t x_1+(1-t)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
}
$$

the conditional velocity is

$$
\boxed{
u_t(x\mid x_1)
=
\frac{x_1-x}{1-t}.
}
$$

Marginalizing over the unknown endpoint gives

$$
\boxed{
v_t(x)
=
\mathbb E\left[
\frac{x_1-x}{1-t}
\;\middle|\;
x_t=x
\right].
}
$$

The marginal score is recoverable from the velocity:

$$
\boxed{
s_t(x)
=
\nabla_x\log p_t(x)
=
\frac{t v_t(x)-x}{1-t}.
}
$$

Therefore, an entire family of SDEs with the same marginals as the Flow Matching ODE is

$$
\boxed{
dX_t
=
\left[
v_t(X_t)
+
\frac{\eta g^2(t)}{2}s_t(X_t)
\right]dt
+
\sqrt{\eta}\,g(t)\,dW_t.
}
$$

Equivalently, using only the Flow Matching velocity,

$$
\begin{aligned}

        dX_t
        & =
        \left[
            v_t(X_t)
            +
            \frac{\eta g^2(t)}{2(1-t)}
            \left(
                t v_t(X_t)-X_t
            \right)
        \right]dt
        +
        \sqrt{\eta}\,g(t)\,dW_t \\
        & =
        \left[
            \left( 1 + \frac{\eta g^2(t) t}{2(1-t)} \right)
            v_t(X_t)
            -
            \frac{\eta g^2(t)}{2(1-t)} X_t
        \right]dt
        +
        \sqrt{\eta}\,g(t)\,dW_t \\.

\end{aligned}
$$

The conceptual chain is therefore

$$
\boxed{
\begin{array}{c}
x_t=t x_1+(1-t)\epsilon\\[4pt]
\Downarrow\\
v_t(x)=
\mathbb E\!\left[
\dfrac{x_1-x}{1-t}\mid x_t=x
\right]\\[10pt]
\Downarrow\\
s_t(x)=
\dfrac{t v_t(x)-x}{1-t}\\[10pt]
\Downarrow\\
dX_t=
\left[
v_t(X_t)+
\dfrac{\eta g^2(t)}{2}s_t(X_t)
\right]dt
+
\sqrt{\eta}\,g(t)dW_t.
\end{array}
}
$$

---

## 11. Derivation in the Noise-Level Index $b$ (the $\alpha_t$-Native Form)

Sections 1–10 index the bridge by the flow-matching time $t\in[0,1]$ ($t=0$ noise, $t=1$ data). The sampler code instead indexes every step by the noise-schedule output $\alpha$, through the **noise fraction**

$$
\boxed{
b=
\begin{cases}
\alpha_t, & \texttt{invert\_time\_convention=true}\ \text{(EFLM default)},\\[2pt]
1-\alpha_t, & \texttt{invert\_time\_convention=false}\ \text{(MDLM-like)},
\end{cases}
}
$$

with which training interpolates $x=(1-b)\,x_1+b\,\epsilon$ (cf. `algo.EFLM.q_xt`). The two indices are related by the deterministic dictionary

$$
\boxed{
b=1-t,
\qquad
t=1-b,
}
$$

so this section derives nothing new — it rewrites §§1–8 directly in $b$, in the quantities the sampler already holds, so that the code needs no $t\leftrightarrow\alpha$ conversion. During sampling, $b$ decreases from $1$ (pure noise) to $0$ (data).

### 11.1 Bridge

$$
\boxed{
x_b=(1-b)x_1+b\,\epsilon,
\qquad
p_b(x\mid x_1)=\mathcal N\!\big((1-b)x_1,\ b^2 I\big).
}
$$

### 11.2 Velocity, Predicted Endpoint, and the ODE Step

Differentiate with the endpoints fixed and eliminate $\epsilon=\frac{x_b-(1-b)x_1}{b}$:

$$
\frac{dx_b}{db}
=\epsilon-x_1
=\frac{x_b-x_1}{b}.
$$

Averaging over the posterior of $x_1$ given $x_b=x$, with $\hat x(x,b):=\mathbb E[x_1\mid x_b=x]$:

$$
\boxed{
v_b(x)=\frac{x-\hat x(x,b)}{b},
\qquad
\hat x=x-b\,v_b(x).
}
$$

The trained model supplies the predicted endpoint $\hat x=p_\theta E$; the code works with the raw direction

$$
\boxed{
\mathrm{vel}:=\hat x-x=-b\,v_b(x).
}
$$

An Euler step of $dX=v_b(X)\,db$ from $b_k$ down to $b_{k+1}$, with $\Delta_k:=b_k-b_{k+1}>0$ (the noise removed this step):

$$
\boxed{
X_{k+1}
=X_k+v_b(X_k)\,(b_{k+1}-b_k)
=X_k+\frac{\Delta_k}{b_k}\,\mathrm{vel}.
}
$$

The prefactor $\Delta_k/b_k$ — the fraction of the remaining noise removed — is exactly the step size returned by `sfm_step_size` under both conventions.

### 11.3 Score

From §11.1, $\nabla_x\log p_b(x\mid x_1)=-\frac{x-(1-b)x_1}{b^2}$, and Fisher's identity gives

$$
s_b(x)
=-\frac{x-(1-b)\hat x}{b^2}.
$$

Substituting $\hat x=x+\mathrm{vel}$:

$$
\boxed{
s_b(x)
=\frac{(1-b)\,\mathrm{vel}-b\,x}{b^2}.
}
$$

(Check: with $b=1-t$ and $\mathrm{vel}=(1-t)v_t$ this is §5's $s_t=\frac{t\,v_t-x}{1-t}$.)

### 11.4 Marginal-Preserving SDE

To reuse the Fokker–Planck argument of §6 with its standard forward-time sign conventions, run the sampler on the increasing clock $s=1-b$ ($ds=-db$) — bookkeeping only; this clock coincides with the flow-matching time $t$ of §§1–10. The marginals $q_s:=p_{1-s}$ satisfy $\partial_s q_s=-\nabla\!\cdot(q_s\,u)$ with the generative velocity

$$
u(x)=-v_b(x)=\frac{\mathrm{vel}}{b},
$$

and, exactly as in §6, the drift $u+\frac{a}{2}s_b$ with diffusion $\sqrt{a}$ preserves $q_s$ for any $a(b)\ge0$. One Euler–Maruyama step ($\Delta_k=b_k-b_{k+1}=\Delta s_k$, $\xi_k\sim\mathcal N(0,I)$):

$$
\boxed{
X_{k+1}
=X_k
+\left[\frac{\mathrm{vel}}{b_k}+\frac{a(b_k)}{2}\,s_b(X_k)\right]\Delta_k
+\sqrt{a(b_k)\,\Delta_k}\;\xi_k.
}
$$

$a\equiv0$ recovers the ODE step of §11.2.

### 11.5 Diffusion Schedule and the Final Code-Native Step

Parameterize $a(b)=\eta\,G^2(b)$. Choose the linear scale $G(b)=b$ (equivalently $g(t)=1-t$ in the index of §7, which left $g$ generic; this is the schedule the sampler implements). Then all $1/b$ factors in the score term cancel:

$$
\frac{\eta G^2(b)}{2}\,s_b(x)
=\frac{\eta}{2}\big((1-b)\,\mathrm{vel}-b\,x\big),
\qquad
\sqrt{a\,\Delta}=b\sqrt{\eta\,\Delta}.
$$

$$
\boxed{
X_{k+1}
=X_k
+\frac{\Delta_k}{b_k}\,\mathrm{vel}
+\frac{\eta}{2}\big((1-b_k)\,\mathrm{vel}-b_k X_k\big)\Delta_k
+b_k\sqrt{\eta\,\Delta_k}\;\xi_k.
}
$$

Every quantity is native to the sampler: $b_k,b_{k+1}$ come straight from the schedule ($b=\alpha$ under the invert convention), $\Delta_k/b_k$ is the existing ODE step size, and $\mathrm{vel}=p_\theta E-X$ is the model direction. The injected noise vanishes as $b\to0$ and the drift contains no singular factors.

(The §9 regular family corresponds to $a(b)=2\lambda b$: drift $\big(1+\lambda(1-b)\big)\frac{\mathrm{vel}}{b}-\lambda X$, diffusion $\sqrt{2\lambda b\,\Delta}$.)
