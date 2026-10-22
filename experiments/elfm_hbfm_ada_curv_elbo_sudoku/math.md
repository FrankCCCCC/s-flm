# The ELBO of EFLM and HBFM

Notation follows `slides/oct07_2026` (and `slides/aug26_2026`, `experiments/hbfm_rlf_ada_curv/math.md`):
a word $v$ with embedding $\mathbf e_v$, a sequence $x=(x^1,\dots,x^L)$, a latent $z_t$ (one block per
position), the model $f_\theta(z_t,t)\in\Delta^V$ per position. References: VDM (Kingma et al., 2021),
Plaid (Gulrajani & Hashimoto, 2023), LangFlow (Chen et al., 2026).

**Result in one line.** For both models the NELBO is
$\mathcal L=\mathcal L_{\mathrm{prior}}+\mathcal L_{\mathrm{diff}}+\mathcal L_{\mathrm{rec}}$, estimated
with one noise level per sequence; $f_\theta=q(\cdot\mid z_t)$ (the Bayes posterior) minimizes every
term and makes $\mathcal L=\mathsf H(q_{\mathrm{data}})$. The CE loss the code trains today has the same
minimizer (posterior matching, LangFlow §3.1) but is not a bound; `algo.loss_type=elbo` trains and
reports the bound.

---

## 0. The template (slides "Discrete-Time Bridge NELBO")

On a grid $0=t_0<\dots<t_T=\mathcal T$ with target transitions $Q=q(z_{t_{i+1}}\mid z_{t_i},x)$ and learned
transitions $P_\theta=\sum_{v'}f_\theta(z_{t_i})_{v'}\,q(z_{t_{i+1}}\mid z_{t_i},v')$,

$$
-\log p_\theta(x)\le
\underbrace{\mathrm{KL}\big(q(z_{\text{noisy}}\mid x)\,\|\,p(z_{\text{noisy}})\big)}_{\mathcal L_{\mathrm{prior}}}
+\underbrace{\sum_i\mathbb E\big[\mathrm{KL}(Q\|P_\theta)\big]}_{\mathcal L_{\mathrm{diff}}}
+\underbrace{\mathbb E\big[-\log f_\theta(z_{\text{clean}})_{x}\big]}_{\mathcal L_{\mathrm{rec}}} .
$$

**Tightness.** With $f_\theta=q(\cdot\mid z_t)$, $P_\theta=q(z_{t_{i+1}}\mid z_{t_i})$ and the Bayes identity
telescopes along the chain:
$\sum_i\log\frac{q(z_{t_{i+1}}\mid z_{t_i},x)}{q(z_{t_{i+1}}\mid z_{t_i})}
=\log\frac{q(x\mid z_{\text{clean}})}{q_{\mathrm{data}}(x)}-\log\frac{q(z_{\text{noisy}}\mid x)}{q(z_{\text{noisy}})}$,
so

$$
\mathcal L^\star=\mathsf H(q_{\mathrm{data}})+\mathrm{KL}\big(q(z_{\text{noisy}})\,\|\,p(z_{\text{noisy}})\big).
$$

The only slack is the mismatch between the aggregate noisy marginal and the model prior; it is
exactly zero for HBFM (both start at the origin) and $\approx 5\cdot10^{-4}$ nats/token for EFLM at the
noise floor of the schedules used here (§1). **The bound is tight at the Bayes posterior whatever the
horizon**, which is why the reconstruction term must be kept when the trained range is truncated.

**Estimator.** The training loop draws one $t\sim\mathcal U[t_{\min},1]$ per sequence (`_sample_t`,
$t_{\min}$ = `training.sampling_eps`). The clean end of the trained range is $t_{\min}$ (MDLM
convention, `invert_time_convention=false`) and the noisy end $t=1$:

$$
\widehat{\mathcal L}(x)=\underbrace{\ell_{\mathrm{prior}}(x)}_{\text{closed form at }t=1}
+\underbrace{w(t)\,\ell_{\mathrm{diff}}(z_t,x)}_{t\sim\mathcal U,\ \text{one draw}}
+\underbrace{-\log f_\theta(z_{t_{\min}})_x}_{\text{a second forward pass}} ,
$$

per token, masked to the solution cells of a Sudoku board like every other loss in `trainer_base._loss`.
The second forward pass is Plaid's "learned conditional likelihood" (§3.4): the decoder of the
reconstruction term is the same categorical readout, so it carries the greedy decode of the sampler.
(Plaid splits the batch between the two terms instead; a second pass is simpler and has no allocation
hyper-parameter.)

---

## 1. EFLM: Gaussian interpolant

Forward process (`EFLM.q_xt`): $x_t=(1-b_t)\,\mathbf e_x+b_t\,\varepsilon$, $\varepsilon\sim\mathcal N(0,I_d)$,
with the noise fraction $b_t=1-\alpha_t$ increasing in $t$. In VDM's terms $\alpha^{\mathrm{VDM}}_t=1-b_t$,
$\sigma^{\mathrm{VDM}}_t=b_t$, $\mathrm{SNR}(t)=(1-b_t)^2/b_t^2$ -- a valid Markov Gaussian process since the
SNR decreases.

**Transition KL.** For $s<t$, $q(x_s\mid x_t,\mathbf e)$ is Gaussian with a mean affine in $\mathbf e$ and a
variance independent of $\mathbf e$, so (VDM App. E, for any $\alpha,\sigma$)

$$
\mathrm{KL}\big(q(x_s\mid x_t,\mathbf e_x)\,\|\,q(x_s\mid x_t,\hat{\mathbf x}_\theta)\big)
=\tfrac12\big(\mathrm{SNR}(s)-\mathrm{SNR}(t)\big)\,\|\mathbf e_x-\hat{\mathbf x}_\theta(x_t)\|^2 .
$$

Summing over the grid and letting $T\to\infty$ (VDM Eq. 17, Plaid Eq. 5):

$$
\boxed{\;
\mathcal L_{\mathrm{diff}}=\int_{t_{\min}}^{1}\underbrace{-\tfrac12\mathrm{SNR}'(t)}_{w(t)}\,
\mathbb E\,\|\mathbf e_x-\hat{\mathbf x}_\theta(x_t)\|^2\,dt,\qquad
w(t)=\frac{(1-b_t)\,|b'_t|}{b_t^{3}}\;}
$$

(`snr_weight`, the weight the `snr_weighted_ce` option already uses). Under the autonomous clock
$b_t=e^{-\tau_{\max}(1-t)}$ this is $w=\tau_{\max}(1-b)/b^2$, bounded by $\tau_{\max}(1-e^{-\tau_{\max}})e^{2\tau_{\max}}$.

**Categorical reparameterization (Plaid §3.2).** The Bayes-optimal denoiser is the posterior mean,
$\mathbb E[\mathbf e_x\mid x_t]=\sum_v q(v\mid x_t)\,\mathbf e_v$, so the model's denoiser is
$\hat{\mathbf x}_\theta=\sum_v f_\theta(x_t)_v\,\mathbf e_v=f_\theta^\top\mathbf E$ with $\mathbf E$ the
radius-rescaled table the forward process uses (`_sc_embed_table`). The quadratic is minimized in
expectation over $x\sim q(\cdot\mid x_t)$ at $f_\theta=q(\cdot\mid x_t)$ (exactly, when the $\mathbf e_v$ are
affinely independent, as $V=12$ vectors in $\mathbb R^{512}$ are).

**Endpoints.**

- Reconstruction, clean end $t_{\min}$: $\mathcal L_{\mathrm{rec}}=\mathbb E[-\log f_\theta(x_{t_{\min}})_x]$, the CE
  of the readout at noise fraction $b_{t_{\min}}$ -- under $\tau_{\max}=0.5$ that is $b=0.61$, the point
  where the sampler decodes greedily, so this term is NOT small: it is the decode posterior's CE.
- Prior, noisy end $t=1$: $p(x_1)=\mathcal N(0,cI)$ ($c$ = `algo.prior_cov`), $q(x_1\mid x)=\mathcal N((1-b_1)\mathbf e_x,b_1^2I)$,

$$
\ell_{\mathrm{prior}}=\tfrac12\Big[d\Big(\tfrac{b_1^2}{c}-1-\log\tfrac{b_1^2}{c}\Big)+\tfrac{(1-b_1)^2}{c}\|\mathbf e_x\|^2\Big],
$$

  $\approx 5\cdot10^{-4}$ nats at $b_1=1-10^{-3}$, $d=512$, $\|\mathbf e\|=8$ (the schedules' `eps` floor).

**Why not the SNR-weighted CE.** `experiments/eflm_rescale_auto_sudoku` §3 found $w(t)\cdot\mathrm{CE}$
collapses on long horizons: the CE of a discrete codebook decays exponentially in the SNR while the MMSE
$\|\mathbf e-\hat{\mathbf x}\|^2$ decays polynomially, so $w$ over-integrates the solved region. The ELBO
weights the quadratic, which is what the bound actually says.

---

## 2. HBFM: the hyperbolic bridge

Per position, product manifold $\prod_m\mathbb H^{d_m}_{K_m}$, $\kappa_m=\sqrt{-K_m}$; the bridge runs from the
origin toward the boundary point $\mathbf e_{x,m}\in\mathbb S^{d_m-1}$ of its word. Heat time $t\in[0,\infty)$,
state $(\rho_{t,m},\omega_{t,m})$ per factor, $u=\kappa\rho$, $D_v=\cosh u-\sinh u\,\langle\omega,\mathbf e_v\rangle$.

**Path KL (slides "Angular Path-KL", "Radial Path-KL").** For $t\to t+dt$ the transition KL splits into a
radial and an angular Girsanov term; summed over factors (independent Brownian motions),

$$
\ell_{\mathrm{diff}}(z_t,x)=\sum_m\frac{(d_m-1)^2\kappa_m^2}{2}\Big[
\underbrace{\Big(\sum_v\mu_v A_v\Big)^2}_{\ell_{\mathrm{rad}}}
+\underbrace{\Big\|\mathcal P_{\omega}\sum_v\frac{\mu_v\,\mathbf e_{v,m}}{D_v}\Big\|^2}_{\ell_{\mathrm{ang}}}\Big],
\qquad \mu_v=f_\theta(z_t)_v-\mathbb 1[v=x],\quad
A_v=\frac{\cosh u\,\langle\omega,\mathbf e_v\rangle-\sinh u}{D_v}.
$$

With $w_v:=A_v\,\omega+\mathcal P_\omega\mathbf e_v/D_v$ -- a unit vector, the direction of $\mathbf e_v$ seen from
$z_t$ -- the two terms are the squared radial and tangential components of one vector:
$\ell_{\mathrm{rad}}+\ell_{\mathrm{ang}}=\frac{(d-1)^2\kappa^2}{2}\|\sum_v\mu_vw_v\|^2$. This is exactly
`numeric.loss.Loss.bridge_loss_elbo_refactor` (`aug26_2026`, "Complete Physical-Time Hyperbolic ELBO"),
whose $e^{\pm u}$ split and float64 are what keep $D_v$ and $w_v$ cancellation-free (unigram2
`numerics.md` §1-3; `||w_v|| = 1` bounds the integrand by $2(d-1)^2\kappa^2$ per factor). Because $\mu$ is
linear in $f_\theta$ and the Bayes posterior is $\mathbb E[\mathbb 1_x\mid z_t]$, $f_\theta=q(\cdot\mid z_t)$
minimizes $\mathbb E_x[\ell_{\mathrm{diff}}\mid z_t]$ (slides "Bayes Optimal"). The readout
`forward` returns is the full log-posterior (horosphere term included, Invariant 1), so the loss adds no
geometry of its own.

**Importance weight.** The heat time comes from the noise schedule, $t=-\log(1-\alpha_s)/r$ for the
schedule time $s\sim\mathcal U[s_{\min},1]$ (`time_conversion`, `exp` mode of rate $r$), whose density is the
exp proposal $q(t)=r e^{-rt}$ restricted to $[t_{\min},T]$; the weight is $1/q(t)$ times the schedule's
$|\alpha'_s|$ (MDLM's change of variables), unchanged from the CE loss:

$$
\boxed{\;
\widehat{\mathcal L}_{\mathrm{diff}}=\frac{|\alpha'_s|}{r\,(1-\alpha_s)}\;\ell_{\mathrm{diff}}(z_t,x),\qquad
T=-\tfrac1r\log(1-\alpha_{s_{\min}}) .\;}
$$

Under `noise=log-linear` ($\alpha_s=\epsilon+(1-\epsilon)(1-s)$, $\epsilon=10^{-3}$) and $r=3$ the trained
heat times are $[3.3\cdot10^{-4},\,2.30]$. `unif` / `trunc_exp` modes change $q(t)$ and $T$ only.

**Endpoints.**

- Prior: zero. The bridge and the generative SDE both start at the origin, a point mass.
- Reconstruction at the horizon: $\mathcal L_{\mathrm{rec}}=\mathbb E[-\log f_\theta(z_T)_x]$ with $z_T$ the bridge
  state at $s_{\min}$ (`reconstruction_loss`, one more forward pass). For the context-free
  $(\mathbb H^3)^3$ posterior at $K=-\tfrac12$ the CE at $t=2.3$ is $\approx 0.01$ nats; for a trained model
  it is smaller still, but the measured NELBO of a trained Sudoku model is only $\sim 0.01$ nats/token
  (`hbfm_rlf_ada_curv/RESULTS.md` §4), so the term is kept rather than argued away.
- The piece of the integral below $t_{\min}$ is not sampled: it is at most
  $\sum_m2(d_m-1)^2\kappa_m^2\,t_{\min}\approx 4\cdot10^{-3}$ nats per token here (the same $\epsilon$-floor MDLM's
  bound carries at both ends). It is reported as the bound's resolution, not corrected.

**Adaptive curvature** (`algo.curvature_bound` $b>1$, `hbfm_rlf_ada_curv/math.md`). Position $i$ runs at
curvature $\lambda_iK$, i.e. the base bridge on its own clock $\lambda_it$, and its rate is
$\lambda_i\,\tilde\ell_i$ with $\tilde\ell_i$ the base-curvature rate of the dimensionless state (slides
"Learned Curvature Is a Clock": both terms are $\kappa^2$ times a curvature-free quantity). Hence

$$
\widehat{\mathcal L}_{\mathrm{diff},i}=\frac{\lambda_i}{q(t)}\,\tilde\ell_i\big(z^i_{\lambda_it}\big),\qquad
\mathcal L_{\mathrm{rec},i}=\mathbb E\big[-\log f_\theta(z_{\lambda T})_{x^i}\big] ,
$$

the same $\lambda_i/q(t)$ weight `q_xt` already returns for the CE, and the terminal state drawn by the
same `q_xt` at $\lambda_iT$. The mean is invariant to a global $\lambda$ (one clock, horizon term included)
and equal to $\mathsf H$ at the Bayes posterior for any $\lambda$; $\lambda$ is still trained on the variance
of $\widehat{\mathcal L}$ through the unchanged gradient hook (now of the full NELBO estimator).

---

## 3. Numerical stability

- HBFM: the geometry stays float64 and cancellation-free -- $\sin^2(\alpha/2)$, $\cos^2(\alpha/2)$ forms,
  $D_v=\sin^2(\alpha/2)e^{u}+\cos^2(\alpha/2)e^{-u}$, $u$ capped at `RHO_MAX` (Invariants 3-5,
  `numerics.md`); all inside `bridge_loss_elbo_refactor`. float64 tensors are exempt from bf16 autocast.
- EFLM: the $(B,L,V)\times(V,d)$ matmul $f_\theta^\top\mathbf E$ runs with autocast disabled, in float32:
  under bf16 the error $\|\mathbf e-\hat{\mathbf x}\|^2$ of a confident model ($\lesssim10^{-6}$) would be pure
  rounding noise, and $w(t)$ reaches $10^3$ at the clean end. `snr_weight` floors $b_t$ at `algo.eps`.
- The loss scales match the CE arms (HBFM: the same $1/q(t)$ weight on a rate bounded by
  $2(d-1)^2\kappa^2$; EFLM: $w\le0.52$ at $\tau_{\max}=0.5$ on a quadratic $\le(2R)^2$), so the gradient
  clip (1.0) and the LR grid are shared; the LR is re-tuned per loss (`EXPERIMENT.md`).

---

## 4. Verification (`tests/test_elbo_loss.py`)

| claim | check |
|---|---|
| §0 tightness, EFLM | unigram with the closed-form Gaussian posterior fed through `forward`: NELBO = $\mathsf H(p)$ within MC error (prior + diffusion + reconstruction, through `nll`) |
| §0 the quantity is a bound | a tempered (EFLM) / biased (HBFM) posterior gives a larger value |
| §0 tightness, HBFM | horosphere readout with zero residual = the context-free Bayes posterior, uniform words: NELBO = $\log V$ within MC error + the $t_{\min}$ floor |
| §2 adaptive curvature keeps the mean | mixed $\lambda\in[1/4,4]$ per position: same mean as $\lambda=1$ and $=\log V$ |
| CE path untouched | `loss_type=ce` returns the (weighted) CE of before; unknown types rejected |
| trainable | real DiTs under bf16 autocast: finite loss, gradients reach embeddings, trunk and curvature readout |

## 5. Implementation map

| math | code |
|---|---|
| $w(t)\|\mathbf e-f_\theta^\top\mathbf E\|^2$ | `EFLM.diffusion_loss` |
| $\mathcal L_{\mathrm{rec}}+\ell_{\mathrm{prior}}$ (EFLM) | `EFLM.endpoint_losses` |
| $\frac{|\alpha'|}{q(t)}\,\ell_{\mathrm{diff}}$ (HBFM) | `HyperbolicBoundaryFM.nll_per_token` -> `diffusion_loss` -> `Loss.bridge_loss_elbo_refactor` |
| $-\log f_\theta(z_{\lambda T})_x$ | `HyperbolicBoundaryFM.reconstruction_loss` |
| loss selection | `algo.loss_type` in `configs/algo/{eflm,hbfm}.yaml`; `LOSS_TYPE` of the sudoku scripts |
