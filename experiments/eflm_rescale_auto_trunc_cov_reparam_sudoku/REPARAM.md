# EFLM Bayes-reparametrised readout ("Gaussian horosphere") — response, math, implementation, plan

Written 2026-10-01 for the agent that will run `experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku`.
Repo: `/home/sc3379/workspace/research/s-flm-dev1/s-flm`, branch `claude/seflm_cov_reparam`, HEAD `7778806`
(merge of `main` into the branch; the HBFM horosphere code is now present in this checkout at `algo.py:953-968`).
All line numbers below are for that HEAD. Spec: `setup.md` in this folder (restored copy of the user's spec).

Question answered here: *can EFLM, like HBFM's `forward_type=horosphere`, reparametrise the model's
$q_\theta(x\mid z_t)$ as $q(z_t\mid x)\,q_\theta(x)$ so that the network only predicts $q(x)$?*

---

## 0. TL;DR

- **Yes, and it is the same operation.** HBFM's horosphere readout is Bayes' rule with the analytic bridge likelihood
  added to the logits. For EFLM the forward process is Gaussian, so the analytic term is the Gaussian log-likelihood
  $\log q(z_t\mid x)=-\lVert z_t-\alpha_t e_x\rVert^2/(2 b_t^2)$. The readout becomes
  $\log q_\theta(x\mid z_t)=\operatorname{log\_softmax}\bigl[\ell_\theta(z_t)+\beta_t\langle z_t,e_x\rangle-\tfrac{\beta_t\alpha_t}{2}\lVert e_x\rVert^2\bigr]$,
  $\beta_t=\alpha_t/b_t^2$. With `rho_min == rho_max = R` (every sudoku run) the norm term is constant and drops.
- **The code already exists** as `SphereDiT._plaid_bias` (`models/sphere_dit.py:242-262`, Plaid's Eq. 44) but is wired only
  for LangFlow (`sphere_dit.py:292`: `if self.logit_bias and lf is not None and lf.r > 0.0`). It uses the RAW
  embedding table and lives inside the backbone. For EFLM do **not** reuse that path; implement the term in
  `EFLM._process_model_output` (`algo.py:412-414`) exactly the way HBFM does it (`algo.py:953-968`), with the
  radius-rescaled table `EFLM._sc_embed_table()` (`algo.py:512-515`).
- **"Only predicts $q(x)$" is exact at the optimum, not an architectural constraint.** Because the noise is independent
  per position given $x_j$, $q(x_j\mid z_t)\propto q(z_t^j\mid x_j)\,q(x_j\mid z_t^{-j})$ holds exactly, so the CE optimum
  of the residual logits is $\log q(x_j\mid z_t^{-j})+c$: the context prior. The network still sees $z_t^j$; it just has
  nothing left to learn from it. Same as HBFM.
- **Honest prior on the effect size.** HBFM horosphere vs naive on sudoku hard is $+1.6$ pt at 3 seeds
  ($61.7\pm0.4$ vs $60.1\pm2.4$, `claude_test_hbfm_rlf_sudoku`), not the $+9$ pt single-seed number quoted earlier;
  on TinyStories it ties. EFLM may differ because its 512-d learned embedding geometry is harder for the network to
  internalise than HBFM's 9-d random codebook, which is exactly what this experiment tests. Decision bar: $>5$ pt on a
  3-seed mean at matched config.
- Cost: one `[B,L,d]x[d,V]` matmul per forward (V=12 here), i.e. free. Sampler, self-conditioning and the final decode
  pick the change up automatically because all of them consume `model.forward`.

---

## 1. Framing: what the horosphere readout does, and the EFLM analogue

HBFM (`algo.py:953-968`): the backbone emits a residual $\ell_\theta(z_t)\in\mathbb R^{V}$; the readout adds the
closed-form log-density of the bridge state under each word, $\mathrm{horo}_v(z_t)=\log P(z_t,\xi_v)$ (Poisson kernel /
Busemann function), divides it by the sampling temperature when one is present, and returns
$\operatorname{log\_softmax}(\ell_\theta+\mathrm{horo})$. The CE loss is unchanged, so the network's target becomes the
prior. `numeric/horosphere.py` docstring: "softmax(horosphere_dists + log p) is exactly the Bayes posterior".

EFLM has the same three ingredients:

| ingredient | HBFM | EFLM (this experiment) |
|---|---|---|
| state | $z_t\in(\mathbb H^3)^3$, Poincaré coordinates | $z_t\in\mathbb R^{512}$, `xt` in rescaled-embedding scale |
| likelihood $q(z_t\mid x)$ | Poisson kernel $P(z_t,\xi_x)$, $t$-free | $\mathcal N(\alpha_t e_x,\,b_t^2 I)$, depends on $t$ through $\alpha_t,b_t$ |
| word table | `self.word_embedding` = `backbone.sphere_embed.weight` (directions only) | `self._sc_embed_table()` = `rescale_radius(sphere_embed.weight, rho_min, rho_max)` |
| hook | `HyperbolicBoundaryFM._process_model_output` | `EFLM._process_model_output` (`algo.py:412-414`, currently `model_output.float().log_softmax(-1)`) |
| temperature | geometry term divided by `context.temperature` | same |
| time conditioning | `False` (posterior is $t$-free) | keep `True` (posterior depends on $t$ via $\beta_t$; the residual's target also depends on $t$) |

---

## 2. Math

### 2.1 Forward process as coded

`EFLM.q_xt` (`algo.py:420-434`) with `_sample_prior` = `randn_like` (`algo.py:416-418`) and `_lerp`
(`algo.py:462-477`, `out = (1-a)*clean + a*noisy` with `a = slerp_t`):

$$
z_t = (1-b_t)\,e_x + b_t\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I_d),\qquad
b_t=\begin{cases}1-\alpha_t & \texttt{invert\_time\_convention=false (all sudoku scripts)}\\ \alpha_t & \texttt{true}\end{cases}
$$

$e_x=$ `get_rescaled_embeddings(x, rho_min, rho_max)` (`models/sphere_dit.py:203-233`); with `rho_min == rho_max = R`,
$\lVert e_v\rVert=R$ for every $v$. `prior_cov` is **not** used in `q_xt` (only in the sampler's initial state), so the
train-time noise is exactly unit-variance. Prompt positions (`valid_tokens == 0`) are pinned to the clean $e_x$.

Hence, writing $a_t:=1-b_t$ (the signal coefficient; $=\alpha_t$ under the MDLM convention):

$$
q(z_t\mid x=v)=\mathcal N\!\bigl(a_t e_v,\;b_t^2 I_d\bigr),\qquad
\log q(z_t\mid v)=-\frac{\lVert z_t\rVert^2}{2b_t^2}+\underbrace{\frac{a_t}{b_t^2}}_{\beta_t}\langle z_t,e_v\rangle-\frac{a_t^2}{2b_t^2}\lVert e_v\rVert^2 .
$$

### 2.2 Bayes posterior and the reparametrised readout

The first term is $v$-independent and drops in the softmax:

$$
q(v\mid z_t)=\operatorname*{softmax}_v\Bigl[\log q(v)+\underbrace{\beta_t\langle z_t,e_v\rangle-\tfrac{\beta_t a_t}{2}\lVert e_v\rVert^2}_{\ell^{\mathrm{geo}}_v(z_t)}\Bigr],
\qquad \beta_t=\frac{a_t}{b_t^2}.
$$

Define the model as $\log q_\theta(v\mid z_t)=\operatorname{log\_softmax}_v\bigl[\ell_{\theta,v}(z_t)+\ell^{\mathrm{geo}}_v(z_t)\bigr]$.
Then:

- **Loss unchanged.** `nll_per_token` (`algo.py:442-460`) is CE on the readout, so it becomes the denoising CE of the
  exact Bayes posterior with learned prior; optional `snr_weighted_ce` / `noise_sched_cov_weight` multiply it as before.
- **Residual target.** With independent per-position noise, $z_t^j\perp(x_{-j},z_t^{-j})\mid x_j$, so exactly
  $q(x_j=v\mid z_t)\propto q(x_j=v\mid z_t^{-j})\,q(z_t^j\mid v)$ and the CE optimum is
  $\ell_{\theta,v}(z_t)=\log q(x_j=v\mid z_t^{-j})+c(z_t)$: the context prior. That is the precise meaning of
  "the model only predicts $q(x)$". It is not enforced architecturally (the network still receives $z_t^j$), and it
  need not be: the optimum is the same.
- **Zero-init head.** `DDiTFinalLayer` zero-initialises weight and bias (`models/dit.py:553-567`), so at step 0
  $\ell_\theta\equiv 0$ and the readout is already the exact uniform-prior Bayes posterior
  $\operatorname{softmax}_v[\beta_t\langle z_t,e_v\rangle]$ (pinned norms). Naive EFLM starts from the uniform
  distribution and has to learn this geometry from scratch. This is the mechanism by which the reparametrisation can
  help: on sudoku the hard part is the combinatorial prior, and the network's capacity is no longer spent on the
  Gaussian likelihood.
- **Pinned norms.** With $\lVert e_v\rVert=R$ the second term of $\ell^{\mathrm{geo}}$ is $v$-constant; keep it in
  the code anyway (it costs nothing and makes `rho_min != rho_max` correct), exactly as `_plaid_bias` does.

### 2.3 Autonomous clock, cov weight, truncation

`Autonomous` (`noise_schedules.py:69-94`): $\alpha_t=1-(1-\epsilon)e^{-\tau_{\max}(1-t)}$, so with
$\tau:=\tau_{\max}(1-t)$ the noise fraction is $b=(1-\epsilon)e^{-\tau}$ and

$$
\beta_\tau=\frac{1-b}{b^2}=\frac{(1-(1-\epsilon)e^{-\tau})\,e^{2\tau}}{(1-\epsilon)^2}\;\le\;\frac{e^{2\tau_{\max}}}{(1-\epsilon)^2}.
$$

The coefficient is bounded on the trained range (e.g. $\tau_{\max}=0.5\Rightarrow\beta\le 2.72$; $\tau_{\max}=3\Rightarrow\beta\le 403$).
Compute $\ell^{\mathrm{geo}}$ in float32; no chunking needed at $V=12$.

`noise_sched_cov_weight` (commit `9b09ed3`, `algo.py:454-458`): `ce_loss *= |alpha_t'|`. On the autonomous clock
$|\alpha'_t|=(1-\epsilon)\tau_{\max}e^{-\tau}=\tau_{\max}\,b_\tau$, so it converts the uniform-in-$\tau$ objective into the
uniform-in-$\alpha$ one (the MDLM ELBO weighting; density $\propto e^{-\tau}$ in heat time), i.e. it moves training mass
toward the noisy end. Under the Gaussian readout the residual at the noisy end is almost purely the context prior, so
the two knobs are orthogonal in code but not in effect (hypothesis H3 below). It is mutually exclusive with
`snr_weighted_ce` (`algo.py:393-396`).

Truncation on the autonomous clock is $\tau_{\max}$ itself (noise-fraction floor $e^{-\tau_{\max}}$);
`noise.alpha_max` also works through `TruncatedScheduleWrapper` but is the log-linear knob.

### 2.4 Decode point is R-invariant under the Gaussian readout

`alpha_star_euclidean` (`noise_schedules.py:112-146`): $z_R=\frac{1}{R}\sqrt{2\ln\frac{2(V-1)}{\delta}}$,
$\alpha^*=\frac{z_R}{1+z_R}$, $\tau^*=\ln(1+z_R)$. At the decode point $b^*=1/(1+z_R)$, $\beta^*=z_R(1+z_R)$, and the
target's tilt is $\beta^* a^* R^2 = z_R^2R^2 = 2\ln\frac{2(V-1)}{\delta}$: **independent of R** ($10.79$ nats for $V=12$,
$\delta=0.1$), while the impostor noise term $\beta^* b^* R\,\langle\varepsilon,\hat e\rangle$ has std $z_R R=3.28$. So
with the exact readout the margin at $\tau^*$ is the same for every $R$; the R-dependence that remains in the naive runs is
the network's difficulty in learning the geometry, which the reparametrisation removes. This is the sharpest prediction of
the experiment (H2): the Gaussian readout should flatten the accuracy-vs-R curve and allow $\tau_{\max}$ to move.

### 2.5 $\tau^*(R)$ for sudoku ($V=12$, $\delta=0.1$; $C=\sqrt{2\ln 220}=3.2844$, $z_R=C/R$)

`setup.md` quotes V=50257 values (0.840/0.397/0.158); for sudoku use these:

| R | 0.05 | 0.1 | 0.5 | 1 | 5 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|---|
| $z_R$ | 65.69 | 32.84 | 6.569 | 3.284 | 0.6569 | 0.4106 | 0.2053 | 0.1173 |
| $\alpha^*=z/(1+z)$ (log-linear `alpha_max`) | 0.985 | 0.970 | 0.868 | 0.767 | 0.396 | 0.291 | 0.170 | 0.105 |
| $\tau^*=\ln(1+z)$ (autonomous `tau_max`) | 4.200 | 3.522 | 2.024 | 1.455 | 0.505 | 0.344 | 0.187 | 0.111 |

R=1 matches `tests/test_truncation_bounds.py:38` (0.767). Prior evidence (`experiments/eflm_rescale_auto_sudoku`):
the optimum $\tau_{\max}$ tracked $\tau^*(R)$ (0.505/0.344/0.187 predicted; 0.5/0.25/0.1 measured best for R=5/8/16) and
$\tau_{\max}\ge1$ at R=16 collapsed (24.3 → 5.9 → 0.4 % for $\tau_{\max}$ 1/2/4). The small-R rows need $\tau_{\max}$ 2–4,
i.e. the regime that collapsed at R=16; treat R ∈ {0.05, 0.1, 28} as single-seed probes, not main-grid cells (§6).

### 2.6 Gradient paths (what the readout changes in training dynamics)

$\ell^{\mathrm{geo}}$ depends on the embedding table through $\langle z_t,e_v\rangle$ and $\lVert e_v\rVert^2$, so CE now
reaches `sphere_embed` through a tied-head-like path in addition to the existing path through `q_xt` (`xt` is a
differentiable function of $e_x$; nothing is detached). With pinned norms the new path can only rotate directions
(HBFM's horosphere term also back-propagates into its codebook, through a custom backward). Keep an ablation flag to
detach the table inside $\ell^{\mathrm{geo}}$ (H4). Do **not** pin or detach by default: mirror HBFM.

### 2.7 What this is and is not, relative to the ELBO

The reparametrisation changes the parametrisation of $q_\theta(x\mid z_t)$, not the objective: training is still the
heat-time-integrated denoising CE, which is a surrogate and not a likelihood bound (see the analysis page cited in §9,
§8). It therefore cannot be compared to MDLM in bits; the metric here is the sudoku solve rate.

---

## 3. What already exists in dev1 (read before coding)

| item | where | status |
|---|---|---|
| Gaussian log-likelihood logit bias | `models/sphere_dit.py:242-262` `_plaid_bias(z_gamma, alpha, sigma, r)`; `coef1 = r*alpha/sigma^2` on $\langle z,e_v\rangle$, `coef2 = r*alpha^2/(2 sigma^2)` on $\lVert e_v\rVert^2$ | correct formula, but uses the RAW `sphere_embed.weight` (wrong for rescaled EFLM), is computed in bf16 logits dtype, lives inside the backbone (so it is divided by temperature in `trainer_base.forward:308-311` automatically), and is gated on `getattr(config.algo,'logit_bias',False)` + a context carrying `.alpha/.sigma/.r` (only `LangFlowContext`, `algo.py:518-524`). Passing EFLM's contexts with `logit_bias=true` would raise `AttributeError`. **Do not reuse; reimplement at the readout.** |
| Readout hook | `algo.py:412-414` `EFLM._process_model_output(model_output, xt, sigma, context)` → `model_output.float().log_softmax(-1)` | receives `xt` (float32, rescaled scale) and processed `sigma` (1-D, $=-\log\alpha_t$ if `time_conditioning` else zeros) |
| Caller | `trainer_base.py:302-317` `forward()`: `_process_sigma` → backbone → `/ context.temperature` (only if the context has a `temperature` field, i.e. at sampling) → `_process_model_output` | train context = `TrainingContext` (`trainer_base.py:28-31`: `kv_cache`, `z_sc`); sample context = `EFLMContext` (`samplers.py:958-961`: `temperature`, `z_sc`). `'temperature' in context` works via `dataclass_patch.__contains__`, which checks declared field names, so new context data must be a declared field |
| Rescaled table | `algo.py:512-515` `EFLM._sc_embed_table()` | the `[V,d]` mean set of $q(z_t\mid v)$; the sampler already uses it for the velocity (`samplers.py:1200`) |
| Train-time $\alpha_t$ | `algo.py:479-507` `EFLM.nll`: `dalpha_t, alpha_t = self.noise(t)`, `[B,1]`; `self._self_cond_pass(...)` then `self.forward(...)` | set `context.alpha_t` **before** `_self_cond_pass` so the self-cond pass also sees the Bayes readout |
| Sample-time $\alpha_t$ | `samplers.py:1180-1186` `EFLMSampler.step`: `_, alpha_t = model.noise(t)` (0-d), `sigma_t` `[1,1]`, `EFLMContext(temperature, z_sc)`, `model.forward(...)` | velocity `elfm_compute_velocity` (`samplers.py:963`): $v=p^\top E-x$ from the readout probabilities; `_last_step_decode` (`samplers.py:1114`) argmax/sample of the readout |
| HBFM template | `algo.py:953-968` (`forward_type == "horosphere"` branch) | copy its structure: compute term from `xt`, divide by temperature, add, `log_softmax` |
| cov weight | `algo.py:384, 393-396, 454-458`; `configs/algo/eflm.yaml:19` | done on this branch |
| Scripts | `scripts/train/sudoku/eflm_rescale_auto.sh`, `eflm_rescale_auto_adaptive.sh`, `scripts/sample/sudoku/eflm_rescale_auto.sh` | knobs `RHO TAU_MAX SNR_CE LR SEED DIFFICULTY OUTPUT_DIR DEVICES` (+adaptive: `refit_every=50`, `buffer=50*256`, `ema=0.9`, `uniform_mix=1e-3`); **no** knob for `noise_sched_cov_weight`, `alpha_max` or a readout flag |
| Conventions | `Agent.md` (repo root) | `EXPERIMENT.md` + `RESULTS.md` per project; `sweep.py` orchestrates `scripts/train|sample/sudoku/*.sh` only, idempotent on `eval/results.json` + `squeue`; partition `thickstun,desa`, `--exclude=desa-compute-01`; all compute on compute nodes; no git operations without the user's approval |

---

## 4. Implementation spec (dev1)

### 4.1 Config — `configs/algo/eflm.yaml`

```yaml
forward_type: naive                # naive: plain logits | gaussian: logits are a residual on log N(z_t; a_t e_v, b_t^2 I)
gaussian_readout_detach: false     # detach the embedding table inside the Gaussian term (ablation H4)
gaussian_readout_warmup_steps: 0   # Plaid-style ramp r: 0 -> 1 over these steps; 0 = full term from step 0 (HBFM has none)
```

### 4.2 Contexts — declare an `alpha_t` field

`trainer_base.py:28-31` `TrainingContext`: add `alpha_t: torch.Tensor | None = None  # [B,1] signal level of the noised batch`.
`samplers.py:958-961` `EFLMContext`: add the same field. Both classes use `dataclass_patch.dataclass`, so the field is
visible to `'alpha_t' in context`.

### 4.3 `algo.py` — `EFLM`

`__init__` (`algo.py:375-387`): read `forward_type`, `gaussian_readout_detach`, `gaussian_readout_warmup_steps`.

`_validate_configuration` (`algo.py:388-410`): `forward_type in {'naive','gaussian'}`; if `gaussian` and not
`time_conditioning`, still fine because $\alpha_t$ comes from the context (see 4.4), but raise if neither a context
`alpha_t` nor a usable `sigma` is available at call time (defensive check inside the hook).

`nll` (`algo.py:479-507`): right after `alpha_t = alpha_t.unsqueeze(-1)` add `context.alpha_t = alpha_t` (before
`_self_cond_pass`).

`_process_model_output` (`algo.py:412-414`) becomes:

```python
def _process_model_output(self, model_output, xt, sigma, context=None):
  logits = model_output.float()
  if self.forward_type == 'gaussian':
    r = self._gaussian_readout_r()                       # 1.0 unless warmup is set
    if r > 0.0:
      geo = self._gaussian_loglik(xt, self._alpha_for_readout(sigma, context))   # [B, L, V] float32
      if context is not None and 'temperature' in context and context.temperature != 1.0:
        geo = geo / context.temperature                  # forward() already divided the residual; temper the whole posterior (HBFM pattern)
      logits = logits + r * geo
  return logits.log_softmax(-1)

def _alpha_for_readout(self, sigma, context):
  if context is not None and 'alpha_t' in context and context.alpha_t is not None:
    return context.alpha_t.reshape(-1)                   # [B] or [1]
  if self.time_conditioning and sigma is not None:
    return torch.exp(-sigma.reshape(-1))                 # sigma = -log alpha_t (trainer_base._sigma_from_alphat)
  raise RuntimeError('gaussian readout needs alpha_t: pass it via context or enable time_conditioning')

def _gaussian_loglik(self, xt, alpha_t):
  # log N(xt; a e_v, b^2 I) up to the v-independent -||xt||^2 / (2 b^2); same convention as q_xt / _lerp.
  b = alpha_t if self.invert_time_convention else 1.0 - alpha_t
  b = b.clamp(min=self.eps).to(torch.float32).reshape(-1, 1, 1)          # [B,1,1] or [1,1,1]
  a = 1.0 - b
  E = self._sc_embed_table()                                               # [V, d] radius-rescaled, the table q_xt used
  if self.gaussian_readout_detach:
    E = E.detach()
  E = E.float()
  inner = torch.einsum('bld,vd->blv', xt.float(), E)                       # <z_t, e_v>
  norm_sq = (E ** 2).sum(-1)                                               # [V]; constant across v when rho_min == rho_max
  return (a / b ** 2) * inner - (a ** 2 / (2.0 * b ** 2)) * norm_sq
```

Notes for the implementer:

- `alpha_t` is `[B,1]` at train time and 0-d / `[1,1]` at sample time; `reshape(-1,1,1)` broadcasts both.
- `b.clamp(min=self.eps)` (`algo.eps` 1e-6): never active on the autonomous clock (floor $e^{-\tau_{\max}}$), only a guard.
- Use `xt` as given (train: `q_xt` output, float32; sample: `state.xt`, float32). Do **not** use the self-conditioned
  input; `xt` in the hook is the raw state, as in HBFM.
- Do not touch `nll_per_token`, the sampler, or `_sc_embed_table`. The sampler (`samplers.py:1180-1227`) and the
  self-cond pass (`algo.py:212-240`) call `model.forward`, so they inherit the readout; the velocity $p^\top E-x$ then uses
  the Bayes posterior automatically.
- Warmup `r` (optional): `_gaussian_readout_r = 1.0 if w <= 0 else min(1.0, global_step / w)` as in
  `LangFlow._logit_bias_r` (`algo.py:561-566`); at eval `global_step` is the checkpoint's, so a finished run has `r = 1`.
  Default 0 (off). Note that with `r < 1` the readout is no longer the exact posterior, so prefer the default.

### 4.4 Sampler — `samplers.py:1180-1186`

```python
_, alpha_t = model.noise(state.t_schedule[state.step_idx])
sigma_t = model._sigma_from_alphat(alpha_t).reshape(-1, 1)
context = EFLMContext(temperature=self.temperature, z_sc=state.z_sc,
                      alpha_t=alpha_t.reshape(1, 1))
log_p = model.forward(xt=state.xt, sigma=sigma_t, context=context)
```

Nothing else changes: prefix pinning (`_project_prefix`), velocity, last-step decode, SDE branch (`eta > 0`) all
consume `log_p`.

### 4.5 Scripts — new stems (one run per script; the sweep parameterises via env)

`scripts/train/sudoku/eflm_rescale_auto_trunc_cov_reparam.sh` = `eflm_rescale_auto.sh` plus knobs

```bash
COV_W="${COV_W:-false}"        # algo.noise_sched_cov_weight
FWD="${FWD:-naive}"            # algo.forward_type  (naive | gaussian)
DETACH="${DETACH:-false}"      # algo.gaussian_readout_detach
ADA="${ADA:-false}"            # false: noise=autonomous ; true: noise=autonomous-adaptive + refit_every=50 buffer=$((50*256)) ema=0.9 uniform_mix=1e-3
ALPHA_MAX="${ALPHA_MAX:-null}" # optional noise.alpha_max on top of tau_max (not used in the main grid)
```

and the matching `scripts/sample/sudoku/eflm_rescale_auto_trunc_cov_reparam.sh` passing **the same** overrides
(`sudoku_eval` rebuilds the model from CLI overrides; a checkpoint trained with `forward_type=gaussian` evaluated with the
default `naive` would silently drop the term). Keep `sampler.noise_removal=greedy`, `STEPS=180`, `VELOCITY=exact`,
`TOPK_VELOCITY=-1`. Add `wandb.project=eflm_rescale_auto_trunc_cov_reparam_sudoku wandb.group=${FWD}-${ADA}-${COV_W}
wandb.name=${RUN_NAME}` (Agent.md standard knobs; the existing eflm_* sudoku scripts never set them, everything lands in
project `debug`).

### 4.6 Sweep — `experiments/eflm_rescale_auto_trunc_cov_reparam_sudoku/sweep.py`

Copy `experiments/eflm_rescale_auto_sudoku/sweep.py` (simple_slurm; `partition='thickstun,desa'`,
`exclude='desa-compute-01'`, `gres='gpu:1'`, 2 CPUs, 16G, 6h, `sbatch --requeue`; skip a cell whose
`eval/results.json` exists or whose tag is in `squeue`). Fix `REPO` to this checkout
(`/home/sc3379/workspace/research/s-flm-dev1/s-flm`; the auto_sudoku sweep points at `/share/thickstun/sychou/...`).
Job body: `export TMPDIR=/tmp; MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID; WANDB_CACHE_DIR=/tmp/wandb-cache-$SLURM_JOB_ID`
(the `eflm_rescale_sudoku` ENOSPC fix), then
`{knobs} OUTPUT_DIR={tdir} DEVICES=1 bash scripts/train/sudoku/<stem>.sh` and
`{knobs} CKPT_PATH={tdir}/checkpoints/last.ckpt OUTPUT_DIR={tdir}/eval DEVICES=1 bash scripts/sample/sudoku/<stem>.sh`.
Tag / job name / output dir:
`eflmrcr_fwd-{naive|gauss}_ada-{0|1}_cov-{0|1}_r-{R}_tau-{tau}_lr-{lr}_d-hard_rs{seed}` →
`outputs/eflm_rescale_auto_trunc_cov_reparam_sudoku/{tag}/eval/results.json["accuracy"]` (fraction; report ×100,
mean ± sd over seeds). Do not set `nice` in the sweep.

### 4.7 Tests — `tests/test_eflm_gaussian_readout.py` (GPU; run via `sbatch --gres=gpu:1` with `TMPDIR=/home/sc3379/tmp/<tag>_$SLURM_JOB_ID`)

1. **Exactness.** Build `EFLM` with `forward_type=gaussian`, `rho_min=rho_max=R`, both `invert_time_convention`
   values; for random `x`, several `t` (incl. $t\to$ clean end), compute `q_xt`, call `_process_model_output` with
   `model_output = zeros` and the train context carrying `alpha_t`; compare to
   `log_softmax(-||xt - a e_v||^2 / (2 b^2))` computed directly from `_sc_embed_table()`; `atol 1e-5` in float32.
2. **Unpinned norms.** Same with `rho_min != rho_max` (and `None/None`): the $\lVert e_v\rVert^2$ term must be present.
3. **Context vs sigma path.** `_alpha_for_readout` from context equals `exp(-sigma)` path when `time_conditioning=True`;
   raises when `time_conditioning=False` and no context field.
4. **Temperature.** With `EFLMContext(temperature=T)`, the readout equals `log_softmax((logits + geo)/T)`.
5. **Zero-init init state.** A freshly built model's readout on `q_xt` samples equals the uniform-prior Bayes posterior
   and its CE at a clean-end $t$ is $\ll \log V$ (naive gives $\log V = 2.485$).
6. **Sampler smoke.** `EFLMSampler` 8 steps, `velocity=exact`, `top_k_velocity=-1`, `forward_type=gaussian`, with a
   prefix; shapes and finiteness; `_last_step_decode` returns prefix tokens unchanged.
7. **Eval rebuild.** Train 20 steps with `forward_type=gaussian`, run the sample script path with and without `FWD=gaussian`
   and assert the readouts differ (guards the "flag not passed at eval" failure).
8. **Naive unchanged.** `forward_type=naive` reproduces the current `model_output.float().log_softmax(-1)` bit-for-bit.

Also run the existing `tests/test_eflm_sde.py`, `test_radius_rescale.py`, `test_self_conditioning.py`,
`test_autonomous_flow.py`, `test_truncation_bounds.py` (unchanged behaviour with the default `naive`).

---

## 5. Experiment plan (within `setup.md`; staged because the full factorial is ~1700 cells)

Axes from `setup.md`: R ∈ {0.05, 0.1, 0.5, 1, 5, 8, 16, 28}, LR ∈ {3e-4, 5e-4, 1e-3}, `noise_sched_cov_weight` ∈ {F, T},
adaptive ∈ {off, on}, truncation $\tau_{\max}=\tau^*(R)$ "varied a little", 3 seeds, 20k steps, bs 256, tiny DiT, exact
velocity, top-k −1, 180 steps, greedy. New axis: `forward_type` ∈ {naive, gaussian}. The `naive` arm at matched
settings **is** the control for every `gaussian` cell; no external baseline is needed for the decision, but see §7 for
the numbers to beat.

| stage | cells | content |
|---|---|---|
| 0 smoke | 2 | `gaussian` and `naive`, R=16, $\tau_{\max}$=0.5, ada on, LR 1e-3, seed 1, `MAX_STEPS=500`: loss curves, init CE ≈ 0 vs $\log 12$, eval runs |
| 1 pilot (seed 1) | 80 | fwd {naive, gaussian} × cov {F, T} × ada {off, on} × R {0.5, 1, 5, 8, 16} × LR {5e-4, 1e-3} at $\tau_{\max}=\tau^*(R)$ |
| 1b R-extremes (seed 1) | 12 | fwd × cov at R {0.05, 0.1, 28}, $\tau_{\max}=\tau^*(R)$, ada on, LR 1e-3 (expect small-R collapse; this is a probe) |
| 2 truncation (seed 1) | ~40 | for the best (cov, ada, LR) per R from stage 1: $\tau_{\max}\in\{0.7, 1.4\}\tau^*(R)$ × fwd |
| 3 replication | ~48 | 3 seeds on the top ~8 (R, $\tau$, cov, ada, LR) configs **in both arms** (fwd matched pairs) |
| 4 ablations (3 seeds) | ~12 | `gaussian_readout_detach=true` on the best gaussian cell; `time_conditioning=False` on the best gaussian cell (t-free residual; the geometry term carries $t$) |

~190 cells × ~3 h ≈ 570 GPU-h; at ~12 concurrent ≈ 2 days wall. LR 3e-4 is dropped from the pilot (second-order in
all previous EFLM sudoku sweeps; add it in stage 3 if a winner sits at 5e-4).

Pre-registered decision rule (same as the previous sudoku projects): a `gaussian` cell beats its matched `naive` cell
if the 3-seed mean is higher by **> 5 pt and in ≥ 2 of 3 seeds**; < 2 pt is a tie. Per-seed sd on this task is 1.4–5
pt (autonomous clock) and up to 8 pt (log-linear).

---

## 6. Hypotheses and predictions

- **H1 (main).** `gaussian` ≥ `naive` at matched (R, $\tau$, cov, ada, LR), with the gain largest at small R and at
  early checkpoints (5k/10k), because the network no longer spends capacity on the Gaussian likelihood. Honest prior
  from HBFM: +1.6 pt (3 seeds) on sudoku; the EFLM-specific reason it could be larger is the learned 512-d geometry.
  Null result = tie within 2 pt everywhere.
- **H2 (R-invariance / truncation).** With the exact readout the decode-point margin is R-free (§2.4), so (i) the
  accuracy-vs-R curve flattens relative to naive, and (ii) the optimal $\tau_{\max}$ shifts below $\tau^*(R)$
  (the $0.7\tau^*$ cells win or tie), since the readout is already exact at the decode point and later training time is
  pure variance.
- **H3 (cov weight interaction).** `noise_sched_cov_weight` moves training mass to the noisy end, where under
  `gaussian` the residual target is the pure context prior; predict cov helps `gaussian` more (or hurts it less) than
  `naive`. If cov hurts both, that is consistent with the TinyStories finding that flatter time weights win on
  sample quality.
- **H4 (ablations).** Detaching the table inside the geometry term changes nothing beyond noise when norms are pinned.
  `time_conditioning=False` costs little under `gaussian` (the geometry term carries $t$; the context prior's $t$-dependence
  is recoverable from the context states' norms $\approx\sqrt{a_t^2R^2+b_t^2 d}$), but is not expected to help.

Failure modes to watch: (a) loss near zero at the clean end makes the loss scale smaller; with `cov` on, the
`eflm_rescale_auto_sudoku` report saw single-seed divergence at LR 1e-3 when the loss scale shrank — keep LR 5e-4 in the
pilot; (b) a `gaussian` checkpoint evaluated without `FWD=gaussian` (test 7); (c) small-R cells need
$\tau_{\max}$ 2–4 where the autonomous clock collapsed at R=16 — report them as probes.

---

## 7. Reference numbers (sudoku hard-30, 2000 boards, exact velocity, top-k −1, 180 steps, greedy, 20k steps, bs 256, 3 seeds)

| config | solve rate (%) | source |
|---|---|---|
| EFLM auto + ada, R=16, $\tau_{\max}$ 0.5, CE, LR 1e-3 (best EFLM on the autonomous clock) | 52.0 ± 1.4 | `experiments/eflm_rescale_auto_sudoku/RESULTS.md` |
| EFLM auto naive, R=8, $\tau_{\max}$ 0.5, CE, LR 5e-4 | 45.1 ± 1.3 | same |
| EFLM auto naive, R=5, $\tau_{\max}$ 0.25, SNR-CE, LR 1e-3 | 51.1 ± 3.5 | same |
| EFLM log-linear + ada, R=5, LR 1e-3 (best EFLM overall) | 53.9 ± 4.1 | `experiments/eflm_rescale_sudoku/RESULTS.md` |
| EFLM log-linear + trunc + ada (R=8, $\alpha_{\max}$ 0.291) | 45.6 ± 1.2 (trunc+ada does not compose on sudoku) | same, round 4 |
| HBFM incumbent (exp rate 3, K −0.5, horosphere, LR 3e-4), Ampere harness | 61.1 ± 1.8 / 61.7 ± 0.4 | `claude_test_hbfm_sudoku` round 2 / `claude_test_hbfm_rlf_sudoku` (primary repo) |
| HBFM naive readout, same | 60.1 ± 2.4 | `claude_test_hbfm_rlf_sudoku` |

Those EFLM runs were executed from other checkouts (`/share/thickstun/sychou/...`, `s-flm-dev1` of an earlier date);
their raw outputs may not be reachable from here, which is why the matched `naive` arm is run inside this sweep.

---

## 8. Caveats and rules for the implementing agent

- Run everything, including tests, on compute nodes (`srun`/`sbatch`), never on the login node; `--exclude=desa-compute-01`.
- No git commits/pushes without the user's approval; the user edits this repo in parallel, so re-read (`md5sum`/mtime)
  any file you are about to modify.
- Pass **every** new `algo.*` flag identically in the train and sample scripts.
- `setup.md`'s $\alpha^*$ numbers are for V=50257; use §2.5.
- The reparametrisation does not make `val/ppl` or the CE a likelihood; do not compare it to MDLM bits.
- Keep `forward_type: naive` as the yaml default so every existing EFLM script is unchanged (test 8).

---

## 9. References (for the math and the HBFM precedent)

- HBFM horosphere readout: `algo.py:953-968` (this checkout, after the merge) and `numeric/horosphere.py` docstring
  ("softmax(horosphere_dists + log p) is exactly the Bayes posterior").
- Plaid logit bias: `models/sphere_dit.py:242-262`; Gulrajani & Hashimoto, *Likelihood-Based Diffusion Language Models*
  (2023), Eq. 44.
- Euclidean-bridge posterior derivation and the CE-vs-ELBO analysis (sections 7–8 of the rendered page
  "Hyperbolic Bridge Angular KL", https://claude.ai/artifact/EYoVyjaVwoRxfnDhmxE94X) and
  `/home/sc3379/workspace/research/s-flm-dev/s-flm/experiments/analysis/hbfm_horosphere_skew_sampler_2026-09-30.md` §7.
- HBFM vs EFLM on sudoku hard, protocol comparability, and the 3-seed naive-vs-horosphere re-measurement:
  `experiments/claude_test_hbfm_rlf_sudoku/RESULTS.md` (primary repo `/share/desa/nfs02/sc3379/workspace/research/s-flm`).
