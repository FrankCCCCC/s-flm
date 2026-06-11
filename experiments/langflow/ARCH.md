# ARCH: LangFlow + learnable Gumbel `UnifInfoSchedule`

Architecture contract for the test-writer and implementer. Authoritative inputs:
`experiments/langflow/EXPERIMENT.md` (scope/metrics/scripts), `/tmp/langflow_spec.md`
(math). Strategy 1 (full faithful) is fixed; do not re-litigate.

This document is concrete and load-bearing: signatures, shapes, stopgrad boundaries,
config keys, and the unit-testable invariants are normative. Match existing repo style
(2-space indent, `dataclass_patch.dataclass`, plain attribute access on
`config.*`, `nn.Parameter`/`register_buffer` patterns, `utils.print_nans`).

---

## 1. Scope

**In scope**
- `LangFlow(trainer_base.Diffusion)` — γ=logNSR path, Gaussian (not slerp) corruption on
  sphere embeddings, self-conditioning, Plaid logit bias, `L_CE + L_Scheduler` combined loss.
- `UnifInfoSchedule(NoiseSchedule)` — self-contained learnable **Gumbel** schedule
  (`P_mu`, `P_beta`, `H_inf`) with a trainable/fixed switch; wired into `get_noise`.
- Backbone deltas in `models/sphere_dit.py`, all flag-gated and zero-init so OFF == today's
  `SphereDiT` byte-for-byte: self-cond input projections `W_in`/`W_SC`; γ→time-embedding feed;
  Plaid logit bias.
- `LangFlowSampler` — Euler-on-γ (Algorithm 2), always-on self-conditioning.
- Configs: `configs/algo/langflow.yaml`, `configs/noise/gumbel.yaml`,
  `configs/sampler/langflow.yaml`. Reuse `small-sphere-dit`/`tiny-sphere-dit`.
- `main.py` algo dispatch + `samplers.get_sampler` dispatch + `eval.sh` `MODEL_TYPE=langflow`.
- Training scripts per EXPERIMENT.md §10 (point to it, do not duplicate).
- New W&B/log keys (§6 of EXPERIMENT.md).

**Out of scope (do NOT touch / do NOT add)**
- Theorem-3.1 ODE-NLL bound (DEFERRED, user-confirmed). Report existing `val/ppl`.
- Embedding normalization (`sphere_normalize` / unit-sphere) — **NOT applied in LangFlow's
  path** (Option A, O1 RESOLVED). LangFlow uses a **raw, free, un-normalized** embedding:
  the clean target is the raw lookup `z = E[x]` directly; `E = sphere_embed.weight` is
  initialized with `std≈1.0` (NOT `ngpt`'s `1/√D`) so `‖z‖≈√D≈27.7` at init, matching the
  `N(0,I)` corruption noise (`‖ε‖≈√D`) so the VP corruption `z_γ=αz+σε` carries signal. The
  magnitude is **free to drift** (no `renormalize_weights`, no point-of-use normalize); the
  **trainable Gumbel `UnifInfoSchedule` is the mechanism that adapts** — it concentrates
  γ-sampling where information gain is large and tracks the informative γ-region as the free
  embedding's scale moves (paper §4.1). H4 fairness is enforced at the backbone / params /
  data / eval level, NOT at the embedding: LangFlow legitimately differs from SFM's SLERP-unit
  embedding because its Gaussian corruption is magnitude-sensitive while SLERP is
  magnitude-invariant. See §8 invariant E and Open Question O1 (resolved).
- Any change to `SFM`/`HFLM`/`FLM`/`CANDI`/`MDLM` algos, the spline `AdaptiveSchedule`, the
  `trainer_base.Diffusion._loss` token-masking math, or the `_get_parameters()` chaining.
- No new abstraction shared between LangFlow and SFM. They diverge (Gaussian vs slerp,
  γ vs t); copy what's needed, do not refactor SFM.
- AR/duo/mdlm sampler paths.

---

## 2. Module layout (exact paths)

| Path | Action | What |
|------|--------|------|
| `noise_schedules.py` | modify | Replace `UnifInfoSchedule` (currently a spline copy, ~L272-453) with the Gumbel schedule. Add a branch in `get_noise` (~L456). |
| `algo.py` | modify | Replace `LangFlow` stub (currently an SFM copy, ~L318-422) with the real implementation. |
| `models/sphere_dit.py` | modify | Add flag-gated self-cond projections, γ feed, logit bias to `SphereDiT.__init__`/`forward`. |
| `samplers.py` | modify | Add `LangFlowState`, `LangFlowSampler`; add `predictor == 'langflow'` branch in `get_sampler` (~L1183). |
| `main.py` | modify | Add `elif config.algo.name == 'langflow': diffusion_model = algo.LangFlow` (~L636-639). |
| `configs/algo/langflow.yaml` | create | LangFlow algo config. |
| `configs/noise/gumbel.yaml` | create | Trainable Gumbel; `trainable` flag toggles the fixed variant (one file + flag, see §7). |
| `configs/sampler/langflow.yaml` | create | Euler-on-γ sampler. |
| `scripts/train/tinystories/langflow.sh` | create | Paper-HP. Per EXPERIMENT.md §10(i). |
| `scripts/train/tinystories/langflow_sfmhp.sh` | create | S-FLM-HP. Per §10(ii). |
| `scripts/train/tinystories/langflow_smoke.sh` | create | Smoke. Per §10(iii). |
| `scripts/sample/tinystories/eval.sh` | modify | Add `langflow)` case to the `MODEL_TYPE` switch. |

No new files beyond these. No new module-level helpers shared across files.

---

## 3. `UnifInfoSchedule` (Gumbel) — `noise_schedules.py`

A `NoiseSchedule` subclass that **owns** the γ=logNSR path. It does NOT use a base schedule,
spline, or `record_time_loss_pair` buffer machinery — all of that spline code is deleted.

### 3.1 Parameterization (positivity-preserving)

Three learnable scalars. We store unconstrained reals and map to the constrained values via
softplus so `P_beta > 0` and `H_inf > 0` hold every step regardless of optimizer:

```
P_mu        := raw_mu                       # location, free real
P_beta      := softplus(raw_beta) + beta_floor   # scale > 0
H_inf       := softplus(raw_H) + H_floor         # entropy scale > 0
```

- `beta_floor = H_floor = 1e-4` (avoids exact-zero collapse; keeps `surrogate_entropy` and
  the γ map finite). Use `F.softplus`, not `torch.nn.Softplus()` instance, for clarity.
- Initialization (paper-faithful, VP-centered): `raw_mu = 0.0`,
  `raw_beta = inv_softplus(1.0) ≈ 0.5413` (so `P_beta ≈ 1.0`),
  `raw_H = inv_softplus(H_inf_init)` with `H_inf_init` from config (default below).
  Provide `inv_softplus(y) = log(expm1(y))` as a private module-level helper or compute inline.

### 3.2 Attributes / construction

```python
class UnifInfoSchedule(NoiseSchedule):
  """Learnable Gumbel schedule over gamma = logNSR (LangFlow).

  Variance-preserving: sigma^2 = sigmoid(gamma), alpha^2 = sigmoid(-gamma).
  gamma ~ Gumbel via inverse-CDF: gamma = P_mu - P_beta * log(-log q).
  Surrogate entropy H_gamma = H_inf * exp(-exp(-(gamma - P_mu)/P_beta)).
  """
  def __init__(self, trainable: bool, q_clip: float, H_inf_init: float,
               beta_floor: float = 1e-4, H_floor: float = 1e-4):
```

- `trainable=True`  → `raw_mu/raw_beta/raw_H` are `nn.Parameter(requires_grad=True)`.
- `trainable=False` → the **same three tensors** are registered as **buffers**
  (`register_buffer`), with `requires_grad=False`. They keep their init values forever;
  `scheduler_loss` returns a scalar-zero tensor; `sample_gamma` still draws from the (fixed)
  Gumbel via the same inverse-CDF, so the fixed variant is "fixed Gumbel params", not a
  separate uniform-in-t mapping. This is the §5 uniform-fixed ablation baseline (A1).
- `q_clip` is the Gumbel quantile clip used both to clip `q` and to define the γ clip `[a,b]`
  (the 1e-5 / 1−1e-5 quantiles): `a = P_mu - P_beta*log(-log(1-q_clip))`,
  `b = P_mu - P_beta*log(-log(q_clip))`. Compute `[a,b]` from the **current** params each call
  (cheap, 3 scalars) — do NOT cache, since params move.

Property accessors (read constrained values; used by all methods + logging):
```python
@property
def P_mu(self)   -> torch.Tensor   # scalar, raw_mu
@property
def P_beta(self) -> torch.Tensor   # scalar, softplus(raw_beta)+beta_floor
@property
def H_inf(self)  -> torch.Tensor   # scalar, softplus(raw_H)+H_floor
```

### 3.3 Methods (signatures + shapes)

```python
def sample_gamma(self, n: int, device, *, antithetic: bool = True) -> torch.Tensor:
  """Draw n gamma values via clipped Gumbel inverse-CDF. Returns gamma[n] (float32).

  q = clip(rand(n) [low-discrepancy if antithetic], q_clip, 1-q_clip)
  gamma = P_mu - P_beta * log(-log q)
  gamma = gamma.clamp(a, b)            # the q_clip quantiles
  Caller is responsible for stopgrad into the CE path (see LangFlow.nll); this method
  returns a tensor that DOES carry grad to P_mu/P_beta when trainable.
  """
```
- Antithetic = the existing low-discrepancy trick (mirror `Diffusion._sample_t`):
  `eps = rand(n); offset = arange(n)/n; q = (eps/n + offset) % 1` then clip. Sharding across
  ranks/accum is handled by `LangFlow.nll` (which passes already-sharded `n`), NOT here.
- `gamma` keeps the grad path to `P_mu`/`P_beta` (needed only for the clip-range bookkeeping;
  CE-path consumers must `.detach()` — enforced in `LangFlow.nll`, §4).

```python
def alpha_sigma_from_gamma(self, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """VP map. Returns (alpha, sigma), each same shape as gamma.
  sigma = sqrt(sigmoid(gamma)); alpha = sqrt(sigmoid(-gamma)). alpha^2 + sigma^2 == 1.
  """
```
- Use `torch.sigmoid`; do NOT recompute `1 - sigmoid` (use `sigmoid(-gamma)`) so the VP
  identity holds to fp tolerance. Clamp the sigmoid args is unnecessary because γ is clipped
  to `[a,b]`; do NOT add extra clamps (keeps invariant A exact).

```python
def surrogate_entropy(self, gamma: torch.Tensor) -> torch.Tensor:
  """H_gamma = H_inf * exp(-exp(-(gamma - P_mu)/P_beta)). Same shape as gamma.
  Carries grad to H_inf/P_mu/P_beta. gamma is passed DETACHED by the scheduler-loss caller.
  """
```

```python
def scheduler_loss(self, gamma: torch.Tensor, ce_detached: torch.Tensor) -> torch.Tensor:
  """L_Scheduler = mean( (stopgrad(ce) - H_gamma)^2 ), a scalar.

  gamma:       per-sample gamma, DETACHED (gamma[B]).
  ce_detached: per-sample CE already detached + reduced to per-sample (ce[B]).
  trainable=False -> returns a zero scalar on gamma.device with the schedule params'
  dtype, requires_grad=False (so adding it to the loss is a no-op).
  """
```
- `H_gamma = self.surrogate_entropy(gamma.detach())` (γ detached so CE-path / VP-path grads
  never reach the schedule through γ; only through `H_inf/P_mu/P_beta` inside `surrogate_entropy`).
- `ce_detached` is `stopgrad(L_CE)` per-sample — the implementer passes it pre-detached.
- Return `((ce_detached - H_gamma) ** 2).mean()`.

### 3.4 Vestigial ABC contract (`alpha_t`/`alpha_prime_t`)

`NoiseSchedule` is `abc.ABC` with abstract `alpha_t`/`alpha_prime_t`, and `forward` returns
`(alpha_prime_t(t), alpha_t(t))`. LangFlow never calls `self.noise(t)`, but the ABC must be
satisfiable and `_get_parameters()` must still iterate `self.noise.parameters()`. Provide:

```python
def alpha_t(self, t):       # vestigial; not on the LangFlow path
  gamma = self.P_mu - self.P_beta * torch.log(-torch.log(t.clamp(self.q_clip, 1 - self.q_clip)))
  alpha, _ = self.alpha_sigma_from_gamma(gamma)
  return alpha
def alpha_prime_t(self, t):
  return torch.zeros_like(t)   # unused; keep finite
```
Rationale: keeps the ABC honest and gives a sane `alpha_t(t)` (the t→γ→α map) if any generic
code path touches it, without pretending to a derivative LangFlow does not use.

### 3.5 Checkpointing

- `trainable=True`: `raw_mu/raw_beta/raw_H` are parameters → auto-saved in `state_dict`, and
  auto-included in EMA + AdamW via `_get_parameters()` (no optimizer change). **EMA covers
  them** — flagged in EXPERIMENT.md §8; acceptable (3 scalars), see Open Question O2.
- `trainable=False`: buffers → auto-saved, `requires_grad=False`, never updated.
- No custom `load_state_dict` override needed (unlike the spline schedule). Delete the spline
  `load_state_dict`/`_maybe_rebuild_spline` machinery entirely.
- Store `q_clip`, `beta_floor`, `H_floor` as plain python attributes (config-derived, not
  buffers) — they are reconstructed from config at build time, like the spline hyperparams were.

### 3.6 `get_noise` branch

```python
elif noise_config.type == 'gumbel':
  noise = UnifInfoSchedule(
    trainable=noise_config.trainable,
    q_clip=noise_config.q_clip,
    H_inf_init=noise_config.H_inf_init)
  return noise   # bypass the TruncatedScheduleWrapper / adaptive block entirely
```
Place this branch **before** the `TruncatedScheduleWrapper` / `adaptive` post-processing and
`return` immediately — those wrappers assume an α(t) base schedule and must not wrap the Gumbel.
The existing `log-linear`/`cosine-squared` branches and post-processing are unchanged.

---

## 4. `LangFlow` algo — `algo.py`

`class LangFlow(trainer_base.Diffusion)`. Overrides `nll` and `training_step`; does NOT reuse
`Diffusion.nll` (γ-path, not t-path) and does NOT reuse `SFM.q_xt` (Gaussian, not slerp).

### 4.1 `__init__`

```python
def __init__(self, config, tokenizer):
  super().__init__(config, tokenizer)            # builds backbone, self.noise (Gumbel), ema
  self.self_conditioning = config.algo.self_conditioning   # bool
  self.p_self_cond = config.algo.p_self_cond               # 0.25
  self.logit_bias = config.algo.logit_bias                 # bool
  self.logit_bias_warmup_steps = config.algo.logit_bias_warmup_steps  # 5000
  self._validate_configuration()
```

```python
def _validate_configuration(self):
  assert self.config.noise.type == 'gumbel', 'LangFlow requires noise=gumbel.'
  assert self.config.model.type == 'sphere-dit', 'LangFlow requires sphere-dit backbone.'
  if self.self_conditioning:
    assert 0.0 <= self.p_self_cond <= 1.0
```
- Do NOT inherit SFM's `invert_time_convention`/`renormalize_weights` checks; LangFlow has
  neither concept. (The training scripts still pass `algo.invert_time_convention=false` and the
  key lives in `langflow.yaml` for the shared eval harness — it is unused by LangFlow.)

### 4.2 Embedding + corruption helpers

```python
def _embed(self, x0: torch.Tensor) -> torch.Tensor:
  """Clean RAW embeddings z = E[x0] (NO normalization, Option A). Returns [B, L, d]."""
  return self.backbone.get_raw_embeddings(x0)           # raw lookup, NOT unit-norm

def q_xt(self, z, alpha, sigma, valid_tokens=None):
  """VP Gaussian corruption: z_gamma = alpha * z + sigma * eps, eps~N(0,I). [B, L, d].

  alpha, sigma: [B, 1] (one noise level per sequence) -> broadcast over L, d.
  Prompt positions (valid_tokens == 0) stay clean (z). Mirrors SFM's valid_tokens handling.
  """
  eps = torch.randn_like(z)
  z_gamma = alpha.unsqueeze(-1) * z + sigma.unsqueeze(-1) * eps
  if valid_tokens is not None:
    z_gamma = torch.where(valid_tokens.bool().unsqueeze(-1), z_gamma, z)
  return z_gamma
```
**Raw embedding accessor (new).** LangFlow must NOT call `get_sphere_embeddings` (which
unit-normalizes). Add a raw accessor to the backbone and use it everywhere LangFlow needs
the embedding:
```python
def get_raw_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:   # in SphereDiT
  """Raw embedding lookup, NO normalization. Returns [B, L, d]."""
  return self.sphere_embed(token_ids)
```
`get_raw_embeddings` is the single LangFlow entry point for clean embeddings. The existing
`get_sphere_embeddings` is left untouched (used by SFM, not LangFlow).
Note the signature **diverges** from `Diffusion.q_xt(x, alpha_t, use_pure_noise, valid_tokens)`.
That is fine: `LangFlow.nll` calls `self.q_xt` directly with the LangFlow signature; nothing in
the base `_loss`/`Diffusion.nll` is reused. (`Diffusion.nll`'s `use_pure_noise`/`_sample_t`/
`self.noise(t)` path is entirely bypassed.) `z_gamma` is NOT re-normalized to the sphere — VP
corruption deliberately moves it off the sphere; the backbone consumes raw `z_gamma`.

### 4.3 Output post-processing (with Plaid logit bias)

The Plaid bias is applied **inside the backbone** (`SphereDiT.forward`, §5), because it needs
`z_gamma`, `alpha`, `sigma`, and the per-token embedding `e_x`. `LangFlow._process_model_output`
only does the log-softmax:

```python
def _process_model_output(self, model_output, xt, sigma, context=None):
  del xt, sigma, context
  return model_output.float().log_softmax(-1)
```
`model_output` already contains the bias term (added in the backbone before this call). Reason
for placing the bias in the backbone: the bias is a function of `(z_gamma, alpha, sigma, E)` —
all available in `SphereDiT.forward` — and Plaid adds it to the **logits**, i.e. pre-softmax;
doing it in `_process_model_output` would require re-threading `z_gamma`/`alpha`/`sigma`/`r`
back out, duplicating state. Keep it where the data lives.

### 4.4 CE loss

```python
def nll_per_token(self, log_x_theta, x0):
  """L_CE per token = -log p_theta(x0). Returns [B, L]."""
  return -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
```
(Same form as SFM; keep the LangFlow-local copy, do not import SFM's.)

### 4.5 Train forward (`nll`) — end-to-end, with explicit stopgrad boundaries

```python
def nll(self, x0, output_tokens, context,
        current_accumulation_step=None, train_mode=False, valid_tokens=None):
  del output_tokens
  B = x0.shape[0]

  # (1) sample gamma (low-discrepancy/antithetic), sharded like _sample_t.
  n = (self.config.loader.global_batch_size
       if current_accumulation_step is not None else B)
  gamma_full = self.noise.sample_gamma(
      n, self.device, antithetic=self.antithetic_sampling)   # [n]
  gamma = self._shard_like_sample_t(gamma_full, B, current_accumulation_step)  # [B]
  #   STOPGRAD BOUNDARY 1: gamma feeding the CE/VP path is detached.
  gamma_ce = gamma.detach()                                  # [B]

  # (2) VP coefficients (from detached gamma -> alpha/sigma carry no schedule grad).
  alpha, sigma_vp = self.noise.alpha_sigma_from_gamma(gamma_ce)   # [B], [B]
  alpha  = alpha.unsqueeze(-1)        # [B, 1]
  sigma_vp = sigma_vp.unsqueeze(-1)   # [B, 1]

  # (3) clean embeddings + Gaussian corruption.
  z = self._embed(x0)                                        # [B, L, d]
  z_gamma = self.q_xt(z, alpha, sigma_vp, valid_tokens)      # [B, L, d]

  # (4) self-conditioning first pass (no grad).
  z_sc = self._self_cond_pass(z_gamma, gamma_ce, alpha, sigma_vp, train_mode)  # [B,L,d] or None

  # (5) main pass (grad). gamma conditions the time-embedding; bias added in backbone.
  log_x_theta = self._forward_langflow(
      z_gamma, gamma_ce, alpha, sigma_vp, z_sc)              # [B, L, V] log-probs
  utils.print_nans(log_x_theta, 'model_output')

  # (6) per-token CE.
  ce = self.nll_per_token(log_x_theta, x0)                   # [B, L]
  return ce, gamma_ce      # return gamma (detached) as the "t" slot for logging
```

Helpers:

```python
def _shard_like_sample_t(self, gamma_full, B, accum_step):
  """If accum_step is None -> return gamma_full[:B]. Else chunk by node/device/accum
  exactly as Diffusion._sample_t does, then slice to B. Returns [B]."""
```
- Copy the chunk logic verbatim from `Diffusion._sample_t` (lines for node_rank/local_rank/
  accumulate_grad_batches). This guarantees the same per-rank sample assignment SFM uses, so
  H4 stays apples-to-apples.

```python
def _self_cond_pass(self, z_gamma, gamma_ce, alpha, sigma_vp, train_mode):
  """Self-conditioning embedding z_sc.

  - self_conditioning False -> return None.
  - train_mode and rand() >= p_self_cond -> return None (z_SC := 0 in backbone).
  - else: first pass under torch.no_grad(), then z_sc = E^T xhat (RAW E, NO normalize),
          DETACHED. STOPGRAD BOUNDARY 2.
  """
  if not self.self_conditioning:
    return None
  if train_mode and torch.rand(()) >= self.p_self_cond:
    return None
  with torch.no_grad():
    log_xhat = self._forward_langflow(z_gamma, gamma_ce, alpha, sigma_vp, z_sc=None)
    z_sc = self._x_to_embed(log_xhat.exp())   # zhat = E^T xhat -> [B, L, d], RAW (not unit-norm)
  return z_sc.detach()
```
- `_x_to_embed(probs)`: `probs @ E` with **raw** `E = self.backbone.sphere_embed.weight`
  (NO `sphere_normalize` on `E` and NO `sphere_normalize` on the result — Option A). Matches
  the sampler's self-cond carry (§6) so train and sample see the same z_SC construction.
  Shape [B, L, V] @ [V, d] -> [B, L, d].
- **No-grad guarantee:** the first pass is inside `torch.no_grad()` AND `z_sc.detach()`, so the
  graph has exactly ONE backward pass (the main pass). Step time ≈ 1× (invariant F).

```python
def _forward_langflow(self, z_gamma, gamma_ce, alpha, sigma_vp, z_sc):
  """Single backbone call. Threads gamma (as the time signal), z_sc, alpha, sigma, and the
  current logit-bias ramp r into the backbone via a LangFlowContext, then log-softmax.
  Returns [B, L, V] log-probs.
  """
  r = self._logit_bias_r()                       # scalar in [0,1]
  ctx = LangFlowContext(z_sc=z_sc, alpha=alpha, sigma=sigma_vp, r=r)
  # gamma is fed as 'sigma' so trainer_base.forward routes it to the time embedder.
  return self.forward(x0=None, xt=z_gamma, sigma=gamma_ce, context=ctx)
```
- **γ as the time signal.** We pass `gamma_ce` in the `sigma=` slot of `TrainerBase.forward`.
  `Diffusion._process_sigma` mean-reduces / shape-normalizes a `[B,1]` or `[B]` tensor to `[B]`
  and zeroes it if `not time_conditioning` — γ is already `[B]`, so it passes through as the
  scalar-per-sequence time conditioning. The backbone's `TimestepEmbedder` then embeds γ. This
  reuses the S-FLM conditioning entry point exactly (fairness) — the only difference is the
  scalar's *meaning* (γ vs `-log α`). Keep `algo.time_conditioning=True`.

```python
def _logit_bias_r(self) -> float:
  """Plaid ramp: 0 if logit_bias off; else min(1, global_step / warmup_steps)."""
  if not self.logit_bias:
    return 0.0
  w = self.logit_bias_warmup_steps
  return 1.0 if w <= 0 else min(1.0, self.global_step / w)
```

### 4.6 Combining `L_CE + L_Scheduler` — override `training_step`

**Decision: override `LangFlow.training_step`, not `_loss`/`nll`.** Reason: `_loss` does the
per-token masking and **divides by token count** (`mean_token_loss`); `L_Scheduler` is a
**scalar in nats² over γ**, NOT a per-token quantity — routing it through the token-mask/divide
path would silently rescale it by `1/num_tokens` and break the H2 mechanism. So we let the
unchanged `_loss` produce `L_CE` (the existing reduction), then add the scheduler scalar on top:

```python
def training_step(self, batch, batch_idx):
  accum = batch_idx % self.trainer.accumulate_grad_batches
  losses = self._loss(batch['input_ids'], batch['attention_mask'],
                      current_accumulation_step=accum, train_mode=True)
  ce_loss = losses.loss                          # scalar, mean-token CE (grad to backbone+E)

  # Recompute per-sample CE + gamma for the scheduler. We need (gamma, per-sample CE)
  # pairs; _loss/nll already produced them but did not surface per-sample CE. Cleanest
  # surgical option: stash them on self during nll.
  gamma = self._last_gamma                       # [B] detached, set in nll
  ce_per_sample = self._last_ce_per_sample       # [B] detached, set in nll
  sched_loss = self.noise.scheduler_loss(gamma, ce_per_sample)   # scalar (0 if fixed)

  loss = ce_loss + sched_loss
  self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
  self._log_langflow_diagnostics(ce_loss, sched_loss, gamma)
  self.log('trainer/loss', loss.item(), on_step=True, on_epoch=False, sync_dist=True)
  return loss
```
- **Surfacing per-sample CE + γ:** in `nll`, after computing `ce` ([B,L]) and `gamma_ce` ([B]),
  store `self._last_gamma = gamma_ce.detach()` and
  `self._last_ce_per_sample = (ce * valid_tokens).sum(-1) / valid_tokens.sum(-1).clamp(min=1)`
  `.detach()` (per-sample mean-token CE, [B]). This is the **same** per-sample reduction
  `_loss` uses for `per_sample_loss`, so the scheduler regresses `H_gamma` against the same CE
  the user sees. Set both before returning from `nll`. (Two instance attributes, written once
  per step; the cleanest surgical surface without widening `_loss`'s return type.)
- **Why not feed `record_time_loss_pair`:** EXPERIMENT.md §9 says prefer LangFlow feeding the
  scheduler directly. We do exactly that — `scheduler_loss` is called inline; the
  `record_time_loss_pair` gate in `_loss` (`config.noise.adaptive`) is left untouched and is
  `False` for `noise=gumbel`, so that path is dead for LangFlow. **No change to `trainer_base`.**
- `sched_loss` is detached from the CE graph by construction (γ and CE both detached inside
  `scheduler_loss`); adding it to `ce_loss` only grows the schedule-param subgraph
  (`raw_mu/raw_beta/raw_H`). **STOPGRAD BOUNDARY 3.**

### 4.7 Gradient / stopgrad boundary summary (normative)

| Boundary | Where | Effect |
|----------|-------|--------|
| B1: γ → CE/VP path | `gamma_ce = gamma.detach()` in `nll` | schedule params get NO gradient from CE/backbone. |
| B2: self-cond first pass | `torch.no_grad()` + `z_sc.detach()` | no second backward; one grad pass. |
| B3: CE → scheduler | `scheduler_loss` uses `stopgrad(ce)` and `surrogate_entropy(gamma.detach())` | backbone/E get NO gradient from `L_Scheduler`. |

Consequence (test invariant): with `trainable=False`, `raw_mu/raw_beta/raw_H` never move; with
`trainable=True`, backbone gradients are identical whether or not `sched_loss` is added.

---

## 5. Backbone deltas — `models/sphere_dit.py`

All additions are gated on config flags and **zero-initialized** so that with the flags off the
module is byte-for-byte the current `SphereDiT`. Existing callers (`SFM`, `HyperbolicBoundaryFM`
do not use sphere-dit's new path; only LangFlow passes a `LangFlowContext`) are unaffected
because the new branches key on `context` being a `LangFlowContext` and on the new config flags
(`config.algo.self_conditioning`, `config.algo.logit_bias`), which only LangFlow's algo config
sets true.

### 5.1 `__init__` additions

**Embedding init (`unit_var`, new — Option A).** Add a new `init` branch to
`SphereDiT.__init__`, additive and gated so existing `random`/`ngpt`/`pretrained` are
byte-for-byte untouched (insert before the `else: raise ValueError(self.init_mode)`):
```python
elif self.init_mode == 'unit_var':
  nn.init.normal_(self.sphere_embed.weight, std=1.0)   # per-coord var~1 -> ||z||~sqrt(D)
```
LangFlow's two non-smoke training scripts and the `eval.sh` `langflow)` case MUST pass
`model.init=unit_var` (NOT `model.init=ngpt`); see §7.4. Rationale: with `std=1` each token
vector has `‖z‖≈√D`, matching the `N(0,I)` corruption noise so VP carries signal; the
magnitude is free to drift and the trainable Gumbel schedule adapts. The weight stays free
(no `renormalize_weights`, no point-of-use normalize).

Read flags once (guard with `getattr` so non-LangFlow algo configs lacking these keys still
build):
```python
self.self_conditioning = getattr(config.algo, 'self_conditioning', False)
self.logit_bias = getattr(config.algo, 'logit_bias', False)
if self.self_conditioning:
  self.W_in = nn.Linear(dim, dim, bias=False)
  self.W_sc = nn.Linear(dim, dim, bias=False)
  self.W_in.weight.data.zero_()     # zero-init -> OFF == identity at step 0
  self.W_sc.weight.data.zero_()
```
- Zero-init means at step 0 `z + W_in z + W_sc z_sc == z` exactly (matches current behavior),
  so a fresh LangFlow with self-cond on starts identical to self-cond off and learns the
  projections from zero. This mirrors the repo's `DDiTFinalLayer`/`adaLN` zero-init convention
  and `FLMBase._zero_init_module`.
- No new params when `self_conditioning=False`.

### 5.2 `forward` signature + additions

Keep the signature `forward(self, x0, xt, sigma, context=None)`. `xt` is `z_gamma`; `sigma` is
γ (already shaped by `_process_sigma`). Branch on `isinstance(context, LangFlowContext)`:

```python
def forward(self, x0, xt, sigma, context=None):
  del x0
  x = xt                                   # [B, L, d]
  lf = context if isinstance(context, LangFlowContext) else None

  # (A) self-conditioning input projection (zero-init -> no-op until trained).
  if self.self_conditioning and lf is not None:
    z_sc = lf.z_sc if lf.z_sc is not None else torch.zeros_like(x)
    x = x + self.W_in(x) + self.W_sc(z_sc)

  if self.adaLN:
    t_cond = F.silu(self.sigma_map(sigma))   # sigma == gamma here
  else:
    t_cond = None
  rotary_cos_sin = self.rotary_emb(x)
  with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c=t_cond)
    logits = self.output_layer(x, c=t_cond)            # [B, L, V]
    if self.out_temperature_scaling:
      ... (unchanged)

  # (B) Plaid logit bias (added pre-softmax; LangFlow log-softmaxes after).
  if self.logit_bias and lf is not None and lf.r > 0.0:
    logits = logits + self._plaid_bias(xt, lf.alpha, lf.sigma, lf.r)
  return logits
```

The bias is the **full Gaussian log-likelihood** (Eq. 44):
```
log p(z_gamma|v) = -||z_gamma - alpha*e_v||^2 / (2 sigma^2)
  = -||z_gamma||^2/(2 sigma^2)  +  (alpha/sigma^2) <z_gamma,e_v>  -  (alpha^2/(2 sigma^2)) ||e_v||^2
```
- The first term `-||z_gamma||^2/(2 sigma^2)` is constant across v → drops in softmax → OMIT.
- The middle term `(alpha/sigma^2) <z_gamma,e_v>` is kept.
- The quadratic term `-(alpha^2/(2 sigma^2)) ||e_v||^2` is **vocab-dependent** under Option A
  (raw, free, drifting embedding norms `||e_v||^2`) and therefore **IS included**. The paper
  drops it only because it pins `||e_v||^2 = D` via √D normalization; under Option A the norms
  are free, so the term does not cancel and must be kept.

```python
def _plaid_bias(self, z_gamma, alpha, sigma, r):
  """Plaid bias term: the full Gaussian log-lik (Eq. 44, minus the const ||z_gamma||^2 term).

  Returns [B, L, V]. E = sphere_embed.weight  RAW ([V, d], NO normalize — Option A).
  bias[b,l,v] = coef1[b]*<e_v, z_gamma[b,l]> - coef2[b]*||e_v||^2,
    coef1 = r*alpha/sigma^2, coef2 = r*alpha^2/(2 sigma^2)  ([B,1,1] each);
    bounded because gamma is clipped (sigma^2>0) AND E is finite.
  """
  E = self.sphere_embed.weight                               # [V, d], RAW (no sphere_normalize)
  inner = torch.einsum('bld,vd->blv', z_gamma, E)            # <z_gamma, e_v> for all v
  norm_sq = (E * E).sum(-1)                                  # [V] = ||e_v||^2
  coef1 = (r * alpha / (sigma ** 2)).unsqueeze(-1)           # [B, 1, 1]
  coef2 = (r * alpha ** 2 / (2 * sigma ** 2)).unsqueeze(-1)  # [B, 1, 1]
  return coef1 * inner - coef2 * norm_sq                     # [B, L, V]
```
- `alpha`/`sigma` arrive as `[B, 1]` from `LangFlowContext`; reshape to `[B,1,1]`. Only the
  `||z_gamma||^2` term (constant across v) drops out under softmax; the `||e_v||^2` quadratic
  term is vocab-dependent (Option A free norms) and is included.
- **Finiteness (invariant G):** `sigma^2 = sigmoid(gamma) > 0` and γ is clipped to `[a,b]`, so
  `alpha/sigma^2` and `alpha^2/sigma^2` are finite; the **raw** `E = sphere_embed.weight` is a
  finite parameter (init `std=1`, free to drift but never inf), so both `coef1 * <e_v, z_γ>` and
  `coef2 * ||e_v||^2` are finite. The γ clip is the only guard needed; do NOT add an extra
  epsilon (would desync from the VP identity).
  Assert finite via `utils.print_nans` upstream.
- `E` is the raw weight, used directly (no per-call recompute — there is no normalization).
  Use `.detach()`? No — the paper's bias participates in the model; let gradients flow to `E`
  through the bias as well as through the embedding lookup. (This matches "logits depend on E".)
  The `z_gamma` in the bias is the same detached-γ-derived input; no extra stopgrad needed.

### 5.3 `LangFlowContext` dataclass

Define in `algo.py` next to the other `*TrainingContext` dataclasses (it is passed by the algo,
consumed by the backbone — same location as `FLMTrainingContext`):
```python
@dataclass
class LangFlowContext:
  z_sc: torch.Tensor | None = None   # [B, L, d] self-cond embeds, None -> zeros
  alpha: torch.Tensor | None = None  # [B, 1]
  sigma: torch.Tensor | None = None  # [B, 1]
  r: float = 0.0                     # logit-bias ramp
  temperature: float = 1.0           # so trainer_base.forward temperature path is harmless
```
- `temperature` field included so the `forward`'s `context.temperature` access
  (`trainer_base.forward` ~L307) doesn't AttributeError; default 1.0 → no-op. Sampler sets it.
- Import `LangFlowContext` into `sphere_dit.py` lazily inside `forward` is undesirable
  (circular). Instead, gate on a duck-typed attribute: treat `context` as a LangFlow context if
  `hasattr(context, 'z_sc')`. Use that check in §5.2 instead of `isinstance` to avoid importing
  `algo` into `models/`. (Repo already does `getattr(context, 'skip_softmax', False)` for FLM.)

---

## 6. Sampler — `samplers.py`

`LangFlowSampler` = Euler on the γ-path (Algorithm 2), always-on self-conditioning, final argmax.

### 6.1 State + Context

```python
@dataclass(kw_only=True)
class LangFlowState(BaseState):
  xt: torch.Tensor          # [B, L, d] during integration; [B, L] ids after final argmax
  gammas: torch.Tensor      # [N+1] precomputed gamma schedule
  alphas: torch.Tensor      # [N+1]
  sigmas: torch.Tensor      # [N+1]
  z_sc: torch.Tensor        # [B, L, d] self-cond carry (zeros at k=0)
  step_idx: int
  nfe: int
  done: bool
```
Reuse `LangFlowContext` (§5.3) for the model call; set `temperature` from `sampler.temperature`.

### 6.2 `init_state`

```python
def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
               prefix_tokens=None, prefix_lengths=None):
  N = num_steps or model.config.sampler.steps
  # gamma schedule from the scheduler quantiles (Algorithm 2):
  #   q_k = clip(1 - k/N, q_clip, 1 - q_clip),  k = 0..N
  k = torch.arange(N + 1, device=model.device)
  q = (1 - k / N).clamp(model.noise.q_clip, 1 - model.noise.q_clip)
  gammas = (model.noise.P_mu - model.noise.P_beta * torch.log(-torch.log(q))).clamp(a, b)
  alphas, sigmas = model.noise.alpha_sigma_from_gamma(gammas)     # [N+1]
  # z_0 ~ N(0, sigma_0^2 I) -- raw D-space prior (per-coord var sigma_0^2). UNCHANGED under
  # Option A: the embedding space is already raw (per-coord var ~1, ||z||~sqrt(D)), so a
  # diagonal-Gaussian prior at the embedding scale is exactly right; no rescale needed.
  xt = torch.randn(num_samples, model.num_tokens, model.backbone.embed_dim,
                   device=model.device, dtype=torch.float32) * sigmas[0]
  z_sc = torch.zeros_like(xt)
  return LangFlowState(xt=xt, gammas=gammas, alphas=alphas, sigmas=sigmas,
                       z_sc=z_sc, step_idx=0, nfe=0, done=False)
```
- k=0 → q≈1 → γ≈a (cleanest end of clip) is **noise**? Note paper §sampling: q=1−k/N starts at
  q=1 (γ small → near clean) — but z_0 ~ N(0, σ_0² I) with σ_0 from γ_0. Follow the spec
  literally: `sigmas[0]` is the prior std. **Open Question O3** flags the γ-direction sanity
  check (the implementer must confirm γ_0 corresponds to the noisy end at sampling start; if the
  q ramp is reversed in the paper, flip to `q = k/N`). The schedule formula and clip are fixed;
  only the k-direction needs a one-line confirmation against `/tmp/langflow_paper.txt` Alg 2.
- Prefix/infilling: out of scope for this experiment (TinyStories unconditional). Keep the
  `_validate_prefix_args` call but treat `prefix_tokens is not None` as unsupported (assert).

### 6.3 `step` (Algorithm 2 Euler update)

```python
def step(self, model, state):
  N = len(state.gammas) - 1
  is_last = (state.step_idx == N)               # decode at (z_N, gamma_N) — Algorithm 2
  k = state.step_idx
  gamma_k = state.gammas[k].expand(state.xt.shape[0])      # [B]
  alpha_k, sigma_k = state.alphas[k], state.sigmas[k]
  ctx = LangFlowContext(z_sc=state.z_sc, alpha=alpha_k.reshape(1,1).expand(B,1),
                        sigma=sigma_k.reshape(1,1).expand(B,1), r=1.0,
                        temperature=self.temperature)   # r=1 always at sampling
  log_p = model.forward(x0=None, xt=state.xt, sigma=gamma_k, context=ctx)  # [B,L,V]
  state.nfe += 1

  if is_last:                                   # final argmax at (z_N, gamma_N)
    state.xt = log_p.argmax(-1)                 # [B, L] ids
    state.done = True
    return state

  # self-cond carry: zhat = E^T xhat (RAW E, NO normalize — matches train _x_to_embed)
  E = model.backbone.sphere_embed.weight                                       # [V,d] raw
  zhat = torch.einsum('blv,vd->bld', log_p.exp(), E)                           # [B,L,d]
  state.z_sc = zhat

  # Euler step (Algorithm 2):
  #   z_{k+1} = sigma_{k+1} * [ z_k / sigma_k + (alpha_{k+1}/sigma_{k+1} - alpha_k/sigma_k) zhat ]
  a_next, s_next = state.alphas[k+1], state.sigmas[k+1]
  state.xt = s_next * (state.xt / sigma_k
                       + (a_next / s_next - alpha_k / sigma_k) * zhat)
  state.step_idx += 1
  return state
```
- **Logit bias at sampling: always on (`r=1.0`).** Self-conditioning: always on (`z_sc` carried,
  `model.self_conditioning` is True for the reference config; if a self-cond-off checkpoint is
  sampled, `z_sc` stays zeros and `W_sc(z_sc)=0` — harmless).
- `zhat` construction matches `LangFlow._x_to_embed` (train/sample parity, invariant H).
- Final step returns ids via argmax (paper: "Final tokens: argmax x̂θ(z_N, γ_N)"). No nucleus/
  top-k by default (langflow.yaml sets `p_nucleus=1.0`); honor them if set, mirroring SFM's
  `top_k_top_p_filtering` on `log_p` before argmax.
- **NFE = N+1 (Algorithm 2).** `num_steps=N` ⇒ N Euler updates (step_idx 0..N−1, each reaching
  `z_{k+1}` up to `z_N`) PLUS one final model eval at step_idx==N that decodes the argmax at the
  **cleanest point `(z_N, γ_N)`**. The `[N]` schedule entries are used (not dead): `is_last`
  fires at `step_idx == N`, and `run_sampler` calls `step` N+1 times. (The earlier off-by-one,
  `is_last = step_idx == N-1`, decoded at `(z_{N-1}, γ_{N-1})` and left the `[N]` entries dead —
  corrected.)

### 6.4 `get_sampler` dispatch

```python
if s.predictor == 'langflow':
  return LangFlowSampler(temperature=s.temperature, p_nucleus=s.p_nucleus, top_k=s.top_k)
```
`LangFlowSampler.__init__(self, temperature, p_nucleus, top_k)`. No `velocity`/`slerp`/
`invert_time_convention` (γ-path Euler needs none). `metadata` inherited (`{'nfe': ...}`).

---

## 7. Config schema (every new key + default)

### 7.1 `configs/noise/gumbel.yaml`
```yaml
type: gumbel
trainable: true        # true -> P_mu/P_beta/H_inf are nn.Parameters; false -> fixed buffers (A1)
q_clip: 1e-5           # Gumbel quantile clip; defines gamma range [a, b]
H_inf_init: 5.0        # initial H_inf (entropy scale, nats); tune if H2 mechanism stalls
adaptive: false        # consumed by trainer_base._loss gate; MUST be false (no spline path)
alpha_min: null        # consumed by get_noise pre-branch; unused for gumbel (we return early)
alpha_max: null        # ditto
```
- **One file + `trainable` flag** (not two files). Rationale: the only difference between
  trainable Gumbel and uniform-fixed is `requires_grad` + whether `scheduler_loss` is active;
  the γ map is the same inverse-CDF. A2 toggles `noise.trainable=false`. Keeping `adaptive`,
  `alpha_min`, `alpha_max` present satisfies the keys `get_noise`/`_loss` read before the gumbel
  branch returns. `H_inf_init=5.0` is a starting guess for nats-scale CE on gpt2 vocab; the
  test only checks positivity/finiteness, not the value.

### 7.2 `configs/algo/langflow.yaml`
```yaml
name: langflow
diffusion_type: sphere
backbone: sphere-dit
parameterization: mean
time_conditioning: True       # gamma is the time signal
loss_type: ce
T: 0
causal_attention: False
adaLN: True
slerp_precision: float64      # unused by LangFlow (no slerp); present for shared eval harness
eps: 1e-6                     # unused by LangFlow; present for shared sampler dispatch
invert_time_convention: false # unused by LangFlow; scripts/eval set it; keep for harness parity
renormalize_weights: False    # unused; present so eval.sh's shared overrides don't KeyError

self_conditioning: true       # user-facing toggle (default true)
p_self_cond: 0.25             # train-time self-cond probability
logit_bias: true              # Plaid logit bias on/off
logit_bias_warmup_steps: 5000 # r ramps 0 -> 1 over these steps
```
- The four "unused" keys (`slerp_precision`, `eps`, `invert_time_convention`,
  `renormalize_weights`) are retained because `get_sampler`/`eval.sh`/shared overrides read them
  generically; omitting them risks `omegaconf` errors on the shared harness. They are inert for
  LangFlow. (`eps`/`slerp_precision` ARE read by `get_sampler` for the sfm/hflm branches but not
  the langflow branch; harmless to keep.)

### 7.3 `configs/sampler/langflow.yaml`
```yaml
predictor: langflow
steps: 1024                # match sfm.yaml for equal-NFE H4 comparison
noise_removal: greedy      # final argmax (Algorithm 2); 'greedy' is the argmax decode
use_float64: false
p_nucleus: 1.0
top_k: -1
temperature: 1.0
num_sample_batches: 2
num_sample_log: 2
```
- `steps` equals `sfm.yaml`'s 1024 so `avg_nfe` is held fixed (EXPERIMENT.md §4 fairness). The
  `noise_removal`/`use_float64`/`velocity` keys SFM uses are not all needed; include only what
  `LangFlowSampler.__init__` + `eval.sh` reference. (`noise_removal: greedy` is documentary —
  the sampler always argmaxes at the last step; keep it for harness parity.)

### 7.4 Training / smoke scripts + eval

Per **EXPERIMENT.md §10** (authoritative — do not duplicate the override lists here):
- `scripts/train/tinystories/langflow.sh` = §10(i) verbatim overrides.
- `scripts/train/tinystories/langflow_sfmhp.sh` = §10(ii) (identical numerics to sfm.sh's
  optim/schedule block, only `+wandb.name=tinystories_langflow_sfmhp`).
- `scripts/train/tinystories/langflow_smoke.sh` = §10(iii) (`tiny-sphere-dit`, 200 steps, 1 GPU,
  `strategy=single-device`); run env `SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1` (cluster memory).
- **Embedding init (Option A).** ALL three LangFlow training scripts MUST pass
  `model.init=unit_var` (NOT `model.init=ngpt`) so `E` is initialized with `std=1`
  (`‖z‖≈√D`, matching the `N(0,I)` corruption). This is the only required override beyond §10.
- Mirror `sfm.sh` scaffolding (REPO_ROOT/CACHE_DIR/OUTPUT_DIR/DEVICES/`set -euo pipefail`/
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`/`hydra.run.dir`).
- **Delta vs §10:** `tiny-sphere-dit.length=180` but tinystories ctx is 1024. The smoke script
  must override `model.length=1024` (or accept truncation) — flag as a one-line script note; the
  smoke run only checks invariants, so either is acceptable, but `model.length` must match the
  data ctx to avoid a shape error. **Implementer: set `model.length=1024` in the smoke override.**

`eval.sh` new case (mirror the `sfm)` line):
```bash
langflow)  MARGS=(model=small-sphere-dit model.init=unit_var algo=langflow noise=gumbel sampler=langflow) ;;
```
- The existing `algo.invert_time_convention=false` shared override in eval.sh is inert for
  LangFlow (key present, unused). No other eval.sh change.

---

## 8. Edge cases, invariants, what is NOT handled

Invariants (assert in-code where cheap; the test-writer encodes them — §9):

- **A — VP identity.** `alpha^2 + sigma^2 == 1` to fp tol for any γ in `[a,b]`. Guaranteed by
  `sigma^2=sigmoid(gamma)`, `alpha^2=sigmoid(-gamma)` (no independent clamps).
- **B — γ clip.** Every sampled/scheduled γ ∈ `[a,b]` (the q_clip quantiles). `sample_gamma`
  and the sampler schedule both `.clamp(a,b)`.
- **C — softplus positivity.** `P_beta > 0`, `H_inf > 0` for ALL real `raw_beta/raw_H`
  (softplus + floor). Holds every step, trainable or not.
- **D — stopgrad separation (B1/B3).** Schedule params get no CE/backbone grad; backbone/E get
  no scheduler grad. With `trainable=false`, params are buffers and never change.
- **E — raw free embedding (Option A).** LangFlow's embedding is **raw and free**: init
  `std=1.0` ⇒ per-coord var≈1 ⇒ `‖z‖≈√D` at init, matching `N(0,I)` (`‖ε‖≈√D`). **No
  normalization anywhere** in LangFlow's path (target, ẑ, self-cond embed, Plaid-bias `E` all
  raw); the magnitude is **free to drift** during training, and the trainable Gumbel schedule
  adapts to wherever the informative γ-region lands. **Testable:** at init, mean `‖z‖ ≈ √D`
  within tolerance; `_embed`/`_x_to_embed` apply NO normalization (outputs are NOT unit-norm).
  (Open Question O1 — RESOLVED.)
- **F — self-cond no-grad == 1× grad pass.** First pass under `torch.no_grad()` + detach; step
  time ≈ 1×; exactly one backward.
- **G — logit-bias finiteness.** `alpha/sigma^2` finite because γ clipped (σ²>0); logits finite.
- **H — train/sample z_SC parity.** `_x_to_embed` (train) and the sampler's `zhat` use the same
  RAW construction `probs @ E` (`E = sphere_embed.weight`, no `sphere_normalize` anywhere).
- **I — OFF flags == baseline.** `self_conditioning=false` and `logit_bias=false` ⇒
  `SphereDiT.forward` is identical to today's (no `W_in/W_sc`, no bias term). Zero-init means
  even `self_conditioning=true` at step 0 == off.

NOT handled (explicit):
- Prefix/infilling sampling (assert unsupported in `LangFlowSampler.init_state`).
- ODE-NLL tight bound (deferred).
- `T > 0` discrete-time path, `use_pure_noise`, `low_var` loss — LangFlow ignores these (γ-path).
- Multi-noise-level-per-token: γ is one scalar per sequence (`[B]`), broadcast over L (matches
  SFM's `alpha_t` `[B,1]`). Per-token γ is not supported.
- `record_time_loss_pair` / spline adaptive schedule for gumbel (dead path; gate stays
  `config.noise.adaptive`, which is false).

---

## 9. Test surface (unit-testable contracts for the test-writer)

Against the §3/§4/§5/§6 interfaces. All on CPU, tiny shapes, `tiny-sphere-dit` or a 2-layer
stub where a full backbone is overkill.

1. **VP identity (A).** For 1000 γ in `[a,b]`: `alpha_sigma_from_gamma` ⇒
   `|alpha^2 + sigma^2 - 1| < 1e-5`.
2. **γ clip (B).** `sample_gamma(10000)` ∈ `[a,b]`; with `antithetic=True` also low-discrepancy
   (empirical CDF close to Gumbel CDF).
3. **softplus positivity (C).** Set `raw_beta = raw_H = -1e3` ⇒ `P_beta`, `H_inf` ≥ floor > 0.
4. **trainable/fixed switch.** `trainable=true` ⇒ 3 params in `noise.parameters()`;
   `trainable=false` ⇒ 0 params, 3 buffers, `scheduler_loss(...) == 0` (and `requires_grad`
   False on the result).
5. **scheduler_loss value + grad (D/B3).** `scheduler_loss(gamma, ce)` =
   `mean((ce - H_inf*exp(-exp(-(gamma-P_mu)/P_beta)))^2)`; its `.backward()` produces grad on
   `raw_mu/raw_beta/raw_H` only, **zero** grad on backbone/E.
6. **B1 stopgrad.** After `LangFlow.nll`, backprop of CE produces grad on backbone+E but
   **None/zero** on `raw_mu/raw_beta/raw_H`.
7. **self-cond no-grad (F).** With `p_self_cond=1.0`, the autograd graph has exactly one
   backward through the backbone (assert via `torch.autograd.grad` count or a fwd-call counter:
   2 forward calls, 1 of them under `no_grad`).
8. **OFF == baseline (I).** Build `SphereDiT` with `self_conditioning=false, logit_bias=false`
   and the current config; assert forward output equals a vanilla `SphereDiT` forward on the
   same input (bitwise / `allclose`). Also: `self_conditioning=true` at step 0 (zero-init)
   ⇒ same output (the `W_in/W_sc` contribute 0).
9. **logit-bias finiteness (G).** For γ at both clip ends, `_plaid_bias` and the final log-probs
   are all finite; no NaN/Inf.
10. **train/sample z_SC parity (H).** `LangFlow._x_to_embed(probs)` ==
    sampler's `zhat` construction for the same `probs`/`E` — both the RAW `probs @ E`
    (assert NEITHER is unit-normalized; result need not have norm 1).
11. **VP corruption shape (q_xt).** `q_xt(z, alpha, sigma)` returns `[B,L,d]`, equals
    `alpha*z + sigma*eps`, and keeps prompt positions clean when `valid_tokens` given.
12. **Euler step closed form.** One `LangFlowSampler.step` matches the Algorithm-2 formula
    against a hand-computed reference for a fixed `zhat`.
13. **combined-loss routing.** `training_step` returns `ce_loss + sched_loss`; with
    `trainable=false`, `sched_loss==0` so `training_step` loss == `_loss().loss`. The scheduler
    scalar is NOT divided by token count (assert `sched_loss` magnitude unchanged if L doubles).
14. **dispatch.** `main.py` maps `algo.name=='langflow'` → `algo.LangFlow`; `get_sampler` maps
    `predictor=='langflow'` → `LangFlowSampler`; `get_noise` maps `type=='gumbel'` →
    `UnifInfoSchedule` and does NOT wrap it in `TruncatedScheduleWrapper`/`AdaptiveSchedule`.
15. **raw free embedding (E, Option A).** Build `SphereDiT` with `model.init=unit_var`; assert
    `mean ‖E[x]‖ ≈ √D` within tolerance at init (per-coord var≈1) AND embeddings are NOT
    unit-norm. `LangFlow._embed` applies NO normalization: for random `x0`, `_embed(x0)` equals
    `backbone.get_raw_embeddings(x0)` (== `sphere_embed(x0)`) exactly, and rows are NOT unit-norm
    (`abs(‖·‖ - 1) > tol`). `_x_to_embed`/`_plaid_bias`/sampler `zhat` use raw
    `E = sphere_embed.weight` (no `sphere_normalize` call). Keep the self-cond/bias OFF==baseline
    tests (#8) unchanged.

---

## 10. Non-obvious choices (justification, ≤2 lines each)

- **Override `training_step`, not `_loss`/`nll`, for `L_Scheduler`.** `_loss` divides by token
  count; the scheduler scalar (nats² over γ) must not be rescaled by `1/num_tokens`.
- **Stash `_last_gamma`/`_last_ce_per_sample` on `self`.** Cleanest surgical way to surface the
  per-sample (γ, CE) pair without widening `_loss`/`nll` return types or touching `trainer_base`.
- **γ in the `sigma=` slot of `forward`.** Reuses S-FLM's exact time-conditioning entry point
  (`TimestepEmbedder`) for fairness; only the scalar's meaning changes (γ vs −log α).
- **Plaid bias inside the backbone.** Needs `(z_gamma, alpha, sigma, E)` pre-softmax; computing
  it there avoids re-threading those tensors back out of `_process_model_output`.
- **One gumbel config + `trainable` flag.** Trainable and uniform-fixed differ only by
  `requires_grad` and an active `scheduler_loss`; same γ map ⇒ no second file.
- **Duck-typed `hasattr(context,'z_sc')` in `sphere_dit.py`.** Avoids importing `algo` into
  `models/` (circular); matches the repo's existing `getattr(context,'skip_softmax',...)` style.
- **Zero-init `W_in`/`W_SC`.** Guarantees OFF==baseline and a from-zero learning start, matching
  the repo's `DDiTFinalLayer`/adaLN zero-init convention.
- **Raw free embedding (Option A).** No `sphere_normalize` anywhere in LangFlow's path; init
  `std=1` gives `‖z‖≈√D` to match `N(0,I)` corruption (magnitude-sensitive VP, unlike SLERP);
  the weight is free to drift and the trainable Gumbel schedule adapts. (O1 resolved.)

---

## 11. BLOCKING open questions (need a human decision)

**O1 — RESOLVED (Option A, user decision).** LangFlow uses a **raw, free, un-normalized**
embedding: clean target `z = E[x]` direct lookup (`get_raw_embeddings`), `E = sphere_embed.weight`
init `std=1.0` (NOT `ngpt`'s `1/√D`) ⇒ `‖z‖≈√D≈27.7` matching `N(0,I)` corruption noise so VP
carries signal; **NO `sphere_normalize` anywhere** (target, ẑ, self-cond, Plaid `E`), weight
stays **free to drift** (no `renormalize_weights`). Rationale: VP Gaussian corruption is
magnitude-sensitive (unlike SLERP), so the embedding must live at the noise scale; the **trainable
Gumbel `UnifInfoSchedule` adapts** to wherever the informative γ-region lands as the free
embedding drifts (paper §4.1). H4 fairness is at backbone / params / data / eval, NOT embedding.
Affects `_embed`, `_x_to_embed`, `_plaid_bias`, the sampler `zhat` carry, the new `unit_var` init,
and §7.4 scripts (`model.init=unit_var`).

**O2 (non-blocking; flag for awareness).** `_get_parameters()` chains `self.noise.parameters()`,
so EMA + AdamW now cover the 3 Gumbel params when `trainable=true`. EMA-ing a 3-scalar schedule
is benign but means eval/sampling uses the EMA'd schedule. EXPERIMENT.md §8 asks to confirm this
"doesn't freeze the schedule." Default: **leave as-is** (params get both raw AdamW updates and an
EMA shadow used only at eval — standard). Override only if you want the schedule excluded from
EMA (would require a custom `_get_parameters` in LangFlow — a `trainer_base`-adjacent change).

**O3 — RESOLVED (confirmed correct).** Algorithm 2's sampling q-ramp uses
`q_k = clip(1 - k/N)`, giving `z_0 ~ N(0, σ_0² I)` (k=0 is the noise end, γ_0 large ⇒ σ_0 near 1).
Confirmed against the paper; the schedule formula, clip, and Euler update are fixed. No change.

**No other blocking questions.** O1 (embedding handling) is RESOLVED to Option A (raw free
embedding); O3 (q-ramp) is confirmed correct; O2 (EMA over the 3 Gumbel params) remains
non-blocking and is left as-is. Scope, interfaces, configs, and stopgrad structure are fully
determined by EXPERIMENT.md (authoritative for scope) and `/tmp/langflow_spec.md` (authoritative
for math).
