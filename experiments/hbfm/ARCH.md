# HBFM — Architecture (minimal direct-bridge `q_xt`)

Design contract for `HyperbolicBoundaryFM`. The geometry already lives in
`geo_bridge.py`; this is thin glue + a sampler + config edits. Read EXPERIMENT.md
for the science.

## Blocking open questions
None. All design points are locked by EXPERIMENT.md §4/§5 and the orchestrator
brief. (OQ-1 perf probe and OQ-2 input_repr=B are deferred / out of scope.)

---

## 1. Scope

**In**
- Rewrite `algo.HyperbolicBoundaryFM` (currently a verbatim SFM/slerp copy at
  `algo.py:317-421`) so `q_xt` is a single differentiable bridge call, heat-time
  is uniform, and `sigma = t/t_max`.
- Add `HBFMSampler` to `samplers.py` + a `get_sampler` branch.
- Edit `configs/algo/hbfm.yaml` comments (d=512 → d=64).
- Three throttled diagnostics from `q_xt`.

**Out (do NOT build)**
- No `_to_ball_cartesian`, no `_reflect_to_target_diff`, no reparam scaffolding
  (removed; do not reintroduce).
- No `_slerp`, no `_sample_prior`, no `invert_time_convention` logic in `q_xt`.
- No log-space radial-marginal fix (d≳80 overflow is deferred; d=64 is finite).
- No input_repr=B (boundary), no weighted-velocity, no embedding renormalization.
- No new noise schedule object; HBFM ignores `self.noise` entirely.
- `main.py:631` registry branch already present — verify only, do not edit.

---

## 2. Module layout

| File | Action |
|---|---|
| `algo.py:317-421` | **Rewrite** `HyperbolicBoundaryFM` (the only substantive change). |
| `samplers.py` | **Add** `HBFMState`, `HBFMContext`, `HBFMSampler`; **add** `get_sampler` branch `predictor=='hbfm'`. Import names already present at `samplers.py:19-20`. |
| `configs/algo/hbfm.yaml` | **Edit** comment lines 17-19 (d=512→d=64). Fields already correct. |
| `configs/sampler/hbfm.yaml` | No change needed (already `predictor: hbfm`, `steps: 180`). |
| `main.py:631` | **Verify only** — branch exists. |

---

## 3. `HyperbolicBoundaryFM` — interfaces

Subclass `trainer_base.Diffusion` (same base as `SFM`). Keep `__init__` /
`_validate_configuration` identical in spirit to SFM but read HBFM fields.

```python
def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.renormalize_weights = config.algo.renormalize_weights   # False (LOCKED)
    self.t_min = config.algo.hbfm_t_min                          # 1e-3
    self.t_max = config.algo.hbfm_t_max                          # 0.05 (2.0 @ d=2 smoke)
    self.bridge_dim = config.algo.bridge_dim or config.model.hidden_size  # null→64
    self.weighted_ce = config.algo.weighted_ce                  # False default
    self.input_repr = config.algo.input_repr                    # 'A' (only A supported)
    self.log_qxt_time = config.algo.hbfm_log_qxt_time           # False
    self._validate_configuration()
```

`_validate_configuration`: assert `input_repr == 'A'`, `proposal_type == 'unif'`,
`backbone == sphere-dit`. (Mirror SFM's sphere-arch guard cheaply.)

### `_process_model_output(self, model_output, xt, sigma, context=None)`
1-line: `return model_output.float().log_softmax(-1)`. Identical to SFM. `xt` is
the ball point — the backbone consumes it as a continuous embedding; no
sphere assumption is enforced.

### `_sample_heat_t(self, n) -> (ts, interval)`
1-line: uniform heat-time sampler, constant importance weight.
- `ts = torch.rand(n, device=self.device) * (t_max - t_min) + t_min` → `(n,)` float32.
- `interval = self.t_max - self.t_min` (python float, the constant proposal weight).
- Replaces `_sample_t` + `self.noise`; HBFM does **not** call the noise schedule.

### `q_xt(self, x, t, use_pure_noise, valid_tokens=None) -> z`
The single mechanistic method. `x`: `(B,L)` long token ids. `t`: `(B,)` heat-time
(this is the literal heat time, NOT `alpha_t`). Returns `z`: `(B,L,d)` float32,
`‖z‖<1`, **`requires_grad` and differentiable w.r.t. the embedding direction**.

Steps:
1. `emb = self.backbone.sphere_embed.weight` — `(V,d)`, trainable, NOT renormalized.
   (Use the raw weight, not `get_sphere_embeddings` which would sphere-normalize and
   also detach nothing but normalizes norm; the bridge normalizes direction itself.)
2. `ts64 = t.to(torch.float64)` — run the bridge in float64 for numerical safety
   (`sample_radial` grid + `_check_lorentz_rho_bound` precision). Grad survives the cast.
3. **Dispatch on `d == 2`** (`self.bridge_dim`):
   - **general d (`d != 2`)** — `targets` is `(B,L)`, `ts` is `(B,)` broadcast over L:
     ```
     rhos, u = HyperbolicHeatKernel.poincare_bridge(
         ts=ts64, targets=x, word_embedding=emb.to(torch.float64),
         output_coord=Coordinate.HYPERBOLIC_POLAR)        # rhos (B,L), u (B,L,d)
     z = GeoUtils.hyperbolic_polar_to_poincare_cartesian(rhos, u)   # (B,L,d), ‖z‖<1
     ```
   - **d == 2** — `binary_*` are flat-batched; flatten L into the batch:
     ```
     B, L = x.shape
     ts_flat = ts64.repeat_interleave(L)                  # (B*L,)
     tgt_flat = x.reshape(-1)                             # (B*L,)
     rhos, thetas = BinaryHyperbolicHeatKernel.binary_poincare_bridge(
         ts=ts_flat, targets=tgt_flat, word_embedding=emb.to(torch.float64),
         output_coord=Coordinate.HYPERBOLIC_POLAR)        # both (B*L,)
     z = GeoUtils.binary_hyperbolic_polar_to_poincare_cartesian(rhos, thetas)  # (B*L,2)
     z = z.reshape(B, L, 2)
     rhos = rhos.reshape(B, L)                            # for diagnostics
     ```
   Splitting polar→Cartesian (rather than `output_coord=CARTESIAN`) gives `rhos`
   for the diagnostics for free; the conversion never trips `_LORENTZ_RHO_MAX`
   (Poincaré conversion has no rho bound — only Lorentz does), so `‖z‖<1` always holds.
4. **`use_pure_noise` branch** (replaces steps 1-3 when true): draw from the *free*
   kernel (no target conditioning) at `t_max`:
   ```
   ts_max = torch.full((B,), self.t_max, dtype=torch.float64, device=self.device)
   z = HyperbolicHeatKernel.free_poincare_heat_kernel(
       ts=ts_max, seq_len=L, embedding_size=d, output_coord=Coordinate.CARTESIAN)  # (B,L,d)
   ```
   (d==2 path uses `BinaryHyperbolicHeatKernel.binary_free_poincare_heat_kernel`
   flattened+reshaped the same way. `rhos` for diagnostics optional here.)
5. **`valid_tokens` clean-pinning** — Sudoku passes all-ones, so make it a no-op
   fast path: `if valid_tokens is not None and not valid_tokens.all():` recompute a
   near-clean bridge at `t_min` and `torch.where(mask, z, z_clean)`. Otherwise skip
   entirely. (Mirrors SFM's `torch.where(valid, x_t, e_clean)` minimally; SFM pins
   to `e_clean` directly but HBFM has no on-manifold clean point, so pin to a
   `t_min` bridge draw of the same tokens — cheap and stays in the ball.)
6. `z = z.to(torch.float32)` (model dtype). Grad survives.
7. Diagnostics: `self._log_diag(rhos)` (throttled; §5). Skip when `rhos` absent
   (pure-noise) or when not training.

**Gradient note:** grad flows emb → bridge direction (`_reflect_to_target` /
`atan2(e[1],e[0])`) → `z`. Radius (`sample_radial`) and the angular Poisson draw
carry no embedding grad — direction-only, intended (same as S-FLM's slerp, where
only the clean endpoint carries embedding grad). No `torch.no_grad()` anywhere in
`q_xt`.

### `nll_per_token(self, log_x_theta, xt, x0, interval, low_var=False, context=None, train_mode=False)`
1-line: plain CE × constant proposal weight.
- `ce = -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)`  → `(B,L)` (the SFM form).
- `return ce * interval if self.weighted_ce else ce`. `interval = t_max - t_min`
  is the constant uniform-proposal weight (`weighted_ce=False` ⇒ plain CE, default).
- Signature swaps SFM's `(alpha_t, dalpha_t)` for the scalar `interval`. `del xt`.

### `nll(self, x0, output_tokens, context, current_accumulation_step=None, train_mode=False, valid_tokens=None) -> (loss, t)`
Mirror SFM/base `nll` structure, swapping the schedule for heat-time:
```
del output_tokens
t, interval = self._sample_heat_t(x0.shape[0])         # but use base batch-sharding?
use_pure_noise = self._use_pure_noise(train_mode, context)
if use_pure_noise: t = torch.full_like(t, self.t_max)
xt = self.q_xt(x0, t, use_pure_noise=use_pure_noise, valid_tokens=valid_tokens)
sigma = (t / self.t_max).unsqueeze(-1)                 # (B,1); NOT _sigma_from_alphat
log_x_theta = self.forward(x0=x0, xt=xt, sigma=sigma, context=context)
utils.print_nans(log_x_theta, 'model_output')
loss = self.nll_per_token(log_x_theta, xt=xt, x0=x0, interval=interval,
                          low_var=..., context=context, train_mode=train_mode)
return loss, t
```
**Batch sharding:** `_sample_heat_t` must produce `x0.shape[0]` times after the
DDP/accum chunking that `_sample_t` does. Simplest minimal choice: draw
`t ~ Uniform[t_min,t_max]` of length `x0.shape[0]` directly (no antithetic / accum
sharding — heat-time has no schedule coupling, and the assert `t.shape[0]==x0.shape[0]`
is trivially satisfied). Keep `_sample_heat_t(n)` = `torch.rand(n)`-based, `n=x0.shape[0]`.

### `optimizer_step(self, *args, **kwargs)`
1-line: `return super().optimizer_step(*args, **kwargs)`. NO renormalization
(`renormalize_weights=False`, LOCKED). May omit the override entirely and inherit
the base; include it only if a guard is wanted (prefer omitting — base already EMA-updates).

### `_log_diag(self, rhos)` (tiny helper)
Throttled diagnostics, ≤8 lines:
```
if not self.training or rhos is None: return
step = self.global_step
if step % 50 != 0: return
self.log('hbfm/mean_rho', rhos.float().mean(), on_step=True, on_epoch=False, sync_dist=True)
self.log('hbfm/rho_saturated_frac', (rhos >= _LORENTZ_RHO_MAX).float().mean(), ...)
self.log('hbfm/emb_norm_mean', self.backbone.sphere_embed.weight.norm(dim=-1).mean(), ...)
```
`_LORENTZ_RHO_MAX = 20.0` imported from `geo_bridge`. `hbfm/qxt_time` (OQ-1) only if
`self.log_qxt_time` — wrap the bridge call in `time.perf_counter()` and log once;
keep it behind the flag so default path has zero timing overhead. Match the
`self.log(...)` kwargs style used at `trainer_base.py:326-335`.

---

## 4. Data flow

```
x0 (B,L) long ──┐
                ├─ q_xt ── bridge(emb.dir, t) ── z (B,L,d) float32, ‖z‖<1, grad→emb.dir
t~U[tmin,tmax] ─┘                                    │
                                                     ▼
sigma=t/t_max (B,1) ───────────────► backbone.forward(xt=z, sigma) ── logits (B,L,V)
                                                     │
                            _process_model_output: log_softmax
                                                     ▼
x0 (B,L) ───────────────► nll_per_token: CE [× interval] ── loss (B,L)
```
State: none persisted in training. The only learned state is
`backbone.sphere_embed.weight` (trains via the bridge in `q_xt`) and the
independent `backbone.output_layer` (vocab head; no tied readout). `self.noise` is
unused dead state inherited from base.

---

## 5. Diagnostics (W&B)

`hbfm/mean_rho`, `hbfm/rho_saturated_frac`, `hbfm/emb_norm_mean` every 50 steps from
`_log_diag`. `hbfm/qxt_time` once, gated by `hbfm_log_qxt_time`. These are the
EXPERIMENT.md §8 keys not currently emitted by `trainer_base`. Throttle to avoid
per-step overhead (radius reduction is cheap but `self.log` cadence matters).

---

## 6. `HBFMSampler` (samplers.py)

Inference only — a `torch.no_grad()` wrapper is fine. Mirror `SFMSampler`'s
state/decoding structure (`samplers.py:599-723`).

```python
@dataclass(kw_only=True)
class HBFMState(BaseState):
    xt: torch.Tensor          # (B,L,d) Poincaré-ball point during integration; (B,L) int at decode
    t_schedule: torch.Tensor  # (steps+1,) heat-times DECREASING from t_max
    start_idx: int; step_idx: int; nfe: int; done: bool
    prefix_lengths: torch.Tensor = None
    prefix_tokens: torch.Tensor = None
    prefix_embeds: torch.Tensor = None   # ball embeddings of prefix (optional; Sudoku has none)

@dataclass
class HBFMContext:
    temperature: float = 1.0
```

### `__init__(self, noise_removal, velocity, use_float64, eps, temperature, p_nucleus, top_k, t_min, t_max, bridge_dim)`
Store flags (mirror SFMSampler). `t_max`/`t_min`/`bridge_dim` from `config.algo`.

### `init_state(self, model, num_samples, *, num_steps=None, eps=1e-5, prefix_tokens=None, prefix_lengths=None) -> HBFMState`
1-line: free-kernel init in the ball, decreasing heat-time grid.
- `d = model.backbone.embed_dim`; `L = model.num_tokens`.
- `ts0 = torch.full((num_samples,), t_max, dtype=torch.float64, device=model.device)`.
- `xt = HyperbolicHeatKernel.free_poincare_heat_kernel(ts0, L, d, CARTESIAN)` (d==2:
  binary free, flatten+reshape) → `(B,L,d)` float32, `‖xt‖<1`.
- `t_schedule = torch.linspace(t_max, max(t_min, eps), num_steps+1, device=...)`
  (decreasing; terminal step decodes).
- Prefix handling: reuse `_project_prefix` exactly as SFMSampler (Sudoku passes none).

### `step(self, model, state) -> HBFMState`
1-line: predict x0, take a hyperbolic-geodesic step toward predicted-target ball point.
- `is_last_step = (step_idx == num_steps-1)`.
- `sigma = (t_schedule[step_idx] / t_max).reshape(-1,1)` (matches training `sigma=t/t_max`).
- `log_p = model.forward(xt=state.xt, sigma=sigma, context=HBFMContext(temperature))`;
  `nfe += 1`; optional top_k/top_p filter (reuse SFMSampler logic).
- `if is_last_step: return self._last_step_decode(state, log_p)` — identical to
  `SFMSampler._last_step_decode` (greedy/ancestral categorical over `log_p`,
  `_project_prefix`, set `state.xt = tokens`, `done=True`).
- Else, move along the geodesic toward the expected target ball point:
  - `E` = ball embedding of vocab — `tanh(‖emb‖? )`… NO: HBFM has no fixed ball
    table. **Decode-then-geodesic:** compute expected destination on the ball as the
    `t_min`-bridge / direction-weighted point. Minimal on-manifold choice that
    reuses geo_bridge: take `dest` = current `xt` advanced by the geodesic kernel.
  - Use `HyperbolicHeatKernel.geodesic(t=frac, src_cartesian=state.xt,
    dest_cartesian=dest, cartesian_model=Geometry.POINCARE, output_coord=CARTESIAN)`
    (d==2: `BinaryHyperbolicHeatKernel.geodesic`, flatten+reshape). `frac` from the
    heat-time ratio `(t[k]-t[k+1])/t[k]`. Result stays in the ball (`‖·‖<1`).
  - `state.xt = result.to(state.xt.dtype)`; `_project_prefix`; `step_idx += 1`.

> **Sampler open detail (NON-blocking, implementer/test-writer to settle):** the
> exact `dest` ball point per step. The geometry-correct destination is the
> argmax/expected target *embedding direction* placed on the ball at radius→0
> (clean), i.e. `dest = tanh(rho_target/2)·dir(E[argmax log_p])` with `rho_target→0`.
> Simplest correct-on-manifold version: `dest` = origin-direction of the decoded
> token's embedding (`dir = emb[tok]/‖emb[tok]‖`, radius from a small `t` bridge or
> 0). This is inference-only and does not affect the H1 training result; pick the
> minimal version that keeps `‖xt‖<1` and decodes sensibly, validated by the
> sampler on-manifold test. Mirror SFMSampler's velocity-then-`exp_map` shape if a
> closer analogue is wanted, but `geodesic` is the on-manifold primitive to use.

### `get_sampler` branch
```python
if s.predictor == 'hbfm':
    return HBFMSampler(noise_removal=s.noise_removal, velocity=s.velocity,
        use_float64=s.use_float64, eps=config.algo.eps, temperature=s.temperature,
        p_nucleus=s.p_nucleus, top_k=s.top_k,
        t_min=config.algo.hbfm_t_min, t_max=config.algo.hbfm_t_max,
        bridge_dim=config.algo.bridge_dim or config.model.hidden_size)
```

---

## 7. Edge cases & invariants

- **`‖z‖<1` always** (Poincaré conversion clamps scale below 1 by one ulp,
  `geo_bridge.py:321`). No `_LORENTZ_RHO_MAX` raise on the Poincaré path (only the
  Lorentz/`free_lorentz`/`*_lorentz_bridge` paths check it). We never request
  Lorentz-Cartesian output → no `ValueError` from large rho. The
  `rho_saturated_frac` diagnostic is informational, not a crash guard.
- **d≳80 overflow** is upstream in `sample_radial` (deferred). d=64 verified finite;
  d=2 uses Gruet series (unaffected). `_validate_configuration` need not guard d
  (EXPERIMENT.md keeps d=64); optionally a soft assert `bridge_dim <= 64`.
- **dtype:** bridge runs in float64 (`ts64`, `emb.to(float64)`); `z` cast to float32
  for the backbone. emb grad survives the float64→float32 round trip.
- **grad route is q_xt only:** if a caller detaches `z`, `emb.grad` is None — the
  vocab head (`output_layer`) is independent and carries no embedding grad. Consistent
  with `renormalize_weights=False`: direction is normalized only *inside* the bridge,
  norm is free and trains via this single grad route.
- **adaLN-Zero head:** the backbone's `DDiTFinalLayer`/block adaLN init to zero gates,
  so on a *fresh* model the first backward through the head can yield zero/near-zero
  emb grad. Tests must perturb the head (one optimizer step, or assert on a
  non-fresh/seeded-nonzero state), NOT assert nonzero emb grad on the first backward
  of a fresh model. (See test surface.)
- **`use_pure_noise`** uses the free kernel at `t_max` (no target) — `valid_tokens`
  pinning is skipped in that branch.
- **NOT handled:** input_repr=B, weighted velocity, log-space marginal, schedule
  coupling, embedding renormalization, antithetic/accum sharding of heat-time.

---

## 8. Integration points

- `main.py:631` already maps `algo.name=='hbfm'` → `algo.HyperbolicBoundaryFM`. Verify.
- `configs/algo/hbfm.yaml` fields are read in `__init__` (§3). Only comment edits needed.
- `configs/sampler/hbfm.yaml` `predictor: hbfm` → new `get_sampler` branch.
- Backbone: `models.sphere_dit.SphereDiT` (via `config.model.type` resolution in
  `trainer_base.__init__`). `backbone.sphere_embed.weight` `(V,d)`, `backbone.embed_dim`,
  `backbone.forward(x0,xt,sigma,context)`. No backbone change.
- `self.forward`/`self._process_model_output`/`self.log`/`self.global_step` inherited
  from `trainer_base.Diffusion`. No shared-code edits.

---

## 9. Test surface (test-writer recreates `tests/test_hbfm_*.py`)

Fixtures in surviving `tests/conftest.py`: `hbfm_d8`/`hbfm_d2`, `config_d8`/`config_d2`
(d=8 general path, d=2 closed-form; `HBFM_T_MAX=2.0`, `n_heads=1` for d=2), `x0`
`(2,4)`, `valid_tokens` all-ones. Reference `tests/test_geo_bridge_overflow.py` for
the d-overflow boundary (xfail at d=512; d=64/d=8/d=2 finite).

Contract items the tests assert:
- **q_xt invariants:** `z = hbfm.q_xt(x0, t, use_pure_noise=False, valid_tokens)`:
  shape `(B,L,d)`, dtype float32, `‖z‖.max() < 1`, `z.requires_grad` True.
- **emb grad nonzero via q_xt backward:** after **perturbing the head** (one optimizer
  step on the model, or seeding non-zero adaLN gates), `z.sum().backward()` ⇒
  `backbone.sphere_embed.weight.grad` is not None and norm > 0. Do NOT assert nonzero
  on a *fresh* model's first backward (adaLN-Zero gates ⇒ may be 0). For the pure
  `q_xt` grad check, backprop directly from `z` (not through the zero-gated head) — a
  loss on `z` itself gives nonzero emb grad even on a fresh model.
- **grad route is q_xt only:** detach `z` before the head ⇒ `sphere_embed.weight.grad`
  is None after backward (no tied-readout leak; head is independent).
- **d=2 vs general-d dispatch:** d=2 fixture exercises `binary_poincare_bridge`
  (flatten `(B,L)→(B*L)`, reshape back `(B*L,2)→(B,L,2)`); d=8 exercises
  `poincare_bridge` `(B,L,d)`. Both yield `‖z‖<1`, correct shape.
- **loss equivalence:** `nll_per_token` with `weighted_ce=False` equals the SFM plain
  CE `-log_softmax.gather(x0)`; with `weighted_ce=True` equals that × `interval`
  (`t_max - t_min`). The uniform proposal weight == `interval` (constant).
- **embedding not renormalized:** after `optimizer_step`, `sphere_embed.weight` norm
  is unchanged by any HBFM-side renormalization (no call to
  `backbone.renormalize_weights`); norms remain free.
- **`sigma = t/t_max`** fed to backbone (not `_sigma_from_alphat`); `_sample_heat_t(n)`
  returns `t ∈ [t_min,t_max]` and constant `interval == t_max - t_min`.
- **pure-noise branch:** `q_xt(..., use_pure_noise=True)` returns ball point from the
  free kernel, `‖z‖<1`, target-independent.
- **sampler init/step on-manifold:** `HBFMSampler.init_state` ⇒ `‖xt‖<1`,
  `t_schedule` length `steps+1` decreasing from `t_max`; `step` keeps `‖xt‖<1` and
  the terminal step decodes to int tokens `(B,L)` (mirror SFMSampler decode test).

---

## 10. Non-obvious choices (1-liners)

- **Polar→Cartesian split instead of `output_coord=CARTESIAN`:** yields `rhos` for the
  diagnostics with no extra bridge call, and the Poincaré conversion has no rho bound.
- **float64 bridge, float32 z:** `sample_radial` grid / rho-bound checks want
  float64; backbone wants float32; grad survives the cast (verified emb.grad ≠ 0).
- **raw `sphere_embed.weight` (not `get_sphere_embeddings`):** the bridge normalizes
  direction internally; pre-normalizing would discard the free-norm signal and is
  redundant. Matches `renormalize_weights=False`.
- **No `_sample_t`/`self.noise`:** heat-time is schedule-free; reusing the base
  sampler would couple to an irrelevant alpha schedule.
- **`geodesic` as the sampler step primitive:** it is the on-manifold interpolator
  already in `geo_bridge.py`; reusing it guarantees `‖xt‖<1` without new geometry.
