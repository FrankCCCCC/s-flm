# ARCH: HFLM (Hyperbolic Flow Language Model)

**Owner:** deuterium1729@gmail.com · **Date:** 2026-06-05 · **Branch:** `hflm`
**Contract for:** test-writer + implementer. This is Direction 1 (hyperbolic analog of S-FLM),
already chosen — this doc specifies it precisely, it does not re-open the design.

**Guiding principle:** minimal, surgical, modular. The HFLM training/sampling backbone must stay
structurally *identical* to S-FLM so that HFLM-vs-S-FLM is a clean A/B over geometry only. Every
delta below is justified in 1–2 lines. Do not add abstraction beyond what is listed.

---

## 1. Scope

### In scope
- `algo.HFLM`: fix the broken WIP into a structural twin of `algo.SFM` with sphere→hyperbolic swaps
  (wrapped-normal prior, hyperbolic geodesic interpolation, CE loss into the embedding table).
- `models.hyperbolic_dit.HyperbolicDiT`: rename `SphereDiT` clone; drop embedding renormalization and
  sphere calibration; consume Poincaré `xt` as-is. (This is the **default** backbone for the run.)
- `samplers.HFLMSampler`: hyperbolic sampler — wrapped-normal prior init, geodesic-step-toward-predicted-
  clean integrator, argmax decode.
- Config + dispatch wiring: `configs/algo/hflm.yaml`, `configs/model/{tiny,small}-hyperbolic-dit.yaml`,
  `configs/sampler/hflm.yaml`, branches in `trainer_base.py`, `main.py`, `samplers.py`,
  `models/__init__.py`, and the W&B `sudoku/exact_match_acc` log line.

### Out of scope (explicitly excluded)
- **`models.hyperbolic_arch` (nGPT clone) is NOT used for the Sudoku run.** Its blocks `justnorm`
  activations onto `S^{d-1}` after every layer, which would discard the radial coordinate after layer 1
  and destroy length-as-radial structure. We rename it for parity but it is *not* wired as the run
  backbone. See §6 caveat. (Justification: keeps the radial signal alive; backbone stays a vanilla DiT.)
- **Exact marginalized hyperbolic velocity** (analog of eq.15, `Σ_k p_k log_{z_t}(e_k)` via a hyperbolic
  log-map over all `V`). The sampler uses the geodesic-step-toward-predicted-clean scheme instead
  (top-1 / sampled token). Full marginalization is OPTIONAL/future; §7 notes what it would require.
- **Truncated / adaptive schedules** beyond the optional run D (`noise.alpha_max` override only — no code).
- Any change to `models.dit` DDiT blocks, the training loop, `noise_schedules`, or `trainer_base.Diffusion`
  beyond the single dispatch branch. The DiT body stays byte-identical to `SphereDiT`.
- New geometry math. All geometry comes from `geo_bridge.py` (`HyperbolicHeatKernel.geodesic`,
  `GeoUtils.wrapped_normal`, converters). Do NOT re-derive.

---

## 2. Module / file map

| File | Add/Edit | Purpose |
|---|---|---|
| `algo.py` | Edit | Rewrite `HFLM` (`317-436`): fix `_sample_prior`, `q_xt`, rename+fix `_hyeprbolic_geodesic`→`_hyperbolic_geodesic`. `nll`/`nll_per_token`/`optimizer_step`/`_process_model_output`/`_validate_configuration` stay as the SFM twin. |
| `models/hyperbolic_dit.py` | Edit | Rename `SphereDiT`→`HyperbolicDiT`; drop `renormalize_weights`, `get_sphere_embeddings` reliance, `init_sphere_embed_from_pretrained`'s sphere normalize; keep `get_hyperbolic_polar_embeddings`; `forward` consumes Poincaré `xt` as-is. Embedding param **stays named `sphere_embed`** (see §6 note). |
| `models/hyperbolic_arch.py` | Edit | Rename `SphereArch`→`HyperbolicArch`, `SphereArchBlock`→`HyperbolicArchBlock` for parity only. NOT wired into the run. Carry the `justnorm` caveat as a module docstring line. |
| `models/__init__.py` | Edit | Add `from . import hyperbolic_dit` and `from . import hyperbolic_arch`. |
| `samplers.py` | Edit | Rewrite `HFLMSampler` (`727-851`); add `HFLMState`/reuse `SFMState`; add `hflm_step_size` helper if step-size differs (it does not — reuse `sfm_step_size`); add `predictor=='hflm'` branch to `get_sampler` (`~1180`). |
| `trainer_base.py` | Edit | Add `model.type=='hyperbolic-dit'` (→`models.hyperbolic_dit.HyperbolicDiT`) and `'hyperbolic-arch'` (→`models.hyperbolic_arch.HyperbolicArch`) branches at `69-85`. |
| `main.py` | Edit | Add `algo.name=='hflm'`→`algo.HFLM` branch at `~629`; add `wandb_logger.log_metrics({'sudoku/exact_match_acc': accuracy*100})` in `_sudoku_eval` rank-0 block (`~583`). |
| `configs/algo/hflm.yaml` | Add | Clone `sfm.yaml`; `backbone: hyperbolic-dit`, `renormalize_weights: False`, add `prior_cov: 0.25`, `rho_max: 12`. |
| `configs/model/tiny-hyperbolic-dit.yaml` | Add | Clone `tiny-sphere-dit.yaml`; `type: hyperbolic-dit`, `init: hyperbolic` (new init mode, std≈0.3 — see §5). |
| `configs/model/small-hyperbolic-dit.yaml` | Add | Clone `small-sphere-dit.yaml`; same swaps. Not used by the Sudoku run; added for parity. |
| `configs/sampler/hflm.yaml` | Add | `predictor: hflm`, `steps: 180`, `noise_removal: greedy`, `velocity: exact`, `top_k_velocity: 1`. (Full spec in EXPERIMENT §4.) |
| `scripts/train/sudoku/hflm.sh` | Add | Clone `sfm.sh` with `model=tiny-hyperbolic-dit algo=hflm algo.prior_cov=0.25 algo.rho_max=12 sampler=hflm` and W&B overrides (EXPERIMENT §4). |
| `scripts/sample/sudoku/hflm.sh` | Add | Sudoku eval invocation mirroring the existing sfm sample script (if present); `mode=sudoku_eval`, `sampler=hflm`. |

**Note on `models/hyperbolic_dit.py` current state:** it is presently a *verbatim copy* of `SphereDiT`
(class still named `SphereDiT`, still has `renormalize_weights`, `get_sphere_embeddings`,
`get_hyperbolic_polar_embeddings`). The edit is a rename + declutter, not a from-scratch write.

---

## 3. Interfaces (exact signatures + shapes)

Notation: `B`=batch, `L`=seq len (180 for Sudoku), `d`=embedding dim (512), `V`=vocab.
All polar `rhos` are `[B,L,1]` (keepdim) unless noted; all directions/Poincaré points are `[B,L,d]`.

### 3.1 `algo.HFLM`

```python
class HFLM(trainer_base.Diffusion):
    def __init__(self, config, tokenizer): ...
        # self.eps, self.renormalize_weights, self.invert_time_convention  (SFM twin)
        # + self.prior_cov = config.algo.prior_cov   # float, default 0.25
        # + self.rho_max   = config.algo.rho_max     # float, default 12.0

    def _validate_configuration(self): ...
        # Identical to SFM. Additionally: assert self.prior_cov > 0 and self.rho_max > 0.
        # Backbone-arch guard: if model.type == 'hyperbolic-arch':
        #   raise ValueError('hyperbolic-arch justnorms the radial away; use hyperbolic-dit.')

    def _process_model_output(self, model_output, xt, sigma, context=None) -> torch.Tensor:
        # IDENTICAL to SFM: return model_output.float().log_softmax(-1).  -> [B,L,V]

    def _sample_prior(self, e_clean_rhos) -> tuple[torch.Tensor, torch.Tensor]:
        """Origin wrapped-normal prior on H^d. Returns (rhos[B,L,1], u[B,L,d])."""
        # shape = (B, L, d) taken from broadcasting e_clean_rhos[...,0] against d.
        # rhos_raw, u = GeoUtils.wrapped_normal(
        #     shape=(B, L, d), mean=0.0, cov=self.prior_cov,
        #     dtype=<clean dtype>, device=<clean device>)
        # wrapped_normal returns rhos[B,L] -> unsqueeze(-1) to [B,L,1]; u is [B,L,d].
        # cov is a VARIANCE: prior_cov=0.25 == std s=0.5, E[rho]≈s·√d≈11.3 (< 20).

    def q_xt(self, x, alpha_t, use_pure_noise, valid_tokens=None) -> torch.Tensor:
        """Poincaré-ball latent z_t at noise level alpha_t. Returns z_t [B,L,d], ‖z_t‖<1."""
        # See §4.1 for the exact data flow. Output is a Poincaré CARTESIAN point.

    def optimizer_step(self, *args, **kwargs):
        # IDENTICAL to SFM (renormalize_weights stays False for HFLM ⇒ no-op).

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                      low_var=False, context=None, train_mode=False) -> torch.Tensor:
        # IDENTICAL to SFM: ce = -log_x_theta.gather(-1, x0[...,None]).squeeze(-1)  -> [B,L].

    def _hyperbolic_geodesic(self, clean_rhos, clean_thetas,
                             noisy_rhos, noisy_thetas, alpha_t) -> torch.Tensor:
        """Constant-speed H^d geodesic, src=noisy → dest=clean, fraction=alpha_t.
        Returns Poincaré-ball CARTESIAN point z_t [B,L,d]. alpha_t convention: see §4.2."""
        # float64 path mirrors SFM._slerp (gated on config.algo.slerp_precision == 'float64').

    def nll(self, x0, output_tokens, context,
            current_accumulation_step=None, train_mode=False,
            valid_tokens=None) -> tuple[torch.Tensor, torch.Tensor]:
        # IDENTICAL to SFM nll (211-315 twin). No changes.
```

### 3.2 `models.hyperbolic_dit.HyperbolicDiT`

```python
class HyperbolicDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int): ...
        # IDENTICAL body to SphereDiT EXCEPT init mode (§5) and no renormalize calls.
        # self.sphere_embed = nn.Embedding(vocab_size, dim)   # name KEPT (see §6 note)
        # init: 'hyperbolic' -> nn.init.normal_(weight, std=0.3); 'ngpt'/'random'/'pretrained' kept.
        # self.embed_dim = dim   # d, UNCHANGED — Poincaré is d-dim, no width change.

    def get_hyperbolic_polar_embeddings(self, token_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Length-as-radial embedding. emb = sphere_embed(ids) [B,L,d];
        rhos = ‖emb‖ [B,L,1]; thetas = emb/‖emb‖ [B,L,d] (unit). Both differentiable. NO clamp here."""
        # KEEP exactly as in current file (lines 180-184). The clamp lives in q_xt, not here,
        # so the raw embedding length is free; gradient flows through ρ_clean AND θ_clean.

    def forward(self, x0, xt, sigma, context=None) -> torch.Tensor:
        """Consume Poincaré xt [B,L,d] as-is → DDiT blocks → output_layer → logits [B,L,V]."""
        # IDENTICAL to SphereDiT.forward (193-216) byte-for-byte: x = xt; rotary; blocks; output_layer;
        # optional learn_temperature_scaling. NO sphere_normalize of xt anywhere.
```

Removed from the class (declutter): `renormalize_weights`, `get_sphere_embeddings` (callers in HFLM use
the polar getter), and the sphere-normalize inside `init_sphere_embed_from_pretrained`
(if `pretrained` init is ever used, copy raw weights — not needed for the Sudoku run, `init: hyperbolic`).
`load_pretrained_from` may stay (unused for this run; the `vocab_embed.embedding -> sphere_embed.weight`
rename still works since the param name is kept).

### 3.3 `samplers.HFLMSampler`

Reuse `SFMState` and `SFMContext` (do not add new dataclasses unless a field is genuinely needed —
HFLMState would be identical, so reuse `SFMState`).

```python
class HFLMSampler(Sampler):
    def __init__(self, noise_removal, velocity, use_float64, slerp_float64, eps,
                 temperature, p_nucleus, top_k, top_k_velocity, invert_time_convention,
                 prior_cov, rho_max):
        # Same fields as SFMSampler PLUS prior_cov, rho_max (read in get_sampler from config.algo).

    def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
                   prefix_tokens=None, prefix_lengths=None) -> SFMState:
        """Wrapped-normal prior → Poincaré xt [num_samples, L, d], ‖xt‖<1. See §4.3."""

    def _last_step_decode(self, state, log_p) -> SFMState:
        # IDENTICAL to SFMSampler (greedy=argmax / ancestral=sample); writes int tokens [B,L]; done=True.

    def _get_step_size(self, model, state) -> torch.Tensor:
        # REUSE sfm_step_size(alpha_t, alpha_s, invert_time_convention, eps). Same schedule math.

    def step(self, model, state) -> SFMState:
        """forward → posterior log_p [B,L,V] → pick token v* (top-1/sample) → geodesic-step z_t a
        fraction dt toward Poincaré(e_{v*}). Last step: argmax decode. See §4.3."""
```

### 3.4 dispatch helpers

```python
# samplers.get_sampler, add before the final raise:
if s.predictor == 'hflm':
    return HFLMSampler(
        noise_removal=s.noise_removal, velocity=s.velocity,
        use_float64=s.use_float64,
        slerp_float64=config.algo.slerp_precision == 'float64',
        eps=config.algo.eps, temperature=s.temperature,
        p_nucleus=s.p_nucleus, top_k=s.top_k,
        top_k_velocity=s.top_k_velocity,
        invert_time_convention=config.algo.invert_time_convention,
        prior_cov=config.algo.prior_cov, rho_max=config.algo.rho_max)
```

---

## 4. Data flow

### 4.1 Training `q_xt` path (the corrected version)

```
x: token ids [B,L]
  │
  ├─ get_hyperbolic_polar_embeddings(x)
  │      e_clean_rhos   [B,L,1]   = ‖sphere_embed(x)‖          (differentiable, NO clamp)
  │      e_clean_thetas [B,L,d]   = sphere_embed(x)/‖·‖        (differentiable, unit)
  │
  ├─ _sample_prior(e_clean_rhos)        # uses e_clean shape (B,L,d) for the draw
  │      e_noisy_rhos   [B,L,1]   = ‖v‖,  v~N(0, prior_cov·I) in T_o H^d
  │      e_noisy_thetas [B,L,d]   = v/‖v‖                       (no grad; pure noise)
  │
  ├─ SOFT RADIAL CLAMP on BOTH endpoints, applied ONLY when forming the geodesic endpoint:
  │      rho_clamp(ρ) = rho_max · tanh(ρ / rho_max)            # smooth, ρ_eff < rho_max < 20
  │      clean_rhos_c = rho_clamp(e_clean_rhos)   [B,L,1]
  │      noisy_rhos_c = rho_clamp(e_noisy_rhos)   [B,L,1]      # prior already <20; clamp is a guard
  │      (thetas unchanged — clamp is radial only)
  │
  ├─ if use_pure_noise:
  │      z_t = hyperbolic_polar_to_poincare_cartesian(noisy_rhos_c, e_noisy_thetas)   [B,L,d]
  │  else:
  │      slerp_t = alpha_t if invert_time_convention else (1 - alpha_t)               [B,L,1]
  │      z_t = _hyperbolic_geodesic(
  │              clean_rhos=clean_rhos_c, clean_thetas=e_clean_thetas,
  │              noisy_rhos=noisy_rhos_c, noisy_thetas=e_noisy_thetas,
  │              alpha_t=slerp_t)                                                       [B,L,d]
  │      # internally: geodesic(t=slerp_t, src=(noisy_rhos_c,e_noisy_thetas),
  │      #             dest=(clean_rhos_c,e_clean_thetas),
  │      #             cartesian_model=Geometry.POINCARE, output_coord=Coordinate.CARTESIAN)
  │
  └─ if valid_tokens is not None:        # keep prompt (givens) clean as a Poincaré point
         e_clean_cart = hyperbolic_polar_to_poincare_cartesian(clean_rhos_c, e_clean_thetas)  [B,L,d]
         z_t = torch.where(valid_tokens.bool().unsqueeze(-1), z_t, e_clean_cart)
  return z_t   # Poincaré CARTESIAN [B,L,d], ‖z_t‖ < 1
```

Key fixes vs the broken WIP (`algo.py:339-404`):
- `_sample_prior` calls **`GeoUtils.wrapped_normal(shape=..., mean=0.0, cov=self.prior_cov, ...)`**
  (not bare `wrapped_normal(m=...)`), returns the **polar pair** `(rhos[B,L,1], u[B,L,d])`.
- `q_xt` returns the **geodesic Poincaré point** `z_t`, NOT `x_t_thetas * x_t_rhos` (the old line 363 bug
  produced a raw Euclidean vector). All branches (pure-noise, geodesic, valid_tokens) are Poincaré cartesian.
- `valid_tokens` masks against `e_clean_cart` (Poincaré), not the undefined `e_clean`.

### 4.2 The `slerp_t` / `invert_time_convention` mapping + t=0/t=1 contract

S-FLM `_slerp(clean, noisy, alpha_t)` uses the convention **alpha_t=0 → clean, alpha_t=1 → noisy**, and
sets `slerp_t = alpha_t if invert_time_convention else 1 - alpha_t`.

`geo_bridge.geodesic` uses the OPPOSITE base convention: **t=0 → source, t=1 → destination**. We make the
geodesic produce the same z_t as S-FLM's slerp by choosing **src=noisy, dest=clean** and passing the SAME
`slerp_t`. Then:

| geometry | t/alpha | endpoint |
|---|---|---|
| `_slerp` (sphere) | `slerp_t = 0` | clean (S-FLM convention 0→clean) |
| `geodesic` (src=noisy, dest=clean) | `t = slerp_t = 0` | **source = noisy** |
| `geodesic` (src=noisy, dest=clean) | `t = slerp_t = 1` | **destination = clean** |

These DISAGREE at the endpoints (slerp 0→clean vs geodesic 0→noisy). To keep the *exact same data-to-noise
mapping as S-FLM* (so the schedule means the same thing), the contract is:

> **`slerp_t = alpha_t if invert_time_convention else (1 - alpha_t)`** (identical formula to SFM),
> with **`src=noisy`, `dest=clean`**, and the geodesic invariant **t=0→noisy, t=1→clean**.

So with `invert_time_convention=false` (the Sudoku run): `alpha_t=1` ⇒ clean (MDLM-like, S-FLM matches),
`slerp_t = 1 - alpha_t`, and:
- `alpha_t = 1` (clean signal) ⇒ `slerp_t = 0` ⇒ geodesic at t=0 ⇒ **noisy**. ❌ MISMATCH with S-FLM.

**Resolution (the load-bearing decision the test must pin):** geodesic endpoints are chosen so that the
sanity contract below holds with `src=noisy, dest=clean` AND a geodesic-t equal to the slerp's
"distance-from-clean". Concretely set the geodesic fraction to the *complement*:

> **`geo_t = 1 - slerp_t`**, with `src=noisy`, `dest=clean`.

Then, for `invert_time_convention=false`, `slerp_t = 1 - alpha_t` ⇒ `geo_t = alpha_t`, and:
- `alpha_t → 1` (clean) ⇒ `geo_t → 1` ⇒ geodesic at destination = **clean**. ✅
- `alpha_t → 0` (noise) ⇒ `geo_t → 0` ⇒ geodesic at source = **noisy**. ✅

Equivalently and more simply: **`geo_t = alpha_t` when `invert_time_convention=false`,
`geo_t = 1 - alpha_t` when `invert_time_convention=true`**, with `src=noisy, dest=clean`. The
implementer MUST encode `geo_t` (not `slerp_t`) inside `_hyperbolic_geodesic`, and the test MUST verify
the two endpoint cases below.

**t=0 / t=1 sanity contract (unit-tested):**
- `geodesic(src=noisy, dest=clean, t=0)` ≈ Poincaré(noisy)  (within `eps`).
- `geodesic(src=noisy, dest=clean, t=1)` ≈ Poincaré(clean)  (within `eps`).
- With `invert_time_convention=false`: `alpha_t=eps` (pure noise) ⇒ `z_t ≈ prior`;
  `alpha_t=1` (clean) ⇒ `z_t ≈ clean embedding (clamped)`.

> **OPEN QUESTION Q1 (blocking for the implementer):** confirm the exact `alpha_t→geo_t` map above against
> the S-FLM slerp at one concrete `alpha_t` (e.g. 0.3) on a tiny example, since S-FLM's slerp and the
> hyperbolic geodesic are different curves — we only need the *endpoints and monotonicity* to agree, not
> the interior. The test below pins endpoints; the orchestrator should confirm we do not also need
> interior-point parity (we believe not — geometry differs by design).

### 4.3 Sampling step path (geodesic-step toward predicted clean)

```
init_state:
  rhos, u = GeoUtils.wrapped_normal(shape=(N,L,d), mean=0, cov=prior_cov, device, dtype)  # N=num_samples
  rhos_c  = rho_max * tanh(rhos / rho_max)            # [N,L,1]  (guard)
  xt      = hyperbolic_polar_to_poincare_cartesian(rhos_c, u)   # [N,L,d], ‖xt‖<1
  prefix: prefix_embeds = Poincaré(get_hyperbolic_polar_embeddings(prefix_tokens)) (clamped) → project
  t_schedule: SAME as SFMSampler (linspace, direction by invert_time_convention), num_steps from config.

step (not last):
  log_p = model.forward(xt, sigma_t, SFMContext(temperature))      # [N,L,V]
  (optional top_k/top_p filter, float64 cast — same as SFMSampler)
  window = log_p[:, start_idx:]                                    # [N,L,V]
  pick v*:
    top_k_velocity == 1 (default): v* = window.argmax(-1)          # [N,Lw]
    velocity == 'sample':          v* = sample_categorical(window.exp())
  dest_rhos, dest_thetas = get_hyperbolic_polar_embeddings(v*)     # [N,Lw,1],[N,Lw,d] (DETACHED table ok)
  dest_rhos_c = rho_max * tanh(dest_rhos / rho_max)
  x = xt[:, start_idx:]                                            # current Poincaré point [N,Lw,d]
  dt = _get_step_size(model, state)                                # sfm_step_size (scalar/[1])
  x_new = geodesic(src_cartesian=x, cartesian_model=POINCARE,
                   dest_radial=dest_rhos_c.squeeze(-1), dest_angular=dest_thetas,
                   t=dt, output_coord=CARTESIAN)                   # one geodesic fraction toward e_{v*}
  xt[:, start_idx:] = x_new
  project prefix; step_idx += 1

step (last):  _last_step_decode(state, log_p)  # argmax posterior (eq.13)
```

Notes:
- `dt` from `sfm_step_size` is the *fraction of remaining distance to clean* per step — exactly the S-FLM
  semantics, reused so the schedule is identical. Geodesic `t=dt` moves a `dt` fraction from the CURRENT
  point toward the predicted-clean endpoint (constant-speed, so this is a partial step, not all the way).
- Embedding table is read with `.detach()` at sample time (S-FLM detaches `E` too, `samplers.py:701`),
  but the *radial* must be preserved: use `get_hyperbolic_polar_embeddings` on the detached weight, NOT
  `sphere_normalize` (the WIP bug). The radial is the length — keep it.
- `geodesic` accepts a CARTESIAN source and a POLAR destination in one call (the API allows mixed forms
  per endpoint); `cartesian_model=POINCARE` governs the cartesian source and the cartesian output.

> **OPEN QUESTION Q2 (non-blocking, record-in-W&B):** the EXPERIMENT config requests `velocity: exact,
> top_k_velocity: 1`. The exact marginalized hyperbolic velocity is OUT OF SCOPE (§1). The sampler
> implements the geodesic-step-toward-predicted-clean scheme; with `top_k_velocity: 1` this is the
> top-1 variant. Record the actual scheme (`hflm_geodesic_top1`) in W&B config so the run is reproducible.

---

## 5. The ρ-bound handling (clamp math + where applied)

Two independent radii must stay `< 20` (the `_LORENTZ_RHO_MAX` guard, `geo_bridge.py:48/81`, which fires
inside `geodesic` when a POLAR endpoint is lifted to Lorentz):

1. **Prior radius.** `prior_cov=0.25` ⇒ tangent std `s=0.5` ⇒ `E[ρ] = s·E[χ_d] ≈ s·√d ≈ 0.5·22.6 ≈ 11.3`,
   `σ_ρ ≈ s/√2 ≈ 0.35`, so batch-max over `256·180` draws ≈ `11.3 + ~5σ ≈ 13 < 20`. ✅ Confirmed feasible.
   Do NOT use `wrapped_normal`'s default `cov=1.0` (gives `E[ρ]≈22.6 > 20`, crashes).

2. **Embedding radius.** Trainable `‖e_v‖` can drift. Soft clamp **`ρ_eff = rho_max · tanh(ρ / rho_max)`**
   (`rho_max=12`) caps it below 20 with headroom, smoothly (tanh ⇒ differentiable, gradient never zero for
   finite ρ). Applied to BOTH endpoints in `q_xt` **only when forming the geodesic input** (§4.1), so the
   raw embedding length the table learns is unconstrained; the clamp shapes only what the geodesic sees.

**Where applied:** in `algo.HFLM.q_xt` and `HFLMSampler` (both the prior draw guard and the predicted-clean
endpoint), NEVER inside `get_hyperbolic_polar_embeddings` (the raw length must stay free for the table to
learn length-as-radial structure, EXPERIMENT §3/§8 "embedding collapse" watch).

**Init scale.** `init: hyperbolic` ⇒ `std=0.3` ⇒ initial `‖e_v‖ ≈ std·√d ≈ 0.3·22.6 ≈ 6.8` (well under
`rho_max=12` and `20`), comparable to `E[ρ_prior]≈11.3` so clean and noisy radii are the same order at
`t≈0`. (Justification: avoids both origin-collapse and immediate clamp saturation at init.)

### Gradient-flow guarantee (the load-bearing invariant)

CE loss → `log_x_theta` → `forward(xt=z_t)` → `z_t` → `_hyperbolic_geodesic` → `clean_rhos_c, clean_thetas`
→ `rho_clamp(e_clean_rhos)`, `e_clean_thetas` → `‖sphere_embed(x)‖`, `sphere_embed(x)/‖·‖` →
**`sphere_embed.weight`**. Every link is differentiable:
- `ρ_clean = ‖emb‖` and `θ_clean = emb/‖emb‖` are both differentiable in `emb` (the polar getter has no
  `detach`).
- `rho_clamp` uses `tanh` — smooth, non-vanishing gradient.
- `geodesic` is a smooth composition of converters (`tanh`, `cosh`, `acosh`, stereographic maps).

**Implementer MUST NOT `.detach()` the clean endpoint** (rhos or thetas) anywhere in `q_xt`. The prior
endpoint (`e_noisy_*`) needs no gradient (pure noise) and may stay attached harmlessly. This mirrors S-FLM:
gradient reaches the embedding table through the geodesic clean endpoint `z₁`, NOT through the loss head.

---

## 6. Backbone deltas (justified) + the arch caveat

- **Rename only** `SphereDiT → HyperbolicDiT`; the DDiT body/forward stays byte-identical so HFLM-vs-S-FLM
  isolates geometry, not architecture. (EXPERIMENT §8 "backbone drift".)
- **Embedding param name stays `sphere_embed`.** Justification: `HFLMSampler` and tests reference
  `model.backbone.sphere_embed.weight` (the test contract below names it), `load_pretrained_from`'s rename
  target is `sphere_embed.weight`, and renaming the param would break checkpoint compatibility for no
  benefit. The *class* is renamed; the *param* is not.
- **Drop `renormalize_weights`** (and the `optimizer_step` hook stays but is a no-op since
  `algo.renormalize_weights=False`). Drop reliance on `get_sphere_embeddings` in the HFLM path.
- **No sphere-normalize of `xt`** in `forward` — `xt` arrives as a Poincaré point and is consumed as-is.
- **Width unchanged:** `d=512`. Poincaré ball is `d`-dimensional (the Lorentz lift to `d+1` happens
  *inside* `geodesic` and is never exposed to the network). No config width change.
- **Init mode `hyperbolic` (std=0.3)** added to the `init_mode` branch; `ngpt`/`random`/`pretrained` kept.

**hyperbolic-arch caveat (documented, NOT wired):** `HyperbolicArchBlock` calls `justnorm(...)` on
activations every block, projecting onto `S^{d-1}` and **destroying the radial coordinate after layer 1**.
For a *length-as-radial* model this discards the very signal HFLM encodes. Therefore the Sudoku run uses
`hyperbolic-dit` (vanilla DiT, no internal normalization). `HFLM._validate_configuration` raises if
`model.type=='hyperbolic-arch'`. (Justification: a silent radial-collapse would invalidate the experiment.)

---

## 7. Numerical / precision notes

- **float64 geodesic path.** `_hyperbolic_geodesic` mirrors `SFM._slerp`: when
  `config.algo.slerp_precision == 'float64'`, cast `alpha_t`, `clean_*`, `noisy_*` to float64, run the
  geodesic, cast back to the original dtype. `configs/algo/hflm.yaml` sets `slerp_precision: float64`.
  The `geodesic` internals already use the cancellation-safe `cosh d - 1 = ⟨x-y,x-y⟩_L/2` form.
- **Poincaré bound `‖z‖<1` is structural:** `hyperbolic_polar_to_poincare_cartesian` clamps `tanh(ρ/2)`
  one ulp below 1, so every `q_xt`/sampler output satisfies `‖z_t‖<1` by construction.
- **`utils.print_nans`** stays in `nll` (`algo.py:428` twin) — any NaN ⇒ EXPERIMENT "refuted".
- **`_LORENTZ_RHO_MAX=20` guard** fires only for POLAR endpoints lifted to Lorentz. The clamp (§5) keeps
  both endpoints under 12, so the guard should never fire; if it does, EXPERIMENT's ρ-bound failure path.
- **Sampler step preserves `‖z‖<1`:** geodesic output is a Poincaré point (`output_coord=CARTESIAN`,
  `cartesian_model=POINCARE`), so the bound holds every step.

**Exact-marginalized hyperbolic velocity (future, if Q2 escalates):** would require a hyperbolic
`log_{z_t}(e_k)` for all `V` (a per-token tangent vector at `z_t` toward each `e_k`), summed weighted by
`p_k`, then `exp_{z_t}(dt·v)`. `geo_bridge` exposes geodesics and converters but not a vectorized
`log_map`/`exp_map` on `H^d` over the full vocab; building that (with the `_LORENTZ_RHO_MAX` guard over
`B·L·V` lifts) is the blocking cost. Out of scope for this experiment.

---

## 8. Edge cases & invariants

- **`use_pure_noise=True`:** `z_t` = Poincaré(prior) directly (no geodesic call); `alpha_t` forced to 1 by
  `nll` (twin behavior). Prior radius already `<20`; the §4.1 clamp is still applied as a guard.
- **`valid_tokens` (Sudoku givens):** prompt positions are forced to the CLEAN Poincaré point
  `e_clean_cart` (clamped), keeping givens uncorrupted — matches S-FLM and closes the WIP `e_clean`
  undefined bug (EXPERIMENT §8 "eval contamination").
- **`d≥2` required:** `wrapped_normal`/`geodesic` need `d≥2` (`_uniform_sphere` rejects `d=1`). `d=512` ✅.
- **Zero-variance prior (`prior_cov=0`)** is not used; `_validate_configuration` asserts `prior_cov>0`.
- **NOT handled:** ρ exceeding 20 despite the clamp (would raise — surfaces as the EXPERIMENT ρ-bound
  failure, by design, not silently swallowed); interior-curve parity with S-FLM's slerp (different
  geometry by design — only endpoints/monotonicity match).

---

## 9. Config / dispatch wiring (concrete)

`configs/algo/hflm.yaml` (clone of `sfm.yaml`):
```yaml
name: hflm
diffusion_type: sphere        # reuses trainer_base.Diffusion path (unchanged); label only
backbone: hyperbolic-dit
parameterization: mean
time_conditioning: True
loss_type: ce
T: 0
causal_attention: False
adaLN: True
slerp_precision: float64
eps: 1e-6
invert_time_convention: true  # overridden to false on the CLI for the Sudoku run
renormalize_weights: False
prior_cov: 0.25               # tangent VARIANCE (s²); s=0.5 ⇒ E[ρ]≈11.3 < 20
rho_max: 12                   # soft radial clamp ρ_eff = 12·tanh(ρ/12)
```

`configs/model/tiny-hyperbolic-dit.yaml` (clone of `tiny-sphere-dit.yaml`):
```yaml
name: tiny
type: hyperbolic-dit
hidden_size: 512
cond_dim: 128
length: 180
n_blocks: 8
n_heads: 8
dropout: 0.1
init: hyperbolic             # std=0.3 ⇒ ‖e_v‖≈6.8 at init
learn_temperature_scaling: False
eps: 1e-6
pretrained_ckpt_path: null
```

`configs/sampler/hflm.yaml`: per EXPERIMENT §4 (predictor: hflm, steps: 180, noise_removal: greedy,
velocity: exact, top_k_velocity: 1, use_float64: true, temperature: 1.0, p_nucleus: 1.0, top_k: -1,
num_sample_batches: 2, num_sample_log: 2).

`trainer_base.py:69-85` — add:
```python
elif self.config.model.type == 'hyperbolic-dit':
    self.backbone = models.hyperbolic_dit.HyperbolicDiT(self.config, vocab_size=self.vocab_size)
elif self.config.model.type == 'hyperbolic-arch':
    self.backbone = models.hyperbolic_arch.HyperbolicArch(self.config, vocab_size=self.vocab_size)
```

`main.py:~629` — add `elif config.algo.name == 'hflm': diffusion_model = algo.HFLM`.

`main.py` `_sudoku_eval` rank-0 block (`~583`, after `accuracy` computed) — add W&B logging. The function
has no `wandb_logger` in scope today; the minimal change is to build a logger from `config.wandb` (mirror
`main.py:218-222`) OR pass the existing one in. Recommended minimal form:
```python
if config.get('wandb', None) is not None:
    import wandb
    wandb.log({'sudoku/exact_match_acc': num_correct / total * 100})
```
> **OPEN QUESTION Q3 (blocking, small):** `_sudoku_eval` currently has no W&B logger object. Confirm with
> orchestrator whether to (a) instantiate `WandbLogger(**config.wandb)` inside `_sudoku_eval` rank-0, or
> (b) re-init `wandb.init/log` directly, or (c) thread the training-time logger through. (a) or (b) are
> both ~3 lines; pick one for consistency with how the repo logs elsewhere.

---

## 10. Test contracts (for the test-writer to encode)

Geometry/shape:
1. `HFLM.q_xt(x, alpha_t, use_pure_noise=False, valid_tokens=None)` returns `[B,L,d]` with `‖z_t‖<1`
   (max norm strictly `<1`) for `d=512`, several `alpha_t` in `(0,1)`.
2. `HFLM.q_xt(..., use_pure_noise=True)` returns Poincaré(prior), `[B,L,d]`, `‖·‖<1`.
3. **Geodesic endpoints:** with `src=noisy, dest=clean`, `geo_t=alpha_t` (invert=false),
   `q_xt` at `alpha_t≈eps` ≈ Poincaré(prior) and at `alpha_t=1` ≈ Poincaré(clean-clamped) within `1e-4`
   (float64). (Pins §4.2.)
4. **`valid_tokens` masking:** prompt positions (`valid_tokens==0`) equal the clean Poincaré embedding
   exactly; generate positions differ. (Closes the givens-leak bug.)

Gradient:
5. A tiny forward+CE loss (`B=2,L=4,d=8`, small vocab) yields a non-zero
   `model.backbone.sphere_embed.weight.grad` after `loss.backward()` — gradient REACHES the embedding
   table through `z_t` (NOT only through the loss head). Assert `.grad is not None` and `.grad.abs().sum()>0`.
6. The clean endpoint is NOT detached: monkeypatch / inspect that `rho_clamp` output `requires_grad` and
   is connected to `sphere_embed.weight`.

ρ-bound:
7. With `prior_cov=0.25, d=512`, a `(256,180,d)` `wrapped_normal` draw has `max(ρ) < 20` (sample-level
   feasibility; allow a generous margin, e.g. assert `<18`).
8. After `rho_clamp` with `rho_max=12`, `max(ρ_eff) < 12` for arbitrarily large input ρ (feed ρ=1e3).
9. `q_xt` never feeds a polar endpoint with `ρ>20` into `geodesic`: with embedding weights manually scaled
   to huge norm, `q_xt` does NOT raise (clamp protects it).

Sampler:
10. `HFLMSampler.init_state` produces `xt` with `‖xt‖<1` and shape `[N,L,d]`.
11. `HFLMSampler.step` (non-last) preserves `‖xt‖<1` and shape; one step moves `xt` (not a no-op).
12. `HFLMSampler.step` (last) returns int tokens `[N,L]` and `state.done=True`; decode = argmax posterior.

Dispatch:
13. `trainer_base` constructs `HyperbolicDiT` for `model.type='hyperbolic-dit'`.
14. `main` selects `algo.HFLM` for `algo.name='hflm'`.
15. `samplers.get_sampler` returns an `HFLMSampler` for `predictor='hflm'` with `prior_cov`/`rho_max` set.
16. `HFLM(model.type='hyperbolic-arch')` raises in `_validate_configuration` (justnorm caveat).

---

## 11. Blocking open questions (for the orchestrator → user)

- **Q1 (geometry-t mapping):** confirm the `alpha_t→geo_t` mapping in §4.2 (`geo_t=alpha_t` when
  `invert_time_convention=false`, `src=noisy, dest=clean`) — specifically that we only require endpoint +
  monotonicity parity with S-FLM's slerp, not interior-curve parity. The test pins endpoints; if interior
  parity is required, the design needs a different (slerp-on-tangent) interpolation. **Recommended: accept
  endpoint-only parity (geometry differs by design).**
- **Q3 (W&B logger in `_sudoku_eval`):** `_sudoku_eval` has no logger object in scope. Pick (a)
  `WandbLogger(**config.wandb)`, (b) `wandb.init/log`, or (c) thread the training logger. Needed before
  the primary metric lands in W&B (EXPERIMENT §6 "Action required"). **Recommended: (b), ~3 lines, rank-0.**

Non-blocking (record-only):
- **Q2 (velocity scheme):** EXPERIMENT requests `velocity: exact, top_k_velocity: 1`; we ship the
  geodesic-step-toward-predicted-clean top-1 scheme (exact marginalized velocity is out of scope). Record
  `hflm_geodesic_top1` in W&B config. No decision needed unless the orchestrator wants exact marginalization
  (large added cost — see §7).
