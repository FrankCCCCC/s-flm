# HBFM — Hyperbolic Bridge Flow Matching (Experiment Spec)

> Status note (authoring): the file referenced by `configs/algo/hbfm.yaml`,
> `configs/sampler/hbfm.yaml`, and `tests/conftest.py` (cross-refs to §5.1, §6.1,
> §10, ablation A3, OQ-1/OQ-2) was missing on disk and absent from git history, so
> this spec was reconstructed to match those existing cross-references and then
> edited for the two locked design changes (d=64 primary; direct differentiable
> bridge `q_xt`). See `TBD` markers in §3 and §11.

HBFM is the hyperbolic analogue of Hyperspherical Flow Matching (S-FLM). It keeps
the S-FLM training recipe — plain token cross-entropy against a model that
denoises a corrupted continuous embedding — but replaces the spherical SLERP
forward process with a **target-conditioned hyperbolic heat-kernel bridge** drawn
from `geo_bridge.py` (`HyperbolicHeatKernel.poincare_bridge` for general `d`,
`BinaryHyperbolicHeatKernel.binary_poincare_bridge` for the `d=2` closed form).

---

## 1. Hypothesis

**H1 (primary).** Corrupting the target word-embedding through the *correct
hyperbolic heat-kernel bridge* (negative-curvature geometry, radius drawn from the
true `H^d` marginal `π(ρ) ∝ sinh^{d-1}(ρ) p_H(ρ;t)`) yields a denoiser that, under
the identical training/eval recipe, matches or beats spherical S-FLM on Sudoku
exact-match accuracy. **Falsifiable:** if HBFM ≥ base S-FLM (same `d`) on
Sudoku-easy exact-match by the success threshold in §2, H1 is supported;
if it is below the refute threshold, H1 is rejected.

**H2 (mechanism).** The hyperbolic bridge concentrates samples near the target
direction with a heavy-tailed radius whose spread is controlled by heat time
`t = σ`. The `hbfm/mean_rho` diagnostic should rise monotonically with `t` and the
denoiser should remain trainable (no radius saturation / NaN), distinguishing the
geometry effect from a mere reparameterization of S-FLM.

---

## 2. Success / failure criteria

Primary metric: **Sudoku-easy validation exact-match accuracy** (all 81 cells),
full 2,000-puzzle valid set, eval recipe in §10. Compared against base S-FLM
**re-baselined at the same `d=64`** (see §3).

| Outcome | Condition (over 3 seeds, mean) |
|---|---|
| **Confirm H1** | HBFM exact-match ≥ S-FLM(d=64) − 0 pt AND ≥ 2 pt absolute improvement on at least one of {easy} OR statistically indistinguishable (paired, p<0.05 across 3 seeds) while training stably |
| **Weak / inconclusive** | within ±2 pt of S-FLM(d=64), no significant difference |
| **Refute H1** | HBFM exact-match ≥ 5 pt **below** S-FLM(d=64) mean, or training diverges (NaN / `mean_rho` saturation) on ≥2 of 3 seeds |

Secondary gate (H2): `hbfm/mean_rho` increases with `t` and
`hbfm/rho_saturated_frac` < 0.01 throughout training (no `ρ > _LORENTZ_RHO_MAX`
clamp pressure). A failed H2 gate invalidates the geometry interpretation even if
H1's accuracy number passes.

Thresholds are intentionally modest: this is a feasibility/parity experiment on a
small (`d=64`) model, not a SOTA chase.

---

## 3. Baselines

| Baseline | Config / script | What it controls for |
|---|---|---|
| **base S-FLM (d=64)** — primary | `scripts/train/sudoku/sfm.sh` with `model.hidden_size=64`, `algo=sfm`, `noise=log-linear` | same backbone, same data, same CE recipe; isolates the forward process (SLERP-on-sphere vs hyperbolic bridge) at matched `d` |
| base S-FLM (d=512) — reference only | `scripts/train/sudoku/sfm.sh` (default) | the reproduced paper row (easy 77.6% / paper 81.5%, see `experiments/sudoku_reproduce/SUDOKU_REPRODUCTION.md`); **not** a fair HBFM comparand because `d` differs |

The d=512 reproduction lives in W&B project `syctw/debug` (runs `sfm_base_*`).
`TBD: exact W&B run IDs for the existing sfm_base easy/medium/hard d=512 runs` —
read from `eval_runs/sudoku/sfm_*/results.json` or the W&B project before report.
The d=64 S-FLM re-baseline does **not exist yet** and must be trained as part of
this experiment (counted in §8 budget).

---

## 4. Variables

**Changed (HBFM vs S-FLM baseline):**
- Forward corruption `q_xt`: spherical SLERP → **hyperbolic heat-kernel bridge**
  (the only mechanistic change).
- Prior/manifold: `S^{d-1}` → Poincaré ball `B^d` (input representation A,
  ball-Cartesian `z`, `‖z‖<1`).
- Noise parameterization: `sigma = t/t_max`, heat time `t ~ Uniform[t_min, t_max]`
  (replaces the S-FLM `alpha_t` log-linear schedule).

**Held fixed:**
- Backbone: `tiny-sphere-dit`, 8 blocks, **hidden 64** (primary; was 512),
  8 heads, seq-len 180, `adaLN`, `parameterization=mean`, `time_conditioning`.
- Loss: plain token cross-entropy (`loss_type=ce`), no reweighting (A3 toggles
  this).
- Optimizer/schedule: AdamW lr 3e-4, EMA 0.9999, bf16, global batch 256,
  20,000 steps.
- Data: `sudoku_generator.py`, 48,000 train / 2,000 valid, seed 42, easy=40 clues.
- Eval recipe: §10 (180 steps, exact velocity, EMA on, full valid set).
- Word embedding is **NOT renormalized** (`renormalize_weights=False`, LOCKED):
  trainable with flexible norm; only the *direction* is normalized inside the
  bridge.

### Locked design — `d = hidden_size = word-embedding dim = 64` (primary)

`geo_bridge.HyperbolicHeatKernel.sample_radial` builds the radial marginal
`sinh^{d-1}(ρ) p_H(ρ;t)` in **linear** (not log) space, which overflows float64
once `ρ ≳ 709/(d-1)`. Empirically verified: **`d=64` produces finite radii;
`d=80` returns NaN** (see `tests/test_geo_bridge_overflow.py`). Since `E[ρ] ~ (d-1)t/2`,
larger `d` pushes the marginal into the overflow regime at the heat times we use.
We therefore fix the **primary `d = 64`** everywhere (was 512). `bridge_dim: null`
→ resolves to `config.model.hidden_size = 64`.

> **Known geo_bridge limitation (deferred):** `d ≳ 80` overflows the linear-space
> marginal in `sample_radial`. Fixing it would require a **log-space marginal**
> (`logsumexp` over the `sinh^{d-1} p_H` integrand). Deferred — out of scope for
> this experiment. The `d=2` closed-form path (`BinaryHyperbolicHeatKernel`) uses
> Gruet's series, not `sample_radial`, and is **unaffected** by the overflow.

### Locked design — `q_xt` is a direct, differentiable bridge call

`geo_bridge.py`'s posterior/bridge methods are now **differentiable** (the
`@torch.no_grad()` decorators were removed). So `q_xt` is a **single direct call**
to the bridge:

- general `d`: `HyperbolicHeatKernel.poincare_bridge(ts, targets, word_embedding, output_coord=CARTESIAN)`
- `d=2`: `BinaryHyperbolicHeatKernel.binary_poincare_bridge(ts, targets, word_embedding, output_coord=CARTESIAN)`

with `ts = sigma * t_max = t` (heat time) and `word_embedding` the trainable table
(not renormalized). The bridge returns the Poincaré-ball point `z` (`‖z‖<1`)
**with gradient flowing to the embedding *direction* through the bridge itself**.
Empirically verified: `emb.grad` norm 2.12 (general-d) / 12.53 (d=2), `‖z‖<1`,
`requires_grad=True`.

The earlier verbose reparameterization (`_to_ball_cartesian` /
`_reflect_to_target_diff` reflect-and-scale) is **REMOVED** — it is no longer
needed now that the bridge is differentiable end-to-end. This matches S-FLM's
"gradient flows through `q_xt` to the embedding" design (there via SLERP; here via
the differentiable bridge). Loss is unchanged: plain CE × constant
uniform-proposal weight (`proposal_type=unif`, `weighted_ce=false` by default).

---

## 5. Algo / config fields (§5.1)

Locked in `configs/algo/hbfm.yaml` (primary, d=64):

```
name: hbfm
diffusion_type: sphere          # reuse the sphere-dit backbone family
backbone: sphere-dit            # MUST be sphere-dit
parameterization: mean
loss_type: ce
renormalize_weights: False      # LOCKED: embedding not renormalized (free norm)
hbfm_t_min: 1e-3                # uniform heat-time lower bound (d=64 primary)
hbfm_t_max: 0.05                # uniform heat-time upper bound (d=64 primary)
bridge_dim: null                # null -> config.model.hidden_size (=64)
input_repr: A                   # A = ball-Cartesian z; B (boundary dir) = OQ-2/deferred
proposal_type: unif             # only 'unif' supported
weighted_ce: false              # A3 ablation toggles this true
hbfm_log_qxt_time: false        # OQ-1 perf probe
```

> **Edit needed:** `configs/algo/hbfm.yaml` lines 17–19 still say `(d=512 primary)`
> / `=512 for tiny-sphere-dit`. Update those comments to `d=64` and set the run to
> use `model.hidden_size=64`. (Comment-only for 17–18; line 19's `null` default is
> correct once `hidden_size=64`.)

`d=2` smoke uses `BINARY_D=2`, `HBFM_T_MAX=2.0`, `n_heads=1` (head_dim≥2 backbone
constraint) — see `tests/conftest.py`.

---

## 6. Run plan (§6.1 d=2 smoke → d=64 primary)

**Phase 0 — d=2 smoke gate (first).** Tiny `d=2` HBFM training run on Sudoku-easy
exercising the closed-form `binary_poincare_bridge` path end-to-end. Gate: loss
decreases, no NaN, `‖z‖<1` holds, `hbfm/mean_rho` finite and `t`-monotone, a few
hundred steps. This validates the differentiable-bridge `q_xt` wiring before
spending GPU-hours on the primary. **Do not proceed to Phase 1 until the smoke
gate passes.**

**Phase 1 — d=64 primary.** Full 20k-step HBFM training on Sudoku-easy, 3 seeds,
plus the d=64 S-FLM re-baseline, 3 seeds. Eval per §10.

---

## 7. Ablations

Kept minimal and mechanism-isolating:

- **A1 — `d=2` vs `d=64`.** Closed-form binary bridge vs general-`d` `sample_radial`
  bridge. Isolates whether accuracy is geometry-dimension-dependent and confirms
  the two code paths agree qualitatively. (d=2 is the smoke gate; promote to a
  full run only if Phase 1 is interesting.)
- **A3 — plain vs weighted CE.** `weighted_ce: false` (default) vs `true`. Tests
  whether the constant uniform-proposal weight is sufficient or the per-token
  importance weight helps.
- **A4 — `t_min`/`t_max` sweep.** Heat-time window for `sigma = t/t_max`. Grid:
  `t_max ∈ {0.02, 0.05, 0.1}` × `t_min ∈ {1e-3, 1e-2}` (single seed each, easy
  only) to find the usable diffusion regime; then re-run the best at 3 seeds.
  Constraint: keep `t` small enough that `ρ < _LORENTZ_RHO_MAX = 20` (Cartesian
  output is refused above it) and below the float64 overflow at `d=64`.

A2 (input_repr B / boundary direction) is **OQ-2, deferred** — not run here.

---

## 8. Metrics (W&B keys)

**Primary:**
- Sudoku-easy exact-match accuracy — reported from `sudoku_eval`
  (`eval_runs/sudoku/<run>/results.json`); logged as the eval summary metric.

**Secondary:**
- `trainer/loss` — train CE per step (existing key, `trainer_base.py:326`).
- `val/loss` (val CE) and the `val/*` nll aggregates emitted on
  `on_validation_epoch_end`.

**Diagnostic (HBFM-specific, NEW — must be logged from `q_xt`):**
- `hbfm/mean_rho` — batch-mean bridge radius `ρ` (H2 monotonicity check).
- `hbfm/rho_saturated_frac` — fraction of `ρ` at/above `_LORENTZ_RHO_MAX` (clamp /
  overflow pressure; gate < 0.01).
- `hbfm/qxt_time` — per-step `q_xt` wall time, logged once when
  `hbfm_log_qxt_time=true` (OQ-1 perf probe).

> **Feasibility flag:** `hbfm/mean_rho`, `hbfm/rho_saturated_frac`, `hbfm/qxt_time`
> are **not currently logged** — `trainer_base` only logs `trainer/loss` and
> `val/*`. Minimal change: in `HyperbolicBoundaryFM.q_xt`, after the bridge call,
> compute `rho = arccosh(z[...,0])`-equivalent (or read the polar `ρ` the bridge
> already produces) and `self.log('hbfm/mean_rho', ...)` / `self.log('hbfm/rho_saturated_frac', ...)`
> on a throttled cadence (e.g. every 50 steps) to avoid per-step overhead.

---

## 9. Compute budget

Hardware: 1× NVIDIA RTX A6000 per run (cluster `gpu` partition; see `sinfo`).

The d=512 S-FLM reproduction ran ~2.5 h/run. The **d=64 primary is a much smaller
model** (hidden 64 vs 512 → roughly an order of magnitude fewer params in the
mixing layers; embedding/attention compute drops accordingly). Expect training to
be **GPU-bound but faster**; budget conservatively at **~1.5 h/run** wall time
(refine after the first run). The bridge `q_xt` adds CPU/GPU cost over SLERP
(general-`d` `sample_radial` does a 2000-pt inverse-CDF per call) — OQ-1 (`hbfm/qxt_time`)
measures whether it dominates; if it does, throttle the radial grid.

| Item | Runs | Wall time/run | GPU-hours |
|---|---|---|---|
| d=2 smoke gate | 1 | ~0.3 h | ~0.3 |
| d=64 HBFM primary (easy, 3 seeds) | 3 | ~1.5 h | ~4.5 |
| d=64 S-FLM re-baseline (easy, 3 seeds) | 3 | ~1.5 h | ~4.5 |
| A3 weighted-CE (easy, 3 seeds) | 3 | ~1.5 h | ~4.5 |
| A4 t-window sweep (6 single-seed) | 6 | ~1.5 h | ~9.0 |
| **Total** | **16** | — | **~23 GPU-hours** |

Plus ~12 min one-time Sudoku data generation (cached). Eval (`sudoku_eval`, full
2,000 puzzles, 180 steps) adds a few minutes/run. This is comfortably within a
single-A6000 day or two and well under any reasonable budget cap.

---

## 10. Eval recipe (§10)

Identical to the S-FLM Sudoku reproduction so numbers are comparable:

- `mode=sudoku_eval`, sampler `predictor=hbfm` (`configs/sampler/hbfm.yaml`).
- `steps=180`, `velocity=exact`, `noise_removal` per config (`ancestral` default;
  S-FLM repro used `greedy` — match the comparand: **use `greedy` for the headline
  number** and report `ancestral` as a secondary).
- `top_k_velocity=-1`, `p_nucleus=1.0`, `temperature=1.0`, `use_float64=true`,
  EMA on.
- Full 2,000-puzzle valid set; metric = exact match of all 81 solution cells.

> **Edit needed:** `configs/sampler/hbfm.yaml` sets `noise_removal: ancestral`. For
> a like-for-like comparison with the reproduced S-FLM rows (which used `greedy`),
> run the headline HBFM eval with `noise_removal=greedy`.

---

## 11. Failure modes to watch for

- **Radial overflow / saturation.** `d≳80` NaNs in `sample_radial`; large `t`
  pushes `ρ > _LORENTZ_RHO_MAX=20` and Cartesian conversion is *refused* (raises
  `ValueError`). Mitigation: `d=64` primary, small `t_max` (≤0.05), monitor
  `hbfm/rho_saturated_frac`. A run that hits the bound is invalid for accuracy
  comparison.
- **Gradient leakage / wrong path.** Verify `‖z‖<1`, `requires_grad=True`, and a
  nonzero `emb.grad` (sanity values 2.12 general-d / 12.53 d=2) on the first
  batch. If `emb.grad` is None/zero, the differentiable-bridge wiring is broken and
  H2 is meaningless.
- **Unfair baseline (data-prep skew).** This repo's `sudoku_generator.py` differs
  from the paper's PRISM prep (documented in `SUDOKU_REPRODUCTION.md`), biasing
  *absolute* numbers. Mitigation: compare HBFM only against the **d=64 S-FLM
  re-baseline trained here on the same generator/seed**, never against paper rows.
- **`d` mismatch contamination.** Comparing HBFM(d=64) to the existing S-FLM(d=512)
  reproduction would confound geometry with capacity. The d=64 re-baseline is
  mandatory; the d=512 rows are reference only.
- **Eval contamination.** Train (48k) and valid (2k) puzzles are disjoint by
  construction (seed 42 split); confirm no puzzle overlap before trusting accuracy.
- **Eval protocol drift.** EMA must be on; checkpoint must be the 20k-step one;
  schedule/sampler must match §10. (The S-FLM repro found a silent adaptive-schedule
  eval bug — re-audit that HBFM's sampler state is actually active at checkpoint
  load.)

`TBD: confirm the d=2 closed-form bridge and the general-d bridge agree on a shared
sanity case (e.g. d=2 via both paths) before the smoke gate.`

---

## 12. W&B project / run naming convention

- **Entity / project:** `syctw/debug` (same as the S-FLM reproduction;
  `configs/config.yaml` → `wandb.project: debug`).
- **Run names (exact strings):**
  - d=2 smoke: `hbfm_d2_smoke_easy`
  - d=64 primary: `hbfm_d64_easy_seed{1,2,3}`
  - d=64 S-FLM re-baseline: `sfm_d64_easy_seed{1,2,3}`
  - A3 weighted CE: `hbfm_d64_easy_wce_seed{1,2,3}`
  - A4 sweep: `hbfm_d64_easy_tmax{0.02,0.05,0.1}_tmin{1e-3,1e-2}`
- **Tags:** `hbfm`, `sudoku`, `d64` (or `d2`), `easy`, plus the existing
  `${noise.type}` auto-tag.
- **Group:** `hbfm_parity` (groups HBFM + the d=64 S-FLM comparand for the H1 test).

---

## Open questions (carried)

- **OQ-1** — does the differentiable bridge `q_xt` (general-`d` 2000-pt inverse-CDF)
  bottleneck training? Probe with `hbfm_log_qxt_time=true` / `hbfm/qxt_time`.
- **OQ-2** — input representation B (boundary direction on `∂B^d`) — deferred,
  not run in this experiment.
