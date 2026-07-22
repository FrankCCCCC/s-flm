# hGPT — a hyperbolic nGPT (design note)

## Context
Porting nGPT to hyperbolic space. nGPT normalizes every representation to the unit sphere `S^{d−1}` and
updates the residual stream with the projection retraction `Norm(h + α(f−h))`. hGPT keeps nGPT's direction
on the sphere but adds a **radius** (the magnitude nGPT discards), making each state a point in hyperbolic
space `H^d` (curvature `K<0`, `R=1/√|K|`), and replaces the sphere retraction with the **geodesic
retraction**. On OWT 0.5B a first cut descended faster than nGPT/GPT for ~1000 steps, then spiked and
plateaued — the spec below is built to remove both failure modes.

## 1. Representation — polar hyperbolic point
A token state is `h = (ρ_h, u_h)`:
- `u_h = Norm(x) ∈ S^{d−1}` — unit direction (exactly nGPT's sphere vector).
- `ρ_h ≥ 0` — geodesic radius (depth / "confidence"), obtained via the arctanh lift (§2).

Charts: Poincaré `z_h = R·tanh(ρ_h/2R)·u_h` (`‖z‖<R`); Lorentz lift `x_h = R(cosh(ρ_h/R), sinh(ρ_h/R)·u_h)`.
`K, R` are hyperparameters (fixed, or learnable per-layer; per-direction ⇒ product manifold).

## 2. Radius from a bounded scalar — the arctanh lift, stabilized
For any bounded scalar `s ∈ [−1,1]` nGPT already produces (a cosine / normalized dot product):
```
ρ = arctanh( clamp(s, −1+ε, 1−ε) )        # ≡ arctanh((1−ε)·s)
```
- `ε` caps `ρ ≤ arctanh(1−ε)` and the gradient `arctanh'(s)=1/(1−s²) ≤ 1/(2ε)` → **kills the spike**.
- Apply arctanh only to a **genuinely bounded** scalar: `arctanh(tanh(x)) = x`, so squashing a free scalar
  then lifting is a no-op. The boundedness must be geometric (unit-norm dot product).
- Keep `ρ_max = arctanh(1−ε) ≪ 20R` (the Lorentz guard `_LORENTZ_RHO_MAX=20`, geo_bridge.py:48).
- `ε` is the capacity↔stability knob: smaller ε = deeper but higher max-grad.

## 3. Residual update — geodesic retraction (the core change)
```
nGPT:  h ← Norm(h + α·(f − h))                       # sphere projection retraction ≈ SLERP
hGPT:  h ← γ(α),  γ(0)=h, γ(1)=f  =  Exp_h(α·Log_h f) # hyperbolic geodesic retraction (sinh, not sin)
```
Lorentz closed form (⟨a,b⟩_L = −a₀b₀ + Σ aᵢbᵢ):
```
δ    = (1/R)·arccosh( −⟨x_h, x_f⟩_L / R² )           # = d(h,f)/R
γ(α) = [ sinh((1−α)δ)·x_h + sinh(αδ)·x_f ] / sinh δ
```
Per sub-block (attention, then MLP):
```
f_A = hyp_point( dir = Norm(ATTN(h)),  ρ = arctanh_cap(s_A) )
h   ← geodesic(t = α_A, src = h, dest = f_A, K)
f_M = hyp_point( dir = Norm(MLP(h)),   ρ = arctanh_cap(s_M) )
h   ← geodesic(t = α_M, src = h, dest = f_M, K)
```
- `α_A, α_M` = nGPT's learnable per-dim eigen-LR — now the geodesic **step fraction** (`t=0→h, t=1→f`).
- **ρ and u move jointly** — the geodesic bows through the origin, so radius and direction cannot be
  retracted separately.
- In the s-flm repo this whole line is one call: `HyperbolicHeatKernel.geodesic(t=α, src=h, dest=f, K)`
  (geo_bridge.py:1245) — no exp/log/Möbius pieces needed.

## 4. Depth control — prevents the plateau
Add a prior/penalty pulling `ρ` off the boundary:
```
L_depth = λ · mean(ρ²)          # or KL to an origin wrapped-normal of a target radius
```
Without it the model collapses every point to the boundary → radii saturate & synchronize (the HFLM
pathology) → gradients vanish → plateau.

## 5. Scalars (carried from nGPT, same rationale) + new
Kept, because direction-normalization removes magnitude: `s_qk` (QK), `s_u`, `s_ν` (MLP), `s_z` (logit
temperature), `α_A/α_M` (eigen-LR). New: per-block radius scalars `s_A, s_M` feeding `arctanh_cap`.

## 6. Output
Direction path unchanged: `logits = s_z · (E_out · u_h)`. Optionally let depth modulate temperature
(`s_z ← s_z·g(ρ_h)`) if you want confidence to sharpen logits.

## 7. Why this removes fast-descent → spike → plateau
- **fast descent** — small ρ ⇒ arctanh≈linear, geodesic≈Euclidean ⇒ stable + extra depth capacity.
- **spike** — removed by the arctanh **input-cap** (bounds `1/(1−s²)` and the `cosh/sinh(ρ/R)` growth).
- **plateau** — removed by the **depth prior** (stops boundary collapse / radius synchronization).
- **geometric consistency** — the geodesic retraction avoids the Poincaré conformal-factor blowup
  `λ_z = 2/(1−‖z‖²/R²) → ∞` at the boundary that a fixed sphere-style ambient step triggers. The sphere is
  compact (no boundary → nGPT stable); hyperbolic space has a boundary at infinity → the retraction must be
  paired with radius control.

## 8. Open choices (yours)
- Source of `s_A, s_M`: a learned gate, the pre-norm magnitude, or a cosine to a learned anchor?
- Are token embeddings hyperbolic too (polar embedding, HFLM-style) or only the residual stream?
- Working chart: Lorentz (avoids Poincaré boundary blowup inside the geodesic) vs Poincaré (cheaper read-off).
- `K` fixed vs learnable; per-layer vs per-direction (product manifold — de-saturates chosen directions
  without a global Euclidean collapse).

## References (this repo)
- Geodesic retraction primitive: `HyperbolicHeatKernel.geodesic(t, gaussian_curvature, src_*, dest_*)` —
  `geo_bridge.py:1245` (`t=0→src, t=1→dest`, constant-speed Lorentz-LERP).
- Coordinate converters: `GeoUtils.hyperbolic_polar_to_{poincare,lorentz}_cartesian`,
  `lorentz_cartesian_to_hyperbolic_polar`, `_curvature_scale` (R=1/√|K|) — `geo_bridge.py`.
- Radial clamp reference (HFLM): `HFLM._rho_clamp` = `rho_max·tanh(rho/rho_max)` — `algo.py:654`
  (note: hGPT uses the arctanh **input**-cap instead, to avoid re-introducing saturation).
