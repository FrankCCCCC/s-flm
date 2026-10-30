## GOAL

Read Agent.md, train HBFM on tinystories with ada curv and discrete-time ELBO loss. Since computing ELBO is time exhausted, implement MC on words to compute ELBO to accelerate it. Choose a proper MC sample count for the MC ELBO.

Shared by both bases
- Model: DiT (768 hidden, 12 layers, 12 heads), naive readout (`FORWARD_TYPE=naive`), readout in float32 (`READOUT_PRECISION=float32`)
- Product Manifold: `FACTOR_DIM=3`, `EMBED_DIM` = 3xM (M = number of components,  M = 16 -> `EMBED_DIM=48`)
- Training: LR 3e-4, 30k steps (`MAX_STEPS=30000`), global batch 512, sequence length 256, bf16, EMA 0.9999, `ce` loss
- Word Embedding: GPT-2 XL / GPT-2 embedding reduced by PCA
- Time proposal: `TIME_CONVERSION_MODE=exp`, (`PROPOSAL_RATE`, `GAUSS_CURV`) from the grid below (anchor: 3, -0.5)

### Time proposal and curvature under the ELBO (`exp`)

**Is `exp` proper for the ELBO? Yes.** The weight 1/(r u) · |α'_s| turns the uniform schedule draw
into ∫ rate dt over the trained heat-time range [t_min, T] exactly, with T = ln(1/ε)/r ≈ 6.91/r and
t_min ≈ ε/r (ε = `sampling_eps` = 1e-3; `elfm_hbfm_ada_curv_elbo_sudoku/math.md` §1). The
reconstruction CE is taken at T (`reconstruction_loss`), and the prior term is zero. The bound is
tight at the Bayes posterior for any T, so NELBOs from different rates and curvatures can be compared
directly (unlike `val/ppl` under CE). Under the adaptive curvature, position i covers [λ t_min, λ T];
the mean is unchanged by a global λ. Caveats:
- The part of the integral in [0, t_min] is not sampled. It adds at most
  16 · 2(d−1)²|K| · t_min · λ = 0.128 λ / r_unit nats/token (r_unit = r/|K|). The reported NELBO
  sits below the true bound by up to that amount; the column "slack" lists it at λ = 1.
- The horizon follows from the rate: T = 6.91/r physical, or 6.91/r_unit in unit time. `trunc_exp`
  cuts it to `ub` and moves the rest into the reconstruction term: the CE-trained (H³)¹⁶ model with
  trunc_exp ub 1 (λ = 1/4) paid 0.43 nats there (`elbo_tune_tinystories_256/RESULTS.md` §1). `exp`
  avoids this, which is why it is the better default for the ELBO, even though `trunc_exp` won
  GenPPL under CE.
- `T_MAX` must be set to T. `hbfm.sh` defaults to UNIT_T_MAX/|K| = 4 at K −0.5, past the trained
  range (2.30 at rate 3), and `exp` does not validate it. The exact NELBO is the likelihood of the
  process that stops at T and decodes there.
- ELBO gradients are noisier than CE gradients. On Sudoku the ELBO gradient SNR is 0.05–0.13 per
  micro-batch. On (H³)³ TinyStories at 10k steps, the ELBO-trained cells scored 2.42–2.62 under the
  bound vs 2.26 for the CE control (`elbo_tune_tinystories_256/RESULTS.md` §3). The 16-word MC rate
  adds variance on top of that, so keep the CE twin as the control.

**Grid** (16 H³ factors, `EMBED_DIM=48`). Rationale from past results:
- CE sweep (`claude_test_hbfm_tinystories_256/RESULTS.md`): rate 3 at K −0.5 (unit rate 6) was the
  best point. Rates 1.5–5 were within 5–8% (one seed each, not resolved). Rates 10–20 lost 13–24%.
  K −0.25 and K −1 at rate 3 lost 10–14%, and that loss was mostly the unit rate changing
  (rate/|K| equivalence held to ~6%).
- Adaptive curvature with `exp` rate 20 left λ at its guard 1/4 and broke the model, GenPPL 24–210
  (`hbfm_rlf_ada_curv_tinystories_256/RESULTS.md`). Rate ≥ 10 is out.
- Under the ELBO, λ on (H³)³ rose to 1.7–2.6, stretching the range from [0, 2.30] toward [0, 6].
  The bound seems to want more heat time than the CE did, so the grid leans toward the low-rate side.

| cell | `PROPOSAL_RATE` | `GAUSS_CURV` | unit rate | R = 1/√\|K\| | T = `T_MAX` | unit horizon (λ ∈ [¼, 4]) | slack (λ = 1) | role |
|---|---:|---:|---:|---:|---:|---|---:|---|
| r3_K0.5 | 3 | −0.5 | 6 | 1.41 | 2.30 | 1.15 (0.29–4.6) | 0.021 | anchor: the CE optimum |
| r1.5_K0.5 | 1.5 | −0.5 | 3 | 1.41 | 4.61 | 2.30 (0.58–9.2) | 0.043 | longer horizon, smaller reconstruction term |
| r5_K0.5 | 5 | −0.5 | 10 | 1.41 | 1.38 | 0.69 (0.17–2.8) | 0.013 | high side of the CE basin |
| r6_K1 | 6 | −1.0 | 6 | 1.00 | 1.15 | 1.15 (0.29–4.6) | 0.021 | same unit proposal as the anchor; isolates the state scale R |

Run the ada arm (`CURV_BOUND=4`) on all four cells. Run the fixed arm (`CURV_BOUND=1`) on the anchor
only, so the curvature's effect is read at the anchor. Rank cells by the exact NELBO; report GenPPL
at `T_MAX` = T alongside it.
<!-- - Eval: validation PPL plus 64 samples with 180 steps and `TOPK_VELOCITY=1`. NELBO is a separate pass over the full validation set with `LOSS_TYPE=elbo SEED=1 EVAL_BS=16` -->

- `ELBO_NUM_WORDS=16` (`algo.elbo_num_words`) replaces the path-KL rate's sum over all V = 50257
  words with a Monte-Carlo estimate over 16 words drawn from p_theta per token
  (`numeric/loss.py: Loss.bridge_loss_elbo_mc`). The estimate is unbiased in value and gradient: a
  U-statistic over pairs of samples, with p_theta(v)/sg(p_theta(v)) factors carrying the gradient.
  At V = 50257 and 16 H³ factors, forward + backward costs 0.11 s vs 12.6 s for the exact rate
  (CPU, 128 tokens). Checked by `tests/test_elbo_loss.py::test_hbfm_path_kl_monte_carlo_is_unbiased`.
  The continuous-time path-KL ELBO itself is unchanged; only the sum over words is sampled.
- Evaluation uses the exact NELBO: the sample script leaves `elbo_num_words` at null.
- Sweep: `python experiments/hbfm_elbo_tinystories_256/sweep.py --site {a100,a6000} [--seeds 1,2,3]`
  (one job per arm × seed: train → val PPL + 64 samples (180 steps, `TOPK_VELOCITY=1`) → exact
  NELBO pass `LOSS_TYPE=elbo SEED=1 EVAL_BS=16` → `eval/nelbo.json` → λ per keep checkpoint for `ada`).
  `--dry-run` prints the job bodies. Nothing has been submitted yet.

---

## Following are reference config in other experiment

| | `ada` | `fix` |
|---|---|---|
| Curvature | learned scale $\lambda = 4^{\tanh(h)}$, between 1/4 and 4; curvature in use is $\lambda K$ | fixed at $K$ ($\lambda$ = 1) |
| `CURV_BOUND` | 4 | 1 |
| `TIME_RANGE_UPPER_BOUND` (ub, where the time proposal is cut off) | 1 | 0.5 |
| `T_MAX` (sampler start time) | 1 | 0.5 |
| Extra step after eval | $\lambda$ read per keep checkpoint | none |

Exact training call (M = 16, default rate 3, K = -0.5)

`ada` arm, seed 1, 4 GPUs (`fix`: `CURV_BOUND=1 TIME_RANGE_UPPER_BOUND=0.5 T_MAX=0.5`):


- `TIME_CONVERSION_MODE=exp`: `TIME_RANGE_UPPER_BOUND` only applies under
  `trunc_exp` / `unif`


Note here is laptop, just implement the code and wirte down the sweep, not submit jobs