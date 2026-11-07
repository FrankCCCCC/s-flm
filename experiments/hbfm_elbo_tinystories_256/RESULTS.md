# hbfm_elbo_tinystories_256 — results

**Final (2026-11-06). Target 1.7 reached: best 1.6832** (wave 42: peak LR 2e-3 held to 20k, then cosine; dropout 0; tied by 3e-3 held to 20k at 1.6833; answer 7). All jobs have finished; cells that failed a gate
or were dominated stopped early, as the stage table shows. Design and gate rule: `EXPERIMENT.md`.
Job log: `status.md`. W&B: https://wandb.ai/syctw/hbfm_elbo_tinystories_256 (one run
per cell; the stage NELBOs and final evals are in each run's summary).

Shared protocol: HBFM (H³)^16, DiT 768/12/12, naive float32 readout, LR 3e-4 unless stated, global
batch 512, bf16, EMA 0.9999, seed 1. NELBO = exact NELBO in nats/token (`LOSS_TYPE=elbo`, words
summed exactly, `SEED=1`, `EVAL_BS=16`).

## Final recipe

The 1.6832 cell: `trunc_exp`, ub 0.6, fix arm, K −0.5, rate 3, S = 64, GPT-2 XL embedding, dropout 0, peak LR 2e-3 held to
20k, then cosine to 0 at 30k. It has one seed. The best recipe with two seeds is the hold to 14k: mean 1.6861 (1.6849, 1.6872).

| Knob | Choice | Evidence (validation NELBO, nats/token) |
|---|---|---|
| `time_conversion_mode` | `trunc_exp` | 30k, ada: 1.828 at ub 0.5 vs 2.175 for `exp` |
| ub | 0.6 | 30k, same recipe: ub 0.5 / 0.6 / 0.7 → 1.7201 / 1.7081 / 1.7146 |
| `CURV_BOUND` | 1 (fix) | 30k, ub 0.5: 1.759 vs 1.828 for ada (bound 4) |
| K | −0.5 | 6k: K −0.25 is −0.16% at LR 3e-4 and +0.31% at 1e-3; K −1 is +0.77% at 3e-4 |
| `PROPOSAL_RATE` | 3 | 10k: rate 4 is +0.26%; 6k: flat proposal is +3.8% |
| `ELBO_NUM_WORDS` | 64 | 30k: 1.712 vs 1.721 for S = 16; S = 256 gains only −0.06% and trains 22% slower |
| Word embedding | GPT-2 XL | 6k: BERT-large 2.123 vs 2.125, a tie |
| Peak LR | 2e-3 or 3e-3 | 30k, held to 20k: 1.6832 vs 1.6833; 4e-3 is past the optimum (linear: 1.6957 vs 1.6876) |
| Schedule | hold the peak, then cosine | 30k, 2e-3: cosine 1.6940 (2 seeds), held to 6k 1.6923 (2 seeds), 14k 1.6861 (2 seeds), 20k 1.6832 (1 seed) |
| Dropout | 0 | 30k: 1.6954 vs 1.7081 at dropout 0.1 |

Baselines (`naive_ar_tinystories_s256`, LR 1e-3, 3-seed mean of `val/nll`, also an ELBO upper bound): MDLM 1.5821 and
DUO 1.6346. The best HBFM is 0.10 and 0.05 nats behind them.

## Answers

1. **trunc_exp vs exp: the finite horizon is far better.** At 30k on the full validation set, `trunc_exp`
   ub 0.5 (ada) scores **1.828**, against 2.175 for `exp` on the same arm (−16%) and 2.064 for the best `exp`
   cell (fix, −11%). Its GenPPL is 10.45 vs 18.56 (`exp` ada) and 14.20 (`exp` fix). At 1/5 of training, `trunc_exp` with
   ub 0.5 (ada) scores 2.125 on 10% of validation. The `exp` fix anchor (horizon 2.30), the best
   `exp` cell, is at 2.753 at the same step (fitted curve) and still at 2.188 at 20k. ub 1 scores
   2.181, ub 0.7 2.132, ub 0.35 2.152 and ub 0.25 2.164 (all ada), so the horizon has a broad optimum
   between 0.5 and 0.7. At 10k, ub 0.7 catches up: 2.007 vs 2.008 for ub 0.5 (+0.3% behind at 6k), and
   **at 30k ub 0.7 ada wins: 1.815 vs 1.828 (−0.71%)** on the full validation set (GenPPL 10.87 vs 10.45).
   On ada this is not a longer horizon than fix's: each position runs on the clock λ·t, and λ falls to
   0.447 at 30k for ub 0.7 (0.545 for ub 0.5), so the mean clock horizon is 0.31 vs 0.27. Both stay below
   fix's 0.5, and ub 0.5 fix (1.759) still leads ub 0.7 ada by 3.1%. On fix, ub 0.7 trails ub 0.5 by 1.2–1.4%
   at 6k, more than ada's 1.0% swing from 6k to 30k. **ub 0.6 fix at LR 1e-3 scores 2.009 at 6k**, +0.20% vs
   ub 0.5's seed mean (2.005; seeds 2.002 / 2.004 / 2.010), inside the seed range. So on fix at LR 1e-3,
   **ub 0.5 / 0.6 / 0.7 → 2.005 / 2.009 / 2.025**: flat from 0.5 to 0.6, then +1%. **At 10k ub 0.6 scores 1.908
   vs ub 0.5's 1.907 (+0.05%, same seed; +0.35% at 6k)**: the longer horizon closes its gap with training, as
   ub 0.7 ada did before it won at 30k. So the best recipe trains to 30k at ub 0.6 (888086/7: 1.985 at 6k vs
   ub 0.5's 1.977, +0.42%; **at 10k 1.883 vs 1.890, −0.37%**, the best 10k), at ub 0.7 (wave 22, 895547/9:
   **1.997 at 6k, +0.98%**, inside the +1.5% horizon limit) and at ub 0.8 (wave 24: **2.015 at 6k, +1.91%**, past
   the limit, stopped). The 6k gap grows with ub, by 0.42%, 0.56% and 0.93% per 0.1. **ub 0.6 vs ub 0.5 under the best recipe: +0.42 /
   −0.37 / −0.48 / −0.38% at 6k / 10k / 14k / 16k** (1.8215 vs 1.8303 at 14k, 1.8000 vs 1.8069 at 16k): the
   longer horizon's gain arrives by 10k and then holds at about −0.4%; **at 30k it is −0.70% (888087, 1.7081 vs 1.7201,
   full validation)**, and **ub 0.7 ends at 1.7146 (+0.38% vs ub 0.6; 895549, full validation, std 1.32; GenPPL
   10.60)**: so **ub 0.5 / 0.6 / 0.7 → 1.7201 / 1.7081 / 1.7146 at 30k**, a parabola through them bottoms out at ub ≈ 0.61. **ub 0.7 scores 1.896 at 10k** (+0.33% vs ub 0.5, +0.70% vs
   ub 0.6; it runs on to 30k), so under the best recipe **ub 0.5 / 0.6 / 0.7 → 1.890 / 1.883 / 1.896 at 10k**:
   the optimum horizon is near 0.6. The NELBO's spread also moves with ub (6k checkpoints, each at its own
   ub; table "Each model at its own horizon" below): the estimator std (across t / noise draws) is small
   (0.19–0.27) and rises with ub on both arms; the between-sequence std is ~5× larger and dominates
   `val/loss_std`. It falls with ub on ada (0.96 → 0.59 from ub 0.25 to 1) but rises on fix (1.04 → 1.21
   from ub 0.35 to 1).
2. **ub changes both the mean and the variance of the NELBO.** On one fixed (exp-trained) model, a
   shorter horizon gives a *higher* mean: the reconstruction CE at ub grows faster than the path-KL
   shrinks. It gives a *lower* variance, in both the estimator (across t, z draws) and across
   sequences (table below). Trained at its own horizon, the low variance more than pays for the
   looser bound (point 1).
3. **ada vs fix:** under `exp` the fix arm (λ = 1) wins: 2.064 vs 2.175 at 30k on the full
   validation set (−5.1%), with a better GenPPL (14.2 vs 18.6). It led by 4% from 10k on. Under
   `trunc_exp` ub 1 the two arms tie at 6k (fix 2.187, ada 2.181), but at ub 0.5, the best horizon,
   fix wins again: **2.081 vs 2.125 at 6k (−2.2%) and 1.959 vs 2.008 at 10k (−2.5%)**, and **1.759 vs 1.828 at
   30k on the full validation set (−3.7%)**, with GenPPL 9.58 vs 10.45. At LR 1e-3 the gap holds: ada scores
   2.053 at 6k vs 2.002 for fix (+2.6%). On fix, ub 0.35
   scores 2.109 at 6k (+1.4% vs ub 0.5; on ada it was +1.2%) and ub 0.7 scores 2.110 (+1.4%; on ada
   +0.3%). So the fix arm's optimum sits at ub 0.5, more sharply than ada's. Under `exp` the learned clock λ shrinks from 0.85
   at 5k to 0.66 at 30k, so it shortens the horizon to about 1.5. Under `trunc_exp` ub 0.5 it falls
   further: 0.87 / 0.66 / 0.61 / 0.555 at 2k / 6k / 10k / 24k. Learning λ costs more than it gains,
   so the remaining knobs are tested on fix (K −1, ub 0.35). **CURV_BOUND** (ub 0.5, 6k): bound 1 (fix) /
   2 / 4 → **2.081** / 2.129 / 2.125. Bound 2 (λ ∈ [1/2, 2]) is no better than bound 4: ada's λ (0.66 at 6k)
   stays inside [1/2, 2], so the tighter bound never binds. The cost is learning λ at all, not its range.
   CURV_BOUND = 1 (fix) is the choice. ada at LR 1e-3 (877672) re-checks this at the chosen LR.
4. **Rate and K (exp):** at 10k on ada, rate 1.5 is clearly worse (2.728), and rates 3 / 5 and r6_K1
   tie (2.561 / 2.570 / 2.562). At 30k r6_K1 ada ends slightly ahead of r3 ada (2.162 vs 2.175). On
   fix, rate 5 at 6k (2.756) matches the rate-3 anchor's curve (2.753). **Under `trunc_exp`** (ub 0.5,
   ada) the rate only shapes the proposal inside [0, 0.5]. Rate 6 scores 2.135 at 6k vs 2.125 for rate 3
   (+0.45%), and `TIME_CONVERSION_MODE=unif` (flat on [0, 0.5], the rate → 0 end) scores 2.207 (+3.8%).
   So rate 3 stays. **Under the best recipe** (fix, ub 0.6, S 64, cosine from 2e-3) **rate 4 scores 1.992 / 1.888 at
   6k / 10k vs 1.985 / 1.883 for rate 3** (+0.33 / +0.26%, inside the 0.32% seed-difference noise; val/loss_std 1.07 vs
   1.29), so it was stopped at 10k. Rate 3's proposal density falls 4.5× over [0, 0.5], close to the 5.6× fall of the
   path-KL rate itself; a flat proposal undersamples the early times where most of the path-KL is.
   **K under `trunc_exp`** (fix): K −1 with rate 6 and ub 0.25 keeps ub 0.5's proposal in unit time
   (unit rate 6, unit horizon 0.25), so only the state scale R changes (1.41 → 1). It scores 2.097 at 6k,
   +0.77% vs K −0.5. That is inside the 1% gate margin, and under `exp` K −1 caught up with training, so
   it continued: at 10k it is 1.966, +0.36% (gap halved). It passed the 1/3 gate to 30k, but was
   cancelled at ~13k once LR 1e-3 led by 3.8% (answer 6); its checkpoints are kept.
   K −0.25 (R 2; rate 1.5, ub 1) scores **2.077 at 6k, the best LR-3e-4 6k score** (−0.16% vs K −0.5). At equal unit
   proposal, K −1 / −0.5 / −0.25 → 2.097 / 2.081 / 2.077: a small, monotone gain from a larger state scale
   at 6k. **At 10k the order holds: 1.966 / 1.959 / 1.955** (K −0.25 −0.20% vs K −0.5). So a smaller |K|
   helps, but by ≤0.6% over a 4× range of K. K −0.25 passed the 1/3 gate; at LR 3e-4 it was not carried
   to 30k (its checkpoints are kept). **At LR 1e-3 the sign flips: K −0.25 scores 2.008 at 6k, +0.31% vs
   K −0.5 (2.002).** K moves the NELBO by under ±0.3% either way at either LR, so K −0.5 stays and K −0.25
   stopped at 6k.
5. **ELBO_NUM_WORDS:** on a fixed model the S-word MC leaves the mean unchanged (S = 4 and 64 give
   the same Δ). Its extra estimator variance is small next to the noise of the heavy-tailed `exp`
   estimate itself (the table below). **In training** (ub 0.5, ada), S = 64 scores 2.117 at 6k vs 2.125
   for S = 16 (−0.39%), but it trains 13% slower (1.60 vs ~1.84 it/s on the same A100 node). In 13% more
   steps S = 16 gains ~0.024 at this point of its curve, three times S = 64's 0.008. So per step S = 64
   is slightly better, per GPU-hour S = 16 is. S = 4 on the same base scores 2.193 at 6k (+3.2% vs S = 16)
   and is no faster (1.78 it/s on the same node). So S = 4 / 16 / 64 → 2.193 / 2.125 / 2.117 at LR 3e-4. The exp-mode S4 / S64 cells were cancelled, since no exp cell could
   pass the 1/5 gate. **At the higher LRs S = 64's gain grows:** −0.58% at 6k at constant LR 1e-3, and from
   a shared 6k checkpoint under cosine from 2e-3, −0.42% at 10k and **−0.53% at 30k (1.712 vs 1.721)**. Past
   64 it stops: **S = 256 scores 1.9780 at 6k vs 1.9772 for S = 64** (same recipe from step 0, +0.04%, inside
   the 0.32% seed noise) and trains 22% slower, so that run (wave 23) was stopped at 6k. **The MC word count
   is S = 64.** Wave 25 (S = 256 from ub 0.6's 10k) checks whether S = 256 gains late in training: **1.8195 at 14k
   vs 1.8215 for S = 64 (−0.11%)**, a quarter of S = 64's gain over S = 16 in the same 4k steps, and **1.7070 at 30k vs
   1.7081 (−0.06%; full validation, std 1.30; GenPPL 11.50, entropy 3.90, 64/64)**: no late gain either, at 22% more time
   per step.
6. **LR (fix, `exp`, 6k):** 6e-4 gives 2.735 and 1e-3 gives 2.722, vs 2.753 on the 3e-4 anchor's
   curve: 0.6% and 1.1% better. That is far less than the horizon's effect (−23%). **Under `trunc_exp`**
   (ub 0.5 fix) the LR matters far more: **LR 1e-3 scores 2.002 at 6k, −3.8% vs 2.081 at 3e-4**, the best
   score of the sweep at 1/5 of training (between-sequence std 1.21 vs 1.16). It continues to 10k (876308).
   **LR 3e-4 / 6e-4 / 1e-3 → 2.081 / 2.026 / 2.002**: monotone, with a shrinking gain (−2.6% for the first
   doubling, −1.2% for the next 1.7×). LR 6e-4 stopped at 6k (+1.2%). **LR 2e-3 scores 1.994 at 6k**
   (−0.36% vs 1e-3; std 1.23): the gain nearly stops at 2e-3, so the constant-LR curve has peaked. **ub 0.7 at LR 1e-3 scores 2.025 at 6k, +1.2% vs ub 0.5** (2.002; at LR 3e-4, fix: +1.4%), so the best
   horizon does not move with the LR: ub 0.5 stays. The schedule is
   constant after a 2500-step warmup. (LR 1e-3 was first cancelled at 4.6k by mistake, on a flat logged
   train loss, ~4.28. The A100 site logs a train loss that is flat or rising for every fix cell, including
   ub 0.7 fix: ~4.3–4.4 at steps 2000–6000, val 2.110 at 6k. It was resubmitted from step 4500.) **BERT-large vs GPT-2 XL:** a tie. On the ub 0.5 ada base,
   BERT-large-uncased (PCA to 48-d, mapped onto GPT-2 words) scores 2.123 at 6k, against 2.125 for
   GPT-2 XL (−0.1%), so the word embedding's source is not a lever. GPT-2 XL stays.
7. **Target 1.7: reached. The best finished cell is ub 0.6, peak LR held to 20k and then cosine to 0 at 30k, S = 64,
   dropout 0: 1.6832 at peak 2e-3** (973872, wave 42, desa A6000; 30k, full validation 1172/1172; std 1.30; GenPPL 13.21,
   entropy 3.77, 64/64 unique), **tied by 1.6833 at peak 3e-3** (973879, wave 44, desa A5000; GenPPL 9.90, entropy 3.90,
   64/64), 1.0% below the target. At 2e-3 the hold length is monotone: held to 6k 1.6901, to 14k 1.6849, to 20k 1.6832
   (−0.41% end to end), but the last step (−0.10%) is inside the seed gap (0.16%) and inside the run-to-run spread at a
   high LR: waves 43 and 44 share the config and seed through 14k, yet scored 0.6–0.7% apart at 6k–16k on different desa
   GPUs, and wave 43 (3e-3 held to 14k) ended at 1.6911. Peak 4e-3 with a linear decay (wave 41) ended at 1.6957, +0.48%
   vs 3e-3 linear: past the optimum. **More steps at a high LR win at 30k**: the longer hold beats the shorter one, and
   peak 3e-3 (linear) beats 2e-3 (cosine), although both score worse or equal at 10k / 16k. The 10k / 16k gates measured
   against a cosine run are biased against such schedules (they are behind mid-training and ahead at 30k), so the
   held-peak cells got only the 6k gate. **Seed 2 of the two earlier holds (same A100 site):** held to 6k 1.6946 (970595,
   wave 39; seed gap 0.27%; GenPPL 10.74, entropy 3.95), held to 14k 1.6872 (973883, wave 45; seed gap 0.13%; GenPPL 11.20,
   entropy 4.06). **Two-seed means: cosine 1.6940, held to 6k 1.6923, held to 14k 1.6861.** The hold to 6k alone is no
   proven gain (−0.10% vs the cosine, inside the seed gaps). The hold to 14k is a gain: −0.37% vs the hold to 6k, with
   seed ranges that do not overlap, and −0.47% vs the cosine (which ran on desa, so that one crosses sites). The 20k holds have one seed each: 1.6832 is
   −0.17% vs the 14k mean, inside the noise, so 20k over 14k is not settled. These were the last waves (no new ones after
   wave 40, per the user; all ran to 30k without gate cancellations). Before wave 42, the best was **peak 2e-3 held to 14k: 1.6849** (921565, wave 31, A100; GenPPL 11.12,
   entropy 3.85), −0.31% vs the hold to 6k (same seed and site), −0.16% vs wave 34 and −0.54% vs the cosine two-seed
   mean (1.6940). Before it, **the best finished cell was ub 0.6,
   peak LR 3e-3 with a linear decay to 0 at 30k (warmup 2500),
   S = 64, dropout 0: 1.6876** (937627, wave 34; 30k, full validation 1172/1172; std 1.29; GenPPL 10.32, entropy 3.87,
   64/64 unique), 0.73% below the target: −0.46% vs wave 28 (cosine from 2e-3, same seed), −0.38% vs the cosine two-seed
   mean (1.6940) and −0.15% vs wave 30. At 6k / 10k / 16k it was −0.40 / −0.11 / +0.07% vs wave 28, so the gain
   arrives after 16k, while its LR decays from 1.5e-3 to 0 (wave 28: from 1.0e-3). Under the cosine the
   same 3e-3 failed its 10k gate (+0.62%). Single seed; wave 41 (4e-3, linear) continues the LR axis. Next is ub 0.6,
   peak LR 2e-3 held to 6k and then cosine to 0 at 30k, S = 64, dropout 0: 1.6901** (917963, wave 30, A100; 30k, full validation; std 1.30; GenPPL 11.25, entropy 3.87, 64/64 unique),
   0.58% below the target and −0.32% vs the same recipe under the cosine from step 0 (wave 28, below). That is inside
   the 0.32% two-seed noise: wave 30 led by −0.16 / −0.11% at 6k / 10k and trailed by +0.09% at 16k, and at 6k the two
   schedules are almost the same LR, so ~0.15% is run (and A100 vs A6000) noise. **Wave 28's seed 2 finished at 1.6927**
   (932948; seed gap 0.16%, two-seed cosine mean 1.6940), so wave 30 is −0.23% vs the cosine mean and −0.15% vs its better
   seed: suggestive, not settled. Wave 39 (wave 30 at seed 2) then erased this lead; the hold to 14k did show a gain (see
   above), and wave 40 (held to 10k) was stopped as dominated. Before it, **the best finished cell was ub 0.6,
   cosine from 2e-3 at step 0, S = 64, dropout 0: 1.6954**
   (915962, wave 28; 30k, full validation; std 1.29; GenPPL 12.13, entropy 3.91, 64/64 unique), 0.27% below the target.
   Against the same recipe at dropout 0.1 (888087, 1.7081) it is −0.74%: dropout 0 led by −1.30 / −1.41 / −1.34% at
   6k / 10k / 16k, and the decay to 0 recovered half of that for dropout 0.1. Its samples score a worse GenPPL (12.13 vs
   10.91; 64 samples, noisy). Its second seed (932948) scored 1.6927 (−0.16%; 6k / 10k / 16k: −0.4 / −0.4 / −0.19%; GenPPL 11.02, entropy 3.91). Before it, **the best finished cell was
   ub 0.6 with a linear decay from 2e-3 to 0 (S = 64, dropout 0.1): 1.7053** (903969, wave 26; 30k, full validation; std 1.30; GenPPL 9.92, entropy 3.94, 64/64 unique), 0.31%
   above the target and −0.17% vs the cosine (−0.56 / −0.29 / −0.11% at 6k / 10k / 16k: inside the 0.32% noise at 30k,
   but the same sign at every stage). Next is the same recipe under the cosine from 2e-3 at step 0 (S = 64,
   dropout 0.1): 1.7081** (888087; 30k, full validation; std 1.30; GenPPL 10.91, entropy 3.94, 64/64 unique), 0.48% above
   the target, −0.70% vs the same recipe at ub 0.5 (wave 21, 1.7201) and −0.23% vs 880403 (1.712). Two levers are still
   running on top of it: 880403's held-peak schedule (wave 29, worth −0.47% at ub 0.5) and **dropout 0 (wave 28: −1.30% /
   −1.41% at 6k / 10k, 1.9597 / 1.8568 vs 1.9855 / 1.8833, the largest gain since the LR)**; wave 30 = both, wave 31 =
   dropout 0 with the peak held to 14k. **The held peak does not carry over to ub 0.6: wave 29 scored 1.8876 at 10k, +0.23%
   vs the cosine from step 0 (888087, 1.8833)**, where at ub 0.5 it was −0.26% at 10k; stopped (dropout 0.1, and past the
   1.8846 bar). Wave 32 put wave 28 (dropout 0) at ub 0.7: 1.9747 at 6k, +0.77% vs ub 0.6 (the dropout-0.1 gap was +0.60%), and with
   ub 0.7's 30k final at +0.38% it was stopped: dropout 0 does not move the horizon optimum. The previous best was the S = 64 twin of the S = 16 cosine run: **1.712** (880403,
   r3_K0.5_ub0.5_lr2e-3_cos_S64; 30k, full validation; std 1.26; GenPPL 10.03, entropy 3.93, 64/64 unique),
   0.7% above the target and −0.53% vs S = 16 (−0.42% at 10k): S = 64's gain grows to the end of training.
   The two levers left stack on it: the horizon (ub 0.6, −0.37% at 10k) and S past 64. Wave 25 continues ub
   0.6's 10k checkpoint at S = 256 on the same schedule (900614, from ~10:15).
   **The same recipe with the cosine from step 0 (wave 21: S = 64 throughout, ub 0.5, peak 2e-3, warmup 2500) finished at
   1.7201** (std 1.26; GenPPL 9.92, entropy 3.99, 64/64), +0.47% vs 880403 (+0.26% at 10k), level with the S = 64 cosine
   from 1e-3 at 6k (1.7202). 880403 held the 2e-3 peak (constant) to 6k and decayed over the last 24k, so its LR stays higher
   to the end (1.26e-3 vs 1.03e-3 at 16k): holding the peak longer is worth ~0.5% at 30k. **The horizon's gain holds to
   30k: ub 0.6 at constant LR 1e-3 (896271, S = 16) finished at 1.7241 vs 1.7323 for ub 0.5 (878861), −0.47%** (std 1.31 vs
   1.27; GenPPL 9.19 vs 9.76, entropy 3.84 vs 3.98, 64/64), as at 10k–16k under the cosine (−0.4%). Every ub 0.6 run so far (888087,
   waves 25, 26, 28) uses the cosine from step 0, so wave 29 puts ub 0.6 on 880403's schedule: constant 2e-3 at S = 64 to
   6k, then cosine over 24k (917213 → 917215).
   **The S = 16 cosine run from peak 2e-3 finished at 1.721**
   (879486; 30k, full validation; std 1.27; GenPPL 10.09, entropy 3.92, 64/64 unique), 1.2% above the target,
   −0.63% vs constant LR 1e-3 (−0.74% at 10k) and 0.006 below its power-law projection (1.727). **Constant LR
   1e-3 finished at 1.732** (878861; std 1.27; GenPPL 9.76, entropy 3.98, 64/64 unique), 1.9% above the
   target and −1.5% vs LR 3e-4 (1.759; GenPPL 9.58). LR 3e-4's three scores (2.081 / 1.959 / 1.759) fit
   1.333 + 0.796 (step/5000)^−0.349. With that exponent, LR 1e-3's two points (2.002 / 1.907) gave 1.752
   at 30k; it finished **0.020 below that projection**, so a higher LR's early gain does *not* wash out by
   30k at a constant LR (it shrinks from −3.8% at 6k to −1.5%). K −1 and K −0.25 project to 1.752 and
   1.754. The cosine runs' projections (below; 1.705–1.735 before any decay gain) carry the same bias, so
   the remaining 1.9% is expected from the LR decay, S = 64 and the horizon.
   Earlier, **ub 0.5 ada finished at 1.828**, beating its two-point projection (1.86). Its three scores (2.125 / 2.008 / 1.828 at 6k / 10k / 30k) fit
   1.534 + 0.640 (step/5000)^−0.435, a flatter curve than the `exp` anchor's. With that exponent, the
   two-point fits give **≈1.77 at 30k for ub 0.5 fix** (LR 3e-4; running, ~10:00) and for K −0.25 / K −1 fix.
   **LR 1e-3 ub 0.5 fix** scores **1.907 at 10k** (−2.65% vs LR 3e-4's 1.959; −3.8% at 6k), so its lead
   shrinks. Its two-point fit gives ≈1.76 at 30k against ≈1.77 for LR 3e-4, with a higher asymptote
   (1.53 vs 1.47). **It finished at 1.732 (878861)**, below both fits: the higher LR's gain shrinks but
   stays (−1.5% at 30k). A cosine decay from the higher LR would keep the early gain and remove the
   late penalty. **That run is r3_K0.5_ub0.5_lr2e-3_cos** (879486, desa, ~Nov 2 05:00 + evals). It is LR
   2e-3's 6k checkpoint continued with a cosine decay from 2e-3 to 1e-6 at 30k. **At 10k it scores 1.893**
   (keep-10000.ckpt; LR 93% of the peak): −0.74% vs constant LR 1e-3 (1.907) and −3.4% vs LR 3e-4 (1.959). It
   is the best 10k score. Its 6k and 10k points on the fix power law (c 0.349) give **1.727 at 30k** before
   any gain from the decay, which acts after 10k. **It finished at 1.721.** **Its S = 64 twin scores 1.885 at 10k, −0.42% vs S = 16**
   (1.893). They share the 6k checkpoint and the schedule, so 4k steps at S = 64 bought 0.42%: S = 64's gain
   holds at a high LR and grows with training. A twin with S = 64 words
   (r3_K0.5_ub0.5_lr2e-3_cos_S64, 880403, ~Nov 2 09:30) starts from the same checkpoint on the same schedule:
   the MC word count at the end of training, where gradient noise matters most. At LR 1e-3, **S = 64 scores
   1.990 at 6k, −0.58% vs S = 16 (2.002)**, the best 6k score. That beats S = 64's −0.39% at LR 3e-4 (ada), so
   the word count matters more at a higher LR. It continues from its 6k checkpoint with a cosine decay from
   1e-3 (r3_K0.5_ub0.5_S64_lr1e-3_cos, 882037). **At 10k it scores 1.893** (LR still 93% of the peak):
   −0.71% vs S = 16 at constant LR 1e-3 (1.907), so S = 64's gain grows from 6k to 10k, and +0.45% vs peak
   2e-3 at S = 64 (1.885), so the higher peak still leads. On the fix power law its 6k/10k points give 1.735
   at 30k, against ~1.705 for the peak-2e-3 S = 64 twin (whose 6k point is the shared S = 16 checkpoint).
   Its 30k gives the peak LR under cosine: 1e-3 vs 2e-3 (wave 16), both at S = 64. **It finished at 1.720**
   (std 1.26; GenPPL 10.26, entropy 3.94, 64/64), +0.47% vs peak 2e-3 (1.712; +0.45% at 10k): the 10k gap
   held to 30k, and the S = 64 decay gained 0.015 over its power-law projection (1.735).
   A third cosine run starts from step 0 at peak **3e-3** with S = 64 (882038). Its warmup is 2500 steps,
   as for the constant runs. Under a decay, the peak can sit above the constant-LR optimum. **At 6k it
   scores 1.995** (LR 96% of the peak): +0.24% vs S = 64 at constant 1e-3 (1.990), and level with S = 16 at
   2e-3 (1.994). So at 6k, 3e-3 is past the optimum. It passes the 1/5 gate (1% margin). Its 10k score
   (~20:20) is the direct comparison with the peak-2e-3 S = 64 run (1.885). **At 10k it scores 1.9005**
   (LR 83% of the peak), +0.84% vs peak 2e-3 at S = 64 (~2.6 seed σ, answer 8), so it fails the 1/3 gate and
   was cancelled. **Under cosine, peak 1e-3 / 2e-3 / 3e-3 at S = 64 → 1.893 / 1.885 / 1.900 at 10k: 2e-3 is the
   peak.** Its slot runs that recipe from step 0 with S = 64 throughout (r3_K0.5_ub0.5_S64_lr2e-3_cos, 889513,
   ~Nov 3 02:00). **At 6k it scores 1.977**, the best 6k (−0.65% vs S = 64 at constant 1e-3, −0.85% vs
   S = 16 at constant 2e-3, the start of the copied peak-2e-3 runs). The same recipe at ub 0.6 (888086) scores
   1.985 (+0.42%, ~1.3 seed σ), as at LR 1e-3 (+0.35%). **At 10k it scores 1.890**, +0.30% vs the copied
   peak-2e-3 S = 64 run (1.885), so it passes the 1/3 gate, but its 6k lead (−0.85% vs that run's S = 16
   6k start) turned into a deficit. Between 6k and 10k the copied run, whose cosine starts at 6k, ran at a
   higher LR (93% vs 83% of the peak at 10k). The EMA is not the cause: its state, warmup counter included,
   is restored with the copied checkpoint. The finals decide (880403 ~08:00, 889513 ~Nov 3 02:00).
   **The same recipe at ub 0.6 scores 1.883 at 10k**, the best 10k: −0.37% vs ub 0.5 (1.890), after +0.42% at
   6k. Its gap swung by 0.79% in 4k steps, against 0.30% at constant LR 1e-3 (answer 1).
   The older two-point fits follow.
   Before ub 0.5 ada finished, the best finished cell was the `exp` fix anchor at 2.064. Its curve
   (1.736 + 1.150 (step/5000)^−0.673) predicted 2.08 at 30k and flattens at 1.74, so no `exp` cell
   reaches 1.7 within 30k. ub 0.5 ada scores 2.008 at 10k, already below that anchor's 30k. With the
   anchor's exponent fitted through its 6k and 10k scores, it projects to **1.86 at 30k**, levelling off
   near 1.72; it ran to 30k (860505) and ended at 1.828. ub 0.5 on fix is 2.5% lower: 2.081 at 6k and **1.959 at
   10k**. Fitted the same way, it projects to **1.80 at 30k**, levelling off near 1.66. That fit uses
   two points and a borrowed exponent, so it is rough. It ran to 30k (864328) and ended at 1.759. K −1
   (rate 6, ub 0.25) fix scored 1.966 at 10k, 0.36% behind ub 0.5 fix, and projected to the same 1.80.
   **LR 1e-3** on ub 0.5 fix is 3.8% below LR 3e-4 at 6k. If that ratio held, it would reach **~1.73 at
   30k**, close to 1.7 (one point only; its 10k score, ~06:30, gives a real fit). It is the lead cell for
   the target; LR 2e-3 checks for more.
8. **Seed noise.** ub 0.5 fix at LR 1e-3, trained with seeds 1 / 2 / 3, scores **2.002 / 2.004 / 2.010 at 6k**
   (mean 2.005, std 0.0045 = **0.22%**). The eval seed and the 10% validation subset are fixed, so this is
   training-seed noise. The difference of two single-seed runs then has a std of ~0.32%. Against that yardstick:
   ub 0.7 vs 0.5 (+1.2–1.4%), ada vs fix (+2.2–3.7%), the uniform proposal (+3.8%), S = 4 (+3.2%) and LR 3e-4
   vs 1e-3 (+3.8%) are real. K (−0.16% to +0.77%, sign flipping with the LR), BERT vs GPT-2 (−0.1%) and
   peak 3e-3 vs 1e-3 at 6k (+0.24%) are inside the noise. Rate 6 vs 3 (+0.45%) and LR 2e-3 vs 1e-3 at 6k
   (−0.36%) are ~1.3σ. S = 64 vs 16 is negative in all four comparisons (−0.39%, −0.58% at 6k; −0.42% from a
   shared 6k checkpoint, where the seed noise is smaller; −0.71% at 10k), so its gain is real.

## Stage scores (exact NELBO, the same random 10% of validation)

val/loss_std (the between-sequence std) is in brackets. Wave-1 cells are scored on their
`keep-{step}.ckpt`.

| cell | arm | horizon | 5k | 6k | 10k | 15k | 20k |
|---|---|---|---:|---:|---:|---:|---:|
| r3_K0.5_ub0.5 | fix | trunc_exp 0.5 | | **2.081** (1.16) | **1.959** (1.20) | | |
| r3_K0.5_ub0.5 | ada | trunc_exp 0.5 | | 2.125 (0.83) | 2.008 (0.82) | | |
| r3_K0.5_ub0.5_lr1e-3 (LR 1e-3) | fix | trunc_exp 0.5 | | **2.002** (1.21) | **1.907** (1.24) | | |
| r3_K0.5_ub0.5_lr1e-3, seed 2 | fix | trunc_exp 0.5 | | 2.004 (1.21) | seed check | | |
| r3_K0.5_ub0.5_lr1e-3, seed 3 | fix | trunc_exp 0.5 | | 2.010 (1.21) | seed check | | |
| r3_K0.5_ub0.5_lr2e-3 (LR 2e-3) | fix | trunc_exp 0.5 | | **1.994** (1.23) | → cosine | | |
| r3_K0.5_ub0.5_lr2e-3_cos (cosine from 2e-3 at 6k) | fix | trunc_exp 0.5 | | (1.994) | **1.893** (1.25) | | |
| r3_K0.5_ub0.5_lr2e-3_cos_S64 (same, S = 64) | fix | trunc_exp 0.5 | | (1.994) | **1.885** (1.25) | | |
| r3_K0.5_ub0.5_S64_lr1e-3 (S = 64, LR 1e-3) | fix | trunc_exp 0.5 | | **1.990** (1.22) | → cosine | | |
| r3_K0.5_ub0.5_S64_lr1e-3_cos (cosine from 1e-3 at 6k) | fix | trunc_exp 0.5 | | (1.990) | **1.893** (1.24) | | |
| r3_K0.5_ub0.5_S64_lr3e-3_cos (S = 64, cosine from 3e-3, step 0) | fix | trunc_exp 0.5 | | 1.995 (1.24) | 1.900 (1.25), stopped | | |
| r3_K0.5_ub0.6_lr1e-3 (LR 1e-3) | fix | trunc_exp 0.6 | | 2.009 (1.25) | 1.908 (1.28) | |30k 1.7241 (full val) |
| r3_K0.5_ub0.5_S64_lr2e-3_cos (S = 64, cosine from 2e-3, step 0) | fix | trunc_exp 0.5 | | **1.977** (1.23) | 1.890 (1.25) | 14k 1.8303, 16k 1.8069 |30k 1.7201 (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_cos (same recipe, ub 0.6) | fix | trunc_exp 0.6 | | 1.985 (1.27) | **1.883** (1.29) | 14k 1.8215, 16k 1.8000 |30k **1.7081** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_lin (the same, linear decay to 0 at 30k) | fix | trunc_exp 0.6 | | 1.974 (1.27) | **1.878** (1.29) |  16k 1.7980 |30k **1.7053** (full val) |
| r4_K0.5_ub0.6_S64_lr2e-3_cos (the same, proposal rate 4) | fix | trunc_exp 0.6 | | 1.992 (1.04) | 1.888 (1.07), stopped | | |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_do0 (the same, dropout 0) | fix | trunc_exp 0.6 | | **1.960** (1.26) | **1.857** (1.28) | 16k **1.7759** (1.29) |30k **1.6954** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_do0, seed 2 | fix | trunc_exp 0.6 | | 1.952 (1.26) | 1.849 (1.28) | 16k 1.7726 (1.29), seed check |30k **1.6927** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0 (dropout 0, peak held to 6k, cosine over the last 24k) | fix | trunc_exp 0.6 | | **1.957** (1.26) | **1.855** (1.28) | 16k 1.7775 (1.29) |30k **1.6901** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0, seed 2 (A100) | fix | trunc_exp 0.6 | | | | |30k 1.6946 (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0 (dropout 0, peak held to 14k, cosine over the last 16k) | fix | trunc_exp 0.6 | | | | |30k **1.6849** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_lin_do0 (dropout 0, linear decay to 0 at 30k) | fix | trunc_exp 0.6 | | **1.954** (1.26) | 1.859 (1.28) | 16k 1.7834 (1.29), stopped | |
| r3_K0.5_ub0.6_S64_lr3e-3_lin_do0 (the same, peak LR 3e-3) | fix | trunc_exp 0.6 | | **1.952** (1.27) | **1.855** (1.28) | 16k 1.7772 (1.29) |30k **1.6876** (full val) |
| r3_K0.5_ub0.6_S64_lr4e-3_lin_do0 (the same, peak LR 4e-3) | fix | trunc_exp 0.6 | | 1.970 (1.26) | 1.869 (1.28) | 16k 1.7915 (1.29) |30k 1.6957 (full val) |
| r3_K0.5_ub0.6_S64_lr3e-3_cos_do0 (wave 28 at peak LR 3e-3) | fix | trunc_exp 0.6 | | 1.968 (1.27) | 1.868 (1.29), stopped | | |
| r3_K0.5_ub0.6_S64_lr4e-3_cos_do0 (wave 28 at peak LR 4e-3) | fix | trunc_exp 0.6 | | 1.977 (1.27), stopped | | | |
| r3_K0.5_ub0.6_S64_lr1.5e-3_cos_do0 (wave 28 at peak LR 1.5e-3) | fix | trunc_exp 0.6 | | 1.964 (1.26) | 1.862 (1.27) | stopped at 12k (dominated) | |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_do0.05 (wave 28 at dropout 0.05) | fix | trunc_exp 0.6 | | 1.967 (1.26) | 1.868 (1.28), stopped | | |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd10k_do0 (dropout 0, peak held to 10k, cosine over the last 20k) | fix | trunc_exp 0.6 | | stopped at ~2.6k (dominated) | | | |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd20k_do0 (dropout 0, peak held to 20k, cosine over the last 10k) | fix | trunc_exp 0.6 | | 1.956 (1.26) | 1.862 (1.28) | 16k 1.7881 (1.30) |30k **1.6832** (full val) |
| r3_K0.5_ub0.6_S64_lr3e-3_wsd14k_do0 (the same at peak LR 3e-3, held to 14k) | fix | trunc_exp 0.6 | | 1.970 (1.27) | 1.871 (1.29) | 16k 1.7961 (1.30) |30k 1.6911 (full val) |
| r3_K0.5_ub0.6_S64_lr3e-3_wsd20k_do0 (peak LR 3e-3 held to 20k) | fix | trunc_exp 0.6 | | 1.956 (1.27) | 1.860 (1.28) | 16k 1.7856 (1.29) |30k **1.6833** (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0, seed 2 (A100) | fix | trunc_exp 0.6 | | | | |30k 1.6872 (full val) |
| r3_K0.5_ub0.6_S64_lr2e-3 → _const6k_cos (880403's schedule at ub 0.6, S = 64: constant 2e-3 to 6k, then cosine) | fix | trunc_exp 0.6 | | 1.984 (1.27) | 1.888 (1.29), stopped | | |
| r3_K0.5_ub0.7_S64_lr2e-3_cos_do0 (wave 28 at ub 0.7) | fix | trunc_exp 0.7 | | 1.975 (1.28), stopped | | | |
| r3_K0.5_ub0.7_S64_lr2e-3_cos (same recipe, ub 0.7) | fix | trunc_exp 0.7 | | 1.997 (1.29) | 1.896 (1.31) | |30k 1.7146 (full val) |
| r3_K0.5_ub0.8_S64_lr2e-3_cos (same recipe, ub 0.8) | fix | trunc_exp 0.8 | | 2.015 (1.30) | stopped | | |
| r3_K0.5_ub0.5_S256_lr2e-3_cos (same recipe, S = 256) | fix | trunc_exp 0.5 | | 1.978 (1.23) | stopped | | |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_S256 (888087's 10k, then S = 256) | fix | trunc_exp 0.6 | | (1.985) | (1.883) | 14k 1.8195 |30k 1.7070 (full val) |
| r3_K0.5_ub0.7_lr1e-3 (LR 1e-3) | fix | trunc_exp 0.7 | | 2.025 (1.27) | stopped | | |
| r3_K0.5_ub0.5_lr1e-3 (LR 1e-3) | ada | trunc_exp 0.5 | | 2.053 (0.83) | stopped | | |
| r1.5_K0.25_ub1_lr1e-3 (K −0.25, LR 1e-3) | fix | trunc_exp 1 | | 2.008 (1.21) | stopped | | |
| r3_K0.5_ub0.5_lr6e-4 (LR 6e-4) | fix | trunc_exp 0.5 | | 2.026 (1.19) | stopped | | |
| r3_K0.5_ub0.5 (CURV_BOUND 2) | ada2 | trunc_exp 0.5 | | 2.129 (0.82) | stopped | | |
| r1.5_K0.25_ub1 (K −0.25, rate 1.5) | fix | trunc_exp 1 | | 2.077 (1.16) | **1.955** (1.21) | | |
| r6_K1_ub0.25 (K −1, rate 6) | fix | trunc_exp 0.25 | | 2.097 (1.16) | 1.966 (1.20) | cancelled ~13k | |
| r3_K0.5_ub0.35 | fix | trunc_exp 0.35 | | 2.109 (1.05) | stopped | | |
| r3_K0.5_ub0.7 | fix | trunc_exp 0.7 | | 2.110 (1.21) | stopped | | |
| r3_K0.5_ub0.5_S64 (S = 64) | ada | trunc_exp 0.5 | | 2.117 (0.82) | stopped | | |
| r3_K0.5_ub0.5_S4 (S = 4) | ada | trunc_exp 0.5 | | 2.193 (0.82) | stopped | | |
| r3_K0.5_unif0.5 (`unif`) | ada | unif 0.5 | | 2.207 (1.14) | stopped | | |
| r6_K0.5_ub0.5 (rate 6) | ada | trunc_exp 0.5 | | 2.135 (0.61) | stopped | | |
| r3_K0.5_ub0.5_bert (BERT-large) | ada | trunc_exp 0.5 | | 2.123 (0.81) | stopped | | |
| r3_K0.5_ub0.7 | ada | trunc_exp 0.7 | | 2.132 (0.72) | 2.007 (0.72) | | |
| r3_K0.5_ub0.35 | ada | trunc_exp 0.35 | | 2.152 (0.91) | stopped | | |
| r3_K0.5_ub0.25 | ada | trunc_exp 0.25 | | 2.164 (0.97) | stopped | | |
| r3_K0.5_ub1 | ada | trunc_exp 1 | | 2.181 (0.61) | stopped | | |
| r3_K0.5_ub1 | fix | trunc_exp 1 | | 2.187 | stopped | | |
| r5_K0.5 | fix | exp 1.38 | | 2.756 | stopped | | |
| r3_K0.5_lr6e-4 (LR 6e-4) | fix | exp 2.30 | | 2.735 | stopped | | |
| r3_K0.5_lr1e-3 (LR 1e-3) | fix | exp 2.30 | | 2.722 | stopped | | |
| r3_K0.5 | fix | exp 2.30 | 2.886 (1.07) | (2.753 fit) | 2.457 (1.21) | 2.286 (1.26) | 2.188 (1.28) |
| r6_K1 | ada | exp 1.15 | 2.976 (0.98) | | 2.562 (0.98) | 2.411 (1.06) | 2.307 (1.03) |
| r3_K0.5 | ada | exp 2.30 | | | 2.561 (0.87) | | |
| r5_K0.5 | ada | exp 1.38 | | | 2.570 (0.87) | | |
| r1.5_K0.5 | ada | exp 4.61 | | | 2.728 (0.83) | | |

Gate (user, 2026-10-31): at 6k, NELBO < 2.77 and within 1% of the best staged cell; at 10k, < 2.46 and
within 0.5%. r1.5 and r5 ada were cancelled after failing it at 10k. ub1 ada stopped at 6k (+2.6%).

## Final evals (30k, full validation set)

| cell | arm | NELBO | val/ppl (CE) | GenPPL (64 samples) | entropy | unique | λ (30k) |
|---|---|---:|---:|---:|---:|---:|---:|
| r3_K0.5 | ada | 2.175 | 2.68 | 18.56 | 4.257 | 64/64 | 0.664 |
| r3_K0.5 | fix | **2.064** | 2.61 | 14.20 | 4.046 | 64/64 | 1 |
| r6_K1 | ada | 2.162 | (1.64)* | 13.81 | 4.028 | 64/64 | 0.670 |
| r3_K0.5_ub0.5 (`trunc_exp` 0.5) | ada | **1.828** | (1.68)* | **10.45** | 3.988 | 64/64 | 0.545 |
| r3_K0.5_ub0.5 (`trunc_exp` 0.5) | fix | **1.759** | (1.92)* | **9.58** | 3.903 | 64/64 | 1 |
| r3_K0.5_ub0.7 (`trunc_exp` 0.7) | ada | 1.815 | (1.73)* | 10.87 | 4.023 | 64/64 | 0.447 |
| r3_K0.5_ub0.5_lr1e-3 (`trunc_exp` 0.5, LR 1e-3) | fix | **1.732** | (1.92)* | 9.76 | 3.975 | 64/64 | 1 |
| r3_K0.5_ub0.6_lr1e-3 (`trunc_exp` 0.6, LR 1e-3) | fix | 1.724 | (2.02)* | 9.19 | 3.839 | 64/64 | 1 |
| r3_K0.5_ub0.5_lr2e-3_cos (cosine from 2e-3, from LR 2e-3's 6k) | fix | **1.721** | (1.91)* | 10.09 | 3.920 | 64/64 | 1 |
| r3_K0.5_ub0.5_lr2e-3_cos_S64 (the same, S = 64 from 6k) | fix | **1.712** | (1.91)* | 10.03 | 3.930 | 64/64 | 1 |
| r3_K0.5_ub0.5_S64_lr1e-3_cos (S = 64 throughout, cosine from 1e-3 at 6k) | fix | 1.720 | (1.91)* | 10.26 | 3.941 | 64/64 | 1 |
| r3_K0.5_ub0.5_S64_lr2e-3_cos (S = 64 throughout, cosine from 2e-3 at step 0) | fix | 1.720 | (1.92)* | 9.92 | 3.985 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_cos (the same at ub 0.6) | fix | **1.708** | (2.02)* | 10.91 | 3.944 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_S256 (the same, S = 256 from 10k) | fix | **1.707** | (2.03)* | 11.50 | 3.898 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_lin (ub 0.6, linear decay to 0 at 30k) | fix | **1.705** | (2.03)* | 9.92 | 3.945 | 64/64 | 1 |
| r3_K0.5_ub0.7_S64_lr2e-3_cos (the same recipe at ub 0.7) | fix | 1.715 | (2.15)* | 10.60 | 4.053 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_do0 (the ub 0.6 cosine recipe, dropout 0) | fix | **1.695** | (2.00)* | 12.13 | 3.908 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_cos_do0, seed 2 | fix | **1.693** | (2.01)* | 11.02 | 3.912 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0 (the same, peak held to 6k, then cosine; A100) | fix | **1.690** | (2.00)* | 11.25 | 3.870 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr3e-3_lin_do0 (peak LR 3e-3, linear decay to 0 at 30k, dropout 0) | fix | **1.688** | (2.00)* | 10.32 | 3.875 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0 (peak 2e-3 held to 14k, then cosine; dropout 0; A100) | fix | **1.685** | (2.00)* | 11.12 | 3.849 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr4e-3_lin_do0 (peak LR 4e-3, linear decay to 0 at 30k, dropout 0) | fix | 1.696 | (2.02)* | 10.64 | 3.934 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd20k_do0 (peak 2e-3 held to 20k, then cosine; dropout 0) | fix | **1.683** | (1.99)* | 13.21 | 3.770 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr3e-3_wsd14k_do0 (peak 3e-3 held to 14k, then cosine; dropout 0) | fix | 1.691 | (2.00)* | 9.82 | 3.888 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr3e-3_wsd20k_do0 (peak 3e-3 held to 20k, then cosine; dropout 0) | fix | **1.683** | (2.00)* | 9.90 | 3.897 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0, seed 2 (A100) | fix | 1.695 | (2.01)* | 10.74 | 3.950 | 64/64 | 1 |
| r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0, seed 2 (A100) | fix | 1.687 | (2.01)* | 11.20 | 4.058 | 64/64 | 1 |

\* val/ppl is exp of the CE objective averaged over the cell's own time proposal (rate 6: horizon
1.15; `trunc_exp`: [0, ub]). It is not a likelihood and is not comparable across rates or horizons; the NELBO is.

## EMA vs raw weights (same 10% of validation, same eval seed: paired)

The protocol evaluates the EMA (0.9999, a ~10k-step window) weights. `eval.disable_ema=True` scores the raw weights of
the same checkpoint (job 936689; scratch copy of the sample script with the switch, the repo script untouched):

| checkpoint | LR at that step | EMA | raw | raw vs EMA |
|---|---|---:|---:|---:|
| wave 26 (linear, dropout 0.1) keep-30000 | 0 | 1.7154 | 1.7114 | −0.23% |
| wave 28 (cosine, dropout 0) keep-16000 | 1.2e-3 | 1.7759 | 1.8227 | +2.6% |

While the LR is high the EMA is worth 2.6%; once the LR has decayed to 0 it lags by 0.2% (its window reaches back to
steps where the NELBO was ~1.78). All reported NELBOs keep the protocol's EMA weights.

## Does ub change the NELBO's mean and variance? (fixed model)

`nelbo_variance.py` on the fix anchor (exp-trained) at 20k and at 30k: the first 256 validation
sequences × 4 draws, the same seeds in every row. recon = the reconstruction CE at the horizon,
path = the path-KL. Δ is paired against `exp` (s.e. in brackets); est. std = the estimator std
(within a sequence, across draws); betw. std = the between-sequence std.

| evaluation | NELBO 20k | NELBO 30k | recon 30k | path 30k | Δ vs exp, 30k | est. std 20k / 30k | betw. std 30k |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp (T = 2.30) | 1.929 | 1.810 | 0.026 | 1.784 | — | 0.303 / 0.294 | 1.297 |
| trunc_exp ub 2 | 1.932 | 1.813 | 0.032 | 1.781 | +0.003 (0.001) | 0.299 / 0.290 | 1.295 |
| trunc_exp ub 1 | 1.946 | 1.832 | 0.089 | 1.744 | +0.023 (0.006) | 0.273 / 0.268 | 1.245 |
| trunc_exp ub 0.5 | 2.006 | 1.890 | 0.282 | 1.608 | +0.080 (0.020) | 0.223 / 0.224 | 1.047 |
| trunc_exp ub 0.25 | 2.262 | 2.137 | 0.755 | 1.382 | +0.327 (0.044) | 0.196 / 0.200 | 0.725 |
| exp, S = 4 | 1.933 | 1.837 | 0.026 | 1.811 | +0.027 (0.015) | 0.324 / 0.392 | 1.291 |
| exp, S = 16 | 1.934 | 1.833 | 0.026 | 1.807 | +0.024 (0.014) | 0.314 / 0.365 | 1.297 |
| exp, S = 64 | 1.936 | 1.834 | 0.026 | 1.808 | +0.024 (0.014) | 0.310 / 0.368 | 1.295 |

- **ub moves the mean up and the variance down, the same at 20k and 30k.** Cutting the horizon from
  2.30 to 0.5 costs +0.08 nats on this model and cuts the estimator std by about a quarter (0.294 →
  0.224). At ub 0.25 it costs +0.33 and the reconstruction term alone is 0.76 nats. The path-KL piles
  up early: 1.38 of its 1.78 nats by t = 0.25 and 1.61 by t = 0.5.
- **S does not move the mean.** The three S rows share one shift (+0.024 to +0.027, 1.7 s.e.), the
  same for S = 4 and S = 64, so it comes from the t / noise draws, which the word sampling offsets
  from the exp row's. It is not the MC estimate. Their std's also differ by more between 20k and 30k
  than between S values. The exp estimator is heavy-tailed (weights up to 1/(rate·ε) = 333), so a
  4-draw std is itself noisy.
- The absolute level is that block's, because the first 256 validation sequences are an easier block.
  The official, shuffled eval gives 2.160 on its own first 256 sequences at 20k (fp32 and bf16 agree,
  job 853558), 2.188 on its 10% and 2.064 on the full set at 30k. The paired Δ's and the stds are what
  this table is for.

**The same on a `trunc_exp`-trained model:** ub 0.5 fix at 30k (keep-30000, the best finished cell), scored at its
own ub 0.5 and at the shorter ub 0.35 / 0.25 (longer ones leave the trained range), and with S words at ub 0.5
(job 888648; same 256 sequences × 4 draws).

| evaluation | NELBO | recon | path | Δ vs ub 0.5 | est. std | betw. std |
|---|---:|---:|---:|---:|---:|---:|
| trunc_exp ub 0.5 (trained) | 1.509 | 0.217 | 1.292 | — | 0.235 | 1.227 |
| trunc_exp ub 0.35 | 1.550 | 0.308 | 1.242 | +0.041 (0.014) | 0.209 | 1.050 |
| trunc_exp ub 0.25 | 1.644 | 0.460 | 1.184 | +0.135 (0.028) | 0.191 | 0.864 |
| ub 0.5, S = 4 | 1.518 | 0.219 | 1.299 | +0.009 (0.010) | 0.232 | 1.246 |
| ub 0.5, S = 16 | 1.518 | 0.219 | 1.300 | +0.010 (0.010) | 0.227 | 1.245 |
| ub 0.5, S = 64 | 1.518 | 0.219 | 1.300 | +0.010 (0.010) | 0.225 | 1.246 |

- **The pattern holds on a model trained at its horizon:** a shorter evaluation ub raises the mean (+0.04 at
  0.35, +0.14 at 0.25; the reconstruction CE grows faster than the path-KL shrinks) and lowers both stds
  (estimator 0.235 → 0.191, between-sequence 1.23 → 0.86).
- **S leaves the mean unchanged** (one shared +0.01 shift within 1 s.e., from the offset t / noise draws, as
  above) and barely moves the estimator std: 0.232 / 0.227 / 0.225 for S = 4 / 16 / 64, −3% over a 16× range.
  At evaluation the word MC is a small part of the estimator noise; most of it comes from t and the bridge
  noise. In training S = 64 still gains 0.4–0.7% over S = 16 (answer 8), so its effect is on the gradient,
  not on the loss value's spread.

**Each model at its own horizon (the 6k checkpoints of the ub sweep):** every cell of the ub sweep at LR 3e-4
(rate 3, K −0.5, S = 16 in training, GPT-2 XL; ada = CURV_BOUND 4, fix = 1), its keep-6000.ckpt (EMA) scored at
its trained ub with the exact rate, on the first 512 validation sequences × 8 draws (the same seeds for every
cell; jobs 907576–81, 907620–2). The three A100 checkpoints were exported to `outputs/.../_var6k/`. The total
std matches each cell's `val/loss_std` at 6k (10% of validation, 1 draw) within 0.01–0.03.

| arm | ub | NELBO (512 seq.) | recon | path | est. std | betw. std | total std | stage score 6k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ada | 0.25 | 2.049 | 0.625 | 1.424 | 0.190 | 0.958 | 0.973 | 2.164 |
| ada | 0.35 | 2.033 | 0.587 | 1.446 | 0.190 | 0.899 | 0.916 | 2.152 |
| ada | 0.5 | **2.011** | 0.527 | 1.484 | 0.192 | 0.810 | 0.829 | **2.125** |
| ada | 0.7 | 2.018 | 0.474 | 1.545 | 0.198 | 0.711 | 0.734 | 2.132 |
| ada | 1 | 2.068 | 0.388 | 1.680 | 0.216 | 0.587 | 0.620 | 2.181 |
| fix | 0.35 | 1.997 | 0.508 | 1.489 | 0.208 | 1.038 | 1.055 | 2.109 |
| fix | 0.5 | **1.976** | 0.383 | 1.593 | 0.229 | 1.157 | 1.176 | **2.081** |
| fix | 0.7 | 2.002 | 0.294 | 1.707 | 0.247 | 1.208 | 1.229 | 2.110 |
| fix | 1 | 2.078 | 0.204 | 1.875 | 0.270 | 1.211 | 1.236 | 2.187 |

- **The estimator variance grows with ub on both arms.** ada 0.190 → 0.216 (flat up to ub 0.5, +14% by ub 1),
  fix 0.208 → 0.270 (+30% from 0.35 to 1, monotone). The longer the horizon, the more of the bound sits in
  the sampled path-KL (fix: 1.49 → 1.88 nats) rather than in the reconstruction CE, which has no t noise.
- **The between-sequence spread is ~4–5× the estimator std, and its trend depends on the arm.** On ada it
  falls by 39% from ub 0.25 to 1; on fix it rises by 17% from 0.35 to 0.7 and then stays flat. It is a
  property of the trained model (how much the per-story NELBO varies), not of the estimator; on one fixed
  model a shorter evaluation ub lowers it (tables above), so the ada trend comes from training at the horizon.
- **The mean ranking reproduces the stage scores** (minimum at ub 0.5 on both arms; the first 512 sequences
  are an easier block, so the level is ~0.11 lower), with the trade-off visible directly: from ub 0.35 to 1 the
  reconstruction CE falls (fix 0.51 → 0.20) while the path-KL rises (1.49 → 1.88).
- A stage score is one draw over 1872 sequences (117 batches × 16), so the t / noise draws alone give it an
  s.e. of 0.19–0.27 / √1872 ≈ 0.005. That is the size of the 0.22% (~0.0045) run-to-run std measured across
  training seeds with the same eval seed, so much of that std is probably estimator noise, not training noise.

**Why learned curvature has the lower std.** λ (κ = λK) is trained only on the variance: its gradient is
2 · loss_b · ∂loss_b/∂λ, i.e. it minimizes E[loss_b²] of each sequence's token-mean loss (`algo.py`, `nll`); the
curvature head reads detached trunk features (`models/hyperbolic_dit.py`, `curvature`). On TinyStories (wrap,
no prompt) its input is the origin at every position, so **κ is one number per model, the same at every
position and in every sequence** (`visualization/curvature_per_position.py`, figures
`figures/curvature_per_position_<cell>_keep-6000.png`; spread over 256 positions × 64 sequences ≤ 1e-15):

| ada, 6k | ub 0.25 | ub 0.35 | ub 0.5 | ub 0.7 | ub 1 |
|---|---:|---:|---:|---:|---:|
| λ | 1.099 | 0.848 | 0.656 | 0.552 | 0.502 |
| κ = λK (K = −0.5) | −0.550 | −0.424 | −0.328 | −0.276 | −0.251 |
| λ · ub (effective horizon at K) | 0.275 | 0.297 | 0.328 | 0.387 | 0.502 |
| between-seq. std | 0.958 | 0.899 | 0.810 | 0.711 | 0.587 |

- The clock acts as a horizon cut: the reconstruction CE follows λ · ub (ada ub 1, λ · ub 0.50: 0.388 vs fix ub
  0.5: 0.383; ada ub 0.5, 0.33: 0.527 vs fix ub 0.35: 0.508). Under `trunc_exp` the clock is therefore not
  mean-neutral (the code's premise holds for the untruncated path only): the variance objective pulls the
  horizon to 0.27–0.50 whatever ub is set to, which is why ada's NELBO is worse than fix's.
- The horizon cut does not explain the low between-sequence std. The fix cell with the same geometry as ada
  ub 1 at 6k (r1.5_K0.25_ub1: κ −0.25, ub 1; rate 1.5 vs 3) has `val/loss_std` 1.164 vs ada's 0.611, and the
  fix K sweep at equal K · ub (K −1 / −0.5 / −0.25) gives 1.157 / 1.162 / 1.164, so a flatter κ by itself does
  not lower it either. The difference is in the trained model, not in where κ ends up; not yet explained.
