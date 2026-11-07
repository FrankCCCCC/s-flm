# hbfm_elbo_tinystories_256 — status / handoff (final, 2026-11-06)

Goal (user, `/goal`): read Agent.md, `setup.md` and this file, and finish the experiment. `setup.md`'s
GOAL was widened by the user on 2026-10-30 19:20/19:37. It now asks to tune K, CURV_BOUND, ub, rate,
time_conversion_mode, ELBO_NUM_WORDS and the word embedding (GPT-2 XL or BERT-large-uncased) for the
lowest validation NELBO, with early stopping at 1/5 (NELBO < 3.0) and 1/3 (< 2.6). It also asks
whether ub changes the NELBO's mean and variance (trunc_exp vs exp), and to upload to wandb project
`hbfm_elbo_tinystories_256`. Design, hypotheses and the gate rule: `EXPERIMENT.md`. Deliverable:
`RESULTS.md`, plus the `## Results` pointer in `setup.md`. Nothing is committed; don't commit unless
the user asks.

**Finished 2026-11-06.** No jobs are left at either site. Best 1.6832 (wave 42); `RESULTS.md` has the final recipe.

## 1. Jobs

2026-10-31 01:00, user: target a final validation NELBO of 1.7; use a stricter early-stopping rule
at 1/5 and 1/3 and cancel the jobs that fail it; keep tuning as jobs finish. The rule, its derivation
and wave 3: EXPERIMENT.md, "Changes made on 2026-10-31".

Wave 1 (to 30k, then full evals inside the job): r3_K0.5 ada 795773 (ch2263, A100; DONE, NELBO 2.175),
r3_K0.5 fix 820570 (desa; DONE 04:45, NELBO 2.064) and r6_K1 ada 820569 (desa; DONE 07:10, NELBO 2.162). r1.5 ada 795774 and r5 ada 795775 were CANCELLED on
2026-10-31 01:10 (failed the 1/3 rule; the user authorized it).

Wave 2 (desa, staged, `--steps 6000` submitted 2026-10-30 19:53): ub1 ada 849443 and ub0.5 ada 849444
running (6k at ~01:45). Pending: fix ub1 849893; bert ada 849445, S64 ada 849446 and S4 ada 849447
at nice 500 (behind wave 3).
Wave 3 (fix arm, staged, `--steps 6000` submitted 2026-10-31 01:15): lr6e-4 853616 and lr1e-3 853617
on ch2263 A100s (running); r5 853618, S64 853620 and bert 853621 on desa (pending).
Wave 4 (ub axis around 0.5, staged, submitted 2026-10-31 01:50; ub0.5 ada scored 2.125 at 6k, the best
cell): ub0.35 ada 853912 and ub0.7 ada 853913 on A100s (running); ub0.5 fix 853911 and ub0.25 ada
853910 (nice 50, behind ub0.5 fix) on desa (pending). ub0.5 ada stage 2 (to 10k) 853909 DONE 08:39,
NELBO 2.008 at 10k, passed the 1/3 gate → 30k as 860505 (desa, running since 09:09; 2.3 it/s, so 30k at ~Nov 1 05:00, then the full evals, 1–2 h); ub1 ada stopped at 6k. nelbo_variance on fix keep-30000: 857397 (pending).
Wave 5 (2026-10-31 03:50): r6_K0.5_ub0.5 ada 856940 on A100 (pending), the proposal rate under trunc_exp.
Wave 6 (07:10): r3_K0.5_ub0.5_bert ada 859788 on desa (running since 07:11), r3_K0.5_ub0.5_S64 ada
860090 on A100 (running since 08:15), r3_K0.5_unif0.5 ada 860761 on A100 (09:15, pending). ub0.35 ada
stopped at 6k (2.152). r6_K0.5_ub0.5 856940 started on A100 at 07:53. lr6e-4 fix stopped at
6k (2.735).
ub0.7 ada scored 2.132 at 6k (+0.31%), passed → 10k as 861620 (A100, running since 10:40, ~13:45).
Wave 7 (10:40): r6_K1_ub0.25 ada 861621 CANCELLED at 12:33 (2 min in; moved to fix).
ub0.5 fix scored **2.081 at 6k** (12:31; best, −2.2% vs ub0.5 ada) → 10k as 864309 (desa, running,
~16:10). Its 30k stage 864328 is queued `afterok:864309` (`sweep.py --after`); **cancel 864328 if the
10k score fails the 1/3 gate**. Wave 7 on fix (12:35): r6_K1_ub0.25 fix 864330 and r3_K0.5_ub0.35 fix
864331 (desa, pending, for the ~13:00 slots). ub0.25 ada stopped at 6k (2.164); r6_K1_ub0.25 fix 864330
running since 12:58; ub0.35 fix 864331 running since 13:00 (bert ada's slot; bert ada 2.123 at 6k = a tie with GPT-2 XL, stopped).
r6_K0.5_ub0.5 ada stopped at 6k (15:08, 2.135, +0.45% vs rate 3). r3_K0.5_ub0.7 fix 867046 on A100
(running since 15:12, 6k ~22:00). ub0.7 ada scored 2.007 at 10k (15:38; ties ub0.5 ada) → 30k as 867339
(A100); cancel it if ub0.5 fix at 10k (~16:10) is more than 0.5% lower.
16:12: ub0.5 fix scored **1.959 at 10k** → its 30k stage 864328 started (desa, ~2.5 it/s: 30k at ~Nov 1
10:30, then the evals). ub0.7 ada 867339 now fails (+2.5%). Cancel it once the unif / S64 ada scores
pick its replacement fix cell (~16:30–17:00).
16:28: unif0.5 ada stopped at 6k (2.207, +3.8%). Wave 8: r3_K0.5_ub0.5_lr1e-3 fix 868664 on A100 (unif's slot).
16:50: ub0.5_S64 ada stopped at 6k (2.117, −0.39% vs S16 but 13% slower). r3_K0.5_ub0.5_S4 ada 869152 on A100
(S64's slot). ub0.7 ada 867339 is **kept** to 30k as the ub pair of 860505 (EXPERIMENT.md 16:50); don't cancel it.
18:26: ub0.35 fix stopped at 6k (2.109, +1.36%). r6_K1_ub0.25 fix scored 2.097 (+0.77%), passed → 10k as
870629 (desa, ~22:30). Wave 9: r1.5_K0.25_ub1 fix 870630 (desa, K −0.25, 6k ~00:00 Nov 1).
22:05: ub0.7 fix stopped at 6k (2.110, +1.43%). LR 1e-3 fix 868664 cancelled at 4.6k by mistake (the A100
train-loss artifact, EXPERIMENT.md 23:35); resubmitted as 874787 (pending, resumes from step 4500). Wave 10
(A100): r3_K0.5_ub0.5_lr6e-4 fix 874030 and r3_K0.5_ub0.5 ada2 874031 (new arm, CURV_BOUND 2), 6k ~05:00
Nov 1. Code synced to ch2263 (the curvature_ts.py fix, for 867339's λ step). **Never judge a run on the
A100's logged train loss**; only the stage NELBOs.
23:20: r6_K1_ub0.25 fix scored 1.966 at 10k (+0.36%), passed → 30k as 874737 (desa, ~Nov 1 17:30 + evals).
23:55: r1.5_K0.25_ub1 fix scored 2.077 at 6k (best 6k), passed → 10k as 875137 (desa, ~03:40).
00:10: S4 ada stopped at 6k (2.193, +3.2% vs S16). LR 1e-3 fix 874787 running (resumed from step 4500; 6k ~02:15).
01:27: **LR 1e-3 fix (ub 0.5) scored 2.002 at 6k** (−3.8% vs LR 3e-4's 2.081; the lead cell) → 10k as 876308
(A100, ~06:30). K −1 fix 874737 cancelled at ~13k (tied at 10k; checkpoints kept, resumable with
`--steps 30000`). Wave 11: r3_K0.5_ub0.5_lr2e-3 fix 876316 on its desa slot (6k ~07:00).
03:10: r1.5_K0.25_ub1 fix scored **1.955 at 10k** (−0.20% vs K −0.5; passes the 1/3 gate) but is not carried
to 30k at LR 3e-4 (checkpoints kept; resume with `--steps 30000`). Wave 12: r1.5_K0.25_ub1_lr1e-3 fix 877057
on its desa slot (K −0.25 at LR 1e-3, 6k ~08:40).
03:42: ub0.5 ada 860505 reached 30k; GenPPL 10.45 (entropy 3.99; exp fix anchor 14.20). Full NELBO and λ running.
03:51: LR 6e-4 fix (ub 0.5) stopped at 6k (2.026, +1.2% vs LR 1e-3). Wave 13: r3_K0.5_ub0.7_lr1e-3 fix 877382 on
its A100 slot (6k ~10:45).
04:21: ub0.5 ada 860505 full NELBO **1.828** (30k, full validation; std 0.79); λ 0.555 at 24k (curvature.json).
04:24: ub0.5 ada2 (CURV_BOUND 2) stopped at 6k (2.129; ada 2.125, fix 2.081). Wave 14: r3_K0.5_ub0.5_lr1e-3 **ada**
877672 (A100, ada2's slot) and r3_K0.5_ub0.5_S64_lr1e-3 fix 877673 (desa, 860505's slot), 6k ~10:00–11:30.
06:04: LR 1e-3 fix (ub 0.5) scored **1.907 at 10k** (−2.65%), passed → 30k as 878861 (A100, ~Nov 2 06:00 + evals).
06:10: `scripts/train/tinystories/hbfm.sh` gained `LR_SCHED` (constant | cosine) and `LR_DECAY_STEPS` (default
constant = unchanged). It was replaced by rename (new inode), so running jobs keep reading the old file. sweep.py
passes `LR_SCHED={cell sched}` and `LR_DECAY_STEPS=30000`, so a staged cosine run keeps one schedule across stages.
Synced to ch2263.
07:20: LR 2e-3 fix (ub 0.5) 876316 scored **1.994 at 6k** (−0.36% vs LR 1e-3; best 6k). Not continued at a constant
LR. Wave 15: r3_K0.5_ub0.5_lr2e-3_cos fix **879486** (desa, ~Nov 2 05:00 + evals). Its checkpoints/last.ckpt and
eval/nelbo_6000.json are copies of LR 2e-3's step-6000 files, so it resumes at 6k, then decays cosine from 2e-3
to 1e-6 at 30k. The launcher gained `LR_SCHED_FROM` (new inode again; synced). A constant-LR checkpoint carries
no cosine step count, so the scheduler counts from 0 at the resume: `LR_SCHED_FROM=6000` sets warmup 0 and
`t_initial` 24000 (verified: "Restored all states" at 6000; logged trainer/lr 1.99992e-3 at 6099 and 1.99966e-3
at 6199, equal to the cosine to 6 digits). Later restarts restore the cosine scheduler's own state.
09:12: r1.5_K0.25_ub1_lr1e-3 fix 877057 scored 2.008 at 6k, +0.31% vs K −0.5 at LR 1e-3 (2.002). It is inside the
1% gate but not continued: it differs from the better LR 1e-3 cell only in K (stricter rule; checkpoints kept).
Wave 16: r3_K0.5_ub0.5_lr2e-3_cos_S64 fix **880403** (desa, ~Nov 2 09:30 + evals). It starts from the same copied
LR 2e-3 step-6000 checkpoint and uses the same cosine schedule, with ELBO_NUM_WORDS 64.
10:17: ub0.5 fix (LR 3e-4) 864328 is final: **NELBO 1.759** (full validation; best final), GenPPL 9.58, 64/64 unique.
Its 3-point fit (c 0.349) puts constant LR 1e-3 at 1.752 at 30k. Its desa slot is held until ub 0.7 at LR 1e-3
(877382, A100) scores at 6k (~10:50); the candidate for the slot is ub 0.7 with the cosine schedule.
10:23: r3_K0.5_ub0.5_S64_lr1e-3 fix 877673 scored **1.990 at 6k** (best 6k; −0.58% vs S = 16 at LR 1e-3) → 10k as
881955 (desa). Cancelled after 8 min and replaced by its cosine twin (below).
10:30: r3_K0.5_ub0.7_lr1e-3 fix 877382 (A100) scored 2.025 at 6k (+1.2% vs ub 0.5): failed the gate; ub 0.5 stays.
10:40: Wave 17: r3_K0.5_ub0.5_S64_lr1e-3_cos fix **882037** (desa). Its last.ckpt is a copy of S64 LR 1e-3's
keep-6000.ckpt (that cell's last.ckpt was re-saved at 6000 by 881955, same weights). Cosine from 1e-3, 6k → 30k,
~Nov 2 11:00 + evals. Wave 18: r3_K0.5_ub0.5_S64_lr3e-3_cos fix **882038** (desa, from step 0, warmup 2500,
~Nov 2 17:00 + evals). It has no stage stops: score its keep-6000 / keep-10000 with `--score-keep`.
Seed replicate: r3_K0.5_ub0.5_lr1e-3 fix seed 2 **882039** (A100, → 6k, ~18:00).
Verified 10:59: 882037 "Restored all states", logged trainer/lr 9.9962e-4 at 6299; 882038 at 399 (warmup)
4.7964e-4. Both equal the schedule.
11:35: r3_K0.5_ub0.5_lr1e-3 **ada** 877672 (A100) scored 2.053 at 6k (+2.6% vs fix at LR 1e-3): stopped, fix stays.
Its A100 slot: seed 3 of r3_K0.5_ub0.5_lr1e-3 fix → 6k (seed noise, with seed 2 882039; ~18:40).
12:15: **Checkpoint fix in the three copied-checkpoint cells** (lr2e-3_cos, lr2e-3_cos_S64, S64_lr1e-3_cos). The copied
checkpoint's ModelCheckpoint state names the source dir's last.ckpt, so Lightning writes the rolling "last" to
`last-v1.ckpt`. `last.ckpt` had stayed the copied 6k file, so the final evals (CKPT_PATH=.../last.ckpt) and any
requeue would have used step 6000. Each `last.ckpt` is now a relative symlink to `last-v1.ckpt` (atomic mv).
The from-scratch 3e-3 cell is unaffected. Before a final eval, check that `last.ckpt -> last-v1.ckpt` resolves to
the 30000 save. Score 879486's keep-10000: job 883472.
12:21: r3_K0.5_ub0.5_lr2e-3_cos 879486 scored **1.893 at 10k** (best 10k; −0.74% vs constant LR 1e-3). The 1/3 gate
now marks constant LR 1e-3 (878861, A100, 10k → 30k) as failing (+0.75% > 0.5%). It is kept on purpose: it is the
only constant high-LR run to 30k, the reference for the decay's effect and a check on the projections, and the
slow A100 slot has no better use. Cancel it if the user prefers.
13:06: the symlink fix holds. The next rolling saves (12:31–13:04) went to last-v1.ckpt in all three cells, and
last.ckpt -> last-v1.ckpt is intact. Scoring 880403's keep-10000 (S = 64 vs 16 under the same cosine).
13:13: r3_K0.5_ub0.5_lr2e-3_cos_S64 880403 scored **1.885 at 10k** (best 10k; −0.42% vs its S = 16 twin).
14:32: r3_K0.5_ub0.5_S64_lr1e-3_cos 882037 scored **1.893 at 10k** (job 885059): +0.45% vs the best, inside the 1/3
margin (0.5%), so it continues to 30k. Peak 2e-3 leads 1e-3 at S = 64. 882038 (peak 3e-3, S = 64, from step 0)
reaches 6k ~16:20; its stage score is submitted automatically (`~/tmp/hbfm_elbo_score_when.sh <cell> <step>`
waits for keep-<step>.ckpt, then runs `sweep.py --site a6000 --arms fix --cells <cell> --score-keep --steps <step>`).
15:03: r3_K0.5_ub0.7 ada 867339 COMPLETED: full NELBO **1.815** (std 0.71; ub 0.5 ada 1.828), GenPPL 10.87, entropy
4.02, 64/64 unique, λ 0.447 at 30k (curvature.json; ub 0.5 ada 0.545). Pulled to desa. Its A100 slot goes to wave 19,
r3_K0.5_ub0.6_lr1e-3 fix to 6k (**885386**, started 15:10, 6k ~22:30): fix ub 0.5 / 0.6 / 0.7 at LR 1e-3.
16:22: r3_K0.5_ub0.5_S64_lr3e-3_cos 882038 scored **1.995 at 6k** (job 886273): +0.24% vs the best 6k (S64 LR 1e-3,
1.990), inside the 1/5 margin, so it continues to 10k. Its keep-10000 (~20:20) is scored automatically.
17:21: seed 2 of r3_K0.5_ub0.5_lr1e-3 fix (882039) scored **2.004 at 6k** (seed 1: 2.002; +0.10%). Its A100 slot goes to
wave 20, r3_K0.5_ub0.6_S64_lr2e-3_cos (best recipe at ub 0.6, from step 0): **888086** to 6k (~01:30), then **888087**
(afterok) to 30k (~Nov 3 09:00). Cancel 888087 if its 6k fails the 1/5 gate, or if ub 0.6 at LR 1e-3 (885386, ~22:30)
trails ub 0.5 by more than the seed spread.
16:57: the watcher had saved an ssh login banner as its baseline (a false event); its check now needs a known A100
job id (`^ch2263 795773 `), and the baseline was rebuilt.
18:27: seed 3 (882922) scored **2.010 at 6k**: seeds 1/2/3 → 2.002 / 2.004 / 2.010, std 0.22%. Cancel rule for 888087:
ub 0.6 at LR 1e-3 (885386) above 2.0155 (seed mean + 2 × 0.26%, the std of a single run minus a 3-seed mean).
18:35: nelbo_variance.py now takes a trunc_exp-trained cell (its own ub first, then shorter ubs, then S words at its
ub). **888648** (desa, 1 GPU): r3_K0.5_ub0.5 fix keep-30000, `--ubs 0.25,0.35` → nelbo_variance_keep-30000.json.
18:37: 888648 COMPLETED (RESULTS.md, "the same on a trunc_exp-trained model"): shorter ub → higher mean, lower stds;
S 4/16/64 → estimator std 0.232 / 0.227 / 0.225, mean unchanged.
20:07: r3_K0.5_ub0.5_S64_lr3e-3_cos 882038 scored **1.9005 at 10k** (job 889484): +0.84% vs the best, fails the 1/3 gate;
**cancelled 20:09** (checkpoints kept). Its slot: wave 21, r3_K0.5_ub0.5_S64_lr2e-3_cos (S 64 from step 0, cosine from 2e-3)
to 30k, **889513** (desa). Its keep-6000 is scored automatically (`hbfm_elbo_score_when.sh`).
21:44: r3_K0.5_ub0.6_lr1e-3 fix 885386 scored **2.009 at 6k** (< 2.0155): 888087 (ub 0.6 best recipe → 30k) stays.
ub 0.6 continues to 10k on the A100 (**891074**) to compare with ub 0.5's 10k (1.907, same seed).
Nov 2 01:13: r3_K0.5_ub0.6_S64_lr2e-3_cos 888086 (cosine from step 0) scored **1.9855 at 6k**: the new best 6k
(−0.23% vs S64 LR 1e-3 constant, 1.990), passes the 1/5 gate. Its 30k stage **888087** started 01:13 (A100);
its keep-10000 is scored by **895024** (A100, afterok on the CPU waiter 895023, which polls /scratch; ~05:50).
02:00: wave 21 r3_K0.5_ub0.5_S64_lr2e-3_cos 889513 scored **1.9772 at 6k** (job 895308): the new best 6k (−0.65% vs
S64 LR 1e-3 constant; ub 0.6 with the same recipe, 888086, is +0.42%, inside the 1/5 margin, so 888087 stays).
Its keep-10000 (~05:50) is scored automatically (`hbfm_elbo_score_when.sh`); the 1/3 gate then compares it with
the copied peak-2e-3 S64 run (880403, 1.8846) and ub 0.6 (895024).
02:21: r3_K0.5_ub0.6_lr1e-3 891074 scored **1.9077 at 10k** vs ub 0.5's 1.9067 (+0.05%; +0.35% at 6k): the longer
horizon closes its gap with training (as ub 0.7 ada did). Its A100 slot: wave 22, r3_K0.5_ub0.7_S64_lr2e-3_cos (best
recipe at ub 0.7, from step 0): **895547** to 6k (~10:30), then **895549** (afterok) to 30k. Horizon cells are
judged against ub 0.5 with the same recipe (889513), not by the plain margin, since their gap shrinks with steps:
cancel 895549 if its 6k is > +1.5% vs 1.9772 (> 2.0069), or at 10k if its gap has not shrunk below +0.6%.
04:14: r3_K0.5_ub0.5_lr1e-3 fix 878861 (constant LR 1e-3, S 16) COMPLETED: **NELBO 1.732** (full validation; GenPPL
9.76, entropy 3.98, 64/64 unique), the best final (−1.5% vs LR 3e-4), 0.020 below its power-law projection (1.752).
Its A100 slot: r3_K0.5_ub0.6_lr1e-3 from its 10k checkpoint to 30k (**896271**, pending for a GPU; same dir), the
constant-LR ub 0.6 partner of 878861 at 30k.
Wave 23, r3_K0.5_ub0.5_S256_lr2e-3_cos (best recipe at S = 256, from step 0, to 30k): **896278** (desa, afterok
879486). Score its keep-6000 / keep-10000 with `hbfm_elbo_score_when.sh` against wave 21 (1.9772 at 6k).
05:41: r3_K0.5_ub0.5_lr2e-3_cos 879486 (cosine from 2e-3, S 16) COMPLETED: **NELBO 1.7212** (GenPPL 10.09, entropy
3.92, 64/64), the best final (−0.63% vs constant LR 1e-3); evals ran on the 30k save (last.ckpt -> last-v1.ckpt,
04:58). 896278 (wave 23, S 256) started 05:40 on its GPUs.
05:48: wave 21 889513 scored **1.8902 at 10k** (job 897130): +0.30% vs the best 10k (880403, 1.8846), passes the
1/3 gate; it runs on to 30k. S 256 (896278) trains at 1.81 it/s vs 2.33 for S 64 (−22%): 6k ~13:00, 30k ~Nov 3 18:30.
06:30: ub 0.6 best recipe 888087 scored **1.8833 at 10k** (job 895024): the best 10k, −0.37% vs ub 0.5 (wave 21,
1.8902) after +0.42% at 6k; it runs on to 30k (~Nov 3 08:30). Wave 24, r3_K0.5_ub0.8_S64_lr2e-3_cos (best recipe at
ub 0.8, from step 0, to 30k): **898230** (desa, afterok 880403). Score its keep-6000 / keep-10000 against ub 0.5
(the horizon rule: > +1.5% at 6k, or a gap not below +0.6% at 10k, cancels).
07:53: 880403 (S 64 cosine from 2e-3) reached 30k; last.ckpt -> last-v1.ckpt is the 07:53 30k save. Its
full-val NELBO and samples evals run now (~1 h); 898230 (wave 24) starts on its GPUs after. 08:00: its samples:
GenPPL 10.03, entropy 3.93. Wave 22's 10k score queued on ch2263: CPU waiter **899373** (polls keep-10000.ckpt on
/scratch) → **899383** (afterok, score-keep 10000). Cancel both with 895549 if wave 22 fails its 6k horizon check.
08:36: 880403 COMPLETED (evals): **NELBO 1.7121** (std 1.26; GenPPL 10.03, entropy 3.93, 64/64), the best final, 0.7%
above 1.7 and −0.53% vs its S 16 twin 879486 (−0.42% at 10k). 898230 (wave 24) started on its GPUs at 08:36.
Wave 25, r3_K0.5_ub0.6_S64_lr2e-3_cos_S256 (888087's keep-10000.ckpt, exported by CPU job 900446 and rsynced in as
checkpoints/last.ckpt; the same cosine, whose scheduler state resumes, at S = 256 from 10k to 30k): **900614**
(desa, afterok 882037, ~10:15). A background shell symlinks its last.ckpt -> last-v1.ckpt after its first save.
Gate at 14k vs 888087 at the same step: waiter **900656** → score **900658** (ch2263) for 888087's keep-14000, and
`hbfm_elbo_score_when.sh` for the twin's. Cancel 900614 if it is not below 888087 at 14k.
09:52: 882037 (S 64, cosine from 1e-3 at 6k) COMPLETED: **NELBO 1.7202** (std 1.26; GenPPL 10.26, entropy 3.94, 64/64),
+0.47% vs peak 2e-3 (880403). Wave 25 900614 started at 09:51 on kuleshov-compute-03 ("Restored all states").
Its cosine resumed in place (lr 1.6465e-3 at step 10099, expected 1.6464e-3); it saved at 10000 on resume
(3-10000.ckpt, keep-10000.ckpt, last-v1.ckpt), and last.ckpt -> last-v1.ckpt since 09:57. ~750 steps/h: 14k
~15:20, 30k ~Nov 3 13:00.
10:11: wave 22 (ub 0.7) scored **1.9966 at 6k** (895547): +0.98% vs ub 0.5 (1.9772), inside the +1.5% horizon limit, so
895549 runs on (started 10:10); its 10k score is queued (899373 → 899383, ~15:30).
11:38: 888087 (ub 0.6, S 64) scored **1.8215 at 14k** (900658): the bar for wave 25 (S 256 from its 10k) at 14k (~15:20).
13:25: wave 23 (S 256 from step 0) scored **1.9780 at 6k** (903869): +0.04% vs S 64 (wave 21, 1.9772), a tie at 22% lower speed.
Cancelled 896278 at 13:30 (checkpoints kept to 6k); wave 25 tests S 256 late in training. In its slot, wave 26,
r3_K0.5_ub0.6_S64_lr2e-3_lin (wave 20's recipe with a linear decay to 0 at 30k; new LR_SCHED=linear in hbfm.sh,
configs/lr_scheduler/linear_decay_warmup.yaml = transformers.get_linear_schedule_with_warmup, warmup 2500):
**903969** (desa, from 13:31, ~30k Nov 3 18:00). Its LR is below cosine's until 16250 and above it after, so its
gate is 16k vs 888087's keep-16000 (A100 waiter **903972** → score **903973**): cancel if not below. Its 6k and 10k
go through the standard gates; `hbfm_elbo_score_when.sh` scores its 6k. Its lr at step 99 is 7.92e-5 (= 2e-3 · 99/2500).
14:34: wave 24 (ub 0.8) scored **2.0150 at 6k** (904473): +1.91% vs ub 0.5, past the +1.5% horizon limit, so 898230 was
cancelled (checkpoints kept to 6k). 888087 scored **1.8000 at 16k** (903973): wave 26's bar. In the freed slot, wave
21's keep-14000 / keep-16000 are scored (904768 / 904769) for the ub 0.6 vs 0.5 gap at 14k / 16k.
14:40: wave 21 (ub 0.5) scored **1.8303 at 14k, 1.8069 at 16k**: ub 0.6 (888087) vs ub 0.5 = +0.42 / −0.37 / −0.48 / −0.38%
at 6k / 10k / 14k / 16k, so the horizon gain has levelled at ~−0.4%. Wave 27, r4_K0.5_ub0.6_S64_lr2e-3_cos (wave 20's
recipe at proposal rate 4): **904783** (desa, from 14:41, ~30k Nov 3 18:00). Cancel at 6k if > +0.5% vs 888087 (1.985
→ > 1.995), at 10k if not below 888087 (1.8833). `hbfm_elbo_score_when.sh` scores its 6k.
15:06: wave 25 (S 256 from 888087's 10k) scored **1.8195 at 14k** (905111): −0.11% vs 888087 (1.8215), below the bar, so
900614 runs on (S 64 vs 16 over the same 4k steps: −0.42%).
15:31: wave 22 (ub 0.7) scored **1.8964 at 10k** (899383): +0.33% vs ub 0.5 (1.8902; +0.98% at 6k), under the +0.6% horizon
limit, so 895549 runs on to 30k (A100, ~Nov 3 16:00). vs ub 0.6: +0.56% at 6k, +0.70% at 10k, so ub 0.6 stays the best horizon.
18:20: (user question) NELBO std vs ub on the 6k checkpoints of the LR 3e-4 ub sweep: `nelbo_variance.py --n 512
--draws 8 --ubs '' --words ''` on all 9 keep-6000.ckpt (907576-81 desa; 907620-2 on the three A100 checkpoints, exported by
CPU job 907583 on ch2263 into `sfm_home/outputs/hbfm_elbo_tinystories_256/_var6k/<cell>`, cache_dir fixed). Estimator std
0.19-0.27, rising with ub on both arms; between-sequence std ~5x larger, falling with ub on ada (0.96 -> 0.59) and rising
on fix (1.04 -> 1.21). Table in RESULTS.md ("Each model at its own horizon").
18:35: (user request) `visualization/curvature_per_position.py` (new): kappa_i = lambda_i K per position, one figure
per checkpoint -> `figures/curvature_per_position_<cell>_keep-6000.png` for the 5 ada 6k checkpoints (jobs 908089, 908111).
kappa is constant over positions and sequences (no prompt, wrap): -0.550 / -0.424 / -0.328 / -0.276 / -0.251 for ub 0.25 ... 1.
19:13: wave 26 (linear decay) scored **1.9738 at 6k** (908538): −0.56% vs cosine (888087, 1.985), as expected from
its lower LR before 16250 (1.75e-3 vs 1.92e-3 at 6k). Passes; its 10k and 16k are queued (`hbfm_elbo_score_when.sh`, ~23:10 and
~05:10). The decision is at 16k vs 888087's 1.8000.
20:28: wave 27 (proposal rate 4) scored **1.9915 at 6k** (910567): +0.33% vs rate 3 (888087, 1.985), under the 1.995 cut,
so 904783 continues; its 10k is queued (`hbfm_elbo_score_when.sh`, ~00:30) and must be below 1.8833 (888087's 10k). Its
val/loss_std is 1.04 vs 1.27 at rate 3: the rate moves the estimator variance, not (so far) the mean.
23:00: wave 26 (linear decay) scored **1.8779 at 10k** (914364): −0.29% vs cosine (888087, 1.8833), passes the 10k
rule; 903969 runs on. Its 16k (~05:10, queued) must be below 1.8000 (888087's 16k), else cancel.
00:14: wave 27 (proposal rate 4) scored **1.8883 at 10k** (915806): +0.26% vs rate 3 (888087, 1.8833; +0.33% at 6k), not
below it, so 904783 was cancelled (checkpoints kept to 10k). Rate 3 stays (rate 3 vs 4 is inside the 0.32% seed-difference noise
twice, so neither side gains). In its slot, wave 28, r3_K0.5_ub0.6_S64_lr2e-3_cos_do0 (wave 20's recipe with model.dropout 0
instead of 0.1; 30k x 512 x 256 is ~8 epochs of TinyStories; new DROPOUT knob in hbfm.sh, default 0.1, copied-edited-moved):
**915962** (desa, kuleshov-compute-02, from 00:17; config.yaml dropout 0 checked). Cancel at 6k if > 1.995 (+0.5% vs 888087's
1.985), at 10k if not below 1.8833. `hbfm_elbo_score_when.sh` scores its 6k and 10k (~08:30, ~13:50).
01:34: wave 21 (889513, S 64 cosine from 2e-3 at step 0, ub 0.5) COMPLETED: **NELBO 1.7201** (std 1.26; GenPPL 9.92,
entropy 3.99, 64/64; last.ckpt = the 00:51 30k save, own dir). +0.47% vs 880403 (constant 2e-3 to 6k, then cosine over 24k;
1.7121), level with 882037 (1.7202). Wave 29 puts ub 0.6 on 880403's schedule, in wave 21's slot: stage
r3_K0.5_ub0.6_S64_lr2e-3 (constant 2e-3, S 64, to 6k + its 6k score) **917213** (kuleshov-compute-02, from 01:44, ~07:30);
CPU copy job **917214** (afterok) copies its keep-6000.ckpt -> r3_K0.5_ub0.6_S64_lr2e-3_const6k_cos/checkpoints/last.ckpt
and its nelbo_6000.json; **917215** (afterok 917214) resumes it with LR_SCHED_FROM 6000 (cosine 2e-3 -> 1e-6 over 24k,
~Nov 4 06:00 + evals); CPU job **917218** (after 917215 starts) points last.ckpt -> last-v1.ckpt once last-v1 is written.
Gates: the stage's 6k vs ub 0.5 constant 2e-3 (S 16, 1.994) and S 64 constant 1e-3 (1.990): cancel 917215 if > 2.004;
the cosine's 10k must not be above 880403's 1.8846.
03:12: 896271 (A100, ub 0.6, constant LR 1e-3, S 16) COMPLETED: **NELBO 1.7241** (std 1.31; GenPPL 9.19, entropy 3.84,
64/64), −0.47% vs ub 0.5 at the same LR (878861, 1.7323): the horizon gain holds to 30k. Pulled (`sweep.py --site a100 --pull`).
Its A100 slot is free; ch2263's other two (888087 ~08:30, 895549 ~16:00) run on.
03:20: wave 30, r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0 (A100, in 896271's slot): wave 29's schedule in one job plus wave 28's
dropout 0. New LR_SCHED=wsd in hbfm.sh (LR_HOLD_STEPS, default 3500) + configs/lr_scheduler/wsd_warmup.yaml =
transformers.get_wsd_schedule: warmup 2500, 2e-3 to 6000, cosine to 0 at 30000 (880403 decayed to 1e-6: the same to
5e-4 of the peak). hbfm.sh copied-edited-moved; `--sync-code` to ch2263 (only the two lr_scheduler yamls and hbfm.sh
changed; rsync renames, so 888087 / 895549 keep their inode). **917963** (pending Resources at 03:20). Cancel it if wave
28's dropout 0 fails its 6k (> 1.995, ~08:20); otherwise its 6k is compared with wave 29's stage 6k (same schedule, dropout
0.1). Check trainer/lr in its log: 7.92e-5 at step 99, 2e-3 from 2500 to 6000, 1e-3 at 18000.
03:47: 917963 started (nlplarge-compute-01); config.yaml: dropout 0, get_wsd_schedule 2500 / 3500 / 24000. ~850 steps/h
(30k ~Nov 4 15:00).
04:36: wave 26 (linear decay) scored **1.7980 at 16k** (918473; its sbatch reported a socket timeout but the job was
created, one copy): −0.11% vs cosine (888087, 1.8000), below the bar, so 903969 runs to 30k (~Nov 3 18:00 + evals).
lin vs cos: −0.56 / −0.29 / −0.11% at 6k / 10k / 16k, shrinking as linear's LR passes cosine's (16250).
06:00: wave 28 (dropout 0) scored **1.9597 at 6k** (919004): −1.30% vs dropout 0.1 (888087, 1.9855), 4x the
seed noise, the largest single-knob gain since LR. Passes (≤ 1.995): 915962 and wave 30 (917963) run on. Its 10k (~09:50)
must stay < 1.8833; dropout 0 over 8 epochs may overfit later, so the 10k / 16k margins decide it. Together with
wave 29 (held schedule, dropout 0.1) and wave 30 (held, dropout 0) this is a 2x2 schedule x dropout grid.
06:10: wave 30 (917963) logged trainer/lr 7.92e-5 at step 99 and 4.0e-4 at 499: the wsd warmup as planned (~1900 steps).
07:19: wave 29 stage (917213, constant 2e-3, S64, ub 0.6) scored **1.9836 at 6k**: −0.54% vs 880403 (ub 0.5, S16,
1.9944), −0.10% vs cosine ub 0.6 (888087, 1.9855; cosine is still at 1.93e-3 at 6k). Passes (≤ 2.004): the chain
runs, copy 917214 started 07:19 → 917215 (cosine from 6000) → 917218 (last.ckpt → last-v1).
07:24: 917215 "Restored all states" from the copied last.ckpt (config: CosineDecayWarmup t_initial 24000, warmup 0,
lr_min 1e-6, dropout 0.1). Lightning saved last-v1.ckpt (and an extra 1-6000.ckpt) at the resume; 917218 linked last.ckpt ->
last-v1.ckpt at 07:22, so the final evals will read the 30k save.
08:02: 917215 trainer/lr 1.99992e-3 at 6099, 1.99694e-3 at 6599 = the cosine from 2e-3 over 24k, as 880403.
08:20: wave 30 (917963, nlplarge, QOS normal) was preempted 5 times (restarts 06:53, 07:10, 07:18, 07:27); each
requeue "Restored all states" from its rolling last.ckpt on the same node, so it lost ~1.5 h, not the run: ~step 2650 at
08:19, 30k ~Nov 4 17:00 if not preempted again. 888087 / 895549 (also nlplarge) were not hit. If it keeps being preempted,
move it to desa when 900614 frees a slot (~13:00): export its last.ckpt (CPU job on nlplarge-claire-highpri), resume there.
888087 (ub 0.6 cosine) finished training: ppl.json, samples_genppl.json written 08:17-08:18, nelbo.json next.
08:32: **888087 final 1.7081** (full val, std 1.30; GenPPL 10.91, entropy 3.94, 64/64): the new best, 0.48% above 1.7;
−0.70% vs ub 0.5 on the same recipe (wave 21, 1.7201), −0.23% vs 880403 (1.7121). Pulled (`sweep.py --site a100 --pull`).
Its A100 slot is free: wave 31 goes there, picked on wave 28's 10k (~09:40). Both cells are in sweep.py (dry-run checked):
r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0 (dropout 0, 2e-3 held to 14k, cosine over the last 16k) if dropout 0 holds its lead,
r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0.05 (wave 30's schedule, dropout 0.05) if it fades. Submit with
`sweep.py --site a100 --arms fix --cells <cell>` (full length; score its 6k with --score-keep at the A100 site).
09:44: wave 28 (dropout 0) scored **1.8568 at 10k** (921531): −1.41% vs dropout 0.1 (888087, 1.8833; −1.30% at 6k), so
the gain grows, no sign of overfitting. Passes: 915962 runs to 30k (~Nov 4 05:00 + evals; if the gap holds, ~1.684).
Wave 31 = **r3_K0.5_ub0.6_S64_lr2e-3_wsd14k_do0** on the A100 (`--sync-code`, then submitted): **921565**. Its 6k should
match wave 30's (same schedule to 6k); then compare at 10k / 16k (wave 30 decays from 6k, wave 31 holds to 14k).
09:50: 921565 config_tree: get_wsd_schedule 2500 / 11500 / 16000, dropout 0, ub 0.6, S 64 (nlplarge-compute-01).
11:12: wave 29 (917215) scored **1.8876 at 10k** (924267): past its 1.8846 bar (880403's 10k, +0.16%) and +0.23% vs the
cosine from step 0 at ub 0.6 (888087, 1.8833), so the held peak's 10k lead at ub 0.5 (−0.26%) does not carry over. It is
dropout 0.1, now 1.4% behind dropout 0, and waves 30 / 31 test the held peak at dropout 0: **cancelled 917215** (~10.9k;
keep-6000/8000/10000 kept), its 16k waiter stopped.
Not moved wave 31 to desa: the W&B loop rsyncs ch2263's wandb/ into the desa dir of the same tag and `--pull` copies its
.hydra / logs, so one tag at two sites would mix. Wave 32 in the freed slot instead: **r3_K0.5_ub0.7_S64_lr2e-3_cos_do0**
(wave 28 at ub 0.7; desa, full length) = **924656** (11:20, kuleshov-compute-02). Waiter scores its 6k / 10k. Gates: 6k ≤
1.989 (wave 28's 1.9597 + 1.5% horizon allowance); 895549's 30k final (ub 0.7, dropout 0.1, ~16:00): cancel 924656 if ub
0.7 ends clearly above 888087's 1.7081 (> +0.3%); 10k vs wave 28's 1.8568 (+0.7% = ub 0.7's 10k gap at dropout 0.1).
12:00: **wave 25 final 1.7070** (900614, S = 256 from 888087's 10k; full val, std 1.30; GenPPL 11.50, entropy 3.90,
64/64): −0.06% vs S = 64 (1.7081), noise; last.ckpt -> last-v1.ckpt = the 11:19 30k save (= keep-30000). S = 64 stays.
Wave 33 in its slot: **r3_K0.5_ub0.6_S64_lr2e-3_lin_do0** (wave 26's linear decay + dropout 0; desa, full length) =
**925761** (12:05, kuleshov-compute-03); waiter scores 6k / 10k. Gates: 6k / 10k vs wave 28 (1.9597 / 1.8568, +0.32%
noise); cancel if wave 26's 30k final (~18:00) ties or loses to 888087 (1.7081).
13:55: wave 30 (917963) past 6.4k (no restart since 07:27): its keep-6000 is scored by **928844** (ch2263; nlplarge has
one node, so it reads the same /scratch). Gate: vs wave 28's 1.9597 (same dropout 0, cosine from 0; wsd holds the peak to
6k). The A100 runs share nlplarge-compute-01 and are slow: wave 31 (921565) 2.8k steps in 3.7 h, 895549 at 27.1k.
13:58: **wave 30 6k = 1.9566** (std 1.26) vs wave 28's 1.9597: −0.16%, inside the 0.32% noise: passes. Holding the peak
to 6k costs nothing at dropout 0 (at dropout 0.1 / ub 0.6 the held peak was −0.05% at 6k, +0.23% at 10k). 10k score
queued: CPU waiter **929133** (polls keep-10000 on /scratch) → **929134** (afterok, score-keep 10000). Gate at 10k: within
+0.32% of wave 28's 1.8568 (≤ 1.8627), else cancel 917963. Wave 31 shares this schedule to 6k, so it skips the 6k score.
16:13: **wave 28 16k = 1.7759** (930868; std 1.29) vs 888087's 1.8000 (−1.34%) and wave 26's linear 1.7980 (−1.23%).
The dropout-0 gap holds (−1.30 / −1.41 / −1.34% at 6k / 10k / 16k): no overfitting by 16k. At −1.34% its 30k projects to
~1.685 (888087 1.7081), below the 1.7 target. Done at ~05:00 Nov 4 plus evals.
17:13: **wave 32 6k = 1.9747** (931709; std 1.28): passes its ≤ 1.989 bar; +0.77% vs wave 28 (ub 0.6, 1.9597), about the
gap ub 0.7 had at dropout 0.1 (+0.60% at 6k, +0.69% at 10k). 895549 (ub 0.7, dropout 0.1) finished training; ppl.json
and samples written, full-val NELBO running: wave 32's horizon gate waits for it.
17:28: **895549 final 1.7146** (ub 0.7, dropout 0.1; full val, std 1.32; GenPPL 10.60, entropy 4.05, 64/64): +0.38% vs
888087 (ub 0.6, 1.7081), past the +0.3% bar. ub 0.5 / 0.6 / 0.7 → 1.7201 / 1.7081 / 1.7146 at 30k (parabola min ≈ 0.61).
**Cancelled wave 32 (924656)** at ~6.5k (keep-2000/4000/6000 kept) and its 10k waiter. The listed knobs (K, bound, ub,
rate, mode, S, embedding) are tuned and the schedule × dropout-0 variants are running (waves 30 / 31 / 33), so the slot
runs a **seed-2 replicate of wave 28** (r3_K0.5_ub0.6_S64_lr2e-3_cos_do0, seed 2) = **932948** (17:35,
kuleshov-compute-02): the 1.7 claim then rests on two seeds. Waiter scores its 6k / 10k (expect within 0.32% of seed 1).
17:55: **wave 33 6k = 1.9539** (934451; std 1.26) vs wave 28's 1.9597: −0.30%, passes (linear vs cosine at dropout 0.1:
−0.56% at 6k). Next: its 10k vs 1.8568, and wave 26's final (~19:00; cancel 925761 if linear ties or loses to 1.7081).
18:14: **wave 30 10k = 1.8548** (929134; std 1.28) vs wave 28's 1.8568: −0.11%, passes (≤ 1.8627). The peak held to 6k
ties the cosine from 0 at dropout 0 (−0.16 / −0.11% at 6k / 10k). 16k queued: waiter **935825** → score **935828**
(afterok); gate vs wave 28's 1.7759.
18:37: **wave 26 final 1.7053** (903969, linear decay to 0, dropout 0.1; full val, std 1.30; GenPPL 9.92, entropy 3.94,
64/64; last.ckpt = the 30k save): **new best finished cell**, −0.17% vs the cosine (888087, 1.7081). Inside the 0.32%
noise, but linear led at all four stages (−0.56 / −0.29 / −0.11 / −0.17%), so I read it as a small win, not a tie, and
**kept wave 33** (925761, linear + dropout 0; −0.30% at 6k); nothing better is waiting for its slot. Its gates: 10k vs
wave 28's 1.8568 (+0.32%), 16k vs 1.7759.
18:53: EMA check (936689, 1 GPU, paired 10% val): wave 26 keep-30000 EMA 1.7154 vs raw 1.7114 (−0.23%); wave 28 keep-16000
EMA 1.7759 vs raw 1.8227 (+2.6%). Protocol keeps EMA; RESULTS.md has the table. A100 sped up after 895549 ended: wave 31
4.6k → 6.7k in 1.7 h (~1.2k steps/h; 30k ~15:00 Nov 4), wave 30 at 10.3k. So wave 31 stays on the A100.
Wave 34 in wave 26's slot: **r3_K0.5_ub0.6_S64_lr3e-3_lin_do0** (peak LR 3e-3, linear, dropout 0) = **937627** (18:56,
kuleshov-compute-03). At dropout 0.1 / cosine 3e-3 was +0.9 / +0.53% at 6k / 10k and stopped while the gap shrank.
Gates vs wave 33 (same recipe at 2e-3; 1.9539 at 6k): 6k ≤ +1.5% (higher-LR allowance, ≤ 1.983), 10k ≤ +0.5% of wave
33's 10k, 16k must be below wave 33's 16k. Waiter scores 6k / 10k / 16k.
21:42: **wave 33 10k = 1.8585** (942625; std 1.28) vs wave 28's 1.8568: +0.09%, passes (≤ 1.8627). Linear vs cosine
at dropout 0: −0.30 / +0.09% at 6k / 10k (at dropout 0.1: −0.56 / −0.29%), level so far. 16k score queued (vs 1.7759).
23:14: **wave 28 seed 2, 6k = 1.9522** (944889; std 1.26) vs seed 1's 1.9597: −0.38%, about the 0.32% two-run noise.
So wave 28's dropout-0 lead replicates at 6k. Its 10k is scored when written.
00:41: **wave 34 (LR 3e-3) 6k = 1.9518** (946431; std 1.27) vs wave 33 (2e-3) 1.9539: −0.11%, passes (≤ 1.983).
At dropout 0.1 / cosine, 3e-3 trailed 2e-3 by +0.9% at 6k; at dropout 0 it is level. 10k gate ≤ 1.8678 (+0.5% of 1.8585).
01:58: A100 is slower than the 18:53 estimate: wave 30's keep-16000 was only written at 01:54 (waiter 935825),
so its 16k score 935828 started then (gate vs wave 28's 1.7759); poller in the scratchpad (wait_935828.sh).
01:57: **wave 30 16k = 1.7775** (935828, 1:43 on the A100; std 1.29) vs wave 28's 1.7759: +0.09%, passes (≤ 1.7816,
+0.32% noise). Held-to-6k vs cosine-from-0 at dropout 0: −0.16 / −0.11 / +0.09% at 6k / 10k / 16k, a tie. 917963 runs
to 30k (~14k more steps at ~0.8-1k/h on the A100: ~18:00 Nov 4 plus evals).
03:01: seed 2's 10k score **950367** queued (Resources: first in line). Another experiment's ~60 1-GPU
`sudoku_reproduce_*` jobs (submitted 01:14, same account; leave them alone) took the two spare desa GPUs, so score
jobs now wait for one of those to end. Our new jobs still outrank them (priority 18254 vs 18224), but when 915962
ends its 4 GPUs go to that backlog unless the next cell is already pending: queue it afterany:915962.
03:10: **wave 28 seed 2, 10k = 1.8487** (950367; std 1.28) vs seed 1's 1.8568: −0.43% (−0.38% at 6k). The seed offset
persists and is a bit above the 0.32% two-run estimate, so gaps under ~0.45% between single runs are not evidence;
waves 30 / 33 / 34 are all within that of wave 28. Seed 2 (932948) runs on to 30k as the replicate of the best recipe.
03:19: **wave 33 16k = 1.7834** (950732; std 1.29) vs wave 28's 1.7759: +0.42%, fails (≤ 1.7816). Same seed as wave 28,
so the seed offset cancels: linear vs cosine at dropout 0 went −0.30 / +0.09 / +0.42% at 6k / 10k / 16k (at dropout 0.1
linear led at every stage and won at 30k by 0.16%). **925761 cancelled** (03:25, keep-16000 kept).
Wave 35 in its slot: **r3_K0.5_ub0.6_S64_lr3e-3_cos_do0** (wave 28 at peak LR 3e-3) = **950777** (03:25,
kuleshov-compute-03; submitted before the cancel so the sudoku backlog could not take the GPUs). Gates vs wave 28
(same seed): 6k ≤ +1.5% (≤ 1.989), 10k ≤ +0.5% (≤ 1.8661), 16k ≤ +0.32% (≤ 1.7816).
Wave 34's gates now refer to wave 28 too, since wave 33 lost: 10k ≤ 1.8661, 16k ≤ 1.7816.
04:15: wave 28 (915962) is slower than estimated: 29k at 03:44 (~1k steps/57 min), so 30k ~04:40, then its sample and
full-val NELBO evals (~06:00-06:30). Wave 36 queued in its slot: **r3_K0.5_ub0.6_S64_lr4e-3_cos_do0** = **952658**
(afterok:915962), so the sudoku backlog cannot take the 4 GPUs. With waves 28 / 35 it makes a 2e-3 / 3e-3 / 4e-3 curve.
If wave 34 (3e-3, linear) fails its 10k, replace 952658 before 915962 ends. Same gates as wave 35.
04:26: **wave 34 10k = 1.8548** (952943; std 1.28): −0.20% vs wave 33 (2e-3, linear, same seed), −0.11% vs wave 28;
passes (≤ 1.8661). Under dropout 0 the peak 3e-3 leads 2e-3 at 6k / 10k (−0.11 / −0.20%), where at dropout 0.1 it
trailed (+0.9 / +0.53%). So wave 36 (4e-3) stays queued. 16k gate ≤ 1.7816; wave 33 had 1.7834 there.
04:47: wave 28 (915962) trained to 30k at 04:43. Sample eval: **GenPPL 12.13**, entropy 3.908, 64/64 unique (wave 26,
dropout 0.1: 9.92 / 3.945 / 64/64). The full-val NELBO started 04:46. (`eval/ppl.json`, 0.694, is the sampling
script's own val metric, not the NELBO.)
05:26: **TARGET REACHED. Wave 28 final = 1.6954** (915962 COMPLETED; full val, std 1.29): 0.27% below 1.7, −0.74% vs
dropout 0.1 (888087, 1.7081), −0.58% vs the previous best (wave 26, 1.7053). Dropout 0's lead halved in the decay
(−1.34% at 16k). RESULTS.md answer 7 and finals updated. Wave 36 (952658, LR 4e-3) started in its slot at 05:25.
Still running: seed 2 (932948), waves 30 / 31 (A100), 34, 35, 36: tuning continues for a lower NELBO.
08:45: **wave 28 seed 2, 16k = 1.7726** (956802; std 1.29) vs seed 1's 1.7759: −0.19% (−0.38 / −0.43% at 6k / 10k).
The seed gap narrows as the LR decays; seed 2's final should land near seed 1's 1.6954.
10:06: **wave 34 (LR 3e-3, linear), 16k = 1.7772** (958298; std 1.29): +0.07% vs wave 28's 1.7759, passes the 16k gate
(≤ 1.7816); −0.35% vs 2e-3 linear (wave 29, 1.7834, stopped). Its lead over wave 28 shrinks: −0.40 / −0.11 / +0.07% at
6k / 10k / 16k. **Wave 35 (LR 3e-3, cosine), 6k = 1.9680** (957083; std 1.27): +0.42% vs wave 28's 1.9597, passes the
6k gate (≤ 1.989). Both score jobs sat behind 4 sudoku jobs on the full desa nodes; moved them to `--partition=desa,gpu
--constraint="ampere|ada"` (scontrol update), and sweep.py's `sbatch1` now asks for the same, so scores backfill the
shared gpu partition (it preempts by requeue; a score takes ~5 min).
11:13: **wave 36 (LR 4e-3, cosine), 6k = 1.9773** (959040, on the gpu partition, 4:43; std 1.27): +0.90% vs wave 28's
1.9597, passes the 6k gate (≤ 1.989). Under cosine the 6k NELBO rises with the peak LR: 2e-3 / 3e-3 / 4e-3 = 1.9597 /
1.9680 / 1.9773 (the LR is still near its peak at 6k; the finals decide).
12:49: **wave 35 (LR 3e-3, cosine), 10k = 1.8683** (960772; std 1.29): +0.62% vs wave 28's 1.8568, **fails the 10k gate**
(≤ 1.8661); its gap widened from +0.42% at 6k. Cancelled 950777 at ~10.3k. **Wave 36 (LR 4e-3) stopped early** (952658, ~7.9k):
at 6k it trailed wave 35 (1.9773 vs 1.9680), which has now failed at 10k, so it would fail too (stricter gate, as the
directive allows). Under cosine and dropout 0 the LR optimum is at or below 2e-3. Submitted first, then cancelled, so
the GPUs went to: **wave 37 = r3_K0.5_ub0.6_S64_lr1.5e-3_cos_do0 (960878, compute-03)**, the LR axis below 2e-3, and
**wave 38 = r3_K0.5_ub0.6_S64_lr2e-3_cos_do0.05 (960879, compute-02)**, wave 28 at dropout 0.05 (dropout 0's lead over
0.1 halved from 16k to 30k: −1.34% → −0.74%). Gates vs wave 28 (seed 1): 6k ≤ 1.989, 10k ≤ 1.8661, 16k ≤ 1.7816.
Score helpers re-armed for both. K, rate, ub, S and the embedding are settled (answers 4, 7, 8); these two are the
remaining open knobs at the final recipe.
18:38: **wave 37 (LR 1.5e-3, cosine), 6k = 1.9641** (968853; std 1.26): +0.23% vs wave 28's 1.9597; **wave 38
(dropout 0.05), 6k = 1.9673** (969052; std 1.26): +0.39%. Both pass the 6k gate (≤ 1.989) and continue to 10k
(~22:15). Under cosine and dropout 0 the 6k NELBO is lowest at LR 2e-3: 1.5e-3 / 2e-3 / 3e-3 / 4e-3 = 1.9641 / 1.9597
/ 1.9680 / 1.9773. By dropout at LR 2e-3: 0 / 0.05 / 0.1 = 1.9597 / 1.9673 / 1.985. Both gaps are within two-seed
noise (seed 2 of wave 28 scored 1.952 at 6k, -0.4%). W&B steps at the 17:42 sync: A100 wave 30 28.8k (final ~20:00),
wave 28 seed 2 26.3k (~22:00), wave 34 25.0k (~23:00), A100 wave 31 25.1k (~00:30).
20:07: **NEW BEST. Wave 30 final = 1.6901** (917963 COMPLETED, A100; full val 1172/1172, std 1.30; GenPPL 11.25,
entropy 3.87): −0.32% vs wave 28's 1.6954, 0.58% below 1.7. Within two-seed noise (6k / 10k / 16k: −0.16 / −0.11 /
+0.09% vs wave 28). Pulled (`sweep.py --site a100 --pull`). ch2263's A100 slot is free; wave 31 (921565) runs on.
20:09: **wave 39 = wave 30 at seed 2** (r3_K0.5_ub0.6_S64_lr2e-3_wsd6k_do0, `--seeds 2`, A100 full length): **970595**
(pending Priority). Wave 30's lead over wave 28 (−0.32%) equals the two-seed noise; with wave 28's seed 2 (932948) this
gives both schedules at two seeds. Code at ch2263 checked identical (md5 of both hbfm.sh, wsd yaml, model, utils).
Score its keep-6000/10000/16000 at the A100 site with `--score-keep`.
22:24: **wave 37 (LR 1.5e-3), 10k = 1.8617** (972024; std 1.27): +0.27% vs wave 28's 1.8568 (+0.23% at 6k), passes the
10k gate (≤ 1.8661); on to 16k (gate ≤ 1.7816). 22:26: **wave 38 (dropout 0.05), 10k = 1.8678** (972082; std 1.28):
+0.59% (+0.39% at 6k), **fails the 10k gate**. Dropout is settled at 0 (0 / 0.05 / 0.1 at 10k: 1.8568 / 1.8678 / 1.8833).
Submitted first, then cancelled 960879 at ~10.3k: **wave 40 = r3_K0.5_ub0.6_S64_lr2e-3_wsd10k_do0 (972096,
compute-02)**, the hold length of wave 30's held peak: 2e-3 held to 10k (LR_HOLD_STEPS 7500), cosine over the last
20k; between wave 30 (6k, 1.6901) and wave 31 (14k, A100, final ~00:30). Its 6k is wave 30's schedule (gate ≤ 1.989);
after 6k its LR stays high, so the cosine gates do not apply; it runs to 30k. Score helper re-armed (6k/10k/16k).
22:33: **wave 28 seed 2 final = 1.6927** (932948 COMPLETED; full val 1172/1172, std 1.29; GenPPL 11.02, entropy 3.91):
−0.16% vs seed 1 (1.6954); the seed gap shrank from −0.4% at 6k/10k to −0.19% at 16k and −0.16% at 30k. Cosine
two-seed mean 1.6940; wave 30 (held peak, 1.6901) is −0.23% vs that mean. 932948's GPUs went to other jobs: at 22:36
both desa nodes are fully allocated (compute-02 and -03: 10/10 GPUs). Next slot: wave 34 (937627), done ~00:00 with evals.
23:51: **NEW BEST. Wave 34 final = 1.6876** (937627 COMPLETED; peak LR 3e-3, linear decay to 0, dropout 0; full val
1172/1172, std 1.29; GenPPL 10.32, entropy 3.87): −0.46% vs wave 28 (same seed), −0.38% vs the cosine two-seed mean,
−0.15% vs wave 30 (held peak). Stages vs wave 28: −0.40 / −0.11 / +0.07% at 6k / 10k / 16k, so the gain is in the tail.
23:55: **wave 41 = r3_K0.5_ub0.6_S64_lr4e-3_lin_do0 (973334, compute-03, in wave 34's slot)**: the LR axis above 3e-3
under the linear decay. Its gates are vs wave 34 (same schedule shape), with the usual margins (1.5 / 0.5 / 0.32%):
6k ≤ 1.981, 10k ≤ 1.864, 16k ≤ 1.783. Score helper armed (6k/10k/16k).
00:51: **NEW BEST. Wave 31 final = 1.6849** (921565 COMPLETED, A100; 2e-3 held to 14k, cosine over the last 16k;
full val 1172/1172, std 1.30; GenPPL 11.12, entropy 3.85): −0.31% vs the hold to 6k (wave 30, same seed and site),
−0.16% vs wave 34. Pulled. More high-LR steps win at 30k (longer hold; 3e-3 linear > 2e-3 cosine), while they trail at
10k / 16k: the cosine-relative 10k / 16k gates are biased against them (wave 35, 3e-3 cosine, may have been a false
negative at 10k). So:
00:57: submitted first, then cancelled as dominated **wave 40** (972096, hold to 10k, between 6k and 14k; ~2.6k) and
**wave 37** (960878, 1.5e-3 cosine, against the winning direction; 12k). New, to 30k from step 0, 6k gate only (≤ 1.989):
**wave 42 = r3_K0.5_ub0.6_S64_lr2e-3_wsd20k_do0 (973872, compute-02)**, 2e-3 held to 20k (LR_HOLD_STEPS 17500);
**wave 43 = r3_K0.5_ub0.6_S64_lr3e-3_wsd14k_do0 (973873, compute-02)**, 3e-3 held to 14k;
**wave 44 = r3_K0.5_ub0.6_S64_lr3e-3_wsd20k_do0 (973879, compute-03)**, 3e-3 held to 20k (the 2 x 2 corner);
**wave 45 = wave 31 at seed 2 (973883, A100, `--seeds 2`)**, its replicate on the same hardware.
transformers 4.45's get_wsd_schedule has no decay_type (cosine only), so a held peak with a linear decay would need a
code change; not done. Score helpers armed for waves 42-44 (6k/10k/16k); waves 37 / 40 helpers stopped.
03:06: **No new waves after wave 40** (user, 2026-11-05: "I think we can stop at wave40", then chose "Let all
finish, no new waves"). Waves 39 and 41-45 run to 30k with no gate cancellations; the score helpers still record
their 6k/10k/16k scores. Write-up when the last one finishes (~Nov 6 morning-noon; A100 wave 45 last).
05:43: wave 41 (4e-3 linear) 6k = 1.9699 (std 1.26), +0.93% vs wave 34 (3e-3 linear, 1.9518) and +0.10% vs
3e-3 cosine (1.9680); at 6k 4e-3 is past the optimum, as 3e-3 was vs 2e-3 at cosine. Running on to 30k.
06:44: wave 44 (3e-3 held to 20k) 6k = 1.9559 (std 1.27): passes the 6k gate (<= 1.989); −0.04% vs 2e-3 held
(wave 30, 1.9566), +0.21% vs wave 34 (3e-3 linear, 1.9518, whose LR at 6k is already 2.6e-3). Running on.
06:47: wave 42 (2e-3 held to 20k) 6k = 1.9560 (std 1.26): the same LR path as wave 30 to 6k (1.9566, A100), so
desa vs A100 agree to 0.03% here. Running on.
06:49: wave 43 (3e-3 held to 14k) 6k = 1.9703 (std 1.27). Waves 43 and 44 have the same config through 14k (only
num_stable/num_decay_steps differ; same seed) yet differ by 0.74% at 6k (1.9703, A6000 compute-02 vs 1.9559, A5000
compute-03), while the 2e-3 pair (waves 30 / 42) agree to 0.03%. So at peak 3e-3 a 6k score carries ~0.7% run noise
(nondeterministic kernels, different GPUs); the 3e-3 / 4e-3 6k scores (1.952-1.970) can't be ranked. Running on.
09:28: wave 41 (4e-3 linear) 10k = 1.8692 (std 1.28): +0.78% vs wave 34 (3e-3 linear, 1.8548; +0.93% at 6k),
level with 3e-3 cosine (1.8683). Running on.
10:31: wave 44 (3e-3 held to 20k) 10k = 1.8596 (std 1.28), still at the full 3e-3: +0.26% vs wave 30 and wave 34
(both 1.8548, LR there ~1.9e-3 / 2.2e-3). Running on.
10:35: wave 42 (2e-3 held to 20k) 10k = 1.8622 (std 1.28): +0.40% vs wave 30 (1.8548, A100; LR 1.93e-3 there vs 2e-3),
though the two matched at 6k; +0.14% vs wave 44 (3e-3). Running on.
10:39: wave 43 (3e-3 held to 14k) 10k = 1.8710 (std 1.29): still +0.61% vs its same-config twin wave 44 (1.8596;
+0.74% at 6k). The gap persists, so it is a run-level divergence, not checkpoint noise. Running on.
15:13: wave 41 (4e-3 linear) 16k = 1.7915 (std 1.29): +0.81% vs wave 34 (1.7772; +0.93% / +0.78% at 6k / 10k), so
the gap is not closing. Wave 34 gained ~0.5% on cosine between 16k and 30k; wave 41 would need more than 0.8% to tie it.
Running on (no cancellations).
16:13: wave 44's 16k score job 995627 FAILED in 5 s on yu-compute-01 (exit 0:53, no log written; node fault).
Resubmitted the same script by hand as 995700 with --exclude=desa-compute-01,yu-compute-01.
16:14: wave 42 (2e-3 held to 20k) 16k = 1.7881 (std 1.30), still at 2e-3: +0.60% vs wave 30 (1.7775, LR 1.26e-3 there).
Expected for a held peak (wave 31 trailed mid-run too). Running on.
16:18: wave 44 (3e-3 held to 20k) 16k = 1.7856 (std 1.29; rescored job 995700): −0.14% vs wave 42 (2e-3 held),
+0.47% vs wave 34 (3e-3 linear, LR 1.53e-3 there). Running on.
16:23: wave 43 (3e-3 held to 14k) 16k = 1.7961 (std 1.30; LR 2.9e-3 there): +0.59% vs wave 44, the same gap to its
twin as at 6k / 10k (+0.74% / +0.61%). All 16k scores are in; next events are the 30k finals (Nov 6).
Nov 6 05:12: **wave 41 final = 1.6957** (973334 COMPLETED; 4e-3 linear; full val 1172/1172, std 1.29; GenPPL 10.64,
entropy 3.93, 64/64): +0.48% vs wave 34 (3e-3 linear, 1.6876), +0.64% vs the best (wave 31, 1.6849). The 16k gap
(+0.81%) narrowed but did not close: 4e-3 is past the optimum; 3e-3 stays the best linear peak.
06:03: **NEW BEST. Wave 42 final = 1.6832** (973872 COMPLETED, desa A6000; 2e-3 held to 20k, cosine over the last
10k; full val 1172/1172, std 1.30; GenPPL 13.21, entropy 3.77, 64/64): −0.10% vs wave 31 (held to 14k, 1.6849, A100),
inside the ~0.16% seed gap, but the hold trend is monotone: 6k 1.6901 > 14k 1.6849 > 20k 1.6832. Samples are worse
(GenPPL 13.2 vs 11.1, entropy 3.77 vs 3.85), though 64 samples are noisy (the cosine seeds differ by 1.1 GenPPL).
06:14: wave 43 final = 1.6911 (973873; 3e-3 held to 14k; full val 1172/1172, std 1.30; GenPPL 9.82, entropy 3.89,
64/64): +0.37% vs wave 31 (2e-3, same hold), +0.47% vs wave 42. This run was the weak twin of wave 44 all along
(+0.6-0.7% at 6k-16k), so wave 44's final is the cleaner read on peak 3e-3 held.
06:21: **wave 44 final = 1.6833** (973879, desa A5000; 3e-3 held to 20k; full val 1172/1172, std 1.30; GenPPL 9.90,
entropy 3.90, 64/64): ties wave 42 (2e-3 held to 20k, 1.6832; +0.004%), with better samples (GenPPL 9.9 vs 13.2).
Twins 43 / 44 end 0.46% apart (0.59% at 16k, LR nearly equal there), so ~0.5% is the run-to-run spread for one config
across desa GPUs at peak 3e-3: the 3e-3 vs 2e-3 and 14k vs 20k hold differences are within it. All desa waves done;
left: A100 waves 39 (wave 30 seed 2) and 45 (wave 31 seed 2).
Nov 6 18:28: both A100 finals pulled. Wave 39 final = 1.6946 (970595 COMPLETED 11:17; held to 6k, seed 2; full val
1172/1172, std 1.29; GenPPL 10.74, entropy 3.95, 64/64): +0.27% vs seed 1. Wave 45 final = 1.6872 (973883 COMPLETED
16:08; held to 14k, seed 2; full val 1172/1172, std 1.29; GenPPL 11.20, entropy 4.06, 64/64): +0.13% vs seed 1.
Two-seed means: cosine 1.6940, held to 6k 1.6923 (−0.10%, inside the seed gaps), held to 14k 1.6861 (−0.37% vs held to
6k, seed ranges do not overlap). Wave 42 stays the best, on one seed. Experiment finished: no new waves, per the user.
CANCELLED 2026-10-31 09:10 (user-approved; exp-mode, could not pass the 1/5 gate): S64 fix 853620,
bert fix 853621, bert / S64 / S4 ada 849445/6/7. They stay in sweep.py's CELLS as a record; don't
resubmit them. Their knobs run on the ub 0.5 base (wave 6).
1/5 gate at 07:10: fix ub1 (2.187) and r5 fix (2.756) stopped; ub0.25 ada 853910 and ub0.5 fix 853911
started on desa at 07:07.
After each stage the job scores `eval/nelbo_{steps}.json`: the exact NELBO on 10% of the validation set.
Results: `RESULTS.md`.

Gate, applied as the scores arrive (`gate.py`; it calls `sweep.py` with `--submit`, at the site that
holds the cell's checkpoints). Pull the A100 cells' scores first:
```bash
cd /share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/experiments/hbfm_elbo_tinystories_256
P=/share/desa/nfs02/sc3379/sfm_home/envs/sfm/bin/python
$P sweep.py --site a100 --pull
$P gate.py --steps 6000  --next 10000 --max 2.77 --margin 0.01 --submit
$P gate.py --steps 10000 --next 30000 --max 2.46 --margin 0.005 --submit   # wave-1 keep-10000 scores included
```
A cell that fails is not continued; a full-length (non-staged) job that fails is cancelled.
When GPUs free up, keep tuning: add fix-arm cells to `CELLS` / `ARMS['fix']` in sweep.py (they are
staged automatically) and submit them with `--steps 6000`.
Wave-1 reference scores (same protocol, `sweep.py --score-keep --steps N`): keep-10000 of all five
cells; keep-5000/15000/20000 of the two desa cells, for ranking stability. The A100 cells'
keep-10000 were exported from nlplarge's /scratch (CPU job 849453) into sfm_home/outputs.

## 2. Analyses

- `nelbo_variance.py` (via `nelbo_variance.sbatch`): NELBO mean (recon. vs path-KL), estimator std
  and between-sequence std vs ub (trunc_exp) and S on a fixed checkpoint → `<cell>/nelbo_variance_<ckpt>.json`.
  It ran on r3 fix keep-20000 (job 849438), then on the 30k checkpoints of r3 fix and ub 0.5 fix. The ada
  anchor's 30k checkpoint was not run; the answer did not need it.
- `trainer/loss_std` (the user's change, 2026-10-30 18:18): logged by the wave-2 runs only.
- `nelbo_variance.py` now accepts `--ubs ''` / `--words ''` (only the trained-horizon row); used for the 6k ub-sweep table.

## 3. Code changes this session (uncommitted)

- `sweep.py --after JOBID`: submits a cell's next stage with `--dependency=afterok:JOBID`, so it takes
  the slot its running stage frees (otherwise any pending job grabs it before the gate is run).

- `experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py`: skips the `keep-N-v1.ckpt`
  re-saves that a resumed stage writes (its step regex crashed on them, which would fail the λ step
  of every staged ada cell that reaches 30k, e.g. 860505).

- `utils.load_pretrained_word_embedding(..., tokenizer_name)`: if the model's vocabulary differs
  from the data's, it maps rows onto GPT-2 words (mean of WordPiece pieces, [UNK] when there are
  none). `models/hyperbolic_dit.py` passes `config.data.tokenizer_name_or_path`. Tested by
  `tests/test_pretrained_embed.py` (2 passed, job 849549; run from `tests/`, since pytest from the
  repo root trips on the unreadable `data_cache` link). pytest was installed into the sfm env copy.
- `sweep.py`: per-cell config dict (`cfg`), wave-2 cells, `--cells`, `--steps` (stages),
  `--score-keep`, `--wandb-sync`; drives the a6000 site locally when run as sc3379; KEEP_EVERY=2000.
- `report.py` reads `cfg()`; new files `gate.py`, `nelbo_variance.py/.sbatch`, `EXPERIMENT.md`.

## 4. Where things are

| | ch2263 (a100) | sc3379 (a6000) |
|---|---|---|
| repo | `/home/ch2263/syc_workspace/s-flm-dev/s-flm` (code synced 2026-10-31 01:12, `sweep.py --site a100 --sync-code`) | `/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm` |
| env | `/home/ch2263/miniconda3/envs/sfm` | `/share/desa/nfs02/sc3379/sfm_home/envs/sfm` |
| run dirs | `<repo>/outputs/hbfm_elbo_tinystories_256/<tag>`, ckpts on nlplarge /scratch | `/share/desa/nfs02/sc3379/sfm_home/outputs/hbfm_elbo_tinystories_256/<tag>` |

Watchers: `~/tmp/hbfm_elbo_watch.sh` (one line per sc3379 job); `~/tmp/hbfm_elbo_event.sh [secs]`
exits when a job changes state at either site or a new JSON appears.

When an A100 cell finishes (log ends `DONE`), copy it, with the wandb dir this time, so it can be
synced. Never copy ch2263's stale `r6_K1` / `r3_K0.5_curv-fix` stubs:
```bash
rsync -a --exclude=checkpoints -e "ssh -i ~/.ssh/unicorn_internal" \
  ch2263@unicorn-login-02.coecis.cornell.edu:/home/ch2263/syc_workspace/s-flm-dev/s-flm/outputs/hbfm_elbo_tinystories_256/<tag>/ \
  /share/desa/nfs02/sc3379/sfm_home/outputs/hbfm_elbo_tinystories_256/<tag>/
```
Then `python sweep.py --site a6000 --wandb-sync`. W&B: https://wandb.ai/syctw/hbfm_elbo_tinystories_256,
one run per cell, first synced 2026-10-30 23:10 while the cells were still training. Re-run it (after
re-copying the A100 cells' `wandb/`) at any time: it uploads only the rows W&B lacks (`synced_rows` in
the summary) and puts the stage NELBOs and final evals in the summary. Then
`python -c "import report; report.OUT='/share/desa/nfs02/sc3379/sfm_home/outputs/hbfm_elbo_tinystories_256'; report.main()"`.

## 5. Gotchas (still true)

- `trainer/loss` is the last micro-batch (a fixed t stratum), not a NELBO; training-history `val/*` are NaN.
- `pymp-*` `OSError: Device or resource busy` tracebacks are harmless (DataLoader temp cleanup on NFS).
- ch2263 is at its GPU cap; never cancel its interactive job 662408. This experiment's own jobs may be
  cancelled without asking (user, 2026-10-31 09:10: "You dont need my manual approval to cancel jobs",
  "For this experiment"), e.g. a cell that fails the gate or would only block better cells.
- sc3379 cannot use the thickstun partition ("User's group not permitted").
- Don't edit `scripts/train|sample/tinystories/hbfm.sh` while jobs run them.
