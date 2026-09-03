#!/bin/bash
# SimpFLM — E-FLM with the word-embedding matrix replaced by the DIAGONAL
# R * I_V: every "embedding" is a one-hot simplex vertex, so the Gaussian
# Euclidean flow x_t = (1 - b_t) e + b_t z runs in the one-hot (logit) space
# R^V. The small-flm (flm-dit) backbone projects that [B, L, V] blend down to
# the model width. R = RHO (algo.rho_min = rho_max = RHO) is E-FLM's own radial
# rescale applied to the diagonal (rho_min and rho_max must be EQUAL), and it
# is the only geometric knob.#
# AUTONOMOUS clock: 1 - alpha_t = exp(-tau), tau = TAU_MAX * (1 - t), so the
# bridge drift is the time-invariant v(X) = y - X and every Euler step advances
# the same d_tau. TAU_MAX *is* the truncation on this clock (noise-fraction
# floor exp(-TAU_MAX)); tau*(R) = -log(1 - alpha_star_euclidean(V=50257,
# embed_norm=R)) = log(1 + C/R) stops at the decode point (C = 5.2575, so
# tau*(1) = 1.834), and TAU_MAX = 6.908 (= -log 1e-3) is the untruncated
# horizon.#
# Plus the ADAPTIVE time remap: a spline reweighting of t onto where |dL/dt| is
# largest (noise_schedules.AdaptiveSchedule). It sits on top of the truncated
# alpha_t, so the decode point is unchanged -- only the density of visited noise
# levels moves. Requires the MDLM time convention (invert_time_convention=false),
# this algo's default. The fitted remap ships in the checkpoint, so train and
# eval must use the same `noise=` config.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/simpflm_auto_truncation_adaptive}"
RUN_NAME="${RUN_NAME:-simpflm_auto_truncation_adaptive}"
WANDB_GROUP="${WANDB_GROUP:-simpflm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
LR="${LR:-3e-4}"
RHO="${RHO:-1.0}"                   # R: the radius of the simplex sphere (algo.rho_min = rho_max = RHO)
TAU_MAX="${TAU_MAX:-1.834}"         # autonomous horizon = the truncation; tau*(R=1) = 1.834

cd "${REPO_ROOT}"
python -u -m main \
    seed=${SEED:-1} \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-flm \
    model.length=${SEQ_LEN:-1024} \
    algo=simpflm \
    algo.invert_time_convention=false \
    algo.rho_min="${RHO}" \
    algo.rho_max="${RHO}" \
    sampler=simpflm \
    noise=autonomous-adaptive \
    noise.tau_max=${TAU_MAX} \
    optim.lr=${LR} \
    loader.global_batch_size=512 \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=60_000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    callbacks.checkpoint_every_n_steps.save_top_k=${SAVE_TOPK:-1} \
    wandb.project=tinystories-flm \
    wandb.group="${WANDB_GROUP}" \
    +wandb.name="${RUN_NAME}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
