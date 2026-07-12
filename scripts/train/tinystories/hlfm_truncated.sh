#!/bin/bash
# H-FLM with truncated noise schedule. Single TinyStories training run.
# ALPHA_MAX default is the hyperbolic analog of the paper's Eq. 17 bound:
# alpha_star_hyperbolic(V=50257, d=768, embed_std=1/sqrt(768)) = 0.979
# (noise_schedules.py; matches the default INIT=ngpt, prior_cov=0.25,
# rho_max=12, delta=0.1 — nearly vacuous because the ngpt clean radius ~1
# is dwarfed by the noise radius ~10). For INIT=hyperbolic (embed_std=0.3)
# set ALPHA_MAX=0.608. NOT the sphere bound 0.121 — that collapses HFLM
# (experiments/hflm/RESULTS.md).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/hflm_truncated}"
RUN_NAME="${RUN_NAME:-hflm_truncated}"
WANDB_GROUP="${WANDB_GROUP:-adv_geo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
INIT="${INIT:-ngpt}"            # ngpt | hyperbolic | custom
INIT_STD="${INIT_STD:-}"        # required only when INIT=custom
PRIOR_COV="${PRIOR_COV:-0.25}"
RHO_MAX="${RHO_MAX:-12}"
GAUSS_CURV="${GAUSS_CURV:--1.0}"    # Gaussian curvature, restrict to < 0.0 for hyperbolic
ALPHA_MAX="${ALPHA_MAX:-0.979}"     # alpha_star_hyperbolic for INIT=ngpt; null = no truncation
if [ "${INIT}" = "custom" ]; then INIT_ARGS="model.init=custom model.init_std=${INIT_STD}"; else INIT_ARGS="model.init=${INIT}"; fi

cd "${REPO_ROOT}"
python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-hyperbolic-dit \
    model.length=${SEQ_LEN:-1024} \
    ${INIT_ARGS} \
    algo=hflm \
    algo.prior_cov=${PRIOR_COV} \
    algo.rho_max=${RHO_MAX} \
    algo.gaussian_curvature=${GAUSS_CURV} \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    sampler=hflm \
    noise=log-linear \
    noise.alpha_max=${ALPHA_MAX} \
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
