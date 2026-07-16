#!/bin/bash
# E-FLM with truncated + adaptive noise schedule. Single TinyStories training
# run. ALPHA_MAX default is the Euclidean analog of the paper's Eq. 17 bound:
# alpha_star_euclidean(V=50257) = 0.840 (noise_schedules.py; ngpt init
# ||e||~=1, N(0,I) prior, delta=0.1).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eflm_truncated_adaptive}"
RUN_NAME="${RUN_NAME:-eflm_truncated_adaptive}"
WANDB_GROUP="${WANDB_GROUP:-adv_geo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
LR="${LR:-3e-4}"
ALPHA_MAX="${ALPHA_MAX:-0.840}"      # alpha_star_euclidean(50257); null = no truncation

cd "${REPO_ROOT}"
python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.length=${SEQ_LEN:-1024} \
    model.init=ngpt \
    algo=eflm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=${ALPHA_MAX} \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
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
