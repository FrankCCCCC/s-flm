#!/bin/bash
# S-FLM with truncated noise schedule. Single TinyStories training run (slides jun25_2026).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/sfm_truncated}"
RUN_NAME="${RUN_NAME:-sfm_truncated}"
WANDB_GROUP="${WANDB_GROUP:-adv_geo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
LR="${LR:-3e-4}"
ALPHA_MAX="${ALPHA_MAX:-0.121}"
SELF_COND="${SELF_COND:-false}"      # LangFlow-style self-conditioning
# self-conditioning leaves the self-cond params unused on ~75% of steps (p_self_cond);
# default ddp strategy (find_unused_parameters=false) errors on that -> enable when self-cond.
if [ "${SELF_COND}" = "true" ]; then SC_STRAT="strategy.find_unused_parameters=true"; else SC_STRAT=""; fi

cd "${REPO_ROOT}"
python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.length=${SEQ_LEN:-1024} \
    model.init=ngpt \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    noise=log-linear \
    noise.alpha_max=${ALPHA_MAX} \
    optim.lr=${LR} \
    loader.global_batch_size=512 \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    ${SC_STRAT} \
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
