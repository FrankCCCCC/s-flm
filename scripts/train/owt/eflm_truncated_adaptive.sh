#!/bin/bash
# E-FLM on OpenWebText with truncated + adaptive noise schedule — mirrors
# scripts/train/owt/sfm.sh (model size 768/12/12, batch 512, adaptive knobs)
# with algo=eflm. ALPHA_MAX default is the Euclidean analog of the paper's
# Eq. 17 bound: alpha_star_euclidean(V=50257) = 0.840 (noise_schedules.py;
# ngpt init ||e||~=1, N(0,I) prior, delta=0.1).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/owt/eflm_truncated_adaptive}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-8}"
MAX_STEPS="${MAX_STEPS:-1000000}"
GLOBAL_BS="${GLOBAL_BS:-512}"
PER_GPU_BS="${PER_GPU_BS:-32}"
ALPHA_MAX="${ALPHA_MAX:-0.840}"      # alpha_star_euclidean(50257); null = no truncation

BUF_SIZE=$((50 * GLOBAL_BS))

cd "${REPO_ROOT}"

python -u -m main \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=ngpt \
    algo=eflm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=${ALPHA_MAX} \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=50_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    wandb.project=eflm-owt \
    hydra.run.dir="${OUTPUT_DIR}"
