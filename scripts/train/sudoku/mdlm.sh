#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/mdlm_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
SEED="${SEED:-1}"
LR="${LR:-3e-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
PER_GPU_BS="${PER_GPU_BS:-256}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"

cd "${REPO_ROOT}"

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny \
    algo=mdlm \
    seed="${SEED}" \
    optim.lr="${LR}" \
    loader.global_batch_size=${GLOBAL_BATCH} \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=${NUM_WORKERS:-8} \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps=${MAX_STEPS} \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    hydra.run.dir="${OUTPUT_DIR}"
