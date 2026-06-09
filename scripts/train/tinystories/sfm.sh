#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/sfm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))

cd "${REPO_ROOT}"

python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=ngpt \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=30_000 \
    trainer.val_check_interval=60_000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2_500 \
    wandb.project=tinystories-flm \
    wandb.group=geometry-vs-tricks \
    +wandb.name=tinystories_sfm_naive \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
