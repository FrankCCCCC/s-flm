#!/bin/bash
# Hyperbolic-FLM on TinyStories — mirrors scripts/train/tinystories/sfm.sh (same model size
# 768/12/12, batch 512, optimizer, EMA) for a fair geometry comparison.
# NECESSARY change vs sfm.sh: noise=log-linear (NO truncation / NO adaptive). The
# sphere-tuned truncation (alpha_max) collapses HFLM (see Sudoku RESULTS.md), so the
# hyperbolic model uses the full schedule. Run a matched naive S-FLM (also plain
# log-linear) as the controlled-geometry baseline.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/hflm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"
MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BS="${GLOBAL_BS:-512}"
PER_GPU_BS="${PER_GPU_BS:-32}"

cd "${REPO_ROOT}"

python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-hyperbolic-dit \
    algo=hflm \
    algo.prior_cov=0.25 \
    algo.rho_max=12 \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    sampler=hflm \
    noise=log-linear \
    loader.global_batch_size=${GLOBAL_BS} \
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
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2_500 \
    wandb.project=tinystories-flm \
    wandb.group=geometry-vs-tricks \
    +wandb.name=tinystories_hflm \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
