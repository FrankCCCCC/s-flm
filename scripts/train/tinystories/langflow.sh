#!/bin/bash
# LangFlow — paper hyperparameters (EXPERIMENT.md §10(i)).
# Embedding init = unit_var (ARCH §7.4, Option A): std=1 so ||z||~sqrt(D)
# matches the N(0,I) VP corruption.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/langflow}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

GLOBAL_BS=512

cd "${REPO_ROOT}"

python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=unit_var \
    algo=langflow \
    algo.invert_time_convention=false \
    noise=gumbel \
    noise.trainable=true \
    algo.self_conditioning=true \
    algo.p_self_cond=0.25 \
    algo.logit_bias=true \
    algo.logit_bias_warmup_steps=5000 \
    lr_scheduler=constant_warmup \
    optim.lr=3e-4 \
    training.ema=0.9999 \
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
    +wandb.name=tinystories_langflow_paperhp \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
