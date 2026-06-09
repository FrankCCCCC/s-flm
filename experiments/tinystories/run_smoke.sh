#!/bin/bash
# Phase-7 smoke train run for the TinyStories integration (EXPERIMENT.md criterion 5).
# Short MDLM run on the full TinyStories dataset: 50 steps, 1 GPU, online W&B.
# NOTE: first run tokenizes the full train split (~2.1M stories) before training.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm"
PY="/home/sc3379/anaconda3/envs/sfm/bin/python"
CACHE_DIR="${REPO_ROOT}/data_cache"
OUTPUT_DIR="${REPO_ROOT}/outputs/tinystories/smoke50"

cd "${REPO_ROOT}"

"${PY}" -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small \
    algo=mdlm \
    strategy=single-device \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    loader.num_workers=4 \
    eval.generate_samples=False \
    eval.compute_generative_perplexity=False \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_steps=50 \
    trainer.log_every_n_steps=5 \
    trainer.num_sanity_val_steps=0 \
    trainer.limit_val_batches=0 \
    trainer.val_check_interval=1000 \
    checkpointing.resume_from_ckpt=False \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
    wandb.project=tinystories-integration \
    wandb.group=dataset-integration \
    hydra.run.dir="${OUTPUT_DIR}"
