#!/bin/bash
# LangFlow smoke run (EXPERIMENT.md §10(iii)) — correctness/invariants only.
# tiny-sphere-dit, 200 steps, 1 GPU, single-device strategy. Embedding init =
# unit_var (ARCH §7.4). model.length overridden to 1024 to match the tinystories
# context (tiny-sphere-dit defaults to 180, which would shape-error on the data).
# Cluster env per the single-GPU workaround: SLURM_JOB_NAME=bash NCCL_P2P_DISABLE=1.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-bash}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/langflow_smoke}"

cd "${REPO_ROOT}"

python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=tiny-sphere-dit \
    model.init=unit_var \
    model.length=1024 \
    algo=langflow \
    algo.invert_time_convention=false \
    noise=gumbel \
    noise.trainable=true \
    algo.self_conditioning=true \
    algo.logit_bias=true \
    loader.global_batch_size=32 \
    loader.batch_size=32 \
    trainer.devices=1 \
    strategy=single-device \
    trainer.max_steps=200 \
    trainer.val_check_interval=200 \
    trainer.num_sanity_val_steps=0 \
    +wandb.name=tinystories_langflow_smoke \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
