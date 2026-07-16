#!/bin/bash
# E-FLM with truncated noise schedule. ALPHA_MAX default is the Euclidean
# analog of the paper's Eq. 17 bound: alpha_star_euclidean(V=12) = 0.767
# (noise_schedules.py; ngpt init ||e||~=1, N(0,I) prior, delta=0.1).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/eflm_truncated_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
SEED="${SEED:-1}"                    # global random seed (L.seed_everything)
LR="${LR:-3e-4}"                     # AdamW learning rate
ALPHA_MAX="${ALPHA_MAX:-0.767}"      # alpha_star_euclidean(12); null = no truncation

cd "${REPO_ROOT}"

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    seed="${SEED}" \
    model=tiny-sphere-dit \
    optim.lr="${LR}" \
    algo=eflm \
    algo.invert_time_convention=false \
    noise=log-linear \
    noise.alpha_max="${ALPHA_MAX}" \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    hydra.run.dir="${OUTPUT_DIR}"
