#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained HFLM sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
GAUSS_CURV="${GAUSS_CURV:--1.0}"     # Gaussian curvature K < 0; must match training
INIT="${INIT:-hyperbolic}"           # embedding init; must match training (custom needs INIT_STD)
INIT_STD="${INIT_STD:-null}"         # std for INIT=custom; ignored otherwise
SEED="${SEED:-1}"                    # eval sampling seed (match training seed for paired runs)
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/hflm_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"         # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"   # 1 = top-1 predicted-clean endpoint

cd "${REPO_ROOT}"

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-hyperbolic-dit \
    model.init="${INIT}" \
    model.init_std="${INIT_STD}" \
    seed="${SEED}" \
    algo=hflm \
    algo.invert_time_convention=false \
    algo.prior_cov=0.25 \
    algo.rho_max=12 \
    algo.gaussian_curvature="${GAUSS_CURV}" \
    noise=log-linear \
    sampler=hflm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.steps="${STEPS}" \
    sudoku.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    sudoku.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
