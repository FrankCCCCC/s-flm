#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained eflm_rescale_variational sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/eflm_rescale_variational_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"         # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"  # 1 = top-1, -1 = full vocab (no top-k)
SELF_COND="${SELF_COND:-false}"      # self-conditioning; must match training
RHO="${RHO:-1.0}"                    # fixed embedding norm R; must match training
VAR_SCOPE="${VAR_SCOPE:-positional}"  # must match training
VAR_CONTEXT="${VAR_CONTEXT:-prompt}"  # must match training
VAR_DEGREE="${VAR_DEGREE:-5}"        # must match training

cd "${REPO_ROOT}"

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit \
    algo=eflm \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.rho_min="${RHO}" \
    algo.rho_max="${RHO}" \
    noise=log-linear-variational \
    noise.var_scope="${VAR_SCOPE}" \
    noise.var_context="${VAR_CONTEXT}" \
    noise.var_degree="${VAR_DEGREE}" \
    sampler=eflm \
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
