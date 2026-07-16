#!/bin/bash
# Eval for H-FLM trained with truncated + adaptive noise schedule. The noise
# config must mirror training so the adapted schedule (alpha_vals buffers)
# loads from the checkpoint and drives the sampling trajectory. GAUSS_CURV /
# INIT / INIT_STD / ALPHA_MAX must match the training run.
# ALPHA_MAX default = alpha_star_hyperbolic(V=12, d=512) = 0.624 for the
# default INIT=hyperbolic, K=-1 (noise_schedules.py); other geometries: see
# experiments/trunc_ada_sudoku/alpha_star_numeric.py.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained HFLM truncated+adaptive sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
GAUSS_CURV="${GAUSS_CURV:--1.0}"     # Gaussian curvature K < 0; must match training
INIT="${INIT:-hyperbolic}"           # embedding init; must match training (custom needs INIT_STD)
INIT_STD="${INIT_STD:-null}"         # std for INIT=custom; ignored otherwise
SEED="${SEED:-1}"                    # eval sampling seed (match training seed for paired runs)
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/hflm_truncated_adaptive_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"         # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"   # 1 = top-1 predicted-clean endpoint
ALPHA_MAX="${ALPHA_MAX:-0.624}"       # must match training
UNIFORM_MIX="${UNIFORM_MIX:-1e-3}"    # must match training

GLOBAL_BS=256
BUF_SIZE=$((50 * GLOBAL_BS))

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
    algo.prior_cov="${PRIOR_COV:-0.25}" \
    algo.rho_max=12 \
    algo.gaussian_curvature="${GAUSS_CURV}" \
    noise="${NOISE:-log-linear-adaptive}" \
    noise.alpha_max="${ALPHA_MAX}" \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix="${UNIFORM_MIX}" \
    noise.adaptive_log_importance="${LOG_IMPORTANCE:-false}" \
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
