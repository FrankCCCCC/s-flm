#!/bin/bash
# HBFM — Sudoku board accuracy (sudoku_eval) of ONE checkpoint. The geometry knobs
# (EMBED_DIM / FACTOR_DIM / GAUSS_CURV / rate / precision) MUST match training.
# The sampler horizon is in UNIT time (t_max = UNIT_T_MAX / |K|); tokens are resolved
# by unit time ~0.1 here, so 180 steps over 0.5 give dtau ~ 0.003 on the decision phase.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained HBFM sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-hard}"
GAUSS_CURV="${GAUSS_CURV:--1.0}"
EMBED_DIM="${EMBED_DIM:-96}"
FACTOR_DIM="${FACTOR_DIM:-3}"
PROD_FACTOR_DIM="${PROD_FACTOR_DIM:-${FACTOR_DIM}}"   # hydra value of algo.prod_factor_dim: an int (equal split) or a list like [3,3,3]
PROD_FACTOR_CURV="${PROD_FACTOR_CURV:-${GAUSS_CURV}}" # hydra value of algo.prod_factor_gaussian_curvature: a float or a list; GAUSS_CURV stays the scalar |K| used for the unit-time conversions
INIT="${INIT:-ngpt}"
SEED="${SEED:-1}"
UNIT_PROPOSAL_RATE="${UNIT_PROPOSAL_RATE:-20}"
READOUT_PRECISION="${READOUT_PRECISION:-float32}"
NOISE="${NOISE:-log-linear}"
FORWARD_TYPE="${FORWARD_TYPE:-horosphere}"  # horosphere: logits are a residual on the Busemann log-densities / naive: plain logits
ADA_REFIT_EVERY="${ADA_REFIT_EVERY:-500}"  # AdaptiveSchedule knobs, defaults = configs/noise/log-linear-adaptive.yaml; only passed when NOISE is adaptive. The repo's 50 / 0.9 recipe (hflm_truncated_adaptive.sh) was A/B-tested for HBFM at rate 0.01 and did not help (experiments/claude_test_hbfm_sudoku/RESULTS.md).
ADA_EMA="${ADA_EMA:-0.0}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"         # training global batch: the adaptive schedule's buffer size (gbs * refit_every // 10) must match the checkpoint
UNIT_T_MAX="${UNIT_T_MAX:-0.5}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"          # exact (posterior-mean drift = top_k_v -1) / sample
NOISE_REMOVAL="${NOISE_REMOVAL:-greedy}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/hbfm_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
PROPOSAL_RATE=$(python -c "print(${UNIT_PROPOSAL_RATE} * abs(${GAUSS_CURV}))")
T_MAX=$(python -c "print(${UNIT_T_MAX} / abs(${GAUSS_CURV}))")

cd "${REPO_ROOT}"
ADA_ARGS=""
case "${NOISE}" in *adaptive*) ADA_ARGS="noise.adaptive_refit_every=${ADA_REFIT_EVERY} noise.adaptive_ema=${ADA_EMA} noise.adaptive_buffer_size=$((50 * GLOBAL_BATCH))";; esac

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-hyperbolic-dit \
    model.embed_dim=${EMBED_DIM} \
    model.init="${INIT}" \
    seed="${SEED}" \
    algo=hbfm \
    algo.prod_factor_dim="${PROD_FACTOR_DIM}" \
    algo.prod_factor_gaussian_curvature="${PROD_FACTOR_CURV}" \
    algo.time_exp_rate=${PROPOSAL_RATE} \
    algo.readout_precision=${READOUT_PRECISION} \
    algo.forward_type=${FORWARD_TYPE} \
    noise=${NOISE} ${ADA_ARGS} \
    sampler=hbfm \
    sampler.noise_removal=${NOISE_REMOVAL} \
    sampler.velocity="${VELOCITY}" \
    sampler.steps="${STEPS}" \
    sampler.t_max=${T_MAX} \
    loader.global_batch_size=${GLOBAL_BATCH} \
    sudoku.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    sudoku.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
