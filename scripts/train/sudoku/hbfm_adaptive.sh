#!/bin/bash
# HBFM (hyperbolic bridge DLM). Single Sudoku training run: tiny-hyperbolic-dit
# with the product manifold (H^FACTOR_DIM_K)^(EMBED_DIM/FACTOR_DIM) lifted into
# the DiT by model.embed_dim (HyperbolicDiT.in_proj). The heat-time proposal is
# stated in UNIT time (exp rate UNIT_PROPOSAL_RATE * |K|): the 32 factors'
# evidence adds, and with a 12-token vocabulary an untrained model already
# resolves a cell by unit time ~0.1, so the default rate is 20 (mean tau 0.05).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-hard}"        # easy / medium / hard
GAUSS_CURV="${GAUSS_CURV:--1.0}"        # Gaussian curvature K < 0 of every factor
EMBED_DIM="${EMBED_DIM:-96}"            # manifold dim; null = DiT hidden size
FACTOR_DIM="${FACTOR_DIM:-3}"           # H^FACTOR_DIM factors, EMBED_DIM/FACTOR_DIM of them
PROD_FACTOR_DIM="${PROD_FACTOR_DIM:-${FACTOR_DIM}}"   # hydra value of algo.prod_factor_dim: an int (equal split) or a list like [3,3,3]
PROD_FACTOR_CURV="${PROD_FACTOR_CURV:-${GAUSS_CURV}}" # hydra value of algo.prod_factor_gaussian_curvature: a float or a list; GAUSS_CURV stays the scalar |K| used for the unit-time conversions
INIT="${INIT:-ngpt}"
LR="${LR:-3e-4}"
SEED="${SEED:-1}"
UNIT_PROPOSAL_RATE="${UNIT_PROPOSAL_RATE:-20}"    # exp rate in unit-model time
READOUT_PRECISION="${READOUT_PRECISION:-float32}" # float64 / float32 (u stays < 3 here)
FORWARD_TYPE="${FORWARD_TYPE:-horosphere}"  # horosphere: logits are a residual on the Busemann log-densities / naive: plain logits
MAX_STEPS="${MAX_STEPS:-20000}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
PER_GPU_BS="${PER_GPU_BS:-256}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/hbfm_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
PROPOSAL_RATE=$(python -c "print(${UNIT_PROPOSAL_RATE} * abs(${GAUSS_CURV}))")

cd "${REPO_ROOT}"
ADA_ARGS=""
case "${NOISE}" in *adaptive*) ADA_ARGS="noise.adaptive_refit_every=${ADA_REFIT_EVERY} noise.adaptive_ema=${ADA_EMA} noise.adaptive_buffer_size=$((50 * GLOBAL_BATCH))";; esac

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-hyperbolic-dit \
    model.embed_dim=${EMBED_DIM} \
    model.init="${INIT}" \
    optim.lr="${LR}" \
    seed="${SEED}" \
    algo=hbfm \
    algo.prod_factor_dim="${PROD_FACTOR_DIM}" \
    algo.prod_factor_gaussian_curvature="${PROD_FACTOR_CURV}" \
    algo.time_exp_rate=${PROPOSAL_RATE} \
    algo.readout_precision=${READOUT_PRECISION} \
    algo.forward_type=${FORWARD_TYPE} \
    sampler=hbfm \
    noise=log-linear-adaptive \
    noise.alpha_max=${ALPHA_MAX} \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BATCH} \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps=${MAX_STEPS} \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    hydra.run.dir="${OUTPUT_DIR}" 
