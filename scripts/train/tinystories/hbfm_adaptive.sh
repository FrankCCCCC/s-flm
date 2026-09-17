#!/bin/bash
# HBFM (hyperbolic bridge DLM). Single TinyStories training run.
# The product manifold H^{FACTOR_DIM}_K x ... over EMBED_DIM channels is lifted
# into the small DiT by model.embed_dim < hidden_size (HyperbolicDiT.in_proj).
# The heat-time proposal is specified in UNIT time: with curvature K the bridge on
# H_K at time t is the unit-model bridge at tau = |K| t, so the exp rate is
# UNIT_PROPOSAL_RATE * |K| and a curvature sweep changes only the scale
# R = 1/sqrt|K| of the Poincare-ball state the DiT sees. The rate is set by the
# manifold: the factors' evidence adds, so (H^3)^32 resolves a token at u < 1
# (untrained CE is 0 past tau ~ 0.5); rate 5 (mean tau 0.2) covers that range.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/hbfm}"
RUN_NAME="${RUN_NAME:-hbfm}"
WANDB_GROUP="${WANDB_GROUP:-hbfm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
MODEL="${MODEL:-small-hyperbolic-dit}"
SEQ_LEN="${SEQ_LEN:-256}"
LR="${LR:-3e-4}"
SEED="${SEED:-1}"
INIT="${INIT:-ngpt}"
EMBED_DIM="${EMBED_DIM:-96}"              # manifold dim; null = DiT hidden size
FACTOR_DIM="${FACTOR_DIM:-3}"             # H^FACTOR_DIM factors, EMBED_DIM/FACTOR_DIM of them
GAUSS_CURV="${GAUSS_CURV:--1.0}"          # Gaussian curvature K < 0 of every factor
UNIT_PROPOSAL_RATE="${UNIT_PROPOSAL_RATE:-5}"     # exp rate in unit-model time (see below)
READOUT_PRECISION="${READOUT_PRECISION:-float64}"  # float64 / float32 (see configs/algo/hbfm.yaml)
ALPHA_MAX="${ALPHA_MAX:-null}"            # noise.alpha_max: null = untruncated; a value < 1 drops the LATE heat times (u = 1 - alpha >= 1 - alpha_max)
PROPOSAL_RATE=$(python -c "print(${UNIT_PROPOSAL_RATE} * abs(${GAUSS_CURV}))")

cd "${REPO_ROOT}"
python -u -m main \
    seed=${SEED} \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=${MODEL} \
    model.length=${SEQ_LEN} \
    model.embed_dim=${EMBED_DIM} \
    model.init=${INIT} \
    algo=hbfm \
    algo.prod_factor_dim=${FACTOR_DIM} \
    algo.prod_factor_gaussian_curvature=${GAUSS_CURV} \
    algo.time_exp_rate=${PROPOSAL_RATE} \
    algo.readout_precision=${READOUT_PRECISION} \
    sampler=hbfm \
    noise=log-linear-adaptive \
    noise.alpha_max=${ALPHA_MAX} \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    optim.lr=${LR} \
    loader.global_batch_size=${GLOBAL_BATCH} \
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
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    callbacks.checkpoint_every_n_steps.save_top_k=${SAVE_TOPK:-1} \
    wandb.project=tinystories-flm \
    wandb.group="${WANDB_GROUP}" \
    +wandb.name="${RUN_NAME}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
