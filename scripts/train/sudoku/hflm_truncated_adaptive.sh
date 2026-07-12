#!/bin/bash
# H-FLM with truncated + adaptive noise schedule. ALPHA_MAX default is the
# hyperbolic analog of the paper's Eq. 17 bound: alpha_star_hyperbolic(
# V=12, d=512) = 0.624 (noise_schedules.py; init=hyperbolic std 0.3,
# prior_cov=0.25, rho_max=12, delta=0.1). NOT the sphere bound 0.093 —
# that collapses HFLM (experiments/hflm/RESULTS.md). Recompute ALPHA_MAX
# if INIT/PRIOR_COV/RHO_MAX change (embed_std for INIT=ngpt is 1/sqrt(512)).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
GAUSS_CURV="${GAUSS_CURV:--1.0}"     # Gaussian curvature, restrict to < 0.0 for hyperbolic
INIT="${INIT:-hyperbolic}"           # embedding init: hyperbolic / ngpt / random / unit_var / custom
INIT_STD="${INIT_STD:-null}"         # std for INIT=custom; ignored otherwise
LR="${LR:-3e-4}"                     # AdamW learning rate
SEED="${SEED:-1}"                    # global random seed (L.seed_everything)
ALPHA_MAX="${ALPHA_MAX:-0.624}"      # alpha_star_hyperbolic(12, 512); null = no truncation
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/hflm_truncated_adaptive_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"

GLOBAL_BS=256
BUF_SIZE=$((50 * GLOBAL_BS))

cd "${REPO_ROOT}"

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-hyperbolic-dit \
    model.init="${INIT}" \
    model.init_std="${INIT_STD}" \
    optim.lr="${LR}" \
    seed="${SEED}" \
    algo=hflm \
    algo.invert_time_convention=false \
    algo.prior_cov="${PRIOR_COV:-0.25}" \
    algo.rho_max=12 \
    algo.gaussian_curvature="${GAUSS_CURV}" \
    sampler=hflm \
    noise="${NOISE:-log-linear-adaptive}" \
    noise.alpha_max="${ALPHA_MAX}" \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps="${MAX_STEPS:-20000}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    hydra.run.dir="${OUTPUT_DIR}"
