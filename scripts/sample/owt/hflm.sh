#!/bin/bash
# Hyperbolic-FLM OWT sampling + Gen-PPL / entropy frontier point.
# Mirrors scripts/sample/owt/sfm.sh. Sweep TEMP over {0.70,0.75,...,1.20} and
# STEPS over {32,1024} to trace the frontier (Suppl. C.8). noise=log-linear (HFLM
# trains without truncation; see scripts/train/owt/hflm.sh).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained HFLM OWT checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/hflm}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
TEMP="${TEMP:-1.0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
VELOCITY="${VELOCITY:-exact}"
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-hyperbolic-dit \
    algo=hflm \
    algo.prior_cov=0.25 \
    algo.rho_max=12 \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear \
    sampler=hflm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.steps="${STEPS}" \
    sampler.temperature="${TEMP}" \
    sampler.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
