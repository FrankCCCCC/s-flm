#!/bin/bash
# HyperbolicBoundaryFM (HBFM) Sudoku accuracy eval.
# Mirrors scripts/sample/sudoku/sfm.sh; uses the HBFM geodesic sampler.
# greedy decode at the final step to match the S-FLM comparand.
#
#   CKPT_PATH=outputs/sudoku/hbfm_d64_easy_seed1/checkpoints/last.ckpt \
#   D=64 DIFFICULTY=easy ./scripts/sample/sudoku/hbfm.sh

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained HBFM sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
D="${D:-64}"                          # must match the trained checkpoint's hidden_size
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/hbfm_d${D}_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"

if [ "${D}" -le 2 ]; then N_HEADS=1; else N_HEADS="${N_HEADS:-8}"; fi

cd "${REPO_ROOT}"

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit \
    model.hidden_size="${D}" \
    model.n_heads="${N_HEADS}" \
    algo=hbfm \
    noise=log-linear \
    sampler=hbfm \
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
