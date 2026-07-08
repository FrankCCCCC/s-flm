#!/bin/bash
# LangFlow sudoku eval — fair-comparison protocol (slides jun10_2026):
# 180 NFE (steps=179: 179 Euler + 1 final decode), greedy argmax decode,
# top-1 velocity (sampler.velocity=exact sampler.top_k_velocity=1 -> the Euler
# update target zhat = argmax-token embedding; same knobs as S-FLM's sfm.sh),
# EMA on, full 2000-puzzle valid set, exact-match over all 81 solution cells.
#
# VARIANT must match the trained checkpoint (model is rebuilt from this config).

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained LangFlow sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
VARIANT="${VARIANT:-full}"            # naive / sc / ada_sched / full
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/langflow_${VARIANT}_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-179}"                 # NFE = STEPS+1 = 180 (parity with S-FLM/HFLM STEPS=180)
VELOCITY="${VELOCITY:-exact}"         # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"   # 1 = top-1 predicted-clean endpoint; -1 = full expectation

case "${VARIANT}" in
  naive)     TRAINABLE=false; SELF_COND=false ;;
  sc)        TRAINABLE=false; SELF_COND=true ;;
  ada_sched) TRAINABLE=true;  SELF_COND=false ;;
  full)      TRAINABLE=true;  SELF_COND=true ;;
  *) echo "unknown VARIANT=${VARIANT}"; exit 1 ;;
esac

cd "${REPO_ROOT}"

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit \
    model.init=unit_var \
    algo=langflow \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.logit_bias=true \
    noise=gumbel \
    noise.trainable="${TRAINABLE}" \
    sampler=langflow \
    sampler.steps="${STEPS}" \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.temperature=1.0 \
    sudoku.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    sudoku.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
