#!/bin/bash
# LangFlow sudoku training — fair-comparison grid vs S-FLM/HFLM (slides jun10_2026).
# Same recipe as scripts/train/sudoku/sfm.sh: tiny DiT (512/8/8, seq 180),
# 20k steps, global batch 256, AdamW lr 3e-4 wd 0, EMA 0.9999, bf16, 1 GPU.
#
# VARIANT (2x2 over the two LangFlow tricks; logit_bias is base LangFlow, ON for all):
#   naive     = fixed Gumbel(0,1) schedule (no info-uniform adaptation), no self-cond
#   sc        = fixed schedule + self-conditioning
#   ada_sched = trainable Gumbel (info-uniform principle), no self-cond
#   full      = trainable Gumbel + self-conditioning

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
VARIANT="${VARIANT:-full}"            # naive / sc / ada_sched / full
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/langflow_${VARIANT}_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-20_000}"
CKPT_EVERY="${CKPT_EVERY:-5_000}"

case "${VARIANT}" in
  naive)     TRAINABLE=false; SELF_COND=false ;;
  sc)        TRAINABLE=false; SELF_COND=true ;;
  ada_sched) TRAINABLE=true;  SELF_COND=false ;;
  full)      TRAINABLE=true;  SELF_COND=true ;;
  *) echo "unknown VARIANT=${VARIANT}"; exit 1 ;;
esac

cd "${REPO_ROOT}"

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    strategy=single-device \
    model=tiny-sphere-dit \
    model.init=unit_var \
    algo=langflow \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.p_self_cond=0.25 \
    algo.logit_bias=true \
    algo.logit_bias_warmup_steps=5000 \
    noise=gumbel \
    noise.trainable="${TRAINABLE}" \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps="${MAX_STEPS}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
    +wandb.name="sudoku_langflow_${VARIANT}_${DIFFICULTY}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
