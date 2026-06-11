#!/bin/bash
# LangFlow TinyStories training — fair-comparison grid vs S-FLM/HFLM
# (slides jun10_2026). Recipe mirrors scripts/train/tinystories/sfm.sh
# verbatim: small DiT (768/12/12), 30k steps, global batch 512 (32/GPU x 4),
# seq 1024, AdamW lr 3e-4 wd 0, EMA 0.9999, bf16, ckpt every 2500.
#
# VARIANT (2x2 over the two LangFlow tricks; logit_bias ON for all):
#   naive     = fixed Gumbel(0,1) schedule, no self-cond
#   sc        = fixed schedule + self-conditioning
#   ada_sched = trainable Gumbel (info-uniform), no self-cond
#   full      = trainable Gumbel + self-conditioning

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
VARIANT="${VARIANT:-full}"            # naive / sc / ada_sched / full
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/langflow_${VARIANT}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

GLOBAL_BS=512
# Microbatch 16 (vs S-FLM's 32): LangFlow's Plaid logit bias adds [B,L,V]
# tensors to the fwd/bwd peak, which OOMs 44GB at B=32/GPU. Global batch is
# unchanged (accumulate_grad_batches auto-doubles to 8) -> identical math.
MICRO_BS="${MICRO_BS:-16}"

case "${VARIANT}" in
  naive)     TRAINABLE=false; SELF_COND=false ;;
  sc)        TRAINABLE=false; SELF_COND=true ;;
  ada_sched) TRAINABLE=true;  SELF_COND=false ;;
  full)      TRAINABLE=true;  SELF_COND=true ;;
  *) echo "unknown VARIANT=${VARIANT}"; exit 1 ;;
esac

cd "${REPO_ROOT}"

# find_unused_parameters: LangFlow's graph legitimately varies per step
# (Bernoulli self-cond branch; Plaid-bias r-ramp activates after step 0), which
# trips DDP's static-graph rebuild check. Gradients are unaffected.
python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=ddp \
    strategy.find_unused_parameters=true \
    model=small-sphere-dit \
    model.init=unit_var \
    algo=langflow \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.p_self_cond=0.25 \
    algo.logit_bias=true \
    algo.logit_bias_warmup_steps=5000 \
    noise=gumbel \
    noise.trainable="${TRAINABLE}" \
    lr_scheduler=constant_warmup \
    optim.lr=3e-4 \
    training.ema=0.9999 \
    loader.global_batch_size=${GLOBAL_BS} \
    loader.batch_size="${MICRO_BS}" \
    loader.eval_batch_size="${MICRO_BS}" \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=30_000 \
    trainer.val_check_interval=60_000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2_500 \
    wandb.project=tinystories-flm \
    wandb.group=geometry-vs-tricks \
    +wandb.name="tinystories_langflow_${VARIANT}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
