#!/bin/bash
# EFLM — eval ONE TinyStories checkpoint: valid PPL (ppl_eval) + GenPPL (sample_eval).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=/abs/path/to/checkpoint.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eval/eflm}"
DEVICES="${DEVICES:-1}"
EVAL_BS="${EVAL_BS:-16}"
STEPS="${STEPS:-180}"
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"
VELOCITY="${VELOCITY:-exact}"
ALPHA_MAX="${ALPHA_MAX:-0.8402}"
SELF_COND="${SELF_COND:-false}"      # self-conditioning; must match training

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

MARGS=(
    model=small-sphere-dit
    model.length=${SEQ_LEN:-1024}
    model.init=ngpt
    algo=eflm
    algo.self_conditioning=${SELF_COND}
    algo.renormalize_weights=False
    noise=log-linear-adaptive
    noise.alpha_max=${ALPHA_MAX}
    noise.adaptive_refit_every=50
    noise.adaptive_buffer_size=25600
    noise.adaptive_ema=0.9
    noise.adaptive_uniform_mix=1e-3
    sampler=eflm
    sampler.velocity=${VELOCITY}
    sampler.top_k_velocity=${TOPK_VELOCITY}
    sampler.steps=${STEPS}
    sampler.noise_removal=greedy
)

# (1) validation perplexity
python -u -m main \
    mode=ppl_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
    algo.invert_time_convention=false \
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.results_json_path="${OUTPUT_DIR}/ppl.json" \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/ppl"

# (2) generative perplexity + samples
python -u -m main \
    mode=sample_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
    algo.invert_time_convention=false \
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    eval.results_json_path="${OUTPUT_DIR}/samples_genppl.json" \
    sampler.num_sample_batches=4 \
    sampler.temperature=1.0 \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/sample"
