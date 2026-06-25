#!/bin/bash
# AR — eval ONE TinyStories checkpoint: valid PPL (ppl_eval) + GenPPL (sample_eval).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=/abs/path/to/checkpoint.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eval/ar}"
DEVICES="${DEVICES:-1}"
EVAL_BS="${EVAL_BS:-16}"
GREEDY="${GREEDY:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
P_NUCLEUS="${P_NUCLEUS:-1.0}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

MARGS=(
    model=small
    model.length=${SEQ_LEN:-1024}
    algo=ar
    sampler=ar
    sampler.greedy=${GREEDY}
    sampler.p_nucleus=${P_NUCLEUS}
)

# (1) validation perplexity
python -u -m main \
    mode=ppl_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
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
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    eval.results_json_path="${OUTPUT_DIR}/samples_genppl.json" \
    sampler.num_sample_batches=4 \
    sampler.temperature=${TEMPERATURE} \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/sample"
