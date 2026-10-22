#!/bin/bash
# MDLM — eval ONE TinyStories checkpoint: valid PPL (ppl_eval) + GenPPL (sample_eval).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=/abs/path/to/checkpoint.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eval/mdlm}"
DEVICES="${DEVICES:-1}"
EVAL_BS="${EVAL_BS:-16}"
STEPS="${STEPS:-180}"   # setup.md: MDLM 180 steps, matched to DUO/FLM/geo-flow NFE
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
RUN_PPL_EVAL="${RUN_PPL_EVAL:-true}"   # false: GenPPL pass only
LOSS_TYPE="${LOSS_TYPE:-elbo}"         # objective of the ppl_eval pass (algo.loss_type): elbo = MDLM's NELBO
PPL_JSON="${PPL_JSON:-${OUTPUT_DIR}/ppl.json}"   # where the ppl_eval metrics go
SEED="${SEED:-1}"                      # seed of the ppl_eval pass (the (t, noise) draws)
RUN_SAMPLE_EVAL="${RUN_SAMPLE_EVAL:-true}"   # false: ppl_eval pass only

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

MARGS=(
    model=small
    model.length=${SEQ_LEN:-1024}
    algo=mdlm
    algo.loss_type=${LOSS_TYPE}
    sampler=ancestral
    sampler.steps=${STEPS}
)

# (1) validation perplexity
if [ "${RUN_PPL_EVAL}" = "true" ]; then
    python -u -m main \
        mode=ppl_eval \
        data=tinystories \
        data.cache_dir="${CACHE_DIR}" \
        strategy=single-device \
        "${MARGS[@]}" \
        eval.checkpoint_path="${CKPT_PATH}" \
        eval.strict_loading=false \
        eval.results_json_path="${PPL_JSON}" \
        seed=${SEED} \
        eval.generate_samples=False \
        loader.eval_batch_size=${EVAL_BS} \
        loader.num_workers=4 \
        trainer.num_nodes=1 \
        trainer.devices="${DEVICES}" \
        +wandb.offline=true \
        hydra.run.dir="${OUTPUT_DIR}/ppl"
fi

# (2) generative perplexity + samples
if [ "${RUN_SAMPLE_EVAL}" = "true" ]; then
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
    sampler.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampler.temperature=${TEMPERATURE} \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/sample"
fi
