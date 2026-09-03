#!/bin/bash
# SimpFLM — E-FLM with the word-embedding matrix replaced by the DIAGONAL
# R * I_V: every "embedding" is a one-hot simplex vertex, so the Gaussian
# Euclidean flow x_t = (1 - b_t) e + b_t z runs in the one-hot (logit) space
# R^V. The small-flm (flm-dit) backbone projects that [B, L, V] blend down to
# the model width. R = RHO (algo.rho_min = rho_max = RHO) is E-FLM's own radial
# rescale applied to the diagonal (rho_min and rho_max must be EQUAL), and it
# is the only geometric knob.#
# Truncated LOG-LINEAR schedule: ALPHA_MAX is the Eq.-17 decode point. For the
# ORTHOGONAL vertices of the diagonal the impostor score is literally Gaussian,
# so alpha_star_euclidean(V=50257, embed_norm=R) applies verbatim -- 0.840 at
# R = 1. null = no truncation.#
# Plus the ADAPTIVE time remap: a spline reweighting of t onto where |dL/dt| is
# largest (noise_schedules.AdaptiveSchedule). It sits on top of the truncated
# alpha_t, so the decode point is unchanged -- only the density of visited noise
# levels moves. Requires the MDLM time convention (invert_time_convention=false),
# this algo's default. The fitted remap ships in the checkpoint, so train and
# eval must use the same `noise=` config.
#
# Eval ONE TinyStories checkpoint: valid PPL (ppl_eval) + GenPPL
# (sample_eval). RHO and the truncation knob must match training.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=/abs/path/to/checkpoint.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eval/simpflm_truncated_adaptive}"
DEVICES="${DEVICES:-1}"
EVAL_BS="${EVAL_BS:-16}"
STEPS="${STEPS:-180}"
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"
VELOCITY="${VELOCITY:-exact}"
RHO="${RHO:-1.0}"                   # R: the simplex-sphere radius; must match training
ALPHA_MAX="${ALPHA_MAX:-0.840}"       # alpha_star_euclidean(50257, embed_norm=RHO); must match training
ETA="${ETA:-0.0}"                   # SDE noise scale; 0 = deterministic ODE
GT_METHOD="${GT_METHOD:-linear}"    # SDE g(t): const / sqrt / linear / quad
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
RUN_PPL_EVAL="${RUN_PPL_EVAL:-true}" # false: GenPPL pass only

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

MARGS=(
    model=small-flm
    model.length=${SEQ_LEN:-1024}
    algo=simpflm
    algo.rho_min=${RHO}
    algo.rho_max=${RHO}
    noise=log-linear-adaptive
    noise.alpha_max=${ALPHA_MAX}
    sampler=simpflm
    sampler.velocity=${VELOCITY}
    sampler.top_k_velocity=${TOPK_VELOCITY}
    sampler.steps=${STEPS}
    sampler.eta=${ETA}
    sampler.gt_method=${GT_METHOD}
    sampler.noise_removal=greedy
)

# (1) validation perplexity
if [ "${RUN_PPL_EVAL}" = "true" ]; then
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
fi

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
    sampler.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampler.temperature=${TEMPERATURE} \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/sample"
