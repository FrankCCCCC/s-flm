#!/bin/bash
# HBFM + adaptive noise — eval ONE TinyStories checkpoint: valid PPL (ppl_eval) + GenPPL
# (sample_eval). MODEL / SEQ_LEN / EMBED_DIM / FACTOR_DIM / GAUSS_CURV / rate / ALPHA_MAX MUST
# match training (scripts/train/tinystories/hbfm_adaptive.sh); noise=log-linear-adaptive with the
# same knobs so the learned schedule is reconstructed from the checkpoint's alpha_vals buffer.
# The sampler horizon is given in UNIT time (UNIT_T_MAX = |K| t_max), like the
# training proposal (see scripts/train/tinystories/hbfm.sh).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=/abs/path/to/checkpoint.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eval/hbfm}"
DEVICES="${DEVICES:-1}"
EVAL_BS="${EVAL_BS:-16}"
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"
TOPK_VELOCITY="${TOPK_VELOCITY:-1}"       # -1 = posterior-mean drift over the whole vocabulary
NOISE_REMOVAL="${NOISE_REMOVAL:-greedy}"
MODEL="${MODEL:-small-hyperbolic-dit}"
SEQ_LEN="${SEQ_LEN:-256}"
INIT="${INIT:-ngpt}"
EMBED_DIM="${EMBED_DIM:-96}"
HYLA_DIM="${HYLA_DIM:-null}"           # HyLa feature dim; must match training (null = off)
HYLA_RADIUS_CAP="${HYLA_RADIUS_CAP:-null}"   # must match training
HYLA_CONCAT_STATE="${HYLA_CONCAT_STATE:-false}"   # must match training
FACTOR_DIM="${FACTOR_DIM:-3}"
GAUSS_CURV="${GAUSS_CURV:--1.0}"
UNIT_PROPOSAL_RATE="${UNIT_PROPOSAL_RATE:-5}"
READOUT_PRECISION="${READOUT_PRECISION:-float64}"
ALPHA_MAX="${ALPHA_MAX:-null}"            # must match training
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"       # training global batch: the adaptive buffer size (50 * batch) must match the checkpoint
UNIT_T_MAX="${UNIT_T_MAX:-2}"    # (H^3)^32 resolves tokens at u < 1; 180 steps -> dtau 0.011 over the decision phase
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"   # <1.0 only for smoke tests
PROPOSAL_RATE=$(python -c "print(${UNIT_PROPOSAL_RATE} * abs(${GAUSS_CURV}))")
T_MAX=$(python -c "print(${UNIT_T_MAX} / abs(${GAUSS_CURV}))")

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

MARGS=(
    model=${MODEL}
    model.length=${SEQ_LEN}
    model.embed_dim=${EMBED_DIM}
    model.init=${INIT}
    model.hyla_dim=${HYLA_DIM}
    model.hyla_radius_cap=${HYLA_RADIUS_CAP}
    model.hyla_concat_state=${HYLA_CONCAT_STATE}
    algo=hbfm
    algo.prod_factor_dim=${FACTOR_DIM}
    algo.prod_factor_gaussian_curvature=${GAUSS_CURV}
    algo.time_exp_rate=${PROPOSAL_RATE}
    algo.readout_precision=${READOUT_PRECISION}
    noise=log-linear-adaptive
    noise.alpha_max=${ALPHA_MAX}
    noise.adaptive_refit_every=50
    noise.adaptive_buffer_size=25600
    noise.adaptive_ema=0.9
    noise.adaptive_uniform_mix=1e-3
    loader.global_batch_size=${GLOBAL_BATCH}
    sampler=hbfm
    sampler.velocity=${VELOCITY}
    sampler.top_k_velocity=${TOPK_VELOCITY}
    sampler.steps=${STEPS}
    sampler.t_max=${T_MAX}
    sampler.noise_removal=${NOISE_REMOVAL}
)

# (1) validation perplexity (the importance-weighted denoising bound)
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
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
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
    sampler.num_sample_batches=${NUM_SAMPLE_BATCHES:-4} \
    sampler.temperature=1.0 \
    loader.eval_batch_size=${EVAL_BS} \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}/sample"
