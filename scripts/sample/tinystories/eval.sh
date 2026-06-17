#!/bin/bash
# Single-GPU eval of one TinyStories checkpoint (eval is forward-only, so 1 GPU is
# enough and avoids the multi-GPU DDP path). Produces the three requested metrics:
#   (1) eval PPL  -> mode=ppl_eval     -> W&B val/ppl + ppl.json
#   (2) GenPPL    -> mode=sample_eval  -> samples_genppl.json (gen_ppl_first_chunk_retok)
#   (3) samples   -> mode=sample_eval  -> samples_genppl.json (text)
# Env: MODEL_TYPE=sfm|sfm_adaptive|hflm  CKPT=/abs/path/to.ckpt  [STEP_TAG]
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
MODEL_TYPE="${MODEL_TYPE:-sfm}"
CKPT="${CKPT:?set CKPT=/abs/path/to/checkpoint.ckpt}"
STEP_TAG="${STEP_TAG:-eval}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/outputs/tinystories/eval/${MODEL_TYPE}_${STEP_TAG}}"

# --- Decode-policy knobs (overridable for the fair-comparison sweep) ---
# All three models must decode IDENTICALLY within a sweep cell; only the geometry
# (predictor) differs. STEPS=NFE; NOISE_REMOVAL final-step rule; TOPKV=top_k_velocity
# (sphere honours it, hyperbolic is structurally top-1 so it is a no-op there).
# Default pins STEPS=1024 for ALL models (this OVERRIDES hflm's native 180; sfm's native is
# already 1024 -> matched NFE). Only noise_removal/top_k_velocity are left empty and are
# thus inherited per-model from sampler=<model> when not set.
STEPS="${STEPS:-1024}"
NOISE_REMOVAL="${NOISE_REMOVAL:-}"           # ''=use sampler default; else ancestral|greedy
TOPKV="${TOPKV:-}"                            # ''=use sampler default; else 1|-1
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"
DO_PPL="${DO_PPL:-1}"                         # 1=run held-out val/ppl (NFE-independent)
DO_SAMPLE="${DO_SAMPLE:-1}"                   # 1=run GenPPL + samples

cd "${REPO_ROOT}"
mkdir -p "${OUTDIR}"

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))
case "${MODEL_TYPE}" in
  sfm)          MARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False noise=log-linear sampler=sfm) ;;
  sfm_adaptive) MARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False noise=log-linear-adaptive noise.alpha_max=0.121 noise.adaptive_refit_every=50 "noise.adaptive_buffer_size=${BUF_SIZE}" noise.adaptive_ema=0.9 noise.adaptive_uniform_mix=1e-3 sampler=sfm) ;;
  hflm)         MARGS=(model=small-hyperbolic-dit algo=hflm algo.prior_cov=0.25 algo.rho_max=12 algo.renormalize_weights=False noise=log-linear sampler=hflm) ;;
  *) echo "unknown MODEL_TYPE=${MODEL_TYPE}"; exit 1 ;;
esac

# (1) eval PPL on the held-out validation split -> W&B val/ppl + JSON
# NFE-independent (no sampling) -> run once per (model, checkpoint); skip in sweep cells.
if [ "${DO_PPL}" = "1" ]; then
python -u -m main \
    mode=ppl_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
    algo.invert_time_convention=false \
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT}" \
    eval.strict_loading=false \
    eval.generate_samples=False \
    eval.results_json_path="${OUTDIR}/ppl.json" \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    wandb.project=tinystories-flm \
    wandb.group=geometry-vs-tricks \
    +wandb.name="${MODEL_TYPE}_${STEP_TAG}_ppl" \
    +wandb.offline=true \
    hydra.run.dir="${OUTDIR}/ppl"
fi

# (2,3) GenPPL + generated samples -> JSON (gen_ppl_first_chunk_retok, entropy, text)
# Pin EVERY decode knob so "identical decode across models" is provable, not inherited.
SAMPLE_OVERRIDES=(
    "sampler.steps=${STEPS}"
    "sampler.num_sample_batches=${NUM_SAMPLE_BATCHES}"
    sampler.velocity=exact
    sampler.temperature=1.0
    sampler.p_nucleus=1.0
    sampler.top_k=-1
)
if [ -n "${NOISE_REMOVAL}" ]; then
    SAMPLE_OVERRIDES+=("sampler.noise_removal=${NOISE_REMOVAL}")
fi
if [ -n "${TOPKV}" ]; then
    SAMPLE_OVERRIDES+=("sampler.top_k_velocity=${TOPKV}")
fi

if [ "${DO_SAMPLE}" = "1" ]; then
python -u -m main \
    mode=sample_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
    algo.invert_time_convention=false \
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    eval.results_json_path="${OUTDIR}/samples_genppl.json" \
    "${SAMPLE_OVERRIDES[@]}" \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +wandb.offline=true \
    hydra.run.dir="${OUTDIR}/sample"
fi

echo "EVAL DONE ${MODEL_TYPE} ${STEP_TAG} steps=${STEPS} nr=${NOISE_REMOVAL:-native} kv=${TOPKV:-native} -> ${OUTDIR}"
