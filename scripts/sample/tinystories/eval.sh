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

cd "${REPO_ROOT}"
mkdir -p "${OUTDIR}"

GLOBAL_BS=512
BUF_SIZE=$((50 * GLOBAL_BS))
case "${MODEL_TYPE}" in
  sfm)          MARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False noise=log-linear sampler=sfm) ;;
  langflow)     MARGS=(model=small-sphere-dit model.init=unit_var algo=langflow noise=gumbel sampler=langflow) ;;
  sfm_adaptive) MARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False noise=log-linear-adaptive noise.alpha_max=0.121 noise.adaptive_refit_every=50 "noise.adaptive_buffer_size=${BUF_SIZE}" noise.adaptive_ema=0.9 noise.adaptive_uniform_mix=1e-3 sampler=sfm) ;;
  hflm)         MARGS=(model=small-hyperbolic-dit algo=hflm algo.prior_cov=0.25 algo.rho_max=12 algo.renormalize_weights=False noise=log-linear sampler=hflm) ;;
  *) echo "unknown MODEL_TYPE=${MODEL_TYPE}"; exit 1 ;;
esac

# (1) eval PPL on the held-out validation split -> W&B val/ppl + JSON
python -u -m main \
    mode=ppl_eval \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    strategy=single-device \
    algo.invert_time_convention=false \
    "${MARGS[@]}" \
    eval.checkpoint_path="${CKPT}" \
    eval.strict_loading=false \
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

# (2,3) GenPPL + generated samples -> JSON (gen_ppl_first_chunk_retok, entropy, text)
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
    sampler.num_sample_batches=4 \
    sampler.temperature=1.0 \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +wandb.offline=true \
    hydra.run.dir="${OUTDIR}/sample"

echo "EVAL DONE ${MODEL_TYPE} ${STEP_TAG} -> ${OUTDIR}"
