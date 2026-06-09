#!/bin/bash
#SBATCH --job-name=owt_eval
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm/outputs/owt_eval_%x.%j.log
# Gen-PPL / entropy frontier for ONE model at ONE checkpoint.
# Sweeps TEMPS x NFES; writes one JSON per point to $OUTDIR. Single-GPU (no NCCL).
# Required env: MODEL_TYPE (sfm|hflm), CKPT, OUTDIR, STEP_TAG.
# Optional env: TEMPS (default "0.8 1.0 1.2"), NFES (default "32").
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
cd /share/thickstun/sychou/workspace/research/s-flm-dev/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
# comma- or space-separated (commas are --export-safe)
TEMPS="${TEMPS:-0.8,1.0,1.2}"; TEMPS="${TEMPS//,/ }"
NFES="${NFES:-32}"; NFES="${NFES//,/ }"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-8}"
EVAL_BS="${EVAL_BS:-8}"
mkdir -p "$OUTDIR"

NOISE_MODE="${NOISE_MODE:-paper}"   # for sfm: paper (adaptive+trunc) | plain (log-linear)
if [ "$MODEL_TYPE" = "sfm" ]; then
  if [ "$NOISE_MODE" = "plain" ]; then
    MODEL_ARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False
                noise=log-linear sampler=sfm sampler.top_k_velocity=1)
  else
    BUF=$((50 * 512))
    MODEL_ARGS=(model=small-sphere-dit model.init=ngpt algo=sfm algo.renormalize_weights=False
                noise=log-linear-adaptive noise.alpha_max=0.121 noise.adaptive_refit_every=50
                "noise.adaptive_buffer_size=${BUF}" noise.adaptive_ema=0.9 noise.adaptive_uniform_mix=1e-3
                sampler=sfm sampler.top_k_velocity=1)
  fi
else
  MODEL_ARGS=(model=small-hyperbolic-dit algo=hflm algo.prior_cov=0.25 algo.rho_max=12
              algo.renormalize_weights=False noise=log-linear sampler=hflm sampler.top_k_velocity=1)
fi

for NFE in $NFES; do
  for T in $TEMPS; do
    RJ="$OUTDIR/${MODEL_TYPE}_${STEP_TAG}_nfe${NFE}_T${T}.json"
    echo "===== ${MODEL_TYPE} step=${STEP_TAG} NFE=${NFE} T=${T} -> $RJ ====="
    $PY -u -m main \
      mode=sample_eval \
      eval.checkpoint_path="$CKPT" \
      eval.strict_loading=false \
      eval.compute_generative_perplexity=True \
      eval.results_json_path="$RJ" \
      data=openwebtext-split data.cache_dir="$PWD/data_cache" \
      algo.invert_time_convention=false \
      "${MODEL_ARGS[@]}" \
      sampler.noise_removal=greedy sampler.velocity=exact \
      sampler.steps="$NFE" sampler.temperature="$T" \
      sampler.num_sample_batches="$NUM_SAMPLE_BATCHES" \
      loader.eval_batch_size="$EVAL_BS" loader.num_workers=4 \
      trainer.num_nodes=1 trainer.devices=1 \
      '+wandb.offline=true' hydra.run.dir="$OUTDIR/hydra_${MODEL_TYPE}_${STEP_TAG}_nfe${NFE}_T${T}" \
      || echo "FAILED ${MODEL_TYPE} NFE=${NFE} T=${T}"
  done
done
echo "FRONTIER EVAL DONE ${MODEL_TYPE} ${STEP_TAG}"
