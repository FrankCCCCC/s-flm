#!/bin/bash
#SBATCH --job-name=owt_sfm
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm/outputs/owt_sfm.%j.log
# S-FLM on OWT — paper/repo config (scripts/train/owt/sfm.sh): standard DiT 768/12,
# init ngpt, noise=log-linear-adaptive + truncation alpha_max=0.121, batch 512.
# NOISE_MODE=plain switches to plain log-linear (matched control vs HFLM).
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash          # -> Lightning native multi-GPU launcher (not srun)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
cd /share/thickstun/sychou/workspace/research/s-flm-dev/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
DEVICES="${DEVICES:-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
RUN="${RUN:-sfm}"
NOISE_MODE="${NOISE_MODE:-paper}"   # paper = adaptive+trunc ; plain = log-linear

if [ "$NOISE_MODE" = "plain" ]; then
  NOISE_ARGS=(noise=log-linear)
else
  BUF=$((50 * 512))
  NOISE_ARGS=(noise=log-linear-adaptive noise.alpha_max=0.121
              noise.adaptive_refit_every=50 "noise.adaptive_buffer_size=${BUF}"
              noise.adaptive_ema=0.9 noise.adaptive_uniform_mix=1e-3)
fi

$PY -u -m main \
    data=openwebtext-split \
    data.cache_dir="$PWD/data_cache" \
    model=small-sphere-dit \
    model.init=ngpt \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    "${NOISE_ARGS[@]}" \
    loader.global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.val_check_interval=50000 \
    trainer.limit_val_batches=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    wandb.project=hflm-owt '+wandb.offline=true' \
    hydra.run.dir="$PWD/outputs/owt/${RUN}"
echo "SFM OWT DONE"
