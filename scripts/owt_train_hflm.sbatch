#!/bin/bash
#SBATCH --job-name=owt_hflm
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm/outputs/owt_hflm.%j.log
# Hyperbolic-FLM on OWT. Matched to S-FLM (768/12, batch 512, AdamW 3e-4, EMA 0.9999)
# but NECESSARY change: noise=log-linear (sphere truncation collapses HFLM; see Sudoku).
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash          # -> Lightning native multi-GPU launcher (not srun)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
cd /share/thickstun/sychou/workspace/research/s-flm-dev/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
DEVICES="${DEVICES:-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
RUN="${RUN:-hflm}"

$PY -u -m main \
    data=openwebtext-split \
    data.cache_dir="$PWD/data_cache" \
    model=small-hyperbolic-dit \
    algo=hflm \
    algo.prior_cov=0.25 \
    algo.rho_max=12 \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    sampler=hflm \
    noise=log-linear \
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
echo "HFLM OWT DONE"
