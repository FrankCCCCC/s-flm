#!/bin/bash
#SBATCH --partition=desa
#SBATCH --exclude=desa-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/experiments/hbfm_ce_tinystories_256/logs/%x_%j.log
# horizon.py on one desa GPU:
#   CELL=<run dir> CKPT=keep-6000.ckpt sbatch --job-name=hbfm_ce_horizon_<cell>_<step> horizon.sbatch [extra args]
set -e
H=/share/desa/nfs02/sc3379/sfm_home
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0
export PATH=$H/envs/sfm/bin:$PATH HF_HOME=$H/hf_cache XDG_CACHE_HOME=$H/cache MPLCONFIGDIR=$H/cache/matplotlib
cd /share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -u experiments/hbfm_ce_tinystories_256/horizon.py "${CELL:?}" --ckpt "${CKPT:-last.ckpt}" "$@"
