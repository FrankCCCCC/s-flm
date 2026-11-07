#!/bin/bash
#SBATCH --partition=desa
#SBATCH --exclude=desa-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/experiments/hbfm_elbo_tinystories_256/logs/%x_%j.log
# nelbo_variance.py on one desa GPU:
#   CELL=<run dir> CKPT=keep-20000.ckpt sbatch --job-name=hbfm_elbo_var_<cell>_<step> nelbo_variance.sbatch [extra args]
set -e
H=/share/desa/nfs02/sc3379/sfm_home
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0
export PATH=$H/envs/sfm/bin:$PATH HF_HOME=$H/hf_cache XDG_CACHE_HOME=$H/cache MPLCONFIGDIR=$H/cache/matplotlib
cd /share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -u experiments/hbfm_elbo_tinystories_256/nelbo_variance.py "${CELL:?}" --ckpt "${CKPT:-last.ckpt}" "$@"
