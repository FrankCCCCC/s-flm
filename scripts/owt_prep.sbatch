#!/bin/bash
#SBATCH --job-name=owt_prep
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm-dev/s-flm/outputs/owt_prep.%j.log
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /share/thickstun/sychou/workspace/research/s-flm-dev/s-flm
/home/sc3379/anaconda3/envs/sfm/bin/python -u scripts/prep_owt_cache.py
echo "PREP SBATCH DONE"
