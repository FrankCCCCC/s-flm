#!/bin/bash
#SBATCH -J hflm_embed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH -t 0:20:00
#SBATCH --partition=desa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --get-user-env
#SBATCH -o /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/embed_%j.out
#SBATCH -e /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/embed_%j.err
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm
cd /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm
# args (sources + --out ...) are passed through from the sbatch command line
python experiments/tinystories/hflm_embed_norms.py "$@"
echo "EMBED_ANALYSIS_DONE"
