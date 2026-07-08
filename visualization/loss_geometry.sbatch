#!/bin/bash
#SBATCH -J loss_geo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH -t 1:00:00
#SBATCH --partition=desa,thickstun
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --get-user-env
#SBATCH -o /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/loss_geometry_vis/loss_geo_%j.out
#SBATCH -e /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/loss_geometry_vis/loss_geo_%j.err
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm
cd /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm
# args (--mode ... --out ...) are passed through from the sbatch command line
python visualization/loss_geometry.py "$@"
echo "LOSS_GEOMETRY_DONE"
