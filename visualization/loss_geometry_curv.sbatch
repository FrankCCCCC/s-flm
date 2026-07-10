#!/bin/bash
#SBATCH -J loss_geo_curv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH -t 1:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --get-user-env
#SBATCH -o /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/loss_geometry_vis/hflm_curv/loss_geo_curv_%j.out
#SBATCH -e /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/loss_geometry_vis/hflm_curv/loss_geo_curv_%j.err
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm
# HFLM curvature: run from the non-dev1 (claude/curv) tree so `import algo` is
# gaussian_curvature-aware; dev1 `main` would silently use K=-1.
cd /share/thickstun/sychou/workspace/research/s-flm
python visualization/loss_geometry_curv.py "$@"
echo "LOSS_GEOMETRY_DONE"
