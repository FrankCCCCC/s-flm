#!/bin/bash
#SBATCH -J embed_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH -t 0:30:00
#SBATCH --partition=desa
#SBATCH --get-user-env
#SBATCH -o /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/embed_%j.out
#SBATCH -e /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/embed_%j.err
#
# CPU-only embedding-distribution analysis (no GPU needed: load to CPU, compute norms, plot).
# Runs on any node with free CPUs even when all GPUs are allocated. Args pass through to the script.
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=""
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm
cd /share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm
python experiments/tinystories/hflm_embed_norms.py "$@"
echo "EMBED_ANALYSIS_DONE"
