#!/bin/bash
#SBATCH -J sfm_sudoku_km1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH -t 4:00:00
#SBATCH --partition=desa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sc3379@cornell.edu
#SBATCH -o /share/thickstun/sychou/workspace/research/s-flm/slurm_logs/%x_%j.out
#SBATCH -e /share/thickstun/sychou/workspace/research/s-flm/slurm_logs/%x_%j.err
#
# S-FLM Sudoku eval at top_k_velocity=-1 (full-vocab "average" velocity) on the SAME
# checkpoints as the top-1 run, so the -1 vs 1 contrast is on identical models.
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export SLURM_JOB_NAME=bash
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm

ROOT="/share/thickstun/sychou/workspace/research/s-flm"
cd "${ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

declare -A SCRIPT=( [sfm]=sfm.sh [sfm_truncated]=sfm_truncated.sh [sfm_truncated_adaptive]=sfm_truncated_adaptive.sh )

for v in sfm sfm_truncated sfm_truncated_adaptive; do
  for d in easy medium hard; do
    CK="${ROOT}/outputs/sudoku/${v}_${d}/checkpoints/last.ckpt"
    OUT="${ROOT}/eval_runs/sudoku/${v}_${d}_km1"
    echo "================ ${v} ${d}  top_k_velocity=-1 ================"
    if [ -f "${CK}" ]; then
      CKPT_PATH="${CK}" DIFFICULTY="${d}" OUTPUT_DIR="${OUT}" TOPK_VELOCITY=-1 STEPS=180 \
        bash "${ROOT}/scripts/sample/sudoku/${SCRIPT[$v]}" \
        && echo "OK ${v} ${d}" || echo "FAILED ${v} ${d}"
    else
      echo "MISSING ${CK}"
    fi
  done
done
echo "ALL_SFM_KM1_DONE"
