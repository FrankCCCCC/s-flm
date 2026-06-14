#!/bin/bash
#SBATCH --job-name=eflm_sudoku
#SBATCH --partition=thickstun,desa
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm/experiments/eflm/logs/%x.%j.log
# EFLM sudoku experiment runner (one job per difficulty; pass DIFFICULTY=easy|medium|hard).
#   Phase 1: train 20k steps  -> scripts/train/sudoku/eflm.sh
#   Phase 2: sudoku_eval @ 180 steps, exact-velocity, greedy, top_k_velocity in {-1, 1}
#            -> scripts/sample/sudoku/eflm.sh
# Env mirrors scripts/owt_train_sfm.sbatch (SLURM_JOB_NAME=bash -> Lightning native launcher).
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PATH=/home/sc3379/anaconda3/envs/sfm/bin:$PATH   # recipes call bare `python`

REPO=/share/thickstun/sychou/workspace/research/s-flm
DIFFICULTY="${DIFFICULTY:-easy}"
cd "${REPO}"
echo "node=$(hostname)  difficulty=${DIFFICULTY}  start=$(date)"

TRAIN_OUT="${REPO}/outputs/sudoku/eflm_${DIFFICULTY}"
echo "##### TRAIN EFLM sudoku ${DIFFICULTY} (20k steps) #####"
DIFFICULTY="${DIFFICULTY}" OUTPUT_DIR="${TRAIN_OUT}" bash scripts/train/sudoku/eflm.sh
P1=$?
echo "##### TRAIN done (exit ${P1}) ##### $(date)"
[ "${P1}" -ne 0 ] && { echo "TRAIN FAILED -> stop"; exit "${P1}"; }

CKPT="${TRAIN_OUT}/checkpoints/last.ckpt"
for TKV in -1 1; do
  EVAL_OUT="${REPO}/eval_runs/sudoku/eflm_${DIFFICULTY}_tkv${TKV}"
  echo "##### SAMPLE EFLM ${DIFFICULTY} exact/greedy/180steps top_k_velocity=${TKV} #####"
  CKPT_PATH="${CKPT}" DIFFICULTY="${DIFFICULTY}" OUTPUT_DIR="${EVAL_OUT}" \
    STEPS=180 VELOCITY=exact TOPK_VELOCITY="${TKV}" \
    bash scripts/sample/sudoku/eflm.sh || echo "SAMPLE tkv=${TKV} FAILED"
  echo "  result(tkv=${TKV}): $(cat ${EVAL_OUT}/results.json 2>/dev/null)"
done
echo "##### EFLM sudoku ${DIFFICULTY} ALL DONE ##### $(date)"
