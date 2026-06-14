#!/bin/bash
#SBATCH --job-name=eflm_smoke
#SBATCH --partition=thickstun
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm/outputs/eflm_smoke.%j.log
# EFLM (Euclidean Flow LM) smoke test on sudoku-easy.
#   Phase 1: train algo.EFLM a few steps (verifies q_xt raw-embedding corruption,
#            nll, backbone, loss run end-to-end without NaN).
#   Phase 2: sudoku_eval with sampler=eflm (verifies samplers.EFLMSampler runs
#            end-to-end incl. prefix conditioning). Non-fatal: Phase 1 is the gate.
set -uo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash          # -> Lightning native launcher (not srun)
export NCCL_P2P_DISABLE=1

REPO=/share/thickstun/sychou/workspace/research/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
CACHE_DIR="${REPO}/data_cache"
OUTPUT_DIR="${REPO}/outputs/sudoku/eflm_smoke"
cd "${REPO}"

echo "=== Phase 1: train EFLM (sudoku-easy, 200 steps) ==="
$PY -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty=easy \
    model=tiny-sphere-dit \
    algo=eflm \
    algo.invert_time_convention=false \
    noise=log-linear \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    strategy=single-device \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_steps=200 \
    trainer.val_check_interval=200 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=100 \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
P1=$?
echo "=== Phase 1 done (exit ${P1}) ==="
[ "${P1}" -ne 0 ] && { echo "Phase 1 FAILED -> stop"; exit "${P1}"; }

CKPT=$(find "${OUTPUT_DIR}" -name "last*.ckpt" 2>/dev/null | head -1)
echo "=== Phase 2: sudoku_eval with EFLM sampler (ckpt=${CKPT}) ==="
if [ -n "${CKPT}" ]; then
  $PY -u -m main \
      mode=sudoku_eval \
      eval.checkpoint_path="${CKPT}" \
      eval.strict_loading=false \
      data=sudoku \
      data.cache_dir="${CACHE_DIR}" \
      data.difficulty=easy \
      model=tiny-sphere-dit \
      algo=eflm \
      algo.invert_time_convention=false \
      noise=log-linear \
      sampler=eflm \
      sampler.noise_removal=greedy \
      sampler.steps=16 \
      sudoku.batch_size=64 \
      loader.eval_batch_size=64 \
      loader.num_workers=4 \
      trainer.num_nodes=1 \
      trainer.devices=1 \
      sudoku.output_dir="${OUTPUT_DIR}/eval" \
      +wandb.offline=True \
      hydra.run.dir="${OUTPUT_DIR}/eval" \
    || echo "Phase 2 (sampler eval) FAILED -- see traceback above"
else
  echo "No checkpoint found under ${OUTPUT_DIR}; skipping Phase 2."
fi
echo "=== EFLM smoke DONE ==="
