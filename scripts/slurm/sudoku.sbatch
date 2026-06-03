#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --mem=60gb
#SBATCH -t 24:00:00
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#
# Parameterized Sudoku training launcher for the desa partition.
# Submit with --export, e.g.:
#   sbatch -J hbfm_smoke_d2  --export=ALL,ALGO=hbfm,D=2,SEED=1,MAX_STEPS=300   scripts/slurm/sudoku.sbatch
#   sbatch -J hbfm_d64_s1    --export=ALL,ALGO=hbfm,D=64,SEED=1,MAX_STEPS=20000 scripts/slurm/sudoku.sbatch
#   sbatch -J sfm_d64_s1     --export=ALL,ALGO=sfm,D=64,SEED=1,MAX_STEPS=20000  scripts/slurm/sudoku.sbatch

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm

ALGO="${ALGO:-hbfm}"
D="${D:-64}"
SEED="${SEED:-1}"
DIFFICULTY="${DIFFICULTY:-easy}"
MAX_STEPS="${MAX_STEPS:-20000}"

REPO_ROOT="/share/thickstun/sychou/workspace/research/s-flm"
cd "${REPO_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "${ALGO}" = "hbfm" ]; then
  SEED="${SEED}" MAX_STEPS="${MAX_STEPS}" D="${D}" DIFFICULTY="${DIFFICULTY}" \
    bash scripts/train/sudoku/hbfm.sh
elif [ "${ALGO}" = "sfm" ]; then
  if [ "${D}" -le 2 ]; then N_HEADS=1; else N_HEADS=8; fi
  CKPT_EVERY=$(( MAX_STEPS < 5000 ? MAX_STEPS : 5000 ))
  python -u -m main \
    seed="${SEED}" \
    data=sudoku data.cache_dir="${REPO_ROOT}/data_cache" data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit model.hidden_size="${D}" model.n_heads="${N_HEADS}" \
    algo=sfm algo.invert_time_convention=false noise=log-linear \
    loader.global_batch_size=256 loader.batch_size=256 loader.eval_batch_size=256 \
    loader.num_workers=8 eval.generate_samples=False \
    trainer.devices=1 trainer.val_check_interval="${MAX_STEPS}" \
    trainer.limit_val_batches=0 trainer.max_steps="${MAX_STEPS}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
    hydra.run.dir="${REPO_ROOT}/outputs/sudoku/sfm_d${D}_${DIFFICULTY}_seed${SEED}"
else
  echo "Unknown ALGO=${ALGO} (want hbfm|sfm)"; exit 1
fi
