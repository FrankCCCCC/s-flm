#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --mem=60gb
#SBATCH -t 2:00:00
#SBATCH --partition=desa
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#
# Sudoku accuracy eval (mode=sudoku_eval). Uses ABSOLUTE checkpoint/output paths
# (Hydra chdir's into hydra.run.dir, which breaks relative paths).
#   sbatch -J hbfm_eval_d64 --export=ALL,ALGO=hbfm,D=64,CKPT=/abs/.../106-20000.ckpt scripts/slurm/sudoku_eval.sbatch
#   sbatch -J sfm_eval_d64  --export=ALL,ALGO=sfm,D=64,CKPT=/abs/.../106-20000.ckpt  scripts/slurm/sudoku_eval.sbatch

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
source /home/sc3379/anaconda3/etc/profile.d/conda.sh
conda activate sfm

ALGO="${ALGO:-hbfm}"
D="${D:-64}"
SEED="${SEED:-1}"
DIFFICULTY="${DIFFICULTY:-easy}"
STEPS="${STEPS:-180}"
REPO_ROOT="/share/thickstun/sychou/workspace/research/s-flm"
cd "${REPO_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "${D}" -le 2 ]; then N_HEADS=1; else N_HEADS=8; fi
CKPT="${CKPT:-${REPO_ROOT}/outputs/sudoku/${ALGO}_d${D}_${DIFFICULTY}_seed${SEED}/checkpoints/last.ckpt}"
OUT="${REPO_ROOT}/eval_runs/sudoku/${ALGO}_d${D}_${DIFFICULTY}"

if [ "${ALGO}" = "hbfm" ]; then SAMPLER=hbfm; EXTRA=(); else SAMPLER=sfm; EXTRA=(algo.invert_time_convention=false); fi

python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT}" \
    eval.strict_loading=false \
    data=sudoku data.cache_dir="${REPO_ROOT}/data_cache" data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit model.hidden_size="${D}" model.n_heads="${N_HEADS}" \
    algo="${ALGO}" "${EXTRA[@]}" noise=log-linear \
    sampler="${SAMPLER}" sampler.noise_removal=greedy sampler.velocity=exact sampler.steps="${STEPS}" \
    sudoku.batch_size=64 loader.eval_batch_size=64 loader.num_workers=4 \
    trainer.devices=1 \
    sudoku.output_dir="${OUT}" \
    +wandb.offline=True \
    hydra.run.dir="${OUT}"
