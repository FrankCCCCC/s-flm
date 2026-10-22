#!/bin/bash
#SBATCH --job-name=hbfm_adats_lambda_local
#SBATCH --constraint=avx
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# lambda per keep-{step}.ckpt of the cells trained on sc3379 (checkpoints on NFS), on CPU (any AVX node: the env's torch dies with an illegal instruction without it).
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PATH=/home/sc3379/anaconda3/envs/sfm/bin:$PATH
cd /share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm
hostname
python experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py \
  $(ls -d outputs/hbfm_rlf_ada_curv_tinystories_256/cb-4_*/checkpoints | xargs -n1 dirname) 2>&1 | grep -v -i warn
