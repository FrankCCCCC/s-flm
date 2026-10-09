#!/bin/bash
#SBATCH --job-name=hbfm_adats_lambda
#SBATCH --partition=nlplarge-claire-highpri
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# lambda per keep-{step}.ckpt of every A100 cell (the checkpoints are on the node's /scratch), CPU only:
# no GPU is allocated and none is made visible, so a training GPU is never touched.
export CUDA_VISIBLE_DEVICES= TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
export PATH=/home/ch2263/miniconda3/envs/sfm/bin:$PATH
cd /home/ch2263/syc_workspace/s-flm-adacurv
hostname
python experiments/hbfm_rlf_ada_curv_tinystories_256/curvature_ts.py \
  outputs/hbfm_rlf_ada_curv_tinystories_256/cb-4_*/ 2>&1 | grep -v -i warn
