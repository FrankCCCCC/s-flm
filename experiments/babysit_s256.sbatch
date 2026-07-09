#!/bin/bash
#SBATCH --job-name=ts256_babysit
#SBATCH --partition=thickstun,desa
#SBATCH --exclude=desa-compute-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm/experiments/babysit_s256_%j.log
# Drives the 4 seq-256 TinyStories experiments to completion: every 30 min, resubmit
# any preempted/dead cells (idempotent sweeps auto-resume from last.ckpt) and refresh
# RESULTS.md reports. CPU-only (no GPU) so it never blocks training. Stop with scancel.
echo "babysitter_s256 start $(date) on $(hostname)"
for i in $(seq 1 672); do   # 672 x 30 min ~= 14 days
  bash /share/thickstun/sychou/workspace/research/s-flm/experiments/maintain_s256.sh
  sleep 1800
done
echo "babysitter_s256 end $(date)"
