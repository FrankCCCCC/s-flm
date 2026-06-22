#!/bin/bash
#SBATCH --job-name=ts_babysit
#SBATCH --partition=thickstun,desa
#SBATCH --exclude=desa-compute-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --output=/share/thickstun/sychou/workspace/research/s-flm/experiments/babysit_%j.log
# Drives the 4 TinyStories experiments to completion: every 30 min, resubmit any
# preempted/dead cells (idempotent sweeps auto-resume from last.ckpt) and refresh
# RESULTS.md reports. CPU-only (no GPU) so it never blocks training. Stop with scancel.
echo "babysitter start $(date) on $(hostname)"
for i in $(seq 1 480); do   # 480 x 30 min ~= 10 days
  bash /share/thickstun/sychou/workspace/research/s-flm/experiments/maintain.sh
  sleep 1800
done
echo "babysitter end $(date)"
