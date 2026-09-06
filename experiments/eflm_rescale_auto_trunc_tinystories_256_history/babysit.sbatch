#!/bin/bash
#SBATCH --job-name=eflmrath_babysit
#SBATCH --partition=thickstun,desa
#SBATCH --exclude=desa-compute-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/share/desa/nfs02/sc3379/workspace/research/s-flm/experiments/eflm_rescale_auto_trunc_tinystories_256_history/babysit_%j.log
# Drives the sweep to completion without a live session: every 30 min, `sweep.py
# --auto` resubmits cells that are neither done nor queued (they resume from
# last.ckpt) and opens the next stage once the current one is complete, then
# analyze.py refreshes the tables. CPU-only, so it never holds a GPU.
# Stop with scancel once RESULTS_tables.md shows all stages done.
set -uo pipefail
export PATH=/usr/local/slurm/current/bin:/home/sc3379/anaconda3/envs/sfm/bin:${PATH:-}
EXP=/share/desa/nfs02/sc3379/workspace/research/s-flm/experiments/eflm_rescale_auto_trunc_tinystories_256_history
for i in $(seq 1 336); do   # 336 x 30 min = 7 days
  echo "===== $(date) pass ${i} ====="
  python "${EXP}/sweep.py" --auto
  python "${EXP}/analyze.py" --md "${EXP}/RESULTS_tables.md" >/dev/null
  sleep 1800
done
