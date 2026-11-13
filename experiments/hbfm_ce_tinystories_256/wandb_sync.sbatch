#!/bin/bash
#SBATCH --job-name=hbfm_ce_ts_wandb_sync
#SBATCH --partition=desa
#SBATCH --nodelist=desa-compute-01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/experiments/hbfm_ce_tinystories_256/logs/%x_%j.log
# Replays the A100 cells' offline W&B runs into the project every 10 minutes, until ch2263 has no job of this sweep left.
# No GPU. Extra args go to sweep.py (--smoke).
H=/share/desa/nfs02/sc3379/sfm_home
export CUDA_VISIBLE_DEVICES= PATH=$H/envs/sfm/bin:$PATH XDG_CACHE_HOME=$H/cache
cd /share/desa/nfs02/sc3379/workspace/research/s-flm-dev/s-flm/experiments/hbfm_ce_tinystories_256
left() {
  ssh -o BatchMode=yes -i /home/sc3379/.ssh/unicorn_internal ch2263@unicorn-login-02.coecis.cornell.edu \
    'bash -lc "squeue -h -u ch2263 -o %j"' | grep -c '^hbfm_ce_ts_'
}
while true; do
  n=$(left) || n=1   # grep -c exits 1 on a count of 0, so read the count, not the status
  python sweep.py --site a100 --pull "$@"
  python sweep.py --wandb-sync "$@"
  [ "${n:-1}" = "0" ] && break
  sleep 600
done
