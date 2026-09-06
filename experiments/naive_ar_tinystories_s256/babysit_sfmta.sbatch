#!/bin/bash
#SBATCH --job-name=nar256_sfmta_babysit
#SBATCH --partition=thickstun,desa
#SBATCH --exclude=desa-compute-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
# Drives the sfmta (S-FLM + trunc + ada) arm of this experiment to completion: every 20
# min, submit the frontier cells whose pretrained checkpoint has landed (frontier_sweep
# skips seeds with no last.ckpt, done cells and queued jobs) and refresh the figure/csv
# from whatever is finished. Exits once all 360 sfmta cells exist. CPU-only, so it never
# holds a GPU. Stop with scancel.
set -uo pipefail
export PATH=/usr/local/slurm/current/bin:/home/sc3379/anaconda3/envs/sfm/bin:${PATH:-}
REPO=/share/desa/nfs02/sc3379/workspace/research/s-flm-dev1/s-flm
EXP=${REPO}/experiments/naive_ar_tinystories_s256
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
TOTAL=360   # 3 seeds x 8 NFE x 15 T
cd "${REPO}"
echo "sfmta babysitter start $(date) on $(hostname)"
for i in $(seq 1 288); do   # 288 x 20 min = 4 days
  {
    echo "===== sfmta maintain $(date) ====="
    ${PY} "${EXP}/frontier_sweep.py" --methods sfmta 2>&1 | grep -E "submitted|skip" | tail -5
    ${PY} visualization/genppl_entropy_frontier_line.py \
        --out "${EXP}/figures/genppl_entropy_frontier_line.png" \
        --csv "${EXP}/frontier_cells.csv" > "${EXP}/frontier_tables.md" 2>&1
    head -1 "${EXP}/frontier_tables.md"
  } >> "${EXP}/babysit_sfmta.log" 2>&1
  n=$(ls -d ${REPO}/outputs/naive_ar_tinystories_s256/m-sfmta_lr-1e-3_sd-*/frontier/nfe-*_t-*/samples_genppl.json 2>/dev/null | wc -l)
  echo "[$(date)] sfmta cells: ${n}/${TOTAL}"
  [ "${n}" -ge "${TOTAL}" ] && { echo "all sfmta cells done $(date)"; break; }
  sleep 1200
done
echo "sfmta babysitter end $(date)"
