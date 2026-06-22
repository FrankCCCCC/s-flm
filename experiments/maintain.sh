#!/bin/bash
# Maintenance pass for the 4 TinyStories experiments (slides jun25_2026).
# Idempotent: (1) re-runs each sweep -> resubmits only cells that are neither done
# (eval/ppl.json present) nor currently queued, which auto-resume from last.ckpt;
# (2) regenerates the per-experiment RESULTS.md reports.
# Intended to run on a ~30-min cron until all 161 cells complete (then it's a no-op
# beyond refreshing reports). Logs to experiments/maintain.log.
set -uo pipefail
export PATH=/usr/local/slurm/current/bin:/home/sc3379/anaconda3/envs/sfm/bin:/usr/bin:/bin:${PATH:-}
REPO=/share/thickstun/sychou/workspace/research/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
cd "${REPO}"
EXPS="naive_ar_tinystories naive_geo_tinystories adv_geo_tinystories hflm_sweep_tinystories"
{
  echo "===== maintain $(date) ====="
  for e in ${EXPS}; do
    echo "-- sweep ${e} --"
    ${PY} "experiments/${e}/sweep.py" 2>&1 | grep -E "submitted|skipped" | tail -3
    ${PY} experiments/report.py "${e}" 2>&1 | tail -1
  done
  done_cells=$(find outputs/{naive_ar,naive_geo,adv_geo,hflm_sweep}_tinystories -name ppl.json 2>/dev/null | wc -l)
  echo "progress: ${done_cells}/161 cells have eval/ppl.json ; queue=$(squeue -u sc3379 -h | wc -l)"
} >> experiments/maintain.log 2>&1
