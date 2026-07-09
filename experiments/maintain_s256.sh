#!/bin/bash
# Maintenance pass for the 4 seq-256 TinyStories experiments (slides jun25_2026).
# Idempotent: (1) re-runs each sweep -> resubmits only cells that are neither done
# (eval/ppl.json present) nor currently queued, which auto-resume from last.ckpt;
# (2) regenerates the per-experiment RESULTS.md reports.
# Submission order = naive first, hflm sweep last (headline baselines land first).
# Logs to experiments/maintain_s256.log.
set -uo pipefail
export PATH=/usr/local/slurm/current/bin:/home/sc3379/anaconda3/envs/sfm/bin:/usr/bin:/bin:${PATH:-}
REPO=/share/thickstun/sychou/workspace/research/s-flm
PY=/home/sc3379/anaconda3/envs/sfm/bin/python
cd "${REPO}"
EXPS="naive_ar_tinystories_s256 naive_geo_tinystories_s256 adv_geo_tinystories_s256 hflm_sweep_tinystories_s256"
{
  echo "===== maintain_s256 $(date) ====="
  for e in ${EXPS}; do
    echo "-- sweep ${e} --"
    ${PY} "experiments/${e}/sweep.py" 2>&1 | grep -E "submitted|skipped|cells" | tail -3
    ${PY} experiments/report.py "${e}" 2>&1 | tail -1
  done
  # Prioritize adv_geo (fair-comparison baseline) over the H-FLM sweep: deprioritize
  # pending H-FLM so freed GPUs go to adv first; keep adv at nice 0. Idempotent.
  # adv_geo is top priority, UNLESS a short areval* probe is queued (then probe runs first).
  if squeue -u sc3379 -h -o "%j" | grep -q '^areval'; then ADVNICE=5000; else ADVNICE=0; fi
  for id in $(squeue -u sc3379 -h -t PD -o "%i %j" | awk '$2 ~ /^hflm256_/ {print $1}'); do
    scontrol update jobid=$id nice=5000 2>/dev/null
  done
  for id in $(squeue -u sc3379 -h -t PD -o "%i %j" | awk '$2 ~ /^adv256_/ {print $1}'); do
    scontrol update jobid=$id nice=$ADVNICE 2>/dev/null
  done
  done_cells=$(find outputs/{naive_ar,naive_geo,adv_geo,hflm_sweep}_tinystories_s256 -name ppl.json 2>/dev/null | wc -l)
  echo "progress: ${done_cells}/162 cells have eval/ppl.json ; queue=$(squeue -u sc3379 -h | wc -l)"
} >> experiments/maintain_s256.log 2>&1
