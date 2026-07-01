#!/bin/bash
# Wait for sfm + sfm_truncated_adaptive TRAINING to finish, then:
#   (1) submit 30k held-out val/ppl for the two sphere models (NFE-independent; hflm done)
#   (2) regenerate the sweep manifest (last.ckpt now == 30k) and submit the 60-cell
#       decode-policy GenPPL sweep array (steps x noise_removal x top_k_velocity).
set -uo pipefail
ROOT="/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm"
LOG="${ROOT}/experiments/tinystories/final_eval_monitor.log"
cd "${ROOT}"
echo "[$(date '+%F %T')] monitor v2 start; waiting for ts-sfm + ts-sfm-trunc to finish..." >> "${LOG}"
for i in $(seq 1 192); do   # up to ~16h at 5min
  n=$(squeue -h -u sc3379 -n ts-sfm,ts-sfm-trunc 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[$(date '+%F %T')] training done (poll $i)." >> "${LOG}"
    # (1) 30k held-out val/ppl for the two sphere models (DO_SAMPLE=0 -> no generation)
    for mt in sfm sfm_adaptive; do
      d=$([ "$mt" = "sfm" ] && echo sfm || echo sfm_truncated_adaptive)
      # explicit 30000-step ckpt (not rolling last.ckpt) -> guarantees the 30k label is honest
      CK=$(ls "${ROOT}/outputs/tinystories/${d}/checkpoints/"*-30000.ckpt 2>/dev/null | head -1)
      if [ -n "${CK}" ] && [ -f "${CK}" ]; then
        J=$(sbatch --parsable --constraint=gpu-high --export=ALL,MODEL_TYPE=${mt},CKPT="${CK}",STEP_TAG=30k,DO_PPL=1,DO_SAMPLE=0 experiments/tinystories/eval.sub)
        echo "[$(date '+%F %T')] submitted ${mt} 30k val/ppl ($(basename "${CK}")) -> $J" >> "${LOG}"
      else
        echo "[$(date '+%F %T')] MISSING *-30000.ckpt for ${mt} in ${d}/checkpoints" >> "${LOG}"
      fi
    done
    # (2) regenerate manifest (resolves last.ckpt -> 30k) + submit GenPPL sweep array
    if bash experiments/tinystories/sweep_gen_manifest.sh >> "${LOG}" 2>&1; then
      A=$(sbatch --parsable experiments/tinystories/sweep_array.sub)
      echo "[$(date '+%F %T')] submitted sweep array -> $A" >> "${LOG}"
      echo "RESULT=SWEEP_SUBMITTED ${A}" >> "${LOG}"
    else
      echo "[$(date '+%F %T')] manifest gen FAILED; sweep NOT submitted" >> "${LOG}"
      echo "RESULT=MANIFEST_FAIL" >> "${LOG}"
    fi
    exit 0
  fi
  sleep 300
done
echo "[$(date '+%F %T')] TIMEOUT waiting for training" >> "${LOG}"; echo "RESULT=TIMEOUT" >> "${LOG}"; exit 2
