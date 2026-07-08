#!/bin/bash
# One sync+prune cycle for the 3-site hflm_curv_init_lr_sudoku sweep.
#   1. rsync finished results both ways (unicorn <-> ARC shared home)
#   2. on every cluster, scancel PENDING hcil_* jobs whose cell already has results
#      (running jobs are left alone; the job-body guard no-ops stale starts)
#   3. print a one-line status
# Run from unicorn (sc3379). Safe to re-run any time.
set -uo pipefail
REPO=/share/thickstun/sychou/workspace/research/s-flm
OUT=$REPO/outputs/hflm_curv_init_lr_sudoku
ARC_OUT=/home/shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku
PROXY='ssh -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="/home/sc3379/bin/tailscale --socket=/home/sc3379/.tailscale/tailscaled-unicorn-login-02.sock nc %h %p" -i /home/sc3379/.ssh/unicorn_internal'
TC=shengyenc@100.89.71.37
FAL=shengyenc@100.66.60.61
FILTER=(--include='d-*/' --include='d-*/eval/' --include='d-*/eval/results.json' --exclude='*')

# 1) two-way results sync (ARC home is shared by tc + falcon)
rsync -az -e "$PROXY" "${FILTER[@]}" "$TC:$ARC_OUT/" "$OUT/" 2>/dev/null
rsync -az -e "$PROXY" "${FILTER[@]}" "$OUT/" "$TC:$ARC_OUT/" 2>/dev/null

# 1b) reap checkpoints of finished cells (jobs submitted before the auto-clean
#     patch don't clean up after themselves; ~1.4-1.8G/cell; user-approved 2026-07-03)
for d in "$OUT"/*/; do
  [ -f "${d}eval/results.json" ] && [ -d "${d}checkpoints" ] && rm -rf "${d}checkpoints"
done
eval timeout 120 "$PROXY" "$TC" bash -s <<'INNER' 2>/dev/null
OUT=/home/shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku
for d in "$OUT"/*/; do
  [ -f "${d}eval/results.json" ] && [ -d "${d}checkpoints" ] && rm -rf "${d}checkpoints"
done
INNER

# 2a) prune unicorn PD jobs for cells that are done
pruned_uni=0
while read -r jid name; do
  tag=${name#hcil_}
  if [ -f "$OUT/$tag/eval/results.json" ]; then scancel "$jid" && pruned_uni=$((pruned_uni+1)); fi
done < <(squeue -u sc3379 -h -t PD -o "%i %j" | grep " hcil_")

# 2b) prune on tc and falcon (same script logic remotely; shared fs)
prune_remote() {
  eval timeout 90 "$PROXY" "$1" bash -s <<'INNER' 2>/dev/null
OUT=/home/shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku
n=0
while read -r jid name; do
  tag=${name#hcil_}
  if [ -f "$OUT/$tag/eval/results.json" ]; then scancel "$jid" && n=$((n+1)); fi
done < <(squeue -u shengyenc -h -t PD -o "%i %j" | grep " hcil_")
echo "$n pruned; R=$(squeue -u shengyenc -h -t R | wc -l) PD=$(squeue -u shengyenc -h -t PD | wc -l)"
INNER
}
tc_stat=$(prune_remote "$TC")
fal_stat=$(prune_remote "$FAL")

done_n=$(ls "$OUT"/*/eval/results.json 2>/dev/null | wc -l)
echo "[sync-prune] done=$done_n/1008 | unicorn: pruned=$pruned_uni R=$(squeue -u sc3379 -h -t R | wc -l) PD=$(squeue -u sc3379 -h -t PD | wc -l) | tc: ${tc_stat:-unreachable} | falcon: ${fal_stat:-unreachable}"
