#!/bin/bash
# Gather the retrained seed=1 / hard CHECKPOINTS from Falcon (ARC /home) onto
# unicorn /share, where the user wants them. Run from unicorn (sc3379).
#   default : rsync ARC d-hard_*_rs1/checkpoints -> unicorn (copy; idempotent)
#   --prune : also delete the ARC-side checkpoints of cells whose rc_ job has
#             finished (bounds ARC /home usage to in-flight cells; ~0.44G/cell)
# Unicorn-native cells (baselines + K in {-0.25,-1.0}) already land on /share.
set -uo pipefail
PRUNE=0; [ "${1:-}" = "--prune" ] && PRUNE=1
REPO=/share/thickstun/sychou/workspace/research/s-flm
OUT=$REPO/outputs/hflm_curv_init_lr_sudoku
ARC_OUT=/home/shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku
PROXY='ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o ProxyCommand="/home/sc3379/bin/tailscale --socket=/home/sc3379/.tailscale/tailscaled-unicorn-login-02.sock nc %h %p" -i /home/sc3379/.ssh/unicorn_internal'
FAL=shengyenc@100.66.60.61

# 1) pull all Falcon hard/seed1 checkpoints -> unicorn /share
rsync -az -e "$PROXY" \
  --include='d-hard_*_rs1/' --include='d-hard_*_rs1/checkpoints/' \
  --include='d-hard_*_rs1/checkpoints/**' --exclude='*' \
  "$FAL:$ARC_OUT/" "$OUT/" 2>/dev/null && echo "[gather] pulled ARC -> unicorn"

have=$(ls $OUT/d-hard_*_rs1/checkpoints/last.ckpt 2>/dev/null | wc -l)
echo "[gather] unicorn now holds $have hard/seed1 H-FLM last.ckpt (of 168)"

# 2) optional: prune ARC checkpoints of finished cells (job no longer in squeue)
if [ "$PRUNE" = 1 ]; then
  eval timeout 120 "$PROXY" "$FAL" bash -s <<'INNER' 2>/dev/null
OUT=/home/shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku
running=$(squeue -u shengyenc -h -o "%j" | grep '^rc_' | sed 's/^rc_//')
n=0
for d in "$OUT"/d-hard_*_rs1/; do
  tag=$(basename "$d")
  [ -f "${d}checkpoints/last.ckpt" ] || continue
  echo "$running" | grep -qx "$tag" && continue        # still training -> keep
  rm -rf "${d}checkpoints" && n=$((n+1))                # done + gathered -> drop
done
echo "[gather] pruned $n finished-cell checkpoints on ARC"
INNER
fi
