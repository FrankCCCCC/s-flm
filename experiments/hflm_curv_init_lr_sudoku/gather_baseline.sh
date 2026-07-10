#!/bin/bash
# gather_baseline.sh — pull the 3-seed baseline re-run outputs from Falcon /scratch
# back to unicorn /share (the persistent, analysable location). Run ON unicorn.
#
# The re-run trains all 63 cells on Falcon with checkpoints RETAINED on
# /scratch/shengyenc/sfm_output/... (scratch is huge + login-visible, no /home quota
# blow-up). scratch is periodically purged, so this script persists the deliverable:
#   - always: bl_*/eval/results.json (+ eval logs)  -> for analyze_baseline.py
#   - with --ckpts: bl_*/checkpoints/last.ckpt too  (~1.8G/cell, ~113G total)
set -euo pipefail

SOCK=/home/sc3379/.tailscale/tailscaled-unicorn-login-02.sock
FALCON=100.66.60.61
SRC=/scratch/shengyenc/sfm_output/hflm_curv_init_lr_sudoku
DST=/share/thickstun/sychou/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku

# rsync -e cannot take the ProxyCommand inline (quoting); wrap it in a script.
WRAP=$(mktemp)
cat > "$WRAP" <<EOF
#!/bin/bash
exec ssh -o "ProxyCommand=/home/sc3379/bin/tailscale --socket=$SOCK nc %h %p" \
  -o BatchMode=yes -o StrictHostKeyChecking=no -i /home/sc3379/.ssh/unicorn_internal "\$@"
EOF
chmod +x "$WRAP"
trap 'rm -f "$WRAP"' EXIT

mkdir -p "$DST"
if [ "${1:-}" = "--ckpts" ]; then
    echo "[gather] results.json + checkpoints (large) ..."
    rsync -av --prune-empty-dirs -e "$WRAP" \
        --include='bl_*/' --include='bl_*/eval/***' \
        --include='bl_*/checkpoints/***' --exclude='*' \
        "shengyenc@${FALCON}:${SRC}/" "${DST}/"
else
    echo "[gather] results.json + eval logs only ..."
    rsync -av --prune-empty-dirs -e "$WRAP" \
        --include='bl_*/' --include='bl_*/eval/***' --exclude='*' \
        "shengyenc@${FALCON}:${SRC}/" "${DST}/"
fi

echo "[gather] done. results present: $(ls "${DST}"/bl_*/eval/results.json 2>/dev/null | wc -l)/63"
