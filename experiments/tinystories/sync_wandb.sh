#!/bin/bash
# Push the offline TinyStories W&B runs to wandb.ai/syctw/tinystories-flm.
# Training runs offline (online wandb is unreliable under DDP on this cluster), so
# run this periodically and once more after training finishes to get the final curves.
set -uo pipefail
REPO_ROOT="/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm"
WANDB=/home/sc3379/anaconda3/envs/sfm/bin/wandb
for d in $(find "${REPO_ROOT}/outputs/tinystories" -maxdepth 3 -type d -name "offline-run-*" 2>/dev/null); do
  echo "--- syncing $d ---"
  "${WANDB}" sync "$d" 2>&1 | grep -iE "syncing|done|error" | head -2
done
echo "sync complete"
