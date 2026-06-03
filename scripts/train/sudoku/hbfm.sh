#!/bin/bash
# HyperbolicBoundaryFM (HBFM) training on Sudoku.
#
# Mirrors sfm.sh but uses the hyperbolic heat-kernel-bridge posterior:
#   q_xt = poincare_bridge / binary_poincare_bridge (differentiable; grad -> embedding direction),
#   plain cross-entropy x uniform-proposal weight, embedding NOT renormalized.
#
# Dimension is set via D (= model.hidden_size = word-embedding dim = bridge dim).
# NOTE: geo_bridge.sample_radial is finite only for D <= ~64 (linear-space sinh^{d-1}
# marginal overflows at D >= 80); the D=2 path uses the closed-form binary kernel and
# is unaffected. Primary run: D=64. Smoke gate: D=2.
#
#   D=2  DIFFICULTY=easy ./scripts/train/sudoku/hbfm.sh     # smoke gate
#   D=64 DIFFICULTY=easy ./scripts/train/sudoku/hbfm.sh     # primary

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
D="${D:-64}"                          # hidden_size = embedding dim = bridge dim (<= 64)
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"       # use a small value (e.g. 300) for a smoke gate
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/hbfm_d${D}_${DIFFICULTY}_seed${SEED}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"

# head_dim must be >= 2 for the rotary fallback; d=2 needs a single head.
if [ "${D}" -le 2 ]; then N_HEADS=1; else N_HEADS="${N_HEADS:-8}"; fi

cd "${REPO_ROOT}"

python -u -m main \
    seed="${SEED}" \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit \
    model.hidden_size="${D}" \
    model.n_heads="${N_HEADS}" \
    algo=hbfm \
    sampler=hbfm \
    noise=log-linear \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval="${MAX_STEPS}" \
    trainer.limit_val_batches=0 \
    trainer.max_steps="${MAX_STEPS}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=$(( MAX_STEPS < 5000 ? MAX_STEPS : 5000 )) \
    hydra.run.dir="${OUTPUT_DIR}"
