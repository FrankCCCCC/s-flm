#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained MDLM sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/mdlm_${DIFFICULTY}}"
PPL_JSON="${PPL_JSON:-${OUTPUT_DIR}/ppl.json}"   # where the ppl_eval metrics go
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
SEED="${SEED:-1}"                      # seed of the ppl_eval pass (the (t, noise) draws) and of the board sampler
STEPS="${STEPS:-180}"
LOSS_TYPE="${LOSS_TYPE:-elbo}"       # objective of the ppl_eval pass (algo.loss_type): elbo = MDLM's NELBO
RUN_PPL_EVAL="${RUN_PPL_EVAL:-false}"   # true: also one mode=ppl_eval pass (val/nll of LOSS_TYPE in nats per solution token, 2000 boards) -> PPL_JSON
RUN_SAMPLE_EVAL="${RUN_SAMPLE_EVAL:-true}"   # board accuracy (mode=sudoku_eval); false: ppl_eval pass only

cd "${REPO_ROOT}"

run_main() {  # $@: the pass-specific overrides (mode, seed, output paths)
python -u -m main \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny \
    algo=mdlm \
    algo.loss_type="${LOSS_TYPE}" \
    sampler=ancestral \
    sampler.steps="${STEPS}" \
    loader.global_batch_size=256 \
    loader.eval_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    +wandb.offline=True \
    "$@"
}

# (1) likelihood bound (RUN_PPL_EVAL=true): one ppl_eval pass on the 2000
# validation boards, one (t, noise) draw per board. To average several passes,
# call the script once per SEED with its own PPL_JSON (the sweep does).
if [ "${RUN_PPL_EVAL}" = "true" ]; then
run_main mode=ppl_eval seed="${SEED}" eval.generate_samples=False \
    eval.results_json_path="${PPL_JSON}" hydra.run.dir="${OUTPUT_DIR}/ppl"
fi

# (2) board accuracy
if [ "${RUN_SAMPLE_EVAL}" = "true" ]; then
run_main mode=sudoku_eval seed="${SEED}" sudoku.batch_size=64 \
    sudoku.output_dir="${OUTPUT_DIR}" hydra.run.dir="${OUTPUT_DIR}"
fi
