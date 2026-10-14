#!/bin/bash
# Eval for E-FLM with fixed embedding norm R on the AUTONOMOUS clock. TAU_MAX
# and RHO must match training: the sampler rebuilds the model from these CLI
# overrides, and the tau grid (uniform t = uniform tau) is what makes each
# Euler step advance the flow by the same amount.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the trained eflm_rescale_auto sudoku checkpoint}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/eflm_rescale_auto_${DIFFICULTY}}"
PPL_JSON="${PPL_JSON:-${OUTPUT_DIR}/ppl.json}"   # where the ppl_eval metrics go
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
SEED="${SEED:-1}"                      # seed of the ppl_eval pass (the (t, noise) draws) and of the board sampler
STEPS="${STEPS:-180}"
VELOCITY="${VELOCITY:-exact}"         # sample / exact
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"  # 1 = top-1, -1 = full vocab (no top-k)
SELF_COND="${SELF_COND:-false}"      # self-conditioning; must match training
RHO="${RHO:-1.0}"                    # fixed embedding norm R; must match training
TAU_MAX="${TAU_MAX:-3.0}"            # autonomous-clock horizon; must match training
SNR_CE="${SNR_CE:-false}"            # loss weighting; only affects the reported bound
LOSS_TYPE="${LOSS_TYPE:-ce}"         # objective of the ppl_eval pass (algo.loss_type): elbo = the VDM NELBO, ce = the denoising CE (not a bound)
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
    model=tiny-sphere-dit \
    algo=eflm \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.rho_min="${RHO}" \
    algo.rho_max="${RHO}" \
    algo.snr_weighted_ce="${SNR_CE}" \
    algo.loss_type="${LOSS_TYPE}" \
    noise=autonomous \
    noise.tau_max="${TAU_MAX}" \
    sampler=eflm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
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
