#!/bin/bash
# Training for E-FLM with fixed embedding norm R on the AUTONOMOUS clock:
# tau = tau_max * (1 - t) with noise fraction 1 - alpha_t = exp(-tau), so the
# bridge drift is the time-invariant v(X) = y - X (slides/jul09_2026) and every
# sampler step advances the same d_tau. SNR_CE=true swaps the plain CE for the
# VDM (Eq. 16) -SNR'(t)/2-weighted CE variational bound.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sudoku/eflm_rescale_auto_${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
SEED="${SEED:-1}"                    # global random seed (L.seed_everything)
LR="${LR:-3e-4}"                     # AdamW learning rate
SELF_COND="${SELF_COND:-false}"      # LangFlow-style self-conditioning
# self-conditioning leaves the self-cond params unused on ~75% of steps (p_self_cond);
# default ddp strategy (find_unused_parameters=false) errors on that -> enable when self-cond.
if [ "${SELF_COND}" = "true" ]; then SC_STRAT="strategy.find_unused_parameters=true"; else SC_STRAT=""; fi
RHO="${RHO:-1.0}"                    # fixed embedding norm R (rho_min = rho_max = RHO)
TAU_MAX="${TAU_MAX:-3.0}"            # autonomous-clock horizon (noise floor exp(-TAU_MAX))
SNR_CE="${SNR_CE:-false}"            # weight the CE by -SNR'(t)/2 (VDM variational bound)
MAX_STEPS="${MAX_STEPS:-20000}"
CKPT_EVERY="${CKPT_EVERY:-5000}"

cd "${REPO_ROOT}"

python -u -m main \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    seed="${SEED}" \
    model=tiny-sphere-dit \
    optim.lr="${LR}" \
    algo=eflm \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.rho_min="${RHO}" \
    algo.rho_max="${RHO}" \
    algo.snr_weighted_ce="${SNR_CE}" \
    noise=autonomous \
    noise.tau_max="${TAU_MAX}" \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    ${SC_STRAT} \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps=${MAX_STEPS} \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    hydra.run.dir="${OUTPUT_DIR}"
