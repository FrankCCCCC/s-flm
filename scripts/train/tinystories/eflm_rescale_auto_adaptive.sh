#!/bin/bash
# E-FLM with fixed embedding norm R (rho_min = rho_max = RHO) on the AUTONOMOUS
# clock: 1 - alpha_t = exp(-tau), tau = TAU_MAX * (1 - t), so the bridge drift is
# the time-invariant v(X) = y - X (slides/jul09_2026) and every sampler step
# advances the same d_tau. TAU_MAX *is* the truncation on this clock:
# tau*(R) = -log(1 - alpha_star_euclidean(V=50257, embed_norm=RHO)) truncates at
# the decode point; TAU_MAX=6.908 (= -log 1e-3) is the untruncated horizon,
# spanning the same SNR range as the log-linear schedule.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinystories/eflm_rescale_auto_adaptive}"
RUN_NAME="${RUN_NAME:-eflm_rescale_auto_adaptive}"
WANDB_GROUP="${WANDB_GROUP:-adv_geo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
PER_GPU_BS="${PER_GPU_BS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2500}"
LR="${LR:-3e-4}"
RHO="${RHO:-1.0}"                   # fixed embedding norm R (rho_min = rho_max = RHO)
SNR_CE="${SNR_CE:-false}"            # weight the CE by -SNR'(t)/2 (VDM Eq. 16)
TAU_MAX="${TAU_MAX:-1.834}"         # autonomous horizon; tau*(R)=-log(1-alpha*(R)) truncates at the decode point
SELF_COND="${SELF_COND:-false}"      # LangFlow-style self-conditioning
# self-conditioning leaves the self-cond params unused on ~75% of steps (p_self_cond);
# default ddp strategy (find_unused_parameters=false) errors on that -> enable when self-cond.
if [ "${SELF_COND}" = "true" ]; then SC_STRAT="strategy.find_unused_parameters=true"; else SC_STRAT=""; fi

cd "${REPO_ROOT}"
python -u -m main \
    data=tinystories \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.length=${SEQ_LEN:-1024} \
    model.init=ngpt \
    algo=eflm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    algo.self_conditioning="${SELF_COND}" \
    algo.rho_min="${RHO}" \
    algo.rho_max="${RHO}" \
    algo.snr_weighted_ce="${SNR_CE}" \
    noise=autonomous-adaptive \
    noise.tau_max=${TAU_MAX} \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    optim.lr=${LR} \
    loader.global_batch_size=512 \
    loader.batch_size=${PER_GPU_BS} \
    loader.eval_batch_size=${PER_GPU_BS} \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    ${SC_STRAT} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=60_000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
    callbacks.checkpoint_every_n_steps.save_top_k=${SAVE_TOPK:-1} \
    wandb.project=tinystories-flm \
    wandb.group="${WANDB_GROUP}" \
    +wandb.name="${RUN_NAME}" \
    +wandb.offline=true \
    hydra.run.dir="${OUTPUT_DIR}"
