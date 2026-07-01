#!/bin/bash
# Generate the decode-policy sweep manifest for the fair geometry comparison.
# Each line is one GenPPL eval cell:  MODEL_TYPE CKPT STEPS NOISE_REMOVAL TOPKV STEP_TAG
#
# Within every cell the decode policy (steps, noise_removal, top_k_velocity, +pinned
# velocity/temperature/p_nucleus/top_k in eval.sh) is IDENTICAL across models; only the
# geometry (predictor) differs. Grid:
#   steps          in {32,64,128,256,512,1024}   (NFE-vs-quality curve)
#   noise_removal  in {ancestral, greedy}        (final-step rule)
#   top_k_velocity = 1 for ALL models            (aligned: both step toward single
#                    argmax-predicted-clean token; the ONLY value HFLM can honour since
#                    its hyperbolic step is structurally top-1)
#   top_k_velocity = -1 additionally for sfm / sfm_adaptive  (sphere-NATIVE reference, so
#                    the aligned gap is not misread as geometry; no-op for HFLM -> skipped)
# Sweep is on the 30k (final) checkpoint only; the 10k/20k checkpoint axis is the separate
# overfitting probe handled by the per-step single-config evals.
set -euo pipefail
ROOT="/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm"
OUT="${ROOT}/experiments/tinystories/sweep_manifest.txt"

# Resolve the explicit 30000-step checkpoint per model. The epoch prefix at step 30000
# differs per model (hflm=33; sfm/sfm_adaptive differ), so glob *-30000.ckpt instead of a
# fixed name. Using the numbered file (NOT the rolling last.ckpt) makes the existence
# pre-check below an automatic same-step (30k) guarantee -- the fair-comparison invariant.
ck30k() { local d="$1" g=( "$1"/*-30000.ckpt ); if [ -e "${g[0]}" ]; then echo "${g[0]}"; else echo "${d}/__NO_30000_CKPT__"; fi; }
declare -A CKPT=(
  [sfm]="$(ck30k "${ROOT}/outputs/tinystories/sfm/checkpoints")"
  [sfm_adaptive]="$(ck30k "${ROOT}/outputs/tinystories/sfm_truncated_adaptive/checkpoints")"
  [hflm]="$(ck30k "${ROOT}/outputs/tinystories/hflm/checkpoints")"
)

STEPS_LIST=(32 64 128 256 512 1024)
NR_LIST=(ancestral greedy)

nr_tag() { case "$1" in ancestral) echo anc;; greedy) echo grd;; *) echo "$1";; esac; }
kv_tag() { case "$1" in -1) echo kvm1;; *) echo "kv$1";; esac; }

emit() {  # model topkv
  local model="$1" kv="$2" ck="${CKPT[$1]}"
  for s in "${STEPS_LIST[@]}"; do
    for nr in "${NR_LIST[@]}"; do
      echo "${model} ${ck} ${s} ${nr} ${kv} 30k_s${s}_$(nr_tag "$nr")_$(kv_tag "$kv")"
    done
  done
}

TMP="${OUT}.tmp"
{
  emit hflm 1                       # hyperbolic: top_k_velocity is a no-op, one pass only
  emit sfm 1;          emit sfm -1          # sphere: aligned (1) + native (-1)
  emit sfm_adaptive 1; emit sfm_adaptive -1
} > "${TMP}"

# Sanity: every referenced 30k checkpoint must exist before we publish the manifest.
missing=0
while read -r model ck _; do
  [ -f "$ck" ] || { echo "MISSING ckpt for ${model}: ${ck}" >&2; missing=1; }
done < <(awk '!seen[$1 SUBSEP $2]++' "${TMP}")
if [ "$missing" -ne 0 ]; then
  rm -f "${TMP}"; echo "ABORT: missing 30k checkpoints; manifest not written"; exit 1
fi
mv "${TMP}" "${OUT}"
echo "wrote ${OUT} ($(wc -l < "${OUT}") cells)"
