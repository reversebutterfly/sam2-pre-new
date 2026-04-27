#!/bin/bash
# v5 M0 R004 — real-SAM2 smoke test on Pro 6000.
# Validates the codex-required deployment gate:
#   1. parity mode (blocked=[], extractor=None) matches base forward
#   2. blocked-frame case runs without error (W_attacked block)
#   3. extractor case captures V/attn (RoPE path on real SAM2.1)
#
# Single clip (dog), reuses existing v5-polished output from prior dev-4
# retest (vadi_runs/v5_pilot_ss7_v41_retest_gpu0/dog/...).
#
# Expected wall: <2 GPU-min on Pro 6000.

set -u
TAG="ss7-v5-m0-smoke"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_paper_m0_smoke}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  # Verify the v5 dev-4 retest dir exists (reusing dog's polished output).
  V5_ROOT=~/sam2-pre-new/vadi_runs/v5_pilot_ss7_v41_retest_gpu0
  if [ ! -d "${V5_ROOT}/dog" ]; then
    echo "[${TAG}] ERROR: ${V5_ROOT}/dog missing; rerun v4.1 retest first"
    exit 2
  fi

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m scripts.run_a3_gating \
      --davis-root ~/sam2-pre-new/data/davis \
      --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
      --v5-root "${V5_ROOT}" \
      --out-root "${OUT_ROOT}" \
      --clips dog \
      --smoke \
      --device cuda
  RC=$?
  echo "[${TAG}] python exited rc=${RC} at $(date)"
  if [ "${RC}" = "0" ]; then
    if [ -f "${OUT_ROOT}/dog/a3_smoke_result.json" ]; then
      echo "[${TAG}] smoke_result:"
      cat "${OUT_ROOT}/dog/a3_smoke_result.json"
    fi
  fi
} > "${LOG}" 2>&1
