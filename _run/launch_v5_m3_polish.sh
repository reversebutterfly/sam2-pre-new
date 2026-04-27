#!/bin/bash
# v5 M3 — full v5 polish on 10 held-out clips (produces x_polished + W*).
# Runs joint placement search + A0 polish + Stage 14 v4.1 + adaptive wrapper
# per clip. Output is consumed by M2 (A3 gating) and M4 (A2 random).
#
# Expected wall: ~8 GPU-h for 10 clips on Pro 6000 (sequential single GPU).
#   Per clip: joint search ~22 min + Stage 14 ~10 min + export ~2 min.
#   Plus A0 polish ~5 min Stage 11-13.

set -u
TAG="ss7-v5-m3-polish"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_paper_m3}
: ${CLIPS:="bear blackswan breakdance cows dance-twirl dog hike horsejump-high india judo"}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT}"
  echo "[${TAG}] CLIPS=${CLIPS}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m scripts.run_vadi_v5 \
      --davis-root ~/sam2-pre-new/data/davis \
      --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
      --clips ${CLIPS} \
      --oracle-trajectory \
      --oracle-traj-v4 \
      --placement-search joint_curriculum \
      --out-root "${OUT_ROOT}" \
      --device cuda
  RC=$?
  echo "[${TAG}] python exited rc=${RC} at $(date)"
  if [ "${RC}" = "0" ] && [ -f "${OUT_ROOT}/v5_summary.json" ]; then
    echo "[${TAG}] === v5_summary.json ==="
    cat "${OUT_ROOT}/v5_summary.json" | head -100
  fi
} > "${LOG}" 2>&1
