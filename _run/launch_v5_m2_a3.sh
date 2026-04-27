#!/bin/bash
# v5 M2 — A3 gating run on 10 held-out clips.
# 3-condition memory-block ablation + d_mem(t) trace per clip.
#
# Pre-registered acceptance (refine-logs/EXPERIMENT_PLAN.md):
#   Strong : >=7/10 with collapse_attacked >= 0.20 AND (att-ctrl) >= 0.10
#   Partial: >=6/10 with collapse_attacked >= 0.10 AND (att-ctrl) >= 0.05
#   Fail   : workshop pivot (re-pre-registered framing)
#
# DEPENDENCIES (must exist before this run):
#   - vadi_runs/v5_paper_m3/<clip>/<config>__ot/processed/   (M3 polished output)
#   - vadi_runs/v5_paper_m3/<clip>/<config>/results.json     (W_attacked)
#
# If running A3 BEFORE M3 polish, change V5_ROOT to point at the v4.1 retest
# dir or run M3 first.
#
# Expected wall: ~3-4 GPU-h for 10 clips × 4 forwards (clean ref + 3
# conditions) on Pro 6000.

set -u
TAG="ss7-v5-m2-a3"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_paper_m2_a3}
: ${V5_ROOT:=vadi_runs/v5_paper_m3}
: ${CLIPS:="bear blackswan breakdance cows dance-twirl dog hike horsejump-high india judo"}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT} V5_ROOT=${V5_ROOT}"
  echo "[${TAG}] CLIPS=${CLIPS}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  if [ ! -d "${V5_ROOT}" ]; then
    echo "[${TAG}] ERROR: V5_ROOT (${V5_ROOT}) does not exist."
    echo "[${TAG}]   Run M3 (scripts.run_vadi_v5 --oracle-traj-v4 ...) first"
    echo "[${TAG}]   OR override V5_ROOT to point at an existing v5 run dir."
    exit 2
  fi

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m scripts.run_a3_gating \
      --davis-root ~/sam2-pre-new/data/davis \
      --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
      --v5-root "${V5_ROOT}" \
      --out-root "${OUT_ROOT}" \
      --clips ${CLIPS} \
      --device cuda \
      --control-seed 0 \
      --top-k 32
  RC=$?
  echo "[${TAG}] python exited rc=${RC} at $(date)"
  if [ "${RC}" = "0" ] && [ -f "${OUT_ROOT}/a3_summary.json" ]; then
    echo "[${TAG}] === a3_summary.json ==="
    cat "${OUT_ROOT}/a3_summary.json"
  fi
} > "${LOG}" 2>&1
