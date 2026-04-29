#!/bin/bash
# Determinism sanity check (codex round 11-14, 2026-04-29).
# Verifies that --deterministic + --seed 0 + CUBLAS_WORKSPACE_CONFIG
# produces stable J-drop across identical-config runs.
#
# 3 runs of dog with duplicate_seed + joint W [1,22,35] + insert_only_100
# + delta_support=off. Same seed each time. Expected: J-drop variance < 5%.
#
# If variance < 5% → determinism works → proceed to clean carrier comparison.
# If variance >= 5% → cuDNN nondeterminism or autocast leakage; debug.
#
# Compute: ~5-7 min/run × 3 = ~15-21 GPU-min on Pro 6000 GPU 1.

set -u
TAG="ss7-determinism"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_determinism}
: ${PROFILE_ROOT:=vadi_runs/v5_go_nogo_profiles}
: ${CLIP:=dog}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] CLIP=${CLIP} OUT_ROOT=${OUT_ROOT}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  RUN_REPLICATE() {
    local replicate=$1
    echo ""
    echo "[${TAG}] === REPLICATE ${replicate} (deterministic, seed=0) ==="
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE python launches for
    # cuBLAS determinism. systemd-run --setenv preserves it across the
    # transient scope.
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    systemd-run --user --scope \
        --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root ~/sam2-pre-new/data/davis \
        --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
        --clips ${CLIP} \
        --insert-base duplicate_seed \
        --use-profiled-placement "${PROFILE_ROOT}" \
        --schedule insert_only_100 \
        --delta-support off \
        --deterministic \
        --seed 0 \
        --out-root "${OUT_ROOT}/run${replicate}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === REPLICATE ${replicate} exited rc=${rc} at $(date) ==="
  }

  RUN_REPLICATE 1
  RUN_REPLICATE 2
  RUN_REPLICATE 3

  echo ""
  echo "[${TAG}] === aggregating ==="
  python - <<EOF
import json, pathlib, statistics
out_root = pathlib.Path("${OUT_ROOT}")
clip = "${CLIP}"

def read_jdrop(rep):
    p = out_root / f"run{rep}" / clip
    if not p.exists():
        return None
    cands = list(p.glob("K*__ot/results.json")) + list(p.glob("K*/results.json"))
    if not cands:
        return None
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    return float(d.get("exported_j_drop") or 0.0)

j = [read_jdrop(i) for i in (1, 2, 3)]
print(f"[${TAG}] DETERMINISM RESULTS:")
for i, v in enumerate(j, 1):
    print(f"[${TAG}]   replicate {i}: J-drop = {v}")
ok = [v for v in j if v is not None]
if len(ok) >= 2:
    mean = statistics.mean(ok)
    std = statistics.pstdev(ok) if len(ok) > 1 else 0.0
    rng = max(ok) - min(ok)
    rel_std = (std / abs(mean)) if mean != 0 else float("inf")
    rel_rng = (rng / abs(mean)) if mean != 0 else float("inf")
    print(f"[${TAG}]   mean = {mean:.4f}")
    print(f"[${TAG}]   std  = {std:.6f}")
    print(f"[${TAG}]   range= {rng:.6f}")
    print(f"[${TAG}]   relative std   = {rel_std:.4f} ({rel_std*100:.2f}%)")
    print(f"[${TAG}]   relative range = {rel_rng:.4f} ({rel_rng*100:.2f}%)")
    if rel_rng < 0.05:
        print(f"[${TAG}]   VERDICT: GO  (rel range < 5%, determinism OK)")
    elif rel_rng < 0.10:
        print(f"[${TAG}]   VERDICT: BORDERLINE  (5%-10% drift, may need further fix)")
    else:
        print(f"[${TAG}]   VERDICT: FAIL  (>=10% drift; cuDNN nondeterminism or autocast leak)")
EOF
  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
