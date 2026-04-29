#!/bin/bash
# Quick attack-effect check on hybrid poisson_hifi carrier (codex round 8 fix).
# Single clip (dog), seed-only (no PGD), 3 arms:
#   1. duplicate_seed seed-only (legacy carrier baseline)
#   2. poisson_hifi seed-only with inner_color_preserve_erode=4 (NEW hybrid)
#   3. poisson_hifi seed-only with erode=0 (pure cv2.seamlessClone — for B comparison)
#
# Compute: ~3-5 min × 3 arms = ~15 min on Pro 6000 GPU 1 (light, no PGD).
# Out: vadi_runs/v5_attack_check_hybrid/{dup, hybrid, pure}/dog/...
#
# Note: seed-only writes ν=0, δ=0 carrier and runs SAM2 forward to compute J.
# This isolates the CARRIER's contribution to attack from any PGD optimization.

set -u
TAG="ss7-attack-check-hybrid"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_attack_check_hybrid}
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

  RUN_ARM() {
    local name=$1; local insert_base=$2
    echo ""
    echo "[${TAG}] === ARM ${name}: insert-base=${insert_base} ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root ~/sam2-pre-new/data/davis \
        --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
        --clips ${CLIP} \
        --insert-base ${insert_base} \
        --allow-ghost-fallback \
        --use-profiled-placement "${PROFILE_ROOT}" \
        --seed-only \
        --out-root "${OUT_ROOT}/${name}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === ARM ${name} exited rc=${rc} at $(date) ==="
  }

  RUN_ARM "dup"     "duplicate_seed"
  RUN_ARM "hybrid"  "poisson_hifi"
  # Note: there is no "erode=0 pure poisson" CLI yet; would need a separate flag.
  # Skipping arm 3 for this quick check — visual comparison already showed
  # pure poisson is washed-out (irrelevant here for attack-effect).

  # Aggregate
  echo ""
  echo "[${TAG}] === aggregating arms ==="
  python - <<EOF
import json, pathlib
out_root = pathlib.Path("${OUT_ROOT}")
clip = "${CLIP}"

def read_jdrop(arm_dir):
    p = arm_dir / clip
    if not p.exists():
        return None
    cands = list(p.glob("K*__ot/results.json")) + list(p.glob("K*/results.json"))
    if not cands:
        return None
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    return float(d.get("exported_j_drop") or 0.0)

j_dup = read_jdrop(out_root / "dup")
j_hybrid = read_jdrop(out_root / "hybrid")

print(f"[${TAG}] CARRIER ATTACK EFFECT (seed-only, no PGD):")
print(f"[${TAG}]   duplicate_seed (legacy carrier): J-drop = {j_dup}")
print(f"[${TAG}]   poisson_hifi hybrid (new):       J-drop = {j_hybrid}")
if j_dup is not None and j_hybrid is not None:
    delta = j_hybrid - j_dup
    pct = 100.0 * j_hybrid / max(j_dup, 1e-6) if j_dup > 0 else 0.0
    print(f"[${TAG}]   delta (hybrid - dup) = {delta:+.4f} ({pct:.0f}% of legacy)")
EOF
  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
