#!/bin/bash
# Disambiguation experiment per codex round 9 (2026-04-29).
# Tests 3 hypotheses for the failed PGD attack-effect comparison:
#   A — fidelity compromise weakens attack
#   B — test setup broken (ghost-fallback + nondeterminism)
#   C — W mismatch (joint W used with bare A0 pipeline)
#   D — dog clip unsuitable for ghost-free carrier
#
# 4 arms × 100-step ν PGD insert-only:
#   arm1: dog,   duplicate_seed, joint W=[1,22,35]    (reproduce 0.06 baseline)
#   arm2: dog,   duplicate_seed, A0 W=[8,18,32]       (if jumps → C confirmed)
#   arm3: libby, duplicate_seed, libby joint W=[1,10,15]   (libby A0 baseline)
#   arm4: libby, poisson_hifi, same W, STRICT no-fallback (ghost-fallback raises)
#
# Decoder:
#   arm2 >> arm1   →  C dominant (W mismatch)
#   arm4 ≈ arm3    →  A weak (carrier doesn't matter much)
#   arm4 << arm3   →  A real (fidelity costs attack)
#   arm4 raises    →  D real (libby still fails border-safety; need lower margin)

set -u
TAG="ss7-disambig-4arm"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_disambig_4arm}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] OUT_ROOT=${OUT_ROOT}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  RUN_ARM() {
    local name=$1; local clip=$2; local insert_base=$3
    local profile_root=$4; local strict=$5  # strict=1 → no --allow-ghost-fallback
    echo ""
    echo "[${TAG}] === ARM ${name}: clip=${clip} carrier=${insert_base} profile=${profile_root} strict=${strict} ==="
    local fb_flag=""
    if [ "${strict}" != "1" ]; then
      fb_flag="--allow-ghost-fallback"
    fi
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root ~/sam2-pre-new/data/davis \
        --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
        --clips ${clip} \
        --insert-base ${insert_base} \
        ${fb_flag} \
        --use-profiled-placement "${profile_root}" \
        --schedule insert_only_100 \
        --delta-support off \
        --out-root "${OUT_ROOT}/${name}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === ARM ${name} exited rc=${rc} at $(date) ==="
  }

  # Arm 1: dog, duplicate_seed, joint W (reproduce 0.06 baseline)
  RUN_ARM "arm1_dog_dup_jointW"  "dog" "duplicate_seed" \
          "vadi_runs/v5_go_nogo_profiles" "0"

  # Arm 2: dog, duplicate_seed, A0 W (if jumps → C confirmed)
  RUN_ARM "arm2_dog_dup_a0W"     "dog" "duplicate_seed" \
          "vadi_runs/v5_placement_profile" "0"

  # Arm 3: libby, duplicate_seed, joint W (libby A0 baseline)
  RUN_ARM "arm3_libby_dup_jointW" "libby" "duplicate_seed" \
          "vadi_runs/v5_go_nogo_profiles" "0"

  # Arm 4: libby, poisson_hifi, joint W, STRICT (no fallback)
  RUN_ARM "arm4_libby_hybrid_jointW" "libby" "poisson_hifi" \
          "vadi_runs/v5_go_nogo_profiles" "1"

  # Aggregate
  echo ""
  echo "[${TAG}] === aggregating ==="
  python - <<EOF
import json, pathlib
out_root = pathlib.Path("${OUT_ROOT}")

def read_jdrop(arm, clip):
    p = out_root / arm / clip
    if not p.exists():
        return None
    cands = list(p.glob("K*__ot/results.json")) + list(p.glob("K*/results.json"))
    if not cands:
        return None
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    return float(d.get("exported_j_drop") or 0.0)

a1 = read_jdrop("arm1_dog_dup_jointW",      "dog")
a2 = read_jdrop("arm2_dog_dup_a0W",         "dog")
a3 = read_jdrop("arm3_libby_dup_jointW",    "libby")
a4 = read_jdrop("arm4_libby_hybrid_jointW", "libby")

print(f"[${TAG}] DISAMBIGUATION RESULTS:")
print(f"[${TAG}]   arm1 dog   dup    joint W [1,22,35]: J = {a1}")
print(f"[${TAG}]   arm2 dog   dup    A0    W [8,18,32]: J = {a2}")
print(f"[${TAG}]   arm3 libby dup    joint W [1,10,15]: J = {a3}")
print(f"[${TAG}]   arm4 libby hybrid joint W [1,10,15]: J = {a4}")

print()
print(f"[${TAG}] === Hypothesis disambiguation ===")
if a1 is not None and a2 is not None:
    if a2 > a1 + 0.10:
        print(f"[${TAG}]   C (W mismatch) CONFIRMED: arm2 ({a2:.3f}) > arm1 ({a1:.3f}) by {a2-a1:+.3f}")
    elif a2 < a1 + 0.05:
        print(f"[${TAG}]   C (W mismatch) WEAK: arm2 ({a2:.3f}) ~= arm1 ({a1:.3f})")
    else:
        print(f"[${TAG}]   C (W mismatch) MARGINAL: arm2 ({a2:.3f}) > arm1 ({a1:.3f}) by {a2-a1:+.3f}")

if a3 is not None and a4 is not None:
    delta = a4 - a3
    pct = 100.0 * a4 / max(a3, 1e-6) if a3 > 0 else 0.0
    if abs(delta) < 0.05:
        print(f"[${TAG}]   A (fidelity cost) WEAK: arm4 hybrid ({a4:.3f}) ~= arm3 dup ({a3:.3f})")
        print(f"[${TAG}]     hybrid retains {pct:.0f}% — carrier change does NOT cost attack much")
    elif delta < -0.10:
        print(f"[${TAG}]   A (fidelity cost) REAL: arm4 hybrid ({a4:.3f}) << arm3 dup ({a3:.3f}) by {-delta:.3f}")
        print(f"[${TAG}]     hybrid retains {pct:.0f}% — carrier change DOES cost attack significantly")
    else:
        print(f"[${TAG}]   A (fidelity cost) MARGINAL: delta = {delta:+.3f}, retains {pct:.0f}%")
elif a4 is None and a3 is not None:
    print(f"[${TAG}]   D (libby still fails border-safety): arm4 raised on poisson_hifi.")
    print(f"[${TAG}]     Need smaller safety_margin or different clip.")
EOF
  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
