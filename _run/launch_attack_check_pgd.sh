#!/bin/bash
# Attack-effect comparison: dup vs hybrid carrier, full PGD optimization.
# Single clip (dog), 2 arms × 100-step ν PGD, no bridge δ.
#
# Replaces the earlier seed-only test, which was uninformative because
# carriers alone produce ~0 J-drop (the attack comes from ν optimization).
#
# Compute: ~10-15 min × 2 arms = ~30 min on Pro 6000 GPU 1.
# Out: vadi_runs/v5_attack_check_pgd/{dup, hybrid}/dog/...
#
# Decision: if hybrid J-drop ≥ 0.55 (within ~13% of dup baseline 0.6351),
# the hybrid carrier preserves attack capacity — proceed to GO/NO-GO with
# state_continuation bridge. If < 0.40, hybrid weakens too much — need to
# tune (smaller erode, alpha-only paste, or ProPainter).

set -u
TAG="ss7-attack-check-pgd"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_attack_check_pgd}
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
    echo "[${TAG}] === ARM ${name}: insert-base=${insert_base} (full PGD) ==="
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
        --schedule insert_only_100 \
        --delta-support off \
        --out-root "${OUT_ROOT}/${name}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === ARM ${name} exited rc=${rc} at $(date) ==="
  }

  RUN_ARM "dup"     "duplicate_seed"
  RUN_ARM "hybrid"  "poisson_hifi"

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
    return float(d.get("exported_j_drop") or 0.0), str(cands[0])

j_dup_t = read_jdrop(out_root / "dup")
j_hybrid_t = read_jdrop(out_root / "hybrid")

print(f"[${TAG}] FULL-PGD ATTACK COMPARISON:")
if j_dup_t:
    print(f"[${TAG}]   duplicate_seed  carrier + PGD ν: J-drop = {j_dup_t[0]:.4f}")
    print(f"[${TAG}]                                    src   = {j_dup_t[1]}")
if j_hybrid_t:
    print(f"[${TAG}]   poisson_hifi    carrier + PGD ν: J-drop = {j_hybrid_t[0]:.4f}")
    print(f"[${TAG}]                                    src   = {j_hybrid_t[1]}")
if j_dup_t and j_hybrid_t:
    j_dup, j_hybrid = j_dup_t[0], j_hybrid_t[0]
    delta = j_hybrid - j_dup
    pct = 100.0 * j_hybrid / max(j_dup, 1e-6) if j_dup > 0 else 0.0
    print()
    print(f"[${TAG}]   delta (hybrid - dup) = {delta:+.4f}")
    print(f"[${TAG}]   hybrid retains       = {pct:.0f}% of dup attack capacity")
    if j_hybrid >= 0.55:
        print(f"[${TAG}]   VERDICT: GO  (hybrid >=0.55, well within range)")
    elif j_hybrid >= 0.40:
        print(f"[${TAG}]   VERDICT: GO-WITH-COST (acceptable J-drop sacrifice)")
    else:
        print(f"[${TAG}]   VERDICT: TUNE (hybrid <0.40, weakens too much)")
EOF
  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
