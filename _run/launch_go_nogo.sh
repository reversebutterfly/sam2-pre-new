#!/bin/bash
# GO/NO-GO test for "Joint > Only" optimized OLD framing (codex round 5+7).
# 4 clips × 2 arms (~6-10 GPU-h on Pro 6000 / V100).
#
# Arm A: A0 ghost-free (poisson_hifi insert + ν only)
# Arm B: A0 ghost-free + state_continuation bridge δ (decoupled optimization)
#
# Pre-registered success bar:
#   B improves over A by ≥ +0.05 J-drop on ≥ 3/4 clips
#   AND inserts visibly cleaner than current oracle-composite (subjective)
#
# Compute: ~6-10 GPU-h.
# Out: vadi_runs/v5_go_nogo/{arm_a, arm_b}/<clip>/...

set -u
TAG="ss7-go-nogo"
cd ~/sam2-pre-new
: ${OUT_ROOT:=vadi_runs/v5_go_nogo}
: ${PROFILE_ROOT:=vadi_runs/v5_go_nogo_profiles}
: ${CLIPS:="camel dog breakdance libby"}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"
{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT} PROFILE_ROOT=${PROFILE_ROOT}"
  echo "[${TAG}] CLIPS=${CLIPS}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  # =================================================================
  # Arm A: A0 ghost-free (poisson_hifi insert + ν only, no δ, no bridge)
  # =================================================================
  echo ""
  echo "[${TAG}] === ARM A: A0 ghost-free (insert-only ν) ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
      --property=ManagedOOMMemoryPressure=auto \
      --property=ManagedOOMSwap=auto \
      python -m scripts.run_vadi_v5 \
      --davis-root ~/sam2-pre-new/data/davis \
      --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
      --clips ${CLIPS} \
      --insert-base poisson_hifi \
      --allow-ghost-fallback \
      --schedule insert_only_100 \
      --delta-support off \
      --use-profiled-placement "${PROFILE_ROOT}" \
      --out-root "${OUT_ROOT}/arm_a" \
      --device cuda
  RC_A=$?
  echo "[${TAG}] === ARM A exited rc=${RC_A} at $(date) ==="

  # =================================================================
  # Arm B: A0 ghost-free + state_continuation bridge δ (decoupled)
  # =================================================================
  echo ""
  echo "[${TAG}] === ARM B: A0 + state_continuation bridge δ ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
      --property=ManagedOOMMemoryPressure=auto \
      --property=ManagedOOMSwap=auto \
      python -m scripts.run_vadi_v5 \
      --davis-root ~/sam2-pre-new/data/davis \
      --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \
      --clips ${CLIPS} \
      --insert-base poisson_hifi \
      --allow-ghost-fallback \
      --schedule insert_only_100 \
      --delta-support off \
      --use-profiled-placement "${PROFILE_ROOT}" \
      --state-continuation \
      --state-continuation-min-improvement 0.02 \
      --out-root "${OUT_ROOT}/arm_b" \
      --device cuda
  RC_B=$?
  echo "[${TAG}] === ARM B exited rc=${RC_B} at $(date) ==="

  # =================================================================
  # Aggregation: per-clip paired comparison
  # =================================================================
  echo ""
  echo "[${TAG}] === aggregating arm_a vs arm_b per-clip ==="
  python - <<EOF
import json, pathlib
out_root = pathlib.Path("${OUT_ROOT}")
clips = "${CLIPS}".split()

def find_results_json(arm_dir, clip):
    p = arm_dir / clip
    if not p.exists():
        return None
    cands = list(p.glob("K*__ot/results.json"))
    if not cands:
        cands = list(p.glob("K*/results.json"))
    return cands[0] if cands else None

def read_jdrop(arm_dir, clip):
    f = find_results_json(arm_dir, clip)
    if f is None:
        return None
    raw = f.read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    return float(d.get("exported_j_drop") or 0.0)

arm_a = out_root / "arm_a"
arm_b = out_root / "arm_b"

per_clip = {}
n_wins = 0
deltas = []
for c in clips:
    ja = read_jdrop(arm_a, c)
    jb = read_jdrop(arm_b, c)
    delta = (jb - ja) if (ja is not None and jb is not None) else None
    if delta is not None and delta >= 0.05:
        n_wins += 1
    per_clip[c] = {"arm_a": ja, "arm_b": jb, "delta_b_minus_a": delta}
    if delta is not None:
        deltas.append(delta)
    print(f"  {c}: A={ja} B={jb} ΔB-A={delta}")

verdict = "GO" if n_wins >= 3 else "NO-GO"
mean_delta = (sum(deltas) / len(deltas)) if deltas else None
summary = {
    "per_clip": per_clip,
    "n_wins_b_over_a_by_0.05": n_wins,
    "n_clips": len(clips),
    "mean_delta_b_minus_a": mean_delta,
    "verdict": verdict,
    "decision_rule": "GO if B - A >= +0.05 on >= 3/4 clips",
}
out_root.joinpath("go_nogo_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print()
print(f"[${TAG}] PRE-REGISTERED VERDICT: {verdict}")
print(f"[${TAG}]   wins (B beats A by >= +0.05): {n_wins}/{len(clips)}")
if mean_delta is not None:
    print(f"[${TAG}]   mean (B - A): {mean_delta:+.4f}")
EOF
  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
