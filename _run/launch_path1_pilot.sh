#!/bin/bash
# Path-1 pilot v2 (codex round 18 RETHINK fixes, variant 1b, 2026-04-29).
#
# 3 DAVIS-17 val clips × per-method joint placement search + ours post-hoc.
#
# Phase 1 — Search (6 cells, ~12 GPU-h):
#   for each clip × {dup, hybrid}:
#     run_vadi_v5 --oracle-trajectory --oracle-traj-v4
#                 --placement-search joint_curriculum
#                 --insert-base <c>
#     Search forward uses Stage-14 v4 (bridge edits inside) and respects
#     config.insert_base_mode → genuinely per-carrier W. Search is strict
#     ghost-free internally (build_decoy_insert_seeds_via_strategy in
#     stage14_helpers does NOT receive allow_ghost_fallback). Final attack
#     PGD at chosen W honors --allow-ghost-fallback for the hybrid carrier
#     (so we still produce a number; we log per-cell ghost-fallback count).
#
# Phase 2 — Ours eval (3 cells, ~0.5 GPU-h):
#   for each clip:
#     synth fake profile.json from hybrid's chosen W
#     run_vadi_v5 --use-profiled-placement <fake-profile-dir>
#                 --insert-base poisson_hifi
#                 --state-continuation --state-continuation-min-improvement 0.02
#                 --schedule insert_only_100 --delta-support off
#                 --allow-ghost-fallback
#     ours = hybrid-search-W + state_continuation post-hoc bridge δ.
#     This is variant 1b (codex round 18 recommended salvage path).
#
# Why we do NOT search "ours" separately:
#   joint_curriculum_search optimizes Stage-14 internal bridge edits
#   (α-warp + R residual), NOT state_continuation's teacher-aligned bridge
#   δ. With current code, "ours search" would be byte-identical to
#   "hybrid search" except for a post-hoc flag, so per-method W (gate c)
#   would fail by design. Variant 1b is the honest framing.
#
# Clips: dog (T=51), camel (T~90), libby (T=49). All DAVIS-2017 val.
#
# Pilot success gates (manual review after run):
#   (a) Search converges on all 6 search cells (no NaN W / crashes).
#   (b) Hybrid-search W has ghost-free pass rate ≥6/9 vs. dup-optimal-W's
#       2/9 in the prior optionc cell 2.
#   (c) Per-carrier W's actually differ across dup vs hybrid on ≥2/3 clips.
#   (d) ours J-drop > hybrid J-drop on ≥2/3 clips (state_continuation adds
#       value beyond hybrid carrier alone).
#
# Output (codex round 18: __ot path is conditional on Stage-14 polish
# being accepted; Phase 2 prep + summary fall back to base K*/ if not):
#   vadi_runs/v5_path1_pilot/
#     search/<method>/<clip>/K*[__ot]/results.json   (Phase 1)
#     ours/<clip>/K*[__ot]/results.json              (Phase 2)
#     fake_profiles/<clip>/profile.json              (synthesised between phases)

set -euo pipefail

TAG="ss7-path1-pilot"
PROJECT_DIR="${HOME}/sam2-pre-new"
cd "${PROJECT_DIR}"

: ${OUT_ROOT:=vadi_runs/v5_path1_pilot}
: ${CLIPS:="dog camel libby"}
: ${SEED:=0}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"

# Pre-flight: clips MUST be in DAVIS-2017 val (held-out), NOT in train.
for c in ${CLIPS}; do
  if [ ! -d "data/davis/JPEGImages/480p/${c}" ]; then
    echo "[${TAG}] PRE-FLIGHT FAIL: clip dir missing: data/davis/JPEGImages/480p/${c}" >&2
    exit 2
  fi
  if grep -qx "${c}" data/davis/ImageSets/2017/train.txt 2>/dev/null; then
    echo "[${TAG}] PRE-FLIGHT FAIL: clip '${c}' is in DAVIS-2017 TRAIN (benchmark contamination)" >&2
    exit 2
  fi
  if ! grep -qx "${c}" data/davis/ImageSets/2017/val.txt 2>/dev/null; then
    echo "[${TAG}] PRE-FLIGHT FAIL: clip '${c}' is NOT in DAVIS-2017 val.txt" >&2
    exit 2
  fi
done
echo "[${TAG}] pre-flight OK: 3/3 clips are DAVIS-2017 val (held-out)"

{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT}"
  echo "[${TAG}] CLIPS=${CLIPS}  SEED=${SEED}"
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] python: $(which python)"
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  # Reproducibility metadata snapshot.
  python - <<EOF || echo "[${TAG}] WARN: metadata snapshot failed (non-fatal)"
import json, pathlib, platform, socket, subprocess, sys, time
out = pathlib.Path("""${OUT_ROOT}""") / "metadata.json"
def safe(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"
import torch  # type: ignore
md = {
    "tag": "ss7-path1-pilot",
    "variant": "1b (search dup+hybrid, ours = hybrid-W + state_continuation)",
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_head": safe("git rev-parse HEAD"),
    "git_short": safe("git rev-parse --short HEAD"),
    "git_dirty": (safe("git status --porcelain") != ""),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
    "gpu_visible": "${CUDA_VISIBLE_DEVICES:-1}",
    "nvidia_smi": safe("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader"),
    "out_root": "${OUT_ROOT}",
    "clips": """${CLIPS}""".split(),
    "seed": int("""${SEED}"""),
    "split": "DAVIS-2017 val (held-out)",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(md, indent=2), encoding="utf-8")
print(f"[ss7-path1-pilot] metadata written to {out}")
EOF

  # =========================================================================
  # Phase 1: per-carrier joint placement search (dup + hybrid)
  # =========================================================================

  RUN_SEARCH() {
    local clip=$1
    local method=$2  # dup | hybrid
    local cell_out="${OUT_ROOT}/search/${method}"
    rm -rf "${cell_out}/${clip}"
    mkdir -p "${cell_out}"
    local insert_base
    case "${method}" in
      dup)    insert_base="duplicate_seed" ;;
      hybrid) insert_base="poisson_hifi" ;;
      *) echo "UNKNOWN SEARCH METHOD: ${method}" >&2; return 1 ;;
    esac
    echo ""
    echo "[${TAG}] === SEARCH clip=${clip} method=${method} ==="
    echo "[${TAG}] insert_base=${insert_base}  start: $(date)"
    # NOTE: --schedule / --delta-support left at Stage-14 v4 defaults (NOT
    # insert_only_100 / off) — search forward is Stage-14 v4 with bridge
    # edits, mismatching schedule_preset against insert_only_100 caused
    # the codex round 18 abort in v1 of this launcher.
    # --allow-ghost-fallback set on the FINAL attack PGD only; search
    # itself is internally strict (build_decoy_insert_seeds_via_strategy
    # in stage14_helpers does NOT thread the flag).
    local fb_flag=""
    if [ "${method}" = "hybrid" ]; then
      fb_flag="--allow-ghost-fallback"
    fi
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root "${PROJECT_DIR}/data/davis" \
        --checkpoint "${PROJECT_DIR}/checkpoints/sam2.1_hiera_tiny.pt" \
        --clips "${clip}" \
        --insert-base "${insert_base}" \
        ${fb_flag} \
        --oracle-trajectory --oracle-traj-v4 \
        --placement-search joint_curriculum \
        --placement-search-prescreen-seed 0 \
        --seed "${SEED}" \
        --out-root "${cell_out}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === SEARCH clip=${clip} method=${method} exited rc=${rc} at $(date) ==="
    return ${rc}
  }

  for clip in ${CLIPS}; do
    for method in dup hybrid; do
      RUN_SEARCH "${clip}" "${method}"
    done
  done

  # =========================================================================
  # Inter-phase: synth fake profile.json from hybrid-search W per clip
  # =========================================================================

  echo ""
  echo "[${TAG}] === Phase 2 prep: synth fake profiles from hybrid-search W ==="
  # Codex round 18 follow-up: `__ot/results.json` is only emitted when
  # Stage 14 polish is accepted; reverted/skipped Stage 14 leaves a
  # base-config `K*/results.json` instead. Try `__ot` first, fall back
  # to base config dir. Also validate fake profiles before Phase 2.
  python - <<EOF
import json, pathlib, sys
root = pathlib.Path("""${OUT_ROOT}""")
profile_root = root / "fake_profiles"
profile_root.mkdir(parents=True, exist_ok=True)
clips = """${CLIPS}""".split()
val_clips = set((root.parent.parent / "data" / "davis" / "ImageSets" / "2017" / "val.txt").read_text(encoding="utf-8").splitlines()) if (root.parent.parent / "data" / "davis" / "ImageSets" / "2017" / "val.txt").is_file() else set()
errs = []
for c in clips:
    base = root / "search" / "hybrid" / c
    if not base.is_dir():
        errs.append(f"{c}: search dir missing at {base}"); continue
    # Prefer __ot eval dir (codex-accepted Stage-14 polish), fall back
    # to base K* dir (Stage-14 reverted/skipped).
    cands = sorted(base.glob("K*__ot/results.json"))
    if not cands:
        cands = sorted(base.glob("K*/results.json"))
    if not cands:
        errs.append(f"{c}: no results.json under {base} (search crashed?)"); continue
    if len(cands) > 1:
        errs.append(f"{c}: ambiguous results.json — {len(cands)} matches: {[str(p) for p in cands]}"); continue
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    W_att_raw = d.get("W")
    if not isinstance(W_att_raw, list):
        errs.append(f"{c}: results.json has no W list (got {type(W_att_raw).__name__})"); continue
    try:
        W_att = sorted(int(x) for x in W_att_raw)
    except (TypeError, ValueError) as e:
        errs.append(f"{c}: W contains non-int: {W_att_raw} ({e})"); continue
    if len(W_att) != 3:
        errs.append(f"{c}: K={len(W_att)} but expected 3"); continue
    W_clean = [w - i for i, w in enumerate(W_att)]
    # Sanity: clean-space W must be sorted, distinct, all ≥ 0.
    if W_clean != sorted(set(W_clean)):
        errs.append(f"{c}: W_clean not strictly increasing/unique: {W_clean}"); continue
    if W_clean[0] < 1:
        errs.append(f"{c}: W_clean[0] < 1 (driver requires 1 <= c < T_clean): {W_clean}"); continue
    # Sanity: clip must be in DAVIS-2017 val (already pre-flighted but
    # double-check after search in case of typo or symlink corruption).
    if val_clips and c not in val_clips:
        errs.append(f"{c}: not in DAVIS-2017 val.txt"); continue
    j_raw = d.get("exported_j_drop")
    j_score = 0.0 if j_raw is None else float(j_raw)
    profile = {
        "clip_name": c,
        "best": {
            "subset": W_clean,
            "score": j_score,
            "metadata": {
                "subset": W_clean,
                "K": 3,
                "source": "hybrid-search W (path-1 pilot variant 1b)",
                "source_results": str(cands[0].relative_to(root)),
                "W_attacked_space": W_att,
            },
        },
    }
    cp = profile_root / c
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "profile.json").write_text(json.dumps(profile, indent=2))
    print(f"  {c}: hybrid-W = {W_clean}  (from {cands[0].relative_to(root)})")
if errs:
    sys.stderr.write(f"FATAL ({len(errs)}):\n  - " + "\n  - ".join(errs) + "\n")
    sys.exit(2)
print(f"fake profiles written to {profile_root}")
EOF

  # =========================================================================
  # Phase 2: ours eval (hybrid-W + state_continuation)
  # =========================================================================

  RUN_OURS_EVAL() {
    local clip=$1
    local cell_out="${OUT_ROOT}/ours"
    rm -rf "${cell_out}/${clip}"
    mkdir -p "${cell_out}"
    echo ""
    echo "[${TAG}] === OURS_EVAL clip=${clip} ==="
    echo "[${TAG}] start: $(date)"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root "${PROJECT_DIR}/data/davis" \
        --checkpoint "${PROJECT_DIR}/checkpoints/sam2.1_hiera_tiny.pt" \
        --clips "${clip}" \
        --insert-base poisson_hifi \
        --schedule insert_only_100 --delta-support off \
        --allow-ghost-fallback \
        --use-profiled-placement "${OUT_ROOT}/fake_profiles" \
        --state-continuation \
        --state-continuation-min-improvement 0.02 \
        --seed "${SEED}" \
        --out-root "${cell_out}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === OURS_EVAL clip=${clip} exited rc=${rc} at $(date) ==="
    return ${rc}
  }

  for clip in ${CLIPS}; do
    RUN_OURS_EVAL "${clip}"
  done

  # =========================================================================
  # Summary
  # =========================================================================

  echo ""
  echo "[${TAG}] === pilot summary ==="
  python - <<EOF
import json, pathlib, re
root = pathlib.Path("""${OUT_ROOT}""")
clips = """${CLIPS}""".split()

def read_cell(rel):
    """Return cell summary or None if no results.json exists.

    Prefers __ot/results.json (codex-accepted Stage-14 polish), falls
    back to base K*/results.json (Stage-14 reverted/skipped).
    """
    base = root / rel
    if not base.is_dir():
        return None
    cands = sorted(base.glob("K*__ot/results.json"))
    if not cands:
        cands = sorted(base.glob("K*/results.json"))
    if not cands:
        return None
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    W_att_raw = d.get("W") or []
    try:
        W_att = sorted(int(x) for x in W_att_raw)
    except (TypeError, ValueError):
        return None
    W_clean = [w - i for i, w in enumerate(W_att)] if W_att else []
    # Codex round 18 final fix: `0.0 or NaN == NaN` (0.0 is falsy in
    # Python). Explicit None check.
    j_raw = d.get("exported_j_drop")
    j_drop = float("nan") if j_raw is None else float(j_raw)
    return {
        "W_clean": W_clean,
        "W_att": W_att,
        "j_drop": j_drop,
        "source": str(cands[0].relative_to(root)),
    }

# Forward-fill ghost counts: bucket [ghost-fallback] warnings between
# successive clip-finish lines, attribute to the clip in the finish.
log_text = ""
log_path = root / "run.log"
if log_path.is_file():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
ghost_counts = {}
cur_section = None  # ("search", method) | ("ours",)
section_re = re.compile(r"=== (?:SEARCH|OURS_EVAL).*clip=(\S+).*method=(\S+) ===|=== OURS_EVAL clip=(\S+) ===")
finish_re = re.compile(r"\[v5\]\s+(\S+):\s*exported_j_drop=")
pending = 0
last_clip = None
last_section = None
for line in log_text.splitlines():
    m = re.search(r"=== SEARCH clip=(\S+) method=(\S+) ===", line)
    if m:
        last_section = ("search", m.group(2)); last_clip = m.group(1); pending = 0; continue
    m = re.search(r"=== OURS_EVAL clip=(\S+) ===", line)
    if m:
        last_section = ("ours",); last_clip = m.group(1); pending = 0; continue
    if "[ghost-fallback]" in line and last_section is not None:
        pending += 1; continue
    m = finish_re.search(line)
    if m and last_section is not None:
        clip = m.group(1)
        if pending > 0:
            key = (last_section, clip)
            ghost_counts[key] = ghost_counts.get(key, 0) + pending
        pending = 0

print(f"\n{'phase':<10} {'clip':<14} {'W (clean)':<22} {'J-drop':>8} {'ghost-fb':>10}")
print("-" * 72)
rows = []
for c in clips:
    for method in ("dup", "hybrid"):
        cell = read_cell(f"search/{method}/{c}")
        if cell is None:
            print(f"{'search/'+method:<10} {c:<14} {'<missing>':<22} {'NaN':>8} {'?':>10}")
            continue
        gh = ghost_counts.get((("search", method), c), 0)
        print(f"{'search/'+method:<10} {c:<14} {str(cell['W_clean']):<22} {cell['j_drop']:>8.4f} {gh:>10}")
        rows.append((method, c, cell['W_clean'], cell['j_drop'], gh))
    cell = read_cell(f"ours/{c}")
    if cell is None:
        print(f"{'ours':<10} {c:<14} {'<missing>':<22} {'NaN':>8} {'?':>10}")
    else:
        gh = ghost_counts.get((("ours",), c), 0)
        print(f"{'ours':<10} {c:<14} {str(cell['W_clean']):<22} {cell['j_drop']:>8.4f} {gh:>10}")
        rows.append(("ours", c, cell['W_clean'], cell['j_drop'], gh))

# Pilot gates
print()
print("=== Pilot success gates ===")
search_dup = {r[1]: r for r in rows if r[0] == "dup"}
search_hyb = {r[1]: r for r in rows if r[0] == "hybrid"}
ours_eval = {r[1]: r for r in rows if r[0] == "ours"}

# Gate (a): explicit per-cell completion (codex round 18 fix)
gate_a_misses = []
for c in clips:
    for tag, table in (("dup", search_dup), ("hybrid", search_hyb), ("ours", ours_eval)):
        if c not in table:
            gate_a_misses.append(f"{tag}/{c}")
print(f"(a) all 9 cells produced results.json: {'YES' if not gate_a_misses else 'NO  missing=' + ','.join(gate_a_misses)}")

# Gate (b): aggregate + per-clip floor (codex round 18 follow-up)
hybrid_ghost_per_clip = {c: int(search_hyb[c][4] or 0) for c in clips if c in search_hyb}
total_inserts = 9  # 3 clips × 3 inserts
total_fb = sum(hybrid_ghost_per_clip.values())
total_pass = total_inserts - total_fb
worst_clip_pass = min((3 - n) for n in hybrid_ghost_per_clip.values()) if hybrid_ghost_per_clip else 0
print(f"(b) hybrid-W ghost-free aggregate: {total_pass}/{total_inserts} (need >=6)")
print(f"(b) hybrid-W ghost-free worst per-clip: {worst_clip_pass}/3 (need >=1, codex round 18 floor)")

# Gate (c): clean-space W comparison
diffs = sum(1 for c in clips
            if c in search_dup and c in search_hyb
            and search_dup[c][2] != search_hyb[c][2])
print(f"(c) per-carrier W differs on: {diffs}/{len(clips)} clips (need >=2)")

# Gate (d): ours > hybrid on majority
ours_wins = sum(1 for c in clips
                if c in ours_eval and c in search_hyb
                and ours_eval[c][3] > search_hyb[c][3])
print(f"(d) ours > hybrid J-drop on: {ours_wins}/{len(clips)} clips (need >=2)")
EOF

  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
