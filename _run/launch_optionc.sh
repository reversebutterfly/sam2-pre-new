#!/bin/bash
# Option C aggregate experiment (codex round 16, 2026-04-29; round 17 fixes
# 2026-04-29). 13 clips × {dup, hybrid, ours} × R=3 replicates on SAM2.1-Tiny.
#
# Methods (all use the same per-clip W from v5_go_nogo_profiles):
#   dup    — duplicate_seed insert + ν only (insert_only_100, δ off)
#   hybrid — poisson_hifi   insert + ν only (insert_only_100, δ off)
#   ours   — poisson_hifi   insert + ν + state_continuation bridge δ
#
# Replicates: --seed in {0,1,2} (no --deterministic — natural noise across runs).
# Ghost-fallback ALLOWED for all methods so every (clip,method,seed) cell
# produces a result; the per-clip ghost_used flag is logged so the aggregator
# can report ghost-free pass rate as a separate metric.
#
# Output:
#   vadi_runs/v5_optionc/seed${r}/${method}/${clip}/K*__ot/results.json
#
# Compute estimate (Pro 6000 GPU 1):
#   13 clips × ~6 min × 3 methods × 3 seeds ≈ 12 GPU-h.
#
# Aggregator: _run/aggregate_optionc.py (paired Wilcoxon + bootstrap CI).
#
# Codex round 17 changes:
#   - set -euo pipefail; fail-fast on bad cd/source/activate
#   - hard-fail when any --use-profiled-placement clip is missing profile.json
#   - hard-fail RUN_CELL on non-zero rc (no silent continue)
#   - per-cell rm -rf BEFORE each run so __ot / __sc / base detection is
#     unambiguous on reruns
#   - method_flags returns into a bash array (no unquoted word-splitting)

set -euo pipefail

TAG="ss7-optionc"
PROJECT_DIR="${HOME}/sam2-pre-new"
cd "${PROJECT_DIR}"

: ${OUT_ROOT:=vadi_runs/v5_optionc}
: ${PROFILE_ROOT:=vadi_runs/v5_go_nogo_profiles}
: ${CLIPS:="bear blackswan bmx-trees breakdance camel cows dance-twirl dog hike horsejump-high india judo libby"}
: ${SEEDS:="0 1 2"}
: ${METHODS:="dup hybrid ours"}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"

# Pre-flight checks before we redirect to LOG so failures hit the terminal.
# Round-17 follow-up: validate profile.json *content* too — the v5 driver
# silently falls back to args.placement if best.subset is malformed or has
# the wrong K, which would change the experiment without a visible failure.
: ${EXPECTED_K:=3}
PRE_PYTHON="${HOME}/miniconda3/envs/memshield/bin/python"
if [ ! -x "${PRE_PYTHON}" ]; then
  PRE_PYTHON=python3  # fall back to whatever python3 is on PATH
fi
"${PRE_PYTHON}" - <<EOF
import json, sys, pathlib
profile_root = pathlib.Path("""${PROFILE_ROOT}""").expanduser()
clips = """${CLIPS}""".split()
expected_k = int("""${EXPECTED_K}""")
errs = []
for c in clips:
    p = profile_root / c / "profile.json"
    if not p.is_file():
        errs.append(f"{c}: missing {p}"); continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        errs.append(f"{c}: parse error in {p}: {type(e).__name__}: {e}"); continue
    subset = d.get("best", {}).get("subset")
    if not isinstance(subset, list):
        errs.append(f"{c}: best.subset not a list in {p}"); continue
    if len(subset) != expected_k:
        errs.append(f"{c}: best.subset has K={len(subset)}, expected {expected_k}"); continue
    if not all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in subset):
        errs.append(f"{c}: best.subset non-integer entries: {subset}"); continue
if errs:
    sys.stderr.write(
        "PRE-FLIGHT FAIL: profile.json content invalid for "
        f"{len(errs)} clip(s):\n  - " + "\n  - ".join(errs) + "\n"
    )
    sys.stderr.write("Hint: rerun python -m _run.synth_go_nogo_profiles\n")
    sys.exit(2)
print(f"pre-flight OK: {len(clips)} profiles validated under {profile_root} (K={expected_k})")
EOF
PRE_RC=$?
if [ ${PRE_RC} -ne 0 ]; then
  exit ${PRE_RC}
fi

{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] PWD=$(pwd) OUT_ROOT=${OUT_ROOT} PROFILE_ROOT=${PROFILE_ROOT}"
  echo "[${TAG}] CLIPS=${CLIPS}"
  echo "[${TAG}] SEEDS=${SEEDS}"
  echo "[${TAG}] METHODS=${METHODS}"
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/memshield
  echo "[${TAG}] python: $(which python)"
  echo "[${TAG}] HEAD: $(git log --oneline -1)"

  # Reproducibility metadata snapshot (codex round 17 B). Captured BEFORE
  # the first cell so a partial run still has a metadata file. Failure
  # to write metadata is non-fatal — it's diagnostic only.
  python - <<EOF || echo "[${TAG}] WARN: metadata snapshot failed (non-fatal)"
import json, pathlib, platform, socket, subprocess, sys, time
out = pathlib.Path("""${OUT_ROOT}""") / "metadata.json"
def safe(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"
import torch  # type: ignore
import scipy  # type: ignore
md = {
    "tag": "ss7-optionc",
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_head": safe("git rev-parse HEAD"),
    "git_short": safe("git rev-parse --short HEAD"),
    "git_dirty": (safe("git status --porcelain") != ""),
    "git_status_porcelain": safe("git status --porcelain"),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "scipy_version": scipy.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
    "gpu_visible": "${CUDA_VISIBLE_DEVICES:-1}",
    "nvidia_smi": safe("nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader"),
    "out_root": "${OUT_ROOT}",
    "profile_root": "${PROFILE_ROOT}",
    "clips": """${CLIPS}""".split(),
    "seeds": [int(s) for s in """${SEEDS}""".split()],
    "methods": """${METHODS}""".split(),
    "method_flags": {
        "dup":    "--insert-base duplicate_seed --schedule insert_only_100 --delta-support off",
        "hybrid": "--insert-base poisson_hifi --schedule insert_only_100 --delta-support off",
        "ours":   "--insert-base poisson_hifi --schedule insert_only_100 --delta-support off --state-continuation --state-continuation-min-improvement 0.02",
    },
    "deterministic": False,
    "allow_ghost_fallback": True,
    "expected_K": int("""${EXPECTED_K:-3}"""),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(md, indent=2), encoding="utf-8")
print(f"[ss7-optionc] metadata written to {out}")
EOF

  # Build the v5 driver flag vector for a given method label, into the
  # named array given as $2.  Using arrays avoids unquoted word-splitting.
  method_flags_into() {
    local method=$1
    local -n out=$2
    case "${method}" in
      dup)
        out=(--insert-base duplicate_seed --schedule insert_only_100 --delta-support off)
        ;;
      hybrid)
        out=(--insert-base poisson_hifi --schedule insert_only_100 --delta-support off)
        ;;
      ours)
        out=(--insert-base poisson_hifi --schedule insert_only_100 --delta-support off
             --state-continuation --state-continuation-min-improvement 0.02)
        ;;
      *)
        echo "[${TAG}] UNKNOWN METHOD: ${method}" >&2
        return 1
        ;;
    esac
  }

  RUN_CELL() {
    local seed=$1
    local method=$2
    local cell_out="${OUT_ROOT}/seed${seed}/${method}"
    # Pre-clean: stale K*__ot / K*__sc / K* dirs from prior runs would
    # confuse _read_jdrop's "first sorted glob" selection.
    rm -rf "${cell_out}"
    mkdir -p "${cell_out}"
    local flags=()
    method_flags_into "${method}" flags
    local clips_arr=( ${CLIPS} )
    echo ""
    echo "[${TAG}] === CELL seed=${seed} method=${method} ==="
    echo "[${TAG}] flags: ${flags[*]}"
    echo "[${TAG}] out:   ${cell_out}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} systemd-run --user --scope \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root "${PROJECT_DIR}/data/davis" \
        --checkpoint "${PROJECT_DIR}/checkpoints/sam2.1_hiera_tiny.pt" \
        --clips "${clips_arr[@]}" \
        "${flags[@]}" \
        --allow-ghost-fallback \
        --use-profiled-placement "${PROFILE_ROOT}" \
        --seed "${seed}" \
        --out-root "${cell_out}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === CELL seed=${seed} method=${method} exited rc=${rc} at $(date) ==="
    return ${rc}
  }

  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      RUN_CELL "${seed}" "${method}"
    done
  done

  echo ""
  echo "[${TAG}] === all cells finished, running aggregator ==="
  agg_clips_arr=( ${CLIPS} )
  agg_seeds_arr=( ${SEEDS} )
  agg_methods_arr=( ${METHODS} )
  python -m _run.aggregate_optionc \
      --out-root "${OUT_ROOT}" \
      --seeds "${agg_seeds_arr[@]}" \
      --methods "${agg_methods_arr[@]}" \
      --clips "${agg_clips_arr[@]}" \
      --baseline dup \
      --baseline hybrid \
      --target ours

  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
