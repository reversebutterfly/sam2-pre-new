#!/bin/bash
# Paper main table: ours on full DAVIS-2017 val (30 clips) × R=3 replicates.
#
# Codex round 19 confirmed the right framing:
# - Main table = ours vs external baselines (clean / random / UAP-SAM2 / HRD)
# - This launcher only produces the "ours" row's data points (the rest
#   require external code adaptation, separate work stream).
# - dup arm dropped — codex confirmed dup is legacy/internal, not a baseline.
#
# Phase 1: per-clip joint_curriculum search (R=1 search, ours = poisson_hifi
#   + Stage-14 v4 search forward with bridge edits in loss). state_continuation
#   is NOT in the search forward (codex round 18 limitation; documented in
#   paper as approximate per-method W). Outputs W per clip.
#
# Phase 2: per-clip ours-eval × R=3 seeds {0,1,2} at fixed W from Phase 1.
#   Eval pipeline: poisson_hifi insert + insert_only_100 ν PGD + post-hoc
#   state_continuation 30-step bridge δ. R=3 handles cuDNN/bf16 noise
#   (147% relative range from determinism sanity check, task #55).
#
# Memory savings (codex round 19): cache=8 + expandable_segments → peak 60→46 GB.
# vLLM (other user) typically holds 40 GB on GPU 1; pilot fits in 56 GB free.
# If clips with T_clean > 80 frames OOM at peak, fall back to GPU 0 if free,
# or add memory_attention checkpointing (deferred code change).
#
# Compute estimate (Pro 6000 RTX 6000 Tiny, single GPU, serial):
#   Phase 1: 30 × ~17 min search ≈ 8.5 GPU-h
#   Phase 2: 30 × 3 × ~10 min eval ≈ 15 GPU-h
#   Total: ~24 GPU-h ≈ 1 day wall-clock
#
# Output:
#   vadi_runs/v5_main_ours/
#     search/<clip>/K*[__ot]/results.json     (Phase 1: search W + diagnostic)
#     fake_profiles/<clip>/profile.json        (clean-space W, between phases)
#     eval/seed{r}/<clip>/K*[__ot]/results.json (Phase 2: J-drop per replicate)
#     metadata.json
#     run.log

set -euo pipefail

TAG="ss7-main-ours"
PROJECT_DIR="${HOME}/sam2-pre-new"
cd "${PROJECT_DIR}"

: ${OUT_ROOT:=vadi_runs/v5_main_ours}
: ${SEEDS:="0 1 2"}
: ${EXPECTED_K:=3}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"

# Read DAVIS-2017 val clip list (30 clips, held-out from SAM2 training).
CLIPS_FILE="${PROJECT_DIR}/data/davis/ImageSets/2017/val.txt"
if [ ! -f "${CLIPS_FILE}" ]; then
  echo "[${TAG}] PRE-FLIGHT FAIL: val.txt missing at ${CLIPS_FILE}" >&2
  exit 2
fi
CLIPS=$(tr '\n' ' ' < "${CLIPS_FILE}" | sed 's/  *$//')
N_CLIPS=$(wc -w <<< "${CLIPS}")
if [ "${N_CLIPS}" != "30" ]; then
  echo "[${TAG}] PRE-FLIGHT FAIL: expected 30 val clips, got ${N_CLIPS}" >&2
  exit 2
fi

# Pre-flight: each clip's data dir exists, none in train.
for c in ${CLIPS}; do
  if [ ! -d "data/davis/JPEGImages/480p/${c}" ]; then
    echo "[${TAG}] PRE-FLIGHT FAIL: clip dir missing: data/davis/JPEGImages/480p/${c}" >&2
    exit 2
  fi
  if grep -qx "${c}" data/davis/ImageSets/2017/train.txt 2>/dev/null; then
    echo "[${TAG}] PRE-FLIGHT FAIL: clip '${c}' is in train.txt (benchmark contamination)" >&2
    exit 2
  fi
done
echo "[${TAG}] pre-flight OK: ${N_CLIPS} DAVIS-2017 val clips"

{
  echo "[${TAG}] starting at $(date)"
  echo "[${TAG}] OUT_ROOT=${OUT_ROOT}"
  echo "[${TAG}] N_CLIPS=${N_CLIPS}  SEEDS=${SEEDS}"
  echo "[${TAG}] CLIPS=${CLIPS}"
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
import torch, scipy
md = {
    "tag": "ss7-main-ours",
    "purpose": "paper main table 'ours' row (30 DAVIS-17 val × R=3)",
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_head": safe("git rev-parse HEAD"),
    "git_short": safe("git rev-parse --short HEAD"),
    "git_dirty": (safe("git status --porcelain") != ""),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "scipy_version": scipy.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
    "gpu_visible": "${CUDA_VISIBLE_DEVICES:-1}",
    "nvidia_smi": safe("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader"),
    "out_root": "${OUT_ROOT}",
    "clips": """${CLIPS}""".split(),
    "seeds": [int(s) for s in """${SEEDS}""".split()],
    "expected_K": int("${EXPECTED_K}"),
    "split": "DAVIS-2017 val (held-out, 30 clips)",
    "method": "ours = poisson_hifi insert + ν 100-step PGD + state_continuation post-hoc bridge δ",
    "phase_1_search": "joint_curriculum (Stage-14 v4 forward, hybrid carrier; state_continuation NOT in search loss — paper limitation)",
    "phase_2_eval": "insert_only_100 + delta_support=off + state_continuation, replayed at fixed W from Phase 1",
    "memory_optimizations": ["--placement-search-cache-size 8", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
    "deterministic": False,
    "allow_ghost_fallback": True,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(md, indent=2), encoding="utf-8")
print(f"[ss7-main-ours] metadata written to {out}")
EOF

  # =========================================================================
  # Phase 1: per-clip joint placement search (30 cells)
  # =========================================================================

  # Codex round 20 fix #2: strict mode (no --allow-ghost-fallback) for the
  # paper main-table row. Any cell where poisson_hifi feasibility fails
  # would otherwise silently degrade to duplicate_object — that contaminates
  # the "ours = poisson_hifi" claim. Strict = method-pure; cell crash on
  # infeasibility is recorded and the clip is excluded from the headline.
  RUN_SEARCH() {
    local clip=$1
    local cell_out="${OUT_ROOT}/search"
    rm -rf "${cell_out}/${clip}"
    mkdir -p "${cell_out}"
    echo ""
    echo "[${TAG}] === SEARCH clip=${clip} ==="
    echo "[${TAG}] start: $(date)"
    systemd-run --user --scope \
        --setenv=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
        --setenv=PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root "${PROJECT_DIR}/data/davis" \
        --checkpoint "${PROJECT_DIR}/checkpoints/sam2.1_hiera_tiny.pt" \
        --clips "${clip}" \
        --insert-base poisson_hifi \
        --oracle-trajectory --oracle-traj-v4 \
        --placement-search joint_curriculum \
        --placement-search-prescreen-seed 0 \
        --placement-search-cache-size 8 \
        --seed 0 \
        --out-root "${cell_out}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === SEARCH clip=${clip} exited rc=${rc} at $(date) ==="
    return ${rc}
  }

  # Codex round 20 fix #3: canary-first protocol. Run the 3 LONGEST DAVIS-17
  # val clips (cows 104f, parkour 100f, soapbox 99f) FIRST under strict mode.
  # If any canary fails → hard-fail (likely a config/strict-mode issue worth
  # investigating before burning the remaining 24h). If all pass → continue-
  # on-fail for the remaining 27 (transient OOMs from vLLM contention OK).
  CANARY_CLIPS="cows parkour soapbox"
  echo "[${TAG}] === Phase 1 canary: ${CANARY_CLIPS} ==="
  for clip in ${CANARY_CLIPS}; do
    if ! RUN_SEARCH "${clip}"; then
      echo "[${TAG}] FATAL: canary search failed on ${clip}. Aborting before burning 24h." >&2
      exit 3
    fi
  done
  echo "[${TAG}] canary passed (3/3). Proceeding to remaining 27 clips."

  # Remaining clips (not in canary)
  REMAINING_CLIPS=""
  for clip in ${CLIPS}; do
    if [[ "${CANARY_CLIPS}" != *"${clip}"* ]]; then
      REMAINING_CLIPS="${REMAINING_CLIPS} ${clip}"
    fi
  done

  SEARCH_FAILED=()
  for clip in ${REMAINING_CLIPS}; do
    if ! RUN_SEARCH "${clip}"; then
      SEARCH_FAILED+=("${clip}")
      echo "[${TAG}] WARN: search failed for ${clip}; continuing"
    fi
  done
  if [ ${#SEARCH_FAILED[@]} -gt 0 ]; then
    echo "[${TAG}] WARN: ${#SEARCH_FAILED[@]} non-canary search cells failed: ${SEARCH_FAILED[*]}"
  fi

  # =========================================================================
  # Inter-phase: synth fake profiles from Phase 1 search W per clip
  # =========================================================================

  echo ""
  echo "[${TAG}] === Phase 2 prep: synth fake profiles from search W ==="
  python - <<EOF
import json, pathlib, sys
root = pathlib.Path("""${OUT_ROOT}""")
profile_root = root / "fake_profiles"
profile_root.mkdir(parents=True, exist_ok=True)
clips = """${CLIPS}""".split()
val_path = pathlib.Path("""${PROJECT_DIR}""") / "data" / "davis" / "ImageSets" / "2017" / "val.txt"
val_clips = set(val_path.read_text(encoding="utf-8").splitlines()) if val_path.is_file() else set()
errs = []
for c in clips:
    base = root / "search" / c
    if not base.is_dir():
        errs.append(f"{c}: search dir missing at {base}"); continue
    cands = sorted(base.glob("K*__ot/results.json"))
    if not cands:
        cands = sorted(base.glob("K*/results.json"))
    if not cands:
        errs.append(f"{c}: no results.json under {base}"); continue
    if len(cands) > 1:
        errs.append(f"{c}: ambiguous results.json — {len(cands)} matches"); continue
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    d = json.loads(raw)
    W_att_raw = d.get("W")
    if not isinstance(W_att_raw, list):
        errs.append(f"{c}: W not a list"); continue
    try:
        W_att = sorted(int(x) for x in W_att_raw)
    except (TypeError, ValueError) as e:
        errs.append(f"{c}: W non-int: {e}"); continue
    if len(W_att) != ${EXPECTED_K}:
        errs.append(f"{c}: K={len(W_att)} != ${EXPECTED_K}"); continue
    W_clean = [w - i for i, w in enumerate(W_att)]
    if W_clean != sorted(set(W_clean)):
        errs.append(f"{c}: W_clean not strictly increasing/unique: {W_clean}"); continue
    if W_clean[0] < 1:
        errs.append(f"{c}: W_clean[0] < 1: {W_clean}"); continue
    if val_clips and c not in val_clips:
        errs.append(f"{c}: not in val.txt"); continue
    j_raw = d.get("exported_j_drop")
    j_score = 0.0 if j_raw is None else float(j_raw)
    profile = {
        "clip_name": c,
        "best": {
            "subset": W_clean,
            "score": j_score,
            "metadata": {
                "subset": W_clean,
                "K": ${EXPECTED_K},
                "source": "ss7-main-ours Phase 1 joint_curriculum search W",
                "source_results": str(cands[0].relative_to(root)),
                "W_attacked_space": W_att,
            },
        },
    }
    cp = profile_root / c
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "profile.json").write_text(json.dumps(profile, indent=2))
    print(f"  {c}: W = {W_clean}  (search J-drop={j_score:.4f})")
if errs:
    sys.stderr.write(f"FATAL ({len(errs)}):\n  - " + "\n  - ".join(errs) + "\n")
    sys.exit(2)
print(f"fake profiles written for {len(clips)} clips → {profile_root}")
EOF

  # =========================================================================
  # Phase 2: ours-eval × R=3 at fixed W (90 cells)
  # =========================================================================

  RUN_EVAL() {
    local clip=$1
    local seed=$2
    local cell_out="${OUT_ROOT}/eval/seed${seed}"
    rm -rf "${cell_out}/${clip}"
    mkdir -p "${cell_out}"
    echo ""
    echo "[${TAG}] === EVAL clip=${clip} seed=${seed} ==="
    echo "[${TAG}] start: $(date)"
    # Codex round 20 fix #2: strict mode (no --allow-ghost-fallback) for
    # eval too; method-purity gate.
    systemd-run --user --scope \
        --setenv=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
        --setenv=PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        --property=ManagedOOMMemoryPressure=auto \
        --property=ManagedOOMSwap=auto \
        python -m scripts.run_vadi_v5 \
        --davis-root "${PROJECT_DIR}/data/davis" \
        --checkpoint "${PROJECT_DIR}/checkpoints/sam2.1_hiera_tiny.pt" \
        --clips "${clip}" \
        --insert-base poisson_hifi \
        --schedule insert_only_100 --delta-support off \
        --use-profiled-placement "${OUT_ROOT}/fake_profiles" \
        --state-continuation \
        --state-continuation-min-improvement 0.02 \
        --seed "${seed}" \
        --out-root "${cell_out}" \
        --device cuda
    local rc=$?
    echo "[${TAG}] === EVAL clip=${clip} seed=${seed} exited rc=${rc} at $(date) ==="
    return ${rc}
  }

  # Phase 2 canary first (3 longest clips × 3 seeds = 9 cells), hard-fail
  # if any canary eval fails. Then continue-on-fail on the remaining 27 × 3.
  echo "[${TAG}] === Phase 2 canary eval: ${CANARY_CLIPS} ==="
  for clip in ${CANARY_CLIPS}; do
    for seed in ${SEEDS}; do
      if ! RUN_EVAL "${clip}" "${seed}"; then
        echo "[${TAG}] FATAL: canary eval failed on ${clip}/seed${seed}. Aborting." >&2
        exit 3
      fi
    done
  done
  echo "[${TAG}] eval canary passed (9/9)."

  EVAL_FAILED=()
  for clip in ${REMAINING_CLIPS}; do
    for seed in ${SEEDS}; do
      if ! RUN_EVAL "${clip}" "${seed}"; then
        EVAL_FAILED+=("${clip}/seed${seed}")
        echo "[${TAG}] WARN: eval failed for ${clip} seed=${seed}; continuing"
      fi
    done
  done
  if [ ${#EVAL_FAILED[@]} -gt 0 ]; then
    echo "[${TAG}] WARN: ${#EVAL_FAILED[@]} non-canary eval cells failed: ${EVAL_FAILED[*]}"
  fi

  # =========================================================================
  # Summary: per-clip mean ± std (R=3); aggregate across 30 clips
  # =========================================================================

  echo ""
  echo "[${TAG}] === main table 'ours' row summary ==="
  python - <<EOF
import json, math, pathlib, statistics, re
root = pathlib.Path("""${OUT_ROOT}""")
clips = """${CLIPS}""".split()
seeds = [int(s) for s in """${SEEDS}""".split()]

def read_jdrop(rel):
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
    j_raw = d.get("exported_j_drop")
    if j_raw is None:
        return None
    try:
        v = float(j_raw)
        return float("nan") if math.isnan(v) else v
    except (TypeError, ValueError):
        return None

# Forward-fill ghost counts onto clip-finish lines per (seed, clip).
log_text = ""
log_path = root / "run.log"
if log_path.is_file():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
ghost_counts = {}
cell_re = re.compile(r"=== EVAL clip=(\S+) seed=(\S+) ===")
finish_re = re.compile(r"\[v5\]\s+(\S+):\s*exported_j_drop=")
cur_clip = None; cur_seed = None; pending = 0
for line in log_text.splitlines():
    m = cell_re.search(line)
    if m:
        cur_clip = m.group(1); cur_seed = int(m.group(2)); pending = 0; continue
    if "[ghost-fallback]" in line and cur_clip is not None and cur_seed is not None:
        pending += 1; continue
    m = finish_re.search(line)
    if m and cur_clip is not None and cur_seed is not None:
        clip = m.group(1)
        if pending > 0:
            ghost_counts[(cur_seed, clip)] = ghost_counts.get((cur_seed, clip), 0) + pending
        pending = 0

print(f"\n{'clip':<22} {'mean ΔJ':>10} {'± std':>10} {'n_ok':>5} {'ghost':>7}  per-seed")
print("-" * 90)
clip_means = []
clip_stds = []
ghost_total = 0
for c in clips:
    vals = []
    per_seed = []
    g_clip = 0
    for s in seeds:
        v = read_jdrop(f"eval/seed{s}/{c}")
        per_seed.append("nan" if (v is None or math.isnan(v)) else f"{v:.4f}")
        if v is not None and not math.isnan(v):
            vals.append(v)
        g_clip += ghost_counts.get((s, c), 0)
    ghost_total += g_clip
    if not vals:
        print(f"{c:<22} {'NaN':>10} {'NaN':>10} {0:>5} {g_clip:>7}  {' '.join(per_seed)}")
        continue
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    clip_means.append(mean); clip_stds.append(std)
    print(f"{c:<22} {mean:>10.4f} {std:>10.4f} {len(vals):>5} {g_clip:>7}  {' '.join(per_seed)}")

print("-" * 90)
if clip_means:
    grand_mean = statistics.fmean(clip_means)
    grand_std = statistics.stdev(clip_means) if len(clip_means) >= 2 else 0.0
    print(f"{'AGGREGATE (n='+str(len(clip_means))+')':<22} {grand_mean:>10.4f} {grand_std:>10.4f}")
print(f"\nTotal ghost-fallback warnings across all eval cells: {ghost_total}")
print(f"Method: poisson_hifi + ν insert-only-100 + state_continuation post-hoc")
print(f"W: per-clip joint_curriculum search at seed=0")

# Codex round 20 publication gates
print("\n=== Publication gates ===")

def read_full(rel):
    base = root / rel
    if not base.is_dir():
        return None
    cands = sorted(base.glob("K*__ot/results.json"))
    if not cands:
        cands = sorted(base.glob("K*/results.json"))
    if not cands or len(cands) > 1:
        return None
    raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
    return json.loads(raw)

# Pre-load fake profiles for Gate 3 (W match check)
fake_W = {}
for c in clips:
    p = root / "fake_profiles" / c / "profile.json"
    if p.is_file():
        try:
            fake_W[c] = sorted(int(x) for x in json.loads(p.read_text(encoding="utf-8"))["best"]["subset"])
        except Exception:
            pass

n_eval_total = len(clips) * len(seeds)
n_ok = 0
n_W_match = 0
n_profiled_source = 0
n_W_check_total = 0
n_src_check_total = 0
for c in clips:
    for s in seeds:
        d = read_full(f"eval/seed{s}/{c}")
        if d is None:
            continue
        j_raw = d.get("exported_j_drop")
        if j_raw is None or (isinstance(j_raw, float) and math.isnan(j_raw)):
            continue
        n_ok += 1
        # Gate 3a: placement_source label
        ps = d.get("placement_source")
        n_src_check_total += 1
        if isinstance(ps, str) and "profil" in ps.lower():
            n_profiled_source += 1
        # Gate 3b: W (attacked-space) → clean-space matches fake profile
        W_att = d.get("W")
        if isinstance(W_att, list) and c in fake_W:
            try:
                W_att_sorted = sorted(int(x) for x in W_att)
                W_clean = [w - i for i, w in enumerate(W_att_sorted)]
                n_W_check_total += 1
                if W_clean == fake_W[c]:
                    n_W_match += 1
            except (TypeError, ValueError):
                pass

gate_complete = n_ok == n_eval_total
gate_no_ghost = ghost_total == 0
gate_src_pure = (n_src_check_total > 0 and n_profiled_source == n_src_check_total)
gate_W_match = (n_W_check_total > 0 and n_W_match == n_W_check_total)

print(f"Gate 1 (zero ghost-fallback for method purity): {'PASS' if gate_no_ghost else 'FAIL'} (count={ghost_total}, must be 0)")
print(f"Gate 2 (all {n_eval_total} eval cells produced J-drop):  {'PASS' if gate_complete else 'FAIL'} ({n_ok}/{n_eval_total})")
print(f"Gate 3a (placement_source = profiled): {'PASS' if gate_src_pure else 'FAIL'} ({n_profiled_source}/{n_src_check_total})")
print(f"Gate 3b (eval W matches fake-profile W): {'PASS' if gate_W_match else 'FAIL'} ({n_W_match}/{n_W_check_total})")
all_pass = gate_complete and gate_no_ghost and gate_src_pure and gate_W_match
print()
print(f"=== Main-table publication policy ===")
print(f"The main-table 'ours' aggregate above is publication-ready ONLY IF all 4 gates PASS.")
print(f"Subset reporting (e.g. 'on the 27/30 clips that passed') is NOT acceptable for the headline number.")
print(f"Overall publication-ready: {'YES' if all_pass else 'NO — investigate failed/contaminated clips before reporting'}")
EOF

  echo "[${TAG}] all done at $(date)"
} > "${LOG}" 2>&1
