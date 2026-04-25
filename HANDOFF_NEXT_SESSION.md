# Handoff for Next Session — Round 5 Bundle B Sub-Session 3

**Last update**: 2026-04-25 22:30
**Last commit**: `b0967ea` — Round 5 Bundle A sub-session 2 (placement profiling + Stage 14)
**Status**: Bundle A sub-session 2 **DONE** + profile job running overnight on Pro 6000.
Bundle B sub-session 3 = NEXT (inpainting model selection + integration).

## How to start the next session

```
开始 → /clear (or new conversation)
→ Claude reads CLAUDE.md (auto)
→ Claude reads this HANDOFF + REVIEW_STATE.json → recovers context
→ Direct: "开始 Round 5 Bundle B sub-session 3"
```

Read in this order to recover context efficiently:
1. `CLAUDE.md` (auto-loaded; hard rules: No-Proxy, code-review-before-GPU, heartbeat-8min, paper-direction-decoy-only, method-effectiveness-over-reuse)
2. `HANDOFF_NEXT_SESSION.md` (this file)
3. `REVIEW_STATE.json` (round/sub-session state)
4. `AUTO_REVIEW.md` last ~250 lines (Round 5 entries — both sub-session 1 and 2)
5. `git log --oneline -10` (recent commits)
6. **First action**: SSH Pro 6000 to check Bundle A profile job status (see below).

## Bundle A status check (FIRST ACTION on next session)

```bash
ssh lvshaoting-pro6000 "cd /datanas01/nas01/Student-home/2025Lv_Zhaoting/sam2-pre-new && \
  ls vadi_runs/v5_placement_profile/ && \
  cat vadi_runs/v5_placement_profile/_summary.json 2>&1 | head -50 && \
  echo '---' && \
  tail -20 vadi_runs/v5_placement_profile/run.log && \
  echo '---HEARTBEAT---' && \
  tail -10 vadi_runs/v5_placement_profile/heartbeat.log && \
  echo '---SCREEN---' && \
  screen -ls && \
  echo '---PROC---' && \
  pgrep -af 'python.*run_placement_profile' | head -3 && \
  echo '---GPU---' && \
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

**Expected outcomes**:
- **Profile complete** (3 clips × profile.json exist + `_summary.json` populated): Bundle B can proceed to sub-session 5 pilot whenever sub-session 3-4 are done.
- **Still running**: pin progress, check ETA. Keep going on sub-session 3 (model selection — pure planning, no GPU).
- **Crashed**: investigate `run.log` last lines, fix root cause, restart. Don't paper-over with `--skip-existing` unless the partial profiles look sane.

**Restart parameters** (current — adjusted from initial 18h estimate):
- `--beam-width 4` (was 8 — reduced for ~5x cost cut)
- `--full-n-steps 30` (was 100 — reduced after measuring 32s/cheap-eval and recomputing ETA)
- Other params unchanged (cheap=12, random-baseline=10, candidates 1..T-1)
- ETA: ~19h, started 22:26:01 on 2026-04-25 → expected complete ~17:30 on 2026-04-26

## Bundle A sub-session 2 deliverables (already committed as b0967ea)

| File | LOC | Tests | Notes |
|---|---|---|---|
| `memshield/polish_gating.py` (NEW) | 674 | 8/8 ✓ | Shared preflight + accept/revert helper. PolishGatingResult dataclass with State A/B/C lifecycle. Used by both Stage 13 (refactored, zero regression) and Stage 14. |
| `memshield/v5_score_fn.py` (NEW) | 439 | 6/6 ✓ | Codex Q1c two-pass score wrapper. cheap_n_steps=12 for K=1/K=2, full_n_steps=100 for K=3 (now 30 in current run). cleanup_export_after_scoring=True bounds disk. |
| `scripts/run_placement_profile.py` (NEW) | 473 | imports OK | 4-phase profiler: raw K=1 over all candidates → beam K=3 → random K=3 baseline → persist profile.json with full beam + raw_k1_scores + run_config. |
| `scripts/run_vadi_v5.py` (MOD) | +756 net | _self_test ✓ | Added `_run_oracle_trajectory_pgd` (~330 LOC), Stage 14 outer block (twin of Stage 13), VADIv5Config oracle_traj_* fields, CLI flags + main() wiring. Stage 13 outer block REFACTORED to use polish_gating helper. |

Codex thread for Bundle A sub-session 2: **`019dc4bd-a6f6-7343-ac04-4f193040bd71`**
(may expire on MCP restart — if so, new thread with context recap is OK).

## Bundle B Sub-Session 3 = NEXT (inpainting model selection + integration)

### Goal
Replace Stage 14's current "duplicate-object Poisson blend" content compositor (`build_duplicate_object_decoy_frame` at `memshield/decoy_seed.py`) with a proper inpainting-model-based **semantic compositor**. Per CLAUDE.md No-Proxy hard rule: this is the full no-proxy upgrade Bundle B promises — must NOT silently downgrade.

### Why this is the right next step
Codex Round 5 design (`auto-review-loop` thread `019dbfe0`/`019dc4bd`) called this out as the second major improvement after trajectory oracleization (Bundle A). Current state:
- Stage 14 PGD machinery is wired (anchor + delta + α + warp + ν joint optimization). ✓
- But the duplicate frame at each bridge step is built by Poisson-blending `x_clean[c_t]`'s object crop at offset `(dy, dx)` — produces the "pasted ghost" artifact that LPIPS easily catches.
- The current `build_duplicate_object_decoy_frame` is **the proxy** flagged in CLAUDE.md No-Proxy section.

### Inpainting model selection — needs codex consult FIRST

Pre-research tradeoff matrix (drafted in sub-session 2 final discussion):

| Model | Size | Latency / call | Quality | Deploy complexity | Notes |
|---|---|---|---|---|---|
| **Stable Diffusion inpaint** | 4-7 GB | ~5 s | SOTA semantic | High (diffusers + scheduler) | Per polish: 270 calls × 5s = 22.5 min. Too heavy for VADI. |
| **LaMa** | ~200 MB | 100-200 ms | Good (purpose-built) | Light (single .pt) | 270 × 0.15s = 40s/polish. **Lead candidate.** |
| **MAT** | ~600 MB | ~500 ms | High | Medium (transformer) | 270 × 0.5s = 2.3 min/polish. Acceptable. |
| **Poisson blend (current)** | 0 | μs | Low (pasted ghost) | 0 | Status quo — the proxy we're removing. |

**User-leaned candidate**: LaMa.
**Open codex questions before code** (consult thread 019dc4bd or fresh):
1. **Which model**: LaMa vs MAT under the constraint that we make ~270 inpaint calls per polish (3 inserts × 3 bridges × 30 PGD steps) + the model is **frozen** (Bundle B does NOT fine-tune).
2. **Compositor pipeline structure**: the task is NOT "pure inpaint" — it's "compose a moved object on a frame". Decompose as:
   - (a) Inpaint to remove the original object (background reconstruction at true object's pose)
   - (b) Crop the object from its original position
   - (c) Composite the cropped object at the trajectory-decoy position
   - (d) Harmonize colors/lighting at the seam (LaB ΔE + soft alpha)
3. **Frozen vs fine-tuned**: codex Round 5 design said "frozen". Confirm + identify which intermediate buffers (object crop / background / mask) carry gradient w.r.t. anchor + delta. Currently anchor/delta gradient flows ONLY via softened decoy mask through `apply_continuation_overlay`. Bundle B should preserve this and add gradient through the **alpha-blending of the inpainted-background + composited-object**. Fully detached compositor is acceptable if the user agrees (per CLAUDE.md No-Proxy rule; record decision).
4. **Memory budget**: SAM2-tiny + LaMa frozen on Pro 6000 GPU 1 (96GB). Should fit. Confirm.
5. **Masked residual** (Bundle B sub-session 4 component): 
   - Codex Round 5 design called for "soft-mask-supported tiny global learnable residual R_k ∈ [H, W, 3]". 
   - Constrained by LPIPS ≤ 0.20 per frame.
   - Initialized to 0; gradient flows from L_margin.
   - Whether R_k is per-bridge-frame or per-clip is an open design point — codex should weigh in.

### Sub-session 3 deliverables (when codex returns verdict)

1. **`memshield/inpainter.py`** (NEW, ~200 LOC):
   - Wrapper around chosen model (LaMa most likely)
   - Lazy load + freeze + cache forward pass (object_crop_cache: f0-conditioned object crops per insert can be reused across PGD steps)
   - API: `inpaint_remove_object(frame, mask) -> frame_no_object`
2. **`memshield/semantic_compositor.py`** (NEW, ~250 LOC):
   - Replaces `build_duplicate_object_decoy_frame` for Stage 14 use
   - Pipeline: inpaint(remove) → crop(object) → composite(at trajectory_offset) → harmonize(seam)
   - Returns `[H, W, 3]` differentiable in α-blending dimension
3. **`scripts/run_vadi_v5.py` Stage 14 integration**: swap `build_duplicate_object_decoy_frame(...)` call inside `_run_oracle_trajectory_pgd` with `semantic_compositor.compose_decoy_at(traj_offset_detached, ...)`. Add config flag `oracle_traj_compositor: str = "semantic"` (vs `"poisson"` for ablation).

LOC budget: ~500-700 new + ~50 changed in v5 driver.

### Sub-session 4 = masked residual + final integration

Adds R_k learnable per-frame residual and L_residual hinge.

### Sub-session 5 = 3-clip pilot (BLOCKED on Bundle A profile completion)

Run command (when ready):
```bash
python scripts/run_vadi_v5.py \
  --davis-root data/davis \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --out-root vadi_runs/v5_bundleAB_pilot \
  --clips dog camel blackswan \
  --K 3 --placement top \
  --use-profiled-placement vadi_runs/v5_placement_profile \
  --oracle-trajectory \
  --oracle-traj-bridge-search \
  --insert-base duplicate_seed
```

**Pre-committed acceptance criterion (Round 5 6th and FINAL)**:
- mean ΔJ over A0 (~0.50) ≥ +0.05 → continue full 10-clip + ablations, paper goes
- mean ΔJ < +0.02 → cut δ permanently (5 prior δ designs failed; this is the last attempt)
- middle: Pareto synergy ablation paper

## Hard rules to remember (from CLAUDE.md)

- **No-Proxy Implementation** (2026-04-25): when implementing Bundle B, do NOT downgrade the inpainter or compositor to placeholders for LOC reasons. If a full version is too expensive, escalate to user with documented capacity loss BEFORE writing the proxy.
- **Code review before GPU deploy** (2026-04-24): any method/experiment code change MUST go through codex review BEFORE commit/push/GPU-deploy.
- **Method effectiveness > code reuse (correctness only)** (2026-04-23): if Bundle B's proper compositor needs significant rewrite, do it. Don't preserve `build_duplicate_object_decoy_frame` just because it's there.
- **Paper direction = decoy-attack, NO audit pivot** (2026-04-24): if codex pushes for audit/falsification paper, pushback. Bundle B refines decoy attack, doesn't pivot.
- **Heartbeat 8min** (2026-04-25): when launching long GPU jobs, set up parallel heartbeat screen.
- **Local Windows has no GitHub key**: push must go via Pro 6000.

## Open coordination items

- ScheduleWakeup chain from sub-session 2 may still be active — when next session opens, check `TaskList` and `CronList` for stale wake-ups; cancel via TaskStop / CronDelete if irrelevant.
- Codex thread `019dc4bd-a6f6-7343-ac04-4f193040bd71` likely expired by next session start — start fresh thread with brief context recap (Round 5 Bundle A sub-session 2 done, Bundle B sub-session 3 starts now).

## Validated baselines (don't break these)

- A0 K3_top mean J-drop: **0.48-0.51** on 10 DAVIS clips (insert-only, no δ).
- All 5 prior δ designs falsified (boundary-bridge, hiera-steering v0/v0.1, SC v0, JT v0/v1).
- Stage 14 (Bundle A only, no Bundle B): forecast 0.55-0.60 — placement profiling could lift, trajectory params have less mileage than expected because the duplicate content is the bottleneck.
- Stage 14 + Bundle B: codex forecast **0.60-0.64** (5th cut-δ override target).

## Files to know about (no need to re-read on session start unless touching them)

- `memshield/oracle_trajectory.py` — sub-session 1, FalseTrajectoryParams + helpers
- `memshield/placement_profiler.py` — sub-session 1, beam_search_K3
- `memshield/polish_gating.py` — sub-session 2, shared gating helper
- `memshield/v5_score_fn.py` — sub-session 2, beam search score wrapper
- `memshield/decoy_continuation.py` — pre-existing, Stage 13 compositor (α-overlay + warp)
- `memshield/decoy_seed.py:build_duplicate_object_decoy_frame` — **the proxy Bundle B replaces**
- `scripts/run_vadi_v5.py` — main driver, ~4150 lines after sub-session 2
- `scripts/run_placement_profile.py` — sub-session 2, profiler driver
