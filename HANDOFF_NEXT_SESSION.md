# Handoff for Next Session — Round 5 Bundle A Sub-Session 2

**Last update**: 2026-04-25 20:30
**Last commit**: `f8518b0` — Round 5 Bundle A modules
**Status**: Sub-session 1/7 of Path 1 (full no-proxy implementation) **DONE**

## How to start the next session

```
开始 → /clear (or just open new conversation)
→ Claude reads CLAUDE.md (auto)
→ Claude reads REVIEW_STATE.json + this HANDOFF doc → recovers context
→ Direct: "继续 Round 5 Bundle A sub-session 2"
```

Claude should read in this order to recover context efficiently:
1. `CLAUDE.md` (auto-loaded; contains hard rules including No-Proxy)
2. `HANDOFF_NEXT_SESSION.md` (this file — quick recovery)
3. `REVIEW_STATE.json` (round + sub-session state)
4. `AUTO_REVIEW.md` last 200 lines (Round 5 entry)
5. `git log --oneline -8` (recent commits)
6. The two modules pushed in sub-session 1 (`memshield/oracle_trajectory.py`, `memshield/placement_profiler.py`) for direct reference

Do NOT re-read every prior round. The above is sufficient.

## Where we are

**Project**: VADI / MemoryShield SAM2 video segmentation adversarial attack
**Current phase**: Auto-review-loop Round 5 of 6, designing Path 1 (full no-proxy method) per user 6th cut-δ override

**Path 1 spec** (from codex Round 5 final design):
- Trajectory source: oracle clean future trace (publisher-side offline threat model)
- Insert scaffold: future-conditioned semantic compositor (inpainting model)
- Insert residual: soft-mask supported + tiny global branch
- Bridge editor: same compositor on bridge frames + masked residual
- Optimization: end-to-end joint
- Forecast: mean J-drop **0.60-0.64** (+0.10-0.12 over A0 ~0.50)

**3 bundles** to implement across 6 remaining sub-sessions:
- **Bundle A** (in progress): trajectory oracleization
  - ✓ Sub-session 1: pure modules (`oracle_trajectory.py`, `placement_profiler.py`) **DONE**
  - **Sub-session 2 = NEXT**: driver integration + placement profiling preprocessing
- **Bundle B** (next): semantic content upgrade
  - Sub-session 3: inpainting model selection + integration
  - Sub-session 4: semantic compositor (insert + bridge) + masked residual
  - Sub-session 5: pilot
- **Bundle C** (last): optimization cleanup
  - Sub-session 6: end-to-end joint opt + LPIPS-native ν
  - Sub-session 7: final 3-clip pilot + decision

**6th and FINAL pre-committed criterion** (after Path 1 complete):
- mean ΔJ ≥ +0.05 over A0 → continue to full 10-clip + ablation, paper goes
- mean ΔJ < +0.02 → cut δ permanently (final, project pivots to insert-only K3_top)

## What sub-session 2 needs to do (concrete)

### Goal
Integrate the Bundle A modules into the v5 driver as a Stage 14, plus a separate driver script to pre-compute placement profiling (overnight GPU run).

### Files to create

#### 1. `scripts/run_placement_profile.py` (NEW, ~400 LOC)

Preprocessing driver. For each clip:
1. Load clip, run clean SAM2 inference, get pseudo_masks
2. Get candidate frame pool (all non-f0 indices)
3. Define `score_fn(subset)`:
   - Build inserts at `subset` positions using existing v5 decoy logic
   - Run **abbreviated A0 PGD** (10 steps instead of 100) on ν
   - Export uint8 + eval J-drop on the exported video
   - Return J-drop scalar
4. Call `beam_search_K3(candidates, score_fn, beam_width=16, min_gap=2)` (codex: 16 minimum real default)
5. Also profile a random K=3 baseline (n=10 random subsets)
6. Save result to `vadi_runs/v5_placement_profile/<clip>/profile.json` (use `serialize_result`)

CLI:
```
python scripts/run_placement_profile.py \
  --davis-root data/davis \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --clips dog camel blackswan \
  --beam-width 16 \
  --inner-pgd-steps 10 \
  --out-root vadi_runs/v5_placement_profile
```

Estimated runtime: ~3 hours/clip × 3 clips = 9 hours (overnight on Pro 6000).

#### 2. `scripts/run_vadi_v5.py` Stage 14 + extensions (~300 LOC)

Add Stage 14 (oracle trajectory polish) that uses:
- `FalseTrajectoryParams` (init from W_clean_sorted, K, L=3, anchor_offsets from v5's existing decoy_offsets)
- `build_oracle_decoy_masks_for_clip` — REPLACES the existing fixed-shift `m_decoy_by_t` construction
- `select_bridge_length_per_insert` — searches L ∈ {2, 3, 4} per clip, picks best by inner-PGD score
- `trajectory_smoothness_loss` added to existing margin/fid loss
- `project_trajectory_to_budget` after each PGD step

Stage 14 integrates with existing JT v0 polish (Stage 13) by:
- If `--oracle-trajectory` flag: replaces JT's static decoy mask with oracle trajectory
- Trajectory params (anchor + delta) become learnable alongside α + warp + ν

CLI additions:
- `--use-profiled-placement` — load cached `profile.json` and use beam-search top-1
- `--oracle-trajectory` — enable Stage 14 (oracle decoy masks instead of fixed-shift)
- `--oracle-traj-max-offset-px FLOAT` (default = some reasonable bound like 200 px)
- `--oracle-traj-bridge-search` — auto-pick bridge length per clip from {2, 3, 4}

#### 3. `memshield/v5_score_fn.py` (NEW, ~150 LOC)

Helper that wraps the v5 PGD core into a `score_fn(subset_tuple) → J_drop` callable. Called by `run_placement_profile.py`. Uses `make_cached_scorer` from `placement_profiler.py` to avoid duplicate work.

### Codex pre-commit review (mandatory per CLAUDE.md)

After writing the 3 files, send to codex thread `019dbfe0-d348-7c72-8e46-ba1139e00347` (continue same thread, codex has full context). Request review on:
- Score function fidelity (abbreviated PGD vs full A0)
- Stage 14 integration correctness
- Trajectory parameter reordering when using profiled placement (caller must sort W + reorder anchor/delta consistently per oracle_trajectory.py:258 assertion)

### Then deploy

After codex GO:
1. Sync to Pro 6000
2. Run `scripts/run_placement_profile.py --clips dog camel blackswan` overnight (~9 hours, GPU 1 dedicated)
3. Next morning: pull cached profiles, examine results
4. Commit results + sub-session 2 wrap-up
5. Move to sub-session 3 (Bundle B Part 1: inpainting model selection)

## Key references (already exist)

- `CLAUDE.md` § "No-Proxy Implementation" — hard rule, do not downgrade silently
- `CLAUDE.md` § "Code Review Protocol" — codex review mandatory before deploy
- `CLAUDE.md` § "Method Design Constraints" — must keep insert + modify, no pure suppression
- `CLAUDE.md` § "Paper Direction Constraint" — decoy paper, no audit pivot
- Codex thread `019dbfe0-d348-7c72-8e46-ba1139e00347` — full context for the entire VADI design conversation
- `memshield/vulnerability_scorer.py` — current placement (will be REPLACED by profiled placement)
- `memshield/decoy_seed.py:build_duplicate_object_decoy_frame` — current insert content (to be replaced in Bundle B)
- `scripts/run_vadi_v5.py` Stage 13 (`_run_joint_trajectory_pgd`) — JT v0 polish, will be UPGRADED in Stage 14

## Validated baselines (don't break these)

- A0 K3_top mean J-drop: **0.48-0.51** on 10 DAVIS clips
- JT v0 on saturated camel: delta_overlap **+0.59**, ΔJ **+0.014** (mechanism works, J-drop bounded by saturation)
- 5 prior δ designs failed: boundary-bridge, hiera-steering v0/v0.1, SC v0, JT v0/v1
- All 5 hit structural ceiling because of f0 conditioning + fresh evidence + FIFO self-healing

## Known issues to remember

- cuDNN bf16 nondeterminism makes A0's ν* land borderline (different runs trip TV vs LPIPS gates differently). The TV-relaxed perceptual gate from Stage 13 (`_jt_perceptual_feasible`) handles this for JT — same logic should apply to Stage 14.
- dog/blackswan A0 sometimes degrades to J-drop 0.07-0.39 (noise/chaotic clips); camel is reliably 0.97-0.98.
- 3-clip apples-to-apples = dog + camel + blackswan (matches all prior pilots).

## Open questions for codex sub-session 2

1. Should `score_fn` use FULL A0 (100-step PGD) for accuracy or ABBREVIATED (10-step) for tractability? Codex Q1 in earlier review answered "low-budget surrogate is acceptable" but didn't specify N. Best ratio probably 10-15 steps for ~5% of full cost.
2. Should we cache only the BEST K=3 placement, or the full beam (top_k1/top_k2/top_k3) for ablation purposes? Cache size implications.
3. Stage 14 entry condition: is it (a) replaces Stage 13 JT entirely, or (b) extends Stage 13 with oracle trajectory params? Probably (a) for cleanest abstraction.

Resolve these with codex BEFORE writing code.
