# Handoff for Next Session — Sub-Session 7: Joint Curriculum Placement Search

**Last update**: 2026-04-26 13:15 (after auto-review-loop R6 design completion)
**Last commit**: `c8b30db` — Round 5 Bundle B sub-session 6 (Bundle C C2 LPIPS-native ν)
**Status**: Bundle B sub-sessions 3+4+6 ALL DONE + dog 1-clip pilot landed +0.171 ΔJ + auto-review-loop R6 design APPROVED. Sub-session 7 = NEXT (joint curriculum placement search implementation).

## How to start the next session

```
开始 → /clear (or new conversation)
→ Claude reads CLAUDE.md (auto)
→ Claude reads this HANDOFF + REVIEW_STATE.json + AUTO_REVIEW.md (last 350 lines for R6 spec)
→ Direct: "开始 sub-session 7：实施 joint curriculum placement search"
```

Read in this order to recover context efficiently:
1. `CLAUDE.md` (auto; hard rules: No-Proxy, code-review-before-GPU, heartbeat-30min-now, paper-direction-decoy-only, method-effectiveness-over-reuse)
2. `HANDOFF_NEXT_SESSION.md` (this file)
3. `REVIEW_STATE.json` (round/sub-session state)
4. `AUTO_REVIEW.md` last ~350 lines — focus on **Round 6 (auto-review-loop)** entry which has the FULL approved spec
5. `ABLATION_DESIGN_REVIEW_2026-04-26.md` — ablation matrix planning (orthogonal to ss7 but informs validation)
6. `git log --oneline -8` (recent commits)
7. **First action**: SSH Pro 6000 to check GPU 1 brute-force profile status (still running for oracle calibration)

## Approved design (auto-review-loop R6 final spec)

**Topic**: Joint curriculum placement-perturbation optimization. Replaces brute-force `beam_search_K3` (9h/clip) with discrete-schedule-interpolated joint optimization (~63 min/clip target, codex says budget 2-4h/clip until proven).

**Core mechanism**:
1. **Continuous learnable τ ∈ R³** (placement positions in clean-space) parameterized **by ordered gaps**:
   ```
   tau[0] = clamp_left + (T - bridge_budget - clamp_left) * sigmoid(g0)   # ≥ 1 by construction
   tau[1] = tau[0] + d_min + softplus(g1)                                 # > tau[0] + d_min
   tau[2] = tau[1] + d_min + softplus(g2)                                 # > tau[1] + d_min
   ```
   d_min ≥ 2 (or 3); bridge_budget = bridge_length + 1 to keep right margin.

2. **Discrete schedule interpolation as timing surrogate** (NOT soft frame averaging — that was R2 NO-GO):
   - Per joint step, enumerate `2^K` neighbor schedules using `floor(τ[k])` and `ceil(τ[k])` per active insert.
   - For each schedule, weight = `product over k of (1 - frac[k])` if floor else `frac[k]`.
   - Filter invalid schedules (those violating strict ordering c[0] < c[1] < c[2]); renormalize remaining weights.
   - Loss = weighted sum of per-schedule Stage 14 forward losses → real differentiable timing signal.

3. **Single-level joint loop** (NOT bilevel; each step does ONE Stage 14 forward per schedule, NOT 30 inner steps):
   ```python
   opt = Adam([g_active, anchor, delta, alpha_logits, warp_s, warp_r, nu_active])  # active per K phase
   for step in curriculum_steps:
       L_step = 0
       valid_corner_count = 0
       valid_weight_mass = 0
       for schedule_W, weight in enumerate_neighbor_schedules(tau, active_inserts, T):
           attack_state = build_attack_state_from_W(schedule_W, x_clean, pseudo_masks, config)
           L_schedule, _ = stage14_forward_loss(attack_state, anchor, delta, alpha, warp, R, nu)
           L_step += weight * L_schedule
           valid_weight_mass += weight; valid_corner_count += 1
       L_step = L_step / valid_weight_mass
       L_step.backward()
       opt.step()
       sign_pgd_update_R_active_slices_only()
       log(valid_corner_count, valid_weight_mass)        # MANDATORY guardrail #2
       if valid_corner_count < 2: project_tau_inward()    # MANDATORY guardrail #2 fallback
   ```

4. **Curriculum** (rebuild optimizer at each transition, NOT just grad zeroing):
   - K=1 phase: 12 steps; only g0, anchor[0], delta[0,:], α[0,:], warp[0,:], R[0,:], ν[0] active. g1, g2 frozen → τ[1], τ[2] held at prescreen init.
   - K=2 phase: 12 steps; activate g1 + insert-1 params.
   - K=3 phase: 15 steps; activate g2 + insert-2 params (full joint).

5. **Prescreen for τ initialization** (1 fwd × ~100 frames, ~5min):
   - For each candidate c, build attack_state with K=1 at c (cheap defaults), 1 forward, score = `-L_margin`.
   - Pick top-K with d_min spacing → init τ values via inverse-sigmoid / inverse-softplus.

6. **Hardening + local refine**:
   - Round τ → discrete c_k = round(τ[k])
   - Enumerate **27 = 3³ joint** ±1 neighbor triples (not 9 independent), filter invalid ordering
   - For each valid triple: 6-step cheap Stage 14, measure J-drop estimate
   - Pick best by J-drop estimate → final triple

7. **Final 30-step Stage 14** on chosen triple (existing code path) → deployment-ready output.

## 6 mandatory implementation guardrails (codex R3 GO conditions)

1. **τ[0] ≥ 1 by construction** (NOT penalty). Bridge_budget margin built into right boundary too.
2. **Log `valid_corner_count` + `valid_weight_mass` every step**. If `valid_corner_count < 2` or weight mass too small → project τ inward before enumeration, OR fallback to single hard schedule (acceptable but record in log).
3. **Bundle C OFF for joint-search v1**: explicit guard `if --placement-search joint_curriculum and --oracle-traj-nu-lpips-native: raise`. LPIPS-native ν line-search assumes fixed W; multi-schedule re-anchor is out of scope.
4. **Wall-clock instrument from day 1**: don't trust 63 min/clip estimate. Budget first run as 2-4h/clip until proven.
5. **Fixed-W parity regression test (MANDATORY, blocks code review)**: existing `_run_oracle_trajectory_pgd(W_fixed)` ≡ `joint_loop_with_W_fixed_to_W_fixed` within tolerance (e.g., final J-drop within ±0.005, all per-step diagnostics within ±1%). Without this, can't isolate "is the new search bad" vs "did the helper extraction break Stage 14".
6. **Multi-seed fallback predeclared**: if dog J-drop > 0.03 below brute-force (0.667 → < 0.637), immediately rerun with **2 more prescreen seeds** (top-2 and top-3 candidates as alternative inits) before declaring design failure.

## Mandatory minimum tests (codex R3)

In `memshield/joint_placement_search.py` self-tests:
- **Fixed-W parity test** (MANDATORY): old wrapper vs new helpers, same clip, same seed, same final state within tolerance.
- **Schedule weight sum test**: `enumerate_neighbor_schedules(tau, [0,1,2], T)` weights sum to 1.0 (modulo invalid filtering); exactness at integer τ.
- **Ordered-τ legality**: monotonicity τ[0] < τ[1] < τ[2], left boundary τ[0] ≥ 1, right boundary τ[2] + bridge_length ≤ T-1.
- **Joint-loop smoke test** through K1/K2/K3 transitions: optimizer rebuild correct, R active-slice mask correct, no crash.
- **R active-slice mask test**: inactive slices unchanged after sign-PGD step.
- **Bundle C incompatibility guard test**: passing `--placement-search joint_curriculum --oracle-traj-nu-lpips-native` raises clear error.

## Refactoring plan (R3 codex Q6: minimal shared extraction)

Extract from `scripts/run_vadi_v5.py:_run_oracle_trajectory_pgd`:
1. `build_attack_state_from_W(W_clean, x_clean, pseudo_masks, config) -> AttackState` (TypedDict)
   - Contains: W_attacked (sorted), decoy_seeds, decoy_offsets, bridge_frames_by_k, m_true_by_t, m_decoy_by_t, clean_refs_for_inserts, oracle_decoy_masks_clean (whatever's W-dependent)
2. `stage14_forward_loss(attack_state, anchor, delta, alpha_logits, warp_s, warp_r, R, nu, config) -> (L_total, diagnostics)`
   - Runs ONE Stage 14 forward + computes loss (not 30 inner steps)

Existing `_run_oracle_trajectory_pgd` becomes thin wrapper:
- Calls `build_attack_state_from_W(W_clean_input)` once
- Loops 30 steps of `stage14_forward_loss` + sign-PGD on R
- Same outputs as before (zero-regression goal)

Both old (`_run_oracle_trajectory_pgd`) and new (`joint_curriculum_search`) paths use the SAME helpers. Fixed-W parity test verifies equivalence.

## Driver integration

Add CLI flag:
```python
p.add_argument("--placement-search", choices=["top", "joint_curriculum", "off"], default="off",
               help="Algorithmic placement search. off=use --use-profiled-placement or --placement. "
                    "joint_curriculum=auto-discover via joint placement-perturbation optimization.")
```

Mutex guard:
```python
if args.placement_search == "joint_curriculum" and args.oracle_traj_nu_lpips_native:
    raise ValueError("joint_curriculum + LPIPS-native ν not supported in v1")
```

## Dog parity test (R4 success criterion)

```bash
ssh lvshaoting-pro6000 "cd <repo> && \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vadi_v5.py \
    --davis-root data/davis --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --out-root vadi_runs/v5_pilot_ss7_dog_joint \
    --clips dog --K 3 --placement off --placement-search joint_curriculum \
    --oracle-trajectory --insert-base duplicate_seed"
```

**Pass criterion**: Stage 14 J-drop ≥ 0.637 (within 0.03 of brute-force 0.667).
**Fallback**: if < 0.637, rerun with multi-seed prescreen.
**Fail criterion**: < 0.55 (more than 0.10 below brute-force) → call design failed, regroup.

## Implementation order (sequential)

1. Refactor: extract `build_attack_state_from_W` + `stage14_forward_loss` from `_run_oracle_trajectory_pgd`. Self-test: existing dog 1-clip pilot reproduces J-drop 0.667 within tolerance.
2. Codex code review of refactor (no algorithm change).
3. New module: `memshield/joint_placement_search.py` with all spec components.
4. Tests: 6 mandatory self-tests pass.
5. Codex code review of new module.
6. Driver flag + main wiring + mutex guard.
7. Sync + commit (refactor + new module + driver) on Pro 6000.
8. Dog parity test on GPU 0 (GPU 1 still running brute-force for oracle).
9. If pass → continue to camel + blackswan; if fail → multi-seed retry; if still fail → regroup.

LOC budget: ~700-900 new + ~200 refactored from existing.

## Compute resources state

- **Pro 6000 GPU 0**: free (after killing brute-force at 13:00). Available for joint-search experiments.
- **Pro 6000 GPU 1**: dog/camel/blackswan brute-force profile continues. ETA ~22:00-04:00 today/tomorrow. Used as oracle calibration:
  - dog: profile already complete (best=[8,18,32], score=0.7153)
  - camel: K=2 phase ongoing (started 08:39)
  - blackswan: not yet started

- **Lab policy**: max 3 concurrent jobs per user; currently 1 (GPU 1 profile). Can add 1-2 more on GPU 0 (joint search experiments).

## Pre-committed criteria (still active)

- **Round 5 6th-and-FINAL δ criterion**: mean ΔJ ≥ +0.05 over A0 on full 10-clip eval → continue + paper goes; < +0.02 → cut δ permanently. Joint placement search redesigns the SEARCH, not the success criterion.
- **Joint search dog parity criterion**: J-drop within 0.03 of brute-force.
- **Wall-clock budget**: 2-4h/clip first pass, optimize down later if convergence permits.

## Files to know about

- `memshield/oracle_trajectory.py` — sub-session 1 trajectory primitives
- `memshield/placement_profiler.py` — brute-force search (KEEP for oracle calibration; not deleted)
- `memshield/polish_gating.py` — Stage 13 + Stage 14 shared gating helper
- `memshield/v5_score_fn.py` — score wrapper used by placement_profiler (REVIEW: may need similar wrapper for joint search)
- `memshield/decoy_continuation.py` — apply_continuation_overlay + apply_translation_warp_roi (Stage 14 inner)
- `memshield/decoy_seed.py:build_duplicate_object_decoy_frame` — legacy compositor (kept as ablation `--oracle-traj-compositor poisson`)
- `memshield/semantic_compositor.py` — alpha_paste + apply_masked_residual + find_max_feasible_nu_scale (Bundle B + Bundle C helpers)
- `scripts/run_vadi_v5.py` — main driver (~4400 lines after sub-session 6); `_run_oracle_trajectory_pgd` at line ~1932 is the refactor target
- `scripts/run_placement_profile.py` — brute-force profiler driver (KEEP; standalone for oracle runs)

## Codex thread

`019dc51a-c71a-7971-bece-116a592de2f5` (gpt-5.4 xhigh, Bundle B + C + auto-review-loop R6) — may expire on MCP restart. Fresh thread OK with brief context recap.

## Validated baselines (don't break these)

- A0 K3_top mean J-drop: **0.48-0.51** on 10 DAVIS clips (insert-only, no Stage 14)
- Dog brute-force profile + Stage 14 (Bundle A+B): **J-drop 0.6665, ΔJ +0.171** (this is the oracle to match)
- Bundle B + C self-tests: 17/17 pass on Pro 6000

## What NOT to do

- Don't delete `memshield/placement_profiler.py` or `scripts/run_placement_profile.py` — kept for oracle calibration + ablation
- Don't break `_run_oracle_trajectory_pgd` for the existing `--use-profiled-placement` path (must remain bit-equivalent within tolerance after refactor)
- Don't enable Bundle C (LPIPS-native ν) with joint search in v1 — explicit guard required
- Don't trust 63 min/clip estimate without instrumentation
- Don't use STE through `round(τ).int()` for indexing — use the schedule interpolation surrogate instead

## Hard rules from CLAUDE.md (re-iterate)

- **Code review before GPU deploy**: any method change → codex review → only then commit + push + run
- **Heartbeat cadence**: now **30 min** per user override (was 8 min)
- **No-Proxy**: don't downgrade joint-search components for LOC reasons
- **Method effectiveness > code reuse (correctness only)**: refactor is correctness-required, do it
- **Local Windows has no GitHub key**: push via Pro 6000

## Session-7 entry checklist

- [ ] Read this HANDOFF + AUTO_REVIEW R6 + REVIEW_STATE
- [ ] Confirm GPU 0 free + GPU 1 still profiling
- [ ] Confirm latest commit is `c8b30db`
- [ ] Start with refactor (extract helpers + parity test)
- [ ] Then new joint_placement_search.py module
- [ ] Codex review at each major commit point
- [ ] Dog parity test on GPU 0
- [ ] Update REVIEW_STATE.json + AUTO_REVIEW.md after each milestone
