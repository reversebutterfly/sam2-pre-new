# Joint > Only — Optimization Path (codex round 5, 2026-04-28)

## TL;DR

**Verdict: "Joint > Only" IS salvageable**, but only if the joint formulation changes. **Free joint ν+δ optimization is NOT the path.** The defensible path:

1. **A0 = strong insert-only base attack** (ν optimized alone, no δ)
2. **δ = differential bridge enhancer** applied AFTER A0, with ν frozen
3. **Accept bridge only if it beats A0** on exported eval (no-regression guard)

Bridge δ gets a **specific role insert-only cannot do**: align bridge-frame state (maskmem_features + obj_ptr) with **cached insert-state teachers**. ν cannot act on real frames after the insert; bridge δ can. This makes joint > only structurally guaranteed under the right loss.

**Crucially**: most of the required code already exists in the repo. This is a **wiring-up job, not a new-implementation job**.

---

## What's already in the codebase

| Component | Location | Status |
|-----------|----------|--------|
| Clean A0-only placement profiling (no δ, no polish) | `memshield/v5_score_fn.py:102` | EXISTS |
| Profile-based placement runner | `scripts/run_placement_profile.py` | EXISTS |
| No-regression bridge accept/reject for state_continuation | `scripts/run_vadi_v5.py:3447` | EXISTS |
| Bridge loss aligning to cached insert-state teachers | `scripts/run_vadi_v5.py:1365` | EXISTS |
| **`create_decoy_base_frame_hifi(...)`** — Poisson-blend hi-fi insert that inpaints original object position before pasting new object | **`memshield/decoy.py:292`** | **EXISTS — directly attacks ghosting** |
| ProPainter middle-frame constructor (backup ghost-fix) | `memshield/propainter_base.py:425` | EXISTS |
| Old joint placement search (with suffix probe overlap problem) | `memshield/joint_placement_search.py:1350` | TO DEMOTE |

## Specific code changes (codex round 5)

1. **Add insert-base-mode option**:
   - In `scripts/run_vadi_v5.py:2677` and `memshield/stage14_helpers.py:233`, expose `insert_base_mode = {"duplicate_seed", "midframe", "poisson_hifi", "propainter"}` instead of just the first two.
   - Wire `poisson_hifi` → `memshield/decoy.py:292` (`create_decoy_base_frame_hifi`).
   - Wire `propainter` → `memshield/propainter_base.py:425`.

2. **Promote profiled placement as default**:
   - Use `scripts/run_placement_profile.py` + `memshield/v5_score_fn.py:102`.
   - Demote `memshield/joint_placement_search.py:1350` from the main method (keep as ablation only).
   - **No suffix probe** in main pipeline.

3. **Promote `state_continuation` bridge stage**:
   - Set `state_continuation_off_switch=True`.
   - Set positive `state_continuation_min_improvement` (NOT 0.0) so bridge is rejected if marginal/negative.
   - Replace oracle-trajectory bridge overlays with state-continuation as the default bridge stage.

4. **Decouple optimization order**:
   - First optimize ν only (insert-only A0 attack).
   - Freeze ν.
   - Then optimize δ on bridge frames only, with `state_continuation` loss.

## Loss decomposition (proposed)

### ν loss (stage 1, frozen after this)
- Standard insert-only decoy loss on insert frames + immediate post-insert behaviour
- **No bridge terms**
- Objective: create the strongest possible A0 insert-only teacher

### δ loss (stage 2, ν frozen)
- **`L_state`**: align bridge-frame `maskmem_features` + `obj_ptr` with cached insert-time teacher states (this is the **non-redundant role** ν cannot perform)
- **`L_margin_bridge`**: attack margin only on bridge frames where A0 rebounds (differential targeting)
- **`L_fid_bridge`**: LPIPS / ε constraints on bridge frames only

### Joint regularizer
- None needed (ν is frozen)
- Accept/reject against A0 as no-regression guard

## Placement metric (codex strong recommendation)

- **Use**: profiled A0 insert-only exported J-drop
- **Do NOT use**: suffix probe (overlapping per round 4)
- **Do NOT use**: old clean-video vulnerability score as primary selector

This eliminates the methodological overlap from `SUFFIX_PROBE_OVERLAP_2026-04-28.md`.

## Ghost-fix ranking (codex round 5)

| Option | Quality | Compute | Recommendation |
|--------|---------|---------|----------------|
| **B1.1 + B1.4: poisson_hifi (object-removal composite + alpha-matted paste)** | ★★★★ | low | **PRIMARY** — already in `decoy.py:292` |
| ProPainter middle-frame base (`ICCV 2023`) | ★★★★ | medium | secondary if poisson_hifi insufficient |
| Interpolation + ν' steering | ★★★ | medium | backup only (round 3 risk: may collapse attack) |
| Diffusion infill | ★★★★★ | high | last resort |

Internal evidence: insert LPIPS improving from 0.161 → 0.134 dropped attack from 0.536 → 0.334-0.485 (`AUTO_REVIEW.md:843`). So **higher fidelity DOES cost some J-drop**, which is acceptable per PI directive.

## A3 mechanism evidence transfer

- Old A3 (4/4 STRONG on oracle composite carrier) **transfers MORE cleanly to poisson_hifi than to interpolation** because the insert is still a strong wrong-evidence frame, just without the original-position residual.
- Still need a **fresh A3 on poisson_hifi carrier** for paper main result.
- Old A3 → motivation in appendix only.

## Realistic AAAI mock score (codex)

If we hit:
- Mean ΔJ ≈ 0.30 (vs old 0.746)
- Paired joint wins ≥ 8/13
- Ghost-free flipbook pass-rate ≥ 70%
- Fresh A3 transfer

→ AAAI prior ≈ **6/10, borderline but credible**.

This is **stronger than the old visible 0.746 story** even with lower attack magnitude:
- Paired joint > only is defensible (per-clip, not aggregate)
- Visuals don't kill the paper
- Bridge δ has scientifically clean role

## GO/NO-GO test (smallest possible)

**4 clips**: camel, dog, breakdance, libby
**Arms**:
- A: A0 ghost-free (poisson_hifi insert + ν only, no δ)
- B: A0 ghost-free + state_continuation bridge δ (decoupled optimization)

**Success bar**:
- B improves over A by ≥ +0.05 on **3/4 clips**
- Inserts visibly cleaner than current oracle-composite (subjective check)

**Compute**: ~6-10 GPU-h on V100.

**If FAIL** → next-best fallback is publisher-side stealth pivot (CHRONOCLOAK_FEASIBILITY_2026-04-28.md). The OLD attacker-side story does not tolerate lower J-drop; the publisher-side does.

## Full experimental sequence (if GO)

| Stage | Clips | Compute | Purpose |
|-------|-------|---------|---------|
| **GO/NO-GO** | 4 (camel/dog/breakdance/libby) | 6-10 GPU-h | Decision gate |
| Paired benchmark | 13 (full set) | 30-50 GPU-h | Main result table: A0 vs joint per-clip |
| Role ablation | 6 | 15-25 GPU-h | A0 / A0+old-free-bridge / A0+state-continuation-bridge |
| Fresh A3 (poisson_hifi carrier) | 4-6 | 10-20 GPU-h | Mechanism transfer |
| Human stealth flipbook | all 13 inserts × 3 raters | 0 GPU | Visual claim |
| **Total** | 60-95 GPU-h | within 3 V100-weeks |

## Defensible AAAI claims under this design

What we CAN claim:
- ✓ "Per-clip paired comparison: joint (A0 + state-continuation bridge) outperforms insert-only A0 on N/13 clips"
- ✓ "Bridge δ has a non-redundant role: aligning bridge-frame memory state with cached insert-state teachers — a perturbation surface ν cannot reach"
- ✓ "Ghost-free insert via Poisson hi-fi composite preserves attack while eliminating visible synthesis artifacts"
- ✓ Fresh A3 on poisson_hifi carrier supports memory-write causality at insert positions
- ✓ "Sacrificing ~50% of J-drop magnitude for clean visuals + defensible joint claim is a worthwhile tradeoff"

What we still CANNOT claim (per round 4):
- ❌ "Joint > Only" in general (only under our specific decoupled formulation)
- ❌ "Bridge δ extends decoy effect onto subsequent frames" (it's a state aligner, not extender)

## Risk analysis

### Highest risk failure mode
Ghost-free insert base weakens A0 too much → cached insert-state teachers are weaker → δ gains sparse + inconsistent → off-switch keeps reverting to A0 → end up with clean visuals but no majority joint wins.

### Medium risk
Paired wins may be 5-7/13 (not the 8/13 target). Borderline AAAI: 5.5/10 instead of 6/10.

### Low risk
poisson_hifi already exists and was probably tested at some point. Wiring it in as `insert_base_mode=poisson_hifi` is a small code change.

## Codex thread

`019dd243-04d2-7111-9eb0-c4eb3fec729d`, round 5, 2026-04-28.
