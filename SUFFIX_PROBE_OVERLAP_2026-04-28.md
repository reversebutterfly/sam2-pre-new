# Suffix Probe vs Bridge δ — Methodological Overlap (codex round 4, 2026-04-28)

## TL;DR

**Verdict: OVERLAPPING (literal, not philosophical).**

Suffix probe and bridge δ are structurally entangled in the v4.1 codebase. The user's diagnostic question was directionally correct, with one key correction: suffix probe is NOT a decoy-only metric — it's a **joint-aware** (or bridge-aware nu-free) proxy that uses the **same bridge-edit machinery** that Stage 14 later optimizes.

This explains the apparent 10-clip "joint hurts" result and exposes a methodological vulnerability in the OLD framing fallback paper.

## Two precise pieces of evidence from our own code

1. **Suffix range definition** (`stage14_helpers.py:512`, `joint_placement_search.py:1560`):
   - "Suffix" frames span from `w_first + 1` to end of processed video, excluding only insert positions.
   - **NOT** "only frames after the last insert."
   - This means suffix-probe loss lands directly inside bridge windows of LATER inserts. The probe is scoring on the same frames bridge δ later modifies.

2. **Probe execution path** (`stage14_helpers.py:726, 941`):
   - Suffix probe is computed inside `stage14_forward_loss`.
   - That function builds edited bridge frames from `traj + alpha/warp + R` and inserted frames from `decoy_seeds + nu`.
   - **Suffix probe is scoring the effect of the same bridge-edit machinery that Stage 14 later optimizes.**

## Probe variants (actual implementation)

- **Phase-3 curriculum** (`joint_placement_search.py:1529, 1547, 1610`): joint-aware. Optimizer includes `tau, traj, edit_params, nu` and `R` (in last phase if residuals enabled).
- **Local refine 27** (`joint_placement_search.py:1252, 1312`): bridge-aware, nu-free proxy. 6-step mini Stage-14 over `traj + alpha/warp` with `nu=0` and `R` inactive, scored by `-(λ_margin·L_margin + λ_suffix·L_suffix)`.

So suffix probe is **definitely not a clean vulnerability-window selector** in the spirit of "where could a decoy alone propagate?" It is a placement selector that reuses the downstream attack mechanism.

## Why this explains 10-clip vs v4.1 discrepancy

| Pipeline | Bridge role | Result |
|---|---|---|
| 10-clip decisive (older `K3_top`) | naive generic δ+ν, no anchored loss | bridge δ is **net hurt** (−0.156 pp vs insert-only) |
| 13-clip v4.1 RAW joint | A0-anchored polish: `L_keep_full` + `L_keep_margin` + `L_gain_suffix`, bridge-local | bridge δ is **stabilizer**, mean ΔJ = 0.746 (vs insert-only ~0.537) |

**The honest causal reading**: v4.1's anchored Stage 14 changed bridge from "free second attack arm" to "anchored, bridge-local non-regressing stabilizer." This is a fundamentally different mechanism from the original "joint" idea. We should not claim "joint > insert-only" in general — that statement only holds under v4.1's specific anchored loss.

## Implications for paper claims

### What we CANNOT claim
- ❌ "Joint insert + δ outperforms insert-only" (in general)
- ❌ "Bridge δ extends decoy effect onto subsequent frames" (suggests workhorse role; not supported)
- ❌ "Placement search finds insertion vulnerability windows" (placement search is bridge-aware, not vulnerability-aware)

### What we CAN claim (narrower)
- ✓ "Naive joint δ+ν hurt vs insert-only in our 10-clip ablation"
- ✓ "An anchored, bridge-local Stage-14 polish improves over insert-only on some clips (6 RAW joint mean 0.746 vs old insert-only 0.537)"
- ✓ "Bridge δ functions as a stabilizer under the v4.1 anchored loss, not as an efficacy booster"
- ✓ A3 mechanism evidence (4/4 STRONG): the **insert-position memory write** is the causal point. Bridge δ position memory blocking was never separately tested.

## Codex's recommended cleanup (option α with modification)

If we end up writing the OLD framing as a fallback paper:

1. **Keep** insert + δ joint mechanism (CLAUDE.md hard rule).
2. **Remove** suffix probe from placement search, OR demote it to tie-breaker only.
3. **Replace** placement metric with one of:
   - clean-video vulnerability score (closest to UAP-SAM2 NeurIPS 2025 / HardRegion VOS TCSVT 2024 spirit)
   - insert-only / decoy-only effect at W (ν optimization at each candidate W, no bridge δ during search)
   - memory-attention sensitivity (mechanism-aligned but NOT currently implemented; reviewer-best in principle but paper-not-ready)
4. **Keep** v4.1 anchored Stage-14 bridge polish as the second stage exactly as-is.

Resulting story: "**placement finds insertion vulnerability; bridge polish provides bounded local stabilization beyond insert-only**" — clean role separation, defensible against reviewer drilling.

## Codex's settling experiment (2×2 factorial on 6 clips)

| | final attack: insert-only A0 | final attack: v4.1 anchored bridge polish |
|---|---|---|
| placement: insert-only | A | B |
| placement: suffix-probe joint-aware | C | D |

Read:
- If D > B and C ≈ A → suffix-probe placement only helps when bridge is on → bridge-dependent → overlapping
- If A ≈ B (insert-only placement competitive) → α wins, suffix probe is removable
- If B > A and D > C (bridge polish helps regardless of placement metric) → bridge is a stabilizer regardless, suffix probe is redundant for bridge's effect

This is **the** experiment that answers the question. ~6 GPU-h on V100.

## Methodological severity

| axis | severity | comment |
|---|---|---|
| Scientific soundness | **MEDIUM-HIGH** | overlap is real and code-level; reviewers will notice |
| Reproducibility | LOW | can be verified by reviewers reading code |
| Defensibility under reviewer drilling | **MEDIUM-HIGH** | the "joint > only" framing is fragile; the v4.1-specific claim is narrow but defensible |
| Fix cost | LOW | "remove suffix probe" is a 1-line config change; placement falls back to phase-1/2 stick-breaking + trust-region |

## Action items

- [ ] **Acknowledge in paper** (if OLD fallback): suffix probe and bridge δ overlap; we report v4.1-specific narrow claim.
- [ ] **Run 2×2 factorial** if writing OLD fallback (~6 GPU-h on V100, after V100 deployment).
- [ ] **Update memory**: this is a hard methodological constraint going forward — don't repeat the "suffix probe + bridge polish" pattern in ChronoCloak. Use a **clean placement metric** (clean-video vulnerability, NOT bridge-aware) to avoid this exact entanglement under publisher-side framing.

## Codex thread

`019dd243-04d2-7111-9eb0-c4eb3fec729d`, round 4, 2026-04-28.
