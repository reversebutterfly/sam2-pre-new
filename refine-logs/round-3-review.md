# Round 3 Review (VADI)

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db8a1-7059-76b1-9958-ba5edc222de5`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.2 |
| Method Specificity | 8.5 |
| Contribution Quality | 8.2 |
| Frontier Leverage | 7.6 |
| Feasibility | 6.9 |
| Validation Focus | 8.7 |
| Venue Readiness | 7.3 |
| **Weighted Overall** | **8.2 / 10** |

## Verdict

**REVISE / strong PILOT-READY, not READY.** "Focused, elegant, implementable proposal. Next score jump depends on empirical evidence, not more proposal polishing."

## Drift Warning

**NONE.**

## P0 Sub-7 Fix

### F11 — Feasibility denominator (Feasibility, 6.9)

"Excluding infeasible clips from success counting" reads as hiding failures.

- **Fix**: primary success denominator = **ALL 10 clips**. Infeasible clips count as **failures**. Report `n_infeasible / 10` and feasible-only performance SEPARATELY as diagnostic, not as primary success metric.

## Other Precision Improvements (not sub-7 but push toward 8.3-8.4)

### F12 — Matched local-δ support via phantom insertion positions

- `top-δ-only (K=0)` and `random-δ-only (K=0)` are ambiguous: what positions define their local δ support?
- **Fix**: use PHANTOM insertion positions (same top-K / random-K ranks) to define the δ support S_δ. The "phantom" inserts themselves are NOT placed; only δ at the neighborhoods of those positions is optimized. This matches the local-δ support between full-method and ablation.

### F13 — Signed anti-suppression decomposition (replaces ratio)

**Before**: `|Δmu_decoy| / |Δmu_true| ≥ 2`

**After** (signed):
```
require:  Δmu_decoy > 0  AND  Δmu_decoy ≥ 2 · max(0, -Δmu_true)
report:   Δmu_true and Δmu_decoy separately
```

Rationale: ratios are unstable when denominators are small; and both-negative cases can pass `|ratio|≥2` trivially. Signed form ensures decoy is actually rising AND the true-suppression contribution is bounded.

### F14 — Absolute + ratio gaps (not just ratio)

**Before**: "ours ≥ 2× random" (ratio)

**After** (ratio + absolute):
```
ours ≥ max(2 × random, random + 0.05)      [placement vs random]
ours ≥ max(3 × bottom, bottom + 0.05)      [placement vs bottom]
ours ≥ top-δ-only + 0.10                  [insert presence — already absolute]
ours ≥ top-base-insert+δ + 0.05           [ν optimization — already absolute]
```

Prevents "ratio satisfied but both near zero" passing.

## Answers (detailed)

1. **4 causal claims cover insert causality?** Mostly yes. Need F12 precision (phantom insertion positions for δ-only baselines).
2. **Rank-sum simple enough?** Yes.
3. **Hard feasibility acceptance honest enough?** No — F11 fix needed.
4. **Ratio anti-suppression sufficient?** No — F13 signed form needed.
5. **Pre-pilot ceiling?** 8.2 → 8.3-8.4 with F11-F14. **Cannot honestly reach ≥ 8.5 without pilot evidence.**

## Raw

<details>
<summary>Click</summary>

R3 score: **8.2 / 10, strong PILOT-READY, not READY**.

Scores:
- Problem Fidelity 9.2, Method Specificity 8.5, Contribution Quality 8.2, Frontier Leverage 7.6, Feasibility 6.9, Validation Focus 8.7, Venue Readiness 7.3.

Sub-7: Feasibility — infeasible-clip accounting. Primary success denominator = 10, infeasible = failure. Feasible-only as side diagnostic.

4 causal claims mostly cover gap. Add: top-δ-only / random-δ-only use PHANTOM insertion positions to match local-δ support.

Rank-sum simple enough. Yes.

Hard feasibility + n/10 good, but excluding from success is hiding failures. Primary denominator = all 10.

Ratio anti-suppression insufficient. Use signed: `Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true)`. Report Δmu_true separately.

Absolute gaps beside ratios: `ours ≥ 2× random AND ours ≥ random + 0.05` etc.

Pre-pilot ceiling around 8.2-8.4. Cannot honestly hit ≥ 8.5 without pilot evidence.

Verdict: REVISE / pilot-ready. No drift. Move to pilot.

</details>
