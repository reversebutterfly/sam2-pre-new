# Round 2 Review

**Reviewer**: GPT-5.4 xhigh (same thread as R1)
**Thread**: `019dcd87-c42b-7b03-9139-34df6b6ebd89`
**Date**: 2026-04-27
**Verdict**: **REVISE** (7.2/10 overall, +1.2 vs R1)

## Scores

| Dimension | R1 | **R2** | Δ |
|---|---|---|---|
| Problem Fidelity | 6 | **8** | +2 |
| Method Specificity | 6 | **7** | +1 |
| Contribution Quality | 5 | **7** | +2 |
| Frontier Leverage | 7 | **8** | +1 |
| Feasibility | 8 | **7** | -1 (more honest scoping) |
| Validation Focus | 5 | **6** | +1 |
| Venue Readiness | 5 | **6** | +1 |
| **Overall** | **6.0** | **7.2** | **+1.2** |

**Drift Warning**: NONE (anchor preserved, raw joint = scientific method).

## Critical findings (R2)

### CRITICAL — A1 still mis-specified
A0 vs full-v5 bundles insertion + search + stabilization + bridge δ. Doesn't isolate the bridge δ contribution.

**Fix**: A1 = "insert-only with SAME placements + SAME ν + SAME decoy family" vs "insert+bridge δ". Bridge δ is the ONLY toggle. Measure both J-drop AND post-insert d_mem persistence.

### IMPORTANT — A3 should run early; narrow framing if weak
A3 collapse threshold (≥0.20 abs on ≥7/10) is aggressive. Run A3 first; if weaker collapse → narrow paper to "memory-mediated persistence evidence" not "dominant failure mode".

### IMPORTANT — d_mem protocol tightening needed
"Object-related tokens = attention weight > median" is condition-dependent → potentially circular. Better: define token set from CLEAN run ONCE, fix layer/block/head/agg ex ante, reuse across clean/insert-only/full.

### IMPORTANT — Pre- vs post-projection value vectors not specified
Value vectors in cross-attention have pre-output-projection and post-output-projection variants. Spec which.

### MINOR — Placement search ownership
Reviewer says "either own E1 more directly OR simplify to deterministic heuristic + curriculum as appendix". I'm keeping E1 as owned but will add explicit text in Discussion explaining why search is the right tool (vulnerability heuristics anti-correlated empirically — falsified earlier).

## Action items for Round 2 refinement

1. **CRITICAL** Fix A1: insert-only-with-ν-and-W-fixed vs insert+bridge δ. Both with v5's placement search to remove placement confound.
2. **IMPORTANT** Tighten d_mem protocol: token set from CLEAN once + freeze + reuse; pre-projection V; layer = `memory_attention.layers[-1].cross_attention`.
3. **IMPORTANT** A3 acceptance language: report as "strong pass" target. Weaker collapse (≥0.10 abs on ≥6/10) → narrowed framing to "memory contribution evidence, partial".
4. **MINOR** Add explicit Discussion text: search ownership + why heuristic anti-correlation forces empirical search.
5. **EXECUTION** Run A3 FIRST; let result determine final framing.

## Raw response

<details>
<summary>Codex R2 (verbatim)</summary>

The Problem Anchor is now preserved, not drifted. Moving the scientific method to RAW joint and relegating wrapper selection to a separate deployment column fixes the biggest Round 1 issue.

Scores: PF 8, MS 7, CQ 7, FL 8, F 7, VF 6, VR 6. Overall 7.2.

The dominant contribution is sharper. This now reads like one mechanism claim with two explicit enablers, instead of three half-overlapping contributions plus a rescue wrapper.

Critiques:
- C1.a is now a real causal test, falsifiable. Concern is acceptance threshold (≥0.20 collapse on ≥7/10) is aggressive.
- C1.b: "object-related tokens = attention weight > median" can become circular. Define token set from clean run once, fix layer/block/head, reuse exact subset.
- RAW joint headline gate plausible but not bankable from 4-clip dev. Most fragile = combined ≥0.55 mean J-drop + 5/10 wins.
- d_mem extraction needs tighter spec: pre/post output projection.
- Placement search hidden complexity remains; either own more directly or simplify.

Action items:
1. Fix A1 — currently bundles insertion+search+stabilization+bridge δ.
2. Tighten d_mem protocol.
3. Don't make C1.a 0.20 threshold the only scientifically valid outcome.
4. Decide on placement search ownership.
5. Run A3 before polishing paper story further.

Verdict: REVISE.

</details>
