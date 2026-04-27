# Round 4 Review

**Reviewer**: GPT-5.4 xhigh (same thread)
**Date**: 2026-04-27
**Verdict**: **REVISE (proposal-stage ceiling confirmed)** — 8.4/10

## Scores

| Dimension | R1 | R2 | R3 | **R4** | R3→R4 |
|---|---|---|---|---|---|
| Problem Fidelity | 6 | 8 | 9 | **9** | 0 |
| Method Specificity | 6 | 7 | 8 | **9** | +1 |
| Contribution Quality | 5 | 7 | 7 | **8** | +1 |
| Frontier Leverage | 7 | 8 | 8 | **8** | 0 |
| Feasibility | 8 | 7 | 8 | **8** | 0 |
| Validation Focus | 5 | 6 | 8 | **9** | +1 |
| Venue Readiness | 5 | 6 | 6 | **7** | +1 |
| **Overall** | 6.0 | 7.2 | 7.8 | **8.4** | +0.6 |

**Drift**: NONE.

## Codex explicit ceiling statement

> "The proposal architecture has reached the natural ceiling for proposal-stage iteration. Further proposal refinement now has diminishing returns. One good A3 run will move this more than another full review round."

Pattern matches the 2026-04-23 v4-vadi run (also ended at 8.4 with "PRE-PILOT CEILING" verdict). Structural feature of "implementation already done + paper claim plausible + validation experiments not yet run" proposals.

## What blocks READY

- Empirical contingency: A3 hasn't run; can't verify "dominant failure mode" claim.
- C2 hasn't been run on held-out 10-clip; can't verify RAW joint headline gates.
- E1 honest ownership accepted but elegance still contingent on strong results.

→ READY is data-bound, not architecture-bound.

## Codex's only proposed proposal-side upgrade (HIGH-VALUE)

**Add a matched non-insert frame negative control to A3**: Block memory writes at K matched non-insert clean frames (matched count = K=3); compare collapse magnitude vs blocking at attacked insert frames. If insert-position blocking collapses J-drop substantially MORE than control-position blocking, memory hijack story is more reviewer-proof.

This is a one-config-extra ablation (~2 GPU-h additional). Codex calls it "the only remaining proposal-side improvement I would still consider high value." I will add it as A3-control.

## Optional modernizations (only if A3 strong-passes)

- One layer-sensitivity check on the A3 hook (multi-layer block) in appendix
- Already-planned T_obj 16/32/64 sensitivity stays

## Final Action Items (final round)

| # | Action | Status |
|---|---|---|
| 1 | Add A3-control (matched non-insert frame block) | round 4 refinement |
| 2 | Freeze A1 + A3 + A3-control implementations before running | round 4 refinement |
| 3 | Stop proposal iteration after R4 | declare ceiling |
| 4 | Move to Phase 5 final reports | next |

## Raw response

<details>
<summary>Codex R4 (verbatim)</summary>

Scores: PF 9, MS 9, CQ 8, FL 8, F 8, VF 9, VR 7. Overall 8.4.

Verdict: REVISE. Drift: NONE.

R3 fixes the main structural issues. The proposal is now well-anchored, implementable, and scientifically sequenced. The remaining blocker is no longer proposal architecture; it is unresolved empirical evidence.

A1 now correctly isolates bridge δ contribution. d_mem token-set-from-clean essentially free of circularity. Dual-threshold A3 honest and sufficient. Putting A3 first right sequencing. Validation Focus reached 9. Method Specificity reached 9. Venue Readiness moved past 6 but not beyond 7 before results exist. Contribution Quality 7→8.

Why not READY: empirical, not structural. A3 hasn't shown "dominant failure mode" justified. C2 hasn't shown raw joint beats A0 on held-out. E1 honest but elegance contingent on results.

**Smallest further change to push READY**: No further proposal-stage edit is the real answer. Smallest meaningful next step is run A3 and let it gate paper framing.

If you insist on one last structural upgrade: add one negative-control hook — block memory writes at matched non-insert clean frames with same count as W*. If collapse is much smaller there than at attacked insert positions, causal story more reviewer-proof. That is the only remaining proposal-side improvement of high value.

**Bottom line**: Yes, proposal architecture has reached natural ceiling for proposal-stage iteration. Further proposal refinement now has diminishing returns. One good A3 run will move this more than another full review round.

Action items:
1. Freeze A1 + A3 implementations before running
2. Run A3 first; commit to Framing A/B/C
3. If A3 partial, do NOT keep "dominant failure mode" language anywhere
4. C2 + A1 main table; wrapper strictly separate
5. STOP proposal iteration after this. Next acceptance-lift comes from data.

</details>
