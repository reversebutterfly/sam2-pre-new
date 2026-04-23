# Round 2 Review (VADI)

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db8a1-7059-76b1-9958-ba5edc222de5`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 8.0 |
| Contribution Quality | 7.4 |
| Frontier Leverage | 7.3 |
| Feasibility | 6.8 |
| Validation Focus | 8.2 |
| Venue Readiness | 6.6 |
| **Weighted Overall** | **7.7 / 10** |

## Verdict

**REVISE / PILOT-READY, not READY.** "If the pilot shows top-K optimized inserts beat random/bottom and beat matched local-δ-only controls, the proposal can move into 8.3-8.7. READY only after empirical evidence confirms insert-specific causal mechanism."

## Drift Warning

**NONE.** Only "hidden drift risk": success may come from local δ placement rather than optimized inserts. That is a **causal identification issue**, not a constraint violation.

## P0 Action Items

### F8 — Causality isolation controls (Venue Readiness, P0)
Top-K placement advantage may come from **LOCAL δ at vulnerable frames**, not inserts. Add:
- **top-δ-only**: δ at top-K vulnerability neighborhoods, **no inserts**
- **random-δ-only**: δ at random-K neighborhoods, no inserts
- **top-base-insert+δ**: insert at top-K, but insert is **unoptimized midframe** (no ν); δ optimized

These three + existing top-K optimized (ours) disentangle:
- inserts vs δ at same positions (ours vs top-δ-only)
- insert OPTIMIZATION vs insert presence (ours vs top-base-insert+δ)
- placement of δ (top-δ-only vs random-δ-only)

### F9 — Hard feasibility acceptance (Feasibility, P0)
Final selection must enforce LPIPS feasibility as **hard acceptance**, not rely on hinge penalties alone. If no PGD step satisfies all fidelity constraints simultaneously, the clip is flagged as "fidelity-infeasible at stated budget" and excluded (or reported separately).

### F10 — Log mu_true and mu_decoy separately
Even with contrastive margin loss, if success comes only from **collapsing mu_true** (while mu_decoy stays flat), reviewers call it "implicit suppression". Log both per-frame trajectories; report at final:
- `Δmu_true = mu_true(attacked) − mu_true(clean)` — should be ≤ 0 but moderate
- `Δmu_decoy = mu_decoy(attacked) − mu_decoy(clean)` — should be ≫ |Δmu_true|

For true "decoy wins" behavior, `Δmu_decoy` must be much larger in magnitude than `|Δmu_true|`.

## Simplifications

- **Scorer math**: rank-based robust-z via IQR is mathematically more elaborate than necessary. Replace with **rank-sum**:
  ```
  rank_conf_m = rank(r_conf_m) among m ∈ {1..T-1}       # 1..T-1
  (similarly rank_mask, rank_feat)
  v_m = rank_conf_m + rank_mask_m + rank_feat_m          # simple sum of ranks
  W = argtop-K non-adjacent
  ```
  Same robustness; less paper noise.

## Specific Q&A (from R2)

1. **Anti-suppression**: "substantially fixed. Removing L_obj right. Log mu_true / mu_decoy separately to preempt implicit-suppression critique."
2. **Scorer**: "Yes, much more principled. Describe simply as robust rank aggregation. Robust-z formula may be more elaborate than necessary."
3. **Local δ**: "Greatly reduces dense-δ confound but doesn't fully prove inserts dominate. Need matched δ-only-by-position controls." → fix F8
4. **Top vs random vs bottom**: "Strong for placement causality. NOT by itself strong for insertion causality. Add optimized-vs-base-insert AND with-vs-without-insert controls." → fix F8
5. **LPIPS-bound ν**: "Clean and better than hard ε. Final selection must enforce feasibility as hard acceptance." → fix F9
6. **Pilot + NO-GO**: "Acceptable and reviewer-positive. Thresholds reasonable for pilot, not for final claims. Full claim needs DAVIS-10 evidence with CIs and absolute gaps."

## Raw Response

<details>
<summary>Click</summary>

Revised weighted score: **7.7 / 10, PILOT-READY but not READY**.

| Dim | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 8.0 |
| Contribution Quality | 7.4 |
| Frontier Leverage | 7.3 |
| Feasibility | 6.8 |
| Validation Focus | 8.2 |
| Venue Readiness | 6.6 |

Sub-7 issues:
- Feasibility: K=3 top ≥ 0.20 J-drop on 2/3 pilot clips uncertain. Pilot as hard gate.
- Venue Readiness: top placement may help because LOCAL δ is at fragile frames. Add matched controls: top-local-δ-only, random-local-δ-only, top-base-insert+δ.

Anti-suppression: fixed. Log mu_true / mu_decoy separately.
Scorer: principled, simplify (rank aggregation, not robust-z).
Local δ: reduces confound but doesn't prove inserts dominate. Need δ-only-by-position controls.
Top/rand/bottom: strong for placement, not insertion. Add "with insert vs no insert" at matched positions.
LPIPS-bound ν: clean. Final selection = hard feasibility acceptance.
Pilot: acceptable. Thresholds OK for pilot, not final.

Verdict: REVISE / PILOT-READY. READY only after empirical evidence.

</details>
