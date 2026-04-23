# Refinement Report — VADI (Vulnerability-Aware Decoy Insertion)

**Problem**: User directive — keep insert+modify strategy; reject pure suppression. Design principled insertion method leveraging publisher's access to clean video for optimal placement.

**Initial approach**: Per-video PGD, K_ins inserts at heuristic-scored vulnerability windows + local δ + contrastive decoy-margin.

**Date**: 2026-04-23

**Rounds**: 4 / 5 (ceiling reached early by reviewer-explicit confirmation)

**Final score**: 8.4 / 10

**Final verdict**: REVISE (formal) / PRE-PILOT CEILING (internal — pilot mandatory)

## Problem Anchor

See `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`. Preserved across all rounds.

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | Prob Fid | Method Spec | Contrib | Frontier | Feasib | Valid | Venue | Overall | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 7.5 | 6.0 | 6.0 | 7.0 | 5.5 | 6.0 | 5.0 | **6.3** | REVISE |
| 2 | 9.0 | 8.0 | 7.4 | 7.3 | 6.8 | 8.2 | 6.6 | **7.7** | PILOT-READY |
| 3 | 9.2 | 8.5 | 8.2 | 7.6 | 6.9 | 8.7 | 7.3 | **8.2** | strong PILOT-READY |
| 4 | 9.4 | 8.7 | 8.4 | 7.7 | 7.2 | 9.0 | 7.7 | **8.4** | ceiling confirmed |

## Round-by-Round Review Record

| Round | Reviewer concerns | Fixes | Outcome |
|---|---|---|---|
| 1 | L_obj = suppression; scorer ad hoc; decoy non-contrastive; dense δ confound; ν ε=8/255 underuses LPIPS; K=3 top unproven | Removed L_obj; rank-z 3-signal; contrastive margin; local δ; LPIPS-TV bound ν; gated pilot | +1.4 |
| 2 | Top-K may win via δ not inserts; ratio-only anti-suppression; scorer over-elaborate; hinge-only feasibility | Added δ-only-top + δ-only-random + base-insert+δ controls; signed decoy decomposition; rank-sum; hard S_feas | +0.5 |
| 3 | Excluding infeasible = hiding failures; phantom positions needed; ratios unstable | Primary denominator = all 10; phantom W for δ-only; absolute+ratio gaps | +0.2 |
| 4 | Internal-float feasibility ≠ exported-artifact feasibility | Re-measure on EXPORTED uint8 artifact; S_feas on export | 0 (ceiling) |

## Final Proposal Snapshot (3-5 bullets)

- **Vulnerability-aware insertion**: rank-sum 3-signal scorer (confidence drop + mask discontinuity + Hiera discontinuity) on clean-SAM2 trace picks top-K non-adjacent placements.
- **Content optimization**: insert content (ν, LPIPS-TV bounded, no ε) + local δ on insert neighborhoods are jointly PGD-optimized under **contrastive decoy-margin** loss `softplus(mu_true - mu_decoy + 0.75)`. No suppression, no object_score margin.
- **Causal isolation**: 10-row main table with top/random/bottom placement + top-δ-only / random-δ-only / top-base-insert+δ insertion-value controls; signed decoy-vs-suppression decomposition.
- **Feasibility discipline**: metrics measured on EXPORTED uint8 artifact, infeasible clips counted as failures in primary denominator.
- **Pilot-gated**: 3 clips × 4 configs; GO requires `J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05` AND `J-drop(K=3 top) ≥ 0.20` on ≥ 2/3 clips; NO-GO triggers attack-surface pivot paper.

## Method Evolution Highlights

1. **Biggest simplification**: fold 4-term weighted vulnerability scorer → rank-sum 3-signal (drop flow, equal weights, pre-registered). Drop ProPainter → temporal midframe.
2. **Biggest mechanism upgrade**: replace `L_decoy + L_obj` (suppression-by-another-name) with pure contrastive decoy-margin `softplus(mu_true - mu_decoy + 0.75)` enforcing strict ordering on masked means.
3. **Biggest validation upgrade**: 10-row main table with causal isolation (placement × insertion-value) + signed anti-suppression + restoration attribution + exported-artifact measurement.

## Pushback / Drift Log

| Round | Reviewer said | Author response | Outcome |
|---|---|---|---|
| 1 | "L_obj is suppression by another name" | Accepted — removed from default (user directive backing: no suppression) | Accepted |
| 2 | "Ratio anti-suppression insufficient" | Accepted — switched to signed decomposition | Accepted |
| 3 | "Excluding infeasible = hiding failures" | Accepted — primary denominator all 10 | Accepted |
| 4 | "Internal float ≠ exported artifact" | Accepted — re-measure on export | Accepted |

No drift incidents. Insert-required + no-suppression constraint preserved. Codex never suggested removing inserts (would have been drift per user directive in CLAUDE.md).

## Remaining Weaknesses

1. **Empirical untested**. Historical J-drop at matched settings was 0.001-0.0013 (R001-R003). The proposal's claim of ≥ 0.35 is ambitious.
2. **Pilot NO-GO possibility**: if top-δ-only ≈ ours, inserts are decorative and the paper pivots honestly to "vulnerability-aware local perturbation" (conflict with user constraint; would require user resolution).
3. **SAM2Long install on Pro 6000** pending (2-3 GPU-hours).

## Raw Reviewer Responses

All round reviews verbatim in `refine-logs/round-N-review.md` (N=1..4).

## Next Steps

### Run the pilot (MANDATORY, ~3-5 GPU-hours on Pro 6000)

3 clips × 4 configs (K=1 top, K=1 random, K=3 top, δ-only-local-random).

**GO path**: proceed to DAVIS-10 main + restoration + appendix (~15-20 GPU-hours total). Paper targets NeurIPS/ICML-class venue.

**NO-GO path**: pivot to "architecture-aware attack-surface analysis" paper using restoration + vulnerability scoring as primary content; honestly retract the attack-success narrative.

### If GO:

1. Write `scripts/run_vadi.py` driver.
2. Write `memshield/vulnerability_scorer.py` (rank-sum scorer).
3. Extend `memshield/ablation_hook.py` with 3 swap hooks (SwapF0Memory, SwapHieraFeatures, SwapBank).
4. Run `/experiment-plan` for detailed execution roadmap or `/run-experiment` for direct execution.
