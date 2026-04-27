# Refinement Report

**Problem**: Adversarial attack on SAM2 video segmentation using BOTH internal frame insertion AND sparse δ on adjacent original frames; address Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA prior arts; AAAI venue.
**Initial Approach**: v4.1 implementation (memory-hijack insertion + bridge δ + adaptive wrapper); needed paper-level reframe per novelty check.
**Date**: 2026-04-27
**Rounds**: 4 / 5 (early termination at proposal-stage ceiling per codex explicit verdict)
**Final Score**: **8.4 / 10**
**Final Verdict**: **REVISE (CEILING)** — architecture at natural max; READY blocked only by unrun A3.
**Codex thread**: `019dcd87-c42b-7b03-9139-34df6b6ebd89`

## Problem Anchor (verbatim)

(See `PROBLEM_ANCHOR_2026-04-27.md`)

## Output Files

- Problem anchor: `refine-logs/PROBLEM_ANCHOR_2026-04-27.md`
- Round files: `refine-logs/round-{0,1,2,3,4}-{initial-proposal/refinement,review}.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- **Final proposal**: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | PF | MS | CQ | FL | F | VF | VR | Overall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 6 | 5 | 7 | 8 | 5 | 5 | **6.0** | REVISE |
| 2 | 8 | 7 | 7 | 8 | 7 | 6 | 6 | **7.2** | REVISE (+1.2) |
| 3 | 9 | 8 | 7 | 8 | 8 | 8 | 6 | **7.8** | REVISE (+0.6) |
| 4 | 9 | 9 | 8 | 8 | 8 | 9 | 7 | **8.4** | REVISE CEILING (+0.6) |

PF=Problem Fidelity, MS=Method Specificity, CQ=Contribution Quality, FL=Frontier Leverage, F=Feasibility, VF=Validation Focus, VR=Venue Readiness.

## Round-by-Round Review Record

| Round | Main reviewer concerns | Main fixes | Result |
|---|---|---|---|
| 1 | Wrapper drift; demoted contributions dishonest; A3 confounded; "first demonstration" overclaim | (initial proposal) | many open |
| 2 | A1 confounded; d_mem circular; A3 0.20/7-clip aggressive; placement ownership | C1+E1+E2 reframe; raw joint = science; A3 dual-threshold; novelty narrowed; memory-causality A3 | mostly resolved |
| 3 | A1 still bundled upstream search/ν; d_mem layer/projection unspecified; conditional framing not pre-committed | A1 operationally locked (same upstream); d_mem pre-registered (last block, pre-projection V, top-32 frozen tokens); A3-first sequencing; conditional Framing A/B/C | venue contingent on A3 |
| 4 | Empirical contingency only — no architectural blocker | Negative-control hook A3-control (matched non-insert frames); A1+A3 implementations frozen; ceiling declared | partial → CEILING |

## Final Proposal Snapshot (3-5 bullets)

- **Thesis**: K=3 internally-inserted semantic decoys hijack SAM2's prompt-conditioned memory; sparse δ on L=4 bridge frames extends the hijack persistence; an adaptive wrapper (deployment, not contribution) guarantees joint ≥ insert-only.
- **C1 (mechanism, paired)**: A3 with negative control (block memory writes at insert vs matched non-insert positions); pre-registered strong/partial/fail tier. Plus C1.b persistence trace via d_mem(t) protocol.
- **E1 (enabling, openly necessary engineering)**: joint curriculum placement search; vulnerability heuristics empirically falsified.
- **E2 (enabling)**: dense L_keep_full no-regression on full suffix.
- **Validation package**: 10-clip held-out RAW joint vs A0 paired (Table 1 main, Table 2 deployment); A1 isolated bridge δ; A2 random vs search; A3 with negative control. ~23 GPU-h total, 3 days.

## Method Evolution Highlights

1. **Most important reframe (R1→R2)**: scientific method = RAW joint, not wrapper-selected. Demoted polish_revert from contribution to deployment policy. Reframed contribution structure to ONE main C1 + TWO enabling E1/E2.
2. **Most important specificity upgrade (R2→R3)**: A1 operationally locked with same upstream W*, ν, decoy_seeds; d_mem protocol pre-registered (last cross-attention block, pre-output-projection V, top-32 clean-derived tokens frozen per clip).
3. **Most important credibility upgrade (R3→R4)**: A3 negative control (matched non-insert memory-write blocking) — collapse must be insert-position-specific, not generic memory-write fragility.
4. **Most important honesty upgrade (across rounds)**: Conditional title/abstract Framing A (strong), B (partial), C (workshop pivot) pre-registered before running A3 — paper claim is gated by data, no post-hoc reframing allowed.

## Pushback / Drift Log

| Round | Reviewer said | Author response | Outcome |
|---|---|---|---|
| 1 | Demote outcome-critical components to "details" is dishonest | ACCEPTED — restructured to 1 main + 2 enabling + 1 deployment | resolved |
| 1 | A3 "all-frames-δ vs insert+bridge" confounds | ACCEPTED — replaced with memory-causality blocking ablation | resolved |
| 1 | "First demonstration" overclaim | ACCEPTED — narrowed to causal evidence + persistence sub-claims | resolved |
| 2 | "Use SAM2 memory readout as auxiliary loss" (modernization) | REJECTED as auxiliary loss (adds gradient pathway, breaks 2-component budget); ACCEPTED as causal diagnostic only | partial — diagnostic |
| 2 | "frozen vision prior for decoy init" | REJECTED for default (adds component); kept available as supplementary "decoy quality" ablation | rejected default |
| 2 | A1 still confounds upstream search/ν | ACCEPTED — locked operationally with same upstream W*, ν, decoy_seeds | resolved R3 |
| 2 | d_mem token-set "circular" | ACCEPTED — token set from CLEAN run once, frozen, reused | resolved R3 |
| 3 | "Either own E1 or simplify aggressively" | ACCEPTED OWN — added Discussion paragraph with empirical falsification of heuristic | resolved R3 |
| 3 | "collapse traj+α+warp+R to simpler masked residual" | REJECTED for now (works in v4.1 dev-4); marked future ablation | accepted with limit |
| 4 | A3 negative control | ACCEPTED — added matched-non-insert blocking | resolved R4 |
| 4 | "STOP proposal iteration after this" | ACCEPTED — declared CEILING at 8.4 | terminated |

## Remaining Weaknesses (honest)

1. **AAAI-level novelty contingent on A3 strong-pass**: paper claim "DOMINANT failure mode" only if A3 strong; otherwise narrower claim "SUBSTANTIAL component". Pre-registered, but the strong-pass branch has higher venue lift.
2. **RAW joint headline gates not yet validated on held-out 10**: v4.1 dev-4 had 75% strict-win on 4 clips; held-out 10 may give 5-7/10, but this is not bankable until run.
3. **bmx-trees-like clips still revert**: lambda_keep_full=50 retry not yet validated. Need ablation in supplementary or pre-eval tuning.
4. **Search-heavy method dependence**: even with E1 honest framing, the elegance argument is weakened because search is non-trivially load-bearing. If reviewers want a heuristic-based or learned placement, paper has to defend the search choice empirically.

## Raw Reviewer Responses

(See round-{1,2,3,4}-review.md for full Codex responses, including detailed scores, per-dimension fixes, simplification/modernization opportunities, action items, and verdicts.)

## Next Steps

**Codex explicit verdict**: "STOP proposal iteration after this. The next acceptance-lift comes from data, not wording. One good A3 run will move this more than another full review round."

Recommended next actions:
1. **Wait for Pro 6000 GPU release** (currently shared with 2025D_ShiGuangze).
2. **Implement R4-spec'd hook + extractor + control sampler** (~5h coding).
3. **Run A3 first** (Day 1 PM, 10 GPU-h) — gates all paper framing.
4. **Run C2 + A1 paired** (Day 2 overnight, 8 GPU-h).
5. **Run A2 + T_obj sensitivity** (Day 3 AM, 5 GPU-h).
6. **Write paper using pre-registered conditional framing**.

If READY → proceed to `/experiment-plan` for full execution roadmap, then `/run-experiment`.
If REVISE on data → `/auto-review-loop` with results.
If RETHINK → revisit core mechanism with `/idea-creator`.

## Pattern Note

This 8.4 ceiling pattern matches the 2026-04-23 v4-vadi run (also ended 8.4 PRE-PILOT CEILING). Structural feature of "implementation already exists + paper claim plausible + validation experiments unrun" proposals. Future research-refine sessions on this project should expect to hit ~8.4 in 3-4 rounds; anything above requires actually running A3 / held-out 10.
