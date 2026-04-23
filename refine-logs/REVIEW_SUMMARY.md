# Review Summary — VADI (Vulnerability-Aware Decoy Insertion)

**Problem**: User rejected prior v3 pure-suppression direction. Keep insert+modify strategy. Design principled insertion method using publisher's access to clean video to find optimal insertion points that make SAM2 lose the target.

**Initial approach**: Per-video PGD with K_ins inserts at rank-sum-scored vulnerability windows + local δ on insert neighborhoods + contrastive decoy-margin loss (no suppression).

**Date**: 2026-04-23

**Rounds**: 4 / 5

**Final score**: 8.4 / 10 (pre-pilot ceiling reached & explicitly confirmed by reviewer)

**Final verdict**: REVISE (formal) / PRE-PILOT CEILING (internal — pilot is mandatory next step)

## Problem Anchor (verbatim, preserved across all rounds)

See `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`. Key invariants:
- Insert+modify (K_ins ∈ {1,2,3}) is required.
- Placement must be principled via vulnerability scoring (not canonical FIFO).
- No suppression loss; contrastive decoy-margin only.
- Insert works via current-frame Hiera pathway (consistent with B2 bank non-causality finding), not via bank poisoning.
- Two-tier fidelity: f0 (ε=2/255+SSIM≥0.98), originals (ε=4/255+LPIPS≤0.20), inserts (LPIPS≤0.35 vs midframe base+TV≤1.2×base, no ε).
- GT-free throughout.

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | What changed | Score | Solved? |
|---|---|---|---|---|
| 1 | L_obj = suppression in disguise; scorer ad hoc; decoy loss not contrastive; dense δ may dominate; scheduler-tuning accusation risk; ν=8/255 underuses LPIPS budget; K=3 top unproven | Removed L_obj; rank-based robust-z over 3 signals; contrastive margin `softplus(mu_true - mu_decoy + 0.75)`; local δ on insert neighborhoods; top/random/bottom + multi-draw random; LPIPS+TV on ν (no ε); gated 3-clip pilot | 6.3 → 7.7 | Mostly |
| 2 | Top-K advantage may come from LOCAL δ at vulnerable frames, not inserts; ratio anti-suppression insufficient; scorer math over-elaborate; hard feasibility | Added top-δ-only / random-δ-only / top-base-insert+δ controls; signed Δmu_decoy vs Δmu_true decomposition; rank-sum (not robust-z); hard S_feas acceptance | 7.7 → 8.2 | Yes |
| 3 | Excluding infeasible from success = hiding failures; need phantom positions for δ-only; ratio-only unstable | Primary denominator = all 10, infeasible = failure; phantom W for δ-only baselines; ratio + absolute gap both required | 8.2 → 8.4 | Yes |
| 4 | Internal-float feasibility may not match exported-artifact feasibility (quantization/JPEG) | Re-measure all metrics on EXPORTED uint8 artifact; S_feas on export, not internal | 8.4 (ceiling) | Yes, plus last tightening |

## Overall Evolution

- **Anchor**: preserved throughout; all drift-warnings flagged by Codex explicitly rejected with user directive backing.
- **Dominant contribution sharpened**: from vague "insertion method" (R0) to "vulnerability-aware insertion with full causal isolation + mechanism attribution + GT-free + exported-artifact feasibility" (R4). One paper thesis, no parallel contributions.
- **Unnecessary complexity removed**: L_obj (suppression in disguise); ProPainter (replaced by temporal midframe); motion-discontinuity scorer term (unjustified); hard ε bound on ν (replaced by LPIPS-TV); flow signal (dropped).
- **Frontier leverage appropriate**: SAM2.1 as both target and pseudo-label source; no LLM/diffusion/RL decoration; gradient-based scorer kept only as fallback modernization.
- **Validation discipline**: top/random/bottom + δ-only-top/random + base-insert-top + canonical + multi-draw random + restoration attribution + signed decoy decomposition + exported-artifact measurement. 10-row main table, every row isolates a specific causal claim.

## Final Status

- **Anchor status**: preserved across 4 rounds.
- **Focus status**: tight. 2 trainable tensors (δ, ν), 1 heuristic scorer, 1 paper thesis.
- **Modernity**: appropriately frontier-aware (SAM2 internals; no decoration).
- **Strongest parts of final method**:
  1. Principled placement via rank-sum 3-signal scorer on clean-SAM2 signals.
  2. Contrastive decoy-margin loss with signed anti-suppression guarantee.
  3. Causal isolation via 3-way placement controls + 3-way insert-value controls + restoration attribution.
  4. Exported-artifact feasibility (closes optimization-vs-delivery loophole).
  5. Pre-committed 8-claim success bar.
- **Remaining weaknesses**:
  1. All claims empirically unproven. Historical J-drop for matched settings was 0.001-0.0013.
  2. If top-δ-only ≈ ours, inserts are decorative (conflicts with user constraint; honestly reported as NO-GO pivot).
  3. Pilot may trigger NO-GO → paper pivots to attack-surface analysis.
  4. SAM2Long install still pending (2-3 GPU-hours).

## Output Files

- **Final proposal**: `refine-logs/FINAL_PROPOSAL.md`
- **Round logs**: `refine-logs/round-0-initial-proposal.md`, `round-N-review.md`, `round-N-refinement.md` (N=1..4)
- **Score evolution**: `refine-logs/score-history.md`
- **Problem anchor**: `refine-logs/PROBLEM_ANCHOR_2026-04-23_v4-insert.md`
- **Archived v3 suppression proposal**: `refine-logs/archive-2026-04-23-v3-suppression__*.md`
- **Design-constraint note in CLAUDE.md**: section "Method Design Constraints (2026-04-23)"
