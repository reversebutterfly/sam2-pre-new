# Refinement Report — MemoryShield v3 (dataset-protection reframe)

**Problem**: user explicitly scoped down the paper claim after prior-proposal falsification — reframe as dataset protection (processed video; attacker cannot segment under high fidelity).

**Initial approach**: per-video PGD vs SAM2 with end-to-end output supervision; architecture-aware budget; restoration attribution.

**Date**: 2026-04-23

**Rounds**: 5 / 5 (MAX_ROUNDS reached)

**Final score**: 8.4 / 10 (pre-pilot ceiling reached and acknowledged by reviewer)

**Final verdict**: REVISE (formal) / CONDITIONALLY READY (internal, pending pilot data)

## Problem Anchor (verbatim)

See `PROBLEM_ANCHOR_2026-04-23.md` and `REVIEW_SUMMARY.md`. Key invariants:
- Dataset protection (not clean-suffix attack).
- Per-video PGD, pixels-only, two-tier fidelity budget.
- f0 + current-frame as causal pathways (bank deliberately skipped per B2).
- DAVIS-10 J&F drop ≥ 0.40 at fidelity as success criterion.

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1     | 8.0 | 6.0 | 6.0 | 8.0 | 6.5 | 6.5 | 6.0 | **6.7** | REVISE |
| 2     | 8.5 | 8.0 | 7.4 | 8.0 | 7.2 | 6.8 | 6.7 | **7.7** | REVISE |
| 3     | 9.0 | 8.7 | 8.1 | 8.2 | 7.8 | 8.4 | 7.6 | **8.4** | REVISE |
| 4     | 8.8 | 8.2 | 8.2 | 8.3 | 7.7 | 8.5 | 7.9 | **8.4** | REVISE (light) |
| 5     | 8.8 | 8.4 | 8.2 | 8.3 | 7.8 | 8.5 | 7.9 | **8.4** | REVISE (ceiling, sign-fix confirmed) |

## Round-by-Round Review Record

| Round | Main reviewer concerns | What was changed | Result |
|-------|---|---|---|
| 1 | Supervision leakage; PGD-only novelty; non-specific ablation; feasibility vague; scope narrow. | GT-free self-supervision; C2 mandatory; restoration counterfactuals; concrete feasibility; SAM2Long promoted. | 6.7 → 7.7 |
| 2 | Sign bug in ΔJ_restore; R1/R2 non-additivity; C2 still parallel. | Sign fixed; R12 joint added; causal-loop framing (C2 folded into C1); soft logits; decoy dropped. | 7.7 → 8.4 |
| 3 | Venue readiness depends on pilot outcome. | Pilot gate promoted to first-class with 4 pre-committed branches; pathway-aligned fallback loss; confidence-weighted pseudo-mask. | 8.4 plateau |
| 4 | Sign bug in fallback loss (`∥h_a − h_c∥` minimized preserves clean features). | Switched to cosine anti-alignment with detached clean features. | 8.4 plateau (ceiling) |
| 5 | Sign-fix validation. | Confirmed correct; specified m_f0 cosine over maskmem ⊕ obj_ptr, L2-normalized per channel. | 8.4 (confirmed ceiling) |

## Final Proposal Snapshot (3-5 bullets)

- **Architecture-aware dataset protection** for SAM2-family VOS.
- **Causal loop**: diagnose (bank non-causal per B2) → attack (pathway-targeted per-video PGD, two-tier fidelity, GT-free soft-logit + confidence-weighted supervision) → verify (five-config restoration attribution with signed ΔJ_restore).
- **Single trainable component** δ; three inference-only swap hooks for restoration.
- **Pilot-gate with 4 pre-committed branches** (B1 strong+attributed, B2 strong+joint, B3 strong+open-mechanism, B4 pivot). Paper narrative committed before pilot runs.
- **Compute**: ~15 GPU-hours to full paper from pilot-pass.

## Method Evolution Highlights

1. **Most important simplification**: merger of C2 (bank marginality) into C1 (attack + attribution) as mechanism evidence, collapsing two-claim structure to one diagnose-attack-verify thesis.
2. **Most important mechanism upgrade**: GT-free clean-SAM2 soft-logit confidence-weighted self-supervision (replaces the earlier DAVIS-GT-risk supervision and hard-threshold pseudo-masks).
3. **Most important modernization**: restoration-counterfactual attribution via precisely-boundary-specified swap hooks (SwapF0MemoryHook / SwapHieraFeaturesHook / SwapBankHook) replacing blunt pathway-removal ablation.

## Pushback / Drift Log

| Round | Reviewer said | Author response | Outcome |
|-------|---|---|---|
| 1 | "SAM2Long main-table sanity row acceptable, not bloat, only if it stays secondary." | Kept as single row not transfer sweep. | Accepted. |
| 2 | "Modernization: pathway-aligned feature losses could replace suppression." | Added as B4 fallback only, NOT as default. Prevents contribution sprawl. | Accepted as fallback. |
| 3 | "Empirical risk remains." | Added pilot-gate first-class with 4 pre-committed branches → paper robust to outcomes. | Accepted (venue readiness +0.3). |
| 4 | "Fallback loss sign wrong — minimizing distance preserves clean features." | Switched to cosine anti-alignment with detached clean features. Equivalent sign-canonical form also provided. | Accepted (sign correct). |

**No drift incidents.** All reviewer-requested changes were method-level improvements. The one conditional drift flag (R2: "checkpoint selection might leak GT") was addressed explicitly.

## Remaining Weaknesses

1. **Empirical feasibility**: mean J&F drop ≥ 0.40 at F_lpips=0.20 is untested. Prior clean-suffix v2 attacks gave J-drop ≈ 0.001; the new whole-video / processed-frame eval regime is different but also unproven. **Pilot is mandatory before full commit.**
2. **B4 pivot** (if pilot fails) is honestly a different paper. The proposal stays coherent under B4 (attack-surface analysis) but the core "dataset protection succeeds" claim is retracted.
3. **SAM2Long main-row** depends on SAM2Long install on Pro 6000 (not yet done; 2-3 GPU-hours).

## Raw Reviewer Responses

All round reviews verbatim under `refine-logs/round-N-review.md` (N=1..4) with expandable `<details>` blocks containing the full Codex MCP response.

## Next Steps

### If pilot hits B1/B2/B3 (strong attack):

1. Proceed to full DAVIS-10 run (~2.5 GPU-hours).
2. Run restoration (R1/R2/R12/R3 + B-control) on all 10 clips (~0.5 GPU-hour).
3. Install SAM2Long + run transfer (~2-3 GPU-hours).
4. Appendix (DAVIS-30 extended, prompt-robustness, SAM2.1-Base transfer; ~5-8 GPU-hours).
5. Invoke `/experiment-plan` for detailed execution roadmap or `/run-experiment` for direct execution.

### If pilot hits B4 (weak attack):

1. Swap to `L_B4` pathway fallback; rerun pilot.
2. If still weak, pivot paper per B4 narrative: "architecture-aware attack-surface analysis of SAM2-family VOS" using the B2 bank-ablation + restoration suite as primary results.
3. Re-run `/research-refine` with the pivoted anchor to tighten the B4-paper's scope and claims.

### In all cases:

- Run the pilot. The core empirical uncertainty cannot be resolved without it.
- The proposal is formally at pre-pilot ceiling and further formulation edits are lower-value than empirical data.
