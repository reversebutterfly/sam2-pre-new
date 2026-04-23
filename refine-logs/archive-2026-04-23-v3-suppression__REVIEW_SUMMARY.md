# Review Summary — MemoryShield v3 (dataset-protection reframe)

**Problem**: Claim of prior proposal was too strong. Reframe as dataset protection: publisher processes video; consumer with SAM2 cannot segment the processed video under high fidelity. Emphasis on high-fidelity damage to SAM2 segmentation.

**Initial approach**: Per-video PGD against SAM2 with end-to-end output supervision, architecture-aware budget allocation, restoration-based attribution — informed by prior falsification of the clean-suffix bank-poisoning narrative.

**Date**: 2026-04-23

**Rounds**: 5 / 5 (MAX_ROUNDS reached at pre-pilot ceiling)

**Final score**: 8.4 / 10 across last three rounds (plateau)

**Final verdict**: REVISE (formal) / CONDITIONALLY READY (internal — pending pilot data)

## Problem Anchor (verbatim, preserved across all rounds)

(per `PROBLEM_ANCHOR_2026-04-23.md`)

- **Bottom-line**: clean video + first-frame mask → processed video that is visually faithful AND causes SAM2 to lose the target across the entire processed video.
- **Must-solve bottleneck**: SAM2.1-Tiny tracks via f0 + current-frame; non-cond FIFO bank architecturally marginal (B2). Attack must target causal pathways under realistic fidelity.
- **Non-goals**: clean-suffix eval; FIFO-self-healing defeat; UAP; runtime hook.
- **Constraints**: white-box SAM2.1-Tiny, per-video PGD, pixels-only, two-tier fidelity budget, DAVIS-2017 val, ≤ 15 GPU-min/clip.
- **Success**: DAVIS-10 — (1) mean J&F drop ≥ 0.40; (2) fidelity met; (3) restoration attribution confirms damage in f0/current-frame.

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | What this round simplified / modernized | Solved? | Remaining risk |
|-------|---|---|---|---|
| 1 | Supervision leakage (m_true_t undefined); method reads as standard PGD; ablation non-specific; feasibility underspecified; scope (SAM2-family) weak. | GT-free clean-SAM2 self-supervision; C2 promoted mandatory; restoration counterfactuals replace drop-based ablation; STE+memory+windowing+pilot-gate; SAM2Long main-row promotion. | Mostly (6.7 → 7.7) | Sign bug in restoration metric; R1/R2 potentially redundant; novelty still at risk. |
| 2 | Sign flipped on ΔJ_restore; R1/R2 non-additivity; causal framing still loose. | Sign fixed; R12 joint restoration added with predefined protocol; causal loop framing; C2 fully folded into C1; soft-logit supervision; decoy regime deleted. | Yes (7.7 → 8.4) | No formulation issues; empirical feasibility still unproven. |
| 3 | Empirical risk (no pilot data); could benefit from pilot-gate as first-class element; fallback loss undefined. | Pilot-gate promoted to first-class (4 pre-committed branches B1-B4); pathway-aligned fallback loss added as SWAP; confidence-weighted pseudo-mask. | Partial (8.4 plateau) | Sign error in fallback loss caught at Round 4. |
| 4 | **Sign bug in fallback loss**: `∥h_attack − h_clean∥` minimized preserves clean features. | Sign corrected via cosine anti-alignment with detached clean features. | Yes (8.4 stable) | Pre-pilot ceiling confirmed; READY requires empirical data. |
| 5 | Sign-fix validation + ceiling confirmation. | Confirmed sign correct; ceiling reached; specified m_f0 cosine over (maskmem ⊕ obj_ptr, L2-normalized per channel). | Yes (8.4 ceiling) | Remaining gap is empirical — run the pilot. |

## Overall Evolution

- **How the method became more concrete**: GT-free self-supervision defined (clean-SAM2 soft logits + confidence weighting), three swap hooks specified at exact SAM2 intercept points, fake-quantize STE, bf16, sliding-window backward, pilot-gate rules all pinned to specific numbers.
- **How the dominant contribution became more focused**: separated-two-claims (C1 + C2) structure collapsed into a single four-part causal-loop thesis: diagnose (bank marginality) → attack (pathway-targeted PGD) → verify (restoration counterfactuals). Pilot-gate branches became the fourth part of C1, preventing post-hoc narrative reshuffle.
- **How unnecessary complexity was removed**: decoy regime (L_rank, m_decoy, offset heuristic) deleted from default. L_stale kept only as diagnostic. Augmented-Lagrangian replaced with hinge + best-feasible checkpoint (avoids the v2 μ-saturation pathology).
- **How modern technical leverage improved**: SAM2.1 itself is the foundation-model primitive (target + white-box surrogate). Clean-SAM2 soft-logit confidence-weighted self-supervision is a natural frontier choice. Restoration-counterfactual attribution is a standard interpretability technique applied in a streaming VOS setting. LLM/diffusion/RL explicitly rejected as decoration.
- **How drift was avoided or corrected**: Round-1 CRITICAL supervision-leakage concern (`m_true_t` potentially GT) addressed by defining GT-free clean-SAM2 pseudo-labels. Round-2 conditional drift flag ("checkpoint selection must also be GT-free") addressed explicitly. No final drift warning.

## Final Status

- **Anchor status**: preserved throughout all 5 rounds.
- **Focus status**: tight. One trainable component (δ), one paper thesis (causal loop).
- **Modernity status**: appropriately frontier-aware. SAM2.1 surrogate + clean-SAM2 soft logits + restoration counterfactuals. No decoration primitives.
- **Strongest parts of final method**:
  1. Architecture-aware design principle (bank is known non-causal → don't attack it).
  2. Two-tier fidelity budget exploiting SAM2's prompt-frame structure.
  3. Pilot-gate with pre-committed 4-branch narrative — paper is robust to any empirical outcome.
  4. Restoration-counterfactual attribution with predefined interpretation protocol.
- **Remaining weaknesses**:
  1. The primary empirical claim (J&F drop ≥ 0.40 at F_lpips=0.20) is untested. Prior history (v2 R001/R002/R003) had J-drop ≈ 0.001 on clean-suffix eval — but that was a different eval regime. On the new (whole-video, processed-frame) eval, we have NO prior data points. Pilot is mandatory before full commit.
  2. B4 branch (pilot-failure pivot) is honestly a different paper. If B4 triggers, the "dataset protection succeeds" claim is retracted in favor of an attack-surface analysis paper.
  3. SAM2Long install on Pro 6000 still pending (2-3 GPU-hours); transfer row in main table depends on it.

## Output Files

- **Full refined proposal**: `refine-logs/FINAL_PROPOSAL.md`
- **Round logs**: `refine-logs/round-0-initial-proposal.md`, `round-N-review.md`, `round-N-refinement.md` (N=1..4)
- **Score evolution**: `refine-logs/score-history.md`
- **Problem anchor**: `refine-logs/PROBLEM_ANCHOR_2026-04-23.md`
- **Prior v2 proposal (archived, falsified)**: `refine-logs/archive-2026-04-22-v2-falsified__*.md`
