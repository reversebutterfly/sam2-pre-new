# Review Summary — MemoryShield Preprocessor Refinement

- **Problem**: User-controlled preprocessor for protecting video data from SAM2-family VOS.
- **Initial Approach**: Insert frames cause SAM2 to lose target + perturbed frames prevent recovery.
- **Date**: 2026-04-22
- **Rounds**: 4 / 5
- **Final Score**: 9.2 / 10
- **Final Verdict**: READY

## Problem Anchor (verbatim, never drifted)

- Preprocessor takes clean video + first-frame mask, outputs modified video causing SAM2-family VOS to lose target and not recover, within tight fidelity budget.
- Bottleneck: FIFO streaming memory (num_maskmem=7) self-heals from single-frame perturbations.
- Non-goals: UAP / backdoor / runtime hook / maximal attack.
- Success condition: eval-window J-drop ≥ 0.55 AND low rebound / post-loss AUC AND fidelity triad AND each phase necessary (≥ 40% relative loss if removed) AND SAM2Long transfer.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | Phase-2 loss hallucinates decoy on clean frames; "any SAM2-style" overclaim; no resonance ablation; SAM2Long optional; monotone-drop metric brittle | Dropped decoy supervision in Phase 2; narrowed to SAM2-family; added schedule ablation; made SAM2Long mandatory; switched to rebound / post-loss AUC | Partial | L_stale and schedule still underspecified |
| 2 | Sign error in confidence lock; three clocks unformalized; L_stale 2-way ratio; off-resonance confounded by recency; short optimization window confused with reporting | Corrected sign with logsumexp; formalized 3 clocks; L_stale → 3-bin KL; recency-matched off-resonance; full-suffix reporting window | Partial | CVaR domain + resolution invariance pending |
| 3 | Confidence-lock not resolution-invariant; CVaR zero-contamination; schedule claim too strong ("pure resonance law"); minor Q rationale | logmeanexp; masked-CVaR over SET; renamed to "write-aligned seed-plus-boundary"; Q rationale + sensitivity optional | Yes | None that block READY |
| 4 | None (final check) | None (READY at 9.2/10) | Yes | Execution risk only |

## Overall Evolution

- **Problem anchor preserved across all 4 rounds.** Reviewer never flagged drift.
- Method became concretely smaller, not larger: dropped one loss term (clean-decoy BCE), one branching fallback (distractor-mode suppression), and one dual-option (Unimatch flow).
- Dominant contribution sharpened from "two-phase attack" (4-ish subcomponents felt) to strict "Phase 1 = loss event, Phase 2 = recovery prevention" with `L_stale` explicitly framed as Phase 2's internal regularizer, not a 3rd contribution.
- Modern technical leverage stable throughout: ProPainter + RAFT + LPIPS, no LLM / VLM / diffusion / RL pressure.
- Schedule claim anchored directly in `num_maskmem`, with three-clock formalization and matched-recency ablation — now a reproducible mechanism claim, not a heuristic.

## Final Status

- **Anchor status**: preserved
- **Focus status**: tight (1 dominant + 1 supporting, with `L_stale` correctly subordinate)
- **Modernity status**: appropriately frontier-aware (ProPainter as frozen generator; no gratuitous modern components)
- **Strongest parts of final method**:
  - Two-phase composition story is causally minimal and directly targets the bottleneck (FIFO self-healing)
  - `L_stale` = 3-bin KL over bank-attention is a principled memory-hijack regularizer
  - Write-aligned seed-plus-boundary schedule is parameterized by `num_maskmem`, not hand-tuned
  - Matched-recency off-resonance ablation is a clean isolation test
  - Four claims cover composition / recovery-prevention / transfer / schedule mechanism
- **Remaining weaknesses** (non-blocking):
  - Execution risk (L_stale gradient stability through memory-attention; ProPainter LPIPS floor compatibility)
  - DAVIS-30 numbers not yet produced
  - SAM2Long transfer assumption unproven at claimed magnitudes
  - Reviewer expects pre-registration of `τ_conf`, `β`, Q sensitivity ranges
