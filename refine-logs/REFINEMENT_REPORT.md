# Refinement Report — MemoryShield Preprocessor

- **Problem**: User-controlled preprocessor for protecting video data from SAM2-family promptable VOS.
- **Initial Approach**: Insert frames cause SAM2 to lose target + perturbed frames prevent recovery.
- **Date**: 2026-04-22
- **Rounds**: 4 / 5
- **Final Score**: 9.2 / 10
- **Final Verdict**: READY
- **Thread**: `019db31e-38dd-7fc1-8d72-c4dea843b254`

## Problem Anchor (verbatim across all 4 rounds)

- Preprocessor takes clean video + first-frame mask of a target to protect, outputs modified video causing SAM2-family promptable VOS to lose target and not recover, within insert LPIPS ≤ 0.10 and attacked-originals SSIM ≥ 0.97.
- Bottleneck: FIFO memory bank (num_maskmem=7) self-heals from single-frame perturbations in ≤ 6 writes.
- Non-goals: UAP / backdoor / runtime hook / maximal attack.
- Success: eval-window J-drop ≥ 0.55 AND low rebound / post-loss AUC AND each phase necessary (≥ 40% relative loss if removed) AND SAM2Long transfer.

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`
- Round files: `round-{0..3}-initial-proposal|review|refinement.md` + `round-4` review in this report
- Problem anchor: `refine-logs/PROBLEM_ANCHOR_2026-04-22.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 6                  | 6                    | 8                 | 6           | 6                | 6               | 6.6     | REVISE  |
| 2     | 9                | 7                  | 8                    | 9                 | 7           | 7                | 7               | 7.9     | REVISE  |
| 3     | 9                | 8                  | 9                    | 9                 | 8           | 8                | 8               | 8.6     | REVISE  |
| 4     | 10               | 9                  | 9                    | 9                 | 9           | 9                | 9               | **9.2** | **READY** |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|---|---|---|---|
| 1 | (CRITICAL) Phase-2 hallucinates decoy on clean frames; claim scope "any SAM2-style" overclaim; FIFO-resonance not directly tested; SAM2Long optional; "monotone drop" brittle. (IMPORTANT) 4-contribution feel; warmup backward; ν search space too large | Dropped clean-decoy BCE → suppression + low-conf lock; narrowed to "SAM2-family"; added schedule ablation; made SAM2Long mandatory; rebound + post-loss AUC; L_stale subordinated; reversed warmup; ν restricted to paste+seam; cache clean-suffix | Partial — R2 pending on sign, clocks, L_stale, deconfound |
| 2 | (CRITICAL) Sign error in confidence lock; three clocks unformalized; reporting window confused with optimization window. (IMPORTANT) L_stale 2-way ratio misses "other"; off-resonance confounded by recency; ROI supervision pending | Fixed sign (logsumexp, later logmeanexp); formalized O/M/W clocks; write-aligned schedule; L_stale → 3-bin KL; matched-recency off-resonance (m_3=14 pinned); full-suffix metrics reported; ROI-BCE; RAFT only; SSIM out of loss | Partial — R3 pending on resolution invariance, CVaR domain, schedule naming |
| 3 | (IMPORTANT) logsumexp scales with HW → resolution-dependent; CVaR on 1[·] zero-contamination; schedule claim too strong. (MINOR) Q rationale | logmeanexp; masked-CVaR over SET `{g(p) : p ∈ C}`; renamed "write-aligned seed-plus-boundary"; added offset sweep; Q rationale + optional sensitivity ablation | Yes — all issues closed |
| 4 | None blocking | Final ready-check confirms 9.2/10 READY | READY |

## Final Proposal Snapshot (canonical version in FINAL_PROPOSAL.md)

- **One-sentence thesis**: Protect a target object by combining (i) K_ins=3 synthetic inserts at a write-aligned seed-plus-boundary schedule that create a tracking-loss event by populating the FIFO bank with mislocated entries, and (ii) L∞ ≤ 4/255 prefix perturbations that suppress target re-acquisition under a 3-bin categorical KL memory-staleness regularizer keeping bank-attention on inserted slots.
- **Dominant contribution**: two-phase preprocessor mechanism against SAM2-family VOS, with each phase mapped to one step of the self-healing attack surface (loss ← inserts; recovery prevention ← prefix perturbation + L_stale).
- **Supporting contribution**: write-aligned seed-plus-boundary schedule parameterized by `num_maskmem`, tested at matched recency.
- **Expected paper-grade results**: DAVIS-10 Full-method J-drop ≥ 0.55 on full suffix, rebound ≤ 0.15, SAM2Long retention ≥ 0.40, resonance ≥ off-resonance by ≥ 20pp.

## Method Evolution Highlights

1. **Phase 2 objective rewrite** (Round 1 → Round 2): from "force decoy tracking on clean frames" (hallucination demand) to "suppress true + low-confidence lock" (no hallucination) — this was the single biggest conceptual fix.
2. **Three-clock formalization** (Round 2): original / modified / write indices made explicit; schedule defined on Clock W. Made the FIFO-resonance claim reproducible rather than descriptive.
3. **L_stale reformulation** (Round 2 → Round 3): from 2-way ratio to 3-bin categorical KL over `{insert, recent, other}`, preventing attention-collapse from looking like success.
4. **Schedule claim name tightening** (Round 3): "FIFO-resonant" → "write-aligned seed-plus-boundary" with matched-recency controls, aligning the proof with what the ablation isolates.
5. **Resolution-invariant confidence lock** (Round 3): logsumexp → logmeanexp, so τ_conf means the same thing at any resolution.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|---|---|---|---|
| 1 | "Force decoy tracking on clean frames" is a harder target than the anchor requires | Accepted fully — changed Phase 2 to suppression + confidence lock | Accepted |
| 1 | Narrow claim to SAM2-family or add second target family | Accepted narrow (avoid sprawl) | Accepted |
| 2 | Schedule must be defined on write count, not absolute frame indices | Accepted; added three-clock formalization | Accepted |
| 3 | Schedule claim tighten to "write-aligned seed-plus-boundary" | Accepted; added offset sweep for extra robustness | Accepted |
| — | NO reviewer suggestion triggered drift of the Problem Anchor | — | — |

## Remaining Weaknesses (non-blocking)

1. **Execution risk**: L_stale gradients through SAM2's memory-attention not yet empirically validated stable; margin-form fallback is pre-declared but may itself need tuning.
2. **ProPainter compatibility with ν LPIPS ≤ 0.10**: realization-gap assumption — reviewer did not press this in the refine thread because R2 focus was on loss design, but the fidelity loop with Pilot A/B/C showed prior attempts had LPIPS ≈ 0.13-0.14 floors with Poisson base; ProPainter is expected to help but unverified.
3. **DAVIS-30 numbers**: proposal sizes the compute at ~24 GPU-hours but no full-scale result yet.
4. **Pre-registration**: reviewer's non-blocking note to declare small sensitivity ranges for τ_conf, β, Q BEFORE full experiments; must be observed when implementing.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review — full verbatim</summary>

[See `round-1-review.md` for full text.]

</details>

<details>
<summary>Round 2 Review — full verbatim</summary>

[See `round-2-review.md` for full text.]

</details>

<details>
<summary>Round 3 Review — full verbatim</summary>

[See `round-3-review.md` for full text.]

</details>

<details>
<summary>Round 4 Ready-Check — full verbatim</summary>

Yes. This is now **READY** as a proposal.

**Scores**
1. **Problem Fidelity:** 10/10
   It is cleanly anchored on the original problem: offline preprocessor protection against **SAM2-family self-healing**.
2. **Method Specificity:** 9/10
   Losses, clocks, slot definitions, optimization stages, and reporting windows are now concrete enough to implement without guessing.
3. **Contribution Quality:** 9/10
   The paper now has one dominant mechanism-level contribution and one clearly subordinate schedule/mechanism support claim.
4. **Frontier Leverage:** 9/10
   ProPainter + flow + LPIPS are appropriate; no gratuitous modern components.
5. **Feasibility:** 9/10
   The ROI-restricted `ν`, cached clean-suffix embeddings, and staged PGD make the stated compute budget credible.
6. **Validation Focus:** 9/10
   The validation is now tight: composition, `L_stale`, mandatory transfer, and schedule mechanism are all directly tested.
7. **Venue Readiness:** 9/10
   If the implementation matches the proposal and the expected gaps materialize, this is sharp enough for a top-venue submission.

**OVERALL:** **9.2/10**

**Verdict:** **READY**

**Drift Warning:** **NONE**

**Why it crosses the bar**
- The dominant contribution is now genuinely sharp: `Phase 1 creates loss`, `Phase 2 prevents reacquisition`.
- The method is simpler than before, not larger.
- The remaining uncertainty is now empirical execution risk, not proposal incoherence or mechanism ambiguity.

**Simplification Opportunities**: NONE
**Modernization Opportunities**: NONE

**Non-blocking notes**
- In the final writeup, define `A_u^recent` explicitly as attention mass on the **currently resident** clean-prefix memory slots, not all historical prefix frames, to avoid reviewer confusion.
- Keep Claim 4 phrased exactly as you revised it: **write-aligned seed-plus-boundary schedule**, not a stronger "pure resonance law" claim.
- Pre-register a small sensitivity range for `τ_conf`, `β`, and `Q` so reviewers cannot dismiss the method as heavily tuned.

At this point I would stop revising the proposal and move to implementation.

</details>

## Next Steps

1. `/experiment-plan` — turn this FINAL_PROPOSAL into a detailed execution roadmap (pre-registered sensitivity ranges, run order, checkpoints).
2. Implementation priority: re-run the Pilot A realization-gap test from the prior fidelity loop but now with ProPainter base (not Poisson) to confirm LPIPS ≤ 0.10 is realistic before committing to the full ablation suite.
3. If ProPainter base fails LPIPS goal, fall back to reviewer's non-blocking relaxation — keep L_stale + schedule as the main mechanism claims, accept Pareto tradeoff for fidelity.
