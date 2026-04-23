# Problem Anchor — MemoryShield v3 (dataset-protection reframe)

**Frozen 2026-04-23.** This anchor supersedes `PROBLEM_ANCHOR_2026-04-22.md`
after experimental invalidation of the prior "clean-suffix poisoning"
narrative (see `AUTO_REVIEW.md` Rounds 1–4 and `RESEARCH_REVIEW_v2_vs_v4.md`).

## Bottom-line problem

Given a clean source video and a first-frame target mask, produce a
**processed video** such that:

1. A downstream SAM2 user, given the processed video AND the same first-
   frame mask as a prompt, obtains **substantially degraded target
   segmentation across the entire processed video** (not just the
   original prefix).
2. The processed video remains **visually faithful** to the source under
   a fidelity budget calibrated to realistic perception (a human
   looking at the processed video should perceive it as a valid /
   unmodified version of the source content).

This is a **dataset-protection** threat model: a data publisher
processes every frame once and ships the result. The consumer sees
only the processed frames — there is no "clean suffix" to recover
from.

## Must-solve bottleneck

Under a fidelity budget that respects the floor study (natural DAVIS
adjacent-frame LPIPS = 0.25-0.38 on moving clips, 0.09 on static clips;
ProPainter insert floor 0.67-0.89), SAM2.1 is still robust. It tracks
the target from the f0 conditioning slot + current-frame Hiera
features. The non-conditioning FIFO memory bank is architecturally
marginal (B2 causal ablation: `|delta_J| < 0.01` across 5 clips when
the bank is removed). Therefore:

- Memory-bank poisoning is **not an available bottleneck** on SAM2.1-Tiny.
- The attack must concentrate on the actual causal pathways: **f0
  conditioning + current-frame image features**.
- It must do so under a **tight fidelity budget** that the floor study
  shows is achievable (not the infeasible 0.10 LPIPS).

## Non-goals

- NOT "defeat FIFO self-healing" — self-healing is not the dominant
  recovery mechanism on SAM2.1-Tiny (B2 falsifies it).
- NOT "future clean frames segment badly because of poisoned memory" —
  the dataset-protection threat model has no clean frames downstream.
- NOT universal perturbation (UAP-SAM2 is universal per-model; we are
  per-video and condition on the target mask).
- NOT runtime hook / training-time backdoor / weight modification.
- NOT requiring the attacker's cooperation in prompt selection — the
  attack must hold under the **same** first-frame prompt the publisher
  assumes the consumer will use.

## Constraints

- **Target model**: SAM2.1-Tiny (primary, available at Pro 6000). Transfer
  to SAM2.1-Base/Large and SAM2Long evaluated as an appendix, not a
  primary claim.
- **Access**: white-box SAM2.1 surrogate (weights locally available).
- **Output format**: per-video pixel perturbation; publisher may also
  insert a small number of synthetic frames (K_ins ≤ 3) within the
  video if it improves the fidelity / attack tradeoff.
- **Fidelity budget** (tentative, to be grounded by an updated floor
  study in Phase 1.5):
  - LPIPS(processed_frame, source_frame) per-frame mean ≤ `F_lpips`,
    with `F_lpips` chosen so that natural DAVIS adjacent-frame LPIPS
    on moving clips is comparable. Preliminary: `F_lpips ≈ 0.15-0.20`.
  - SSIM(processed, source) per-frame mean ≥ 0.95 for non-insert
    frames, ≥ 0.90 for insert frames.
  - ε_∞ per-frame: 4/255 on non-insert frames, 2/255 on f0 (the
    conditioning-prompt frame) to keep the mask prompt honest.
- **Dataset**: DAVIS-2017 val (10-clip primary subset; full 30-clip
  appendix).
- **Compute**: per-video PGD ≤ ~10 minutes per clip on a single
  Pro 6000; full 10-clip experiment in ~2 GPU-hours.

## Success condition

A run is considered evidence that the method addresses the actual
problem when **all three** hold on the 10-clip DAVIS subset:

1. **Segmentation damage**: mean J&F on the processed video (SAM2
   propagate from the same first-frame mask as prompt) drops by at
   least **0.40** compared to the clean baseline.
2. **Fidelity**: all per-frame LPIPS ≤ `F_lpips`, all non-insert SSIM
   ≥ 0.95 (f0 ≥ 0.98 to keep the mask prompt honest).
3. **Mechanism attribution** (what makes this a paper, not just a
   number): a causal ablation distinguishes WHICH SAM2 pathway the
   attack damages. Minimum: an ablation showing the J-drop
   concentrates on the current-frame image-feature pathway (not the
   non-cond bank), confirming the attack targets the pathway that
   actually matters.

Transfer evaluation (SAM2Long + larger SAM2.1 variant) is a
**secondary** success condition — the paper can still land if J-drop
≥ 0.40 holds on SAM2.1-Tiny even without transfer, provided the
mechanism attribution is clean.

## What fails this anchor (i.e. what counts as drift)

- Re-introducing "clean-suffix eval" as the primary evaluation (that
  was the v2 framing; it is now a non-goal).
- Claiming "FIFO self-healing defeated" or "memory bank hijack is the
  mechanism" without a causal demonstration (B2 data currently says
  otherwise on SAM2.1-Tiny).
- Accepting a fidelity budget that contradicts the floor study.
- Adding learned generators / diffusion models / LLM planners / RL
  agents without a specific bottleneck they solve.
- Contribution sprawl: more than one dominant mechanism claim + one
  supporting claim.

## Key facts carried forward from Rounds 1–4 of the prior refinement loop

- `L_stale` (3-bin KL on memory-attention mass) is **diagnostic only**.
  High `A_insert` does not cause J-drop on SAM2.1-Tiny.
- ProPainter-quality inserts have LPIPS 0.67-0.89 to predecessor,
  independent of decoy offset. Inserts are fidelity-expensive.
- Fake uint8 quantization in the PGD loop and `track_step`-based
  surrogate are the right infrastructure (per "MemoryShield
  breakthrough" memory).
- The `SAM2VideoAdapter` (Chunk 5b-ii) correctly bypasses
  `@torch.inference_mode` to enable autograd through the full SAM2
  graph; this is reusable.
- Augmented-Lagrangian μ saturation to 10000 under infeasible budgets
  is a known pathology — the new method must use a feasible budget
  AND/OR check-pointed return of best-feasible state.
