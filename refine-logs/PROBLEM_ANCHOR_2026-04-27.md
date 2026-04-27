# Problem Anchor (frozen, 2026-04-27)

This anchor is reused VERBATIM in every round. Reviewer suggestions that change this anchor are flagged as drift.

---

## Bottom-line problem

Design a publisher-side adversarial attack on SAM2 video segmentation that, given a clean video and a first-frame prompt mask, produces a modified video which (a) drives SAM2's per-frame Jaccard score down on the post-export uint8 artifact, and (b) does so using BOTH internal decoy frame insertion AND sparse δ on original (bridge) frames adjacent to inserts. The mechanism must be defendable as a publishable contribution at a top venue (AAAI target).

## Must-solve bottleneck

Existing prior art is insufficient on the SAM2 + insert-+-original-δ combination:

- **Chen WACV 2021** appends dummy frames + δ on inserts only → targets video classification, no original-frame modification, frames APPENDED at end.
- **UAP-SAM2 NeurIPS 2025** (arxiv 2510.24195) attacks SAM2 with DENSE δ on existing frames, NO insertion, universal perturbation. mIoU 76→33.67%.
- **Li T-CSVT 2023** (Hard Region Discovery, arxiv 2309.13857) attacks pre-SAM2 VOS (STM/HMMN/STCN) with first-frame δ only, no insertion. J&F drops 4-7 points.
- **PATA arxiv 2310.10010** is single-image SAM v1, not video, no insertion.

NONE combines (i) internal insertion, (ii) sparse δ on adjacent ORIGINAL frames, (iii) explicit exploitation of SAM2's prompt-conditioned temporal memory propagation, (iv) per-clip targeted attack with adaptive selection at export.

The empirical bottleneck observed in our v4.1 dev-4: 3/4 strong-A0 clips strictly improve under joint (apply); 1/4 reverts to A0. Hence the wrapper-level claim is defensible but the mechanism-level claim ("bridge δ on originals is uniformly beneficial") is NOT YET established. We need the method + experiments to make the mechanism-level contribution as defensible as the wrapper-level one, OR explicitly accept the wrapper-level framing.

## Non-goals (explicitly NOT pursued)

- Pure suppression / object-score margin attack (CLAUDE.md hard rule)
- Pure-δ method without insertion (CLAUDE.md hard rule)
- Pure insertion without original-frame δ (this round's user constraint)
- First-frame-only attack (Li 2023 territory)
- Universal perturbation (UAP-SAM2 territory)
- Single-image SAM v1 (PATA territory)
- Audit / falsification / negative-results paper (CLAUDE.md hard rule)
- Pivoting to "joint hurts, just use insert-only" (RESEARCH_REVIEW_JOINT_VS_ONLY hard rule)
- Bank-poisoning narrative (B2 ablation falsified)
- Defeat-FIFO-self-healing narrative (also falsified)

## Constraints

- White-box SAM2.1-Tiny on Pro 6000 96GB ×2 (1 GPU per user default; 2 with admin OK; share when other students online).
- Per-clip targeted attack on DAVIS-2017 evaluation set; TEN-clip held-out set is the headline experiment.
- Two-tier fidelity: f0 (prompt frame) protected at ε=2/255 + SSIM≥0.98; other originals at ε=4/255 + per-frame LPIPS≤0.20; insert frames at LPIPS≤0.35 vs neighbors + TV ≤ 1.2× base.
- Joint method must keep BOTH insertion and original-frame δ active.
- Adaptive accept/revert wrapper allowed and is part of the method.
- AAAI submission target. Must address all 4 cited prior arts (Chen WACV 2021, UAP-SAM2 NeurIPS 2025, Li T-CSVT 2023, PATA 2310.10010) in related work + experimental comparison where applicable.
- Reviewers must be told the method via threat model + mechanism, not via 5 sub-component contributions.

## Success condition

The held-out 10-clip eval (post v4.1 / v5 method finalization) must satisfy ALL of:

| Gate | Threshold |
|---|---|
| Strict joint > only on | ≥ 5/10 clips |
| Mean paired lift (joint − only) | ≥ +0.05 |
| Median paired lift | > 0 |
| Top contributing clip share of total lift | < 40% |
| Polish_applied rate | ≥ 60% |
| Mean exported J-drop (joint) | ≥ 0.55 |

If ALL satisfied → headline "adaptive joint substantively beats insert-only" claim is defensible.

PLUS, the method must support ≥ 3 reviewer-proof ablations:

1. **Insert-only vs Insert+bridge-δ paired comparison** (proves bridge δ contributes).
2. **Vulnerability-aware vs random placement** (proves placement matters).
3. **Mechanism analysis**: per-frame J-drop trajectory + SAM2 memory feature divergence over time, comparing insert-only and insert+δ (supports "memory hijack" mechanism story).

The novelty positioning per `NOVELTY_CHECK_AAAI_2026-04-27.md` must be defendable: "First **internal-insertion** attack on prompt-driven VOS / SAM2 with sparse bridge δ exploiting temporal memory propagation."
