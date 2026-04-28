# Idea Discovery Report — ChronoCloak (publisher-side temporal state-injection cloak for SAM2-class VOS)

**Direction**: Deep-read 4 anchor papers (Liu CVPR 2025 / UAP-SAM2 NeurIPS 2025 / BB-SAM TMM 2025 / HardRegion VOS TCSVT 2024) and propose ONE feasible AAAI scheme.
**Date**: 2026-04-28
**Pipeline used**: research-review (codex round 2 in thread `019dd243-04d2-7111-9eb0-c4eb3fec729d`) — full /idea-discovery flow compressed because direction was already pinned by 2026-04-28 publisher-side pivot.
**Companion docs**: `PIVOT_REVIEW_2026-04-28.md` (publisher-side direction approval), `CHRONOCLOAK_REVIEW_2026-04-28.md` (codex's compact review of this proposal).
**Prior version**: `IDEA_REPORT_2026-04-15.md` (archived).

---

## Executive Summary

**Recommended idea**: **ChronoCloak** — A publisher-side temporal state-injection cloak for prompt-driven video segmentation, where natural interstitial frame insertion is the primary attack surface and invisible bridge perturbations only stabilize the induced state corruption.

**Codex novelty score (round 2)**: 6/10 as initially drafted; ≥7/10 if the "temporal state injection" axis is elevated to the central scientific contribution and proven with a matched-budget causal ablation.

**Mock AAAI score**: 5/10 → 6.5/10 with the recommended changes.

**Final verdict**: **GO-WITH-CHANGES** — direction is correct; framing must demote engineering to implementation detail and elevate the science.

---

## Anchor-paper landscape (codex round 2)

### A1. Liu et al. CVPR 2025 — "Protecting Your Video Content"
- **Contribution**: proactive publisher-side video watermark; two attack families (`Rambling-F/L` for wrong captions, `Mute-S/N` for premature EOS).
- **Threat model**: video owner protects against unauthorized video-LLM annotation (`Video-ChatGPT`, `Video-LLaMA`, `Video-Vicuna`); white-box on victim video-LLM by default.
- **Metrics**: CLIP, BLEU, caption length, EOS rate, downstream VQAA/VQAT.
- **Most damaging overlap**: **the publisher-side framing itself**.
- **Our delta**: target service is prompt-driven *video segmentation*, not text-output video-LLMs; perturbation surface is *temporal state injection through inserted interstitial frames*, not additive watermarks.

### A2. UAP-SAM2 / "Vanish into Thin Air" NeurIPS 2025 Spotlight
- **Contribution**: cross-prompt UNIVERSAL adversarial perturbation for SAM2 via target-scanning + dual semantic deviation (semantic confusion + feature shift + memory misalignment).
- **Threat model**: open-source SAM2 surrogate + public datasets; cross-prompt + cross-dataset transfer.
- **Metrics**: mIoU on YouTube-VOS / DAVIS / MOSE under point + box prompts.
- **Most damaging overlap**: **SAM2-specific prompt transfer + temporal/memory disruption already claimed**.
- **Our delta**: **non-native-frame insertion as the attack surface**. UAP-SAM2 is a classical additive UAP. We claim that realistic interstitial frame *insertion* (not perturbation) plus short bridge stabilization opens a new perturbation surface.

### A3. Black-Box Targeted SAM (BB-SAM, IEEE TMM 2025)
- **Contribution**: prompt-agnostic targeted attack on image SAM via encoder-space PATA / PATA++ regularizer; transfer-based black-box (NOT classical query-based — codex correction).
- **Threat model**: image SAM, targeted (force imitation of target mask), transfer black-box from SAM-B to SAM-L/H.
- **Metrics**: IoU/mIoU between adv mask and target mask (higher is better).
- **Most damaging overlap**: prompt-agnostic encoder-space optimization.
- **Our delta**: video, memory-bearing model, publisher-side, non-targeted, temporal-state corruption (not feature mimicry).

### A4. Hard Region Discovery VOS (TCSVT 2024)
- **Contribution**: ARA — first-frame attack that learns hardness-map from gradients and concentrates δ on hard fg/bg boundary regions.
- **Threat model**: white-box semi-supervised VOS targeting STM / HMMN / STCN / AOT; black-box variant studied.
- **Metrics**: J&F on DAVIS 2016/2017 + YouTube-VOS official.
- **Most damaging overlap**: sparse-in-time attack story (early-frame perturbation hurts later).
- **Our delta**: target SAM2 (memory-bank class); inserted-frame *memory-write hijack* (we have A3 4/4 STRONG causal evidence) vs spatial hard-region perturbation.

---

## Gap Matrix

| Method            | Target service              | Perturbation surface                                        | Fidelity budget                            | Primary metric                          | Mechanism evidence                              | Transferability claim                |
|-------------------|-----------------------------|-------------------------------------------------------------|--------------------------------------------|-----------------------------------------|-------------------------------------------------|--------------------------------------|
| Liu CVPR25        | Video-LLM annotation        | Additive watermark on native frames                         | L∞=16/255                                  | CLIP, BLEU, caption length, EOS, VQAA/T  | Caption / logit / EOS manipulation; no temporal causal proof | Across 3 video-LLMs; prompt transfer |
| UAP-SAM2          | SAM2 image+video VOS        | Additive UAP + sample-wise variant                          | UAP 10/255; sample-wise 8/255              | mIoU (lower better)                     | Semantic confusion + feature shift + memory misalignment | Cross-prompt, cross-dataset, cross-model, → SAM2long |
| BB-SAM            | Image SAM                    | Additive image perturbation via encoder attack              | 4/255–16/255 (8/255 highlighted)           | mIoU to target mask (higher better)     | Feature mimicry + dominance regularizer         | Cross-prompt, cross-model            |
| HardRegion VOS    | Classic VOS (STM/HMMN/STCN/AOT) | First-frame additive δ weighted by hardness map             | small / imperceptible (default ε not verified) | J&F DAVIS, YT-VOS official              | First-frame gradient hardness learner            | White-box main; black-box variant    |
| **ChronoCloak**   | Publisher-side prompt-driven SAM2-class VOS | **Inserted interpolation interstitials + L=2 bridge δ on real frames + accept/revert wrapper** | insert ε'=2/255, bridge ε=2/255, mean LPIPS≤0.03, SSIM≥0.99 | **original-frames-only J + UTR + SFR + re-prompt burden + human stealth** | A3 memory-write blocking 4/4 STRONG (need to extend) | Limited same-family prompt transfer (don't overclaim) |

---

## ChronoCloak — Final Method (codex GO-WITH-CHANGES)

### One-line pitch (verbatim from codex)
> "A publisher-side temporal state-injection cloak for prompt-driven video segmentation, where natural interstitial frame insertion is the primary attack surface and invisible bridge perturbations only stabilize the induced state corruption."

### Core scientific axis (THE thing the paper sells)
**Temporal state injection** is a new, distinct perturbation surface for memory-bank-class video segmentation models — not additive δ on native frames (UAP-SAM2 family), not first-frame spatial concentration (HardRegion family), not text-token EOS manipulation (Liu family). It exploits a property unique to memory-bank prompt-driven segmentation: that the model writes per-frame state into a queue used by future frames, so a *single inserted frame's* memory-write can be hijacked while leaving the visible video stream perceptually clean.

### Method components (priority-ordered)

1. **[CORE] Interpolated decoy frame** — between original I_t and I_{t+1}, generate an interstitial frame via a frozen interpolator (RIFE / FILM / IFRNet) and apply a small adversarial steering ν' (ε'=2/255) optimized to corrupt SAM2's cross-attention readout when SAM2 processes that decoy. **Replaces** oracle-trajectory composite (which produced visible double-object ghosting in the v4.1 viz).
2. **[CORE] Sparse bridge δ** — only on the next L=2 real frames after each insert; ε=2/255, mean LPIPS ≤ 0.03, 95th-%ile ≤ 0.05, SSIM ≥ 0.99. Frames outside the bridge window untouched.
3. **[IMPLEMENTATION DETAIL] Robust placement** — vulnerability-aware top-K with random fallback to fix the inversion problem. Demoted from flagship to implementation detail per codex.
4. **[IMPLEMENTATION DETAIL] GT-free training** — clean-SAM2 pseudo-labels with confidence weighting; GT only for evaluation.
5. **[IMPLEMENTATION DETAIL] Adaptive accept/revert wrapper** — accept perturbation only if (effect ≥ θ_eff AND stealth ≥ θ_stealth); else revert that clip to insert-only or no-op.
6. **[SCIENTIFIC SUPPORT] Memory-hijack mechanism evidence** — A3 causal ablation extended from current 4 clips to all 13.

### Fidelity budgets (publisher-side, tight)
- Prompt frame: ε ≤ 1/255
- Bridge frames: ε ≤ 2/255
- Mean LPIPS on perturbed real frames: ≤ 0.03
- 95th-percentile LPIPS: ≤ 0.05
- Mean SSIM: ≥ 0.99
- Inserted frames: not judged by ε; require human flipbook pass-rate + temporal-consistency check

### Primary metrics
- **Original-frames-only J vs DAVIS GT** (excludes inserted frames from J)
- **Unusable-Track Rate** (UTR): fraction of clips with mean original-frame J < 0.5
- **Sustained Failure Rate** (SFR): fraction of clips with ≥5 consecutive original frames at J < 0.3
- **Re-prompt burden**: extra prompts user must issue to recover usable tracking
- **Human stealth pass-rate**: fraction of inserted frames not flagged by raters in flipbook test

---

## Defensible AAAI Claim (narrowed)

> *"ChronoCloak — first publisher-side temporal-state-injection cloak for prompt-driven memory-bank video object segmentation (SAM2-class)."*

What's NOT defensible:
- ❌ "First publisher-side video cloak" — broken by Liu CVPR 2025
- ❌ "First memory-based SAM2 attack" — broken by UAP-SAM2 (memory misalignment is one of their components)
- ❌ "Joint insert + δ outperforms insert-only" — current evidence does NOT support this; bridge δ must be repositioned as a *stealth-preserving stabilizer*, not an efficacy booster.

What IS defensible (under tight wording):
- ✓ "Frame insertion (vs additive perturbation) as a distinct, complementary perturbation surface for memory-bank VOS"
- ✓ "Causal evidence (memory-write blocking ablation) that the inserted-frame insertion point is the mechanism"
- ✓ "Tight fidelity budget (LPIPS ≤ 0.03) compatible with publisher use"

---

## Minimum Experiment Package (codex round 2, prioritized for V100-only 3 GPU-week budget)

### MUST-RUN (in order)

**M1. Go/No-Go pilot on 6 clips** (~3 GPU-days)
- Methods: clean / additive-only native-sequence baseline / old oracle composite insert / interpolation insert-only / ChronoCloak (interpolation + bridge δ)
- Fixed simple placements (use existing W from v5_paper_all_merged); skip placement search
- Metrics: original-frames-only J, internal stealth screen
- **Decision gate**: if interpolation does NOT remove ghosting OR kills the attack, the pivot is dead.

**M2. Core causal ablation on the same 6 clips** (~3 GPU-days)
- Methods: additive-only / insert-only / insert+bridge — **matched LPIPS and matched modified-frame count**
- Plus memory-write blocking at insert positions vs matched non-insert positions (extends current A3)
- This is **the central scientific experiment** of the paper.

**M3. Main 13-clip benchmark** (~5 GPU-days)
- Methods: clean / additive-only / insert-only / ChronoCloak
- Primary: original-frames-only J, UTR, SFR
- Secondary: whole-processed-video J-drop

**M4. Limited transfer / robustness on 6 clips** (~2 GPU-days)
- Point-prompt optimize, box-prompt eval
- SAM2-Tiny → SAM2.1-Tiny transfer
- Wrapper ablation: with vs without accept/revert

**M5. Human stealth study** (~1 hour, no GPU)
- 39 inserted events (3 inserts × 13 clips) flipbook + 39 clean controls
- 3 raters each
- Report pass-rate + obvious-fake rate

### NICE-TO-HAVE (drop in this order if compute slips)
1. Full placement-search ablations
2. Cross-dataset transfer
3. Large baseline reproduction of UAP-SAM2 / HardRegion
4. Extra wrapper variants

### DO NOT DROP
- Matched-budget additive vs insert causal ablation (M2)
- 13-clip original-frame GT table (M3)
- Human stealth test (M5)

---

## What survives the pivot from existing assets

### Reuse as-is
- 13 polished DAVIS clip raw frames (`vadi_runs/v5_paper_all_merged/`)
- Clean-SAM2 pseudo-labels for optimization
- A3 causal mechanism code (`memshield/causal_diagnostics.py`)
- 4-clip A3 STRONG results — moves from "main result" to "preliminary mechanism evidence" in appendix
- viz pipeline (5 side-by-side MP4) — kept to demonstrate the *failure mode* that motivates the pivot
- joint_placement_search code — runs as preprocessing, demoted from flagship

### Must rerun
- All main attack numbers (under new tight budget)
- All visuals (with interpolation-based decoys)
- All fidelity metrics (against the new LPIPS≤0.03 target)
- Final results table

### Build new (low compute)
- Interpolator integration (RIFE pretrained, no training)
- LPIPS / SSIM evaluation hooks
- UTR / SFR / re-prompt-burden metric implementations
- Human stealth study harness (web flipbook)

---

## AAAI risk analysis (codex round 2)

| Risk axis                 | Where ChronoCloak is weakest                        | Mitigation                                                       |
|---------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| Over-claimed firstness    | "First publisher-side video cloak" / "first memory-based SAM2 attack" | Use narrowed claim above; explicitly cite Liu CVPR25 and UAP-SAM2 in intro |
| Engineering combination   | interpolator + PGD + placement + pseudo-labels + wrapper looks like assembly | Elevate "temporal state injection" as the science; demote others |
| Weak ablations            | prior evidence says insert-only > joint            | Reposition bridge δ as *stealth-preserving stabilizer*; M2 must show δ doesn't HURT |

---

## Mock AAAI Review (codex round 2)

- **Score**: 5/10 as drafted; 6.5/10 with the 7 priority changes applied.
- **Confidence**: 4/5
- **What moves to accept**:
  1. Matched-budget proof that temporal state injection is the true novelty
  2. Full 13-clip original-frames-only GT benchmark
  3. Small but credible human stealth study
  4. Cleaner narrative that demotes placement/wrapper engineering and centers the scientific claim on inserted-frame temporal state corruption

---

## Next steps

### Phase A (no GPU needed, do now)
1. Implement RIFE / FILM / IFRNet interpolation hook (`memshield/interp_decoy.py`)
2. Tighten fidelity budgets in `VADIv5Config` (task #42 — already pending)
3. Implement UTR / SFR / re-prompt-burden metrics
4. Build human stealth study harness (web flipbook)
5. Position vs CVPR 2025 paper (task #43 — already pending)

### Phase B (V100, when available)
6. Run M1 (6-clip Go/No-Go pilot) — task #41 already pending
7. Run M2 (matched-budget causal ablation) — codex says THIS IS THE CRITICAL EXPERIMENT
8. Extend A3 from 4 clips to 13 clips (when Pro 6000 swap clears)

### Phase C (writing)
9. Draft paper around the temporal-state-injection axis
10. /paper-writing pipeline once M2 + M3 + M5 complete

---

## Companion files

- `PIVOT_REVIEW_2026-04-28.md` — codex's publisher-side pivot direction approval
- `CHRONOCLOAK_REVIEW_2026-04-28.md` — codex's compact review of this proposal (round 2)
- `paper/method_explainer_zh.tex` / `.pdf` — Chinese explainer with current experimental data (旧 framing, kept for reference)
- `viz/{camel,bear,breakdance,dance-twirl,dog}_clean_vs_attacked.mp4` — current side-by-side visualizations
- Memory: `feedback_publisher_side_pivot.md` (HARD directive 2026-04-28)
- Prior idea report: `IDEA_REPORT_2026-04-15.md` (archived)
