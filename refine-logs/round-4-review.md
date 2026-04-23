# Round 4 Review (VADI)

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db8a1-7059-76b1-9958-ba5edc222de5`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.4 |
| Method Specificity | 8.7 |
| Contribution Quality | 8.4 |
| Frontier Leverage | 7.7 |
| Feasibility | 7.2 |
| Validation Focus | 9.0 |
| Venue Readiness | 7.7 |
| **Weighted Overall** | **8.4 / 10** |

## Verdict

**Pre-pilot ceiling confirmed: 8.4 / 10.** All R3 sub-7s now above threshold. "Not READY ≥ 9 because core causal claims remain empirical. Further proposal polishing has diminishing returns; run the gated pilot."

## Drift Warning

**NONE.**

## One Final Non-Empirical Tightening

### F16 — Lock metrics to exported uint8 artifact, not internal float tensors

**Weakness**: LPIPS/SSIM/TV feasibility measured on internal float tensors during PGD may succeed while the FINAL EXPORTED uint8 JPEG video (after quantization + JPEG encoding) may violate the constraints due to quantization / compression losses.

**Fix**:
```
After PGD returns (δ*, ν*):
  processed_video_uint8 = build_and_export_uint8_JPEGs(δ*, ν*)
  Re-measure ALL fidelity on the EXPORTED artifact:
    LPIPS_orig_t_exported = LPIPS(processed_video_uint8[t], x_t)
    SSIM_f0_exported = SSIM(processed_video_uint8[0], x_0)
    LPIPS_ins_k_exported = LPIPS(processed_video_uint8[W_k], base_insert_k)
    TV_ins_k_exported    = TV(processed_video_uint8[W_k])
  
  If ANY exported metric violates its budget → clip = INFEASIBLE (even if internal float was feasible)
  Then evaluate SAM2 and all causal claims on the EXPORTED video.
```

This prevents a "feasible in optimization, infeasible in delivered artifact" loophole. Causal claims are measured on exactly what the consumer would receive.

## Raw

<details>
<summary>Click</summary>

R4 score: **8.4 / 10, pre-pilot ceiling confirmed**.

All R3 sub-7s now above threshold. Counting infeasible as failures fixes honesty. Phantom δ positions fix δ-only ambiguity. Signed decoy metrics fix anti-suppression diagnostic.

Pre-pilot ceiling ~8.3-8.4; score 8.4. Not higher without empirical evidence.

One final non-empirical tightening: lock all success metrics to the final exported processed video, not internal float tensors. Predefine LPIPS/SSIM/TV tolerances after quantization/export. Prevents "feasible in optimization, infeasible in delivered artifact" loophole.

No blocking conceptual issue. Proposal is clean, focused, pilot-ready.

Verdict: pre-pilot ceiling confirmed: 8.4. Further proposal polishing has diminishing returns; run the gated pilot.

</details>
