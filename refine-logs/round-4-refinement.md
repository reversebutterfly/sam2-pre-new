# Round 4 Refinement (VADI, final pre-pilot)

## Problem Anchor (verbatim)

(per `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

## Anchor Check

Preserved. Codex R4: NONE drift, pre-pilot ceiling confirmed 8.4/10.

## Simplicity Check

No new components. F16 is a measurement discipline addition only.

## Changes Made

### F16 — Exported-artifact feasibility + evaluation

```
After PGD returns (δ*, ν*):
  processed_uint8 = build_and_export_uint8_JPEGs(δ*, ν*)
  
  Re-measure all fidelity metrics on the EXPORTED uint8 artifact:
    LPIPS_orig_exported_t = LPIPS(processed_uint8[t], x_t)           for t in S_δ, t≥1
    SSIM_f0_exported      = SSIM(processed_uint8[0], x_0)
    LPIPS_ins_exported_k  = LPIPS(processed_uint8[W_k], base_insert_k)
    TV_ins_exported_k     = TV(processed_uint8[W_k])
  
  Feasibility check (HARD):
    If any exported metric violates its budget → clip := INFEASIBLE
  
  SAM2 evaluation:
    processed_for_SAM2 = processed_uint8   (the actual artifact the consumer gets)
    J_attacked = SAM2(processed_uint8, m_0) evaluated against DAVIS GT (at eval time only)
```

This closes the "feasible in optimization, infeasible in delivered artifact" loophole. All causal claims now measured on exactly what the consumer receives.

## Revised Proposal (round-4, final pre-pilot)

### Problem Anchor

(see `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

### Method Thesis

Vulnerability-aware insertion for SAM2 dataset protection. Rank-sum 3-signal scorer (clean-SAM2 confidence derivative + mask discontinuity + Hiera discontinuity) selects top-K non-adjacent positions. Insert content (ν, LPIPS-TV bounded) + local δ on insert neighborhoods are jointly PGD-optimized under a **contrastive decoy-margin loss** (no suppression, no object_score margin). Causal isolation via 10-config matched-budget study; mechanism attribution via restoration counterfactuals. All metrics measured on the exported uint8 artifact.

### Contribution Focus

- **C1 (dominant)**: vulnerability-aware insertion for per-video SAM2 dataset protection, with causal isolation of both placement (top/random/bottom) and insert value (vs δ-only, vs base-insert), plus signed decoy-vs-suppression decomposition. Target: J-drop ≥ 0.35 at fidelity triad on DAVIS-10 (primary denominator = all 10, infeasible = failure).
- **C2 (supporting)**: restoration-counterfactual attribution showing damage lives in current-frame Hiera pathway at insert positions (R2 ≥ +0.20), not in the non-cond bank (R3 ≤ +0.02), consistent with prior B2 observation.
- **Non-contributions**: no new generator, no UAP, no runtime hook, no bank poisoning, no learned net, no LLM/RL/diffusion, **no suppression, no object_score margin**.

### Complexity Budget

- Frozen: SAM2.1-Tiny, SAM2VideoAdapter, LPIPS(alex), fake_uint8_quantize STE, DropNonCondBankHook.
- New trainable (2): δ on local support S_δ (two-tier ε), ν per-insert (LPIPS-TV bound, no ε).
- New non-trainable: rank-sum 3-signal vulnerability scorer.
- New inference-only: 3 restoration swap hooks.
- Intentionally not used: ProPainter, L_stale, L_suppress, L_obj, learned scorer, LLM/diffusion/RL.

### System Overview

```
Offline (GT-free, one-time):
  clean_SAM2 on x_0..x_{T-1} with m_0 prompt →
    m̂_true_t = sigmoid(pred_logits_t) ∈ [0,1]^{HxW}
    confidence_t = sigmoid(object_score_t) · mean(m̂_true_t > 0.5)
    H_t = Hiera encoder output (cached)
  
  For m ∈ {1, ..., T-1}:
    r_conf_m = |confidence_m − confidence_{m-1}|
    r_mask_m = 1 − IoU(m̂_true_{m-1}, m̂_true_m)
    r_feat_m = ||H_{m-1} − H_m||_2 / mean(||H||_2)
  
  rank_x_m = rank(r_x_m among {1..T-1})
  v_m = rank_conf_m + rank_mask_m + rank_feat_m
  W = argtop-K non-adjacent (|m_i - m_j| ≥ 2), K ∈ {1,2,3}
  
  decoy_offset geometric; m̂_decoy_t = shift_mask(m̂_true_t, offset)
  c_t = |2·m̂_true_t - 1|                             # pseudo-mask confidence weight

Per-video PGD (100 steps, 3 stages):
  Initialize: δ = 0 (supported on S_δ = ∪_k NbrSet(W_k) ∪ {0})
              ν_k = small Gaussian noise (std 0.02/255)
  base_insert_k = 0.5·x_{W_k-1} + 0.5·x_{W_k}        # temporal midframe
  
  For step = 1..100:
    x'_t = clamp(x_t + δ_t, 0, 1) ∘ fake_quantize_STE           for t ∈ S_δ
    insert_k = clamp(base_insert_k + ν_k, 0, 1) ∘ fake_quantize_STE
    processed_t' = interleave(x', insert_k at W)
    
    SAM2VideoAdapter(processed, m_0) with bf16 autocast, no inference_mode
    → pred_logits per processed-time frame
    
    mu_true_t  = Σ_pixels pred_logits_t · m̂_true_t  · c_t / normalization
    mu_decoy_t = Σ_pixels pred_logits_t · m̂_decoy_t · c_t / normalization
    
    L_margin_insert   = Σ_k softplus(mu_true_{W_k} − mu_decoy_{W_k} + 0.75)
    L_margin_neighbor = Σ_{t ∈ NbrSet\inserts} 0.5 · softplus(mu_true_t − mu_decoy_t + 0.75)
    L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
    L_fid_ins  = Σ_k max(0, LPIPS(insert_k, base_insert_k) − 0.35)
    L_fid_TV   = Σ_k max(0, TV(insert_k) − 1.2 · TV(base_insert_k))
    L_fid_f0   = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
    
    L = L_margin_insert + L_margin_neighbor
      + λ(step) · (L_fid_orig + L_fid_ins + L_fid_TV) + λ_0 · L_fid_f0
    
    (δ, ν) ← (δ, ν) − η · sign(∇_{δ,ν} L)
    clip δ_0 to ±2/255, δ_{t≥1, t∈S_δ} to ±4/255 (ν unbounded by ε)
    
    log: mu_true_trace_t, mu_decoy_trace_t, surrogate_J_drop_step,
         per-frame internal LPIPS, f0 SSIM
  
  # (F16) Export + re-measure on artifact
  processed_uint8 = build_and_export_uint8_JPEGs(δ, ν)  for each step
  Re-measure fidelity on EXPORTED artifact
  S_feas = { step : ALL exported metrics meet budget }   # HARD acceptance on EXPORT
  
  If S_feas empty → clip = INFEASIBLE (counts as failure in primary denominator)
  Else → (δ*, ν*) = argmax surrogate_J_drop_exported over S_feas

Stages:
  N_1 = 30 (attack only, λ=0)
  N_2 = 40 (fidelity regularization: λ init 10, grow 2x per 10 steps when hinge violated)
  N_3 = 30 (Pareto-best: η halved; tracking)
```

### Pilot Gate (unchanged)

3 clips × 4 configs, ~3-5 GPU-hours. GO if BOTH:
- `J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05` on ≥ 2/3 clips
- `J-drop(K=3 top) ≥ 0.20` on ≥ 2/3 clips

Plus anti-suppression check: `Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true)` on ≥ 2/3.

NO-GO → pivot to attack-surface analysis paper.

### Validation Suite (F15 consolidated + F16 measurement discipline)

**Main table (10 rows, all eval on EXPORTED uint8 artifact)**:
| # | Config | ν-opt | δ-opt | Positions |
|---|---|:---:|:---:|---|
| 1 | Clean | — | — | — |
| 2 | **Ours K=1 top** | ✓ | ✓ (top nbr) | top-1 |
| 3 | Ours K=3 top | ✓ | ✓ | top-3 |
| 4 | K=1 random (×5 paired bootstrap) | ✓ | ✓ | random-1 |
| 5 | K=3 random (×5) | ✓ | ✓ | random-3 |
| 6 | K=3 bottom | ✓ | ✓ | bottom-3 |
| 7 | top-δ-only K=0 (phantom top positions) | N/A | ✓ (top nbr) | W_phantom = top |
| 8 | random-δ-only K=0 (phantom random) | N/A | ✓ | W_phantom = random |
| 9 | top-base-insert+δ (ν=0 midframe) | ✗ | ✓ | top-3 |
| 10 | Canonical {6,12,14} | ✓ | ✓ | fixed |

**Restoration** (on processed_uint8 from row 2/3):
- R2: attacked + clean Hiera at insert positions (W) only
- R2b: attacked + clean Hiera at all frames
- R3: attacked + clean non-cond bank
- B-control: attacked + drop non-cond bank

**Appendix**: UAP-SAM2 per-clip, SAM2Long transfer, SAM2.1-Base transfer, DAVIS-30.

### Success Criteria (pre-committed, primary denominator = 10)

Paper's headline requires (≥ 7/10 feasible clips AND infeasible = failure):

| Claim | Condition |
|---|---|
| J-drop | mean(J-drop(ours)) ≥ 0.35 on EXPORTED artifact |
| Placement vs random | ours ≥ max(2·random, random + 0.05) |
| Placement vs bottom | ours ≥ max(3·bottom, bottom + 0.05) |
| Insert presence | ours ≥ top-δ-only + 0.10 |
| ν optimization | ours ≥ top-base-insert+δ + 0.05 |
| Decoy vs suppression | Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true) |
| Hiera attribution | R2 ≥ +0.20 restoration |
| Bank non-attribution | R3 ≤ +0.02 |

All conditions measured on the **exported uint8 artifact**, not internal float tensors.

### Failure Mode Handling

| Failure | Detection | Handling |
|---|---|---|
| Pilot Δ_top < 0.05 or K=3 top < 0.20 | 3-clip pilot | NO-GO → pivot paper |
| Clip infeasible (S_feas empty) | After PGD | Counted as failure in primary denominator |
| Only ν drives success (top-δ-only ≈ ours) | DAVIS-10 result | Paper becomes "vulnerability-aware local perturbation", not "insertion"; user-constraint check |
| Δmu_decoy ≤ 0 or ratio < 2 | mu trace | Investigate margin weighting; may retrain with larger margin |
| R2 recovers < 0.20 (Hiera not where damage lives) | Restoration | Report honestly; mechanism is distributed; weaker attribution claim |

### Compute & Timeline

- Pilot: 3-5 GPU-hours on Pro 6000.
- DAVIS-10 main (10 configs with multi-draw random): 5-8 GPU-hours.
- Restoration: 0.5 GPU-hour.
- Appendix: 8-12 GPU-hours.
- Total: ~20 GPU-hours. Timeline: 4-5 focused days from PILOT-PASS.

### Status

**Pre-pilot ceiling 8.4/10 confirmed by reviewer.** No further non-empirical improvements available. Next step: run the gated pilot.
