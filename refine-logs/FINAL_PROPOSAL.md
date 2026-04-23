# Research Proposal: Vulnerability-Aware Decoy Insertion (VADI) for SAM2 Dataset Protection

**Status**: Pre-pilot ceiling confirmed (8.4/10 at R4, Codex explicit "further proposal polishing has diminishing returns").
**Verdict**: REVISE (formal) / PRE-PILOT CEILING (internal — pending pilot).
**Thread**: `019db8a1-7059-76b1-9958-ba5edc222de5`
**Frozen anchor**: `refine-logs/PROBLEM_ANCHOR_2026-04-23_v4-insert.md`

## Problem Anchor

- **Bottom-line**: clean video + first-frame mask → processed video (K_ins inserts + δ-perturbed originals) that (a) degrades SAM2 target segmentation across the whole processed video, (b) remains visually faithful under a floor-grounded fidelity budget.
- **Must-solve bottleneck**: prior insertion attacks placed inserts at FIFO-canonical positions (motivation B2-falsified); ProPainter-decoy content had LPIPS floor 0.67-0.89 regardless of decoy offset; bank poisoning is non-causal on SAM2.1-Tiny (|ΔJ_bank-drop| < 0.01). Missing: **principled placement via vulnerability analysis of the clean video**, insert content optimized for current-frame Hiera pathway, amplified by **local** δ on insert neighborhoods, under a **contrastive decoy-margin loss** — no suppression, no object_score margin.
- **Non-goals**: clean-suffix eval; defeat FIFO self-healing (falsified); universal perturbation; runtime hook; pure-δ method; pure-suppression method.
- **Constraints**: white-box SAM2.1-Tiny; per-video PGD; K_ins ∈ {1,2,3} at rank-sum-scored positions; two-tier fidelity (f0: ε=2/255 + SSIM≥0.98; originals t≥1 in S_δ: ε=4/255 + LPIPS≤0.20; inserts: LPIPS≤0.35 vs temporal midframe + TV ≤ 1.2× base; no ε bound on ν); GT-free supervision; DAVIS-10.
- **Success condition (pre-committed)**: on DAVIS-10 (primary denominator = all 10, infeasible = failure), ≥ 7/10 clips satisfy the 8-claim success bar below.

## Method Thesis

Vulnerability-aware insertion for SAM2 dataset protection: rank-sum 3-signal scorer (confidence drop, mask discontinuity, Hiera discontinuity) on the clean-video SAM2 trace picks top-K non-adjacent placements; insert content (ν, LPIPS-TV bounded, no ε) + local δ on insert neighborhoods are jointly PGD-optimized under a **contrastive decoy-margin loss** on confidence-weighted masked means; all metrics measured on the **exported uint8 artifact**, with infeasible clips counted as failures.

## Contribution Focus

- **C1 (dominant)**: vulnerability-aware insertion with **causal isolation of both placement AND insert optimization** — top/random/bottom placement controls + top-δ-only / random-δ-only / top-base-insert+δ insertion-value controls — achieving mean J-drop ≥ 0.35 at fidelity triad on DAVIS-10, with signed decoy-vs-suppression decomposition.
- **C2 (supporting)**: restoration-counterfactual attribution showing damage concentrates in SAM2's current-frame Hiera pathway at insert positions (R2 ≥ +0.20), not in the non-cond bank (R3 ≤ +0.02).
- **Non-contributions**: no new generator, no UAP, no runtime hook, no bank poisoning, no learned net, no LLM/RL/diffusion, no suppression, no object_score margin.

## Proposed Method

### Complexity Budget

- **Frozen/reused**: SAM2.1-Tiny, `SAM2VideoAdapter` (Chunk 5b-ii), LPIPS(alex), `fake_uint8_quantize` (STE), `DropNonCondBankHook`.
- **New trainable tensors (2)**: δ on local support S_δ (two-tier ε); ν per-insert (LPIPS-TV bound, no ε).
- **New non-trainable**: rank-sum 3-signal vulnerability scorer.
- **New inference-only hooks**: `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook` for restoration.
- **Intentionally NOT used**: ProPainter, `L_stale`, `L_suppress`, `L_obj`, learned scorer, LLM/diffusion/RL, insertions at canonical FIFO positions (falsified).

### System Overview

```
Offline publisher analysis (GT-free, one-time):
  clean_SAM2 on x_0..x_{T-1} with m_0 prompt →
    m̂_true_t    = sigmoid(pred_logits_t) ∈ [0,1]^{H×W}
    confidence_t = sigmoid(object_score_t) · mean(m̂_true_t > 0.5)
    H_t         = Hiera encoder output (cached for restoration hooks)

Vulnerability scoring (rank-sum 3-signal, GT-free):
  For m ∈ {1, ..., T-1}:
    r_conf_m = |confidence_m − confidence_{m-1}|
    r_mask_m = 1 − IoU(m̂_true_{m-1}, m̂_true_m)
    r_feat_m = ||H_{m-1} − H_m||_2 / mean(||H||_2)

    rank_x_m = rank(r_x_m among {1..T-1})
    v_m      = rank_conf_m + rank_mask_m + rank_feat_m
  W = argtop-K non-adjacent (|m_i - m_j| ≥ 2), K ∈ {1,2,3}

  decoy_offset = argmax_{(dy,dx) feasible}  geometric-distance(shift_mask(m̂_true, (dy,dx)),  m̂_true)
  m̂_decoy_t    = shift_mask(m̂_true_t, decoy_offset)
  c_t          = | 2·m̂_true_t − 1 |                              # pseudo-mask confidence weight

Per-video PGD (100 steps, 3 stages):
  S_δ = ∪_k NbrSet(W_k) ∪ {0}      NbrSet(m) = {m-2, m-1, m+1, m+2} ∩ [0, T-1]
  Initialize: δ_t = 0 for t ∈ S_δ else fixed 0
              ν_k = small Gaussian noise (std 0.02/255)
  base_insert_k = 0.5·x_{W_k - 1} + 0.5·x_{W_k}       # temporal midframe (no ProPainter)

  For step = 1..100:
    x'_t       = clamp(x_t + δ_t, 0, 1) ∘ fake_quantize_STE    for t ∈ S_δ
    insert_k   = clamp(base_insert_k + ν_k, 0, 1) ∘ fake_quantize_STE
    processed  = interleave(x', insert_k at positions W)

    Forward SAM2VideoAdapter(processed, m_0) with bf16 autocast → pred_logits per processed-time frame

    # Confidence-weighted masked means
    mu_true_t  = Σ_pixels pred_logits_t · m̂_true_t  · c_t  /  (Σ m̂_true_t · c_t + eps)
    mu_decoy_t = Σ_pixels pred_logits_t · m̂_decoy_t · c_t  /  (Σ m̂_decoy_t · c_t + eps)

    # Contrastive decoy-margin (no suppression, no object_score)
    L_margin_insert   = Σ_k       softplus(mu_true_{W_k} − mu_decoy_{W_k} + 0.75)
    L_margin_neighbor = Σ_{t ∈ NbrSet\inserts}  0.5 · softplus(mu_true_t − mu_decoy_t + 0.75)

    # Fidelity hinges (internal, during PGD)
    L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
    L_fid_ins  = Σ_k               max(0, LPIPS(insert_k, base_insert_k) − 0.35)
    L_fid_TV   = Σ_k               max(0, TV(insert_k) − 1.2 · TV(base_insert_k))
    L_fid_f0   =                    max(0, 1 − SSIM(x'_0, x_0) − 0.02)

    L = L_margin_insert + L_margin_neighbor
      + λ(step) · (L_fid_orig + L_fid_ins + L_fid_TV) + λ_0 · L_fid_f0

    (δ, ν) ← (δ, ν) − η · sign(∇_{δ,ν} L)
    clip δ_0 to ±2/255, δ_{t≥1, t∈S_δ} to ±4/255 (ν unbounded by ε)

    log per-step:
      mu_true_trace_t, mu_decoy_trace_t
      surrogate_J_drop_internal = 1 − J(sigmoid(pred_logits) > 0.5, m̂_true_remapped)
      per-frame LPIPS, f0 SSIM

Stages:
  N_1 = 30  (attack-only, λ=0)
  N_2 = 40  (fidelity regularization: λ init=10, grow 2× per 10 steps when any hinge violated)
  N_3 = 30  (Pareto-best tracking: η halved)

# (F16) Export + re-measure on uint8 artifact (HARD acceptance)
For each step's (δ, ν):
  processed_uint8 = export_uint8_JPEG_sequence(δ, ν)
  Re-measure on EXPORTED artifact:
    LPIPS_orig_exp_t, SSIM_f0_exp, LPIPS_ins_exp_k, TV_ins_exp_k
  step_feasible = (all exported metrics meet budget)

S_feas = { step : step_feasible }
If S_feas == empty → clip = INFEASIBLE (primary-denominator failure)
Else → (δ*, ν*) at argmax_{step ∈ S_feas} surrogate_J_drop_on_EXPORTED
```

### Pilot Gate (pre-committed)

**Scope**: 3 clips (dog, cows, bmx-trees — easy/static/fast motion) × 4 configs (K=1 top, K=1 random, K=3 top, δ-only-local-random). ~3-5 GPU-hours total.

**GO condition (AND of both)**:
- `J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05` on ≥ 2/3 clips (placement-vs-random at K=1)
- `J-drop(K=3 top) ≥ 0.20` on ≥ 2/3 clips (absolute strength)

**Additional diagnostic check** (does not block GO but flagged at pilot):
- `Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true)` on ≥ 2/3 clips (true decoy, not implicit suppression)

**NO-GO** → pivot paper to "architecture-aware attack-surface analysis of SAM2" using restoration + vulnerability scoring as primary content (honest fallback).

### Main Validation (10-row table, all metrics on EXPORTED artifact)

| # | Config | ν-opt | δ-opt | Positions | Isolates |
|---|---|:---:|:---:|---|---|
| 1 | Clean | — | — | — | Baseline |
| 2 | **Ours K=1 top** | ✓ | ✓ (top nbr) | rank-sum top-1 | **Centerpiece** |
| 3 | Ours K=3 top | ✓ | ✓ (top nbr) | top-3 | Stronger variant |
| 4 | K=1 random (5 draws, paired bootstrap) | ✓ | ✓ (random nbr) | random-1 | Placement causality 1 |
| 5 | K=3 random (5 draws) | ✓ | ✓ | random-3 | Placement causality 1, K=3 |
| 6 | K=3 bottom | ✓ | ✓ | rank-sum bottom-3 | Placement causality 2 |
| 7 | top-δ-only K=0 (phantom top positions) | N/A | ✓ (top nbr) | W_phantom = top | **Insert necessity 1** |
| 8 | random-δ-only K=0 (phantom random) | N/A | ✓ | W_phantom = random | **Insert necessity 2 (placement-matched)** |
| 9 | top-base-insert+δ (ν=0 midframe) | ✗ | ✓ (top nbr) | top-3 | **ν optimization necessity** |
| 10 | Canonical {6,12,14} | ✓ | ✓ | fixed legacy | Legacy comparison |

### Restoration Attribution (on ours exported artifacts)

| Config | Swap | Expected | Interpretation |
|---|---|---|---|
| R2 | clean Hiera at insert positions (W) | ≥ +0.20 | damage lives in current-frame pathway at inserts |
| R2b | clean Hiera at ALL frames | ≥ R2 | joint upper bound |
| R3 | clean non-cond bank | ≤ +0.02 | bank non-causal on attacked too |
| B-control | drop non-cond bank | ≤ +0.02 | confirms B2 on attacked |

Metric: `ΔJ_restore = J(attacked + swap) − J(attacked)`. Positive = restoration works.

### Success Criteria (pre-committed, primary denominator = 10)

Paper's headline requires on **all 10 DAVIS clips** (infeasible = failure, ≥ 7/10 must satisfy):

| Claim | Condition |
|---|---|
| J-drop | mean(J-drop(ours)) ≥ 0.35 on exported artifact |
| Placement vs random | ours ≥ max(2·random, random + 0.05) |
| Placement vs bottom | ours ≥ max(3·bottom, bottom + 0.05) |
| Insert necessity (vs δ-only-top) | ours ≥ top-δ-only + 0.10 |
| ν optimization necessity | ours ≥ top-base-insert+δ + 0.05 |
| Decoy vs suppression | Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true) |
| Hiera attribution | R2 ≥ +0.20 |
| Bank non-attribution | R3 ≤ +0.02 |

### Novelty and Elegance Argument

Closest work:
- **UAP-SAM2** (NeurIPS 2025): universal, inference-time. Different threat model.
- **Internal v4 decoy insertion**: canonical FIFO (falsified motivation); ProPainter (LPIPS-expensive); reported 92.5% J-drop was eval-overlap artifact, not insertion causality.
- **Adversarial patches**: image domain; no temporal/Hiera structure.
- **Video adversarial attacks on trackers**: uniform-per-frame; no principled placement.

Exact novelties:
1. **Principled placement via clean-SAM2 vulnerability scoring** — foundation-model-native attack design.
2. **Insert-as-current-frame-pathway attack** (B2-informed): inserts corrupt the causal pathway directly, not the non-cond bank.
3. **Causal isolation suite**: 10 matched-budget configs + signed decoy-vs-suppression decomposition + restoration attribution. Most attacks do not separate placement-vs-content-vs-optimization.
4. **Hard exported-artifact feasibility**: metrics are measured on the delivered uint8 video, not internal floats. Closes the common "feasible in optimization, infeasible in artifact" loophole.
5. **GT-free throughout**: optimization, selection, scoring — all from clean-SAM2 pseudo-labels.

Why focused: 2 trainable tensors (δ, ν); 1 heuristic scorer; 1 paper thesis; no parallel contributions.

### Failure Modes and Diagnostics

| Failure | Detection | Handling |
|---|---|---|
| Pilot Δ_top < 0.05 or K=3 top < 0.20 | 3-clip pilot | NO-GO → pivot paper |
| Clip infeasible (S_feas empty on exported artifact) | Post-PGD export check | Counted as failure in primary denominator |
| ν optimization adds < 0.05 over base-insert+δ | Main-table row 9 | Paper honestly narrows claim to "placement + local δ matters; ν refinement is marginal" |
| Only δ drives success (top-δ-only ≈ ours) | Main-table row 7 | Conflicts with user's insert-required constraint; paper pivots honestly |
| Δmu_decoy ≤ 0 or ratio < 2 | mu trace | Investigate margin weighting; possibly retrain with larger margin |
| R2 recovers < 0.20 (Hiera not where damage lives) | Restoration | Mechanism distributed; weaker attribution claim; honestly reported |

### Compute & Timeline

- **Pilot**: 3-5 GPU-hours.
- **DAVIS-10 main** (10 configs, multi-draw random): ~5-8 GPU-hours.
- **Restoration**: 0.5 GPU-hour.
- **Appendix** (UAP-SAM2 per-clip, SAM2Long transfer, SAM2.1-Base transfer, DAVIS-30): ~8-12 GPU-hours.
- **Total**: ~20 GPU-hours on a single Pro 6000; 4-5 focused days from PILOT-PASS to full results.
