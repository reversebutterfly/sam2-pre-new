# Round 3 Refinement (VADI)

## Problem Anchor (verbatim)

(see `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

## Anchor Check

Preserved. No drift.

## Simplicity Check

No new components. R3 fixes are precision on existing elements (denominators, decomposition signs, absolute gaps, phantom positions).

## Changes Made

### F11 — Primary denominator = 10, infeasible = failure

```
n_success := count of clips meeting ALL causal criteria WITH all fidelity constraints satisfied
n_infeasible := count of clips where S_feas is empty
n_feasible_but_fail := count of clips with S_feas non-empty but causal criteria not met

Primary headline: n_success / 10 (infeasible counts as failure)
Report also: n_infeasible / 10, n_feasible_but_fail / 10, feasible-only mean J-drop (diagnostic)
```

Headline claim in paper is AGAINST THE FULL DAVIS-10 DENOMINATOR. Infeasible = failure.

### F12 — Phantom insertion positions for δ-only baselines

```
For "top-δ-only (K_ins=0)":
  W_phantom = top-K positions from rank-sum scorer (same as "Ours")
  S_δ = ∪_k NbrSet(W_phantom_k) ∪ {0}
  No inserts placed. δ optimized on S_δ.

For "random-δ-only (K_ins=0)":
  W_phantom = K random non-adjacent positions (5 draws, paired bootstrap)
  S_δ = ∪_k NbrSet(W_phantom_k) ∪ {0}
  No inserts placed.
```

This matches local-δ support between full method and ablation.

### F13 — Signed anti-suppression decomposition

**Before**: `|Δmu_decoy| / |Δmu_true| ≥ 2`

**After**:
```
Δmu_true_t  = mu_true_t(attacked)  − mu_true_t(clean)     per t ∈ (inserts ∪ NbrSet)
Δmu_decoy_t = mu_decoy_t(attacked) − mu_decoy_t(clean)    same t

Aggregated (mean over t):
  Δmu_true  = mean_t Δmu_true_t
  Δmu_decoy = mean_t Δmu_decoy_t

Report both separately.

Anti-implicit-suppression guarantee:
  Δmu_decoy > 0
  Δmu_decoy ≥ 2 · max(0, -Δmu_true)
```

First condition: decoy is actually rising. Second: if true-suppression happens (-Δmu_true > 0), decoy rise is at least 2× that amount. This ensures the contrastive margin wins via decoy elevation, not true collapse alone.

### F14 — Absolute + ratio gaps for placement

```
Placement causality (replaces ratio-only):
  ours ≥ max(2.0 · random_mean,  random_mean + 0.05)
  ours ≥ max(3.0 · bottom,       bottom      + 0.05)

Insert presence (already absolute):
  ours ≥ top-δ-only + 0.10

ν optimization (already absolute):
  ours ≥ top-base-insert+δ + 0.05
```

Ratios alone unstable when denominators are near zero; absolute guards prevent false positives.

### F15 — Summary of all causal claims (consolidated)

Paper's pre-committed success bar (≥ 7/10 feasible clips, primary denominator all 10):

| Claim | Condition |
|---|---|
| Headline J-drop | J-drop(ours) ≥ 0.35 |
| Placement vs random | ours ≥ max(2·rand, rand+0.05) |
| Placement vs bottom | ours ≥ max(3·bot, bot+0.05) |
| Insert presence | ours ≥ top-δ-only + 0.10 |
| ν optimization | ours ≥ top-base-insert+δ + 0.05 |
| Decoy-not-suppression | Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true) |
| Mechanism attribution (Hiera) | R2 restoration ≥ +0.20 |
| Mechanism attribution (bank) | R3 restoration ≤ +0.02 |

## Revised Proposal (round-3)

### Problem Anchor
(verbatim, unchanged)

### Method Thesis

Vulnerability-aware insertion for SAM2 dataset protection: rank-sum 3-signal scorer picks top-K non-adjacent positions from clean-SAM2 signals; insert content (ν, LPIPS-TV bound) + local δ on insert neighborhoods are jointly optimized under a contrastive decoy-margin loss (no suppression, no object_score). Causal isolation via 10-config matched-budget study; mechanism attribution via restoration counterfactuals.

### Contribution Focus

- **C1 (dominant)**: vulnerability-aware insertion with full causal isolation — placement (top/random/bottom), insert presence (vs δ-only), and insert optimization (vs base-insert) — achieving J-drop ≥ 0.35 at fidelity triad on DAVIS-10, with signed decoy-vs-suppression decomposition.
- **C2 (supporting)**: restoration-counterfactual attribution confirming damage lives in current-frame Hiera pathway at insert positions.
- **Non-contributions**: as before.

### Complexity Budget

Unchanged.

### System Overview

```
Offline (GT-free):
  clean_SAM2 forward → m̂_true_t, confidence_t, H_t
  Rank-sum 3-signal scorer → W = top-K non-adjacent
  decoy_offset geometric; m̂_decoy_t = shift_mask(m̂_true_t)

Per-video PGD (100 steps, 3 stages):
  δ supported on S_δ = ∪_k NbrSet(W_k) ∪ {0}
  base_insert_k = 0.5·x_{W_k-1} + 0.5·x_{W_k}
  insert_k = clamp(base_insert_k + ν_k) ∘ fake_quant
  x'_t = clamp(x_t + δ_t) ∘ fake_quant for t ∈ S_δ; else x_t
  processed = interleave(x', insert_k at W)
  
  Forward → pred_logits_t per processed-time frame
  mu_true_t  = confidence-weighted mean(pred_logits_t on m̂_true_t)
  mu_decoy_t = confidence-weighted mean(pred_logits_t on m̂_decoy_t)
  
  L_margin_insert   = Σ_k softplus(mu_true_{W_k} − mu_decoy_{W_k} + 0.75)
  L_margin_neighbor = Σ_{t ∈ NbrSet\inserts} 0.5·softplus(mu_true_t − mu_decoy_t + 0.75)
  L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
  L_fid_ins  = Σ_k max(0, LPIPS(insert_k, base_insert_k) − 0.35)
  L_fid_TV   = Σ_k max(0, TV(insert_k) − 1.2·TV(base_insert_k))
  L_fid_f0   = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
  
  L = L_margin_insert + L_margin_neighbor + λ(step)·(L_fid_orig + L_fid_ins + L_fid_TV) + λ_0·L_fid_f0
  
  PGD step; clip δ_0 ε=2/255, δ_{t≥1, t∈S_δ} ε=4/255; ν unbounded by ε
  log: mu_true_trace_t, mu_decoy_trace_t, surrogate_J_drop, per-frame LPIPS, f0 SSIM
  
  S_feas = {steps : L_fid_orig=L_fid_ins=L_fid_TV=L_fid_f0 = 0}
  If S_feas empty → clip = INFEASIBLE (failure in primary denominator)
  Else → (δ*, ν*) = argmax surrogate_J_drop over S_feas
```

### Pilot Gate (unchanged)

3 clips × 4 configs. GO if (a) Δ_top ≥ 0.05 on 2/3 AND (b) J-drop(K=3 top) ≥ 0.20 on 2/3, AND pilot anti-suppression check: `Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0,-Δmu_true)` on ≥ 2/3.

### Validation (round-2 10-row main table unchanged in content; add F11 reporting)

Main headline: J-drop (primary denominator = **ALL 10** DAVIS-10 clips, infeasible = failure). Plus side-table: n_infeasible / 10, feasible-only mean.

### Compute

Unchanged ~20 GPU-hours.
