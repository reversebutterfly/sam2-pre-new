# Round 2 Refinement (VADI)

## Problem Anchor (verbatim, unchanged)

(see `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

## Anchor Check

- Preserved. Codex R2 confirmed no drift. "Hidden drift risk" (placement vs insert attribution) is a causal-ID issue addressed via F8 controls.

## Simplicity Check

- Dominant contribution tighter: **vulnerability-aware insert placement + LPIPS-bound insert optimization** where causality is **isolated via matched δ-only / base-insert controls**, not just top/random/bottom.
- Simplifications applied: scorer math (rank-sum, not robust-z); explicit mu_true / mu_decoy logging.

## Changes Made

### F8 — Isolation controls (CRITICAL venue-readiness fix)

Added 3 controls to main table to prove **inserts (optimized) matter, not just "local δ at vulnerable positions"**:

| # | Config | Purpose | What it isolates |
|---|---|---|---|
| 1 | Clean | baseline | — |
| 2 | Ours K=1 top | mechanistic centerpiece | full method |
| 3 | Ours K=3 top | stronger variant | scale |
| 4 | K=1 random (5 draws) | placement causality 1 | insert+δ at random vulnerability-irrelevant positions |
| 5 | K=3 random (5 draws) | placement causality 1 | same, K=3 |
| 6 | K=3 bottom | placement causality 2 | inserts at LOWEST vulnerability → should be weakest |
| 7 | **top-δ-only (K_ins=0)** | **insert causality 1** | δ at top vulnerability neighborhoods, NO inserts |
| 8 | **random-δ-only (K_ins=0)** | **insert causality 2 (placement control)** | δ at random neighborhoods, NO inserts |
| 9 | **top-base-insert+δ (unoptimized ν=0)** | **insert-optimization causality** | top-K midframe insert with no ν, δ optimized around |
| 10 | Canonical {6,12,14} | legacy comparison | — |

Ours (row 2/3) is strongest if ALL the following hold:
- **beat 4/5** (placement matters): top ≥ 2× random J-drop.
- **beat 6** (bottom is weakest): top ≥ 3× bottom J-drop.
- **beat 7** (inserts matter): ours J-drop ≥ top-δ-only J-drop + 0.10.
- **beat 9** (ν optimization matters): ours J-drop ≥ top-base-insert+δ J-drop + 0.05.

If ours does NOT beat 7 → insert's causal role is NOT established (δ at vulnerable positions is sufficient). That would move the paper's framing from "vulnerability-aware insertion" to "vulnerability-aware local perturbation" (not the user's preferred direction; honest finding regardless).

### F9 — Hard feasibility acceptance

**Before**: L_fid hinge penalties. Final selection = argmax surrogate_J_drop over "fidelity-feasible steps".

**After**: feasibility is HARD:
```
After PGD loop, filter step_history to only steps where ALL hinges = 0:
  S_feas = { step : L_fid_orig[step] = L_fid_ins[step] = L_fid_TV[step] = L_fid_f0[step] = 0 }

If S_feas is empty:
  Flag clip as "fidelity-infeasible at (ε, F_orig, F_ins)". Report its attainable LPIPS/SSIM pair.
  Do NOT include this clip in the main DAVIS-10 success count. Report as "n infeasible / 10".

Else:
  (δ*, ν*) = step with argmax surrogate_J_drop over S_feas.
```

This prevents a "best infeasible δ" from silently being reported as a success.

### F10 — Log mu_true, mu_decoy separately

Per-step diagnostic log includes:
```
mu_true_trace_t  = mu_true_t(step) for t ∈ NbrSet∪inserts
mu_decoy_trace_t = mu_decoy_t(step) for same
```

Final paper reports:
```
Δmu_true  = mean_t [ mu_true_t(attacked) − mu_true_t(clean) ]
Δmu_decoy = mean_t [ mu_decoy_t(attacked) − mu_decoy_t(clean) ]
Ratio = |Δmu_decoy| / |Δmu_true|
```

Expected: `Δmu_decoy >> 0, Δmu_true ≤ 0 but |Δmu_true| small`. Ratio ≥ 2 confirms true decoy behavior, not implicit suppression.

### Scorer math simplification (Codex simplification)

**Before**: rank-based robust-z via IQR.

**After** (rank-sum):
```
For m ∈ {1, ..., T-1}:
  rank_conf_m = rank(r_conf_m among all m)    # 1 = smallest, T-1 = largest
  rank_mask_m = rank(r_mask_m)
  rank_feat_m = rank(r_feat_m)

v_m = rank_conf_m + rank_mask_m + rank_feat_m

W = argtop-K of v_m with |m_i − m_j| ≥ 2 (non-adjacency).
```

Equivalent robustness (rank-based) with simpler math.

### F7 pilot thresholds tightened per Codex feedback

Codex said "pilot thresholds reasonable for pilot, not for final claims". Minor adjustment to pilot GO condition:

**Before**: GO if `Δ_top ≥ 0.05 OR Δ_insert ≥ 0.05 on ≥ 2/3` AND `J-drop(C) ≥ 0.20 on ≥ 2/3`.

**After**: GO if **BOTH**: (a) `Δ_top := J-drop(A) − J-drop(B) ≥ 0.05` on ≥ 2/3 clips; (b) `J-drop(C, K=3 top) ≥ 0.20` on ≥ 2/3 clips. (AND instead of OR — more conservative.)

If only one condition holds, proceed cautiously with scope-narrowed claims.

## Revised Proposal (round-2, full)

### Problem Anchor
(see `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

### Method Thesis

**Vulnerability-aware insertion** for SAM2 dataset protection: rank-sum 3-signal scorer → top-K non-adjacent placement → LPIPS-TV-bound insert content optimization (ν) + local δ on insert neighborhoods → contrastive decoy-margin loss (no suppression, no object_score margin). K=1 is the mechanistic centerpiece.

### Contribution Focus

- **C1 (dominant)**: vulnerability-aware insertion with **causal isolation of both placement AND insert optimization** via a 10-config matched-budget ablation study, achieving J-drop ≥ 0.35 at fidelity triad on DAVIS-10.
- **C2 (supporting)**: restoration-counterfactual attribution showing damage concentrates in current-frame Hiera pathway at insert positions.
- **Non-contributions**: no new generator, no UAP, no runtime hook, no bank poisoning, no learned net, no LLM/RL/diffusion, no suppression, no object_score penalty.

### Complexity Budget

Unchanged. 2 trainable (δ local-support, ν LPIPS-bound). Non-trainable rank-sum scorer. 3 inference-only swap hooks.

### System Overview

```
Offline (GT-free):
  Clean SAM2 → m̂_true_t, confidence_t, H_t
  Rank-sum scorer → W = top-K non-adjacent positions
  decoy_offset geometric; m̂_decoy_t = shift_mask(m̂_true_t)

Per-video PGD (100 steps, 3 stages):
  δ supported on S_δ = ∪_k NbrSet(m_k) ∪ {0} (NbrSet = m±1, m±2; T_δ ≈ 12-13 frames)
  base_insert_k = 0.5·x_{m_k-1} + 0.5·x_{m_k}
  insert_k = clamp(base_insert_k + ν_k) ∘ fake_quantize
  x'_t = clamp(x_t + δ_t) ∘ fake_quantize  (for t ∈ S_δ; else x_t)
  processed = interleave(x', insert_k at W)
  
  SAM2 forward → pred_logits_t per processed-time frame
  
  mu_true_t  = confidence-weighted mean of pred_logits_t on m̂_true_t
  mu_decoy_t = confidence-weighted mean of pred_logits_t on m̂_decoy_t
  
  L_margin_insert   = Σ_k softplus(mu_true_{m_k} − mu_decoy_{m_k} + 0.75)
  L_margin_neighbor = Σ_{t ∈ NbrSet\inserts} 0.5 · softplus(mu_true_t − mu_decoy_t + 0.75)
  L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
  L_fid_ins  = Σ_k max(0, LPIPS(insert_k, base_insert_k) − 0.35)
  L_fid_TV   = Σ_k max(0, TV(insert_k) − 1.2 · TV(base_insert_k))
  L_fid_f0   = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
  
  L = L_margin_insert + L_margin_neighbor
    + λ(step)·(L_fid_orig + L_fid_ins + L_fid_TV) + λ_0·L_fid_f0
  
  PGD step on (δ, ν); clip δ_0 to ±2/255, δ_{t≥1, t∈S_δ} to ±4/255; ν unbounded by ε (LPIPS constrains)
  Log mu_true_trace, mu_decoy_trace, surrogate_J_drop, per-frame LPIPS, f0 SSIM
  
  S_feas = {steps where ALL L_fid_* = 0}
  If empty → clip flagged infeasible
  Else → (δ*, ν*) = argmax surrogate_J_drop over S_feas
```

### Validation (F8-expanded)

**Main table** (10 rows):
| # | Config | ν-opt? | δ-opt? | Positions |
|---|---|:---:|:---:|---|
| 1 | Clean | — | — | — |
| 2 | **Ours K=1 top** | ✓ | ✓ (top nbr) | top-1 |
| 3 | Ours K=3 top | ✓ | ✓ (top nbr) | top-3 |
| 4 | K=1 random (×5 draws, paired bootstrap) | ✓ | ✓ (random nbr) | random-1 |
| 5 | K=3 random (×5 draws) | ✓ | ✓ | random-3 |
| 6 | K=3 bottom | ✓ | ✓ | bottom-3 |
| 7 | **top-δ-only K=0** | N/A | ✓ (top nbr) | no inserts |
| 8 | **random-δ-only K=0** (×5) | N/A | ✓ (random nbr) | no inserts |
| 9 | **top-base-insert+δ (ν=0)** | ✗ | ✓ (top nbr) | top-3 midframe |
| 10 | Canonical {6,12,14} | ✓ | ✓ | fixed |

Appendix: UAP-SAM2 per-clip, SAM2Long transfer, SAM2.1-Base transfer, DAVIS-30.

### Success criteria (updated per F8-F10)

Ours = row 2 or 3. Paper's headline claim requires:
- `J-drop(ours) ≥ 0.35` mean on DAVIS-10 feasibly-selected clips
- Fidelity triad: all L_fid_* = 0 per S_feas (hard acceptance)
- **Placement causality**: `J-drop(ours) ≥ 2 × J-drop(K-random)` AND `≥ 3 × J-drop(K-bottom)` on ≥ 7/10 clips
- **Insert causality**: `J-drop(ours) ≥ J-drop(top-δ-only) + 0.10` on ≥ 7/10 clips
- **Insert-optimization causality**: `J-drop(ours) ≥ J-drop(top-base-insert+δ) + 0.05` on ≥ 7/10 clips
- **Decoy vs suppression check**: `|Δmu_decoy| / |Δmu_true| ≥ 2` (decoy truly beats true, not just collapses it)
- **Mechanism attribution**: R2 (Hiera at inserts) recovers ≥ 0.20; R3 (bank) ≤ 0.02

### Pilot Gate (F7-updated)

3 clips × 4 configs (K=1 top, K=1 random, K=3 top, δ-only-local-random). ~3-5 GPU-hours.

**GO** if BOTH: (a) `J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05` on ≥ 2/3 clips; (b) `J-drop(K=3 top) ≥ 0.20` on ≥ 2/3 clips.

Also check at pilot: `|Δmu_decoy| / |Δmu_true|` ratio. If ratio < 2, the margin loss isn't achieving true decoy behavior — investigate loss weighting.

**NO-GO**: pivot to attack-surface analysis paper (restoration + vulnerability scoring as main tools; not an attack-success paper).

### Compute

Same ~20 GPU-hours total (multi-draw random baselines dominate).
