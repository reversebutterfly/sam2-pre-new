# Round 4 Refinement (final pre-pilot)

## Problem Anchor (verbatim)

(unchanged, per `PROBLEM_ANCHOR_2026-04-23.md`)

## Anchor Check

- Preserved. NO drift.
- B4 pivot branch IS a different paper; flagged in Round 4 review. We retain it as an honest pilot-failure path but the primary proposal is the B1/B2/B3 narrative.

## Simplicity Check

- Unchanged: 1 trainable component (δ), 1 loss pipeline (with B4 fallback as SWAP not add).
- Reviewer said "NONE" for simplification at Round 3; no further trimming.

## Changes Made

### 1. Fallback loss sign fix (CRITICAL, Round 4 bug)

**Reviewer said**: `L_suppress_pathway` minimizes distance, which is the OPPOSITE of attack.

**Action**: rewrite fallback loss with correct signs + detached clean features.

**Before (WRONG)**:
```
L_suppress_pathway = α · ⟨∥h_attack_t − h_clean_t∥_2⟩_support
                   + β · ∥m_f0_attack − m_f0_clean∥_2^2
```
Minimizing this preserves clean features. Backwards for attack.

**After (correct)**:
```
L_pathway_attack = α · ⟨ cosine(h_attack_t, h_clean_t.detach()) ⟩_{t ∈ support(m̂_true)}
                 + β · cosine( m_f0_attack, m_f0_clean.detach() )
```

Minimizing cosine pushes it toward −1 (anti-aligned with clean features). Clean tensors `.detach()`-ed so gradients do not flow back through the clean forward.

Equivalently (sign-canonical):
```
L_pathway_attack = - α · ⟨ 1 − cosine(h_attack_t, h_clean_t.detach()) ⟩_support
                  - β · ( 1 − cosine(m_f0_attack, m_f0_clean.detach()) )
```

Both forms are minimized by making attack features MAXIMALLY DISSIMILAR from clean features on the foreground support.

**Impact**: fallback path is now correctly-oriented for PGD descent.

## Revised Proposal (final pre-pilot)

### Problem Anchor

(see `PROBLEM_ANCHOR_2026-04-23.md`)

### Method Thesis

Architecture-aware dataset protection for SAM2-family VOS via a **diagnose → attack → verify** causal loop, with pilot-gate-committed narrative.

- **Diagnose**: B2 causal ablation shows SAM2.1-Tiny's non-cond FIFO bank is architecturally marginal (`|ΔJ_bank-drop| < 0.01` on 5 clips).
- **Attack**: per-video PGD on `δ` with two-tier ε budget (f0: 2/255 + SSIM≥0.98; t≥1: 4/255 + LPIPS≤0.20), GT-free soft-logit confidence-weighted supervision via clean-SAM2 pseudo-labels, 3-stage training, best-feasible checkpoint.
- **Verify**: five-config restoration study (R1 f0 swap, R2 Hiera swap, R12 joint, R3 bank swap, B-control bank drop) on attacked videos with `ΔJ_restore(c) = J(attacked + swap) − J(attacked)`.

### Contribution Focus

- **C1 (single, four-part)**: (i) causal diagnosis of SAM2; (ii) pathway-targeted per-video PGD; (iii) restoration-counterfactual attribution; (iv) pre-committed pilot-gate decision rule with 4 branches (B1-B4).
- **Non-contributions**: no new generator, UAP, runtime hook, bank poisoning, FIFO-self-healing, LLM/diffusion/RL.

### Complexity Budget

- **Frozen/reused**: SAM2.1-Tiny, `SAM2VideoAdapter`, LPIPS(alex), fake uint8 quantize (STE), `DropNonCondBankHook`.
- **New trainable (1)**: `δ ∈ R^{T × H × W × 3}` with two-tier ε.
- **New inference-time infrastructure (3 hooks, NO trainable)**: `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook`.

### Supervision (GT-free, confidence-weighted soft logits)

```
m̂_true_t = sigmoid(clean_SAM2(x, m_0).pred_masks_high_res_t)   ∈ [0,1]
c_t      = | 2 · m̂_true_t − 1 |                                 # confidence weight, downweights boundary
L_suppress = Σ_{t≥1} softplus( (1 / Σ_pixels c_t) · Σ_pixels c_t · pred_logits_t · m̂_true_t )
L_obj      = Σ_{t≥1} softplus( object_score_logits[t] + 0.5 )
```

### Primary Loss (B1/B2/B3 branches)

```
L_fid_frame[t] = max(0, LPIPS(x'_t, x_t) − 0.20)       for t ≥ 1
L_fid_f0       = max(0, 1 − SSIM(x'_0, x_0) − 0.02)

L = L_suppress + 0.3 · L_obj + λ(step) · Σ_{t≥1} L_fid_frame[t] + λ_0 · L_fid_f0
```

### Fallback Loss (B4 branch only; REPLACES L_suppress, not adds)

```
# Clean features precomputed once and detached
h_clean_t = image_encoder.forward_image(x_t).detach()
m_f0_clean = SAM2_memory_encode_from_f0(x_0, m_0).detach()

# Attack features updated per PGD step
h_attack_t = image_encoder.forward_image(x'_t)
m_f0_attack = SAM2_memory_encode_from_f0(x'_0, m_0)

L_pathway_attack =
    α · ⟨ cosine( h_attack_t, h_clean_t ) ⟩_{t ≥ 1, pixels ∈ support(m̂_true_t)}
  + β · cosine( m_f0_attack, m_f0_clean )

L_B4 = L_pathway_attack + 0.3 · L_obj + λ(step) · Σ L_fid_frame + λ_0 · L_fid_f0
```

Minimizing cosine pushes attack features toward anti-alignment with clean features on the foreground support. α, β chosen so the magnitude at step 0 matches L_suppress on the pilot clip.

### Two-tier Fidelity Budget

| Frame | ε_∞ | Fidelity |
|---|---|---|
| f0 (prompt) | 2/255 | SSIM ≥ 0.98 (hinge margin 0.02) |
| t ≥ 1 | 4/255 | LPIPS ≤ 0.20 (hinge) |

### Pilot-Gate Decision Rule (first-class, pre-committed)

1 clip, ≤ 1 GPU-hour. Measurements: surrogate J-drop at δ*, per-frame LPIPS, f0 SSIM, wall-clock, memory, R1/R2/R12 on pilot.

| Branch | Trigger | Action | Paper narrative |
|---|---|---|---|
| **B1** | J-drop ≥ 0.40 + fidelity met + R12 ≥ 0.40 + R1,R2 ≥ 0.15 each | Proceed to full DAVIS-10 main | Causal-loop dataset protection works. Full narrative. |
| **B2** | J-drop ≥ 0.40 + fidelity met + R12 ≥ 0.40 but R1 < 0.10 and R2 < 0.10 | Proceed to DAVIS-10 | Causal-loop attack works; attribution is joint-only. |
| **B3** | J-drop ≥ 0.40 + fidelity met but ALL R1/R2/R12 < 0.05 | Proceed to DAVIS-10 | Attack works; attribution open question in discussion. |
| **B4** | J-drop < 0.30 at feasibility OR fidelity infeasible at F_lpips=0.20 | Swap to `L_B4` (fallback). Rerun pilot. If still < 0.30, pivot paper. | Paper pivots to "architecture-aware attack-surface analysis on SAM2-family", honestly reporting the ceiling. |

No post-hoc branch reshuffle.

### Restoration Attribution (signed)

```
ΔJ_restore(config) = J(attacked + swap) − J(attacked)   # positive = restoration
```

| Config | Swap | Expected | Interpretation |
|---|---|---|---|
| R1 | clean f0 maskmem/obj_ptr | ≥ +0.25 | damage in f0 pathway |
| R2 | clean Hiera per frame | ≥ +0.30 | damage in current-frame pathway |
| R12 | R1 + R2 joint | ≥ max(R1,R2); target ≥ +0.40 | joint upper bound |
| R3 | clean non-cond bank | ≤ +0.02 | bank not damage location |
| B-ctrl | drop non-cond bank | ≤ +0.02 | bank marginal on attacked |

Protocol: dominated / additive / jointly-non-identifiable.

Swap boundaries:
- SwapF0MemoryHook: intercepts `_prepare_memory_conditioned_features`, replaces the cond-slot maskmem/obj_ptr with clean-cached f0 values.
- SwapHieraFeaturesHook: intercepts `image_encoder.forward_image` output.
- SwapBankHook: intercepts `non_cond_frame_outputs` dict, replaces entries with clean cache.

### Training Plan

100 PGD steps, stages 30/40/30. η=1/255 (stages 1-2), 0.5/255 (stage 3). Sliding 30-frame backward window if T>40. bf16 autocast. STE uint8 quantize. `δ*` = argmax surrogate_J_drop over fidelity-feasible steps. No DAVIS GT anywhere.

### Validation

Main table (5 rows): Clean / Ours / Uniform-δ / UAP-SAM2 per-clip / Ours-on-SAM2Long.
Restoration: R1 / R2 / R12 / R3 / B-control.
Mechanism evidence (embedded in C1): bank-drop on attacked DAVIS-10 confirms `|ΔJ| < 0.02`.
Appendix: prompt-robustness, SAM2.1-Base transfer, DAVIS-30 extended.

### Compute & Timeline

- Pilot: 1 GPU-hour (decision gate, branch commit happens here).
- DAVIS-10 C1 (on B1/B2/B3): ~2.5 GPU-hours.
- Restoration (R1/R2/R12/R3 + B-control on 10 clips): ~0.5 GPU-hour.
- SAM2Long install + transfer: ~2-3 GPU-hours.
- Appendix: ~5-8 GPU-hours.
- **Total budget: ~12-15 GPU-hours from PILOT-PASS to full results** (3-4 focused days).

### Status

Pre-pilot formulation ceiling achieved. Structural proposal stable across R3 and R4 (8.4/10). Remaining gap to READY is empirical (pilot must show J-drop ≥ 0.40 at feasibility). Proposal is pre-committed to all four pilot outcomes.
