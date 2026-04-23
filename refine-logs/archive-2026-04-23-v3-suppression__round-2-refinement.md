# Round 2 Refinement

## Problem Anchor (verbatim)

(per `PROBLEM_ANCHOR_2026-04-23.md` — unchanged)

- **Bottom-line**: clean video + first-frame mask → processed video that is visually faithful AND causes SAM2 to lose the target across the entire processed video.
- **Must-solve bottleneck**: f0 + current-frame features are causal; non-cond FIFO bank is architecturally marginal (B2). Attack targets causal pathways under realistic fidelity.
- **Non-goals**: clean-suffix eval; FIFO self-healing defeat; UAP; runtime hook.
- **Constraints**: white-box SAM2.1-Tiny, per-video PGD, pixels-only, two-tier fidelity budget, DAVIS-2017 val, ≤ 15 GPU-min / clip.
- **Success**: on DAVIS-10 — (1) mean J&F drop ≥ 0.40; (2) fidelity triad satisfied; (3) restoration attribution confirms damage lives in f0/current-frame.

## Anchor Check

- Original bottleneck preserved.
- All Round-2 reviewer asks are method/framing refinements; no drift risk.
- The reviewer's "conditional NONE" drift warning (GT-free optimization + checkpoint selection) is addressed by explicitly stating both are computed from clean-SAM2 pseudo-labels, never DAVIS annotations.

## Simplicity Check

- **Dominant contribution after revision**: the full causal loop — bank diagnosis (B2) → path-targeted PGD attack → restoration attribution. This is ONE story, not two parallel contributions.
- **Components removed or merged**:
  - `L_rank` + decoy regime now deleted from default method (reintroduced only if pilot shows suppression-only fails).
  - C2 (bank marginality) fully folded into C1 as mechanism evidence; no standalone contribution.
  - R1/R2 restoration counterfactuals gain a joint R1+R2 upper-bound swap for interpretability.
- **Reviewer suggestions rejected as unnecessary complexity**: modernization option 2 ("pathway-aligned Hiera/f0 feature-corruption losses") rejected as a DEFAULT — this would be a second trainable component essentially. Kept as a fallback only if output loss underperforms at pilot.
- **Why still the smallest adequate**: one trainable component (δ), one paper thesis (causal loop), one supervision source (clean-SAM2 soft logits), 4 ablation/restoration configs for attribution.

## Changes Made

### 1. Validation sign fix (IMPORTANT #1)

**Reviewer said**: sign is backwards.

**Action**: redefine `ΔJ_restore(config) = J(attacked + swap) − J(attacked)`. Positive = swap restores tracking. All thresholds updated accordingly.

### 2. Joint R1+R2 restoration (IMPORTANT #2)

**Reviewer said**: R1/R2 may be non-additive / redundant.

**Action**: add `R12` (swap both f0 memory AND Hiera features — joint restoration). Interpretation protocol:
- If `ΔJ(R12) ≈ max(ΔJ(R1), ΔJ(R2))` → attribution is dominated by ONE of the two pathways (report which).
- If `ΔJ(R12) > ΔJ(R1) + ΔJ(R2) − overlap` → pathways are additive; both independently contribute.
- If `ΔJ(R12) ≈ 0` despite individual swaps working → joint non-identifiability; honestly report as "damage is distributed; restoration attribution inconclusive at individual pathway level".

Four restoration configs now: R1 (f0 only), R2 (Hiera only), R12 (f0+Hiera joint), R3 (bank only) + B-control (bank dropped).

### 3. Causal-loop framing (IMPORTANT #3)

**Reviewer said**: don't let C2 compete with C1.

**Action**: restate the paper thesis explicitly as a SINGLE causal loop:

> **Thesis**: Dataset protection against SAM2-family VOS must be architecture-aware. We (1) empirically show SAM2.1-Tiny's non-cond FIFO memory bank is architecturally marginal for tracking (causal ablation); (2) design a per-video preprocessor whose perturbation targets only the causal pathways (f0 + current-frame) under a two-tier fidelity budget; (3) verify via restoration counterfactuals that the resulting attack's damage genuinely lives in the targeted pathways. The three-step diagnose → attack → verify loop is the paper's contribution.

C1 now embeds (1) as mechanism evidence. No parallel C2.

### 4. GT-free checkpoint selection (explicit)

**Reviewer specific-check**: "if checkpoint selection also uses clean-SAM2 pseudo masks/logits rather than DAVIS GT, state explicitly".

**Action**: checkpoint selection now explicit.

> At each PGD step, compute `surrogate_J_drop(δ)` = 1 − J(SAM2 prediction on processed video, `m̂_true`). J is measured against **clean-SAM2 pseudo-labels** `m̂_true_t`, not DAVIS GT. `δ*` = argmax surrogate_J_drop over all steps where all fidelity hinges are zero. **Zero DAVIS access at any point of optimization or selection.**

### 5. Soft-logit supervision (Modernization)

**Reviewer modernization #1**: replace hard-threshold pseudo-labels with soft logits.

**Action**: `m̂_true_t` is now defined as `sigmoid(clean_SAM2.pred_masks_high_res_t)` — a soft `[0, 1]`-valued confidence map, not a binary threshold. `L_suppress` uses this as soft weight:

```
L_suppress[t] = Σ_pixels softplus(pred_logits_t · m̂_true_t)
```

This gives smoother gradients and avoids discretization loss. Hard-threshold still used at evaluation (for J computation only).

### 6. Decoy regime deleted from default (Simplification #1)

**Reviewer said**: delete decoy unless suppression fails.

**Action**: decoy regime (`L_rank`, `m̂_decoy_t`, offset construction) removed from default loss. Stays available as an appendix fallback if pilot shows suppression-only cannot reach J-drop ≥ 0.40.

### 7. SAM2Long kept as single stress-row (Simplification #3)

Unchanged from Round 1 — already a single main-table row.

### 8. Tensor-swap boundaries specified (from specific-check)

For restoration hooks, the swap boundary is specified precisely:

- **SwapF0MemoryHook**: at each non-init frame's call to `_prepare_memory_conditioned_features`, the conditioning-slot entry (output of `memory_encoder` for f0) is replaced with the clean-video f0's maskmem + obj_ptr. Other pathway components (recent bank, current-frame Hiera) are attack-origin.
- **SwapHieraFeaturesHook**: at each specified frame, the `image_features` output of `image_encoder.forward_image` is replaced with the clean frame's precomputed output. Everything downstream sees clean current-frame features but attacked memory pathway.
- **SwapBankHook**: for each eval frame, the `non_cond_frame_outputs[t-k]` entries (k=1..7) are replaced with the corresponding `clean_non_cond_frame_outputs[t-k]`. Conditioning slot and current-frame features stay attack-origin.

These are narrowly scoped inference-time intercepts. They do NOT change model weights. Reversible via context manager (same pattern as `DropNonCondBankHook`).

## Revised Proposal (full)

### Problem Anchor (verbatim)

[see top]

### Technical Gap

SAM2.1-Tiny's segmentation is dominated by f0 conditioning memory + current-frame Hiera features. The non-cond FIFO bank contributes `|delta_J| < 0.01` to tracking (measured by removal on 5 DAVIS clips, B2). Therefore:

- UAP-SAM2-style universal attacks pay fidelity budget without targeting causal pathways and are dominated by per-video attacks.
- Prior internal decoy-insert methods poisoned a pathway that does not matter (bank). High `A_insert` does not imply high J-drop.
- The missing mechanism is a **per-video preprocessor whose perturbation concentrates on the actually-causal pathways**, validated by restoration counterfactuals.

### Method Thesis

**Architecture-aware dataset protection via a causal loop.** We (1) diagnose which SAM2 pathway is causally responsible for segmentation (bank-ablation B2 already done), (2) attack only the causal pathways (f0 + current-frame) via per-video PGD with a two-tier fidelity budget and clean-SAM2 soft-logit self-supervision, and (3) verify via restoration counterfactuals that the resulting attack's damage concentrates in the targeted pathways. The paper's contribution is the three-step diagnose→attack→verify loop realized on SAM2, not the PGD solver per se.

### Contribution Focus

- **Dominant contribution (C1, three-part)**: causal-loop dataset protection for SAM2. The three parts are (i) causal diagnosis (bank-marginality measurement on SAM2.1-Tiny; this becomes a design principle, not a standalone contribution); (ii) pathway-targeted PGD with two-tier fidelity + GT-free self-supervision; (iii) restoration-based attribution of the attack's damage to the targeted pathways.
- **Explicit non-contributions**: no new generator; no UAP; no runtime hook; no bank poisoning; no FIFO-self-healing narrative; no LLM/diffusion/RL; no parallel supporting contribution competing with C1.

### Complexity Budget

- **Frozen / reused**: SAM2.1-Tiny, `SAM2VideoAdapter` (Chunk 5b-ii), LPIPS(alex), `fake_uint8_quantize` (STE), `DropNonCondBankHook`.
- **New trainable components (exactly 1)**: `δ ∈ R^{T × H × W × 3}` with two-tier ε (f0=2/255, others=4/255).
- **New inference-time infrastructure (3 hooks, NO trainable components)**: `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook` — inference-only intercepts for restoration counterfactuals.
- **Intentionally not used**: ν inserts, L_stale, decoy regime (in default), teacher, learned scheduler, LLM/diffusion/RL.

### System Overview

```
(Publisher-side, one-time, NO GT)
  m̂_true_t = sigmoid(clean_SAM2(x, m_0).pred_masks_high_res_t)   ∈ [0,1]^{H×W} per frame
  (optional) cache clean Hiera features, clean f0 maskmem, clean bank entries (for restoration tests only)

(PGD loop, 100 steps, 3 stages, per-video)
  Initialize δ = 0
  For step = 1..100:
    x'_t = clip(x_t + δ_t, 0, 1) via fake_uint8_quantize_STE
    forward SAM2VideoAdapter(x', m_0):
      pred_logits_t, object_score_logits_t
    L_suppress = Σ_{t≥1} softplus( ⟨pred_logits_t · m̂_true_t⟩_mean )
    L_obj      = Σ_{t≥1} softplus( object_score_logits[t] + 0.5 )
    L_fid_t    = max(0, LPIPS(x'_t, x_t) - 0.20)     for t ≥ 1
    L_fid_f0   = max(0, 1 - SSIM(x'_0, x_0) - 0.02)
    L          = L_suppress + 0.3 · L_obj + λ(step) · Σ_{t≥1} L_fid_t + λ_0 · L_fid_f0
    δ ← δ - η · sign(∇_δ L)
    clip δ_0 to ±2/255, δ_{t≥1} to ±4/255
    log:   surrogate_J_drop = 1 - J(SAM2_pred(x', m_0), m̂_true)
           per-frame LPIPS, f0 SSIM
  δ* = argmax surrogate_J_drop  over steps where all fidelity hinges = 0
  (no DAVIS access throughout)

(Publisher output)
  x'_0..x'_{T-1} (uint8)

(Restoration attribution, post-hoc, once per processed clip)
  R1 = J(attacked + SwapF0Memory)   - J(attacked)
  R2 = J(attacked + SwapHieraFeats) - J(attacked)
  R12= J(attacked + Swap[F0+Hiera]) - J(attacked)
  R3 = J(attacked + SwapBank)       - J(attacked)
  B_ctrl = J(attacked + DropBank)   - J(attacked)
```

### Loss

```
m̂_true_t = sigmoid(clean_SAM2(x, m_0).pred_masks_high_res_t)   # soft, in [0,1]

L_suppress = Σ_{t≥1} softplus( (1/(H·W)) · Σ_pixels pred_logits_t · m̂_true_t )
L_obj      = Σ_{t≥1} softplus( object_score_logits[t] + 0.5 )
L_fid_frame[t] = (LPIPS(x'_t, x_t) − 0.20)_+
L_fid_f0       = (1 − SSIM(x'_0, x_0) − 0.02)_+

L = L_suppress + 0.3 · L_obj + λ(step) · Σ_{t≥1} L_fid_frame[t] + λ_0 · L_fid_f0
```

Stages:
1. `N_1 = 30`: attack-only (λ=0). Push δ into attack manifold.
2. `N_2 = 40`: fidelity regularization. λ starts at 10, grows 2× every 10 steps when any `L_fid_frame[t] > 0`. λ_0 fixed at 20 (f0 is non-negotiable).
3. `N_3 = 30`: Pareto-best tracking. η halved. Log per-step surrogate_J_drop + fidelity; return `δ*` at argmax feasible surrogate_J_drop.

### Two-tier Fidelity Budget

| Frame | ε_∞ | Fidelity |
|---|---|---|
| f0 (prompt) | 2/255 | SSIM ≥ 0.98 (hinge margin 0.02) |
| t ≥ 1 | 4/255 | LPIPS ≤ 0.20 (hinge); SSIM ≥ 0.95 implied by ε |

### Restoration-Based Pathway Attribution (fixed sign, joint R12 added)

Metric: `ΔJ_restore(config) = J(attacked + swap) − J(attacked)`. Positive = restoration works.

| Config | Swap content | Expected ΔJ_restore | Interpretation |
|---|---|---|---:|
| R1  | f0 maskmem + obj_ptr → clean | ≥ 0.25 | damage lives in f0 conditioning |
| R2  | Hiera features per frame → clean | ≥ 0.30 | damage lives in current-frame features |
| R12 | both f0 and Hiera → clean | ≥ max(R1, R2); ideally ≥ 0.40 | joint upper bound |
| R3  | non-cond bank → clean | ≤ 0.02 | bank is NOT where damage lives (reconfirms B2) |
| B-ctrl | drop non-cond bank (B2 repeat on attacked) | ≤ 0.02 | bank marginal on attacked too |

Interpretation protocol:
- `ΔJ(R12) ≈ max(ΔJ(R1), ΔJ(R2))` → one pathway dominates (report which).
- `ΔJ(R12) > ΔJ(R1) + ΔJ(R2) − small_overlap` → pathways jointly contribute.
- `ΔJ(R1) ≈ ΔJ(R2) ≈ ΔJ(R12) ≈ 0` despite J-drop >> 0 → damage distributed beyond these pathways; attribution inconclusive (still reportable as finding).

Tensor-swap boundaries (for reviewer confidence):
- **SwapF0MemoryHook**: replaces conditioning-slot output of `memory_encoder` on f0 with clean-video f0's maskmem/obj_ptr. Intercepts at `_prepare_memory_conditioned_features`.
- **SwapHieraFeaturesHook**: replaces `image_encoder.forward_image(x'_t)` output with clean `image_encoder.forward_image(x_t)`. Intercepts at Hiera output.
- **SwapBankHook**: replaces `non_cond_frame_outputs[t-k]` entries with their clean-video counterparts. Intercepts at `_prepare_memory_conditioned_features`.

All hooks are inference-only; no gradient flow needed.

### Training Plan

- 100 PGD steps, stages 30/40/30.
- η = 1/255 (stages 1-2), 0.5/255 (stage 3).
- F_lpips = 0.20 (floor-grounded); may per-clip adapt at pilot.
- f0 SSIM threshold = 0.98; hinge margin 0.02.
- Windowing: if T > 40, sliding 30-frame backward window with 10-frame stride. Forward always full-video.
- **Pilot gate (mandatory)**: 1 clip on Pro 6000. Measure: (i) wall-clock per 10 steps; (ii) peak memory; (iii) trajectory of surrogate_J_drop and LPIPS; (iv) feasibility for F_lpips=0.20. If wall-clock > 15 min or peak > 80GB, scale to 50 steps or 20-frame window.

### Novelty and Elegance Argument

The paper's contribution is NOT "white-box PGD on SAM2". The contribution is the **causal loop**:

1. Causal diagnosis of SAM2's tracking pathway (bank marginality — first published for SAM2-family as far as we know).
2. Architecture-aware attack design grounded in (1) — NO insert-based bank poisoning, NO L_stale, YES two-tier budget targeting the causal f0 + current-frame path.
3. Restoration-counterfactual verification that the attack's damage lives where the design targeted.

Closest work:
- UAP-SAM2: universal, pathway-agnostic, different threat model.
- Pre-SAM2 adversarial VOS: no analogue to SAM2's memory structure.
- Attribution/IG-style interpretability: typically on classifiers, not on streaming VOS memory pathways.

Exact novelty: a **diagnosed + targeted + verified** per-video attack on SAM2 with (a) two-tier fidelity budget exploiting SAM2's prompt-frame structure, (b) zero-GT clean-SAM2 self-supervision, (c) restoration-counterfactual pathway attribution.

### Claim-Driven Validation Sketch

#### C1 (dominant): causal-loop dataset protection

Main table (5 rows):
| # | Config | Model | Notes |
|---|---|---|---|
| 1 | Clean | SAM2.1-Tiny | Baseline J&F on DAVIS-10 |
| 2 | Ours | SAM2.1-Tiny | Target: mean J&F drop ≥ 0.40 + fidelity triad met |
| 3 | Uniform-δ (single ε=4/255, no f0 SSIM) | SAM2.1-Tiny | Shows two-tier budget matters |
| 4 | UAP-SAM2 per-clip | SAM2.1-Tiny | Universal-vs-per-video comparison |
| 5 | Ours transfer | SAM2Long | Stress-test sanity row |

Fidelity table: per-frame LPIPS (mean, max), SSIM (mean, f0), f0 ε_∞.

Restoration attribution table (signed):
| Config | Expected ΔJ_restore |
|---|---:|
| R1 f0 only | ≥ +0.25 |
| R2 Hiera only | ≥ +0.30 |
| R12 joint | ≥ max(R1,R2), target ≥ +0.40 |
| R3 bank only | ≤ +0.02 |
| B-control (drop bank) | ≤ +0.02 |

#### Embedded mechanism evidence (from C2 fold-in)

In-paper table: bank-ablation on all 10 attacked clips shows `|ΔJ_bank-drop|< 0.02`. Confirms bank non-causality on BOTH clean and attacked inputs; justifies design choice.

#### (Appendix) prompt robustness + SAM2.1-Base transfer + DAVIS-30 extended

### Experiment Handoff Inputs

- **Must-prove**: C1 (J-drop ≥ 0.40 + fidelity + restoration attribution).
- **Must-run**: uniform-δ baseline; UAP-SAM2 baseline; R1/R2/R12/R3 + B-control; SAM2Long transfer.
- **Datasets / metrics**: DAVIS-2017 val; mean J&F, per-frame LPIPS/SSIM, f0 SSIM, ΔJ_restore for each pathway.
- **Highest-risk assumptions**:
  1. F_lpips = 0.20 is feasible under two-tier budget (pilot-gate validates).
  2. Mean J&F drop ≥ 0.40 is reachable (no prior clean-eval evidence; pilot gate confirms or falsifies).
  3. R1 + R2 individually restore ≥ 0.25 / 0.30. If both ≈ 0 but R12 is high, paper reports joint non-identifiability honestly.

### Compute & Timeline

- Pilot: ~1 GPU-hour.
- DAVIS-10 C1: ~2.5 GPU-hours.
- Restoration (R1/R2/R12/R3 + B-control) on 10 clips: ~0.5 GPU-hour.
- SAM2Long install + 10-clip transfer: ~2-3 GPU-hours.
- Appendix (DAVIS-30, SAM2.1-Base, prompt-robustness): ~5-8 GPU-hours.
- **Total: ~12-15 GPU-hours on a single Pro 6000.**
- Timeline: 3-4 focused days from READY to full results.
