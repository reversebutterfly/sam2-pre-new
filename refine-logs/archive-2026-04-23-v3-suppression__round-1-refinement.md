# Round 1 Refinement

## Problem Anchor (verbatim from `PROBLEM_ANCHOR_2026-04-23.md`)

- **Bottom-line problem**: given a clean video + first-frame target mask, produce a processed video that (a) looks visually faithful to humans under a fidelity budget and (b) causes SAM2 (given the same mask as prompt) to substantially lose the target across the entire processed video.
- **Must-solve bottleneck**: SAM2.1-Tiny tracks via f0 conditioning + current-frame features; the non-cond FIFO bank is architecturally marginal (B2: `|delta_J| < 0.01` on 5 clips). Attacks must target the causal pathways under a fidelity budget grounded in the DAVIS LPIPS floor.
- **Non-goals**: clean-suffix eval; defeating FIFO self-healing; universal perturbation; runtime hook; requiring attacker prompt cooperation.
- **Constraints**: white-box SAM2.1-Tiny surrogate; per-video PGD; pixels-only; fidelity budget grounded in floor study; DAVIS-2017 val; ≤ ~10 GPU-min per clip.
- **Success condition**: on DAVIS-10, (1) mean J&F drop ≥ 0.40 on processed videos vs clean; (2) fidelity triad met; (3) causal pathway attribution.

## Anchor Check

- **Original bottleneck**: damage SAM2's segmentation of the protected target across the processed video, under a realistic fidelity budget, without touching the non-causal FIFO bank.
- **Does the revised method still solve it?** Yes. The revisions tighten the mechanism but do not change the target or budget.
- **Reviewer suggestions rejected as drift**: NONE. All Round-1 asks are method-level improvements, not problem redefinitions. The critical "supervision leakage" concern, if we had ignored it, WOULD have drifted the method into cheating (using test-time GT at optimization) — addressing it is the OPPOSITE of drift.

## Simplicity Check

- **Dominant contribution after revision**: an architecture-aware dataset-protection preprocessor for SAM2 — (i) empirically diagnose which SAM2 pathway is causal, (ii) attack only that pathway via per-video PGD under a realistic fidelity budget, (iii) verify attribution via restoration counterfactuals. Three-step story, one paper thesis.
- **Components removed or merged**:
  - Removed `L_rank` as a standalone term (folded into a single `L_suppress` via pseudo-label supervision; decoy construction made optional).
  - Merged C2 into C1 — mechanism attribution is mandatory, not appendix.
  - Merged SSIM projection + LPIPS hinge into a single "soft hinge + best-feasible checkpoint" framework.
- **Reviewer suggestions rejected as unnecessary complexity**: NONE on new additions. One clarification: we add **restoration counterfactuals** (new experiment) but not new trainable components — this is a measurement tool, not a mechanism.
- **Why the remaining mechanism is still the smallest adequate route**: one trainable component (δ), one dominant supervision source (clean-SAM2 self-supervision), no learned modules.

## Changes Made

### 1. Supervision leakage — fixed (CRITICAL)

**Reviewer said**: loss uses `m_true_t` and `m_decoy_t` without defining where they come from; if GT, the method leaks evaluation annotations.

**Action**: define pseudo-label regime and drop GT entirely from optimization.

**Reasoning**: a dataset publisher does not have GT at publish time; they have the clean video and the first-frame mask the consumer will use as a prompt. The clean-SAM2 pseudo-labels are exactly what the publisher can compute and are the right supervision target.

**Impact**:
- Define `m̂_true_t` = sigmoid(clean-SAM2 pred_masks_high_res at frame t), computed ONCE on the clean video via vanilla `propagate_in_video` with `m_0` as prompt. Binary-thresholded at 0.5. No DAVIS access.
- Optional `m̂_decoy_t` = shift_mask(m̂_true_t, (dy, dx)) with decoy offset computed via the same geometric heuristic as the prior decoy pipeline (no learned selection). Used only when fidelity-amortization experiments show decoy targeting helps.
- **`L_rank` retained ONLY when decoy is used**; dropped from the default method. Default `L_suppress` = `softplus(mean(pred_logits ∘ m̂_true_t))` per frame — push the true-location confidence down. Cleaner and GT-free.

### 2. Contribution reframing — C2 promoted to mandatory (IMPORTANT)

**Reviewer said**: method risks reading as "standard white-box per-video PGD on evaluated frames".

**Action**: restructure the contribution as **path-specific causal attack design**.

**Reasoning**: the paper's real novelty is the three-step methodology — diagnose → attack → verify — not the PGD solver itself.

**Impact**:
- Restated thesis: "Architecture-aware dataset protection against SAM2 — cause diagnosis (B2 shows bank non-causal), pathway-matched attack (no inserts, no L_stale), and restoration-based attribution."
- C2 (pathway attribution) is now **in the main table**, not the appendix. It is the evidence that the design choices are CORRECT, not arbitrary.
- Title sketch: "Pathway-Specific Dataset Protection against SAM2: What Memory Structures Actually Matter for Adversarial VOS".

### 3. Restoration counterfactuals — ablation redesigned (IMPORTANT)

**Reviewer said**: "drop f0 conditioning" harms BOTH clean and attacked; doesn't isolate attack location.

**Action**: replace drop-based ablation with **restoration-based counterfactuals**.

**Reasoning**: dropping a pathway's MAGNITUDE is non-specific; swapping the clean version of the pathway INTO the attacked run is specific ("does the attack's damage go away if we fix THIS pathway?").

**Impact**: new C2 experiment suite (3 restoration configs + 1 control):
- **R1** (attacked + clean f0): swap the attacked video's f0 conditioning memory with what SAM2 would have written if f0 were clean. If J recovers → attack's damage lives in the f0 pathway.
- **R2** (attacked + clean Hiera at eval frames): swap the current-frame image-feature output for each eval frame with what the clean-video Hiera would produce. If J recovers → attack's damage lives in current-frame features.
- **R3** (attacked + clean non-cond memory bank): at each eval frame, synthesize the non-cond memory entries from the clean video's prior predictions. If J recovers, bank matters after all — this would flag our B2 reading as wrong.
- **B-control** (attacked + bank dropped): repeat B2 on attacked videos to confirm the bank remains marginal.

Implementation: extends `memshield/ablation_hook.py` with pathway-swap hooks. Infrastructure is already in place (RuntimeProvenanceHook, DropNonCondBankHook).

### 4. Feasibility — concrete implementation + pilot gate (IMPORTANT)

**Reviewer said**: 100-step full-video SAM2 backprop may exceed 10 min/clip. Specify STE, memory, resolution/windowing, measured pilot gate.

**Action**: pin down implementation details and add a pilot-gate stopping rule.

**Reasoning**: the prior proposal's time estimate was extrapolated from R002 (K_ins=1, T=22 frames, 200 steps in 421s). Full-clip DAVIS (T=60-100) is 3-5× more frames; backprop over the full video may OOM.

**Impact**:
- **Quantization**: fake uint8 via straight-through estimator: `x_q = quantize(x).detach() + (x - x.detach())`. Gradient flows through `x`. Already implemented in `memshield/losses.py::fake_uint8_quantize`.
- **Memory**: bf16 autocast on SAM2VideoAdapter (already the default path). Expected: 22-frame × 1024² clips ≈ 33GB; 60-frame DAVIS clips at 480p would be ~50GB if we backprop over all frames simultaneously. Pro 6000 has 96GB — fits one full clip at a time.
- **Windowing**: if a clip has > 40 frames, backprop over a **sliding window of 30 frames** (10-frame stride). The SAM2 forward always runs on the full video (no windowing for measurement); only the BACKWARD pass is windowed. For each window, PGD updates δ for that subset of frames; frames outside the current window are fixed. ~50 steps per window; total steps stay ≤ 100 per clip.
- **Resolution**: DAVIS 480p native. SAM2.1-Tiny internal resolution is 1024 (after upsample from 480).
- **Pilot gate (mandatory before any full DAVIS-10 run)**: 1 clip on Pro 6000 measuring: (i) wall-clock per 10 PGD steps, (ii) peak memory, (iii) surrogate J-drop trajectory, (iv) fidelity feasibility. If 100 steps > 15 min/clip or memory > 80GB, downgrade to 50 steps / sliding-window 20 frames. Pilot gate is 1 GPU-hour.

### 5. Scope sharpening — SAM2Long sanity test promoted (IMPORTANT)

**Reviewer said**: novelty depends too much on execution magnitude; either narrow title to SAM2.1-Tiny or promote transfer to main.

**Action**: promote a single SAM2Long sanity test to the main table; keep larger SAM2.1 variants as appendix.

**Reasoning**: SAM2Long is the stronger SAM2-family variant that reviewers will ask about. A single number showing the attack transfers (or not) is more defensible than a long transfer sweep.

**Impact**: main results table now has 3 rows instead of 1:
- Row 1: attack on SAM2.1-Tiny (surrogate, direct).
- Row 2: transfer to SAM2.1-Tiny with a different first-frame prompt sampling (robustness check — does the attack depend on the exact mask prompt?).
- Row 3: transfer to SAM2Long (same attack video, different model). If J-drop attenuates to ~0.5× surrogate, that's a good sanity signal of "the attack is real, not surrogate-overfit".

SAM2Long install on Pro 6000 estimated at 2-3 hours; included in compute budget.

## Revised Proposal

### Problem Anchor (unchanged)

(see `PROBLEM_ANCHOR_2026-04-23.md` and summary at top of this document)

### Technical Gap (unchanged)

As in round-0. Key points preserved:
- UAP-SAM2 is universal; we are per-video.
- Prior internal decoy work was invalidated: bank is architecturally marginal (B2), and v4's 92.5% was an eval-overlap confound.
- Frontier primitives (LLM, diffusion) are decoration here; the bottleneck is architectural mismatch, not planning / generation.

### Method Thesis (refined)

**One-sentence thesis**: Architecture-aware dataset protection for SAM2-family VOS — diagnose which SAM2 pathway is causally responsible for segmentation (via bank-ablation), attack only that pathway via per-video PGD with two-tier fidelity budget and clean-SAM2 self-supervision, and verify attribution via restoration counterfactuals.

**Why this is the smallest adequate intervention**: one trainable component (δ), no learned modules, no new generators, all supervision from clean-SAM2 pseudo-labels (zero GT access at optimization), pathway attribution measured from already-built causal ablation infrastructure.

### Contribution Focus (refined)

- **Dominant contribution (C1)**: a **pathway-diagnosed per-video preprocessor** for SAM2-family dataset protection achieving mean J&F drop ≥ 0.40 on DAVIS-10 under the two-tier fidelity budget, with restoration-based attribution showing damage is concentrated in the current-frame / f0 pathway, not in the non-cond bank. (Note: C1 now embeds the mechanism attribution that was previously C2.)
- **Supporting contribution (C2)**: a causal-ablation study of SAM2.1-Tiny that shows its non-cond FIFO bank is architecturally marginal for segmentation — the first published observation as far as we are aware, and the basis on which we design our attack to skip the bank pathway. This is scope-independent of the attack; usable on its own as a measurement paper or as a design principle for future preprocessor attacks on SAM2-family models.
- **Explicit non-contributions**: no new generator; no UAP; no runtime hook; no bank-poisoning primitive; no FIFO-self-healing narrative; no LLM/diffusion/RL components.

### Proposed Method

#### Complexity Budget

- **Frozen / reused**: SAM2.1-Tiny, SAM2VideoAdapter (Chunk 5b-ii), LPIPS(alex), fake uint8 quantize (from `memshield/losses.py`), DropNonCondBankHook + new restoration hooks.
- **New trainable components (exactly 1)**: `δ ∈ R^{T × H × W × 3}` with two-tier ε budget.
- **Intentionally not used**: ν inserts, L_stale, teacher cooperation, learned scheduler, LLM/diffusion/RL.

#### System Overview

```
Input: clean video x_0..x_{T-1}, first-frame mask m_0
  │
  ├── (one-time, publisher-side, NO GT)
  │   m̂_true_{0..T-1} = sigmoid(SAM2(x)(m_0)) > 0.5         # clean SAM2 pseudo-labels
  │   if decoy regime: m̂_decoy_t = shift_mask(m̂_true_t, offset)
  │
  ├── PGD loop (100 steps, 3 stages; sliding-window backward for T > 40)
  │     for step in 1..N:
  │         x'_t = clip(x_t + δ_t, 0, 1) ∘ fake_uint8_quantize_STE()
  │         with SAM2VideoAdapter (bf16 autocast, no inference_mode):
  │             forward from x'_0 with m_0 prompt, collect per-frame
  │             pred_masks_high_res, object_score_logits, Hiera features
  │         L_suppress  = Σ_{t≥1} softplus(mean(pred_logits ∘ m̂_true_t))
  │         L_obj       = Σ_{t≥1} softplus(object_score_logits[t] + 0.5)
  │         (optional) L_decoy = Σ_{t≥1} softplus(-mean(pred_logits ∘ m̂_decoy_t))
  │         L_fid_frame[t] = (LPIPS(x'_t, x_t) - F_lpips)_+
  │         L_fid_f0    = (1 − SSIM(x'_0, x_0) − 0.02)_+
  │         L_attack = L_suppress + 0.3 · L_obj + (0 or 0.5 · L_decoy)
  │         L = L_attack + λ(step) · Σ_t L_fid_frame[t] + λ_0 · L_fid_f0
  │         δ ← δ - η · sign(∇_δ L); clip per-frame ε; project f0 SSIM
  │         log per-step: surrogate J-drop, per-frame LPIPS, f0 SSIM
  │     δ* = argmax J-drop across all fidelity-feasible steps
  │
  └── Output: x'_0..x'_{T-1} (uint8)
```

#### Core Loss (self-supervised, GT-free)

```
# All targets derive from clean-SAM2 pseudo-labels m̂_true_t (frozen, computed once)
L_suppress = Σ_{t≥1} softplus(mean(pred_logits ∘ m̂_true_t))      # push true-location confidence ≤ 0
L_obj      = Σ_{t≥1} softplus(object_score_logits[t] + 0.5)       # push object_score below -0.5

# Optional, only in decoy-regime runs
L_decoy    = Σ_{t≥1} softplus(-mean(pred_logits ∘ m̂_decoy_t))    # raise decoy-location confidence

L_fid_frame[t] = (LPIPS(x'_t, x_t) - F_lpips)_+                   # per-frame LPIPS hinge
L_fid_f0       = (1 − SSIM(x'_0, x_0) − 0.02)_+                   # f0 special (SSIM ≥ 0.98)

L_total = L_suppress + 0.3·L_obj + γ_decoy · L_decoy              # γ_decoy ∈ {0, 0.5}
        + λ(step) · Σ_t L_fid_frame[t] + λ_0 · L_fid_f0
```

Stages:
1. `N_1 = 30`: λ=0 (attack-only).
2. `N_2 = 40`: λ starts at 10, grows 2× every 10 steps when any LPIPS violated.
3. `N_3 = 30`: Pareto-best tracking; η halved; return `δ*` = argmax J-drop on steps where all fidelity constraints satisfied.

#### Two-tier Fidelity Budget (key design choice)

| Frame | L∞ ε | LPIPS / SSIM |
|---|---|---|
| f0 (prompt frame) | **2/255** | SSIM ≥ 0.98 (hinge margin 0.02) — keep the mask prompt honest |
| t ≥ 1 (non-prompt) | 4/255 | LPIPS ≤ `F_lpips` = 0.20 (floor-study grounded) |

Rationale: f0 hosts the user-provided mask prompt. If f0 is too perturbed, the consumer can visually re-draw the mask on the processed f0 and defeat the attack. Tightly constraining f0 preserves the attack's "the consumer cannot easily escape" property.

#### Restoration-Based Pathway Attribution (C2)

New infrastructure (extends `memshield/ablation_hook.py`):

- `SwapF0MemoryHook`: at each subsequent frame, replace the f0 conditioning slot's `maskmem_features`/`obj_ptr` with the values SAM2 would have written if f0 were the CLEAN image. Requires running clean SAM2 forward once offline to cache the clean f0 memory.
- `SwapHieraFeaturesHook`: at each specified frame, replace the Hiera encoder output with the one computed from the CLEAN frame's image. Requires clean Hiera cache.
- `SwapBankHook`: at each eval frame, replace non-cond bank entries (maskmem from previous attacked predictions) with the corresponding clean bank entries.

Experiment configs:
- **R1** (swap clean f0): attacked video + clean f0 memory. Expected: J recovers substantially → attack damages f0 pathway.
- **R2** (swap clean Hiera per frame): attacked video + clean Hiera features at each t. Expected: J recovers substantially → attack damages current-frame pathway.
- **R3** (swap clean bank): attacked video + clean bank. Expected: J barely moves → bank is NOT where the damage lives (consistent with B2).
- **B-control** (drop bank on attacked): attacked video + no bank. Expected: J barely moves → reconfirms B2 on attacked inputs.

This four-way restoration study attributes the damage to a specific SAM2 pathway. It is the mechanism-level evidence that the paper needs.

#### Integration

- Reuse `memshield/sam2_forward_adapter.py::SAM2VideoAdapter` for differentiable forward.
- Reuse `memshield/losses.py::decoy_target_loss`, `object_score_positive_loss`, `fake_uint8_quantize`.
- Extend `memshield/ablation_hook.py` with three swap hooks (SwapF0Memory, SwapHieraFeatures, SwapBank).
- Drop `memshield/losses_v2.py::l_stale`, `memshield/optimize_v2.py`'s augmented-Lagrangian (replaced by hinge + best-feasible checkpoint).
- New per-video driver `scripts/run_datasetprotect.py` that orchestrates: clean-SAM2 pseudo-label computation → PGD → best-feasible selection → output uint8 processed video.

#### Training Plan

- Per-video PGD ≤ 100 steps, ≤ 15 min per clip on RTX Pro 6000 (to be pilot-validated).
- Stages: 30 / 40 / 30 (attack / regularized / Pareto-best).
- Step size η = 1/255 (stages 1-2), η/2 (stage 3).
- F_lpips = 0.20 (floor-grounded); may per-clip adapt based on clip's natural adjacent-frame LPIPS (measure at pilot).
- SSIM threshold f0 = 0.98; others = 0.95 (via hinge).
- Windowing: if T > 40 frames, backprop over sliding 30-frame windows with 10-frame stride; forward stays full-video.
- **Pilot gate**: 1 clip (e.g. dog @ T=60). Measure: wall-clock, memory, surrogate J-drop, fidelity feasibility. If any metric exceeds budget, scale down (50 steps or 20-frame windows).

#### Failure Modes and Diagnostics

| Failure mode | How to detect | Fallback / mitigation |
|---|---|---|
| F_lpips=0.20 infeasible for a clip | Pareto frontier shows no feasible step | Flag; report clip's fidelity floor; adapt per-clip F_lpips to max(0.20, 1.3× natural_floor) |
| f0 SSIM projection doesn't hold | SSIM(f0) drops below 0.98 mid-training | Stop δ_0 updates when SSIM at threshold; cheaper than projection |
| On-surrogate attack succeeds but R1 restoration does NOT recover J | Attack damages are distributed beyond f0/current-frame — maybe coupling effect | Report honestly; expand R1/R2 to joint swaps; weaker attribution claim |
| Windowed PGD diverges vs full-video PGD on pilot | J-drop shows large jumps at window boundaries | Increase window overlap to 15 frames or use weighted blend at boundaries |
| Clean-SAM2 pseudo-labels differ from GT too much | Low J on clean vs GT (baseline) | Report both; paper claim is against clean-SAM2 segmentation regime; pseudo-label quality IS the publisher's operating regime |

#### Novelty and Elegance Argument

Closest work:
- **UAP-SAM2 (NeurIPS 2025)**: universal, inference-time noise. Different threat model.
- **Adversarial VOS attacks pre-SAM2**: smaller trackers; no f0-conditioning / memory-bank structure to exploit.
- **Model-stealing / dataset-watermarking literature**: different goal (detect or authenticate), similar threat model flavor but non-adversarial methods.

Exact differences:
1. **Pathway-diagnosed attack design**. We MEASURE which SAM2 pathway matters, THEN attack. Prior work treats the model as a black box or attacks arbitrary pathways.
2. **Two-tier fidelity budget** matched to SAM2's prompt-frame structure.
3. **Restoration-counterfactual attribution**. Our attack comes with a causal decomposition showing which pathway it damages — mechanism-level evidence.
4. **Clean-SAM2 self-supervision**. No GT access at optimization; honest publisher-side setup.

Why this is focused: ONE new trainable component (δ), ONE primary claim (pathway-specific attack + mechanism attribution), ONE supporting measurement contribution (B2 bank-marginality).

### Claim-Driven Validation Sketch

#### Claim 1 (primary, C1): Pathway-diagnosed dataset protection.

- **Minimal experiment**: DAVIS-10 (5 hard + 5 easy: dog, cows, bmx-trees, blackswan, breakdance, car-shadow, breakdance-flare, bear, judo, camel). Run per-clip PGD.
- **Main table rows**:
  1. Clean baseline: SAM2.1-Tiny J&F on original DAVIS-10.
  2. Our method: SAM2.1-Tiny J&F on processed DAVIS-10.
  3. Uniform-δ (no two-tier budget): same PGD with ε=4/255 everywhere including f0. Tests whether two-tier matters.
  4. UAP-SAM2 per-clip: apply released UAP-SAM2 perturbation as a universal-baseline comparison.
  5. **Sanity** (promoted from appendix): transfer to SAM2Long on the processed videos, no re-optimization.
- **Fidelity report**: per-frame mean LPIPS, SSIM; f0 SSIM; max-frame LPIPS violations.
- **Metric**: mean J&F drop (EVAL_START=1 to cover whole video excluding prompt f0).
- **Expected evidence**: mean J&F drop ≥ 0.40 (target) with all fidelity constraints met on ≥ 8/10 clips. Uniform-δ achieves similar J-drop BUT violates f0 fidelity (implies our two-tier budget matters). SAM2Long transfer shows ≥ 0.25 J-drop (partial but non-zero).

#### Claim 2 (supporting, C2): SAM2.1-Tiny's non-cond FIFO bank is architecturally marginal.

- **Minimal experiment** (already 80% done from R4 B2-multi):
  - B2-clean (done): 5 clips × clean SAM2 × (normal, bank-dropped). Result: `|delta_J| < 0.01` across all.
  - B2-attacked (new): same 10 clips × attacked SAM2 × (normal, bank-dropped). Expected: still `|delta_J| < 0.02`.
- **Metric**: per-clip delta_J from bank ablation.
- **Expected evidence**: on the 10 published clips, dropping the non-cond bank at eval time changes J by less than 0.02 on attacked videos too → confirms bank-marginal regardless of input.

#### Claim 1 evidence-cont., restoration attribution (fused into C1):

- **Experiment**: four restoration configs (R1 / R2 / R3 / B-control) on the 10 processed clips.
- **Metric**: `ΔJ_restore(config) = J(attacked) - J(attacked + swap)`. Positive means the swap restores J.
- **Expected evidence**: `ΔJ_restore(R1 swap clean f0) ≥ 0.25`, `ΔJ_restore(R2 swap clean Hiera) ≥ 0.30`, `ΔJ_restore(R3 swap clean bank) ≤ 0.02`, `ΔJ_restore(B-control) ≤ 0.02`. This is the attribution test: damage lives in current-frame + f0 pathways, not in the bank.

### Experiment Handoff Inputs

- **Must-prove claims**: C1 (pathway-diagnosed attack + attribution); C2 (bank marginality on attacked inputs).
- **Must-run ablations**: uniform-δ vs two-tier-δ; restoration R1/R2/R3; B-control on attacked inputs.
- **Critical datasets / metrics**: DAVIS-2017 val; mean J&F, per-frame LPIPS, SSIM, f0 SSIM.
- **Highest-risk assumptions**:
  1. F_lpips = 0.20 is achievable under the two-tier budget (pilot-gate validates).
  2. Mean J&F drop ≥ 0.40 is achievable at all (need pilot evidence; v4's 92.5% was an eval confound so unknown on honest eval).
  3. Restoration attributions behave as predicted (`R1`/`R2` recover; `R3`/`B-control` don't) — if not, we have a harder-to-tell-story but potentially interesting finding (distributed damage).

### Compute & Timeline Estimate

- **Pilot** (1 clip, ~15 min): validates time/memory; ~1 GPU-hour.
- **DAVIS-10 primary**: 10 clips × 15 min = 2.5 GPU-hours.
- **DAVIS-10 C2 attribution**: 10 clips × 4 restoration configs × ~30s each ≈ 20 min = 0.5 GPU-hour.
- **SAM2Long transfer**: SAM2Long install 2-3 GPU-hours, then 10 clips × 30s ≈ 5 min = 0.2 GPU-hour.
- **Appendix** (DAVIS-30 + SAM2.1-Base transfer): ~5-8 GPU-hours.
- **Total realistic**: ~12-15 GPU-hours to full paper.
- **Timeline**: 3-4 focused days from round-1-READY to full results, on a single Pro 6000.
