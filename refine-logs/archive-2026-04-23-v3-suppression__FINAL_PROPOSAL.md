# Research Proposal: Architecture-Aware Dataset Protection against SAM2-Family Promptable VOS

**Status**: Pre-pilot ceiling (8.4/10 over 5 GPT-5.4 xhigh review rounds). Formal verdict REVISE until pilot data; internal execution label CONDITIONALLY READY.

**Thread**: `019db862-fbe6-7250-94f0-946474202938`

**Frozen anchor**: `refine-logs/PROBLEM_ANCHOR_2026-04-23.md`

## Problem Anchor

- **Bottom-line**: given a clean video + first-frame target mask, produce a processed video that (a) looks visually faithful under a fidelity budget and (b) causes SAM2 (given the same mask as prompt) to substantially lose the target across the entire processed video.
- **Must-solve bottleneck**: SAM2.1-Tiny tracks via f0 conditioning + current-frame Hiera features; the non-cond FIFO memory bank is architecturally marginal (B2 causal ablation: `|delta_J| < 0.01` across 5 clips). Attacks must target the causal pathways under a fidelity budget grounded in the DAVIS LPIPS floor (mean 0.25 on dog, 0.38 on bmx-trees; ProPainter insert floor 0.67-0.89).
- **Non-goals**: clean-suffix eval (the v2 framing, empirically falsified); defeating FIFO self-healing (not the causal bottleneck); universal perturbation (UAP-SAM2 territory); runtime hook; requiring attacker prompt cooperation.
- **Constraints**: white-box SAM2.1-Tiny surrogate; per-video PGD; pixels-only output; two-tier fidelity (f0: ε=2/255 + SSIM≥0.98; t≥1: ε=4/255 + LPIPS≤0.20 + SSIM≥0.95); DAVIS-2017 val; ≤ ~15 GPU-min / clip.
- **Success condition** (DAVIS-10 main): (1) mean J&F drop ≥ 0.40 on processed videos vs clean; (2) fidelity triad met; (3) restoration-counterfactual attribution confirms damage concentrates in f0 / current-frame pathway, not in the non-cond bank.

## Method Thesis

**Architecture-aware dataset protection for SAM2-family VOS via a diagnose → attack → verify causal loop, with pilot-gate-committed narrative.**

- **Diagnose**: B2 causal ablation shows SAM2.1-Tiny's non-cond FIFO bank is architecturally marginal for segmentation. This is the design principle: do NOT attack the bank.
- **Attack**: per-video PGD on δ with two-tier ε budget, GT-free soft-logit confidence-weighted supervision from clean-SAM2 pseudo-labels, three-stage training, best-feasible checkpoint selection.
- **Verify**: five-config restoration study (R1 f0, R2 Hiera, R12 joint, R3 bank, B-control bank-drop) measuring `ΔJ_restore = J(attacked + swap) − J(attacked)` on attacked videos.

## Contribution Focus

- **C1 (single, four-part)**: (i) causal diagnosis of SAM2's tracking pathway; (ii) pathway-targeted per-video PGD with two-tier budget and GT-free self-supervision; (iii) restoration-counterfactual attribution; (iv) pre-committed pilot-gate decision rule with 4 narrative branches.
- **Non-contributions**: no new generator/diffusion, no UAP, no runtime hook, no bank-poisoning primitive, no FIFO-self-healing narrative, no LLM/RL components.

## Proposed Method

### Complexity Budget
- **Frozen/reused**: SAM2.1-Tiny, `memshield/sam2_forward_adapter.py::SAM2VideoAdapter` (Chunk 5b-ii), LPIPS(alex), `memshield/losses.py::fake_uint8_quantize` (STE), `memshield/ablation_hook.py::DropNonCondBankHook`.
- **New trainable component (exactly 1)**: `δ ∈ R^{T × H × W × 3}` per-video, per-frame, two-tier ε budget (f0=2/255, others=4/255).
- **New inference-only infrastructure (3 hooks, no trainable)**: `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook` — context-manager intercepts for restoration counterfactuals.

### System Overview

```
(Publisher-side, one-time, GT-FREE)
  m̂_true_t = sigmoid(clean_SAM2(x, m_0).pred_masks_high_res_t)   ∈ [0,1]  per frame
  c_t      = | 2 · m̂_true_t − 1 |                                           # confidence weight
  (Optional, for restoration tests): cache clean Hiera features, clean f0 maskmem, clean bank.

(PGD loop, 100 steps, 3 stages, per-video)
  Initialize δ = 0
  For step = 1..100:
    x'_t = clip(x_t + δ_t, 0, 1) via fake_uint8_quantize_STE
    Forward SAM2VideoAdapter(x', m_0) → pred_logits, object_score_logits
    
    L_suppress = Σ_{t≥1} softplus( (1 / Σ_pixels c_t) · Σ_pixels c_t · pred_logits_t · m̂_true_t )
    L_obj      = Σ_{t≥1} softplus( object_score_logits[t] + 0.5 )
    L_fid_frame[t] = max(0, LPIPS(x'_t, x_t) − 0.20)   for t ≥ 1
    L_fid_f0       = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
    
    L = L_suppress + 0.3 · L_obj + λ(step) · Σ_{t≥1} L_fid_frame[t] + λ_0 · L_fid_f0
    
    δ ← δ − η · sign(∇_δ L)
    clip δ_0 to ±2/255, δ_{t≥1} to ±4/255
    log per-step: surrogate_J_drop = 1 − J(SAM2_pred(x', m_0), m̂_true)
                   per-frame LPIPS, f0 SSIM
  
  δ* = argmax surrogate_J_drop over steps where ALL fidelity hinges equal 0
  (Zero DAVIS GT access at any point.)

(Publisher output)
  x'_0..x'_{T-1} as uint8 JPEGs

(Restoration attribution, post-hoc, once per processed clip)
  ΔJ_restore(config) = J(attacked + swap_config) − J(attacked)
  configs: R1, R2, R12, R3, B-control
```

### Primary Loss (B1/B2/B3 branches)

```
L = L_suppress + 0.3 · L_obj + λ(step) · Σ_{t≥1} L_fid_frame[t] + λ_0 · L_fid_f0
```

Stages:
1. `N_1 = 30` steps: attack-only (λ=0). Push δ into attack manifold.
2. `N_2 = 40` steps: fidelity regularization. λ starts at 10, grows 2× every 10 steps when any `L_fid_frame[t] > 0`. λ_0 = 20 (f0 non-negotiable).
3. `N_3 = 30` steps: Pareto-best tracking. η halved. Per-step logging → `δ*` = argmax surrogate_J_drop over fidelity-feasible steps.

### Two-tier Fidelity Budget

| Frame | ε_∞ | Fidelity metric |
|---|---|---|
| f0 (prompt) | 2/255 | SSIM ≥ 0.98 (hinge margin 0.02) |
| t ≥ 1 | 4/255 | LPIPS ≤ 0.20 (hinge) |

Rationale: f0 hosts the user's mask prompt. If f0 is too perturbed, the consumer can redraw the mask on the processed f0 and defeat the attack. Tight f0 keeps the attack unescapable.

### Fallback Loss (B4 branch only; REPLACES L_suppress, not adds)

If pilot triggers B4 (weak attack), swap `L_suppress` for pathway-aligned feature-corruption:

```
h_clean_t  = image_encoder.forward_image(x_t).detach()       # precomputed once
h_attack_t = image_encoder.forward_image(x'_t)               # per PGD step
m_f0_clean  = clean_SAM2 f0 memory encode (maskmem + obj_ptr concatenated, L2-normalized per channel).detach()
m_f0_attack = fresh-SAM2 f0 memory encode on x'_0

L_pathway_attack = α · ⟨ cosine(h_attack_t, h_clean_t) ⟩_{t≥1, pixels ∈ support(m̂_true_t)}
                 + β · cosine(m_f0_attack, m_f0_clean)

L_B4 = L_pathway_attack + 0.3·L_obj + λ(step)·Σ L_fid_frame + λ_0·L_fid_f0
```

Minimizing cosine pushes attack features toward anti-alignment with clean features on the foreground support. Clean features detached so grad doesn't flow through the clean path. α, β chosen so initial magnitude matches L_suppress on the pilot clip. Masks resized to Hiera resolution for the cosine average.

### Pilot-Gate Decision Rule (first-class, pre-committed)

1 clip, ≤ 1 GPU-hour. Measure: surrogate J-drop at δ*, per-frame LPIPS, f0 SSIM, wall-clock/memory, R1/R2/R12 on pilot.

| Branch | Trigger | Action | Paper narrative |
|---|---|---|---|
| **B1** | J-drop ≥ 0.40 + fidelity met + R12 ≥ 0.40 + R1,R2 ≥ 0.15 each | Full DAVIS-10 | Causal-loop dataset protection works. Full narrative. |
| **B2** | J-drop ≥ 0.40 + fidelity met + R12 ≥ 0.40 + R1<0.10, R2<0.10 | Full DAVIS-10 | Causal-loop attack; attribution joint-only. |
| **B3** | J-drop ≥ 0.40 + fidelity met + ALL R1/R2/R12 < 0.05 | Full DAVIS-10 | Attack works; attribution open question. |
| **B4** | J-drop < 0.30 OR fidelity infeasible at F_lpips=0.20 | Swap to L_B4. Rerun pilot. If still < 0.30, pivot. | Paper pivots to "architecture-aware attack-surface analysis of SAM2" honestly reporting ceiling. |

Branches committed BEFORE pilot runs. No post-hoc reshuffle.

### Restoration Attribution (signed)

```
ΔJ_restore(config) = J(attacked + swap_config) − J(attacked)   # positive = restoration
```

| Config | Swap content | Expected | Interpretation |
|---|---|---|---|
| R1 | clean f0 maskmem + obj_ptr | ≥ +0.25 | damage in f0 pathway |
| R2 | clean Hiera per frame | ≥ +0.30 | damage in current-frame pathway |
| R12 | R1 + R2 joint | ≥ max(R1,R2); target ≥ +0.40 | joint upper bound |
| R3 | clean non-cond bank | ≤ +0.02 | bank not damage location |
| B-control | drop non-cond bank | ≤ +0.02 | bank marginal on attacked too |

Protocol: dominated / additive / jointly-non-identifiable (pre-committed, no post-hoc interpretation).

Swap boundaries:
- **SwapF0MemoryHook**: intercepts `SAM2Base._prepare_memory_conditioned_features`. Replaces the conditioning-slot `maskmem_features` + `obj_ptr` with clean-cached f0 values.
- **SwapHieraFeaturesHook**: intercepts `image_encoder.forward_image` output. Replaces with clean-cached Hiera.
- **SwapBankHook**: intercepts `non_cond_frame_outputs` dict. Replaces entries with clean-cached bank.

All hooks are inference-only context managers; no gradient flow, no weight changes.

### Training Plan

- 100 PGD steps, stages 30/40/30.
- η = 1/255 (stages 1-2), 0.5/255 (stage 3).
- F_lpips = 0.20 (floor-grounded); may per-clip adapt at pilot.
- Windowing: if T > 40, backprop over sliding 30-frame windows with 10-frame stride. Forward always full-video.
- bf16 autocast (memory: 60-frame 480p ≈ 50 GB; fits 96 GB Pro 6000).
- STE for fake uint8 quantize.
- **Pilot gate (mandatory, ≤ 1 GPU-hour)**: before full DAVIS-10, run one clip end-to-end and measure. Downgrade to 50 steps or 20-frame window if time/memory exceeds budget.

### Integration

- Reuse `memshield/sam2_forward_adapter.py` for differentiable forward (bypasses `@torch.inference_mode`).
- Reuse `memshield/losses.py` fake uint8 quantize + object_score_positive_loss (adapted signs).
- Extend `memshield/ablation_hook.py` with three restoration swap hooks.
- Drop `memshield/losses_v2.py::l_stale` and `memshield/optimize_v2.py`'s augmented-Lagrangian path from the new driver (replaced by hinge + best-feasible checkpoint).
- New driver: `scripts/run_datasetprotect.py` orchestrating clean-SAM2 pseudo-label computation → PGD → best-feasible selection → uint8 output.

### Validation

**Main table (5 rows)**:
| # | Config | Model | Notes |
|---|---|---|---|
| 1 | Clean | SAM2.1-Tiny | Baseline J&F on DAVIS-10 |
| 2 | Ours | SAM2.1-Tiny | Target: mean J&F drop ≥ 0.40 + fidelity met |
| 3 | Uniform-δ (single ε=4/255, no f0 SSIM) | SAM2.1-Tiny | Shows two-tier budget matters |
| 4 | UAP-SAM2 per-clip | SAM2.1-Tiny | Universal-vs-per-video comparison |
| 5 | Ours transfer | SAM2Long | Stress-test sanity row |

**Restoration attribution table** (signed): R1 / R2 / R12 / R3 / B-control.

**Embedded mechanism evidence**: bank-drop on attacked DAVIS-10 `|ΔJ| < 0.02` confirms bank non-causality on attacked inputs.

**Appendix**: prompt-robustness (different first-frame masks on same processed video), SAM2.1-Base transfer, DAVIS-30 extended.

### Failure Modes and Diagnostics

| Failure mode | Detection | Fallback |
|---|---|---|
| F_lpips=0.20 infeasible on a clip | Pareto frontier has no feasible step | Per-clip-adapt F_lpips = max(0.20, 1.3× natural_adj_LPIPS_mean) |
| f0 SSIM projection fails | SSIM(f0) < 0.98 mid-training | Freeze δ_0 updates once SSIM hits margin (projection ≡ freeze) |
| Surrogate attack succeeds but R1/R2 do NOT recover J | Restoration shows damage is elsewhere | Report honestly; Branch B3 narrative |
| Windowed PGD diverges vs full-video PGD (pilot compare) | J-drop trajectories diverge at window boundaries | Increase window overlap to 15 frames |
| Clean-SAM2 pseudo-labels disagree with DAVIS GT too much | `m̂_true` quality test vs GT on baseline | Publisher's operating regime is to TRUST its own clean-SAM2 segmentation; paper claim is against that regime, not against GT. Report both. |

### Novelty and Elegance Argument

Closest work:
- **UAP-SAM2 (NeurIPS 2025)**: universal, pathway-agnostic, inference-time. Different threat model.
- **Pre-SAM2 adversarial VOS**: smaller trackers; no f0-conditioning / memory-bank structure to exploit.
- **Attribution / IG-style interpretability**: typically on classifiers; not on streaming VOS memory pathways.

Exact novelty: a **diagnosed + targeted + verified** per-video attack on SAM2-family VOS, with (a) two-tier fidelity budget exploiting SAM2's prompt-frame structure, (b) zero-GT clean-SAM2 self-supervision with confidence weighting, (c) restoration-counterfactual pathway attribution, (d) pre-committed pilot-gate narrative.

Why this is focused: ONE trainable component (δ), ONE primary claim (causal loop), ONE supporting measurement (bank marginality, embedded as mechanism evidence).

## Experiment Handoff Inputs

- **Must-prove**: C1 (J-drop ≥ 0.40 with fidelity triad + restoration attribution).
- **Must-run**: uniform-δ baseline, UAP-SAM2 per-clip baseline, restoration R1/R2/R12/R3 + B-control, SAM2Long transfer, bank-ablation on attacked inputs.
- **Critical datasets/metrics**: DAVIS-2017 val; mean J&F, per-frame LPIPS/SSIM, f0 SSIM, ΔJ_restore for each pathway, ΔJ_bank-drop on attacked.
- **Highest-risk assumptions** (to be burned-off by pilot):
  1. F_lpips = 0.20 is feasible under two-tier budget.
  2. Mean J&F drop ≥ 0.40 reachable.
  3. R1 + R2 individually recover ≥ 0.25 / 0.30. If both ≈ 0 but R12 high, paper honestly reports joint non-identifiability (B2 branch).

## Compute & Timeline

- **Pilot**: 1 GPU-hour (branch decision made here).
- **DAVIS-10 C1 (B1/B2/B3)**: ~2.5 GPU-hours.
- **Restoration (R1/R2/R12/R3 + B-control × 10 clips)**: ~0.5 GPU-hour.
- **SAM2Long install + transfer**: ~2-3 GPU-hours.
- **Appendix (DAVIS-30, SAM2.1-Base, prompt-robustness)**: ~5-8 GPU-hours.
- **Total budget**: ~12-15 GPU-hours on a single Pro 6000; 3-4 focused days from PILOT-PASS to full results.
