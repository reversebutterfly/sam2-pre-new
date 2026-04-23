# Research Proposal: Dataset Protection against SAM2-Family Promptable VOS via Architecture-Aware Per-Video Perturbation

## Problem Anchor

(verbatim from `refine-logs/PROBLEM_ANCHOR_2026-04-23.md`)

- **Bottom-line problem**: given a clean video + first-frame target mask, produce a processed video that (a) looks visually faithful to humans under a fidelity budget and (b) causes SAM2 (given the same mask as prompt) to substantially lose the target across the entire processed video.
- **Must-solve bottleneck**: SAM2.1-Tiny tracks via f0 conditioning + current-frame features; the non-cond FIFO bank is architecturally marginal (B2: `|delta_J| < 0.01` on 5 clips). Attacks must target the causal pathways under a fidelity budget grounded in the DAVIS LPIPS floor (mean 0.25 on dog, 0.38 on bmx-trees).
- **Non-goals**: clean-suffix eval; defeating FIFO self-healing; universal perturbation; runtime hook; requiring attacker prompt cooperation.
- **Constraints**: white-box SAM2.1-Tiny surrogate; per-video PGD; pixels-only output; fidelity: ε_0=2/255 + SSIM_0≥0.98 on f0, ε=4/255 + LPIPS ≤ `F_lpips` (~0.15–0.20) + SSIM ≥ 0.95 on t≥1; DAVIS-2017 val; ≤ 10 GPU-min per clip.
- **Success condition**: on DAVIS-10, all three hold — (1) mean J&F drop ≥ 0.40 on processed videos vs clean; (2) fidelity triad met; (3) mechanism attribution via ablation confirms the damage concentrates on current-frame / f0 pathway, not the non-cond bank.

## Technical Gap

Current methods do not solve per-video SAM2 dataset protection under realistic fidelity:

- **UAP-SAM2 (NeurIPS 2025)** is universal — one perturbation for every video. This is a different threat model: it does not condition on the specific video or target mask, so it cannot leverage per-instance signal. For a data publisher shipping a specific video with a specific protected target, a per-video attack can achieve much higher J-drop at a tighter fidelity budget.
- **UAP-SAM2 and variants also embed perturbation at inference time as a one-shot noise**, whereas dataset protection requires that the perturbation travels with the video file — it IS the file the consumer receives.
- **Prior internal "decoy-insert" work (MemoryShield v4, v2)** tried to exploit SAM2's FIFO memory bank by inserting synthetic frames. Two invalidations by this project's own experiments:
  1. The FIFO bank is architecturally marginal for segmentation on SAM2.1-Tiny (B2 ablation shows `|delta_J| < 0.01` across 5 clips when the entire non-cond bank is removed). Poisoning a non-causal pathway cannot yield large J-drop.
  2. Prior "92.5% J-drop" results came from direct δ on eval frames (the v4 perturb_set overlapped the `EVAL_START=10` window), not from bank poisoning. Under honest clean-suffix eval the same machinery delivers J-drop ≈ 0.001 (v2 R001/R002/R003 on dog).

Why naive bigger systems are not enough:

- Stacking more modules (schedulers, teachers, generators) does not help — the bottleneck is not "not enough parts"; it is "the parts are on the wrong pathway".
- Larger PGD budget (ε=8/255, K_ins=9) does not help — the attention probe (D1) shows the bank is already being saturated at `A_insert ≈ 0.54`, and J still does not move.
- Prompting, agentic LLM planning, diffusion priors — these do not address the architectural mismatch. Using them would be decoration, not mechanism.

The missing mechanism is an **architecture-aware attack** that (a) skips the non-causal bank pathway entirely and (b) concentrates its fidelity budget where it actually damages SAM2's tracking signal.

## Method Thesis

**One-sentence thesis**: Per-video PGD on SAM2's structured causal inputs (f0 conditioning at tight budget + current-frame image features at moderate budget) with end-to-end supervision on SAM2's own output logits is the minimum adequate mechanism for dataset protection against SAM2-family VOS under realistic fidelity.

**Why this is the smallest adequate intervention**: a single trainable component (per-video pixel perturbation δ with a two-tier per-frame budget). No new modules, no new generators, no learned schedulers, no memory-poisoning primitives. The attack directly supervises the signal SAM2 actually uses for segmentation.

**Why this route is timely in the foundation-model era**: SAM2 is the first widely-deployed foundation-scale promptable VOS and is being adopted for video annotation, surveillance, medical, and consumer applications. Dataset protection at the data layer — publish processed video, reader with SAM2 cannot segment — is the right threat surface for users who want to retain control over their video's ML-readability. The attack must be architecture-aware precisely BECAUSE SAM2's scale makes naive approaches fail.

## Contribution Focus

- **Dominant contribution (C1)**: a per-video preprocessor for SAM2 dataset protection that achieves mean J&F drop ≥ 0.40 on DAVIS-10 under ε=4/255 + LPIPS ≤ 0.20 + SSIM ≥ 0.95, with structured dual-budget δ (ε_0=2/255 on the prompt frame, ε=4/255 elsewhere) and end-to-end output-level supervision.
- **Optional supporting contribution (C2)**: an architecture-aware attack-surface analysis (B2-style causal ablation on both clean and attacked videos) showing the attack damage concentrates on SAM2's current-frame image-feature pathway, not on the non-cond FIFO memory bank. This is what justifies the design choice "no inserts, no bank poisoning".
- **Explicit non-contributions**: no new SAM2 variant; no new generator or diffusion model; no UAP; no runtime hook; no bank-poisoning mechanism (deliberately excluded per B2); no FIFO-self-healing narrative.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: SAM2.1-Tiny (Hiera encoder + memory_attention + MaskDecoder). No architectural modification; used as a white-box surrogate only.
- **New trainable components** (exactly 1): `δ ∈ R^{T × H × W × 3}` — per-video, per-frame L∞-bounded pixel perturbation. Two-tier budget: `ε_0 = 2/255` on f0, `ε = 4/255` on t ≥ 1.
- **Tempting additions intentionally not used**:
  - No `ν` (insert perturbation) in the default method. ProPainter insert floor is 0.67-0.89 LPIPS — inserts are fidelity-expensive. Default K_ins = 0.
  - No `L_stale` (3-bin KL on attention) — B2 shows this is non-causal. Kept ONLY as a diagnostic in the ablation study.
  - No teacher memory cooperation (v3's idea) — dormant in v4 anyway; no evidence it helps.
  - No learned scheduler / LLM planner / diffusion prior.

### System Overview

```
Input: clean video x_0..x_{T-1}, first-frame mask m_0
  │
  ├── (one-time) freeze SAM2.1-Tiny surrogate; cache Hiera features
  │   on un-perturbed frames for initialization (optional speedup).
  │
  ├── PGD loop (3 stages, total ~100 steps):
  │     Initialize: δ = zeros
  │     For step = 1..N:
  │         x'_t = clip(x_t + δ_t, 0, 1) ∘ fake_uint8_quantize()
  │         forward SAM2VideoAdapter on x'_0..x'_{T-1} using m_0 as f0 prompt
  │         collect per-frame pred_masks_high_res, object_score_logits
  │         compute L_attack + λ · L_fid
  │         δ ← δ - η · sign(∇_δ L)
  │         clip δ_0 to [-2/255, 2/255], δ_{t≥1} to [-4/255, 4/255]
  │         enforce SSIM(x'_0, x_0) ≥ 0.98 via projection
  │         enforce per-frame LPIPS(x'_t, x_t) ≤ F_lpips via hinge penalty
  │     Return best-feasible δ (by J-drop at highest fidelity-feasible step)
  │
  └── Output: x'_0..x'_{T-1} (per-frame quantized to uint8)
```

### Core Mechanism

- **Input / output**: δ is the single learnable tensor. Output is `x' = clip(x + δ)` per frame, uint8-quantized.
- **Architecture / policy**: two-tier L∞ projection + SSIM-honest f0 projection + per-frame LPIPS hinge. No learned sub-modules.
- **Training signal / loss**:
  ```
  L_margin  = Σ_{t≥1} softplus(object_score_logits[t] + 0.5)        # push scores negative
  L_rank    = Σ_{t≥1} softplus(mean(pred_logits ∘ m_true_t) 
                              - mean(pred_logits ∘ m_decoy_t) + 0.75)
  L_bg      = Σ_{t≥1} ⟨-pred_logits⟩ over 1 − (m_true ∪ m_decoy)    # no-spurious
  L_fid_frame[t] = (LPIPS(x'_t, x_t) - F_lpips)_+
  L_fid_f0     = (1 − SSIM(x'_0, x_0) − 0.02)_+
  
  L = (L_margin + L_rank + 0.1 · L_bg) + λ · (Σ_t L_fid_frame[t]) + λ_0 · L_fid_f0
  ```
  Three stages:
  1. `N_1 = 30` steps: attack-only (λ=0). Get δ into the attack manifold.
  2. `N_2 = 40` steps: introduce fidelity regularization (λ=10 init, grow 2× every 10 steps when LPIPS violated).
  3. `N_3 = 30` steps: Pareto-best tracking. Log J-drop (on surrogate) and LPIPS at every step; return the step's δ that achieves the **highest J-drop subject to all fidelity constraints being satisfied**.
- **Why this is the main novelty**: end-to-end output supervision against SAM2's own logits is standard adversarial-attack practice, but applying it with (a) a two-tier per-frame fidelity budget that respects SAM2's prompt-frame dependency and (b) deliberately skipping the non-causal bank pathway (as justified by causal ablation) is, to our knowledge, new. The design is LITERALLY architecture-aware: we empirically measure which SAM2 pathway is causal, then attack that one.

### Optional Supporting Component

None in the default method. `ν` (insert perturbation) may be reintroduced in an appendix if fidelity-amortization experiments show inserts strictly dominate pure δ for a subset of clips. In that case, K_ins ≤ 3 inserts at a simple temporal midpoint schedule (not the canonical write-aligned one — that was motivated by FIFO self-healing, which B2 invalidated).

### Modern Primitive Usage

- **Pretrained SAM2.1 surrogate** is used as the white-box adversarial target. This is the only foundation-model primitive in the method, and it is THE thing being attacked — not a decoration.
- **LPIPS(alex)** is used for fidelity (standard, pretrained AlexNet feature extractor). Not a novelty, just a loss component.
- **Explicit non-uses**:
  - No LLM for decoy selection (target selection is geometric, not linguistic).
  - No diffusion prior (adding one would violate the "smallest adequate mechanism" principle — uniform δ perturbation suffices).
  - No RL / inference-time scaling (a per-video PGD loop is the natural solver).

### Integration into the Adversarial Pipeline

- Reuse `memshield/sam2_forward_adapter.py::SAM2VideoAdapter` for the differentiable forward (bypassing `@torch.inference_mode`). No new SAM2 code.
- Reuse `memshield/losses.py::decoy_target_loss` and `object_score_positive_loss` (adapted signs) for the attack objective.
- Reuse the fake uint8 quantize trick from the breakthrough commit.
- Drop `memshield/losses_v2.py::l_stale` from the loss stack in the core method (kept for diagnostic ablation).
- Drop `memshield/optimize_v2.py`'s augmented-Lagrangian LPIPS formulation and replace with straight hinge-penalty + best-feasible checkpoint selection. The Lagrangian µ-saturation pathology from R002 is avoided by construction.

### Training Plan

- Per-video PGD, ~100 steps, ~10 minutes per clip on RTX Pro 6000. Full DAVIS-10 run ≈ 2 GPU-hours.
- Stages: 30 attack-only / 40 fidelity-regularized / 30 Pareto-best tracking.
- Step size: η = 1/255 for δ, halved at stage 3 for fine tracking.
- Fidelity schedule: F_lpips = 0.20 initially. If floor study update (Step 1.5 below) shows tighter feasible, reduce.
- **Checkpoint policy**: at every step, log on-surrogate J-drop (compared to clean) and per-frame LPIPS/SSIM. Return `δ*` = argmax J-drop over all feasible (fidelity-satisfying) steps. This is the Pareto-best selection that R002 lacked.

### Failure Modes and Diagnostics

| Failure mode | How to detect | Fallback / mitigation |
|---|---|---|
| On-surrogate attack but poor on-model transfer (shouldn't happen since white-box) | J-drop(surrogate) >> J-drop(eval) | Validate surrogate is using exactly the deployed SAM2.1-Tiny weights |
| Fidelity infeasibility: F_lpips=0.20 not achievable for a given clip | Pareto frontier shows no feasible step | Flag the clip; report its "fidelity floor" (LPIPS at clean-input) — possibly extend F_lpips per-clip based on its natural adjacent-frame LPIPS |
| f0 SSIM violation destroys mask prompt | J-drop happens, but eval user can compute corrected mask from processed f0 and restore tracking | Tight ε_0=2/255 + SSIM_0 ≥ 0.98 enforcement; measure "prompt-robustness" as an appendix ablation |
| Attack damages ONLY specific-object-index, not a general "SAM2 can't segment" | Running SAM2 with a different first-frame prompt on the same processed video still works | Evaluate "attack transfer across prompts" on the same processed video as appendix ablation |
| PGD oscillation / no convergence | History shows J-drop bouncing step-to-step | Three-stage schedule + best-feasible checkpoint return handles this by construction; if still bad, decrease η in stage 3 |

### Novelty and Elegance Argument

Closest work:

- **UAP-SAM2 (NeurIPS 2025)**: universal perturbation on SAM2. Different threat model (universal vs per-video), different budget structure (single perturbation vs per-frame), different evaluation (noise at inference vs processed-video dataset).
- **Per-instance adversarial VOS attacks (pre-SAM2 era, e.g. OSVOS attacks)**: generally PGD against a smaller tracker. Different target model; no analogue to SAM2's f0-conditioning + memory-bank structure.

Exact differences:

1. **Threat model**: dataset-level publisher protection, not inference-time noise. The processed video IS the artifact.
2. **Architecture-aware attack surface**: empirically grounded in causal ablation showing SAM2's bank is non-causal for segmentation. No insert-based bank poisoning. No `L_stale`.
3. **Two-tier fidelity budget**: f0 is tightly budgeted because it hosts the mask prompt; other frames are looser. Prior per-frame UAP-SAM2 does not differentiate.
4. **Pareto-best checkpoint**: returns the δ with highest feasible J-drop across training, avoiding the µ-saturation pathology we demonstrated in v2 R002.

Why this is focused, not a module pile-up: ONE trainable component (δ), ONE primary claim (J-drop ≥ 0.40 under realistic fidelity), ONE supporting analysis (causal ablation attribution). No parallel contributions.

## Claim-Driven Validation Sketch

### Claim 1 (primary, C1): Per-video PGD on SAM2 under dual-budget fidelity achieves substantial segmentation damage.

- **Minimal experiment**: DAVIS-10 subset (5 hard + 5 easy: dog, cows, bmx-trees, blackswan, breakdance, car-shadow, breakdance-flare, bear, judo, camel). Run the method per clip with stated hyperparameters. Evaluate J&F on the processed video vs the original clean baseline using SAM2.1-Tiny with the same first-frame mask prompt.
- **Baselines / ablations**:
  - `Clean` (no attack; reference J&F).
  - `Uniform-δ` (no architecture-aware budget tiers; single ε=4/255 on all frames including f0). Tests whether the two-tier budget contributes.
  - `UAP-SAM2-calibrated` (apply the released UAP-SAM2 perturbation to the same 10 clips as a per-video baseline). Tests whether universal competes with per-video.
- **Metric**: mean J&F drop on full processed video (EVAL_START=1 to cover essentially the whole video; the f0 frame is excluded since it's the prompt). Fidelity triad: per-frame LPIPS mean, per-frame SSIM mean, f0 SSIM specifically.
- **Expected evidence**: our method achieves mean J&F drop ≥ 0.40 with fidelity satisfied on 8/10 clips. Uniform-δ underperforms on J-drop or violates f0 fidelity. UAP-SAM2-calibrated is weaker (universal can't match per-video under the same budget).

### Claim 2 (supporting, C2): The attack damages the current-frame / f0 pathway, not the non-cond FIFO bank.

- **Minimal experiment**: on the processed videos from Claim 1, re-run SAM2 with three pathway ablations (same infrastructure as B2):
  - **Ablation A** (baseline): processed + full SAM2 — measure J.
  - **Ablation B** (bank dropped at eval): processed + `DropNonCondBankHook` on all frames — measure J.
  - **Ablation C** (f0 conditioning dropped at eval): processed + hook that drops the conditioning slot — measure J.
- **Metric**: per-clip `ΔJ(B−A)` (J change from removing bank) and `ΔJ(C−A)` (J change from removing f0).
- **Expected evidence**: `|ΔJ(B−A)| < 0.02` (bank is still non-causal even on attacked videos) and `ΔJ(C−A) ≥ 0.20` (removing f0 conditioning makes things WORSE — confirming f0 is what's left holding tracking together and is the actual attack victim). This is the mechanism-attribution test.

### (Appendix) Claim 3: Cross-variant transfer holds within SAM2-family.

- Apply the processed videos from Claim 1 (generated against SAM2.1-Tiny surrogate) as-is to SAM2.1-Base and SAM2Long. Measure J&F drop.
- Expected: substantial attenuation (transfer is harder), but non-trivial J-drop if the attack generalizes.
- Not a primary claim; appendix-only.

## Experiment Handoff Inputs

- **Must-prove claims**: C1 (J-drop ≥ 0.40 with fidelity); C2 (attack damages f0/current-frame, not bank).
- **Must-run ablations**: uniform-δ baseline; UAP-SAM2 baseline; B2-style three-way pathway ablation on attacked videos.
- **Critical datasets / metrics**: DAVIS-2017 val; mean J&F, per-frame LPIPS, per-frame SSIM, f0 SSIM.
- **Highest-risk assumptions**:
  1. F_lpips = 0.20 is actually achievable under the attack budget (to be confirmed by a pilot on dog + cows + bmx-trees).
  2. J-drop ≥ 0.40 is achievable at all (v4's 92.5% was inflated by eval overlap; the honest number on all-frames eval is unknown).
  3. C2's ablation attribution holds (if `ΔJ(B−A)` is NOT small on attacked videos — i.e. the attack actually DOES end up corrupting the bank meaningfully — then C2 needs re-framing).

## Compute & Timeline Estimate

- Per-clip PGD: ~10 minutes on RTX Pro 6000 (per B2 observation that one propagate ≈ 2 sec, PGD 100 steps ≈ 200 sec forward + ~200 sec backward + LPIPS).
- DAVIS-10 primary: ~2 GPU-hours.
- DAVIS-10 B2-style attribution: ~1 GPU-hour (3 propagates × 10 clips × ~2 sec each ≈ 1 minute, times 3 configs).
- Pilot: ~3 clips × 15 minutes ≈ 45 minutes to validate fidelity feasibility before full run.
- Full DAVIS-30 appendix: ~6 GPU-hours.
- SAM2Long transfer: ~2 GPU-hours.
- Total realistic budget to full paper: ~15 GPU-hours (not counting SAM2Long install time).
- Timeline: 2-3 days of focused work from round-0-READY.
