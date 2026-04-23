# Research Proposal: MemoryShield — A Two-Phase Preprocessor for Protecting Video Data from Promptable Foundation-Model Segmenters

## Problem Anchor (verbatim — see PROBLEM_ANCHOR_2026-04-22.md)

- **Bottom-line problem**: User-controlled preprocessor that takes a clean video + first-frame mask of a target to protect, and outputs a modified video which causes any downstream SAM2-style promptable VOS model to lose the target and not recover, while remaining visually acceptable.
- **Must-solve bottleneck**: streaming memory banks self-heal from single-frame perturbations; single-component attacks fail under SAM2's memory-rollover dynamics.
- **Non-goals**: UAP / backdoor / runtime hook / maximal attack; all of these violate the threat model or the fidelity constraint.
- **Constraints**: white-box architecture knowledge, per-video PGD, pixels-only output, insert LPIPS ≤ 0.15 (goal 0.10), SSIM ≥ 0.95 on attacked originals, DAVIS-2017 val, 30-clip or DAVIS-10 subset.
- **Success condition**: attacked eval-window J-drop ≥ 0.55 AND no-recovery (monotone drop after loss) AND fidelity triad met AND each of the two phases shown necessary via ablation (≥ 40% relative loss if either removed) AND transfer to SAM2Long.

## Technical Gap

Current generation of **memory-based VOS foundation models** (SAM2, SAM2Long, XMem-family) maintain a streaming FIFO of recent-frame feature maps plus a privileged conditioning-frame memory (f0). At each new frame, memory-attention cross-attends over the bank + current image features to produce masks, then writes back features. This design is **actively self-healing**:

- A single-frame perturbation briefly degrades the mask, but the next clean frame writes a corrected feature into memory, evicting the poisoned slot within `num_maskmem=7` frames.
- Universal perturbations (UAP-SAM2 "Vanish into Thin Air", NeurIPS 2025) bypass this by degrading EVERY frame globally — but they require perturbing the entire video and are not suited to a preprocessor protecting a specific target with low LPIPS.
- First-frame-only attacks (one-shot VOS attacks, ACMM 2023) exploit the privileged conditioning memory but are defeated when the downstream pipeline re-prompts or when the memory naturally refreshes.

**The gap**: existing preprocessor-style protection cannot survive self-healing. What is missing is a mechanism that (a) forces a catastrophic loss event AND (b) keeps the memory bank in a post-loss-only steady state so that SAM2 never re-acquires, all within a tight visual-fidelity budget.

## Method Thesis

**One-sentence thesis**: We protect video data by combining two minimal interventions — (i) synthetic frames inserted at FIFO-resonant positions that drive an immediate tracking-loss event by creating mislocated memory entries, and (ii) bounded perturbations on original frames adjacent to each insert that keep the FIFO in a post-loss steady state so the tracker cannot self-heal — yielding a cleanly two-phase protection mechanism that directly targets SAM2's memory-rollover vulnerability.

- **Why this is the smallest adequate intervention**: single-component baselines (insert-only or perturb-only) fail by construction under FIFO self-healing; two components are the minimum because one must induce loss and the other must prevent recovery. Adding more would be contribution sprawl.
- **Why timely in foundation-model era**: SAM2 is THE current reference for video-segmentation foundation models. Its streaming memory is the new architectural primitive that prior VOS attacks (MaskTrack / STM era) did not address. Protection methods must target the memory-bank vulnerability to remain relevant.

## Contribution Focus

- **Dominant contribution**: a **two-phase preprocessor-style protection mechanism** against SAM2-style VOS foundation models, with formal characterization of the loss-induction and recovery-prevention roles and their composition. Both behaviors are empirically **necessary** (ablation) and **sufficient together** (no further modules needed).
- **Optional supporting contribution**: a **memory-staleness regularization term** `L_stale` that operationalizes recovery-prevention by penalizing memory-attention mass on clean-anchor slots.
- **Explicit non-contributions**: no new SAM2 architecture, no new generator (use off-the-shelf inpainter), no UAP / transfer / backdoor claims, no runtime hook, no learned scheduler.

## Proposed Method

### Complexity Budget

- **Frozen / reused**: SAM2.1 Hiera-Tiny (attack surrogate), LPIPS(AlexNet) for fidelity, RAFT/Unimatch optical-flow (frozen), ProPainter video inpainter (frozen, for insert-content base).
- **New trainable components** (≤ 2 per skill rule):
  1. Per-video adversarial perturbation `δ_orig` on original frames (per-video PGD over pixels, no learned model)
  2. Per-video insert-content tensor `ν_k` (per-video PGD over pixels on top of ProPainter base)
- **Tempting additions intentionally not used**:
  - NO learned scheduler (fixed FIFO-resonant positions are sufficient)
  - NO learned generator (ProPainter used frozen)
  - NO auxiliary teacher model (prior work falsified independent value)
  - NO bilevel virtual-state optimization (unnecessary under anchor; two-phase story does not require tensor-level state control)

### System Overview

```
Clean video x_0:T + first-frame mask m_0
   │
   ├── Phase 1 (LOSS INDUCTION — inserted frames)
   │   ├── Pick K_ins = 3 insert positions at FIFO-resonant offsets {after f3, after f7, after f11}
   │   ├── Build insert base per slot: ProPainter-inpaint object out of f_prev,
   │   │   paste object crop at a spatially displaced "decoy" location
   │   └── Per-video PGD on pixel tensor ν_k within LPIPS ≤ 0.10 vs f_prev
   │
   ├── Phase 2 (RECOVERY PREVENTION — original-frame perturbations)
   │   ├── Attack window = first N_pref = 15 original frames (f0..f14)
   │   └── Per-video PGD on δ_orig within L∞ ≤ 4/255 (f0: 2/255)
   │
   ├── Joint optimization stage: alternating-freeze PGD on (ν, δ) minimizing a
   │   shared two-phase objective (see losses below)
   │
   └── Output: modified video x_0:T' = attacked originals interleaved with inserts
```

### Core Mechanism

**Input**: clean frames `x_0..x_{T-1}`, binary mask `m_0`. **Output**: modified video. **Eval**: J&F on frames f15..end under SAM2.

**Architecture / policy**: per-video PGD only — no learned cross-video model. Surrogate = SAM2 open checkpoint, white-box gradients.

**Training signal** (per-video loss). Let `g_u(x)` = SAM2 output logits for frame u under modified video x. Let `C_u` = true-object region from clean SAM2 run, `D_u` = displaced decoy region (from m_0 shifted by per-video decoy offset chosen at video start).

$$
\mathcal{L}(\nu, \delta) = \mathcal{L}_{\text{loss}} + \lambda_r \mathcal{L}_{\text{rec}} + \lambda_f \mathcal{L}_{\text{fid}}
$$

**Phase 1 — `L_loss`** (drives SAM2 to mislocalize on insert frames themselves):

$$
\mathcal{L}_{\text{loss}} = \frac{1}{K_{\text{ins}}} \sum_{k=1}^{K_{\text{ins}}} \Big[ \text{BCE}(g_{\text{ins}_k}, \mathbb{1}[D_{\text{ins}_k}]) + \alpha \cdot \text{softplus}(\text{CVaR}_{0.5}(g_{\text{ins}_k} \cdot \mathbb{1}[C_{\text{ins}_k}]) + m) \Big]
$$

First term: push mask to decoy region. Second term: soft CVaR (median-gated) on top half of logits inside true region, pushing them below margin m. Soft CVaR validated stable in prior loop (vs hard top-k which caused clip-specific collapse).

**Phase 2 — `L_rec`** (measured on CLEAN post-prefix eval frames while optimizing δ on prefix):

$$
\mathcal{L}_{\text{rec}} = \frac{1}{|U|} \sum_{u \in U} \Big[ \text{BCE}(g_u, \mathbb{1}[D_u]) + \alpha \cdot \text{softplus}(\text{CVaR}_{0.5}(g_u \cdot \mathbb{1}[C_u]) + m) \Big] + \beta \cdot \mathcal{L}_{\text{stale}}
$$

where `U` = eval window f15..f15+H-1 with H=7.

**The NEW piece — `L_stale`** (memory-staleness; keeps wrong memory alive):

$$
\mathcal{L}_{\text{stale}} = \frac{1}{|V|} \sum_{u \in V} \log \frac{A_u^{\text{clean-recent}}}{A_u^{\text{insert-memory}} + \epsilon}
$$

`V` = first 3 clean frames after each insert. `A_u^{·}` = fraction of memory-attention mass from foreground queries landing on each slot type. **This is the only non-trivial new loss term** — everything else in Phase 2 mirrors Phase 1 on different frames.

**Fidelity**: `L_fid = L_fid_ins + L_fid_orig` with LPIPS bound on inserts (≤ 0.10), SSIM ≥ 0.97 on attacked originals, + Lab ΔE on insert edit-mask seam band.

**Why this is the main novelty**: the specific combination — insert-only loss-induction + prefix PGD with memory-staleness regularization — is what defeats FIFO self-healing. Neither component in isolation is new; their COMPOSITION targeted at FIFO streaming memory IS new and verifiable via necessity ablations.

### Optional Supporting Component: position policy

Fixed inserts at `{f3, f7, f11}` (FIFO-resonant: period ≤ `num_maskmem - 1 = 6`, so at any eval time u ≥ 15 the bank always contains at least one insert). Decoy offset at video start = 1-of-8 direction maximizing background coverage + color-similarity to target. Original-frame schedule: f0 ε=2/255, f1..f14 ε=4/255. No learned scheduler.

### Modern Primitive Usage

- **ProPainter** (CVPR 2024, flow-guided video inpainter) used FROZEN as insert-content base generator. Role: generates a plausible `f_prev`-with-object-removed-and-relocated starting point so PGD only has to do ε-ball refinement rather than attack-from-scratch. Replaces our old Poisson seamless-clone base which had LPIPS floor ≈ 0.10.
- **RAFT / Unimatch** (frozen) for optical-flow into ProPainter.
- **LPIPS (AlexNet)** as perceptual fidelity metric.
- **NOT used**: LLMs, VLMs, video diffusion (too expensive for per-video PGD), RL-trained scheduler, learned generator — all add mass without addressing the anchor.

### Integration into Preprocessor Pipeline

Run at **publish-time** (offline, per-video). User supplies video + first-frame mask; preprocessor runs ProPainter once (seconds) to produce `K_ins=3` insert bases, then runs per-video PGD (~200 steps, ~2-6 GPU-min on single RTX Pro 6000) to refine `(ν, δ)`, then emits modified video. No SAM2 runtime interaction. Deployment binary: SAM2 surrogate checkpoint + ProPainter + preprocessor code.

### Training Plan

Per-video PGD only — no cross-video training.

1. Run clean SAM2 once → `C_u` for u=1..T, `D_u` via decoy-offset shift.
2. ProPainter forward → insert bases for 3 slots.
3. Initialize `δ_orig = 0, ν_k = 0`.
4. **Stage 1** (steps 1-40): perturb-only warmup; optimize δ with only L_rec (no inserts applied yet).
5. **Stage 2** (steps 41-80): insert-only warmup; optimize ν with only L_loss.
6. **Stage 3** (steps 81-200): joint optimization with full two-phase loss, alternating 2:1 δ:ν updates.
7. Project to fidelity constraints every step (L∞ clamp for δ, small-step LPIPS projection for ν).

### Failure Modes and Diagnostics

- **F1: FIFO self-heal still wins** (phase-2 too weak). Diagnostic: per-frame J rises back after f20. Fallback: increase β on L_stale, or extend prefix to 22 frames (still FIFO-compatible).
- **F2: insert too visible** (LPIPS > 0.10). Diagnostic: per-frame LPIPS on inserts at end of PGD. Fallback: lower ε_ins, accept lower attack strength.
- **F3: decoy direction off-scene** (object at border). Diagnostic: shifted mask falls off image. Fallback: next-best offset direction OR "suppression" mode (Phase 1 blanks out mask).
- **F4: natural-distractor clip** (same-class instance present, e.g. cows). Diagnostic: `is_natural_distractor=True` at video start via color-sim test. Fallback: disable explicit decoy; Phase 1 becomes "suppress current"; Phase 2 carries recovery-prevention burden.

### Novelty and Elegance Argument

**Closest work**:
- **UAP-SAM2** "Vanish into Thin Air" (NeurIPS 2025): universal SAM2 attack via dual semantic deviation. **Delta**: we are per-instance preprocessor for SPECIFIC target, not universal; combine insertion + prefix perturbation (UAP-SAM2 does perturbation only on all frames); formalize self-healing mechanism and our response.
- **Chen et al. WACV 2021** "Appending Adversarial Frames for Universal Video Attack": appends dummy frames for classification. **Delta**: target streaming VOS memory (not classification); inserts are targeted scene-edit in the middle (not appended dummy); coupled with prefix perturbations.
- **One-shot VOS attacks** (ACMM 2023): perturb first frame only. **Delta**: 15-frame prefix + insertion targeting streaming memory; necessity proof for each phase.
- **BadVSFM** (arXiv 2025): backdoor at training time. **Delta**: inference-time only, no training compromise.

**Elegance**: two-component system, each with one clean role. Each role justified by the bottleneck (FIFO self-healing). No other modules needed. Main technical novelty beyond the combination is `L_stale` making Phase 2 actively counter self-healing, and FIFO-resonant scheduling giving Phase 1 persistent presence in the bank.

## Claim-Driven Validation Sketch

### Claim 1 (DOMINANT): Two-phase protection defeats FIFO self-healing

- **Minimal experiment**: 4-condition ablation on DAVIS-10 hard subset
  - baseline (clean)
  - Phase-1-only (inserts, no prefix perturbation)
  - Phase-2-only (prefix perturbation, no inserts)
  - Full (both)
- **Baselines / ablations**: insert-only / perturb-only vs full
- **Metric**: mean SAM2 J-drop on eval window (f15..end), + per-frame J trajectory
- **Expected evidence**: Full drop > 0.60; Phase-1-only drop < 0.30 (bank self-heals); Phase-2-only drop < 0.30 (no loss event). Necessity confirmed if either single-phase < 50% of full.

### Claim 2 (SUPPORTING): `L_stale` is necessary for no-recovery

- **Minimal experiment**: Full vs Full-without-L_stale on DAVIS-10
- **Metric**: per-frame J trajectory; mean J at f20, f25, f30 (late eval)
- **Expected evidence**: Full keeps J monotone-down; Full-no-L_stale shows J rising back toward clean by f25+.

### Optional Claim 3 (TRANSFER): Mechanism transfers to SAM2Long

- **Minimal experiment**: SAM2Long (num_pathway=3) on attacked videos from Claim 1
- **Metric**: SAM2Long J-drop; retention = SAM2Long-drop / SAM2-drop
- **Expected evidence**: SAM2Long J-drop ≥ 0.30, retention ≥ 0.40, confirming attack targets memory architecture shared across variants.

## Experiment Handoff Inputs

- **Must-prove claims**: (1) two-phase composition is necessary; (2) `L_stale` is necessary for Phase 2; (3) transfer to SAM2Long.
- **Must-run ablations**: insert-only / perturb-only / no-L_stale / Poisson-base vs ProPainter-base.
- **Critical datasets**: DAVIS-10 hard subset (blackswan, breakdance, bmx-trees, bike-packing, camel, car-roundabout, cows, dance-twirl, dog, car-shadow). Full DAVIS-30 for final numbers.
- **Critical metrics**: J&F on eval window, per-frame J trajectory, insert/orig LPIPS, SAM2Long transfer.
- **Highest-risk assumptions**: (a) ProPainter base is good enough LPIPS floor under ε-ball PGD; (b) `L_stale` gradients through memory-attention are stable; (c) eval window long enough to show recovery-vs-no-recovery.

## Compute & Timeline Estimate

- Per-video PGD: ~6 GPU-min × 30 clips = ~3 GPU-hours full DAVIS
- ProPainter forward: ~5 sec × 3 slots × 30 clips = 7 GPU-min
- Ablation suite: 5 conditions × DAVIS-10 = ~5 GPU-hours
- SAM2Long eval: ~15 min × 30 clips = ~7 GPU-hours
- **Total paper evidence**: ≤ 1 GPU-day on single Pro 6000.
- Timeline: 2 weeks method + 1 week ablations + 1 week paper writing.
