# MemoryShield: A Two-Phase Preprocessor for Protecting Video Data from SAM2-Family Streaming Promptable Segmenters

**Status**: READY (GPT-5.4 xhigh score 9.2/10, verdict READY after 4 rounds)
**Date**: 2026-04-22
**Thread**: `019db31e-38dd-7fc1-8d72-c4dea843b254`

## Problem Anchor

- **Bottom-line problem**: Preprocessor that takes a clean video + first-frame mask of a target to protect, outputs a modified video causing any SAM2-family promptable VOS model to lose the target and not recover, within a strict visual-fidelity budget.
- **Must-solve bottleneck**: FIFO streaming memory (num_maskmem = 7) self-heals from single-frame perturbations in ≤ 6 writes. Single-component attacks fail.
- **Non-goals**: UAP / backdoor / runtime hook / maximal attack; all violate the threat model or fidelity constraint.
- **Constraints**: white-box architecture knowledge, per-video PGD, pixels-only output, insert LPIPS ≤ 0.10, SSIM ≥ 0.97 on attacked originals, DAVIS-2017 val.
- **Success condition**: eval-window J-drop ≥ 0.55 AND low rebound / post-loss AUC AND fidelity triad met AND each phase necessary (≥ 40% relative loss if removed) AND SAM2Long transfer.

## Technical Gap

SAM2-family streaming VOS (SAM2.1, SAM2Long, Hiera variants) uses a FIFO memory bank of size `num_maskmem = 7` plus a privileged f0 conditioning slot. At each new frame, memory-attention cross-attends over the bank + current image features to produce masks, then writes a new entry and evicts the oldest non-conditioning slot. This architecture **actively self-heals** from single-frame perturbations: a poisoned frame is evicted within 6 writes, and intervening clean frames write correct entries that restore tracking. Existing attacks sidestep rather than defeat self-healing: UAP-SAM2 (NeurIPS 2025) degrades every frame globally (not preprocessor-compatible for a specific target with tight LPIPS); one-shot first-frame attacks (ACMM 2023) fail when memory refreshes; BadVSFM (2025) requires training-time compromise. Preprocessor-style protection for user video data needs a compose-able mechanism: induce a loss event AND prevent recovery.

## Method Thesis

MemoryShield protects a specific target object in a user video by (i) inserting `K_ins = 3` synthetic frames at a write-aligned seed-plus-boundary schedule that together drive a tracking-loss event by populating the bank with mislocated memory entries, and (ii) perturbing prefix originals within L∞ ≤ 4/255 to suppress target re-acquisition under a memory-staleness regularizer that keeps bank-attention mass on inserted-slot residuals. Each phase is individually insufficient; their composition is the minimum counter to FIFO self-healing.

## Contribution Focus

- **Dominant contribution**: a two-phase preprocessor mechanism against SAM2-family VOS. Phase 1 creates the loss event via inserts; Phase 2 prevents re-acquisition via prefix perturbations + a memory-staleness regularizer `L_stale` on bank-attention mass.
- **Supporting contribution**: a write-aligned seed-plus-boundary insert schedule parameterized by `num_maskmem`, tested against off-resonance and offset-swept controls at matched recency.
- **Non-contributions**: no new SAM2 architecture; no new generator (ProPainter frozen); no UAP / backdoor / runtime-hook; no learned scheduler; no cross-family claim.

## Complexity Budget

- **Frozen / reused**: SAM2.1 Hiera-Tiny surrogate (+ SAM2Long for transfer eval), ProPainter video inpainter, RAFT optical flow, LPIPS (AlexNet).
- **New trainable components** (2):
  1. Per-video `δ_orig` L∞-bounded pixel perturbation on f0..f14 (ε_f0 = 2/255, f1..f14 = 4/255)
  2. Per-video `ν_k` pixel perturbation on INSERT paste region + 5 px seam band only (augmented-Lagrangian LPIPS penalty)
- **Intentionally excluded**: learned scheduler, learned generator, teacher model, bilevel virtual-state control, LLM / VLM / diffusion / RL.

## Three Clocks (schedule formalization)

- **Clock O** — original-frame index `o ∈ {0, …, T-1}` (clean input)
- **Clock M** — modified-sequence index (output video after insertions)
- **Clock W** — memory-write count (monotone count of non-cond memory writes; f0 is write 0, first FIFO write is w = 1)

**Insert schedule on Clock W** — *write-aligned seed-plus-boundary schedule*:

    w_k = (num_maskmem - 1) · k   for k = 1, …, K_ins - 1
    w_{K_ins} = W_total - 1        (force final insert adjacent to eval start)

For `num_maskmem = 7`, `K_ins = 3`, `T_prefix = 15`: canonical `m = {6, 12, 14}` → inserts after original frames {5, 10, 11}.

## Proposed Method

### System Overview

```
Clean video x_0:T + first-frame mask m_0
  │
  ├── (one-time) clean SAM2 forward → C_u per clock-O frame (flow-warped + erode-2)
  ├── (one-time) decoy-offset selection at video start → D_u
  ├── (one-time) ProPainter × K_ins → insert bases b_k
  │
  ├── Insert positions on Clock W: write-aligned seed-plus-boundary (above)
  │
  ├── PGD Stage 1 (1-40): ν-only, L_loss — creates loss event first
  ├── PGD Stage 2 (41-80): δ-only, L_rec — learns to preserve loss (inserts frozen)
  ├── PGD Stage 3 (81-200): joint 2:1 δ:ν, full L — refines composition
  │
  └── Output: modified video = attacked originals + 3 optimized inserts
```

### Loss (per-video PGD objective)

$$ \mathcal{L}(\nu, \delta) = \mathcal{L}_{\text{loss}} + \lambda_r \mathcal{L}_{\text{rec}} + \lambda_f \mathcal{L}_{\text{fid}} $$

**Phase 1 — L_loss (inserts only, ROI-restricted)**:

$$ \mathcal{L}_{\text{loss}} = \frac{1}{K_{\text{ins}}} \sum_k \Big[ \text{BCE}_{\text{ROI}}(g_{\text{ins}_k}, \mathbb{1}[D_{\text{ins}_k}]) + \alpha \cdot \text{softplus}\big( \text{CVaR}_{0.5}(\{g_{\text{ins}_k}(p) : p \in C_{\text{ins}_k}\}) + m \big) \Big] $$

- ROI = `D_box ∪ C_box` dilated 10 px (no background supervision mass).
- Masked CVaR over the SET of logits inside true-object region; no zero-contamination.
- `g_u` is a logit map (pre-sigmoid) throughout.

**Phase 2 — L_rec (clean post-prefix eval frames u ∈ U = f15..f21)**:

$$ \mathcal{L}_{\text{rec}} = \frac{1}{|U|} \sum_{u \in U} \Big[ \alpha_{\text{supp}} \cdot \text{CVaR}_{0.5}(\{g_u(p) : p \in C_u\})^+ + \alpha_{\text{conf}} \cdot \text{softplus}\big(\text{logmeanexp}(g_u) - \tau_{\text{conf}}\big) \Big] + \beta \cdot \mathcal{L}_{\text{stale}} $$

- **Suppress term**: masked CVaR_0.5 on true region, positive part → push top-half logits below 0 (no hallucination demand).
- **Low-confidence lock**: `logmeanexp(g_u) = logsumexp(g_u) − log(HW)` is resolution-invariant; penalizes high global logit mass.
- **`L_stale`**: Phase 2's internal memory-hijack regularizer.

**L_stale (3-bin categorical KL)**:

$$ \mathcal{L}_{\text{stale}} = \frac{1}{|V|} \sum_{u \in V} \text{KL}(Q \| P_u), \quad P_u = [A_u^{\text{ins}}, A_u^{\text{recent}}, A_u^{\text{other}}], \quad Q = [0.6, 0.2, 0.2] $$

- `V` = first 3 clean post-last-insert frames (covers self-heal window).
- `A_u^{ins}`: attention mass from foreground queries on bank slots sourced from inserted frames `{m_k}`.
- `A_u^{recent}`: attention mass on **currently resident** clean-prefix memory slots (not all historical prefix).
- `A_u^{other}`: conditioning slot + image-feature fallback.
- Foreground queries = pixels inside `erode(flow_warp(C_u), 2)`.
- Attention averaged across all heads in SAM2's FINAL memory-attention block.
- Q rationale: insert > recent by margin 0.4, with non-zero "other" preventing collapse of all attention to a single slot type.

**Fallback form** (if training noisy): `L_stale^margin = softplus(γ + A^recent − A^ins) + λ · A^other`.

**L_fid (augmented Lagrangian)**:

$$ \mathcal{L}_{\text{fid}} = \mu_\nu \cdot (\text{LPIPS}(x_{\text{ins}_k}, f_{\text{prev}_k}) - 0.10)^+ + \mu_s \cdot \Delta E_{\text{seam}} $$

- δ hard-L∞-clamped every step (not in L_fid).
- μ increases exponentially when budget exceeded.
- SSIM reported as metric only (not optimized).

### Training Plan

1. Clean SAM2 forward → `C_u` per clock-O frame.
2. Decoy-offset selection at video start (1-of-8 direction maximizing bg-coverage + color-sim to target).
3. ProPainter × K_ins → insert bases `b_k`.
4. **Stage 1 (steps 1-40)**: `ν`-only with `L_loss` — creates loss event.
5. **Stage 2 (steps 41-80)**: `δ`-only with `L_rec` (inserts frozen) — learns to preserve loss.
6. **Stage 3 (steps 81-200)**: joint 2:1 δ:ν with full `L` — refines composition.
7. δ L∞-clamp every step; ν LPIPS penalty via augmented Lagrangian.
8. Cache clean-suffix image embeddings (pixels never change; only attention path through bank changes).

### Pipeline Integration

Run offline at publish-time. User provides video + m_0 → preprocessor emits modified video. DAVIS 480p, SAM2.1-Tiny, batch 1, ~3-8 GPU-min per video on single RTX Pro 6000. No SAM2 runtime interaction.

### Failure Modes

- **F1** (Phase-2 fails to suppress): post-loss AUC high → raise β or extend prefix to 22 frames.
- **F2** (insert LPIPS > 0.10): tighter μ_ν, accept lower attack.
- **F3** (decoy direction off-scene): next-best offset.
- **F4** (`L_stale` gradient unstable): swap to margin fallback form.

### Novelty and Elegance

**Closest prior**: UAP-SAM2 (universal, all-frame perturbation — not per-instance preprocessor); Chen 2021 WACV (appended dummy frames for classification — not VOS memory); ACMM 2023 one-shot (first-frame only — defeated by refresh); BadVSFM (training-time backdoor — different threat model).

**Elegance**: two components, one role each (Phase 1 creates the loss, Phase 2 prevents recovery). All other pieces frozen/reused. `L_stale` is Phase 2's INTERNAL regularizer that makes recovery-prevention work — not a third contribution. The dominant mechanism claim is *insert + perturb composition defeats FIFO self-healing*, proved by a 4-condition ablation with each single-component trivially recoverable.

## Claim-Driven Validation

### Claim 1 (DOMINANT): Two-phase composition defeats FIFO self-healing
- 4 conditions on DAVIS-10 hard subset: clean / Phase-1-only / Phase-2-only / Full.
- Metric: mean J-drop on **full suffix f15..end**, per-frame J trajectory, rebound, post-loss AUC.
- Expected: Full J-drop ≥ 0.55; each single-phase ≤ 50% of Full; Full rebound ≤ 0.15; single-phase rebound ≥ 0.35.

### Claim 2 (SUPPORTING): `L_stale` is necessary for no-recovery
- Full vs Full-no-L_stale on DAVIS-10.
- Metric: rebound, post-loss AUC, **P_u breakdown at f16, f17, f18**.
- Expected: Full keeps `A^ins ≥ 0.5`; no-L_stale collapses `A^ins ≤ 0.2`; rebound gap ≥ 0.2.

### Claim 3 (MANDATORY): Transfer to SAM2Long
- SAM2Long (num_pathway = 3) on attacked videos from Claim 1 Full.
- Metric: SAM2Long J-drop, retention = SAM2Long-drop / SAM2-drop.
- Expected: SAM2Long J-drop ≥ 0.25; retention ≥ 0.40.

### Claim 4 (MECHANISM): Write-aligned seed-plus-boundary schedule
- **4a**: resonance `m = {6, 12, 14}` vs off-resonance `m = {4, 8, 14}` (matched last-insert m=14 → matched recency; only early-insert spacing differs).
- **4b**: offset sweep `m ∈ {5,11,14}, {6,12,14}, {7,13,14}` — peak at FIFO-period alignment.
- Expected: canonical beats off-resonance by ≥ 20pp J-drop; sweep peaks at (6,12,14).

### (Optional) Claim 4c: Q sensitivity
- Test `Q ∈ {[0.5, 0.25, 0.25], [0.6, 0.2, 0.2], [0.7, 0.15, 0.15]}` on 3 clips; report stability.

## Experiment Handoff

- **Must-prove claims**: 1–4 above.
- **Must-run ablations**: insert-only / perturb-only / no-L_stale / off-resonance / offset sweep / ProPainter-vs-Poisson base.
- **Critical datasets**: DAVIS-10 hard subset (blackswan, breakdance, bmx-trees, bike-packing, camel, car-roundabout, cows, dance-twirl, dog, car-shadow). DAVIS-30 for final numbers.
- **Critical metrics**: mean J-drop (full suffix), post-loss AUC, rebound, per-frame J trajectory, insert LPIPS, orig LPIPS/SSIM, SAM2Long J-drop + retention, P_u bank-attention breakdown.
- **Highest-risk assumptions**: (a) L_stale gradient stability through memory-attention; (b) ProPainter base compatible with ν LPIPS ≤ 0.10; (c) matched-recency off-resonance comparison is a fair isolation test.
- **Pre-registration note**: `τ_conf`, `β`, `Q` are declared in a small sensitivity range BEFORE full experiments, to rule out reviewer concern of heavy tuning.

## Compute & Timeline

- Per-video PGD: ~3-8 GPU-min × 30 clips ≈ 3-4 GPU-hours
- ProPainter forward total: ~7 GPU-min
- Ablation suite: 7 conditions × DAVIS-10 = ~6 GPU-hours
- SAM2Long eval: ~7 GPU-hours
- **Total**: ~24 GPU-hours ≈ 1 GPU-day on single RTX Pro 6000 Blackwell
- Timeline: 2 weeks method + 1 week ablations + 1 week writing = 4 weeks total
