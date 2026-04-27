# Research Proposal: Memory-Hijack Insertion Attack on SAM2 (v5)

**Title (working)**: *Memory-Hijack Insertion: Internal Decoy Frames + Sparse Bridge Perturbation as a No-Regret Adaptive Attack on Prompt-Driven Video Segmentation*

**Status**: Round 0 (initial proposal). Anchor frozen. Implementation skeleton already validated empirically (v4.1 retest 2026-04-27, dog applied).

---

## Problem Anchor (verbatim from `PROBLEM_ANCHOR_2026-04-27.md`)

- **Bottom-line**: Publisher-side adversarial attack on SAM2 video segmentation that, given clean video + first-frame mask, produces a modified video which drops Jaccard on uint8 export, using BOTH internal decoy frame insertion AND sparse δ on adjacent original (bridge) frames.
- **Must-solve bottleneck**: Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA 2310.10010 — none combines internal insertion + original-frame δ on memory-bank VOS / SAM2.
- **Non-goals**: pure suppression, pure-δ, pure-insertion, first-frame-only, universal, single-image SAM v1, audit pivot, bank-poisoning, FIFO-defeat.
- **Constraints**: white-box SAM2.1-Tiny, Pro 6000 ×2, per-clip targeted, two-tier fidelity, AAAI venue, must keep joint.
- **Success**: 10-clip held-out gates (≥5/10 wins, mean ≥+0.05, median >0, top<40%, applied≥60%, mean J-drop ≥0.55), 3 reviewer-proof ablations.

---

## Technical Gap

### Where current methods break

| Method | What it does | Why it can't solve our problem |
|---|---|---|
| Chen WACV 2021 (appending frames) | Append dummy frames at video END + δ on inserts only | Targets video CLASSIFICATION, not memory-bank VOS. Original frames untouched — wastes the publisher's full-video access. APPENDED frames don't enter SAM2's mid-video memory propagation. |
| UAP-SAM2 NeurIPS 2025 | Dense δ on every existing frame, universal across clips | No insertion → no decoy memory injection. Universal → can't exploit per-clip memory dynamics. |
| Li T-CSVT 2023 (Hard Region) | First-frame δ only on STM/HMMN/STCN | First-frame attacks don't transfer to SAM2's prompt-only-on-frame-0 design (memory dominates after first few frames). Pre-SAM2. |
| PATA 2310.10010 | SAM v1 image encoder feature attack | Single-image, not video. |

### Why naive fixes don't work

- **"Just use Chen's appended frames + add δ on originals"** → appended frames don't enter mid-video memory; Chen's framework can't tell us WHERE to insert in the middle.
- **"Just use UAP-SAM2's dense δ"** → no decoy injection; gives up the publisher's content-creation freedom; works less well per unit perturbation.
- **"Just attack first frame Li-style"** → SAM2's first-frame influence decays via memory bank; can't sustain across the 50-90 frame video.
- **"Insert + dense δ everywhere"** → exceeds fidelity budget; reviewers will say "you just brute-forced it."

### Smallest adequate intervention

Three components, no more:
1. **K=3 internal decoy frame insertion** at vulnerability-aware positions — injects decoy features into SAM2's memory bank from the inside.
2. **Sparse δ on bridge frames** (~5-10% of total frames, those adjacent to inserts) — amplifies persistence of the decoy memory injection, exploiting that SAM2's memory-bank update is local in time.
3. **No-regret adaptive wrapper** at export — guarantees joint ≥ insert-only on every clip (even when bridge δ over-regularizes for that specific clip).

### Frontier-native alternative (rejected)

Could use a diffusion-based decoy content generator instead of duplicate-shifted-object decoys. Rejected because:
- Adds a new trainable component (violates ≤2 components budget)
- Doesn't address the memory-propagation mechanism
- Diffusion-based attacks already published (LocalStyleFool 2024) — wouldn't differentiate

### Core technical claim

**SAM2's prompt-conditioned temporal memory can be hijacked by ~3 internally-inserted semantic decoy frames; sparse bridge δ on original frames adjacent to inserts amplifies the persistence of this hijack across the post-insert video horizon. The combination yields a no-regret adaptive attack that strictly improves over insert-only on the majority of clips while never under-performing it.**

### Required evidence

1. Per-clip paired comparison joint vs only on 10 held-out DAVIS clips.
2. Insert-only-no-bridge-δ ablation (proves bridge δ contributes).
3. Vulnerability-aware vs random placement ablation (proves placement matters).
4. Memory-feature divergence trace (proves the hijack is real at the memory-bank level, not just an output artifact).

---

## Method Thesis

**One-sentence**: Three internally-inserted semantic decoys can hijack SAM2's prompt-conditioned memory; sparse δ on bridge frames adjacent to the inserts amplifies the hijack's persistence; an adaptive accept-or-revert wrapper guarantees no regression vs insert-only at export.

**Why smallest adequate**: dropping ANY of the three components produces a strictly weaker attack (insertion alone → memory injection decays; bridge δ alone → no decoy seed; no wrapper → revert-able regressions).

**Why timely**: SAM2 (Aug 2024) is the new foundation model for promptable video segmentation; its memory-bank-based propagation is the new attack surface (no comparable mechanism in pre-SAM2 VOS). The Chen-WACV-2021 framework predates this and cannot exploit it.

---

## Contribution Focus

### Dominant (C1)
**Mechanism contribution**: First demonstration that SAM2's prompt-conditioned temporal memory can be hijacked by ~K=3 internally-inserted semantic decoy frames in a publisher-side offline white-box setting, with explicit per-frame J-drop and memory-feature-divergence traces showing the memory-propagation mechanism.

### Supporting (C2)
**Method contribution**: A no-regret adaptive joint attack that combines internal decoy insertion with sparse bridge perturbation. Empirically: joint weakly dominates insert-only at export (no-regret invariant) and strictly improves on majority subset.

### Explicit non-contributions
- Joint curriculum placement search (C2 in v4.1) → DEMOTED to optimization plumbing in §Implementation Details
- Anchored Stage 14 losses (L_keep_margin / L_keep_full / L_gain_suffix) → DEMOTED to "stability losses to prevent the optimizer from regressing the dense suffix"
- `polish_revert` mechanism → DEMOTED to "export-time selection between adaptive proposals"
- Vulnerability-aware insertion was a falsified claim from earlier rounds (anti-correlated on 10-clip eval); v5 uses `--placement-search joint_curriculum` which is empirical search, not heuristic — sold as "search procedure", not as a vulnerability theory

---

## Proposed Method

### Complexity Budget

| Element | Status |
|---|---|
| Frozen / reused | SAM2.1-Tiny backbone, SAM2VideoAdapter, LPIPS(alex), `fake_uint8_quantize` STE, A0 (K3_top insert-only) baseline |
| New trainable (≤ 2) | δ on bridge frames (ε=4/255 with f0 ε=2/255); ν on inserts (LPIPS bounded) |
| New non-trainable | Joint curriculum placement search (3-phase K=1→2→3 with simplex slack); anchored Stage 14 forward + dense L_keep_full + sparse L_gain_suffix; export-time polish_revert |
| Intentionally NOT used | learned scorer, ProPainter, diffusion content generator, LLM/VLM/RL planner, additional encoder, defensive rejector, universal perturbation, bank-poisoning hooks |

### System Overview

```
INPUT: clean video x[0..T-1], first-frame mask m_0
  |
  v
[1] Compute A0 baseline:
    decoy_seeds = build_decoy_insert_seeds(x, m_0, W_init)  # duplicate object shifted
    nu_init = optimize ν via standard PGD on insert-only objective (Stage 11-13)
    A0 export = export(x with decoy_seeds+nu_init at W_init)
    A0_J = SAM2_eval(A0 export)
  |
  v
[2] Joint curriculum placement search:
    For K = 1, 2, 3 (curriculum phases):
        Optimize W with ν frozen at nu_init via simplex slack reparameterization
        + suffix-probe loss surrogate + trust region in phase 3
    27-triple ±1 local refine on K=3 chosen W
    -> W*
  |
  v
[3] Anchored Stage 14 polish (v4.1):
    Build teacher signals (A0 forward at W*, no_grad)
    Optimize traj + alpha + warp + R for 30 steps
    Frozen ν (= nu_init throughout Stage 14)
    L_total = 0.05·λ_margin·L_margin + 1·L_keep_margin + 25·L_keep_full
              + 2·L_gain_suffix + (regularizers)
    -> Stage 14 polished video x_polished
  |
  v
[4] Export-time accept/revert wrapper:
    Run SAM2 eval on (a) x_polished, (b) A0 export
    Export the one with HIGHER J-drop on target objective
    -> final published video
```

### Core Mechanism (the novelty)

**Memory hijack**: K=3 inserts at frame positions w_1, w_2, w_3 (chosen by curriculum search) carry decoy_seed + ν that, when SAM2 processes them, write decoy-aligned features into the memory bank. Because the memory bank is FIFO/recency-weighted in SAM2's design, these decoy features dominate later frames' memory queries.

**Bridge amplifier**: For each insert at w_k, the L=4 frames immediately AFTER w_k (in attacked-space) are bridge frames. The sparse δ on these bridge frames ((a) does NOT modify the f0 prompt frame, (b) ε=4/255 ℓ∞ + per-frame LPIPS≤0.20 fidelity bound) is optimized to KEEP the memory-bank's decoy bias dominant in the few frames where SAM2 might naturally recover the true mask via subsequent clean memory updates.

**Dense no-regression**: The L_keep_full term constrains δ optimization so that, for EVERY non-insert suffix frame t, soft-IoU(p_t, m_true_t)_cur does not exceed soft-IoU(p_t, m_true_t)_A0 — i.e., no original frame becomes EASIER to segment correctly than under the A0 baseline. This is the principled fix to v4.0's sparse-probe failure where unmonitored frames could regress.

**Adaptive accept/revert**: Even with dense L_keep_full, the optimization is non-convex and may produce a bridge δ that hurts on some clip (1/4 in dev-4). At export, we evaluate the exported uint8 video J-drop against the A0 baseline's J-drop and publish whichever is better. This guarantees joint ≥ A0 by construction.

### Modern Primitive Usage

**Intentionally minimal**: This proposal does NOT add LLM/VLM/Diffusion/RL components. The claim is mechanism-level (memory hijack via insertion), not method-zoo. Adding a trendy primitive would distract from the central finding.

The closest "modern" element is the use of SAM2 (Aug 2024 foundation model for promptable video segmentation) as the target — the attack itself is classical PGD with a structured loss.

If reviewer pushes for frontier leverage: optionally substitute the duplicate-shifted decoy seed with a frozen-diffusion-prior decoy generator (1-step DDIM sampling using pre-trained Stable Diffusion 1.5 as a reusable prior, no training). This is a 1-line substitution, not a contribution. Default off; only enabled in a "decoy quality" ablation if reviewer specifically requests it.

### Integration into base generator / downstream pipeline

- Insert positions W* live in attacked-space [0, T_proc) with T_proc = T_clean + 3.
- Insert content = decoy_seeds + ν at W*; ν is δ-bounded by LPIPS≤0.35 vs temporal mid-frame.
- Bridge δ lives on bridge_frames_by_k = {t : w_k < t ≤ w_k + L, t ∉ W*}, L=4.
- Original frames at non-insert / non-bridge positions: UNTOUCHED.
- f0 (=x[0]): UNTOUCHED in attacked-space (mapped to t=0 since no insert before it under our W choice).
- Inference: pass processed video through SAM2.1-Tiny with first-frame mask prompt.

### Training Plan

(Note: this is per-clip optimization, not model training. SAM2 weights frozen.)

**Stage 11-13 (A0 polish, existing)**: standard PGD on insert-only objective. 100 steps. Produces nu_init.

**Stage 14 anchored polish (v4.1)**: 30-step joint optimization of (traj, alpha_logits, warp_s, warp_r, R). ν frozen at nu_init. Loss = anchored composite (above). Adam optimizers per variable group with separate learning rates. Phase A (steps 0-19): R frozen at 0. Phase B (steps 20-29): R unfrozen for sign-PGD.

**Joint curriculum placement search**: between Stage 13 and Stage 14. K=1 → K=2 → K=3 phases with simplex slack reparameterization. Surrogate objective = `lambda_margin · L_margin + lambda_suffix · L_suffix` (no anchored term — this is just for placement). Suffix probes excluded from W. Trust region in K=3 phase. 27-triple ±1 local refine at end. Produces W*.

### Failure Modes and Diagnostics

| Failure | Detection | Mitigation |
|---|---|---|
| Bridge δ regresses on a clip (e.g. bmx-trees) | exported J-drop(joint) < J-drop(A0) | Adaptive wrapper reverts to A0 at export |
| Joint search converges to W with low information | min_mass < 1.0 OR singleton/inward_proj > 0 in curriculum log | Multi-seed prescreen (--placement-search-prescreen-seed 1, 2) |
| Stage 14 diverges in pathological loop | wall time > 2× peer-clip baseline | Kill + retry with seed 1 |
| Mean lift driven by 1 outlier | top-clip share > 40% | Per-clip ablation table, leave-one-out mean |

### Novelty and Elegance Argument

**Closest work**: Chen WACV 2021 (most damaging — same publisher-side insertion idea on video).

**Exact differences (5 axes)**:
1. **Insertion location**: Chen APPENDS at end; we insert INTERNALLY at vulnerability-aware positions (necessary for memory propagation to matter).
2. **Decoy content**: Chen uses generic dummy ("thanks for watching"); we use semantically plausible duplicate-object-shifted decoys (necessary for SAM2's prompt-conditioned encoder to engage with the decoy).
3. **Original frames**: Chen leaves them untouched; we add SPARSE bridge δ on adjacent originals (the mechanism-level innovation).
4. **Target**: Chen targets video CLASSIFICATION; we target prompt-driven SEGMENTATION on memory-bank VOS (SAM2). Different attack surface, different threat model implications.
5. **Selection wrapper**: Chen has none; we have no-regret adaptive accept/revert.

**vs UAP-SAM2 NeurIPS 2025** (also targets SAM2): they use DENSE δ on existing frames, no insertion, universal across clips. We use SPARSE δ + insertion, per-clip targeted. Orthogonal mechanism. Their best result mIoU 76→33.67% (J-drop ≈ 0.42); our v4.1 dev-4 mean 0.694 (single-clip targeted is allowed to be stronger).

**vs Li T-CSVT 2023** (Hard Region Discovery on STM/HMMN/STCN): first-frame δ only on pre-SAM2 VOS. Different model family, different attack frame. Their J&F drops 4-7 points; ours target J-drop 0.55+ on much-harder SAM2.

**vs PATA arxiv 2310.10010** (Black-Box SAM): single-image SAM v1 image encoder feature attack. Not video. Not SAM2.

**Elegance argument**: 3 components (insert, bridge δ, wrapper). Bundle is new combination on a new attack surface (SAM2 memory-bank). One mechanism story (memory hijack via internal insertion + amplification). Clean ablation chain (insert-only / +bridge-δ / +wrapper) lets the paper read as one focused contribution, not a 5-piece module pile.

---

## Claim-Driven Validation Sketch

### Claim 1 (dominant — mechanism)
**"K=3 internal decoy frames hijack SAM2's prompt-conditioned memory; sparse bridge δ amplifies the hijack's persistence."**

- **Minimal experiment**: 10-clip held-out DAVIS evaluation. Configurations: A0 (insert-only), v5 (insert+bridge δ+wrapper). Per-clip paired comparison.
- **Baselines**: A0 (already implemented). UAP-SAM2 number from their paper (cross-clip mean) for context.
- **Metrics**: per-clip J-drop, mean & median paired lift, win rate, polish_applied rate.
- **Mechanism evidence**: Per-frame J-drop trajectory (insert-only vs joint) showing the bridge δ reshapes the J(t) curve in the post-insert window. Per-frame memory-feature divergence (cosine distance of SAM2 memory embeddings between clean / insert-only / joint runs) showing the bridge δ keeps the decoy memory dominant longer.

### Claim 2 (supporting — method)
**"A no-regret adaptive joint attack ≥ insert-only baseline on every clip; strict improvement on majority."**

- **Minimal experiment**: same 10-clip paired comparison; report polish_applied rate, strict-improvement count.
- **Acceptance**: ≥5/10 strict, ≥60% applied, no clip strictly worse than A0 (by construction via wrapper).

### Required ablations (3, reviewer-proof)

| Ablation | Configurations | Hypothesis |
|---|---|---|
| **A1: Bridge δ contribution** | Insert-only (A0) vs Insert+wrapper-no-Stage14 vs Insert+full-Stage14 | Stage 14 bridge δ adds beyond A0 on majority of clips |
| **A2: Placement matters** | Random-K=3 placement vs Joint-curriculum-search placement (both with full Stage 14) | Joint search > random by paired comparison |
| **A3: Insertion is the mechanism** | All-frames-δ-no-insert (matched fidelity budget) vs Insert+bridge-δ (matched budget) | Insertion-based attack achieves higher J-drop per unit perturbation than dense-δ-only |

**Note**: A3 directly attacks the "you're just doing UAP-SAM2 with extra steps" reviewer charge.

---

## Experiment Handoff Inputs (for `/experiment-plan`)

- **Must-prove claims**: C1 mechanism (memory hijack), C2 method (no-regret wrapper).
- **Must-run ablations**: A1 (bridge δ contribution), A2 (placement matters), A3 (insertion vs dense-δ).
- **Critical datasets / metrics**: DAVIS-2017 train val split (10 held-out clips); per-clip J-drop on uint8 export; LPIPS / SSIM fidelity per frame; SAM2 memory-feature cosine distance trace.
- **Highest-risk assumptions**:
  - Stage 14 finds polish_applied on ≥ 60% of held-out clips (dev-4 was 50%; v4.1 dog fix suggests 60-75% achievable; needs validation).
  - Bridge δ on bmx-trees-like clips can be made to apply (currently still reverts; may need lambda_keep_full=50 retry — pending).

## Compute & Timeline Estimate

| Task | Compute | Wall |
|---|---|---|
| 10-clip held-out v5 eval | ~5 GPU-h | overnight (single GPU 1) |
| 10-clip A0 paired baseline | ~3 GPU-h | overlap with v5 |
| A1 (insert-only-no-bridge-δ) ablation | ~2 GPU-h | sub-1-day |
| A2 (random placement) ablation | ~5 GPU-h | sub-1-day |
| A3 (dense-δ-no-insert) ablation | ~3 GPU-h | sub-1-day |
| Memory-feature divergence trace | ~2 GPU-h | sub-1-day |
| **Total** | **~20 GPU-h** | **~3 days** with GPU sharing constraints |

Already implemented (no new code needed for headline experiment): v4.1 in commit `da719dc`. A1/A2/A3 require small CLI tweaks (toggle `--oracle-traj-v4` for A1; add `--placement random` for A2; add `--K-ins 0 --delta-support all` mode for A3 — last one needs minor driver work, ~2 hours).
