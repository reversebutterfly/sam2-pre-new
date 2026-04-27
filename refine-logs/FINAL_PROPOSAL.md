# Research Proposal: Memory-Mediated Failure of Prompt-Driven Video Segmentation: Causal Evidence from Internal Decoy Insertion with Sparse Bridge Perturbation (v5 — final)

**Status**: Proposal-stage ceiling 8.4/10 (codex R6 thread `019dcd87-c42b-7b03-9139-34df6b6ebd89`, 2026-04-27).
**Implementation**: v4.1 commit `da719dc` on Pro 6000.
**Frozen anchor**: `PROBLEM_ANCHOR_2026-04-27.md`.

---

## Problem Anchor (verbatim)

- **Bottom-line**: Publisher-side adversarial attack on SAM2 video segmentation — clean video + first-frame mask → modified video that drops Jaccard on uint8 export, using BOTH internal decoy frame insertion AND sparse δ on adjacent original (bridge) frames.
- **Must-solve bottleneck**: Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA 2310.10010 — none combines internal insertion + original-frame δ on memory-bank VOS / SAM2.
- **Non-goals**: pure suppression, pure-δ, pure-insertion, first-frame-only, universal, single-image SAM v1, audit pivot, bank-poisoning, FIFO-defeat.
- **Constraints**: white-box SAM2.1-Tiny on Pro 6000 ×2; per-clip targeted DAVIS; two-tier fidelity. AAAI venue. Must keep BOTH insertion AND original-frame δ active.
- **Success**: 10-clip held-out gates ≥5/10 wins, mean ≥+0.05, median >0, top<40%, applied≥60%, mean J-drop ≥0.55. 3 reviewer-proof ablations.

---

## Title (CONDITIONAL on A3 outcome — pre-registered)

| A3 outcome | Title | Thesis |
|---|---|---|
| Strong pass | *Memory-Mediated Failure of Prompt-Driven Video Segmentation: Causal Evidence from Internal Decoy Insertion with Sparse Bridge Perturbation* | "memory propagation is DOMINANT failure mode" |
| Partial pass | *Memory-Mediated Persistence in Adversarial Attacks on Prompt-Driven Video Segmentation* | "memory propagation is a SUBSTANTIAL component" |
| Fail | (workshop pivot) *Engineered Insertion + Sparse Perturbation Attack on SAM2* | empirical effectiveness only |

---

## Method Thesis

Three internally inserted semantic decoys at empirically-searched positions provide causal evidence (via memory-write blocking with matched non-insert negative control) that SAM2's prompt-conditioned memory propagation is the (dominant / substantial — A3-conditional) failure mode under publisher-side adversarial attack, and sparse δ on the L=4 bridge frames immediately following each insert measurably extends the memory divergence beyond the insert-only baseline.

---

## Contribution Focus

- **C1 (main, mechanism, paired)**:
  - **C1.a** *(causal, insert-position-specific)*: J-drop COLLAPSES under blocking of memory writes at insert positions, by a margin SUBSTANTIALLY larger than under matched non-insert control blocking. Pre-registered:
    - Strong pass: collapse_attacked ≥0.20 abs AND (collapse_attacked − collapse_control) ≥0.10 abs, both on ≥7/10 clips.
    - Partial pass: collapse_attacked ≥0.10 abs AND (collapse_attacked − collapse_control) ≥0.05 abs, both on ≥6/10 clips.
    - Fail: either threshold misses on majority → workshop pivot.
  - **C1.b** *(persistence)*: bridge δ extends d_mem(t) above insert-only on ≥75% of held-out clips — measurable as integral of (d_mem_full − d_mem_only) over (w_K, w_K + L) > 0.
- **E1 (enabling, openly necessary engineering)**: vulnerability-aware joint curriculum placement search. Honestly framed as engineering necessity (vulnerability heuristic 3-signal scorer was empirically falsified — anti-correlated 0.488 ranked vs 0.534 random on 10 clips).
- **E2 (enabling)**: dense no-regression stabilization L_keep_full. Without it, optimization regresses unmonitored frames (v4.0 50% revert evidence).
- **Deployment policy (separately reported, NOT a contribution)**: export-time `polish_revert` selector — publishes max(joint, A0). Reported in deployment column only.
- **Explicit non-contributions**: no learned scorer, no diffusion gen, no LLM/VLM/RL planner, no UAP, no bank-poisoning, no first-frame-only, no defensive rejector.

---

## Complexity Budget

| Element | Status |
|---|---|
| Frozen / reused | SAM2.1-Tiny, SAM2VideoAdapter, LPIPS(alex), STE quantize, A0 (K3_top insert-only) baseline, deterministic decoy alpha-paste compositor |
| New trainable (≤2) | δ on bridge frames (ε=4/255 with f0 ε=2/255, per-frame LPIPS≤0.20); ν on inserts (LPIPS≤0.35 vs neighbors) |
| New non-trainable | Joint curriculum placement search (3-phase K=1→2→3 + simplex slack); anchored Stage 14 forward + dense L_keep_full + sparse L_gain_suffix; export wrapper (deployment-only) |
| Diagnostic only (no training signal) | `BlockInsertMemoryWritesHook` (R4 spec); memory-readout extractor (R2 protocol); negative-control sampler |
| Intentionally NOT used | learned scorer, ProPainter, diffusion content gen, LLM/VLM/RL planner, additional encoder, defensive rejector, universal perturbation, bank-poisoning hooks |

---

## Core Mechanism (mathematically locked)

### Decoy family — ONE explicit deterministic choice

For each insert position w_k with corresponding clean-space `c_k = w_k − k`:

```
decoy_seed[k] = alpha_paste(
    x[c_k],
    shifted_object(m_true_at_c_k, dy_k, dx_k),
    feather_radius=5, feather_sigma=2.0
)
```

where `(dy_k, dx_k) = compute_decoy_offset_from_mask(m_true_at_c_k)` — deterministic centroid-based shift to non-overlapping location. NO learned content.

### Optimized variables

- `ν[k] ∈ R^{H × W × 3}` per insert k. Bounds: per-insert LPIPS(decoy_seed[k] + ν[k], decoy_seed[k]) ≤ 0.35 + per-insert TV ≤ 1.2× base.
- `δ[t] ∈ R^{H × W × 3}` per bridge frame t. Bounds: ε_∞ = 4/255 (or 2/255 if t==0) AND per-frame LPIPS(x_clean[t] + δ[t], x_clean[t]) ≤ 0.20.
- δ parameterized as **masked bridge-edit composite**: learned alpha-paste of duplicate-object decoy onto each bridge frame, with small spatial warp and masked residual confined to decoy support. Full equations: Appendix B.

### Loss (frozen ν, optimize δ + bridge-edit composite)

```
L_total = 0.05 · λ_margin · L_margin            (local surrogate, attenuated)
        + 1.0   · L_keep_margin                  (no-regression on attacked window)
        + 25.0  · L_keep_full                    (DENSE no-regression on suffix — v4.1 hot-fix)
        + 2.0   · L_gain_suffix                  (sparse gain, 6 evenly-spaced probes)
        + (regularizers: alpha, warp, residual_TV, traj smoothness, fidelity hinges)
```

with `L_keep_full = mean_{t ∈ keep_suffix_frames} relu(u_cur(t) − u_A0(t))` over ALL non-insert attacked-space frames after w_first.

### Memory readout d_mem(t) protocol (R2 + post-impl refinement 2026-04-27)

**Refinement after codex M0 review**: the original R2 wording said "T_obj from clean run, FROZEN per clip, reused across conditions". On implementation we found that bank composition differs across conditions (Nk varies because attacked runs have K extra inserts contributing to memory). Literally re-using clean indices in attacked conditions is undefined when Nk_attacked ≠ Nk_clean. The honest interpretation pre-registered in code (`memshield.causal_diagnostics.aggregate_V_top_attended`):

- **Layer**: `memory_attention.layers[-1].cross_attn_image` — last cross-attention block.
- **Value extraction point**: PRE-output-projection V tensor (after `v_proj` and `_separate_heads`, before SDPA mixes Q/K/V or `out_proj`). Shape (B, H, Nk, d_head).
- **Selection rule (FIXED across conditions)**: top-K (default K=32) memory positions by total received attention `attn.sum(dim=(B, H, Nq))`.
- **Selected positions (CONDITION-SPECIFIC)**: re-derived per condition because bank composition differs. This is the honest interpretation of "selection criterion frozen across conditions".
- **Aggregation**: per condition, V averaged over its top-K positions and over (B, H) → (d_head,) vector.
- **Metric**: `d_mem(t) = 1 − cos(M_clean[t], M_attacked[t])` ∈ [0, 2].
- **Sensitivity (appendix)**: K ∈ {16, 32, 64}.

Implementation: `compute_d_mem_trace(V_attn_clean, V_attn_attacked, top_k=32)` in `memshield/causal_diagnostics.py`. 12/12 Windows self-tests pass.

### A3 hook spec (R4-pre-registration, refined post-impl 2026-04-27)

**Refinement after codex M0 review**: the original wording said "block memory writes". The actual implementation skips the per-frame entry into `obj_output_dict["non_cond_frame_outputs"]`. SAM2 reads this dict for BOTH (a) mask-memory chunk assembly AND (b) object-pointer token assembly when `use_obj_ptrs_in_encoder=True`. The honest pre-registration is therefore:

> **A3 intervention**: at t ∈ W_blocked, the per-frame `current_out` is computed normally (Hiera forward, mask decoder, memory encoder all run; current frame's mask is correctly predicted using prior bank), but is NOT appended to `obj_output_dict`. Subsequent frames querying SAM2's temporal state see no contribution from t in either mask-memory or obj_ptr assembly. This is "block ALL future temporal state contributions from blocked frames", a slight broadening of "memory write blocking".

```python
def make_blocking_forward_fn(base_forward_fn, *, blocked_frames=(), extractor=None):
    """Wraps VADIForwardFn so that, at fid in blocked_frames, the per-frame
    current_out is NOT appended to obj_output_dict["non_cond_frame_outputs"].
    Excludes blocked frames from BOTH mask-memory chunks AND obj_ptr tokens.

    Frame 0 (prompt) is not blockable. Implementation in
    memshield/causal_diagnostics.py; 12/12 Windows self-tests pass."""
```

This is the SCIENTIFIC mechanism still: any inserted-frame contribution to SAM2's prompt-conditioned temporal memory propagation is suppressed at blocked positions. The matched non-insert control (A3-control) tests whether the effect is insert-position-specific.

---

## Validation (R4 final)

### C1.a — memory causality WITH negative control (10 clips × 3 configs)

- A3-baseline: full v5 (no blocking)
- A3-attacked: full v5 + hook with W_blocked = W_attacked
- A3-control: full v5 + hook with W_blocked = W_control (matched random non-insert non-bridge, seed=0)

Pre-registered acceptance:

| Tier | collapse_attacked | collapse_attacked − collapse_control | Clips |
|---|---|---|---|
| Strong | ≥ 0.20 abs | ≥ 0.10 abs | ≥ 7/10 (both) |
| Partial | ≥ 0.10 abs | ≥ 0.05 abs | ≥ 6/10 (both) |
| Fail | < 0.10 OR control comparable | — | otherwise |

### C1.b — persistence extension

Per-clip d_mem(t) trace under R2 protocol, 3 conditions (clean / insert-only / full-v5).
Acceptance: `∫(d_mem_full(t) − d_mem_only(t)) dt over (w_K, w_K + L) > 0` on ≥7/10 clips.
T_obj 16/32/64 sensitivity in appendix.

### C2 — RAW joint headline

**Table 1 (MAIN — RAW JOINT v5 vs A0 paired)**:

| Clip | A0 J-drop | RAW v5 J-drop | Lift | Win? | polish_applied? |
|---|---|---|---|---|---|

Headline gates (RAW v5):
- ≥ 5/10 strict wins
- mean paired lift ≥ +0.05
- median paired lift > 0
- top-contributing clip < 40% of total lift
- mean RAW v5 J-drop ≥ 0.55

**Table 2 (DEPLOYMENT — wrapper-selected, separate)**:

| Clip | RAW v5 J-drop | A0 J-drop | Wrapper publishes | Final J-drop |
|---|---|---|---|---|

Wrapper = max(RAW v5, A0). NOT used for headline gates.

### Ablations (R4 final)

| # | Ablation | Configurations | Acceptance |
|---|---|---|---|
| **A1** | Bridge δ contribution, isolated | (i) insert-only at W* with same nu_init, ALL bridge variables zeroed (traj=0, alpha_logits=-1e9, warp=0, R=0); (ii) insert + full Stage 14 polish at SAME W*, SAME nu_init | A1-full mean paired lift ≥ +0.05; ≥6/10 strict; positive median |
| **A2** | Placement matters | Random K=3 vs joint curriculum search (both with full v5 polish) | Search > random ≥ +0.10 mean lift |
| **A3** | Memory-causality with negative control | A3-baseline / A3-attacked / A3-control | Strong / Partial / Fail tier above |

---

## Failure Modes & Pre-registered Decisions

| Failure | Detection | Decision |
|---|---|---|
| Bridge δ regress per clip | RAW v5 J-drop < A0 J-drop | Report honestly; wrapper-selected column for deployment readers |
| A3 strong | collapse_att ≥0.20 AND att−ctrl ≥0.10 on ≥7/10 | Framing-A "DOMINANT" |
| A3 partial | collapse_att ≥0.10 AND att−ctrl ≥0.05 on ≥6/10 | Framing-B "SUBSTANTIAL" |
| A3 control comparable | att−ctrl < 0.05 on majority | Framing-C workshop pivot |
| A3 attacked too weak | collapse_att < 0.10 on majority | Framing-C workshop pivot |
| Joint search low information | min_mass<1 / singleton>0 | Multi-seed prescreen |
| Stage 14 pathological loop | wall>2× peer | Kill+retry seed 1 |
| Outlier-driven mean | top-clip>40% | Leave-one-out reporting |

---

## Discussion (E1 honest ownership)

> "Why search, not heuristic: We rely on a joint curriculum search for placement because vulnerability heuristics adapted from per-frame fragility (a 3-signal scorer combining confidence drop, mask discontinuity, and Hiera discontinuity) were empirically falsified on a 10-clip ranked-vs-random comparison (mean J-drop 0.488 ranked vs 0.534 random — anti-correlated). The search is necessary engineering for this attack surface; we do not claim it as a primary contribution."

---

## Compute & Timeline (3 days, ~23 GPU-h)

| Day | Task | GPU-h | Decision gate |
|---|---|---|---|
| Day 1 AM | Implement R4 hook spec + extractor + control sampler (~3h impl) | minimal | code review |
| Day 1 PM | A3-baseline + A3-attacked + A3-control + d_mem trace on 10 clips | 10 GPU-h | **A3 verdict gates framing** |
| Day 2 | C2 RAW joint v5 + A0 paired + A1 (paired ablation) overnight | 8 GPU-h | Headline gates check |
| Day 3 AM | A2 random + T_obj sensitivity + reporting | 5 GPU-h | Tables done |
| Day 3 PM | Writeup with conditional Framing | 4 author-h | Submit |
| **Total** | | **~23 GPU-h** | **3 days** |

Implementation status: v5 = v4.1 (commit `da719dc`) already on Pro 6000. New code: A3 hook (~2h), memory-readout extractor (~1h), control-frame sampler (~30min), reporting scripts (~2h).

---

## Experiment Handoff Inputs (for `/experiment-plan` if needed)

- **Must-prove claims**: C1.a strong/partial/fail (with negative control), C1.b persistence integral, C2 RAW joint headline.
- **Must-run ablations**: A1 (R3-locked bridge isolation), A2 (random placement), A3 (R4 with negative control).
- **Must-do appendices**: T_obj 16/32/64 sensitivity.
- **Critical datasets / metrics**: DAVIS-2017, 10 held-out clips, J-drop on uint8 export, d_mem(t) trace per R2 protocol.
- **Highest-risk assumptions**: A3 collapse magnitude (pre-registered narrowing); RAW joint headline gates (v4.1 dev-4 75% on 4 clips suggests 5-7/10 likely on 10); hook implementation modularity (~50 LOC, low risk).

---

## Compute & Implementation Status Summary

- **v4.1 method already implemented and validated on 4 clips dev-4** (commit `da719dc` on Pro 6000):
  - dog: applied, J=0.6351 (+0.13 vs v4.0)
  - bmx-trees: still reverts, J=0.5987
  - camel: applied, J=0.9589 (v4.0 result)
  - libby: applied, J=0.5813 (v4.0 result)
- **NEW code for paper experiments** (~5 hours impl + 23 GPU-h compute):
  1. `BlockInsertMemoryWritesHook` (R4 spec, ~50 LOC)
  2. Memory-readout extractor with R2 protocol (~30 LOC)
  3. Control-frame sampler with seed (~20 LOC)
  4. Reporting scripts for paired tables + d_mem trace plots (~200 LOC)
