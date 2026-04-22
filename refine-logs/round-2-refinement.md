# Round 2 Refinement

## Problem Anchor (verbatim)
[Unchanged from PROBLEM_ANCHOR_2026-04-22.md; see round-0 + round-1 for full text.]

## Anchor Check
Preserved. All fixes target mechanism precision, not scope.

## Simplicity Check
- Dominant contribution unchanged: 2-phase preprocessor. Phase 1 = loss event (inserts + ROI decoy-target supervision). Phase 2 = recovery prevention (prefix perturbation + L_stale internal regularizer).
- Components removed: full-frame BCE on inserts (→ ROI only); dual flow option (→ RAFT only); SSIM / ΔE in optimization loss (→ reported metrics only, if not binding).
- Components merged / fixed: L_stale now 3-bin categorical KL (not 2-way ratio).
- No new contribution added.

## Changes Made

### C1 — Low-confidence lock sign error fixed (CRITICAL)
- **Was**: `softplus(-τ_conf - max(g_u))` → this rewards HIGH max logit (wrong direction).
- **Now**: `softplus(logsumexp(g_u) - τ_conf)` with `τ_conf` a logit ceiling. `logsumexp` instead of `max` so a single-pixel spike cannot satisfy the term trivially; `logsumexp` is a smooth upper bound on max.
- **Reasoning**: penalizes high logit mass anywhere in the frame, forcing SAM2's confidence down globally → "no salient foreground" behavior.

### C2 — Three clocks formalized (CRITICAL)
Three clocks made explicit:

- **Clock O** — original-frame index `o ∈ {0, 1, ..., T-1}` (clean input).
- **Clock M** — modified-sequence index `m ∈ {0, 1, ..., T + K_ins - 1}` (output video after insertions).
- **Clock W** — memory-write index `w` (monotone count of non-cond memory writes SAM2 performs when processing the modified sequence; cond frame f0 is write 0, first FIFO write is w = 1, etc.).

**Insert schedule defined on Clock W**:
> Place inserts at write positions `w_k = (num_maskmem - 1) · k` for `k = 1, 2, ..., K_ins`, with one insert shifted by `w_K ← min(w_K, W_total - 1)` to force the final insert to be the last write before eval.

For `num_maskmem = 7`, `K_ins = 3`: `w_k ∈ {6, 12, ...}` mapped to `W_total` that the modified sequence produces. Clock-M positions are derived by forward-simulating SAM2 write behavior: before inserting at write `w_k`, we know the corresponding modified-index `m_k` because every frame in the modified sequence triggers one write. Thus `m_k = w_k`. Clock-O positions come from the reverse insertion map: `o_k = m_k - (number of inserts before position m_k)`.

Explicit for 15-frame prefix (`T_prefix = 15`), `K_ins = 3`:
- `m_1 = 6, m_2 = 12, m_3 = 13` (w_3 forced to last-write)
- Insertions occur BETWEEN original frames: insert #1 is modified-index 6 → between clock-O f5 and f6; insert #2 is modified-index 12 → between O f10 and f11; insert #3 is modified-index 13 → between O f10 and f11 (two inserts between same original pair) ... this is wrong.

**Corrected mapping rule**:
- Write `w_k` corresponds to modified-sequence frame `m_k = w_k` (since each frame produces one write after cond).
- If insertion happens AT modified-index `m_k`, the number of inserts at or before `m_k` determines the original frame: `o_k = m_k - |{k' ≤ k : m_{k'} ≤ m_k}|`.

Cleanly: insert-k is placed such that it lands at modified-index `m_k`. Between originals, so the insert displaces everything after. For `{m_1, m_2, m_3} = {6, 12, 13}` we get inserts placed AFTER original `{o_1 = 5, o_2 = 10, o_3 = 10}` — two inserts between f10 and f11.

If the paper prefers inserts all between different originals, shift: `m = {6, 12, 14}` → inserts after `{o = 5, o = 10, o = 11}`. This is the canonical schedule; published in the paper in these exact three clocks.

- **Reasoning**: schedule claim and experiment are now reproducible from `num_maskmem` + `T_prefix` + `K_ins` alone, no hand-tuned `r`.
- **Impact**: off-resonance ablation (I2) can be defined precisely in the same clock system.

### I1 — L_stale reformulated as 3-bin categorical KL (IMPORTANT)
- **Was**: $\log(A^{\text{clean-recent}} / A^{\text{insert-memory}})$ — 2-way ratio.
- **Now**: build distribution `P_u = [A_u^\text{ins}, A_u^\text{recent}, A_u^\text{other}]` (normalized over all bank slots + conditioning + image-feature fallback). Target distribution `Q = [q_\text{ins}, q_\text{recent}, q_\text{other}]` with `q_\text{ins} = 0.6, q_\text{recent} = 0.2, q_\text{other} = 0.2`.

$$ L_\text{stale} = \frac{1}{|V|} \sum_{u \in V} \text{KL}(Q \| P_u) $$

- **Reasoning**: 3-bin explicitly favors insert slots over clean-recent AND tracks where rest of attention flows (prevents attention collapse to unrelated slots from looking like success).
- **Impact**: L_stale now a proper categorical target, not a raw ratio.

### I2 — Off-resonance deconfounded (IMPORTANT)
- **Was**: period-4 vs period-6 changes resonance AND last-insert-to-eval distance (confounded).
- **Now**: the ablation matches:
  - `K_ins = 3` (same)
  - `T_prefix = 15` (same)
  - **Last-insert modified-index `m_{K_ins}` = 14** (same, so distance to first eval `m = 15` is identical)
  - Only the spacing of earlier inserts varies:
    - **Resonance condition**: `m_k = 6 · k`, shift last to 14 → `{6, 12, 14}` → write periodicity = num_maskmem - 1
    - **Off-resonance condition**: `m_k = 4 · k`, shift last to 14 → `{4, 8, 14}` → write periodicity = 4 (breaks FIFO-period logic)
- **Reasoning**: recency is equalized; the only difference is whether the earlier 2 inserts sit at FIFO-resonant multiples.
- **Impact**: Claim 4 (FIFO resonance matters) becomes a clean mechanism test.

### I3 — Whole-suffix reporting metrics (IMPORTANT)
- **Optimization** `L_rec` stays on f15..f21 (compute).
- **Reported** rebound + post-loss AUC computed on f15..end (full suffix, may be 50-100 frames).
- Add **long-horizon J trajectory** plot in paper showing attack persists through end of video.
- **Reasoning**: claim of "no recovery" (not "delayed recovery") requires whole-suffix evidence.

### M1 — A_u extraction specified (MINOR)
- **Query set**: foreground queries = pixel positions inside `erode(C_u, 2)` after optical-flow warp from f0 mask.
- **Attention aggregation**: average across all memory-attention heads in the FINAL memory-attention block of SAM2 (closest to mask decoder).
- **Slot types**:
  - `insert` slots: any bank entry whose source modified-index `m_k` coincides with an inserted frame
  - `recent-clean` slots: any bank entry from a modified-index corresponding to an original prefix frame (f0..f14)
  - `other`: remaining attention to conditioning frame + image-feature fallback
- **Reasoning**: reproducible measurement; no hidden hyperparameters.

### Simplifications applied
- **S1**: L_loss ROI restriction. Was `BCE(g_ins, 1[D_ins])` over full frame. Now `ROI-BCE`: BCE computed only on pixels inside the union of the decoy box `D_box` and the original-target box `C_box`, both dilated by 10 pixels. Background mass removed.
- **S2**: Flow stack simplified to RAFT only (drop Unimatch).
- **S3**: Fidelity optimization stack collapsed to: δ L∞ clamp (hard) + ν LPIPS soft penalty + seam-band ΔE. SSIM kept as **reported metric only** (drop from loss).

## Revised Proposal (final form this round)

### Title
**MemoryShield: A Two-Phase Preprocessor for Protecting Video Data from SAM2-Family Streaming Promptable Segmenters**

### Problem Anchor
[Unchanged, see PROBLEM_ANCHOR_2026-04-22.md]

### Method

**Per-video PGD objective**:
$$ L(\nu, \delta) = L_\text{loss} + \lambda_r L_\text{rec} + \lambda_f L_\text{fid} $$

**Phase 1 — L_loss (inserts only, ROI)**:
$$ L_\text{loss} = \frac{1}{K_\text{ins}} \sum_k \Big[ \text{BCE}_{\text{ROI}}(g_{\text{ins}_k}, 1[D_{\text{ins}_k}]) + \alpha \cdot \text{softplus}(\text{CVaR}_{0.5}(g_{\text{ins}_k} \cdot 1[C_{\text{ins}_k}]) + m) \Big] $$

ROI = union of `D_box` and `C_box` dilated by 10 px.

**Phase 2 — L_rec (clean post-prefix eval frames u ∈ U = f15..f21)**:
$$ L_\text{rec} = \frac{1}{|U|} \sum_u \Big[ \alpha_\text{supp} \cdot \text{CVaR}_{0.5}(g_u \cdot 1[C_u])^+ + \alpha_\text{conf} \cdot \text{softplus}(\text{logsumexp}(g_u) - \tau_\text{conf}) \Big] + \beta \cdot L_\text{stale} $$

**L_stale (3-bin categorical KL)**:
$$ L_\text{stale} = \frac{1}{|V|} \sum_{u \in V} \text{KL}(Q \| P_u), \quad P_u = [A_u^\text{ins}, A_u^\text{recent}, A_u^\text{other}], \quad Q = [0.6, 0.2, 0.2] $$

V = first 3 clean post-last-insert frames.

**L_fid**:
$$ L_\text{fid} = \mu_\nu \cdot (\text{LPIPS}(x_{\text{ins}_k}, f_{\text{prev}_k}) - 0.10)^+ + \mu_s \cdot \Delta E_\text{seam} $$

δ hard-clamped to L∞ ball every step (not in L_fid). SSIM is reported, not optimized.

**Schedule (three-clock formalization)**:
- Clock W (memory writes), Clock M (modified-sequence index), Clock O (original-frame index).
- Insert positions on Clock W: `w_k = 6 · k` for k=1,2,3, with `w_3 ← w_3 + 2` when prefix ends at w=14 (forces final insert adjacent to eval start).
- For T_prefix = 15, num_maskmem = 7, K_ins = 3:
  - Canonical schedule: `m = {6, 12, 14}` → inserts after original frames {5, 10, 11}.
  - Off-resonance control (same last-insert distance): `m = {4, 8, 14}` → inserts after original {3, 6, 11}.

**Training Plan**:
1. Clean SAM2 run → `C_u` per clock-O frame (flow-warped + erode-2).
2. Decoy-offset selection at video start.
3. ProPainter × K_ins → insert bases.
4. Stage 1 (1-40): ν-only with L_loss.
5. Stage 2 (41-80): δ-only with L_rec (inserts frozen).
6. Stage 3 (81-200): joint alternating 2:1 δ:ν with full L.
7. δ L∞-clamp every step; ν LPIPS penalty via aug-Lagrangian.
8. Cache clean-suffix image embeddings.

**A_u extraction (fixed)**:
- Foreground queries = pixel positions inside `erode(flow_warp(C_u), 2)`.
- Attention = average of all heads in SAM2's FINAL memory-attention block.
- Slot types:
  - `insert`: bank entries sourced from modified frames in `{m_k}`
  - `recent-clean`: bank entries from modified frames corresponding to original prefix (f0..f14)
  - `other`: conditioning slot + image-feature fallback

### Validation

**Claim 1 (DOMINANT)**: 4-condition ablation on DAVIS-10 hard — clean / Phase-1-only / Phase-2-only / Full. Metric: mean J-drop (full suffix) + rebound + post-loss AUC. Expect Full ≥ 0.55 J-drop, singles ≤ 50% Full, Full rebound ≤ 0.15.

**Claim 2 (SUPPORTING)**: Full vs Full-no-L_stale on DAVIS-10. Metric: rebound + post-loss AUC + bank-attention breakdown `P_u`. Expect Full keeps `P_u^\text{ins} ≥ 0.5` on V; no-L_stale collapses to `P_u^\text{ins} ≤ 0.2`; rebound gap ≥ 0.2.

**Claim 3 (MANDATORY transfer)**: SAM2Long on attacked videos. Expect SAM2Long J-drop ≥ 0.25, retention ≥ 0.40.

**Claim 4 (MECHANISM)**: Resonance `m={6,12,14}` vs off-resonance `m={4,8,14}`, fixed K_ins, prefix, last-insert-distance, ε, LPIPS. Expect resonance ≥ off-resonance by ≥ 20pp J-drop.

### Compute & Timeline
~24 GPU-hours total (~1 GPU-day on Pro 6000). 4-week timeline.

### Novelty re-statement
Closest prior: UAP-SAM2 (universal, all-frame); Chen 2021 (video classification, appended dummy); ACMM 2023 (first-frame only); BadVSFM (training-time). MemoryShield is per-instance preprocessor targeting FIFO self-healing via loss-induction + recovery-prevention composition, with FIFO-resonant schedule parameterized by num_maskmem. `L_stale` (3-bin KL on bank-attention mass) is the Phase-2 mechanism that makes recovery-prevention work.
