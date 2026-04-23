# Round 1 Refinement (VADI)

## Problem Anchor (verbatim from `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

- **Bottom-line**: clean video + first-frame mask → processed video with K_ins synthetic inserts + δ-perturbed originals; degrade SAM2 segmentation across the whole processed video; stay visually faithful.
- **Must-solve bottleneck**: prior insertions at FIFO-canonical positions (B2-falsified); ProPainter floor 0.67-0.89 LPIPS. Missing: principled placement via vulnerability analysis + insert-as-current-frame attack + decoy-targeted amplification on adjacent originals.
- **Non-goals**: clean-suffix eval, FIFO defeat, UAP, runtime hook, pure δ, pure suppression.
- **Constraints**: white-box SAM2.1-Tiny; per-video PGD; K_ins ∈ {1,2,3}; two-tier fidelity; GT-free.
- **Success**: J-drop ≥ 0.35; fidelity triad met; top-K beats random-K by ≥ 2×; restoration attribution confirms Hiera-at-inserts pathway.

## Anchor Check

- Anchor preserved. Codex R1 drift warnings align with our design constraints (insert-required, suppression-banned).
- **Self-audit**: Codex correctly flagged that my round-0 `L_obj = softplus(object_score + 0.5)` is effectively suppression. The user's anti-suppression rule applies at the mechanism level, not just the name level. I remove `L_obj` from default and replace with a **strictly contrastive decoy-margin loss** — this is an actual "pull SAM2 toward decoy", not "push object score down".

## Simplicity Check

- **Dominant contribution after revision**: vulnerability-aware insertion where (i) placement comes from a pre-registered rank-based 3-signal scorer, (ii) insert content is optimized under LPIPS-bound fidelity, (iii) δ is LOCAL to insert neighborhoods, (iv) single loss = decoy-margin contrastive.
- **Components removed**:
  - `L_obj` (suppression in disguise, per Codex F2 + drift warning).
  - Global δ perturbation across all frames (reduced to insert-local neighborhoods).
  - Hard ε=8/255 on ν (replaced by LPIPS-as-real-constraint).
  - Motion-discontinuity term in vulnerability scorer (not justified pre-pilot).
- **Reviewer suggestions rejected**: none flat-rejected; F7 (gated pilot) incorporated as a mandatory first step.

## Changes Made

### 1. Vulnerability scorer: rank-based robust-z over 3 signals (F1 P0)

**Before**: ad-hoc weighted sum with undefined α,β,γ,δ and 4 heterogeneous signals.

**After**:
```
For each candidate insert position m ∈ {1, ..., T-1} (insert between orig t=m-1 and t=m):
  r_conf_m = |confidence_m − confidence_{m-1}|           # object-score confidence derivative
  r_mask_m = 1 − IoU(m̂_true_{m-1}, m̂_true_m)             # pseudo-mask discontinuity
  r_feat_m = ||H_{m-1} − H_m||_2 / mean(||H||_2)         # Hiera feature discontinuity (normalized)

For each signal r_x, compute rank-based robust z-score:
  rz_x_m = (rank(r_x_m) − (T-1)/2) / (0.7413 · IQR(rank) / 2)
  (robust-z via IQR-based scale, symmetric around median rank)

v_m = rz_conf_m + rz_mask_m + rz_feat_m                  # equal weight, pre-registered

Positions: W = argtop-K_ins of v_m, with |m_i − m_j| ≥ 2.
```

All weights pre-registered as equal. No clip-specific tuning. **Flow term dropped** (Codex simplification).

### 2. Contrastive decoy-margin loss (F2 P0; replaces L_obj)

**Before**: `L_decoy = softplus(-mean(pred_logits · m̂_decoy) + 0.5)` — raises decoy without suppressing true.

**After**:
```
# Pre-registered pseudo-label masks
m̂_true_t  = sigmoid(clean_SAM2(x, m_0).pred_logits_t)    ∈ [0,1]^{HxW}
m̂_decoy_t = shift_mask(m̂_true_t, decoy_offset)          # geometric only, GT-free
c_t       = | 2·m̂_true_t − 1 |                           # confidence weight

# Masked means of processed-video pred_logits
mu_true_t  = (Σ_pixels pred_logits_t · m̂_true_t  · c_t) / (Σ m̂_true_t · c_t + eps)
mu_decoy_t = (Σ_pixels pred_logits_t · m̂_decoy_t · c_t) / (Σ m̂_decoy_t · c_t + eps)

L_margin_insert   = Σ_k softplus( mu_true_{m_k} − mu_decoy_{m_k} + 0.75 )
L_margin_neighbor = Σ_{t ∈ NbrSet \ inserts} 0.5 · softplus( mu_true_t − mu_decoy_t + 0.75 )

L_fid_orig = Σ_{t ∈ NbrSet originals} max(0, LPIPS(x'_t, x_t) − 0.20)
L_fid_ins  = Σ_k max(0, LPIPS(insert_k, base_insert_k) − 0.35)
L_fid_f0   = max(0, 1 − SSIM(x'_0, x_0) − 0.02)

L = L_margin_insert + L_margin_neighbor
  + λ(step) · (L_fid_orig + L_fid_ins) + λ_0 · L_fid_f0
```

**Key property**: minimizing `softplus(mu_true − mu_decoy + 0.75)` requires `mu_decoy` to be at least `mu_true + 0.75`. This ENFORCES a strict ordering (decoy location MUST beat true location by margin), which is the actual condition for SAM2 to switch its prediction. Raising decoy alone (old loss) doesn't satisfy this.

NO L_obj. NO L_suppress. Just contrastive decoy margin.

### 3. Local δ support (F4 P0; dense δ removed)

**Before**: δ over all frames f0..f_{T-1}.

**After**:
- **Neighborhood set** per insert: `NbrSet(m_k) = {m_k - 2, m_k - 1, m_k + 1, m_k + 2}` clipped to `[1, T-1]`.
- **Global δ support**: `S_δ = ∪_k NbrSet(m_k) ∪ {0}`. Only frames in `S_δ` get optimized; all others have δ_t = 0 hard-coded.
- For K_ins=3 at non-adjacent positions, |S_δ| ≈ 12-13 of T frames. Much sparser than v4's dense δ.
- Per-frame L∞: ε_0=2/255 (f0 is always in S_δ), ε_{t≥1}=4/255.

Ablation (as F5 requires): **"Local δ support" vs "Global δ support"** to check whether locality is necessary.

### 4. LPIPS-bound inserts, not hard ε (F6 P0)

**Before**: ν ε=8/255 hard clip.

**After**:
```
base_insert_k = temporal_midframe(x_{m_k-1}, x_{m_k})
                          # = 0.5·x_{m_k-1} + 0.5·x_{m_k} (simple; no motion prediction needed pre-pilot)

insert_k = clamp( base_insert_k + ν_k, 0, 1 ) ∘ fake_uint8_quantize_STE

Fidelity constraint: per-insert LPIPS(insert_k, base_insert_k) ≤ F_ins = 0.35    (hinge in loss)
                     Also: per-insert TV penalty to discourage high-frequency noise:
                                TV(insert_k) ≤ TV(base_insert_k) · (1 + 0.2)      (hinge)
ν has NO ε bound; LPIPS + TV handle it.
```

ν starts at zero + small Gaussian noise (`std=0.02/255`) to avoid stuck-at-zero gradient.

### 5. Placement causality: top-K vs random-K vs bottom-K, multi-draw (F5 P0)

**Main table** (revised):
| # | Config | Purpose |
|---|---|---|
| 1 | Clean | baseline J&F |
| 2 | **Ours K=1 top** | **mechanistic centerpiece** |
| 3 | Ours K=3 top | stronger attack variant |
| 4 | Random-K=1 (5 draws, paired bootstrap) | placement causality: same ν PGD, random position |
| 5 | Random-K=3 (5 draws, paired bootstrap) | same, K=3 |
| 6 | **Bottom-K=3** | isolates "placement matters at all" from "any placement + PGD works" |
| 7 | Canonical {6,12,14} | legacy comparison |
| 8 | δ-only on local neighborhoods (no inserts) | isolates "inserts add value" |

Move UAP-SAM2 per-clip baseline + SAM2Long transfer to **appendix** (Codex V1 simplification).

### 6. Surrogate selection metric explicit (F3 P0)

```
surrogate_J_drop(δ, ν, step) = 1 - mean_{t ∈ eval_range} J( sigmoid(SAM2_pred(processed(δ,ν))_t) > 0.5 ,
                                                            m̂_true_t_processed_time )
eval_range = all processed-time frames except f0 (prompt).
J = Jaccard of binary masks.
m̂_true_t_processed_time = clean-SAM2 pseudo-label remapped to processed-time index (original t → processed t' via insertion schedule).
```

**Zero DAVIS GT anywhere**. At each step, log this + per-frame LPIPS + f0 SSIM. Return `(δ*, ν*)` at argmax surrogate_J_drop over fidelity-feasible steps.

### 7. Gated 3-clip pilot (F7 P0)

```
Pilot: 3 clips (dog, cows, bmx-trees — spanning easy/static/fast motion).
  For each clip, run:
    A. K=1 top (vulnerability scorer pick)
    B. K=1 random (1 draw)
    C. K=3 top
    D. δ-only local (K=0; δ on arbitrary 12 mid-prefix frames)

Decision rule after pilot:
  Δ_top = J_drop(A) − J_drop(B)                 # top vs random (placement causality, K=1)
  Δ_insert = J_drop(C) − J_drop(D)              # inserts vs δ-only (inserts add value)
  Δ_scale  = J_drop(C) − J_drop(A)              # K=3 scales over K=1

  GO: if Δ_top ≥ 0.05 OR Δ_insert ≥ 0.05 on ≥ 2/3 clips, AND J-drop(C) ≥ 0.20 on ≥ 2/3 clips.
  NO-GO: if none of the above hold — fallback to "architecture-aware attack-surface analysis" paper.

Gate is 3-5 GPU-hours, BEFORE full DAVIS-10.
```

### 8. Modernization: optional gradient-based vulnerability scorer (Codex M2)

**Rank-based robust-z (default)** is the pre-registered method. If pilot shows v_m is weak, replace with:
```
For each candidate m, compute:
  perturb x_m with small Gaussian ε (once, no PGD)
  forward SAM2 → Δ_J_decoy_m = change in mu_decoy − mu_true at frames near m
  v_grad_m = |Δ_J_decoy_m|

Use rank-based robust-z of v_grad_m in place of v_m heuristic.
```

**Cost**: 1 forward + backward per candidate frame = ~T × 2 seconds ≈ 2 minutes per video. Well within budget.

**Kept as fallback**, not default — default remains heuristic 3-signal scorer per Codex's "don't introduce learnability if heuristic suffices".

## Revised Proposal (round-1 full)

### Problem Anchor
(see `PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

### Method Thesis

Vulnerability-aware insertion: identify SAM2's intrinsically weak windows on the clean video via a **pre-registered rank-based 3-signal scorer**, place K_ins inserts at top-K non-adjacent windows, optimize insert content (ν, LPIPS-bound) + **local** δ on insert neighborhoods under a **contrastive decoy-margin loss** (no suppression, no object_score penalty). K=1 is the mechanistic centerpiece; K=3 is the stronger variant.

### Contribution Focus

- **C1 (dominant, 2-part)**: (i) pre-registered vulnerability-aware placement on SAM2 using clean-SAM2 signals; (ii) end-to-end insert+local-δ PGD under contrastive decoy margin, achieving mean J-drop ≥ 0.35 at fidelity triad on DAVIS-10.
  - Causal claim (pre-gated): top-K placement beats random-K by ≥ 2× AND beats bottom-K by ≥ 3× at matched compute and content budget.
- **C2 (supporting)**: restoration-counterfactual attribution showing damage lives in current-frame Hiera pathway at insert positions (R2 ≥ +0.20 recovery), consistent with B2 bank non-causality.
- **Non-contributions**: no new generator, no UAP, no runtime hook, no bank poisoning, no learned components beyond δ,ν, no LLM/RL/diffusion, **no suppression loss**, **no object_score margin**.

### Complexity Budget

- Frozen/reused: SAM2.1-Tiny, SAM2VideoAdapter, LPIPS(alex), fake_uint8_quantize STE.
- New trainable (2): δ (over local support S_δ), ν (K_ins inserts).
- New non-trainable: rank-based vulnerability scorer (3 signals).
- New inference-only: 3 restoration swap hooks.
- Intentionally not used: ProPainter, L_stale, L_suppress, L_obj, learned scorer (unless pilot demands), LLM/diffusion/RL.

### System Overview (condensed)

```
Offline (GT-free):
  Clean SAM2 → m̂_true_t, confidence_t, H_t (cached)
  Vulnerability score v_m via rank-based robust-z of {conf_drop, mask_discont, Hiera_discont}
  W = top-K non-adjacent argmax v_m
  decoy_offset via geometric distance max
  m̂_decoy_t = shift_mask(m̂_true_t)

Per-video PGD (100 steps, 3 stages):
  δ supported on S_δ = ∪_k NbrSet(m_k) ∪ {0}     (local perturbation)
  base_insert_k = 0.5·x_{m_k−1} + 0.5·x_{m_k}    (temporal midframe)
  insert_k = clamp(base_insert_k + ν_k) ∘ quant
  x'_t = clamp(x_t + δ_t) ∘ quant                (for t ∈ S_δ; else x_t)
  processed = interleave(x', insert_k at W)
  Forward SAM2VideoAdapter(processed, m_0) → pred_logits_t
  
  L_margin_insert   = Σ_k softplus(mu_true_{m_k} − mu_decoy_{m_k} + 0.75)
  L_margin_neighbor = Σ_{t ∈ NbrSet\inserts} 0.5 · softplus(mu_true_t − mu_decoy_t + 0.75)
  L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
  L_fid_ins  = Σ_k max(0, LPIPS(insert_k, base_insert_k) − 0.35)
  L_fid_TV   = Σ_k max(0, TV(insert_k) − 1.2 · TV(base_insert_k))
  L_fid_f0   = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
  
  L = L_margin_insert + L_margin_neighbor
    + λ(step) · (L_fid_orig + L_fid_ins + L_fid_TV) + λ_0 · L_fid_f0
  
  δ, ν ← PGD step (ν unbounded by ε; constrained via L_fid_ins/TV)
  clip δ_0 to ±2/255, δ_{t≥1,t∈S_δ} to ±4/255
  surrogate_J_drop = 1 − J(SAM2_pred(processed), m̂_true_processed)
  δ*, ν* = argmax surrogate_J_drop over fidelity-feasible steps
  (NO DAVIS GT.)

Validation:
  Main (8 configs): Clean, Ours K=1 top, Ours K=3 top, Random-K=1 (5 draws),
                    Random-K=3 (5 draws), Bottom-K=3, Canonical {6,12,14}, δ-only local
  Restoration: R2 (Hiera at inserts), R2b (Hiera all), R3 (bank).
  Appendix: UAP-SAM2 per-clip, SAM2Long transfer, SAM2.1-Base transfer, DAVIS-30.
```

### Pilot Gate (mandatory pre-main-run)

3 clips × 4 configs (K=1 top, K=1 random, K=3 top, δ-only local). ~3-5 GPU-hours total.

**GO** if Δ_top ≥ 0.05 OR Δ_insert ≥ 0.05 on ≥ 2/3 clips AND J-drop(K=3 top) ≥ 0.20 on ≥ 2/3 clips.

**NO-GO**: fallback paper on "architecture-aware attack-surface analysis" using restoration + vulnerability scoring as main content (not an attack-success paper).

### Compute

- Pilot: ~3-5 GPU-hours.
- DAVIS-10 main (8 configs with multi-draw random): ~5-8 GPU-hours.
- Restoration: ~0.5 GPU-hour.
- Appendix (UAP, SAM2Long install+transfer, SAM2.1-Base, DAVIS-30): ~8-12 GPU-hours.
- **Total**: ~20 GPU-hours on Pro 6000 (larger than v3 due to multi-draw random baselines). Timeline 4-5 focused days.
