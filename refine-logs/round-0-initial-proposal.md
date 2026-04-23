# Research Proposal: Vulnerability-Aware Decoy Insertion for SAM2 Dataset Protection

**Title**: VADI — Vulnerability-Aware Decoy Insertion for Per-Video SAM2 Dataset Protection.

## Problem Anchor

(verbatim from `refine-logs/PROBLEM_ANCHOR_2026-04-23_v4-insert.md`)

- **Bottom-line**: clean video + first-frame mask → processed video that (a) degrades SAM2 target segmentation across the whole processed video and (b) remains visually faithful. Processed video contains K_ins synthetic insert frames + δ-perturbed originals.
- **Must-solve bottleneck**: prior insertion attacks placed inserts at FIFO-canonical positions (motivation falsified by B2) and used LPIPS-expensive ProPainter decoy content. Missing: **principled insert placement via vulnerability analysis of the clean video**, content optimized for **current-frame Hiera pathway corruption**, amplified by δ on neighboring originals. No bank poisoning, no suppression.
- **Non-goals**: clean-suffix eval; defeat FIFO self-healing; UAP; runtime hook; pure-δ or pure-suppression method (user directive: insert+modify ONLY; no suppression).
- **Constraints**: white-box SAM2.1-Tiny; per-video PGD; K_ins ∈ {1,2,3} at heuristic-scored positions; two-tier fidelity (f0 ε=2/255, originals ε=4/255 + LPIPS≤0.20, inserts LPIPS≤0.35 vs temporal interpolation); GT-free supervision; DAVIS-10.
- **Success**: (1) mean J&F drop ≥ 0.35; (2) fidelity triad met; (3) **vulnerability-scorer causality**: top-K positions give ≥ 2× the J-drop of random-K at matched budget; (4) restoration attribution: damage concentrates in current-frame Hiera pathway at insert positions.

## Technical Gap

Current methods fail at per-video SAM2 dataset protection under the "insert + modify" constraint:

- **v4 decoy insertion** (prior internal): placed inserts at canonical FIFO positions motivated by "defeat FIFO self-healing" — B2 causal ablation falsifies this (bank contributes `|ΔJ|<0.01`). Position choice is therefore arbitrary. Also ProPainter-decoy content gave LPIPS 0.67-0.89, and the reported 92.5% J-drop was mostly direct-δ-on-eval, not insertion effect.
- **v2 clean-suffix eval**: a different regime entirely, empirically falsified (J-drop ≈ 0.001 on 3 runs).
- **UAP-SAM2**: universal, not per-video, different threat model.
- **v3 pure-δ + suppression**: user explicitly rejected. Inserts are required.

Why naive fixes don't work:

- **Random insertion positions**: no reason to beat canonical; same LPIPS cost, weak signal.
- **More inserts (K_ins=6,9)**: fidelity budget explodes; and B2 tells us filling the bank has near-zero segmentation effect anyway.
- **ProPainter decoy quality improvement**: the LPIPS floor is structural (ProPainter inpainting quality), not a tuning issue.
- **Just "bigger PGD"**: prior R003 with K_ins=3 canonical + 200 steps + LPIPS off gave J-drop 0.0013. The optimizer is not the bottleneck.

**The missing mechanism**: placement of inserts at positions where SAM2 is **already vulnerable on the clean video** (scene transitions, confidence dips, motion discontinuities), with insert content optimized **end-to-end to corrupt SAM2's current-frame Hiera features**, amplified by δ on neighboring originals via **decoy targeting** (not suppression). The principle: **don't fight SAM2's stable regions; tip it over where it was already teetering**.

## Method Thesis

**One-sentence thesis**: Vulnerability-aware insertion — use the publisher's access to the clean video to identify where SAM2 is intrinsically unstable, place synthetic frames there whose content is adversarially optimized to corrupt SAM2's current-frame pathway when processed, and amplify with decoy-targeted δ on adjacent originals, under a two-tier fidelity budget calibrated to the natural DAVIS LPIPS floor.

**Why this is the smallest adequate intervention under user constraint**: exactly 2 trainable tensors (δ per-frame, ν per-insert). Insertion positions from a **heuristic scorer** (no learned network). No ProPainter, no diffusion prior, no learned scheduler. Under the user's "must use insert+modify" constraint, this is the most minimal realization.

**Why this route is timely**: SAM2.1 is foundation-scale promptable VOS being adopted broadly. Dataset protection with embedded synthetic frames is a natural threat model for publishers. The frontier move is using **clean-SAM2's own vulnerability structure** as an attack signal — i.e., attack where the model is already weak. This is closer to recent "feature attribution-guided attacks" in adversarial-ML than to old UAP-style approaches.

## Contribution Focus

- **Dominant contribution (C1)**: vulnerability-aware insertion method for per-video SAM2 dataset protection achieving mean J&F drop ≥ 0.35 on DAVIS-10 at fidelity triad, with a causal demonstration that heuristic vulnerability scoring beats random placement by ≥ 2× at matched budget.
- **Supporting contribution (C2)**: restoration-counterfactual attribution showing the J-drop concentrates in SAM2's current-frame Hiera pathway at insert positions (not in the bank), validating that "insert-as-current-frame-attack" is the operative mechanism — consistent with the B2 causal-diagnosis finding.
- **Non-contributions**: no new generator (inserts are simple temporal-interpolation base + ν); no new SAM2 variant; no UAP; no runtime hook; no bank poisoning; no FIFO-self-healing claim; no learned components beyond δ,ν; no LLM/diffusion/RL.

## Proposed Method

### Complexity Budget

- **Frozen/reused**: SAM2.1-Tiny, `SAM2VideoAdapter` (Chunk 5b-ii), LPIPS(alex), `fake_uint8_quantize` (STE), `DropNonCondBankHook` + new swap hooks.
- **New trainable tensors (2)**:
  - `δ ∈ R^{T × H × W × 3}` — per-frame perturbation on originals, two-tier ε budget.
  - `ν ∈ R^{K × H × W × 3}` — per-insert content perturbation on top of temporal-midframe base.
- **New non-trainable infrastructure**: heuristic vulnerability scorer (pure algorithmic, no learned weights); three restoration swap hooks (inference-only).
- **Intentionally not used**: ProPainter (replaced by temporal midframe), L_stale, L_suppress, learned scheduler, LLM/diffusion/RL.

### System Overview

```
(Offline publisher analysis, one-time, GT-free)
  Run clean SAM2 on x_0..x_{T-1} with m_0 prompt
  → m̂_true_t   = sigmoid(pred_logits)                           ∈ [0,1]^{H×W}
  → s_t        = object_score_logits_t                          ∈ R
  → H_t        = Hiera encoder output                           (cached for restoration hooks)
  → confidence = sigmoid(s_t) * mean(m̂_true_t > 0.5)            ∈ [0,1]

(Vulnerability scoring, heuristic, no learned parameters)
  For each candidate insertion position m ∈ {1, 2, ..., T-1} (insert between orig t=m-1 and t=m):
    v_conf_m     = |confidence_m - confidence_{m-1}|              # confidence drop/jump
    v_mask_m     = 1 - IoU(m̂_true_{m-1}, m̂_true_m)                # mask discontinuity
    v_feat_m     = ||H_{m-1} - H_m||_2 / (mean(||H||) + eps)      # feature discontinuity (normalized)
    v_motion_m   = optical_flow_magnitude_change(m-1, m)          # motion discontinuity
    v_m          = α·v_conf + β·v_mask + γ·v_feat + δ·v_motion
    (α,β,γ,δ chosen so each term ∈ [0,1] and equally weighted; tune α-δ on pilot if needed)

  Select insertion positions W = argtop-K_ins of {v_m}, with non-adjacency constraint
  (|m_i - m_j| ≥ 2 to avoid clustering). K_ins ∈ {1, 2, 3}.

(Decoy target construction)
  For each insert position m_k: decoy_offset_k = argmax over shifts {(dy,dx)} the distance from m̂_true_{m_k}
    subject to border safety (decoy must fit in frame).
  m̂_decoy_t = shift_mask(m̂_true_t, decoy_offset)   for t near each insert

(Per-video PGD — 100 steps, 3 stages)
  Initialize δ = 0, ν = 0
  base_insert_k = 0.5·(x_{m_k-1} + x_{m_k}) + small_motion_predicted_delta    # temporal midframe
  
  For step = 1..100:
    Build processed video by interleaving:
      x'_t      = clip(x_t + δ_t) ∘ fake_quant                  # perturbed originals
      insert_k  = clip(base_insert_k + ν_k) ∘ fake_quant        # optimized inserts
    
    SAM2 forward via SAM2VideoAdapter(processed, m_0) → pred_logits per processed-time frame
    
    L_decoy_insert = Σ_k softplus(-mean(pred_logits_{m_k} · m̂_decoy_{m_k}) + 0.5)   # insert frame: pull to decoy
    L_decoy_neighbor = Σ_{t ∈ NbrSet} 0.5 · softplus(-mean(pred_logits_t · m̂_decoy_t) + 0.5)
                                                        # neighbor frames soft-pull to decoy
    L_obj        = Σ_{t in processed, t≠0} softplus(object_score_logits_t + 0.5)
    L_fid_orig   = Σ_{t≥1} max(0, LPIPS(x'_t, x_t) - 0.20)
    L_fid_ins    = Σ_k max(0, LPIPS(insert_k, base_insert_k) - 0.35)
    L_fid_f0     = max(0, 1 - SSIM(x'_0, x_0) - 0.02)
    
    L = L_decoy_insert + L_decoy_neighbor + 0.3·L_obj 
      + λ(step)·(L_fid_orig + L_fid_ins) + λ_0·L_fid_f0
    
    δ, ν ← (δ, ν) - η·sign(∇ L)
    clip δ_0 to ±2/255, δ_{t≥1} to ±4/255, ν_k to ±8/255 (inserts get wider ε since content is synthetic)
    log surrogate_J_drop, per-frame LPIPS, f0 SSIM
  
  Return (δ*, ν*) at argmax surrogate_J_drop over fidelity-feasible steps

NbrSet for each insert m_k: {m_k - 2, m_k - 1, m_k + 1, m_k + 2} clipped to [1, T-1]
```

### Core Mechanism

**Input → output**:
- Input: x_0..x_{T-1} (clean video), m_0 (prompt mask).
- Output: processed_video of length T + K_ins (originals + inserts interleaved at W positions), uint8.

**Policy**:
- **Position selection**: heuristic v_m scorer (no learned weights). Top-K_ins non-adjacent.
- **Content optimization**: (δ, ν) jointly via PGD with decoy targeting + fidelity hinges.
- **Return state**: Pareto-best over feasible steps (no augmented-Lagrangian μ pathology).

**Training signal**:
- Primary: `L_decoy_insert` + `L_decoy_neighbor` (pull SAM2 toward decoy at and near inserts).
- Secondary: `L_obj` (suppress object confidence).
- Constraint: fidelity hinges on (originals, inserts, f0).
- **Zero GT use**: all targets (m̂_true, m̂_decoy, confidence, features) derived from clean-SAM2 pseudo-labels, not DAVIS annotations.

**Why this is the main novelty**:
1. **Vulnerability-aware placement** replaces the falsified canonical-FIFO placement. Positions are chosen because SAM2 is naturally weak there.
2. **Insert-as-current-frame attack**: B2 proved bank is non-causal; we place inserts so that when SAM2 processes them, their bad Hiera features directly corrupt the causal pathway.
3. **Decoy (no suppression)**: respects user directive. Pulls SAM2 toward a specific wrong location rather than making pred_logits generically negative.

### Modern Primitive Usage

- **SAM2.1** is the attacked model AND the pseudo-label source. Natural foundation-model usage.
- **LPIPS(alex)** for fidelity (standard).
- **Optical flow** for motion-discontinuity scoring (pretrained RAFT if available; otherwise finite-difference).
- **Explicit non-uses**: no LLM, no diffusion prior, no RL. Position selection is heuristic because (a) K_ins is tiny (≤3), (b) the vulnerability signal is geometric/per-frame and doesn't need learning to be useful at this scale.

### Integration

- Reuse `memshield/sam2_forward_adapter.py::SAM2VideoAdapter` for differentiable forward.
- Reuse `memshield/losses.py::decoy_target_loss` (5-region form) simplified to just (decoy positive term + background suppress term).
- Reuse `memshield/propainter_base.py::find_decoy_region` + `shift_mask` for decoy offset selection.
- Drop ProPainter insert-base generator. Replace with simple temporal midframe + ν.
- Extend `memshield/ablation_hook.py` with `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook` for restoration attribution.
- New driver `scripts/run_vadi.py`.
- New module `memshield/vulnerability_scorer.py` implementing v_m heuristic.

### Training Plan

- Per-video PGD, 100 steps total, stages 30/40/30:
  1. `N_1 = 30`: attack-only (λ=0).
  2. `N_2 = 40`: fidelity regularization (λ starts 10, grows 2× per 10 steps when violated).
  3. `N_3 = 30`: Pareto-best tracking.
- Step size η = 1/255 (stages 1-2), 0.5/255 (stage 3).
- ε_∞ budgets: δ_0=2/255, δ_{t≥1}=4/255, ν_k=8/255 (inserts get wider budget because their content is entirely synthetic — the fidelity metric for inserts is LPIPS vs the temporal-midframe BASE, not vs a "specific source frame").
- F_orig = 0.20, F_ins = 0.35, f0 SSIM ≥ 0.98.
- Windowing: T ≤ 40 full backward. If T > 40, sliding 30-frame backward window.
- **Pilot gate (mandatory)**: 1 clip (dog, T=22 or 60), measure wall-clock, memory, surrogate J-drop, fidelity feasibility, top-K vs random-K vulnerability check.

### Failure Modes and Diagnostics

| Failure mode | Detection | Fallback |
|---|---|---|
| Vulnerability scorer selects useless positions (v_m ≈ uniform) | Top-K vs random-K pilot experiment: no J-drop gap | Fall back to canonical positions; flag that vulnerability signal is weak on this clip |
| Temporal midframe base is too blurry (ν can't rescue it) | LPIPS(insert, interpolated_neighbors) high even at ν=0 | Use ν init = random-gaussian (0, 0.02) and larger α_decoy |
| F_ins=0.35 infeasible | Stage-3 Pareto frontier empty | Per-clip adapt F_ins to `1.3 × natural_adjacent_LPIPS_mean` |
| δ amplification swamps fidelity | L_fid_orig grows without L_decoy_insert growing | Reduce neighbor weight from 0.5 to 0.3; prioritize insert frames over neighbors |
| Restoration attribution fails: R2 (clean Hiera at inserts) does NOT recover J | Mechanism is not current-frame pathway — damage distributed | Report honestly; proposal has a pilot-branch narrative for this outcome |
| PGD oscillation at stage 3 entry | L_decoy_insert bounces step-to-step | Pareto-best checkpoint selection handles by construction; else reduce η |

### Novelty and Elegance Argument

Closest work:
- **UAP-SAM2 (NeurIPS 2025)**: universal inference-time noise. Different threat model.
- **Adversarial patch attacks** (image domain): place a patch at a strategic location. Closest analogue but for static images, not video VOS; no notion of "vulnerability window" in time.
- **Video adversarial attacks on trackers**: generally target ALL frames uniformly. No principled placement.
- **Internal v4 decoy insertion**: placed inserts at canonical positions (motivation falsified); used ProPainter (expensive).

Exact novelties:
1. **Vulnerability-aware placement** using clean-SAM2's own tracking-quality signal as attack guidance. This is foundation-model-native.
2. **Insert-as-current-frame attack**: re-interprets insertion as corrupting the causal pathway (B2-informed) rather than as a bank-poisoning tool.
3. **Two-tier + insert-specific fidelity budget** that correctly treats insert frames as content-vs-interpolation comparisons, not as comparisons to a non-existent source frame.
4. **GT-free optimization and selection** via clean-SAM2 pseudo-labels + confidence weighting.

Why this is focused: two trainable tensors (δ, ν); one heuristic scorer; one paper thesis (vulnerability-aware insertion); one supporting attribution (restoration). No parallel contributions.

## Claim-Driven Validation Sketch

### Claim 1 (dominant, C1): Vulnerability-aware insertion achieves J-drop ≥ 0.35 at fidelity triad on DAVIS-10.

- **Minimal experiment**: DAVIS-10 subset (dog, cows, bmx-trees, blackswan, breakdance, car-shadow, breakdance-flare, bear, judo, camel). Per-clip VADI run with stated budgets.
- **Main table**:
  1. Clean (baseline J&F)
  2. **Ours** (K_ins=3, top-3 vulnerability positions)
  3. **Ours K_ins=1** (ablation: how much does scale matter?)
  4. **Random-position baseline** (K_ins=3, 3 random non-adjacent positions, same PGD)
  5. **Canonical-schedule baseline** (K_ins=3, {6,12,14}-style FIFO-canonical)
  6. **δ-only (no inserts)** (K_ins=0; pure per-video perturbation)
  7. **UAP-SAM2 per-clip** (universal attack as external baseline)
  8. **SAM2Long transfer** (ours, on SAM2Long)
- **Metric**: mean J&F drop on full processed video (f0 excluded). Fidelity triad per-frame.
- **Expected evidence**: Ours ≥ 0.35 J-drop with fidelity met on ≥ 7/10 clips. Top-K vulnerability beats random-K by ≥ 2× J-drop (C1 causal claim). Beats canonical-FIFO by ≥ 30% J-drop. Beats δ-only by ≥ 0.10 J-drop (inserts add value).

### Claim 2 (supporting, C2): The attack's damage lives in SAM2's current-frame Hiera pathway at insert positions.

- **Experiment**: restoration counterfactuals on ours DAVIS-10:
  - **R1** attacked + clean f0 memory
  - **R2** attacked + clean Hiera at insert positions ONLY
  - **R2b** attacked + clean Hiera at ALL frames
  - **R3** attacked + clean bank
  - **B-control** drop non-cond bank on attacked
- **Metric**: `ΔJ_restore(c) = J(attacked + swap) − J(attacked)`. Positive = restoration works.
- **Expected**: `ΔJ(R2) ≥ +0.20` (Hiera-at-inserts restores most damage), `ΔJ(R2b) ≥ ΔJ(R2)` (replacing all Hiera restores more — upper bound), `ΔJ(R1) ~ +0.10` (f0 is secondary), `ΔJ(R3) ≤ +0.02`, `ΔJ(B-ctrl) ≤ +0.02`.

### (Appendix) Claim 3: Cross-variant transfer on SAM2Long.

## Experiment Handoff Inputs

- **Must-prove**: C1 (J-drop ≥ 0.35, top-K beats random-K by 2×) and C2 (Hiera-at-inserts attribution).
- **Must-run ablations**: random-K placement, canonical-FIFO placement, δ-only, K_ins sweep (0/1/2/3), restoration R1/R2/R2b/R3, UAP-SAM2 baseline, SAM2Long transfer.
- **Datasets/metrics**: DAVIS-2017 val; mean J&F; per-frame LPIPS (original and insert); f0 SSIM; ΔJ_restore.
- **Highest-risk assumptions**:
  1. Vulnerability scorer is actually predictive of where insertion works (falsifiable via top-K vs random-K).
  2. F_ins=0.35 is feasible at ν ε=8/255 (pilot-checkable).
  3. Mean J&F drop ≥ 0.35 at fidelity triad is reachable.

## Compute & Timeline

- **Pilot**: 1 clip (dog), ~1 GPU-hour. Validates time, memory, feasibility, top-K vs random-K gap.
- **DAVIS-10 C1 primary** (8 configs above × 10 clips): ~3-4 GPU-hours.
- **Restoration C2** (4 configs × 10 clips): ~0.5 GPU-hour.
- **SAM2Long install + transfer**: 2-3 GPU-hours.
- **Appendix** (DAVIS-30, prompt-robustness, SAM2.1-Base): 5-8 GPU-hours.
- **Total**: ~15 GPU-hours on one Pro 6000; 3-4 focused days from PILOT-PASS.
