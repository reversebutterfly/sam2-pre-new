# Round 3 Refinement

## Problem Anchor (verbatim)

(per `PROBLEM_ANCHOR_2026-04-23.md` — unchanged)

## Anchor Check

- Anchor preserved. Drift warning remains NONE.
- Round-3 reviewer did not request changes that would drift the problem.

## Simplicity Check

- Dominant contribution unchanged: causal loop (diagnose → attack → verify).
- No components added in Round 3. Changes are all decision-plan polish + fallback spec.
- Reviewer explicitly says "NONE" for simplification — further trimming would hurt.

## Changes Made

The reviewer identified no formulation issues at Round 2. Remaining gap (to 9+) is **empirical risk**. Round-3 refinement therefore bulletproofs the proposal against the empirical outcome of the mandatory pilot gate. Two structured additions:

### 1. Pilot gate promoted to first-class design element (addresses Venue Readiness 7.6)

**What**: move the pilot from a "to-be-done" footnote to an explicit **Phase-0 design element** with four predefined outcome branches and the corresponding paper narrative for each.

**Reasoning**: reviewer flagged that venue readiness hinges on pilot outcome. We cannot pre-empt the empirical outcome, but we CAN pre-empt the paper narrative so the same proposal carries through all four branches without post-hoc storytelling.

**Addition to proposal**:

```
Phase 0 — Pilot gate (mandatory, 1 GPU-hour, 1 clip on Pro 6000)

  Measure: (i) surrogate_J_drop(δ*) at the best-feasible step,
           (ii) per-frame LPIPS mean / max,
           (iii) f0 SSIM final,
           (iv) wall-clock per 10 PGD steps,
           (v) peak GPU memory,
           (vi) ΔJ_restore(R1), ΔJ_restore(R2), ΔJ_restore(R12) on the pilot clip.

  Four predefined branches:

  B1. STRONG ATTACK + ATTRIBUTED CAUSAL PATHWAY
      (surrogate J-drop ≥ 0.40, fidelity satisfied,
       ΔJ(R12) ≥ +0.40 and R1/R2 individually ≥ +0.15)
      → proceed to DAVIS-10 main run as planned.
      Paper narrative: causal-loop dataset protection works.

  B2. STRONG ATTACK + JOINT-DISTRIBUTED DAMAGE
      (surrogate J-drop ≥ 0.40, fidelity satisfied,
       R12 strong but R1 and R2 each < +0.10)
      → proceed to DAVIS-10, but paper reports joint non-identifiability
        between f0 and current-frame pathways.
      Paper narrative: causal-loop attack works, attribution is joint-only.

  B3. STRONG ATTACK + NO CAUSAL ATTRIBUTION
      (surrogate J-drop ≥ 0.40 with fidelity, but ALL ΔJ_restore < +0.05)
      → proceed, but paper reports "empirical attack without pathway
        attribution". Weakens the "verify" leg of the causal loop.
      Paper narrative: attack is strong but mechanism is not what we
      thought; propose in discussion what the actual mechanism might be.

  B4. WEAK ATTACK
      (surrogate J-drop < 0.30 at feasibility OR fidelity infeasible
       under stated budget on the pilot clip)
      → trigger the Hiera/f0 feature-corruption fallback (see below).
      If the fallback still does not reach J-drop ≥ 0.30, STOP and
      declare the decoy-free suppression attack inadequate on
      SAM2.1-Tiny at these fidelity thresholds. Paper pivots to
      "architecture-aware attack-surface analysis of SAM2-family
      VOS" using the causal ablation (B2 + restoration swaps on
      attacked pilot video), honestly reporting that preprocessor-
      level per-video attacks under LPIPS ≤ 0.20 + SSIM ≥ 0.95 face
      a hard ceiling on SAM2.1-Tiny.

  The four branches are committed BEFORE the pilot runs. No post-hoc
  branch reshuffling.
```

**Impact on score dimensions**:
- Venue Readiness: +1 (paper survives any pilot outcome with a pre-defined narrative — no "pivot roulette").
- Validation Focus: +0.3 (commitment to pre-defined decision rule is a methodological strength).

### 2. Structured Hiera/f0 feature-corruption fallback (addresses Contribution Quality 8.1)

**What**: if the pilot triggers B4, replace `L_suppress` with a pathway-aligned loss, NOT as a parallel addition.

**Reasoning**: the reviewer explicitly said "if suppression underperforms, REPLACE rather than add". The fallback is a structured drop-in substitute, not a second loss.

**Addition to proposal**:

```
Alternative loss L_suppress_pathway (used ONLY if pilot triggers B4):

  Let   h_clean_t = image_encoder.forward_image(x_t)  (precomputed once)
        h_attack_t = image_encoder.forward_image(x'_t)  (fresh per PGD step)
        m_f0_clean  = clean SAM2's f0 maskmem
        m_f0_attack = SAM2 run on attacked video's f0 maskmem

  L_suppress_pathway =
        α · ⟨ ∥ h_attack_t − h_clean_t ∥_2 ⟩_{t over m̂_true support}        # Hiera corruption under true-foreground
      + β · ∥ m_f0_attack − m_f0_clean ∥_2^2                                # f0 maskmem drift

  Replaces L_suppress entirely (not added). α, β set so that initial
  magnitudes match L_suppress on the pilot clip. No other loss terms
  change. The rest of the pipeline (L_obj, L_fid, three stages,
  best-feasible checkpoint) stays identical.
```

This is a clean "swap suppression objective" move. One trainable component (still δ). No new modules.

**Impact**:
- Contribution Quality: +0.2 (fallback is not bolt-on; it is pathway-aligned in the same framework).
- The paper's primary narrative uses `L_suppress`. The fallback is only mentioned in the pilot-branch section.

### 3. Explicit confidence-weighted pseudo-mask (Modernization #2)

**What**: use soft logits AND confidence weighting from clean-SAM2's sigmoid.

**Reasoning**: reviewer modernization suggestion; cheap and well-aligned with soft-logit supervision.

**Addition**: `m̂_true_t = sigmoid(...)` already soft. Add confidence mask `c_t = abs(2·m̂_true_t − 1)` to downweight pseudo-mask uncertain boundaries:

```
L_suppress[t] = softplus( (1/Σc_t) · Σ_pixels c_t · pred_logits_t · m̂_true_t )
```

This makes supervision focus on pixels the clean SAM2 is confident about, and smoothly ignores borderline pixels. Still GT-free.

## Revised Proposal (final pre-pilot version)

### Problem Anchor

(see `PROBLEM_ANCHOR_2026-04-23.md`)

### Method Thesis

Architecture-aware dataset protection for SAM2 via the **diagnose → attack → verify** causal loop. Pilot-gate-committed: paper narrative pre-committed to four pilot outcomes to prevent post-hoc reframing.

### Contribution Focus

- **C1 (single, four-part)**: (i) causal diagnosis (B2 bank marginality); (ii) pathway-targeted per-video PGD (two-tier fidelity, GT-free soft-logit + confidence-weighted supervision); (iii) restoration-counterfactual attribution (R1/R2/R12/R3/B-control); (iv) pre-committed pilot-gate decision rule.
- **Non-contributions**: unchanged from R2.

### Supervision (final)

```
m̂_true_t = sigmoid(clean_SAM2(x, m_0).pred_masks_high_res_t) ∈ [0,1]
c_t      = |2·m̂_true_t − 1|                                  # confidence weight
L_suppress = Σ_{t≥1} softplus( (1 / Σ_pixels c_t) · Σ_pixels c_t · pred_logits_t · m̂_true_t )
L_obj      = Σ_{t≥1} softplus( object_score_logits[t] + 0.5 )
L_fid_frame[t] = max(0, LPIPS(x'_t, x_t) − 0.20)
L_fid_f0       = max(0, 1 − SSIM(x'_0, x_0) − 0.02)
L = L_suppress + 0.3·L_obj + λ(step)·Σ_t L_fid_frame[t] + λ_0·L_fid_f0
```

`surrogate_J_drop(δ)` uses `m̂_true` for evaluation. Zero GT access.

### Pilot-Gate Decision Rule (new, first-class)

1 clip, ≤ 1 GPU-hour. Four predefined outcome branches:
- **B1** (strong attack + clear attribution): proceed to full DAVIS-10 main.
- **B2** (strong attack + joint-distributed damage): proceed; paper reports attribution as joint only.
- **B3** (strong attack + no attribution): proceed; paper flags mechanism as open question.
- **B4** (weak attack or fidelity-infeasible): trigger L_suppress_pathway fallback. If fallback still < 0.30, STOP and pivot paper to "architecture-aware attack-surface analysis", honestly reporting the ceiling.

No post-hoc branch reshuffle.

### Fallback Loss (invoked only under B4)

```
L_suppress_pathway =
    α · ⟨ ∥h_attack_t − h_clean_t∥_2 ⟩_{t over m̂_true support}
  + β · ∥m_f0_attack − m_f0_clean∥_2^2
```

Replaces (NOT adds to) L_suppress.

### Restoration Attribution (unchanged from R2, fixed sign)

R1 / R2 / **R12** / R3 + B-control. Protocol: dominated / additive / jointly-non-identifiable.

### Two-tier Fidelity Budget (unchanged)

| Frame | ε_∞ | Fidelity |
|---|---|---|
| f0 (prompt) | 2/255 | SSIM ≥ 0.98 (hinge margin 0.02) |
| t ≥ 1 | 4/255 | LPIPS ≤ 0.20 (hinge) |

### Training Plan (unchanged)

100 PGD steps, stages 30/40/30, η=1/255 (stage 3: 0.5/255), sliding 30-frame backward window for T>40, bf16 autocast, STE uint8 quantize, best-feasible checkpoint.

### Validation

**Main table (5 rows)**: Clean / Ours / Uniform-δ / UAP-SAM2 per-clip / Ours-on-SAM2Long.
**Restoration attribution table**: R1 / R2 / R12 / R3 / B-control (signed).
**Embedded mechanism evidence**: bank-drop on attacked DAVIS-10 confirms `|ΔJ| < 0.02`.
**Appendix**: prompt-robustness, SAM2.1-Base transfer, DAVIS-30 extended.

### Compute & Timeline

- Pilot: 1 GPU-hour (decision gate).
- DAVIS-10 C1: ~2.5 GPU-hours.
- Restoration: ~0.5 GPU-hour.
- SAM2Long install + transfer: ~2-3 GPU-hours.
- Appendix: ~5-8 GPU-hours.
- **Total: ~12-15 GPU-hours on one Pro 6000; 3-4 focused days from PILOT-PASS to full results.**
