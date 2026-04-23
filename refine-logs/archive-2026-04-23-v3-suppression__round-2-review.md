# Round 2 Review

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db862-fbe6-7250-94f0-946474202938`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.5 |
| Method Specificity | 8.0 |
| Contribution Quality | 7.4 |
| Frontier Leverage | 8.0 |
| Feasibility | 7.2 |
| Validation Focus | 6.8 |
| Venue Readiness | 6.7 |
| **Weighted Overall** | **7.7 / 10** |

## Verdict

**REVISE.** "Credible, focused proposal. Not READY because the validation metric needs correction and the paper still needs empirical proof that the causal framing lifts it above a standard per-video adversarial attack."

## Drift Warning

**NONE, conditional on no DAVIS GT use during optimization or checkpoint selection.**

## Sub-7 Action Items

### IMPORTANT — Validation sign bug
`ΔJ_restore = J(attacked) − J(attacked + swap)` is backwards. If swap restores J, `J(attacked+swap)` increases, so current metric is negative when recovery works.
- **Fix**: `ΔJ_restore = J(attacked + swap) − J(attacked)`. Positive = restoration works.

### IMPORTANT — Restoration non-additivity
R1/R2 may be partially redundant (f0 and Hiera are coupled). Single-pathway swaps may be ambiguous.
- **Fix**: predefine interpretation with an R1+R2 joint (f0+Hiera) upper-bound restoration. Use it to check whether single swaps are interpretable or whether attribution is jointly distributed.

### IMPORTANT — Contribution framing
Novelty still depends on whether this is more than "white-box per-video PGD against SAM2". C2 as standalone risks competing with C1.
- **Fix**: make the paper's central claim the **full causal loop**: bank diagnosis → path-targeted attack → restoration proof. C2 (bank marginality) becomes MECHANISM EVIDENCE for C1, not a parallel contribution.

## Specific Checks

| Check | Result |
|---|---|
| GT-free optimization | Yes, but checkpoint selection must also use clean-SAM2 pseudo-labels/logits. State explicitly. |
| Dominant contribution sharper | "Substantially sharper. Diagnose → attack → verify is the right framing." |
| Restoration ablation causal | Yes, specific causal attribution test. Fix sign + specify tensor swap boundaries. |
| Feasibility concrete | Much more credible (STE, bf16, memory estimates, windowed backward, pilot gate). |
| Scope | SAM2Long main-row sanity check acceptable. Do not make SAM2-family transfer a primary claim until results exist. |

## Simplification Opportunities

1. Delete the decoy regime entirely from the main method unless pilot shows suppression fails.
2. Fold C2 fully into C1 as mechanism evidence; no standalone "bank marginality" contribution.
3. Keep SAM2Long as ONE stress-test row, not a transfer sweep.

## Modernization Opportunities

1. Replace hard-threshold pseudo-labels with clean-SAM2 **soft logits or confidence-weighted masks**.
2. If output loss underperforms, **replace** rather than add: use pathway-aligned Hiera/f0 feature-corruption losses.
3. No LLM/diffusion/RL needed.

## Raw Reviewer Response

<details>
<summary>Click to expand</summary>

**Anchor Check**
Preserved. The revised proposal still solves the 2026-04-23 anchor: processed-video dataset protection, same first-frame mask, no clean-suffix recovery claim, no bank-poisoning narrative, per-video pixel-only PGD, f0/current-frame causal pathway focus.

Caveat: SAM2Long in the main table is acceptable only as a secondary stress-test row, not part of the primary success condition.

**Scores**
| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.5 |
| Method Specificity | 8.0 |
| Contribution Quality | 7.4 |
| Frontier Leverage | 8.0 |
| Feasibility | 7.2 |
| Validation Focus | 6.8 |
| Venue Readiness | 6.7 |

**Weighted Overall: 7.7 / 10**

**Sub-7 Issues**
- **Validation Focus, 6.8, IMPORTANT:** `ΔJ_restore = J(attacked) − J(attacked + swap)` has the sign backwards. If clean swap restores tracking, `J(attacked + swap)` increases, so the metric becomes negative. Fix: define `ΔJ_restore = J(attacked + swap) − J(attacked)`.
- **Validation Focus, 6.8, IMPORTANT:** restoration swaps are much better than pathway removal, but R1/R2 can still be non-additive and partially redundant. Fix: predefine interpretation: f0-only, Hiera-only, bank-only, and optionally f0+Hiera as an upper-bound restoration if singles are ambiguous.
- **Venue Readiness, 6.7, IMPORTANT:** the story is sharper, but top-venue novelty still depends on whether this is more than "white-box per-video PGD against SAM2." Fix: make the paper's central claim the full causal loop: bank diagnosis → path-targeted attack → restoration proof. Do not let C2 become a separate contribution competing with C1.

**Specific Checks**
- **GT-free optimization:** yes, if checkpoint selection also uses clean-SAM2 pseudo masks/logits rather than DAVIS GT. This must be stated explicitly.
- **Dominant contribution:** substantially sharper. "Diagnose → attack → verify" is the right framing.
- **Restoration ablation:** yes, it is now a specific causal attribution test, not a blunt pathway-removal test. Fix the sign and specify tensor swap boundaries.
- **Feasibility:** much more concrete. STE, bf16, memory estimates, windowed backward, and pilot gate are credible enough for proposal stage.
- **Scope:** SAM2Long main-row sanity check is acceptable. Do not make SAM2-family transfer a primary claim until results exist.

**Simplification Opportunities**
- Delete the decoy regime entirely from the main method unless suppression fails.
- Fold C2 fully into C1 as mechanism evidence; avoid presenting "bank marginality" as a second standalone paper.
- Keep SAM2Long as one stress-test row, not a transfer sweep.

**Modernization Opportunities**
- Replace hard threshold pseudo-labels with clean-SAM2 soft logits or confidence-weighted masks.
- If output loss underperforms, replace rather than add: use pathway-aligned Hiera/f0 feature corruption losses.
- No LLM/diffusion/RL needed.

**Drift Warning**
NONE, conditional on no DAVIS GT use during optimization or checkpoint selection.

**Verdict: REVISE**

This is now a credible, focused proposal. It is not READY because the validation metric needs correction and the paper still needs empirical proof that the causal framing lifts it above a standard per-video adversarial attack.

</details>
