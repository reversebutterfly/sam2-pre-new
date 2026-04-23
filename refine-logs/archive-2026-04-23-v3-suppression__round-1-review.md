# Round 1 Review

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db862-fbe6-7250-94f0-946474202938`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.0 |
| Method Specificity | 6.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 8.0 |
| Feasibility | 6.5 |
| Validation Focus | 6.5 |
| Venue Readiness | 6.0 |
| **Weighted Overall** | **6.7 / 10** |

## Verdict

**REVISE.** "A valid pivot, but not READY. The core mechanism is plausible; the paper risk is that it becomes a polished standard adversarial PGD attack unless supervision, causal attribution, and 'SAM2-family' scope are tightened."

## Drift Warning

**Conditional.** The reframe itself does not drift. But if `m_true_t` is ground-truth DAVIS masks during optimization, the method drifts from the anchored input setting.

## Action Items

### CRITICAL (Method Specificity, 6.0)

**Supervision leakage** in `L_rank` — `m_true_t` / `m_decoy_t` undefined, potentially leaks DAVIS GT annotations at optimization time.

- **Fix**: define `m_true_t` = frozen clean-SAM2 pseudo masks (publisher runs SAM2 on the clean video once, uses its per-frame predictions as supervision). Define `m_decoy_t` algorithmically (offset from `m_true_t`, or just drop if not worth the extra cost). Either way, **no DAVIS GT at optimization time**.

### IMPORTANT (Contribution Quality, 6.0)

**Dominant method risks reading as "standard white-box per-video PGD on evaluated frames".**

- **Fix**: make C2 mandatory, not optional. Frame the novelty as **path-specific causal attack design**: measure (B2) which SAM2 pathway is causal → attack only that pathway → verify (via restoration experiments) that the attack's damage lives there. This three-step story is the paper.

### IMPORTANT (Feasibility, 6.5)

**100-step full-video SAM2 backprop + LPIPS + SSIM + fake quantization + best-feasible checkpointing may exceed 10 min/clip.**

- **Fix**: specify STE for quantization; specify memory strategy; specify resolution/windowing for DAVIS clips >30 frames; add a measured pilot gate before claiming 10-min.

### IMPORTANT (Validation Focus, 6.5)

**"Drop f0 conditioning" ablation does not prove the attack damage is on f0** — dropping f0 harms clean tracking too.

- **Fix**: use **counterfactual restoration** instead of dropping:
  - attacked video + CLEAN f0 embedding swapped in (does J recover to clean? → attack lives in f0)
  - attacked video + CLEAN current-frame Hiera features swapped in (does J recover? → attack lives in current-frame)
  - attacked video + bank dropped (control — bank was already marginal on clean)
  
  Restoration > removal for pathway attribution.

### IMPORTANT (Venue Readiness, 6.0)

**Novelty depends too much on execution magnitude.**

- **Fix**: either narrow scope to SAM2.1-Tiny only (honest, smaller claim), or promote SAM2.1-Base + SAM2Long transfer from appendix to a minimal main-table sanity check.

## Simplification Opportunities

1. **Delete `L_rank`** unless `m_decoy_t` has a clean annotation-free definition.
2. **Merge C2 into C1** — do not present mechanism attribution as optional.
3. **Replace hard SSIM "projection" language** with hinge penalties + best-feasible checkpoint (unifies all fidelity constraints into one framework).

## Modernization Opportunities

1. Use **clean-SAM2 pseudo-labels / logits as self-supervision** instead of DAVIS GT. Publisher runs SAM2 on clean video, uses its own outputs as the "supervision target" at optimization time. Consumer sees GT at evaluation time; they are disjoint.
2. Add **pathwise SAM2 feature losses / restoration tests** on Hiera features and f0 memory embeddings (grounded in SAM2's actual internal representations). Do NOT add LLM/diffusion.

## Raw Reviewer Response

<details>
<summary>Click to expand</summary>

**Scores**
| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.0 |
| Method Specificity | 6.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 8.0 |
| Feasibility | 6.5 |
| Validation Focus | 6.5 |
| Venue Readiness | 6.0 |

**Weighted Overall: 6.7 / 10**

**Sub-7 Issues**
- **Method Specificity, 6.0, CRITICAL:** loss uses `m_true_t` and `m_decoy_t`, but the anchor only gives clean video + first-frame mask. If `m_true_t` means DAVIS GT, the method leaks evaluation annotations. Fix: define `m_true_t` as frozen clean-SAM2 pseudo masks/logits from the same first-frame prompt; define `m_decoy_t` algorithmically or delete `L_rank`.
- **Contribution Quality, 6.0, IMPORTANT:** dominant method risks reading as "standard white-box per-video PGD on evaluated frames." Fix: make C2 mandatory, not optional, and frame novelty as path-specific causal attack design: f0/current-feature intervention, clean-vs-attacked pathway restoration, and why bank routes are excluded.
- **Feasibility, 6.5, IMPORTANT:** 100-step full-video SAM2 backprop with LPIPS, SSIM, fake quantization, and best-feasible checkpointing may exceed 10 min/clip. Fix: specify STE for quantization, memory strategy, resolution/windowing, and a measured pilot gate before claiming 10 min.
- **Validation Focus, 6.5, IMPORTANT:** current ablation "drop f0 conditioning" does not prove the attack damage is on f0/current pathways; dropping f0 harms clean tracking too. Fix: use counterfactual restoration: attacked video + clean f0 embedding, attacked video + clean current-frame embeddings, attacked video + bank dropped.
- **Venue Readiness, 6.0, IMPORTANT:** sharp but not yet top-venue ready because novelty depends too much on execution magnitude. Fix: either narrow title to SAM2.1-Tiny, or move SAM2.1-Base/SAM2Long transfer from appendix to a minimal main-table sanity check.

**Simplification Opportunities**
- Delete `L_rank` unless `m_decoy_t` has a clean, annotation-free definition.
- Merge C2 into C1; do not present mechanism attribution as optional.
- Replace hard SSIM "projection" language with hinge penalties + best-feasible checkpoint selection.

**Modernization Opportunities**
- Use clean SAM2 pseudo-labels/logits as self-supervision instead of DAVIS GT.
- Add pathwise SAM2 feature losses/restoration tests on Hiera and f0 memory embeddings; do not add LLM/diffusion.

**Drift Warning**
Conditional. The reframe itself does not drift. But if `m_true_t` is ground-truth DAVIS masks during optimization, the method drifts from the anchored input setting.

**Verdict: REVISE**

This is a valid pivot, but not READY. The core mechanism is plausible; the paper risk is that it becomes a polished standard adversarial PGD attack unless supervision, causal attribution, and "SAM2-family" scope are tightened.

</details>
