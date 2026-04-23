# Round 1 Review (VADI)

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db8a1-7059-76b1-9958-ba5edc222de5`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 7.5 |
| Method Specificity | 6.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 7.0 |
| Feasibility | 5.5 |
| Validation Focus | 6.0 |
| Venue Readiness | 5.0 |
| **Weighted Overall** | **6.3 / 10** |

## Verdict

**REVISE.** "Coherent with the v4 anchor, much less narratively false than FIFO bank poisoning. Biggest risk: becoming 'adaptive placement + dense δ PGD' rather than a clean method showing vulnerability-aware inserts are the causal intervention."

## Drift Warnings Issued

- **L_obj = softplus(object_score + 0.5) is suppression by another name.** Remove or make ablation-only.
- Dropping inserts = DRIFT.
- Returning to FIFO / bank poisoning / clean-suffix = DRIFT.
- Turning into a generator/scheduler system = contribution sprawl.

## Critical Fixes (P0)

| # | Weakness | Fix |
|---|---|---|
| F1 | Vulnerability scorer is ad hoc; weights undefined. | Pre-registered **rank-based robust-z normalization over 2-3 signals** (confidence drop, clean mask discontinuity, Hiera residual). Drop flow unless pilot justifies. Define original-time vs processed-time indices. |
| F2 | Decoy loss not contrastive — can raise decoy WITHOUT making SAM2 abandon true target. | Replace with **margin form**: `softplus(mean_true_logit - mean_decoy_logit + margin)` on insert + neighbor frames. |
| F3 | `surrogate_J_drop` for checkpoint selection undefined; could leak GT. | Define explicitly: pseudo-label J-drop against clean-SAM2 masks OR decoy-margin improvement. **No DAVIS GT anywhere.** |
| F4 | Dense δ over ALL originals may dominate → inserts decorative. | **Local δ support**: perturb only frames in `∪_k NbrSet(m_k)` (insert neighborhoods, ~3-4 frames each). Global δ as ablation only. |
| F5 | "Top-K beats random" alone smells like scheduler tuning. | Add **bottom-K placement baseline** + report absolute drops (not just ratios). Multiple random draws (≥5) per clip, paired bootstrap. |
| F6 | ν ε=8/255 may underuse LPIPS≤0.35 budget (inserts become adversarial noise). | Use LPIPS+TV+temporal as **real binding fidelity** on inserts. Relax ν ε or drop ε on inserts, let LPIPS do the constraining. |
| F7 | J-drop ≥ 0.35 with K≤3 inserts is unproven. | **Gated pilot**: 3 clips × {K=1 top, K=1 random, K=3 top, δ-only local}. If top doesn't clearly win → stop expanding. |

## Simplifications

- Collapse vulnerability scorer to 2-3 signals. Drop flow unless pilot shows it helps.
- **K=1 as mechanistic centerpiece; K=3 as stronger variant.** Paper's "clean story" uses K=1.
- Remove `L_obj` default (or justify why it's not suppression).
- Keep restoration minimal: R2 (Hiera at inserts) + R2b (Hiera all) + R3 (bank). Drop hook-taxonomy bloat.

## Modernization

- **Gradient-based vulnerability scoring** (one clean backward per candidate frame to estimate local decoy susceptibility). Still heuristic, still white-box, still GT-free, BUT more principled than hand-weighted sum.
- Diffusion/inpainting: only as a FIDELITY CEILING if LPIPS-constrained optimized inserts can't produce enough attack capacity. Not default.

## Raw Response

<details>
<summary>Click</summary>

Overall weighted score: **6.3 / 10, REVISE**. Coherent with v4 anchor, much less narratively false than FIFO bank poisoning, but not venue-ready. Biggest risk: "adaptive placement + dense δ PGD" rather than a clean method showing vulnerability-aware inserts are the causal intervention.

Sub-7 fixes:
- Vulnerability score ad hoc; weights/indexing undefined. → Pre-registered rank-based robust-z over 2-3 signals.
- Decoy loss not contrastive; can raise decoy without making SAM2 abandon true. → Margin form `softplus(mean_true − mean_decoy + margin)`.
- surrogate_J_drop undefined. → Pseudo-label drop; no GT.
- Dense δ may dominate. → Local δ to insert neighborhoods.
- Top-K vs random alone = scheduler tuning. → Add bottom-K; multi-draw random.
- ν ε=8/255 may underuse LPIPS budget. → LPIPS as real constraint, relax or drop ε on inserts.
- 0.35 J-drop unproven. → Gated 3-clip pilot.
- SAM2Long/UAP in main table distracting. → Move to appendix.

Simplifications: 2-3 scorer terms; K=1 centerpiece, K=3 stronger variant; remove L_obj default; minimal restoration hooks.

Modernization: gradient-based vulnerability scoring (one clean backward per candidate frame) as principled upgrade. No LLM/RL.

Drift warnings: dropping inserts, adding L_suppress, bank poisoning, clean-suffix — all DRIFT.

Verdict REVISE. Paper lives or dies on: "vulnerability-aware insert placement causes more damage than matched random/canonical placement, AND the damage routes through current-frame Hiera rather than memory-bank poisoning or dense δ". Prove that with pilot before adding any modules.

</details>
