# Round 4 Review

**Reviewer**: gpt-5.4 @ xhigh reasoning
**Thread**: `019db862-fbe6-7250-94f0-946474202938`
**Date**: 2026-04-23

## Parsed Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.8 |
| Method Specificity | 8.2 |
| Contribution Quality | 8.2 |
| Frontier Leverage | 8.3 |
| Feasibility | 7.7 |
| Validation Focus | 8.5 |
| Venue Readiness | 7.9 |
| **Weighted Overall** | **8.4 / 10** |

## Verdict

**REVISE, lightly.** "The proposal is structurally strong and probably at the ceiling before running the pilot. It is not READY because READY requires ≥ 9 and the central empirical claim remains unproven after prior falsification."

## Critical bug caught

**Sign error in fallback loss** (L_suppress_pathway):

As written:
```
L_suppress_pathway = α · ⟨∥h_attack_t − h_clean_t∥_2⟩_support
                   + β · ∥m_f0_attack − m_f0_clean∥_2^2
```

PGD minimizes L. Minimizing `∥h_attack − h_clean∥` pushes h_attack TOWARDS h_clean — this PRESERVES clean features, the OPPOSITE of attack. Correct form:

```
L_pathway_attack = -α · ⟨ 1 − cosine(h_attack_t, h_clean_t.detach()) ⟩_support
                 - β · ( 1 − cosine(m_f0_attack, m_f0_clean.detach()) )
```

or equivalently: `L = α·cosine(h_attack, h_clean.detach()) + β·cosine(m_f0_attack, m_f0_clean.detach())` minimized (pushes cosine → -1 = feature dissimilarity).

Clean features must be **detached** so gradients don't flow through the clean forward.

## Other checks

- Pre-committed pilot-gate: improves rigor, reduces post-hoc narrative risk. Does NOT fully lift venue readiness because B4 = different paper (no dataset-protection success).
- Fallback preserves "one trainable component" property (it replaces L_suppress, adds no module). Sign fix is the only issue.
- Confidence-weighted soft pseudo-mask supervision is sufficient modernization. No further need for LLM/diffusion/RL.
- Score stuck at 8.4 across R3 and R4 — **pre-pilot ceiling**.

## Drift Warning

**NONE** for B1/B2/B3. **B4 is an honest fallback project, but no longer the original attack-success paper.**

## Remaining Action

- Fix fallback loss sign (see above).
- Accept 8.4 as pre-pilot ceiling OR stop loop and proceed to pilot-as-experiment-gate.

## Raw Reviewer Response

<details>
<summary>Click to expand</summary>

**Scores**
| Dimension | Score |
|---|---:|
| Problem Fidelity | 8.8 |
| Method Specificity | 8.2 |
| Contribution Quality | 8.2 |
| Frontier Leverage | 8.3 |
| Feasibility | 7.7 |
| Validation Focus | 8.5 |
| Venue Readiness | 7.9 |

**Weighted Overall: 8.4 / 10**

**Answers**
1. The four-branch pilot gate improves rigor and reduces post-hoc narrative risk, but it does not fully lift venue readiness. B4 is a different paper: useful, honest attack-surface analysis, but it no longer satisfies the anchored success condition of dataset protection with large J&F drop.
2. The fallback preserves the one-trainable-component property because it replaces the loss and adds no module. But as written, if you minimize `L_suppress_pathway`, it preserves clean features rather than corrupting them. The sign/objective must be fixed.
3. Confidence-weighted soft pseudo-mask supervision is sufficient modernization. No need for LLM/diffusion/RL.
4. This is near the pre-pilot ceiling, but not at 9. The smallest non-empirical fix is to correct and specify the fallback objective: use negative clean-feature similarity / positive target corruption under descent, with clean features detached.
5. Verdict: **REVISE**, but only lightly.

**Required Fix**
The fallback should be written as an attack loss, e.g.

```text
L_pathway_attack =
  - α · mean_{t, support}(1 - cosine(h_attack_t, h_clean_t.detach()))
  - β · (1 - cosine(m_f0_attack, m_f0_clean.detach()))
```

or equivalently define the optimizer as maximizing the feature-distance objective. Right now the formula's sign is inconsistent with PGD descent.

**Drift Warning**
NONE for B1/B2/B3. B4 is an honest fallback project, but it is no longer the original attack-success paper.

**Verdict: REVISE**
The proposal is structurally strong and probably at the ceiling before running the pilot. It is not READY under your rule because READY requires ≥9 and the central empirical claim remains unproven after prior falsification.

</details>
