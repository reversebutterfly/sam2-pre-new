# Research Review — Joint vs Only Framing (2026-04-27)

**Reviewer:** GPT-5.4 xhigh (Codex MCP)
**Thread:** `019dc51a-c71a-7971-bece-116a592de2f5` (R6 implementation+research)
**Trigger:** Post v4.1 retest of dog + bmx-trees. User question: "is joint better than only?"

---

## TL;DR

We do NOT have evidence for the broad headline "original-frame modification beats insert-only".
We DO have evidence for the narrower defensible claim
"an adaptive joint attack (with no-regret revert) weakly dominates insert-only at export and strictly improves on 75% of strong-A0 clips."

Reframe the paper as **"No-Regret Adaptive Joint Attack"** — make `polish_revert` part of the method, not a fallback.

---

## Data Recap (post v4.1 retest)

A0 baseline = K3_top + insert-only. Mean A0 J-drop ~0.537 across prior 10 clips.

v4.x = anchored Stage 14 with frozen-ν + auto-revert when joint hurts.

| Clip | A0 (only) | latest joint J-drop | polish_applied | joint > only? |
|---|---|---|---|---|
| camel | ~0.4-0.5* | **0.9589** (v4.0) | Yes | YES, +~0.5 (big) |
| libby | ~0.4-0.5* | 0.5813 (v4.0) | Yes | YES, +~0.1-0.2 |
| dog | 0.5005 | **0.6351** (v4.1) | Yes (v4.1 fix) | YES, +0.13 |
| bmx-trees | 0.5963 | 0.5987 (v4.1) | **No (revert)** | NO, ties via revert |

* camel/libby A0 not directly measured in dev-4; estimated from prior 10-clip data. The v4.0 polish_applied=True implies joint > A0 by construction.

**dev-4 mean**: joint ~0.694 vs only ~0.55-0.60. +0.10-0.15 mean lift.
**Win rate**: 3/4 (75%) strict improvement. 1/4 tie via revert. 0/4 strict regression.
**Top-clip share**: camel contributes ~35% of total lift — near the 40% reviewer warning threshold.

---

## Paper Claim Matrix (verdict from codex)

### What the data CURRENTLY supports

| Claim | Defensible? |
|---|---|
| Adaptive joint wrapper > insert-only at export | ✅ promising, needs 10-clip confirm |
| Adaptive wrapper achieves no-regret (joint ≥ only) | ✅ by construction (revert) |
| When polish applied, joint gives substantial gain | ✅ camel +0.5, dog +0.13 |
| Strict improvement on a substantial subset of clips | ✅ 3/4 = 75% on dev-4 |

### What the data does NOT yet support

| Claim | Reason |
|---|---|
| Joint method universally beats insert-only | bmx-trees revert is counterexample |
| Original-frame δ is uniformly beneficial | mechanism-level claim, would need polish_applied on most clips |
| Stage 14 is a consistent improvement | Same as above |
| Headline-level mean improvement | 4 clips too thin; needs 10-clip; camel outlier dominates |

---

## Reframing: "No-Regret Adaptive Joint Attack"

**Old framing (broken)**: "Joint method (insert + δ) beats insert-only baseline."
- Falsified by bmx-trees revert. Polish actively hurts on some clips.

**New framing (sharp)**: 
> "Original-frame modification is beneficial but clip-dependent. We therefore
> cast the final attack as an adaptive joint procedure that invokes Stage 14
> only when it improves the target-model objective, yielding export-time
> dominance over insert-only and substantial gains on a majority subset."

This converts the revert mechanism from "ad hoc fallback" → "core algorithmic design principle." White-box adaptive attacks have legitimate algorithmic freedom to choose between proposals.

---

## Two-Layer Claim Structure (paper main result)

### Layer 1 — Method-level (the wrapper)
> "Adaptive joint attack weakly dominates insert-only and strictly improves on X% of clips."

Required reports:
- mean paired lift
- median paired lift
- strict-improvement rate
- polish-applied rate

### Layer 2 — Mechanism-level (when polish kicks in)
> "When the joint polish is accepted, it yields substantial additional lift over insert-only."

Required reports:
- conditional mean gain on applied clips
- conditional median gain on applied clips

**Do NOT hide the distinction between raw Stage 14 proposal and final exported wrapper.** That separation makes the paper more credible, not less.

---

## Held-out 10-clip Acceptance Gates (next experiment)

### For headline claim "adaptive joint substantively beats insert-only"

| Gate | Full 10 clips | Held-out-7 (excl dev-4) |
|---|---|---|
| Strict improvement on | ≥ 5/10 | ≥ 4/7 |
| Mean paired lift | ≥ +0.05 | ≥ +0.04 |
| Median paired lift | > 0 | > 0 |
| Top clip share of total lift | < 40% | (n/a) |

**If pass**: paper headline "adaptive joint substantively beats insert-only".
**If fail (e.g. 3/10 wins, mean driven by camel)**: narrower claim "joint helps on some clips, wrapper prevents harm" — paper viable only if rest of paper is very strong.

---

## Required Reporting Schema

To be reviewer-proof, any results table must include:

| Metric | Why |
|---|---|
| Mean paired lift | headline number |
| Median paired lift | guards against outlier-driven mean |
| Win rate (% strict improvement) | clip-level signal |
| Polish-applied rate | how often the joint stage actually fires |
| Leave-one-out mean without top contributor (camel) | guards against single-clip dominance |
| Raw OT proposal vs A0 (per-clip) | mechanism-level transparency |
| Exported adaptive wrapper vs A0 (per-clip) | method-level claim |

---

## Strongest Defensible Claim Right Now (pre-10-clip)

> "On the development clips studied so far, the adaptive joint attack
> never underperforms insert-only at export and yields strict gains on 3
> of 4 clips, including substantial gains on camel and dog."

Honest limitation:
> "These gains currently come from an adaptive accept/revert wrapper
> around the joint stage; the raw original-frame modification stage is
> not yet shown to be uniformly beneficial across held-out clips."

---

## Prioritized TODO

| # | Task | GPU cost | Output |
|---|---|---|---|
| 1 | Finish v4.1 winners retest (camel + libby) | ~25-50 min | confirm v4.1 didn't break v4.0 wins |
| 2 | bmx-trees retry with `--oracle-traj-v4-lambda-keep-full 50` | ~30 min | last attempt to make 4/4 strong-A0 apply |
| 3 | **Held-out-7 + dev-4 = 10-clip eval** of v4.1 | ~5 GPU-hours overnight | the headline experiment |
| 4 | Compute paired A0 baseline on same 10 clips | ~3 GPU-hours | paired-lift comparison |
| 5 | Write up reporting schema (table + figure) | 0 (paper) | reviewer-proof presentation |

Items 3 + 4 should ideally share a single eval pass (run A0 + v4.1 back-to-back per clip, log both J-drops).

---

## Hard Constraints (CLAUDE.md, NOT negotiable)

- Paper direction LOCKED to positive-method decoy attack. NO audit pivot.
- Method LOCKED: insert + δ on originals. NOT pure suppression.
- v4 evolution driven by "joint must do no harm vs A0" — that constraint is the paper's correctness claim, not a tuning concern.

These constraints are the reason we ended up at the "no-regret adaptive wrapper" framing instead of dropping the joint approach entirely.

---

## Open Risks for the Held-out 10-clip

1. **camel-driven mean**: if the pattern holds (1 outlier carries the win), reviewers will discount.
2. **Strong-A0 clips revert**: if more than 25-30% of held-out clips revert, the wrapper claim weakens to "occasional rescue."
3. **Median ≤ 0**: if half the lift is concentrated in <50% of clips, paper becomes a "specific clips" story.

Mitigation: report the leave-one-out and median up front, don't bury them in appendix.

---

## Decision Point After 10-clip

| Outcome | Paper claim |
|---|---|
| ≥5/10 wins, mean ≥+0.05, median >0, top<40% | Full headline: "adaptive joint substantively beats insert-only" |
| 4/10 wins, mean ≥+0.03, median >0 | Modest claim + ablation paper |
| 3/10 wins or mean driven by 1-2 clips | Narrow claim only — paper viable only with strong other contributions |
| ≤2/10 wins or mean ≤0 | Reframe needed — possibly "vulnerability-aware insert-only is the contribution; joint is ablation" |

The bottom row violates CLAUDE.md "no audit pivot" if we lose the joint claim entirely. Mitigation if we hit that branch: still keep joint as the method, but reposition the paper around the *adaptive policy* itself rather than around the J-drop number. (This is a fallback, not a primary path.)

---

## Conversation thread

`019dc51a-c71a-7971-bece-116a592de2f5` (Codex MCP gpt-5.4 xhigh). Full Q&A + the v4.0 / v4.1 implementation conversation history is preserved there for follow-up rounds.
