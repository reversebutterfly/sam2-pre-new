# Round 1 Review

**Reviewer**: GPT-5.4 xhigh
**Thread**: `019dcd87-c42b-7b03-9139-34df6b6ebd89`
**Date**: 2026-04-27
**Verdict**: **REVISE** (6.0/10 overall)

---

## Scores

| Dimension | Score | Notes |
|---|---|---|
| Problem Fidelity | 6 | Conditional drift via accept/revert wrapper |
| Method Specificity | 6 | Decoy family + memory readout under-specified |
| Contribution Quality | 5 | Demoting outcome-critical components dishonest |
| Frontier Leverage | 7 | Modern enough; SAM2 attack surface is right |
| Feasibility | 8 | v4.1 already implemented |
| Validation Focus | 5 | A3 confounds; need memory-causality ablation |
| Venue Readiness | 5 | "First demonstration" overclaim |
| **Overall (weighted)** | **6.0** | REVISE |

---

## Critical Findings

### CRITICAL #1 — Conditional drift via wrapper
The `max(joint, A0)` accept/revert rule means on reverted clips, the final published attack is NOT the joint method anymore. This conditionally violates the anchored constraint "must use BOTH insertion AND original-frame δ".

**Fix (codex)**: Define the scientific method as the **raw joint attack**. Require positive median lift on raw joint (no wrapper) with a non-trivial applied rate. Report `accept/revert` only as a separate deployment policy column.

### CRITICAL #2 — Dishonest contribution demotion
Demoting placement search, L_keep_full, polish_revert to "implementation details" is dishonest because they appear outcome-critical (L_keep_full was the v4.1 hot-fix that made dog apply; polish_revert salvaged 50% of v4.0 dev-4 clips).

**Fix (codex)**: Reframe to ONE main contribution + TWO enabling components.
- Main: internal insertion can causally bias SAM2 memory; bridge δ extends bias.
- Enabling 1: placement search.
- Enabling 2: no-regression stabilization (L_keep_full).
- Drop "no-regret adaptive attack" as a named scientific contribution.

### CRITICAL #3 — A3 ablation confounded
"all-frames-δ-no-insert vs insert+bridge-δ" confounds memory writes / temporal discontinuity / sparsity / budget allocation.

**Fix (codex)**: Replace A3 with a **memory-causality ablation**. Keep same inserted visuals BUT disable/clear SAM2 memory writes on insert frames (or prevent inserts from being banked). If the attack effect collapses → "memory hijack" mechanism credible.

---

## Important Findings

### IMPORTANT #1 — Method specificity (decoy + memory readout)
"Semantic decoy", "traj+α+warp+R", "memory-feature-divergence trace" are under-specified.

**Fix (codex)**: 
- Lock decoy to one explicit family (e.g. duplicate-object spatially shifted by trajectory anchor offset, alpha-paste compositor with feathering parameters specified).
- Define optimized variables mathematically (full math notation for ν, δ, traj, α, R).
- Pre-register the memory readout: which SAM2 layer, what similarity (cosine), how it ties to failure propagation.

### IMPORTANT #2 — Venue novelty overclaim
"First demonstration" is too strong without causal ablation.

**Fix (codex)**: Narrow to "**evidence** of memory-mediated failure; bridge δ **increases persistence**" — defensible after causal ablation.

---

## Simplification Opportunities

1. Remove `polish_revert` from core contribution. Keep as deployment selector only.
2. Either own placement search as a method component, or simplify aggressively.
3. Fix decoy construction to one deterministic family.

## Modernization Opportunities

1. Use SAM2's own memory/readout signals as auxiliary loss or causal diagnostic.
2. Optional: frozen vision prior for decoy initialization/filtering only.

## Drift Warning

**NOT NONE**. Accept/revert wrapper causes conditional drift on reverted clips. Move it from "method" to "deployment policy".

---

## Action Items for Round 1 refinement (priority order)

| # | Priority | Action |
|---|---|---|
| 1 | CRITICAL | Reframe contribution: ONE main (memory-mediated failure mechanism) + TWO enabling (placement search, no-regression stabilization). Drop "no-regret adaptive attack" as named contribution. |
| 2 | CRITICAL | Define scientific method = raw joint. Wrapper = deployment policy. Headline gates apply to RAW joint. Report wrapper-selected result separately. |
| 3 | CRITICAL | Replace A3 with memory-causality ablation: disable memory writes on insert frames; if effect collapses → mechanism confirmed. |
| 4 | IMPORTANT | Specify decoy family mathematically (one family, one set of params). Pre-register memory readout (layer, similarity, frame-trace metric). |
| 5 | IMPORTANT | Narrow novelty claim from "first demonstration" → "evidence of memory-mediated failure" + "bridge δ increases persistence". |

---

## Raw Reviewer Response

<details>
<summary>Full Codex round 1 response (verbatim)</summary>

**Overall Score**: `6.0/10`

This is a plausible attack direction, and it is much better anchored than most early-stage adversarial proposals. But in its current form it reads more like a strong attack recipe with a mechanism story attached than a paper that has actually proven the claimed mechanism. That distinction matters a lot for AAAI.

1. Problem Fidelity: `6/10` — wrapper creates conditional escape from anchored problem.
2. Method Specificity: `6/10` — decoy and memory readout under-specified.
3. Contribution Quality: `5/10` — demoting outcome-critical components dishonest.
4. Frontier Leverage: `7/10` — modern enough.
5. Feasibility: `8/10` — v4.1 implemented.
6. Validation Focus: `5/10` — A3 confounded.
7. Venue Readiness: `5/10` — "first demonstration" overclaim.

**Critical fixes**: (1) raw joint = scientific method, wrapper = deployment policy. (2) reframe to 1+2 contribution structure. (3) replace A3 with memory-causality ablation.

**Simplifications**: (1) polish_revert deployment-only. (2) own placement search or simplify. (3) one decoy family.

**Modernization**: (1) SAM2 memory readout as auxiliary loss / diagnostic. (2) frozen vision prior for decoy init only.

**Drift Warning**: not NONE — wrapper causes conditional drift.

**Verdict**: REVISE.

</details>
