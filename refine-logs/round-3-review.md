# Round 3 Review

**Reviewer**: GPT-5.4 xhigh (same thread)
**Thread**: `019dcd87-c42b-7b03-9139-34df6b6ebd89`
**Date**: 2026-04-27
**Verdict**: **REVISE** (7.8/10, +0.6 from R2)

## Scores

| Dimension | R1 | R2 | **R3** | Δ R2→R3 |
|---|---|---|---|---|
| Problem Fidelity | 6 | 8 | **9** | +1 |
| Method Specificity | 6 | 7 | **8** | +1 |
| Contribution Quality | 5 | 7 | **7** | 0 |
| Frontier Leverage | 7 | 8 | **8** | 0 |
| Feasibility | 8 | 7 | **8** | +1 |
| Validation Focus | 5 | 6 | **8** | +2 |
| Venue Readiness | 5 | 6 | **6** | 0 |
| **Overall** | 6.0 | 7.2 | **7.8** | +0.6 |

**Drift**: NONE.

## Key codex assessments

- **A1 fix**: correctly isolating bridge δ (with caveat: enforce same upstream W* and ν, ONLY zero bridge variables in control)
- **d_mem fix**: largely free of circularity. Remaining issue is arbitrariness (top-32, c_K_clean) — pre-registered + appendix sensitivity is enough
- **A3 pre-registration**: honest, sufficient
- **A3 first**: right scientific sequencing
- **READY blocker**: empirical dependence of headline on A3 + nontrivial search dependence. "Now a credible AAAI candidate contingent on results, not READY-level by proposal quality alone."

## Remaining issues (action items)

### IMPORTANT — Operationally lock A1
Same W*, same ν, same fidelity accounting; **ONLY** bridge variables (traj, α, warp, R) zeroed. Add pseudocode showing the same upstream branch.

### IMPORTANT — Pre-register hook behavior for A3
`BlockInsertMemoryWritesHook` must specify EXACTLY what it modifies (so reviewers can't claim intervention changed more than memory writes).

### IMPORTANT — Conditional paper framing pre-committed
Title / abstract / claim language explicitly conditional on A3 outcome:
- Strong pass: "dominant failure mode"
- Partial pass: "substantial memory-mediated component"
- Fail: workshop pivot
Pre-register both templates.

### MINOR — Main vs deployment table separation
Raw-joint = main table. Wrapper-selected = separate "Deployment" column / table.

### MINOR — E1 honest engineering label
Don't hide E1; don't elevate it to compete with C1; just label as "the empirically necessary search procedure for placement."

### MINOR — Appendix sensitivity on T_obj (16/32/64 tokens)
Strengthens diagnostic, no main-story bloat.

### MINOR — Compress traj+α+warp+R in main paper
Use "masked bridge-edit parameterization" in main; full equations in Appendix B.

## Raw response

<details>
<summary>Codex R3 (verbatim, key passages)</summary>

Scores: PF 9, MS 8, CQ 7, FL 8, F 8, VF 8, VR 6. Overall 7.8.

A1 is now correctly isolating bridge δ, with caveat: W* and ν must be frozen from same upstream branch, control arm only zeros/disables bridge-edit variables.

d_mem fix largely free of circularity. Top-32 and c_K_clean are design choices, acceptable if pre-registered with appendix sensitivity.

Dual-threshold A3 pre-registration honest. Strong/partial/fail much better than post-hoc reframing.

Putting A3 first is right scientific sequencing.

Contribution structure now sharper but not at AAAI ≥9 ex ante. Credible AAAI candidate contingent on results, not READY by proposal quality alone.

Why not READY: empirical dependence on A3 + nontrivial search dependence. C1 properly framed but paper lives/dies on memory-write-blocking collapse magnitude.

Action items:
1. Lock A1 operationally
2. Pre-register exact hook behavior for A3
3. Final paper framing conditional on A3 strong vs partial
4. Raw-joint main, wrapper deployment column
5. E1 openly necessary engineering, not hide

Verdict: REVISE.

</details>
