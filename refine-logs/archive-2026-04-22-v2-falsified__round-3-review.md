# Round 3 Review — 2026-04-22

- **Overall**: 8.6 / 10 (up from 7.9)
- **Verdict**: REVISE
- **Drift Warning**: NONE

| Dim | R1 | R2 | R3 |
|---|---|---|---|
| Problem Fidelity | 8 | 9 | 9 |
| Method Specificity | 6 | 7 | 8 |
| Contribution Quality | 6 | 8 | 9 |
| Frontier Leverage | 8 | 9 | 9 |
| Feasibility | 6 | 7 | 8 |
| Validation Focus | 6 | 7 | 8 |
| Venue Readiness | 6 | 7 | 8 |
| **Overall** | **6.6** | **7.9** | **8.6** |

## Remaining action items

### IMPORTANT

**I1. Schedule claim precision**: With `m_3 = 14` forced, we are not isolating pure period-6. We test "early FIFO-aligned seed inserts + one boundary-adjacent insert." Either name it that way OR add one matched-shape control / small offset sweep.

**I2. Resolution-invariant confidence lock**: `softplus(logsumexp(g_u) - τ_conf)` scales with spatial HW. Use `logmeanexp(g_u) = logsumexp(g_u) - log(HW)` so `τ_conf` is invariant to resolution/crop.

**I3. CVaR domain**: `CVaR_0.5(g_u · 1[C_u])^+` contaminated by zeros outside C_u. Write explicitly as CVaR over the set `{g_u(p) : p ∈ C_u}` (masked), state whether g_u is logit or probability.

### MINOR

**M1**: Justify `Q = [0.6, 0.2, 0.2]` in one line OR swap to margin form `softplus(γ + A^recent - A^ins) + λ · A^other`.

### Optional simplification
Margin-style L_stale is less elegant diagnostically but simpler and more robust if training noisy.

### Modernization: NONE
