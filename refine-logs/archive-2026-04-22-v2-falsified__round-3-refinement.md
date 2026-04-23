# Round 3 Refinement (final polish before ready-check)

## Problem Anchor
[Unchanged — see PROBLEM_ANCHOR_2026-04-22.md]

## Anchor / Simplicity Check
- Anchor: preserved.
- Dominant contribution: unchanged (2-phase preprocessor against SAM2-family self-healing).
- Remaining changes are reproducibility polish, NOT new mechanisms.

## Changes

### I1. Schedule claim re-named + offset sweep added
- **Old name**: "FIFO-resonant schedule"
- **New precise name**: **"Write-aligned seed-plus-boundary schedule"** — K_ins-1 inserts placed on multiples of `num_maskmem - 1` (the FIFO period), plus one force-placed boundary insert adjacent to eval start.
- **Validation tightened**: Claim 4 now tests TWO comparisons (both matched on K_ins, prefix length, last-insert modified-index):
  - 4a: resonance `m = {6, 12, 14}` vs off-resonance `m = {4, 8, 14}` (period 6 vs 4)
  - 4b: offset sweep with canonical shape — `m ∈ {5, 11, 14}` / `{6, 12, 14}` / `{7, 13, 14}` — shows the mechanism depends on FIFO-period alignment, not just "put inserts somewhere."
- Claim now reads: "the write-aligned seed-plus-boundary schedule (period = num_maskmem − 1, K_ins − 1 early inserts + 1 boundary insert) dominates non-aligned schedules at matched budget and matched last-insert recency."

### I2. Resolution-invariant confidence lock
$$ L_\text{conf} = \text{softplus}(\text{logmeanexp}(g_u) - \tau_\text{conf}) $$
where `logmeanexp(g_u) = logsumexp(g_u) - log(HW)`. `τ_conf` now interpretable as a logit ceiling in natural units, invariant to spatial resolution.

### I3. CVaR domains made explicit (masked statistics)
All CVaR terms now defined over masked sets, not over full frame with 1[·] contamination.

**Phase 1 insert-side suppression**:
$$ \text{SuppressCore}_{\text{ins}_k} = \text{softplus}\big( \text{CVaR}_{0.5}\big( \{g_{\text{ins}_k}(p) : p \in C_{\text{ins}_k}\} \big) + m \big) $$

**Phase 2 clean-suffix suppression**:
$$ \text{Suppress}_u = \text{CVaR}_{0.5}\big( \{g_u(p) : p \in C_u\} \big)^+ $$

All `g_u` are **logit maps** (pre-sigmoid). Masked CVaR_0.5 = median-gated top half inside the region, not contaminated by zeros outside.

### M1. Q rationale line added + ablation fallback
Q = [0.6, 0.2, 0.2] chosen so that (i) insert slots exceed recent-clean with margin (0.6 vs 0.2), (ii) `other` has non-negligible target preventing collapse of bank attention onto any single slot type. This is a regularization target, not a learned distribution. If training proves noisy, fall back to margin form:
$$ L_\text{stale}^\text{margin} = \text{softplus}(\gamma + A^\text{recent} - A^\text{ins}) + \lambda \cdot A^\text{other} $$
One-line sensitivity ablation can be added: test Q ∈ {[0.5, 0.25, 0.25], [0.6, 0.2, 0.2], [0.7, 0.15, 0.15]} on 3 clips, report stability of attack result.

## Final Loss (polished)

$$
\boxed{\; L(\nu, \delta) = L_\text{loss} + \lambda_r L_\text{rec} + \lambda_f L_\text{fid} \;}
$$

**Phase 1**:
$$ L_\text{loss} = \frac{1}{K_\text{ins}} \sum_k \Big[ \text{BCE}_\text{ROI}(g_{\text{ins}_k}, \mathbb{1}[D_{\text{ins}_k}]) + \alpha \cdot \text{softplus}\big(\text{CVaR}_{0.5}(\{g_{\text{ins}_k}(p) : p \in C_{\text{ins}_k}\}) + m\big) \Big] $$

**Phase 2**:
$$ L_\text{rec} = \frac{1}{|U|} \sum_u \Big[ \alpha_\text{supp} \cdot \text{CVaR}_{0.5}(\{g_u(p) : p \in C_u\})^+ + \alpha_\text{conf} \cdot \text{softplus}\big(\text{logmeanexp}(g_u) - \tau_\text{conf}\big) \Big] + \beta \cdot L_\text{stale} $$

**L_stale** (3-bin, with margin fallback):
$$ L_\text{stale} = \frac{1}{|V|} \sum_u \text{KL}(Q \| P_u), \quad Q = [0.6, 0.2, 0.2] $$

**L_fid**:
$$ L_\text{fid} = \mu_\nu (\text{LPIPS}(x_{\text{ins}_k}, f_{\text{prev}_k}) - 0.10)^+ + \mu_s \Delta E_\text{seam} $$

δ hard-L∞-clamped per step.

## Final Schedule

Canonical (paper): `m = {6, 12, 14}` → inserts after original {5, 10, 11}.
Off-resonance ablation: `m = {4, 8, 14}` → {3, 6, 11}.
Offset sweep: `m = {5,11,14} / {6,12,14} / {7,13,14}` for extra robustness.

## Everything else unchanged from round-2-refinement.md
- PGD stages 1-40 ν-only → 41-80 δ-only → 81-200 joint
- δ L∞ = 4/255 (f0: 2/255), ν LPIPS ≤ 0.10
- 3 clocks explicit, A_u extraction spec'd
- ROI-BCE on D_box ∪ C_box dilated 10px
- RAFT only, SSIM reported not optimized
- Whole-suffix f15..end reporting
- Mandatory SAM2Long transfer
