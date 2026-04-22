# Round 2 Review — 2026-04-22

- **Overall**: 7.9 / 10 (up from 6.6)
- **Verdict**: REVISE (need ≥ 9 for READY)
- **Drift Warning**: NONE

## Dimension scores

| Dim | R1 | R2 | Δ |
|---|---|---|---|
| Problem Fidelity | 8 | **9** | +1 |
| Method Specificity | 6 | **7** | +1 |
| Contribution Quality | 6 | **8** | +2 |
| Frontier Leverage | 8 | **9** | +1 |
| Feasibility | 6 | **7** | +1 |
| Validation Focus | 6 | **7** | +1 |
| Venue Readiness | 6 | **7** | +1 |

## Reviewer's status verdict

- Problem Anchor: preserved
- Dominant contribution: materially sharper
- Method simplicity: improved
- Frontier leverage: still appropriate

## CRITICAL action items remaining

### C1. Sign error in low-confidence lock
`softplus(-τ_conf - max(g_u))` rewards HIGH `max(g_u)` — OPPOSITE of intent. Fix: `softplus(max(g_u) - τ_conf)` with `τ_conf` as a logit ceiling. Also replace `max` (easily satisfied by 1-pixel spike) with `topk-mean` or `logsumexp`.

### C2. Three clocks formalization
`p_k = 6k + r` + "force one at prefix boundary" + "after originals ≈ {5,11,12}" does NOT form one reproducible schedule. Need: schedule defined on **write count in modified sequence**, algorithmically mapped to original-frame positions. Three clocks to name: (i) original-frame index, (ii) modified-sequence index, (iii) memory-write index.

## IMPORTANT action items

### I1. `L_stale` → 3-bin distribution
2-way ratio `log(A_clean_recent / A_insert_memory)` ignores where rest of attention goes. Use 3-bin `{insert, recent-clean, other}` + KL / cross-entropy target favoring insert.

### I2. Off-resonance deconfound
Period 4 vs 6 changes resonance AND last-insert-to-eval-start distance. Match K_ins + prefix length + last-insert-to-first-eval distance across schedules; vary only write periodicity.

### I3. Whole-suffix metrics
Optimize `L_rec` on f15..f21 window is fine for compute, but REPORTED rebound / post-loss AUC must be on f15..end, else claim is "delayed recovery" not "no recovery."

## MINOR

### M1. Specify `A_u` extraction
One sentence fixing layer/head aggregation + query set definition (e.g. "foreground queries = pixels inside flow-warped eroded clean mask C_u; attention averaged across all memory-attention heads in the final decoder block").

## Simplifications

- Replace full-frame `BCE(g_ins, 1[D_ins])` with ROI loss on decoy box + target box (no background supervision mass).
- Fidelity story collapses to: δ L∞ clamp + ν LPIPS + seam-band penalty. Drop SSIM/ΔE from optimization (keep as reported metrics only if not binding).
- Pick ONE flow stack (RAFT OR Unimatch).

## Modernization: NONE.
