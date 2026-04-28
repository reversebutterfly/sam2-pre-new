# Ablation Design Review — Bundle B sub-session 5 prep

**Date**: 2026-04-26 (after dog 1-clip pilot landed +0.171 ΔJ)
**Reviewer**: Codex MCP gpt-5.4 (xhigh reasoning)
**Thread**: `019dc51a-c71a-7971-bece-116a592de2f5`
**Trigger**: User concern "ΔJ change is still not persuasive enough" — single-clip result is anecdote.

## Verdict (one-liner)

A clean paired story over **10 clips × 6 configs (≈80 runs, 8 GPU-hours)** is the smallest credible package. Don't blow the overnight on a full 2×2×2 factorial.

## Pilot anchor result (committed-and-tuning point — DO NOT report this on test data)

| Quantity | Value |
|---|---|
| Clip | dog (DEV; treat as contaminated for hyperparameter selection) |
| Profiled placement (Bundle A1) | [8, 18, 32] |
| A0 (insert-only, profiled placement) J-drop | 0.4953 |
| Stage 14 (Bundle A+B, alpha_paste + R_k) J-drop | 0.6665 |
| **ΔJ vs A0** | **+0.1712** |
| polish_applied / polish_reverted | True / False |

## The 80-run overnight matrix (gpt-5.4 prescription)

| Cell | Description | Runs |
|---|---|---|
| `A0_rand[s]` | A0 with random K=3 placement, 3 seeds (s ∈ {0, 1, 2}) | 10 × 3 = 30 |
| `A0_prof` | A0 with profiled placement | 10 |
| `OT_poisson_noR` | Stage 14, poisson compositor, R_k OFF | 10 |
| `OT_alpha_noR` | Stage 14, alpha_paste, R_k OFF | 10 |
| `OT_full` | Stage 14, alpha_paste, R_k ON (default Bundle B) | 10 |
| `OT_full_C2` | OT_full + LPIPS-native ν (Bundle C ablation) | 10 |
| **TOTAL** | | **80 runs** |

Cost: 80 × ~6 min = 8 GPU-hours overnight on 1 GPU; 4 hours if parallel across 2 GPUs.

## Comparisons enabled

| Pair | Conclusion |
|---|---|
| `A0_prof - mean(A0_rand)` | Bundle A1 (profiled placement) effect |
| `OT_full - A0_prof` | Stage 14 polish (Bundle A2 + B) total effect |
| `OT_alpha_noR - OT_poisson_noR` | B1 alpha_paste vs poisson (halo-fix) effect |
| `OT_full - OT_alpha_noR` | B2 R_k masked residual effect |
| `OT_full_C2 - OT_full` | C2 LPIPS-native ν effect |

## Statistical reporting (paired per-clip analysis)

For the headline ΔJ = `OT_full - A0_prof`:

- **Mean ΔJ + median ΔJ**
- **95% bootstrap CI on mean** (resample paired-difference distribution)
- **Exact paired permutation test** on mean ΔJ (primary)
- **Exact Wilcoxon signed-rank test** as robustness
- **Win count** (e.g., "8/10 clips improved" or "10/10 clips improved")
- **Scatter plot**: A0_prof J-drop vs OT_full J-drop with y=x reference line
- **DO NOT lead with Cohen's d** — paired ΔJ is more interpretable for attack papers

Reporting slices (compute once, report all):
1. **All 10 clips** (full set)
2. **Motion-valid 8 clips** (excluding car-roundabout + drift-straight, where A0 already collapses to ~0.03-0.04)
3. **Held-out (dog excluded)** — directly counters cherry-picking complaint at zero extra GPU cost

## Hyperparameter discipline (Q4 — critical)

**Freeze ALL hyperparameters BEFORE running the 10-clip overnight matrix.** Includes:
- R_k ε bound (8/255 default — frozen)
- R_k support dilate_px (4) + feather_sigma (3.0) — frozen
- α_max (0.30) — frozen
- max_disp_px (2.0) — frozen
- λ_residual_tv (0.001) — frozen
- LPIPS-native cap (0.35) — frozen
- Bridge length L (3 default) — frozen, OR enable bridge_search if confident

**Treat dog as DEV-CONTAMINATED.** It was used to validate the pilot. Don't tune anything else on dog. Held-out slice = 9 clips excluding dog.

**Sensitivity sweeps** (ε_R, α_max, dilate_px) are appendix-only. Run on dev clips ONLY, report full curve, never re-tune from sweep results to test set.

**DO NOT report worst-cell of a sweep.** That doesn't buy credibility. What buys credibility: predeclared default + full curve in appendix.

## Mock NeurIPS Review (after 10-clip + matrix)

### Strengths
- Clear paired gain over a strong A0 baseline
- Mechanism is understandable: placement + trajectory-guided bridge edits + masked residual
- Fidelity constraints are explicit and bounded (LPIPS ≤ 0.35 insert / 0.20 orig, SSIM ≥ 0.98 on f0)
- Component story testable with clean ablations
- Reproducible (deterministic seeds, exported processed frames)

### Weaknesses
- Single target model family (SAM2 only)
- Small-scale evaluation (DAVIS-10)
- Threat model is niche, easy to challenge ("inserted decoys are detectable")
- Many moving parts (placement profiler + Stage 14 with α/warp/anchor/delta/R_k/ν)
- Expensive placement-profile preprocessing (~5 GPU-hours per clip)
- No evidence vs trivial frame-removal / temporal-dedup detection

### Predicted Score
**5-6/10 (borderline weak reject ↔ borderline weak accept)** with the planned 10-clip + matrix.

### What moves toward 7+:
1. **Cross-VOS-model transfer** — show attack lifts on SAM2-large or another VOS model. Cost: re-eval on different checkpoint, no PGD rerun.
2. **Anti-removal / detection appendix** — naive temporal frame-dedup detector vs attacked frames. ~2 GPU-hours for 1 simple detector.
3. **Broader clip set** — add 10-20 more DAVIS-2017 valid clips. Profile cost is the bottleneck.
4. **Held-out confirmation** — already covered by reporting dog-excluded slice.

## Detectability objection (Q6)

User's draft response (publisher-side tripwire, out-of-scope detection) is **NOT SUFFICIENT alone**.

Reviewer will ask: "If I can remove the 3 inserts, why is this useful?"

Required:
- **Explicit threat-model paragraph** clarifying publisher-side tripwire (NOT stealth from publisher)
- **Cheap appendix experiment** vs naive temporal detector / frame-dedup heuristic

Possible cheap detector to test against:
- Forward LPIPS between consecutive frames; flag inserts as outliers
- Cumulative motion residual; inserts break temporal smoothness

If our inserts evade these naive detectors, the threat model is more credible. If they don't, frame-it as "publisher accepts trade-off; sophisticated detector is future work".

## Pre-committed Round 5 criterion

`Mean ΔJ ≥ +0.05` over A0 → continue (paper goes); `< +0.02` → cut δ permanently. dog at +0.17 is encouraging but **N=1 is not the criterion**.

The criterion applies to the **full 10-clip matrix** mean. Cell to evaluate: `OT_full mean - A0_prof mean` over 10 clips.

## Acceptance-lift priority (do tonight; cut from bottom if budget tightens)

| Priority | Cells | Total runs | Reason |
|---|---|---|---|
| **P1** | A0_prof + OT_full on 10 clips | 20 | The headline claim. mean ΔJ. |
| **P2** | OT_alpha_noR on 10 clips | 10 | Does R_k actually matter? Tests 6th-and-FINAL δ shot. |
| **P3** | OT_poisson_noR on 10 clips | 10 | Halo-fix real or cosmetic? |
| **P4** | A0_rand × 3 seeds on 10 clips | 30 | Placement story rigor. Without this, A0_prof = single random draw is weak. |
| **P5** | OT_full_C2 on 10 clips | 10 | Bundle C — likely low lift. **CUT FIRST** if budget tightens. |

If only P1+P2+P3 land, paper still has the core ablation skeleton.

## Critical bottleneck for the 10-clip overnight matrix

**Placement profile is REQUIRED for cells `A0_prof`, `OT_poisson_noR`, `OT_alpha_noR`, `OT_full`, `OT_full_C2`** (5 of 6 cells = 50 of 80 runs).

Currently only dog has profile. Camel + blackswan still profiling (ETA ~17:30 today). The other 7 clips (cows, bmx-trees, motocross-jump, dance-twirl, soapbox, car-roundabout, drift-straight) have NO profile.

**Profile cost ≈ 5 GPU-hours / clip.** 7 more clips × 5h = ~35 GPU-hours = ~1.5 days on 1 GPU; ~17h on 2 GPUs in parallel.

### Options to unblock the 10-clip matrix
| Option | Effort | Tradeoff |
|---|---|---|
| (a) Wait for full 10-clip profile (extend current job to all 10) | ~35 GPU-hours, 1-2 days | Cleanest, but delays paper |
| (b) Run ablations on profiled-clips-only (dog + camel + blackswan + any others profiled by then) | Ready when profiles land | N=3 too small for paired stats; need at least 8 |
| (c) Trim profile budget (beam_width=2, full_n_steps=20) for the 7 remaining clips | ~3 GPU-hours/clip × 7 = 21h | Risk: beam misses best K=3 subset; report degraded profile in caveats |
| (d) Skip A1 placement story for the 7 non-profiled clips; use random K=3 + Stage 14 | Available immediately | Loses Bundle A1 contribution claim on 70% of clips; weakens placement ablation |

**Recommendation**: launch (c) — start trimmed profile for remaining 7 clips on GPU 0 RIGHT NOW. The 7 clips parallel with the existing 3-clip job on GPU 1; finish ~21h from now. Tomorrow morning the full 10-clip profile + ablation matrix is ready.

Alternative: if user is OK with the 3-clip preliminary, we can run the matrix on dog/camel/blackswan only as a "preliminary 3-clip pilot" while extended profile runs.

## Saved Artifacts

- This document (review).
- gpt-5.4 thread `019dc51a-c71a-7971-bece-116a592de2f5` (may expire on MCP restart).
- Pilot result: `vadi_runs/v5_pilot_ss5_dog/dog/K3_top_R8_b-dup_l-dc_o-ad_d-post_s-fs__ot/results.json`.
