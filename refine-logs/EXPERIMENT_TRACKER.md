# Experiment Tracker — MemoryShield

**Date**: 2026-04-22
**Planned total**: 18-24 GPU-hours on RTX Pro 6000 Blackwell

Statuses: `TODO` / `RUN` / `DONE` / `FAIL` / `GATED` (waiting on upstream)

## Must-run (paper core)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | sanity — pipeline end-to-end, loss forms, attention-mass extraction | Full, dog, K_ins=1, 50 steps | DAVIS-10:dog | no NaN, logs sane, P_u extractable | MUST | TODO | blocks all subsequent |
| R002 | M1 | ProPainter-base LPIPS realization-gap gate | Full, dog, K_ins=1, 200 steps | DAVIS-10:dog | insert LPIPS, J-drop | MUST | TODO | after R001 |
| R003 | M1 | Poisson-base LPIPS floor confirmation | Full-PoissonBase, dog, K_ins=1, 200 steps | DAVIS-10:dog | insert LPIPS, J-drop | MUST | TODO | after R001; for B5 comparison |
| R004 | M2 | UAP-SAM2 baseline reproduction | UAP-SAM2 trained | DAVIS-train → DAVIS-10 hard | mean J-drop, LPIPS, query footprint | MUST | TODO | reuse `reproduction_report.json` + training |
| R005 | M3.B1 | Full method on DAVIS-10 (seed 42) | Full, all clips, 200 steps | DAVIS-10 hard (10) | J-drop full suffix, rebound, post-loss AUC, LPIPS, SSIM | MUST | TODO | GPU0 |
| R006 | M3.B1 | Full method on DAVIS-10 (seed 43) | Full | DAVIS-10 hard | same | MUST | TODO | GPU0 after R005 |
| R007 | M3.B1 | Full method on DAVIS-10 (seed 44) | Full | DAVIS-10 hard | same | MUST | TODO | GPU0 after R006 |
| R008 | M3.B1 | Phase-1-only on DAVIS-10 (seed 42) | inserts only, δ=0 | DAVIS-10 hard | same metrics | MUST | TODO | parallel GPU1 |
| R009 | M3.B1 | Phase-2-only on DAVIS-10 (seed 42) | δ only, no inserts | DAVIS-10 hard | same metrics | MUST | TODO | parallel GPU1 after R008 |
| R010 | M3.B1 | Clean SAM2 eval on DAVIS-10 | no attack | DAVIS-10 hard | upper-bound J-drop = 0 | MUST | TODO | quick |
| R011 | M3.B2 | Resonance `m={6,12,14}` (already = Full in R005) | Full | DAVIS-10 hard | J-drop gap vs R012 | MUST | TODO | shares with R005 |
| R012 | M3.B2 | Off-resonance `m={4,8,14}` (matched recency) | Full, schedule=off-res | DAVIS-10 hard | J-drop, rebound | MUST | TODO | parallel GPU1 |
| R013 | M3.B2 | Offset sweep `m={5,11,14}` | Full, schedule=early-shift | DAVIS-10 hard | J-drop | MUST | TODO | after R012 |
| R014 | M3.B2 | Offset sweep `m={7,13,14}` | Full, schedule=late-shift | DAVIS-10 hard | J-drop | MUST | TODO | after R013 |
| R015 | M4.B3 | Full-no-L_stale (β=0) on DAVIS-10 | β=0 | DAVIS-10 hard | rebound, post-loss AUC, P_u @ f16/f17/f18 | MUST | TODO | after R005 |
| R016 | M4.B4 | SAM2Long eval on R005 attacked videos | SAM2Long, num_pathway=3 | DAVIS-10 hard attacked | SAM2Long J-drop, retention | MUST | TODO | after R005 |
| R017 | M4.B4 | SAM2Long clean eval | SAM2Long, clean input | DAVIS-10 hard | upper bound | MUST | TODO | cached if possible |
| R018 | M5.B6 | UAP-SAM2 attack → SAM2 eval for comparison | UAP-SAM2 attacked | DAVIS-10 hard | mean J-drop, LPIPS | MUST | TODO | after R004 |
| R019 | M6.B8 | Full DAVIS-30 final run | Full, all 30 clips | DAVIS-30 | full metric suite | MUST | TODO | after M3-M5 pass |

## Nice-to-have (appendix)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R020 | M7.B5 | ProPainter-base Full (already R005) vs Poisson-base Full | Full-PoissonBase | DAVIS-5 subset | LPIPS, J-drop | NICE | TODO | 5 clips |
| R021 | M7.B7 | τ_conf sweep ∈ {-1.0, -0.5, 0.0} | Full, vary τ_conf | 3 clips | J-drop, rebound | NICE | TODO | 3×3 = 9 runs |
| R022 | M7.B7 | β sweep ∈ {0.1, 0.3, 1.0} | Full, vary β | 3 clips | J-drop, rebound | NICE | TODO | 9 runs |
| R023 | M7.B7 | Q sweep ∈ {[.5,.25,.25], [.6,.2,.2], [.7,.15,.15]} | Full, vary Q | 3 clips | J-drop, rebound | NICE | TODO | 9 runs |
| R024 | M7.B9 | Failure-case qualitative | Full, selected failures | 2-3 hand-picked from DAVIS-10 | J trajectory, attention viz, insert inspection | NICE | TODO | visualization-heavy |

## First 3 runs to launch

1. **R001** (M0 sanity): dog K_ins=1 50 steps — verifies new loss forms and P_u extraction without burning budget.
2. **R002 + R003** (M1 realization-gap gate): dog K_ins=1 200 steps ProPainter-base vs Poisson-base. One GPU-hour total. **DECISION GATE** — if R002 insert LPIPS > 0.10 → relax the hard 0.10 bar BEFORE committing M3 runs.
3. **R004** (M2 UAP-SAM2 baseline): reproduce UAP-SAM2 on DAVIS-10 for positioning comparison. Can run GPU0 background while R001/R002 are on GPU0 sequentially.

## Gate logic

- R001 FAIL → fix loss implementation; halt.
- R002 PASS (LPIPS ≤ 0.10) → proceed M3 with LPIPS ≤ 0.10 claim.
- R002 FAIL (0.10 < LPIPS ≤ 0.15) → proceed M3 with relaxed LPIPS ≤ 0.15 claim documented.
- R002 FAIL (LPIPS > 0.15) → halt and revisit generator choice before M3.
- R005+R008+R009 any FAIL success criterion → halt, diagnose (likely β mis-tuned); do not proceed to M4.
- R015 no clear rebound gap → pivot: keep L_stale or swap to margin form; re-run once.
- R016 retention < 0.3 → halt, narrow claim scope; re-write paper intro before M6.

## Budget accounting

| Milestone | Run IDs | Est. GPU-hours | Parallel? |
|---|---|---|---|
| M0 | R001 | 0.25 | — |
| M1 | R002, R003 | 1.0 | seq on GPU0 |
| M2 | R004 | 2-3 | GPU1 parallel with M0/M1 |
| M3 | R005-R014 | 6-8 | GPU0+GPU1 parallel |
| M4 | R015, R016, R017 | 4 | GPU0 |
| M5 | R018 | ~0 (reuses R004) | — |
| M6 | R019 | 3-4 | GPU0 |
| M7 | R020-R024 | 3-4 | GPU1 while M6 on GPU0 |
| **Total must-run** | R001-R019 | **17-21** | |
| **Total with nice-to-have** | R001-R024 | **20-25** | |
