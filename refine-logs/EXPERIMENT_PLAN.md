# Experiment Plan — MemoryShield Preprocessor

- **Problem**: User-controlled preprocessor that takes a clean video + first-frame target mask, outputs modified video causing SAM2-family promptable VOS to lose the protected target and not recover, within insert LPIPS ≤ 0.10 and attacked-originals SSIM ≥ 0.97.
- **Method thesis**: Phase 1 inserts (3 frames, write-aligned seed-plus-boundary schedule) create a tracking-loss event; Phase 2 prefix perturbations (f0..f14, L∞ ≤ 4/255) + `L_stale` 3-bin KL on memory-attention prevent recovery. Per-video PGD, no runtime hook.
- **Date**: 2026-04-22
- **Proposal**: `refine-logs/FINAL_PROPOSAL.md` (score 9.2/10, verdict READY)

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| **C1 (primary)** — Two-phase composition defeats FIFO self-healing | Dominant contribution; single-component attacks provably fail under SAM2 memory rollover | 4-condition ablation on DAVIS-10: clean / Phase-1-only / Phase-2-only / Full. Each single-phase ≤ 50% of Full J-drop. Full rebound ≤ 0.15 over full suffix f15..end | B1, B3 |
| **C2 (supporting)** — Write-aligned seed-plus-boundary schedule is the right placement rule | Proves the mechanism story (FIFO period matters) is causal, not descriptive | Resonance `m={6,12,14}` vs off-resonance `m={4,8,14}` at matched last-insert recency: gap ≥ 20pp J-drop. Offset sweep `{5,11,14}/{6,12,14}/{7,13,14}` peaks at canonical | B2 |
| **Anti-claim 1** — "gain is only from ProPainter realistic base, not from the two-phase mechanism" | Reviewer will attack if uncontrolled | Appendix: Full(ProPainter) vs Full(Poisson-base) — mechanism still beats self-heal under both bases, only fidelity differs | B5 |
| **Anti-claim 2** — "gain is just larger perturbation budget" | Reviewer will attack | Phase-1-only (inserts, no δ) and Phase-2-only (δ, no inserts) are budget-matched to each other in Block 1; each individually fails to break self-heal | covered in B1 |
| **Anti-claim 3** — "`L_stale` is decoration, any Phase-2 loss works" | Reviewer will attack | Full vs Full-no-L_stale (Claim 2 in proposal): rebound gap ≥ 0.2, bank-attention `A^ins` collapses without L_stale | B3 |
| **Transfer claim** — Mechanism generalizes to SAM2-family | Needed for the "SAM2-family" paper framing | Attack videos from B1 Full → SAM2Long (num_pathway=3). SAM2Long J-drop ≥ 0.25, retention ≥ 0.40 | B4 |

## Paper Storyline

**Main paper must prove**:
- Table 1: 4-condition composition ablation + rebound/post-loss AUC (C1 necessity of two-phase)
- Table 2: SAM2Long transfer (generalization)
- Table 3 (or Fig): schedule ablation 4a (resonance vs off-resonance) + sweep 4b
- Table 4: L_stale ablation + bank-attention P_u breakdown (the mechanism proof)
- Figure: per-frame J trajectory for 3 representative clips (blackswan / cows / bmx-trees) showing Full holds, singles recover
- Figure: UAP-SAM2 baseline comparison at matched LPIPS

**Appendix can support**:
- ProPainter-vs-Poisson base study (defends simplicity — mechanism ≠ generator choice)
- τ_conf / β / Q sensitivity sweeps (pre-registered ranges)
- Per-clip detailed results DAVIS-30
- Failure-mode qualitative analysis on natural-distractor clips

**Experiments intentionally cut**:
- Learned scheduler comparison (proposal explicitly rejects; no need to compare against something we refuse to build)
- LLM / VLM / diffusion primitive ablation (proposal is intentionally non-frontier beyond ProPainter; no "frontier necessity check" needed)
- Cross-family transfer to non-SAM2 VOS (scope narrowed in refine Round 1)
- Runtime attack / dynamic protection scenarios (threat model is offline preprocessor)

## Experiment Blocks

### Block 0 — Pre-gate: ProPainter realization-gap pilot
- **Claim tested**: prerequisite feasibility — can ProPainter + `ν` PGD reach insert LPIPS ≤ 0.10 WHILE `L_loss` converges on a single easy clip?
- **Why this block exists**: prior fidelity loop showed Poisson-base LPIPS floor ≈ 0.13-0.14. If ProPainter base does not close the gap under ν PGD, LPIPS ≤ 0.10 is not achievable and the whole plan rescopes to LPIPS ≤ 0.15 with explicit disclosure.
- **Dataset / split**: DAVIS-10, `dog` clip (easy non-distractor, large object)
- **Compared systems**: ProPainter-base-Full vs Poisson-base-Full at 200 PGD steps, measured at single insert slot m=6
- **Metrics**: per-frame insert LPIPS, per-slot final LPIPS, J-drop on eval f15..f21 (sanity)
- **Setup**: SAM2.1-Tiny, RTX Pro 6000, DAVIS 480p, batch 1, K_ins=1, prefix=15
- **Success criterion**: ProPainter-base final LPIPS ≤ 0.10 AND J-drop ≥ 0.3 on insert window
- **Failure interpretation**: if LPIPS floor > 0.10, the LPIPS ≤ 0.10 claim in FINAL_PROPOSAL must be relaxed to ≤ 0.15; entire plan proceeds but paper fidelity claim softened
- **Table / figure target**: not in paper; a single row in REFINEMENT_REPORT for decision-gate record
- **Priority**: MUST-RUN FIRST (1 GPU-hour gate)

### Block 1 — Main: 4-condition composition ablation (C1)
- **Claim tested**: C1 — two-phase composition defeats FIFO self-healing; each phase individually insufficient
- **Why this block exists**: the dominant paper claim. Without this, nothing else matters.
- **Dataset / split**: DAVIS-10 hard subset (blackswan, breakdance, bmx-trees, bike-packing, camel, car-roundabout, cows, dance-twirl, dog, car-shadow)
- **Compared systems** (4):
  - `Clean` (no attack)
  - `Phase-1-only` (3 inserts + ProPainter base + `L_loss`; no δ; δ_orig=0)
  - `Phase-2-only` (no inserts, but δ on f0..f14; `L_rec` + `L_stale`; schedule-related losses degrade to on-original supervision)
  - `Full` (inserts + δ + full `L`)
- **Metrics**:
  - Primary: mean J-drop over full suffix f15..end (per-clip, averaged)
  - Primary: `post-loss AUC = (1/|U_late|) Σ J_atk(u)` for u ≥ first-loss-frame
  - Primary: `rebound = max_{u∈U_late}(J_clean - J_atk) - min_{u∈U_early}(J_clean - J_atk)` (low = no recovery)
  - Secondary: per-clip J trajectory (qualitative)
  - Secondary: insert LPIPS, orig LPIPS, orig SSIM (confirms fidelity triad)
- **Setup**: per-video PGD 200 steps, Stages 1(40)/2(40)/3(120). DAVIS 480p, SAM2.1-Tiny, 3 seeds (42, 43, 44). Pro 6000 GPU0.
- **Success criterion** (all must hold):
  - `Full` mean J-drop ≥ 0.55
  - `Phase-1-only` mean J-drop ≤ 50% of Full
  - `Phase-2-only` mean J-drop ≤ 50% of Full
  - `Full` rebound ≤ 0.15
  - `Phase-1-only` OR `Phase-2-only` rebound ≥ 0.30 (self-heal visible)
  - Fidelity: insert LPIPS ≤ 0.10 (or relaxed 0.15 per B0), orig SSIM ≥ 0.97
- **Failure interpretation**:
  - If Full < 0.55 but singles comparable → method not strong enough; re-tune β / prefix length / K_ins
  - If singles ≈ Full → composition unnecessary; this is a strong negative for the paper; consider scope pivot
- **Table / figure target**: Paper Table 1 + Paper Figure 2 (J trajectory)
- **Priority**: MUST-RUN

### Block 2 — Schedule ablation (C2, two sub-experiments)
- **Claim tested**: C2 — write-aligned seed-plus-boundary schedule (period = num_maskmem - 1 = 6) is causally necessary, not arbitrary
- **Why this block exists**: without this, Claim 4 reduces to "we picked some positions and it worked"; paper story weakens
- **Dataset / split**: DAVIS-10 hard subset, Full method
- **Compared systems**:
  - **2a**: resonance `m={6,12,14}` vs off-resonance `m={4,8,14}` (same K_ins=3, same prefix=15, same ε, same LPIPS, same last-insert recency since m_3=14 in both)
  - **2b**: offset sweep `m ∈ {5,11,14} / {6,12,14} / {7,13,14}` showing alignment peak
- **Metrics**: mean J-drop over full suffix, rebound, post-loss AUC
- **Setup**: per-video PGD 200 steps, Full method, 3 seeds, DAVIS-10. Pro 6000 GPU1 (parallel with B1).
- **Success criterion**:
  - 2a: canonical beats off-resonance by ≥ 20pp J-drop
  - 2b: sweep peaks at `{6,12,14}` (canonical) with monotone drop on either side
- **Failure interpretation**:
  - If 2a gap < 10pp → schedule claim is wrong; pivot to "any dense enough schedule works" (weaker paper framing)
  - If 2b has no clear peak → resonance is real but not specifically period-6; reframe as "coarse alignment" instead of precise resonance
- **Table / figure target**: Paper Table 3 (2a) + Paper Figure 3 or Table 3b (sweep)
- **Priority**: MUST-RUN

### Block 3 — L_stale necessity + mechanism diagnostic
- **Claim tested**: anti-claim 3 — `L_stale` is not decoration; specifically the mechanism keeping bank-attention on insert slots
- **Why this block exists**: reviewer explicitly asked this in Round 2 / 3. Without it, "Phase 2 internal regularizer" is not justified.
- **Dataset / split**: DAVIS-10 hard, Full vs Full-no-L_stale (β=0)
- **Compared systems**:
  - `Full` (β as tuned)
  - `Full-no-L_stale` (β=0; everything else identical)
- **Metrics**:
  - Primary: rebound + post-loss AUC
  - Primary: P_u breakdown `[A^ins, A^recent, A^other]` at f16, f17, f18 (measured by logging memory-attention mass)
  - Secondary: mean J-drop
- **Setup**: per-video PGD 200 steps, Full method with β∈{tuned, 0}, DAVIS-10, 3 seeds. Pro 6000 GPU0 (after B1).
- **Success criterion**:
  - Full: `A^ins ≥ 0.5` on V = {f16, f17, f18} (bank attention locked on inserts)
  - Full-no-L_stale: `A^ins ≤ 0.2`
  - Rebound gap (Full-no-L_stale rebound − Full rebound) ≥ 0.2
- **Failure interpretation**:
  - If Full-no-L_stale rebound is low AND J-drop similar to Full → L_stale is redundant; remove it and simplify method claim
  - If bank-attention numbers noisy → swap to margin-form L_stale (pre-declared fallback in proposal)
- **Table / figure target**: Paper Table 4 + attention-mass heatmap figure
- **Priority**: MUST-RUN

### Block 4 — SAM2Long transfer (mandatory per reviewer)
- **Claim tested**: transfer claim — mechanism generalizes within SAM2-family, not tied to SAM2.1-Tiny output head
- **Why this block exists**: Round 1 reviewer elevated this from optional to mandatory; paper framing "SAM2-family" demands it.
- **Dataset / split**: B1 Full attack videos evaluated under SAM2Long (num_pathway=3, iou_thre=0.1, uncertainty=2) on DAVIS-10
- **Compared systems**:
  - Clean SAM2Long (upper bound)
  - Attacked video (from B1 Full) → SAM2Long
- **Metrics**:
  - Primary: mean SAM2Long J-drop on full suffix
  - Primary: retention = SAM2Long J-drop / SAM2 J-drop
  - Secondary: per-clip breakdown
- **Setup**: SAM2Long checkpoint already on Pro 6000 at `~/SAM2Long/`; reuse prior eval pipeline from `scripts/sam2long_eval.py`. 1 seed (videos fixed from B1).
- **Success criterion**:
  - SAM2Long mean J-drop ≥ 0.25
  - Retention ≥ 0.40
- **Failure interpretation**:
  - Retention < 0.3 → attack is SAM2-specific (likely output-head coupled); narrow claim further to "SAM2.1-Tiny streaming VOS"
- **Table / figure target**: Paper Table 2 (transfer)
- **Priority**: MUST-RUN

### Block 5 — Appendix: ProPainter vs Poisson base (simplicity / anti-claim 1)
- **Claim tested**: anti-claim 1 — gain is from the two-phase mechanism, not just from a nicer insert base
- **Why this block exists**: defends simplicity — "we could have used a better generator, but the mechanism is what drives the attack"
- **Dataset / split**: DAVIS-10 hard (subset of 5 clips to save compute; choose blackswan, dog, camel, bike-packing, car-roundabout)
- **Compared systems**: Full (ProPainter base) vs Full (Poisson base, legacy) at matched everything else
- **Metrics**: insert LPIPS, mean J-drop, rebound
- **Setup**: reuse B1 pipeline; swap insert base generator. 1 seed.
- **Success criterion**: both beat Clean; LPIPS on ProPainter base lower (obvious); J-drop similar magnitude or ProPainter slightly higher (mechanism works under both)
- **Failure interpretation**: if Poisson-base Full J-drop ≫ ProPainter Full → surprising; suggests realization-gap matters more than expected; needs further analysis
- **Table / figure target**: Appendix table
- **Priority**: NICE-TO-HAVE

### Block 6 — UAP-SAM2 baseline comparison
- **Claim tested**: positioning vs the closest published work
- **Why this block exists**: reviewer will ask; we MUST have a side-by-side comparison at matched fidelity
- **Dataset / split**: DAVIS-10 hard
- **Compared systems**:
  - UAP-SAM2 (reproduced from our prior work — see `reproduction_report.json`)
  - MemoryShield Full (from B1)
- **Metrics**: mean SAM2 J-drop at comparable visible-perturbation budget (UAP-SAM2 uses ε=10/255 on ALL frames; MemoryShield uses ε=4/255 on 15 frames + insert LPIPS ≤ 0.10). Report both J-drop and total modified-pixel budget.
- **Setup**: UAP reproduced locally; evaluate on identical DAVIS-10 videos. 1 seed (UAP is deterministic once trained).
- **Success criterion** (positioning, not strict dominance):
  - MemoryShield reaches comparable mean J-drop to UAP-SAM2 on protected target
  - MemoryShield modifies ≪ pixels than UAP-SAM2 (locality advantage)
  - Per-target targeting advantage evident (MemoryShield fails other objects less)
- **Failure interpretation**:
  - MemoryShield J-drop ≪ UAP-SAM2 → reposition paper as "targeted protection" rather than "stronger attack"
- **Table / figure target**: Paper Table 1 (side row in main composition table)
- **Priority**: MUST-RUN

### Block 7 — Pre-registered sensitivity sweeps (`τ_conf`, `β`, `Q`)
- **Claim tested**: robustness to hyperparameter choice (defends against "heavily tuned")
- **Why this block exists**: reviewer Round 4 non-blocking note pre-registered ranges; must not look post-hoc.
- **Dataset / split**: 3 clips (dog, cows, bmx-trees) covering easy / distractor / hard
- **Compared systems**:
  - `τ_conf ∈ {-1.0, -0.5, 0.0}` (3 values, fix others)
  - `β ∈ {0.1, 0.3, 1.0}` (3 values)
  - `Q ∈ {[0.5, 0.25, 0.25], [0.6, 0.2, 0.2], [0.7, 0.15, 0.15]}` (3 values)
- **Metrics**: mean J-drop, rebound
- **Setup**: Full method on 3 clips, 1 seed each. 9 runs per dimension × 3 dimensions = 27 total but most share work via cached embeddings.
- **Success criterion**: J-drop varies < 0.10 across each sweep; best configuration stable.
- **Failure interpretation**:
  - If J-drop varies > 0.15 across Q → swap to margin-form L_stale (pre-declared fallback)
  - If τ_conf or β instability → tighten sensitivity analysis language in paper
- **Table / figure target**: Appendix table
- **Priority**: NICE-TO-HAVE (but reviewer recommends strongly)

### Block 8 — Full DAVIS-30 headline numbers
- **Claim tested**: all claims (final paper numbers)
- **Why this block exists**: paper needs full-benchmark headline, not just DAVIS-10 subset
- **Dataset / split**: DAVIS-2017 val full 30 clips
- **Compared systems**: Clean / UAP-SAM2 / MemoryShield Full
- **Metrics**: mean J-drop, rebound, post-loss AUC, insert LPIPS, orig LPIPS/SSIM, SAM2Long J-drop + retention
- **Setup**: after B1-B4 confirm DAVIS-10 numbers; extend to 30 clips using identical pipeline. 1 seed.
- **Success criterion**: match DAVIS-10 results (within ± 0.10 on mean J-drop); passes success triad of FINAL_PROPOSAL
- **Failure interpretation**:
  - DAVIS-30 mean ≪ DAVIS-10 mean → selection bias in DAVIS-10 hard subset; disclose
- **Table / figure target**: Paper Table 1 (main) + Appendix per-clip table
- **Priority**: MUST-RUN (after B1-B4 pass gates)

### Block 9 — Failure-mode qualitative analysis
- **Claim tested**: honest limitations, failure case understanding
- **Why this block exists**: reviewers appreciate honest failure discussion; improves venue readiness
- **Dataset / split**: 2-3 clips where Full method underperforms (likely natural-distractor cases)
- **Compared systems**: Full method
- **Metrics**: qualitative J trajectory + attention-mass visualization + insert visual inspection
- **Setup**: hand-selected 2-3 failures from B1
- **Success criterion**: qualitatively identify what mechanism failure happened (self-heal won? L_stale diverged? paste off-scene?)
- **Table / figure target**: Appendix discussion + Figure 5 (failure cases)
- **Priority**: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| **M0 — Sanity** | verify pipeline, metric computation, one-clip PGD runs end-to-end with new loss forms | 1 run: dog, K_ins=1, 50 PGD steps | pipeline correct, no NaN, LPIPS computable, memory-attention mass extractable | 15 GPU-min | low; if fails → loss implementation bug |
| **M1 — B0 Realization-gap** | decide LPIPS budget feasibility | 2 runs: ProPainter-base-Full, Poisson-base-Full on dog at K_ins=1 | ProPainter-base final LPIPS ≤ 0.10 + J-drop ≥ 0.3 | 1 GPU-hour | medium; if fails → relax LPIPS to 0.15, continue |
| **M2 — Baselines** | UAP-SAM2 reproduced on DAVIS-10 | 1 run: UAP-SAM2 trained on DAVIS-train + eval on DAVIS-10 hard | UAP J-drop matches published ≈ 0.40-0.55 on DAVIS-10 | 2-3 GPU-hours | low; prior work exists |
| **M3 — B1 Main (GPU0) + B2 Schedule (GPU1) in parallel** | core ablation + schedule study | B1 Full + singles + B2 2a + 2b = 4+3 conditions × 10 clips = 70 per-video PGD runs | B1 Full J-drop ≥ 0.55 AND singles ≤ 50% Full; B2 resonance gap ≥ 20pp | 6-8 GPU-hours parallel | medium; if fails → re-tune |
| **M4 — B3 L_stale + B4 SAM2Long** | mechanism proof + transfer | B3 Full-no-L_stale × 10 clips; B4 SAM2Long eval × 10 clips | B3 rebound gap ≥ 0.2; B4 retention ≥ 0.40 | 4 GPU-hours | medium |
| **M5 — B6 UAP baseline comparison** | paper positioning | UAP-SAM2 attacked videos under SAM2 + compare vs MemoryShield Full | side-by-side table populated | already done in M2 | low |
| **M6 — B8 Full DAVIS-30** | headline numbers | Full on 20 remaining clips | passes success triad | 3-4 GPU-hours | low (after M3-M5) |
| **M7 — B5 + B7 + B9 appendix** | simplicity defense + sensitivity + failure analysis | 15 runs | all appendix panels populated | 3-4 GPU-hours | low; skipped if M3-M6 prove tight |

**Total for must-run (M0 → M6)**: ~ 18-20 GPU-hours
**Total with nice-to-have (M0 → M7)**: ~ 24 GPU-hours

Stop gates:
- **After M1**: if ProPainter-base LPIPS floor > 0.10, document and relax to 0.15 cap in paper; do NOT continue to M3 under pretense the hard 0.10 bar is met.
- **After M3**: if Full J-drop < 0.45 OR schedule gap < 10pp, STOP and return to method design (likely a bug or miscalibration of β).
- **After M4**: if SAM2Long retention < 0.3, STOP and narrow claim to SAM2.1-Tiny; re-write paper framing before continuing to M6.

## Compute and Data Budget

- **Total GPU-hours**: 18-24 on single RTX Pro 6000 Blackwell
- **Data prep**: DAVIS-2017 val (already on Pro 6000), ProPainter checkpoint (need to install; ~1 hour), RAFT flow (already in many codebases, likely installable via pip in minutes)
- **Human evaluation**: none in main paper; optional visual-quality A/B study on 5 clips if reviewer requests
- **Biggest bottleneck**: sequential per-video PGD on a single GPU. M3 can parallelize on GPU0 + GPU1 (2× speedup). M4 and M6 must share because SAM2Long eval is heavy.

## Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| ProPainter LPIPS floor exceeds 0.10 | High | B0 gates before commit; pre-declared relaxation to 0.15 |
| `L_stale` gradient unstable through memory-attention | Medium | Margin-form fallback declared in proposal; swap at M3 stage 1 if grad-norm spikes |
| Natural-distractor clips (cows) regress attack | Medium | Reviewer does not require uniform DAVIS-10 success; suppression-as-default handles; report per-clip breakdown honestly |
| GPU non-determinism invalidates tight comparisons | Low-Medium | 3 seeds on B1 + median-over-seeds reporting; EMA on perturbation |
| SAM2Long install / compatibility on Blackwell cu128 | Low | already validated in prior fidelity loop (works) |
| UAP-SAM2 reproduction differs from published | Medium | existing `reproduction_report.json` gives baseline; disclose gap in paper |
| `A^ins / A^recent / A^other` slot-tagging bug | High | extensive unit tests at M0; visualize on known-clean clip with zero-insert to confirm slot accounting |

## Final Checklist

- [x] Main paper tables are covered: Table 1 (B1 + B6), Table 2 (B4), Table 3 (B2), Table 4 (B3)
- [x] Novelty is isolated: B1 singles (composition necessity) + B3 (L_stale necessity) + B2 (schedule necessity) all decouple
- [x] Simplicity is defended: B5 (mechanism vs base), implicit in B1 (no bilevel needed)
- [x] Frontier contribution: ProPainter usage justified in proposal; no further "frontier necessity" block needed (method intentionally non-frontier beyond insert generation)
- [x] Nice-to-have separated: B5, B7, B9 appendix; B6+B8 still must-run for paper
- [x] Failure analysis included: B9
- [x] Pre-registered hyperparameter sensitivity: B7
- [x] Transfer claim: B4 mandatory
