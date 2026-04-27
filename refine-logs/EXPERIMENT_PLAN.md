# Experiment Plan — v5 (Memory-Hijack Insertion)

**Problem**: Adversarial attack on SAM2 video segmentation using internal decoy insertion + sparse bridge δ on adjacent original frames; target AAAI; must address Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA prior arts.

**Method Thesis**: Three internally-inserted semantic decoys at empirically-searched positions provide causal evidence that SAM2's prompt-conditioned memory propagation is the (DOMINANT / SUBSTANTIAL — A3-conditional) failure mode under publisher-side adversarial attack, and sparse δ on the L=4 bridge frames immediately following each insert measurably extends the memory divergence beyond the insert-only baseline.

**Date**: 2026-04-27
**Implementation status**: v4.1 commit `da719dc` already on Pro 6000.

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1.a (causal)** | Mechanism-level; memory-write blocking should collapse the attack at insert positions specifically (vs matched non-insert positions). Foundation for paper title/thesis A vs B vs C framing. | A3 dual-threshold pre-registered: collapse_attacked ≥0.20 AND (att−ctrl) ≥0.10 on ≥7/10 (strong) OR collapse_attacked ≥0.10 AND (att−ctrl) ≥0.05 on ≥6/10 (partial). | B3 |
| **C1.b (persistence)** | Bridge δ adds beyond insert-only at memory-bank level (not just output). Supports "bridge δ extends hijack" claim. | Per-clip d_mem(t) trace integral of (full − only) > 0 over (w_K, w_K+L) on ≥7/10 clips. T_obj 16/32/64 sensitivity. | B5 |
| **C2 (method, RAW joint headline)** | Paper headline number. RAW joint (no wrapper) must beat insert-only A0 in paired comparison on held-out 10. | ≥5/10 strict wins; mean paired lift ≥+0.05; median >0; top-clip <40%; mean RAW J-drop ≥0.55. | B1 + B2 |
| **C2-supporting (deployment)** | For deployment readers: wrapper-selected ≥ A0 by construction. NOT a contribution. | Reported separately in Table 2; no headline gate. | B1 |
| **Anti-claim 1**: "the gain is from search, not from bridge δ" | RULE OUT. A1 isolates bridge δ at fixed upstream W*+ν+decoy. | A1: A1-full mean lift over A1-only ≥+0.05; ≥6/10 strict; median >0. | B2 |
| **Anti-claim 2**: "any memory-write block degrades attack, not insert-specific" | RULE OUT. A3 negative control with matched non-insert frames. | A3-attacked − A3-control ≥0.10 (strong) / ≥0.05 (partial) on majority. | B3 |
| **Anti-claim 3**: "search complexity carries the gain, heuristic could too" | RULE OUT. A2 random vs search. | Search > random by ≥+0.10 mean lift. Plus: vulnerability heuristic (3-signal scorer) was empirically anti-correlated 0.488 vs 0.534 in earlier rounds — referenced. | B4 |

---

## Paper Storyline

### Main paper must prove
- **Table 1 (RAW joint headline)**: 10-clip paired RAW joint v5 vs A0 (B1)
- **Table 3 (A3 with neg control, mechanism)**: 10-clip × {baseline, attacked-block, control-block} (B3)
- **Table 4 (A1 bridge δ isolation)**: 10-clip paired A1-only vs A1-full (B2)
- **Table 5 (A2 placement)**: 10-clip search vs random (B4)
- **Figure 1 (overview)**: pipeline diagram + d_mem(t) trace example
- **Figure 2 (d_mem persistence)**: per-clip d_mem trace, 3 conditions, with mean ± std (B5)

### Appendix can support
- **Table A1 (T_obj sensitivity)**: d_mem trace integrals at T_obj ∈ {16, 32, 64} on 10 clips
- **Table A2 (Stage 14 wall-clock + polish_applied rate)**: dev-4 + held-out
- **Table A3 (deployment policy column)**: wrapper-selected J-drop per clip
- **Figure A1 (qualitative)**: per-frame decoy + bridge-δ visualizations
- **Table A4 (lambda_keep_full sensitivity)**: 25 vs 50 retry on bmx-trees-like clips (if time permits)
- **Discussion paragraph (E1 ownership)**: heuristic falsification, search necessity, not claimed as primary contribution

### Experiments intentionally CUT
- Cross-architecture transfer (AOTL, STM, etc.) — not in scope; SAM2 is the new attack surface and existing prior arts already saturate non-SAM2 VOS
- Universal perturbation comparison — UAP-SAM2 numbers cited from their paper; per-clip targeted is a different threat model
- Black-box transfer — out of threat-model scope
- Diffusion-based decoy generator — explicitly demoted; supplementary "decoy quality" only if reviewer asks
- Per-layer A3 hook (block writes layer-by-layer) — appendix only IF A3 strong-passes

---

## Experiment Blocks

### Block 1 (B1) — RAW joint v5 vs A0, 10-clip paired headline

- **Claim tested**: C2 (method-level "RAW joint weakly dominates insert-only on held-out"); supports deployment-policy column.
- **Why this block exists**: paper headline. Without ≥5/10 strict wins + mean ≥+0.05 RAW joint, the method claim collapses.
- **Dataset / split / task**: DAVIS-2017 val, 10 held-out clips. Proposed list: bear, blackswan, breakdance, cows, dance-twirl, dog, hike, horsejump-high, india, judo. (Avoid dev-4 redundancy if possible; if forced to overlap, mark dev-4 clips as "in-distribution" in paper; held-out-7 = clips NOT in dev-4.)
- **Compared systems**: 
  - A0 (K3_top + insert-only, frozen — Stage 11-13 of v5)
  - RAW joint v5 (full Stage 14 + L_keep_full anchored, NO wrapper)
  - (Deployment column only) Wrapper-selected = max(joint, A0)
- **Metrics**: 
  - Decisive: per-clip exported J-drop on uint8 video
  - Secondary: polish_applied rate, mean and median paired lift, top-contributor share, leave-one-out mean
- **Setup details**: SAM2.1-Tiny frozen, 1 GPU per clip, ε_∞=4/255 (f0=2/255) + per-frame LPIPS≤0.20, 1 seed (det. infrastructure already in `da719dc`), Stage 14 = 30 steps with v4.1 anchored loss.
- **Success criterion**: ≥5/10 strict wins; mean lift ≥+0.05; median >0; top contributor <40%; mean RAW J-drop ≥0.55.
- **Failure interpretation**: 
  - 3-4/10 wins → narrow paper to "specific clips story"; honest reporting; consider workshop venue
  - <3/10 wins → fundamental method failure; must rethink (low probability per dev-4 75% rate)
- **Table / figure target**: **Table 1 (main, headline)** + Table A3 (deployment policy column).
- **Priority**: **MUST-RUN**.

### Block 2 (B2) — A1 bridge δ contribution, isolated

- **Claim tested**: Anti-claim 1 — "the gain comes from bridge δ, not from upstream search/ν."
- **Why this block exists**: if A0 → full v5 lift could be due to placement search alone (with same ν, same decoy_seeds, just different bridge handling), the bridge δ claim is unsubstantiated. A1 holds W*, ν, decoy_seeds constant; only toggles bridge variables.
- **Dataset / split / task**: same 10-clip held-out as B1.
- **Compared systems**:
  - **A1-only**: insert at W* with `nu_init` (from A0 polish), ALL bridge variables zeroed (`traj=0, alpha_logits=-1e9, warp=0, R=0`); skip Stage 14 entirely; export bridge frames untouched.
  - **A1-full**: insert at SAME W* with SAME `nu_init` + full Stage 14 polish (commit da719dc). Bridge δ is the ONLY toggle.
- **Metrics**: per-clip exported J-drop paired; mean / median / win count.
- **Setup**: shared upstream W* (joint search) and nu_init (A0 polish 100 steps). Both arms run SAM2 eval on uint8 export.
- **Success criterion**: A1-full mean paired lift over A1-only ≥+0.05; ≥6/10 strict wins; median >0.
- **Failure interpretation**: bridge δ contribution rejected. Paper must drop "bridge δ extends" claim and resort to wrapper-only deployment story (very weak for AAAI; → workshop pivot).
- **Table / figure target**: **Table 4 (main)**.
- **Priority**: **MUST-RUN**.
- **Compute note**: A1-only runs in <30s per clip (no Stage 14); A1-full = same as B1's joint v5 → can SHARE B1's runs (no extra GPU-h beyond B1).

### Block 3 (B3) — A3 with negative control, memory-causality

- **Claim tested**: C1.a (causal mechanism) + Anti-claim 2 (insert-position specificity).
- **Why this block exists**: paper title/thesis Framing A vs B vs C is gated entirely by this experiment. Without negative control, "memory hijack" story is reviewer-vulnerable.
- **Dataset / split / task**: same 10-clip held-out.
- **Compared systems**:
  - **A3-baseline**: full v5 (no blocking) — same as B1 RAW joint
  - **A3-attacked**: full v5 + `make_blocking_forward_fn(blocked_frames=W_attacked)` — at t ∈ W_attacked, the per-frame `current_out` is NOT appended to `obj_output_dict`, suppressing the inserted frames' contributions to BOTH future mask-memory chunks AND obj_ptr token assembly.
  - **A3-control**: full v5 + same wrapper with `blocked_frames=W_control` (3 matched non-insert non-bridge non-zero frames; uniform random sample, fixed seed=0 per clip; via `build_control_frames`)
- **Metrics**: per-clip exported J-drop. Compute `collapse_attacked = baseline_J − attacked_J` and `collapse_control = baseline_J − control_J` per clip.
- **Setup**: implementation in `memshield/causal_diagnostics.py` (M0, codex GO 2026-04-27). Hook is `make_blocking_forward_fn(base, blocked_frames, extractor)`; skips `obj_output_dict["non_cond_frame_outputs"][fid] = current_out` for fid in blocked. Everything else (Hiera forward, mask decoder, memory encoder, prior bank queries) untouched. **Important**: this is broader than "block memory writes" — it suppresses ALL future temporal-state contributions (maskmem AND obj_ptr) from blocked frames, because SAM2 reads both from `non_cond_frame_outputs`. The pre-registered claim in §3.1 has been refined to match.
- **Success criterion** (pre-registered, dual threshold):
  - **Strong pass**: collapse_attacked ≥ 0.20 abs AND (collapse_attacked − collapse_control) ≥ 0.10 abs, both on ≥7/10 → **Framing-A "DOMINANT failure mode"**
  - **Partial pass**: collapse_attacked ≥ 0.10 abs AND (collapse_attacked − collapse_control) ≥ 0.05 abs, both on ≥6/10 → **Framing-B "SUBSTANTIAL component"**
  - **Fail**: either threshold misses majority → **Framing-C workshop pivot**
- **Failure interpretation**: pre-registered framing decision — paper is RE-TITLED, not over-claimed.
- **Table / figure target**: **Table 3 (main, mechanism)**; embed strong/partial decision in §Results paragraph 1.
- **Priority**: **MUST-RUN, FIRST** (gates paper framing).

### Block 4 (B4) — A2 placement matters

- **Claim tested**: Anti-claim 3 — "search-based placement matters; random K=3 doesn't suffice."
- **Why this block exists**: defends E1 enabling component. Reviewer will ask "why not random?" given prior heuristic falsification.
- **Dataset / split / task**: same 10-clip held-out.
- **Compared systems**:
  - **Random K=3**: 3 positions sampled uniformly at random from valid clean-space positions [1, T_clean), fixed seed=0 per clip. Then full v5 (Stage 14 polish) on this random W.
  - **Joint search**: joint curriculum K=1→2→3 + simplex slack + suffix-probe + trust region. Then full v5 polish.
- **Metrics**: per-clip exported J-drop paired; mean lift.
- **Setup**: 1 seed for random; same Stage 14 polish parameters in both arms.
- **Success criterion**: search > random by ≥+0.10 mean lift on 10 clips.
- **Failure interpretation**: 
  - <0.05 lift → search not necessary → simplify to random (BUT this would mean placement is empirically irrelevant, which is a paper finding in itself; rewrite §Discussion accordingly)
  - 0.05-0.10 lift → marginal; mention as caveat
  - ≥0.10 lift → as expected; E1 ownership justified
- **Table / figure target**: **Table 5 (main)**.
- **Priority**: **MUST-RUN** (lower priority than B1-B3; can run after).

### Block 5 (B5) — d_mem(t) persistence trace

- **Claim tested**: C1.b (bridge δ extends d_mem above insert-only).
- **Why this block exists**: complements B3 with a per-frame mechanistic view; supports "persistence" half of paired claim.
- **Dataset / split / task**: same 10-clip held-out (3 conditions per clip → 30 forward passes).
- **Compared systems**:
  - Clean SAM2 forward (memory bank from clean video)
  - Insert-only SAM2 forward (decoy_seeds + nu_init at W*, NO bridge δ — same as A1-only)
  - Full v5 SAM2 forward (full Stage 14 polish — same as B1)
- **Metrics**: per-frame `d_mem(t) = 1 − cos(M_clean[t], M_attacked[t])` averaged over `T_obj(clip)` (top-32 memory tokens by attention at clean c_K_clean, FROZEN per clip).
- **Setup**: extractor at `memory_attention.layers[-1].cross_attention`, PRE-output-projection V tensor.
- **Success criterion**: integral of (d_mem_full − d_mem_only) over t ∈ (w_K, w_K + L) > 0 on ≥7/10 clips.
- **Failure interpretation**: bridge δ doesn't actually extend memory divergence → C1.b weakened; paper narrows to mechanism causality (B3) only.
- **Table / figure target**: **Figure 2 (main)** + Table A1 (T_obj sensitivity, appendix).
- **Priority**: **MUST-RUN** (alongside B3 since same forward passes can extract memory in same run).

### Block 6 (B6) — Appendix sensitivity & qualitative (NICE-TO-HAVE)

- **Claim tested**: robustness of d_mem protocol; visual evidence of attack effect.
- **Why this block exists**: deflects reviewer "what if you chose wrong T_obj" / "what does the attack actually look like" questions.
- **Dataset / split / task**: 10-clip subset (or 3 representative clips for qualitatives).
- **Compared systems**: 
  - T_obj ∈ {16, 32, 64} on B5's d_mem analysis
  - Per-frame qualitative visualizations: clean / insert-only / full v5 (3 representative clips; e.g. dog, camel, bmx-trees)
- **Metrics**: d_mem integral; visual fidelity; LPIPS / SSIM per frame
- **Setup**: B5 reuse + visualization scripts.
- **Success criterion**: T_obj results consistent in sign across all three; qualitatives readable.
- **Failure interpretation**: T_obj inconsistent → C1.b protocol fragile; report and weaken claim.
- **Table / figure target**: Table A1 + Figure A1 (appendix).
- **Priority**: **NICE-TO-HAVE**.

### Block 7 (B7) — lambda_keep_full=50 retry on bmx-trees-like (NICE-TO-HAVE)

- **Claim tested**: tunability of L_keep_full weight; can revert clips be made to apply with stronger weight?
- **Why this block exists**: dev-4 found bmx-trees still reverts with default lambda=25; codex's tuning indication was 50. Worth one supplementary check before paper submission.
- **Dataset / split / task**: 1-3 clips that revert in B1 (likely bmx-trees + 1-2 others).
- **Compared systems**: full v5 with `--oracle-traj-v4-lambda-keep-full 50` vs default 25.
- **Metrics**: per-clip J-drop; polish_applied flag.
- **Setup**: CLI flag toggle, no code change.
- **Success criterion**: at least one previously-reverting clip applies at lambda=50 → mention in §Discussion as tuning trade-off.
- **Failure interpretation**: lambda=50 doesn't help → mention bmx-trees-like clips as honest limitation in paper.
- **Table / figure target**: Discussion paragraph or Table A4.
- **Priority**: **NICE-TO-HAVE**.

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0 (Sanity)** | Implement & smoke-test new code | Hook + extractor + control sampler implementation; 1-clip dry run on dog (A3 + d_mem) | Code review + dry run shows non-zero d_mem trace + memory-block actually changes mask | 5h impl + 1 GPU-h | LOW — modular SAM2VideoAdapter |
| **M1 (Baseline)** | A0 baseline on full 10 held-out | A0 (Stage 11-13 only) on 10 clips | A0 mean J-drop reasonable (~0.45-0.55, in line with prior 10-clip data) | 3 GPU-h overnight | LOW — already implemented |
| **M2 (Decisive: A3 first)** | **Gate paper framing** via A3 + d_mem | A3-baseline + A3-attacked + A3-control + d_mem trace on 10 clips (3 forward passes per clip × 10 clips = 30 forwards + 30 d_mem extractions) | **A3 strong / partial / fail decides Framing A / B / C** | 10 GPU-h | MEDIUM — collapse magnitude unknown ex ante; pre-registered narrowing if weak |
| **M3 (Headline)** | C2 RAW joint v5 + A1 paired (shared upstream) | Joint search → A0 polish → Stage 14 (= A1-full = RAW joint v5) → export. Plus A1-only: skip Stage 14. Both per clip × 10 clips. | Headline gates: ≥5/10 wins + mean ≥+0.05 + median >0 + top<40% + mean ≥0.55 | 8 GPU-h overnight | MEDIUM — bmx-trees-like reverts are real; 75% on dev-4 → expect 5-7/10 |
| **M4 (Decisive)** | A2 placement | Random K=3 + full v5 polish on 10 clips | Search > random ≥+0.10 mean lift | 5 GPU-h | LOW — random placement is well-defined |
| **M5 (Polish)** | B6 + B7 nice-to-haves | T_obj sensitivity (B5 reuse with 16/64); qualitative figure 3 clips; lambda=50 retry 1-3 clips | Nice-to-have results in appendix | 4 GPU-h | LOW |
| **Total** |  | | | **~31 GPU-h** | |
| **Paper write** | Write conditional Framing | Author-time | conditional title/abstract/claims pre-registered | 8 author-h | (no compute) |

### Wall-clock plan (3-4 days, GPU-shared)

| Day | Phase | Tasks |
|---|---|---|
| Day 1 AM | M0 | Implement R4-spec hook + extractor + control sampler (~5h coding); dry run on dog |
| Day 1 PM | M2 | A3 + d_mem on 10 clips (~10 GPU-h) — read verdict at end of day |
| Day 1 evening | gate | Read A3 verdict; commit to Framing A/B/C; if Fail → workshop pivot decision |
| Day 2 | M1 + M3 | A0 baseline overnight on 10 clips (3h); RAW joint v5 + A1 (paired) overnight (8h) |
| Day 3 AM | M4 | A2 random placement (5h); collect numbers |
| Day 3 PM | M5 + write | T_obj sens; qualitatives; lambda=50 retry; start writeup |
| Day 4 | finalize | Paper Tables 1/3/4/5 + Fig 1/2; appendices; conditional Framing language; submit |

---

## Compute and Data Budget

- **Total estimated GPU-hours**: ~31 GPU-h (M0 1h + M1 3h + M2 10h + M3 8h + M4 5h + M5 4h)
- **GPU constraint**: Pro 6000 ×2; currently shared with 2025D_ShiGuangze. Realistic single-GPU access: 8-12 GPU-h overnight; 16 GPU-h if dual access.
- **Wall**: 3-4 days assuming single-GPU access most of the time; 2-3 days if dual.
- **Data preparation needs**: DAVIS-2017 val already on disk at `~/sam2-pre-new/data/davis`. No new data.
- **Human evaluation needs**: NONE.
- **Biggest bottleneck**: M2 A3 + d_mem (10 GPU-h, gates everything else); wait for GPU release.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| A3 collapse weaker than partial threshold (<0.10 abs on majority) | MEDIUM | HIGH (paper framing C-pivot) | Pre-registered Framing-C workshop pivot; partial pass still a viable AAAI paper with narrowed claim |
| A3 negative control comparable to attacked (att−ctrl <0.05 majority) | MEDIUM | HIGH (mechanism claim NOT insert-specific) | Same as above — Framing-C; honest "memory writes generally affect, not insert-specific" |
| RAW joint mean lift < +0.05 on held-out | LOW-MEDIUM | HIGH (C2 fails) | dev-4 had 75% strict-win on 4 clips; held-out 10 likely 5-7/10. If <5/10, narrow C2 claim or workshop |
| Stage 14 pathological-loop slowness on some held-out clip (like dog 25min in v4.0) | LOW | MEDIUM (run wall) | Kill+retry with `--placement-search-prescreen-seed 1`; v4.1 dense L_keep_full largely fixes this |
| Hook implementation bug breaks SAM2 forward | LOW | MEDIUM (delays M2) | M0 sanity test on 1 clip catches this |
| GPU contention extends timeline | MEDIUM | LOW | Plan accommodates 8-12h/day single-GPU access; if blocked, write paper in parallel |
| T_obj sensitivity reveals fragile d_mem | LOW | LOW | Only affects appendix; can drop or report |
| bmx-trees-like reverts hurt mean | KNOWN | MEDIUM | A1-full polish_applied rate reported; use leave-one-out as secondary metric; consider lambda=50 retry (B7) |
| Reviewer asks for cross-architecture transfer | MEDIUM | LOW | Cite Li 2023 / Hard Region Discovery for non-SAM2 baseline; argue SAM2 is the target |

---

## Final Checklist

- [x] **Main paper tables are covered**: Tables 1 (RAW headline), 3 (A3 mechanism), 4 (A1 isolation), 5 (A2 placement).
- [x] **Novelty is isolated**: A1 isolates bridge δ (anti-claim 1); A3 isolates insert-position specificity (anti-claim 2); A2 isolates placement value (anti-claim 3).
- [x] **Simplicity is defended**: A1 IS the simplification check (insert-only at fixed W*+ν is the "without bridge δ" baseline). NO additional bloat-comparison needed.
- [x] **Frontier contribution is justified or explicitly not claimed**: NO LLM/VLM/Diffusion/RL component; SAM2 is the foundation-era target; memory readout used as causal diagnostic only. Explicitly stated.
- [x] **Nice-to-have runs are separated from must-run**: B1-B5 must-run; B6-B7 nice-to-have.
- [x] **A3 is sequenced FIRST**: gates paper framing (per codex R4 explicit).
- [x] **Conditional Framing A/B/C pre-registered**: title/abstract/claim language ready for each A3 outcome.
- [x] **Wrapper demoted**: deployment column only, NOT a contribution.
- [x] **E1 honestly owned**: search is enabling engineering, heuristic falsification cited as motivation.
- [x] **All 4 prior arts addressed**: Chen WACV 2021, UAP-SAM2 NeurIPS 2025, Li T-CSVT 2023, PATA — each in §Related Work + numerical comparison where applicable.
