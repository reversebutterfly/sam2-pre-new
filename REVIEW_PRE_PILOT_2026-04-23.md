# Pre-Pilot Review (GPT-5.4 xhigh) — 2026-04-23

**Thread ID**: `019dbabd-4803-7752-95cc-99f3ac08999c`
**Repo HEAD**: `6e00d5e` (VADI wiring landed, forward-smoke passed on Pro 6000)
**Question asked**: Is the pilot ready to launch? Identify silent failure modes, method gaps, code risks, and pilot-gate issues before burning 1-2 GPU-hours.

**Overall verdict**: **HALT**. Minimum fixes required before launch.

---

## Key findings (verified against code)

### 1. Evaluator mismatch (HALT-severity)

`best_surrogate_J_drop` (`memshield/vadi_optimize.py:498-506`) is:

- Computed ONLY on `insert_ids_processed ∪ neighbor_ids_processed` (not whole video)
- Against `m_hat_true_by_t` (pseudo-mask from `clean_SAM2`, NOT DAVIS GT)
- NOT re-run on exported uint8 PNGs

And the pilot GO gate (`scripts/run_vadi_pilot.py:194, 201`) reads directly from this surrogate.

Export re-measure (`scripts/run_vadi.py:remeasure_exported_feasibility`) only re-checks LPIPS/SSIM/TV budgets — it does NOT re-run SAM2 on exported PNGs to compute post-quantization J-drop. So the "delivered bytes defeat SAM2" paper claim is NOT validated by the current code, even at pilot scale.

### 2. Rank-tie bias (HALT-severity, trivial fix)

`_ordinal_rank` uses `argsort.argsort` which gives later-tied entries higher rank (self-test at `vulnerability_scorer.py:354` confirms `[1.0, 1.0, 1.0] → [1, 2, 3]`). If the 3 raw signals are flat/noisy, `top` systematically drifts to later frames. SAM2 naturally degrades over time, so a later-frame placement could produce fake J-drop that masquerades as vulnerability-scoring signal.

Fix: use average/midrank for ties (`scipy.stats.rankdata(x, method='average')`).

### 3. Δmu diagnostic is weak

Uses `last_feasible_step − first_logged_step` (`run_vadi_pilot.py:96`), not best/exported vs a dedicated clean baseline. Can pass/fail independently of the actually-selected attack.

### 4. K=1 random with 1 draw is too variance-prone for n=3 clips

Pilot's condition 1 (top − random ≥ 0.05) relies on a single random draw per clip. With 3 clips this is too noisy to be a reliable signal. Codex recommends 5 draws or a time-matched random control.

### 5. Contrastive margin can collapse to suppression

`softplus(mu_true − mu_decoy + 0.75)` can be minimized by reducing `mu_true` alone (= suppression). Need to explicitly LOG `Δmu_decoy > 0` as a hard check, not just a diagnostic; ideally add a small anchor term like `softplus(-mu_decoy + 0.5)`.

### 6. Insert pseudo-target is weak as eval target

`m_hat_true` at insert position W_k is `0.5·clean[c_k-1] + 0.5·clean[c_k]` — OK as PGD supervision but inappropriate for evaluation, since an OOD inserted frame has no ground truth. Report original-frame-only J separately from insert-frame-only J.

### 7. 0.20 K3_top threshold not calibrated

On the surrogate (self-consistency J-drop against clean_SAM2 pseudo-masks), 0.20 is not empirically anchored. On DAVIS GT whole-video, it might be too strict or too loose — we don't know without running the actual exported-artifact eval.

---

## Minimum fixes before launch (per codex)

1. **Fix `_ordinal_rank` ties** — average rank method. Trivial, ~10 min.
2. **Add exported-artifact SAM2 re-evaluation** — after export, load uint8 PNGs, run clean SAM2 on interleaved {original × inserts} + attacked SAM2, compute actual whole-video J-drop. Gate on this, not surrogate. ~1-2 h coding + extra ~30s per (clip, config) at pilot time.
3. **Clarify J-drop denominator** — either use DAVIS per-frame GT (load full `Annotations/480p/<clip>/*.png`) OR label the pilot clearly as "pseudo-mask surrogate, not DAVIS-GT".
4. **5 random draws per clip** for K1_random, or time-matched random control.
5. **Compute Δmu at best/exported step vs dedicated clean baseline**, not PGD-internal proxy.

---

## Our response / triage

### Accept + implement before launch (HALT-severity)
- **Fix 1 (rank ties)**: trivial, clear correctness win. **DO IT.**
- **Fix 2 (exported-artifact re-eval)**: this is the biggest scientific-validity gap. Must implement before pilot — without it, GO/NO-GO decision has no relationship to the paper claim. **DO IT.**
- **Fix 3 (DAVIS GT vs pseudo-mask labeling)**: at minimum, rename fields and clearly document in pilot_decision.json. For pilot gate purposes, pseudo-mask self-consistency is a valid kill-switch (if attack can't beat clean-SAM2's own output, it definitely won't beat GT). But MUST be documented to avoid accidental misuse. **DO IT (labeling; DAVIS-GT eval can be main-table work).**

### Defer to main-table rigor, not pilot (LOW-severity for kill-switch purpose)
- **Fix 4 (5 random draws)**: pilot is cheap kill-switch, not final evidence. 1 random draw is adequate to detect "no signal at all". Main table gets 5 draws. Document decision.
- **Fix 5 (Δmu vs clean baseline)**: the diagnostic is non-blocking anyway. Defer to main-table evaluation where we have full baseline runs.

### Document but don't block (method-level concerns)
- **Suppression-collapse** (6): log Δmu_decoy hard; if Δmu_decoy ≤ 0 on all pilot clips, that's informational. Don't block pilot on it.
- **Insert-frame eval target** (8): report original-only J as primary; insert-frame J as secondary.

---

## Revised launch plan

1. Fix 1 (rank ties) → code
2. Fix 2 (exported re-eval) → code
3. Fix 3 (rename fields + doc) → code
4. Codex re-review of fixes
5. Forward-smoke re-verify on Pro 6000
6. Launch pilot

Estimated effort: **4-6 hours of coding + review + re-smoke** before the 1-2 GPU-hour pilot. Total: half a day.

---

## Why this is the right call

The codex review caught a gap that was invisible to the A-I review I did on just the new wiring module (Codex 1 only saw `vadi_sam2_wiring.py` + pilot `build_pilot_adapters`). Codex 2 saw the full orchestration and spotted that the KILL-SWITCH gates on a metric that doesn't correspond to the paper claim. Launching now would have produced a number that either (a) misleadingly says GO when the exported artifact is actually defanged by uint8 quantization, or (b) misleadingly says NO-GO because the surrogate is too strict and real J-drop would have been higher.

Half a day of code to fix the evaluator is cheap insurance against a pilot that doesn't tell us what we need to know.
