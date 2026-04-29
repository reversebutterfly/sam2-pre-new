# ChronoCloak Feasibility Audit — 2026-04-28 (codex round 3)

Codex thread `019dd243-04d2-7111-9eb0-c4eb3fec729d`, round 3.

## TL;DR

**Verdict: GO-WITH-CONCERNS, but mostly CONCERNS.**

The publisher-side threat-model framing is sound; the *mechanism* under the tight budget (ε=2/255, LPIPS≤0.03) is **NOT YET CREDIBLE**.

The old VADI v4.1 J-drop (mean 0.746 on 6 RAW joint clips) is most likely powered by **carrier salience** (visible composite artifacts) rather than by a small adversarial steering term. When the carrier is replaced with a natural interpolation and the budget is tightened, the attack effect may collapse to noise.

**Three internal warning signs from our own data** (codex pulled from `AUTO_REVIEW.md`):

1. **Line 843**: When insert fidelity improved from LPIPS 0.161 → 0.134 (a *modest* fidelity gain), attack dropped from 0.536 to 0.334/0.485. Steep fidelity↔effect tradeoff already visible.
2. **Line 1266**: A natural-carrier branch with K=3 inserts produced J-drop=0.0013 on dog clip **even without LPIPS constraints**. **The natural-carrier hypothesis has already failed empirically once.**
3. **Line 1283**: Past runs showed substantial insert-slot attention without segmentation change — **attention hijack ≠ segmentation failure**.

## What we MUST do BEFORE M1-M5

Codex's revised plan: do **NOT** start the M1 6-clip Go/No-Go pilot. Instead, run a much smaller **4-clip ceiling pilot** to test whether ν' on a clean carrier has any meaningful effect at all.

### Ceiling pilot (4 clips × 6 arms)

| Arm | Method | Budget | Purpose |
|-----|--------|--------|---------|
| 1 | clean (no attack) | — | baseline J on original frames |
| 2 | additive-only on native frames | ε=2/255 | additive baseline at the new tight budget |
| 3 | interp insert only, no ν' | — | does pure interpolation insert do anything? |
| 4 | interp + ν' @ 2/255, no bridge | ε'=2/255 | the proposed method's core — is ν' alone enough? |
| 5 | interp + ν' + bridge @ 2/255, L=2 | both 2/255 | full ChronoCloak at intended budget |
| 6 | **interp + ν' @ 8/255, no bridge** | **ε'=8/255 (CEILING)** | **does ν' work at ALL, even under loose budget?** |

**Decision rule (codex):**
- If arm 6 (ε=8/255 ceiling) produces **< 0.15 mean original-frame J-drop** on 4 clips → **the pivot is technically dead. Stop. Reformulate or abandon.**
- If arm 6 produces ≥ 0.15 but arm 5 (the intended-budget version) produces < 0.10 → tuning room exists but the method is fragile.
- If arm 5 produces ≥ 0.15 → cautious GO; proceed to M2 fresh causal A3 under new carrier.

This pilot is approximately **2-3 GPU-h**, far smaller than M1.

## Key calibrations

### Realistic J-drop bands under ε=2/255, LPIPS≤0.03
- Optimistic band: **0.25-0.35**
- Realistic band: **0.15-0.30**
- Failure band: **0.05-0.15**
- Retention of old 0.746 numbers: **plan for 20-40%, with 10% being plausible**

### Effect attribution (if it works)
Most of the effect must come from inserted carrier + ν' (60-80%). Bridge δ at L=2 frames ε=2/255 is a **stabilizer**, contributing 20-40%. **If ν' fails, bridge cannot rescue it.**

### Compute realism
- Codex correction: end-to-end is **80-150 GPU-h** (not 22). 100-step PGD per clip × 13 clips × multiple ablation arms.
- Still fits in 3 V100-weeks **if access is continuous**, but no slack for re-tuning interpolator / loss weights / placements / wrapper.

## Mechanism evidence transfer

The current 4/4 STRONG A3 result is **scientifically MISMATCHED with the new framing**. It proves something about the high-amplitude composite carrier, not about interpolated carriers.

- **Old A3 (4/4 STRONG)**: appendix-only as preliminary motivation ("on a stronger but visibly synthetic carrier, insert-position memory writes are causally important").
- **New A3 needed**: fresh causal ablation under interp+ν' carrier, on a fresh subset of clips, under the new ε=2/255 budget.
- Reviewers WILL call out the mismatch if the old A3 is used as main evidence.

## Stealth ground truth

Codex correction to my Round 2 framing:
- ❌ "Decoy MUST be visually identical to a natural next frame" (impossible bar, kills the pivot definitionally)
- ✓ "Decoy MUST not disrupt normal-speed viewing AND must not look obviously synthetic under casual inspection"

Liu CVPR 2025 sidesteps this entirely with additive-only — they have no inserted frames.

## The 3 top technical risks (codex round 3, ranked)

1. **Effect collapse**: J-drop may collapse once we remove the artifact energy that the old composite carrier was providing.
2. **Mechanism evidence non-transfer**: old A3 doesn't apply to new carrier; we have one alarming internal precedent of attention-without-segmentation-failure (`AUTO_REVIEW.md:1283`).
3. **Tight-budget optimization instability**: even if a weak effect exists, finding a repeatable Pareto point under ε=2/255 + LPIPS=0.03 may be too fragile.

## Updated action plan

### Phase A (no GPU, prep only)
- Implement RIFE/FILM interpolation hook (task #44)
- Tighten budgets in code (task #42)
- Implement UTR / SFR metrics (task #45)
- Build human stealth flipbook harness (task #46)

### Phase B0 — CEILING PILOT (NEW, INSERT BEFORE M1)
- 4 clips × 6 arms (table above), ~2-3 GPU-h
- **Decision gate**: arm 6 (ε=8/255 ceiling) ≥ 0.15 mean J-drop, else KILL the pivot

### Phase B1 — FRESH A3 (NEW, INSERT BEFORE M2)
- Memory-write blocking ablation under new interp+ν' carrier
- 4 clips, baseline + attacked + control
- ~6 GPU-h
- **Decision gate**: collapse_attacked ≥ 0.20 AND att-ctrl ≥ 0.10 on at least 2/4 clips, else mechanism claim is dead

### Phase B2-B5 — original M1-M5 plan
- Only proceed if B0 AND B1 both pass
- Compute realistic = 80-150 GPU-h total

### Kill triggers
- B0 arm 6 < 0.15 mean J-drop → STOP, reformulate or abandon
- B1 collapse_attacked < 0.20 → mechanism claim cannot be defended
- B0 + B1 pass but M3 mean J-drop < 0.20 → soft kill (not enough effect for AAAI even with stealth claim)

## Bottom line

The publisher-side framing is right. The current mechanism is **probably driven by artifact salience**, not by stealth-compatible adversarial steering. **Spend 2-3 GPU-h on the ceiling pilot before committing to anything bigger.**

If the ceiling pilot dies, we have three reformulation options to consider (codex did not endorse these — the PI must decide):
1. Loosen the budget back to ε=4/255 + LPIPS≤0.10 and accept the visible-decoy compromise (gives up some publisher-side credibility but preserves attack).
2. Replace interpolation with a more aggressive carrier (diffusion infill) — paper risk goes up but mechanism may survive.
3. Drop the insert mechanism entirely and pivot to additive-only publisher-side cloak — but that violates CLAUDE.md hard constraint AND walks straight into Liu CVPR25 territory.
