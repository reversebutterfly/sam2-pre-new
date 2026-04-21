# Review: Why Decoy Fails on breakdance + car-shadow (Round 5)

**Reviewer**: gpt-5.4 xhigh via Codex MCP
**Thread**: 019da669-70ff-7562-8d14-925a94dcfbaf
**Date**: 2026-04-19
**Status**: Diagnosis complete; fix plan ready

---

## 1. Presenting evidence

10-DAVIS reduced sweep at `insert_strength=1.0`:

| video | clean | decoy | drop | SSIM |
|---|---|---|---|---|
| bike-packing | 0.785 | 0.040 | 0.745 | 0.972 |
| blackswan | 0.925 | 0.012 | 0.914 | 0.972 |
| bmx-trees | 0.518 | 0.216 | 0.302 | 0.966 |
| **breakdance** | 0.430 | 0.503 | **−0.073** | 0.975 |
| camel | 0.978 | 0.002 | 0.976 | 0.980 |
| car-roundabout | 0.981 | 0.061 | 0.920 | 0.969 |
| **car-shadow** | 0.987 | 0.986 | **0.001** | 0.966 |
| cows | 0.981 | 0.006 | 0.975 | 0.985 |
| dance-twirl | 0.321 | 0.235 | 0.086 | 0.975 |
| dog | 0.978 | 0.357 | 0.621 | 0.975 |
| **mean** | 0.788 | 0.241 | **0.547** | 0.974 |

Aggregate signatures: PosScore=0.94, DecoyHit=0.84, CentShift=0.91. Signatures look successful; J&F drop is not.

---

## 2. Diagnosis

### car-shadow (decoy-validity failure)

1. **Invalid decoy geometry**. Offset is chosen *once at f0* then applied as a constant parallel track. On rigid moving objects, a 0.5× bbox shift can coincidentally match the car's motion vector, so by the eval window the "decoy region" is where the car is actually heading. Result: high DecoyHit, near-zero J&F drop.
2. **Non-exclusive relocation**. Bridge term + positive objectness makes "car + translated extension" an easier solution than "exclusive reassignment to a wrong place". Loss rewards shifted-but-still-overlapping masks.
3. Inside SAM2: memory attention biases the spatial prior, but current-frame visual tokens + object pointer still let the mask decoder snap to the true car. Suppression works here (drop=0.985) because it only needs to kill objectness; decoy needs a plausible *competing* hypothesis.

### breakdance (FIFO self-healing)

1. **Clean frames after f14 keep writing fresh correct memories**. Frame 0 remains privileged conditioning memory; FIFO refills with clean-frame writes that overwrite our poisoned slots. The v4 loss shapes *attacked-frame* logits but does not target what *future clean frames* will write.
2. **Decoy-track implausibility**. Constant-offset "same dancer half a box away" is a weak hypothesis for a tiny, blurred, articulated subject. This is not hijacking a natural distractor; it is asking SAM2 to believe a bad alternative track.
3. Low clean J&F (0.43) is a factor but not the main mechanism — it means the clean tracker is already fragile, so the decoy hypothesis is also hard to make self-consistent.

### User's hypotheses — graded

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | car-shadow motion alignment | **likely** |
| 2 | car-shadow shadow-as-decoy | plausible, secondary |
| 3 | breakdance motion-blur robustness | **likely** (via memory continuity) |
| 4 | breakdance ceiling effect | partly, not main |
| 5 | myopic (f0-only) decoy selection | **definitely true** |
| 6 | rank margin too small | possible, lower priority |

### Missing from user's mental model

- Current "success signatures" (DecoyHit, CentShift) **overcount partial translation** as success.
- The loss **rewards shifted-but-still-overlapping** masks (suppress mask uses full-region mean).
- Supervision is on **outputs**, not on the **memory write/read state** that controls persistence.

---

## 3. Critical code findings (verified)

After the reviewer flagged "your teacher features are built but unused", we audited the code:

### Finding 1: Teacher cooperation is dormant in v4

`teacher_features` is generated in `_build_decoy_bases_and_targets` (`run_two_regimes.py:187`) and passed through function signatures of `_decoy_write` (line 324) and `_decoy_read` (line 443), **but never referenced inside either function body**. The "v3 memory cooperation" claim in recent commit messages is effectively dropped in the v4 main path. Output-only supervision.

### Finding 2: No annulus term

`_decoy_write` (insert loss) uses a single `suppress = GT \ decoy` mask with full-region mean for both the absolute negative term and the rank term. This lets the optimizer satisfy `decoy_mean > true_mean` via smeared activation covering most of GT. The richer `core`/`bridge`/`ring` decomposition *exists* in `build_role_targets` but is flattened to the single `suppress` key by the time `_decoy_write` reads it.

### Finding 3: EVAL_END truncation

`EVAL_END = 15` in `run_two_regimes.py:97` limits optimization rollout. To test post-prefix healing we need supervision extended to at least f17 or f20.

---

## 4. Fix plan (ranked by expected impact / effort)

### F1. Temporal separability selector (replaces `find_decoy_region`)

Use `f0:f14` masks, not f0 alone. For each candidate `(dy, dx)` in 8 directions × 2–3 radii:

```
sep   = 1 − mean_t IoU(shift(m_t, dy, dx), m_t)       # future-overlap penalty
orth  = 1 − |cos((dx, dy), mean_object_velocity)|     # motion-orthogonal preference
plaus = mean color/objectness similarity on shifted support
score = w_sep * sep + w_orth * orth + w_plaus * plaus
```

Reject offsets with `mean_t IoU > 0.25`. Expected primary effect: fixes car-shadow.

### F2. Exclusive-relocation loss + top-k rank

Rewrite `_decoy_write` to split true support and use top-k mean instead of full-region mean:

```python
ring     = GT & ~core & ~decoy      # annulus
core_neg = core & ~decoy
bridge   = bridge & ~GT & ~decoy

L = (
    w_dec   * pos(decoy,    margin=0.9)
  + w_br    * pos(bridge,   margin=0.2)
  + w_ring  * neg(ring,     margin=0.5)
  + w_core  * neg(core_neg, margin=0.5)
  + w_rank  * softplus(topk_mean(logits[ring], 0.2)  - topk_mean(logits[decoy], 0.2) + 0.8)
  + w_score * object_score_positive_loss(score, margin=0.8)
)
```

Suggested weights: `w_dec=1.0, w_ring=1.5, w_core=0.5, w_br=0.05, w_rank=1.0, w_score=0.4`.

Note: current `run_two_regimes.py:_decoy_write` is the path producing the main-table results, *not* `decoy_target_loss` in `memshield/losses.py`. The annulus must be added in the runner, not just in the library function.

### F3. Wire teacher memory supervision (b-lite)

- Actually consume `teacher_features` in `_decoy_write`.
- Match `maskmem_features` on inserted frames **and** the first 2–3 clean frames after the last insert (f15:f17). This directly targets post-prefix self-healing.
- Extend `EVAL_END` or add a separate supervision horizon to cover f15–f17.

```
L_mem_insert = 1.0 * memory_teacher_loss(maskmem_features[insert_positions], teacher[insert_positions])
L_mem_post   = 0.5 * memory_teacher_loss(maskmem_features[f15:f17],          teacher[f15:f17])
L_ptr        = 0.1 * obj_ptr_teacher_loss(...)   # cap at 0.1
```

"No gain on blackswan" is not evidence against this — blackswan is easy. Validate on breakdance and dog.

---

## 5. Diagnostic ablations (before F2/F3 rework)

### A1 — car-shadow decoy-validity audit (no GPU)

- Compute `IoU(shift(GT_t, dy, dx), GT_t)` for t ∈ {0..end}, focus on t ≥ 10.
- Bands: `>0.25` invalid, `0.15–0.25` borderline, `<0.15` viable.
- Compute motion alignment: `cos(offset, mean_velocity_f0:f14)`. `|cos| > 0.7` ⇒ decoy ≈ "where object is going anyway".

### A2 — car-shadow orthogonal rerun (~15 min GPU)

Force offset orthogonal to motion and low overlap. Rerun current v4 on car-shadow only. If drop > 0.3, the failure is selector geometry, not optimizer.

### A3 — breakdance memory-healing test (~20 min GPU)

In `memshield/surrogate.py:154`, `run_mem_encoder` is hardcoded `True`. Gate writes at the source for f15:f24: `run_mem_encoder=False`, keep reads. Rerun breakdance.

- Drop rises substantially ⇒ self-healing is real, F3 is the right fix.
- No change ⇒ bottleneck is decoy plausibility or transfer, F3 is wrong investment.

---

## 6. Paper framing

"Decoy works on clips with a **decoy-viable alternative track**" is defensible **only if**:

- `X` is defined **from clean clip geometry alone**, before attack runs.
- Full-benchmark results reported first.
- Stratification by `X` shown second.
- `X` predicts success on **held-out clips**, not post-hoc.

Honest candidate definitions for `X`:
- `mean_t IoU(shifted_GT_t, GT_t) < τ_overlap`  (decoy separability)
- `|cos(offset, velocity)| < τ_motion`            (motion misalignment)
- Teacher viability: SAM2 run on a synthetic shifted video actually tracks it

If F2+F3 do not recover breakdance/car-shadow, honest claim is:
1. **Suppression** is the dominant universal regime.
2. **Decoy** is a distinct, high-confidence mislocalization regime that works when the clip admits a separable, SAM2-trackable alternative trajectory.

That does not kill the paper. Over-claiming universality would.

---

## 7. Go / no-go plan (≤ 5 GPU-days)

| Step | Task | GPU cost | Decision gate |
|------|------|----------|--------------|
| 1 | A1 audit (no GPU) | 0 | If overlap>0.25 or \|cos\|>0.7 on car-shadow, go to 2 |
| 2 | A2 orthogonal rerun on car-shadow | ~0.5h | Drop > 0.3 confirms selector fix (F1) is sufficient for this clip |
| 3 | Implement F1 + F2 | 0 (code) | — |
| 4 | F1+F2 targeted rerun on bike-packing, blackswan, car-shadow, breakdance | ~2h | car-shadow drop > 0.3 without regressing easy clips |
| 5 | A3 memory-healing test (if step 4 did not fix breakdance) | ~0.5h | Drop rises ⇒ implement F3 |
| 6 | Implement F3; rerun breakdance + dog + 2 easy | ~1.5h | Breakdance drop > 0.3 |
| 7 | Full 10-video rerun with F1+F2(+F3) | ~2h | Mean drop ≥ 0.65 ⇒ accept |
| 8 | If step 7 mean drop < 0.55, rescope paper to conditional-regime framing | — | Honest fallback |

Total: ~6.5 GPU-h for steps 1–7, plus buffer.

---

## 7b. A1 audit results (2026-04-20) — reviewer's h1/h2 refuted

Ran `scripts/audit_decoy_validity.py` on all 10 clips. Key findings:

| video | offset | overlap | cos(offset, vel) | drop |
|---|---|---|---|---|
| bike-packing | (-236,-236) | 0.006 | −0.72 | 0.745 ✓ |
| blackswan | (-287,-287) | 0.000 | −0.83 | 0.914 ✓ |
| camel | (0,-336) | 0.001 | +0.84 | 0.976 ✓ |
| car-roundabout | (0,-332) | 0.088 | +1.00 | 0.920 ✓ |
| cows | (0,+305) | 0.015 | −1.00 | 0.975 ✓ |
| **car-shadow** | (+252,0) | **0.000** | **+0.12** | **0.001 ❌** |
| **breakdance** | (0,-339) | **0.002** | **−0.04** | **−0.073 ❌** |

**Refutations**:
- "Decoy overlaps future GT" — all 10 clips have overlap < 0.1, including car-shadow (0.000).
- "Motion alignment causes failure" — 5/5 motion-aligned clips (|cos|>0.7) produce strong attacks.
- "car-shadow decoy coincides with car's destination" — zero overlap proves otherwise.

**Consequence**: F1 (temporal separability selector) would be a **no-op**; killed from the plan.

## 7c. Updated diagnosis (round 5b)

New primary bottleneck: **decoy trackability, not selector geometry**.

### car-shadow: non-maskable decoy (H-CS-d + H-CS-a)

The true location has a strong, rigid `car + cast shadow` template; the decoy location has no such support. Memory attention biases SAM2 toward the decoy, but the **mask decoder refuses to sustain a real mask there once clean frames resume**. car-roundabout works because its decoy location has trackable features; car-shadow's does not.

### breakdance: inadvertent memory regularization

On a weak tracker (clean J&F=0.43), forcing `object_score > 0` + structured inserts act as **memory refresh that stabilizes the weak tracker**, not as relocation. That explains the *negative* drop. Fixed-offset decoy of an articulated, blurred human is a poor self-consistent alternative track.

### Why `X` should not be simple clean-J&F thresholds

camel, car-roundabout, cows all have clean J&F in (0.978, 0.981) range and succeed — `J&F > 0.95` alone does not predict failure. The real latent variable is teacher-viability: can SAM2 stably track a synthetic video in which the object is pasted at the decoy location?

## 7d. New diagnostic order (replaces old F1 / A2)

### D1. Teacher-viability scan (no PGD, ~20 min GPU)

For car-shadow and breakdance, test 3–5 offset variants without running the full attack:
- Ratios: {0.4, 0.6, 0.75}
- Directions: 4 cardinals × 3 ratios = 12 candidates per clip
- For each, build synthetic decoy video (paste object at shifted location every frame f0:f_end).
- Run official SAM2 with shifted-mask prompt at f0.
- Metrics: `shifted-mask J&F` over f10:end, `PosScore` rate.
- Thresholds: `J&F > 0.7` viable, `< 0.5` non-viable.

Outcome decides:
- car-shadow: all offsets non-viable ⇒ clip is fundamentally unattackable via decoy; report it as failure mode evidence, do NOT rerun.
- car-shadow: some offset viable ⇒ rerun with that offset; expect drop > 0.3.
- breakdance: some offset viable ⇒ proceed to F3.
- breakdance: all non-viable ⇒ the decoy regime is fundamentally incompatible with fast articulated motion; paper scope limits to that condition.

### D2. Surrogate-vs-official replay on existing attacks (~30 min)

Take the attacked videos already saved from the reduced sweep. For car-shadow and breakdance, run both surrogate and official SAM2. Compute per-frame decoy signature metrics on each.
- If surrogate relocates but official does not: transfer/persistence is the bottleneck ⇒ F3 is the right fix.
- If neither relocates: output loss is fooling the signatures ⇒ F2 (annulus + top-k) is the right fix.
- If both relocate but J&F doesn't drop: it's the mask-decoder snap-back (H-CS-a) ⇒ the regime is fundamentally limited.

### D3. breakdance base-only test (~10 min GPU)

Evaluate breakdance with *inserted bases only*, no PGD deltas. If J&F still rises above clean, the negative drop is from memory refresh effect, not failed decoy optimization. Confirms the "inadvertent regularization" hypothesis.

## 7e. Updated go/no-go plan

| Step | Task | GPU cost | Decision |
|---|---|---|---|
| 1 | D1 teacher-viability scan (car-shadow + breakdance) | ~20 min | Identifies viable / non-viable clips |
| 2 | D2 surrogate-vs-official replay | ~30 min | Distinguishes F2 vs F3 as right fix |
| 3 | D3 breakdance base-only | ~10 min | Confirms regularization hypothesis |
| 4a | If D1 viable + D2 says F3: implement F3, rerun breakdance/dog | ~1.5h | Drop > 0.3 on breakdance |
| 4b | If D2 says F2: implement F2, rerun the 5 strong + 2 failures | ~2h | No regression, car-shadow drop > 0.3 |
| 5 | Full 10-video rerun with best fix | ~2h | Mean drop ≥ 0.65 |
| 6 | If step 5 fails mean threshold, rescope to conditional regime | — | — |

Total still ~6 GPU-h; F1 removed from plan.

## 7f. Round 5c — what v4 is *actually* doing + is teacher necessary?

Deep audit of the current pipeline reveals v4 is **not** "memory-feature poisoning" in the strong sense. It is a **teacher-free, short-horizon output attack** that uses the memory bank only as the gradient path — not as a supervision target. The comment in the code explicitly says *"v4 suppress+redirect: no teacher needed, output-level losses only"*. v4 was an intentional simplification, not just a silent dropout.

**End-to-end leverage point**:
- Inserts start from a visually decoy-like base frame (real object pasted via Poisson blending at shifted location) — the base itself gives a strong local prior.
- Original-frame perturbations weaken the true anchor (suppress GT logits, band objectness).
- The *only* signal that asks for persistence is `L_r` on f10:f14 (5 frames) — backprops through unrolled track_step chain via memory attention.
- No explicit memory-state target. Any hidden state yielding decoy outputs on f10:f14 is accepted, even if brittle after f14.

**Key consequence**: v4 will fail when many hidden memory states give similar f10:f14 outputs but diverge after f14. That is exactly the breakdance self-healing signature.

### Are teacher features necessary?

Crisp condition from reviewer:

> Teacher features are worth implementing **iff** all three hold:
> 1. D1 shows a viable positive target (SAM2 can track the synthetic decoy video with decent shifted-mask quality).
> 2. Alt A (extending the read horizon) is insufficient.
> 3. The failure looks like hidden-state ambiguity, not non-maskability.

If D1 fails → teacher is garbage-in-garbage-out. If Alt A succeeds → teacher is unnecessary engineering. Teacher is justified **only** when D1 passes and Alt A still fails.

### Three candidate alternatives to F3

| Alt | Description | Cost | Assessment |
|---|---|---|---|
| **A** | Extend `EVAL_END` 15 → 20 or 22. More clean frames covered by `L_r`. Gradient via existing memory attention, no new machinery. | 1 line | **Right first move.** Directly tests self-healing. Uses existing gradient path. Cheap. No dependency on teacher viability. |
| B | Divergence-from-clean loss on `maskmem_features[insert_positions]`. Untargeted — "be different from clean". | ~20 LOC | **Bad for inserts.** Easiest solution is "write junk / suppression-like", which collapses decoy to absence. Only useful as anti-anchor on originals (separate role). |
| C | F3 full teacher features (b-lite scope: inserts + f15:f17). Requires building synthetic teacher and running SAM2 on it. | ~100 LOC + teacher build | **Too contingent for first move.** Depends on D1. Overkill if Alt A works. |

### Simpler things reviewer flagged

These are **cheaper than any loss change** and should be checked first:

1. **Base-frame maskability** — the insert base already pastes a real object at the decoy location. Evaluate the raw pasted bases with NO PGD deltas. If results improve, PGD is partially destroying a useful decoy template. If they stay bad, the base itself isn't giving SAM2 a viable decoy (especially relevant for car-shadow).
2. **Reduce offset_ratio** for car-shadow specifically — the `0.75` shift may push decoy off-frame or into non-maskable region. Try `0.4`–`0.5` and check that the pasted base is fully on-frame with textured support.

### Final verdict on strategy

Current v4 optimizes **short-horizon decoy outputs**, not **stable decoy memory**. The cheapest, sharpest diagnostic is Alt A — extend the read horizon. Run that before ever implementing F3.

## 7g. Updated go/no-go plan (replaces 7e)

| Step | Task | GPU cost | Decision |
|---|---|---|---|
| 1 | **Alt A**: `EVAL_END = 20`, adjust late-frame weight decay floor from 0.3 → 0.5, rerun breakdance + dog + blackswan (sanity) | ~45 min | breakdance drop > 0.3 → Alt A is the fix; skip F3 |
| 2 | Raw-base eval on car-shadow (no PGD, just pasted inserts + original frames) | ~5 min | Improves → PGD is destroying the decoy template, reduce PGD stages or strengths. Stays bad → non-maskable base, reduce offset_ratio |
| 3 | If step 2 says non-maskable: try `offset_ratio=0.4` for car-shadow, rerun | ~20 min | Drop > 0.3 → clip-specific ratio tuning sufficient |
| 4 | Only if breakdance still fails after step 1: D1 teacher-viability scan | ~20 min | Teacher viable → implement F3; not viable → rescope clip as non-decoyable |
| 5 | If D1+F3 done: full 10-video rerun with best config | ~2h | Mean drop ≥ 0.65 → accept |

Total ~3 GPU-h in the 80% case (steps 1–3 suffice).

## 7h. Summary: do I need teacher features?

**Tentative answer: no, not as the first move, and possibly not at all.**

- Current v4 is output-only short-horizon. Its failures on breakdance/car-shadow have specific mechanisms (self-healing / non-maskability) that are cheaper to test than to pre-fix.
- Alt A (extend horizon to f20) is the right first experiment — it tests the self-healing hypothesis directly and costs 1 line of code.
- F3 (teacher features) is justified **only** when D1 says a viable teacher exists AND Alt A alone doesn't recover the clip.

## 8. Claims matrix (outcomes of the rework)

| Outcome of F1+F2 | Outcome of F3 | Allowed paper claim |
|---|---|---|
| Fixes car-shadow, ≥9/10 strong | Fixes breakdance, ≥10/10 strong | **Universal decoy poisoning** as primary method |
| Fixes car-shadow only | Does not fix breakdance | **Conditional decoy** + full diagnostic framework; suppression as stronger-but-simpler companion |
| Does not fix car-shadow | Does not fix breakdance | **Two-regime study** with honest failure analysis; rescope to SAM2Long transfer story |
| Regresses easy clips | — | Revert F2 weights; keep F1 only |
