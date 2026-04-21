## Failure-Mode Review: `breakdance` and `car-shadow`

Date: 2026-04-19
Scope: Decoy v4 gentle, matched-budget mislocalization poisoning on SAM2

### Bottom line

- `car-shadow` is most likely not a memory-poisoning failure. It is a decoy-definition and metric-validity failure: the chosen decoy is too close to the natural future trajectory and/or too overlapping with the true support, so `DecoyHit` can be high while `J&F` barely changes.
- `breakdance` is most likely a real poisoning failure. Clean SAM2 is already unstable, the target object is small and blurred, and the current objective mainly shapes output logits on attacked frames instead of making future clean frames rewrite a stable decoy memory state.

### Ranked diagnosis

#### `car-shadow`

1. **Trajectory-aligned or overlapping decoy** is the leading explanation.
   - The selector chooses one fixed offset from `f0` using only local color/background heuristics.
   - If that offset points along the car's true motion direction, the decoy can become "future GT shifted backward/forward" rather than a semantically wrong place.
   - If the shifted support still overlaps the car body or attached shadow support, `DecoyHit` can rise without hurting J or F much.

2. **Your current decoy signatures are permissive enough to overcount success.**
   - `DecoyHit` only requires `IoU(pred, shifted_GT) > IoU(pred, GT)`.
   - That does not require low overlap with GT, and centroid shift also does not require exclusive relocation.

3. **The loss is encouraging "confident displacement", not "exclusive wrong support".**
   - Positive object score plus mean-logit ranking can produce a translated or widened mask that still covers the object.
   - For compact rigid objects with stable appearance, SAM2 can keep high-quality shape support while shifting the centroid.

#### `breakdance`

1. **Post-prefix self-healing is the main mechanism.**
   - The current v4 gentle loss is mostly output-level relocation pressure on inserted frames and short-horizon rollout pressure.
   - It does not explicitly force the next clean frames to write decoy-like memories.
   - On a difficult clip, SAM2 can momentarily wobble toward the decoy, then recover once clean evidence dominates.

2. **The chosen decoy is likely semantically weak for a small blurred target.**
   - A shifted background patch is a poor alternative anchor when the object itself is tiny and appearance is ambiguous.
   - In this regime, SAM2 likely relies more on continuity and prompt-conditioned identity than on the exact inserted-frame texture.

3. **There is some ceiling/noise effect, but it is not the main story.**
   - Clean `J&F=0.43` does make small absolute changes harder to interpret.
   - But the stronger point is that low clean accuracy means the model already has weak, noisy memories, so your decoy target is also unstable and hard to make persistent.

### Hypothesis check

1. Car-shadow motion alignment: **plausible and likely**.
2. Car-shadow shadow-as-decoy: **plausible, but secondary unless the shadow materially overlaps the annotated mask.**
3. Breakdance motion-blur robustness / reliance on continuity: **plausible and likely**.
4. Breakdance ceiling effect: **partly true, but not sufficient as the main explanation**.
5. Decoy region selection is myopic: **definitely true**.
6. Rank loss margin too small: **possible, but lower priority than decoy validity and memory persistence.**

### Minimal fixes with highest expected return

1. **Constrain decoy validity before optimization.**
   - Reject any decoy whose average overlap with future GT over `f0:f14` exceeds a threshold.
   - Reject any decoy whose motion-projected centroid is too aligned with the object's natural velocity.
   - This is the cheapest fix and should directly address `car-shadow`.

2. **Add an exclusive-relocation term on eval-prefix frames.**
   - Penalize overlap with GT outside the decoy support, not just relative ranking.
   - Example: `L_excl = IoU(pred, GT)` or BCE on true mask outside bridge, alongside the decoy-positive term.
   - This should reduce "translated but still overlapping" masks.

3. **Turn on teacher-based memory supervision for inserted frames and the first 2-3 clean frames after each insert.**
   - Match `maskmem_features` and optionally `obj_ptr` to a decoy teacher.
   - This is the most important fix for `breakdance`.
   - It is already architecturally closer to the real attack surface than output-only v4.

### Fast diagnostic ablations

1. **Decoy-overlap audit on `car-shadow`.**
   - For the chosen offset, compute mean `IoU(shifted_GT_t, GT_t)` over `t=0..14`.
   - If it is high, the failure is mostly decoy invalidity, not optimization failure.

2. **Orthogonal-decoy rerun on `car-shadow`.**
   - Force the decoy direction to be orthogonal to estimated object motion, or choose the minimum-overlap direction over `f0:f14`.
   - If drop appears immediately, hypothesis 1 is confirmed.

3. **Memory-vs-output ablation on `breakdance`.**
   - Compare output-only v4 against output+teacher-memory on the same clip with 3 seeds.
   - If only the memory version improves mid-horizon drop, self-healing is the real bottleneck.

### Claim boundary

- Mislocalization poisoning is **not** uniformly reliable under FIFO memory with a short attack prefix.
- The regime is strongest when the decoy location is:
  - spatially separated from the object's future trajectory,
  - low-overlap with future GT,
  - visually object-like or distractor-like,
  - and the target is not already near the clean tracker's failure floor.

### Recommended paper framing if failures persist

Claim decoy success only on clips satisfying a measurable **decoy separability** condition:

- low future-overlap: `mean_t IoU(shifted_GT_t, GT_t) < tau1`
- motion misalignment: decoy direction cosine with mean object velocity `< tau2`
- non-fragile clean tracking: clean `J&F > tau3`

This makes the story honest: decoy poisoning is a distinct and real failure mode, but its effectiveness depends on whether the chosen decoy defines a genuinely different tracking hypothesis rather than a nearby continuation of the true one.
