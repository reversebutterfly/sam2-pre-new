# Experiment Plan

**Problem**: Establish whether SAM2 video segmentation admits two distinct adversarial memory-poisoning regimes under matched attack budget, and test whether a confidence-gated memory tree (SAM2Long) changes which regime survives.
**Method Thesis**: Under the same frame schedule, perturbation budget, surrogate, and optimization procedure, SAM2 can be driven into either an `absence poisoning` regime or a `mislocalization poisoning` regime. FIFO memory favors absence poisoning, while confidence/tree selection should relatively preserve mislocalization poisoning.
**Date**: 2026-04-16

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: Two poisoning regimes exist under matched budget. | Turns the paper from "two losses" into a mechanistic contribution. | On the same videos and same attack schedule, Suppression and Decoy both degrade future clean-frame tracking but produce different future-frame signatures: Suppression yields object absence / area collapse, Decoy yields positive objectness with displaced masks. | B1, B2, B4 |
| C2: Memory policy changes which regime persists. | Makes the paper predictive rather than descriptive. | When the same attacked clips are evaluated on SAM2Long, Decoy retains a larger fraction of its attack effect than Suppression. Ideally the gap closes sharply or reverses. | B3, B4 |
| Anti-claim A1: The difference is just because one loss is stronger. | Reviewer will otherwise dismiss the study as unequal optimization difficulty. | Strictly matched frame set, PGD steps, surrogate, fake quantization, clip length, prompting, and perturbation bounds; paired per-video tests on the same eligible subset. | B1 |
| Anti-claim A2: The attack is not memory-mediated. | Without this, the paper is not really about memory poisoning. | Disjoint attack/eval windows, benign insertion controls, and memory-reset recovery. | B2 |

## Paper Storyline
- Main paper must prove:
  1. There are two matched-budget poisoning regimes with distinct future-frame behaviors.
  2. These effects are memory-mediated, not just direct corruption of scored frames.
  3. Changing memory selection from FIFO to confidence/tree selection shifts the relative advantage toward Decoy.
- Appendix can support:
  1. Whole-clip metrics including attacked frames.
  2. SAM2Long parameter sweeps beyond the default.
  3. Extra qualitative cases and per-frame curves.
- Experiments intentionally cut:
  1. Large baseline zoo.
  2. Defense benchmark suite.
  3. Cross-model transfer beyond SAM2/SAM2Long, unless core results are already strong.

## Shared Evaluation Framework
- **Dataset**: DAVIS 2017 validation, 20 clips, 15 original frames per clip.
- **Tracking setting**: single prompted target appearing at frame 0, same point prompt extraction for every method.
- **Attack schedule**: same for both regimes.
  Attackable originals: `{0, 1, 2, 3, 7, 8, 9}`.
  Insertions: 2 frames, after original `f1` and `f7`.
- **Surrogate**: official SAM2 `track_step()`-based surrogate with full memory pipeline.
- **Optimization**: same sign-PGD, same fake uint8 quantization, same number of steps, same step-size heuristic, same stop rule.
- **Perturbation bounds**: keep exactly the current matched deployment budget.
  `f0`: 2/255.
  attacked originals `f1,f2,f3,f7,f8,f9`: 4/255.
  inserted frames: current strong/weak insert bounds from the deployed hybrid setting.
- **Primary clean evaluation window**: untouched original frames `f10:f14`.
- **Eligible subset definition**: pre-register before attack.
  Primary subset: videos with clean `J&F(f10:f14) >= 0.60`.
  Secondary report: all 20 videos, plus coverage rate of the eligible subset.
- **Statistics**: paired mean, median, bootstrap 95% CI across videos, and Wilcoxon signed-rank for Suppression vs Decoy on the matched eligible subset.

## Metrics and Regime Signatures
- **Primary outcome**: `J&F` drop on clean future frames `f10:f14`.
- **Secondary outcomes**: `J`, `F`, attack success rate (`J&F drop >= 20 points`), SSIM on attacked frames.
- **Suppression signature metrics**:
  1. `NegScoreRate`: fraction of eval frames with `object_score < 0`.
  2. `CollapseRate`: fraction of eval frames with predicted area below 1% of GT area or NO_OBJ sentinel if instrumented.
- **Decoy signature metrics**:
  1. `PosScoreRate`: fraction of eval frames with `object_score > 0`.
  2. `DecoyHitRate`: fraction of eval frames where `IoU(pred, shifted_GT) > IoU(pred, GT)`.
  3. `CentroidShift`: normalized distance from GT centroid toward the decoy centroid.
- **SAM2Long transfer metric**:
  `RetentionRatio = Drop_on_SAM2Long / Drop_on_SAM2`.
  Core prediction: `RetentionRatio_Decoy > RetentionRatio_Suppression`.

## Experiment Blocks

### Block 1: Core Matched Regime Comparison
- **Claim tested**: C1.
- **Why this block exists**: This is the anchor table. It establishes that the two regimes are real under identical attack conditions, not just differently tuned attacks.
- **Dataset / split / task**: DAVIS 2017 val, 20 clips, 15-frame single-target VOS.
- **Compared systems**:
  1. Clean.
  2. Suppression Hybrid.
  3. Decoy Hybrid v5.
- **Metrics**:
  1. `J&F`, `J`, `F` on `f10:f14`.
  2. `J&F` drop relative to clean.
  3. SSIM on attacked frames.
  4. Coverage rate of eligible videos.
- **Setup details**:
  1. Same frame schedule, same perturbation bounds, same surrogate, same PGD, same quantization, same prompt.
  2. No per-video or per-method schedule retuning.
  3. Report results on both the all-video set and the eligible subset.
- **Success criterion**:
  1. Both regimes materially reduce future-frame `J&F`.
  2. Suppression is stronger on vanilla SAM2 by at least a modest paired margin.
  3. Signature metrics cleanly separate the regimes.
- **Failure interpretation**:
  1. If Decoy barely degrades performance, the paper becomes mostly about absence poisoning.
  2. If signatures overlap, the regime story is too weak for a full paper.
- **Table / figure target**:
  1. `Table 1`: all-video and eligible-subset comparison.
  2. `Figure 3`: signature bars or violin plots for `NegScoreRate`, `CollapseRate`, `PosScoreRate`, `DecoyHitRate`.
- **Priority**: MUST-RUN.

### Block 2: Mechanism Isolation and Causal Tests
- **Claim tested**: C1 and A2.
- **Why this block exists**: It proves that both regimes are memory-mediated and clarifies how inserts and perturbed originals cooperate.
- **Dataset / split / task**: 8 representative eligible DAVIS clips if runtime is acceptable; otherwise fall back to the pre-registered 5-video pilot subset.
- **Compared systems**:
  1. Clean.
  2. Clean + memory reset at `f10`.
  3. Benign insertions at `f1` and `f7`.
  4. Suppression, perturb-only.
  5. Suppression, insert-only.
  6. Suppression, hybrid.
  7. Suppression, hybrid + memory reset at `f10`.
  8. Decoy, perturb-only.
  9. Decoy, insert-only.
  10. Decoy, hybrid.
  11. Decoy, hybrid + memory reset at `f10`.
- **Metrics**:
  1. `J&F` on `f10:f14`.
  2. Signature metrics from the previous section.
  3. Optional per-frame curves over `f10:f14`.
- **Setup details**:
  1. Use the exact same attackable frames and insertion positions as the full hybrids.
  2. Evaluation is strictly on untouched future originals.
  3. Memory reset must be compared to `clean+reset`, not to plain clean.
- **Success criterion**:
  1. Benign insertion is much weaker than adversarial insertion.
  2. Reset recovers most of the damage for both regimes.
  3. Hybrid is stronger than at least each of its components individually for both regimes.
  4. Suppression shows absence signatures; Decoy shows relocation signatures.
- **Failure interpretation**:
  1. If reset does not help, the memory-poisoning claim weakens substantially.
  2. If insert-only dominates everything, the cooperation story is weaker than expected.
- **Table / figure target**:
  1. `Table 2`: mechanism isolation matrix.
  2. `Figure 4`: causal diagrams + 2 to 4 representative frame sequences.
- **Priority**: MUST-RUN.

### Block 3: SAM2Long Zero-Shot Transfer Test
- **Claim tested**: C2.
- **Why this block exists**: This is the prediction-validation block. It upgrades the paper from a study of SAM2 to a study of how memory policy reshapes the attack surface.
- **Dataset / split / task**: Same 20 attacked clips from Block 1.
- **Compared systems**:
  1. SAM2-tiny evaluation on clean / Suppression / Decoy.
  2. SAM2Long-tiny evaluation on the exact same clean / Suppression / Decoy clips.
- **Metrics**:
  1. `J&F` drop on `f10:f14`.
  2. `RetentionRatio`.
  3. Relative gap closure: `(Drop_Supp - Drop_Decoy)` on SAM2 versus SAM2Long.
- **Setup details**:
  1. Use SAM2Long's default public inference setting first: `--num_pathway 3 --iou_thre 0.1 --uncertainty 2`.
  2. Use the same SAM2 checkpoint family for both SAM2 and SAM2Long.
  3. Do not re-optimize attacks on SAM2Long for the main result. This is a zero-shot transfer test.
  4. Install SAM2Long in a separate environment, as recommended by the repo.
- **Success criterion**:
  1. Suppression weakens on SAM2Long relative to SAM2.
  2. Decoy retains a larger fraction of its effect than Suppression.
  3. Ideally the performance gap between Suppression and Decoy narrows sharply, or Decoy overtakes Suppression.
- **Failure interpretation**:
  1. If both regimes weaken equally, the memory-policy prediction is not supported.
  2. If Suppression still dominates by the same margin, the tree-selection story needs revision.
- **Table / figure target**:
  1. `Table 3`: SAM2 versus SAM2Long transfer comparison.
  2. `Figure 5`: retention-ratio bar chart and 2 qualitative examples.
- **Priority**: MUST-RUN.

### Block 4: SAM2Long Targeted Parameter Sweep
- **Claim tested**: C2 mechanistically.
- **Why this block exists**: It links the observed transfer behavior to the actual selection rules in SAM2Long rather than treating SAM2Long as a black box.
- **Dataset / split / task**: 5 representative clips from the eligible subset.
- **Compared systems**:
  1. SAM2Long default: `num_pathway=3`, `iou_thre=0.1`, `uncertainty=2`.
  2. Weak gate: `iou_thre=0.0`.
  3. Strong gate: `iou_thre=0.2`.
- **Metrics**:
  1. `J&F` drop.
  2. `RetentionRatio`.
  3. If instrumentation is feasible: fraction of attacked frames admitted to the memory bank, mean selected `o_i`, mean selected predicted IoU.
- **Setup details**:
  1. Reuse the same attacked videos from Block 1.
  2. Keep `num_pathway` fixed to isolate the effect of confidence gating.
  3. If there is spare budget, appendix-only sweep `num_pathway in {1, 3, 5}`.
- **Success criterion**:
  1. Increasing `iou_thre` hurts Suppression more than Decoy.
  2. Decoy remains comparatively stable because it preserves positive objectness and usable IoU.
- **Failure interpretation**:
  1. If both regimes react the same way, the policy-shift interpretation is too coarse.
- **Table / figure target**:
  1. `Appendix Table A1` or main paper if Block 3 is especially strong.
- **Priority**: NICE-TO-HAVE, but highly recommended.

### Block 5: Qualitative and Signature Analysis
- **Claim tested**: C1 and C2 visually.
- **Why this block exists**: Reviewers need to see that the two failures look different, not just score different.
- **Dataset / split / task**: 4 videos: best Suppression case, best Decoy case, one ambiguous case, one failure case.
- **Compared systems**:
  1. Clean.
  2. Suppression.
  3. Decoy.
  4. SAM2Long outputs for the same attacked clips.
- **Metrics**:
  1. Frame triptychs with mask overlays.
  2. Object-score / area / centroid-shift curves.
- **Setup details**:
  1. Use fixed visual templates and the same frames across methods.
  2. Include the decoy target overlay for Decoy cases.
- **Success criterion**:
  1. Visuals make the regime distinction obvious in one glance.
- **Failure interpretation**:
  1. If visuals are ambiguous, the interpretability story weakens.
- **Table / figure target**:
  1. `Figure 1` teaser and `Figure 6` qualitative comparison.
- **Priority**: MUST-RUN for the final paper, but low GPU cost.

## Minimum Submission Package
- **MUST**:
  1. Block 1 on all 20 clips.
  2. Block 2 on 8 clips if feasible, otherwise 5 clips with pre-registered selection.
  3. Block 3 on the same 20 attacked clips.
  4. Block 5 qualitative and signature figures.
- **SHOULD**:
  1. Block 4 on 5 clips.
- **CUT FIRST if runtime slips**:
  1. Extra pathway sweeps.
  2. Whole-clip metrics.
  3. Any additional backbone or dataset.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Sanity + eligibility | Clean SAM2, clean SAM2Long on 20 clips; verify `J&F`, signature extraction, reset pipeline | Proceed only if clean `J&F` is stable and eligible subset size is at least 8 | 2 GPU-h | Metric bugs or broken reset implementation |
| M1 | Core SAM2 comparison | Block 1 on 20 clips for Suppression and Decoy | Proceed only if both attacks beat clean on the eligible subset and Suppression remains strong | 18-22 GPU-h | Decoy too weak; mitigate by limiting paper scope or retuning Decoy once |
| M2 | Mechanism isolation | Block 2 on 8 clips, fallback to 5 if M1 runtime is too slow | Proceed only if reset clearly recovers and benign insertions are weak | 8-10 GPU-h | Too many conditions; mitigate by reusing hybrid attacks and shrinking to 5 clips |
| M3 | SAM2Long transfer | Block 3 on the exact attacked clips from M1 | Proceed only if `RetentionRatio_Decoy > RetentionRatio_Suppression` or the gap closes materially | 3-4 GPU-h | SAM2Long setup friction; mitigate with separate environment and default public settings |
| M4 | SAM2Long sweep + figures | Block 4 on 5 clips, Block 5 figures | Keep only if remaining budget is at least 6 GPU-h after M3 | 4-6 GPU-h | Instrumentation cost; mitigate by reporting only output-space metrics |
| M5 | Buffer | One retune pass or rerun failed clips | Use only if a decisive block underperforms due to implementation error | 6-8 GPU-h | Budget overrun |

## Compute and Data Budget
- **Total estimated GPU-hours**: 41-46 GPU-h for the minimum convincing package.
- **Buffer within remaining budget**: 4-9 GPU-h.
- **Data preparation needs**:
  1. Freeze the exact 20 DAVIS clip list.
  2. Freeze the 8-video mechanism subset before looking at mechanism results.
  3. Save clean and attacked protected clips so SAM2Long can reuse them without re-optimization.
- **Human evaluation needs**: none.
- **Biggest bottleneck**: attack optimization time for the extra mechanism variants.

## Fairness Checklist
- Same prompt, same clip, same frame schedule, same attackable originals.
- Same surrogate and same PGD implementation.
- Same quantization-aware optimization and export path.
- Same perturbation bounds and number of optimization steps.
- Same eligible subset for both regimes.
- Same SAM2 checkpoint family for SAM2 and SAM2Long evaluation.
- No per-method schedule retuning.

## Risks and Mitigations
- **Risk**: Decoy remains much weaker than Suppression on SAM2 and also on SAM2Long.
- **Mitigation**: Frame the paper around a dominant absence regime plus an incomplete but interpretable mislocalization regime only if Block 3 still shows the predicted relative retention.

- **Risk**: Memory reset implementation is not exact.
- **Mitigation**: Always compare against `clean+reset`; inspect one manually verified video before batch runs.

- **Risk**: Eligible subset is too small.
- **Mitigation**: Report all-video and eligible-subset results together; use coverage as an explicit metric instead of hiding it.

- **Risk**: SAM2Long setup consumes too much time.
- **Mitigation**: Use the repo defaults first and delay any instrumentation or parameter sweep until the default result is reproduced.

- **Risk**: Reviewers say "this is just two losses."
- **Mitigation**: Make signature metrics and SAM2Long prediction central, not appendix material.

## Final Checklist
- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] Memory mediation is causally tested
- [ ] SAM2Long prediction is tested under matched conditions
- [ ] Nice-to-have runs are separated from must-run runs
