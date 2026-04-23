# Research Review: Why MemoryShield v2 Doesn't Work (vs Decoy v4)

**Date**: 2026-04-23
**Invoked via**: `/research-review 为什么现在的方法没有效果？decoyv4的效果不是还可以吗？`
**Codex thread**: `019db850-aff6-7250-8572-121d78271dad`
**Reviewer model**: gpt-5.4 at `model_reasoning_effort: xhigh`

## TL;DR

**Decoy v4's 92.5% J-drop was very likely an eval-window illusion**, not genuine memory-bank poisoning. v4's `perturb_set` directly δ-perturbed ~4-5 of its eval frames (with EVAL_START=10, attack range overlapping eval). MemoryShield v2 is architecturally cleaner: eval window is strictly clean-suffix AFTER all perturbations, so the test measures only bank-poisoning effect in isolation. That effect, on SAM2.1-Tiny, is approximately zero — confirmed by the multi-clip B2 causal ablation showing `|delta_J| < 0.01` when the non-cond FIFO bank is removed entirely.

v4 was not "a stronger bank-poisoning attack" than v2. It was "direct δ damage on eval frames, packaged with inserts for narrative". That narrative doesn't hold up under v2's clean-suffix eval.

## Context: the two methods in the same repo

### Decoy v4 (`memshield/generator_v4.py`, commit `283f0ea`)

- **Attack surface**: sparse δ on `perturb_set = {0} ∪ {pos, pos+1, pos+2 for each insert pos}`. For 3 inserts at {6, 12, 14} in a T=21 prefix, `perturb_set = {0, 6, 7, 8, 12, 13, 14, 15, 16}` — 9 of 21 frames directly δ-perturbed.
- **K_ins inserts**: 3 at scheduler-selected positions.
- **Loss**: `decoy_target_loss` on pre-clamp `pred_masks_high_res` + `object_score_logits` — direct end-to-end supervision.
  - `support`/`bridge`/`decoy` positive pressure + `suppress` (true-location) negative pressure + rank term (decoy_mean > true_mean).
- **Surrogate**: `track_step()` called directly. `fake_uint8_quantize` in the PGD loop.
- **Eval**: `EVAL_START = 10` hardcoded in `sam2long_eval.py`. For T_prefix=21, frames 10-20 are within the attacked prefix AND in the eval window. Only frames 21+ are genuinely "clean suffix".
- **Claimed result**: 92.5% J-drop on DAVIS with SSIM > 0.93 (per `project_memshield_breakthrough.md` memory file).
- **Artifact on Pro 6000**: `~/sam2-pre-new/results_regimes/regimes_results.json`. `bear` suppression regime shows `mean_jf = 0.000` over `n_eval_frames = 5`.

### MemoryShield v2 (`memshield/*_v2.py`, commit `d5e7b6d`)

- **Attack surface**: dense δ on ALL of prefix f0..f14 at ε=4/255 (ε_f0=2/255).
- **K_ins inserts**: 3 at canonical write-aligned seed-plus-boundary schedule w_positions={6, 12, 17}.
- **T_prefix=15, eval_window=7, T_full=22**. Eval is DAVIS frames 15..21. **Zero δ on eval frames. Zero overlap between perturb range and eval range.**
- **Loss**: `L_loss` (CVaR insert targeting) + `L_rec` (eval-logit suppression) + `L_stale` (3-bin KL on memory-attention mass pushing A^insert up) + `L_fid` (augmented-Lagrangian LPIPS penalty).
- **Surrogate**: `SAM2VideoAdapter` bypasses `@torch.inference_mode` decorators; bf16 autocast; full graph from inserts → memory → memory-attention → MaskDecoder → eval-logits.
- **Results on dog**: R001 (K=1, no LPIPS) J-drop 0.0009; R002 (K=1, LPIPS 0.10) J-drop 0.0004; R003 (K=3 canonical, no LPIPS) J-drop 0.0013.

### Cross-check evidence (auto-review Rounds 1-4)

- **LPIPS floor study**: natural DAVIS adjacent-frame LPIPS mean 0.25 (dog), 0.38 (bmx-trees). ProPainter insert floor 0.67-0.89 regardless of decoy_dx. The "LPIPS ≤ 0.10" budget is impossible by construction.
- **Attention trace (D1)**: on R003 eval frames, `A_insert` mean = 0.515 across 6 of 7 eval frames (0.000 on frame 24 after FIFO eviction). **Inserts ARE in the bank and ARE being attended at half of foreground mass, yet J is unchanged.**
- **Bank causal ablation (B2)**: removing the entire non-cond FIFO bank during eval on clean videos changes mean J by `|delta| < 0.01` across 5 DAVIS clips (dog, cows, bmx-trees, blackswan, breakdance). On blackswan and breakdance, ablation actually _improves_ J slightly. **The bank is not a causal input to SAM2.1-Tiny segmentation on these clips.**

## Codex Round 1 Response (summary)

**Hypothesis ranking for v2 failure**:

1. **Eval-window illusion (#1) + mechanism-loss non-causality (#2)** are the joint dominant explanation. v4 conflated direct δ-on-eval with "the decoy attack worked". v2 removes that confound and exposes the true bank-poisoning strength ≈ 0.
2. **End-to-end output loss vs mechanism loss (#2)** is the main mechanistic reason v2 fails. `L_stale` successfully raises `A_insert` but `A_insert` is not causal for segmentation on Tiny. High attention to poison does not imply high J-drop.
3. **Loss dominance from L_fid saturation (#4)** matters only for R002 (LPIPS on); R001/R003 had LPIPS off and still failed.
4. **Dense vs sparse perturbation (#3)** is lowest. Sparse v4 worked because sparse frames were counted in eval; density is not the decisive variable.

**Was v4 really doing bank poisoning?**

> Probably not in the claimed sense. v4 used direct `decoy_target_loss` on SAM2 logits/object scores and evaluated from `EVAL_START=10`, while its perturb set includes later eval frames. That means the reported 92.5% is not clean evidence of "poisoned memory attacks future frames." It is evidence that direct low-epsilon perturbations plus inserted frames can break SAM2 on/near attacked frames.

**Minimum honest redesign** (if target remains SAM2.1-Tiny + eval must be clean suffix):

> There is no small redesign of bank poisoning that should be expected to resurrect v4-level performance. The minimum honest redesign is to change the attack surface:
> - Target a causal path: frame-0 conditioning memory, current-frame pixels, prompt pathway, or a different VOS model where bank ablation shows `delta_J >= 0.10-0.15`.
> - If staying with clean suffix, bring back v4-style end-to-end eval-logit / object-score loss on the clean suffix, but only after confirming the target model actually depends on the memory bank.
> - Drop `L_stale` as a success proxy. Treat it as a diagnostic only.
> - Remove or relax the infeasible LPIPS 0.10 insert constraint.
> - Select checkpoints by actual clean-suffix J-drop, not training loss.

**Single verification experiment**:

> Re-evaluate the exact v4 artifacts on a clean-suffix window starting after the last δ-perturbed original, e.g. `eval_start = max(perturb_set) + 1` around frame 17. Also report the original v4 window split into δ-touched vs non-δ frames.
>
> Expected result: J-drop collapses from ~0.92 to near v2 scale, likely `~0.001-0.02` on dog, maybe a small transient on the first few post-δ frames.
>
> Interpretation:
> - If it collapses: v4 was an eval-window confound, not proof of future-frame bank poisoning.
> - If it remains high on clean suffix: v2's loss/implementation is missing a real mechanism, and B2 must be reconciled.
> - If original v4 window no longer reproduces: the old result was a pipeline or metric artifact.

**Brutal assessment**:

> As evidence that "decoy bank attacks work on future clean SAM2.1-Tiny frames," v4 is misleading. As evidence that direct output-space adversarial perturbations can break SAM2 on evaluated frames, it is meaningful but much less novel.

## Practical conclusions

1. **Stop treating v4's 92.5% as a benchmark to match.** It was measuring a different, easier quantity — direct frame damage with insert packaging.
2. **v2's setup is the honest test.** The near-zero J-drop is the correct answer for "can prefix-only preprocessor inserts damage clean future-frame segmentation on SAM2.1-Tiny". The paper proposal's claim of J-drop ≥ 0.55 under clean-suffix eval is architecturally infeasible on this target.
3. **The single useful verification experiment** (Codex recommendation): re-evaluate v4 artifacts on v2-style clean-suffix window. Expected J-drop → ~0.001-0.02 on dog. Running cost: minutes if artifacts are still on Pro 6000 (`~/sam2-pre-new/results_regimes/videos/`).
4. **If the user rejects both pivot-target-model (Shape A) and change-attack-surface (Shape C) options from the auto-review loop, the project is architecturally blocked** regardless of further tuning. This is not a method-detail issue.

## Status of v4 artifacts on Pro 6000

- `~/sam2-pre-new/results_regimes/regimes_results.json`: full JSON of per-clip per-regime J scores, `n_eval_frames=5` per clip.
- `~/sam2-pre-new/results_regimes/videos/`: the saved attacked JPEG sequences (decoy + suppression regimes).
- Clean-suffix re-evaluation requires: load attacked JPEGs, identify the last attacked frame index per clip, run SAM2 propagate, compute J on frames after that index against DAVIS GT.

Writing a dedicated driver is a ~1-hour task and the experiment itself is minutes per clip. Artifacts are not deleted and the verification is tractable.

## Open question for the user

Having now seen Codex's reasoning, do you want:

- **V1**: Run the verification experiment — re-evaluate v4 artifacts on clean-suffix eval window. If Codex's prediction holds (J-drop collapses to ~v2 scale), you have the smoking gun that v4 was an eval-window confound.
- **V2**: Accept Codex's analytical argument without running the verification. The B2 result (bank is architecturally marginal) already implies v4 couldn't have been bank poisoning. Move to the decision point: pivot target model, pivot attack surface, or stop the project.
- **V3**: Revisit the proposal paper. If the paper's core claim depends on v4-style results, acknowledge that those results don't hold under honest eval, and scope-narrow the claim to what the data supports (essentially nothing on SAM2.1-Tiny under clean-suffix eval).
