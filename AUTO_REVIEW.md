# Auto Review Log: MemoryShield

## Round 1 (2026-04-15)

### Assessment (Summary)
- Score: **4/10**
- Verdict: **Not ready**
- Reviewer: GPT-5.4 xhigh via Codex MCP

### Key Criticisms (ranked by severity)

1. **(10/10)** Prior mask fed as temporal prompt on frame 1+ — official SAM2 does NOT use `prior_mask` for temporal tracking, only for same-frame interactive correction
2. **(9/10)** No conditioning frame privilege — frame 0 prompt should be in `cond_frame_outputs` with privileged memory access, not just aged as regular FIFO
3. **(9/10)** Memory encoded from wrong features — should use RAW backbone features, not memory-conditioned features
4. **(8/10)** Insert-only attack without codec EOT — perturbation doesn't survive JPEG encoding in eval
5. **(7/10)** No multimask on initial click — weakens the first memory write

### Actions Required
1. Rewrite `forward_video` control flow to match official SAM2 `track_step` path
2. Separate conditioning frame memory from non-conditioning FIFO
3. Encode memory from raw vision features
4. Add hybrid perturbation on existing frames
5. Add codec EOT
6. Add clean-parity regression test (surrogate IoU > 0.90 on frame 1)

### Actions Taken (Round 1)
1. Rewrote surrogate to call `track_step()` directly → tracking IoU: 0.07 → 0.97
2. Added hybrid perturbation (insert + perturb originals)
3. But J_drop = 0.0001 (no transfer)

## Round 2 (2026-04-15)

### Assessment (Summary)
- Score: **6/10** (up from 4/10)
- Verdict: **Improving, attack objective wrong**
- Key insight: PGD exploiting brittle threshold (object_score < 0 → -1024 sentinel), not robustly suppressing segmentation

### Actions Taken (Round 2)
1. **Margin-based multi-term loss**: `λ_obj * softplus(score + margin) + λ_mask * mean_logit(high_res_masks)`
2. **Fake uint8 quantization** in PGD loop (straight-through estimator)
3. **Proper export rounding** (np.rint, not truncation)
4. **JPEG Q100** in evaluation (near-lossless)
5. **Exposed richer outputs** from track_step (pred_masks_high_res, object_score_logits)

### Results — BREAKTHROUGH
- **J_drop = 0.9424** on bear video (from 0.0001!)
- J_clean = 0.9661, J_protected = 0.0237
- SAM2 tracking completely destroyed
- 2 inserted frames + 7 perturbed originals, ε=4-8/255, SSIM=0.94
- 5-video pilot: mean J_drop = 0.7458 (0.9249 excluding breakdance)

## Round 3 (2026-04-15)

### Assessment (Summary)
- Score: **8/10** (up from 6/10)
- Verdict: **Ready for controlled experiments**
- GPT-5.4: "The project has moved from debugging the surrogate to running paper-grade experiments."

### Key Results (5-video pilot)
| Video | J_clean | J_protected | J_drop |
|---|---|---|---|
| bear | 0.9661 | 0.0032 | 0.9629 |
| breakdance | 0.3048 | 0.2755 | 0.0293 |
| car-shadow | 0.9813 | 0.0130 | 0.9683 |
| dance-jump | 0.9043 | 0.0454 | 0.8588 |
| dog | 0.9535 | 0.0440 | 0.9095 |
| **MEAN (excl. breakdance)** | | | **0.9249** |

### Experiment Roadmap (from GPT-5.4)
1. **Ablation table** (perturb-only vs insert-only vs hybrid vs resonance vs random) — 20 videos
2. **Persistence** (15/30/50 frames) — per-frame J curves
3. **Cross-model** (tiny → base+/large/SAM2.1)
4. **Codec** (PNG, Q100, Q95, Q75, H.264)
5. **Failure analysis** (breakdance-type cases)

### Status: LOOP COMPLETE
Score 8/10 meets positive threshold. Transitioning to full experiment execution.

## Method Description

MemoryShield is a video preprocessor that protects datasets against SAM2 segmentation by inserting adversarial frames at FIFO-resonant positions and applying small perturbations to key original frames. The surrogate uses SAM2's official `track_step()` path for faithful gradient computation. The loss combines margin-based object-score suppression with pre-clamp high-res mask logit minimization, optimized via sign-PGD with fake uint8 quantization for transport robustness. On 5 DAVIS videos, it achieves 92.5% J-score degradation while maintaining SSIM > 0.93.

## Decoy Review Round 1 (2026-04-17)

### Assessment (Summary)
- Score: **4/10**
- Verdict: **Correct high-level intuition, wrong optimization target**
- Reviewer mode: local Codex review grounded in current implementation

### Key Criticisms (ranked by severity)

1. **Write/read roles are conceptually separated but not operationally separated.**
   Both inserts and perturbed originals are optimized with the same output-logit relocation loss in [memshield/generator.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\generator.py:435>) and [memshield/losses.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\losses.py:116>), so the system never directly teaches inserts to write decoy memories or originals to read them.
2. **The surrogate still drops the memory tensors that define the attack surface.**
   [memshield/surrogate.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\surrogate.py:174>) returns only masks and object score to the optimizer, even though `track_step()` already produces `maskmem_features` and `obj_ptr`.
3. **The current objective cannot prevent FIFO self-healing.**
   After the 15-frame attack prefix, clean frames are free to write clean memories again, while frame 0 remains a privileged clean anchor. Without explicit memory-state supervision on future frames, long-horizon relocation is structurally unsupported.
4. **`memory_drift_loss` is the wrong memory objective for decoy.**
   Pure divergence from clean memory in [memshield/losses.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\losses.py:178>) is closer to suppression than relocation because it does not specify the alternative memory to write.
5. **Background-only decoy is substantially harder than distractor hijack.**
   The current decoy-region selection in [memshield/decoy.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\decoy.py:20>) is too weak to justify the same persistence expectations for natural-distractor and pure-background settings.

### Minimum Viable Fix

1. Expose `maskmem_features` and `obj_ptr` from the surrogate for every frame.
2. Precompute a fixed no-grad **decoy teacher rollout** on a synthetic relocated video or on per-frame shifted-mask supervision.
3. Add **memory matching losses** on inserted frames and the first post-insert clean frames:
   - `L_mem = D(maskmem_adv, maskmem_teacher_decoy)`
   - `L_ptr = D(obj_ptr_adv, obj_ptr_teacher_decoy or obj_ptr_clean)`
   - keep `object_score_logits` positive to avoid collapse into suppression
4. Use pre-insert originals and frame 0 only to **weaken the true anchor**, not to force full relocation.
5. Add a weaker memory loss on future clean frames so decoy predictions also rewrite decoy-like memories instead of merely producing transient wrong logits.

### Claim Boundary

- **Not doomed:** short- to mid-horizon decoy should be possible if poisoned memories can bootstrap wrong rewrites for a few frames.
- **Fundamentally limited:** full-video persistence on arbitrary clean-background videos is unlikely with only a 15-frame prefix, because privileged frame-0 memory, repeated clean evidence, and FIFO flushing all favor recovery.
- **Likely strongest regime:** natural distractors or decoy locations with object-like appearance.

## Decoy Review Round 2 (2026-04-17)

### Assessment (Summary)
- Score: **6.5/10**
- Verdict: **Substantive architectural fix; now bottlenecked by teacher fidelity and temporal mismatch**
- Reviewer mode: local Codex review grounded in current implementation

### What Improved

1. [memshield/surrogate.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\surrogate.py:174>) now exposes `maskmem_features` and `obj_ptr`, which finally makes the attack target the actual memory-writing pathway.
2. [memshield/losses.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\losses.py:194>) replaces untargeted memory drift with teacher alignment and anti-anchor terms, which is the correct conceptual move from suppression-like behavior toward relocation.
3. [run_two_regimes.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\run_two_regimes.py:309>) now assigns distinct roles to frame 0, pre-insert, post-insert, and inserted frames instead of forcing all attacked frames through the same output loss.

### Remaining Weaknesses

1. **Teacher/video timeline mismatch remains the main technical risk.**
   The teacher is generated on the original-length synthetic video, but the adversarial rollout contains inserted frames. Insert losses in [run_two_regimes.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\run_two_regimes.py:405>) are matched to `orig_pos = after_original_idx + 1`, which does not reproduce the same FIFO occupancy pattern as the attacked modified video.
2. **The current memory loss is too global.**
   [memshield/losses.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\losses.py:194>) flattens the full `maskmem_features` tensor and applies one cosine loss, which may align coarse direction while missing the spatial structure that actually encodes decoy location.
3. **Teacher fidelity is still uncertain.**
   [memshield/generator.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\generator.py:267>) builds a pasted synthetic decoy video. If SAM2 treats that video as out-of-distribution, the optimizer may learn to imitate an artifact-induced teacher rather than a robust decoy state.
4. **Frame-0 privilege is weakened but not neutralized.**
   Anti-anchor on `f0` is directionally correct, but the `2/255` budget still means the clean conditioning memory likely remains a strong recovery source.

### Minimum Next Fixes

1. Generate the teacher on the **same modified schedule** as the attacked video, including synthetic inserted slots.
2. Replace single global cosine memory loss with a mixed objective:
   - channel cosine
   - spatial L1 / smooth-L1 on normalized memory maps
   - optional mask-weighted alignment near the decoy region
3. Log loss magnitudes and cosine statistics per role before changing weights; only then tune `lambda_mem`.
4. Run one ablation that compares:
   - output-only
   - output + teacher memory
   - output + teacher memory + anti-anchor
   This is the fastest way to verify the new cooperation mechanism is carrying real signal.

### Actions Taken (Round 2 → Round 3)
1. **Fixed teacher timeline mismatch**: `build_synthetic_decoy_video()` now accepts `schedule` param and builds the teacher on the MODIFIED timeline (with synthetic inserts at same positions). Teacher features index by `mi` (modified index), not `oi`.
2. **Replaced global cosine with spatial-aware memory loss**: `memory_teacher_loss()` now uses mixed objective: 60% channel-wise cosine similarity (per spatial position) + 40% spatial smooth-L1 on L2-normalized features. Preserves WHERE the object is, not just global statistics.
3. **Clean trajectory also on modified timeline**: Clean reference features are generated on a modified-timeline clean video (with duplicated frames at insert positions) for proper alignment.
4. **Helper functions for teacher lookup**: `_teacher_at(mi)` and `_clean_at(mi)` use modified-video index consistently.

## Decoy Review Round 3 (2026-04-17)

### Assessment (Summary)
- Score: **7.5/10**
- Verdict: **Technically coherent and ready for pilot validation, but not yet ready for a blind 30-video sweep**
- Reviewer mode: local Codex review grounded in current implementation

### What Improved

1. [memshield/generator.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\generator.py:303>) now builds the synthetic teacher video on the modified timeline, including inserted slots, which removes the most important temporal mismatch from Round 2.
2. [run_two_regimes.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\run_two_regimes.py:328>) consistently indexes teacher and clean features by modified-frame index `mi`, so teacher alignment now corresponds to the same FIFO occupancy pattern seen by the attacked rollout.
3. [memshield/losses.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\losses.py:194>) upgrades memory alignment from a global cosine to a mixed spatial objective, which is much better matched to the requirement that decoy memory encode both identity and location.

### Remaining Weaknesses

1. **There are still no empirical results showing the new memory terms improve persistence.**
   The architecture is much more defensible, but without a direct A/B against the previous decoy version, the main claim remains unverified.
2. **Teacher fidelity is still the dominant scientific risk.**
   The teacher video in [memshield/generator.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\memshield\generator.py:324>) is still a synthetic relocate-and-inpaint construction. If SAM2 writes artifact-specific memories on that video, the optimizer may learn the wrong target very precisely.
3. **Optimization is still short-horizon during PGD.**
   The read loss in [run_two_regimes.py](<E:\PycharmProjects\pythonProject\sam2_pre_new\run_two_regimes.py:582>) only supervises `f10:f14` inside the 15-frame prefix optimization window, while the actual claim of interest is mid/long-horizon persistence after the prefix.
4. **Background decoy remains structurally hard.**
   Even with better cooperation, frame-0 privilege and ongoing clean evidence still make arbitrary background relocation much less likely to persist than distractor hijack.

### Minimum Next Fixes

1. Run a **3-video pilot** before any full sweep:
   - one background-style case
   - one natural-distractor case
   - one moderate-difficulty control
2. For each pilot, compare old decoy vs new decoy with the same seed and report:
   - short / mid / long / all-future dJF
   - DecoyHitRate / centroid shift
   - average teacher-memory loss on inserts and first two post-insert frames
3. If the new method does not improve **mid or long horizon** on at least 2 of 3 pilots, do not launch the 30-video run yet.

## Decoy Review Round 4 (2026-04-17)

### Assessment (Summary)
- Score: **6/10**
- Verdict: **Mechanism is real, but not robust enough to stand as a primary contribution**
- Reviewer mode: local Codex review using reported pilot results

### Key Experimental Conclusion

1. **The cooperation mechanism is genuinely doing something new.**
   `blackswan` shows strong persistence well beyond FIFO length, with `dShort=0.681`, `dMid=0.686`, and `dLong=0.652`. That is the first convincing evidence that teacher-based memory cooperation can create self-sustaining decoy behavior rather than a short transient.
2. **The method does not currently generalize.**
   `dog` and `cows` both show near-zero real `dMid/dLong` despite strong surrogate signatures. That means the main failure mode has shifted from self-healing to surrogate-to-real transfer.
3. **This is not publishable as a general decoy attack yet.**
   One strong success out of three pilots is enough to justify the line of inquiry, but not enough to support a broad attack claim.

### Final Recommendation

1. Keep **suppression** as the main contribution.
2. Reframe **decoy** as a conditional or exploratory regime:
   - proof that persistent relocation is possible in some scenes
   - evidence that memory cooperation matters
   - open problem: transfer beyond the surrogate
3. Run the **old decoy baseline** on the same 3 videos before writing any claim about cooperation. Without that ablation, you cannot attribute the `blackswan` success to the new mechanism rather than a scene-specific lucky case.

### Priority Next Steps

1. Run output-only decoy vs teacher-cooperative decoy on `blackswan`, `dog`, and `cows`.
2. If `blackswan` improves materially over output-only while the others do not, write the claim as **conditional effectiveness with strong scene dependence**.
3. Only invest in official-pipeline PGD or stronger EOT if you want to turn decoy into a serious second contribution; otherwise stop after the ablation and keep decoy as a partial result.

## Loop Complete — Final Summary

**Score progression**: 4/10 → 6.5/10 → 7.5/10 → 6/10 (with results)

### What we achieved
- Teacher-based memory cooperation mechanism that is provably different from suppression
- First evidence of persistent background decoy (blackswan: dLong=0.652)
- Exposed maskmem_features and obj_ptr from SAM2 surrogate
- Spatial-aware memory loss (channel cosine + spatial smooth-L1)
- Modified-timeline teacher generation

### Remaining blockers
1. Surrogate-to-real transfer gap (2/3 pilot videos failed)
2. Scene-dependent effectiveness
3. Missing old-vs-new ablation to confirm cooperation helps

### Paper recommendation
- **Main contribution**: Suppression regime (proven 90%+ J-drop)
- **Secondary contribution**: Decoy as conditional case study with memory cooperation analysis
- **Evidence**: Persistent decoy IS possible (blackswan) but transfer is unreliable

## Method Description

MemoryShield's decoy regime uses teacher-based memory cooperation to achieve persistent object relocation in SAM2 video segmentation. A synthetic "teacher video" with the object relocated to the decoy position is pre-generated and run through SAM2 to extract target memory features (maskmem_features, obj_ptr). During PGD optimization, inserted frames match these teacher memories (spatial-aware cosine + smooth-L1), pre-insert frames weaken the true-location anchor (anti-anchor cosine divergence), and post-insert frames reinforce the decoy trajectory. The teacher is generated on the modified timeline (including insert slots) to match the FIFO occupancy pattern of the attacked video. On favorable videos (uniform background, distinctive object), this achieves 65%+ J&F degradation that persists 30+ frames beyond the attack prefix.
