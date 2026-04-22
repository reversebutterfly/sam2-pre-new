# Research Review — Next Directions (2026-04-22)

User question: (Q1) 固定插帧位置是否太笨？能否用 target prior 自动选最优位置？(Q2) SAM2 用什么 IoU 写 memory bank？(Q3) 能否让 insert 帧在 SAM2 的 predicted IoU 视角下成为最高分，从而写入 memory 并污染后续帧？

Reviewer: `gpt-5.4` xhigh via codex MCP, thread `019db2e2-27d0-7961-8930-4d1ab3de3b6e`.

## Key technical correction on SAM2 memory mechanics

Our FIFO-everything surrogate (memshield/surrogate.py:136-180) is **correct for vanilla SAM2**. There is no cross-frame "only write high-quality frames" admission gate. `track_step` always calls `_encode_new_memory(pred_masks_high_res=high_res_masks, object_score_logits=object_score_logits)` — the FIFO admits every frame, and future reads use `cond_frame_outputs + last (num_maskmem-1) non-conditioning outputs`.

**The predicted-IoU selection is within-frame, NOT across-frames**:
- SAM2 mask decoder outputs multiple mask candidates per frame (typically 3, plus the single-mask token)
- A learned MLP head predicts an IoU scalar per candidate
- If `multimask_output_in_sam=True` AND `multimask_output_for_tracking=True`, the argmax-IoU candidate's high-res mask is fed into `_encode_new_memory` → that mask's features go into `maskmem_features`
- **Default for tracking frames is OFF** — open-source SAM2 does NOT use multimask during tracking by default. Need to verify our runtime config before pursuing Q3.
- **`obj_ptr` comes from the single-mask token by default** (`use_multimask_token_for_obj_ptr=False`) — so even if we flip the selected mask, the pointer memory may be unchanged. Two memory channels can move independently.
- `object_score_logits` (object presence / occlusion head, which we DO already optimize via margins) ≠ predicted IoU (mask quality head). Different heads, different attack surfaces.

## Verdict per direction

### Q1 — Dynamic insertion position (from target prior)

- **Not publishable on its own**; incremental attack engineering. Fixed `f3/f7/f11` IS already an architecture-aware baseline tuned to FIFO capacity.
- **Minimum convincing experiment chain**:
  1. Oracle gap on 5-8 clips: brute-force search insert positions in the 15-frame prefix at same 3-insert budget + min-gap constraint, cheap screening PGD first, top few full PGD. If oracle barely beats fixed → stop.
  2. If oracle wins: 20-clip practical scheduler recovering meaningful fraction at equal compute.
  3. Correlate chosen frames with clean-pass uncertainty / memory-change / motion to EXPLAIN why.
- **Recommended design pattern**: "dynamic strong anchor + resonance sustain" — pick only the FIRST strong insert adaptively, keep weak inserts on the 6-period resonance. Preserves the memory story.
- **Scoring function on clean forward pass**: `score_t = α*uncertainty_t + β*memory_change_t + γ*motion_t`
- **Do NOT build an RL scheduler** — looks like overfitting, distracts from the attack mechanism.
- **Prior art (adaptive keyframe selection in video attacks)**: ICASSP 2022 gradient-based keyframe selection; Sensors 2022 gradient-feedback key-frame/pixel; DeepSAVA Neural Networks 2024 Bayesian critical-frame search. All video classification — nothing for SAM2/VOS memory poisoning specifically.

### Q3 — Predicted-IoU-targeted attack on mask candidate selection

- **Conditional win**: real new attack vector ONLY if `multimask_output_for_tracking=True` in our runtime. If off → Q3 is dead (no candidate race to hijack).
- **Distinctness from current attack**: not automatic. Current CVaR + object_score margins may already indirectly make the decoy candidate win. Q3 is truly new only if it changes candidate selection OR long-horizon persistence beyond what logit shaping does.
- **Simplest implementation** (once surrogate exposes candidate-level outputs):
  ```python
  # On insert frames, after SAM heads expose high_res_multimasks (3 cands) and ious (3 cands):
  S_m = decoy_score(high_res_multimasks_m) - true_score(high_res_multimasks_m)   # per candidate
  w   = softmax(S / tau)                                                           # soft best-decoy weighting
  L_iou = -(w * ious).sum()                                                        # reward high pred-IoU on decoy-like cands
  ```
  Soft over candidates, NOT hard argmax-through-selection (avoids gradient blockage).
- **First-test distinctness** (before investing a week):
  1. Expose candidate-level outputs from `_forward_sam_heads()`: `high_res_multimasks`, `ious`, selected idx, `maskmem_features`, `obj_ptr`
  2. Compare `Base`, `Base + IoU-aux`, `IoU-aux only` at same schedule + budget
  3. Measure: candidate-flip rate, post-prefix memory drift / teacher alignment, long-horizon decoy metrics
  4. If `Base + IoU-aux` improves persistence WITHOUT just making insert-frame decoy logits larger → Q3 is real.
- **Important measurement discipline**: measure BOTH `maskmem_features` AND `obj_ptr` — because `obj_ptr` may come from single-mask token by default, Q3 could poison spatial memory only.

### Publishability

- Q1 alone: optimizer policy improvement. Not top-venue-level.
- Q3 alone (if real): architectural vulnerability — "SAM2's own mask-quality selector seeds poisoned memory" — can lift the paper IF framed that way.
- Q1 + Q3 together + (a) mechanism isolation, (b) persistence past attack prefix, (c) memory-policy defense implications, (d) mitigation angle → plausible NeurIPS/ICML.
- Without (a)-(d), still reads as "better attack tuning on SAM2 with fidelity trade-off analysis."

## Failure modes

### Q1 worst cases (3 ranked):
1. Dynamic scheduler rediscovers roughly fixed positions → null result
2. Gain from extra schedule-search compute, not from memory-awareness → methodology confounder
3. Oracle beats fixed only on unstable clips; effect disappears under seed averaging

### Q3 worst cases (3 ranked):
1. `multimask_output_for_tracking` OFF in our runtime → entire direction attacks a dormant branch
2. IoU optimization changes the head numerically but never flips the selected mask → no downstream memory change
3. Only `maskmem_features` moves, not `obj_ptr` → weaker effect than story implies

### Generic red flag
Reviewers will scrutinize original-timeline-index vs modified-timeline-index consistency in the "period-6 resonance" claim. Must be clean.

## Priority allocation (one GPU-week on Pro 6000 Blackwell)

Reviewer's staged plan:
1. **Day 0-1: multimask check** — instrument the SAM2 surrogate, log per-frame `multimask_output` status + `ious` tensor shape + selected candidate index on 1 clean clip. This is THE gating question for Q3.
2. **Day 0-2 (parallel)**: Oracle-gap pilot for Q1 on 5 clips — brute-force position search with reduced PGD.
3. **Day 2-4**: If Q3 is active → instrument candidate outputs + soft IoU-aux implementation + 3-clip Base-vs-Base+IoU comparison. If Q3 dormant → abandon Q3, pivot remaining budget to Q1 scheduler design.
4. **Day 4-7**: Remaining week on the one direction showing a real gap.

## Two exact sanity checks (minimum viable)

1. **Multimask check**: on one clean clip, log per tracking frame whether multimask is active, shape of `ious`, selected candidate index.
2. **Q1 oracle-gap**: on 5 pilot clips, compare fixed schedule to small oracle search over insert positions using reduced PGD.

## Next action

User to decide whether to pursue this package. If yes, next step is the multimask instrumentation — cheapest and most informative single experiment. My recommendation is to run that one BEFORE committing to either Q1 or Q3 code work.

## References

- SAM2 source (memory encoding + multimask): https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
- SAM2 mask decoder (IoU head): https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/modeling/sam/mask_decoder.py
- SAM2 paper (Ravi et al. 2024): https://openreview.net/pdf?id=Ha6RTeWMd0
- DeepSAVA Bayesian critical-frame search: https://www.sciencedirect.com/science/article/pii/S0893608023006792
- Gradient-feedback keyframe selection (Sensors 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC9144869/
- Gradient-based keyframe selection (ICASSP 2022): https://xiaolei.tech/papers/ICASSP22.pdf
