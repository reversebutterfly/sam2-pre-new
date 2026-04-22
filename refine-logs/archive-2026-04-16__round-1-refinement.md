# Round 1 Refinement

## Problem Anchor (verbatim)
Design a rigorous experiment suite proving whether adversarial frame insertion can independently corrupt SAM2's memory-based tracking, separate from direct per-frame perturbation.

## Anchor Check
- Original bottleneck: cannot separate memory poisoning from direct perturbation
- Revised design adds benign insertion control + memory reset — still addresses the anchor
- No drift

## Simplicity Check
- Dominant contribution: clean mechanism evidence via disjoint windows + causal controls
- Removed: Table C (insert ceiling) → appendix; Table E (codec) → appendix
- Kept core: Table A (mechanism) + benign/memory controls + persistence

## Changes Made

### 1. Added benign insertion baselines
- Reviewer said: "Without a benign insertion control, you cannot attribute drop to adversarial content"
- Action: Add A5b (benign-insert-2) and A6b (benign-insert-4) with clean interpolated frames, no adversarial perturbation
- Key test: A5 - A5b = adversarial contribution of insertion

### 2. Added causal memory reset
- Reviewer said: "For a memory-poisoning paper, you need at least one causal intervention"
- Action: Add A8 (hybrid + memory-reset-at-f4): run full hybrid attack, but reset SAM2's memory bank at frame 4 before evaluation
- Key test: if A8 ≈ A0 (clean), the attack is purely memory-mediated

### 3. Budget-matched comparisons
- Reviewer said: "A1 uses 2/255 on 1 frame, A5 uses 8/255 on 2 frames — not fair"
- Action: Add a budget-matched row: orig-2 at ε=8/255 vs insert-2 at ε=8/255 (same total distortion envelope)

### 4. Cut peripheral tables
- Reviewer said: "Too much is peripheral to the core narrative"
- Action: Tables C, E → appendix. Core = revised Table A + persistence

## Revised Experiment Design

### Table A: Mechanism Decomposition (CORE)

All conditions: attack window f0-f3, evaluate ONLY f4-f14 (J, F, J&F)

| ID | Condition | What happens in f0-f3 | ε | Tests |
|---|---|---|---|---|
| A0 | clean | nothing | 0 | upper bound |
| A1 | frame0-only | perturb f0 | 2/255 | conditioning hijack |
| A2 | orig-2 | perturb f1,f2 | 4/255 | memory poisoning (no cond) |
| A3 | orig-2-strong | perturb f1,f2 | 8/255 | budget-matched with A5 |
| A4 | frame0+orig-2 | perturb f0+f1,f2 | 2+4/255 | conditioning + memory |
| A5 | insert-2-adv | insert 2 adv frames after f1,f3 | 8/255 | adversarial insertion |
| A5b | insert-2-benign | insert 2 CLEAN interpolated frames after f1,f3 | 0 | benign insertion control |
| A6 | insert-4-adv | insert 4 adv frames after f0,f1,f2,f3 | 8/255 | denser adversarial insertion |
| A6b | insert-4-benign | insert 4 CLEAN frames | 0 | denser benign control |
| A7 | hybrid | perturb f0,f1,f2 + insert 2 adv | mixed | full method |
| A8 | hybrid+mem-reset | same as A7 but reset memory at f4 | mixed | causal memory test |

**Key Comparisons (the paper's main evidence)**:

| Comparison | What it proves |
|---|---|
| A5 vs A5b | Adversarial content in insertions matters (not just extra frames) |
| A5 vs A3 | Insert-only vs perturb-only at matched budget |
| A8 vs A7 | Attack is memory-mediated (reset kills it) |
| A1 vs A0 | Conditioning frame is vulnerable |
| A7 vs A4+A5 | Hybrid is synergistic |

### Table P: Persistence (CORE)

5-10 videos, attack in f0-f5, evaluate per-frame on f6-f29 or f6-f49

| ID | Length | What |
|---|---|---|
| P1 | 30 frames | Per-frame J&F curve: hybrid vs clean |
| P2 | 50 frames | Longer persistence test |

Key output: plot showing J&F over time, with degradation persisting (or recovering).

### Table D: Cross-Model Transfer (SECONDARY)

10 videos, craft on tiny, eval on tiny/base+/large

### Tables C, E: Insert Ceiling + Codec (APPENDIX)

Moved to supplementary material.

## Protocol Details

1. **Frame alignment**: Score only original frames. After insertion, use mod_to_orig mapping to identify which output frames correspond to originals.
2. **Prompt**: Interior point of GT mask on frame 0 (same for all conditions).
3. **Object selection**: First object (id=1) per video, or all objects merged (anno > 0).
4. **SAM2 version**: sam2.1_hiera_tiny.pt via build_sam2_video_predictor.
5. **PGD**: 50 steps, sign-based, fake uint8 quant, margin-based multi-term loss.
6. **Loss**: Backprop through attack window; loss on eval window via track_step graph.
7. **Memory reset** for A8: after PGD optimization, manually clear memory bank state at frame 4 during official evaluation.

## Compute Estimate (revised)

| Table | Conditions | Videos | GPU-hrs |
|---|---|---|---|
| A (core) | 11 | 20 | ~30h |
| P (persistence) | 2 | 5 | ~5h |
| D (transfer) | 3 | 10 | ~2h |
| **Total core** | | | **~37h** |

## Claims Matrix

| If A5 >> A5b (Δ(J&F) ≥ 5%) | **Adversarial insertion corrupts memory beyond benign insertion** |
|---|---|
| If A5 ≈ A3 | Insert-only ≈ perturb-only at matched budget → both mechanisms valid |
| If A5 << A3 | Insertion less efficient than perturbation → reframe as supplementary |
| If A8 ≈ A0 | **Attack is memory-mediated** (memory reset recovers tracking) |
| If A8 ≈ A7 | Attack is NOT memory-mediated → fundamental problem |
| If A1 >> A0 (≥ 20%) | **Conditioning frame is SAM2's critical vulnerability** |
| If P1 shows persistent ≥10 frames | **Memory corruption has lasting temporal effect** |
