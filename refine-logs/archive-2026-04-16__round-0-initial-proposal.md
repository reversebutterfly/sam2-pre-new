# Experiment Design: MemoryShield Fair Evaluation

## Problem Anchor

- **Bottom-line problem**: Design a rigorous experiment suite that proves whether adversarial frame insertion can independently corrupt SAM2's memory-based tracking, separate from direct per-frame perturbation effects.
- **Must-solve bottleneck**: The current 20-video ablation conflates two mechanisms — (1) direct perturbation of scored frames and (2) memory propagation from poisoned frames to future clean frames. We must disentangle them.
- **Non-goals**: Not optimizing for maximum J_drop numbers; not building a real deployment system yet. The goal is a clean mechanism study.
- **Constraints**: V100 32GB × 1 GPU; 15-frame clips (OOM at 30 with full graph); SAM2 hiera_tiny checkpoint; DAVIS 2017.
- **Success condition**: A table where (a) no scored frame is directly attacked, (b) insert-only shows meaningful J&F drop on untouched future frames, and (c) the contributions of conditioning-frame corruption, memory poisoning, and direct perturbation are clearly separated.

## Current Findings

### What we know:
1. **Hybrid (insert + perturb)**: J_drop = 0.87 on eligible videos (J_clean ≥ 0.6)
2. **Perturb-only**: J_drop = 0.86 — almost as strong as hybrid
3. **Insert-only**: J_drop = 0.25 — much weaker
4. **Confound**: 5 of 13 scored frames are also attacked in perturb-only
5. **Frame 0 (conditioning frame)** gets ε=2/255 perturbation — potentially the dominant mechanism
6. **Only J is measured**, not F or J&F
7. **JPEG Q100** evaluation introduces unknown codec effects
8. **Insert-only uses weak ε=2/255** for the second frame

### What we don't know:
- How much of the J_drop comes from frame-0 conditioning corruption vs memory propagation?
- Does insert-only work better with more frames, higher ε, or better interpolation?
- Does the attack persist beyond the immediate attack window?
- Does it generalize across SAM2 model sizes?

## Experiment Design

### Principle: Disjoint Attack-Evaluate Windows

```
Video: [f0, f1, f2, f3 | f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]
        ↑ ATTACK WINDOW  ↑ EVALUATION WINDOW (untouched, J&F measured here only)
        (frames 0-3)       (frames 4-14)
```

No frame in the evaluation window is directly perturbed. All J&F measurements are on clean, untouched original frames. Any degradation must come through SAM2's memory mechanism.

### Metrics

For EVERY condition, report:
- **J** (region IoU) — per-frame and mean
- **F** (boundary accuracy) — per-frame and mean  
- **J&F** = 0.5*(J+F) — the DAVIS standard metric
- **ΔJ** = J_clean - J_protected
- **ΔF** = F_clean - F_protected
- **Δ(J&F)** = J&F_clean - J&F_protected
- **SSIM** of attacked frames (visual quality)

Report on:
- All 20 DAVIS videos
- Eligible subset (J&F_clean ≥ 0.6)
- Per-video breakdown

### Experiment Table A: Mechanism Decomposition

**Purpose**: Isolate where the attack power comes from.

| ID | Condition | Attack Window (f0-f3) | Eval Window (f4-f14) | What it tests |
|---|---|---|---|---|
| A0 | clean | nothing | measure J&F | upper bound |
| A1 | frame0-only | perturb f0 (ε=2/255) | measure J&F | conditioning frame hijack |
| A2 | early-orig-2 | perturb f1,f2 (ε=4/255) | measure J&F | early memory poisoning (no conditioning) |
| A3 | early-orig-3 | perturb f1,f2,f3 (ε=4/255) | measure J&F | more early memory poisoning |
| A4 | frame0+early | perturb f0 (2/255) + f1,f2 (4/255) | measure J&F | conditioning + memory combined |
| A5 | insert-2-strong | insert 2 frames after f1,f3 (both ε=8/255) | measure J&F | pure insertion effect |
| A6 | insert-4 | insert 4 frames after f0,f1,f2,f3 (all ε=8/255) | measure J&F | denser insertion |
| A7 | hybrid-early | perturb f0,f1,f2 + insert 2 after f1,f3 | measure J&F | full mechanism in attack window |

**Key comparisons**:
- A1 vs A0: conditioning frame contribution
- A2 vs A0: pure memory poisoning (no conditioning attack)
- A5 vs A2: insertion vs perturbation at matched frame count (2 controls each)
- A6 vs A3: insertion vs perturbation at matched frame count (4 vs 3)
- A7 vs A4 vs A5: does hybrid beat the sum of parts?

### Experiment Table B: Schedule Comparison

**Purpose**: Test whether FIFO-resonant positioning matters.

On 15-frame clips, attack only in f0-f3, evaluate on f4-f14:

| ID | Condition | Insertion Positions |
|---|---|---|
| B1 | resonant | After f1, f7 (FIFO-aligned) — but this violates attack-window rule |
| B1' | resonant-early | After f1, f3 (within attack window) |
| B2 | random | After 2 random positions in f0-f3 |
| B3 | even | After f1, f2 (evenly spaced within window) |

**Note**: True resonance testing requires longer videos (30-50 frames) where inserts at f1 and f7 can be compared to random placement, with evaluation on f8-30+. This is a Phase 2 priority.

### Experiment Table C: Insert-Only Optimization

**Purpose**: Find the ceiling of insert-only attack.

| ID | Condition | # Inserts | ε per insert | Base frame |
|---|---|---|---|---|
| C1 | insert-2-weak | 2 | 8/255 + 2/255 | blend |
| C2 | insert-2-strong | 2 | 8/255 + 8/255 | blend |
| C3 | insert-4-strong | 4 | all 8/255 | blend |
| C4 | insert-6-strong | 6 | all 8/255 | blend |
| C5 | insert-2-flow | 2 | 8/255 + 8/255 | optical flow warp |

**Key question**: At what insertion density does insert-only become competitive with perturb-only?

### Experiment Table D: Cross-Model Transfer

**Purpose**: Does the attack generalize?

| ID | Craft on | Evaluate on |
|---|---|---|
| D1 | hiera_tiny | hiera_tiny (white-box) |
| D2 | hiera_tiny | hiera_base+ |
| D3 | hiera_tiny | hiera_large |
| D4 | hiera_tiny | SAM2.1 (if available) |

Use the best hybrid condition from Table A.

### Experiment Table E: Codec Robustness

**Purpose**: Does the attack survive compression?

| ID | Eval format | Details |
|---|---|---|
| E1 | PNG | lossless (true transfer test) |
| E2 | JPEG Q100 | near-lossless |
| E3 | JPEG Q95 | standard high quality |
| E4 | JPEG Q75 | standard web quality |
| E5 | H.264 CRF23 | video codec |

### Experiment Table F: Persistence (requires longer clips)

**Purpose**: Does memory corruption persist over time?

On 5 videos at 30-50 frames (use gradient checkpointing or windowed optimization):

| ID | Clip length | Attack window | Eval window |
|---|---|---|---|
| F1 | 30 frames | f0-f5 | f6-f29 |
| F2 | 50 frames | f0-f5 | f6-f49 |

Report per-frame J&F curve showing degradation over time.

## Priority Order

1. **Table A** (mechanism decomposition) — most critical for paper narrative
2. **Table C** (insert-only ceiling) — determines if insertion is viable
3. **Table E** (codec) — practical relevance
4. **Table D** (cross-model) — generalizability
5. **Table F** (persistence) — memory story
6. **Table B** (schedule) — resonance claim

## Implementation Changes Needed

1. **Add F metric**: Use `src/metrics.py:f_measure()` already available
2. **Add disjoint eval**: New parameter `--eval_start_frame` to exclude attack window from scoring
3. **Add PNG eval option**: Write PNG instead of JPEG in evaluation
4. **Add insert-count sweep**: Parameterize number of insertions
5. **Add per-frame J&F logging**: Return full curve, not just mean
6. **Fix quality regularizer parity**: All conditions should use the same quality constraint

## Compute Estimate

| Table | # Conditions | # Videos | PGD steps | Est. GPU-hrs |
|---|---|---|---|---|
| A (mechanism) | 8 | 20 | 50 | ~24h |
| C (insert ceil) | 5 | 10 | 50 | ~8h |
| E (codec) | 5 | 10 | 0 (reuse A7) | ~1h |
| D (cross-model) | 4 | 10 | 0 (reuse A7) | ~2h |
| F (persistence) | 2 | 5 | 50 | ~8h |
| B (schedule) | 3 | 10 | 50 | ~5h |
| **Total** | | | | **~48h** |

## Expected Outcomes and Claims Matrix

| If A5 (insert-2) >> A0 by ≥10% Δ(J&F) | Claim: Frame insertion alone corrupts SAM2 memory |
|---|---|
| If A1 (frame0-only) >> A0 by ≥30% | Claim: Conditioning frame is SAM2's Achilles heel |
| If A7 > A4 + A5 | Claim: Hybrid is synergistic, not just additive |
| If C4 >> C2 | Claim: Insertion density matters; more = stronger |
| If D2/D3 retains ≥30% of D1 | Claim: Attack transfers across model sizes |
| If E3 retains ≥50% of E1 | Claim: Attack survives practical compression |
