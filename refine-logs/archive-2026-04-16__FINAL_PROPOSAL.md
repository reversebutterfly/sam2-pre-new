# MemoryShield Experiment Design — Final Version

**Status**: READY (GPT-5.4 score 8.5/10, verdict READY)
**Date**: 2026-04-16

## Core Question

Does adversarial frame insertion independently corrupt SAM2's memory-based tracking, and is the effect memory-mediated?

## Design Principle: Disjoint Attack-Evaluate Windows

```
Video: [f0, f1, f2, f3 | f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]
        ↑ ATTACK WINDOW  ↑ EVALUATION WINDOW (untouched, J&F measured here only)
```

No scored frame is directly attacked. Any degradation must propagate through SAM2's memory.

## Table A: Mechanism Decomposition (CORE TABLE)

11 conditions × 20 DAVIS videos, 15 frames, 50 PGD steps

| ID | Condition | Attack (f0-f3) | ε | Mem reset f4? | Purpose |
|---|---|---|---|---|---|
| A0 | clean | — | 0 | No | upper bound |
| A0r | clean+reset | — | 0 | Yes | normalize reset effect |
| A1 | frame0-only | perturb f0 | 2/255 | No | conditioning hijack |
| A2 | orig-2 | perturb f1,f2 | 4/255 | No | memory poisoning |
| A3 | orig-2-strong | perturb f1,f2 | 8/255 | No | budget-matched with A5 |
| A4 | frame0+orig-2 | perturb f0+f1,f2 | 2+4/255 | No | conditioning+memory |
| A5 | insert-2-adv | insert 2 adv after f1,f3 | 8/255 | No | adversarial insertion |
| A5b | insert-2-benign | insert 2 clean after f1,f3 | 0 | No | benign control |
| A5r | insert-2-adv+reset | insert 2 adv after f1,f3 | 8/255 | Yes | **CAUSAL TEST** |
| A7 | hybrid | perturb f0,f1,f2 + insert 2 | mixed | No | full method |
| A8 | hybrid+reset | same as A7 | mixed | Yes | full method causal |

### Decisive Evidence Pattern

```
Claim 1: "Adversarial insertion degrades tracking"
  Evidence: A5 >> A0 (meaningful Δ(J&F))

Claim 2: "The effect is adversarial, not just extra frames"
  Evidence: A5 >> A5b

Claim 3: "The effect is memory-mediated"
  Evidence: A5r ≈ A0r (memory reset recovers tracking)

Claim 4: "Conditioning frame is critical"
  Evidence: A1 >> A0

Claim 5: "Hybrid is stronger than either alone"
  Evidence: A7 > max(A4, A5)
```

## Table P: Persistence (SECONDARY)

5 videos × 30 frames, attack f0-f5, evaluate per-frame f6-f29

Output: per-frame J&F curve showing temporal persistence of degradation.

## Table D: Cross-Model Transfer (SECONDARY)

10 videos, craft on hiera_tiny, evaluate on tiny/base+/large

## Metrics

For every condition:
- J (region IoU), F (boundary accuracy), J&F = 0.5*(J+F)
- ΔJ, ΔF, Δ(J&F) relative to clean
- SSIM on attacked frames
- Per-frame curves for persistence
- Paired per-video stats with confidence intervals

## Compute: ~37 GPU-hours on V100-32GB

## Claims Matrix

| Result pattern | Allowed claim |
|---|---|
| A5 >> A5b AND A5r ≈ A0r | **Adversarial insertion corrupts SAM2 via memory poisoning** |
| A5 ≈ A5b | Insertion mechanism is not adversarial → reframe |
| A5r ≈ A5 (reset doesn't help) | Effect is NOT memory-mediated → fundamental problem |
| A1 >> A0 (≥20%) | Conditioning frame is SAM2's critical vulnerability |
| A3 ≈ A5 | Perturbation and insertion equally effective at matched budget |
| A7 > A4 + A5 (super-additive) | Hybrid is synergistic |
