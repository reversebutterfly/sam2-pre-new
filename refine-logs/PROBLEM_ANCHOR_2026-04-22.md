# Problem Anchor — 2026-04-22

**This anchor is immutable across all rounds of this refinement. Any reviewer suggestion that would change what problem is being solved is DRIFT and must be pushed back.**

## Bottom-line problem

Users publish video data (e.g., private content on social platforms, surveillance footage, video-call recordings) that they do NOT want **promptable video-object-segmentation foundation models** like SAM2 to be able to segment and track without consent. We want a **preprocessor** that takes a clean video + a first-frame prior mask of the target object the user wants to protect, and produces a modified video that, when consumed by any downstream SAM2-style tracker, causes the tracker to **lose the protected object and fail to recover it**, while the video remains visually acceptable to human viewers.

## Must-solve bottleneck

The current generation of prompt-driven video segmentation foundation models (SAM2 is the reference) have **streaming memory banks** that continuously refresh tracker state. This makes them **robust to single-frame perturbations** — even if one frame is attacked, the memory smoothly re-acquires the target from surrounding clean frames. Any preprocessor-style data protection must therefore defeat this self-healing property:

1. Drive SAM2 off the target at some time `t` in the prefix
2. Prevent SAM2 from re-acquiring the target from f_{t+1} onwards via memory rollover

The bottleneck: **single-frame adversarial perturbations alone are insufficient** against a memory-based tracker that self-heals. Prior universal-attack work (UAP-SAM2) operates by degrading frame semantics globally; it's not a preprocessor protecting a specific target with minimal visual impact.

## Non-goals

- **NOT** a universal adversarial perturbation (UAP) competing with UAP-SAM2 on generic SOTA attack strength
- **NOT** a backdoor / training-time attack (e.g. BadVSFM)
- **NOT** a runtime hook into SAM2 (inference-time pixel output only; user's deployment cannot modify SAM2)
- **NOT** maximizing attack aggressiveness at the cost of fidelity; this is a protection tool, not an attack paper
- **NOT** a generic video-classification attack like frame-appending (Chen 2021); we target VOS memory mechanisms specifically

## Constraints

- **Threat model**: white-box knowledge of SAM2 architecture (it's open-source, we can compute gradients through it during preprocessor training / per-video PGD); **no runtime interference** with the deployed tracker
- **Input**: clean video + first-frame GT mask of the protected target (user-provided)
- **Output**: modified video pixels (attacked originals + inserted frames if any)
- **Fidelity bar**: modified video must be **visually acceptable to human viewers** — insert LPIPS ≤ 0.15 (target 0.10), attacked-originals LPIPS ≤ 0.03, SSIM ≥ 0.95 on protected frames. Fidelity is a hard constraint, not a trade-off.
- **Compute**: GPU-minutes-per-video budget (user processes own content, not SOTA benchmarking)
- **Dataset**: DAVIS-2017 val (30 clips), prior 10-clip hard subset for pilots
- **Venue**: top ML / privacy-and-security venue (NeurIPS / ICML / ICLR / USENIX Security / CCS / S&P)

## Success condition

After running on DAVIS-10:
1. **Drop**: SAM2 J&F tracking of the protected target drops from clean ≈ 0.90 to attacked ≤ 0.35 on the **eval window AFTER the attacked prefix** (f15+)
2. **No-recovery**: once SAM2 loses the target, it stays lost — i.e., per-frame J drop grows (or at worst stays flat) over time, does not bounce back
3. **Fidelity**: insert LPIPS ≤ 0.15 (goal ≤ 0.10), attacked-originals LPIPS ≤ 0.03
4. **Transfer**: at least partial transfer to SAM2Long (demonstrating the method exploits a general memory-streaming vulnerability, not a SAM2-specific output head quirk)
5. **Mechanism clarity**: a controlled ablation shows that BOTH components (insert frames causing loss + prefix perturbations preventing recovery) are necessary — removing either reduces attack strength by ≥ 40% relative

## User's proposed mechanism (two-phase)

The user's explicit framing — the method must tell a clean two-part story:

**Phase 1 — "Loss induction"**: Inserted synthetic frames drive SAM2 off the target. The inserted frames are written into SAM2's FIFO memory bank. Their purpose is to create a wrong memory entry (object mislocated or absent) that forces SAM2's next-frame cross-attention to produce a wrong mask.

**Phase 2 — "Recovery prevention"**: Perturbations on ORIGINAL frames surrounding the inserts prevent SAM2 from self-healing. When SAM2 reads the next clean frame and tries to re-acquire the target from visual cues, the perturbed frames (a) keep the wrong memory entries alive longer by reinforcing the mislocated signal in FIFO via their own memory writes, and (b) disrupt the cross-attention residual path that would otherwise snap back onto the true object.

Two phases must work TOGETHER; neither alone defeats the memory bank's self-healing property.

## What this anchor locks in

- **Framing**: privacy-preserving preprocessor, not "better attack"
- **Mechanism**: 2-phase (insert → loss; perturb → no recovery); reviewer cannot push toward single-component schemes (UAP / insert-only / perturb-only) without violating the anchor
- **Target**: specific protected object (user-provided mask), not a class; not universal
- **Output**: pixels only (no runtime hook)
- **Measurement**: drop + no-recovery + fidelity TRIAD; any scheme that passes 1 but fails the others is INSUFFICIENT
