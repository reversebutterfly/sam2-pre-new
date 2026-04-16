# Idea Discovery Report

**Direction**: 视频数据集保护 — 通过架构感知的对抗帧插入策略防御SAM2分割攻击
**Date**: 2026-04-15
**Pipeline**: research-lit → idea-creator (GPT-5.4 xhigh) → novelty-check → research-review → method refinement

---

## Executive Summary

We propose **MemoryShield**, a unified framework for protecting video datasets against SAM2's video object segmentation through **architecture-aware adversarial frame insertion**. The core insight is that SAM2's FIFO memory bank has no quality gating — adversarial frames enter memory unconditionally and corrupt all downstream predictions. Our approach inserts a small number of carefully timed adversarial frames (strong anchor + weak sustain) that exploit three SAM2-specific vulnerabilities: FIFO memory dynamics, conditioning frame priority, and occlusion-transition fragility. All 7 generated ideas passed deep novelty verification against 30+ papers.

**Best idea**: Memory Resonance (记忆共振) — synchronize adversarial frame insertion with FIFO eviction schedule to create persistent memory corruption with minimal insertion budget.

**Recommended next step**: Implement the unified MemoryShield framework and run experiment block E2 (core validation) on V100 GPU.

---

## Literature Landscape

### Surveyed Papers: 30+ across 4 dimensions

#### Dimension 1: Adversarial Attacks on SAM/SAM2 (Image-level)
| Paper | Venue | Key Method |
|---|---|---|
| Attack-SAM (Zhang et al.) | arXiv 2023 | White-box FGSM on SAM encoder |
| Black-box Targeted Attack on SAM (Zheng et al.) | arXiv 2023 | Targeted mask attack, cross-model transfer |
| Red-Teaming SAM (Jankowski et al.) | CVPR 2024 WS | FIGA sparse attack |
| SAM Meets UAP (Han et al.) | arXiv 2024 | Contrastive UAP generation |
| DarkSAM (Chen et al.) | NeurIPS 2024 | Prompt-free UAP, semantic decoupling + frequency |
| Robust SAM | AAAI 2025 | Cross-prompt attack, -40 mIoU on SAM2 |

#### Dimension 2: Temporal/Video Adversarial Attacks
| Paper | Venue | Key Method |
|---|---|---|
| VOS Hard Region Discovery (Li et al.) | IEEE TCSVT 2024 | First-frame attack propagates through memory |
| DeepSAVA (Mu et al.) | Neural Networks 2023 | Single-frame perturbation, 99.5% fooling |
| Flickering Attack (Pony et al.) | CVPR 2021 | Temporal brightness modulation |
| Vanish into Thin Air | 2025 | Cross-prompt universal SAM2 attacks |
| Time-Constrained Attacks | Machine Vision 2025 | Sparse temporal mask, K frames only |

#### Dimension 3: Data Protection Paradigm
| Paper | Venue | Key Method |
|---|---|---|
| Glaze (Shan et al.) | USENIX 2023 | Style protection in CLIP latent space |
| PhotoGuard (Salman et al.) | ICML 2023 | Encoder-level perturbation vs. image editing |
| I2VGuard (Gui et al.) | CVPR 2025 | First video protection against I2V diffusion |
| Anti-I2V | arXiv 2026 | Improved I2V protection |
| PAP | NeurIPS 2024 | Prompt-agnostic perturbation |

#### Dimension 4: SAM2 Architecture Vulnerabilities
- **FIFO memory bank** with NO quality gating → unconditional memory poisoning
- **Memory encoder** fuses masks + frame embeddings → corrupted mask cascades
- **No motion model** → purely appearance-based, fragile to adversarial features
- **Error accumulation** → single bad prediction propagates indefinitely
- **Fixed temporal window** → burst of bad frames flushes good memories

### Structural Gap
**No existing work addresses protecting video datasets from SAM2 via frame insertion.** All prior attacks perturb existing frames; none exploit the insertion attack surface. No prior work uses FIFO-synchronized scheduling, topology-targeted attack objectives, or event-triggered insertion timing.

---

## Ranked Ideas (7 validated)

### 🏆 Idea 1: Memory Resonance 记忆共振 — RECOMMENDED

**Core**: Treat SAM2's FIFO memory bank as a resonant cavity. One strong frame maximally shifts memory embeddings; weak frames arrive precisely before the poison ages out, creating a standing wave of corruption.

| Aspect | Design |
|---|---|
| Temporal Pattern | Strong at t₀; weak at t₀+(N-1), t₀+2(N-1)... synced with FIFO window N |
| Strong Frame | Surrogate-ensemble optimized memory-embedding drift on target contours |
| Weak Frame | Same perturbation basis at 15-25% energy |
| Black-box | SAM/SAM2.1/HQ-SAM ensemble + random prompt EOT + codec augmentation |
| Visual Quality | Perturbation confined to object contours, global PSNR >40dB |
| Novelty | **CONFIRMED** — No prior work syncs adversarial scheduling with FIFO window |
| Closest Work | Vanish into Thin Air (uniform perturbation, no FIFO awareness) |
| Reviewer Risk | "What if SAM2 changes FIFO to tree-based?" → Test both variants |
| Feasibility | ★★★★★ |

---

### 🥈 Idea 2: Conditioning Shadow 条件影随 — RECOMMENDED

**Core**: Insert a "shadow" frame immediately after the prompt frame. SAM2's memory attention preferentially attends to temporally closest conditioning frames, so the shadow hijacks the prompt's influence.

| Aspect | Design |
|---|---|
| Temporal Pattern | Strong frame right after prompt; weak every 3-5 frames while object visible |
| Strong Frame | Preserves click-local appearance but shifts downstream mask memory |
| Weak Frame | Local echo around prompt neighborhood and object centroid |
| Black-box | Randomized point/box prompting on surrogates |
| Visual Quality | Nearly identical to prompt frame, looks like natural inter-frame variation |
| Novelty | **CONFIRMED** — SAM2's conditioning frame privilege unexploited |
| Reviewer Risk | "Attacker doesn't know prompt position" → Pre-insert at every K frames |
| Feasibility | ★★★★ |

---

### 🥉 Idea 3: Topology Split Seed 拓扑分裂种子 — RECOMMENDED

**Core**: Don't make SAM2 fail — make it produce topologically wrong masks. Seed a bad split/hole/bridge at narrow structures, then maintain the topological error with minimal perturbation.

| Aspect | Design |
|---|---|
| Temporal Pattern | Strong at articulated poses; weak every 2-4 frames until topology stabilizes |
| Strong Frame | Boundary attack on necks, limbs, handles exploiting multi-mask ambiguity |
| Weak Frame | Tiny signed-distance perturbation at genus-changing pixels only |
| Black-box | Surrogate connected-component / topology proxy losses |
| Visual Quality | Extremely sparse (few pixels), nearly invisible |
| Novelty | **CONFIRMED** — Topology as attack objective is entirely novel |
| Reviewer Risk | "Simple convex objects have no topology to exploit" → Combine with other modules |
| Feasibility | ★★★☆ |

---

### Idea 4: Occlusion Ghost 遮挡幽灵

**Core**: Event-triggered insertion at natural occlusion/reappearance moments. Exploits SAM2's known fragility at temporal transitions.

| Aspect | Design |
|---|---|
| Temporal Pattern | Strong at occlusion onset; weak on next 3-5 visible frames |
| Strong Frame | Suppress object evidence while preserving background continuity |
| Weak Frame | Low-rank contour weakening + background texture borrowing |
| Novelty | **CONFIRMED** — Event-triggered timing is novel |
| Feasibility | ★★★★★ (optical flow detection is mature) |

---

### Idea 5: Phase-Locked Echo 锁相回声

**Core**: Perturb local Fourier phase (not magnitude) at object boundary harmonics. Phase carries shape information; small phase shifts are invisible but distort memory-encoded boundaries.

| Aspect | Design |
|---|---|
| Strong Frame | Phase-only twist at contour harmonics |
| Weak Frame | Same-sign phase echo with exponential decay |
| Novelty | **CONFIRMED** — Local contour-harmonic phase attack is new |
| Risk | Codec robustness needs validation |
| Feasibility | ★★★★ |

---

### Idea 6: Spectral Hole Punch 频谱空洞冲击

**Core**: Estimate the object's most discriminative texture frequency bands, then carve a narrow adversarial notch. A "subtractive" attack — remove spectral support rather than add noise.

| Aspect | Design |
|---|---|
| Strong Frame | Content-adaptive notch on dominant texture PSD |
| Weak Frame | Same notch at shallow depth (narrow-band = high quality) |
| Novelty | **CONFIRMED** — Subtractive spectral attack is a new primitive |
| Risk | Notch may be partially recovered by codecs |
| Feasibility | ★★★★ |

---

### Idea 7: Bottleneck Trap 信息瓶颈陷阱

**Core**: Information-theoretic optimal control — strong frame maximizes KL divergence of memory embeddings under rate budget; weak frames are minimum-energy control inputs fired adaptively when feature distance collapses.

| Aspect | Design |
|---|---|
| Strong Frame | Optimize memory-embedding drift |
| Weak Frame | Nudge along slowest-correcting principal directions, sparse and nearly invisible |
| Novelty | **CONFIRMED** — Information-theoretic control of VOS memory is new |
| Risk | Theory-to-implementation gap, needs accurate surrogate memory encoder |
| Feasibility | ★★★ |

---

## Eliminated Ideas

| Idea | Phase Eliminated | Reason |
|---|---|---|
| Booster Rhythm 加强针节律 | Phase 2 (not selected) | Subsumed by Memory Resonance's more principled scheduling |
| Chameleon Drift 变色龙漂移 | Phase 2 (not selected) | Interesting but less architecturally targeted |
| Nyquist Drift 奈奎斯特漂移 | Phase 2 (not selected) | Motion illusion is creative but hard to validate black-box |

---

## Refined Proposal: MemoryShield Unified Framework

### Problem Anchor
Video datasets face unauthorized processing by video segmentation foundation models (SAM2). Existing protection methods (DarkSAM, Vanish into Thin Air) only perturb existing frames without exploiting frame insertion as an attack surface, and lack architecture-aware temporal scheduling.

### Method Thesis
Architecture-aware adversarial frame insertion using FIFO memory resonance scheduling as the backbone, combined with topology seed / conditioning shadow / occlusion ghost modules, to persistently disable SAM2's video segmentation while maintaining visual quality.

### Framework Architecture

```
Input Video
    │
    ▼
┌──────────────────────────────────────┐
│        Content Analyzer              │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │Scene/Event│ │ Topology │ │Texture│ │
│  │ Detection │ │ Analysis │ │Profile│ │
│  └─────┬────┘ └────┬─────┘ └──┬───┘ │
└────────┼───────────┼──────────┼──────┘
         │           │          │
         ▼           ▼          ▼
┌──────────────────────────────────────┐
│       Insertion Scheduler (M1)       │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │   FIFO   │ │  Event   │ │Adapt.│ │
│  │Resonance │ │ Trigger  │ │ Decay│ │
│  └─────┬────┘ └────┬─────┘ └──┬───┘ │
└────────┼───────────┼──────────┼──────┘
         │           │          │
         ▼           ▼          ▼
┌──────────────────────────────────────┐
│     Frame Generator (M2)             │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ Strong   │ │  Weak    │ │Quality│ │
│  │Frame Gen │ │Frame Gen │ │Control│ │
│  └─────┬────┘ └────┬─────┘ └──┬───┘ │
└────────┼───────────┼──────────┼──────┘
         │           │          │
         ▼           ▼          ▼
    Protected Video (with inserted frames)
```

### Key Design Choices

1. **Surrogate Ensemble**: SAM + SAM2.1-tiny + HQ-SAM + XMem for black-box transfer
2. **EOT Augmentation**: Random prompt types, positions, codec compression, resize
3. **Quality Constraint**: L∞ ≤ 8/255, SSIM ≥ 0.95, LPIPS ≤ 0.05
4. **Insertion Ratio**: Target ≤ 15% additional frames (e.g., 3 inserted per 20 original)
5. **Perturbation Generation**: Phase-Locked Echo as default, Spectral Hole Punch as variant

---

## Experiment Plan

### Setup
- **Datasets**: DAVIS 2017 (val), YouTube-VOS 2019 (val), MOSE (long video with occlusions)
- **Target Models**: SAM2 (hiera_tiny/base+/large), SAM2.1, SAM2Long
- **Surrogate Models**: SAM-ViT-H, HQ-SAM, XMem, Cutie
- **Hardware**: Tesla V100 (per CLAUDE.md resource policy)

### Metrics
| Category | Metric | Target |
|---|---|---|
| Attack Success | mIoU drop | ≥ 30 points |
| Attack Success | J&F degradation | ≥ 25 points |
| Visual Quality | PSNR | ≥ 38 dB |
| Visual Quality | SSIM | ≥ 0.95 |
| Visual Quality | LPIPS | ≤ 0.05 |
| Efficiency | Insertion ratio | ≤ 15% |
| Persistence | Frames until recovery | ≥ 50 frames |

### Experiment Blocks

| Block | Experiment | Purpose | GPU-hrs |
|---|---|---|---|
| E1 | Baseline reproduction | DarkSAM, Vanish into Thin Air, naive frame insert | 4h |
| E2 | Memory Resonance core | FIFO-sync vs. random/periodic insertion | 3h |
| E3 | Module ablation | +Topology, +Shadow, +Ghost incremental | 6h |
| E4 | Frequency variant | Phase-Locked Echo vs. Spectral Hole Punch vs. PGD | 4h |
| E5 | Surrogate transfer | SAM→SAM2, ensemble→SAM2.1 | 4h |
| E6 | Quality sweep | ε: 2-16/255, insertion ratio: 5-30% | 3h |
| E7 | Codec robustness | H.264/H.265 at CRF 18-28 | 2h |
| E8 | Cross-model | XMem, Cutie, DEVA (non-SAM VOS) | 4h |
| E9 | Defense resistance | Correction-based defense, adversarial purification | 3h |

**Total estimated**: ~33 GPU-hours on V100
**Run order**: E1 → E2 → E3 → E4 (parallel) → E5 → E6 → E7 → E8 → E9
**First 3 runs**: E1, E2, E4

### Claims the Experiments Must Support

| Claim | Required Evidence |
|---|---|
| C1: Frame insertion is a viable attack surface | E2: mIoU drop ≥ 20 with insertion only |
| C2: FIFO-sync outperforms naive scheduling | E2: Resonance > random/periodic by ≥ 5 mIoU |
| C3: Topology attack is self-reinforcing | E3: Topology module alone persists ≥ 30 frames |
| C4: Multi-module combination is synergistic | E3: Full > sum of individual modules |
| C5: Black-box transfer is effective | E5: Surrogate→SAM2 drop ≥ 70% of white-box |
| C6: Visually imperceptible | E6: PSNR ≥ 38 at ε=8/255 |
| C7: Codec-robust | E7: Attack survives H.264 CRF ≤ 23 |

---

## Next Steps

- [ ] Implement MemoryShield framework core (Memory Resonance scheduler + surrogate ensemble)
- [ ] Run E1 (baseline reproduction) to validate experimental setup
- [ ] Run E2 (Memory Resonance core) to prove main thesis
- [ ] Use `/run-experiment` to deploy on GPU server
- [ ] Use `/auto-review-loop` after experiments for iterative paper improvement
- [ ] Or invoke `/research-pipeline` for complete end-to-end flow

---

## References

1. Ravi et al. "SAM 2: Segment Anything in Images and Videos." arXiv 2408.00714, 2024.
2. Chen et al. "DarkSAM: Fooling Segment Anything Model to Segment Nothing." NeurIPS 2024.
3. "Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks on SAM2." arXiv 2510.24195, 2025.
4. Gui et al. "I2VGuard: Safeguarding Images Against Misuse in I2V Models." CVPR 2025.
5. Li et al. "Adversarial Attacks on VOS with Hard Region Discovery." IEEE TCSVT 2024.
6. "Robust SAM: On the Adversarial Robustness of Vision Foundation Models." AAAI 2025.
7. Shan et al. "Glaze: Protecting Artists from Style Mimicry." USENIX Security 2023.
8. Salman et al. "Raising the Cost of Malicious AI-Powered Image Editing." ICML 2023.
9. "SAM2Long: Enhancing SAM2 for Long Video Segmentation with a Training-Free Memory Tree." arXiv 2410.16268, 2024.
10. Mu et al. "DeepSAVA: Sparse Adversarial Video Attacks." Neural Networks 2023.
11. Song et al. "Correction-Based Defense Against Adversarial Video Attacks." USENIX Security 2024.
12. Hu et al. "Topology-Preserving Deep Image Segmentation." MICCAI 2019.
