# Experiment Plan: MemoryShield

## Setup

### Datasets
| Dataset | # Videos | Avg Length | Purpose |
|---|---|---|---|
| DAVIS 2017 val | 30 | 69 frames | Standard VOS benchmark |
| YouTube-VOS 2019 val | 507 | 130 frames | Scale + diversity |
| MOSE | 431 | 80+ frames | Occlusions + complex scenes |

### Target Models (Black-box)
- SAM2 hiera_tiny (checkpoint: sam2.1_hiera_tiny.pt)
- SAM2 hiera_base_plus
- SAM2 hiera_large
- SAM2.1 (latest)
- SAM2Long (tree-based memory variant)

### Surrogate Models (White-box access)
- SAM-ViT-H (vit_h)
- HQ-SAM
- XMem
- Cutie

### Hardware
- Primary: Tesla V100 (per resource policy)
- 1 GPU for profiling pilot, scale if needed
- Total budget: ~33 GPU-hours

### Metrics
| Category | Metric | Good | Great |
|---|---|---|---|
| Attack | mIoU drop | ≥ 20 | ≥ 30 |
| Attack | J&F degradation | ≥ 15 | ≥ 25 |
| Quality | PSNR | ≥ 36 | ≥ 40 |
| Quality | SSIM | ≥ 0.93 | ≥ 0.97 |
| Quality | LPIPS | ≤ 0.08 | ≤ 0.03 |
| Efficiency | Insertion ratio | ≤ 20% | ≤ 10% |
| Persistence | Frames until recovery | ≥ 30 | ≥ 100 |

---

## Experiment Blocks

### E1: Baseline Reproduction (4 GPU-hrs)
**Goal**: Validate setup + establish baselines

| Run | Method | Dataset | Notes |
|---|---|---|---|
| E1.1 | Clean video (no attack) | DAVIS 2017 | Upper bound for SAM2 |
| E1.2 | DarkSAM applied per-frame | DAVIS 2017 | Image-level UAP baseline |
| E1.3 | Vanish into Thin Air | DAVIS 2017 | SOTA SAM2 attack baseline |
| E1.4 | Naive frame insertion (random noise) | DAVIS 2017 | Insertion baseline |
| E1.5 | Naive frame insertion (duplicate frame + PGD) | DAVIS 2017 | Better insertion baseline |

**Success criteria**: Reproduce DarkSAM/Vanish numbers within 5% of reported

### E2: Memory Resonance Core Validation (3 GPU-hrs)
**Goal**: Prove FIFO-synchronized scheduling outperforms alternatives

| Run | Scheduling | ε | Insertion % | Expected |
|---|---|---|---|---|
| E2.1 | Random timing | 8/255 | 15% | Weak baseline |
| E2.2 | Periodic (every K frames) | 8/255 | 15% | Medium |
| E2.3 | **FIFO Resonance (N-1 sync)** | 8/255 | 15% | Best |
| E2.4 | FIFO Resonance, N estimation off by ±2 | 8/255 | 15% | Robustness check |
| E2.5 | FIFO Resonance, reduced budget | 8/255 | 8% | Efficiency check |

**Claim supported**: C1 (insertion viable) + C2 (FIFO-sync > naive)

### E3: Module Ablation (6 GPU-hrs)
**Goal**: Show each module contributes and combination is synergistic

| Run | Modules Active | Dataset |
|---|---|---|
| E3.1 | Resonance only | DAVIS + MOSE |
| E3.2 | Resonance + Topology Seed | DAVIS + MOSE |
| E3.3 | Resonance + Conditioning Shadow | DAVIS + MOSE |
| E3.4 | Resonance + Occlusion Ghost | MOSE (rich in occlusions) |
| E3.5 | **Full MemoryShield** (all modules) | DAVIS + MOSE |
| E3.6 | Topology Seed alone (no Resonance) | DAVIS |
| E3.7 | Persistence test: measure frames until SAM2 recovers | DAVIS |

**Claim supported**: C3 (topology self-reinforcing) + C4 (synergistic combination)

### E4: Frequency Variant Comparison (4 GPU-hrs)
**Goal**: Determine best perturbation generation method

| Run | Perturbation Method | ε | Quality |
|---|---|---|---|
| E4.1 | PGD (spatial domain, standard) | 8/255 | Baseline |
| E4.2 | Phase-Locked Echo (phase-only) | 8/255 | Expected best quality |
| E4.3 | Spectral Hole Punch (subtractive notch) | 8/255 | Expected unique |
| E4.4 | DIM + MI-FGSM (transfer-enhanced PGD) | 8/255 | Transfer baseline |
| E4.5 | SSA (spectrum simulation) | 8/255 | Frequency transfer |

**Can run in parallel with E3**

### E5: Surrogate Transferability (4 GPU-hrs)
**Goal**: Validate black-box attack effectiveness

| Run | Surrogate | Target | Expected |
|---|---|---|---|
| E5.1 | SAM-ViT-H only | SAM2-tiny | Transfer baseline |
| E5.2 | SAM-ViT-H only | SAM2-large | Harder transfer |
| E5.3 | Full ensemble (SAM+HQ-SAM+XMem) | SAM2-tiny | Better transfer |
| E5.4 | Full ensemble | SAM2-large | Key result |
| E5.5 | Full ensemble | SAM2.1 | Generalization |
| E5.6 | Full ensemble | SAM2Long (tree memory) | Robustness to arch change |

**Claim supported**: C5 (black-box transfer effective)

### E6: Visual Quality Sweep (3 GPU-hrs)
**Goal**: Map the attack-quality Pareto frontier

| Run | ε | Insertion % | Key Metric |
|---|---|---|---|
| E6.1 | 2/255 | 10% | Quality upper bound |
| E6.2 | 4/255 | 10% | |
| E6.3 | 8/255 | 10% | Sweet spot candidate |
| E6.4 | 12/255 | 10% | |
| E6.5 | 16/255 | 10% | Attack upper bound |
| E6.6 | 8/255 | 5% | Minimal insertion |
| E6.7 | 8/255 | 15% | |
| E6.8 | 8/255 | 25% | Dense insertion |

**Claim supported**: C6 (visually imperceptible at operating point)

### E7: Codec Robustness (2 GPU-hrs)
**Goal**: Verify attack survives video compression

| Run | Codec | CRF | Expected |
|---|---|---|---|
| E7.1 | None (raw) | — | Upper bound |
| E7.2 | H.264 | 18 | High quality |
| E7.3 | H.264 | 23 | Standard quality |
| E7.4 | H.264 | 28 | Low quality |
| E7.5 | H.265 | 23 | Modern codec |
| E7.6 | VP9 | 31 | Web codec |

**Claim supported**: C7 (codec-robust)

### E8: Cross-Model Generalization (4 GPU-hrs)
**Goal**: Show attack works beyond SAM2 family

| Run | Target Model | Type |
|---|---|---|
| E8.1 | XMem | Memory-based VOS |
| E8.2 | Cutie | Attention-based VOS |
| E8.3 | DEVA | Decoupled VOS |
| E8.4 | DeAOT | Associative VOS |

### E9: Defense Resistance (3 GPU-hrs)
**Goal**: Test against known defenses

| Run | Defense | Method |
|---|---|---|
| E9.1 | No defense | Baseline |
| E9.2 | JPEG compression purification | Per-frame JPEG at quality 75 |
| E9.3 | Correction-based defense (Song et al., USENIX 2024) | Temporal smoothing |
| E9.4 | DiffPure (diffusion purification) | Denoise-then-segment |
| E9.5 | Frame-dropping detection | Detect + remove inserted frames |

---

## Run Order & Dependencies

```
E1 (baselines) ──→ E2 (core validation) ──→ E3 (ablation) ──→ E5 (transfer)
                                         ├─→ E4 (frequency)     ├─→ E6 (quality)
                                         │                      ├─→ E7 (codec)
                                         │                      ├─→ E8 (cross-model)
                                         │                      └─→ E9 (defense)
                                         └─→ (merge best frequency variant into E3+)
```

## First 3 Runs to Launch

1. **E1.1-E1.5**: Baseline reproduction (validate setup)
2. **E2.1-E2.5**: Memory Resonance core (prove main thesis)
3. **E4.1-E4.5**: Frequency variant comparison (determine best generator)

## Go/No-Go Criteria

After E2:
- **GO** if FIFO Resonance achieves mIoU drop ≥ 15 on DAVIS with ≤ 15% insertion
- **PIVOT** if mIoU drop < 10 → investigate whether the surrogate ensemble is too weak
- **KILL** if insertion attack shows no measurable degradation → fundamental approach may be flawed
