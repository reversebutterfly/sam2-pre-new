# Manual Phase Experiment Plan

> Source: GPT-5.4 xhigh review via Codex MCP (thread `019dae7c-be42-78b3-b045-7a9ee7b3ec39`).
> Generated: 2026-04-21 after `/auto-review-loop` closed at 6.0/10 "Almost".
> Goal: execute top-down to push from 6.0 → 7.5+ for a top-venue standalone decoy paper.

---

## TL;DR — Execution Order

1. Finish **10-clip CVaR-only baseline** (first priority, resolves the "3-clip subset" weakness).
2. Measure **fidelity + targeted metrics on CVaR-10 outputs** (piggyback, ~1 GPU-h).
3. Re-eval SAM2Long on 10-clip CVaR attacks (~1-2 GPU-h). [**Note**: our `sam2long_eval.py` already uses full-window `f10:end`, so the reviewer's warning about eval window is moot — existing 3-clip retention 0.40 is valid.]
4. **Module 4 perceptual pilot** on 3 hard clips (insert fidelity fix).
5. **Module 6 transfer hardening pilot** on 3 hard clips (absolute SAM2Long lift).
6. If both pilots pass, scale the winner to 10 clips.

---

## Platform Rule

- Shared V100 cluster is the main path. OOMs are expected.
- **Rule**: if **2 consecutive OOMs** on V100, switch to Pro 6000 setup (conda install + repo sync + data copy, ~4-6h overhead, worth it on a 40-80 GPU-h week).

---

## Prioritized Experiment List

### E1. Finish 10-clip CVaR-only baseline [MUST]

- **Config**: `decoy` regime, `q=0.5` soft CVaR, full video length, `n_steps=50`, `seed=42`, save videos, `--no_teacher`.
- **Command** (7 remaining clips, re-launch after prior OOM):
  ```bash
  python run_two_regimes.py --regime decoy \
    --videos blackswan,breakdance,car-shadow,bike-packing,camel,car-roundabout,dance-twirl \
    --device cuda:0 --n_steps 50 --seed 42 --no_teacher --save_videos \
    --output_dir results_cvar7
  ```
- **Runtime**: 4-7 GPU-h for the 7 remaining; 6-10 GPU-h if all 10 from scratch.
- **Claim**: "CVaR fixes Stage-2 regression and matches or improves v4 on 10 representative DAVIS clips."
- **Success**: mean SAM2 drop ≥ 0.60, ≥ v4's 0.547, no catastrophic collapse on cows/dog, ≥ 8/10 clips non-trivially positive.
- **Failure**: mean < 0.55, or ≥ 2 strong regressions vs v4.

### E2. Re-eval SAM2Long on CVaR-10 (full window) [MUST]

- **Config**: our existing `scripts/sam2long_eval.py` uses `range(EVAL_START=10, len(gt_masks))` — already full-window. No patch needed.
- **Command**:
  ```bash
  python scripts/sam2long_eval.py \
    --attacks_dir results_cvar10 \
    --output_dir results_cvar10/sam2long \
    --videos cows,dog,bmx-trees,blackswan,breakdance,car-shadow,bike-packing,camel,car-roundabout,dance-twirl \
    --regimes decoy --device cuda:0 \
    --sam2_baselines results_cvar10/regimes_results.json
  ```
- **Runtime**: 0.5-1.5 GPU-h (small memory, can share GPU).
- **Claim**: "CVaR decoy is or is not persistent beyond short window on SAM2Long."
- **Success**: full-window mean SAM2Long drop ≥ 0.20; dog ≥ 0.20; bmx-trees ≥ 0.25.
- **Failure**: mean < 0.15, or only bmx-trees survives.

### E3. Fidelity on CVaR-10 outputs [MUST]

- **Config**: run our `measure_fidelity.py` on the actual CVaR outputs, not the Stage-2 proxy.
- **Command**:
  ```bash
  python scripts/measure_fidelity.py \
    --attacks_dir results_cvar10 --davis_root data/davis \
    --videos cows,dog,bmx-trees,blackswan,breakdance,car-shadow,bike-packing,camel,car-roundabout,dance-twirl \
    --regime decoy --device cuda:0 \
    --output_json results_cvar10/fidelity.json
  ```
- **Runtime**: 0.2-0.5 GPU-h (LPIPS inference only).
- **Claim**: "Attacked originals already pass publication-grade; fidelity problem localized to inserts."
- **Success**: originals LPIPS ≤ 0.02 (match current 0.016), inserts honestly quantified.
- **Failure**: originals degrade unexpectedly, or insert LPIPS much worse than Stage-2's 0.16.

### E4. Targeted mislocalization metrics [MUST, free]

- **Config**: reuse `pos_score_rate`, `collapse_rate`, `decoy_hit_rate`, `centroid_shift` from `regimes_results.json` signatures. Optional small patch: add `wrong_region_occupancy = |pred ∩ decoy_gt| / |pred|` and `true_occupancy = |pred ∩ gt| / |pred|`.
- **Runtime**: ~0 GPU-h (parse existing JSON), ≤ 0.2 GPU-h if re-score.
- **Claim**: "This is targeted mislocalization with object-present, not suppression-by-proxy."
- **Success**: `pos_score_rate ≥ 0.8`, `collapse_rate ≤ 0.2`, `decoy_hit_rate ≥ 0.6`, `centroid_shift ≥ 0.35`; optional wrong-region > true occupancy on most eval frames.
- **Failure**: low positive-objectness or high collapse — decoy reads as suppression.

### E5. Module 4 pilot: perceptual insert loss [SHOULD]

- **Config**: 3 hard clips (cows, dog, bmx-trees); keep CVaR, no teacher, `n_steps=50`; add insert-only perceptual losses.
- **Initial weights**: `w_LPIPS=1.0`, `w_DeltaE_seam=0.5`, `w_outside_identity=2.0`, legacy SSIM still active, budgets unchanged.
- **Needs implementation**: LPIPS(alex) on device + edit-mask extraction from Poisson base + Lab DeltaE + seam band TV. Estimated 3-4h coding work.
- **Runtime**: 2-4 GPU-h after code lands.
- **Claim**: "Insert visibility can be fixed without destroying the targeted attack."
- **Success**: mean insert LPIPS ≤ 0.10 on 3 clips (ideally ≤ 0.08); SAM2 drop within 0.05 of CVaR-only; targeted metrics stable.
- **Failure**: insert LPIPS barely moves, or SAM2 drop loses > 0.08.

### E6. Module 6 pilot: transfer hardening [SHOULD]

- **Config**: 3 hard clips; CVaR, no teacher, `n_steps=75` (longer because EOT inflates steps); final joint stage only → `MI-FGSM + TI-FGSM + DI²-FGSM + admix + mild JPEG/blur EOT`.
- **Initial settings**: momentum 1.0, TI Gaussian 5×5 σ=1.0, DI resize 0.9-1.1, admix α=0.2 on 2 nearby frames, JPEG Q95-100 with p=0.3, blur σ=0.5 with p=0.2.
- **Needs implementation**: momentum accumulator + Gaussian gradient smoothing + random resize-pad wrapper in forward + admix frame mix + EOT branch over JPEG/blur transforms. Estimated 4-6h coding.
- **Runtime**: 3-5 GPU-h after code lands.
- **Claim**: "Absolute SAM2Long damage can be raised without weakening SAM2 attack."
- **Success**: full-window mean SAM2Long drop ≥ CVaR-only + 0.05; retention ≥ 0.50; SAM2 loss < 0.05.
- **Failure**: SAM2Long absolute gain < 0.02, or retention up only via SAM2 denominator effect.

### E7. Scale winner to 10 clips [IF E5/E6 pass]

- **Config**: choose winner from E5/E6 (or combine if both pass cleanly). Run on all 10 DAVIS clips + fidelity + SAM2Long.
- **Runtime**: 6-10 GPU-h.
- **Claim**: "Pilot gains scale to the full test set."
- **Success**: SAM2 mean ≥ 0.60, SAM2Long mean ≥ 0.30, retention ≥ 0.50, insert LPIPS ≤ 0.10.
- **Failure**: pilot improvements vanish at scale.

### E8. CVaR `q` sensitivity [LOW PRIORITY]

- **Config**: 3 clips, `q ∈ {0.3, 0.5, 0.7}` + one LSE variant, `n_steps=30` (quick).
- **Runtime**: 2-3 GPU-h.
- **Claim**: appendix-level "no single magic q".
- **Success**: all q within ±0.05 of q=0.5 on mean SAM2 drop.
- **Failure**: high sensitivity → do NOT foreground this in paper.

---

## Claims Matrix (top 5 experiments)

`E1`=10-clip CVaR baseline, `E2`=full-window SAM2Long, `E3`=targeted metrics, `E4`=perceptual pilot, `E5`=transfer pilot

| E1 | E2 | E3 | E4 | E5 | Allowed paper story |
|---|---|---|---|---|---|
| Fail | * | * | * | * | **No decoy-only paper.** Keep decoy as appendix/diagnostic. |
| Pass | Fail | Pass | Fail | Fail | Strong short-horizon targeted attack. Not top-venue standalone. |
| Pass | Fail | Pass | Pass | Fail | High-fidelity targeted SAM2 attack, weak defense resistance. Almost. |
| Pass | Pass | Fail | Pass | Pass | Persistent degradation exists but targeted claim weak. Needs suppression baseline to anchor. |
| Pass | Pass | Pass | Fail | Pass | Good targeted + transfer, but visible inserts draw realism criticism. Not ready. |
| Pass | Pass | Pass | Pass | Fail | High-fidelity targeted SAM2 attack, weak SAM2Long. Borderline. |
| **Pass** | **Pass** | **Pass** | **Pass** | **Pass** | **Standalone decoy story is defensible. Suppression stays as baseline, not backup regime.** |

---

## Minimum Viable Paper (20 GPU-h floor)

1. **E1**: finish 10-clip CVaR-only (4-7 h)
2. **E2**: full-window SAM2Long on existing 3 clips + new 10-clip run (1-2 h, piggyback)
3. **E3 + E4**: fidelity + targeted metrics (free/cheap)
4. **E5**: Module 4 perceptual pilot on 3 hard clips (2-4 h)
5. If budget remaining: E6 on same 3 clips

Total budget: ~20 GPU-h. This answers the 4 reviewer-critical questions:
- Does CVaR generalize beyond 3 clips? (E1)
- Is the attack really targeted? (E3 + E4)
- Is the insert visibly bad? (E3)
- Can that visibility be fixed without killing the attack? (E5)

---

## Stretch Plan (80 GPU-h)

1. E1-E7 complete.
2. Final best model on **DAVIS-30**.
3. Full-window SAM2Long on DAVIS-30 (or 15-clip subset).
4. One 5-clip cross-backbone check on Base+ or Large (only if checkpoints local).
5. One 5-clip codec check on final model: PNG, JPEG Q95, H.264 (skip H.265 unless ffmpeg/eval smooth).
6. Small q/LSE robustness appendix on 3 clips, ONLY IF mainline story already solid.

---

## Honest Flags

**Can backfire:**
- **E8 CVaR q/LSE sweep**: useful internally, but if sensitivity is high it hurts more than helps. Appendix-only.
- **Cross-backbone transfer**: if fails on Base+/Large, weakens story fast. Stretch only.

**Optional nice-to-have (reviewers unlikely to demand):**
- Codec robustness beyond one small final-model subset.
- Full DAVIS-30 multi-seed.

**Explicitly out of scope this week:**
- Full `backbone × codec × DAVIS-30` grid.
- Teacher resurrection branch (A/B already killed it).

---

## Notes for Implementation

### E5 perceptual loss integration points

In `run_two_regimes.py` `optimize_unified()` around line 808-827 (current quality block):

```python
# Current (keep as helper):
lq = F.relu(ssim_threshold - differentiable_ssim(insert_base, adv))

# ADD (new perceptual stack):
# Extract edit_mask from Poisson blend region (available in role_data["targets"])
# Outside edit_mask: match frame_after
lq_identity = F.mse_loss(adv * (1 - edit_mask), frame_after * (1 - edit_mask))
# Inside edit_mask: match insert_base
lq_fit = F.mse_loss(adv * edit_mask, insert_base * edit_mask)
# Seam band (edit_mask dilated minus edit_mask eroded): DeltaE in Lab space
lq_deltaE = lab_deltaE_seam(adv, frame_after, edit_mask)
# LPIPS on full frame
lq_lpips = lpips_fn(adv * 2 - 1, frame_after * 2 - 1)

lq_perc = w_LPIPS * lq_lpips + w_DeltaE * lq_deltaE + w_identity * lq_identity + w_fit * lq_fit
lq = lq + lq_perc  # add to existing SSIM quality
```

### E6 transfer hardening integration points

In `optimize_unified()` Stage 3 PGD update (around line 840-850):

```python
# Current sign-PGD:
insert_deltas[si].data -= alpha_ins[si] * g.sign()

# ADD for Stage 3 joint only (if step >= stage2_end):
# 1. Momentum accumulator (MI-FGSM)
g_normalized = g / (g.abs().mean() + 1e-6)
momentum[si] = momentum_decay * momentum[si] + g_normalized
# 2. TI-FGSM: Gaussian smooth the gradient
g_smoothed = gaussian_conv2d(momentum[si], kernel_size=5, sigma=1.0)
insert_deltas[si].data -= alpha_ins[si] * g_smoothed.sign()

# DI²-FGSM: applied in forward pass, not here
# - Random resize + pad in forward with probability 0.7
# - admix: blend adv with nearby clean frame at α=0.2

# EOT: wrap forward with random JPEG/blur transform
```

---

## GPU Availability Strategy

- Polling: every 5-10 min check `nvidia-smi --query-gpu=memory.free`.
- If ≥ 16G free on any GPU, launch pending experiment immediately.
- If 2 OOMs in a row: switch to Pro 6000 setup (see CLAUDE.md for SSH/env).
- When Pro 6000 ready: bulk-run stretch experiments (DAVIS-30, cross-backbone).
