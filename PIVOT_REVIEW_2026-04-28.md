# Pivot Review - 2026-04-28

## Verdict

Proceed with the publisher-side pivot, but only with a narrower and more honest claim:

- Defensible: publisher-side cloaking for prompt-driven video object segmentation / SAM2-style extraction.
- Not defensible: first publisher-side video cloak in general.
- Not defensible with current evidence: joint `insert + delta` is stronger than insert-only.

Current package status:

- Pre-pivot framing: reject.
- Post-pivot potential: borderline AAAI if fidelity becomes the primary axis and evaluation is rebuilt around it.

## Main reviewer concerns

1. The current inserted frames are visibly fake. Under publisher-side framing, that is a fatal issue, not a cosmetic issue.
2. Your strongest current ablation says `insert-only > joint`. If the paper is forced to keep joint mechanism, delta must be repositioned as a minimal bridge/stabilizer under strict fidelity, not as the main attack booster.
3. Whole-processed-video `J-drop` is too weak as the main metric because bad masks on inserted synthetic frames can inflate the result. Main evaluation must move to original frames only, ideally against DAVIS GT.
4. The current vulnerability-aware top-K placement is a liability. Random placement beating it clip-wise means this module should not stay a flagship contribution unless fixed.
5. The "Glaze/Nightshade lineage" helps only if you use it carefully. The closest image analogue is actually PhotoGuard, not Glaze.

## Framing guidance

- Publisher / content owner = protector.
- Unauthorized SAM2-based extractor / scraper / indexer = adversary.
- Success = released video remains acceptable to humans and ordinary publishing transforms, while one-shot prompt-driven mask extraction becomes unreliable or requires repeated manual reprompting.

Use "stealthy" for the full method.
Use "imperceptible" only for the small bridge perturbation on original frames.

## Method recommendation

Replace the current decoy carrier with a natural interstitial frame:

1. Start from interpolation / flow-warp / occlusion-aware mid-frame between `I_t` and `I_{t+1}`.
2. Apply only a small localized adversarial steering on the inserted frame.
3. Keep the bridge perturbation on the next `L=2` real frames under a much tighter budget.

Practical ranking under current compute:

- Naturalness under real engineering constraints: interpolation/flow-warp > fixed composite with explicit object removal > diffusion infill.
- Compute cost: fixed composite < interpolation/flow-warp << diffusion infill.
- Paper novelty: interpolation/flow-warp and composite-fix are both fine; diffusion infill risks looking outsourced.

## Budget guidance

For publisher-side framing, current `LPIPS <= 0.20` is too loose.

Suggested targets for original perturbed frames:

- Prompt frame: `eps <= 1/255`
- Other bridge frames: `eps <= 2/255`
- Mean LPIPS: `<= 0.03`
- 95th percentile LPIPS: `<= 0.05`
- Mean SSIM: `>= 0.99`

Inserted frames should not be judged by `eps`; they need a realism / stealth evaluation:

- human flipbook pass-rate
- temporal consistency with neighbors

## Success criteria

Raw whole-video `J-drop` should become secondary.

Primary criteria should be on original frames only:

1. Unusable-track rate: attacked `J` or `J&F` falls below a usability threshold on a large fraction of clips.
2. Sustained failure: long consecutive runs of low-quality masks after an insert.
3. Optional practical metric: extra prompts needed to recover usable tracking.

## Minimal experiment package

1. Decoy carrier comparison on a small dev set:
   - current oracle composite
   - interpolation/flow-warp insert
   - composite + explicit object removal/inpainting
2. Main attack table on the 13-clip set with:
   - DAVIS GT evaluation
   - original-frames-only metrics
   - whole-processed-video as secondary
3. Human stealth test for inserted frames / short clips.
4. Tight-budget ablation:
   - insert-only
   - insert + bridge delta
5. Robustness to mundane transforms:
   - re-encode
   - frame-rate normalization / resampling

## Highest-leverage immediate experiment

Run a feasibility pilot on 4-6 representative clips using:

- interpolation-based inserted frames
- existing or simple anchor-aware windows
- bridge delta only on the next `L=2` real frames at `eps=2/255`

Measure:

- GT `J` on original frames only
- visible ghosting / human inspection

If this cannot retain roughly moderate degradation while eliminating obvious composites, the publisher-side pivot is probably not viable.

## Prior-art notes

- `Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models`, USENIX Security 2023
- `Raising the Cost of Malicious AI-Powered Image Editing` / PhotoGuard, ICML 2023
- `Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models`, IEEE S&P 2024
- `Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples` (Mist / AdvDM), ICML 2023
- `Black-Box Targeted Adversarial Attack on Segment Anything (SAM)`, IEEE TMM 2025
- `Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2`, NeurIPS 2025 Spotlight
- `Adversarial Attacks on Video Object Segmentation With Hard Region Discovery`, IEEE TCSVT 2024
- Additional dangerous analog: `Protecting Your Video Content: Disrupting Automated Video-based LLM Annotations`, CVPR 2025
- Additional structural analog: `Practical protection against video data leakage via universal adversarial head`, Pattern Recognition 2022

## Bottom line

Accept the pivot.

Push back on:

- broad firstness claims
- keeping current oracle-composite visuals
- using whole-processed-video `J-drop` as the main evidence
- presenting vulnerability-aware top-K as a mature contribution
- implying delta is already validated as an efficacy booster
