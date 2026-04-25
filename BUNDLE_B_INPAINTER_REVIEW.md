# Bundle B Sub-Session 3 — Inpainter Choice Review

**Date**: 2026-04-25 night
**Reviewer**: Codex MCP gpt-5.4 (xhigh reasoning), thread `019dc51a-c71a-7971-bece-116a592de2f5`
**Prior consult**: gpt-5.2 thread `019dc511-5154-7a82-9083-525ffb078442` (recommended Option B)
**User question**: "I think as long as the method gets good results that's fine — wouldn't choosing Stable Diffusion be better?"

## Decision

**Commit Option B (silhouette alpha-feather paste, no inpainter).** Optionally keep Option C (LaMa) as a run-level ablation flag. **Do NOT build Option D (Stable Diffusion).**

Rationale fully grounded in Stage 14's `apply_continuation_overlay` math: under the current ≤0.35 alpha blend gated by `soft_decoy`, the duplicate frame's pixels at the true-object region are dropped by the (1-α·soft_decoy) ≈ 1 weighting and never reach SAM2's input. Inpainting only differs from B at that dropped region. Therefore B ≡ C ≡ D in the disjoint-region regime (typical case under Round 5's vulnerability-aware top-K placement).

## Compositor Options Evaluated

| Option | Description | LOC | Checkpoint | Latency / polish | Verdict |
|---|---|---|---|---|---|
| A | Poisson seamless clone (current proxy) | — | 0 | μs | REPLACE — bbox halo artifact |
| **B** | **Silhouette alpha-feather paste, no model** | **~150** | **0** | **negligible** | **CHOSEN** |
| C | LaMa frozen inpaint(remove) + paste | ~500 | 200 MB | ~1 s | Optional ablation flag |
| D | Stable Diffusion inpaint + paste | ~700 | 5 GB | ~45 s | REJECTED — dead weight |

## gpt-5.4 Reviewer Verdict (Brutally Honest)

> **Choose Option B. Under the current Stage 14 mixer, SD inpaint is almost certainly dead weight.** My honest prior for the mean pilot gain from B → SD is ~0, realistically [-0.005, +0.005] ΔJ, with a plausible negative tail from extra spill/LPIPS/stochasticity. I would NOT spend Bundle B time or GPU budget on SD unless you first change the attack so the inpainted true-position background actually survives into `x_edited`.

### Key Reasoning Points

1. **Mathematical argument (decisive):** Let `R = support(soft_decoy)` and `T = region where inpainting changes pixels` (true-object removal area). If `R ∩ T = ∅`, then on every pixel where `w = α · soft_decoy` is nonzero, `duplicate_B = duplicate_C = duplicate_D`. Since `x_edited = x_warped + w · (duplicate - x_warped)`, the attacked frame is identical. SD's quality advantage does NOT reach SAM2. Exceptions: overlap, feather spill, global inpaint bleed.

2. **Frame quality ≠ attack effectiveness:** Better-looking standalone duplicate frames do NOT automatically buy stronger attack loss. `L_margin`, α updates, and trajectory gradients depend on pixels that enter SAM2 via `x_edited`, not on hidden parts of `duplicate`.

3. **Memory attention doesn't rescue SD:** If bridge-frame input tokens are the same, stored memory is the same. SD's non-decoy hallucination cannot propagate if it never enters the encoded frame.

4. **LPIPS budget — no SD advantage:** The cap applies to `x_edited`, not raw `duplicate`. If `x_edited_B ≈ x_edited_D`, LPIPS is the same, so SD does NOT buy extra α budget. SD may actually HURT via spill outside intended mask, forcing α down.

5. **SD improves the wrong region:** SD's strength is hallucinating PLAUSIBLE BACKGROUND at the true-object removal region. The attack is driven by DECOY-SITE OBJECT EVIDENCE. Under current gating, SD improves a region SAM2 never sees.

6. **Redesign options that COULD make SD matter:**
   - `α_max = 1.0`: amplifier knob, not SD-specific. Boosts B and SD equally at decoy site.
   - **Background-first then paste object** (replaces `apply_continuation_overlay`'s blend with hard-replace): this is a NEW relocated-object attack, not a Bundle B compositor swap. Out of scope for sub-session 3.
   - Whole-frame low-α: bad trade — diffuse background changes spend LPIPS on weak signal.

7. **Expected effect size:** `E[ΔJ_SD - ΔJ_B] ≈ 0`. Would not assign meaningful probability to SD turning a true +0.04 pilot into a robust +0.05. If +0.01 observed once, suspect overlap or variance, not SD causation.

8. **No-Proxy rule interaction:** B is not globally the same duplicate frame as C/D, but after current Stage 14 mixer it is effectively equivalent in the disjoint regime. That justifies B IF we document the exact condition: `support(soft_decoy)` does not overlap the region where inpainting changes pixels. If Stage 14 changes later, this justification expires.

9. **Worst-case-honest:** SD is dead weight under current math.

10. **Cheaper validation gate (NOT a full ablation):**
    - First measure `sum(soft_decoy * true_mask)` or directly compare `x_edited_B` vs `x_edited_C`.
    - If difference ~0, stop. No SAM2 ablation needed.
    - If overlap nontrivial, run B vs C first. If LaMa doesn't beat B, SD won't rescue.
    - Only test SD specifically if C already shows replicated gain.

## Implementation Path Forward

1. **Build `memshield/semantic_compositor.py` with Option B as primary** — silhouette alpha-feather paste.
2. **Expose CLI flag**: `--oracle-traj-compositor {alpha_paste, poisson, lama}` (default `alpha_paste`); LaMa as optional fallback wired but not built unless pilot fails.
3. **Skip Option D entirely** unless Stage 14 redesigns to make true-object-region pixels survive into `x_edited` (e.g., background-first-then-paste).
4. **Document in commit message** the gpt-5.4 reviewer verdict + the disjoint-region condition that justifies B over C/D.

## Future Trigger Conditions

If pilot mean ΔJ comes in below +0.05 with B, the next investigation is NOT "switch to SD" — it's:
- Measure overlap between soft_decoy support and true-object region empirically.
- If overlap nontrivial: try Option C (LaMa) as the cheaper inpainter ablation. Cost: ~12 GPU-hours.
- If LaMa shows real gain: only THEN consider SD, scoped as a Stage 14 redesign (background-first-then-paste).

If Stage 14's overlay math is later changed to make duplicate's non-decoy pixels visible to SAM2 (e.g., α_max → 1, hard replacement, background-first composition), this analysis must be revisited.

## Saved Artifacts

- This review document (this file).
- gpt-5.4 thread ID: `019dc51a-c71a-7971-bece-116a592de2f5` (may expire on MCP restart).
- Prior gpt-5.2 thread ID: `019dc511-5154-7a82-9083-525ffb078442`.
