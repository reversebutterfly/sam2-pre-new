# Problem Anchor — MemoryShield v4-insert (vulnerability-aware insertion)

**Frozen 2026-04-23.** Supersedes `PROBLEM_ANCHOR_2026-04-23.md` (v3 suppression)
after user directive: keep insert+δ strategy; reject pure suppression; find
optimal insertion points using access to the original video.

## Bottom-line problem

Given a clean source video + first-frame target mask, produce a
**processed video** such that:

1. A downstream SAM2 user with the same first-frame mask prompt obtains
   **substantially degraded target segmentation across the entire
   processed video**.
2. The processed video remains **visually faithful** under a calibrated
   fidelity budget (insert frames + δ-perturbed original frames).

The processed video contains both (a) a small number of **synthetic
insert frames** placed at **vulnerability-optimal positions** chosen via
analysis of the clean video, and (b) the original frames with a
**moderate L∞-bounded δ perturbation**.

## Must-solve bottleneck

Current insertion-based attacks on SAM2 (prior internal v4 / v2) either:

- Place inserts at **FIFO-canonical positions** ({6,12,14} write-aligned)
  motivated by "defeat FIFO self-healing" narrative — this narrative is
  **falsified** (B2 causal ablation: non-cond FIFO bank contributes
  `|ΔJ| < 0.01` to SAM2 segmentation on 5 DAVIS clips). Position choice
  is therefore **unjustified**.
- Use **ProPainter with foreground-shift decoy content**, yielding insert
  LPIPS floor 0.67-0.89 regardless of decoy offset — fidelity-expensive
  for a mechanism (bank poisoning) that does not matter.
- Evaluate on a mixture of attacked + eval frames that overlap (v4's
  `EVAL_START=10` with `perturb_set` extending to frame 16), so reported
  J-drop is dominated by direct δ-on-eval damage, not by insertion.

The missing mechanism: **insertions placed where SAM2 is intrinsically
most vulnerable** (identified from the clean video), with content
optimized end-to-end to attack SAM2's **current-frame Hiera pathway at
the moment SAM2 processes the insert** (the one causal pathway that B2
confirmed is decisive). Combined with moderate δ on neighboring frames
to amplify each insert's effect.

## Non-goals

- NOT "defeat FIFO self-healing" (falsified).
- NOT bank poisoning via insert memory writes (B2: bank marginal).
- NOT clean-suffix eval (the problem is dataset protection; entire
  processed video is eval).
- NOT universal perturbation (per-video).
- NOT runtime hook / backdoor.
- NOT pure δ-only method (explicitly rejected by user — insertions are
  part of the design).
- NOT suppression-only loss (explicitly rejected by user — decoy /
  targeting logic allowed).

## Constraints

- **Target**: SAM2.1-Tiny (white-box surrogate, weights local).
- **Access**: publisher sees full clean video + first-frame mask. Runs
  clean SAM2 once offline to produce per-frame pseudo-labels and
  per-frame vulnerability metrics. **Zero DAVIS GT at any time.**
- **Insertion budget**: `K_ins ∈ {1, 2, 3}`. Positions chosen by the
  vulnerability scorer, not fixed schedule.
- **Fidelity budget** (two-tier, updated for insertion):
  - **f0 (prompt)**: ε_∞ ≤ 2/255, SSIM ≥ 0.98. Tight.
  - **Original frames t ≥ 1 (non-insert)**: ε_∞ ≤ 4/255, LPIPS ≤
    `F_orig = 0.20`.
  - **Insert frames**: LPIPS(insert, temporal_interpolation(neighbors))
    ≤ `F_ins = 0.35`. **Looser than originals** because the insert is
    content that was never in the source — the fidelity reference is
    "what a plausible interpolation of its neighbors would look like",
    not "the specific source frame". 0.35 is chosen to be (i) below
    the bmx-trees natural adjacent-frame LPIPS floor (0.38) so the
    insert does not visually jump more than natural motion would, and
    (ii) well above the ProPainter-decoy 0.7+ floor so we have a
    meaningful budget to negotiate with ν.
- **Eval**: mean J&F over the full processed video (f0 excluded as it
  is the prompt). No "clean suffix" carve-out.
- **Compute**: per-video pipeline ≤ ~15 GPU-min on Pro 6000; full
  DAVIS-10 ≤ 3 GPU-hours primary.

## Success condition

On DAVIS-10 hard subset, all four hold:

1. **Attack strength**: mean J&F drop ≥ 0.35 (slightly relaxed from
   v3's 0.40 because insertion is more constrained than dense δ; can
   be tightened after pilot).
2. **Fidelity triad**: all original-frame LPIPS ≤ `F_orig`, f0 SSIM ≥
   0.98, all insert LPIPS vs interpolated-neighbors ≤ `F_ins`.
3. **Vulnerability-specific claim**: inserts placed at the
   vulnerability scorer's top-K positions achieve ≥ **2×** the J-drop
   of inserts placed at random positions (same K, same ν optimization
   budget). This validates that position choice is causal, not
   arbitrary.
4. **Causal attribution**: restoration counterfactuals on attacked
   videos confirm the J-drop is attributable to the current-frame
   Hiera pathway at insert positions (R2-Hiera-swap recovers ≥ 0.25;
   R3-bank-swap recovers ≤ 0.02).

## Drift guardrails (this anchor MUST survive refinement rounds)

- No reviewer may remove insertions from the default method. If the
  reviewer says "drop inserts, keep pure δ" → **pushback with user
  directive in CLAUDE.md**.
- No reviewer may remove suppression rejection and introduce
  `L_suppress` as the default attack loss. The method uses **decoy
  targeting** as the primary loss (insert pulls SAM2 toward wrong
  location / identity).
- Insertion position selection must be **principled** (vulnerability
  scoring from clean SAM2). The canonical FIFO schedule is banned as a
  DEFAULT choice but may appear as a BASELINE for ablation.
- Fidelity budgets must be **floor-grounded**; the 0.10 LPIPS target
  is banned as infeasible by construction.

## Key experimental facts carried forward

- R001/R002/R003 (prior clean-suffix eval regime): J-drop 0.0004-0.0013.
  Not directly comparable because eval regime differs, but confirms
  "insertion + δ against clean-suffix future is structurally hard".
- B2 causal ablation: non-cond FIFO bank contributes `|ΔJ| < 0.01` on
  SAM2.1-Tiny segmentation across 5 DAVIS clips.
- D1 attention trace on R003: `A_insert = 0.515` on 6 eval frames, yet
  J unchanged. **This tells us**: inserts are being attended but
  attention-level hijack alone doesn't work. The insert must ALSO cause
  the current-frame Hiera pathway to produce bad features when SAM2
  processes the insert as "the current frame".
- LPIPS floor study: ProPainter decoy-shift floor 0.67-0.89; natural
  DAVIS adjacent-frame LPIPS mean 0.09-0.38 depending on clip.

## Working hypothesis (to test via pilot)

When SAM2 processes a synthetic insert at frame position m, three
effects compound:

1. **Current-frame feature attack**: the insert's Hiera features are
   adversarially optimized → SAM2's per-frame mask prediction at m is
   bad.
2. **Downstream propagation**: SAM2's next few frames (m+1, m+2) use
   the insert's bad prediction as part of their memory → error
   propagates briefly.
3. **Vulnerability amplification**: if m is a scene-transition or
   confidence-dip window, SAM2 is already near a tracking loss event
   → small insert nudge tips it over.

Combined with δ on surrounding originals (frames m-1, m+1, m+2) the
total effect should be substantial J-drop **without** relying on bank
poisoning (which B2 says is futile).

This anchor is the frozen statement of what the paper is about.
