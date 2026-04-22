# Round 1 Refinement

## Problem Anchor (verbatim from PROBLEM_ANCHOR_2026-04-22.md)

- **Bottom-line problem**: Preprocessor takes clean video + first-frame mask of a target to protect, outputs modified video that causes any SAM2-family promptable VOS model to lose the target and not recover, while remaining visually acceptable.
- **Must-solve bottleneck**: FIFO streaming memory self-heals from single-frame perturbations.
- **Non-goals**: UAP / backdoor / runtime hook / maximal attack.
- **Constraints**: white-box architecture, per-video PGD, pixels-only, insert LPIPS ≤ 0.15 (goal 0.10), SSIM ≥ 0.95 on attacked originals.
- **Success condition**: eval-window J-drop ≥ 0.55 AND no-recovery (low rebound / low post-loss AUC) AND fidelity triad met AND both phases necessary (≥ 40% relative loss if either removed) AND SAM2Long transfer.

## Anchor Check

- **Original bottleneck**: FIFO memory self-heals → single-component attacks fail → preprocessor needs a compose-able mechanism.
- **Why revised method still addresses it**: the revision sharpens Phase 2 from "force decoy tracking on clean frames" to "suppress true target + reinforce memory staleness" — a STRONGER fit to the bottleneck (you do not have to explain why a decoy would appear in clean pixels; you just need SAM2 to not re-lock).
- **Reviewer suggestions rejected as drift**: NONE. All reviewer suggestions are within the anchor.

## Simplicity Check

- **Dominant contribution after revision**: a two-phase preprocessor mechanism whose Phase 2 is a *memory-staleness-regularized target-suppression objective*, purpose-built to neutralize FIFO self-healing. `L_stale` becomes Phase 2's internal regularizer (not a 3rd contribution).
- **Components removed / merged**:
  - DELETED: decoy-supervision term `BCE(g_u, 1[D_u])` from Phase 2 clean frames (no more hallucination).
  - DELETED: fallback "suppression mode" branch (reviewer 6). Suppression is now the DEFAULT Phase 2 objective for all clips — distractor vs non-distractor distinction removed from main algorithm.
  - MERGED: `L_stale` positioned as Phase 2 internal regularizer, not a co-equal contribution.
- **Reviewer suggestions rejected as unnecessary complexity**: NONE.
- **Why still smallest adequate**: Phase 1 (insert, decoy-target BCE on insert frames only) + Phase 2 (suppress + memory-staleness reg on clean frames) = minimum composition that defeats self-healing, no extras.

## Changes Made

### Change 1 — Phase 2 loss redesign (CRITICAL)

- **Reviewer said**: Phase 2 `BCE(g_u, 1[D_u])` on clean frames asks SAM2 to hallucinate a decoy where none exists; hardest target possible, not what the anchor requires. Fix: target-suppression (low logit on `C_u` + background-lock / low-confidence) + `L_stale` as memory-hijack mechanism.
- **Action**: replaced `L_rec` clean-suffix formula. New form:

  $$ \mathcal{L}_{\text{rec}}^{\text{new}} = \frac{1}{|U|}\sum_{u\in U}\Big[ \alpha_{\text{supp}} \cdot \text{CVaR}_{0.5}(g_u \cdot \mathbb{1}[C_u])^+ + \alpha_{\text{bg}} \cdot \text{softplus}(-\tau_{\text{conf}} - \max(g_u)) \Big] + \beta \cdot \mathcal{L}_{\text{stale}} $$

  where `CVaR_0.5(·)^+` is the median-gated top-half positive logit mass inside `C_u` (push down); `max(g_u)` is the peak logit across the frame (push below a low-confidence threshold `τ_conf` so SAM2 reports "no salient foreground" rather than picking up any object); `L_stale` unchanged.

- **Reasoning**: suppression + low-confidence directly matches "tracker cannot re-acquire" — no hallucination demand on clean pixels.
- **Impact**: Phase 2 is now a pure recovery-prevention loss. No decoy signal needed on clean suffix. Cleaner mechanism story.

### Change 2 — Insert schedule parameterized by memory writes (IMPORTANT / Simplification 3)

- **Reviewer said**: hard-coded `{f3, f7, f11}` is less transferable and does not DIRECTLY prove the FIFO-resonance claim. Use a rule based on memory writes: "insert every `num_maskmem - 1` writes after conditioning."
- **Action**: replaced schedule rule.

  **New rule**: for SAM2 with `num_maskmem = 7`, insert every 6 memory writes after conditioning (= f0). With a 15-frame attack prefix that gives candidate positions after original frame `{0 + 6, 0 + 12}` plus an extra insert force-placed into the last 2 prefix frames per bank-occupancy heuristic → inserts after original `{f5, f11, f13}`. Writing this as `p_k = 6k + r` with `r` chosen to force one slot near the prefix boundary.

- **Reasoning**: the rule IS the claim, so the schedule ablation (shifted off-resonance = `{f2, f5, f8}` or `{f4, f8, f12}` with non-multiple period) is a direct mechanism test.
- **Impact**: claim and test become the same concept; simpler narrative.

### Change 3 — Narrow claim scope (CRITICAL)

- **Reviewer said**: "any SAM2-style VOS" is overclaim given the schedule + loss are tuned to SAM2-family memory internals.
- **Action**: title / thesis / abstract text throughout now reads **"SAM2-family streaming promptable VOS"**. Evaluation commits to SAM2.1 (attack surrogate) + SAM2Long (transfer evidence). If reviewers want cross-family transfer, that becomes a followup, not the current paper's claim.
- **Impact**: aligns claim scope with evidence. Removes the easy-reject surface.

### Change 4 — Warmup order reversed + compute tightening (IMPORTANT)

- **Reviewer said**: current `δ → ν → joint` warmup is backward; should be `ν → δ → joint` so the loss event exists before δ tries to preserve it. Also: cache clean-suffix embeddings; optimize `ν_k` in paste region + seam only.
- **Action**:
  - NEW warmup: Stage 1 (1-40) `ν`-only with L_loss; Stage 2 (41-80) `δ`-only with L_rec (inserts frozen at Stage-1 output); Stage 3 (81-200) joint 2:1 δ:ν.
  - `ν_k` optimization space restricted to bounding box of pasted target + 5px seam band (everything else = frozen ProPainter output).
  - Clean-suffix image embeddings CACHED (pixels never change during PGD; only attention path through memory bank changes via δ + ν). This cuts per-step forward cost significantly.
  - Fidelity constraint: L∞ clip for δ (hard) + augmented-Lagrangian penalty term for LPIPS on ν (not hard projection) — reviewer 8.
- **Impact**: tighter compute envelope; warmup matches causal mechanism; much smaller ν search space.

### Change 5 — Validation additions (CRITICAL)

- **Reviewer said**: add schedule-off-resonance ablation; make SAM2Long transfer mandatory; replace "monotone drop" metric with `max rebound` or `post-loss AUC`.
- **Action**:
  - NEW Claim 3 (was Optional): SAM2Long transfer is required for all experiments.
  - NEW Claim 4 (mechanism test): schedule ablation — Full method with resonance-aligned vs off-resonance positions at fixed K_ins, budget, prefix length. Expected: ≥ 20pp J-drop gap.
  - REPLACED "monotone drop" primary metric with:
    - `Rebound = max_{u ∈ U_late} (J_clean(u) - J_atk(u)) - min_{u ∈ U_early} (J_clean(u) - J_atk(u))`, low = no-recovery.
    - `Post-loss AUC = (1/|U_late|) · Σ_{u ∈ U_late} J_atk(u)`, low = tracking stays lost.
    - Per-frame J trajectory still shown as qualitative evidence.
- **Impact**: validation now directly tests both composition necessity (Claim 1) AND the FIFO-resonance mechanism claim (Claim 4) AND the cross-model transfer claim (Claim 3).

### Change 6 — Contribution re-framing (IMPORTANT)

- **Reviewer said**: 4 things feel like 4 contributions weakening elegance claim.
- **Action**:
  - Dominant contribution = "two-phase preprocessor mechanism" — single contribution.
  - `L_stale` = Phase-2 internal regularizer (not labelled as separate supporting contribution).
  - FIFO-resonance schedule = supporting contribution (one, not two).
  - Suppression-as-default replaces fallback mode; no branching in main algorithm.
- **Impact**: genuinely 2 components × 1 role each, with L_stale inside Phase 2 as its memory-hijack mechanism.

---

## Revised Proposal

### Title
**MemoryShield: A Two-Phase Preprocessor for Protecting Video Data from SAM2-Family Streaming Promptable Segmenters**

### Problem Anchor
[Copy verbatim from above]

### Technical Gap
SAM2-family streaming VOS (SAM2.1, SAM2Long, Hiera variants) use a FIFO memory bank of size `num_maskmem = 7` plus a privileged f0 conditioning slot. At each frame the memory-attention cross-attends over bank tokens + current image features, writes a new entry, and evicts the oldest non-conditioning slot. This design **self-heals from single-frame perturbations**: a poisoned frame is evicted within 6 memory writes, and the intervening clean frames write correct entries that restore tracking. Preprocessor-style protection (outputting pixels, no runtime hook) cannot rely on single-frame attacks. Universal per-frame perturbations (UAP-SAM2) sidestep self-healing by degrading every frame, but are not suited to targeted per-user data protection with a tight LPIPS budget.

### Method Thesis
MemoryShield protects a specific target object in a user video by (i) inserting `K_ins` synthetic frames at FIFO-resonant positions that together drive a tracking-loss event by populating the bank with mislocated memory entries, and (ii) perturbing the prefix originals within L∞ ≤ 4/255 to suppress target re-acquisition under a memory-staleness regularizer that keeps the bank-attention mass on inserted-slot residuals. Each phase is individually insufficient; their composition is the minimum counter to FIFO self-healing.

### Contribution Focus
- **Dominant contribution**: a two-phase preprocessor mechanism against SAM2-family VOS, with each phase mapped to one step of the self-healing attack surface (loss event ← inserts; recovery prevention ← perturbed prefix + memory-staleness reg).
- **Supporting contribution**: FIFO-resonant insert scheduling parameterized directly by `num_maskmem`, tested via an off-resonance control.
- **Non-contributions**: no new SAM2 architecture, no new generator (ProPainter frozen), no UAP / backdoor / runtime hook, no learned scheduler, no cross-family claim.

### Proposed Method

#### Complexity Budget
- Frozen / reused: SAM2.1 Hiera-Tiny surrogate (+ SAM2Long for transfer eval), ProPainter inpainter, RAFT/Unimatch flow, LPIPS(AlexNet).
- New trainable components (≤ 2):
  1. Per-video `δ_orig` L∞-bounded pixel perturbation on f0..f14 (ε: f0 = 2/255, f1..f14 = 4/255).
  2. Per-video `ν_k` pixel perturbation on INSERT paste region + 5px seam band only (not full frame). LPIPS penalty under augmented Lagrangian.

#### System Overview
```
x_0:T + m_0
  │
  ├── (One-time) clean SAM2 run → C_u for u = 1..T (flow-warped + lightly eroded)
  ├── (One-time) decoy offset selection → D_u for u = 1..T
  ├── (One-time) ProPainter forward × K_ins → insert bases b_k
  │
  ├── Insert positions p_k = 6k + r (mod num_maskmem - 1), K_ins = 3 for 15-prefix
  │
  ├── PGD Stage 1 (1-40): ν-only with L_loss (creates loss event first)
  ├── PGD Stage 2 (41-80): δ-only with L_rec (ν frozen)
  ├── PGD Stage 3 (81-200): joint alternating 2:1 δ:ν with full L
  │
  └── Output: modified video = attacked originals + 3 optimized inserts
```

#### Core Mechanism

Per-video loss:

$$ \mathcal{L}(\nu, \delta) = \mathcal{L}_{\text{loss}} + \lambda_r \mathcal{L}_{\text{rec}} + \lambda_f \mathcal{L}_{\text{fid}} $$

**Phase 1 — `L_loss`** (inserts only):

$$ \mathcal{L}_{\text{loss}} = \frac{1}{K_{\text{ins}}} \sum_k \Big[ \text{BCE}(g_{\text{ins}_k}, \mathbb{1}[D_{\text{ins}_k}]) + \alpha \cdot \text{softplus}(\text{CVaR}_{0.5}(g_{\text{ins}_k} \cdot \mathbb{1}[C_{\text{ins}_k}]) + m) \Big] $$

Drives SAM2 to segment the decoy region on insert frames and write the decoy mask into the FIFO bank. Same as before.

**Phase 2 — `L_rec`** (clean post-prefix eval frames `U = f15..f15+H-1`, H=7):

$$ \mathcal{L}_{\text{rec}} = \frac{1}{|U|} \sum_{u \in U} \Big[ \alpha_{\text{supp}} \cdot \text{CVaR}_{0.5}(g_u \cdot \mathbb{1}[C_u])^+ + \alpha_{\text{bg}} \cdot \text{softplus}(-\tau_{\text{conf}} - \max(g_u)) \Big] + \beta \cdot \mathcal{L}_{\text{stale}} $$

- `CVaR_0.5(g_u · 1[C_u])^+`: median-gated positive logit mass inside true region → suppression (not hallucination).
- `softplus(-τ_conf - max(g_u))`: push global peak logit below a low-confidence threshold → SAM2 reports "no clear foreground."
- `L_stale`: memory-staleness regularizer (below) — the MECHANISM inside Phase 2 that actively counters self-healing.

**`L_stale`** (Phase 2 internal):

$$ \mathcal{L}_{\text{stale}} = \frac{1}{|V|} \sum_{u \in V} \log \frac{A_u^{\text{clean-recent}}}{A_u^{\text{insert-memory}} + \epsilon} $$

`V` = first 3 clean post-insert frames (not the whole eval window; `V` covers the self-heal window, `U` covers post-heal survival). `A_u^{·}` = fraction of memory-attention mass from queries in true-region pixels landing on each slot type (clean-recent = non-insert clean slots in the bank; insert-memory = insert-sourced slots). Low `L_stale` = bank-attention dominated by inserted slots; insert memory overrides clean-recent signal.

**Fidelity — augmented Lagrangian, not hard projection**:

$$ \mathcal{L}_{\text{fid}} = \mu_\delta \cdot \|\delta\|_\infty^+\text{clip} + \mu_\nu \cdot (\text{LPIPS}(x_{\text{ins}_k}, f_{\text{prev}_k}) - \text{LPIPS}_{\text{goal}})^+ + \mu_s \cdot \Delta E_{\text{seam}} $$

With µ increasing exponentially when the budget is exceeded (penalty method). `δ` is hard-clipped via L∞ clamp every step (fast), but `ν` uses soft penalty to avoid hard-projection oscillation.

#### Position Policy (supporting contribution)

Insert positions parameterized by memory-write count: `p_k = (num_maskmem - 1) · k + r` for `k = 1, 2, ..., K_ins` with offset `r ∈ {2, 3}` tuned to force one insert into prefix frame 13 or 14 (near eval boundary). For `num_maskmem = 7` and 15-prefix: `{p_1, p_2, p_3} = {6, 12, 13}` → inserts placed after original f5, f11, f12 (modified-timeline indices to be computed). This IS the FIFO-resonance claim; the off-resonance ablation uses shifted positions breaking the period-6 rule.

#### Modern Primitive Usage
ProPainter (frozen) as insert-content base generator; RAFT/Unimatch (frozen) for flow conditioning; LPIPS(AlexNet) for perceptual fidelity. No LLM / VLM / diffusion / RL — reviewer flagged these as unnecessary decoration.

#### Preprocessor Pipeline Integration
Run offline at publish-time. User supplies video + m_0. One-time: clean SAM2 forward (→ C_u), decoy offset (→ D_u), ProPainter forward × K_ins (seconds). Per-video PGD: 200 steps at DAVIS 480p, SAM2.1-Tiny, batch 1, ~3-8 GPU-min on single Pro 6000. Output: modified video. No SAM2 runtime hook.

#### Training Plan (reversed warmup)
1. Clean SAM2 run → `C_u` per frame (flow-warped + erode-2).
2. Decoy-offset selection once at video start.
3. ProPainter forward × K_ins → insert bases `b_k`.
4. **Stage 1** (steps 1-40): `ν`-only PGD with `L_loss`. Creates loss event.
5. **Stage 2** (steps 41-80): `δ`-only PGD with `L_rec` (inserts frozen). Learns to preserve loss.
6. **Stage 3** (steps 81-200): joint alternating 2:1 δ:ν with full `L`. Refines composition.
7. L∞ clip for δ every step; augmented-Lagrangian on LPIPS for ν.
8. Cache clean-suffix image embeddings — they never change during PGD.

#### Failure Modes & Diagnostics
- **F1: Phase-2 fails to suppress** (recovery by f20). Diagnostic: `post-loss AUC` high. Fallback: increase β, extend prefix to 22 frames.
- **F2: insert LPIPS > 0.10**. Diagnostic: per-frame LPIPS at PGD end. Fallback: tighter µ_ν, accept lower attack.
- **F3: decoy direction off-scene**. Diagnostic: shifted mask leaves image. Fallback: next-best offset.
- **F4: `L_stale` gradient unstable** (memory-attention attention mass not differentiable cleanly). Diagnostic: grad norm spikes in Stage 3. Fallback: swap to KL between foreground-query softmax distributions over bank slots.

(Reviewer 6 removed: no more F4 = "natural distractor" branching; suppression-as-default handles this.)

#### Novelty and Elegance Argument

- **Closest work**:
  - **UAP-SAM2** (NeurIPS 2025): universal frame-wise perturbation on SAM2. DELTA: per-instance preprocessor; insert+perturb composition; formal self-healing characterization.
  - **Chen et al. (WACV 2021)**: appended dummy frames for classification. DELTA: targeted scene-edit inserts in the middle + VOS memory target + Phase 2 recovery-prevention.
  - **ACMM 2023 one-shot VOS attacks**: first-frame only. DELTA: prefix + insertion specifically targeting streaming-memory self-heal.
  - **BadVSFM**: training-time backdoor. DELTA: inference-time preprocessor.
- **Elegance**: two components, one role each (Phase 1 creates the loss, Phase 2 prevents recovery). Everything else frozen/reused. `L_stale` is Phase 2's INTERNAL regularizer that makes the recovery-prevention mechanism work — it is not a third contribution.

### Claim-Driven Validation

#### Claim 1 (DOMINANT): Two-phase composition defeats FIFO self-healing
- **Minimal experiment**: 4-condition ablation on DAVIS-10 hard subset
  - Clean / Phase-1-only (inserts, no δ) / Phase-2-only (δ, no inserts) / Full
- **Metric**: mean SAM2 J-drop on eval window + per-frame J trajectory + post-loss AUC + rebound.
- **Expected evidence**: Full J-drop ≥ 0.55; each single-phase J-drop ≤ 50% of Full; Full rebound low, single-phase rebound high.

#### Claim 2 (SUPPORTING): `L_stale` is necessary to prevent recovery
- **Minimal experiment**: Full vs Full-no-L_stale on DAVIS-10.
- **Metric**: rebound + post-loss AUC + bank-attention mass distribution.
- **Expected evidence**: Full rebound ≤ 0.15 (no recovery); no-L_stale rebound ≥ 0.35; bank-attention mass on insert slots drops without L_stale.

#### Claim 3 (MANDATORY transfer): Mechanism transfers to SAM2Long
- **Minimal experiment**: attacked videos from Claim 1 Full → SAM2Long (num_pathway=3).
- **Metric**: SAM2Long J-drop; retention = SAM2Long-drop / SAM2-drop.
- **Expected evidence**: SAM2Long J-drop ≥ 0.25; retention ≥ 0.40.

#### Claim 4 (MECHANISM): FIFO-resonance matters
- **Minimal experiment**: Full method with resonance-aligned `p_k = 6k+r` vs off-resonance `p_k = 4k+r` (period 4 breaks FIFO-period logic), fixed K_ins = 3, fixed prefix 15, fixed ε, fixed LPIPS bound.
- **Metric**: mean J-drop gap.
- **Expected evidence**: resonance ≥ off-resonance by ≥ 20pp J-drop.

### Experiment Handoff

- **Must-prove claims**: Claims 1-4 above.
- **Must-run ablations**: insert-only / perturb-only / no-L_stale / off-resonance / ProPainter-vs-Poisson base.
- **Critical datasets**: DAVIS-10 hard subset for main ablations; DAVIS-30 for final numbers.
- **Critical metrics**: mean J-drop (eval window), post-loss AUC, rebound, per-frame J trajectory, insert LPIPS, orig LPIPS/SSIM, SAM2Long J-drop + retention, bank-attention mass breakdown.
- **Highest-risk assumptions**: (a) L_stale gradient stability through memory-attention; (b) ProPainter LPIPS floor compatible with ≤ 0.10 budget; (c) resonance-off comparison at budget-matched setting is fair.

### Compute & Timeline
- Per-video PGD: 3-8 GPU-min × 30 clips ≈ 3-4 GPU-hours
- ProPainter forward: 7 GPU-min total
- Ablation suite: 7 conditions × DAVIS-10 = ~6 GPU-hours
- SAM2Long eval: 7 GPU-hours
- **Total ~24 GPU-hours on single Pro 6000 ≈ 1 GPU-day.**
- Timeline: 2 weeks method + 1 week ablations + 1 week writing = 4 weeks total.
