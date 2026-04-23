# Round 1 Review — 2026-04-22

- **Reviewer**: gpt-5.4 xhigh
- **Thread**: `019db31e-38dd-7fc1-8d72-c4dea843b254`
- **Overall Score**: 6.6 / 10
- **Verdict**: REVISE
- **Drift Warning**: NONE (problem anchor preserved)

## Dimension scores

| Dimension | Score | Weight |
|---|---|---|
| Problem Fidelity | 8 | 15% |
| Method Specificity | 6 | 25% |
| Contribution Quality | 6 | 25% |
| Frontier Leverage | 8 | 15% |
| Feasibility | 6 | 10% |
| Validation Focus | 6 | 5% |
| Venue Readiness | 6 | 5% |

## CRITICAL action items

1. **Phase 2 loss is mis-specified**. `BCE(g_u, 1[D_u])` on clean suffix asks the model to HALLUCINATE a decoy where no decoy physically exists. This conflates "prevent reacquisition" with "force wrong tracking." Fix: replace with target-suppression objective (low logit mass on `C_u` + background-lock / low-confidence); keep `L_stale` as internal regularizer of Phase 2.
2. **Claim scope overclaim**. "Any downstream SAM2-style VOS" broader than evidence supports. Narrow to "SAM2-family streaming promptable VOS preprocessors" OR add a second genuinely different target family.
3. **Validation gap — resonance ablation missing**. The FIFO-resonant scheduling claim is not directly tested. Add one schedule ablation: same inserts + budget, positions shifted OFF resonance. Resonance story only real if shifted underperforms.
4. **SAM2Long transfer must become MANDATORY**, not optional — if title/thesis says SAM2-family, must show transfer.
5. **"Monotone drop" too brittle**. Quantify no-recovery as `max rebound after first loss` or `post-loss AUC` instead.

## IMPORTANT action items

6. **Contribution quality — 4 pieces feeling like 4 contributions**. Frame `L_stale` strictly as Phase-2 internal regularizer; drop the "suppression mode" branch from main algorithm; if needed, present suppression as default Phase 2 for all clips. Goal: 2 components × 1 role each, for real.
7. **Feasibility — warmup order backwards**. Reverse to `ν-only → δ-only → joint` so optimizer first creates a real loss event before learning to preserve it. Also: cache image embeddings of clean suffix (never change during PGD); optimize `ν_k` only INSIDE paste box + seam band (not full frame).
8. **Hard LPIPS/SSIM projection unstable**. Replace with L∞ clip + penalty or augmented-Lagrangian constraints.

## Simplification opportunities

1. Delete decoy supervision from Phase 2 clean suffix (cuts one objective, prevents hallucination).
2. Freeze most insert pixels; only optimize pasted-target patch + seam band (shrinks PGD search space dramatically).
3. Parameterize insert schedule by MEMORY WRITES ("insert every `num_maskmem - 1` writes after conditioning"), not absolute frame indices — simpler, more transferable, directly proves mechanism claim.

## Modernization opportunities

**NONE**. ProPainter + RAFT + LPIPS is the right foundation-model-era level. Adding LLM/VLM/diffusion/RL would dilute contribution.

<details>
<summary>Full raw reviewer response</summary>

This is pointed in the right direction. It stays on the anchored preprocessor problem and the two-phase decomposition is the right abstraction. The main weakness is not "lack of modernity"; it is that Phase 2 is still partially solving the wrong subproblem, and several core interfaces are not yet sharp enough for a clean implementation or a top-venue narrative.

**Scores**
1. **Problem Fidelity: 8/10**. It remains a preprocessor-style protection method aimed at SAM2 memory self-healing, not a generic attack paper. The only fidelity risk is scope inflation in the phrase "any downstream SAM2-style model."
2. **Method Specificity: 6/10**. There is enough to start building, but the losses and state interfaces are still internally mismatched.
3. **Contribution Quality: 6/10**. The intended contribution is focused, but the current objective design makes it feel less elegant than claimed.
4. **Frontier Leverage: 8/10**. ProPainter/flow/LPIPS are the right primitives here. No LLM/VLM/diffusion/RL is the correct choice.
5. **Feasibility: 6/10**. The current optimization budget and control dimensionality look optimistic for stable per-video PGD through SAM2 memory.
6. **Validation Focus: 6/10**. The package is close to minimal, but it misses one mechanism-proof ablation and one scope-proof evaluation.
7. **Venue Readiness: 6/10**. Timely and potentially sharp, but not yet tight enough in claim scope or mechanism definition for NeurIPS/ICML/ICLR/USENIX/CCS/S&P.

**OVERALL SCORE: 6.6/10**

**Dimensions Below 7**
- **Method Specificity — 6/10**
  Weakness: `L_rec` currently uses `BCE(g_u, 1[D_u])` on clean post-prefix frames. That means Phase 2 is asking the model to segment a decoy location even when no decoy physically exists in those frames. This conflates "prevent reacquisition" with "hallucinate a wrong object," which is a harder and less clean target than the anchored problem requires. Also, `g_t`, `D_t`, `C_t`, slot provenance in `L_stale`, modified-video indexing after inserts, and "project to LPIPS/SSIM every step" are not concretely specified.
  Concrete fix: define `g_t` as the decoder foreground logit map; define `C_t` as a flow-warped, slightly eroded clean target mask; use Phase 1 to supervise decoy occupancy only on insert frames; use Phase 2 on clean suffix frames to minimize target occupancy plus recovery, not to force decoy occupancy. Concretely, replace the clean-suffix `BCE(g_u,1[D_u])` term with a target-suppression term such as low logit mass on `C_u` plus a background-lock / low-confidence objective, and keep `L_stale` as the mechanism enforcing memory hijack. Replace hard LPIPS/SSIM "projection" with `L_inf` clipping plus penalty or augmented-Lagrangian constraints.
  Priority: **CRITICAL**

- **Contribution Quality — 6/10**
  Weakness: the paper claims "2 components × 1 role each," but the actual mechanism currently looks like four things: resonant inserts, decoy-tracking on clean suffix, `L_stale`, and fallback mode branching. That weakens the elegance claim.
  Concrete fix: make the story strictly causal and minimal: Phase 1 creates the first loss event; Phase 2 prevents reacquisition. Treat `L_stale` as Phase-2's internal regularizer, not a semi-separate contribution. Drop explicit decoy supervision on clean suffix frames and remove the "suppression mode" branch from the main algorithm; if needed, present suppression as the default Phase-2 objective for all clips.
  Priority: **IMPORTANT**

- **Feasibility — 6/10**
  Weakness: 200-step joint PGD through SAM2 memory across 15 attacked originals, 3 inserts, and clean suffix rollouts is likely slower and less stable than the stated `2-6` GPU minutes, especially if `ν_k` optimizes whole insert frames. The warmup order is also backwards relative to the causal mechanism: you warm up `δ` before you have a reliable loss-inducing insert.
  Concrete fix: reverse the warmup order to `ν-only -> δ-only -> joint`, so the optimizer first creates a real loss event and only then learns to preserve it. Cache image embeddings for all clean suffix frames, since those pixels never change during PGD. Optimize `ν_k` only inside the pasted-object box plus a seam band, not the full insert frame. State a fixed training resolution explicitly, e.g. DAVIS 480p with batch size 1 and SAM2.1-Tiny.
  Priority: **IMPORTANT**

- **Validation Focus — 6/10**
  Weakness: the current package does not directly prove the "FIFO-resonant positions" claim, and "Optional Claim 3" is not optional if the paper title/thesis still says SAM2-style models. Also, "monotone drop" is too brittle as the main quantitative no-recovery metric.
  Concrete fix: add one tiny schedule ablation on the hard subset: same number of inserts, same perturbation budget, but shifted off resonance. If the memory-rollover story is real, that ablation should visibly underperform. Make SAM2Long transfer mandatory, even if only for the Full method. Keep the per-frame trajectory plots, but quantify no-recovery with `max rebound after first loss` or `post-loss AUC`, not strict monotonicity as the primary scalar metric.
  Priority: **CRITICAL**

- **Venue Readiness — 6/10**
  Weakness: "any downstream SAM2-style promptable VOS model" is broader than what the current method and validation support. Reviewers will call this overclaim immediately because the schedule and loss are tuned to SAM2-family memory internals.
  Concrete fix: narrow the claim to **SAM2-family streaming promptable VOS preprocessors** unless you are willing to add a second genuinely different target family. Given your anti-sprawl objective, narrowing the claim is the better move.
  Priority: **CRITICAL**

**Simplification Opportunities**
1. Delete decoy-location supervision from `L_rec` on clean suffix frames. Phase 2 should enforce non-recovery, not wrong-location tracking.
2. Freeze most insert pixels and optimize only the pasted target patch plus seam band. This keeps ProPainter as the generator and PGD as the attack, instead of blending them into one large control space.
3. Replace hard-coded `{f3,f7,f11}` with one rule based on actual memory write events, e.g. "insert every `num_maskmem - 1` writes after conditioning." That is simpler and more transferable than absolute frame indices.

**Modernization Opportunities**
NONE. The proposal is already using the right level of foundation-model-era machinery. Adding LLM/VLM/diffusion/RL would mostly add compute and dilute the contribution.

**Drift Warning**
NONE. This still solves the anchored preprocessor/self-healing problem. The issue is claim scope, not problem drift.

**Verdict**
**REVISE**

The core direction is good. The highest-value revision is to make Phase 2 strictly about preventing reacquisition on clean suffix frames, not about forcing a persistent decoy track. If you make that change, parameterize the insert schedule by memory writes, and narrow the claim scope to SAM2-family models, the proposal becomes much cleaner and substantially more venue-plausible.

</details>
