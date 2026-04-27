# Round 1 Refinement

## Problem Anchor (verbatim, not changed)

- Bottom-line: Publisher-side adversarial attack on SAM2 video segmentation — clean video + first-frame mask → modified video that drops Jaccard on uint8 export, using BOTH internal decoy frame insertion AND sparse δ on adjacent original (bridge) frames.
- Must-solve bottleneck: Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA 2310.10010 — none combines internal insertion + original-frame δ on memory-bank VOS / SAM2.
- Non-goals: pure suppression, pure-δ, pure-insertion, first-frame-only, universal, single-image SAM v1, audit pivot, bank-poisoning, FIFO-defeat.
- Constraints: white-box SAM2.1-Tiny on Pro 6000 ×2; per-clip targeted DAVIS; two-tier fidelity. AAAI venue. Must keep BOTH insertion AND original-frame δ active.
- Success: 10-clip held-out gates ≥5/10 wins, mean ≥+0.05, median >0, top<40%, applied≥60%, mean J-drop ≥0.55. 3 reviewer-proof ablations.

---

## Anchor Check

- **Original bottleneck**: SAM2 attack via internal insertion + bridge δ. Memory-bank propagation must be the attack surface; the publisher's full-video access must be exploited (not just first-frame, not just dense δ).

- **Why the revised method still addresses it**: We still insert K=3 internal decoys + sparse bridge δ on originals. The change is at the CLAIM level: scientific method = raw joint (not wrapper-selected); wrapper is deployment policy.

- **Reviewer suggestions rejected as drift**: NONE. All codex CRITICAL items are accepted as fixes (none of them removes insertion or removes original-frame δ).

- **Reviewer suggestion accepted with care**: codex's "use SAM2 memory readout as auxiliary loss" is tempting but adds complexity. We use it ONLY as a CAUSAL DIAGNOSTIC (ablation A3), NOT as a training signal. This keeps complexity budget at ≤2 trainable components.

---

## Simplicity Check

- **Dominant contribution after revision**: ONE main mechanism finding ("internal insertion causally biases SAM2's memory; bridge δ extends the bias"). Two ENABLING components (placement search, no-regression stabilization). Wrapper is NOT a contribution — it's deployment policy reported separately.

- **Components removed or merged**:
  - "no-regret adaptive attack" demoted from contribution → deployment policy.
  - "First demonstration" claim narrowed → "evidence + persistence extension".
  - A3 "all-frames-δ vs insert+bridge-δ" REPLACED with memory-causality ablation (cleaner mechanism evidence).

- **Reviewer suggestions rejected as unnecessary complexity**:
  - Codex modernization #1 (SAM2 memory readout as **auxiliary loss**) — REJECTED as training signal (adds gradient pathway, threatens 2-component budget). ACCEPTED as causal diagnostic only.
  - Codex modernization #2 (frozen vision prior for decoy init) — REJECTED for default; OK as backup if reviewer requests "decoy quality" ablation.
  - "Own placement search OR simplify aggressively" — we now own it as enabling component E1 with explicit ablation; no further simplification.

- **Why remaining mechanism is still smallest adequate**: drop ANY of (insertion, bridge δ, no-regression stabilization) and either (a) attack collapses (insertion gone → no decoy memory) or (b) optimization regresses unmonitored frames (no-regression gone → 50% revert as in v4.0). Placement search (E1) is necessary because random-K3 was empirically anti-correlated on prior 10-clip data — paper needs to honestly report this, not hide it.

---

## Changes Made

### Change 1 — Contribution restructure (CRITICAL #1+#2)

- **Reviewer said**: Demoting placement search / L_keep_full / polish_revert to "details" is dishonest. Reframe to ONE main + TWO enabling.
- **Action**: Restructured §Contribution Focus:
  - **C1 (main, mechanism)**: internal insertion causally biases SAM2 memory; bridge δ extends bias persistence.
  - **E1 (enabling)**: vulnerability-aware joint curriculum placement search (paired ablation in main paper).
  - **E2 (enabling)**: dense no-regression stabilization (L_keep_full).
  - **DEMOTED to deployment policy** (separate column in results table, not a contribution): export-time accept/revert wrapper.
- **Reasoning**: Honest framing matches the actual outcome dependence. The paper still has one focused mechanism story (C1); E1/E2 are clearly framed as "what makes C1 land", not as additional contributions.

### Change 2 — Scientific method = raw joint (CRITICAL #1)

- **Reviewer said**: max(joint, A0) wrapper causes conditional drift. Define scientific method = raw joint.
- **Action**: 
  - **Headline gates apply to RAW joint** (no wrapper). Updated success bar: ≥5/10 strict wins on RAW joint, mean RAW joint ≥+0.05, median RAW joint > 0.
  - Wrapper-selected results reported in **separate column** of main results table, labeled "deployment policy".
  - Failure mode "Bridge δ regresses on clip" no longer says "mitigated by wrapper" — instead says "report failure honestly; wrapper-selected column is for deployment readers, not for headline claim".
- **Reasoning**: Anchored problem requires the method itself to deliver, not a max-selector over a baseline. Wrapper is fine as engineering; not as the science.

### Change 3 — Memory-causality ablation replaces A3 (CRITICAL #3)

- **Reviewer said**: A3 confounds memory writes / temporal discontinuity / sparsity / budget.
- **Action**: New ablation A3:
  - **A3 (memory-causality)**: With same v5 method (full v5: insert + bridge δ + Stage 14), intervene at SAM2's memory-bank update step on the K=3 insert frame indices: clear / null-out the memory writes from those frames so they CANNOT enter the bank. Keep all other frames (bridges, originals) unchanged. Re-run the segmentation forward with this hooked memory bank.
  - **Hypothesis**: If the attack's J-drop COLLAPSES (drops by >0.20 absolute) when insert-frame memory writes are blocked → "memory hijack" mechanism is causally supported. If J-drop persists at near-baseline → mechanism is something else (e.g., feature corruption at the insert frame's own forward pass) and we must narrow the claim.
  - **Implementation**: SAM2 has `memory_attention.encode_memory()` that writes per-frame into a memory dict. Add a hook `BlockInsertMemoryWritesHook` that nulls the memory write at t ∈ W_attacked and uses the previous time step's memory cache instead. Estimated 2-hour implementation.
- **Reasoning**: This is a CAUSAL ablation that isolates "did the insert frame's MEMORY contribution drive the failure?" from "did the insert frame's own FORWARD output drive the failure?" — critical for the mechanism claim.

### Change 4 — Method specificity (decoy family + memory readout) (IMPORTANT #1)

- **Reviewer said**: "semantic decoy" / variables / memory readout under-specified.
- **Action**: §Core Mechanism now contains:
  - **Decoy family — one explicit choice**: duplicate-object alpha-paste with feathered mask. For each insert position w_k with corresponding clean-space c_k = w_k - k, the decoy seed is built as: `decoy_seed[k] = alpha_paste(x[c_k], shifted_object(m_true_at_c_k, dy_k, dx_k), feather_radius=5, feather_sigma=2.0)` where `(dy_k, dx_k)` is computed by `compute_decoy_offset_from_mask(m_true_at_c_k)` — a deterministic centroid-based shift to put the duplicate in a non-overlapping location. NO learned generator.
  - **Optimized variables (mathematical notation)**:
    - `ν[k] ∈ R^{H × W × 3}` per insert k, bounded by per-insert LPIPS(decoy_seed[k] + ν[k], decoy_seed[k]) ≤ 0.35 + per-insert TV ≤ 1.2× base.
    - `δ[t] ∈ R^{H × W × 3}` per bridge frame t ∈ bridge_frames, bounded by ε_∞=4/255 (or 2/255 if t==0) AND per-frame LPIPS(x_clean[t] + δ[t], x_clean[t]) ≤ 0.20.
    - Bridge-frame δ is parameterized via `(traj.anchor_offset, traj.delta_offset, edit_params.alpha_logits, edit_params.warp_s, edit_params.warp_r, R[k, l, :, :, :])` per Stage 14 (alpha-paste compositor + warp + masked residual). Full equations in §Implementation.
  - **Memory readout (pre-registered for C1 evidence)**:
    - SAM2.1's memory bank is queried in `memory_attention.cross_attention` during each forward step. Extract the per-token memory-bank value vectors at frame t (before mixing with current Hiera features).
    - For each frame t in [0, T_proc), compute `M_clean[t]`, `M_attacked_no_polish[t]` (insert-only), `M_attacked_full[t]` (insert+bridge δ).
    - **Metric**: cosine distance `d_mem(t) = 1 - cos(M_clean[t], M_attacked[t])` per frame, averaged over object-related tokens (i.e., tokens whose attention weight to current-frame query > median).
    - **Trace plot**: `d_mem(t)` for t=0..T-1, three curves (clean reference at 0; insert-only; insert+full-v5). The C1 narrative demands `d_mem_full(t) > d_mem_only(t)` for t in (w_K, T_proc) — i.e., bridge δ holds the memory divergence open longer.
- **Reasoning**: Reviewers can now mentally simulate the attack and the diagnostic. No further "semantic decoy" hand-waving.

### Change 5 — Novelty claim narrowing (IMPORTANT #2)

- **Reviewer said**: "First demonstration" is too strong without causal ablation.
- **Action**: 
  - DROP "first demonstration" framing.
  - REPLACE with two paired claims, each with explicit evidence requirement:
    - **C1.a**: "Internal decoy insertion at K=3 vulnerability-aware positions provides evidence that SAM2's prompt-conditioned memory propagation is the dominant failure mode, with attack effect collapsing when insert-frame memory writes are blocked (A3)."
    - **C1.b**: "Sparse δ on bridge frames adjacent to inserts measurably extends the duration of the memory-divergence above the insert-only baseline, by holding `d_mem(t)` elevated for L=4 frames after each insert."
- **Reasoning**: Each sub-claim is now tied to a specific ablation / measurement. No more aspirational "first demonstration" wording.

---

## Revised Proposal (full)

### Title (working)

*Memory-Mediated Failure of Prompt-Driven Video Segmentation: Evidence from Internal Decoy Insertion with Sparse Bridge Perturbation*

(Title change reflects narrowed claim from "first demonstration of memory hijack" → "evidence of memory-mediated failure".)

### Method Thesis

**Three internally inserted semantic decoys at vulnerability-aware positions provide causal evidence that SAM2's prompt-conditioned memory propagation is the dominant failure mode of the segmenter, and sparse δ on the L=4 bridge frames immediately following each insert measurably extends the memory divergence beyond the insert-only baseline.**

(Note: deliberately paired claim. C1.a = mechanism evidence via memory-block ablation. C1.b = persistence extension via bridge δ.)

### Contribution Focus

- **C1 (main, mechanism)** — paired:
  - C1.a: Internal-insertion attacks on SAM2 produce a J-drop that collapses (>0.20 absolute) when the inserted frames' memory writes are blocked (A3 ablation), supporting the memory-mediated failure mechanism.
  - C1.b: Sparse δ on L=4 bridge frames after each insert measurably extends `d_mem(t)` above the insert-only level for the L bridge frames in 75%+ of held-out clips.
- **E1 (enabling)** — vulnerability-aware joint curriculum placement search (3-phase K=1→2→3 + simplex slack + suffix-probe surrogate). Paired ablation A2 vs random K=3 confirms placement matters (≥+0.10 mean lift over random).
- **E2 (enabling)** — dense no-regression stabilization via L_keep_full (mean over all non-insert suffix of relu(u_cur(t) - u_A0(t))). Without it, optimization regresses unmonitored frames (v4.0 50% revert evidence).
- **Deployment policy (separately reported, NOT a contribution)**: export-time `polish_revert` selector — publishes max(joint, A0) for deployed-attack scenarios; not part of the scientific claim.
- **Explicit non-contributions**: no learned scorer, no diffusion generator, no LLM/VLM/RL planner, no UAP, no bank-poisoning, no first-frame-only attack. Decoy is one deterministic family (duplicate-object alpha-paste, no learned content).

### Complexity Budget (unchanged from round 0, ≤2 trainable)

- Frozen / reused: SAM2.1-Tiny, SAM2VideoAdapter, LPIPS(alex), STE quantize, A0 baseline.
- New trainable (2): δ on bridge frames; ν on inserts.
- New non-trainable: joint curriculum placement search; anchored Stage 14 forward + dense L_keep_full + sparse L_gain_suffix; export-time wrapper (deployment-only).
- New diagnostic only (no training signal): `BlockInsertMemoryWritesHook` for A3; memory-readout extractor for d_mem(t) trace.

### System Pipeline (unchanged)

INPUT → A0 polish → joint curriculum placement search → anchored Stage 14 (frozen ν) → optionally export-time wrapper.

### Core Mechanism (now mathematically specified)

**Decoy family**: duplicate-object alpha-paste, deterministic, no learned content. Equations and parameters per Change 4.

**Bridge amplifier**: bridge_frames_by_k = {t : w_k < t ≤ w_k + L, t ∉ W_attacked}, L=4. δ on these frames optimized via Stage 14 with composite parameterization (traj+α+warp+R).

**Anchored loss (frozen ν, optimize δ)**:

```
L_total = 0.05 · λ_margin · L_margin                     # local surrogate, attenuated
        + 1.0 · L_keep_margin                             # no-regression on attacked window
        + 25.0 · L_keep_full                              # DENSE no-regression on suffix
        + 2.0 · L_gain_suffix                             # sparse gain (6 probes)
        + (regularizers: alpha, warp, residual_TV, traj, fid)
```

with L_keep_full = mean_{t ∈ keep_suffix_frames} relu(u_cur(t) - u_A0(t)) over ALL non-insert attacked-space frames after w_first.

**Memory readout (for C1 evidence)**: cosine distance per frame between attacked and clean SAM2 memory bank, averaged over object-related tokens. Pre-registered.

### Modern Primitive Usage

INTENTIONALLY MINIMAL. SAM2 (2024 foundation model) is the target. SAM2's memory readout is used as **causal diagnostic**, NOT as auxiliary loss. No LLM/VLM/Diffusion/RL components.

Optional fallback: if reviewers request decoy realism, swap duplicate-object alpha-paste with 1-step DDIM Stable Diffusion 1.5 sampling (frozen, no training) — default OFF, only enable in supplementary "decoy quality" ablation.

### Failure Modes and Diagnostics (revised)

| Failure | Detection | Mitigation |
|---|---|---|
| Bridge δ regresses on clip (raw joint < A0) | per-clip RAW joint J-drop < A0 J-drop | **Report honestly** as a clip where bridge δ doesn't help. Wrapper exists but is deployment-only. |
| Joint search converges to W with low information | min_mass < 1.0, singleton/inward_proj > 0 | Multi-seed prescreen (--prescreen-seed 1,2) |
| Stage 14 diverges in pathological loop | wall > 2× peer | Kill + retry with seed 1 |
| Mean lift driven by 1 outlier | top-clip share > 40% | Per-clip ablation table, leave-one-out mean |
| Memory-block ablation (A3) does NOT collapse attack | A3 J-drop within 0.20 of full-v5 J-drop | **Mechanism narrowed**: claim "feature corruption at insert frame" not "memory hijack". Honest fallback. |

### Novelty Argument (narrowed)

Same 5-axis comparison vs Chen WACV 2021 / UAP-SAM2 / Li T-CSVT 2023 / PATA, BUT:
- Replace "first demonstration of memory hijack" → "first **causal evidence** via memory-write blocking that internal-insertion-induced J-drop on SAM2 propagates through the memory bank".
- Add: "first attack on SAM2 (or any memory-bank VOS) that combines internal insertion with sparse bridge δ on adjacent originals; UAP-SAM2 (NeurIPS 2025) uses dense δ everywhere with no insertion".

### Claim-Driven Validation (revised)

#### C1 — Memory-mediated failure mechanism
- **C1.a (block-write ablation)**: 10-clip held-out + memory-block hook. Hypothesis: full-v5 J-drop − blocked-write J-drop ≥ 0.20 absolute on ≥7/10 clips.
- **C1.b (persistence extension)**: per-clip d_mem(t) trace comparison insert-only vs full-v5. Hypothesis: integral of (d_mem_full − d_mem_only) over t ∈ (w_K, w_K + L) is positive on ≥7/10 clips.

#### C2 — Method (no-regret in honest reporting)
- 10-clip held-out paired comparison RAW joint (v5 without wrapper) vs A0 (insert-only).
- Headline gates apply to RAW joint:
  - ≥ 5/10 strict wins (joint J-drop > A0 J-drop)
  - mean paired lift (joint − A0) ≥ +0.05
  - median paired lift > 0
  - top-contributing clip < 40% of total lift
  - mean joint J-drop ≥ 0.55
- Wrapper-selected column reported separately for deployment readers.

#### Ablations (3, all reviewer-proof)
| # | Ablation | Configurations | Hypothesis |
|---|---|---|---|
| **A1** | Bridge δ contribution | A0 vs full-v5 (RAW joint) | Bridge δ adds beyond A0 on majority of clips, mean lift ≥ +0.05 |
| **A2** | Placement matters | Random K=3 vs joint curriculum search (both with full-v5 polish) | Mean lift of search over random ≥ +0.10 on 10-clip |
| **A3** | Memory-causality | Full-v5 vs Full-v5 with `BlockInsertMemoryWritesHook` | Blocking insert-frame memory writes COLLAPSES the attack (drop ≥ 0.20 absolute on ≥7/10 clips), confirming memory-mediated failure |

### Experiment Handoff Inputs

- **Must-prove claims**: C1.a (memory-block collapse), C1.b (persistence extension), C2 (RAW joint ≥ A0 with non-trivial applied rate).
- **Must-run ablations**: A1 (bridge δ), A2 (placement), A3 (memory-causality).
- **Critical datasets / metrics**: DAVIS-2017, 10 held-out clips, J-drop on uint8 export, d_mem(t) trace.
- **Highest-risk assumptions**: 
  - Memory-block hook achievable with ~2-hour implementation (likely; SAM2's memory_attention is modular).
  - A3 collapse-magnitude assumption (≥0.20 drop) untested but plausible based on the mechanism logic.
  - RAW joint can satisfy headline gates without wrapper crutch (this is the key risk; v4.1 dev-4 had 75% strict-win rate on RAW joint, suggesting ~5/10 likely on held-out).

### Compute & Timeline

| Task | Compute | Wall |
|---|---|---|
| 10-clip RAW joint v5 + A0 paired (already-run-able) | 5+3=8 GPU-h | overnight |
| A1 (bridge δ contribution, paired with above) | overlap | 0 extra |
| A2 (random K=3 placement) | 5 GPU-h | overnight |
| A3 (memory-block hook + 10-clip rerun) | 2 GPU-h impl + 5 GPU-h rerun | sub-day |
| d_mem(t) trace extraction (clean+only+full × 10 clips) | 3 GPU-h | sub-day |
| **Total** | **~26 GPU-h** | **~3-4 days** |

Implementation status: v5 = v4.1 commit `da719dc` already on Pro 6000. New code: `BlockInsertMemoryWritesHook` (~2 hours), memory readout extractor (~1 hour), reporting scripts (~2 hours).
