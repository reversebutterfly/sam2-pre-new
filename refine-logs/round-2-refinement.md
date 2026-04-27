# Round 2 Refinement

## Problem Anchor (verbatim, unchanged)

[Same as Round 0/1; see `PROBLEM_ANCHOR_2026-04-27.md`]

## Anchor Check

- **Original bottleneck**: SAM2 attack via insertion + bridge δ; memory-bank propagation must be the attack surface.
- **Why revised method still addresses it**: Same architecture (K=3 internal insertion + bridge δ + frozen ν + dense L_keep_full). Only the ABLATION DESIGN and DIAGNOSTIC PROTOCOL change.
- **Reviewer suggestions rejected as drift**: NONE. All R2 critical/important fixes are scientific tightening, not goal change.
- **Anchored constraint preserved**: scientific method still = RAW joint with both insertion AND bridge δ active. Wrapper still deployment-only.

## Simplicity Check

- **Dominant contribution after R2**: ONE main C1 (paired claim a + b) + TWO enabling E1/E2.
- **Components removed/merged**: NONE in R2 (R1 already restructured contributions).
- **Reviewer suggestions rejected as unnecessary complexity**:
  - "Make causal intervention granular by memory stage/layer separately" → ACCEPTED in part: A3 will report layer-aggregated AND last-block isolated. NOT exhaustive per-layer (would explode to 4-6 sub-experiments; appendix-only).
  - "Either own placement search or simplify aggressively" → KEEP own as E1; add Discussion paragraph explaining the ownership.
  - "Collapse traj+α+warp+R to simpler masked residual" → REJECTED for now: dev-4 v4.1 used the full parameterization to land dog J=0.6351; collapsing it risks regressing what already works. Mark as "future simplification ablation" but not in main paper.
- **Why remaining mechanism is still smallest adequate**: drop ANY of (insertion, bridge δ, dense no-regression) → attack collapses or regresses.

---

## Changes Made

### Change 1 — Fix A1 to isolate bridge δ contribution (CRITICAL)

- **R2 reviewer**: A0 vs full-v5 bundles insertion+search+stabilization+bridge δ. Doesn't isolate bridge δ.
- **Action**: A1 now compares **two configurations sharing identical (W*, ν, decoy_seeds)**, differing ONLY by whether bridge δ is active:
  - **A1-only**: insert at W* with ν (from A0 polish), NO Stage 14 (bridge frames untouched). This is "insert-only with v5's selected placement and ν".
  - **A1-full**: insert at W* with same ν + full v5 Stage 14 bridge δ. This is the RAW joint v5.
  - Both with the same placement search output W* (no random vs search confound).
  - Both with the same A0-polished ν (no ν optimization confound).
  - Bridge δ is the ONLY toggle.
- **Acceptance**: A1-full mean paired lift over A1-only ≥ +0.05 on 10 held-out clips, AND positive median lift, AND ≥6/10 strict wins.
- **Reasoning**: This is the cleanest isolation of "did bridge δ help, given the same insertion infrastructure?" Reviewers cannot blame search or ν optimization for the gain.

### Change 2 — Tighten d_mem protocol (IMPORTANT)

- **R2 reviewer**: "Object-related tokens = attention weight > median" is condition-dependent → potentially circular. Pre/post-projection unspecified.
- **Action**: Pre-registered d_mem protocol:
  - **Token set fixing**: Run CLEAN SAM2 forward on each held-out clip. At frame `c_K_clean = max(W_clean_sorted)` (last clean-space insert anchor, mid-video), record the cross-attention from current-frame query tokens to memory-bank value tokens. Sort memory tokens by total attention received (sum across query tokens). Take the TOP-32 memory tokens. Call this set `T_obj(clip)`. This set is FROZEN per clip and reused across clean / insert-only / full-v5.
  - **Layer/block fixing**: extract from `memory_attention.layers[-1].cross_attention` — the LAST cross-attention block in SAM2.1's memory mixer (where memory has fully participated). Pre-registered.
  - **Value-vector point**: PRE-output-projection (i.e., the value tensor V immediately after the linear projection of memory features, BEFORE the multi-head attention output projection). Pre-registered.
  - **Aggregation**: For each frame t, average the V vectors over `T_obj(clip)`, giving a single d_emb-dim vector per frame.
  - **Metric**: `d_mem(t) = 1 - cos(M_clean[t], M_attacked[t])`, where M is the aggregated value vector.
- **Reasoning**: Token set is now derived from ONE condition (clean) and reused → not circular. Layer/block/projection point are all named. Reviewers can audit the diagnostic precisely.

### Change 3 — A3 acceptance language softened to dual threshold (IMPORTANT)

- **R2 reviewer**: ≥0.20 abs collapse on ≥7/10 is aggressive. Make it strong-pass target, not only valid outcome.
- **Action**: Two-tier acceptance for A3:
  - **Strong pass** (preferred): full-v5 J-drop − blocked-write J-drop ≥ 0.20 abs on ≥7/10 → headline "memory-mediated failure is the dominant mechanism"
  - **Partial pass** (fallback): collapse ≥ 0.10 abs on ≥6/10 → narrowed framing "memory contribution is a substantial component of attack effectiveness, partially supporting memory-mediated mechanism"
  - **Fail** (collapse < 0.10 abs on majority): mechanism story RETIRED. Paper narrows to "engineered insertion + sparse perturbation attack on SAM2 with empirical effectiveness; mechanism analysis inconclusive". This forces a paper-direction decision (likely accept partial framing or move to workshop). Pre-registered.
- **Reasoning**: Honest pre-registration prevents paper from over-claiming when data doesn't support it.

### Change 4 — Run A3 first; let it gate framing

- **R2 reviewer**: "Run A3 before polishing paper story further; it is the claim-gating experiment."
- **Action**: Experiment order updated:
  1. **Day 1**: Implement `BlockInsertMemoryWritesHook` (~2h). Run A3 on 10-clip (~5h). 
  2. **Day 1 evening**: Read A3 verdict. If strong pass → proceed with full headline framing. If partial → narrow framing + continue. If fail → workshop pivot decision.
  3. **Day 2**: Run A1 (insert-only-with-W*-and-ν vs full-v5). Run A2 (random vs search placement). Run d_mem trace extraction.
  4. **Day 3**: Aggregate, write up.
- **Reasoning**: Putting the gating experiment first prevents wasted GPU time on a story that A3 might not support.

### Change 5 — Discussion paragraph on placement search ownership

- **R2 reviewer**: "Either own E1 more directly OR simplify aggressively."
- **Action**: Add §Discussion paragraph: "**Why search, not heuristic**: Earlier rounds of this project explored a 3-signal vulnerability scorer (confidence drop, mask discontinuity, Hiera discontinuity) for placement. Empirically, this heuristic was anti-correlated with attack effectiveness on a 10-clip ranked vs random comparison (mean J-drop 0.488 ranked vs 0.534 random) — falsified. We therefore use the joint curriculum search not as a methodological luxury but as the only empirically reliable placement strategy on memory-bank VOS. This is itself a finding: vulnerability heuristics adapted from per-frame fragility analysis fail to predict insertion success on SAM2 because the failure mechanism is memory-propagation, not per-frame sensitivity."
- **Reasoning**: Owns E1 directly with empirical justification. Reviewers see we know our own search-vs-heuristic tradeoff.

---

## Revised Proposal (full, R2)

### Title

*Memory-Mediated Failure of Prompt-Driven Video Segmentation: Causal Evidence from Internal Decoy Insertion with Sparse Bridge Perturbation*

### Method Thesis

**Three internally inserted semantic decoys at empirically-searched positions provide causal evidence that SAM2's prompt-conditioned memory propagation is the dominant failure mode, and sparse δ on the L=4 bridge frames immediately following each insert measurably extends the memory divergence beyond the insert-only baseline.**

### Contribution Focus (R2-locked)

- **C1 (main, mechanism)** — paired:
  - **C1.a** *(causal)*: Internal-insertion attacks on SAM2 produce a J-drop that COLLAPSES (≥0.20 abs strong / ≥0.10 abs partial pre-registered) when the inserted frames' memory writes are blocked, supporting the memory-mediated failure mechanism.
  - **C1.b** *(persistence)*: Sparse δ on L=4 bridge frames after each insert measurably extends d_mem(t) above the insert-only level for the L bridge frames in 75%+ of held-out clips.
- **E1 (enabling)**: vulnerability-aware joint curriculum placement search. Owned with empirical justification (per-frame heuristics anti-correlated; search is the only reliable strategy).
- **E2 (enabling)**: dense no-regression stabilization L_keep_full. Without it, optimization regresses unmonitored frames.
- **Deployment policy (separately reported)**: export-time accept/revert. NOT a contribution.
- **Explicit non-contributions**: no learned scorer, no diffusion generator, no LLM/VLM/RL planner, no UAP, no bank-poisoning, no first-frame-only attack.

### Complexity Budget (≤2 trainable)

- Frozen: SAM2.1-Tiny, SAM2VideoAdapter, LPIPS(alex), STE, A0 baseline, decoy alpha-paste compositor.
- New trainable (2): δ on bridge frames; ν on inserts.
- New non-trainable: joint curriculum placement search; anchored Stage 14 forward + dense L_keep_full + sparse L_gain_suffix; export wrapper (deployment-only).
- Diagnostic only (no training signal): `BlockInsertMemoryWritesHook` (A3); memory-readout extractor with R2-pre-registered protocol.

### Core Mechanism (mathematically specified, R1-locked + R2 d_mem tightening)

**Decoy family** (R1-locked, deterministic):
```
decoy_seed[k] = alpha_paste(
    x[c_k], shifted_object(m_true_at_c_k, dy_k, dx_k),
    feather_radius=5, feather_sigma=2.0
)
```
where `(dy_k, dx_k)` from `compute_decoy_offset_from_mask`. NO learned content.

**Optimized variables** (R1-locked):
- `ν[k] ∈ R^{H × W × 3}`, LPIPS(decoy+ν, decoy) ≤ 0.35, TV ≤ 1.2× base.
- `δ[t] ∈ R^{H × W × 3}` per bridge frame t, ε_∞ = 4/255 (or 2/255 if t==0), per-frame LPIPS ≤ 0.20.
- δ parameterized via (traj.anchor_offset, traj.delta_offset, edit_params.alpha_logits, edit_params.warp_s, edit_params.warp_r, R[k, l, :, :, :]).

**Loss** (R1-locked):
```
L_total = 0.05 · λ_margin · L_margin
        + 1.0 · L_keep_margin
        + 25.0 · L_keep_full
        + 2.0 · L_gain_suffix
        + (regularizers: alpha, warp, residual_TV, traj, fid)
```

**Memory readout d_mem(t) protocol (R2-tightened, pre-registered)**:
- Layer: `memory_attention.layers[-1].cross_attention`
- Value extraction point: PRE-output-projection (V tensor immediately after V linear projection, before multi-head output projection)
- Token set `T_obj(clip)`: top-32 memory tokens by total attention received (sum across query tokens) at clean-run frame `c_K_clean = max(W_clean_sorted)`. FROZEN per clip; reused across clean / insert-only / full.
- Aggregation: per-frame mean of V over `T_obj(clip)`.
- Metric: `d_mem(t) = 1 − cos(M_clean[t], M_attacked[t])`.

### Failure Modes (R2)

| Failure | Detection | Mitigation |
|---|---|---|
| Bridge δ regresses on clip (raw joint < A0) | per-clip RAW joint J-drop < A0 | Report honestly. Wrapper-selected column for deployment readers. |
| A3 weak collapse (<0.10 abs on majority) | A3 results | **Narrowed framing**: "memory contribution evidence, partial; not dominant mechanism." Pre-registered. |
| A3 fail (<0.10 abs on majority) | A3 results | **Workshop pivot decision**. Pre-registered. |
| Joint search low information | min_mass<1, singleton>0 | Multi-seed prescreen |
| Stage 14 pathological loop | wall>2× peer | Kill+retry seed 1 |
| Outlier-driven mean | top-clip>40% | Leave-one-out reporting |

### Validation (R2-tightened)

#### C1.a — memory causality
- **Experiment**: 10-clip held-out + `BlockInsertMemoryWritesHook` (nulls memory writes from W* frames; uses previous timestep's memory cache).
- **Strong pass**: collapse ≥0.20 abs on ≥7/10.
- **Partial pass**: collapse ≥0.10 abs on ≥6/10.
- **Fail**: <0.10 abs on majority → workshop pivot.

#### C1.b — persistence extension
- **Experiment**: per-clip d_mem(t) trace under R2 protocol, three conditions (clean / insert-only / full-v5).
- **Acceptance**: integral of (d_mem_full − d_mem_only) over t ∈ (w_K, w_K + L) is positive on ≥7/10 clips.

#### C2 — RAW joint headline
- **Experiment**: 10-clip RAW joint v5 vs A0 paired comparison.
- **Headline gates** (RAW JOINT, not wrapper):
  - ≥5/10 strict wins
  - mean paired lift ≥ +0.05
  - median paired lift > 0
  - top-contributor < 40%
  - mean joint J-drop ≥ 0.55
- Wrapper-selected reported separately (deployment column).

#### Ablations (R2, all reviewer-proof)
| # | Ablation | Configurations | Hypothesis |
|---|---|---|---|
| **A1** *(R2-fixed)* | Bridge δ contribution, isolated | (i) insert-only at W* with same ν vs (ii) insert+bridge δ at W* with same ν | A1-full mean paired lift ≥ +0.05, ≥6/10 strict wins, positive median |
| **A2** | Placement matters | Random K=3 vs joint curriculum search (both with full v5 Stage 14) | Search > random by ≥+0.10 mean lift |
| **A3** *(R2-pre-registered)* | Memory-causality, dual threshold | full-v5 vs full-v5 + memory-block hook | Strong pass 0.20/7-clip OR partial 0.10/6-clip, else workshop pivot |

### Discussion (E1 ownership, R2 added)

**Why search, not heuristic**: Earlier rounds of this project explored a 3-signal vulnerability scorer (confidence drop, mask discontinuity, Hiera discontinuity) for placement. Empirically, this heuristic was anti-correlated with attack effectiveness on a 10-clip ranked vs random comparison (mean J-drop 0.488 ranked vs 0.534 random) — falsified. We therefore use the joint curriculum search not as a methodological luxury but as the only empirically reliable placement strategy on memory-bank VOS. This is itself a small finding: vulnerability heuristics adapted from per-frame fragility analysis fail to predict insertion success on SAM2 because the failure mechanism is memory-propagation, not per-frame sensitivity.

### Compute & Timeline (R2 with A3-first)

| Day | Task | Compute | Decision gate |
|---|---|---|---|
| **Day 1 AM** | Implement `BlockInsertMemoryWritesHook` (~2h impl) + memory-readout extractor (~1h impl) | minimal | code review |
| **Day 1 PM** | A3 + d_mem trace on 10 held-out clips | 7 GPU-h | **A3 verdict: strong / partial / fail** |
| **Day 2** | C2 (RAW joint v5 + A0 paired) + A1 (paired ablation) overnight | 8 GPU-h overnight | Headline gates check |
| **Day 3** | A2 (random placement) + reporting scripts + writeup | 5 GPU-h + 4 author-h | Ablation table done |
| **Total** | | **~20 GPU-h** | **3 days** |

### Experiment Handoff Inputs

- **Must-prove claims**: C1.a (strong or partial), C1.b, C2 RAW joint headline gates.
- **Must-run ablations**: A1 (R2-fixed), A2, A3 (FIRST).
- **Critical datasets / metrics**: DAVIS-2017 10 held-out clips, J-drop on uint8 export, d_mem(t) trace per R2 protocol.
- **Highest-risk assumptions**:
  - A3 collapse magnitude unknown until experiment; pre-registered narrowing if weak.
  - RAW joint headline gates achievable from v4.1 dev-4 (75% on 4 clips) → likely 5-7/10 on held-out 10, but not bankable.
  - Memory-block hook achievable in ~2 hours; SAM2's memory_attention is modular per-layer.
