# Round 3 Refinement

## Problem Anchor (verbatim, unchanged)

[See `PROBLEM_ANCHOR_2026-04-27.md`]

## Anchor Check

- **Original bottleneck**: SAM2 attack via insertion + bridge δ; memory-bank propagation is attack surface.
- **Why revised method still addresses it**: Same architecture as R0/R1/R2. R3 changes are operational locking + conditional paper framing — no method change.
- **Reviewer suggestions rejected as drift**: NONE.

## Simplicity Check

- **Dominant contribution**: same C1 + E1 + E2 structure from R2.
- **Components removed/merged**: none.
- **R3 reviewer suggestions accepted**: operational A1 lock, A3 hook pseudocode, conditional paper framing, table/column separation, T_obj sensitivity appendix.
- **R3 simplification ACCEPTED**: compress traj+α+warp+R to "masked bridge-edit parameterization" in main paper; full math in Appendix B.
- **R3 modernization ACCEPTED partially**: T_obj 16/32/64 sensitivity in appendix.
- **R3 modernization REJECTED**: layer-sensitivity per-block hook → only run if A3 strong-passes; otherwise out of scope.

## Changes Made

### Change 1 — Operationally lock A1 (IMPORTANT)

- **R3 reviewer**: A1 correct in concept but enforce same upstream W* and ν, ONLY zero bridge variables.
- **Action**: Pseudocode + pre-registration:
```python
# Shared upstream (run once per clip):
W_star = joint_curriculum_placement_search(clip, ...)
nu_init = A0_polish(clip, W_star, ...)            # 100-step PGD
decoy_seeds = build_decoy_insert_seeds(clip, W_star)
state = build_attack_state_from_W(W_star, ...)

# A1-only arm (insert-only at W*, NO bridge δ):
x_only = export_with_inserts(clip, decoy_seeds + nu_init, W_star)
# ALL bridge-edit variables zeroed:
#   traj.anchor_offset = 0; traj.delta_offset = 0
#   edit_params.alpha_logits = -inf (alpha_max → 0)
#   edit_params.warp_s = 0; edit_params.warp_r = 0
#   R = 0
# I.e., bridges are unmodified clean frames; inserts are decoy_seeds+nu_init.

# A1-full arm (insert + bridge δ via Stage 14):
traj_*, edit_*, R_* = anchored_stage14_polish(state, nu_init, ...)
x_full = export_with_inserts_and_bridge_delta(...)

# Compare J-drop(x_only) vs J-drop(x_full).
# Same SAM2 eval, same fidelity check, paired per clip.
```
- **Acceptance pre-registered**: A1-full mean paired lift over A1-only ≥ +0.05; ≥6/10 strict wins; positive median.
- **Reasoning**: Both arms share W*, ν, decoy seeds, and SAM2 eval. The ONLY difference is bridge variables = 0 vs Stage-14-optimized. Reviewers cannot blame placement search or insert ν optimization for the lift.

### Change 2 — Pre-register A3 hook behavior (IMPORTANT)

- **R3 reviewer**: pre-register exactly what `BlockInsertMemoryWritesHook` modifies, so reviewers can't claim intervention changed more than memory writes.
- **Action**: Hook specification:
```python
class BlockInsertMemoryWritesHook:
    """Nulls memory-bank writes from frames at insert positions W*.
    
    SAM2.1's `predictor.process_frame(t, ...)` calls 
    `memory_attention.encode_memory(t, features_t, mask_t)` 
    which appends a per-frame memory entry to the bank.
    
    THIS HOOK: at frames t ∈ W_attacked (the insert positions in attacked-space),
    skip the encode_memory call. The memory bank does NOT receive an entry from t.
    
    Subsequent frames t' > t still query the bank as usual; the bank just lacks 
    the would-be-decoy entry. The current-frame Hiera forward at t still runs 
    normally (so the inserted frame is still SEEN by SAM2; only its memory 
    contribution is suppressed).
    
    Everything else unchanged: 
      - Hiera encoder forward (yes)
      - mask decoder forward (yes)  
      - cross-attention to existing bank entries (yes)
      - bank entries from previous t' < t (yes, untouched)
    """
```
Pre-registered. Implementation = ~50 LOC patch on `SAM2VideoAdapter`. Reviewers cannot accuse the intervention of modifying more than the targeted memory-write step.
- **Reasoning**: Hook scope is now PR-quality precise. The intervention removes EXACTLY the inserted-frame memory contribution — and nothing else.

### Change 3 — Conditional paper framing pre-committed (IMPORTANT)

- **R3 reviewer**: title/abstract/claim language explicitly conditional on A3 outcome.
- **Action**: Pre-register both framings:

**Framing-A (Strong pass: A3 ≥ 0.20 abs collapse on ≥7/10)**
```
TITLE: Memory-Mediated Failure of Prompt-Driven Video Segmentation: 
       Causal Evidence from Internal Decoy Insertion

ABSTRACT thesis sentence:
"We show that SAM2's prompt-conditioned temporal memory propagation is the 
DOMINANT failure mode under publisher-side adversarial attack, providing 
causal evidence via memory-write blocking that internal decoy insertion 
attacks lose ≥0.20 absolute J-drop when the inserted frames cannot enter 
the memory bank."
```

**Framing-B (Partial pass: 0.10 ≤ collapse < 0.20 OR <7/10 clips)**
```
TITLE: Memory-Mediated Persistence in Adversarial Attacks on Prompt-Driven 
       Video Segmentation

ABSTRACT thesis sentence:
"We provide evidence that SAM2's prompt-conditioned temporal memory 
propagation is a SUBSTANTIAL component of failure mode under publisher-
side adversarial attack, with internal decoy insertion attacks losing 
≥0.10 absolute J-drop when inserted-frame memory writes are blocked, 
while bridge-frame perturbations measurably extend the memory divergence 
beyond the insert-only baseline."
```

**Framing-C (Fail: <0.10 abs collapse on majority)**
```
DECISION: Pre-registered workshop pivot. The paper retires the memory-
mediated mechanism claim; resubmit as workshop paper "Engineered Insertion 
+ Sparse Perturbation Attack on SAM2" with empirical effectiveness as the 
contribution and mechanism analysis explicitly inconclusive.
```

- **Reasoning**: Reviewers see honest pre-commitment. Result-driven framing is built in.

### Change 4 — Main vs Deployment results separation (MINOR)

- **R3 reviewer**: raw-joint = main table; wrapper-selected = separate column.
- **Action**: Results structure pre-registered:

**Table 1 (MAIN — RAW JOINT v5 vs A0, paired)**:
| Clip | A0 J-drop | RAW v5 J-drop | Lift | Win? | polish_applied? |
|---|---|---|---|---|---|

Headline gates apply to RAW v5 column ONLY. polish_applied column documents whether bridge δ was kept by the optimizer's internal best-step selection (this is internal Stage 14 best-of-30; NOT the export-time wrapper).

**Table 2 (DEPLOYMENT — wrapper-selected, separate)**:
| Clip | RAW v5 J-drop | A0 J-drop | Wrapper publishes | Final J-drop |
|---|---|---|---|---|

Wrapper-selected = max(RAW v5, A0) per clip. Reported for deployment readers; NOT used for headline gates.

### Change 5 — E1 framed as openly necessary engineering (MINOR)

- **R3 reviewer**: Don't hide E1; don't compete with C1; openly label as enabling search procedure.
- **Action**: §Discussion E1 paragraph slightly tweaked. Removes "this is itself a finding" elevation; kept as "engineering necessity given empirically validated failure of vulnerability heuristics on memory-bank VOS."
- New text:
> "Why search, not heuristic: We rely on a joint curriculum search for placement because vulnerability heuristics adapted from per-frame fragility (a 3-signal scorer combining confidence drop, mask discontinuity, and Hiera discontinuity) were empirically falsified on a 10-clip ranked-vs-random comparison (mean J-drop 0.488 ranked vs 0.534 random — anti-correlated). The search is necessary engineering for this attack surface; we do not claim it as a primary contribution."

### Change 6 — Compress traj+α+warp+R in main paper (MINOR)

- **R3 reviewer**: compress to "masked bridge-edit parameterization"; full math in Appendix B.
- **Action**: Main paper §Method now says:
> "Bridge δ is parameterized as a **masked bridge-edit composite**: a learned alpha-paste of a duplicate-object decoy onto each bridge frame, with a small spatial warp and a masked residual confined to the decoy support. Full equations in Appendix B."

Appendix B retains the (traj.anchor_offset, traj.delta_offset, alpha_logits, warp_s, warp_r, R[k, l]) parameterization details.

### Change 7 — T_obj sensitivity appendix (MINOR modernization)

- **R3 reviewer**: appendix sensitivity on T_obj (16/32/64 tokens).
- **Action**: Add appendix experiment running A3 + d_mem trace with T_obj = 16, 32, 64. Same hardware, ~1 GPU-h additional.
- **Reasoning**: Strengthens d_mem diagnostic robustness without main-story bloat.

---

## Revised Proposal (full, R3-locked)

### Title (CONDITIONAL)

- **Strong pass**: *Memory-Mediated Failure of Prompt-Driven Video Segmentation: Causal Evidence from Internal Decoy Insertion*
- **Partial pass**: *Memory-Mediated Persistence in Adversarial Attacks on Prompt-Driven Video Segmentation*
- **Fail**: workshop pivot

### Method Thesis (CONDITIONAL)

(Strong pass) — "memory propagation is the DOMINANT failure mode; causal evidence ≥0.20 collapse."
(Partial pass) — "memory propagation is a SUBSTANTIAL component; ≥0.10 collapse + bridge δ extends d_mem."

### Contribution Focus (R2-locked, R3-conditional language)

- **C1** (main, mechanism, paired):
  - C1.a: J-drop COLLAPSES (≥0.20 strong / ≥0.10 partial pre-registered) under memory-write blocking
  - C1.b: bridge δ extends d_mem(t) above insert-only on ≥75% clips
- **E1** (enabling search, openly necessary engineering — heuristic anti-correlated)
- **E2** (enabling no-regression L_keep_full)
- **Deployment policy** (separately reported wrapper)
- **Non-contributions**: same as R2

### Complexity Budget (≤2 trainable, R2-locked)

- Frozen: SAM2.1-Tiny, SAM2VideoAdapter, LPIPS, STE, A0, decoy alpha-paste compositor.
- New trainable (2): δ on bridge frames; ν on inserts.
- Diagnostic-only (no training signal): `BlockInsertMemoryWritesHook` (R3 hook spec); memory-readout extractor (R2 protocol).
- Non-contribution components: joint curriculum placement search (E1); dense L_keep_full (E2); export wrapper (deployment).

### Core Mechanism (mathematically locked, R2 d_mem)

(decoy family, optimized variables, loss, d_mem protocol — ALL UNCHANGED from R2)

### Validation (R2 + R3 operational locks)

#### C1.a — memory causality
- **Experiment**: 10-clip held-out + `BlockInsertMemoryWritesHook` (R3 spec).
- **Strong / Partial / Fail** (R2 pre-registered).

#### C1.b — persistence extension
- **Experiment**: per-clip d_mem(t) trace per R2 protocol.
- **Acceptance**: integral of (d_mem_full − d_mem_only) over (w_K, w_K + L) > 0 on ≥7/10.
- **Sensitivity (R3 appendix)**: T_obj ∈ {16, 32, 64}.

#### C2 — RAW joint headline (R3 locked)
- **MAIN TABLE = RAW JOINT v5 vs A0 paired**
- Wrapper-selected = SEPARATE deployment table
- Headline gates apply to RAW JOINT only:
  - ≥5/10 strict wins
  - mean ≥+0.05
  - median > 0
  - top contributor < 40%
  - mean joint J-drop ≥ 0.55

#### Ablations (R3 locked)

| # | Ablation | R3 lock | Acceptance |
|---|---|---|---|
| **A1** | Bridge δ contribution, isolated | Same upstream W*, ν, decoy_seeds; ONLY bridge variables (traj/α/warp/R) zeroed in control arm | ≥+0.05 mean lift, ≥6/10 strict, positive median |
| **A2** | Placement matters | Random K=3 vs joint search, both with full v5 polish | Search > random by ≥+0.10 mean lift |
| **A3** | Memory-causality | Full v5 vs full v5 + R3-spec'd hook | Strong (≥0.20/7-clip) / Partial (≥0.10/6-clip) / Fail |

### Failure Modes & Decisions (R2 + R3)

| Failure | Detection | Pre-registered decision |
|---|---|---|
| Bridge δ regress per clip | RAW joint < A0 | Report honestly |
| A3 strong | ≥0.20/7-clip | Framing-A |
| A3 partial | ≥0.10/6-clip OR <7/10 | Framing-B |
| A3 fail | <0.10/majority | Framing-C (workshop pivot) |

### Compute & Timeline (R2 ordering)

| Day | Task | Compute | Decision gate |
|---|---|---|---|
| Day 1 AM | Implement R3-spec hook (~2h) + extractor (~1h) | minimal | code review |
| Day 1 PM | A3 + d_mem trace 10 clips | 7 GPU-h | **A3 verdict gates framing** |
| Day 2 | C2 RAW joint v5 + A0 paired + A1 (paired) overnight | 8 GPU-h | Headline gates |
| Day 3 AM | A2 random + T_obj sensitivity | 5+1 GPU-h | Ablation table |
| Day 3 PM | Writeup with conditional framing | 4 author-h | Submit |
| **Total** | | **~21 GPU-h** | **3 days** |

### Discussion (R3 E1 ownership re-tweaked)

> "Why search, not heuristic: We rely on a joint curriculum search for placement because vulnerability heuristics adapted from per-frame fragility (3-signal scorer: confidence drop, mask discontinuity, Hiera discontinuity) were empirically falsified on 10-clip ranked-vs-random (mean J-drop 0.488 vs 0.534 — anti-correlated). The search is necessary engineering for this attack surface; we do not claim it as a primary contribution."

### Experiment Handoff Inputs

- **Must-prove claims**: C1.a (strong / partial / fail tier), C1.b, C2 RAW joint headline gates.
- **Must-run ablations**: A1 (R3-locked), A2, A3 (FIRST, R3-locked hook).
- **Must-do appendices**: T_obj 16/32/64 sensitivity.
- **Critical datasets / metrics**: DAVIS-2017 10 held-out, J-drop on uint8 export, d_mem(t) per R2 protocol.
- **Highest-risk assumptions**: A3 collapse magnitude (pre-registered narrowing); RAW joint headline gates; hook implementation modularity.
