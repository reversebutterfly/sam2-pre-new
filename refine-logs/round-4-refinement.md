# Round 4 Refinement (FINAL)

## Problem Anchor (verbatim)

[See `PROBLEM_ANCHOR_2026-04-27.md`]

## Anchor Check

- **Original bottleneck**: same. SAM2 attack via insertion + bridge δ; memory propagation = attack surface.
- **R4 changes**: ONE small structural addition (A3-control negative-control hook). No method change.

## Simplicity Check

- **Dominant contribution**: same C1 + E1 + E2.
- **R4 codex verdict**: "proposal architecture reached natural ceiling; further refinement has diminishing returns."
- **R4 NEW addition**: A3-control matched-non-insert-frame negative control hook (codex's only remaining HIGH-VALUE proposal-side upgrade).

## Changes Made (R4 final)

### Change 1 — Add A3-control negative-control hook (HIGH-VALUE)

- **Codex R4**: "If you insist on one last structural upgrade, add one negative-control hook: block memory writes at matched non-insert clean frames, with the same count as W*. If collapse is much smaller there than at attacked insert positions, the causal story becomes more reviewer-proof."

- **Action**: Add **A3-control** as a third configuration in the A3 ablation:
  - **A3-attacked**: full-v5 + `BlockInsertMemoryWritesHook` blocking at W_attacked (3 frames, the inserts themselves)
  - **A3-control**: full-v5 + `BlockInsertMemoryWritesHook` blocking at W_control (3 matched non-insert frames, drawn uniformly at random from non-insert non-bridge attacked-space frames; FROZEN per clip per seed)
  - **A3-baseline**: full-v5 (no blocking)

- **New acceptance** for A3 strong/partial: collapse_attacked − collapse_control ≥ 0.10 absolute on ≥7/10 clips. (I.e., blocking attacked-frame memory writes must hurt MORE than blocking control-frame memory writes.)

- **Updated dual-threshold**:
  - **Strong pass**: collapse_attacked ≥ 0.20 abs AND (collapse_attacked − collapse_control) ≥ 0.10 abs, both on ≥7/10
  - **Partial pass**: collapse_attacked ≥ 0.10 abs AND (collapse_attacked − collapse_control) ≥ 0.05 abs, both on ≥6/10
  - **Fail**: either < threshold on majority

- **Compute**: +2 GPU-h (one extra config × 10 clips). Total budget 21 → 23 GPU-h. Still 3 days.

- **Why this matters**: Without A3-control, a reviewer can argue "blocking ANY memory write at ANY frame might collapse the attack — this isn't insert-frame-specific." With A3-control showing collapse is INSERT-POSITION-specific, the causal mechanism story is much harder to attack.

### Change 2 — Freeze A1 + A3 + A3-control implementations (codex AI #1)

- **Action**: Pre-commit the following implementation specs (NO further changes after this round):

**A1 implementation freeze**:
- Same upstream W* (from joint curriculum search)
- Same upstream nu_init (from A0 polish)
- Same upstream decoy_seeds (from build_decoy_insert_seeds)
- A1-only: `traj.anchor_offset = 0; traj.delta_offset = 0; alpha_logits = -1e9; warp_s = 0; warp_r = 0; R = 0` (zero ALL bridge variables); skip Stage 14 entirely; export with bridge frames untouched.
- A1-full: full Stage 14 polish (commit `da719dc` v4.1).

**A3 hook implementation freeze**:
- Patch on `SAM2VideoAdapter.process_frame(t, ...)`: add `if t in self.W_blocked: return self.predictor.run_hiera_and_decoder_only(...)` — runs Hiera encoder + mask decoder; SKIPS `memory_attention.encode_memory(...)` for those t.
- Other frames untouched. Bank entries from t' < t untouched. Cross-attention to existing bank entries untouched.
- Configurable `W_blocked` via constructor.

**A3-control implementation freeze**:
- Same hook; W_blocked = W_control = 3 frames sampled uniformly at random from {t ∈ [0, T_proc) : t ∉ W_attacked AND t ∉ bridge_frames} with fixed seed=0.
- W_control frozen per (clip, seed).

### Change 3 — Stop proposal iteration after R4 (codex AI #5)

Per codex explicit verdict: "STOP proposal iteration after this. The next acceptance-lift comes from data, not wording."

This is the FINAL refinement. Phase 5 begins next.

---

## Revised Proposal (R4-final)

### Title (CONDITIONAL — pre-registered)

- **Strong pass**: *Memory-Mediated Failure of Prompt-Driven Video Segmentation: Causal Evidence from Internal Decoy Insertion with Sparse Bridge Perturbation*
- **Partial pass**: *Memory-Mediated Persistence in Adversarial Attacks on Prompt-Driven Video Segmentation*
- **Fail**: workshop pivot — *Engineered Insertion + Sparse Perturbation Attack on SAM2*

### Method Thesis (CONDITIONAL on A3 outcome)

(Strong) "memory propagation is DOMINANT failure mode; collapse at insert positions specifically (not at matched control positions)"
(Partial) "memory propagation is SUBSTANTIAL component; insert-position-specific collapse + bridge δ extends d_mem"

### Contribution Focus (R4 final, R3 + R4 negative-control hardening)

- **C1** (main, mechanism, paired):
  - C1.a (causal, insert-position-SPECIFIC): collapse_attacked − collapse_control ≥ 0.10 on ≥7/10 (strong) / ≥0.05 on ≥6/10 (partial). Pre-registered.
  - C1.b (persistence): bridge δ extends d_mem(t) above insert-only on ≥75% clips
- **E1** (enabling search; openly necessary engineering)
- **E2** (enabling no-regression L_keep_full)
- **Deployment policy** (separately reported wrapper, NOT a contribution)

### Complexity Budget (≤2 trainable, R2 locked)

Same as R2/R3.

### Core Mechanism (R2 locked)

Same.

### Validation (R4 final)

#### C1.a — memory causality WITH negative control
- 10-clip + R4 hook spec.
- Three configs per clip: A3-baseline / A3-attacked (block W_attacked) / A3-control (block W_control matched random non-insert).
- Pre-registered:
  - **Strong pass**: collapse_attacked ≥ 0.20 AND (collapse_attacked − collapse_control) ≥ 0.10 on ≥7/10
  - **Partial pass**: collapse_attacked ≥ 0.10 AND (collapse_attacked − collapse_control) ≥ 0.05 on ≥6/10
  - **Fail**: either threshold misses on majority

#### C1.b — persistence (R2 unchanged)
Per-clip d_mem(t) trace; integral (full − only) > 0 on ≥7/10. T_obj 16/32/64 sensitivity in appendix.

#### C2 — RAW joint headline (R3 unchanged)
Table 1 RAW v5 vs A0 paired, headline gates apply only here. Wrapper deployment column separate.

#### Ablations (R4)

| # | Ablation | R4 lock |
|---|---|---|
| **A1** | Bridge δ isolated (R3) | same upstream W*+ν+decoy; A1-only zeros ALL bridge variables; A1-full = full Stage 14 |
| **A2** | Placement matters | random K=3 vs joint search |
| **A3** | Memory-causality with negative control | A3-baseline + A3-attacked + A3-control (matched random non-insert) |

### Compute & Timeline (R4 — 3 days, ~23 GPU-h)

| Day | Task | GPU-h | Gate |
|---|---|---|---|
| 1 AM | impl A3 hook + extractor + control sampler | minimal | code review |
| 1 PM | A3-baseline + A3-attacked + A3-control on 10 clips + d_mem trace | 10 GPU-h | A3 verdict gates framing |
| 2 | C2 RAW joint v5 + A0 paired + A1 (paired) overnight | 8 GPU-h | headline gates |
| 3 AM | A2 random + T_obj sens + reporting | 5 GPU-h | tables |
| 3 PM | writeup with conditional framing | author | submit |
| **Total** | | **~23 GPU-h** | **3 days** |

### Discussion (R3 E1 ownership unchanged)

### Failure Modes & Pre-registered Decisions (R4)

| Failure | Detection | Decision |
|---|---|---|
| Bridge δ regress per clip | RAW < A0 | Report honestly |
| A3 strong pass | collapse_att ≥0.20 AND att−ctrl ≥0.10 on ≥7/10 | Framing-A |
| A3 partial pass | collapse_att ≥0.10 AND att−ctrl ≥0.05 on ≥6/10 | Framing-B |
| A3 control comparable to attacked | att−ctrl < 0.05 on majority | Framing-C workshop pivot (mechanism not specific to inserts) |
| A3 attacked too weak | collapse_att < 0.10 on majority | Framing-C workshop pivot |

---

## Ceiling Declaration

Per codex R4: **"The proposal architecture has reached the natural ceiling for proposal-stage iteration. Further proposal refinement now has diminishing returns. One good A3 run will move this more than another full review round."**

Final proposal-stage score: **8.4/10 REVISE (CEILING)**.

Same pattern as 2026-04-23 v4-vadi run (also ended at 8.4 PRE-PILOT CEILING). This is structural — when implementation exists + paper claim plausible + validation experiments unrun, proposal-stage scores asymptote ~8.4. The remaining 0.6 to READY=9 is data-bound.

**Termination**: Skipping round 5 review. Moving to Phase 5 final reports.
