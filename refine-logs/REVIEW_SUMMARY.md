# Review Summary

**Problem**: Adversarial attack on SAM2 video segmentation (publisher-side, white-box, per-clip targeted) using BOTH internal frame insertion AND sparse δ on adjacent original (bridge) frames.
**Initial Approach**: Memory-hijack insertion + bridge δ + adaptive wrapper (v4.1 retest evidence).
**Date**: 2026-04-27
**Rounds**: 4 / 5
**Final Score**: **8.4 / 10** (proposal-stage ceiling)
**Final Verdict**: **REVISE (CEILING)** — architecture at natural max; READY blocked only by unrun A3.

## Problem Anchor

(See `PROBLEM_ANCHOR_2026-04-27.md`)

Bottom-line: publisher-side SAM2 attack via internal insertion + sparse bridge δ on memory-bank VOS, must address Chen WACV 2021 / UAP-SAM2 NeurIPS 2025 / Li T-CSVT 2023 / PATA 2310.10010 prior arts, must keep BOTH insert + bridge δ (CLAUDE.md hard rule + this round user constraint), AAAI venue.

## Round-by-Round Resolution Log

| Round | Score | Main reviewer concerns | What this round simplified / modernized | Solved? | Remaining risk |
|---|---|---|---|---|---|
| 1 | 6.0 | Wrapper drift; demoted contributions dishonest; A3 confounded; "first demonstration" overclaim | — (initial proposal) | — | many |
| 2 | 7.2 | A1 confounded; d_mem circular; A3 0.20/7-clip too aggressive; placement ownership | Reframed C1 + E1/E2; raw joint = science; A3 dual-threshold; d_mem token set from clean fix; novelty narrowed; memory-causality replaces "all-frames-δ" | mostly | A1 still confounded; T_obj arbitrariness |
| 3 | 7.8 | A1 still bundles upstream search/ν; d_mem layer/projection unspecified; conditional framing not pre-committed | A1 operationally locked (same upstream W*, ν, decoy_seeds); d_mem pre-registered (last cross-attention block, pre-projection V, top-32 frozen tokens); A3-first sequencing; conditional Framing A/B/C pre-registered | yes | venue readiness contingent on A3 |
| 4 | **8.4** | Empirical contingency only — no architectural blocker | Added negative-control hook A3-control (matched non-insert frames blocking); froze A1+A3 implementations; ceiling declared | partial | A3 not yet run |

## Overall Evolution

- **R1 → R2**: massive reframe — wrapper demoted from contribution to deployment policy; raw joint became scientific method; A3 replaced with memory-causality ablation. +1.2 score.
- **R2 → R3**: operational locking — A1 isolation tightened, d_mem protocol pre-registered, A3 dual-threshold pre-committed, conditional framings pre-written. +0.6.
- **R3 → R4**: final structural upgrade — negative control for A3 (matched non-insert frame blocking); insert-position-specific collapse claim. +0.6. CEILING.

## Final Status

- **Anchor**: preserved (raw joint = scientific method; wrapper = deployment).
- **Focus**: tight (1 main C1 + 2 enabling E1/E2 + 1 deployment).
- **Modernity**: appropriate (SAM2 = foundation-era target; memory-readout instrumentation as causal diagnostic; no LLM/VLM/Diffusion/RL bolt-ons).
- **Strongest parts**: causal A3 ablation with negative control; pre-registered Framing A/B/C; complete d_mem protocol; honest E1 ownership.
- **Remaining weaknesses**: AAAI-level novelty contingent on A3 strong-pass; raw joint headline gates not yet validated on held-out 10 clips; bmx-trees-like clips still revert occasionally (needs lambda_keep_full=50 tuning ablation in supplementary).

## Pattern Note

Score plateau at 8.4 matches the 2026-04-23 v4-vadi run (different reframe, same ceiling). Structural feature: when implementation already exists + paper claim plausible + validation experiments unrun → proposal-stage scores asymptote ~8.4. The 0.6 gap to READY=9 is data-bound, not architecture-bound.

Codex explicit guidance: **"STOP proposal iteration after this. The next acceptance-lift comes from data, not wording."**
