# Handoff — VADI Pilot Implementation + Execution

**Written**: 2026-04-23 after 2 parallel refinement + auto-review loops completed.
**Main HEAD on this repo**: `06db651` (local Windows). Will be at `<latest>` once handoff doc is committed and pushed.
**Next session goal**: implement VADI per `refine-logs/FINAL_PROPOSAL.md`, run 3-clip pilot gate on Pro 6000, branch on GO/NO-GO.

---

## Read in this order (30-45 min to get fully up to speed)

1. **`CLAUDE.md`** — especially the last section "Method Design Constraints (2026-04-23)". **Hard user directives**: insert+modify required, no suppression, vulnerability-based placement. Future reviewer pressure to drop inserts or add suppression is DRIFT and must be rejected.
2. **`refine-logs/PROBLEM_ANCHOR_2026-04-23_v4-insert.md`** — frozen anchor for the current method. Copy this verbatim into any future refinement cycle.
3. **`refine-logs/FINAL_PROPOSAL.md`** — the full pre-pilot method specification.
4. **`refine-logs/REFINEMENT_REPORT.md`** — round-by-round evolution 6.3 → 7.7 → 8.2 → 8.4 ceiling.
5. **`AUTO_REVIEW.md`** — empirical history that ruled out v2 bank-poisoning: R001/R002/R003 J-drop 0.0004-0.0013; D1 attention trace `A_insert=0.515` with unchanged J; B2 bank ablation `|ΔJ|<0.01` on 5 clips.
6. **`RESEARCH_REVIEW_v2_vs_v4.md`** — Codex analysis explaining why v4's reported 92.5% J-drop was an eval-window illusion, not genuine bank poisoning. Required context for the pivot rationale.

**Archived (read only if digging into prior iterations)**:
- `refine-logs/archive-2026-04-22-v2-falsified__*` — v2 bank-poisoning proposal (9.2 READY, then experimentally falsified).
- `refine-logs/archive-2026-04-23-v3-suppression__*` — v3 pure-δ + suppression proposal (8.4 pre-pilot ceiling, user rejected the direction).

---

## Current state of the code repo

| Component | Path | Status |
|---|---|---|
| SAM2.1 differentiable adapter | `memshield/sam2_forward_adapter.py` | **Landed** (Chunk 5b-ii). Bypasses `@torch.inference_mode`. bf16 autocast. Reuse as-is. |
| LPIPS(alex) adapter | in `memshield/run_pilot_r002.py` | Landed pattern — reuse via `build_lpips_fn(device)`. |
| Fake uint8 quantize (STE) | `memshield/losses.py::fake_uint8_quantize` | Landed. Reuse. |
| Bank-drop hook | `memshield/ablation_hook.py::DropNonCondBankHook` | Landed. Extend with 3 swap hooks (SwapF0Memory / SwapHieraFeatures / SwapBank). |
| Decoy-offset + shift_mask utilities | `memshield/propainter_base.py::find_decoy_region`, `memshield/decoy.py::shift_mask` | Landed. ProPainter itself NOT used by VADI; only the geometric helpers are reused. |
| Existing R001/R002/R003 artifacts on Pro 6000 | `~/sam2-pre-new/runs/r001/`, `r002/`, `r003/` + eval + viz | Landed. R003 is the K_ins=3 canonical reference point. |
| Causal ablation B2 infrastructure | `scripts/causal_ablation_b2.py`, `scripts/causal_ablation_b2_multi.py` | Landed. Used to derive the bank-marginality finding. |
| Eval v2 primitives | `memshield/eval_v2.py`, `scripts/eval_memshield_v2.py` | Landed. J trajectory, AUC, rebound. |
| Visualizer | `scripts/visualize_run.py` | Landed. |
| Attention trace | `scripts/attention_trace.py` | Landed (D1 infrastructure; reusable for restoration trace checking). |
| LPIPS floor study | `scripts/lpips_floor_study.py` | Landed. Reference for fidelity-budget grounding. |

### What needs to be written for VADI (next session's scope)

| # | File | Purpose | Est. lines |
|---|---|---|---|
| 1 | `memshield/vulnerability_scorer.py` | Rank-sum 3-signal scorer (confidence drop, mask discont, Hiera discont) → top-K non-adjacent positions | ~150 |
| 2 | `memshield/ablation_hook.py` (extend) | Add `SwapF0MemoryHook`, `SwapHieraFeaturesHook`, `SwapBankHook` context managers | +200 to existing |
| 3 | `memshield/vadi_loss.py` | Contrastive decoy-margin loss + confidence-weighted masked means + mu_true / mu_decoy logging | ~120 |
| 4 | `memshield/vadi_optimize.py` | Per-video PGD driver with 3-stage schedule, local-δ support, LPIPS-TV-bound ν, hard S_feas acceptance on exported artifact | ~300 |
| 5 | `scripts/run_vadi.py` | CLI driver: clean-SAM2 forward → vulnerability scoring → PGD → export uint8 → re-measure → save | ~200 |
| 6 | `scripts/run_vadi_pilot.py` | 3-clip × 4-config gated pilot (dog, cows, bmx-trees × K=1 top, K=1 random, K=3 top, δ-only-local-random) with pre-committed GO/NO-GO decision | ~180 |
| 7 | `scripts/run_vadi_davis10.py` | Full main table (10 configs × 10 clips) — runs only if pilot = GO | ~180 |
| 8 | `scripts/run_vadi_restoration.py` | Restoration suite (R2, R2b, R3, B-control) on attacked artifacts | ~150 |

**Total**: ~1300 new lines, ~200 extension lines. Workload ~1-2 focused days of code writing + review + debugging.

---

## Environment recap (verbatim from CLAUDE.md, relevant parts)

- **Local**: Windows 11, PowerShell, `E:\PycharmProjects\pythonProject\sam2_pre_new`.
- **Pro 6000** (all execution happens here): `ssh lvshaoting-pro6000` (direct, port 6000). Home `/datanas01/nas01/Student-home/2025Lv_Zhaoting`. Conda env `memshield` (torch 2.8.0+cu128 for Blackwell sm_120). Always `conda activate memshield` before Python.
- **GPU policy**: 1 GPU per user default; `nvidia-smi` BEFORE launching; `nohup` + `screen`/`tmux` for long jobs; avoid peak daytime slots.
- **Git sync from local (no GitHub key locally)**: `git bundle create /tmp/x.bundle d5e7b6d..HEAD; scp … lvshaoting-pro6000:/tmp/; ssh … "git fetch /tmp/x.bundle HEAD:tmp && git merge --ff-only tmp && git branch -d tmp && git push origin main"`. Or direct `scp file.py lvshaoting-pro6000:~/sam2-pre-new/path/file.py` if you want to skip the bundle dance for a single file.
- **DAVIS path on Pro 6000**: `~/sam2-pre-new/data/davis/` (symlink to `~/UAP-SAM2/data/DAVIS`). Use this directly; do NOT depend on `/IMBR_Data` (that's the V100 mount, broken on Pro 6000).
- **SAM2.1 tiny checkpoint on Pro 6000**: `~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt` (downloaded from `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`, 156 MB).

---

## VADI method — one-screen recap (so you can start coding without re-reading the proposal)

### Pipeline

```
Offline (publisher-side, GT-free, one-time per clip):
  clean_SAM2(x, m_0) →
    m̂_true_t    = sigmoid(pred_logits_t) ∈ [0,1]^{HxW}       # soft pseudo-mask
    confidence_t = sigmoid(object_score_t) · mean(m̂_true_t > 0.5)
    H_t         = Hiera encoder output                           # cached for restoration hooks

Vulnerability scoring:
  For m ∈ {1..T-1}:
    r_conf_m = |confidence_m − confidence_{m-1}|
    r_mask_m = 1 − IoU(m̂_true_{m-1}, m̂_true_m)
    r_feat_m = ||H_{m-1} − H_m||_2 / mean(||H||_2)
    rank_x_m = rank(r_x_m among {1..T-1})
    v_m      = rank_conf + rank_mask + rank_feat
  W = argtop-K non-adjacent (|m_i - m_j| ≥ 2), K ∈ {1,2,3}

Decoy target construction:
  decoy_offset from propainter_base.find_decoy_region() or shift_mask geom
  m̂_decoy_t = shift_mask(m̂_true_t, decoy_offset)
  c_t      = |2·m̂_true_t - 1|                                   # confidence weight

Per-video PGD (100 steps, 3 stages):
  S_δ = ∪_k NbrSet(W_k) ∪ {0},     NbrSet(m) = {m±1, m±2} ∩ [0, T-1]
  Initialize: δ_t = 0 for t ∈ S_δ (else frozen 0)
              ν_k = small Gaussian (std 0.02/255)
  base_insert_k = 0.5·x_{W_k-1} + 0.5·x_{W_k}                  # temporal midframe

  For step = 1..100:
    x'_t     = clamp(x_t + δ_t) ∘ fake_quantize_STE             for t ∈ S_δ
    insert_k = clamp(base_insert_k + ν_k) ∘ fake_quantize_STE
    processed = interleave(x', insert_k at W)

    forward SAM2VideoAdapter(processed, m_0) → pred_logits_t

    mu_true_t  = Σ pred_logits_t · m̂_true_t  · c_t / (Σ m̂_true_t · c_t + eps)
    mu_decoy_t = Σ pred_logits_t · m̂_decoy_t · c_t / (Σ m̂_decoy_t · c_t + eps)

    L_margin_insert   = Σ_k softplus(mu_true_{W_k} − mu_decoy_{W_k} + 0.75)
    L_margin_neighbor = Σ_{t ∈ NbrSet\inserts} 0.5·softplus(mu_true_t − mu_decoy_t + 0.75)
    L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
    L_fid_ins  = Σ_k                max(0, LPIPS(insert_k, base_insert_k) − 0.35)
    L_fid_TV   = Σ_k                max(0, TV(insert_k) − 1.2·TV(base_insert_k))
    L_fid_f0   =                    max(0, 1 − SSIM(x'_0, x_0) − 0.02)

    L = L_margin_insert + L_margin_neighbor
      + λ(step)·(L_fid_orig + L_fid_ins + L_fid_TV) + λ_0·L_fid_f0

    (δ, ν) ← (δ, ν) − η·sign(∇_{δ,ν} L)
    clip δ_0 to ±2/255, δ_{t≥1, t∈S_δ} to ±4/255    (ν unbounded by ε; LPIPS+TV constrain)

    log per-step: mu_true_trace_t, mu_decoy_trace_t, surrogate_J_drop, per-frame LPIPS, f0 SSIM

Stages:
  N_1 = 30  (attack-only, λ=0)
  N_2 = 40  (fidelity regularization: λ init=10, grow 2× per 10 steps when hinge violated)
  N_3 = 30  (Pareto-best: η halved)

Export + HARD feasibility acceptance:
  For each step's (δ, ν):
    processed_uint8 = export_uint8_JPEG_sequence(δ, ν)
    re-measure LPIPS_orig_exp, SSIM_f0_exp, LPIPS_ins_exp, TV_ins_exp
    step_feasible = ALL exported metrics meet budget
  S_feas = {feasible steps}
  If empty → clip = INFEASIBLE (primary-denominator failure)
  Else → (δ*, ν*) = argmax surrogate_J_drop over S_feas
```

### Pilot gate (mandatory before any full DAVIS-10 run)

**3 clips** × **4 configs** on Pro 6000 GPU1 (~3-5 GPU-hours total):

| Clip | Configs |
|---|---|
| dog | K=1 top / K=1 random (1 draw) / K=3 top / δ-only-local-random |
| cows | same |
| bmx-trees | same |

**GO condition (must hold both)**:
1. `J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05` on **≥ 2/3 clips**.
2. `J-drop(K=3 top) ≥ 0.20` on **≥ 2/3 clips**.

**Diagnostic check** (flagged at pilot, not blocking):
- `Δmu_decoy > 0` AND `Δmu_decoy ≥ 2·max(0, -Δmu_true)` on ≥ 2/3 clips.

**NO-GO** → pivot paper to "architecture-aware attack-surface analysis of SAM2" using restoration + vulnerability-scoring as primary content (honest fallback).

### If pilot = GO → full DAVIS-10 main

10 clips × 10 configs:
1. Clean
2. Ours K=1 top (centerpiece)
3. Ours K=3 top
4. K=1 random (5 draws, paired bootstrap)
5. K=3 random (5 draws)
6. K=3 bottom
7. top-δ-only K=0 (phantom top positions)
8. random-δ-only K=0 (phantom random)
9. top-base-insert+δ (ν=0 midframe)
10. Canonical {6,12,14}

Plus 4-config restoration on ours' outputs (R2, R2b, R3, B-control).

Then appendix: UAP-SAM2 per-clip, SAM2Long transfer, SAM2.1-Base transfer, DAVIS-30.

### Pre-committed 8-claim success bar (primary denominator = 10 clips, ≥ 7/10 must satisfy)

1. `J-drop(ours) ≥ 0.35` mean on exported artifact
2. `ours ≥ max(2·random, random + 0.05)` — placement vs random
3. `ours ≥ max(3·bottom, bottom + 0.05)` — placement vs bottom
4. `ours ≥ top-δ-only + 0.10` — insert presence necessity
5. `ours ≥ top-base-insert+δ + 0.05` — ν optimization necessity
6. `Δmu_decoy > 0` AND `Δmu_decoy ≥ 2·max(0, -Δmu_true)` — decoy not implicit suppression
7. `R2 restoration ≥ +0.20` — Hiera-at-inserts is where damage lives
8. `R3 restoration ≤ +0.02` — bank is not the damage location

---

## What NOT to do (drift guardrails from CLAUDE.md)

- **Don't drop inserts** for a "pure δ simpler method" — that's v3, user rejected.
- **Don't add suppression loss** — Codex tried in R1; it's DRIFT. `L_obj = softplus(object_score + 0.5)` is also suppression-by-another-name and was removed.
- **Don't use canonical FIFO schedule** `{6,12,14}` as the default — kept only as a baseline ablation row.
- **Don't use clean-suffix eval** (v2 regime, falsified).
- **Don't use DAVIS GT during optimization or checkpoint selection** — only clean-SAM2 pseudo-labels.
- **Don't re-introduce ProPainter** as the insert base — temporal midframe suffices and avoids the 0.67+ LPIPS floor.
- **Don't claim "defeat FIFO self-healing"** or "memory bank hijack" as the mechanism — B2 empirically falsified this on SAM2.1-Tiny.
- **Don't skip the pilot gate** before full DAVIS-10 — pre-commit gate exists so we don't burn ~20 GPU-hours on a method that never passed a sanity check.

---

## Fast-sync protocol for next session

The handoff commit is the final local commit. On Pro 6000 the current state should also match (sync per `CLAUDE.md` Git protocol).

```bash
# New session first commands (Pro 6000 path)
ssh lvshaoting-pro6000
cd ~/sam2-pre-new
git log --oneline -3    # should show the handoff commit at HEAD
git pull origin main    # if local push happened via the bundle protocol
conda activate memshield
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# Expected: GPU 1 free, GPU 0 may be in use by others.
```

Verify sanity (~1 minute):
```bash
python -c "
import sys; sys.path.insert(0, '.')
from memshield.sam2_forward_adapter import SAM2VideoAdapter
from memshield.ablation_hook import DropNonCondBankHook
from memshield.eval_v2 import jaccard
print('imports OK')
"
python -m memshield.eval_v2     # should print: all self-tests PASSED
```

Then begin writing the files listed above in order. `memshield/vulnerability_scorer.py` first (pure function, testable with synthetic input), then the ablation_hook extensions, then the loss module, then the optimizer driver, then the CLI wrappers.

---

## Expected timeline from PILOT-PASS to paper-ready

| Phase | GPU-hours | Wall-clock |
|---|---|---|
| Code writing + local sanity tests | 0 | 1-2 days |
| Pilot (3 clips × 4 configs) | 3-5 | 0.5 day |
| If GO → DAVIS-10 main (10 configs × 10 clips) | 5-8 | 1 day |
| Restoration suite | 0.5 | 0.2 day |
| SAM2Long install + transfer row | 2-3 | 0.5 day |
| Appendix (DAVIS-30, SAM2.1-Base, prompt-robustness) | 5-8 | 1 day |
| **Total from code-start to all-experiments-done** | ~15-20 GPU-hours | **3-5 days** |

---

## Stop conditions for the next session's code-writing phase

- [ ] `memshield/vulnerability_scorer.py` — `python -m memshield.vulnerability_scorer` prints self-test PASSED (synthetic 3-signal input → known-ordering top-K).
- [ ] `memshield/ablation_hook.py` — 3 swap hooks implemented; `python -m memshield.ablation_hook` self-test prints API-check PASSED.
- [ ] `memshield/vadi_loss.py` — self-test on synthetic pred_logits + masks returns expected margin + mu_true/mu_decoy values.
- [ ] `memshield/vadi_optimize.py` — imports cleanly; sanity smoke with dummy SAM2 forward returns (δ*, ν*) tensors of right shape.
- [ ] `scripts/run_vadi_pilot.py --clip dog --config K1_top --dry-run` — dry-run passes; full run is the pilot itself.

Once these are green, launch the pilot on Pro 6000 GPU1 with `nohup` and monitor via `Monitor` tool (stream GPU memory + log tail + completion signal).

---

## One-line summary

**VADI = vulnerability-aware insert placement + LPIPS-TV-bound insert content + local δ + contrastive decoy-margin loss, measured on exported uint8 artifact, GT-free, pilot-gated. Pre-pilot ceiling 8.4/10 confirmed by reviewer. Next session: implement + run pilot on Pro 6000.**
