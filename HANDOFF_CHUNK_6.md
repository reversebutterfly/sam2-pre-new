# Handoff: Chunk 6 — eval metrics + R002 downstream

**Written**: 2026-04-22 end of 5b-ii session
**Previous commits**: `6d5baab` (5b-ii adapter + probe fix), `10ad2db` (R001 pilot + auto-prep)
**Git HEAD on main**: `10ad2db` — local Windows, Pro 6000, GitHub all synced.

---

## How to resume

```bash
# Pro 6000, first command:
cat HANDOFF_CHUNK_6.md
python -m memshield.scheduler
python -m memshield.losses_v2
python -m memshield.optimize_v2
python -m memshield.sam2_forward_v2
python -m memshield.sam2_forward_adapter
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_5b_ii.py  # ~90 s
```

If all five module tests + smoke pass, the 5b-ii scaffolding is healthy. The R001 artifacts from the prior session live at `runs/r001/` (modified_video.npy, final_nu.npy, final_delta.npy, diagnostics.json) — Chunk 6 mostly reads these.

---

## What's landed (end of 5b-ii session)

| Chunk | File | Role |
|---|---|---|
| 1 | `memshield/mem_attn_probe.py` | Observational side-channel for P_u. **Bug fixed this session**: patched_forward kwargs must be `q/k/v` not `q_in/k_in/v_in`. |
| 2 | `memshield/scheduler.py` | Three-clock schedule, write-aligned seed+boundary. |
| 3 | `memshield/propainter_base.py` | ProPainter dispatcher + real forward. Installed at `~/ProPainter`. |
| 4 | `memshield/losses_v2.py` | L_loss / L_rec / L_stale / L_fid. |
| 5a | `memshield/optimize_v2.py` | 3-stage PGD loop, model-agnostic. |
| 5b-i | `memshield/sam2_forward_v2.py` | RuntimeProvenanceHook — monkey-patches SAM2Base._prepare_memory_conditioned_features. |
| **5b-ii** | `memshield/sam2_forward_adapter.py` | **SAM2VideoAdapter** — full Sam2ForwardFn implementation; bypasses inference_mode decorators; bf16 autocast; suffix Hiera cache. **Codex R8: 8.5/10 READY**. |
| — | `scripts/smoke_5b_ii.py` | 15-frame dog smoke; grad flow + P_u invariants + 10-step PGD. |
| — | `memshield/run_pilot_r001.py` | **R001 sanity driver** — full pipeline end-to-end, 50 steps, no NaN. |

---

## Chunk 6 scope

Evaluation metrics for the paper. Three deliverables, all thin:

### 6.1 — `scripts/eval_clips_sam2.py` extensions

Current file is the legacy UAPSAM evaluator. Extend (don't replace) to add MemoryShield v2 metrics:

- **J trajectory per eval frame**: for each u ∈ U, compute Jaccard(pred_u, gt_u). Returns a [|U|] vector.
- **post-loss AUC**: normalized area under the J-trajectory curve on the eval window. AUC = mean(J_u) with bounds [0, 1].
- **rebound@k**: given the attack-time J dip, measure how fast J recovers. Define rebound@k = first u in eval window where J_u ≥ max(J_first_eval, threshold). Reports −1 if J never recovers.

Input contract: a directory of `runs/r001/` style artifacts + clean DAVIS frames + GT masks. Output: JSON with per-clip metrics.

### 6.2 — Modified-video sanity visualizer

Small script that takes `runs/<run_id>/modified_video.npy` and dumps jpgs + an optional mp4 so you can eyeball the inserts and δ. The inserts should look visually plausible (ProPainter-quality); δ should be barely visible at `eps_other=4/255`.

### 6.3 — R002 gate run

Per `refine-logs/EXPERIMENT_PLAN.md` R002 = full pipeline with LPIPS ON at budget 0.10. Measures whether the attack actually achieves target J-drop while staying within the fidelity budget. After 6.1 lands, run R002 and record results in `refine-logs/EXPERIMENT_RESULTS.md`.

---

## Known caveats carrying forward

1. **SAM2 pred_masks vs upsample quality.** Adapter currently returns bilinear-upsampled low-res logits (256 → video_res). For eval metrics, apply sigmoid + 0.5 threshold *after* the upsample — don't threshold at low-res first.

2. **Clean SAM2 pass order.** R001 driver's `clean_sam2_forward` uses vanilla `propagate_in_video` (inference_mode) — fine for producing reference masks, but masks are thresholded (sigmoid > 0.5). If Chunk 6 wants the raw logit distribution, make a separate helper that returns `video_res_masks` directly without thresholding.

3. **`bundle.C_u` interpretation.** C_u was proxied from clean-SAM2 predictions, not GT. For honest eval, Chunk 6 should evaluate against the actual GT annotation, not against the clean-prediction proxy.

4. **`L_loss` step-50 spike in R001.** Stage 3 joint oscillation is expected but worth logging. Chunk 6 should plot `L_loss` and `L_rec` trajectories per PGD step for the paper's "optimization dynamics" subfigure.

5. **GPU memory budget.** bf16 + 22 frames @ 1024² uses 33 GB steady-state. Scaling to K_ins=3 and longer eval windows (e.g., |U|=14) needs a budget recheck — could push beyond 60 GB. Start R003+ on cuda:0 only after `nvidia-smi` confirms no other user.

---

## Codex continuity

The original thread `019db350-32bb-74a2-b6dc-d9491bf24025` was unreachable from this session's MCP server. We opened a new thread `019db4aa-fa37-7030-b115-9b0428e33758` (Codex R8, scored 8.5/10). For Chunk 6, either:

- Use `019db4aa-…` if it persists on the next session's MCP server — it has R8 context.
- Or start a fresh thread; 5b-ii is frozen at `10ad2db`, so any new Codex review can just read the committed files.

---

## Stop-conditions for Chunk 6

- [ ] `scripts/eval_clips_sam2.py` writes a JSON with `{clip, J_per_eval_frame, AUC, rebound_at_1}` for R001 artifacts.
- [ ] Modified-video visualization shows plausible inserts + invisible δ at eps=4/255.
- [ ] R002 gate run completes with LPIPS ON, budget 0.10, and results recorded in `refine-logs/EXPERIMENT_RESULTS.md`.
- [ ] Codex review at 8.0+/10 READY on eval code.

Then the implementation phase is done and the remaining work is paper writing (probably via `/paper-writing` skill).
