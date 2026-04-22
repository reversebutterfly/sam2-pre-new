# Handoff: Chunk 5b-ii — full SAM2 video-predictor adapter

**Written**: 2026-04-22 at end of previous session
**Previous session commits**: `7aa0739..b6b54d9` (8 commits for chunks 3b v2 / 4 / 4 v2 / 5a / 5a v2 / 5a v3 / 5b-i / 5b-i v2)
**Git HEAD on main**: `b6b54d9`
**Codex thread (keep for continuity)**: `019db350-32bb-74a2-b6dc-d9491bf24025`

---

## How to resume

Read `refine-logs/FINAL_PROPOSAL.md` (what to build), `refine-logs/EXPERIMENT_PLAN.md` (run order), this doc, and then inspect the five already-landed modules. Everything downstream of Chunk 5b-ii is the R001 sanity pilot.

```bash
# In the new session, first command:
cat HANDOFF_CHUNK_5B_II.md
python -m memshield.scheduler        # should print "all invariants OK"
python -m memshield.losses_v2        # ditto
python -m memshield.optimize_v2      # 3 smoke tests pass
python -m memshield.sam2_forward_v2  # 5 provenance-tag invariants pass
```

If all four print OK, the scaffold is healthy and you can start Chunk 5b-ii immediately.

---

## What's landed

| Chunk | File | Lines | Role |
|---|---|---|---|
| 1 | `memshield/mem_attn_probe.py` | ~400 | Observational side-channel to extract `P_u` from SAM2 memory-attention |
| 2 | `memshield/scheduler.py` (v2 appended) | +400 | Three-clock schedule (Clock O/M/W), write-aligned seed+boundary |
| 3 | `memshield/propainter_base.py` | ~530 | ProPainter dispatcher + real forward. **INSTALLED on Pro 6000 at `~/ProPainter`** |
| 4 | `memshield/losses_v2.py` | ~730 | L_loss / L_rec / L_stale / L_fid + augmented-Lagrangian |
| 5a | `memshield/optimize_v2.py` | ~960 | Per-video PGD loop (3 stages, model-agnostic via `Sam2ForwardFn` protocol) |
| 5b-i | `memshield/sam2_forward_v2.py` | ~440 | `RuntimeProvenanceHook` — monkey-patches `_prepare_memory_conditioned_features` and builds `slot_tag` for `MemAttnProbe` |

Plus `AUTO_REVIEW.md`, `IMPLEMENTATION_NOTES.md`, and all of `refine-logs/` (PROBLEM_ANCHOR, FINAL_PROPOSAL, EXPERIMENT_PLAN, EXPERIMENT_TRACKER, REVIEW_SUMMARY, REFINEMENT_REPORT, 4 rounds of review).

---

## Chunk 5b-ii scope (Codex guidance: Option A, narrowly scoped)

Implement `Sam2ForwardFn` from `memshield/optimize_v2.py` — nothing more. Do NOT drag in `run_two_regimes.py` legacy state.

The protocol is:

```python
def sam2_forward_fn(
    modified_video: Tensor[T_prefix_mod, H, W, 3] in [0,1],
    mode: str,                   # "attack" during PGD, "clean" at setup
    cfg: OptimizeConfig,
    bundle: VideoBundle,
) -> Dict:
    return {
      "insert_logits": List[Tensor[H, W]],           # len == cfg.K_ins
      "eval_logits":   List[Tensor[H, W]],           # len == cfg.eval_window_size
      "P_u_list":      List[Optional[Tensor[3]]],    # len == cfg.stale_window_size
      "pred_masks":    Optional[List[Tensor[H, W]]], # only when mode=="clean"
    }
```

### Required behaviors

1. **Build full video** = modified prefix (from `optimize_v2`) + clean eval suffix (`bundle.frames_orig[cfg.T_prefix_orig : cfg.T_prefix_orig + cfg.eval_window_size]`). The optimizer does not modify the suffix; the adapter appends it internally.

2. **Install the two context managers around the forward pass**, outer to inner:
   ```python
   with RuntimeProvenanceHook(sam2.model, insert_frame_ids=schedule.m_positions,
                              probe=probe, fg_mask_by_frame=..., HW_mem=H_feat*W_feat) as hook:
       with probe:
           # run SAM2 video prediction frame-by-frame
           # collect logits at insert positions (m_positions) and eval positions
           # after propagate, probe.P_u_by_frame is populated for frames in V
   ```

3. **Foreground-query mask** per frame: must be aligned with SAM2's feature-grid resolution (typically 64×64 at 1024-res or 48×48 at 768-res). Convert `bundle.C_u` (image-space) to feature-space by downsampling with the same factor SAM2 uses internally. The probe expects a 1-D bool tensor of length `Nq = H_feat * W_feat`.

4. **Hiera embedding cache** for the clean eval suffix. The suffix pixels don't change across PGD steps, so cache the image-encoder output once at setup (mode=="clean") and reuse. The memory-attention path is what changes — so we re-run memory_attention + mask decoder on the cached embeddings each step, but skip the expensive Hiera forward.

5. **Insert frame ids on Clock W** = `schedule.m_positions` (same as w_positions in 5a/5b-i; M==W under our streaming assumption). Pass that set to `RuntimeProvenanceHook.insert_frame_ids`.

6. **Gradient flow**: the returned `insert_logits` / `eval_logits` / `P_u_list` must be differentiable w.r.t. `modified_video`. This means running memory-attention + mask decoder in grad mode. The Hiera cache can stay detached on the suffix; the prefix Hiera must be in grad mode so δ flows.

### Smoke test (mandatory before shipping 5b-ii)

```bash
# On Pro 6000:
ssh lvshaoting-pro6000
source ~/miniconda3/etc/profile.d/conda.sh && conda activate memshield
cd ~/sam2-pre-new
python scripts/smoke_5b_ii.py  # You will write this.
```

Must verify:
- Forward runs end-to-end on a 15-frame dog clip with K_ins=1 at 480p, no OOM.
- `insert_logits[0].requires_grad == True` and `.grad_fn` chain traces back to both `state.nu` and `state.delta`.
- `P_u_list[0].sum() ≈ 1.0` (softmax normalization sanity).
- Calling `optimize_unified_v2(...)` for 10 steps does NOT crash with "backward through freed graph" or "memory concat length mismatch" errors.

### R001 pilot after 5b-ii

From `refine-logs/EXPERIMENT_TRACKER.md`:

> R001 (M0): Full, dog, K_ins=1, 50 steps → no NaN, logs sane, P_u extractable

Launch on Pro 6000 GPU1 (GPU0 usually has another user):

```bash
ssh lvshaoting-pro6000
CUDA_VISIBLE_DEVICES=1 python -m memshield.run_pilot_r001 --clip dog --n_steps 50
```

You'll need to write `memshield/run_pilot_r001.py` too — a thin driver that:
1. Loads DAVIS dog frames + GT mask
2. Calls `create_insert_base(strategy="propainter", ...)` × 1 (K_ins=1, boundary-only)
3. Runs clean-SAM2 forward once (mode="clean") to populate `C_u`
4. Calls `optimize_unified_v2(...)` with n_steps=50
5. Saves the modified video + diagnostics JSON

---

## Environment facts (verify on resume)

**Pro 6000** (`ssh lvshaoting-pro6000`, HostName 183.175.157.243 port 6000):
- conda env `memshield`: torch 2.8.0+cu128, Blackwell sm_120
- ProPainter installed at `~/ProPainter`, weights SHA-verified
- GPU0 often held by another user; default to `CUDA_VISIBLE_DEVICES=1`
- mihomo proxy at `127.0.0.1:7890` for GitHub release downloads
- Our clone at `~/sam2-pre-new`; SAM2 source at `~/sam2_repo`
- NAS home; I/O moderate — keep experiment artifacts local to `/tmp/` or compress before moving to home

**Local Windows** has NO GitHub SSH key: push via Pro 6000 using git bundle + ff-merge + push pattern (see any recent commit's push flow in the previous session's tool calls).

---

## Codex MCP continuity

Use the same thread ID for all Chunk 5b-ii and Chunk 6 review rounds:
```
threadId = "019db350-32bb-74a2-b6dc-d9491bf24025"
```

Tool: `mcp__codex__codex-reply`. Codex already knows:
- Chunks 1-5b-i are READY
- The `Sam2ForwardFn` contract and stated expected shapes
- SAM2's memory-layout specifics at `sam2_base.py:497-680`
- User preferred sign-PGD, 2:1 δ:ν stage-3 schedule, max(lpips) for Lagrange update, clean f_prev for LPIPS
- `track_in_reverse=True` is intentionally `NotImplementedError` in the hook

Start the 5b-ii review with a prompt like:
> "Chunk 5b-ii ready for review: `memshield/sam2_forward_adapter.py` implements `Sam2ForwardFn` per the protocol defined in `optimize_v2.py`. Key design choices: [A/B/C]. Smoke test on Pro 6000 cuda:1 with K_ins=1 dog 15-frame clip: [results]. Please review for correctness w.r.t. SAM2 video-predictor state machine + gradient flow through memory_attention."

---

## Known risks and gotchas

1. **SAM2 VideoPredictor state**. The predictor maintains an `inference_state` dict with frame buffers, etc. Running PGD means repeatedly modifying the input video while keeping inference_state consistent. The cleanest approach: `reset_state()` each step, re-initialize, re-add the first-frame prompt, re-run propagate. Expensive but correct. Optimization (later): warm-start the state with a cached first-frame prompt and only re-run from the first non-cond frame.

2. **Modified prefix frames must go through Hiera in grad mode**. SAM2's `image_embedding_dict` caches per-frame embeddings. Clearing the cache entries for modified prefix frames forces re-encoding. Do this explicitly — don't rely on the video predictor's cache eviction.

3. **`insert_frame_ids` on Clock W vs M**. Under our M==W assumption they're the same set, but do the comparison against `schedule.w_positions` explicitly (not `.m_positions`) to keep the semantic intent in the code.

4. **`fg_mask_by_frame` resolution**. Must match `Nq = H_feat * W_feat` of the memory-attention tokens (not the raw image). Use `memory_attention.d_model`-appropriate downsampling.

5. **Numerical stability on bf16/fp16**. SAM2 runs bf16 by default on modern GPUs. L_stale's softmax path in the probe is already float32 for stability; verify that `fg_mask_by_frame` tensors are cast correctly.

6. **LPIPS on Pro 6000**. We haven't validated `lpips` package is installed in the `memshield` env. Check `pip list | grep lpips` on first run; if absent, install via `pip install --no-deps lpips` and manually grab its alexnet weights if the default download path is blocked.

7. **First PGD step LPIPS will spike**. ν starts at 0, so `x_ins == base` and LPIPS(base, clean_prev) is whatever the ProPainter output looks like vs. its clean neighbor — could be > 0.10 budget, which immediately triggers the Lagrange growth. That's expected; μ_ν stabilizes around step 20-30.

---

## Stop-conditions for 5b-ii

Declare 5b-ii done when:
- [ ] `memshield/sam2_forward_adapter.py` (or similar) implements the full `Sam2ForwardFn` protocol
- [ ] Smoke script `scripts/smoke_5b_ii.py` passes on Pro 6000 cuda:1
- [ ] Codex R8 review at 8.0+/10 READY
- [ ] R001 sanity pilot runs 50 steps without NaN/crash

Then move to Chunk 6 (eval metrics: rebound, post-loss AUC, per-frame J trajectory) which is much smaller.
