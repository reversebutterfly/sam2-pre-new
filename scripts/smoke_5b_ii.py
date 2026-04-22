"""Chunk 5b-ii smoke test: real SAM2 adapter on a 15-frame DAVIS dog clip.

Verifies (per HANDOFF_CHUNK_5B_II.md §Smoke test):
  1. End-to-end forward runs with K_ins=1 on Pro 6000 cuda:1, no OOM.
  2. `insert_logits[0].requires_grad == True` and the .grad_fn chain traces
     back to BOTH `state.nu` and `state.delta`.
  3. `P_u_list[0].sum() ≈ 1.0` (softmax normalization invariant).
  4. `optimize_unified_v2(...)` for 10 steps does not crash with
     "backward through freed graph" / "memory concat length mismatch".

Intentionally NOT exercised here (out of scope for 5b-ii):
  * ProPainter insert bases — we use a trivial "blurred neighbor copy" so the
    adapter smoke doesn't depend on ProPainter weights being resident.
  * LPIPS — set to None; the Lagrangian path is unit-tested in
    `memshield.optimize_v2` already.
  * Full 200-step schedule — 10 steps is enough to hit all three optimizer
    stages (Stage1 = 4 steps, Stage2 = 3 steps, Stage3 = 3 steps).

Usage (Pro 6000):
    conda activate memshield
    cd ~/sam2-pre-new
    CUDA_VISIBLE_DEVICES=1 python scripts/smoke_5b_ii.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# --- path bootstrap (run from repo root) -------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memshield.optimize_v2 import OptimizeConfig, VideoBundle, optimize_unified_v2      # noqa: E402
from memshield.scheduler import compute_schedule_v2                                      # noqa: E402
from memshield.sam2_forward_adapter import SAM2VideoAdapter                              # noqa: E402


DAVIS_ROOT = REPO / "data" / "davis"
CLIP = "dog"
CHECKPOINT = REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _load_dog_clip(n_frames: int):
    """Load `n_frames` frames of DAVIS dog at 480p + first-frame GT mask.

    Returns:
        frames:  [T, H, W, 3] uint8
        mask0:   [H, W]        uint8 (0/1)
    """
    jpg_dir = DAVIS_ROOT / "JPEGImages" / "480p" / CLIP
    ann_dir = DAVIS_ROOT / "Annotations" / "480p" / CLIP
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    assert len(jpgs) == n_frames, f"only {len(jpgs)} frames found"
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])
    # First-frame annotation: pick any non-zero object id as FG.
    ann0 = np.array(Image.open(ann_dir / "00000.png"))
    mask0 = (ann0 > 0).astype(np.uint8)
    return frames, mask0


def _make_trivial_insert_base(frame: np.ndarray) -> np.ndarray:
    """Shift-and-blur of the supplied frame; stands in for a ProPainter
    output. Fine for smoke-testing the adapter — the optimizer treats it as
    an opaque uint8 array."""
    # Shift by 8 px and blur lightly via a 3x3 box filter.
    H, W, _ = frame.shape
    shifted = np.roll(frame, shift=(0, 8), axis=(0, 1))
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    from scipy.signal import convolve2d
    out = np.stack(
        [convolve2d(shifted[..., c], kernel, mode="same", boundary="symm")
         for c in range(3)], axis=-1,
    )
    return out.clip(0, 255).astype(np.uint8)


def _build_semantic_masks(mask0_480: np.ndarray, decoy_offset):
    """Build (D, C, ROI) for a single insert, plus C_u for eval frames.

    For smoke purposes we reuse the first-frame GT mask as an approximation
    of the object location; D is the GT mask shifted by `decoy_offset`; ROI
    is the dilated union of D and C.
    """
    H, W = mask0_480.shape
    C = mask0_480.astype(np.uint8)
    dy, dx = decoy_offset
    D = np.zeros_like(C)
    y_src_lo = max(0, -dy); y_src_hi = min(H, H - dy)
    x_src_lo = max(0, -dx); x_src_hi = min(W, W - dx)
    y_dst_lo = max(0, dy); y_dst_hi = min(H, H + dy)
    x_dst_lo = max(0, dx); x_dst_hi = min(W, W + dx)
    D[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] = \
        C[y_src_lo:y_src_hi, x_src_lo:x_src_hi]
    ROI = ((C > 0) | (D > 0)).astype(np.uint8)
    # Edit mask = ROI (where the attacker may edit pixels on the insert).
    edit_mask = ROI.copy()
    return D, C, ROI, edit_mask


def main():
    torch.manual_seed(0)
    # CUDA_VISIBLE_DEVICES=1 maps physical GPU1 to logical cuda:0; default
    # accordingly. Users pick the physical GPU via the env var.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    assert CHECKPOINT.exists(), f"missing checkpoint: {CHECKPOINT}"
    assert DAVIS_ROOT.exists(), f"missing DAVIS: {DAVIS_ROOT}"

    T_prefix = 15
    eval_window = 7
    T_full = T_prefix + eval_window
    frames, mask0 = _load_dog_clip(n_frames=T_full)
    print(f"[smoke] loaded dog: frames {frames.shape} mask0 {mask0.shape}")

    # ----- build SAM2 predictor -----------------------------------------------
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device)
    predictor.eval()
    print(f"[smoke] SAM2 loaded. image_size={predictor.image_size} "
          f"num_maskmem={predictor.num_maskmem}")

    # ----- schedule + insert base --------------------------------------------
    K_ins = 1
    schedule = compute_schedule_v2(
        T_prefix_orig=T_prefix, num_maskmem=7, K_ins=K_ins, variant="canonical",
    )
    print(f"[smoke] schedule: w_positions={schedule.w_positions} "
          f"T_mod={schedule.T_prefix_mod}")

    decoy_offset = (0, 40)
    insert_base = _make_trivial_insert_base(frames[schedule.slots[0].o_after])
    D_ins, C_ins, ROI_ins, edit_mask = _build_semantic_masks(mask0, decoy_offset)
    insert_bases = [insert_base]
    edit_masks = [edit_mask]

    # C_u for eval frames = GT mask (approx — smoke only)
    masks_gt = np.stack([mask0] * T_full, axis=0)
    C_u = [mask0] * eval_window

    # ----- cfg ----------------------------------------------------------------
    # stage3_delta_per_nu_ratio=2 exercises the R5/R6 default "2:1 δ:ν"
    # substep pattern — each Stage-3 logical step runs δ, δ, joint. Smoke
    # covers 10 logical steps (4 stage1 + 3 stage2 + 3 stage3) with the
    # 2:1 ratio, so Stage 3 fires 9 forward/backward pairs total.
    cfg = OptimizeConfig(
        K_ins=K_ins, num_maskmem=7,
        T_prefix_orig=T_prefix, eval_window_size=eval_window, stale_window_size=3,
        n_steps=10, stage1_end=4, stage2_end=7, stage3_delta_per_nu_ratio=2,
        lagrange_update_every=5, log_every=1, device=device,
    )

    # ----- adapter ------------------------------------------------------------
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    adapter = SAM2VideoAdapter(
        predictor=predictor, cfg=cfg,
        first_frame_mask_video_res=mask0,
        video_H=H_vid, video_W=W_vid,
    )

    # --- (1) end-to-end forward, verify shapes + grad flow --------------------
    print("[smoke] step 1/4: one forward pass (grad-enabled) ...")
    # Build a dummy VideoBundle to pass through; prepare_from_clean needs
    # frames_orig + schedule.
    bundle = VideoBundle(
        frames_orig=frames, masks_gt=masks_gt, schedule=schedule,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=[D_ins], C_ins=[C_ins], ROI_ins=[ROI_ins], C_u=C_u,
    )
    adapter.prepare_from_clean(bundle)
    print(f"[smoke] adapter.HW_mem = {adapter.HW_mem} "
          f"(H_feat={adapter._H_feat}, W_feat={adapter._W_feat})")

    # Build a fake modified prefix with a tiny perturbation tensor so we can
    # check gradient flow.
    from memshield.optimize_v2 import build_modified_video, PGDState
    nu = torch.zeros(K_ins, H_vid, W_vid, 3, device=device,
                     dtype=torch.float32, requires_grad=True)
    # `state.delta` is sized to frames_orig (full video) by convention —
    # suffix slice is allocated but never reaches `build_modified_video`'s
    # output (mod_to_orig only maps mod positions to prefix originals).
    delta = torch.zeros(T_full, H_vid, W_vid, 3, device=device,
                        dtype=torch.float32, requires_grad=True)
    nu.data.add_(0.01)
    delta.data.add_(0.005)
    state = PGDState(nu=nu, delta=delta, mu_nu=10.0, step=0)
    mod_video = build_modified_video(bundle, state, cfg)
    assert mod_video.requires_grad

    out = adapter(mod_video, mode="attack", cfg=cfg, bundle=bundle)

    # Shape checks.
    assert len(out["insert_logits"]) == K_ins
    assert len(out["eval_logits"]) == eval_window
    assert len(out["P_u_list"]) == cfg.stale_window_size
    il = out["insert_logits"][0]
    el = out["eval_logits"][0]
    assert il.shape == (H_vid, W_vid), il.shape
    assert el.shape == (H_vid, W_vid), el.shape
    print(f"[smoke] shapes OK: insert {tuple(il.shape)} eval {tuple(el.shape)}")

    # --- (2) grad flow: backward touches BOTH nu and delta --------------------
    assert il.requires_grad, "insert_logits[0] not differentiable"
    s_ins = il.sum()
    s_ins.backward(retain_graph=True)
    assert state.nu.grad is not None and state.nu.grad.abs().sum() > 0, \
        "nu.grad did not receive signal from insert_logits"
    assert state.delta.grad is not None and state.delta.grad.abs().sum() > 0, \
        "delta.grad did not receive signal from insert_logits"
    print(f"[smoke] grad check 1 OK: |nu.grad|={state.nu.grad.abs().sum():.4e} "
          f"|delta.grad|={state.delta.grad.abs().sum():.4e}")
    state.nu.grad.zero_(); state.delta.grad.zero_()

    # Backward from eval_logits — must also reach nu (via memory chain) and
    # delta (directly + via memory).
    s_eval = el.sum()
    s_eval.backward(retain_graph=True)
    assert state.nu.grad.abs().sum() > 0, \
        "nu.grad did not receive signal from eval_logits (memory chain broken)"
    assert state.delta.grad.abs().sum() > 0, \
        "delta.grad did not receive signal from eval_logits"
    print(f"[smoke] grad check 2 OK: eval→nu {state.nu.grad.abs().sum():.4e} "
          f"eval→delta {state.delta.grad.abs().sum():.4e}")
    state.nu.grad.zero_(); state.delta.grad.zero_()

    # --- (3) P_u normalization invariant -------------------------------------
    P0 = out["P_u_list"][0]
    assert P0 is not None, "P_u_list[0] is None — probe did not fire"
    assert P0.shape == (3,), P0.shape
    total = float(P0.detach().sum())
    print(f"[smoke] P_u_list[0] = {P0.detach().cpu().numpy()}  sum={total:.6f}")
    assert abs(total - 1.0) < 1e-3, f"P_u[0] sum={total} not ≈ 1"

    # Free memory before the 10-step optimize run. We must also drop
    # lingering references inside state.*.grad + probe caches, because the
    # earlier backward(retain_graph=True) kept the direct-forward autograd
    # graph pinned through P_u_by_frame tensors.
    import gc
    state.nu.grad = None
    state.delta.grad = None
    del mod_video, out, s_ins, s_eval, state, nu, delta
    adapter.probe.reset()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[smoke] GPU mem after cleanup: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    # --- (4) 10-step optimize_unified_v2 sanity -------------------------------
    print("[smoke] step 4/4: optimize_unified_v2 for 10 steps ...")
    final, diag = optimize_unified_v2(
        frames_orig=frames, masks_gt=masks_gt,
        sam2_forward_fn=adapter, cfg=cfg,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=[D_ins], C_ins=[C_ins], ROI_ins=[ROI_ins], C_u=C_u,
        lpips_fn=None,
    )
    assert final.dtype == np.uint8
    assert final.shape == (T_prefix + K_ins, H_vid, W_vid, 3), final.shape
    history = diag["history"]
    assert len(history) == cfg.n_steps
    stages = [h["stage"] for h in history]
    assert stages[:4] == [1, 1, 1, 1]
    assert stages[4:7] == [2, 2, 2]
    assert stages[7:] == [3, 3, 3]
    for h in history:
        for k in ("L_loss", "total"):
            v = h.get(k)
            if v is not None:
                assert np.isfinite(v), f"non-finite {k} at step {h['step']}: {v}"
    print(f"[smoke] 10-step optimize PASS  "
          f"stages={stages}  "
          f"nu_L1={np.abs(diag['final_nu']).mean():.4f}  "
          f"delta_L1={np.abs(diag['final_delta']).mean():.4f}")

    print("\n[smoke] ALL 5b-ii CHECKS PASSED")


if __name__ == "__main__":
    main()
