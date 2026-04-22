"""R001 sanity pilot (EXPERIMENT_TRACKER M0): Full, dog, K_ins=1, 50 steps.

Goal: confirm no NaN, logs sane, P_u extractable after 50 PGD steps of the
full MemoryShield v2 pipeline with the real SAM2.1 adapter (5b-ii). This
is the gate between Chunk 5b-ii and Chunk 6 (eval metrics).

Pipeline:
  1. Load DAVIS dog frames (22 = 15 prefix + 7 eval) + first-frame GT mask.
  2. Clean SAM2 forward over all 22 frames to produce per-frame pred_masks;
     use these as proxies for bundle.C_u (eval foreground) and for
     ProPainter's mask_prev / mask_after inputs.
  3. ProPainter insert base at slot k=0 (K_ins=1, canonical schedule gives
     o_after=11 for T_prefix=15, num_maskmem=7).
  4. Build semantic masks D_ins, C_ins, ROI_ins from the clean masks and
     the chosen decoy_offset.
  5. Run SAM2VideoAdapter.prepare_from_clean(bundle) to cache suffix Hiera.
  6. optimize_unified_v2(n_steps=50). Stages scaled for the 50-step budget:
     stage1=10, stage2=20, stage3=20 (with 2:1 δ:ν in stage 3 → 60
     forward/backward pairs for stage 3, 10+20+60 = 90 total).
  7. Save modified video + diagnostics JSON under runs/r001/.

Usage:
    conda activate memshield
    cd ~/sam2-pre-new
    CUDA_VISIBLE_DEVICES=1 python -m memshield.run_pilot_r001 \
        --clip dog --n_steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from memshield.optimize_v2 import OptimizeConfig, VideoBundle, optimize_unified_v2
from memshield.scheduler import compute_schedule_v2
from memshield.sam2_forward_adapter import SAM2VideoAdapter
from memshield.propainter_base import (
    ProPainterConfig, create_insert_base, is_propainter_available,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DAVIS_ROOT = REPO_ROOT / "data" / "davis"
CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_clip(clip: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    jpg_dir = DAVIS_ROOT / "JPEGImages" / "480p" / clip
    ann_dir = DAVIS_ROOT / "Annotations" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    if len(jpgs) < n_frames:
        raise RuntimeError(
            f"{clip} only has {len(jpgs)} frames < required {n_frames}")
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])
    ann0 = np.array(Image.open(ann_dir / "00000.png"))
    mask0 = (ann0 > 0).astype(np.uint8)
    return frames, mask0


def stage_frames_to_tempdir(frames: np.ndarray, tmpdir: Path) -> Path:
    """SAM2's `init_state` reads jpg files from a directory. Write the first
    N frames to a temp dir so the vanilla SAM2 flow can propagate over just
    the first 22 frames without seeing downstream-unused frames."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(tmpdir / f"{i:05d}.jpg", quality=95)
    return tmpdir


def clean_sam2_forward(predictor, frames: np.ndarray, mask0: np.ndarray,
                       device: str) -> List[np.ndarray]:
    """Run vanilla SAM2 over `frames` with `mask0` first-frame prompt;
    return per-frame binary masks (video resolution)."""
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    with tempfile.TemporaryDirectory() as td:
        tmpdir = stage_frames_to_tempdir(frames, Path(td))
        state = predictor.init_state(video_path=str(tmpdir))
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1, mask=mask0,
        )
        masks_per_frame: List[np.ndarray] = [None] * len(frames)
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            # video_res_masks: [num_obj, 1, H_vid, W_vid], logits — sigmoid >0.5 = fg.
            mask_fg = (video_res_masks[0, 0].float().sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            masks_per_frame[frame_idx] = mask_fg
        # Any unfilled entries → treat as empty.
        for i in range(len(masks_per_frame)):
            if masks_per_frame[i] is None:
                masks_per_frame[i] = np.zeros((H_vid, W_vid), dtype=np.uint8)
    return masks_per_frame


# ---------------------------------------------------------------------------
# Semantic mask helpers
# ---------------------------------------------------------------------------


def shift_mask(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = m.shape
    out = np.zeros_like(m)
    y_src_lo = max(0, -dy); y_src_hi = min(H, H - dy)
    x_src_lo = max(0, -dx); x_src_hi = min(W, W - dx)
    y_dst_lo = max(0, dy); y_dst_hi = min(H, H + dy)
    x_dst_lo = max(0, dx); x_dst_hi = min(W, W + dx)
    out[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] = \
        m[y_src_lo:y_src_hi, x_src_lo:x_src_hi]
    return out


def dilate_u8(m: np.ndarray, radius: int) -> np.ndarray:
    import cv2
    if radius <= 0:
        return m.astype(np.uint8)
    ker = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(m.astype(np.uint8), ker, iterations=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="dog")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--K_ins", type=int, default=1)
    parser.add_argument("--T_prefix", type=int, default=15)
    parser.add_argument("--eval_window", type=int, default=7)
    parser.add_argument("--decoy_dy", type=int, default=0)
    parser.add_argument("--decoy_dx", type=int, default=80)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "runs" / "r001"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[r001] device = {device} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    assert CHECKPOINT.exists(), f"missing checkpoint: {CHECKPOINT}"
    assert DAVIS_ROOT.exists(), f"missing DAVIS: {DAVIS_ROOT}"

    T_full = args.T_prefix + args.eval_window
    frames, mask0 = load_clip(args.clip, n_frames=T_full)
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    print(f"[r001] loaded {args.clip}: frames {frames.shape} mask0 {mask0.shape}")

    # --- build SAM2 predictor -------------------------------------------------
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device)
    predictor.eval()

    # --- clean forward → per-frame masks --------------------------------------
    print("[r001] clean SAM2 forward ...")
    t0 = time.time()
    clean_masks = clean_sam2_forward(predictor, frames, mask0, device)
    print(f"[r001] clean forward done in {time.time()-t0:.1f}s; "
          f"fg area mask0={int(mask0.sum())} "
          f"mask[-1]={int(clean_masks[-1].sum())}")

    # --- schedule + insert base (ProPainter) ---------------------------------
    schedule = compute_schedule_v2(
        T_prefix_orig=args.T_prefix, num_maskmem=7, K_ins=args.K_ins,
        variant="canonical",
    )
    print(f"[r001] schedule: w_positions={schedule.w_positions} "
          f"T_mod={schedule.T_prefix_mod} slots={[(s.m_k, s.o_after) for s in schedule.slots]}")

    assert is_propainter_available(), "ProPainter not installed on this machine"
    decoy_offset = (args.decoy_dy, args.decoy_dx)
    insert_bases: List[np.ndarray] = []
    edit_masks:   List[np.ndarray] = []
    D_ins_list:   List[np.ndarray] = []
    C_ins_list:   List[np.ndarray] = []
    ROI_ins_list: List[np.ndarray] = []

    for k, slot in enumerate(schedule.slots):
        o_after = slot.o_after
        frame_prev = frames[o_after]
        frame_after = frames[o_after + 1]
        mask_prev = clean_masks[o_after]
        mask_after = clean_masks[o_after + 1]
        print(f"[r001] ProPainter slot k={k}: o_after={o_after}, "
              f"fg_prev={int(mask_prev.sum())}, fg_after={int(mask_after.sum())}")
        result = create_insert_base(
            strategy="propainter",
            frame_prev=frame_prev, frame_after=frame_after,
            mask_prev=mask_prev, mask_after=mask_after,
            decoy_offset=decoy_offset,
            seam_dilate_px=5, safety_margin=8, feather_px=3,
        )
        if result is None:
            raise RuntimeError(
                f"ProPainter insert-base returned None at slot {k} "
                f"(decoy_offset={decoy_offset} may violate border safety).")
        base, edit_mask = result
        insert_bases.append(base)
        edit_masks.append(edit_mask)
        C = (mask_after > 0).astype(np.uint8)
        D = shift_mask(C, *decoy_offset)
        ROI = dilate_u8(((C | D) > 0).astype(np.uint8), radius=5)
        D_ins_list.append(D)
        C_ins_list.append(C)
        ROI_ins_list.append(ROI)

    # C_u for the 7 eval frames = clean masks at [T_prefix : T_prefix + eval].
    C_u = [clean_masks[args.T_prefix + i] for i in range(args.eval_window)]
    masks_gt = np.stack(clean_masks[:T_full], axis=0)

    # --- cfg / adapter --------------------------------------------------------
    # 50-step budget: stage1 = 10, stage2 = 20, stage3 = 20 (with 2:1 ratio).
    cfg = OptimizeConfig(
        K_ins=args.K_ins, num_maskmem=7,
        T_prefix_orig=args.T_prefix,
        eval_window_size=args.eval_window, stale_window_size=3,
        n_steps=args.n_steps, stage1_end=10, stage2_end=30,
        stage3_delta_per_nu_ratio=2,
        lagrange_update_every=10, log_every=5, device=device,
    )

    adapter = SAM2VideoAdapter(
        predictor=predictor, cfg=cfg,
        first_frame_mask_video_res=mask0,
        video_H=H_vid, video_W=W_vid,
    )
    # Explicit prep: compute suffix Hiera cache + establish HW_mem. (The
    # adapter auto-preps on first __call__ if skipped, but doing it here
    # keeps the one-time cost outside the optimize loop's first step.)
    prep_bundle = VideoBundle(
        frames_orig=frames, masks_gt=masks_gt, schedule=schedule,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=D_ins_list, C_ins=C_ins_list, ROI_ins=ROI_ins_list, C_u=C_u,
    )
    adapter.prepare_from_clean(prep_bundle)
    print(f"[r001] adapter ready; HW_mem={adapter.HW_mem} "
          f"(H_feat={adapter._H_feat}, W_feat={adapter._W_feat})")

    # --- run optimize ---------------------------------------------------------
    print(f"[r001] optimize_unified_v2 for {args.n_steps} steps ...")
    t0 = time.time()
    final, diag = optimize_unified_v2(
        frames_orig=frames, masks_gt=masks_gt,
        sam2_forward_fn=adapter, cfg=cfg,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=D_ins_list, C_ins=C_ins_list, ROI_ins=ROI_ins_list, C_u=C_u,
        lpips_fn=None,    # LPIPS optional for sanity; off for R001
    )
    elapsed = time.time() - t0

    # --- validation checks ----------------------------------------------------
    assert final.dtype == np.uint8
    history = diag["history"]
    assert len(history) == args.n_steps
    finite = all(
        all(np.isfinite(v) for k, v in h.items()
            if isinstance(v, (int, float)))
        for h in history
    )
    if not finite:
        offenders = [
            (h["step"], k) for h in history for k, v in h.items()
            if isinstance(v, (int, float)) and not np.isfinite(v)
        ]
        raise RuntimeError(f"non-finite loss values: {offenders[:5]}")

    # --- save artifacts -------------------------------------------------------
    # Raw modified video as numpy array (uint8) — avoid pulling in ffmpeg
    # for the sanity run. Downstream chunks can compress if needed.
    np.save(out_dir / "modified_video.npy", final)
    np.save(out_dir / "final_nu.npy", diag["final_nu"])
    np.save(out_dir / "final_delta.npy", diag["final_delta"])
    json.dump(
        {
            "clip": args.clip, "n_steps": args.n_steps,
            "K_ins": args.K_ins, "decoy_offset": list(decoy_offset),
            "schedule": diag["schedule"],
            "stage_boundaries": list(diag["stage_boundaries"]),
            "mu_nu_final": diag["mu_nu_final"],
            "elapsed_sec": elapsed,
            "history": history,
        },
        open(out_dir / "diagnostics.json", "w"), indent=2, default=float,
    )

    stages = [h["stage"] for h in history]
    nu_L1 = float(np.abs(diag["final_nu"]).mean())
    delta_L1 = float(np.abs(diag["final_delta"]).mean())
    print(f"[r001] DONE in {elapsed:.1f}s  "
          f"nu_L1={nu_L1:.4f} delta_L1={delta_L1:.4f} "
          f"mu_nu_final={diag['mu_nu_final']:.3f}")
    print(f"[r001] stages summary: "
          f"stage1={stages.count(1)} stage2={stages.count(2)} stage3={stages.count(3)}")
    print(f"[r001] artifacts saved to {out_dir}")
    print("[r001] SANITY PASS (no NaN, logs sane)")


if __name__ == "__main__":
    main()
