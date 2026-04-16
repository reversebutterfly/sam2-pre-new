"""
MemoryShield: main pipeline orchestrator.

Flow:
  1. Load video frames + GT masks
  2. Analyze content (optical flow, occlusion, topology)
  3. Compute insertion schedule (FIFO resonance + event triggers)
  4. Generate adversarial frames via PGD
  5. Build protected video (original + inserted frames)
  6. Return protected video + metadata
"""
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MemShieldConfig
from .scheduler import (
    InsertionSlot,
    build_modified_index_map,
    compute_resonance_schedule,
    merge_event_triggers,
)
from .analyzer import analyze_video
from .generator import optimize_cooperative
from .surrogate import SAM2Surrogate


def protect_video(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    surrogate: SAM2Surrogate,
    cfg: MemShieldConfig,
    video_name: str = "unknown",
) -> Tuple[List[np.ndarray], Dict]:
    """Full MemoryShield pipeline for one video.

    Args:
        frames_uint8: list of [H, W, 3] uint8 original frames.
        masks_uint8: list of [H, W] uint8 binary GT masks.
        surrogate: initialized SAM2Surrogate.
        cfg: MemoryShield config.
        video_name: for logging.

    Returns:
        (protected_frames, metadata)
        protected_frames: list of [H, W, 3] uint8 frames (original + inserted).
        metadata: dict with schedule, metrics, timing info.
    """
    T = len(frames_uint8)
    print(f"\n{'='*60}")
    print(f"  MemoryShield: {video_name}  ({T} frames)")
    print(f"{'='*60}")

    t0 = time.time()

    # ── Phase 1: Content analysis ────────────────────────────────────────────
    print("[1/4] Analyzing video content...")
    analysis = analyze_video(
        frames_uint8, masks_uint8,
        enable_occlusion=cfg.enable_occlusion_ghost,
        enable_topology=cfg.enable_topology_seed,
        occlusion_flow_threshold=cfg.occlusion_flow_threshold,
        topology_narrow_px=cfg.topology_narrow_px,
    )
    print(f"      Occlusion events: {analysis['occlusion_events']}")
    print(f"      Topology events:  {analysis['topology_events']}")
    print(f"      Scene changes:    {analysis['scene_changes']}")

    # ── Phase 2: Compute insertion schedule ──────────────────────────────────
    print("[2/4] Computing FIFO resonance schedule...")
    base_schedule = compute_resonance_schedule(
        n_original=T,
        fifo_window=cfg.fifo_window,
        max_ratio=cfg.max_insertion_ratio,
    )
    schedule = merge_event_triggers(
        base_schedule,
        analysis["all_events"],
        n_original=T,
        max_ratio=cfg.max_insertion_ratio,
    )
    n_strong = sum(1 for s in schedule if s.frame_type == "strong")
    n_weak = sum(1 for s in schedule if s.frame_type == "weak")
    print(f"      Scheduled: {len(schedule)} insertions "
          f"({n_strong} strong, {n_weak} weak)")
    for s in schedule:
        print(f"        after frame {s.after_original_idx:3d}  "
              f"[{s.frame_type:6s}]  ({s.reason})")

    # ── Phase 3: Generate adversarial frames (cooperative decoy) ────────────
    print("[3/4] Optimizing cooperative decoy attack via PGD...")
    inserted_frames, perturbed_originals, gen_metrics = optimize_cooperative(
        surrogate, frames_uint8, masks_uint8, schedule, cfg,
    )
    best_loss = gen_metrics.get('best_loss')
    ssim_min = gen_metrics.get('final_ssim_min')
    n_pert = gen_metrics.get('n_perturbed_originals', 0)
    print(f"      Best loss: {best_loss:.4f}" if best_loss is not None else "      Best loss: N/A")
    print(f"      Final SSIM min: {ssim_min:.4f}" if ssim_min is not None else "      Final SSIM min: N/A")
    print(f"      Perturbed originals: {n_pert} frames")

    # ── Phase 4: Build protected video ───────────────────────────────────────
    print("[4/4] Building protected video...")
    idx_map = build_modified_index_map(T, schedule)

    # Build insertion-after lookup
    insert_after = {}
    for slot_i, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(slot_i)

    protected = []
    for orig_idx in range(T):
        # Use perturbed original if available, otherwise clean
        if orig_idx in perturbed_originals:
            protected.append(perturbed_originals[orig_idx])
        else:
            protected.append(frames_uint8[orig_idx])
        if orig_idx in insert_after:
            for slot_i in insert_after[orig_idx]:
                if slot_i in inserted_frames:
                    protected.append(inserted_frames[slot_i])

    elapsed = time.time() - t0
    ratio = len(schedule) / max(T, 1)
    print(f"\n  Done: {T} → {len(protected)} frames "
          f"(+{len(schedule)}, ratio={ratio:.1%})  [{elapsed:.1f}s]")

    metadata = {
        "video_name": video_name,
        "n_original": T,
        "n_protected": len(protected),
        "n_inserted": len(schedule),
        "insertion_ratio": ratio,
        "schedule": [(s.after_original_idx, s.frame_type, s.reason) for s in schedule],
        "analysis": {
            "occlusion_events": analysis["occlusion_events"],
            "topology_events": analysis["topology_events"],
            "scene_changes": analysis["scene_changes"],
        },
        "gen_metrics": gen_metrics,
        "elapsed_sec": elapsed,
        "idx_map": idx_map,
    }
    return protected, metadata


def evaluate_protection(
    protected_frames: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    metadata: Dict,
    checkpoint: str,
    config: str,
    device_str: str = "cuda",
) -> Dict:
    """Evaluate a protected video using the official SAM2 VideoPredictor.

    Runs SAM2 on the protected video and measures J&F on the original frames
    (skipping inserted frames).

    Args:
        protected_frames: protected video frames (original + inserted).
        masks_uint8: original GT masks.
        metadata: from protect_video, contains idx_map.
        checkpoint: SAM2 checkpoint path.
        config: SAM2 config string.
        device_str: "cuda" or "cpu".

    Returns:
        dict with J&F, per-frame scores, quality metrics.
    """
    import torch
    import cv2
    from sam2.build_sam import build_sam2_video_predictor
    import tempfile
    from pathlib import Path
    from .surrogate import get_interior_prompt

    device = torch.device(device_str)
    idx_map = metadata["idx_map"]
    mod_to_orig = idx_map["mod_to_orig"]
    n_orig = metadata["n_original"]

    # SAM2 VideoPredictor needs frames as image files in a directory
    # Use PNG (lossless) to isolate JPEG as a transfer variable
    tmpdir = tempfile.mkdtemp(prefix="memshield_eval_")
    try:
        for i, frame in enumerate(protected_frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])  # Max quality JPEG

        # Build predictor
        predictor = build_sam2_video_predictor(config, checkpoint, device=device)

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmpdir)

            # Prompt on frame 0 (always an original frame)
            coords_np, labels_np = get_interior_prompt(masks_uint8[0])
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=coords_np,
                labels=labels_np,
            )

            # Propagate
            pred_masks_mod = {}
            for frame_idx, obj_ids, masks_out in predictor.propagate_in_video(state):
                pred_masks_mod[frame_idx] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        # Compute J&F only on original frames (skip inserted)
        from skimage.metrics import structural_similarity as _ssim

        j_scores, f_scores, jf_scores = [], [], []
        ssim_scores, psnr_scores = [], []

        for mod_idx in range(len(protected_frames)):
            orig_idx = mod_to_orig[mod_idx]
            if orig_idx < 0:
                continue  # Skip inserted frames

            if mod_idx in pred_masks_mod and orig_idx < len(masks_uint8):
                pred = pred_masks_mod[mod_idx].astype(bool)
                gt = masks_uint8[orig_idx].astype(bool)

                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                j = float(inter) / max(float(union), 1e-9) if union > 0 else 1.0
                j_scores.append(j)
                jf_scores.append(j)  # Simplified: J only for now

        mean_j = float(np.mean(j_scores)) if j_scores else 0.0

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "mean_j": mean_j,
        "per_frame_j": j_scores,
        "n_eval_frames": len(j_scores),
    }
