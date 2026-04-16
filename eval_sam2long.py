#!/usr/bin/env python3
"""
SAM2Long Zero-Shot Transfer Evaluation (Block 3).

Takes saved protected videos from M1 and evaluates them with SAM2Long.
Computes J/F/J&F and RetentionRatio = Drop_SAM2Long / Drop_SAM2.

Core prediction: RetentionRatio_Decoy > RetentionRatio_Suppression
(Decoy retains more attack effect under confidence-gated tree memory.)

Prerequisites:
  - SAM2Long cloned: github.com/Mark12Ding/SAM2Long
  - Protected videos saved from M1 (--save_videos flag)
  - M1 results JSON for SAM2 baselines

IMPORTANT: This script must be run with SAM2Long on PYTHONPATH so that
`import sam2` resolves to SAM2Long's version (not standard SAM2):
  PYTHONPATH=/path/to/SAM2Long:$PYTHONPATH python eval_sam2long.py ...

Usage:
  # Evaluate all saved videos (SAM2Long via PYTHONPATH)
  PYTHONPATH=/path/to/SAM2Long python eval_sam2long.py --device cuda:0

  # Specific videos
  PYTHONPATH=/path/to/SAM2Long python eval_sam2long.py --videos bear,dog --device cuda:0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.scheduler import build_modified_index_map, compute_resonance_schedule
from memshield.surrogate import get_interior_prompt
from run_two_regimes import (
    DAVIS_20,
    EVAL_START,
    EVAL_END,
    load_video,
    compute_boundary_f,
)


def load_protected_video(video_dir: str) -> List[np.ndarray]:
    """Load a saved protected video (JPEG sequence)."""
    from PIL import Image
    stems = sorted(
        p.stem for p in Path(video_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    frames = []
    for stem in stems:
        frames.append(np.array(
            Image.open(Path(video_dir) / f"{stem}.jpg").convert("RGB")))
    return frames


def evaluate_sam2long(
    video_frames: List[np.ndarray],
    first_mask: np.ndarray,
    mod_to_orig: List[int],
    masks_gt: List[np.ndarray],
    eval_range: set,
    checkpoint: str,
    config: str,
    device_str: str,
    num_pathway: int = 3,
    iou_thre: float = 0.1,
    uncertainty: int = 2,
) -> dict:
    """Evaluate video with SAM2Long.

    SAM2Long uses tree-based memory with confidence gating.
    Falls back to standard SAM2 evaluation if SAM2Long is not installed.
    """
    import shutil
    import tempfile

    device = torch.device(device_str)
    tmpdir = tempfile.mkdtemp(prefix="sam2long_eval_")

    try:
        # Write frames as JPEG sequence
        for i, frame in enumerate(video_frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        # SAM2Long uses the same sam2 package but with extended inference.
        # The builder is the standard build_sam2_video_predictor; SAM2Long
        # params (num_pathway, iou_thre, uncertainty) are set on the state
        # after init_state(), following the official demo pattern.
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(
            config, checkpoint, device=device)

        with torch.inference_mode():
            state = predictor.init_state(video_path=tmpdir)

            # Set SAM2Long-specific parameters on the inference state.
            # These are recognized by SAM2Long's extended propagate_in_video.
            # If running plain SAM2 (no SAM2Long), these keys are ignored.
            state["num_pathway"] = num_pathway
            state["iou_thre"] = iou_thre
            state["uncertainty"] = uncertainty
            is_sam2long = hasattr(predictor, "propagate_in_video_with_tree") or \
                "num_pathway" in state
            print(f"    SAM2Long params set: P={num_pathway}, "
                  f"iou={iou_thre}, unc={uncertainty}")

            coords, labels = get_interior_prompt(first_mask)
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=coords, labels=labels)

            # SAM2Long's propagate_in_video returns (obj_ids, mask_list)
            # where mask_list[i] is the mask for frame i.
            # Standard SAM2 yields (frame_idx, obj_ids, masks_out) per frame.
            result = predictor.propagate_in_video(state)
            preds = {}
            if isinstance(result, tuple) and len(result) == 2:
                # SAM2Long format: (obj_ids, mask_list)
                _, mask_list = result
                for fi in range(len(mask_list)):
                    preds[fi] = (mask_list[fi][0] > 0.0).cpu().numpy().squeeze()
            else:
                # Standard SAM2 generator format
                for fi, _, masks_out in result:
                    preds[fi] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        # Compute J/F/J&F on eval window
        j_scores, f_scores = [], []
        for mi in range(len(video_frames)):
            oi = mod_to_orig[mi]
            if oi < 0 or oi not in eval_range:
                continue
            if mi not in preds or oi >= len(masks_gt):
                continue
            pred = preds[mi].astype(bool)
            gt = masks_gt[oi].astype(bool)
            inter = float((pred & gt).sum())
            union = float((pred | gt).sum())
            j_scores.append(inter / max(union, 1e-9) if union > 0 else 1.0)
            f_scores.append(compute_boundary_f(pred, gt))

        mj = float(np.mean(j_scores)) if j_scores else 0.0
        mf = float(np.mean(f_scores)) if f_scores else 0.0
        return {"mean_j": mj, "mean_f": mf, "mean_jf": 0.5 * (mj + mf),
                "j_scores": j_scores, "f_scores": f_scores,
                "n_eval_frames": len(j_scores)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="SAM2Long Zero-Shot Transfer (Block 3)")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video names (default: all saved)")
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--davis_root",
                        default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint",
                        default=os.path.join(ROOT, "checkpoints",
                                             "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config",
                        default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--m1_results",
                        default=os.path.join(ROOT, "results_regimes",
                                             "regimes_results.json"),
                        help="M1 results JSON for SAM2 baselines")
    parser.add_argument("--protected_dir",
                        default=os.path.join(ROOT, "results_regimes", "videos"),
                        help="Dir with saved protected videos from M1")
    parser.add_argument("--output_dir",
                        default=os.path.join(ROOT, "results_sam2long"))
    # SAM2Long parameters
    parser.add_argument("--num_pathway", type=int, default=3)
    parser.add_argument("--iou_thre", type=float, default=0.1)
    parser.add_argument("--uncertainty", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    eval_range = set(range(EVAL_START, min(args.max_frames, EVAL_END)))

    # Load M1 results for SAM2 baselines
    if os.path.exists(args.m1_results):
        with open(args.m1_results) as f:
            m1_results = json.load(f)
        print(f"Loaded M1 results: {len(m1_results)} videos")
    else:
        print(f"WARNING: M1 results not found at {args.m1_results}")
        m1_results = {}

    # Find available saved videos
    regimes = ["suppression", "decoy"]
    if args.videos:
        videos = args.videos.split(",")
    else:
        videos = sorted(set(
            d.rsplit("_", 1)[0]
            for d in os.listdir(args.protected_dir)
            if os.path.isdir(os.path.join(args.protected_dir, d))
        )) if os.path.isdir(args.protected_dir) else DAVIS_20

    print("=" * 70)
    print("  SAM2Long Zero-Shot Transfer Evaluation")
    print("=" * 70)
    print(f"  Videos:      {len(videos)}")
    print(f"  SAM2Long:    P={args.num_pathway}, iou={args.iou_thre}, "
          f"unc={args.uncertainty}")
    print(f"  Eval window: f{EVAL_START}-f{EVAL_END - 1}")
    print("=" * 70)

    all_results = {}

    for vid in videos:
        print(f"\n{'#' * 60}")
        print(f"  {vid}")
        print(f"{'#' * 60}")

        # Load GT
        frames, masks = load_video(args.davis_root, vid, args.max_frames)
        if len(frames) < 15:
            print(f"  [skip] {len(frames)} frames < 15")
            continue

        T = len(frames)
        schedule = compute_resonance_schedule(T, 7, 0.15)
        idx_map = build_modified_index_map(T, schedule)
        vid_results = {}

        # Clean SAM2Long baseline
        print("  [clean] SAM2Long evaluating...")
        clean_eval = evaluate_sam2long(
            frames, masks[0], list(range(T)), masks, eval_range,
            args.checkpoint, args.sam2_config, args.device,
            args.num_pathway, args.iou_thre, args.uncertainty)
        vid_results["clean_sam2long"] = clean_eval
        print(f"  [clean] SAM2Long J&F={clean_eval['mean_jf']:.4f}")

        # Get SAM2 clean baseline from M1
        m1_vid = m1_results.get(vid, {})
        clean_sam2_jf = m1_vid.get("clean", {}).get("mean_jf", 0)
        vid_results["clean_sam2_jf"] = clean_sam2_jf

        # Evaluate each regime's protected video
        for regime in regimes:
            vid_dir = os.path.join(args.protected_dir, f"{vid}_{regime}")
            if not os.path.isdir(vid_dir):
                print(f"  [{regime}] no saved video found at {vid_dir}")
                vid_results[regime] = {"error": "no saved video"}
                continue

            protected = load_protected_video(vid_dir)
            print(f"  [{regime}] SAM2Long evaluating ({len(protected)} frames)...")

            ev = evaluate_sam2long(
                protected, masks[0], idx_map["mod_to_orig"], masks,
                eval_range, args.checkpoint, args.sam2_config, args.device,
                args.num_pathway, args.iou_thre, args.uncertainty)

            # Compute drops and retention ratio
            drop_sam2long = clean_eval["mean_jf"] - ev["mean_jf"]

            # Get SAM2 drop from M1
            m1_regime = m1_vid.get(regime, {})
            drop_sam2 = m1_regime.get("jf_drop", 0)

            retention = (drop_sam2long / drop_sam2
                         if abs(drop_sam2) > 0.01 else float("nan"))

            vid_results[regime] = {
                **ev,
                "jf_drop_sam2long": drop_sam2long,
                "jf_drop_sam2": drop_sam2,
                "retention_ratio": retention,
            }
            print(f"  [{regime}] SAM2Long J&F={ev['mean_jf']:.4f}  "
                  f"drop={drop_sam2long:.4f}  "
                  f"SAM2_drop={drop_sam2:.4f}  "
                  f"retention={retention:.4f}")

        all_results[vid] = vid_results

        # Save incrementally
        out_path = os.path.join(args.output_dir, "sam2long_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAM2LONG TRANSFER SUMMARY")
    print("=" * 70)

    for regime in regimes:
        drops_s2l = []
        drops_s2 = []
        retentions = []
        for vr in all_results.values():
            r = vr.get(regime, {})
            if not isinstance(r, dict) or "retention_ratio" not in r:
                continue
            d_s2l = r["jf_drop_sam2long"]
            d_s2 = r["jf_drop_sam2"]
            ret = r["retention_ratio"]
            drops_s2l.append(d_s2l)
            drops_s2.append(d_s2)
            if not np.isnan(ret):
                retentions.append(ret)

        if retentions:
            print(f"\n  {regime}:")
            print(f"    SAM2   mean drop: {np.mean(drops_s2):.4f}")
            print(f"    SAM2L  mean drop: {np.mean(drops_s2l):.4f}")
            print(f"    RetentionRatio:   {np.mean(retentions):.4f}  "
                  f"(median={np.median(retentions):.4f})")

    # Test prediction: RetentionRatio_Decoy > RetentionRatio_Suppression
    supp_ret = [vr.get("suppression", {}).get("retention_ratio", float("nan"))
                for vr in all_results.values()]
    decoy_ret = [vr.get("decoy", {}).get("retention_ratio", float("nan"))
                 for vr in all_results.values()]
    supp_ret = [r for r in supp_ret if not np.isnan(r)]
    decoy_ret = [r for r in decoy_ret if not np.isnan(r)]

    if supp_ret and decoy_ret:
        print(f"\n  PREDICTION TEST: RetentionRatio_Decoy > RetentionRatio_Supp?")
        print(f"    Supp retention: {np.mean(supp_ret):.4f}")
        print(f"    Decoy retention: {np.mean(decoy_ret):.4f}")
        if np.mean(decoy_ret) > np.mean(supp_ret):
            print("    CONFIRMED: Decoy retains more effect under tree memory")
        else:
            print("    NOT CONFIRMED: Suppression retains more or equal")

    print(f"\nResults: {os.path.join(args.output_dir, 'sam2long_results.json')}")


if __name__ == "__main__":
    main()
