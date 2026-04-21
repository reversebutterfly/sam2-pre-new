"""Evaluate attacked clips on SAM2Long, compute drop vs SAM2.

Prerequisite: attacks already saved via `run_two_regimes.py --save_videos`
to `{output_dir}/videos/{vid}_{regime}/` as JPEG sequences.

What this script does:
  1. For each (vid, regime) pair, runs SAM2Long vos_inference on the saved
     JPEG sequence using the DAVIS f0 annotation as prompt.
  2. Reads the predicted masks, computes J&F against DAVIS ground truth over
     the same eval window used for SAM2 (EVAL_START..end).
  3. Writes per-clip results to `results_sam2long/{vid}_{regime}.json`.
  4. Also evaluates CLEAN (unmodified DAVIS) on SAM2Long for baselines.

Outputs `sam2long_summary.json`:
  {
    "blackswan": {
      "clean_sam2long": 0.93, "supp_sam2long": 0.05, "decoy_sam2long": 0.02,
      "clean_sam2": 0.93,     "supp_sam2": 0.00,     "decoy_sam2": 0.01,
      "drop_supp_sam2": 0.93, "drop_supp_sam2long": 0.88, "retention_supp": 0.95,
      "drop_decoy_sam2": 0.91,"drop_decoy_sam2long": 0.91,"retention_decoy": 1.00,
    },
    ...
  }
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# SAM2Long repo layout: /IMBR_Data/Student-home/2025M_LvShaoting/SAM2Long/
SAM2LONG_ROOT = Path("/IMBR_Data/Student-home/2025M_LvShaoting/SAM2Long")
sys.path.insert(0, str(SAM2LONG_ROOT))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam2.build_sam import build_sam2_video_predictor

EVAL_START = 10


def load_davis_masks(ann_dir, target_obj=None):
    """Load full DAVIS annotation sequence for target object."""
    paths = sorted(ann_dir.iterdir())
    masks = []
    for i, p in enumerate(paths):
        ann = np.array(Image.open(p))
        if i == 0 and target_obj is None:
            ids = sorted(set(ann.flat) - {0})
            target_obj = ids[0] if ids else 1
        masks.append((ann == target_obj).astype(np.uint8))
    return masks, target_obj


def db_eval_iou(pred, gt):
    """DAVIS-style J metric (IoU)."""
    pred_b = pred > 0
    gt_b = gt > 0
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def db_eval_f(pred, gt, bound_pix=0.008):
    """DAVIS-style F metric (boundary). Simplified: boundary IoU via 1px dilation."""
    from scipy.ndimage import binary_dilation
    pred_b = pred > 0
    gt_b = gt > 0
    r = max(1, int(bound_pix * min(pred.shape)))
    struct = np.ones((3, 3), dtype=bool)
    pred_bd = binary_dilation(pred_b, iterations=r) & ~binary_dilation(~pred_b, iterations=r)
    gt_bd = binary_dilation(gt_b, iterations=r) & ~binary_dilation(~gt_b, iterations=r)
    inter = (pred_bd & gt_bd).sum()
    p_area = pred_bd.sum()
    g_area = gt_bd.sum()
    if p_area + g_area == 0:
        return 1.0
    p = inter / max(p_area, 1)
    r_ = inter / max(g_area, 1)
    if p + r_ == 0:
        return 0.0
    return float(2 * p * r_ / (p + r_))


def run_sam2long_on_video(predictor, frame_dir, init_mask, device,
                          num_pathway=3, iou_thre=0.1, uncertainty=2):
    """Run SAM2Long tracking on a JPEG sequence, return per-frame predicted masks.

    SAM2Long requires num_pathway/iou_thre/uncertainty to be set on
    inference_state AFTER init_state and BEFORE add_new_mask.
    """
    frame_paths = sorted(frame_dir.iterdir())
    n = len(frame_paths)

    state = predictor.init_state(video_path=str(frame_dir))
    state["num_pathway"] = num_pathway
    state["iou_thre"] = iou_thre
    state["uncertainty"] = uncertainty

    ann_frame_idx = 0
    ann_obj_id = 1
    predictor.add_new_mask(
        inference_state=state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=torch.from_numpy(init_mask).to(device).bool(),
    )
    # SAM2Long's propagate_in_video RETURNS (obj_ids, list_of_masks) instead of
    # yielding per-frame — it runs the multi-pathway tree internally and picks
    # the best trajectory, returning the selected pred_masks per frame at
    # original video resolution.
    obj_ids, masks_list = predictor.propagate_in_video(state)
    pred_masks = [None] * n
    for idx, m in enumerate(masks_list):
        if torch.is_tensor(m):
            arr = (m > 0.0).cpu().numpy().astype(np.uint8)
        else:
            arr = (np.asarray(m) > 0).astype(np.uint8)
        # Squeeze all singleton leading dimensions to get a 2D [H,W] mask.
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        pred_masks[idx] = arr
    return pred_masks


def eval_clip(predictor, frame_dir, davis_ann_dir, eval_start, device):
    """Run SAM2Long + compute mean J&F over eval window."""
    gt_masks, target_obj = load_davis_masks(davis_ann_dir)
    init_mask = gt_masks[0]
    pred_masks = run_sam2long_on_video(predictor, frame_dir, init_mask, device)

    j_scores, f_scores = [], []
    for t in range(eval_start, len(gt_masks)):
        if pred_masks[t] is None:
            continue
        # Resize pred to match GT if needed
        if pred_masks[t].shape != gt_masks[t].shape:
            from PIL import Image as _Im
            p_img = _Im.fromarray(pred_masks[t] * 255)
            p_img = p_img.resize((gt_masks[t].shape[1], gt_masks[t].shape[0]),
                                  _Im.NEAREST)
            pred_masks[t] = (np.array(p_img) > 0).astype(np.uint8)
        j_scores.append(db_eval_iou(pred_masks[t], gt_masks[t]))
        f_scores.append(db_eval_f(pred_masks[t], gt_masks[t]))
    mean_j = float(np.mean(j_scores)) if j_scores else float("nan")
    mean_f = float(np.mean(f_scores)) if f_scores else float("nan")
    return {"mean_j": mean_j, "mean_f": mean_f, "mean_jf": (mean_j + mean_f) / 2,
            "n_eval": len(j_scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks_dir", required=True,
                    help="Dir with videos/{vid}_{regime}/ subdirs of attacked JPEGs")
    ap.add_argument("--davis_root", default=str(ROOT / "data" / "davis"))
    ap.add_argument("--output_dir", default=str(ROOT / "results_sam2long"))
    ap.add_argument("--videos", required=True,
                    help="Comma-separated video names")
    ap.add_argument("--regimes", default="suppression,decoy",
                    help="Comma-separated regimes to evaluate")
    ap.add_argument("--sam2_cfg",
                    default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt",
                    default=str(ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sam2_baselines", type=str, default=None,
                    help="Path to SAM2 results JSON (for RetentionRatio)")
    args = ap.parse_args()

    videos = args.videos.split(",")
    regimes = args.regimes.split(",")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device,
    )

    davis_ann_root = Path(args.davis_root) / "Annotations" / "480p"
    attacks_root = Path(args.attacks_dir) / "videos"
    summary = {}

    for vid in videos:
        summary[vid] = {}
        davis_ann_dir = davis_ann_root / vid

        # Clean baseline on SAM2Long: use DAVIS JPEGImages directly
        clean_jpeg_dir = Path(args.davis_root) / "JPEGImages" / "480p" / vid
        print(f"\n=== {vid} [clean SAM2Long] ===")
        summary[vid]["clean_sam2long"] = eval_clip(
            predictor, clean_jpeg_dir, davis_ann_dir, EVAL_START, device)

        for regime in regimes:
            atk_dir = attacks_root / f"{vid}_{regime}"
            if not atk_dir.exists():
                print(f"[WARN] skip {vid}/{regime}: {atk_dir} missing")
                summary[vid][f"{regime}_sam2long"] = {"error": "missing"}
                continue
            print(f"\n=== {vid} [{regime} SAM2Long] ===")
            summary[vid][f"{regime}_sam2long"] = eval_clip(
                predictor, atk_dir, davis_ann_dir, EVAL_START, device)

    # Merge with SAM2 baselines and compute retention
    if args.sam2_baselines and Path(args.sam2_baselines).exists():
        sam2_all = json.load(open(args.sam2_baselines))
        for vid in videos:
            if vid not in sam2_all:
                continue
            s = sam2_all[vid]
            entry = summary[vid]
            if "clean" in s:
                entry["clean_sam2"] = s["clean"].get("mean_jf", float("nan"))
            for regime in regimes:
                if regime in s:
                    entry[f"{regime}_sam2"] = s[regime].get("mean_jf", float("nan"))
            # Compute drops + retention ratio
            c2 = entry.get("clean_sam2", float("nan"))
            cL = entry["clean_sam2long"]["mean_jf"]
            for regime in regimes:
                a2 = entry.get(f"{regime}_sam2", float("nan"))
                aL = entry.get(f"{regime}_sam2long", {}).get("mean_jf", float("nan"))
                if not (np.isnan(c2) or np.isnan(a2)):
                    entry[f"drop_{regime}_sam2"] = c2 - a2
                if not (np.isnan(cL) or np.isnan(aL)):
                    entry[f"drop_{regime}_sam2long"] = cL - aL
                d2 = entry.get(f"drop_{regime}_sam2")
                dL = entry.get(f"drop_{regime}_sam2long")
                if d2 is not None and dL is not None and d2 > 1e-4:
                    entry[f"retention_{regime}"] = dL / d2

    out_path = out_dir / "sam2long_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    # Print summary table
    print("\n{:>18s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
        "video", "drop_s2", "drop_sL", "ret_supp", "drop_d2", "drop_dL", ))
    for vid in videos:
        e = summary[vid]
        ds2 = e.get("drop_suppression_sam2", float("nan"))
        dsL = e.get("drop_suppression_sam2long", float("nan"))
        ret_s = e.get("retention_suppression", float("nan"))
        dd2 = e.get("drop_decoy_sam2", float("nan"))
        ddL = e.get("drop_decoy_sam2long", float("nan"))
        ret_d = e.get("retention_decoy", float("nan"))
        print(f"{vid:>18s} {ds2:>8.3f} {dsL:>8.3f} {ret_s:>8.3f} {dd2:>10.3f} {ddL:>10.3f} {ret_d:>10.3f}")


if __name__ == "__main__":
    main()
