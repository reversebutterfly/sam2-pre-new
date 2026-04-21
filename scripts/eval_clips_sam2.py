"""Evaluate saved JPEG sequences on SAM2 (video predictor).

Input: dir structure  {root}/videos/{vid_name}/00000.jpg, 00001.jpg, ...
For each vid_name, use DAVIS ground truth annotations at f0 as prompt, run
SAM2 video propagation, compute J&F over eval window.

Output: {output_dir}/sam2_eval.json with per-clip mean_j / mean_f / mean_jf.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sam2"))

from sam2.build_sam import build_sam2_video_predictor
from scripts.sam2long_eval import (
    load_davis_masks, db_eval_iou, db_eval_f, run_sam2long_on_video, eval_clip,
    EVAL_START,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True,
                    help="Parent of videos/ subdir with attacked JPEG seqs")
    ap.add_argument("--vid_suffix", default="_uapsam",
                    help="Suffix in saved dir names (e.g. blackswan_uapsam)")
    ap.add_argument("--davis_root", default=str(ROOT / "data" / "davis"))
    ap.add_argument("--videos", required=True)
    ap.add_argument("--sam2_cfg",
                    default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt",
                    default=str(ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device,
    )

    davis_ann_root = Path(args.davis_root) / "Annotations" / "480p"
    clips_root = Path(args.clips_dir) / "videos"
    davis_jpeg_root = Path(args.davis_root) / "JPEGImages" / "480p"
    results = {}

    for vid in args.videos.split(","):
        davis_ann_dir = davis_ann_root / vid
        results[vid] = {}

        # Clean baseline
        clean_dir = davis_jpeg_root / vid
        print(f"\n=== {vid} [clean SAM2] ===")
        results[vid]["clean"] = eval_clip(
            predictor, clean_dir, davis_ann_dir, EVAL_START, device)
        print(f"  clean J&F = {results[vid]['clean']['mean_jf']:.4f}")

        # Attacked
        atk_dir = clips_root / f"{vid}{args.vid_suffix}"
        if not atk_dir.exists():
            print(f"  [WARN] {atk_dir} missing, skipping attack eval")
            continue
        print(f"=== {vid} [attacked SAM2] ===")
        results[vid]["attacked"] = eval_clip(
            predictor, atk_dir, davis_ann_dir, EVAL_START, device)
        c = results[vid]["clean"]["mean_jf"]
        a = results[vid]["attacked"]["mean_jf"]
        results[vid]["drop"] = c - a
        print(f"  attacked J&F = {a:.4f}  drop = {c - a:+.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output_json, "w"), indent=2)
    print(f"\nWrote {args.output_json}")

    print("\n{:>18s} {:>7s} {:>7s} {:>7s}".format("video", "clean", "adv", "drop"))
    for vid, r in results.items():
        c = r.get("clean", {}).get("mean_jf", float("nan"))
        a = r.get("attacked", {}).get("mean_jf", float("nan"))
        d = r.get("drop", float("nan"))
        print(f"{vid:>18s} {c:>7.3f} {a:>7.3f} {d:>7.3f}")


if __name__ == "__main__":
    main()
