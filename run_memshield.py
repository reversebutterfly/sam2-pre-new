#!/usr/bin/env python3
"""
MemoryShield: Protect video datasets against SAM2 segmentation.

Usage:
  # Quick pilot (5 videos, 30 frames each)
  python run_memshield.py --mode pilot --device cuda:0

  # Full DAVIS run
  python run_memshield.py --mode full --device cuda:0

  # Single video
  python run_memshield.py --videos bear --max_frames 50 --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.config import MemShieldConfig
from memshield.surrogate import SAM2Surrogate
from memshield.shield import protect_video, evaluate_protection


# ── DAVIS dataset loader ─────────────────────────────────────────────────────

def load_davis_video(davis_root, video_name, resolution="480p", max_frames=-1):
    """Load frames and masks for one DAVIS video."""
    from PIL import Image

    img_dir = Path(davis_root) / "JPEGImages" / resolution / video_name
    anno_dir = Path(davis_root) / "Annotations" / resolution / video_name

    stems = sorted(p.stem for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg"))
    if max_frames > 0:
        stems = stems[:max_frames]

    frames, masks = [], []
    for stem in stems:
        frame = np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB"))
        frames.append(frame)

        anno_path = anno_dir / f"{stem}.png"
        if anno_path.exists():
            anno = np.array(Image.open(anno_path))
            mask = (anno > 0).astype(np.uint8)  # All objects as foreground
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        masks.append(mask)

    return frames, masks, stems


# ── Video lists ──────────────────────────────────────────────────────────────

DAVIS_PILOT = ["bear", "breakdance", "car-shadow", "dance-jump", "dog"]
DAVIS_FULL = [
    "bear", "bike-packing", "blackswan", "bmx-bumps", "bmx-trees",
    "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-jump", "dance-twirl", "dog",
]


def main():
    parser = argparse.ArgumentParser(description="MemoryShield: protect videos from SAM2")
    parser.add_argument("--mode", choices=["pilot", "full", "custom"], default="pilot")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video names (overrides --mode)")
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--davis_root", type=str, default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT, "results_memshield"))

    # MemoryShield hyperparameters
    parser.add_argument("--fifo_window", type=int, default=7)
    parser.add_argument("--max_insertion_ratio", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="L∞ perturbation budget (in /255)")
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda_quality", type=float, default=5.0)
    parser.add_argument("--codec_in_loop", action="store_true")
    parser.add_argument("--no_occlusion", action="store_true")
    parser.add_argument("--no_topology", action="store_true")

    # Evaluation
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip official SAM2 evaluation (faster)")

    # Baselines
    parser.add_argument("--baseline", choices=["none", "random", "periodic"],
                        default="none",
                        help="Run a baseline schedule instead of resonance")

    args = parser.parse_args()

    # Build video list
    if args.videos:
        videos = args.videos.split(",")
    elif args.mode == "pilot":
        videos = DAVIS_PILOT
    elif args.mode == "full":
        videos = DAVIS_FULL
    else:
        videos = DAVIS_PILOT

    # Build config
    cfg = MemShieldConfig(
        fifo_window=args.fifo_window,
        max_insertion_ratio=args.max_insertion_ratio,
        epsilon_strong=args.epsilon / 255.0,
        n_steps_strong=args.n_steps,
        lr=args.lr,
        lambda_quality=args.lambda_quality,
        codec_in_loop=args.codec_in_loop,
        enable_occlusion_ghost=not args.no_occlusion,
        enable_topology_seed=not args.no_topology,
        device=args.device,
    )

    print("=" * 60)
    print("  MemoryShield — Adversarial Frame Insertion")
    print("=" * 60)
    print(f"  Videos: {videos}")
    print(f"  FIFO window: {cfg.fifo_window}")
    print(f"  Max insertion ratio: {cfg.max_insertion_ratio:.0%}")
    print(f"  Epsilon: {args.epsilon}/255")
    print(f"  PGD steps: {cfg.n_steps_strong}")
    print(f"  Device: {args.device}")
    print(f"  Baseline: {args.baseline}")
    print("=" * 60)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize SAM2 surrogate
    print("\nLoading SAM2 surrogate...")
    device = torch.device(args.device)
    surrogate = SAM2Surrogate(args.checkpoint, args.sam2_config, device)
    print(f"  FIFO bank size: {surrogate.num_maskmem}")

    # Process each video
    all_results = {}
    for vid_name in videos:
        print(f"\n{'#'*60}")
        print(f"  Processing: {vid_name}")
        print(f"{'#'*60}")

        # Load video
        frames, masks, stems = load_davis_video(
            args.davis_root, vid_name, max_frames=args.max_frames,
        )
        if len(frames) < 5:
            print(f"  [skip] {vid_name}: too few frames ({len(frames)})")
            continue

        # Protect
        t_start = time.time()
        protected, meta = protect_video(
            frames, masks, surrogate, cfg, video_name=vid_name,
        )
        protect_time = time.time() - t_start

        # Evaluate with official SAM2
        eval_results = {}
        if not args.skip_eval:
            print("\n  Evaluating with official SAM2 predictor...")
            try:
                eval_results = evaluate_protection(
                    protected, masks, meta,
                    args.checkpoint, args.sam2_config, args.device,
                )
                print(f"  Mean J (protected): {eval_results['mean_j']:.4f}")
            except Exception as e:
                print(f"  [error] Evaluation failed: {e}")
                eval_results = {"error": str(e)}

        # Also evaluate clean baseline
        clean_eval = {}
        if not args.skip_eval:
            print("  Evaluating clean baseline...")
            try:
                clean_meta = {
                    "idx_map": {
                        "mod_to_orig": list(range(len(frames))),
                        "orig_to_mod": list(range(len(frames))),
                        "insert_mod_indices": [],
                        "n_modified": len(frames),
                    },
                    "n_original": len(frames),
                }
                clean_eval = evaluate_protection(
                    frames, masks, clean_meta,
                    args.checkpoint, args.sam2_config, args.device,
                )
                print(f"  Mean J (clean):     {clean_eval['mean_j']:.4f}")
            except Exception as e:
                print(f"  [error] Clean eval failed: {e}")
                clean_eval = {"error": str(e)}

        # Compute J drop
        j_drop = 0.0
        if "mean_j" in clean_eval and "mean_j" in eval_results:
            j_drop = clean_eval["mean_j"] - eval_results["mean_j"]
            print(f"  ** J drop: {j_drop:.4f} **")

        all_results[vid_name] = {
            "n_original": len(frames),
            "n_protected": len(protected),
            "insertion_ratio": meta["insertion_ratio"],
            "protect_time_sec": protect_time,
            "clean_j": clean_eval.get("mean_j"),
            "protected_j": eval_results.get("mean_j"),
            "j_drop": j_drop,
            "schedule": meta["schedule"],
            "gen_metrics": {
                k: v for k, v in meta["gen_metrics"].items()
                if k != "history"
            },
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Video':<20s} {'Frames':>6s} {'Ins%':>5s} "
          f"{'J_clean':>8s} {'J_prot':>8s} {'J_drop':>8s}")
    print("-" * 60)

    j_drops = []
    for vid, r in all_results.items():
        j_c = f"{r['clean_j']:.4f}" if r['clean_j'] is not None else "N/A"
        j_p = f"{r['protected_j']:.4f}" if r['protected_j'] is not None else "N/A"
        j_d = f"{r['j_drop']:.4f}" if r['j_drop'] else "N/A"
        print(f"  {vid:<20s} {r['n_original']:>6d} {r['insertion_ratio']:>5.1%} "
              f"{j_c:>8s} {j_p:>8s} {j_d:>8s}")
        if r['j_drop'] is not None:
            j_drops.append(r['j_drop'])

    if j_drops:
        print("-" * 60)
        print(f"  {'MEAN':<20s} {'':>6s} {'':>5s} "
              f"{'':>8s} {'':>8s} {np.mean(j_drops):>8.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
