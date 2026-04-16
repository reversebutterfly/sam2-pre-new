#!/usr/bin/env python3
"""
Two Memory-Poisoning Regimes Experiment Runner.

Runs matched-budget comparison of Suppression vs Decoy on DAVIS 2017.
Evaluates ONLY on untouched future frames f10:f14 (disjoint from attack window).
Extracts regime-specific signature metrics.

Usage:
  # Block 1: Core comparison (20 videos)
  python run_two_regimes.py --block core --device cuda:0

  # Block 2: Mechanism isolation (8 videos)
  python run_two_regimes.py --block isolation --device cuda:0

  # Single regime test
  python run_two_regimes.py --regime suppression --videos bear,dog --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.config import MemShieldConfig
from memshield.surrogate import SAM2Surrogate, get_interior_prompt
from memshield.scheduler import compute_resonance_schedule, build_modified_index_map
from memshield.generator import optimize_cooperative, select_perturb_originals
from memshield.losses import fake_uint8_quantize
from PIL import Image


# ── Dataset ──────────────────────────────────────────────────────────────────

DAVIS_20 = [
    "bear", "bike-packing", "blackswan", "bmx-bumps", "bmx-trees",
    "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-jump", "dance-twirl", "dog",
]

DAVIS_PILOT = ["bear", "car-shadow", "dance-jump", "dog", "cows"]


def load_video(davis_root, vid, max_frames=15):
    img_dir = Path(davis_root) / "JPEGImages/480p" / vid
    anno_dir = Path(davis_root) / "Annotations/480p" / vid
    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if max_frames > 0:
        stems = stems[:max_frames]
    frames, masks = [], []
    for stem in stems:
        frames.append(np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB")))
        anno = np.array(Image.open(anno_dir / f"{stem}.png"))
        masks.append((anno > 0).astype(np.uint8))
    return frames, masks


# ── Suppression generator (matched budget, uses compute_attack_loss) ─────────

def optimize_suppression(surrogate, frames_uint8, masks_uint8, schedule, cfg):
    """Suppression regime: push object_score negative + suppress GT logits."""
    from memshield.losses import compute_attack_loss, differentiable_ssim
    device = surrogate.device
    T = len(frames_uint8)
    H, W = frames_uint8[0].shape[:2]

    frames_t = [torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                for f in frames_uint8]

    idx_map = build_modified_index_map(T, schedule)
    perturb_set = select_perturb_originals(schedule, T)

    # Deltas for inserts
    insert_deltas, insert_eps, insert_bases = {}, {}, {}
    for si, slot in enumerate(schedule):
        pos = slot.after_original_idx
        idx_after = min(pos + 1, T - 1)
        base = 0.5 * (frames_t[pos] + frames_t[idx_after])
        insert_bases[si] = base.detach()
        insert_deltas[si] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[si] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    # Deltas for originals
    orig_deltas, orig_eps = {}, {}
    for oi in perturb_set:
        orig_deltas[oi] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[oi] = 2.0 / 255 if oi == 0 else 4.0 / 255

    # Eval on f10:f14
    eval_orig_indices = [i for i in range(10, min(T, 15))]
    eval_mod_indices = [idx_map["orig_to_mod"][j] for j in eval_orig_indices]

    insert_after = {}
    for si, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(si)

    n_steps = cfg.n_steps_strong
    alpha_ins = max(cfg.epsilon_strong / max(n_steps // 3, 1), 1.0 / 255)
    alpha_orig = max(4.0 / 255 / max(n_steps // 3, 1), 0.5 / 255)
    best_loss = float("inf")
    best_id, best_od = {}, {}

    for step in range(n_steps):
        for d in list(insert_deltas.values()) + list(orig_deltas.values()):
            if d.grad is not None:
                d.grad.zero_()

        mod_frames = []
        for oi in range(T):
            if oi in orig_deltas:
                f = fake_uint8_quantize((frames_t[oi] + orig_deltas[oi]).clamp(0, 1))
            else:
                f = frames_t[oi].detach()
            mod_frames.append(f)
            if oi in insert_after:
                for si in insert_after[oi]:
                    adv = fake_uint8_quantize((insert_bases[si] + insert_deltas[si]).clamp(0, 1))
                    mod_frames.append(adv)

        all_outs = surrogate.forward_video(mod_frames, masks_uint8[0])

        loss = compute_attack_loss(
            all_outs, masks_uint8, eval_mod_indices, eval_orig_indices,
            device, persistence_weighting=True,
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_id = {si: insert_deltas[si].detach().clone() for si in insert_deltas}
            best_od = {oi: orig_deltas[oi].detach().clone() for oi in orig_deltas}

        loss.backward()

        with torch.no_grad():
            for si in insert_deltas:
                if insert_deltas[si].grad is not None:
                    insert_deltas[si].data -= alpha_ins * insert_deltas[si].grad.sign()
                    insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])
            for oi in orig_deltas:
                if orig_deltas[oi].grad is not None:
                    orig_deltas[oi].data -= alpha_orig * orig_deltas[oi].grad.sign()
                    orig_deltas[oi].data.clamp_(-orig_eps[oi], orig_eps[oi])

        if step % 20 == 0 or step == n_steps - 1:
            print(f"    [supp] step {step:3d}  L={loss.item():.4f}")

    # Build output
    ins_uint8 = {}
    for si in best_id:
        arr = (insert_bases[si] + best_id[si]).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        ins_uint8[si] = np.rint(arr).clip(0, 255).astype(np.uint8)
    orig_uint8 = {}
    for oi in best_od:
        arr = (frames_t[oi] + best_od[oi]).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        orig_uint8[oi] = np.rint(arr).clip(0, 255).astype(np.uint8)

    return ins_uint8, orig_uint8


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_f_measure(pred, gt, bound_thresh=0.02):
    """Boundary F-measure following DAVIS protocol."""
    from scipy.ndimage import binary_dilation, binary_erosion
    import math

    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    h, w = pred_b.shape[-2:]
    bound_pix = max(1, math.ceil(bound_thresh * math.sqrt(h * w)))

    d = 2 * bound_pix + 1
    y, x = np.ogrid[-bound_pix:bound_pix+1, -bound_pix:bound_pix+1]
    disk = (x*x + y*y <= bound_pix*bound_pix).astype(bool)

    def boundary(m):
        if m.sum() == 0:
            return np.zeros_like(m, dtype=bool)
        return np.logical_xor(m, binary_erosion(m, disk))

    gt_bd = boundary(gt_b)
    pred_bd = boundary(pred_b)

    if gt_bd.sum() == 0 and pred_bd.sum() == 0:
        return 1.0

    gt_dil = binary_dilation(gt_bd, disk) if gt_bd.sum() > 0 else np.zeros_like(gt_bd)
    pred_dil = binary_dilation(pred_bd, disk) if pred_bd.sum() > 0 else np.zeros_like(pred_bd)

    prec = float((pred_bd & gt_dil).sum()) / max(float(pred_bd.sum()), 1e-9)
    rec = float((gt_bd & pred_dil).sum()) / max(float(gt_bd.sum()), 1e-9)

    if prec + rec < 1e-9:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def evaluate_official(
    protected_frames, masks_uint8, mod_to_orig,
    eval_frame_range, checkpoint, config, device_str,
):
    """Evaluate with official SAM2 VideoPredictor on eval_frame_range only."""
    import tempfile, shutil
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(device_str)
    tmpdir = tempfile.mkdtemp(prefix="regime_eval_")
    try:
        for i, frame in enumerate(protected_frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmpdir)
            coords, labels = get_interior_prompt(masks_uint8[0])
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                            points=coords, labels=labels)
            preds = {}
            for fi, obj_ids, masks_out in predictor.propagate_in_video(state):
                preds[fi] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        # Compute J, F only on eval frames
        j_scores, f_scores = [], []
        for mod_idx in range(len(protected_frames)):
            orig_idx = mod_to_orig[mod_idx]
            if orig_idx < 0 or orig_idx not in eval_frame_range:
                continue
            if mod_idx not in preds or orig_idx >= len(masks_uint8):
                continue

            pred = preds[mod_idx].astype(bool)
            gt = masks_uint8[orig_idx].astype(bool)

            # J (IoU)
            inter = (pred & gt).sum()
            union = (pred | gt).sum()
            j = float(inter) / max(float(union), 1e-9) if union > 0 else 1.0
            j_scores.append(j)

            # F (boundary)
            f = compute_f_measure(pred, gt)
            f_scores.append(f)

        mean_j = float(np.mean(j_scores)) if j_scores else 0.0
        mean_f = float(np.mean(f_scores)) if f_scores else 0.0
        mean_jf = 0.5 * (mean_j + mean_f)

        # Signature extraction
        scores_neg, scores_pos, collapse = 0, 0, 0
        total_eval = 0
        for mod_idx in range(len(protected_frames)):
            orig_idx = mod_to_orig[mod_idx]
            if orig_idx < 0 or orig_idx not in eval_frame_range:
                continue
            if mod_idx not in preds or orig_idx >= len(masks_uint8):
                continue
            total_eval += 1
            pred_area = preds[mod_idx].astype(bool).sum()
            gt_area = masks_uint8[orig_idx].astype(bool).sum()
            if pred_area < 0.01 * max(gt_area, 1):
                collapse += 1
                scores_neg += 1
            else:
                scores_pos += 1

        signatures = {
            "neg_score_rate": scores_neg / max(total_eval, 1),
            "pos_score_rate": scores_pos / max(total_eval, 1),
            "collapse_rate": collapse / max(total_eval, 1),
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "mean_j": mean_j, "mean_f": mean_f, "mean_jf": mean_jf,
        "j_scores": j_scores, "f_scores": f_scores,
        "signatures": signatures,
        "n_eval_frames": len(j_scores),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_regime(regime, surrogate, frames, masks, cfg, schedule):
    """Run one regime (suppression or decoy) and return protected frames + metadata."""
    T = len(frames)
    idx_map = build_modified_index_map(T, schedule)

    if regime == "suppression":
        ins_uint8, orig_uint8 = optimize_suppression(surrogate, frames, masks, schedule, cfg)
    elif regime == "decoy":
        ins_uint8, orig_uint8, _ = optimize_cooperative(surrogate, frames, masks, schedule, cfg)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Build protected video
    insert_after = {}
    for si, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(si)

    protected = []
    for oi in range(T):
        protected.append(orig_uint8.get(oi, frames[oi]))
        if oi in insert_after:
            for si in insert_after[oi]:
                if si in ins_uint8:
                    protected.append(ins_uint8[si])

    return protected, idx_map


def main():
    parser = argparse.ArgumentParser(description="Two Regimes Experiment")
    parser.add_argument("--block", choices=["core", "isolation", "single"], default="single")
    parser.add_argument("--regime", choices=["suppression", "decoy", "both"], default="both")
    parser.add_argument("--videos", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--davis_root", default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results_regimes"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.videos:
        videos = args.videos.split(",")
    elif args.block == "core":
        videos = DAVIS_20
    elif args.block == "isolation":
        videos = DAVIS_PILOT
    else:
        videos = DAVIS_PILOT

    regimes = ["suppression", "decoy"] if args.regime == "both" else [args.regime]

    cfg = MemShieldConfig(
        epsilon_strong=8.0 / 255, n_steps_strong=args.n_steps, device=args.device,
    )

    eval_frame_range = set(range(10, min(args.max_frames, 15)))

    print("=" * 70)
    print("  Two Memory-Poisoning Regimes Experiment")
    print("=" * 70)
    print(f"  Videos: {len(videos)}")
    print(f"  Regimes: {regimes}")
    print(f"  Eval window: f{min(eval_frame_range)}-f{max(eval_frame_range)}")
    print(f"  PGD steps: {args.n_steps}")
    print("=" * 70)

    device = torch.device(args.device)
    surrogate = SAM2Surrogate(args.checkpoint, args.sam2_config, device)

    all_results = {}

    for vid in videos:
        print(f"\n{'#' * 60}")
        print(f"  {vid}")
        print(f"{'#' * 60}")

        frames, masks = load_video(args.davis_root, vid, args.max_frames)
        if len(frames) < 15:
            print(f"  [skip] too few frames ({len(frames)})")
            continue

        vid_results = {}

        # Clean baseline
        print("  [clean] evaluating...")
        T = len(frames)
        clean_mod_to_orig = list(range(T))
        clean_eval = evaluate_official(
            frames, masks, clean_mod_to_orig, eval_frame_range,
            args.checkpoint, args.sam2_config, args.device)
        vid_results["clean"] = clean_eval
        print(f"  [clean] J={clean_eval['mean_j']:.4f} F={clean_eval['mean_f']:.4f} "
              f"J&F={clean_eval['mean_jf']:.4f}")

        schedule = compute_resonance_schedule(T, cfg.fifo_window, cfg.max_insertion_ratio)

        for regime in regimes:
            print(f"  [{regime}] optimizing...")
            try:
                t0 = time.time()
                protected, idx_map = run_regime(regime, surrogate, frames, masks, cfg, schedule)
                opt_time = time.time() - t0

                print(f"  [{regime}] evaluating ({len(protected)} frames)...")
                eval_r = evaluate_official(
                    protected, masks, idx_map["mod_to_orig"], eval_frame_range,
                    args.checkpoint, args.sam2_config, args.device)

                jf_drop = clean_eval["mean_jf"] - eval_r["mean_jf"]
                j_drop = clean_eval["mean_j"] - eval_r["mean_j"]

                vid_results[regime] = {
                    **eval_r,
                    "jf_drop": jf_drop,
                    "j_drop": j_drop,
                    "opt_time": opt_time,
                }
                print(f"  [{regime}] J={eval_r['mean_j']:.4f} F={eval_r['mean_f']:.4f} "
                      f"J&F={eval_r['mean_jf']:.4f}  Δ(J&F)={jf_drop:.4f}")
                print(f"            Signatures: {eval_r['signatures']}")

            except Exception as e:
                print(f"  [{regime}] ERROR: {e}")
                vid_results[regime] = {"error": str(e)}

        all_results[vid] = vid_results

        # Save incrementally
        with open(os.path.join(args.output_dir, "regimes_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TWO REGIMES SUMMARY (eval on f10:f14 only)")
    print("=" * 70)
    print(f"  {'Video':<18s} {'J&F_clean':>9s}", end="")
    for r in regimes:
        print(f" {'Δ(J&F)_'+r:>15s}", end="")
    print()
    print("-" * 70)

    for vid, vr in all_results.items():
        jf_c = vr.get("clean", {}).get("mean_jf", 0)
        print(f"  {vid:<18s} {jf_c:>9.4f}", end="")
        for r in regimes:
            d = vr.get(r, {}).get("jf_drop", 0)
            print(f" {d:>15.4f}", end="")
        print()

    print(f"\nResults saved to: {os.path.join(args.output_dir, 'regimes_results.json')}")


if __name__ == "__main__":
    main()
