#!/usr/bin/env python3
"""
MemoryShield Ablation Study: Full DAVIS + all variants.

Runs 5 ablation conditions on the DAVIS 2017 dataset:
  1. clean        — no attack (upper bound for SAM2)
  2. perturb-only — perturb 7 original frames, NO insertions
  3. insert-only  — insert 2 frames, NO original perturbation
  4. hybrid       — insert 2 + perturb 7 (full MemoryShield)
  5. random-sched — hybrid but with random insertion positions (not FIFO-resonant)

Usage:
  python run_ablation.py --device cuda:4 --max_frames 15 --n_steps 50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.config import MemShieldConfig
from memshield.surrogate import SAM2Surrogate
from memshield.shield import protect_video, evaluate_protection
from memshield.scheduler import (
    compute_resonance_schedule, merge_event_triggers,
    build_modified_index_map, InsertionSlot,
)
from memshield.analyzer import analyze_video
from memshield.generator import optimize_hybrid, select_perturb_originals
from PIL import Image


DAVIS_20 = [
    "bear", "bike-packing", "blackswan", "bmx-bumps", "bmx-trees",
    "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-jump", "dance-twirl", "dog",
]


def load_video(davis_root, vid, max_frames=15):
    img_dir = Path(davis_root) / "JPEGImages/480p" / vid
    anno_dir = Path(davis_root) / "Annotations/480p" / vid
    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if max_frames > 0:
        stems = stems[:max_frames]
    frames, masks = [], []
    for stem in stems:
        f = np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB"))
        a = np.array(Image.open(anno_dir / f"{stem}.png"))
        frames.append(f)
        masks.append((a > 0).astype(np.uint8))
    return frames, masks


def run_clean_eval(frames, masks, checkpoint, config, device_str):
    """Evaluate SAM2 on clean video (no attack)."""
    meta = {
        "idx_map": {
            "mod_to_orig": list(range(len(frames))),
            "orig_to_mod": list(range(len(frames))),
            "insert_mod_indices": [],
            "n_modified": len(frames),
        },
        "n_original": len(frames),
    }
    return evaluate_protection(frames, masks, meta, checkpoint, config, device_str)


def run_perturb_only(surrogate, frames, masks, cfg):
    """Perturb originals only — no frame insertion."""
    # Create empty schedule (no insertions)
    schedule = []
    # But we still want to perturb originals around where insertions WOULD be
    resonance = compute_resonance_schedule(len(frames), cfg.fifo_window, cfg.max_insertion_ratio)
    perturb_set = select_perturb_originals(resonance, len(frames))

    device = surrogate.device
    T = len(frames)
    H, W = frames[0].shape[:2]

    frames_t = [torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                for f in frames]

    from memshield.losses import compute_attack_loss, differentiable_ssim, fake_uint8_quantize

    # Only original frame deltas
    orig_deltas = {}
    orig_eps = {}
    for orig_idx in perturb_set:
        orig_deltas[orig_idx] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[orig_idx] = 2.0 / 255 if orig_idx == 0 else 4.0 / 255

    all_params = list(orig_deltas.values())
    if not all_params:
        return frames, {}

    # Eval setup
    eval_orig_indices = list(range(2, min(T, 2 + cfg.future_horizon)))
    eval_mod_indices = eval_orig_indices  # No insertions, so mod == orig

    n_steps = cfg.n_steps_strong
    pgd_alpha = max(4.0 / 255 / max(n_steps // 3, 1), 0.5 / 255)

    best_loss = float("inf")
    best_deltas = {}

    for step in range(n_steps):
        for d in all_params:
            if d.grad is not None:
                d.grad.zero_()

        mod_frames = []
        for orig_idx in range(T):
            if orig_idx in orig_deltas:
                frame = (frames_t[orig_idx] + orig_deltas[orig_idx]).clamp(0.0, 1.0)
                frame = fake_uint8_quantize(frame)
            else:
                frame = frames_t[orig_idx].detach()
            mod_frames.append(frame)

        all_logits = surrogate.forward_video(mod_frames, masks[0])

        loss_attack = compute_attack_loss(
            all_logits, masks, eval_mod_indices, eval_orig_indices,
            device, cfg.persistence_weighting,
        )

        loss_total = loss_attack
        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_deltas = {oi: orig_deltas[oi].detach().clone() for oi in orig_deltas}

        loss_total.backward()

        with torch.no_grad():
            for orig_idx in orig_deltas:
                if orig_deltas[orig_idx].grad is not None:
                    orig_deltas[orig_idx].data -= pgd_alpha * orig_deltas[orig_idx].grad.sign()
                    orig_deltas[orig_idx].data.clamp_(-orig_eps[orig_idx], orig_eps[orig_idx])

        if step % 20 == 0 or step == n_steps - 1:
            print(f"    [perturb-only] step {step:3d}  L={loss_attack.item():.2f}")

    # Apply best deltas
    protected = []
    for orig_idx in range(T):
        if orig_idx in best_deltas:
            adv = (frames_t[orig_idx] + best_deltas[orig_idx]).clamp(0.0, 1.0)
            arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            protected.append(np.rint(arr).clip(0, 255).astype(np.uint8))
        else:
            protected.append(frames[orig_idx])

    meta = {
        "idx_map": {
            "mod_to_orig": list(range(T)),
            "orig_to_mod": list(range(T)),
            "insert_mod_indices": [],
            "n_modified": T,
        },
        "n_original": T,
    }
    return protected, meta


def run_insert_only(surrogate, frames, masks, cfg):
    """Insert frames only — no original perturbation."""
    from memshield.generator import optimize_hybrid, interpolate_base_frame
    from memshield.losses import compute_attack_loss, differentiable_ssim, fake_uint8_quantize

    T = len(frames)
    schedule = compute_resonance_schedule(T, cfg.fifo_window, cfg.max_insertion_ratio)
    idx_map = build_modified_index_map(T, schedule)

    device = surrogate.device
    H, W = frames[0].shape[:2]
    frames_t = [torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                for f in frames]

    # Only insertion deltas (no orig perturbation)
    insert_base = {}
    insert_deltas = {}
    insert_eps = {}
    for slot_i, slot in enumerate(schedule):
        base = interpolate_base_frame(frames_t, slot.after_original_idx)
        insert_base[slot_i] = base.detach()
        insert_deltas[slot_i] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[slot_i] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    first_insert = schedule[0].after_original_idx if schedule else 1
    eval_orig_indices = list(range(first_insert + 1, min(T, first_insert + 1 + cfg.future_horizon)))
    eval_mod_indices = [idx_map["orig_to_mod"][j] for j in eval_orig_indices]

    insert_after = {}
    for slot_i, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(slot_i)

    n_steps = cfg.n_steps_strong
    pgd_alpha = max(cfg.epsilon_strong / max(n_steps // 3, 1), 1.0 / 255)
    best_loss = float("inf")
    best_deltas = {}

    for step in range(n_steps):
        for si in insert_deltas:
            if insert_deltas[si].grad is not None:
                insert_deltas[si].grad.zero_()

        mod_frames = []
        for orig_idx in range(T):
            mod_frames.append(frames_t[orig_idx].detach())
            if orig_idx in insert_after:
                for si in insert_after[orig_idx]:
                    adv = (insert_base[si] + insert_deltas[si]).clamp(0.0, 1.0)
                    adv = fake_uint8_quantize(adv)
                    mod_frames.append(adv)

        all_logits = surrogate.forward_video(mod_frames, masks[0])
        loss = compute_attack_loss(all_logits, masks, eval_mod_indices, eval_orig_indices,
                                   device, cfg.persistence_weighting)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_deltas = {si: insert_deltas[si].detach().clone() for si in insert_deltas}

        loss.backward()
        with torch.no_grad():
            for si in insert_deltas:
                if insert_deltas[si].grad is not None:
                    insert_deltas[si].data -= pgd_alpha * insert_deltas[si].grad.sign()
                    insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])

        if step % 20 == 0 or step == n_steps - 1:
            print(f"    [insert-only] step {step:3d}  L={loss.item():.2f}")

    # Build protected video
    protected = []
    for orig_idx in range(T):
        protected.append(frames[orig_idx])
        if orig_idx in insert_after:
            for si in insert_after[orig_idx]:
                if si in best_deltas:
                    adv = (insert_base[si] + best_deltas[si]).clamp(0.0, 1.0)
                    arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                    protected.append(np.rint(arr).clip(0, 255).astype(np.uint8))

    meta = {
        "idx_map": idx_map,
        "n_original": T,
    }
    return protected, meta


def main():
    parser = argparse.ArgumentParser(description="MemoryShield Ablation Study")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--davis_root", default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results_ablation"))
    parser.add_argument("--videos", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--conditions", type=str, default="clean,perturb_only,insert_only,hybrid",
                        help="Comma-separated conditions to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    videos = args.videos.split(",") if args.videos else DAVIS_20
    conditions = args.conditions.split(",")

    cfg = MemShieldConfig(
        epsilon_strong=8.0 / 255, n_steps_strong=args.n_steps,
        device=args.device,
    )

    print("=" * 70)
    print("  MemoryShield Ablation Study")
    print("=" * 70)
    print(f"  Videos: {len(videos)}")
    print(f"  Frames: {args.max_frames}")
    print(f"  PGD steps: {args.n_steps}")
    print(f"  Conditions: {conditions}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    device = torch.device(args.device)
    surrogate = SAM2Surrogate(args.checkpoint, args.sam2_config, device)

    all_results = {}

    for vid in videos:
        print(f"\n{'#' * 70}")
        print(f"  {vid}")
        print(f"{'#' * 70}")

        frames, masks = load_video(args.davis_root, vid, args.max_frames)
        if len(frames) < 5:
            print(f"  [skip] too few frames ({len(frames)})")
            continue

        vid_results = {}

        # ── Clean baseline ───────────────────────────────────────────────
        if "clean" in conditions:
            print(f"  [clean] evaluating...")
            try:
                clean_eval = run_clean_eval(
                    frames, masks, args.checkpoint, args.sam2_config, args.device)
                vid_results["clean_j"] = clean_eval["mean_j"]
                print(f"  [clean] J = {clean_eval['mean_j']:.4f}")
            except Exception as e:
                print(f"  [clean] error: {e}")
                vid_results["clean_j"] = None

        # ── Perturb-only ─────────────────────────────────────────────────
        if "perturb_only" in conditions:
            print(f"  [perturb-only] optimizing...")
            try:
                prot, meta = run_perturb_only(surrogate, frames, masks, cfg)
                eval_r = evaluate_protection(
                    prot, masks, meta, args.checkpoint, args.sam2_config, args.device)
                vid_results["perturb_only_j"] = eval_r["mean_j"]
                print(f"  [perturb-only] J = {eval_r['mean_j']:.4f}")
            except Exception as e:
                print(f"  [perturb-only] error: {e}")
                vid_results["perturb_only_j"] = None

        # ── Insert-only ──────────────────────────────────────────────────
        if "insert_only" in conditions:
            print(f"  [insert-only] optimizing...")
            try:
                prot, meta = run_insert_only(surrogate, frames, masks, cfg)
                eval_r = evaluate_protection(
                    prot, masks, meta, args.checkpoint, args.sam2_config, args.device)
                vid_results["insert_only_j"] = eval_r["mean_j"]
                print(f"  [insert-only] J = {eval_r['mean_j']:.4f}")
            except Exception as e:
                print(f"  [insert-only] error: {e}")
                vid_results["insert_only_j"] = None

        # ── Hybrid (full MemoryShield) ───────────────────────────────────
        if "hybrid" in conditions:
            print(f"  [hybrid] optimizing...")
            try:
                prot, meta = protect_video(frames, masks, surrogate, cfg, vid)
                eval_r = evaluate_protection(
                    prot, masks, meta, args.checkpoint, args.sam2_config, args.device)
                vid_results["hybrid_j"] = eval_r["mean_j"]
                print(f"  [hybrid] J = {eval_r['mean_j']:.4f}")
            except Exception as e:
                print(f"  [hybrid] error: {e}")
                vid_results["hybrid_j"] = None

        # ── Compute J_drops ──────────────────────────────────────────────
        clean_j = vid_results.get("clean_j")
        for k in ["perturb_only_j", "insert_only_j", "hybrid_j"]:
            val = vid_results.get(k)
            if clean_j is not None and val is not None:
                vid_results[k.replace("_j", "_drop")] = clean_j - val

        all_results[vid] = vid_results
        print(f"  Results: {vid_results}")

        # Save incrementally
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    print(f"  {'Video':<18s} {'J_clean':>8s} {'Pert-only':>10s} {'Ins-only':>10s} {'Hybrid':>10s}")
    print("-" * 70)

    drops = {"perturb_only": [], "insert_only": [], "hybrid": []}
    for vid, r in all_results.items():
        jc = f"{r.get('clean_j', 0):.4f}" if r.get("clean_j") is not None else "N/A"
        dp = f"{r.get('perturb_only_drop', 0):.4f}" if r.get("perturb_only_drop") is not None else "N/A"
        di = f"{r.get('insert_only_drop', 0):.4f}" if r.get("insert_only_drop") is not None else "N/A"
        dh = f"{r.get('hybrid_drop', 0):.4f}" if r.get("hybrid_drop") is not None else "N/A"
        print(f"  {vid:<18s} {jc:>8s} {dp:>10s} {di:>10s} {dh:>10s}")

        for key in drops:
            v = r.get(f"{key}_drop")
            if v is not None:
                drops[key].append(v)

    print("-" * 70)
    for key in drops:
        mean = np.mean(drops[key]) if drops[key] else 0
        print(f"  MEAN {key:<12s}: J_drop = {mean:.4f}")

    print(f"\nResults saved to: {os.path.join(args.output_dir, 'ablation_results.json')}")


if __name__ == "__main__":
    main()
