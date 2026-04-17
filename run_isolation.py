#!/usr/bin/env python3
"""
Mechanism Isolation Experiments (Block 2).

Tests whether attack effects are memory-mediated by running controlled variants:
  - benign:       insert base frames without PGD (isolates insertion effect)
  - perturb_only: PGD on originals only, no insertions (isolates perturbation)
  - insert_only:  PGD on inserts only, originals clean (isolates memory poisoning)
  - hybrid:       normal M1 attack (reference, same as run_two_regimes.py)
  - hybrid_reset: normal attack + memory reset at eval start (tests memory mediation)
  - clean_reset:  clean video + memory reset (baseline for reset comparison)

Each variant runs for both regimes on 8 eligible clips (fallback to 5 pilot).

Usage:
  # All variants on pilot videos
  python run_isolation.py --device cuda:0

  # Specific variant
  python run_isolation.py --variant perturb_only --regime suppression --device cuda:0

  # Custom video list
  python run_isolation.py --videos bear,car-roundabout,dog --device cuda:0
"""
import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.config import MemShieldConfig
from memshield.losses import (
    decoy_target_loss,
    differentiable_ssim,
    fake_uint8_quantize,
    mean_logit_loss,
    object_score_margin_loss,
    object_score_positive_loss,
)
from memshield.decoy import shift_mask
from memshield.generator import build_role_targets, select_perturb_originals
from memshield.scheduler import (
    InsertionSlot,
    build_modified_index_map,
    compute_resonance_schedule,
)
from memshield.surrogate import SAM2Surrogate, get_interior_prompt

from run_two_regimes import (
    DAVIS_VAL,
    EVAL_START,
    EVAL_END,
    load_video,
    _build_suppression_bases,
    _build_decoy_bases_and_targets,
    _to_gt,
    _resize_td,
    _supp_write,
    _supp_read,
    _decoy_write,
    _decoy_read,
    build_protected_video,
    evaluate_official,
    extract_signatures,
    compute_boundary_f,
    compute_ssim_attacked,
    optimize_unified,
)

# Pilot subset for quick tests
ISOLATION_PILOT = [
    "blackswan", "car-roundabout", "car-shadow", "cows", "dog",
]

ALL_VARIANTS = [
    "benign", "perturb_only", "insert_only",
    "hybrid", "hybrid_reset", "clean_reset",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Variant: Benign Insertion
# ══════════════════════════════════════════════════════════════════════════════

def run_benign(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
    regime: str,
    perturb_set: Set[int],
    device: torch.device,
) -> Tuple[List[np.ndarray], dict]:
    """Insert base frames at scheduled positions WITHOUT any PGD.

    Suppression bases: inpainted (object removed).
    Decoy bases: object relocated.
    No perturbation on any frame.
    """
    if regime == "suppression":
        bases_np = _build_suppression_bases(frames, masks, schedule)
    else:
        bases_np, _, _ = _build_decoy_bases_and_targets(
            frames, masks, schedule, perturb_set, device)

    # Assemble: clean originals + unoptimized base inserts
    protected = build_protected_video(frames, bases_np, {}, schedule)
    idx_map = build_modified_index_map(len(frames), schedule)
    return protected, idx_map


# ══════════════════════════════════════════════════════════════════════════════
#  Variant: Perturb-Only (no insertions)
# ══════════════════════════════════════════════════════════════════════════════

def run_perturb_only(
    surrogate: SAM2Surrogate,
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
    regime: str,
) -> Tuple[List[np.ndarray], dict]:
    """PGD on original frames only. No insertions in the video.

    Modified video has T frames (same as clean), some perturbed.
    Tests: is perturbation alone sufficient without memory poisoning?
    """
    device = surrogate.device
    T = len(frames)
    H, W = frames[0].shape[:2]

    frames_t = [
        torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        for f in frames
    ]

    # Use same perturb_set as M1 hybrid (from the original schedule)
    perturb_set = select_perturb_originals(schedule, T)

    # No inserts → identity index mapping
    idx_map = build_modified_index_map(T, [])
    eval_orig = list(range(EVAL_START, min(T, EVAL_END)))
    eval_mod = [idx_map["orig_to_mod"][j] for j in eval_orig]

    # Build regime-specific targets for decoy (suppression doesn't need them)
    decoy_targets = None
    decoy_offset = None
    if regime == "decoy":
        role_data = build_role_targets(
            masks, frames, schedule, perturb_set, device)
        decoy_targets = role_data["targets"]
        decoy_offset = role_data["decoy_offset"]

    # Init deltas for originals only
    orig_deltas, orig_eps = {}, {}
    for oi in perturb_set:
        orig_deltas[oi] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[oi] = 2.0 / 255 if oi == 0 else 4.0 / 255

    n_steps = cfg.n_steps_strong
    alpha_orig = max(4.0 / 255 / max(n_steps // 3, 1), 0.5 / 255)
    best_loss = float("inf")
    best_od = {}

    for step in range(n_steps):
        for oi in orig_deltas:
            if orig_deltas[oi].grad is not None:
                orig_deltas[oi].grad.zero_()

        # Build modified video: perturbed originals, NO inserts
        mod_frames = []
        for oi in range(T):
            if oi in orig_deltas:
                f = fake_uint8_quantize(
                    (frames_t[oi] + orig_deltas[oi]).clamp(0, 1))
            else:
                f = frames_t[oi].detach()
            mod_frames.append(f)

        all_outs = surrogate.forward_video(mod_frames, masks[0])

        # Loss (same as M1 but with empty schedule → no insert loss terms)
        # Pass REAL schedule for correct post-insert weighting on originals.
        # Insert loss terms naturally skip because idx_map has no insert_mod_indices.
        if regime == "suppression":
            lw = _supp_write(all_outs, masks, idx_map, perturb_set,
                             schedule, device)
            lr = _supp_read(all_outs, masks, eval_mod, eval_orig, device)
        else:
            lw = _decoy_write(all_outs, decoy_targets, idx_map, perturb_set,
                              schedule, device)
            lr = _decoy_read(all_outs, masks, eval_mod, eval_orig,
                             decoy_offset, device)

        # Quality loss on originals
        lq = torch.tensor(0.0, device=device)
        nq = 0
        for oi in orig_deltas:
            adv = (frames_t[oi] + orig_deltas[oi]).clamp(0, 1)
            sv = differentiable_ssim(frames_t[oi], adv)
            lq = lq + F.relu(0.97 - sv)
            nq += 1
        if nq > 0:
            lq = lq / nq

        loss = lw + 1.3 * lr + cfg.lambda_quality * lq

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_od = {oi: d.detach().clone() for oi, d in orig_deltas.items()}

        loss.backward()

        with torch.no_grad():
            for oi in orig_deltas:
                g = orig_deltas[oi].grad
                if g is not None:
                    orig_deltas[oi].data -= alpha_orig * g.sign()
                    orig_deltas[oi].data.clamp_(-orig_eps[oi], orig_eps[oi])

        if step % 10 == 0 or step == n_steps - 1:
            print(f"    [pert|{regime[:4]}] step {step:3d}  "
                  f"Lw={lw.item():.4f}  Lr={lr.item():.4f}")

    # Export: perturbed originals only, no inserts
    orig_u8 = {}
    for oi, delta in best_od.items():
        arr = (frames_t[oi] + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        orig_u8[oi] = (arr.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)

    protected = [orig_u8.get(i, frames[i]) for i in range(T)]
    return protected, idx_map


# ══════════════════════════════════════════════════════════════════════════════
#  Variant: Insert-Only (originals untouched)
# ══════════════════════════════════════════════════════════════════════════════

def run_insert_only(
    surrogate: SAM2Surrogate,
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
    regime: str,
) -> Tuple[List[np.ndarray], dict]:
    """PGD on inserts only. Originals are clean.

    Modified video has T + len(schedule) frames.
    Tests: can memory poisoning alone (via inserts) corrupt future tracking?
    """
    device = surrogate.device
    T = len(frames)
    H, W = frames[0].shape[:2]

    frames_t = [
        torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        for f in frames
    ]

    perturb_set = select_perturb_originals(schedule, T)
    idx_map = build_modified_index_map(T, schedule)
    eval_orig = list(range(EVAL_START, min(T, EVAL_END)))
    eval_mod = [idx_map["orig_to_mod"][j] for j in eval_orig]

    # Build regime-specific bases/targets
    if regime == "suppression":
        insert_bases_np = _build_suppression_bases(frames, masks, schedule)
        decoy_targets = None
        decoy_offset = None
    else:
        insert_bases_np, role_data, decoy_offset = _build_decoy_bases_and_targets(
            frames, masks, schedule, perturb_set, device)
        decoy_targets = role_data["targets"]

    # Init deltas for inserts only — NO orig_deltas
    insert_deltas, insert_eps, insert_bases_t = {}, {}, {}
    for si, slot in enumerate(schedule):
        base_np = insert_bases_np[si]
        base_t = torch.from_numpy(base_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        insert_bases_t[si] = base_t.detach()
        insert_deltas[si] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[si] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    alpha_ins = {si: max(eps / max(cfg.n_steps_strong // 3, 1), eps * 0.1)
                 for si, eps in insert_eps.items()}

    insert_after = {}
    for si, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(si)

    n_steps = cfg.n_steps_strong
    best_loss = float("inf")
    best_id = {}

    for step in range(n_steps):
        for si in insert_deltas:
            if insert_deltas[si].grad is not None:
                insert_deltas[si].grad.zero_()

        # Build video: CLEAN originals + PGD inserts
        mod_frames = []
        for oi in range(T):
            mod_frames.append(frames_t[oi].detach())
            if oi in insert_after:
                for si in insert_after[oi]:
                    adv = fake_uint8_quantize(
                        (insert_bases_t[si] + insert_deltas[si]).clamp(0, 1))
                    mod_frames.append(adv)

        all_outs = surrogate.forward_video(mod_frames, masks[0])

        # Loss (write-path only has insert terms since originals are clean)
        if regime == "suppression":
            lw = _supp_write(all_outs, masks, idx_map, set(), schedule, device)
            lr = _supp_read(all_outs, masks, eval_mod, eval_orig, device)
        else:
            lw = _decoy_write(all_outs, decoy_targets, idx_map, set(),
                              schedule, device)
            lr = _decoy_read(all_outs, masks, eval_mod, eval_orig,
                             decoy_offset, device)

        # Quality loss on inserts
        lq = torch.tensor(0.0, device=device)
        nq = 0
        for si in insert_deltas:
            adv = (insert_bases_t[si] + insert_deltas[si]).clamp(0, 1)
            sv = differentiable_ssim(insert_bases_t[si], adv)
            thresh = (cfg.ssim_threshold_strong if schedule[si].frame_type == "strong"
                      else cfg.ssim_threshold_weak)
            lq = lq + F.relu(thresh - sv)
            nq += 1
        if nq > 0:
            lq = lq / nq

        loss = lw + 1.3 * lr + cfg.lambda_quality * lq

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_id = {si: d.detach().clone() for si, d in insert_deltas.items()}

        loss.backward()

        with torch.no_grad():
            for si in insert_deltas:
                g = insert_deltas[si].grad
                if g is not None:
                    insert_deltas[si].data -= alpha_ins[si] * g.sign()
                    insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])

        if step % 10 == 0 or step == n_steps - 1:
            print(f"    [ins|{regime[:4]}] step {step:3d}  "
                  f"Lw={lw.item():.4f}  Lr={lr.item():.4f}")

    # Export
    ins_u8 = {}
    for si, delta in best_id.items():
        arr = (insert_bases_t[si] + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        ins_u8[si] = (arr.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)

    protected = build_protected_video(frames, ins_u8, {}, schedule)
    return protected, idx_map


# ══════════════════════════════════════════════════════════════════════════════
#  Variant: Memory Reset Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_reset(
    protected_frames: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    mod_to_orig: List[int],
    eval_range: set,
    checkpoint: str,
    config: str,
    device_str: str,
    reset_orig_idx: int = 10,
) -> dict:
    """Evaluate with memory reset: fresh prompt at reset_orig_idx.

    Instead of tracking from frame 0 through the attacked region, we start
    a FRESH tracking session at the eval window start. This removes all
    memory from the attack window, testing if the effect is memory-mediated.

    If reset recovers tracking, the attack is memory-mediated.
    If reset doesn't help, the attack works through other mechanisms.
    """
    import shutil
    import tempfile
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(device_str)

    # Find modified index of reset_orig_idx
    reset_mod_idx = None
    for mi, oi in enumerate(mod_to_orig):
        if oi == reset_orig_idx:
            reset_mod_idx = mi
            break
    if reset_mod_idx is None:
        return {"error": f"reset_orig_idx {reset_orig_idx} not in mod_to_orig"}

    tmpdir = tempfile.mkdtemp(prefix="reset_eval_")
    try:
        # Write only frames from reset point onwards
        reset_frames = protected_frames[reset_mod_idx:]
        for i, frame in enumerate(reset_frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmpdir)
            # SAM2Long compat: set defaults (num_pathway=1 = standard SAM2)
            state.setdefault("num_pathway", 1)
            state.setdefault("iou_thre", 0.0)
            state.setdefault("uncertainty", 0)
            # Prompt at first frame of the reset sequence using GT mask
            coords, labels = get_interior_prompt(masks_uint8[reset_orig_idx])
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=coords, labels=labels)
            preds = {}
            result = predictor.propagate_in_video(state)
            if isinstance(result, tuple):
                _, mask_list = result
                for fi in range(len(mask_list)):
                    preds[fi] = (mask_list[fi][0] > 0.0).cpu().numpy().squeeze()
            else:
                for fi, _, masks_out in result:
                    preds[fi] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        # Map reset-relative indices back to original indices.
        # SKIP the prompt frame (oi == reset_orig_idx) — it would have
        # near-perfect prediction and inflate the reset score.
        j_scores, f_scores = [], []
        for ri in range(len(reset_frames)):
            global_mi = reset_mod_idx + ri
            if global_mi >= len(mod_to_orig):
                continue
            oi = mod_to_orig[global_mi]
            if oi < 0 or oi not in eval_range or oi == reset_orig_idx:
                continue
            if ri not in preds or oi >= len(masks_uint8):
                continue
            pred = preds[ri].astype(bool)
            gt = masks_uint8[oi].astype(bool)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Main Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mechanism Isolation (Block 2)")
    parser.add_argument("--variant", nargs="+",
                        choices=ALL_VARIANTS + ["all"],
                        default=["all"],
                        help="Which variants to run")
    parser.add_argument("--regime", choices=["suppression", "decoy", "both"],
                        default="both")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video names")
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--davis_root",
                        default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint",
                        default=os.path.join(ROOT, "checkpoints",
                                             "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config",
                        default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output_dir",
                        default=os.path.join(ROOT, "results_isolation"))
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible PGD")
    parser.add_argument("--full", action="store_true",
                        help="Use all 14 eligible videos instead of 8")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.videos:
        videos = args.videos.split(",")
    elif args.full:
        videos = DAVIS_VAL
    else:
        videos = ISOLATION_PILOT
    variants = ALL_VARIANTS if "all" in args.variant else args.variant
    regimes = (["suppression", "decoy"] if args.regime == "both"
               else [args.regime])
    cfg = MemShieldConfig(
        epsilon_strong=8.0 / 255,
        n_steps_strong=args.n_steps,
        device=args.device,
    )
    eval_range = set(range(EVAL_START, min(args.max_frames, EVAL_END)))

    print("=" * 70)
    print("  Mechanism Isolation Experiments (Block 2)")
    print("=" * 70)
    print(f"  Videos:   {len(videos)}")
    print(f"  Variants: {variants}")
    print(f"  Seed:     {args.seed}")
    print(f"  Regimes:  {regimes}")
    print(f"  Steps:    {args.n_steps}")
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
            print(f"  [skip] {len(frames)} frames < 15")
            continue

        T = len(frames)
        vid_results = {}
        schedule = compute_resonance_schedule(
            T, cfg.fifo_window, cfg.max_insertion_ratio)
        perturb_set = select_perturb_originals(schedule, T)
        full_idx_map = build_modified_index_map(T, schedule)

        # Clean baseline (always needed)
        print("  [clean] evaluating...")
        clean_eval = evaluate_official(
            frames, masks, list(range(T)), eval_range,
            args.checkpoint, args.sam2_config, args.device)
        vid_results["clean"] = clean_eval
        print(f"  [clean] J&F={clean_eval['mean_jf']:.4f}")

        # Clean + reset baseline (for reset comparison)
        if "clean_reset" in variants or "hybrid_reset" in variants:
            print("  [clean_reset] evaluating...")
            cr_eval = evaluate_with_reset(
                frames, masks, list(range(T)), eval_range,
                args.checkpoint, args.sam2_config, args.device,
                reset_orig_idx=EVAL_START)
            vid_results["clean_reset"] = cr_eval
            print(f"  [clean_reset] J&F={cr_eval['mean_jf']:.4f}")

        for regime in regimes:
            for variant in variants:
                if variant == "clean_reset":
                    continue  # Already computed above
                key = f"{regime}_{variant}"
                print(f"  [{key}] running...")

                try:
                    # Fixed seed for reproducible PGD
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed)

                    t0 = time.time()

                    if variant == "benign":
                        protected, idx_map = run_benign(
                            frames, masks, schedule, cfg, regime,
                            perturb_set, device)

                    elif variant == "perturb_only":
                        protected, idx_map = run_perturb_only(
                            surrogate, frames, masks, schedule, cfg, regime)

                    elif variant == "insert_only":
                        protected, idx_map = run_insert_only(
                            surrogate, frames, masks, schedule, cfg, regime)

                    elif variant == "hybrid":
                        ins_u8, orig_u8, _ = optimize_unified(
                            surrogate, frames, masks, schedule, cfg, regime)
                        protected = build_protected_video(
                            frames, ins_u8, orig_u8, schedule)
                        idx_map = full_idx_map

                    elif variant == "hybrid_reset":
                        ins_u8, orig_u8, _ = optimize_unified(
                            surrogate, frames, masks, schedule, cfg, regime)
                        protected = build_protected_video(
                            frames, ins_u8, orig_u8, schedule)
                        idx_map = full_idx_map

                    else:
                        print(f"    Unknown variant: {variant}")
                        continue

                    opt_time = time.time() - t0

                    # Evaluate
                    if variant == "hybrid_reset":
                        ev = evaluate_with_reset(
                            protected, masks, idx_map["mod_to_orig"],
                            eval_range, args.checkpoint, args.sam2_config,
                            args.device, reset_orig_idx=EVAL_START)
                    else:
                        mod_to_orig = (idx_map["mod_to_orig"]
                                       if isinstance(idx_map, dict)
                                       else idx_map)
                        ev = evaluate_official(
                            protected, masks, mod_to_orig, eval_range,
                            args.checkpoint, args.sam2_config, args.device)

                    djf = clean_eval["mean_jf"] - ev["mean_jf"]

                    vid_results[key] = {
                        **ev,
                        "jf_drop": djf,
                        "opt_time": opt_time,
                        "variant": variant,
                        "regime": regime,
                    }
                    print(f"  [{key}] J&F={ev['mean_jf']:.4f}  "
                          f"drop={djf:.4f}  time={opt_time:.0f}s")

                except Exception as e:
                    traceback.print_exc()
                    vid_results[key] = {"error": str(e)}

        all_results[vid] = vid_results

        # Save incrementally
        out_path = os.path.join(args.output_dir, "isolation_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MECHANISM ISOLATION SUMMARY")
    print("=" * 70)

    # Collect per-variant means
    for regime in regimes:
        print(f"\n  --- {regime} ---")
        for variant in variants:
            if variant == "clean_reset":
                drops = [vr.get("clean_reset", {}).get("mean_jf", 0)
                         - vr.get("clean", {}).get("mean_jf", 0)
                         for vr in all_results.values()
                         if "clean_reset" in vr and "clean" in vr]
                if drops:
                    print(f"  clean_reset: mean dJF from clean = "
                          f"{np.mean(drops):.4f}")
                continue

            key_pattern = f"{regime}_{variant}"
            drops = [vr[key_pattern]["jf_drop"]
                     for vr in all_results.values()
                     if key_pattern in vr
                     and isinstance(vr[key_pattern], dict)
                     and "jf_drop" in vr[key_pattern]]
            if drops:
                print(f"  {variant}: mean dJF={np.mean(drops):.4f}  "
                      f"n={len(drops)}")

    print(f"\nResults: {os.path.join(args.output_dir, 'isolation_results.json')}")


if __name__ == "__main__":
    main()
