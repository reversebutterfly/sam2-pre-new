"""Evaluate a MemoryShield v2 attack run on vanilla SAM2.1.

Input:
    --run_dir runs/<run_id>/   containing
        modified_video.npy  [T_mod, H, W, 3] uint8 — modified prefix
        diagnostics.json    (provides clip, K_ins, T_mod)

The modified video is ONLY the prefix. For eval, this script concatenates
the clean DAVIS suffix (orig frames [T_prefix_orig : T_prefix_orig +
eval_window]) to form the full attacked video, writes it to a temporary
JPEG directory, runs vanilla SAM2.1 `propagate_in_video` with the DAVIS
first-frame GT as the prompt, and computes:

    * J_per_eval_frame  — length eval_window Jaccard vs DAVIS GT
    * AUC               — mean of J_per_eval_frame
    * rebound_at_1      — first u ≥ 1 where J_u ≥ max(J_0, threshold)

A clean SAM2 baseline is run on the unmodified DAVIS frames for the same
physical eval window to give `mean_j_clean` / `auc_clean`, so the J-drop
can be read off a single JSON.

Usage:
    python -m scripts.eval_memshield_v2 --run_dir runs/r001 \\
        --output_json runs/r001/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sam2"))

from memshield.eval_v2 import evaluate_run
# sam2 is imported lazily in main() so --help / argparse works in envs
# without SAM2 installed (e.g. local Windows dev box).


# -----------------------------------------------------------------------------
# DAVIS loaders (duplicated small helpers — avoids importing run_pilot_r001)
# -----------------------------------------------------------------------------


def load_davis_clip(davis_root: Path, clip: str, n_frames: int
                    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Returns (frames [N,H,W,3] uint8, gt_masks list of [H,W] uint8,
    first-frame GT mask [H,W] uint8).

    Target semantics follow `run_pilot_r001.py` / `run_pilot_r002.py`: the
    foreground is the UNION of all non-zero annotation labels, not a
    specific object id. This matters on multi-object DAVIS clips — using
    a single id here would silently evaluate against a different target
    than the one the attack was generated for.
    """
    jpg_dir = davis_root / "JPEGImages" / "480p" / clip
    ann_dir = davis_root / "Annotations" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    anns = sorted(ann_dir.glob("*.png"))[:n_frames]
    if len(jpgs) < n_frames:
        raise RuntimeError(
            f"{clip} JPEGs: {len(jpgs)} < required {n_frames}")
    if len(anns) < n_frames:
        raise RuntimeError(
            f"{clip} Annotations: {len(anns)} < required {n_frames}")
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])
    gt_masks = [(np.array(Image.open(p)) > 0).astype(np.uint8) for p in anns]
    mask0 = gt_masks[0]
    return frames, gt_masks, mask0


def stage_frames_to_tempdir(frames: np.ndarray, tmpdir: Path) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(tmpdir / f"{i:05d}.jpg", quality=95)
    return tmpdir


# -----------------------------------------------------------------------------
# SAM2 propagation
# -----------------------------------------------------------------------------


def sam2_propagate(predictor, frames: np.ndarray, mask0: np.ndarray,
                   device: str) -> List[np.ndarray]:
    """Run vanilla SAM2.1 propagate over `frames` with mask0 as prompt.

    Returns per-frame binary masks at video resolution (threshold sigmoid > 0.5).
    Uses a tempdir so we avoid polluting the DAVIS tree.
    """
    H, W = frames.shape[1], frames.shape[2]
    with tempfile.TemporaryDirectory() as td:
        stage_frames_to_tempdir(frames, Path(td))
        state = predictor.init_state(video_path=td)
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1,
            mask=torch.from_numpy(mask0).to(device).bool(),
        )
        masks: List[np.ndarray] = [None] * len(frames)
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            m = (video_res_masks[0, 0].float().sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            masks[frame_idx] = m
    for i in range(len(masks)):
        if masks[i] is None:
            masks[i] = np.zeros((H, W), dtype=np.uint8)
    return masks


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Directory with modified_video.npy + diagnostics.json")
    ap.add_argument("--davis_root", default=str(REPO_ROOT / "data" / "davis"))
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt",
                    default=str(REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--device", default=None,
                    help="Default: cuda:0 if available, else cpu")
    ap.add_argument("--clip", default=None,
                    help="Override clip name (default: read from diagnostics.json)")
    ap.add_argument("--eval_window", type=int, default=7,
                    help="Number of eval frames (default: 7, matches R001)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Rebound-at-k absolute floor on J")
    ap.add_argument("--output_json", default=None,
                    help="Default: <run_dir>/eval.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    diag_path = run_dir / "diagnostics.json"
    mod_path = run_dir / "modified_video.npy"
    assert diag_path.exists(), f"missing {diag_path}"
    assert mod_path.exists(), f"missing {mod_path}"

    diag = json.loads(diag_path.read_text())
    clip = args.clip or diag["clip"]
    K_ins = int(diag["K_ins"])
    T_mod = int(diag["schedule"]["T_mod"])
    T_prefix_orig = T_mod - K_ins
    eval_window = args.eval_window
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_json) if args.output_json else run_dir / "eval.json"

    print(f"[eval] run_dir={run_dir} clip={clip} K_ins={K_ins} "
          f"T_mod={T_mod} T_prefix_orig={T_prefix_orig} "
          f"eval_window={eval_window} device={device}")

    # --- load modified prefix + DAVIS -----------------------------------------
    mod_prefix = np.load(mod_path)                                             # [T_mod, H, W, 3] uint8
    assert mod_prefix.ndim == 4 and mod_prefix.shape[0] == T_mod, (
        f"modified_video.npy shape {mod_prefix.shape} != expected "
        f"({T_mod}, H, W, 3)")
    assert mod_prefix.dtype == np.uint8, (
        f"expected uint8, got {mod_prefix.dtype}")
    H_mod, W_mod = mod_prefix.shape[1], mod_prefix.shape[2]

    n_needed = T_prefix_orig + eval_window
    davis_root = Path(args.davis_root)
    frames_orig, gt_masks_orig, mask0_orig = load_davis_clip(
        davis_root, clip, n_frames=n_needed,
    )
    H_orig, W_orig = frames_orig.shape[1], frames_orig.shape[2]
    print(f"[eval] loaded DAVIS {clip}: frames {frames_orig.shape} "
          f"mod_prefix {mod_prefix.shape}")
    if (H_mod, W_mod) != (H_orig, W_orig):
        raise RuntimeError(
            f"modified_video HxW ({H_mod}x{W_mod}) != DAVIS ({H_orig}x{W_orig}); "
            f"run-pipeline resolution mismatch — can't concat clean suffix.")

    # --- build full attacked video --------------------------------------------
    clean_suffix = frames_orig[T_prefix_orig:T_prefix_orig + eval_window]
    attacked = np.concatenate([mod_prefix, clean_suffix], axis=0)              # [T_mod + U, H, W, 3]
    T_atk = attacked.shape[0]
    assert T_atk == T_mod + eval_window

    # --- SAM2 predictor -------------------------------------------------------
    from sam2.build_sam import build_sam2_video_predictor
    print(f"[eval] building SAM2.1 predictor ({args.sam2_cfg})")
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device)
    predictor.eval()

    # --- attacked run ---------------------------------------------------------
    print(f"[eval] propagating attacked video ({T_atk} frames) ...")
    t0 = time.time()
    pred_atk = sam2_propagate(predictor, attacked, mask0_orig, device)
    t_atk = time.time() - t0
    print(f"[eval]   attacked propagate done in {t_atk:.1f}s")

    # --- clean baseline -------------------------------------------------------
    print(f"[eval] propagating clean DAVIS video ({n_needed} frames) ...")
    t0 = time.time()
    pred_clean = sam2_propagate(predictor, frames_orig, mask0_orig, device)
    t_clean = time.time() - t0
    print(f"[eval]   clean propagate done in {t_clean:.1f}s")

    # --- align GT to attacked timeline ---------------------------------------
    # Attacked eval window = [T_mod, T_mod + eval_window)
    #                      = physical orig frames [T_prefix_orig, T_prefix_orig+eval_window).
    # Build an attacked-aligned GT list of length T_atk so evaluate_run can
    # slice [T_mod, T_mod+eval_window) directly. Prefix entries are never
    # read by the eval; fill with zeros just to keep the list dense.
    H_gt, W_gt = gt_masks_orig[0].shape
    gt_atk_aligned: List[np.ndarray] = [
        np.zeros((H_gt, W_gt), dtype=np.uint8) for _ in range(T_mod)
    ] + gt_masks_orig[T_prefix_orig:T_prefix_orig + eval_window]
    assert len(gt_atk_aligned) == T_atk

    metrics_atk = evaluate_run(
        pred_masks=pred_atk, gt_masks=gt_atk_aligned,
        T_prefix=T_mod, eval_window_size=eval_window,
        threshold=args.threshold,
    )
    metrics_clean = evaluate_run(
        pred_masks=pred_clean, gt_masks=gt_masks_orig,
        T_prefix=T_prefix_orig, eval_window_size=eval_window,
        threshold=args.threshold,
    )

    # --- write output ---------------------------------------------------------
    result = {
        "run_dir": str(run_dir),
        "clip": clip,
        "K_ins": K_ins,
        "T_mod": T_mod,
        "T_prefix_orig": T_prefix_orig,
        "eval_window": eval_window,
        "threshold": float(args.threshold),
        "attacked": metrics_atk.to_dict(),
        "clean": metrics_clean.to_dict(),
        "j_drop": float(metrics_clean.mean_j - metrics_atk.mean_j),
        "timings": {
            "attacked_sec": float(t_atk),
            "clean_sec": float(t_clean),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[eval] wrote {out_path}")

    print("\n=== summary ===")
    print(f"  clean   : AUC={metrics_clean.auc:.4f} "
          f"rebound@1={metrics_clean.rebound_at_1} "
          f"J_per_frame={[f'{x:.3f}' for x in metrics_clean.j_per_eval_frame]}")
    print(f"  attacked: AUC={metrics_atk.auc:.4f} "
          f"rebound@1={metrics_atk.rebound_at_1} "
          f"J_per_frame={[f'{x:.3f}' for x in metrics_atk.j_per_eval_frame]}")
    print(f"  J-drop (mean_J_clean - mean_J_attacked) = {result['j_drop']:+.4f}")


if __name__ == "__main__":
    main()
