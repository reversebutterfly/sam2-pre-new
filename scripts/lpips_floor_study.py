"""Fidelity floor study for MemoryShield v2 (AUTO_REVIEW Round 1 fix #1).

Addresses Codex's #1 weakness: "LPIPS budget may be infeasible by construction".

Measures three kinds of LPIPS floors before any PGD work:
  1. Natural adjacent-frame LPIPS on clean DAVIS clips —
     LPIPS(f[t], f[t-1]) averaged over the prefix. Sets the "realistic
     video" reference — if this is > 0.10, the paper's insert budget is
     naturally unreachable.
  2. ProPainter insert-base LPIPS vs predecessor, at the canonical
     schedule slots (K_ins=3, {6,12,14}) — LPIPS(base[k], f[o_after])
     with NO ν. Reveals the ProPainter-generator floor; the augmented-
     Lagrangian can only pull ν toward this floor, not below.
  3. decoy_dx sweep (dog only): LPIPS(base, f_prev) vs decoy_dx ∈
     {0, 20, 40, 60, 80, 120} at the last slot (o_after=11 for K_ins=3
     canonical, the boundary insert). Shows how fidelity scales with
     decoy aggressiveness.

Outputs runs/floor_study/floors.json with per-clip / per-measurement
LPIPS values. Reading this file lets us pick a feasible LPIPS budget
for R003.

Usage:
    conda activate memshield
    CUDA_VISIBLE_DEVICES=1 python scripts/lpips_floor_study.py \\
        --clips dog,cows,bmx-trees --decoy_sweep_clip dog
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memshield.scheduler import compute_schedule_v2
from memshield.propainter_base import create_insert_base, is_propainter_available


def load_clip_frames(davis_root: Path, clip: str, n_frames: int) -> np.ndarray:
    jpg_dir = davis_root / "JPEGImages" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    if len(jpgs) < n_frames:
        raise RuntimeError(f"{clip}: got {len(jpgs)} JPGs, need {n_frames}")
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])


def load_clip_masks(davis_root: Path, clip: str, n_frames: int) -> List[np.ndarray]:
    ann_dir = davis_root / "Annotations" / "480p" / clip
    anns = sorted(ann_dir.glob("*.png"))[:n_frames]
    return [(np.array(Image.open(p)) > 0).astype(np.uint8) for p in anns]


def build_lpips_fn(device: str):
    import lpips as _lpips
    model = _lpips.LPIPS(net="alex", verbose=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def lpips_fn(x_hwc: np.ndarray, y_hwc: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.from_numpy(x_hwc).to(device).float() / 255.0
            y = torch.from_numpy(y_hwc).to(device).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
            y = y.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
            return float(model(x, y).squeeze().item())
    return lpips_fn


def measure_natural_floor(frames: np.ndarray, lpips_fn) -> List[float]:
    """LPIPS(f[t], f[t-1]) for t=1..T-1."""
    T = frames.shape[0]
    return [lpips_fn(frames[t], frames[t - 1]) for t in range(1, T)]


def measure_propainter_floor(frames: np.ndarray, masks: List[np.ndarray],
                              K_ins: int, T_prefix: int, num_maskmem: int,
                              decoy_offset: Tuple[int, int],
                              lpips_fn) -> List[dict]:
    """Build ProPainter insert base at each canonical slot; compare to
    predecessor frame via LPIPS."""
    schedule = compute_schedule_v2(
        T_prefix_orig=T_prefix, num_maskmem=num_maskmem,
        K_ins=K_ins, variant="canonical",
    )
    out = []
    for k, slot in enumerate(schedule.slots):
        o = slot.o_after
        if o < 0 or o + 1 >= len(frames):
            out.append({"slot_k": k, "o_after": o, "error": "out_of_range"})
            continue
        result = create_insert_base(
            strategy="propainter",
            frame_prev=frames[o], frame_after=frames[o + 1],
            mask_prev=masks[o], mask_after=masks[o + 1],
            decoy_offset=decoy_offset,
            seam_dilate_px=5, safety_margin=8, feather_px=3,
        )
        if result is None:
            out.append({"slot_k": k, "o_after": o, "error": "propainter_null"})
            continue
        base, _edit = result
        lp_vs_prev = lpips_fn(base, frames[o])
        lp_vs_after = lpips_fn(base, frames[o + 1])
        # Also natural-neighbor reference for same t=o
        if o >= 1:
            lp_natural = lpips_fn(frames[o], frames[o - 1])
        else:
            lp_natural = None
        out.append({
            "slot_k": k, "o_after": o,
            "lpips_insert_vs_prev": lp_vs_prev,
            "lpips_insert_vs_after": lp_vs_after,
            "lpips_natural_at_o": lp_natural,
        })
    return out


def measure_decoy_sweep(frames: np.ndarray, masks: List[np.ndarray],
                         o_after: int, decoy_dxs: List[int],
                         lpips_fn) -> List[dict]:
    """Sweep decoy_dx at a fixed slot (the last / boundary insert)."""
    out = []
    for dx in decoy_dxs:
        result = create_insert_base(
            strategy="propainter",
            frame_prev=frames[o_after], frame_after=frames[o_after + 1],
            mask_prev=masks[o_after], mask_after=masks[o_after + 1],
            decoy_offset=(0, dx),
            seam_dilate_px=5, safety_margin=8, feather_px=3,
        )
        if result is None:
            out.append({"decoy_dx": dx, "error": "propainter_null"})
            continue
        base, _ = result
        lp = lpips_fn(base, frames[o_after])
        out.append({"decoy_dx": dx, "lpips_vs_prev": lp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="dog",
                    help="Comma-separated DAVIS clip names")
    ap.add_argument("--davis_root", default=str(REPO_ROOT / "data" / "davis"))
    ap.add_argument("--K_ins", type=int, default=3)
    ap.add_argument("--T_prefix", type=int, default=15)
    ap.add_argument("--num_maskmem", type=int, default=7)
    ap.add_argument("--decoy_dy", type=int, default=0)
    ap.add_argument("--decoy_dx", type=int, default=80)
    ap.add_argument("--decoy_sweep_clip", default="dog")
    ap.add_argument("--decoy_sweep_dxs", default="0,20,40,60,80,120")
    ap.add_argument("--out_json", default=str(REPO_ROOT / "runs" / "floor_study" / "floors.json"))
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    davis_root = Path(args.davis_root)
    clips = args.clips.split(",")
    n_frames = args.T_prefix + 7  # match R001/R002's T_full

    assert is_propainter_available(), "ProPainter not installed"
    print(f"[floors] device={device} clips={clips} n_frames={n_frames}")
    lpips_fn = build_lpips_fn(device)

    # Probe: identical inputs should give ~0
    zero_probe = lpips_fn(np.zeros((100, 100, 3), np.uint8),
                          np.zeros((100, 100, 3), np.uint8))
    print(f"[floors] lpips_fn self-check: identical inputs → {zero_probe:.4f}")

    result = {
        "config": {
            "K_ins": args.K_ins, "T_prefix": args.T_prefix,
            "num_maskmem": args.num_maskmem,
            "decoy_offset": [args.decoy_dy, args.decoy_dx],
            "n_frames_loaded": n_frames,
        },
        "per_clip": {},
        "decoy_sweep": None,
    }

    for clip in clips:
        print(f"\n=== {clip} ===")
        frames = load_clip_frames(davis_root, clip, n_frames)
        masks = load_clip_masks(davis_root, clip, n_frames)

        natural = measure_natural_floor(frames, lpips_fn)
        natural_mean = float(np.mean(natural))
        natural_max = float(np.max(natural))
        print(f"  natural LPIPS(f[t], f[t-1]):  "
              f"mean={natural_mean:.4f} max={natural_max:.4f}  "
              f"(per-t: {[f'{x:.3f}' for x in natural[:8]]}...)")

        propainter = measure_propainter_floor(
            frames, masks, args.K_ins, args.T_prefix, args.num_maskmem,
            (args.decoy_dy, args.decoy_dx), lpips_fn,
        )
        for p in propainter:
            if "error" in p:
                print(f"  ProPainter slot k={p['slot_k']}: {p['error']}")
            else:
                print(f"  ProPainter slot k={p['slot_k']} o={p['o_after']}: "
                      f"LPIPS(ins, f_prev)={p['lpips_insert_vs_prev']:.4f}  "
                      f"LPIPS(ins, f_after)={p['lpips_insert_vs_after']:.4f}")

        result["per_clip"][clip] = {
            "natural_per_t": natural,
            "natural_mean": natural_mean,
            "natural_max": natural_max,
            "propainter_floor_at_canonical_slots": propainter,
        }

    # decoy_dx sweep on the last (boundary) slot of the sweep clip
    sweep_clip = args.decoy_sweep_clip
    print(f"\n=== decoy_dx sweep on {sweep_clip} ===")
    frames_sw = load_clip_frames(davis_root, sweep_clip, n_frames)
    masks_sw = load_clip_masks(davis_root, sweep_clip, n_frames)
    # Use the last slot (boundary insert m_{K-1} = T_prefix - 1)
    schedule = compute_schedule_v2(
        T_prefix_orig=args.T_prefix, num_maskmem=args.num_maskmem,
        K_ins=args.K_ins, variant="canonical",
    )
    o_boundary = schedule.slots[-1].o_after
    print(f"  sweeping at o_after={o_boundary} (boundary slot)")
    dxs = [int(x) for x in args.decoy_sweep_dxs.split(",")]
    sweep = measure_decoy_sweep(frames_sw, masks_sw, o_boundary, dxs, lpips_fn)
    for s in sweep:
        if "error" in s:
            print(f"  dx={s['decoy_dx']}: {s['error']}")
        else:
            print(f"  dx={s['decoy_dx']}: LPIPS(base, f_prev)={s['lpips_vs_prev']:.4f}")
    result["decoy_sweep"] = {"clip": sweep_clip, "o_after": o_boundary,
                             "results": sweep}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[floors] wrote {out_path}")


if __name__ == "__main__":
    main()
