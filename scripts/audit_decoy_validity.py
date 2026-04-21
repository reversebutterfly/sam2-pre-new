"""A1 audit: decoy-validity + motion alignment for each DAVIS clip.

For each video, we replicate the exact offset that find_decoy_region would pick
at f0, then ask two questions using only the clean annotations:

  1. overlap = mean_{t in eval window} IoU( shift(GT_t, dy, dx), GT_t )
     > 0.25 -> invalid decoy (shifted mask overlaps real future GT)
     0.15-0.25 -> borderline
     < 0.15 -> viable

  2. motion_cos = cos( (dx, dy), mean_velocity_f0_to_f14 )
     |cos| > 0.7 -> decoy direction aligns with object motion

No GPU needed.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memshield.decoy import find_decoy_region, shift_mask

DAVIS_ROOT = ROOT / "data" / "davis"
VIDEOS = [
    "bike-packing", "blackswan", "bmx-trees", "breakdance", "camel",
    "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog",
]
EVAL_START = 10


def load_clip(vid):
    img_dir = DAVIS_ROOT / "JPEGImages" / "480p" / vid
    ann_dir = DAVIS_ROOT / "Annotations" / "480p" / vid
    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    frames, masks = [], []
    target_obj = None
    for i, stem in enumerate(stems):
        frame = np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB"))
        ann = np.array(Image.open(ann_dir / f"{stem}.png"))
        if i == 0:
            ids = sorted(set(ann.flat) - {0})
            target_obj = ids[0] if ids else 1
        masks.append((ann == target_obj).astype(np.uint8))
        frames.append(frame)
    return frames, masks


def mask_iou(a, b):
    a = a > 0
    b = b > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / max(1, float(union))


def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def audit(vid):
    frames, masks = load_clip(vid)
    T = len(masks)

    # Replicate the find_decoy_region call used in build_role_targets:
    # ref_idx = min(1, T-1) unless mask_f1 is too small, fallback to f0.
    ref_idx = min(1, T - 1)
    if masks[ref_idx].sum() < 100 and masks[0].sum() > 100:
        ref_idx = 0
    _, offset, is_distractor = find_decoy_region(masks[ref_idx], frames[ref_idx], 0.75)
    dy, dx = offset

    # Overlap over eval window
    overlaps = []
    for t in range(EVAL_START, T):
        if masks[t].sum() == 0:
            continue
        shifted = shift_mask(masks[t], dy, dx)
        overlaps.append(mask_iou(shifted, masks[t]))
    mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")

    # Motion over attack window f0:f14 (or shorter if clip is short)
    end = min(14, T - 1)
    cents = [centroid(masks[t]) for t in range(0, end + 1) if masks[t].sum() > 0]
    cents = [c for c in cents if c is not None]
    if len(cents) >= 2:
        vy = (cents[-1][0] - cents[0][0]) / max(1, len(cents) - 1)
        vx = (cents[-1][1] - cents[0][1]) / max(1, len(cents) - 1)
        off_norm = float(np.hypot(dy, dx))
        vel_norm = float(np.hypot(vy, vx))
        if off_norm > 0 and vel_norm > 0:
            cos = (dy * vy + dx * vx) / (off_norm * vel_norm)
        else:
            cos = float("nan")
    else:
        vy = vx = cos = float("nan")

    # Verdict
    if np.isnan(mean_overlap):
        verdict = "N/A"
    elif mean_overlap > 0.25:
        verdict = "INVALID"
    elif mean_overlap > 0.15:
        verdict = "borderline"
    else:
        verdict = "viable"

    if not np.isnan(cos) and abs(cos) > 0.7:
        verdict += " + motion-aligned"

    return {
        "vid": vid,
        "T": T,
        "offset": offset,
        "obj_bbox_hw": tuple(int(x) for x in (masks[ref_idx].any(axis=1).sum(), masks[ref_idx].any(axis=0).sum())),
        "mean_overlap": mean_overlap,
        "motion": (vy, vx),
        "cos": cos,
        "is_distractor": is_distractor,
        "verdict": verdict,
    }


def main():
    print(f"{'video':18s} {'T':>3s} {'offset(dy,dx)':>14s} {'bbox hw':>10s} "
          f"{'overlap':>8s} {'vel(dy,dx)':>14s} {'cos':>7s} {'distr':>6s} verdict")
    print("-" * 110)
    for vid in VIDEOS:
        r = audit(vid)
        off = f"({r['offset'][0]:+d},{r['offset'][1]:+d})"
        bbx = f"{r['obj_bbox_hw'][0]}x{r['obj_bbox_hw'][1]}"
        vel = (f"({r['motion'][0]:+.2f},{r['motion'][1]:+.2f})"
               if not np.isnan(r['motion'][0]) else "N/A")
        cos = f"{r['cos']:+.2f}" if not np.isnan(r['cos']) else "N/A"
        distr = "yes" if r['is_distractor'] else "no"
        print(f"{r['vid']:18s} {r['T']:>3d} {off:>14s} {bbx:>10s} "
              f"{r['mean_overlap']:>8.3f} {vel:>14s} {cos:>7s} {distr:>6s} {r['verdict']}")


if __name__ == "__main__":
    main()
