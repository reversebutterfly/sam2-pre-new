"""Preview poisson_hifi hybrid (Poisson edge + inner color preserve) on dog clip.

Outputs side-by-side PNGs for visual inspection:
  preview/dog_pure_poisson.png       (inner_color_preserve_erode_px=0 = legacy)
  preview/dog_hybrid.png             (inner_color_preserve_erode_px=4 = new default)
  preview/dog_duplicate_seed.png     (legacy duplicate_seed for context)
  preview/dog_inputs.png             (frame_prev, frame_after, masks)
  preview/dog_grid.png               (4-panel comparison)
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import cv2
from PIL import Image

from memshield.decoy import create_decoy_base_frame_hifi
from memshield.decoy_seed import (
    compute_decoy_offset_from_mask,
    build_duplicate_object_decoy_frame,
)

CLIP = "dog"
C_PREV = 0
C_AFTER = 1   # build decoy at insert position c_k = c_after, using prev as frame_prev
DAVIS_ROOT = pathlib.Path.home() / "sam2-pre-new" / "data" / "davis"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_frame(idx):
    p = DAVIS_ROOT / "JPEGImages" / "480p" / CLIP / f"{idx:05d}.jpg"
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_mask(idx):
    p = DAVIS_ROOT / "Annotations" / "480p" / CLIP / f"{idx:05d}.png"
    m = Image.open(p).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    # DAVIS palette PNGs: 1 = first object, 0 = bg. Single-object clip → just >0.
    return (arr > 0).astype(np.uint8)


def save_rgb(img_uint8, path):
    Image.fromarray(img_uint8).save(path)


def main():
    print(f"Loading clip={CLIP}, prev frame {C_PREV}, after frame {C_AFTER}")
    frame_prev = load_frame(C_PREV)
    frame_after = load_frame(C_AFTER)
    mask_prev = load_mask(C_PREV)
    mask_after = load_mask(C_AFTER)
    H, W = frame_prev.shape[:2]
    print(f"  shape: {frame_prev.shape}")

    # Compute auto offset (same as run_vadi_v5.py uses).
    decoy_offset = compute_decoy_offset_from_mask(
        mask_after.astype(np.float32))
    decoy_offset = (int(decoy_offset[0]), int(decoy_offset[1]))
    print(f"  auto decoy_offset: {decoy_offset}")

    # ---------- 1. Pure poisson (legacy = erode=0) ----------
    print("Generating pure Poisson (legacy)...")
    res_pure = create_decoy_base_frame_hifi(
        frame_prev, frame_after, mask_after, decoy_offset,
        mask_prev=mask_prev,
        inner_color_preserve_erode_px=0,  # disable hybrid → legacy seamlessClone only
    )
    if res_pure is None:
        print("  PURE FAILED border-safety, retrying with sign-flipped offset")
        decoy_offset = (-decoy_offset[0], -decoy_offset[1])
        res_pure = create_decoy_base_frame_hifi(
            frame_prev, frame_after, mask_after, decoy_offset,
            mask_prev=mask_prev,
            inner_color_preserve_erode_px=0,
        )
    base_pure, edit_pure = res_pure
    save_rgb(base_pure, OUT_DIR / "dog_pure_poisson.png")
    print(f"  saved: {OUT_DIR / 'dog_pure_poisson.png'}")

    # ---------- 2. Hybrid (Poisson edge + inner color preserve) ----------
    print("Generating hybrid (Poisson edge + inner color preserve, erode=4)...")
    res_hybrid = create_decoy_base_frame_hifi(
        frame_prev, frame_after, mask_after, decoy_offset,
        mask_prev=mask_prev,
        inner_color_preserve_erode_px=4,  # new default
    )
    base_hybrid, edit_hybrid = res_hybrid
    save_rgb(base_hybrid, OUT_DIR / "dog_hybrid.png")
    print(f"  saved: {OUT_DIR / 'dog_hybrid.png'}")

    # ---------- 3. duplicate_seed (legacy duplicate_object decoy) ----------
    print("Generating duplicate_seed (legacy, both objects visible)...")
    x_ref_t = torch.from_numpy(frame_after.astype(np.float32) / 255.0)
    mask_t = torch.from_numpy(mask_after.astype(np.float32))
    seed_t = build_duplicate_object_decoy_frame(
        x_ref_t, mask_t, decoy_offset,
        feather_radius=5, feather_sigma=2.0,
    )
    seed_uint8 = (seed_t.numpy().clip(0, 1) * 255.0).astype(np.uint8)
    save_rgb(seed_uint8, OUT_DIR / "dog_duplicate_seed.png")
    print(f"  saved: {OUT_DIR / 'dog_duplicate_seed.png'}")

    # ---------- 4. Inputs ----------
    save_rgb(frame_prev, OUT_DIR / "dog_frame_prev.png")
    save_rgb(frame_after, OUT_DIR / "dog_frame_after.png")

    # ---------- 5. 4-panel grid ----------
    pad = 10
    label_h = 40
    panel_w, panel_h = W, H
    grid_w = 2 * panel_w + 3 * pad
    grid_h = 2 * (panel_h + label_h) + 3 * pad
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    def paste_at(img, label, ix, iy):
        # ix, iy in {0, 1}
        x = pad + ix * (panel_w + pad)
        y = pad + iy * (panel_h + label_h + pad) + label_h
        grid[y:y + panel_h, x:x + panel_w] = img
        # label above
        from PIL import Image as PI, ImageDraw as PD, ImageFont as PF
        for fp in [r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                font = PF.truetype(fp, 24)
                break
            except Exception:
                font = PF.load_default()
        # We render label by composing via PIL once at the end.

    paste_at(frame_after, "frame_after", 0, 0)
    paste_at(seed_uint8, "duplicate_seed", 1, 0)
    paste_at(base_pure, "pure poisson", 0, 1)
    paste_at(base_hybrid, "hybrid", 1, 1)

    # Render labels with PIL.
    pil = Image.fromarray(grid)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
    labels = [
        ("frame_after (clean original)", 0, 0),
        ("duplicate_seed (legacy, two-object)", 1, 0),
        ("pure poisson_hifi (washed-out)", 0, 1),
        ("hybrid (poisson edge + inner color preserve)", 1, 1),
    ]
    for label, ix, iy in labels:
        x = pad + ix * (panel_w + pad)
        y = pad + iy * (panel_h + label_h + pad) + 10
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    pil.save(OUT_DIR / "dog_grid.png")
    print(f"  saved: {OUT_DIR / 'dog_grid.png'}")
    print()
    print(f"Visual inspection: open {OUT_DIR}")


if __name__ == "__main__":
    main()
