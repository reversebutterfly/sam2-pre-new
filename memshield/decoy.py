"""
Decoy Frame Generator: create semantically misleading inserted frames.

Instead of blurry 50/50 blends + noise, create sharp frames that:
1. Weaken the real object region (push toward background appearance)
2. Create a "bridge" connecting the object to a nearby decoy region
3. Strengthen the decoy region (push toward object-like appearance)

The goal: SAM2 writes a CONFIDENT but WRONG memory entry that
associates the object identity with the decoy location.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def find_decoy_region(
    mask: np.ndarray,
    frame: np.ndarray,
    offset_ratio: float = 0.5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Find a nearby background region to use as decoy.

    Strategy: shift the object mask by 0.3-1.0x object width into background.
    Choose the direction with best color similarity to the object.

    Args:
        mask: [H, W] binary uint8 mask of the target object.
        frame: [H, W, 3] uint8 RGB frame.
        offset_ratio: shift distance as fraction of object width.

    Returns:
        (decoy_mask, (dy, dx)): shifted mask and the offset used.
    """
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask.copy(), (0, 0)

    # Object bounding box and size
    cy, cx = int(ys.mean()), int(xs.mean())
    obj_h = ys.max() - ys.min() + 1
    obj_w = xs.max() - xs.min() + 1
    shift_px = max(int(max(obj_h, obj_w) * offset_ratio), 10)

    # Try 8 directions, pick the one with most background + similar color
    obj_color = frame[mask > 0].mean(axis=0) if (mask > 0).any() else np.zeros(3)
    best_score = -1
    best_offset = (0, shift_px)

    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    for dy_dir, dx_dir in directions:
        dy = int(dy_dir * shift_px)
        dx = int(dx_dir * shift_px)

        # Shift mask
        shifted = np.zeros_like(mask)
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)

        h_len = min(src_y1 - src_y0, dst_y1 - dst_y0)
        w_len = min(src_x1 - src_x0, dst_x1 - dst_x0)
        if h_len <= 0 or w_len <= 0:
            continue

        shifted[dst_y0:dst_y0+h_len, dst_x0:dst_x0+w_len] = \
            mask[src_y0:src_y0+h_len, src_x0:src_x0+w_len]

        # Score: prefer directions where shifted region is in background
        overlap = (shifted > 0) & (mask > 0)
        bg_fraction = 1.0 - overlap.sum() / max(shifted.sum(), 1)

        # Color similarity of decoy region to object
        decoy_pixels = frame[shifted > 0]
        if len(decoy_pixels) > 0:
            decoy_color = decoy_pixels.mean(axis=0)
            color_sim = 1.0 / (np.linalg.norm(obj_color - decoy_color) + 1.0)
        else:
            color_sim = 0.0

        # Reject candidates with too little retained area
        retained = shifted.sum() / max(mask.sum(), 1)
        if retained < 0.3:
            continue

        score = 0.7 * bg_fraction + 0.3 * color_sim
        if score > best_score:
            best_score = score
            best_offset = (dy, dx)

    # Fallback: if no valid direction found, retry at reduced shifts
    if best_score < 0:
        for fallback_frac in [0.5, 0.3, 0.2]:
            fb_shift = max(int(max(obj_h, obj_w) * fallback_frac), 5)
            for dy_dir, dx_dir in directions:
                dy_fb = int(dy_dir * fb_shift)
                dx_fb = int(dx_dir * fb_shift)
                shifted_fb = np.zeros_like(mask)
                s0 = max(0, -dy_fb); s1 = min(H, H - dy_fb)
                s2 = max(0, -dx_fb); s3 = min(W, W - dx_fb)
                d0 = max(0, dy_fb); d1 = min(H, H + dy_fb)
                d2 = max(0, dx_fb); d3 = min(W, W + dx_fb)
                hl_fb = min(s1 - s0, d1 - d0)
                wl_fb = min(s3 - s2, d3 - d2)
                if hl_fb > 0 and wl_fb > 0:
                    shifted_fb[d0:d0+hl_fb, d2:d2+wl_fb] = mask[s0:s0+hl_fb, s2:s2+wl_fb]
                retained_fb = shifted_fb.sum() / max(mask.sum(), 1)
                if retained_fb >= 0.3:
                    best_offset = (dy_fb, dx_fb)
                    break
            if best_score >= 0 or best_offset != (0, shift_px):
                break
        else:
            best_offset = (0, 0)  # Last resort: no shift

    # Create decoy mask with best offset
    dy, dx = best_offset
    decoy_mask = np.zeros_like(mask)
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    h_len = min(src_y1 - src_y0, dst_y1 - dst_y0)
    w_len = min(src_x1 - src_x0, dst_x1 - dst_x0)
    if h_len > 0 and w_len > 0:
        decoy_mask[dst_y0:dst_y0+h_len, dst_x0:dst_x0+w_len] = \
            mask[src_y0:src_y0+h_len, src_x0:src_x0+w_len]

    return decoy_mask, best_offset


def create_bridge_mask(
    true_mask: np.ndarray,
    decoy_mask: np.ndarray,
    bridge_width: int = 15,
) -> np.ndarray:
    """Create a bridge connecting true object core to decoy region.

    The pseudo-target for the inserted frame is:
      small_true_core ∪ bridge ∪ decoy_region
    """
    H, W = true_mask.shape

    # Shrink true mask to a small core (eroded version)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    true_core = cv2.erode(true_mask, kernel, iterations=2)
    if true_core.sum() < 50:
        true_core = true_mask.copy()  # Object too small, keep full

    # Create bridge by connecting centroids with a thick line
    ys_t, xs_t = np.where(true_core > 0)
    ys_d, xs_d = np.where(decoy_mask > 0)
    if len(ys_t) == 0 or len(ys_d) == 0:
        return (true_core | decoy_mask).astype(np.uint8)

    cy_t, cx_t = int(ys_t.mean()), int(xs_t.mean())
    cy_d, cx_d = int(ys_d.mean()), int(xs_d.mean())

    bridge = np.zeros((H, W), dtype=np.uint8)
    cv2.line(bridge, (cx_t, cy_t), (cx_d, cy_d), 1, thickness=bridge_width)

    # Combine: core + bridge + decoy
    pseudo_target = (true_core | bridge | decoy_mask).astype(np.uint8)
    return pseudo_target


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a binary mask by (dy, dx) pixels, clipping to image bounds."""
    H, W = mask.shape
    shifted = np.zeros_like(mask)
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    dy0, dy1 = max(0, dy), min(H, H + dy)
    dx0, dx1 = max(0, dx), min(W, W + dx)
    hl = min(sy1 - sy0, dy1 - dy0)
    wl = min(sx1 - sx0, dx1 - dx0)
    if hl > 0 and wl > 0:
        shifted[dy0:dy0+hl, dx0:dx0+wl] = mask[sy0:sy0+hl, sx0:sx0+wl]
    return shifted


def create_decoy_base_frame(
    frame_after: np.ndarray,
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
) -> np.ndarray:
    """Create a sharp base frame with decoy relocation using a SHARED offset.

    Args:
        frame_after: [H, W, 3] uint8 — sharp base (not blurry blend).
        mask_after: [H, W] uint8 — GT mask for frame_after.
        decoy_offset: (dy, dx) — the shared decoy direction from build_role_targets.

    Returns:
        base_frame: [H, W, 3] uint8.
    """
    H, W = frame_after.shape[:2]
    base = frame_after.copy()

    mask_ref = mask_after
    dy, dx = decoy_offset
    decoy_mask = shift_mask(mask_ref, dy, dx)

    # Weaken real object: blend toward local background
    obj_region = mask_ref > 0
    if obj_region.any():
        # Dilate mask to get surrounding background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated = cv2.dilate(mask_ref, kernel, iterations=2)
        bg_ring = (dilated > 0) & (~obj_region)

        if bg_ring.any():
            bg_color = frame_after[bg_ring].mean(axis=0).astype(np.uint8)
            # Blend object region 70% toward background color
            base[obj_region] = (0.3 * base[obj_region].astype(float) +
                                0.7 * bg_color.astype(float)).clip(0, 255).astype(np.uint8)

    # Strengthen decoy: copy object texture to decoy location
    decoy_region = decoy_mask > 0
    if decoy_region.any() and obj_region.any():
        obj_pixels = frame_after[obj_region]
        if len(obj_pixels) > 0:
            obj_mean = obj_pixels.mean(axis=0).astype(np.uint8)
            base[decoy_region] = (0.6 * base[decoy_region].astype(float) +
                                  0.4 * obj_mean.astype(float)).clip(0, 255).astype(np.uint8)

    return base
