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
) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    """Find a nearby background region to use as decoy.

    Strategy: shift the object mask by 0.3-1.0x object width into background.
    Choose the direction with best color similarity to the object.

    Returns:
        (decoy_mask, (dy, dx), is_natural_distractor)
        is_natural_distractor: True if color_sim > 0.15 (shifted region
        looks like the object — e.g. another instance of the same class).

    Args:
        mask: [H, W] binary uint8 mask of the target object.
        frame: [H, W, 3] uint8 RGB frame.
        offset_ratio: shift distance as fraction of object width.

    """
    H, W = mask.shape
    best_color_sim = 0.0
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
            best_color_sim = color_sim

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

    is_natural_distractor = best_color_sim > 0.15
    return decoy_mask, best_offset, is_natural_distractor


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


def find_decoy_candidates(
    mask: np.ndarray,
    frame: np.ndarray,
    offset_ratio: float = 0.5,
    top_k: int = 6,
) -> Tuple[list, bool]:
    """Return top-K decoy offsets ranked by (bg_fraction, color_sim).

    Same scoring as find_decoy_region but keeps all candidates. Used by
    build_role_targets to iterate until a border-safe placement is found.

    Returns:
        (candidates, is_natural_distractor) where candidates is a list of
        ((dy, dx), score) tuples sorted by score descending.
    """
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [((0, 0), 0.0)], False

    obj_h = ys.max() - ys.min() + 1
    obj_w = xs.max() - xs.min() + 1
    obj_color = frame[mask > 0].mean(axis=0) if (mask > 0).any() else np.zeros(3)

    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    ratios = [offset_ratio, offset_ratio * 0.7, offset_ratio * 1.3]

    candidates = []
    best_color_sim = 0.0
    for r in ratios:
        shift_px = max(int(max(obj_h, obj_w) * r), 10)
        for dy_dir, dx_dir in directions:
            dy = int(dy_dir * shift_px)
            dx = int(dx_dir * shift_px)
            shifted = shift_mask(mask, dy, dx)
            retained = shifted.sum() / max(mask.sum(), 1)
            if retained < 0.3:
                continue
            overlap = (shifted > 0) & (mask > 0)
            bg_fraction = 1.0 - overlap.sum() / max(shifted.sum(), 1)
            decoy_pixels = frame[shifted > 0]
            if len(decoy_pixels) > 0:
                decoy_color = decoy_pixels.mean(axis=0)
                color_sim = 1.0 / (np.linalg.norm(obj_color - decoy_color) + 1.0)
            else:
                color_sim = 0.0
            score = 0.7 * bg_fraction + 0.3 * color_sim
            candidates.append(((dy, dx), score))
            best_color_sim = max(best_color_sim, color_sim)

    candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    dedup = []
    for off, sc in candidates:
        if off in seen:
            continue
        seen.add(off)
        dedup.append((off, sc))
        if len(dedup) >= top_k:
            break
    if not dedup:
        dedup = [((0, 0), 0.0)]
    is_natural_distractor = best_color_sim > 0.15
    return dedup, is_natural_distractor


def _is_border_safe(
    mask_after_bbox: Tuple[int, int, int, int],
    paste_center: Tuple[int, int],
    H: int,
    W: int,
    safety_margin: int = 8,
) -> bool:
    """Check that the paste bbox with safety margin is fully inside [H, W]."""
    y0, y1, x0, x1 = mask_after_bbox
    cy, cx = paste_center
    half_h = (y1 - y0) // 2
    half_w = (x1 - x0) // 2
    py0 = cy - half_h - safety_margin
    py1 = cy + half_h + safety_margin
    px0 = cx - half_w - safety_margin
    px1 = cx + half_w + safety_margin
    return (py0 >= 0 and py1 < H and px0 >= 0 and px1 < W)


def create_decoy_base_frame_hifi(
    frame_prev: np.ndarray,
    frame_after: np.ndarray,
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """High-fidelity decoy base: use frame_prev as identity anchor.

    Differs from create_decoy_base_frame:
      - Background = clean frame_prev (NO inpainting of frame_after → no
        artifacts at the original object location, LPIPS vs f_prev stays low).
      - Object crop sourced from frame_after (preserves real appearance).
      - Border safety check: if paste region + safety_margin exits image,
        return None (caller should try another offset; NO alpha fallback).
      - Returns (base_frame, edit_support_mask) where edit_support_mask is
        the pasted object region dilated by seam_dilate_px. PGD should
        restrict delta to this region to preserve identity elsewhere.

    Args:
        frame_prev: [H, W, 3] uint8 RGB — clean frame just before insertion
                    point (identity anchor).
        frame_after: [H, W, 3] uint8 RGB — clean frame just after insertion
                     point (source of object appearance).
        mask_after: [H, W] uint8 GT mask of object in frame_after.
        decoy_offset: (dy, dx) shared decoy direction.
        seam_dilate_px: dilation radius for the edit support ring.
        safety_margin: pixels of margin required between paste bbox and image border.

    Returns:
        (base_frame [H,W,3] uint8, edit_mask [H,W] uint8) on success, or
        None if placement is not border-safe.
    """
    H, W = frame_prev.shape[:2]
    mask_ref = (mask_after > 0).astype(np.uint8)
    dy, dx = decoy_offset

    ys, xs = np.where(mask_ref > 0)
    if len(ys) == 0:
        return frame_prev.copy(), np.zeros((H, W), dtype=np.uint8)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy_obj = (y0 + y1) // 2 + dy
    cx_obj = (x0 + x1) // 2 + dx

    if not _is_border_safe((y0, y1, x0, x1), (cy_obj, cx_obj), H, W, safety_margin):
        return None

    frame_after_bgr = cv2.cvtColor(frame_after, cv2.COLOR_RGB2BGR)
    frame_prev_bgr = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2BGR)

    obj_crop_bgr = frame_after_bgr[y0:y1, x0:x1]
    mask_crop = mask_ref[y0:y1, x0:x1] * 255

    try:
        result_bgr = cv2.seamlessClone(
            obj_crop_bgr, frame_prev_bgr, mask_crop,
            (cx_obj, cy_obj), cv2.NORMAL_CLONE)
    except cv2.error:
        return None

    result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Edit support = paste region dilated by seam_dilate_px
    paste_region = shift_mask(mask_ref, dy, dx)
    if seam_dilate_px > 0:
        ker_sz = max(3, 2 * seam_dilate_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ker_sz, ker_sz))
        edit_mask = cv2.dilate(paste_region, kernel, iterations=1)
    else:
        edit_mask = paste_region
    edit_mask = (edit_mask > 0).astype(np.uint8)

    return result, edit_mask


def create_decoy_base_frame(
    frame_after: np.ndarray,
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
) -> np.ndarray:
    """Create a high-fidelity base frame with object relocated to decoy position.

    Uses Poisson seamless cloning for photorealistic paste (no ghostly alpha
    blending, no artificial bridges). The result should look like a natural
    frame where the object simply appears at a different location.

    Pipeline:
      1. Inpaint the true object region (large radius for clean removal)
      2. Extract object with tight mask
      3. Poisson seamless clone at decoy location (adapts color/lighting)
      4. Fallback to alpha blend if seamlessClone fails (e.g., near border)

    Args:
        frame_after: [H, W, 3] uint8 RGB frame.
        mask_after: [H, W] uint8 GT mask.
        decoy_offset: (dy, dx) shared decoy direction.

    Returns:
        base_frame: [H, W, 3] uint8 — high-fidelity relocated frame.
    """
    H, W = frame_after.shape[:2]
    mask_ref = (mask_after > 0).astype(np.uint8)
    dy, dx = decoy_offset

    obj_region = mask_ref > 0
    if not obj_region.any():
        return frame_after.copy()

    # Step 1: Inpaint true object with larger radius for clean removal
    # Dilate mask slightly so inpainting covers edge artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inpaint_mask = cv2.dilate(mask_ref, kernel, iterations=1) * 255
    frame_bgr = cv2.cvtColor(frame_after, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(frame_bgr, inpaint_mask, 10, cv2.INPAINT_TELEA)

    # Step 2: Extract object pixels for paste
    ys, xs = np.where(mask_ref > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    obj_crop_bgr = frame_bgr[y0:y1, x0:x1]
    mask_crop = mask_ref[y0:y1, x0:x1] * 255

    # Step 3: Compute decoy paste center
    cy_obj = (y0 + y1) // 2 + dy
    cx_obj = (x0 + x1) // 2 + dx
    # Clamp to image bounds with margin
    margin_y = (y1 - y0) // 2 + 5
    margin_x = (x1 - x0) // 2 + 5
    cy_obj = max(margin_y, min(H - margin_y, cy_obj))
    cx_obj = max(margin_x, min(W - margin_x, cx_obj))

    # Step 4: Poisson seamless clone (photorealistic paste)
    try:
        result_bgr = cv2.seamlessClone(
            obj_crop_bgr, inpainted_bgr, mask_crop,
            (cx_obj, cy_obj), cv2.NORMAL_CLONE)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # Fallback: alpha blend if seamlessClone fails (e.g., near image border)
        base = inpainted_bgr.astype(np.float32)
        obj_alpha = mask_ref.astype(np.float32)
        obj_layer = frame_bgr.astype(np.float32)
        affine = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_alpha = cv2.warpAffine(
            obj_alpha, affine, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shifted_obj = cv2.warpAffine(
            obj_layer, affine, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha3 = shifted_alpha[..., None]
        base = base * (1.0 - alpha3) + shifted_obj * alpha3
        result = cv2.cvtColor(
            np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    return result
