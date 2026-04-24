"""Duplicate-object decoy insert frame construction for VADI-v5 (DIRE).

Replaces the v4 temporal-midframe baseline. Codex priority-2 fix
(2026-04-24 auto-review-loop Round 1): the previous midframe base was
an in-distribution benign frame that the PGD had to FIGHT against to
reach decoy semantics. The new base is itself a decoy-semantic frame
(the tracked object is duplicated at a spatially-translated location),
so ν-optimization starts from a frame that already confuses SAM2.

## Construction

For insert k at clean-space position c_k, we use `x_ref = x_{c_k}` as the
backbone (the frame immediately AFTER the insert). We build:

    decoy_seed_k = composite of:
      - x_ref outside the decoy mask
      - x_ref[object region] translated by `decoy_offset` into the decoy mask

Put simply: we keep the tracked object at its original position AND add
a duplicate of it at `original_position + decoy_offset`. SAM2 now sees
TWO plausible instances of the same identity; its memory-conditioned
pointer has to choose, and the ν-optimization can refine the duplicate's
appearance so SAM2 prefers it.

## Why this is a strong decoy base (Codex rationale)

- Content comes entirely from `x_ref`, so it is temporally + photometrically
  consistent with neighbors (LPIPS(seed, x_ref) is small by construction).
- Presence of a second identical object is exactly the identity-confusion
  cue that exploits SAM2's memory-bank design.
- Alpha-feathering at the duplicate's boundary prevents hard copy-paste
  edges that would trivially trigger SAM2's feature extractor to mark it
  as artificial.
- Unlike ProPainter-inpainted decoys, no LPIPS floor (~0.67 for inpainted
  content) — we stay firmly in the photometric neighborhood of x_ref.

## Relationship to `ν` optimization

The decoy seed is the INITIAL insert content. `ν` is then a learnable
residual on top (ν ∈ ℝ^{H×W×3}), optimized to maximize the probability
that SAM2 latches onto the duplicate as the tracked object. ν can:
  - Enhance the duplicate's features (sharper edges, higher contrast)
  - Suppress the original object's features (dim the original)
  - Adjust the spatial ambiguity boundary

Fidelity constraint during ν optimization: LPIPS(final_insert, x_ref) ≤
F_ins_cap (default 0.35), applied on the FINAL frame vs x_ref (not vs
seed — the seed is already an intentional departure).

## Self-test

`python -m memshield.decoy_seed` → runs synthetic sanity checks on a
small-image decoy construction. No SAM2 dependency.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Spatial-shift primitives
# =============================================================================


def shift_2d(x: Tensor, dy: int, dx: int, *, fill: float = 0.0) -> Tensor:
    """Translate a [H, W, *] tensor by (dy, dx) with zero-padding.

    Positive dy moves content DOWN (toward higher row index).
    Positive dx moves content RIGHT (toward higher column index).

    Out-of-bounds regions are filled with `fill`. Content that falls off
    the frame is lost — we do NOT wrap. This matches SAM2's inductive
    bias: the duplicate should look like it's simply at a different
    place in the scene, not like a torus.
    """
    if x.dim() < 2:
        raise ValueError(f"shift_2d needs ≥2 dims; got {tuple(x.shape)}")
    H, W = x.shape[:2]
    out = torch.full_like(x, fill_value=fill)

    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)

    if (src_y1 > src_y0) and (src_x1 > src_x0):
        out[dst_y0:dst_y1, dst_x0:dst_x1] = x[src_y0:src_y1, src_x0:src_x1]
    return out


def shift_mask_np(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """numpy version of shift_2d for [H, W] masks. Used for pseudo-mask shift."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D; got shape {mask.shape}")
    H, W = mask.shape
    out = np.zeros_like(mask)
    src_y0 = max(0, -dy); src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx); src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy); dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx); dst_x1 = min(W, W + dx)
    if (src_y1 > src_y0) and (src_x1 > src_x0):
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


# =============================================================================
# Alpha-feather (soft mask boundary)
# =============================================================================


def _gaussian_kernel_1d(radius: int, sigma: float, device, dtype) -> Tensor:
    """Return a 1-D Gaussian kernel of length 2·radius+1, normalized to 1."""
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur_mask(mask: Tensor, radius: int = 5, sigma: float = 2.0) -> Tensor:
    """Separable Gaussian blur on a [H, W] float mask → feathered [H, W].

    Zero-padded at borders (same as cv2.GaussianBlur BORDER_CONSTANT=0).
    Used to soften copy-paste edges so SAM2's feature extractor can't
    latch onto a perfectly hard rectangle.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be [H, W]; got {tuple(mask.shape)}")
    if radius <= 0:
        return mask
    k = _gaussian_kernel_1d(radius, sigma, mask.device, mask.dtype)
    # F.conv2d expects [N, C, H, W] + kernel [Cout, Cin, kH, kW].
    m = mask[None, None, :, :]
    kx = k[None, None, None, :]
    ky = k[None, None, :, None]
    m = F.conv2d(m, kx, padding=(0, radius))
    m = F.conv2d(m, ky, padding=(radius, 0))
    return m[0, 0].clamp(0.0, 1.0)


# =============================================================================
# Decoy seed construction
# =============================================================================


def build_duplicate_object_decoy_frame(
    x_ref: Tensor,              # [H, W, 3] float in [0, 1]
    object_mask: Tensor,        # [H, W] float in [0, 1] (soft) or {0, 1}
    decoy_offset: Tuple[int, int],  # (dy, dx) translation for the duplicate
    feather_radius: int = 5,
    feather_sigma: float = 2.0,
) -> Tensor:
    """Construct a duplicate-object decoy frame from x_ref.

    Result looks like x_ref but with a SECOND copy of the tracked object
    pasted at `x_original_location + decoy_offset`. The original object
    remains in place — this is identity-confusion, not relocation.

    Alpha-feathering is applied to the TRANSLATED mask boundary before
    compositing, so the pasted duplicate blends smoothly into the
    background and does not expose a hard rectangle edge.

    Args:
        x_ref: `[H, W, 3]` float in `[0, 1]`. Background + original object.
        object_mask: `[H, W]` float in `[0, 1]`. Pseudo-mask of the tracked
            object at x_ref. Soft OK — will be thresholded at 0.5 for the
            extraction, then alpha-feathered on the translated copy.
        decoy_offset: (dy, dx) integer translation. dy > 0 moves down,
            dx > 0 moves right. Typically on the order of
            `0.5 × object_bbox_diag` for clear duplication with low overlap.
        feather_radius: Gaussian kernel half-width for alpha feathering.
            Larger = softer edge. Default 5 px.
        feather_sigma: Gaussian sigma for feathering. Default 2.0.

    Returns:
        `[H, W, 3]` float in `[0, 1]` — the composite decoy frame.
    """
    if x_ref.dim() != 3 or x_ref.shape[-1] != 3:
        raise ValueError(
            f"x_ref must be [H, W, 3]; got {tuple(x_ref.shape)}")
    if object_mask.dim() != 2 or object_mask.shape != x_ref.shape[:2]:
        raise ValueError(
            f"object_mask must be [H, W]={x_ref.shape[:2]}; "
            f"got {tuple(object_mask.shape)}")

    # Hard-binarize the source mask for extraction.
    hard_mask = (object_mask > 0.5).to(x_ref.dtype)

    # Extract the object pixels from x_ref (RGB).
    object_pixels = x_ref * hard_mask.unsqueeze(-1)

    # Translate both the pixels and the mask.
    dy, dx = int(decoy_offset[0]), int(decoy_offset[1])
    object_pixels_shifted = shift_2d(object_pixels, dy, dx, fill=0.0)
    mask_shifted = shift_2d(hard_mask, dy, dx, fill=0.0)

    # Alpha-feather the translated mask boundary.
    if feather_radius > 0:
        alpha = gaussian_blur_mask(mask_shifted, feather_radius, feather_sigma)
    else:
        alpha = mask_shifted

    # Composite: outside mask → x_ref; inside mask → shifted object pixels.
    # alpha is a soft boundary; the composite is a convex combination so
    # pixels exactly inside always come from the shift, outside from x_ref,
    # and near the boundary we blend.
    decoy_frame = (
        (1.0 - alpha).unsqueeze(-1) * x_ref
        + alpha.unsqueeze(-1) * object_pixels_shifted
    )
    return decoy_frame.clamp(0.0, 1.0)


def compute_decoy_offset_from_mask(
    mask: np.ndarray, min_fraction: float = 0.5,
) -> Tuple[int, int]:
    """Choose a (dy, dx) for the duplicate placement.

    Strategy: translate the object by approximately `min_fraction × bbox_diag`
    in whichever direction gives the most free background. Default 0.5×
    means the duplicate and original have low (but nonzero) bbox overlap
    — enough spatial separation that SAM2 cannot treat them as one blob,
    but close enough that they remain within the tracking field.

    Args:
        mask: `[H, W]` float / uint8 binary mask.
        min_fraction: minimum translation as fraction of bbox diagonal.

    Returns:
        (dy, dx) integer offsets.
    """
    m = (np.asarray(mask) > 0.5)
    if not m.any():
        return (0, 0)
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = y1 - y0 + 1
    bbox_w = x1 - x0 + 1
    H, W = m.shape

    # Try 4 candidate offsets at ±min_fraction × bbox in each axis; pick
    # the one whose translated bbox stays mostly inside the frame and has
    # minimum IoU with the original bbox. In ties, prefer the longer axis.
    base_dy = max(1, int(round(min_fraction * bbox_h)))
    base_dx = max(1, int(round(min_fraction * bbox_w)))

    candidates = [
        (+base_dy, 0), (-base_dy, 0),
        (0, +base_dx), (0, -base_dx),
    ]

    def _score(dy: int, dx: int) -> float:
        # Percent of translated bbox that falls inside the frame.
        ty0, ty1 = y0 + dy, y1 + dy
        tx0, tx1 = x0 + dx, x1 + dx
        iy0, iy1 = max(0, ty0), min(H - 1, ty1)
        ix0, ix1 = max(0, tx0), min(W - 1, tx1)
        if iy1 < iy0 or ix1 < ix0:
            return -1.0
        in_h = iy1 - iy0 + 1
        in_w = ix1 - ix0 + 1
        frac_in = (in_h * in_w) / (bbox_h * bbox_w)
        # bbox overlap with original, expressed as fraction of original.
        ov_y0, ov_y1 = max(y0, ty0), min(y1, ty1)
        ov_x0, ov_x1 = max(x0, tx0), min(x1, tx1)
        if ov_y1 < ov_y0 or ov_x1 < ov_x0:
            overlap = 0.0
        else:
            overlap = ((ov_y1 - ov_y0 + 1) * (ov_x1 - ov_x0 + 1)) \
                / (bbox_h * bbox_w)
        # High frac_in and LOW overlap are good.
        return float(frac_in - overlap)

    best = max(candidates, key=lambda c: _score(*c))
    return (int(best[0]), int(best[1]))


def build_decoy_insert_seeds(
    x_clean: Tensor,                    # [T, H, W, 3]
    pseudo_masks: Sequence[np.ndarray],  # len T; each [H, W] float/uint8
    W_clean_positions: Sequence[int],   # K clean-space insert anchors c_k
    *,
    feather_radius: int = 5,
    feather_sigma: float = 2.0,
    decoy_offset: Tuple[int, int] = None,  # shared across all K if given
) -> Tuple[Tensor, list]:
    """Build K decoy insert seed frames, one per `c_k`.

    For each c_k, the seed is constructed from `x_ref = x_clean[c_k]` and
    `object_mask = pseudo_masks[c_k]`. If `decoy_offset` is None, each
    seed computes its own offset via `compute_decoy_offset_from_mask`
    on its own frame.

    Returns:
        - seeds: `[K, H, W, 3]` float tensor on the same device as x_clean.
        - offsets: `[(dy_k, dx_k), ...]` list of the actual offsets used.
    """
    K = len(W_clean_positions)
    if K == 0:
        return x_clean.new_zeros((0, *x_clean.shape[1:])), []
    T = x_clean.shape[0]
    H, W = x_clean.shape[1], x_clean.shape[2]
    seeds: list = []
    offsets_used: list = []
    for c_k in W_clean_positions:
        c_k = int(c_k)
        if not (0 <= c_k < T):
            raise ValueError(
                f"insert anchor c_k={c_k} out of [0, {T})")
        x_ref = x_clean[c_k]
        mask_np = np.asarray(pseudo_masks[c_k], dtype=np.float32)
        offset_k = (decoy_offset if decoy_offset is not None
                    else compute_decoy_offset_from_mask(mask_np))
        mask_t = torch.from_numpy(mask_np).to(x_clean.device).to(x_clean.dtype)
        seed = build_duplicate_object_decoy_frame(
            x_ref, mask_t, offset_k,
            feather_radius=feather_radius,
            feather_sigma=feather_sigma,
        )
        seeds.append(seed)
        offsets_used.append(offset_k)
    return torch.stack(seeds, dim=0), offsets_used


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # -- shift_2d round-trip + edge handling
    H, W = 16, 16
    x = torch.rand(H, W, 3)
    y = shift_2d(x, 3, 2)
    # Content that was at (5, 5) should now be at (8, 7).
    assert torch.allclose(y[8, 7], x[5, 5]), "shift_2d content moved wrong"
    # First 3 rows and first 2 cols are zero-filled.
    assert y[:3, :, :].abs().max().item() == 0.0
    assert y[:, :2, :].abs().max().item() == 0.0
    # Negative shift.
    z = shift_2d(x, -4, -3)
    assert torch.allclose(z[0, 0], x[4, 3])

    # -- shift_mask_np matches torch-tensor shift_2d
    m_np = np.zeros((H, W), dtype=np.float32); m_np[4:8, 4:8] = 1.0
    m_shift_np = shift_mask_np(m_np, 3, 2)
    m_shift_t = shift_2d(torch.from_numpy(m_np), 3, 2).numpy()
    assert np.allclose(m_shift_np, m_shift_t)

    # -- Gaussian blur mask: hard mask becomes feathered on boundary,
    # interior and exterior stay near 1 / 0 respectively.
    mask = torch.zeros(H, W); mask[6:10, 6:10] = 1.0
    blurred = gaussian_blur_mask(mask, radius=3, sigma=1.5)
    assert blurred.max().item() < 1.0 + 1e-5 and blurred.min().item() >= 0.0
    # Interior (away from boundary) should still be strong.
    assert blurred[7, 7].item() > 0.5
    # Far from boundary is still zero.
    assert blurred[0, 0].item() < 0.05

    # -- compute_decoy_offset_from_mask: for a centered object, picks a
    # reasonable offset inside frame, low overlap with original.
    big_mask = np.zeros((32, 32), dtype=np.float32); big_mask[10:20, 10:20] = 1.0
    off = compute_decoy_offset_from_mask(big_mask, min_fraction=0.5)
    # Should be ±5 (half of 10-pixel bbox) in one axis.
    assert abs(off[0]) + abs(off[1]) >= 4, \
        f"compute_decoy_offset too small: {off}"

    # Zero-mask case returns (0, 0).
    zero_mask = np.zeros((32, 32), dtype=np.float32)
    assert compute_decoy_offset_from_mask(zero_mask) == (0, 0)

    # -- build_duplicate_object_decoy_frame: single frame end-to-end
    H2, W2 = 32, 32
    x_ref = torch.rand(H2, W2, 3)
    obj_mask = torch.zeros(H2, W2); obj_mask[10:20, 10:20] = 1.0
    offset = (5, 0)
    decoy = build_duplicate_object_decoy_frame(x_ref, obj_mask, offset)
    # Shape + range
    assert decoy.shape == x_ref.shape
    assert 0.0 <= decoy.min().item() and decoy.max().item() <= 1.0
    # At positions outside both original AND translated mask, decoy == x_ref.
    # (15+5, 15) is well inside the translated mask (center of duplicate);
    # should NOT equal x_ref there (it should be object pixels).
    # (0, 0) is outside both → should equal x_ref.
    assert torch.allclose(decoy[0, 0], x_ref[0, 0], atol=1e-6), \
        "background outside both masks must equal x_ref"
    # (15+5, 15) = (20, 15): inside translated mask; decoy ≠ x_ref.
    diff_in_duplicate = (decoy[20, 15] - x_ref[20, 15]).abs().max().item()
    assert diff_in_duplicate > 0.0, \
        "decoy inside duplicate mask must differ from x_ref"

    # -- build_decoy_insert_seeds: batch version
    T = 12
    x_clean = torch.rand(T, H2, W2, 3)
    pseudo_masks = []
    for t in range(T):
        m = np.zeros((H2, W2), dtype=np.float32)
        y0 = min(t, H2 - 8); x0 = min(t, W2 - 8)
        m[y0:y0 + 8, x0:x0 + 8] = 1.0
        pseudo_masks.append(m)
    W_clean_positions = [2, 6, 10]
    seeds, offsets = build_decoy_insert_seeds(
        x_clean, pseudo_masks, W_clean_positions,
    )
    assert seeds.shape == (3, H2, W2, 3)
    assert len(offsets) == 3
    # Each seed should differ from the corresponding x_clean[c_k].
    for i, c_k in enumerate(W_clean_positions):
        diff = (seeds[i] - x_clean[c_k]).abs().sum().item()
        assert diff > 0.0, \
            f"seed {i} (c_k={c_k}) identical to x_clean[{c_k}]"

    # -- Explicit shared-offset variant
    seeds_shared, offsets_shared = build_decoy_insert_seeds(
        x_clean, pseudo_masks, W_clean_positions, decoy_offset=(4, 0),
    )
    assert all(o == (4, 0) for o in offsets_shared)

    # -- Empty K handled
    empty_seeds, empty_offsets = build_decoy_insert_seeds(
        x_clean, pseudo_masks, [],
    )
    assert empty_seeds.shape == (0, H2, W2, 3)
    assert empty_offsets == []

    print("memshield.decoy_seed: all self-tests PASSED "
          "(shift_2d, shift_mask_np parity, gaussian_blur_mask, "
          "compute_decoy_offset_from_mask, duplicate_object_decoy_frame "
          "composition, build_decoy_insert_seeds batch + shared-offset + empty-K)")


if __name__ == "__main__":
    _self_test()
