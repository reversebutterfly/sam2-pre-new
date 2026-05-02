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

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Typed errors (codex round 29 D-fix, 2026-05-02)
# =============================================================================


class HybridInfeasibleError(ValueError):
    """No hybrid-feasible offset exists at a given anchor under the
    requested strategy + safety_margin contract.

    Subclass of ValueError so existing `except ValueError` paths in
    callers (e.g. prescreen_tau_init) still catch it — but search-side
    code that wants to distinguish "this anchor is geometrically
    infeasible" from "any other ValueError" can catch this specifically
    via `except HybridInfeasibleError`.

    Attributes:
        c_k: the clean-space insert anchor that failed.
        strategy: insert_base_mode that was tried (e.g. 'poisson_hifi').
        reason: human-readable explanation (border-too-close vs strategy
            internal-check rejection).
    """

    def __init__(self, c_k: int, strategy: str, reason: str) -> None:
        self.c_k = int(c_k)
        self.strategy = str(strategy)
        self.reason = str(reason)
        super().__init__(
            f"insert anchor c_k={c_k}: {reason} for strategy "
            f"{strategy!r}. This anchor is hybrid-infeasible. "
            "Drop it from W_clean (recommended), or pass "
            "allow_ghost_fallback=True to fall back to "
            "duplicate_object decoy (ghosted, violates "
            "ghost-free contract).")


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


def compute_hybrid_safe_offset_from_mask(
    mask: np.ndarray,
    H: int,
    W: int,
    safety_margin: int = 8,
    relax_iou: Tuple[float, ...] = (0.3, 0.4, 0.5),
    min_shift_frac: float = 0.15,
    min_shift_px: int = 4,
) -> Optional[Tuple[int, int]]:
    """Pick a feasibility-first offset for ghost-free hybrid (poisson_hifi /
    propainter) modes.

    Codex round 10 (2026-04-29): the legacy `compute_decoy_offset_from_mask`
    is explicitly aggressive (min_fraction=0.5, prefers IoU≈0). Ghost-free
    modes require the paste bbox to stay fully within image bounds with
    `safety_margin`, so the legacy offsets routinely fail border-safety.

    This function inverts the priority: instead of demanding maximum
    separation, we adaptively relax the IoU target (0.3 → 0.4 → 0.5)
    and accept the FIRST feasible IoU bucket. Within that bucket, among
    feasible candidates (4 axis-only + 4 diagonals = up to 8), pick the
    one with the LARGEST spatial separation (max coord magnitude, with
    sum-of-magnitudes as tie-break). Returns None if no offset satisfies
    even the loosest IoU target — the caller should treat this anchor
    as hybrid-infeasible (drop it from W or pick a different anchor).

    1D IoU geometry (for shift d along an axis with bbox width w):
        IoU = (w - d) / (w + d)  for 0 <= d <= w
        d_min(IoU=τ) = w · (1 - τ) / (1 + τ)
            τ=0.3 → d_min = 0.538 w
            τ=0.4 → d_min = 0.429 w
            τ=0.5 → d_min = 0.333 w

    Args:
        mask: [H, W] float/uint8 binary mask of the object.
        H, W: image dimensions.
        safety_margin: same as `_is_border_safe`. Default 8.
        relax_iou: IoU upper-bound schedule. Tries each in order; the
            first that yields a feasible offset wins.
        min_shift_frac: lower bound on |d| as fraction of bbox axis
            length (avoids near-zero shifts that paste duplicate onto
            original).
        min_shift_px: absolute lower bound on |d| in pixels.

    Returns:
        (dy, dx) integer offset, or None if no feasible offset exists
        even at the loosest IoU.
    """
    # Codex round 11 fix: align bbox convention with the renderer's
    # `create_decoy_base_frame_hifi` (memshield/decoy.py:347-350) which
    # uses EXCLUSIVE y1/x1 (`ys.max() + 1`). Using inclusive max would
    # produce a 1-pixel mismatch on even-sized bboxes and could flip
    # near-border pass/fail decisions vs `_is_border_safe`.
    m = (np.asarray(mask) > 0.5)
    if not m.any():
        return None
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1   # exclusive y1
    x0, x1 = int(xs.min()), int(xs.max()) + 1   # exclusive x1
    bbox_h = y1 - y0
    bbox_w = x1 - x0

    # Match renderer: cy_obj = (y0 + y1) // 2 (using exclusive y1), then
    # paste cy = cy_obj + dy. half_h = (y1 - y0) // 2 = bbox_h // 2.
    cy0 = (y0 + y1) // 2
    cx0 = (x0 + x1) // 2
    half_h = bbox_h // 2
    half_w = bbox_w // 2

    # Feasible range for dy with safety_margin: paste y-bounds in [0, H).
    # py0 = cy0 + dy - half_h - safety_margin >= 0
    # py1 = cy0 + dy + half_h + safety_margin <= H - 1
    dy_lo = -(cy0 - half_h - safety_margin)            # so py0 >= 0
    dy_hi = (H - 1) - (cy0 + half_h + safety_margin)   # so py1 <= H-1
    dx_lo = -(cx0 - half_w - safety_margin)
    dx_hi = (W - 1) - (cx0 + half_w + safety_margin)

    # Codex round 11 fix: do NOT short-circuit on "d=0 infeasible in both
    # axes". A y-axis-only shift (dx=0) can still succeed even if x has
    # no valid range, and vice-versa. Per-candidate feasibility is
    # enforced inside the loop below.

    # Effective lower bound on |d| (avoid near-zero shifts).
    dy_min_abs = max(int(min_shift_px), int(round(min_shift_frac * bbox_h)))
    dx_min_abs = max(int(min_shift_px), int(round(min_shift_frac * bbox_w)))

    # For each IoU target τ, the d_min that achieves IoU ≤ τ on a single
    # axis with bbox extent `w`: d = w · (1 - τ) / (1 + τ). We then ALSO
    # honor min_shift_*_abs (the floor below which a shift is too small).
    def _d_for_iou(tau: float, w: int) -> int:
        return int(round(w * (1.0 - tau) / (1.0 + tau)))

    # Helper: clip a signed shift to its feasible range, applying min-abs.
    # Returns clipped int or None if no clipped value satisfies min-abs
    # OR if the feasible interval is empty (lo > hi). Codex round 13 fix
    # (2026-04-29): without the lo > hi guard, a positive branch could
    # return `hi` even when no actual feasible value exists, producing a
    # false-positive offset that the renderer would reject downstream.
    def _clip_signed(d_signed, sign, lo, hi, min_abs):
        if int(lo) > int(hi):
            return None
        if sign > 0:
            d_clipped = min(d_signed, int(hi))
            return d_clipped if d_clipped >= min_abs else None
        else:
            d_clipped = max(d_signed, int(lo))
            return d_clipped if d_clipped <= -min_abs else None

    candidates_by_axis = []  # list of (tau, candidate_offset)
    for tau in relax_iou:
        dy_target = max(dy_min_abs, _d_for_iou(tau, bbox_h))
        dx_target = max(dx_min_abs, _d_for_iou(tau, bbox_w))

        # Axis-only candidates (4): pure ±y / ±x. Each requires the
        # perpendicular axis to be feasible at d=0.
        axis_only_dirs = [
            ("y", +1, dy_target),
            ("y", -1, dy_target),
            ("x", +1, dx_target),
            ("x", -1, dx_target),
        ]
        for axis, sign, d_t in axis_only_dirs:
            if axis == "y":
                # Codex round 11 fix: y-axis-only requires dx=0 feasible.
                if not (dx_lo <= 0 <= dx_hi):
                    continue
                d_clipped = _clip_signed(
                    sign * d_t, sign, int(dy_lo), int(dy_hi), dy_min_abs)
                if d_clipped is None:
                    continue
                cand = (int(d_clipped), 0)
            else:
                # x-axis-only requires dy=0 feasible.
                if not (dy_lo <= 0 <= dy_hi):
                    continue
                d_clipped = _clip_signed(
                    sign * d_t, sign, int(dx_lo), int(dx_hi), dx_min_abs)
                if d_clipped is None:
                    continue
                cand = (0, int(d_clipped))
            candidates_by_axis.append((tau, cand))

        # Diagonal candidates (4): ±dy AND ±dx simultaneously.
        # Codex round 11 noted: required for corner-near anchors where
        # d=0 is infeasible in BOTH axes (object touching two borders).
        # Both dy_clipped and dx_clipped are filtered by their own
        # feasible range + min_abs, so the resulting (dy, dx) is
        # guaranteed border-safe.
        diagonal_dirs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        for sy, sx in diagonal_dirs:
            dy_clipped = _clip_signed(
                sy * dy_target, sy, int(dy_lo), int(dy_hi), dy_min_abs)
            if dy_clipped is None:
                continue
            dx_clipped = _clip_signed(
                sx * dx_target, sx, int(dx_lo), int(dx_hi), dx_min_abs)
            if dx_clipped is None:
                continue
            cand = (int(dy_clipped), int(dx_clipped))
            candidates_by_axis.append((tau, cand))

        # If at least one feasible candidate emerged at this τ, prefer
        # the one with the largest absolute shift (cleanest separation
        # within IoU bound). Tie-break on long axis.
        feas = [c for (t, c) in candidates_by_axis if t == tau]
        if feas:
            # Selection key: (max-coord-magnitude, sum-of-magnitudes) for
            # deterministic tie-break when multiple candidates have equal
            # max-coord. Codex round 14 doc-vs-code alignment fix.
            best = max(
                feas,
                key=lambda c: (max(abs(c[0]), abs(c[1])),
                               abs(c[0]) + abs(c[1])),
            )
            return (int(best[0]), int(best[1]))

    # No feasible offset found at any IoU target.
    return None


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
# Strategy-based seed builder (codex round 5 ghost-fix wiring, 2026-04-28)
# =============================================================================


def build_decoy_insert_seeds_via_strategy(
    strategy: str,
    x_clean: Tensor,
    pseudo_masks: Sequence[np.ndarray],
    W_clean_positions: Sequence[int],
    *,
    feather_px: int = 3,
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
    decoy_offset: Tuple[int, int] = None,
    propainter_config=None,
    allow_ghost_fallback: bool = False,
) -> Tuple[Tensor, list]:
    """Build K decoy insert seeds via insert-base strategy dispatcher.

    Drop-in alternative to `build_decoy_insert_seeds` for ghost-free
    synthesis. Wraps `memshield.propainter_base.create_insert_base` to
    produce K seeds using either:
      - "poisson_hifi": Poisson-blend with explicit original-object inpaint
        (memshield.decoy.create_decoy_base_frame_hifi).
      - "propainter":  ProPainter 3-frame middle-slot synthesis
        (memshield.propainter_base.create_insert_base_propainter).

    Codex round 5 (JOINT_OPTIMIZATION_2026-04-28.md) recommends
    poisson_hifi as primary ghost-fix; propainter as backup.

    On per-insert border-safety failure, retries alternate offsets
    (sign-flipped on each axis). If ALL attempts fail, raises ValueError
    rather than silently falling back to a ghosted seed — codex round 6
    correctness fix (2026-04-28). To allow the legacy silent fallback,
    pass `allow_ghost_fallback=True` (logs a warning per fallback).

    Args:
        strategy: "poisson_hifi" or "propainter".
        x_clean: [T, H, W, 3] float in [0, 1].
        pseudo_masks: T per-frame masks (np or array-like, [H, W]).
        W_clean_positions: K clean-space insert anchors. Must satisfy
            `c_k >= 1` so we can use `x_clean[c_k - 1]` as `frame_prev`
            (matches upstream contract in stage14_helpers.py:252).
        feather_px: ProPainter feather radius (ignored by poisson_hifi).
        seam_dilate_px: edit-mask dilation passed to both strategies.
        safety_margin: border safety pixel margin.
        decoy_offset: shared (dy, dx) override; None => per-insert via
            `compute_decoy_offset_from_mask`.
        propainter_config: optional ProPainterConfig for propainter strategy.
        allow_ghost_fallback: if True, fall back to duplicate_object decoy
            (which has ghosting) when all offset retries fail. Default
            False — raises ValueError on total failure, so a run labeled
            `poisson_hifi` cannot silently contain ghosted inserts.

    Returns:
        (seeds, offsets) — same shape contract as `build_decoy_insert_seeds`.

    Raises:
        ValueError: if `c_k < 1` for any anchor, or if all offset retries
            fail and `allow_ghost_fallback=False`.
        RuntimeError: if any underlying `create_insert_base` call raises
            (including `InstallationError` from missing ProPainter when
            strategy="propainter"). The wrapper re-raises with anchor /
            offset context to ease debugging long jobs.
    """
    import warnings
    from memshield.propainter_base import create_insert_base

    K = len(W_clean_positions)
    if K == 0:
        return x_clean.new_zeros((0, *x_clean.shape[1:])), []
    T = int(x_clean.shape[0])
    seeds: list = []
    offsets_used: list = []

    for c_k in W_clean_positions:
        c_k = int(c_k)
        # Codex round 6: enforce same anchor contract as upstream
        # (stage14_helpers.py:252; run_vadi_v5.py:2626).
        if not (1 <= c_k < T):
            raise ValueError(
                f"insert anchor c_k={c_k} out of [1, {T}) — strategy "
                "wrapper needs x_clean[c_k - 1] as frame_prev")
        c_prev = c_k - 1
        frame_prev_t = x_clean[c_prev].clamp(0, 1)
        frame_after_t = x_clean[c_k].clamp(0, 1)
        frame_prev_np = (frame_prev_t.detach().cpu().numpy() * 255.0).astype(
            np.uint8)
        frame_after_np = (frame_after_t.detach().cpu().numpy() * 255.0).astype(
            np.uint8)

        # Codex round 21 (2026-04-30): pseudo_masks are SAM2 sigmoid
        # probabilities ∈ [0, 1] with continuous values everywhere (mean
        # 0.10-0.20 on DAVIS, but every pixel has a tiny non-zero prob).
        # Thresholding at `> 0` collapses to nearly all-ones → Poisson
        # seamlessClone receives a whole-frame mask → strategy-internal
        # check rejects → ValueError percolates up as "100% prescreen
        # failure" (canary cows). Other thresholds in this file already
        # use `> 0.5` (lines 194, 239, 339); these two were inconsistent.
        mask_prev_raw = np.asarray(pseudo_masks[c_prev])
        mask_after_raw = np.asarray(pseudo_masks[c_k])
        mask_prev_np = (mask_prev_raw > 0.5).astype(np.uint8)
        mask_after_np = (mask_after_raw > 0.5).astype(np.uint8)

        kwargs = dict(
            seam_dilate_px=int(seam_dilate_px),
            safety_margin=int(safety_margin),
        )
        if strategy == "propainter":
            kwargs["feather_px"] = int(feather_px)
            if propainter_config is not None:
                kwargs["config"] = propainter_config

        # Codex round 10 (2026-04-29): feasibility-first offset selection
        # for ghost-free hybrid modes. The legacy aggressive heuristic
        # (`compute_decoy_offset_from_mask`, min_fraction=0.5, IoU≈0)
        # routinely produces offsets that fail border-safety. Replace it
        # with `compute_hybrid_safe_offset_from_mask`, which adaptively
        # relaxes IoU (0.3 → 0.4 → 0.5) and respects feasibility bounds
        # so the returned offset is GUARANTEED border-safe (or None,
        # meaning this anchor is hybrid-infeasible).
        if decoy_offset is not None:
            # Caller-supplied override: respect it as the only candidate.
            unique_candidates = [(int(decoy_offset[0]),
                                  int(decoy_offset[1]))]
        else:
            feasible = compute_hybrid_safe_offset_from_mask(
                np.asarray(pseudo_masks[c_k], dtype=np.float32),
                int(x_clean.shape[1]),
                int(x_clean.shape[2]),
                safety_margin=int(safety_margin),
            )
            if feasible is None:
                unique_candidates = []
            else:
                unique_candidates = [feasible]

        result = None
        chosen_offset = None
        for cand in unique_candidates:
            try:
                result = create_insert_base(
                    strategy=strategy,
                    frame_prev=frame_prev_np,
                    frame_after=frame_after_np,
                    mask_prev=mask_prev_np,
                    mask_after=mask_after_np,
                    decoy_offset=cand,
                    **kwargs,
                )
            except Exception as exc:
                # Codex round 6: surface InstallationError (propainter)
                # and other strategy errors with context, do NOT silently
                # swallow them.
                raise RuntimeError(
                    f"create_insert_base(strategy={strategy!r}, "
                    f"c_k={c_k}, offset={cand}) raised {type(exc).__name__}: "
                    f"{exc}"
                ) from exc
            if result is not None:
                chosen_offset = cand
                break

        if result is None:
            # No hybrid-feasible offset OR the only candidate's
            # `create_insert_base` rejected (rare — feasibility check
            # already validated border safety, but per-strategy strict
            # checks may still fail).
            n_cand = len(unique_candidates)
            reason = (
                "no hybrid-feasible offset under any IoU target — "
                "object is too close to a frame border or too large "
                "for safety_margin"
                if n_cand == 0 else
                f"the {n_cand} feasible-offset candidate(s) were "
                "rejected by create_insert_base's strategy-internal "
                "checks (e.g., propainter flow estimation failure)"
            )
            if not allow_ghost_fallback:
                # codex round 29 D-fix (2026-05-02): typed exception
                # so search-side callers can soft-fail this anchor and
                # try alternative W combinations instead of crashing.
                # Subclass of ValueError → existing except-ValueError
                # paths (prescreen_tau_init line 754) still catch it.
                raise HybridInfeasibleError(
                    c_k=c_k, strategy=strategy, reason=reason)
            warnings.warn(
                f"[ghost-fallback] strategy={strategy!r} c_k={c_k}: "
                f"{reason}; falling back to duplicate_object decoy "
                "(ghosted). This violates the ghost-free contract of "
                "the strategy.",
                stacklevel=2,
            )
            x_ref = x_clean[c_k]
            mask_t = torch.from_numpy(
                np.asarray(pseudo_masks[c_k], dtype=np.float32)
            ).to(x_clean.device).to(x_clean.dtype)
            # For fallback only, fall back to legacy aggressive offset
            # since duplicate_object_decoy has no border-safety check.
            legacy_offset = compute_decoy_offset_from_mask(
                np.asarray(pseudo_masks[c_k], dtype=np.float32))
            seed = build_duplicate_object_decoy_frame(
                x_ref, mask_t, legacy_offset,
                feather_radius=5,
                feather_sigma=2.0,
            )
            offsets_used.append(
                (int(legacy_offset[0]), int(legacy_offset[1])))
        else:
            base_frame_uint8, _edit_mask = result
            seed = torch.from_numpy(
                base_frame_uint8.astype(np.float32) / 255.0
            ).to(x_clean.device).to(x_clean.dtype)
            offsets_used.append(chosen_offset)

        seeds.append(seed)

    return torch.stack(seeds, dim=0), offsets_used


# =============================================================================
# Try-variant: soft-fail on hybrid-infeasibility (codex round 29 D-fix)
# =============================================================================


def try_build_decoy_insert_seeds_via_strategy(
    strategy: str,
    x_clean: Tensor,
    pseudo_masks: Sequence[np.ndarray],
    W_clean_positions: Sequence[int],
    *,
    feather_px: int = 3,
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
    decoy_offset: Tuple[int, int] = None,
    propainter_config=None,
) -> Tuple[Optional[Tensor], Optional[list], List[int]]:
    """Soft-fail variant of build_decoy_insert_seeds_via_strategy.

    On HybridInfeasibleError at any anchor, returns
    `(None, None, [c_k_failed, ...])` instead of raising. Otherwise
    returns `(seeds, offsets, [])` — same shape contract as the
    raise-version's success path, with an empty failure list.

    Used by joint placement search to try alternative W combinations
    when a particular W tuple has an infeasible anchor, instead of
    crashing the whole search.

    Note: `allow_ghost_fallback` is intentionally NOT exposed — search-
    side code must respect the no-ghost contract; falling back to
    ghosted decoys silently from inside the search would re-introduce
    the same correctness violation the contract is meant to prevent.
    Other (non-hybrid) errors (RuntimeError from create_insert_base,
    ValueError from c_k bounds check) are NOT caught — they indicate
    real bugs and should propagate normally.
    """
    try:
        seeds, offsets = build_decoy_insert_seeds_via_strategy(
            strategy=strategy,
            x_clean=x_clean,
            pseudo_masks=pseudo_masks,
            W_clean_positions=W_clean_positions,
            feather_px=feather_px,
            seam_dilate_px=seam_dilate_px,
            safety_margin=safety_margin,
            decoy_offset=decoy_offset,
            propainter_config=propainter_config,
            allow_ghost_fallback=False,
        )
    except HybridInfeasibleError as exc:
        return None, None, [int(exc.c_k)]
    return seeds, offsets, []


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
