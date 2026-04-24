"""Boundary-band construction for VADI-v5 δ polish stage (2026-04-24).

Codex Round 1 design #1: spend the tiny ε=4/255 budget exactly where
J-drop is lost — the mask boundary. Given per-frame pseudo-masks
m_true and m_decoy, we build a spatial support mask for δ that
covers:

  * `∂m_true` band  — a narrow ring around the true object boundary
  * `∂m_decoy` band — a narrow ring around the decoy target boundary
  * (optional) corridor between centroids — to aid "flow" from
    true-boundary → decoy-boundary

The result is a `[H, W]` float mask in `[0, 1]` — binary at the core,
softly feathered at the edges (to avoid δ-induced hard rectangle
artifacts). δ is then spatially gated by this mask: `δ_effective =
δ ⊙ support_mask`, and the clip stays hard at ε=4/255 elsewhere.

## Why a BAND and not the FULL mask

Inside the mask interior, SAM2's decoder is already well-saturated
(high-confidence positive logits). Small perturbations won't shift
classification there. Outside the mask, same story with negative
saturation. Classification is decided AT the boundary — that is the
decision surface.

Spending the ε budget on interior pixels is wasted; spending it on
the band gives direct boundary-gradient leverage.

## Why a CORRIDOR between centroids

When SAM2's prediction is "directionally correct but mask-shape wrong"
(high align_cos but low J_vs_decoy — exactly the "degraded" mode we
see on most 10-clip runs), the prediction tends to drift partially
toward decoy but remain connected to the true region. A thin corridor
along the centroid-connecting line gives δ a region where it can
"persuade" the prediction to move as a contiguous blob rather than
fragment into two.

## Self-test

`python -m memshield.boundary_bands` → synthetic-input sanity
checks for each function.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Morphological primitives (scipy-backed with numpy fallback)
# =============================================================================


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a disk-like structuring element of `radius`.

    Uses `scipy.ndimage.binary_dilation` with a circular structuring
    element when scipy is available; falls back to a numpy-only pair of
    max-shifts for portability.

    Input: `[H, W]` uint8/bool/float in `{0, 1}`.
    Output: `[H, W]` uint8 in `{0, 1}`.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if radius <= 0:
        return m
    try:
        from scipy.ndimage import binary_dilation
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = (x * x + y * y <= radius * radius).astype(np.uint8)
        return binary_dilation(m, structure=struct).astype(np.uint8)
    except ImportError:
        # Pure numpy fallback: square dilation via shifts (less accurate
        # than disk but monotonic; only hit when scipy missing).
        out = m.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx > radius * radius:
                    continue
                shifted = np.roll(m, shift=(dy, dx), axis=(0, 1))
                # Undo wrap-around on the edges.
                if dy > 0:
                    shifted[:dy] = 0
                elif dy < 0:
                    shifted[dy:] = 0
                if dx > 0:
                    shifted[:, :dx] = 0
                elif dx < 0:
                    shifted[:, dx:] = 0
                out = np.maximum(out, shifted)
        return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion = dilation of the complement, complemented back."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if radius <= 0:
        return m
    return (1 - dilate(1 - m, radius)).astype(np.uint8)


# =============================================================================
# Boundary band
# =============================================================================


def boundary_band(
    mask: np.ndarray, band_width: int = 5,
) -> np.ndarray:
    """Return the `band_width`-wide ring around the mask boundary.

    Constructed as `dilate(mask, r) ∧ ¬erode(mask, r)` where `r =
    band_width // 2`. Result includes a symmetric strip on both sides
    of the boundary: `r` pixels outside + `r` pixels inside.

    Returns `[H, W]` uint8 in `{0, 1}`.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m)
    r = max(1, band_width // 2)
    d = dilate(m, r)
    e = erode(m, r)
    band = (d & (1 - e)).astype(np.uint8)
    return band


# =============================================================================
# Corridor between centroids
# =============================================================================


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """(y, x) centroid. None if mask empty."""
    m = (np.asarray(mask) > 0)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return (float(ys.mean()), float(xs.mean()))


def _rasterize_line(
    y0: float, x0: float, y1: float, x1: float,
    H: int, W: int,
) -> np.ndarray:
    """Rasterize a 1-pixel-thick line between two points in `[H, W]`.

    Uses numpy-friendly parametric sampling (oversampled along the
    longer axis); avoids importing PIL/cv2 for a tiny operation.
    """
    out = np.zeros((H, W), dtype=np.uint8)
    dy, dx = y1 - y0, x1 - x0
    steps = int(max(abs(dy), abs(dx)) * 2) + 1   # oversample 2×
    if steps == 0:
        y, x = int(round(y0)), int(round(x0))
        if 0 <= y < H and 0 <= x < W:
            out[y, x] = 1
        return out
    ts = np.linspace(0.0, 1.0, steps)
    ys = (y0 + ts * dy).round().astype(int)
    xs = (x0 + ts * dx).round().astype(int)
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    out[ys[valid], xs[valid]] = 1
    return out


def corridor_between(
    mask_a: np.ndarray, mask_b: np.ndarray,
    corridor_width: int = 5,
) -> np.ndarray:
    """Return a `corridor_width`-wide strip connecting the centroids of
    the two masks. Empty if either mask is empty.

    Returns `[H, W]` uint8 in `{0, 1}`.
    """
    c_a = _centroid(mask_a)
    c_b = _centroid(mask_b)
    if c_a is None or c_b is None:
        return np.zeros(mask_a.shape, dtype=np.uint8)
    H, W = mask_a.shape
    line = _rasterize_line(c_a[0], c_a[1], c_b[0], c_b[1], H, W)
    if corridor_width <= 1:
        return line
    r = max(1, corridor_width // 2)
    return dilate(line, r)


# =============================================================================
# Feathering (soft transition for perceptual smoothness)
# =============================================================================


def feather(
    mask: np.ndarray, sigma: float = 2.0, radius: Optional[int] = None,
) -> np.ndarray:
    """Gaussian-blur a binary mask to produce a soft `[0, 1]` alpha.

    If scipy unavailable, falls back to a separable numpy box-blur
    approximation.

    Returns `[H, W]` float in `[0, 1]`.
    """
    m = np.asarray(mask, dtype=np.float32)
    if sigma <= 0:
        return m
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    try:
        from scipy.ndimage import gaussian_filter
        return np.clip(
            gaussian_filter(m, sigma=sigma, truncate=3.0), 0.0, 1.0)
    except ImportError:
        # Numpy box-blur approximation, twice → pseudo-Gaussian.
        k = 2 * radius + 1
        kernel = np.ones(k, dtype=np.float32) / k
        blurred = m
        for _ in range(2):
            padded = np.pad(blurred, ((radius, radius), (0, 0)),
                            mode="reflect")
            blurred = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="valid"),
                axis=0, arr=padded)
            padded = np.pad(blurred, ((0, 0), (radius, radius)),
                            mode="reflect")
            blurred = np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="valid"),
                axis=1, arr=padded)
        return np.clip(blurred, 0.0, 1.0)


# =============================================================================
# Combined support-mask builder
# =============================================================================


def build_delta_support_mask(
    m_true: np.ndarray,
    m_decoy: np.ndarray,
    *,
    band_width: int = 5,
    use_corridor: bool = True,
    corridor_width: int = 5,
    feather_sigma: float = 2.0,
) -> np.ndarray:
    """Build the per-frame δ spatial support mask.

    Union of:
      - boundary band of `m_true`
      - boundary band of `m_decoy`
      - (optional) corridor between centroids

    Then feathered with a Gaussian of `feather_sigma`. The feathering
    softens hard edges on the binary union so that δ, which gets
    multiplied by this mask at each PGD step, does not produce a
    visible rectangular artifact.

    Returns `[H, W]` float in `[0, 1]`.
    """
    true_band = boundary_band(m_true, band_width=band_width)
    decoy_band = boundary_band(m_decoy, band_width=band_width)
    parts = [true_band, decoy_band]
    if use_corridor:
        parts.append(corridor_between(
            m_true, m_decoy, corridor_width=corridor_width))
    union = np.zeros_like(true_band, dtype=np.uint8)
    for p in parts:
        union |= p
    if feather_sigma <= 0:
        return union.astype(np.float32)
    return feather(union, sigma=feather_sigma)


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    np.random.seed(0)
    H, W = 64, 64

    # -- dilate / erode round-trip (on a simple box)
    m = np.zeros((H, W), dtype=np.uint8); m[20:40, 20:40] = 1
    d = dilate(m, radius=3)
    assert d.sum() > m.sum(), "dilation must enlarge"
    e = erode(m, radius=3)
    assert e.sum() < m.sum(), "erosion must shrink"
    # Dilate of empty is empty.
    assert dilate(np.zeros((H, W)), 3).sum() == 0
    # Erode of full frame stays full.
    full = np.ones((H, W), dtype=np.uint8)
    assert erode(full, 3).sum() == H * W

    # -- boundary_band: ring around the box
    band = boundary_band(m, band_width=4)
    assert band.shape == m.shape
    # Interior (far from boundary) is 0.
    assert band[25, 25] == 0
    # Just outside the boundary is 1.
    assert band[18, 30] == 1
    # Way outside is 0.
    assert band[0, 0] == 0
    # Empty mask → empty band.
    assert boundary_band(np.zeros((H, W))).sum() == 0

    # -- centroid
    assert _centroid(m) == (29.5, 29.5)
    assert _centroid(np.zeros((H, W))) is None

    # -- rasterize line: from (10, 10) to (50, 50)
    line = _rasterize_line(10, 10, 50, 50, H, W)
    assert line[10, 10] == 1
    assert line[50, 50] == 1
    assert line[30, 30] == 1   # midpoint on the diagonal
    # Non-line pixels stay 0.
    assert line[0, 0] == 0

    # -- corridor between two boxes
    m_b = np.zeros((H, W), dtype=np.uint8); m_b[10:20, 40:50] = 1
    corr = corridor_between(m, m_b, corridor_width=5)
    # Corridor nonempty.
    assert corr.sum() > 0
    # Corridor connects the two centroids (approximately).
    c_a = _centroid(m); c_b = _centroid(m_b)
    mid_y = int((c_a[0] + c_b[0]) / 2)
    mid_x = int((c_a[1] + c_b[1]) / 2)
    # At the midpoint (or close) should be in the corridor.
    nearby = corr[max(0, mid_y - 3):mid_y + 3, max(0, mid_x - 3):mid_x + 3]
    assert nearby.sum() > 0

    # -- feather: binary → soft
    soft = feather(m.astype(np.float32), sigma=1.5)
    assert 0.0 <= soft.min() and soft.max() <= 1.0
    # Interior stays near 1.
    assert soft[30, 30] > 0.99
    # Just outside the boundary is 0 < value < 1 (feathered).
    assert 0.0 < soft[17, 30] < 1.0
    # Far outside is ≈ 0.
    assert soft[0, 0] < 0.01

    # -- build_delta_support_mask: union with corridor + feather
    support = build_delta_support_mask(
        m, m_b, band_width=4, use_corridor=True, corridor_width=5,
        feather_sigma=1.5,
    )
    assert support.shape == m.shape
    assert 0.0 <= support.min() and support.max() <= 1.0
    # Boundary of m_true has high values.
    assert support[18, 30] > 0.5
    # Boundary of m_decoy has high values too.
    assert support[9, 45] > 0.5
    # Corridor midpoint has nonzero.
    assert support[mid_y, mid_x] > 0.1
    # Far outside (away from all components) is near 0.
    assert support[60, 0] < 0.05
    # Without corridor, true-interior is below band cutoff (band r=2 only
    # goes 2 px into the mask, so (30,30) is 10 px inside → 0 contribution).
    support_nocor = build_delta_support_mask(
        m, m_b, band_width=4, use_corridor=False, feather_sigma=1.5,
    )
    assert support_nocor[30, 30] < 0.05

    # -- Empty mask case: both empty → all-zero support
    empty = np.zeros((H, W), dtype=np.uint8)
    support_empty = build_delta_support_mask(empty, empty, feather_sigma=1.0)
    assert support_empty.sum() < 1e-3

    # -- One empty: still builds from the non-empty side (no corridor)
    support_partial = build_delta_support_mask(
        m, empty, band_width=4, use_corridor=True, feather_sigma=1.0,
    )
    # Non-empty side's band should still be there.
    assert support_partial[18, 30] > 0.5
    # m_b's band is empty (no mask), so no contribution from there.

    print("memshield.boundary_bands: all self-tests PASSED "
          "(dilate/erode, boundary_band, centroid, rasterize_line, "
          "corridor_between, feather, build_delta_support_mask "
          "including empty-mask edge cases)")


if __name__ == "__main__":
    _self_test()
