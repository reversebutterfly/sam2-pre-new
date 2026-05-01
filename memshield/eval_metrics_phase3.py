"""Phase 3 fidelity metrics for AAAI main table (codex round 27, 2026-04-30).

Codex round 27 confirmed that LPIPS-AlexNet alone is insufficient as the
fidelity claim because it is partly endogenous to the attack (we constrain
LPIPS during PGD). This module supplies the missing post-hoc fidelity
metrics that strengthen the AAAI submission:

  • PSNR per frame — pixel-domain, reviewer-comforting standard.
  • LPIPS-VGG per frame — secondary perceptual; uses a backbone
    different from the LPIPS-Alex we constrained during PGD.
  • Boundary interpolation error — for each insert, predict the
    "intended midpoint" between F_{c-1} and F_c (linear baseline; can
    swap RIFE/FILM later) and compare to the actual inserted decoy.
    This is the single best automated test for "does the decoy belong
    between its neighbors in the original timeline".
  • Motion-compensated flicker — per consecutive (t, t+1) pair compute
    Farneback optical flow t→t+1, warp F_{t+1} backward to F_t's
    coordinate frame, and report the residual. Compare attacked-flicker
    vs clean-flicker; ratio > 1 means attack added temporal jitter.

All metrics operate on EXISTING saved artifacts:
  • exported PNGs at  vadi_runs/<run>/<clip>/<config>/processed/
  • DAVIS x_clean tensor (loaded from data/davis/JPEGImages/480p/<clip>)
  • W_attacked from results.json's exported_j_drop_details

NO SAM2 forward needed. Self-contained, can run anytime AFTER a main-table
run completes; does not interfere with running experiments.

Self-test entrypoint: `python -m memshield.eval_metrics_phase3`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Codex round 28 fix #4: module-level OpenCV availability sentinel so
# every flicker call site can early-return / no-op consistently instead
# of raising at import-of-cv2 time deep inside the call stack.
try:
    import cv2 as _cv2  # noqa: WPS433
    _HAS_CV2 = True
except ImportError:
    _cv2 = None  # type: ignore
    _HAS_CV2 = False


# ===========================================================================
# PSNR
# ===========================================================================


def psnr_uint8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    """PSNR in dB between two `[H, W, 3]` uint8 frames.

    Returns +inf if the frames are bit-identical.
    """
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"shape mismatch: {a_u8.shape} vs {b_u8.shape}")
    if a_u8.dtype != np.uint8 or b_u8.dtype != np.uint8:
        raise ValueError(
            f"PSNR-uint8 expects uint8 inputs; got "
            f"{a_u8.dtype} / {b_u8.dtype}")
    a = a_u8.astype(np.float64)
    b = b_u8.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _pair_to_unit_float(
    a: np.ndarray, b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Codex round 28 fix #5 (round-28 follow-up): per-side normalization.
    Each operand independently lifted to float64 in [0, 1]:
      uint8 → /255
      float → clip(0, 1)
    This handles MIXED-DTYPE pairs correctly (e.g., float decoy vs
    uint8 midpoint), which the round-28 v1 aggregator mishandled by
    deciding scale from `pairs[0][0].dtype` only.
    """
    def _norm(x: np.ndarray) -> np.ndarray:
        if x.dtype == np.uint8:
            return x.astype(np.float64) / 255.0
        return np.asarray(x, dtype=np.float64).clip(0.0, 1.0)
    return _norm(a), _norm(b)


def aggregate_psnr_via_mse(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Codex round 28 fix #1: aggregate PSNR by averaging MSE across
    frame pairs first, then converting to PSNR.

    Required because plain `np.mean(per_frame_psnr)` collapses to `+inf`
    as soon as a single frame is bit-identical (a common case for
    untouched original frames after PNG round-trip in our pipeline).
    The mean-MSE then PSNR-conversion is the standard reporting method
    and stays finite as long as ANY pair has non-zero MSE.

    Mixed dtypes per pair (e.g., float vs uint8) are normalized via
    `_pair_to_unit_float` (codex r28 follow-up fix #5).

    Args:
      pairs: list of (a, b) frame pairs. Each side independently uint8
        or float in [0, 1]; mixed across sides within one pair is OK.

    Returns:
      Aggregate PSNR in dB. Returns +inf only if every pair is exactly
      identical (a degenerate but not impossible case — e.g., methods
      that leave all original frames untouched).
    """
    if not pairs:
        return float("nan")
    mses: List[float] = []
    for a, b in pairs:
        if a.shape != b.shape:
            raise ValueError(
                f"shape mismatch in aggregator: {a.shape} vs {b.shape}")
        af, bf = _pair_to_unit_float(a, b)
        mses.append(float(np.mean((af - bf) ** 2)))
    mean_mse = float(np.mean(mses))
    if mean_mse == 0.0:
        return float("inf")
    # Always report on the [0, 1]-normalized scale (PEAK = 1.0).
    return 10.0 * math.log10(1.0 / mean_mse)


def psnr_float01(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two `[H, W, 3]` float frames in `[0, 1]`."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    af = np.asarray(a, dtype=np.float64).clip(0.0, 1.0)
    bf = np.asarray(b, dtype=np.float64).clip(0.0, 1.0)
    mse = float(np.mean((af - bf) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ===========================================================================
# LPIPS-VGG (independent backbone — not optimized against during PGD)
# ===========================================================================


def build_lpips_vgg_fn(device: str) -> Callable:
    """LPIPS with VGG backbone (the conservative variant per Zhang et al.
    2018; the Alex variant is the PGD-constrained one in our pipeline so
    we use VGG here as an independent fidelity check).

    Returned callable matches the existing project contract:
      f(x[H,W,3] ∈ [0,1], y[H,W,3] ∈ [0,1]) → scalar Tensor.
    """
    import torch
    import lpips as _lpips
    model = _lpips.LPIPS(net="vgg", verbose=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Codex round 28 fix #2: explicit device attribute (avoids
    # closure-walking the .__closure__ which is unsupported API).
    target_device = next(model.parameters()).device

    def lpips_vgg_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must be [H,W,3]; got {tuple(x.shape)}")
        if y.dim() != 3 or y.shape[-1] != 3:
            raise ValueError(f"y must be [H,W,3]; got {tuple(y.shape)}")
        # Move inputs to the model device before forward.
        x_d = x.to(target_device, non_blocking=True)
        y_d = y.to(target_device, non_blocking=True)
        x_b = x_d.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        y_b = y_d.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        with torch.no_grad():
            return model(x_b, y_b).squeeze()

    # Attach .device for callers that want to query (e.g., aggregator).
    lpips_vgg_fn.device = target_device  # type: ignore[attr-defined]
    return lpips_vgg_fn


# ===========================================================================
# Boundary interpolation error
# ===========================================================================


def linear_midpoint(
    f_prev: np.ndarray, f_after: np.ndarray,
) -> np.ndarray:
    """Simple-baseline midpoint predictor: pixelwise mean of two neighbors.

    Caveat: this is the WEAK baseline. Swap with RIFE/FILM for a stronger
    "what should the in-between frame look like" prediction. Documented as
    such in the paper.
    """
    if f_prev.shape != f_after.shape:
        raise ValueError(
            f"shape mismatch: {f_prev.shape} vs {f_after.shape}")
    return ((f_prev.astype(np.float32) + f_after.astype(np.float32)) / 2.0
            ).astype(f_prev.dtype)


def boundary_interpolation_error(
    exported_attacked: np.ndarray,         # [T_proc, H, W, 3] uint8 OR float
    x_clean: np.ndarray,                    # [T_clean, H, W, 3] uint8 OR float
    W_attacked: Sequence[int],
    interp_fn: Callable = linear_midpoint,
    lpips_vgg_fn: Optional[Callable] = None,
) -> Dict[str, object]:
    """Per-insert "does the decoy belong between its neighbors" metric.

    For each attacked-space insert position w_att in W_attacked:
      • Determine the corresponding clean-space neighbors: F_{c-1} and F_c
        where c = w_clean = w_att - k_index (k = index of this insert in
        sorted W).
      • Predict midpoint = interp_fn(F_{c-1}, F_c).
      • Compare exported_attacked[w_att] (the actual decoy after ν) to
        the predicted midpoint via PSNR + (optionally) LPIPS-VGG.

    Args:
      exported_attacked: full attacked video, T_proc frames.
      x_clean: T_clean original frames (no inserts).
      W_attacked: insert positions in attacked-space.
      interp_fn: midpoint predictor; defaults to linear average.
      lpips_vgg_fn: optional LPIPS-VGG callable on torch tensors. If
          None, only PSNR is computed.

    Returns dict:
      per_insert: List[Dict] — one entry per w_att with PSNR + LPIPS-VGG
      mean_psnr: float
      mean_lpips_vgg: Optional[float]
    """
    import torch  # type: ignore
    W_att_sorted = sorted(int(w) for w in W_attacked)
    T_clean = int(x_clean.shape[0])

    # Map each w_att to its (clean-space) neighbor frame indices.
    # w_clean[k] = w_att[k] - k; the "before" neighbor is w_clean - 1
    # and the "after" neighbor is w_clean (in clean space, NOT attacked).
    per_insert: List[Dict[str, object]] = []
    psnrs: List[float] = []
    decoy_mid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []  # for codex r28 mse-based aggregator
    lpips_vals: List[float] = []

    for k_idx, w_att in enumerate(W_att_sorted):
        w_clean = w_att - k_idx
        # Boundary neighbors in clean-space:
        c_prev = max(0, w_clean - 1)
        c_after = min(T_clean - 1, w_clean)
        if c_prev == c_after:
            # Edge case (insert at clean position 0 OR T_clean-1):
            # collapse to single-neighbor reference; our convention is
            # to use that neighbor as the "interpolated" target.
            mid = (np.asarray(x_clean[c_after]).astype(np.float32)
                   ).astype(x_clean.dtype)
        else:
            mid = interp_fn(x_clean[c_prev], x_clean[c_after])

        # Compare actual decoy to predicted midpoint.
        decoy = exported_attacked[w_att]
        if decoy.dtype == np.uint8 and mid.dtype == np.uint8:
            psnr = psnr_uint8(decoy, mid)
        else:
            decoy_f = (np.asarray(decoy, dtype=np.float32) / 255.0
                       if decoy.dtype == np.uint8
                       else np.asarray(decoy, dtype=np.float32))
            mid_f = (np.asarray(mid, dtype=np.float32) / 255.0
                     if mid.dtype == np.uint8
                     else np.asarray(mid, dtype=np.float32))
            psnr = psnr_float01(decoy_f, mid_f)
        psnrs.append(psnr)
        # Save raw arrays for mean-MSE aggregator (codex r28 fix #1).
        decoy_mid_pairs.append((decoy, mid))

        lpips_v: Optional[float] = None
        if lpips_vgg_fn is not None:
            # Codex r28 fix #2: lpips_vgg_fn now moves inputs to its
            # own device internally; CPU tensors are safe.
            decoy_t = (torch.from_numpy(np.asarray(decoy, dtype=np.float32))
                       / (255.0 if decoy.dtype == np.uint8 else 1.0))
            mid_t = (torch.from_numpy(np.asarray(mid, dtype=np.float32))
                     / (255.0 if mid.dtype == np.uint8 else 1.0))
            lp = lpips_vgg_fn(decoy_t, mid_t)
            lpips_v = float(
                lp.detach().item() if hasattr(lp, "detach") else lp)
            lpips_vals.append(lpips_v)

        per_insert.append({
            "w_attacked": w_att,
            "w_clean": w_clean,
            "c_prev": c_prev,
            "c_after": c_after,
            "psnr_vs_midpoint": psnr,
            "lpips_vgg_vs_midpoint": lpips_v,
        })

    return {
        "per_insert": per_insert,
        # Codex r28 fix #1: mean-MSE aggregator stays finite even when
        # individual pairs are bit-identical (PSNR=+inf).
        "mean_psnr": aggregate_psnr_via_mse(decoy_mid_pairs),
        "mean_lpips_vgg": (float(np.mean(lpips_vals))
                           if lpips_vals else None),
        "n_inserts": len(per_insert),
    }


# ===========================================================================
# Motion-compensated flicker
# ===========================================================================


def _require_cv2():
    """Codex r28 fix #4: single guard for OpenCV availability."""
    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV not installed; flicker metric requires `pip install "
            "opencv-python`. Pass skip_flicker=True or install opencv.")
    return _cv2


def _farneback_flow(
    src_gray: np.ndarray, dst_gray: np.ndarray,
) -> np.ndarray:
    """Farneback optical flow from src → dst. Returns [H, W, 2] (dx, dy)."""
    cv2 = _require_cv2()
    return cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray,
        None,
        0.5,        # pyr_scale
        3,          # levels
        15,         # winsize
        3,          # iterations
        5,          # poly_n
        1.2,        # poly_sigma
        0,          # flags
    )


def _warp_with_flow(
    src: np.ndarray, flow: np.ndarray,
) -> np.ndarray:
    """Backward-warp `src` (HxWx3 uint8) with `flow` (HxWx2 dx,dy).
    Each output pixel `out[y,x] = src[y + flow_y, x + flow_x]` via
    bilinear remap. Out-of-bounds → border-replicate.
    """
    cv2 = _require_cv2()
    H, W = src.shape[:2]
    map_y, map_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (map_x + flow[..., 0]).astype(np.float32)
    map_y = (map_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        src, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def motion_compensated_flicker(
    video: np.ndarray,                  # [T, H, W, 3] uint8 OR float
    sample_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """For consecutive frame pairs (t, t+1), compute motion-compensated
    residual: warp F_{t+1} backward to F_t coordinates via Farneback flow,
    then |F_t - warped_F_{t+1}|. Sum / mean over t gives a global flicker
    score. High values = motion-prediction breaks down = visual jitter.

    Args:
      video: T frames, [T, H, W, 3] uint8 or float in [0, 1].
      sample_pairs: optional list of (t1, t2) pairs to evaluate. If None,
          use all consecutive pairs (0,1), (1,2), ..., (T-2, T-1).

    Returns dict:
      per_pair: List[Dict] — {t_src, t_dst, mean_residual}
      mean_residual: float — mean over all pairs (in [0, 1] units)
      total_flicker: float — sum over all pairs (rough scale)
    """
    cv2 = _require_cv2()
    T = int(video.shape[0])
    if T < 2:
        return {"per_pair": [], "mean_residual": float("nan"),
                "total_flicker": 0.0}

    # Convert to uint8 if needed.
    if video.dtype != np.uint8:
        v_u8 = (np.asarray(video, dtype=np.float32).clip(0, 1) * 255.0
                + 0.5).astype(np.uint8)
    else:
        v_u8 = video

    if sample_pairs is None:
        pairs = [(t, t + 1) for t in range(T - 1)]
    else:
        pairs = [(int(a), int(b)) for a, b in sample_pairs]

    per_pair: List[Dict[str, object]] = []
    residuals: List[float] = []
    for t_src, t_dst in pairs:
        if not (0 <= t_src < T and 0 <= t_dst < T):
            continue
        f_src = v_u8[t_src]
        f_dst = v_u8[t_dst]
        gray_src = cv2.cvtColor(f_src, cv2.COLOR_RGB2GRAY)
        gray_dst = cv2.cvtColor(f_dst, cv2.COLOR_RGB2GRAY)
        flow = _farneback_flow(gray_src, gray_dst)
        warped_dst = _warp_with_flow(f_dst, flow)
        # Residual in [0, 1] units.
        resid = float(np.mean(
            np.abs(f_src.astype(np.float32) - warped_dst.astype(np.float32))
        )) / 255.0
        residuals.append(resid)
        per_pair.append({
            "t_src": t_src,
            "t_dst": t_dst,
            "mean_residual": resid,
        })

    return {
        "per_pair": per_pair,
        "mean_residual": (float(np.mean(residuals))
                          if residuals else float("nan")),
        "total_flicker": float(np.sum(residuals)),
    }


# ===========================================================================
# Per-clip aggregator (entrypoint)
# ===========================================================================


def evaluate_phase3_per_clip(
    clip_name: str,
    davis_root: Path,
    exported_video_uint8: np.ndarray,   # [T_proc, H, W, 3] uint8
    W_attacked: Sequence[int],
    lpips_vgg_fn: Optional[Callable] = None,
    skip_lpips: bool = False,
    skip_flicker: bool = False,
) -> Dict[str, object]:
    """Bundled per-clip Phase 3 fidelity report.

    Args:
      clip_name: DAVIS clip name (used to locate x_clean).
      davis_root: DAVIS root path.
      exported_video_uint8: full attacked video as uint8 [T_proc, H, W, 3].
      W_attacked: insert positions in attacked-space.
      lpips_vgg_fn: pre-built VGG LPIPS callable (or None to skip).
      skip_lpips, skip_flicker: per-metric skip switches.

    Returns dict with keys (codex r28 fix #3 — exact API):
      clip_name, T_clean, T_proc, K, W_attacked: meta
      psnr_orig_per_frame: List[float] — per original frame, PSNR
          (exported[orig_to_mod[c]], x_clean[c]); +inf possible per frame
      psnr_orig_mean: float — aggregate via mean-MSE→PSNR; +inf only in
          the degenerate case where EVERY original frame is bit-identical
          (rare in practice, but possible for methods that don't touch
          original frames at all). Otherwise finite.
      lpips_vgg_orig_per_frame: List[float] (only when lpips_vgg_fn given)
      lpips_vgg_orig_mean: float (mean of per-frame, finite)
      boundary_interpolation: dict from boundary_interpolation_error
      flicker_clean_mean, flicker_attacked_mean, flicker_ratio: floats
          (only when skip_flicker=False; ratio > 1 means attack added jitter)

    NOTE: this aggregator does NOT split "originals modified by δ" vs
    "untouched originals" — that requires separate metadata about which
    frames state_continuation touched (W_attacked alone is insufficient).
    Callers wanting the split should pass that metadata separately.
    """
    from PIL import Image  # noqa: WPS433
    img_dir = Path(davis_root) / "JPEGImages" / "480p" / clip_name
    if not img_dir.is_dir():
        raise FileNotFoundError(f"DAVIS clip dir missing: {img_dir}")

    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No JPGs in {img_dir}")
    frames_clean = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
                    for f in files]
    x_clean_u8 = np.stack(frames_clean, axis=0)

    if exported_video_uint8.dtype != np.uint8:
        raise ValueError(
            f"exported_video must be uint8; got {exported_video_uint8.dtype}")
    T_clean = int(x_clean_u8.shape[0])
    T_proc = int(exported_video_uint8.shape[0])
    K = len(W_attacked)
    if T_proc != T_clean + K:
        raise ValueError(
            f"T_proc={T_proc} != T_clean+K={T_clean + K} for clip {clip_name}")

    out: Dict[str, object] = {
        "clip_name": clip_name,
        "T_clean": T_clean,
        "T_proc": T_proc,
        "K": K,
        "W_attacked": list(W_attacked),
    }

    # PSNR on original-frame timeline (exported at orig_to_mod[c] vs x_clean[c]).
    from memshield.eval_metrics_extended import (
        attacked_to_clean_W, orig_to_attacked_indices,
    )
    W_clean = attacked_to_clean_W(W_attacked)
    orig_to_mod = orig_to_attacked_indices(W_clean, T_clean)
    W_att_set = set(int(w) for w in W_attacked)

    # Codex r28 fix #1 + #3: PSNR aggregation via mean-MSE (handles
    # +inf safely) and explicit per-frame storage. We do NOT split into
    # "originals modified" vs "untouched" here because W_attacked alone
    # doesn't define which originals state_continuation δ touched —
    # caller would need to pass that metadata separately.
    psnr_per_orig: List[float] = []
    psnr_orig_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for c in range(T_clean):
        t = orig_to_mod[c]
        psnr_per_orig.append(
            psnr_uint8(exported_video_uint8[t], x_clean_u8[c]))
        psnr_orig_pairs.append(
            (exported_video_uint8[t], x_clean_u8[c]))
    out["psnr_orig_per_frame"] = psnr_per_orig
    out["psnr_orig_mean"] = aggregate_psnr_via_mse(psnr_orig_pairs)

    if lpips_vgg_fn is not None and not skip_lpips:
        import torch  # type: ignore
        # Codex r28 fix #2: read .device attribute (set by
        # build_lpips_vgg_fn) instead of closure-walking. lpips_vgg_fn
        # also handles per-call device move internally; passing CPU
        # tensors is now safe.
        lpips_per_orig: List[float] = []
        for c in range(T_clean):
            t = orig_to_mod[c]
            x_t = (torch.from_numpy(exported_video_uint8[t]).float() / 255.0)
            y_t = (torch.from_numpy(x_clean_u8[c]).float() / 255.0)
            lp = lpips_vgg_fn(x_t, y_t)
            lpips_per_orig.append(
                float(lp.detach().item() if hasattr(lp, "detach") else lp))
        out["lpips_vgg_orig_per_frame"] = lpips_per_orig
        out["lpips_vgg_orig_mean"] = float(np.mean(lpips_per_orig))

    # Boundary interpolation error at insert positions.
    bie = boundary_interpolation_error(
        exported_video_uint8, x_clean_u8, W_attacked,
        interp_fn=linear_midpoint,
        lpips_vgg_fn=lpips_vgg_fn if not skip_lpips else None,
    )
    out["boundary_interpolation"] = bie

    # Motion-compensated flicker on attacked + clean.
    if not skip_flicker:
        flicker_clean = motion_compensated_flicker(x_clean_u8)
        flicker_att = motion_compensated_flicker(exported_video_uint8)
        ratio = (
            flicker_att["mean_residual"] / flicker_clean["mean_residual"]
            if flicker_clean["mean_residual"] > 0 else float("nan")
        )
        out["flicker_clean_mean"] = flicker_clean["mean_residual"]
        out["flicker_attacked_mean"] = flicker_att["mean_residual"]
        out["flicker_ratio"] = ratio
        # Strip per-pair detail from main dict (keep only summaries) — caller
        # can compute again if needed.

    return out


# ===========================================================================
# Self-tests (numerical sanity, not GPU-dependent)
# ===========================================================================


def _test_psnr_identical():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    assert math.isinf(psnr_uint8(a, a)), "identical → PSNR=+inf"
    af = np.zeros((10, 10, 3), dtype=np.float32)
    assert math.isinf(psnr_float01(af, af))


def _test_psnr_max_diff():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.full_like(a, 255)
    p = psnr_uint8(a, b)
    # MSE = 255²; PSNR = 10*log10(1) = 0
    assert abs(p) < 1e-9, f"max-diff PSNR should be 0, got {p}"


def _test_psnr_known():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.full_like(a, 51)  # pixel diff = 51 → MSE = 51² = 2601
    p = psnr_uint8(a, b)
    expected = 10.0 * math.log10(255.0 ** 2 / (51.0 ** 2))
    assert abs(p - expected) < 1e-6, f"got {p}, expected {expected}"


def _test_linear_midpoint():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full_like(a, 100)
    m = linear_midpoint(a, b)
    assert m.dtype == np.uint8
    assert np.all(m == 50)


def _test_boundary_interp_error_pure_midpoint():
    # If the inserted decoy IS the linear midpoint, PSNR should be +inf.
    T_clean = 5
    H, W = 4, 4
    x_clean = np.zeros((T_clean, H, W, 3), dtype=np.uint8)
    for t in range(T_clean):
        x_clean[t] = t * 50  # frame value = 0, 50, 100, 150, 200
    K = 1
    W_clean = [2]                 # insert at clean-space 2
    W_att = [2]                   # attacked-space 2 (after k=0 inserts)
    T_proc = T_clean + K
    exported = np.zeros((T_proc, H, W, 3), dtype=np.uint8)
    # Original frames at attacked positions:
    exported[0] = x_clean[0]
    exported[1] = x_clean[1]
    exported[3] = x_clean[2]   # F2 in attacked at index 3
    exported[4] = x_clean[3]
    exported[5] = x_clean[4]
    # Insert: linear midpoint of F1 (=50) and F2 (=100) → 75
    exported[2] = 75

    bie = boundary_interpolation_error(
        exported, x_clean, W_att, lpips_vgg_fn=None)
    assert math.isinf(bie["mean_psnr"]), \
        f"pure-midpoint insert should give PSNR=+inf, got {bie['mean_psnr']}"


def _test_aggregate_psnr_via_mse_handles_inf():
    """Mix one bit-identical pair and one max-diff pair → mean MSE
    is non-zero, so aggregate PSNR is finite (not +inf)."""
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b_same = a.copy()
    b_diff = np.full_like(a, 255)
    p = aggregate_psnr_via_mse([(a, b_same), (a, b_diff)])
    assert math.isfinite(p), f"should be finite, got {p}"
    # mean MSE = (0 + 255²) / 2; PSNR = 10 log10(255² / (255²/2)) = 10 log10(2) ≈ 3.01
    expected = 10.0 * math.log10(2.0)
    assert abs(p - expected) < 1e-6, f"got {p}, expected {expected}"


def _test_aggregate_psnr_via_mse_all_identical():
    """If every pair is bit-identical, aggregate IS +inf (preserved)."""
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    p = aggregate_psnr_via_mse([(a, a.copy()), (a, a.copy())])
    assert math.isinf(p)


def _test_aggregate_psnr_mixed_dtypes():
    """Codex r28 follow-up: mixed-dtype pairs (e.g., float decoy vs
    uint8 mid) must give the same numerical answer as same-dtype
    equivalents."""
    a_u8 = np.full((4, 4, 3), 100, dtype=np.uint8)
    b_u8 = np.full((4, 4, 3), 150, dtype=np.uint8)
    a_f = a_u8.astype(np.float32) / 255.0
    b_f = b_u8.astype(np.float32) / 255.0
    p_uu = aggregate_psnr_via_mse([(a_u8, b_u8)])
    p_ff = aggregate_psnr_via_mse([(a_f, b_f)])
    p_uf = aggregate_psnr_via_mse([(a_u8, b_f)])
    p_fu = aggregate_psnr_via_mse([(a_f, b_u8)])
    # Tolerance 1e-4 accounts for float32 round-trip on the mixed paths
    # (uint8 → float32 / 255 introduces ~1e-7 quantization per pixel).
    assert abs(p_uu - p_ff) < 1e-4, f"uint-uint vs float-float: {p_uu} vs {p_ff}"
    assert abs(p_uu - p_uf) < 1e-4, f"uint-uint vs uint-float: {p_uu} vs {p_uf}"
    assert abs(p_uu - p_fu) < 1e-4, f"uint-uint vs float-uint: {p_uu} vs {p_fu}"


def _test_motion_flicker_static():
    """All-identical frames → motion flicker ≈ 0 (modulo tiny numerical noise)."""
    H, W = 32, 32
    T = 5
    video = np.full((T, H, W, 3), 128, dtype=np.uint8)
    res = motion_compensated_flicker(video)
    assert res["mean_residual"] < 1e-3, \
        f"static video flicker should be ~0, got {res['mean_residual']}"


if __name__ == "__main__":
    _test_psnr_identical()
    _test_psnr_max_diff()
    _test_psnr_known()
    _test_aggregate_psnr_via_mse_handles_inf()
    _test_aggregate_psnr_via_mse_all_identical()
    _test_aggregate_psnr_mixed_dtypes()
    _test_linear_midpoint()
    _test_boundary_interp_error_pure_midpoint()
    _test_motion_flicker_static()
    print("eval_metrics_phase3 self-tests PASSED (9 tests)")
