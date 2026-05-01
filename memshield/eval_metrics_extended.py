"""Extended eval metrics for AAAI main table (Phase B, codex round 24, 2026-04-30).

Codex round 24 confirmed J-drop alone is insufficient for AAAI submission.
Codex round 25 (2026-04-30 follow-up): the canonical paper baseline must
also be `clean-ORIGINAL` (no inserts, no perturbation, evaluated on the
original-frame timeline against DAVIS GT), not the existing
`clean-PROCESSED` (inserts at W with ν=δ=0). The processed-space metric
remains useful as an internal "marginal contribution of ν+δ" diagnostic
for the appendix; the original-timeline metric is the headline number.

This module supplies the missing metrics:

  • DAVIS-style F-boundary score per frame (region IoU is `J`; boundary
    score is `F`; standard DAVIS ranking is by mean `J&F`).
  • Per-frame LPIPS / SSIM fidelity (exported vs `processed_clean` baseline).
  • UTR@τ (Unusable-Track Rate, fraction of frames with `J_attacked < τ`).
  • SFR@τ (Success-Frame Rate, fraction of frames with `J_attacked > τ`).
  • `orig_to_attacked_indices` + `eval_original_timeline_metrics` for the
    canonical clean-ORIGINAL vs attacked comparison on the user's
    original-frame timeline.
  • `load_davis_gt_per_frame` to load the DAVIS-2017 per-frame ground-
    truth annotations (binarized to single-object foreground).

The DAVIS F-boundary implementation follows the original davis-evaluation
tools convention:
  • Contour extraction via 2-of-3-neighbor differences (`_seg2bmap`).
  • Bipartite matching with bandwidth = `bound_th * sqrt(H² + W²)`.
  • F = 2·precision·recall / (precision + recall).
  • Both-empty masks → F = 1.0 (consistent with `J` convention in
    memshield.eval_v2.jaccard).
  • One-sided empty masks → F = 0.0.

Self-test entrypoint: `python -m memshield.eval_metrics_extended`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# DAVIS F-boundary
# ===========================================================================


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Convert binary segmentation mask to boundary map (DAVIS convention).

    Boundary pixel = any pixel where seg[i,j] differs from one of its
    east, south, or south-east neighbors.

    Args:
      seg: [H, W] binary mask. Will be cast via `> 0`.

    Returns:
      [H, W] bool boundary map.
    """
    seg_b = np.asarray(seg) > 0
    H, W = seg_b.shape
    e = np.zeros_like(seg_b)
    s = np.zeros_like(seg_b)
    se = np.zeros_like(seg_b)
    e[:, :-1] = seg_b[:, 1:]
    s[:-1, :] = seg_b[1:, :]
    se[:-1, :-1] = seg_b[1:, 1:]
    b = (seg_b ^ e) | (seg_b ^ s) | (seg_b ^ se)
    # The right and bottom edge bits in e/s/se are zero-padded above; mask
    # them out cleanly so boundary pixels there are not double-counted.
    b[-1, :] = seg_b[-1, :] ^ e[-1, :]
    b[:, -1] = seg_b[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def f_boundary(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    bound_th: float = 0.008,
) -> float:
    """DAVIS F-boundary score.

    Args:
      mask_pred, mask_gt: [H, W] binary masks (uint8 / bool / 0-1 float).
      bound_th: bandwidth as fraction of frame diagonal. DAVIS default 0.008.

    Returns:
      F in [0, 1]. Both-empty masks → 1.0; single-empty → 0.0.
    """
    pred = np.asarray(mask_pred) > 0
    gt = np.asarray(mask_gt) > 0
    if pred.shape != gt.shape:
        raise ValueError(
            f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    bm_pred = _seg2bmap(pred)
    bm_gt = _seg2bmap(gt)
    n_pred = int(bm_pred.sum())
    n_gt = int(bm_gt.sum())

    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    H, W = gt.shape
    bound_pix = max(1, int(round(bound_th * float(np.sqrt(H * H + W * W)))))

    # Distance transform: at each pixel, distance to the NEAREST boundary.
    # We invert the boundary map (boundary=0, non-boundary=1) so that
    # distance_transform_edt computes distance-to-boundary.
    from scipy.ndimage import distance_transform_edt  # type: ignore
    dist_to_pred_b = distance_transform_edt(np.logical_not(bm_pred))
    dist_to_gt_b = distance_transform_edt(np.logical_not(bm_gt))

    # A pred boundary pixel is matched if there's a gt boundary pixel
    # within `bound_pix` (inclusive). Symmetric for gt boundary pixels.
    pred_match = bm_pred & (dist_to_gt_b <= bound_pix)
    gt_match = bm_gt & (dist_to_pred_b <= bound_pix)

    precision = float(pred_match.sum()) / float(n_pred)
    recall = float(gt_match.sum()) / float(n_gt)
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def jf(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    bound_th: float = 0.008,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """Compute (J, F, J&F) — region IoU + boundary F + their mean.

    Args:
      mask_pred, mask_gt: [H, W] binary masks.

    Returns:
      (J, F, JF_mean) — JF_mean = (J + F) / 2 per DAVIS convention.
    """
    pred = (np.asarray(mask_pred) > 0).astype(np.uint8)
    gt = (np.asarray(mask_gt) > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    if union == 0:
        # Both empty → IoU = 1.0 by DAVIS convention.
        J = 1.0
    else:
        J = float(inter) / float(union)
    F = f_boundary(pred, gt, bound_th=bound_th)
    return J, F, 0.5 * (J + F)


# ===========================================================================
# Aggregator-only metrics (do NOT need re-eval; computable from per-frame J)
# ===========================================================================


def utr_at(per_frame_j_attacked: Sequence[float], tau: float = 0.3) -> float:
    """Unusable-Track Rate: fraction of frames with `J_attacked < tau`.

    From an operator's perspective, frames at `J<0.3` are "tracking lost"
    and would prompt user re-prompting. Higher UTR = more disruptive attack.
    """
    if not per_frame_j_attacked:
        return float("nan")
    n = len(per_frame_j_attacked)
    n_unusable = sum(1 for j in per_frame_j_attacked
                     if not _isnan(j) and j < tau)
    return float(n_unusable) / float(n)


def sfr_at(per_frame_j_attacked: Sequence[float], tau: float = 0.7) -> float:
    """Success-Frame Rate: fraction of frames with `J_attacked > tau`.

    Higher SFR = SAM2 still tracking acceptably. SFR + UTR + middle-band
    should sum to 1.0 for tau_unusable < tau_success.
    """
    if not per_frame_j_attacked:
        return float("nan")
    n = len(per_frame_j_attacked)
    n_success = sum(1 for j in per_frame_j_attacked
                    if not _isnan(j) and j > tau)
    return float(n_success) / float(n)


def _isnan(x: float) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


# ===========================================================================
# Per-frame fidelity (LPIPS / SSIM) — invoked during eval, needs lpips_fn /
# ssim_fn callables matching memshield.vadi_sam2_wiring's contracts.
# ===========================================================================


def per_frame_lpips_ssim(
    exported_t: "Tensor",
    processed_clean_t: "Tensor",
    lpips_fn: Optional[callable],
    ssim_fn: Optional[callable],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute LPIPS + SSIM between two single frames.

    Both inputs are [H, W, 3] in [0, 1] (matching the lpips_fn contract
    in memshield.vadi_sam2_wiring). ssim_fn expects [1, 3, H, W].

    Returns (lpips, ssim), with None for any disabled fn.
    """
    import torch  # type: ignore

    lp_val = None
    ss_val = None
    if lpips_fn is not None:
        with torch.no_grad():
            lp = lpips_fn(exported_t, processed_clean_t)
            lp_val = float(lp.detach().item() if hasattr(lp, "detach") else lp)
    if ssim_fn is not None:
        with torch.no_grad():
            x = exported_t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            y = processed_clean_t.permute(2, 0, 1).unsqueeze(0)
            ss = ssim_fn(x, y)
            ss_val = float(ss.detach().item() if hasattr(ss, "detach") else ss)
    return lp_val, ss_val


# ===========================================================================
# Aggregator helpers — compute summary stats grouped by is_insert
# ===========================================================================


def aggregate_by_insert(
    per_frame: Dict[int, Dict[str, float]],
    field: str,
) -> Dict[str, float]:
    """Group per-frame metric by `is_insert` and return means.

    Skips frames where `field` is missing or NaN. Returns dict with keys:
      mean        — over all frames
      mean_inserts  — mean over frames where is_insert=True
      mean_originals  — mean over frames where is_insert=False
      n_inserts, n_originals — sample counts (for reporting)
    """
    all_vals: List[float] = []
    insert_vals: List[float] = []
    original_vals: List[float] = []
    for t, rec in per_frame.items():
        if field not in rec:
            continue
        v = rec[field]
        if v is None or _isnan(v):
            continue
        v = float(v)
        all_vals.append(v)
        if rec.get("is_insert", False):
            insert_vals.append(v)
        else:
            original_vals.append(v)
    return {
        "mean": float(np.mean(all_vals)) if all_vals else float("nan"),
        "mean_inserts": (
            float(np.mean(insert_vals)) if insert_vals else float("nan")),
        "mean_originals": (
            float(np.mean(original_vals)) if original_vals else float("nan")),
        "n_inserts": len(insert_vals),
        "n_originals": len(original_vals),
    }


# ===========================================================================
# Self-tests
# ===========================================================================


def _test_seg2bmap_simple():
    """A 5x5 filled square produces a hollow boundary."""
    seg = np.zeros((5, 5), dtype=np.uint8)
    seg[1:4, 1:4] = 1
    bmap = _seg2bmap(seg)
    # Interior (2,2) is fully surrounded by 1s — NOT a boundary.
    assert not bmap[2, 2], f"interior pixel marked as boundary: {bmap}"
    # Edge transitions ARE boundary pixels.
    assert bmap[1, 1] or bmap[3, 3] or bmap[1, 3] or bmap[3, 1], \
        f"no corner boundary detected: {bmap}"


def _test_f_boundary_perfect_match():
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[5:15, 5:15] = 1
    f = f_boundary(seg, seg)
    assert abs(f - 1.0) < 1e-9, f"perfect match should give F=1.0, got {f}"


def _test_f_boundary_both_empty():
    z = np.zeros((10, 10), dtype=np.uint8)
    assert f_boundary(z, z) == 1.0, "both-empty should give F=1.0"


def _test_f_boundary_one_empty():
    seg = np.zeros((10, 10), dtype=np.uint8)
    seg[3:7, 3:7] = 1
    z = np.zeros((10, 10), dtype=np.uint8)
    assert f_boundary(seg, z) == 0.0
    assert f_boundary(z, seg) == 0.0


def _test_f_boundary_translated():
    """A 1-pixel shift should give very high F (boundaries within bandwidth)."""
    a = np.zeros((50, 50), dtype=np.uint8)
    a[20:30, 20:30] = 1
    b = np.zeros((50, 50), dtype=np.uint8)
    b[20:30, 21:31] = 1  # 1-pixel right shift
    f = f_boundary(a, b)
    # bandwidth = ceil(0.008 * sqrt(50²+50²)) = ceil(0.566) = 1 pixel; thus
    # F should be high (most boundary pixels match within 1px).
    assert f > 0.5, f"1-pixel shift F too low: {f}"


def _test_jf_combined():
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[5:15, 5:15] = 1
    j, f, jf_mean = jf(seg, seg)
    assert abs(j - 1.0) < 1e-9
    assert abs(f - 1.0) < 1e-9
    assert abs(jf_mean - 1.0) < 1e-9


def _test_utr_sfr():
    j_per_frame = [0.1, 0.25, 0.5, 0.8, 0.95]
    assert abs(utr_at(j_per_frame, 0.3) - 2 / 5) < 1e-9
    assert abs(sfr_at(j_per_frame, 0.7) - 2 / 5) < 1e-9


def _test_aggregate_by_insert():
    per_frame = {
        0: {"f": 0.9, "is_insert": False},
        1: {"f": 0.3, "is_insert": True},
        2: {"f": 0.85, "is_insert": False},
        3: {"f": 0.4, "is_insert": True},
    }
    agg = aggregate_by_insert(per_frame, "f")
    assert abs(agg["mean"] - 0.6125) < 1e-6
    assert abs(agg["mean_inserts"] - 0.35) < 1e-6
    assert abs(agg["mean_originals"] - 0.875) < 1e-6
    assert agg["n_inserts"] == 2
    assert agg["n_originals"] == 2


# ===========================================================================
# Original-timeline evaluation (codex round 25 — paper headline metric)
# ===========================================================================


def orig_to_attacked_indices(
    W_clean: Sequence[int], T_clean: int,
) -> List[int]:
    """Map each original-clip frame index `c` to its attacked-video position.

    The convention (matching `build_processed`): an insert at clean-space
    position `w_clean` appears BEFORE original frame `w_clean` in the
    attacked video. So if we have W_clean = [2, 12], the attacked
    sequence looks like

        [F0, F1, D0, F2, F3, ..., F11, D1, F12, F13, ...]

    and original frame `c` ends up at attacked index
    `t = c + #{w_clean in W_clean : w_clean <= c}`.

    Args:
      W_clean: insert positions in clean-space (sorted or unsorted).
      T_clean: number of original-clip frames.

    Returns:
      List of length T_clean with attacked-video index per original frame.
    """
    W_sorted = sorted(int(w) for w in W_clean)
    result: List[int] = []
    for c in range(int(T_clean)):
        n_before = sum(1 for w in W_sorted if w <= c)
        result.append(c + n_before)
    return result


def attacked_to_clean_W(W_attacked: Sequence[int]) -> List[int]:
    """Convert attacked-space W to clean-space via `W_clean[i] = W_att[i] - i`."""
    W_att_sorted = sorted(int(w) for w in W_attacked)
    return [w - i for i, w in enumerate(W_att_sorted)]


def load_davis_gt_per_frame(
    davis_root, clip_name: str,
) -> List[np.ndarray]:
    """Load DAVIS-2017 per-frame ground-truth annotations.

    Returns a list of length T_clean of `[H, W]` uint8 binary masks
    (foreground = any non-zero object id; multi-object → union).

    Args:
      davis_root: Path to DAVIS dataset root (containing
          `Annotations/480p/<clip>/<frame>.png`).
      clip_name: e.g., "cows".
    """
    import pathlib
    from PIL import Image
    root = pathlib.Path(davis_root)
    ann_dir = root / "Annotations" / "480p" / clip_name
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"DAVIS annotations missing for clip {clip_name!r}: {ann_dir}")
    files = sorted(ann_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(
            f"DAVIS annotations dir empty for clip {clip_name!r}: {ann_dir}")
    masks: List[np.ndarray] = []
    for f in files:
        a = np.asarray(Image.open(f))
        if a.ndim > 2:
            a = a[..., 0]
        masks.append((a > 0).astype(np.uint8))
    return masks


def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Standard binary IoU; both-empty → 1.0."""
    ai = (a > 0).astype(np.uint8)
    bi = (b > 0).astype(np.uint8)
    inter = int((ai & bi).sum())
    union = int((ai | bi).sum())
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def eval_original_timeline_metrics(
    sam2_eval_fn,
    prompt_mask: np.ndarray,
    x_clean,                         # [T_clean, H, W, 3] float in [0, 1]
    exported,                        # [T_proc, H, W, 3] float in [0, 1]
    W_attacked: Sequence[int],
    gt_original: Sequence[np.ndarray],   # T_clean ground-truth masks
    masks_attacked: Optional[Sequence[np.ndarray]] = None,
    lpips_fn: Optional[callable] = None,
    ssim_fn: Optional[callable] = None,
) -> Dict[str, object]:
    """Codex r25 paper-headline evaluator: compare clean-ORIGINAL vs
    attacked on the original-frame timeline, against DAVIS GT.

    Per original frame `c`:
      • J_clean_orig[c]    = jaccard(SAM2(x_clean)[c], gt_original[c])
      • J_attacked_orig[c] = jaccard(SAM2(exported)[orig_to_mod[c]], gt_original[c])
      • F similarly via DAVIS-style F-boundary.
      • JF = (J + F) / 2.
      • J_drop_orig[c]     = J_clean_orig[c] - J_attacked_orig[c]
      • LPIPS_upload[c]    = lpips(exported[orig_to_mod[c]], x_clean[c])
      • SSIM_upload[c]     = ssim(...) (per the existing fn contract).

    Aggregates: mean over c ∈ [0, T_clean) (exclusively original frames;
    inserted decoy frames are NOT in this metric — they go in the
    processed-space appendix metric).

    Args:
      sam2_eval_fn: SAM2 propagation callable (video, prompt) → list of masks.
      prompt_mask: [H, W] uint8 binary first-frame prompt.
      x_clean: [T_clean, H, W, 3] original clip (no inserts, no perturbation).
      exported: [T_proc, H, W, 3] reloaded attacked video.
      W_attacked: insert positions in attacked-space.
      gt_original: T_clean ground-truth masks.
      masks_attacked: pre-computed SAM2 output on `exported`. If None,
          this function will run sam2_eval_fn(exported, prompt_mask).
          (Allows reuse from the processed-space evaluator to avoid a
          duplicate forward pass.)
      lpips_fn / ssim_fn: optional fidelity callables.

    Returns:
      Dict with `_orig` aggregates + `per_orig_frame` dict.
    """
    import torch
    T_clean = int(x_clean.shape[0])
    T_proc = int(exported.shape[0])
    if len(gt_original) != T_clean:
        raise ValueError(
            f"gt_original length {len(gt_original)} != T_clean {T_clean}")

    W_clean = attacked_to_clean_W(W_attacked)
    orig_to_mod = orig_to_attacked_indices(W_clean, T_clean)
    if max(orig_to_mod) >= T_proc:
        raise ValueError(
            f"orig_to_mod max {max(orig_to_mod)} >= T_proc {T_proc}; "
            f"W_attacked={list(W_attacked)} inconsistent with exported length")

    # SAM2 on the unmodified original clip — the user's reference
    # baseline (what they would have gotten without any attack).
    masks_clean_orig = sam2_eval_fn(x_clean, prompt_mask)
    if len(masks_clean_orig) != T_clean:
        raise RuntimeError(
            f"sam2_eval_fn(x_clean) returned {len(masks_clean_orig)} masks; "
            f"expected {T_clean}")

    if masks_attacked is None:
        masks_attacked = sam2_eval_fn(exported, prompt_mask)
        if len(masks_attacked) != T_proc:
            raise RuntimeError(
                f"sam2_eval_fn(exported) returned {len(masks_attacked)} "
                f"masks; expected {T_proc}")

    per_orig: Dict[int, Dict[str, object]] = {}
    Js_clean: List[float] = []
    Js_att: List[float] = []
    Fs_clean: List[float] = []
    Fs_att: List[float] = []

    for c in range(T_clean):
        t = orig_to_mod[c]
        gt = (np.asarray(gt_original[c]) > 0).astype(np.uint8)
        m_clean = (np.asarray(masks_clean_orig[c]) > 0).astype(np.uint8)
        m_att = (np.asarray(masks_attacked[t]) > 0).astype(np.uint8)

        j_c = _jaccard_binary(m_clean, gt)
        j_a = _jaccard_binary(m_att, gt)
        f_c = f_boundary(m_clean, gt)
        f_a = f_boundary(m_att, gt)
        Js_clean.append(j_c)
        Js_att.append(j_a)
        Fs_clean.append(f_c)
        Fs_att.append(f_a)

        # LPIPS / SSIM on the upload-vs-original axis: how much did the
        # attacker's upload visibly change relative to the original clip?
        # For original frames, exported[orig_to_mod[c]] vs x_clean[c]
        # measures publisher-side stealth.
        lp_val: Optional[float] = None
        ss_val: Optional[float] = None
        if lpips_fn is not None or ssim_fn is not None:
            with torch.no_grad():
                if lpips_fn is not None:
                    lp = lpips_fn(exported[t], x_clean[c])
                    lp_val = float(
                        lp.detach().item() if hasattr(lp, "detach") else lp)
                if ssim_fn is not None:
                    x = exported[t].permute(2, 0, 1).unsqueeze(0)
                    y = x_clean[c].permute(2, 0, 1).unsqueeze(0)
                    ss = ssim_fn(x, y)
                    ss_val = float(
                        ss.detach().item() if hasattr(ss, "detach") else ss)

        per_orig[c] = {
            "J_clean_orig": j_c,
            "J_attacked_orig": j_a,
            "J_drop_orig": j_c - j_a,
            "F_clean_orig": f_c,
            "F_attacked_orig": f_a,
            "F_drop_orig": f_c - f_a,
            "JF_clean_orig": 0.5 * (j_c + f_c),
            "JF_attacked_orig": 0.5 * (j_a + f_a),
            "JF_drop_orig": 0.5 * (j_c + f_c) - 0.5 * (j_a + f_a),
            "lpips_upload": lp_val,
            "ssim_upload": ss_val,
            "attacked_index": t,
        }

    J_clean_orig_mean = float(np.mean(Js_clean))
    J_att_orig_mean = float(np.mean(Js_att))
    F_clean_orig_mean = float(np.mean(Fs_clean))
    F_att_orig_mean = float(np.mean(Fs_att))
    JF_clean_orig_mean = 0.5 * (J_clean_orig_mean + F_clean_orig_mean)
    JF_att_orig_mean = 0.5 * (J_att_orig_mean + F_att_orig_mean)

    utr_03_orig = utr_at(Js_att, tau=0.3)
    sfr_07_orig = sfr_at(Js_att, tau=0.7)

    lpips_vals = [v["lpips_upload"] for v in per_orig.values()
                  if v["lpips_upload"] is not None]
    ssim_vals = [v["ssim_upload"] for v in per_orig.values()
                 if v["ssim_upload"] is not None]

    return {
        "J_clean_orig_mean": J_clean_orig_mean,
        "J_attacked_orig_mean": J_att_orig_mean,
        "J_drop_orig_mean": J_clean_orig_mean - J_att_orig_mean,
        "F_clean_orig_mean": F_clean_orig_mean,
        "F_attacked_orig_mean": F_att_orig_mean,
        "F_drop_orig_mean": F_clean_orig_mean - F_att_orig_mean,
        "JF_clean_orig_mean": JF_clean_orig_mean,
        "JF_attacked_orig_mean": JF_att_orig_mean,
        "JF_drop_orig_mean": JF_clean_orig_mean - JF_att_orig_mean,
        "UTR_at_03_orig": utr_03_orig,
        "SFR_at_07_orig": sfr_07_orig,
        "LPIPS_upload_mean": (
            float(np.mean(lpips_vals)) if lpips_vals else float("nan")),
        "SSIM_upload_mean": (
            float(np.mean(ssim_vals)) if ssim_vals else float("nan")),
        "n_original_frames": T_clean,
        "per_orig_frame": per_orig,
        # for downstream reuse so callers can avoid re-running SAM2:
        "masks_clean_original": masks_clean_orig,
        "masks_attacked": masks_attacked,
    }


# ===========================================================================
# Self-tests for original-timeline metrics
# ===========================================================================


def _test_orig_to_attacked_basic():
    # W_clean=[2], T_clean=5  →  attacked = [F0, F1, D, F2, F3, F4]
    mapping = orig_to_attacked_indices([2], 5)
    assert mapping == [0, 1, 3, 4, 5], f"got {mapping}"


def _test_orig_to_attacked_multi():
    # W_clean=[2, 12, 26], T_clean=50
    mapping = orig_to_attacked_indices([2, 12, 26], 50)
    assert mapping[0] == 0 and mapping[1] == 1, f"got {mapping[:2]}"
    assert mapping[2] == 3, f"c=2 expected 3, got {mapping[2]}"
    assert mapping[12] == 14, f"c=12 expected 14, got {mapping[12]}"
    assert mapping[26] == 29, f"c=26 expected 29, got {mapping[26]}"
    assert mapping[49] == 52, f"c=49 expected 52, got {mapping[49]}"


def _test_attacked_to_clean_W_basic():
    # W_att=[2,13,28] → W_clean=[2, 12, 26]
    assert attacked_to_clean_W([2, 13, 28]) == [2, 12, 26]
    assert attacked_to_clean_W([28, 2, 13]) == [2, 12, 26]   # unsorted input ok


if __name__ == "__main__":
    _test_seg2bmap_simple()
    _test_f_boundary_perfect_match()
    _test_f_boundary_both_empty()
    _test_f_boundary_one_empty()
    _test_f_boundary_translated()
    _test_jf_combined()
    _test_utr_sfr()
    _test_aggregate_by_insert()
    _test_orig_to_attacked_basic()
    _test_orig_to_attacked_multi()
    _test_attacked_to_clean_W_basic()
    print("eval_metrics_extended self-tests PASSED (11 tests)")
