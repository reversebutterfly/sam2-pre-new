"""Decoy-semantic evaluation metrics for VADI-v5.A0 (2026-04-24).

Codex Round 3 requirement: J-drop alone cannot distinguish between
"redirected toward decoy" and "true object suppressed to empty". A
"decoy redirection" paper claim needs per-frame decomposition of attack
mode, so appendix / main-table can report what the attack is ACTUALLY
doing mechanistically.

## Modes of attack outcome per frame

Given on a single frame:
  - `pred_clean`:   SAM2 mask on the clean-processed baseline, {0,1}^HxW
  - `pred_attacked`: SAM2 mask on the exported-attacked video, {0,1}^HxW
  - `m_true`:       pseudo-GT from clean SAM2 on the original, {0,1}^HxW
  - `m_decoy`:      spatially-shifted pseudo-GT (decoy target), {0,1}^HxW

We compute:
  - `retention = area(pred_attacked) / area(pred_clean)` — does the
    prediction still exist or did it shrink?
  - `J_attacked_vs_true`  = Jaccard(pred_attacked, m_true)
  - `J_attacked_vs_decoy` = Jaccard(pred_attacked, m_decoy)
  - `centroid_displacement` = centroid(pred_attacked) − centroid(pred_clean)
  - `decoy_alignment` = cosine(displacement, decoy_offset_vector)

And classify the frame into ONE mode:
  - `intact`:     `J_attacked_vs_true ≥ INTACT_TRUE_J_THR` (no effective change)
  - `suppressed`: `retention < SUPPRESS_RETENTION_THR` (pred collapsed)
  - `redirected`: `J_attacked_vs_decoy ≥ REDIRECT_DECOY_J_THR`
                  AND `J_attacked_vs_decoy > J_attacked_vs_true`
                  AND `retention ≥ REDIRECT_MIN_RETENTION`
  - `split`:      both `J_attacked_vs_decoy ≥ SPLIT_J_THR`
                  AND `J_attacked_vs_true ≥ SPLIT_J_THR`
                  AND `retention ≥ 0.5`
  - `degraded`:   everything else (attack changed pred but not cleanly
                  suppressed / redirected / split)

Thresholds are hyperparameters documented below. For the paper we want
`redirected` rate to be high compared to `suppressed` rate — that is
the empirical support for the decoy-redirection claim.

## Aggregation

Over N frames: count per mode, report fractions + mean J_attacked_vs_decoy
and mean J_attacked_vs_true separately. Also report mean retention and
mean decoy alignment (restricted to frames where both centroids exist).

## Self-test

`python -m memshield.decoy_semantic_metrics` → synthetic-input sanity
checks for each mode + aggregation. No SAM2 dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Classification thresholds (pre-committed; documenting as constants so
# paper tables can cite them without having to dig through code)
# =============================================================================


INTACT_TRUE_J_THR: float = 0.80       # "intact" if J(pred_a, m_true) ≥ this
SUPPRESS_RETENTION_THR: float = 0.20  # "suppressed" if retention < this

REDIRECT_DECOY_J_THR: float = 0.30    # "redirected" requires J(pred_a, m_decoy) ≥ this
REDIRECT_MIN_RETENTION: float = 0.40  # plus retention ≥ this (not an empty mask)

SPLIT_J_THR: float = 0.20             # "split" if pred overlaps both ≥ this
SPLIT_MIN_RETENTION: float = 0.50


# =============================================================================
# Primitive metrics
# =============================================================================


def _as_binary(m: np.ndarray) -> np.ndarray:
    """Coerce to {0, 1} uint8 for robust set math."""
    return (np.asarray(m) > 0).astype(np.uint8)


def mask_area(m: np.ndarray) -> int:
    """Count of positive pixels."""
    return int(_as_binary(m).sum())


def mask_centroid(m: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (y, x) centroid of positive pixels. None if mask is empty."""
    mb = _as_binary(m)
    if mb.sum() == 0:
        return None
    ys, xs = np.where(mb > 0)
    return (float(ys.mean()), float(xs.mean()))


def jaccard_binary(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Binary Jaccard IoU. Both-empty → 1.0 (matches eval_v2 convention)."""
    ab = _as_binary(a); bb = _as_binary(b)
    if ab.shape != bb.shape:
        raise ValueError(f"shape mismatch: {ab.shape} vs {bb.shape}")
    inter = int(np.logical_and(ab, bb).sum())
    union = int(np.logical_or(ab, bb).sum())
    if union == 0:
        return 1.0
    return inter / (union + eps) if eps > 0 else inter / union


def retention_ratio(pred_attacked: np.ndarray, pred_clean: np.ndarray) -> float:
    """`area(pred_attacked) / max(area(pred_clean), 1)`. Clipped to [0, inf).

    If the clean prediction itself is empty (degenerate clip), returns
    `inf` — caller can treat this as "unusable" and exclude from means.
    """
    a_area = mask_area(pred_attacked)
    c_area = mask_area(pred_clean)
    if c_area == 0:
        return float("inf")
    return float(a_area) / float(c_area)


def centroid_displacement(
    pred_attacked: np.ndarray, pred_clean: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """centroid(pred_attacked) − centroid(pred_clean) as (dy, dx).

    Returns None if either mask is empty (displacement undefined).
    """
    ca = mask_centroid(pred_attacked)
    cc = mask_centroid(pred_clean)
    if ca is None or cc is None:
        return None
    return (ca[0] - cc[0], ca[1] - cc[1])


def decoy_alignment(
    displacement: Optional[Tuple[float, float]],
    decoy_offset: Tuple[int, int],
) -> Optional[float]:
    """Cosine similarity between prediction's centroid displacement and
    the nominal decoy offset vector `(dy, dx)`.

    Returns None if `displacement` is None (empty mask) or if either
    vector has zero norm.
    """
    if displacement is None:
        return None
    dy_p, dx_p = displacement
    dy_d, dx_d = float(decoy_offset[0]), float(decoy_offset[1])
    n_p = (dy_p * dy_p + dx_p * dx_p) ** 0.5
    n_d = (dy_d * dy_d + dx_d * dx_d) ** 0.5
    if n_p < 1e-8 or n_d < 1e-8:
        return None
    return (dy_p * dy_d + dx_p * dx_d) / (n_p * n_d)


# =============================================================================
# Per-frame classification
# =============================================================================


def classify_attack_mode(
    retention: float, j_vs_true: float, j_vs_decoy: float,
) -> str:
    """Return one of {'intact', 'suppressed', 'redirected', 'split', 'degraded'}.

    Precedence order (first match wins):
      1. intact (no effective attack)
      2. suppressed (pred collapsed)
      3. redirected (pred aligned with decoy and not just the true)
      4. split (pred covers both)
      5. degraded (everything else)

    The order matters: a frame could technically satisfy multiple
    conditions (e.g., intact AND split), so we commit to precedence.
    """
    if j_vs_true >= INTACT_TRUE_J_THR:
        return "intact"
    if retention < SUPPRESS_RETENTION_THR:
        return "suppressed"
    redirected = (
        j_vs_decoy >= REDIRECT_DECOY_J_THR
        and j_vs_decoy > j_vs_true
        and retention >= REDIRECT_MIN_RETENTION
    )
    if redirected:
        return "redirected"
    split_cond = (
        j_vs_decoy >= SPLIT_J_THR
        and j_vs_true >= SPLIT_J_THR
        and retention >= SPLIT_MIN_RETENTION
    )
    if split_cond:
        return "split"
    return "degraded"


# =============================================================================
# Per-frame record + aggregate
# =============================================================================


@dataclass
class FrameDecoySemantic:
    """Per-frame decoy-semantic record.

    `valid`=True indicates this frame contributes to the aggregate mode
    rates. Excluded frames (pred_clean empty OR pre-first-insert) carry
    `valid=False` and an `exclusion_reason` tag; they are reported in
    separate counters in the aggregate, not mixed into the main rates.
    """

    t: int                         # processed-space frame index
    is_insert: bool                # whether t is an insert position
    mode: str                      # one of the 5 modes, or "excluded"
    valid: bool                    # True if frame contributes to mode rates
    exclusion_reason: Optional[str]  # None if valid, else one of:
                                   # "empty_pred_clean", "pre_first_insert"
    retention: float
    j_vs_true: float
    j_vs_decoy: float
    centroid_pred_clean: Optional[Tuple[float, float]]
    centroid_pred_attacked: Optional[Tuple[float, float]]
    centroid_displacement: Optional[Tuple[float, float]]
    decoy_alignment_cos: Optional[float]


def per_frame_decoy_semantic(
    t: int, is_insert: bool,
    pred_clean: np.ndarray,
    pred_attacked: np.ndarray,
    m_true: np.ndarray,
    m_decoy: np.ndarray,
    decoy_offset: Tuple[int, int],
    *,
    is_pre_first_insert: bool = False,
) -> FrameDecoySemantic:
    """Compute full per-frame record.

    `is_pre_first_insert=True` marks frames before the first insert
    position: m_decoy is zeros by construction there, so the decoy-
    semantic classification does not apply. Such frames are flagged
    `valid=False`, `exclusion_reason="pre_first_insert"`.

    Frames where `pred_clean` is empty are also flagged
    `valid=False`, `exclusion_reason="empty_pred_clean"` (SAM2 failed
    on the baseline for that frame; retention and several centroids
    are undefined). Per codex R1 post-fix 2026-04-24.
    """
    j_t = jaccard_binary(pred_attacked, m_true)
    j_d = jaccard_binary(pred_attacked, m_decoy)
    c_clean = mask_centroid(pred_clean)
    c_att = mask_centroid(pred_attacked)
    disp = centroid_displacement(pred_attacked, pred_clean)
    align = decoy_alignment(disp, decoy_offset)

    # Exclusion checks BEFORE classification. Empty-clean → unusable.
    empty_clean = (mask_area(pred_clean) == 0)
    if empty_clean:
        return FrameDecoySemantic(
            t=int(t), is_insert=bool(is_insert),
            mode="excluded", valid=False,
            exclusion_reason="empty_pred_clean",
            retention=float("nan"),
            j_vs_true=float(j_t), j_vs_decoy=float(j_d),
            centroid_pred_clean=c_clean, centroid_pred_attacked=c_att,
            centroid_displacement=disp, decoy_alignment_cos=align,
        )
    if is_pre_first_insert:
        r = retention_ratio(pred_attacked, pred_clean)
        return FrameDecoySemantic(
            t=int(t), is_insert=bool(is_insert),
            mode="excluded", valid=False,
            exclusion_reason="pre_first_insert",
            retention=float(r),
            j_vs_true=float(j_t), j_vs_decoy=float(j_d),
            centroid_pred_clean=c_clean, centroid_pred_attacked=c_att,
            centroid_displacement=disp, decoy_alignment_cos=align,
        )

    r = retention_ratio(pred_attacked, pred_clean)
    # retention is a finite positive number here (empty_clean excluded above).
    r_clamped = min(r, 10.0)
    mode = classify_attack_mode(r_clamped, j_t, j_d)
    return FrameDecoySemantic(
        t=int(t), is_insert=bool(is_insert), mode=mode,
        valid=True, exclusion_reason=None,
        retention=float(r), j_vs_true=float(j_t), j_vs_decoy=float(j_d),
        centroid_pred_clean=c_clean, centroid_pred_attacked=c_att,
        centroid_displacement=disp, decoy_alignment_cos=align,
    )


@dataclass
class DecoySemanticAggregate:
    """Aggregate metrics.

    Only `valid=True` frames contribute to `mode_counts` / `mode_rates`
    / `mean_*`. Excluded frames are tallied separately in
    `n_excluded_*` so the reader can see what was filtered.
    """
    n_frames_total: int            # total records seen
    n_frames_valid: int            # records contributing to rates
    n_excluded_empty_pred_clean: int
    n_excluded_pre_first_insert: int
    mode_counts: Dict[str, int]    # only over valid frames
    mode_rates: Dict[str, float]
    mean_retention: float
    mean_j_vs_true: float
    mean_j_vs_decoy: float
    mean_decoy_alignment_cos: Optional[float]
    mode_rates_inserts_only: Dict[str, float]
    mode_rates_originals_only: Dict[str, float]


def aggregate_decoy_semantic(
    records: Sequence[FrameDecoySemantic],
) -> DecoySemanticAggregate:
    """Aggregate per-frame records. Excluded frames are tracked separately
    and do NOT bias the mode rates (codex R1 post-fix 2026-04-24).
    """
    total = len(records)
    valid_records = [r for r in records if r.valid]
    n_valid = len(valid_records)
    n_empty = sum(1 for r in records
                  if r.exclusion_reason == "empty_pred_clean")
    n_pre = sum(1 for r in records
                if r.exclusion_reason == "pre_first_insert")

    MODES = ["intact", "suppressed", "redirected", "split", "degraded"]
    counts = {m: 0 for m in MODES}
    counts_ins = {m: 0 for m in MODES}
    counts_orig = {m: 0 for m in MODES}
    j_true_sum = 0.0; j_decoy_sum = 0.0; retention_sum = 0.0
    align_sum = 0.0; align_n = 0
    n_ins = 0; n_orig = 0
    for r in valid_records:
        counts[r.mode] += 1
        if r.is_insert:
            counts_ins[r.mode] += 1; n_ins += 1
        else:
            counts_orig[r.mode] += 1; n_orig += 1
        j_true_sum += r.j_vs_true; j_decoy_sum += r.j_vs_decoy
        retention_sum += r.retention
        if r.decoy_alignment_cos is not None:
            align_sum += r.decoy_alignment_cos; align_n += 1

    def _rates(cs: Dict[str, int], t: int) -> Dict[str, float]:
        if t == 0:
            return {k: 0.0 for k in cs}
        return {k: v / t for k, v in cs.items()}

    return DecoySemanticAggregate(
        n_frames_total=total,
        n_frames_valid=n_valid,
        n_excluded_empty_pred_clean=n_empty,
        n_excluded_pre_first_insert=n_pre,
        mode_counts=counts,
        mode_rates=_rates(counts, n_valid),
        mean_retention=(retention_sum / n_valid) if n_valid else 0.0,
        mean_j_vs_true=(j_true_sum / n_valid) if n_valid else 0.0,
        mean_j_vs_decoy=(j_decoy_sum / n_valid) if n_valid else 0.0,
        mean_decoy_alignment_cos=(align_sum / align_n) if align_n > 0 else None,
        mode_rates_inserts_only=_rates(counts_ins, n_ins),
        mode_rates_originals_only=_rates(counts_orig, n_orig),
    )


# =============================================================================
# Helper: build m_decoy trajectory from pseudo-masks + decoy offsets
# =============================================================================


def build_decoy_mask_trajectory(
    pseudo_masks: Sequence[np.ndarray],      # per clean-frame, [H, W] float/binary
    W_clean_positions: Sequence[int],        # clean-space insert anchors c_k
    decoy_offsets: Sequence[Tuple[int, int]],  # (dy_k, dx_k) per insert
) -> List[np.ndarray]:
    """Rebuild the decoy mask trajectory used during v5 optimization.

    For each clean frame t ∈ [0, T_clean):
      - if t < min(W_clean_positions): m_decoy[t] = zeros (pre-first-insert)
      - else: k_cover = max k such that c_k ≤ t
              m_decoy[t] = shift(pseudo_masks[t], offset_{k_cover})

    This replicates the v5 driver's m_decoy_clean_np construction so we
    can post-hoc recompute decoy-semantic metrics on an existing run.

    Returns a list of `[H, W]` uint8 binary masks.
    """
    from memshield.decoy_seed import shift_mask_np

    T = len(pseudo_masks)
    W_sorted = sorted(int(c) for c in W_clean_positions)
    offsets_sorted = [decoy_offsets[i] for i, _ in
                      sorted(enumerate(W_clean_positions),
                             key=lambda kv: int(kv[1]))]
    out: List[np.ndarray] = []
    for t in range(T):
        k_cover = -1
        for k, c_k in enumerate(W_sorted):
            if c_k <= t:
                k_cover = k
            else:
                break
        if k_cover == -1:
            out.append(np.zeros(pseudo_masks[t].shape, dtype=np.uint8))
        else:
            dy, dx = int(offsets_sorted[k_cover][0]), \
                     int(offsets_sorted[k_cover][1])
            shifted = shift_mask_np(
                (np.asarray(pseudo_masks[t]) > 0.5).astype(np.float32),
                dy, dx,
            )
            out.append((shifted > 0.5).astype(np.uint8))
    return out


def remap_masks_to_processed_space_decoy(
    m_decoy_clean: Sequence[np.ndarray],
    pseudo_masks_clean: Sequence[np.ndarray],   # source for insert overrides
    W_attacked: Sequence[int],
    decoy_offsets: Sequence[Tuple[int, int]],
) -> Dict[int, np.ndarray]:
    """Remap m_decoy to processed-space with v5 override at insert positions.

    v5 rule: at insert t=W_k (processed index), m_decoy[W_k] =
    shift(m_true_clean[c_k], offset_k); everywhere else m_decoy[t] =
    m_decoy_clean[attacked_to_clean(t, W)].

    Returns dict processed_t → `[H, W]` uint8 binary.
    """
    from memshield.decoy_seed import shift_mask_np
    from memshield.vadi_optimize import attacked_to_clean

    T_clean = len(m_decoy_clean)
    W_sorted = sorted(int(w) for w in W_attacked)
    T_proc = T_clean + len(W_sorted)
    insert_set = set(W_sorted)
    # Map each insert's processed position to its clean index c_k.
    # c_k = W_k - k after sorting (since k inserts with positions < W_k
    # are before it in processed order).
    proc_to_c_k: Dict[int, int] = {
        w: (w - k) for k, w in enumerate(W_sorted)
    }
    # Sort offsets to match sorted W.
    offs_sorted = [decoy_offsets[i] for i, _ in
                   sorted(enumerate(W_attacked),
                          key=lambda kv: int(kv[1]))]
    k_for_wsorted: Dict[int, int] = {w: i for i, w in enumerate(W_sorted)}

    out: Dict[int, np.ndarray] = {}
    for t in range(T_proc):
        if t in insert_set:
            c_k = proc_to_c_k[t]
            k = k_for_wsorted[t]
            dy, dx = int(offs_sorted[k][0]), int(offs_sorted[k][1])
            src = (np.asarray(pseudo_masks_clean[c_k]) > 0.5).astype(np.float32)
            out[t] = (shift_mask_np(src, dy, dx) > 0.5).astype(np.uint8)
        else:
            c = attacked_to_clean(t, W_sorted)
            out[t] = (np.asarray(m_decoy_clean[c]) > 0.5).astype(np.uint8)
    return out


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    np.random.seed(0)

    # -- mask_area / mask_centroid / jaccard
    m = np.zeros((10, 10), dtype=np.uint8); m[3:7, 3:7] = 1
    assert mask_area(m) == 16
    c = mask_centroid(m); assert c == (4.5, 4.5)
    assert mask_centroid(np.zeros((10, 10))) is None
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert jaccard_binary(empty, empty) == 1.0
    assert jaccard_binary(m, m) > 0.999
    m2 = np.zeros_like(m); m2[3:5, 3:5] = 1
    j = jaccard_binary(m, m2); assert 0.2 < j < 0.3

    # -- retention ratio
    pred_c = np.zeros((10, 10), dtype=np.uint8); pred_c[2:8, 2:8] = 1  # area 36
    pred_a = np.zeros((10, 10), dtype=np.uint8); pred_a[3:6, 3:6] = 1  # area 9
    r = retention_ratio(pred_a, pred_c)
    assert abs(r - 9/36) < 1e-6
    # empty clean → inf
    assert retention_ratio(pred_a, empty) == float("inf")

    # -- centroid displacement + decoy alignment
    # pred_a = [3:6, 3:6] → centroid (4.0, 4.0); pred_c = [2:8, 2:8] → (4.5, 4.5)
    disp = centroid_displacement(pred_a, pred_c)
    assert disp is not None
    assert abs(disp[0] - (-0.5)) < 1e-6 and abs(disp[1] - (-0.5)) < 1e-6
    # Build two aligned masks (both centered at (4, 4)) so disp==(0,0).
    pred_c_small = pred_a.copy()
    disp_zero = centroid_displacement(pred_a, pred_c_small)
    assert disp_zero == (0.0, 0.0)
    # Move pred_a right by 2 pixels relative to pred_c_small
    pred_a2 = np.zeros((10, 10), dtype=np.uint8); pred_a2[3:6, 5:8] = 1
    disp2 = centroid_displacement(pred_a2, pred_c_small)
    assert abs(disp2[0]) < 1e-6
    assert abs(disp2[1] - 2.0) < 1e-6   # dx=+2
    # alignment with (0, +5) decoy offset should be ~+1 (perfectly aligned)
    a = decoy_alignment(disp2, (0, 5))
    assert a is not None and a > 0.99
    # alignment with (0, -5) decoy should be ~-1
    a_neg = decoy_alignment(disp2, (0, -5))
    assert a_neg < -0.99
    # Zero-displacement → None
    zero_disp = (0.0, 0.0)
    assert decoy_alignment(zero_disp, (0, 5)) is None
    # None displacement → None alignment
    assert decoy_alignment(None, (0, 5)) is None

    # -- classify_attack_mode: each category
    # INTACT: pred matches true completely
    assert classify_attack_mode(retention=1.0, j_vs_true=0.95, j_vs_decoy=0.05) == "intact"
    # SUPPRESSED: retention < 0.2
    assert classify_attack_mode(retention=0.1, j_vs_true=0.4, j_vs_decoy=0.1) == "suppressed"
    # REDIRECTED: j_vs_decoy high, j_vs_true low, retention ≥ 0.4
    assert classify_attack_mode(retention=0.8, j_vs_true=0.15, j_vs_decoy=0.5) == "redirected"
    # SPLIT: both ≥ 0.2
    assert classify_attack_mode(retention=0.8, j_vs_true=0.3, j_vs_decoy=0.3) == "split"
    # DEGRADED: low j_vs_decoy + not-empty pred + not intact
    assert classify_attack_mode(retention=0.5, j_vs_true=0.4, j_vs_decoy=0.1) == "degraded"
    # Edge: intact wins over suppressed when both ostensibly hold (shouldn't
    # occur in practice, but precedence defined)
    assert classify_attack_mode(retention=0.1, j_vs_true=0.9, j_vs_decoy=0.05) == "intact"

    # -- per_frame_decoy_semantic
    H, W = 20, 20
    m_true = np.zeros((H, W), dtype=np.uint8); m_true[5:10, 5:10] = 1
    m_decoy = np.zeros((H, W), dtype=np.uint8); m_decoy[5:10, 13:18] = 1  # shifted +8 in x
    pred_clean = m_true.copy()
    pred_attacked = m_decoy.copy()   # perfectly redirected
    rec = per_frame_decoy_semantic(
        t=5, is_insert=False, pred_clean=pred_clean, pred_attacked=pred_attacked,
        m_true=m_true, m_decoy=m_decoy, decoy_offset=(0, 8),
    )
    assert rec.mode == "redirected", f"expected redirected, got {rec.mode}"
    assert rec.j_vs_decoy > 0.99
    assert rec.j_vs_true < 0.05
    assert rec.decoy_alignment_cos is not None and rec.decoy_alignment_cos > 0.99

    # -- aggregation (3 valid frames + 1 pre-first-insert excluded + 1 empty-clean excluded)
    recs = [
        per_frame_decoy_semantic(t=0, is_insert=True, pred_clean=pred_clean,
                                 pred_attacked=pred_attacked,
                                 m_true=m_true, m_decoy=m_decoy, decoy_offset=(0, 8)),
        per_frame_decoy_semantic(t=1, is_insert=False, pred_clean=pred_clean,
                                 pred_attacked=pred_clean,
                                 m_true=m_true, m_decoy=m_decoy, decoy_offset=(0, 8)),
        per_frame_decoy_semantic(t=2, is_insert=False, pred_clean=pred_clean,
                                 pred_attacked=np.zeros_like(pred_clean),
                                 m_true=m_true, m_decoy=m_decoy, decoy_offset=(0, 8)),
        # Exclusion: pre-first-insert
        per_frame_decoy_semantic(t=3, is_insert=False, pred_clean=pred_clean,
                                 pred_attacked=pred_clean,
                                 m_true=m_true, m_decoy=np.zeros_like(m_decoy),
                                 decoy_offset=(0, 0), is_pre_first_insert=True),
        # Exclusion: empty pred_clean (SAM2 failed on baseline)
        per_frame_decoy_semantic(t=4, is_insert=False,
                                 pred_clean=np.zeros_like(pred_clean),
                                 pred_attacked=pred_attacked,
                                 m_true=m_true, m_decoy=m_decoy, decoy_offset=(0, 8)),
    ]
    # modes: redirected, intact, suppressed, excluded (pre), excluded (empty-clean)
    assert recs[0].mode == "redirected"
    assert recs[1].mode == "intact"
    assert recs[2].mode == "suppressed"
    assert recs[3].mode == "excluded"
    assert recs[3].exclusion_reason == "pre_first_insert"
    assert recs[4].mode == "excluded"
    assert recs[4].exclusion_reason == "empty_pred_clean"
    agg = aggregate_decoy_semantic(recs)
    assert agg.n_frames_total == 5
    assert agg.n_frames_valid == 3      # excluded 2
    assert agg.n_excluded_pre_first_insert == 1
    assert agg.n_excluded_empty_pred_clean == 1
    assert agg.mode_counts["redirected"] == 1
    assert agg.mode_counts["intact"] == 1
    assert agg.mode_counts["suppressed"] == 1
    # Rates over VALID frames only.
    assert abs(agg.mode_rates["redirected"] - 1/3) < 1e-6
    assert abs(agg.mode_rates["intact"] - 1/3) < 1e-6
    # Insert-only (1 valid is_insert → redirected): 1/1
    assert agg.mode_rates_inserts_only["redirected"] == 1.0
    # Originals-only (2 valid !is_insert: intact + suppressed)
    assert abs(agg.mode_rates_originals_only["intact"] - 0.5) < 1e-6
    assert abs(agg.mode_rates_originals_only["suppressed"] - 0.5) < 1e-6
    # alignment over valid frames with valid alignment (only redirected qualifies)
    assert agg.mean_decoy_alignment_cos is not None
    assert agg.mean_decoy_alignment_cos > 0.99

    # -- build_decoy_mask_trajectory
    T = 10
    pseudo_masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(T)]
    for t in range(T):
        pseudo_masks[t][5:10, t:t + 5] = 1  # object drifts right by t each frame
    W_clean = [2, 5, 8]
    decoy_offsets = [(0, 8), (0, 6), (0, 4)]
    m_decoy_traj = build_decoy_mask_trajectory(pseudo_masks, W_clean, decoy_offsets)
    assert len(m_decoy_traj) == T
    # t=0, 1: pre-first-insert → zeros
    assert m_decoy_traj[0].sum() == 0
    assert m_decoy_traj[1].sum() == 0
    # t=2: uses offset (0, 8) on pseudo_masks[2]. Object at [5:10, 2:7] → shifted to [5:10, 10:15]
    assert m_decoy_traj[2][5:10, 10:15].sum() > 0
    # t=5: k_cover=1, offset (0, 6). Object at [5:10, 5:10] → [5:10, 11:16]
    assert m_decoy_traj[5][5:10, 11:16].sum() > 0
    # t=9: k_cover=2, offset (0, 4). Object at [5:10, 9:14] → [5:10, 13:18]
    assert m_decoy_traj[9][5:10, 13:18].sum() > 0

    # -- remap_masks_to_processed_space_decoy
    W_att = [c + k for k, c in enumerate(sorted(W_clean))]  # [2+0, 5+1, 8+2] = [2, 6, 10]
    m_decoy_clean = m_decoy_traj
    decoy_proc = remap_masks_to_processed_space_decoy(
        m_decoy_clean, pseudo_masks, W_att, decoy_offsets,
    )
    T_proc = T + len(W_att)
    assert set(decoy_proc.keys()) == set(range(T_proc))
    # At insert position W_att[0]=2 (c_k=2, offset (0,8)): shifted pseudo_masks[2]
    # = [5:10, 10:15]. Test that.
    assert decoy_proc[2][5:10, 10:15].sum() > 0

    # Non-insert processed index: e.g., t=0 (before any insert) in processed
    # space maps to clean 0 → zeros (pre-insert decoy).
    assert decoy_proc[0].sum() == 0

    print("memshield.decoy_semantic_metrics: all self-tests PASSED "
          "(area/centroid/jaccard/retention/displacement/alignment, "
          "5-mode classification + precedence, per_frame + aggregation, "
          "build_decoy_mask_trajectory, remap_masks_to_processed_space_decoy)")


if __name__ == "__main__":
    _self_test()
