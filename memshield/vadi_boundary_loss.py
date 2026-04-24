"""Boundary-weighted decoy tracking loss for VADI-v5 δ polish stage.

Codex Round 1 design #1 (2026-04-24): the loss for the short δ-polish
stage after A0's ν-only run. Key differences from the standard
`vadi_v5_loss` aggregate:

  1. **Boundary-weighted Dice/BCE**: the supervision is weighted by the
     per-frame δ support mask (from `memshield.boundary_bands`). This
     focuses the gradient signal on the same spatial region δ is
     allowed to act in — so ν + δ coordinate to move the mask boundary
     rather than fight across disjoint regions.

  2. **Anti-true on the true-boundary band only**: instead of penalizing
     pred overlap with the ENTIRE true mask, penalize only the pixels
     that are in the true-boundary band. The interior of m_true is
     NOT where the prediction-vs-truth contest is fought; the band is.

  3. **Optional signed-distance / contour loss**: push pred toward
     m_decoy using a signed-distance field (|dist_to_∂m_decoy|
     weighted sign). This can help when the decoy mask is far from the
     original pred — Dice/BCE gradient on disjoint regions can vanish.
     Disabled by default (gated by `contour_weight`).

## Loss form

For each queried frame `t`:

    band_true_t  = boundary_band(m_true_t, band_width)
    band_decoy_t = boundary_band(m_decoy_t, band_width)
    support_t    = band_true_t ∪ band_decoy_t ∪ corridor (the δ mask)

    L_dice_decoy_band_t = DiceLoss(
        pred_logits_t * support_t, m_decoy_t * support_t)
    L_bce_decoy_band_t = BCEWithLogitsLoss(
        pred_logits_t, m_decoy_t, weight=support_t)
    L_anti_true_band_t = Σ σ(pred_logits_t) · band_true_t   /   (Σ band_true_t + eps)

    L_t = α·L_dice_decoy_band_t + β·L_bce_decoy_band_t + γ·L_anti_true_band_t

Aggregate over selected frames (`insert_ids ∪ polish_frames`).

## Self-test

`python -m memshield.vadi_boundary_loss` → synthetic-input checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# Per-frame boundary-weighted primitives
# =============================================================================


def boundary_weighted_dice(
    pred_logits: Tensor,          # [H, W]
    m_decoy: Tensor,              # [H, W] in [0, 1]
    support_mask: Tensor,         # [H, W] in [0, 1]
    eps: float = 1.0,
) -> Tensor:
    """Dice loss computed only on the support region.

    Weights both `p = σ(pred)` and `m_decoy` by `support_mask` before
    computing intersection / union. Result is a scalar.
    """
    p = torch.sigmoid(pred_logits) * support_mask
    t = m_decoy * support_mask
    inter = (p * t).sum()
    total = p.sum() + t.sum()
    dice = (2.0 * inter + eps) / (total + eps)
    return 1.0 - dice


def boundary_weighted_bce(
    pred_logits: Tensor, m_decoy: Tensor, support_mask: Tensor,
) -> Tensor:
    """BCE with per-pixel weight = support_mask. Normalized so that
    empty-support returns 0 (not NaN) to avoid degenerate gradients.
    """
    # F.binary_cross_entropy_with_logits already supports weight.
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, m_decoy, weight=support_mask, reduction="sum",
    )
    denom = support_mask.sum().clamp(min=1.0)
    return bce / denom


def anti_true_boundary(
    pred_logits: Tensor, band_true: Tensor, eps: float = 1.0,
) -> Tensor:
    """Soft fraction of the true-boundary band covered by the prediction.

    L = Σ σ(pred)·band_true / (Σ band_true + eps). Encourages pred to
    NOT overlap the boundary of the true mask (the contested region).
    Inside the true mask interior is NOT penalized here (that's handled
    implicitly by the Dice term making pred match m_decoy).
    """
    p = torch.sigmoid(pred_logits)
    overlap = (p * band_true).sum()
    total = band_true.sum() + eps
    return overlap / total


# =============================================================================
# Per-frame combined loss
# =============================================================================


@dataclass
class PerFrameBoundaryLossRecord:
    L_dice: float
    L_bce: float
    L_anti_true: float
    L_total: float
    support_area: float              # sum of support_mask — diagnostic


def boundary_decoy_loss_per_frame(
    pred_logits: Tensor,         # [H, W]
    m_decoy: Tensor,             # [H, W]
    band_true: Tensor,           # [H, W]
    support_mask: Tensor,        # [H, W]  (band_true ∪ band_decoy ∪ corridor)
    *,
    alpha_dice: float = 1.0,
    beta_bce: float = 0.5,
    gamma_anti_true: float = 0.3,
) -> tuple[Tensor, PerFrameBoundaryLossRecord]:
    """Per-frame boundary-aware decoy loss.

    Weighted Dice + BCE on the combined support mask, plus an
    anti-true-boundary overlap term. Returns (scalar loss tensor,
    detached record).

    All loss terms are normalized to a comparable scale: BCE by
    support area, Dice in [0, 1], anti-true in [0, 1].
    """
    L_dice = boundary_weighted_dice(pred_logits, m_decoy, support_mask)
    L_bce = boundary_weighted_bce(pred_logits, m_decoy, support_mask)
    L_anti = anti_true_boundary(pred_logits, band_true)
    L_total = (
        alpha_dice * L_dice
        + beta_bce * L_bce
        + gamma_anti_true * L_anti
    )
    rec = PerFrameBoundaryLossRecord(
        L_dice=float(L_dice.detach().item()),
        L_bce=float(L_bce.detach().item()),
        L_anti_true=float(L_anti.detach().item()),
        L_total=float(L_total.detach().item()),
        support_area=float(support_mask.sum().detach().item()),
    )
    return L_total, rec


# =============================================================================
# Multi-frame aggregate
# =============================================================================


@dataclass
class AggregateBoundaryLossRecord:
    L_margin: float
    L_polish_mean: float
    n_frames: int
    per_frame: Dict[int, PerFrameBoundaryLossRecord]


def aggregate_boundary_loss(
    pred_logits_by_t: Dict[int, Tensor],
    m_decoy_by_t: Dict[int, Tensor],
    band_true_by_t: Dict[int, Tensor],
    support_by_t: Dict[int, Tensor],
    polish_frame_ids: Iterable[int],
    *,
    alpha_dice: float = 1.0,
    beta_bce: float = 0.5,
    gamma_anti_true: float = 0.3,
) -> tuple[Tensor, AggregateBoundaryLossRecord]:
    """Mean boundary-loss over the set of frames selected for δ polish.

    Typically these are "degraded" frames flagged by the decoy-semantic
    classifier after A0, plus all insert positions. All per-frame
    weights are 1.0 (no insert-vs-neighbor split like the standard
    vadi_v5_loss; at this stage all polish frames are treated equally).
    """
    polish_set = sorted(int(t) for t in polish_frame_ids)
    if not polish_set:
        # Edge: no polish frames → return zero scalar (no gradient).
        dev = next(iter(pred_logits_by_t.values())).device \
            if pred_logits_by_t else torch.device("cpu")
        dt = next(iter(pred_logits_by_t.values())).dtype \
            if pred_logits_by_t else torch.float32
        z = torch.zeros((), device=dev, dtype=dt)
        return z, AggregateBoundaryLossRecord(
            L_margin=0.0, L_polish_mean=0.0, n_frames=0, per_frame={})

    per_frame_losses: List[Tensor] = []
    per_frame_records: Dict[int, PerFrameBoundaryLossRecord] = {}
    for t in polish_set:
        if t not in pred_logits_by_t:
            raise KeyError(
                f"pred_logits_by_t missing frame t={t} in polish set")
        for d, name in [
            (m_decoy_by_t, "m_decoy_by_t"),
            (band_true_by_t, "band_true_by_t"),
            (support_by_t, "support_by_t"),
        ]:
            if t not in d:
                raise KeyError(f"{name} missing frame t={t}")
        L_t, rec = boundary_decoy_loss_per_frame(
            pred_logits_by_t[t], m_decoy_by_t[t],
            band_true_by_t[t], support_by_t[t],
            alpha_dice=alpha_dice, beta_bce=beta_bce,
            gamma_anti_true=gamma_anti_true,
        )
        per_frame_losses.append(L_t)
        per_frame_records[t] = rec

    L_margin = torch.stack(per_frame_losses).mean()
    record = AggregateBoundaryLossRecord(
        L_margin=float(L_margin.detach().item()),
        L_polish_mean=float(L_margin.detach().item()),
        n_frames=len(polish_set),
        per_frame=per_frame_records,
    )
    return L_margin, record


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    torch.manual_seed(0)
    H, W = 32, 32

    # Build canonical synthetic masks
    m_true = torch.zeros((H, W)); m_true[8:16, 8:16] = 1.0
    m_decoy = torch.zeros((H, W)); m_decoy[8:16, 20:28] = 1.0
    band_true = torch.zeros((H, W))
    band_true[6:18, 6:10] = 1.0                         # left edge of true
    band_true[6:18, 14:18] = 1.0                        # right edge of true
    band_decoy = torch.zeros((H, W))
    band_decoy[6:18, 18:22] = 1.0                       # left edge of decoy
    band_decoy[6:18, 26:30] = 1.0                       # right edge of decoy
    support = (band_true + band_decoy).clamp(0, 1)

    # --- boundary_weighted_dice: pred=+∞ on decoy, -∞ elsewhere (within support)
    pred_good = torch.where(m_decoy > 0.5, torch.full_like(m_decoy, +10.0),
                            torch.full_like(m_decoy, -10.0))
    L_dice = boundary_weighted_dice(pred_good, m_decoy, support)
    assert L_dice.item() < 0.1, f"good pred should give low dice: {L_dice.item()}"

    # Suppressed pred (all -∞) → high dice loss
    pred_sup = torch.full((H, W), -10.0)
    L_dice_sup = boundary_weighted_dice(pred_sup, m_decoy, support)
    assert L_dice_sup.item() > 0.9, \
        f"suppressed pred should give high dice: {L_dice_sup.item()}"

    # --- boundary_weighted_bce: ranges correct
    L_bce_good = boundary_weighted_bce(pred_good, m_decoy, support)
    L_bce_sup = boundary_weighted_bce(pred_sup, m_decoy, support)
    assert L_bce_good.item() < L_bce_sup.item() / 10, \
        "good pred must have much lower BCE than suppressed"
    # Empty support → zero (no NaN)
    empty_support = torch.zeros((H, W))
    L_empty = boundary_weighted_bce(pred_good, m_decoy, empty_support)
    assert L_empty.item() == 0.0

    # --- anti_true_boundary: pred on true → high L_anti
    pred_on_true = torch.where(m_true > 0.5, torch.full_like(m_true, +10.0),
                               torch.full_like(m_true, -10.0))
    L_anti_high = anti_true_boundary(pred_on_true, band_true)
    # Pred is +∞ on true mask interior; band_true surrounds the true interior
    # but doesn't overlap the interior. So σ(pred) ≈ 1 inside m_true and ≈ 0
    # outside. Overlap with band_true (which is outside m_true by construction)
    # should be low.
    # To properly test: put pred on band_true itself.
    pred_on_band = torch.where(band_true > 0.5, torch.full_like(band_true, +10.0),
                               torch.full_like(band_true, -10.0))
    L_anti_onband = anti_true_boundary(pred_on_band, band_true)
    assert L_anti_onband.item() > 0.9, \
        f"pred on band_true must have high L_anti: {L_anti_onband.item()}"
    # Pred far from band_true → low L_anti
    L_anti_low = anti_true_boundary(pred_good, band_true)
    assert L_anti_low.item() < L_anti_onband.item() / 5, \
        "pred on decoy (far from true band) must have much lower L_anti"

    # --- per-frame record
    L_t, rec = boundary_decoy_loss_per_frame(
        pred_good, m_decoy, band_true, support,
        alpha_dice=1.0, beta_bce=0.5, gamma_anti_true=0.3,
    )
    assert rec.L_dice < 0.1
    assert rec.L_total < 0.5
    assert rec.support_area > 0

    # --- gradient flow
    pred_g = torch.randn(H, W, requires_grad=True)
    L_g, _ = boundary_decoy_loss_per_frame(
        pred_g, m_decoy, band_true, support,
    )
    L_g.backward()
    assert pred_g.grad is not None and pred_g.grad.abs().sum().item() > 0.0
    # Gradient is zero OUTSIDE the support region (by construction: both
    # Dice and BCE are zero-weighted there, and anti_true uses only the
    # true-boundary band which is inside support).
    grad_outside_support = pred_g.grad * (1.0 - support)
    grad_outside_anti = grad_outside_support * (1.0 - band_true)
    # Outside the union support AND outside band_true, gradient should be
    # near zero.
    assert grad_outside_anti.abs().max().item() < 1e-4, \
        f"gradient must be ~zero outside support ∪ band_true: " \
        f"max = {grad_outside_anti.abs().max().item()}"

    # --- aggregate: 3 frames, polish set = {1, 2}
    pred_by_t = {0: pred_good, 1: pred_sup, 2: pred_good}
    decoy_by_t = {0: m_decoy, 1: m_decoy, 2: m_decoy}
    btrue_by_t = {0: band_true, 1: band_true, 2: band_true}
    supp_by_t = {0: support, 1: support, 2: support}
    L_agg, rec_agg = aggregate_boundary_loss(
        pred_by_t, decoy_by_t, btrue_by_t, supp_by_t,
        polish_frame_ids=[1, 2],
    )
    # mean of {sup, good}
    expected = 0.5 * (rec_agg.per_frame[1].L_total + rec_agg.per_frame[2].L_total)
    assert abs(rec_agg.L_margin - expected) < 1e-5
    assert rec_agg.n_frames == 2

    # Missing frame raises
    try:
        aggregate_boundary_loss(
            pred_by_t, decoy_by_t, btrue_by_t, supp_by_t,
            polish_frame_ids=[1, 99],
        )
        raise AssertionError("missing polish frame should raise")
    except KeyError:
        pass

    # Empty polish set → zero
    L_empty_agg, rec_empty = aggregate_boundary_loss(
        pred_by_t, decoy_by_t, btrue_by_t, supp_by_t,
        polish_frame_ids=[],
    )
    assert L_empty_agg.item() == 0.0
    assert rec_empty.n_frames == 0

    print("memshield.vadi_boundary_loss: all self-tests PASSED "
          "(boundary_weighted_dice, boundary_weighted_bce, anti_true_boundary, "
          "per-frame record, gradient masked to support ∪ band_true, "
          "aggregate over polish set, missing-frame KeyError, empty-polish-set "
          "zero)")


if __name__ == "__main__":
    _self_test()
