"""Direct decoy-tracking loss for VADI-v5 (DIRE).

Replaces the v4 contrastive decoy margin. Codex priority-3 fix
(2026-04-24 auto-review-loop Round 1): the v4 `softplus(μ_true - μ_decoy
+ margin)` on masked-logit means is too indirect and collapsed to
ground-truth suppression in practice (|Δμ_true| > |Δμ_decoy| in all 9
pilot runs). v5 uses a direct mask-level objective: make SAM2's
predicted mask at each insert + post-insert frame match the DECOY mask.

## Loss

For each queried frame t (insert or post-insert) with:
  - pred_logits[t]: `[H, W]` raw logits from SAM2
  - m_decoy[t]:      `[H, W]` float in {0,1}, decoy mask (translated
                     version of the clean-SAM2 pseudo-mask)
  - m_true[t]:       `[H, W]` float in {0,1}, clean-SAM2 pseudo-mask
                     (the ground-truth we want SAM2 to MISS)

Compute:

    p[t]           = σ(pred_logits[t])                             [H, W]

    L_dice_decoy[t]= 1 - (2·Σ p·m_decoy + ε) / (Σ p + Σ m_decoy + ε)
    L_bce_decoy[t] = BCE(pred_logits[t], m_decoy[t])               mean over H×W

    L_anti_true[t] = (Σ p·m_true) / (Σ m_true + ε)       ∈ [0, 1]
                     # soft fraction of the true object the prediction still covers

    L_t = α·L_dice_decoy[t] + β·L_bce_decoy[t] + γ·L_anti_true[t]

Default weights: α=1.0, β=0.5, γ=0.5.

Aggregation across frames:
    L_margin_v5 = mean_{t ∈ insert_ids ∪ post_insert_ids} L_t

The auxiliary v4 contrastive-margin term is OPTIONAL (weight defaults to
0.0 in v5 — disabled). We keep it callable so ablations can re-enable it.

## Why direct instead of contrastive

- **Grounded in the paper claim**: the paper says "SAM2 predicts the
  decoy mask after insert". The loss MUST drive `pred → m_decoy`.
- **No collapse to suppression**: BCE + Dice are both symmetric in the
  sense that reducing `p` everywhere to 0 does NOT minimize them —
  there's a positive "fill the decoy shape" signal.
- **Direct per-pixel supervision**: the v4 masked-means reduce each
  frame to two scalars before the loss kicks in, throwing away spatial
  structure. Dice/BCE are fully per-pixel.
- **Anti-true term handles the "suppression is cheap" failure mode**:
  γ penalizes predictions that still overlap the true mask region. When
  SAM2 correctly predicts the DECOY shape, this term is small; when it
  predicts a blob covering both true and decoy, it's large.

## Self-test

`python -m memshield.vadi_v5_loss` → synthetic-input assertions
covering: exact-match sanity, suppression-only penalty, joint-prediction
penalty, anti-true term correctness, gradient flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# Per-frame loss primitives
# =============================================================================


def dice_loss_on_decoy(
    pred_logits: Tensor,      # [H, W] or [B, H, W]
    m_decoy: Tensor,          # same shape as pred_logits
    eps: float = 1.0,
) -> Tensor:
    """Soft-Dice loss pushing sigmoid(pred) → m_decoy.

    L = 1 - 2·|p ∩ m| / (|p| + |m|), computed continuously.
    Scalar per batch element collapsed to 0-D.
    """
    if pred_logits.shape != m_decoy.shape:
        raise ValueError(
            f"pred_logits {tuple(pred_logits.shape)} vs m_decoy "
            f"{tuple(m_decoy.shape)}")
    p = torch.sigmoid(pred_logits)
    # Flatten all spatial dims; keep batch dim if present.
    if p.dim() == 2:
        p_flat = p.reshape(-1)
        m_flat = m_decoy.reshape(-1)
    else:
        p_flat = p.reshape(p.shape[0], -1)
        m_flat = m_decoy.reshape(m_decoy.shape[0], -1)
    inter = (p_flat * m_flat).sum(dim=-1)
    total = p_flat.sum(dim=-1) + m_flat.sum(dim=-1)
    dice = (2.0 * inter + eps) / (total + eps)
    return (1.0 - dice).mean()


def bce_on_decoy(
    pred_logits: Tensor, m_decoy: Tensor,
) -> Tensor:
    """Binary-cross-entropy with logits pushing pred → m_decoy.

    Numerically stable via `F.binary_cross_entropy_with_logits`.
    Mean-reduced.
    """
    return F.binary_cross_entropy_with_logits(
        pred_logits, m_decoy, reduction="mean",
    )


def anti_true_overlap(
    pred_logits: Tensor, m_true: Tensor, eps: float = 1.0,
) -> Tensor:
    """Soft fraction of the true mask covered by the prediction.

    L_anti = (Σ p·m_true) / (Σ m_true + eps) ∈ [0, 1].

    Zero when pred never overlaps m_true (desired: decoy prediction).
    One when pred covers all of m_true (undesired: still tracking true).
    """
    p = torch.sigmoid(pred_logits)
    if p.dim() == 2:
        p_flat = p.reshape(-1); t_flat = m_true.reshape(-1)
    else:
        p_flat = p.reshape(p.shape[0], -1)
        t_flat = m_true.reshape(m_true.shape[0], -1)
    overlap = (p_flat * t_flat).sum(dim=-1)
    total_true = t_flat.sum(dim=-1) + eps
    return (overlap / total_true).mean()


# =============================================================================
# Per-frame combined loss
# =============================================================================


@dataclass
class PerFrameDecoyLoss:
    """Per-frame loss decomposition (detached values for logging)."""

    L_dice: float
    L_bce: float
    L_anti_true: float
    L_total: float


def decoy_tracking_loss_per_frame(
    pred_logits: Tensor,     # [H, W]
    m_decoy: Tensor,         # [H, W] ∈ [0, 1]
    m_true: Tensor,          # [H, W] ∈ [0, 1]
    *,
    alpha_dice: float = 1.0,
    beta_bce: float = 0.5,
    gamma_anti_true: float = 0.5,
) -> tuple[Tensor, PerFrameDecoyLoss]:
    """Combined per-frame loss. Returns (loss_tensor, detached record)."""
    L_dice = dice_loss_on_decoy(pred_logits, m_decoy)
    L_bce = bce_on_decoy(pred_logits, m_decoy)
    L_anti = anti_true_overlap(pred_logits, m_true)
    L_total = (
        alpha_dice * L_dice
        + beta_bce * L_bce
        + gamma_anti_true * L_anti
    )
    record = PerFrameDecoyLoss(
        L_dice=float(L_dice.detach().item()),
        L_bce=float(L_bce.detach().item()),
        L_anti_true=float(L_anti.detach().item()),
        L_total=float(L_total.detach().item()),
    )
    return L_total, record


# =============================================================================
# Multi-frame aggregate
# =============================================================================


@dataclass
class AggregateDecoyLossRecord:
    """Record for the multi-frame aggregate, with per-frame breakdowns."""

    L_margin: float                       # the aggregate — driven loss
    L_insert: float                       # mean over insert positions
    L_post_insert: float                  # mean over post-insert positions
    per_frame: Dict[int, PerFrameDecoyLoss]


def aggregate_decoy_tracking_loss(
    pred_logits_by_t: Dict[int, Tensor],
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    insert_ids: Iterable[int],
    post_insert_ids: Iterable[int],
    *,
    alpha_dice: float = 1.0,
    beta_bce: float = 0.5,
    gamma_anti_true: float = 0.5,
    insert_weight: float = 1.0,
    post_insert_weight: float = 1.0,
) -> tuple[Tensor, AggregateDecoyLossRecord]:
    """Aggregate per-frame decoy-tracking losses over insert + post-insert.

    Both groups are weighted (insert_weight, post_insert_weight) in the
    mean. Default weights 1:1 — insert positions and post-insert positions
    contribute equally when averaged.

    Returns the scalar loss tensor (requires_grad if inputs did) + an
    AggregateDecoyLossRecord for logging.
    """
    insert_set = {int(t) for t in insert_ids}
    post_set = {int(t) for t in post_insert_ids}

    per_frame: Dict[int, PerFrameDecoyLoss] = {}
    insert_losses: List[Tensor] = []
    post_losses: List[Tensor] = []

    def _handle(t: int) -> Tensor:
        if t not in pred_logits_by_t:
            raise KeyError(f"pred_logits_by_t missing entry for t={t}")
        if t not in m_decoy_by_t or t not in m_true_by_t:
            raise KeyError(
                f"m_decoy_by_t / m_true_by_t missing entry for t={t}")
        L_t, rec = decoy_tracking_loss_per_frame(
            pred_logits_by_t[t], m_decoy_by_t[t], m_true_by_t[t],
            alpha_dice=alpha_dice, beta_bce=beta_bce,
            gamma_anti_true=gamma_anti_true,
        )
        per_frame[t] = rec
        return L_t

    for t in insert_set:
        insert_losses.append(_handle(t))
    for t in post_set - insert_set:     # don't double-count if overlap
        post_losses.append(_handle(t))

    dev = next(iter(pred_logits_by_t.values())).device
    dt = next(iter(pred_logits_by_t.values())).dtype
    z = torch.zeros((), device=dev, dtype=dt)

    L_insert = torch.stack(insert_losses).mean() if insert_losses else z
    L_post = torch.stack(post_losses).mean() if post_losses else z

    if insert_losses and post_losses:
        L_margin = (insert_weight * L_insert
                    + post_insert_weight * L_post) \
                   / (insert_weight + post_insert_weight)
    elif insert_losses:
        L_margin = L_insert
    elif post_losses:
        L_margin = L_post
    else:
        L_margin = z

    record = AggregateDecoyLossRecord(
        L_margin=float(L_margin.detach().item()),
        L_insert=float(L_insert.detach().item()),
        L_post_insert=float(L_post.detach().item()),
        per_frame=per_frame,
    )
    return L_margin, record


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    torch.manual_seed(0)
    H, W = 16, 16

    # -- Exact-match sanity: pred_logits = +∞ on m_decoy, −∞ outside → both
    # Dice and BCE near zero. L_anti_true = 0 because pred ∩ m_true = 0
    # (assume decoy and true are disjoint).
    m_true = torch.zeros(H, W); m_true[2:8, 2:8] = 1.0
    m_decoy = torch.zeros(H, W); m_decoy[8:14, 8:14] = 1.0
    pred_good = torch.where(m_decoy > 0.5, torch.full_like(m_decoy, +10.0),
                            torch.full_like(m_decoy, -10.0))
    L, rec = decoy_tracking_loss_per_frame(pred_good, m_decoy, m_true)
    assert rec.L_dice < 0.05, f"dice too high on exact match: {rec.L_dice}"
    assert rec.L_bce < 0.05, f"bce too high on exact match: {rec.L_bce}"
    assert rec.L_anti_true < 0.05, f"anti_true should be 0: {rec.L_anti_true}"

    # -- Suppression-only: pred_logits = -10 everywhere. Dice = 1 (no overlap
    # with decoy), BCE = strictly positive (strongly wrong on decoy), L_anti
    # = 0 (pred ∩ true = 0 too). Should NOT minimize the loss.
    pred_suppressed = torch.full((H, W), -10.0)
    _, rec_sup = decoy_tracking_loss_per_frame(
        pred_suppressed, m_decoy, m_true)
    assert rec_sup.L_dice > 0.9, \
        f"suppression should have high dice: {rec_sup.L_dice}"
    assert rec_sup.L_bce > rec.L_bce * 10, \
        "suppression must have much higher BCE than exact match"
    assert rec_sup.L_total > rec.L_total + 0.5, \
        "suppression must NOT reach exact-match-level loss"

    # -- Tracking-true failure: pred matches m_true, not m_decoy. Dice high
    # (no overlap), BCE high (wrong). L_anti = high (pred covers true).
    pred_still_tracking = torch.where(m_true > 0.5, torch.full_like(m_true, +10.0),
                                      torch.full_like(m_true, -10.0))
    _, rec_track = decoy_tracking_loss_per_frame(
        pred_still_tracking, m_decoy, m_true)
    assert rec_track.L_anti_true > 0.9, \
        f"still-tracking-true must have high L_anti: {rec_track.L_anti_true}"

    # -- Joint prediction (covers both true AND decoy): dice somewhat low
    # (good decoy coverage), BCE gets penalty on non-decoy regions that
    # still have high prob, L_anti is strictly positive.
    m_both = ((m_true + m_decoy).clamp(0, 1))
    pred_both = torch.where(m_both > 0.5, torch.full_like(m_both, +10.0),
                            torch.full_like(m_both, -10.0))
    _, rec_both = decoy_tracking_loss_per_frame(pred_both, m_decoy, m_true)
    # Must be BETWEEN exact-match and suppression/tracking-true.
    assert rec.L_total < rec_both.L_total < rec_sup.L_total, \
        f"joint prediction loss out of order: {rec.L_total} / " \
        f"{rec_both.L_total} / {rec_sup.L_total}"
    assert rec_both.L_anti_true > 0.9, \
        "joint prediction covers true → L_anti should be large"

    # -- Gradient flow: loss is differentiable w.r.t. pred_logits.
    pred_g = torch.randn(H, W, requires_grad=True)
    L_g, _ = decoy_tracking_loss_per_frame(pred_g, m_decoy, m_true)
    L_g.backward()
    assert pred_g.grad is not None and pred_g.grad.abs().sum().item() > 0.0, \
        "loss must have nonzero gradient w.r.t. pred_logits"

    # -- aggregate: insert and post-insert positions, proper weighting
    pred_by_t = {
        0: pred_good,
        1: pred_suppressed,
        2: torch.randn(H, W),
        3: torch.randn(H, W),
    }
    m_decoy_by_t = {t: m_decoy for t in [0, 1, 2, 3]}
    m_true_by_t = {t: m_true for t in [0, 1, 2, 3]}
    L_agg, rec_agg = aggregate_decoy_tracking_loss(
        pred_by_t, m_decoy_by_t, m_true_by_t,
        insert_ids=[0, 1], post_insert_ids=[2, 3],
    )
    # L_insert = mean([L_good, L_suppressed])
    expected_insert = 0.5 * (rec_agg.per_frame[0].L_total + rec_agg.per_frame[1].L_total)
    assert abs(rec_agg.L_insert - expected_insert) < 1e-5
    # Equal weights 1:1 → L_margin == mean(L_insert, L_post).
    expected_margin = 0.5 * (rec_agg.L_insert + rec_agg.L_post_insert)
    assert abs(rec_agg.L_margin - expected_margin) < 1e-5, \
        f"aggregate mismatch: {rec_agg.L_margin} vs {expected_margin}"

    # -- Missing frame in pred_logits_by_t raises
    try:
        aggregate_decoy_tracking_loss(
            {0: pred_good}, m_decoy_by_t, m_true_by_t,
            insert_ids=[0, 1], post_insert_ids=[],
        )
        raise AssertionError("missing pred key should raise")
    except KeyError:
        pass

    # -- Empty post_insert: L_margin == L_insert
    L_only_ins, rec_only_ins = aggregate_decoy_tracking_loss(
        pred_by_t, m_decoy_by_t, m_true_by_t,
        insert_ids=[0], post_insert_ids=[],
    )
    assert abs(rec_only_ins.L_margin - rec_only_ins.L_insert) < 1e-5
    assert rec_only_ins.L_post_insert == 0.0

    # -- Batched inputs [B, H, W] also work for per-frame primitives.
    pred_batched = torch.randn(4, H, W, requires_grad=True)
    m_decoy_b = m_decoy.unsqueeze(0).expand(4, -1, -1)
    m_true_b = m_true.unsqueeze(0).expand(4, -1, -1)
    L_b = dice_loss_on_decoy(pred_batched, m_decoy_b)
    assert L_b.dim() == 0

    print("memshield.vadi_v5_loss: all self-tests PASSED "
          "(exact-match, suppression penalty, tracking-true penalty, "
          "joint-prediction ordering, gradient flow, aggregate weighting, "
          "missing-key raise, empty-post-insert, batched primitives)")


if __name__ == "__main__":
    _self_test()
