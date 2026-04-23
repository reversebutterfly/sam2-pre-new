"""VADI loss primitives: contrastive decoy-margin + fidelity hinges.

Pure-functional building blocks the `vadi_optimize.py` PGD driver composes
into the full objective:

    L = L_margin_insert + L_margin_neighbor
      + λ(step) · (L_fid_orig + L_fid_ins + L_fid_TV)
      + λ_0 · L_fid_f0

Margin terms (attack signal, GT-free via clean-SAM2 pseudo-masks):
    c_t       = |2·m_hat_true_t − 1|                                    # confidence weight
    mu_true_t = Σ pred_logits_t · m_hat_true_t  · c_t  /  (Σ ... + eps)
    mu_decoy_t = Σ pred_logits_t · m_hat_decoy_t · c_t  /  (Σ ... + eps)
    L_margin_insert   = Σ_k                softplus(mu_true_{W_k} − mu_decoy_{W_k} + 0.75)
    L_margin_neighbor = Σ_{t ∈ NbrSet \\ W} 0.5 · softplus(mu_true_t − mu_decoy_t + 0.75)

Fidelity hinges (budget compliance during optimization — exported-artifact
feasibility is re-checked afterward by `vadi_optimize.py`):
    L_fid_orig = Σ_{t ∈ S_δ, t≥1} max(0, LPIPS(x'_t, x_t) − 0.20)
    L_fid_ins  = Σ_k               max(0, LPIPS(insert_k, base_insert_k) − 0.35)
    L_fid_TV   = Σ_k               max(0, TV(insert_k) − 1.2 · TV(base_insert_k))
    L_fid_f0   =                    max(0, 0.98 − SSIM(x'_0, x_0))

LPIPS and SSIM are passed in as callables from the driver (matches the
`build_lpips_fn(device)` pattern in `memshield/run_pilot_r002.py`); this
module does not import them so it stays testable without those
heavyweights installed.

Design note: neighbor vs insert disjointness is enforced by
`aggregate_margin_loss` (NbrSet \\ W must be disjoint from W).

Run `python -m memshield.vadi_loss` for self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Confidence-weighted masked mean
# -----------------------------------------------------------------------------


def confidence_weight(m_hat: Tensor) -> Tensor:
    """c_t = |2·m_hat_t − 1| ∈ [0, 1]. Zero at decision boundary (m_hat=0.5),
    max at hard 0/1."""
    return (2.0 * m_hat - 1.0).abs()


def confidence_weighted_logit_mean(
    pred_logits: Tensor,
    mask: Tensor,
    c: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """mu = Σ (pred_logits · mask · c) / (Σ (mask · c) + eps).

    Inputs must have matching shape (enforced; no implicit broadcast);
    typically [H, W] or [B, H, W]. Reduction is over the last two dims —
    returns a 0-d scalar if inputs are 2-D, or [B] if 3-D.

    Numerics: use fp32 or bf16. fp16 can flush `eps=1e-5` to zero and
    produce 0/0 NaN when mask · c reduces to zero. The upstream adapter
    uses bf16 autocast, which is safe.
    """
    if pred_logits.shape != mask.shape or pred_logits.shape != c.shape:
        raise ValueError(
            f"shape mismatch: pred_logits={tuple(pred_logits.shape)}, "
            f"mask={tuple(mask.shape)}, c={tuple(c.shape)}")
    w = mask * c
    num = (pred_logits * w).sum(dim=(-1, -2))
    den = w.sum(dim=(-1, -2)) + eps
    return num / den


# -----------------------------------------------------------------------------
# Per-frame decoy margin
# -----------------------------------------------------------------------------


@dataclass
class FrameMarginOutput:
    """Per-frame contrastive-margin record (all tensors have grad where
    `pred_logits` did; pseudo-masks are treated as constants)."""

    margin_loss: Tensor     # scalar: softplus(mu_true - mu_decoy + margin)
    mu_true: Tensor         # scalar: logit mean under m_hat_true
    mu_decoy: Tensor        # scalar: logit mean under m_hat_decoy


def decoy_margin_per_frame(
    pred_logits: Tensor,
    m_hat_true: Tensor,
    m_hat_decoy: Tensor,
    margin: float = 0.75,
    eps: float = 1e-5,
) -> FrameMarginOutput:
    """Compute the contrastive decoy-margin for one frame.

    Uses the confidence weight `c = |2·m_hat_true − 1|` built from the
    TRUE pseudo-mask (both sides of the contrast re-use the same c, so
    the weighting remains symmetric and does not itself bias the
    gradient toward either mu).

    Returns scalar `margin_loss`, `mu_true`, `mu_decoy` (the latter two
    are detached-friendly for logging trace plots).
    """
    c = confidence_weight(m_hat_true)
    mu_true = confidence_weighted_logit_mean(pred_logits, m_hat_true, c, eps)
    mu_decoy = confidence_weighted_logit_mean(pred_logits, m_hat_decoy, c, eps)
    margin_loss = F.softplus(mu_true - mu_decoy + margin)
    return FrameMarginOutput(
        margin_loss=margin_loss, mu_true=mu_true, mu_decoy=mu_decoy,
    )


# -----------------------------------------------------------------------------
# Aggregate margin loss over insert + neighbor sets
# -----------------------------------------------------------------------------


@dataclass
class AggregatedMarginLoss:
    """Decomposition of the aggregate margin term.

    `L_margin = L_insert + L_neighbor` is the value fed to PGD. Each
    component is also logged per-frame for diagnostic mu-trace plots.
    """

    L_insert: Tensor       # scalar: Σ_k softplus(...) over insert frames
    L_neighbor: Tensor     # scalar: 0.5 · Σ_{t ∈ neighbor} softplus(...)
    L_margin: Tensor       # L_insert + L_neighbor
    insert_ids: List[int]
    neighbor_ids: List[int]


def aggregate_margin_loss(
    margins_by_t: Dict[int, FrameMarginOutput],
    insert_ids: Iterable[int],
    neighbor_ids: Iterable[int],
    neighbor_weight: float = 0.5,
) -> AggregatedMarginLoss:
    """Combine per-frame `FrameMarginOutput`s into the aggregate margin loss.

    Args:
        margins_by_t: {frame_idx → FrameMarginOutput}. Must contain every
                      id in insert_ids and neighbor_ids.
        insert_ids:   the W positions (full weight). Deduplicated here —
                      driver passing the same k twice is accepted silently.
        neighbor_ids: NbrSet \\ W (must be disjoint from insert_ids after
                      dedup). Driver typically builds this by UNIONING
                      `NbrSet(W_k)` across k and subtracting W.
        neighbor_weight: proposal uses 0.5.

    Enforces NbrSet \\ W disjointness per the VADI spec (frames in W
    count at full weight; their neighbors count at 0.5x — double-counting
    would inflate the loss and distort the PGD signal).
    """
    # Dedup while preserving ascending ordering for deterministic logs.
    insert_ids = sorted(set(int(i) for i in insert_ids))
    neighbor_ids = sorted(set(int(i) for i in neighbor_ids))
    overlap = set(insert_ids) & set(neighbor_ids)
    if overlap:
        raise ValueError(
            f"neighbor_ids must be NbrSet \\ insert_ids (disjoint); "
            f"got overlap {sorted(overlap)}")
    for t in insert_ids + neighbor_ids:
        if t not in margins_by_t:
            raise KeyError(
                f"margins_by_t missing frame {t}; have keys "
                f"{sorted(margins_by_t.keys())}")

    # Use a sample tensor to get dtype/device for the zero scalar.
    sample = next(iter(margins_by_t.values())).margin_loss
    zero = torch.zeros((), dtype=sample.dtype, device=sample.device)

    L_insert = zero.clone()
    for t in insert_ids:
        L_insert = L_insert + margins_by_t[t].margin_loss

    L_neighbor = zero.clone()
    for t in neighbor_ids:
        L_neighbor = L_neighbor + margins_by_t[t].margin_loss
    L_neighbor = neighbor_weight * L_neighbor

    return AggregatedMarginLoss(
        L_insert=L_insert, L_neighbor=L_neighbor,
        L_margin=L_insert + L_neighbor,
        insert_ids=insert_ids, neighbor_ids=neighbor_ids,
    )


# -----------------------------------------------------------------------------
# Fidelity: total variation + hinges
# -----------------------------------------------------------------------------


def total_variation(x: Tensor) -> Tensor:
    """Anisotropic L1 total variation.

    - 3-D `[C, H, W]` → scalar (sum over C, H, W).
    - 4-D `[N, C, H, W]` → `[N]` per-sample (sum over C, H, W only). This
      keeps each insert_k's TV separate so the driver can apply the
      per-k hinge `Σ_k max(0, TV_k − 1.2·TV_base_k)` without
      accidentally cross-contaminating across k.

    Proposal uses sum-TV (not mean) because the budget is relative
    (1.2 × TV(base_insert)).
    """
    if x.dim() == 3:
        tv_h = (x[:, 1:, :] - x[:, :-1, :]).abs().sum()
        tv_w = (x[:, :, 1:] - x[:, :, :-1]).abs().sum()
        return tv_h + tv_w
    if x.dim() == 4:
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))
        return tv_h + tv_w
    raise ValueError(
        f"total_variation expects 3D [C,H,W] or 4D [N,C,H,W]; got {x.dim()}D")


def tv_hinge(
    insert: Tensor,
    base_insert: Tensor,
    multiplier: float = 1.2,
) -> Tensor:
    """max(0, TV(insert) − multiplier · TV(base_insert)).

    Returns a scalar for 3-D inputs, or `[N]` for 4-D inputs (caller
    sums over k if building `L_fid_TV = Σ_k max(0, TV_k − 1.2·TV_base_k)`).
    """
    return F.relu(total_variation(insert) - multiplier * total_variation(base_insert))


def lpips_cap_hinge(lpips_value: Tensor, cap: float) -> Tensor:
    """max(0, LPIPS(x', x) − cap). Caller precomputes LPIPS."""
    return F.relu(lpips_value - cap)


def ssim_floor_hinge(ssim_value: Tensor, floor: float) -> Tensor:
    """max(0, floor − SSIM(x', x)). Caller precomputes SSIM.

    For the VADI f0 constraint (SSIM ≥ 0.98), pass `floor=0.98`. Equivalent
    to the proposal's `max(0, 1 − SSIM − 0.02)` form.
    """
    return F.relu(floor - ssim_value)


# -----------------------------------------------------------------------------
# Logging aggregate
# -----------------------------------------------------------------------------


@dataclass
class MarginTrace:
    """Per-frame mu-trace for diagnostic plots. Values are detached.

    Used by `vadi_optimize.py` to populate the mu_true_trace / mu_decoy_trace
    log series that the paper's signed decoy-vs-suppression claim depends on
    (Δmu_decoy > 0 AND Δmu_decoy ≥ 2 · max(0, −Δmu_true)).
    """

    mu_true: Dict[int, float] = field(default_factory=dict)
    mu_decoy: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_margins_by_t(
        cls, margins_by_t: Dict[int, FrameMarginOutput],
    ) -> "MarginTrace":
        return cls(
            mu_true={t: float(m.mu_true.detach().item())
                     for t, m in margins_by_t.items()},
            mu_decoy={t: float(m.mu_decoy.detach().item())
                      for t, m in margins_by_t.items()},
        )


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def _self_test() -> None:
    torch.manual_seed(0)

    # --- confidence_weight: hard mask → c=1 everywhere; exactly-0.5 → 0
    m = torch.tensor([[0.0, 0.5, 1.0]])
    c = confidence_weight(m)
    assert torch.allclose(c, torch.tensor([[1.0, 0.0, 1.0]]))

    # --- confidence_weighted_logit_mean: uniform mask + uniform c → plain mean
    H, W = 4, 4
    logits = torch.arange(H * W, dtype=torch.float32).reshape(H, W)
    mask = torch.ones(H, W)
    c = torch.ones(H, W)
    mu = confidence_weighted_logit_mean(logits, mask, c)
    # eps=1e-5 introduces ~4e-6 relative bias at |logits|~10 — well below any
    # threshold that matters for PGD.
    assert abs(mu.item() - logits.mean().item()) < 1e-4
    # Zero mask → 0 / eps = ~0
    mu_zero = confidence_weighted_logit_mean(logits, torch.zeros(H, W), c)
    assert abs(mu_zero.item()) < 1e-4
    # Shape mismatch raises.
    try:
        confidence_weighted_logit_mean(logits, torch.ones(H + 1, W), c)
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass

    # --- decoy_margin_per_frame: attacker wants mu_true − mu_decoy NEGATIVE
    # (loss pushes mu_decoy above mu_true). Check that softplus(x) decreases
    # monotonically as mu_decoy grows (with mu_true held fixed).
    logits_attack = torch.full((H, W), -2.0)    # favors decoy
    logits_defend = torch.full((H, W), +2.0)    # favors true
    m_true = torch.zeros(H, W); m_true[:2, :2] = 1.0
    m_decoy = torch.zeros(H, W); m_decoy[2:, 2:] = 1.0

    out_att = decoy_margin_per_frame(logits_attack, m_true, m_decoy, margin=0.75)
    out_def = decoy_margin_per_frame(logits_defend, m_true, m_decoy, margin=0.75)
    # Attack: mu_true = -2, mu_decoy = -2 → softplus(0 + 0.75) = ~1.12
    # Wait, logits are uniform, so both means are -2; margin loss is the same
    # regardless of mask. Use different-value regions instead.
    logits_split = torch.full((H, W), 0.0)
    logits_split[:2, :2] = -3.0                 # true region: low logits
    logits_split[2:, 2:] = +3.0                 # decoy region: high logits
    out_split = decoy_margin_per_frame(
        logits_split, m_true, m_decoy, margin=0.75)
    # mu_true ≈ -3, mu_decoy ≈ +3. Input to softplus = -3 - 3 + 0.75 = -5.25.
    # softplus(-5.25) ≈ 0.00525.
    assert abs(out_split.mu_true.item() - (-3.0)) < 1e-4
    assert abs(out_split.mu_decoy.item() - (+3.0)) < 1e-4
    assert out_split.margin_loss.item() < 0.01
    # Swap: defender layout. mu_true ≈ +3, mu_decoy ≈ -3. Input = 3 + 3 + 0.75 = 6.75.
    # softplus(6.75) ≈ 6.75.
    out_def_split = decoy_margin_per_frame(
        -logits_split, m_true, m_decoy, margin=0.75)
    assert 6.0 < out_def_split.margin_loss.item() < 7.0

    # Gradient flows through pred_logits → margin_loss.
    logits_g = torch.zeros(H, W, requires_grad=True)
    out_g = decoy_margin_per_frame(logits_g, m_true, m_decoy, margin=0.75)
    out_g.margin_loss.backward()
    assert logits_g.grad is not None
    assert logits_g.grad.abs().sum() > 0

    # --- aggregate_margin_loss: insert full, neighbor × 0.5, disjointness
    margins_by_t = {
        2: decoy_margin_per_frame(logits_split, m_true, m_decoy),
        3: decoy_margin_per_frame(-logits_split, m_true, m_decoy),
        4: decoy_margin_per_frame(-logits_split, m_true, m_decoy),
        5: decoy_margin_per_frame(logits_split, m_true, m_decoy),
    }
    agg = aggregate_margin_loss(
        margins_by_t,
        insert_ids=[3],
        neighbor_ids=[2, 4, 5],
        neighbor_weight=0.5,
    )
    expected_insert = margins_by_t[3].margin_loss.item()
    expected_neighbor = 0.5 * sum(
        margins_by_t[t].margin_loss.item() for t in (2, 4, 5))
    assert abs(agg.L_insert.item() - expected_insert) < 1e-6
    assert abs(agg.L_neighbor.item() - expected_neighbor) < 1e-6
    assert abs(agg.L_margin.item()
               - (expected_insert + expected_neighbor)) < 1e-6

    # Disjointness violation raises.
    try:
        aggregate_margin_loss(margins_by_t, insert_ids=[3], neighbor_ids=[3, 4])
        raise AssertionError("overlap must raise")
    except ValueError:
        pass
    # Missing margin raises.
    try:
        aggregate_margin_loss(margins_by_t, insert_ids=[99], neighbor_ids=[])
        raise AssertionError("missing frame must raise")
    except KeyError:
        pass
    # Dedup: passing the same k twice in insert_ids / same t twice in
    # neighbor_ids should NOT double-count.
    agg_dup = aggregate_margin_loss(
        margins_by_t,
        insert_ids=[3, 3],
        neighbor_ids=[2, 2, 4, 4, 5, 5],
        neighbor_weight=0.5,
    )
    assert abs(agg_dup.L_insert.item() - agg.L_insert.item()) < 1e-6, \
        "aggregate_margin_loss must dedup insert_ids"
    assert abs(agg_dup.L_neighbor.item() - agg.L_neighbor.item()) < 1e-6, \
        "aggregate_margin_loss must dedup neighbor_ids"
    assert agg_dup.insert_ids == [3]
    assert agg_dup.neighbor_ids == [2, 4, 5]

    # --- total_variation
    # 3-D scalar.
    uniform = torch.zeros(3, 8, 8)
    assert total_variation(uniform).dim() == 0
    assert total_variation(uniform).item() == 0.0
    checker = torch.zeros(1, 4, 4)
    checker[0, ::2, ::2] = 1.0
    checker[0, 1::2, 1::2] = 1.0
    tv_checker = total_variation(checker)
    assert tv_checker.item() > 0
    # 4-D per-batch semantics: stack two different images; each insert
    # must produce its own TV, not a summed-over-batch value.
    uniform_44 = torch.zeros(3, 4, 4)
    stacked = torch.stack([uniform_44, checker.expand(3, 4, 4)], dim=0)
    tv_stacked = total_variation(stacked)
    assert tv_stacked.shape == (2,)
    assert tv_stacked[0].item() == 0.0                    # uniform sample
    assert tv_stacked[1].item() > 0.0                     # checker sample
    # Per-batch TV must equal solo TV for each sample.
    assert abs(tv_stacked[0].item() - total_variation(uniform_44).item()) < 1e-6
    assert abs(tv_stacked[1].item()
               - total_variation(checker.expand(3, 4, 4)).item()) < 1e-6
    # 2-D or 5-D raises.
    try:
        total_variation(torch.zeros(4, 4))
        raise AssertionError("2-D must raise")
    except ValueError:
        pass
    try:
        total_variation(torch.zeros(2, 3, 4, 4, 4))
        raise AssertionError("5-D must raise")
    except ValueError:
        pass

    # --- tv_hinge: below multiplier → 0; above → excess
    base = torch.ones(1, 4, 4) * 0.5
    insert_small = base.clone()
    assert tv_hinge(insert_small, base, multiplier=1.2).item() == 0.0
    insert_big = torch.zeros(1, 4, 4)
    insert_big[0, ::2, :] = 1.0                      # high-frequency content
    hinge_big = tv_hinge(insert_big, base, multiplier=1.2)
    assert hinge_big.item() > 0
    # 4-D per-k hinge: should be [N], each element independent.
    insert_batch = torch.stack([insert_small, insert_big], dim=0)
    base_batch = torch.stack([base, base], dim=0)
    tvh_batch = tv_hinge(insert_batch, base_batch, multiplier=1.2)
    assert tvh_batch.shape == (2,)
    assert tvh_batch[0].item() == 0.0
    assert tvh_batch[1].item() > 0.0
    # TV hinge gradient flows — 3-D path.
    insert_big_g = insert_big.clone().requires_grad_(True)
    tv_hinge(insert_big_g, base, multiplier=1.2).backward()
    assert insert_big_g.grad is not None
    assert insert_big_g.grad.abs().sum().item() > 0
    # TV hinge gradient flows — 4-D path (per-k stacked).
    insert_batch_g = insert_batch.clone().requires_grad_(True)
    tv_hinge(insert_batch_g, base_batch, multiplier=1.2).sum().backward()
    assert insert_batch_g.grad is not None
    # Element 0 was in-budget → no grad; element 1 violates → nonzero grad.
    assert insert_batch_g.grad[0].abs().sum().item() == 0.0
    assert insert_batch_g.grad[1].abs().sum().item() > 0

    # --- lpips_cap_hinge / ssim_floor_hinge
    assert lpips_cap_hinge(torch.tensor(0.15), cap=0.20).item() == 0.0
    assert abs(lpips_cap_hinge(torch.tensor(0.30), cap=0.20).item() - 0.10) < 1e-6
    assert ssim_floor_hinge(torch.tensor(0.99), floor=0.98).item() == 0.0
    assert abs(ssim_floor_hinge(torch.tensor(0.96), floor=0.98).item()
               - 0.02) < 1e-6

    # Fidelity hinges retain grad (required for PGD backward).
    lp = torch.tensor(0.30, requires_grad=True)
    lpips_cap_hinge(lp, cap=0.20).backward()
    assert lp.grad is not None and lp.grad.item() > 0
    # SSIM floor hinge grad: dL/dSSIM = -1 below floor, 0 above.
    ss = torch.tensor(0.96, requires_grad=True)
    ssim_floor_hinge(ss, floor=0.98).backward()
    assert ss.grad is not None and ss.grad.item() < 0

    # --- Composite backward (PGD-like): margin + λ·LPIPS + λ_0·SSIM all
    # depending on a single learnable tensor; verifies one combined
    # .backward() delivers grad through every active term.
    torch.manual_seed(1)
    delta = (torch.randn(H, W) * 0.3).clone().requires_grad_(True)  # leaf
    # An independent insert tensor driving the TV term, also a leaf.
    insert_leaf = (base.clone() + 0.1 * torch.randn_like(base)).clone() \
        .requires_grad_(True)
    # Margin term: pred_logits = delta (spatially varying → mu_true ≠ mu_decoy).
    margin_out = decoy_margin_per_frame(delta, m_true, m_decoy)
    # LPIPS proxy: L2 norm of delta, above a tiny cap → hinge active.
    lpips_proxy = delta.pow(2).mean()
    # SSIM proxy: 1 - L2, below a floor → hinge active.
    ssim_proxy = 1.0 - delta.pow(2).mean()
    # TV: force insert above the base's 1.2× budget by construction.
    L = (margin_out.margin_loss
         + 0.1 * lpips_cap_hinge(lpips_proxy, cap=0.001)             # active
         + 10.0 * ssim_floor_hinge(ssim_proxy, floor=0.999)          # active
         + 1.0 * tv_hinge(insert_leaf, base, multiplier=0.5))        # active
    L.backward()
    assert delta.grad is not None
    assert delta.grad.abs().sum().item() > 0, \
        "composite backward must deliver nonzero grad through δ"
    assert insert_leaf.grad is not None
    assert insert_leaf.grad.abs().sum().item() > 0, \
        "composite backward must deliver nonzero grad through TV"

    # --- MarginTrace: detached floats, correct keys
    trace = MarginTrace.from_margins_by_t(margins_by_t)
    assert set(trace.mu_true) == {2, 3, 4, 5}
    assert set(trace.mu_decoy) == {2, 3, 4, 5}
    assert all(isinstance(v, float) for v in trace.mu_true.values())

    print("memshield.vadi_loss: all self-tests PASSED "
          "(confidence weight, masked mean, decoy margin, aggregate, "
          "TV, LPIPS/SSIM hinges, MarginTrace)")


if __name__ == "__main__":
    _self_test()
