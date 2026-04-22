"""
MemoryShield Chunk 4: loss stack (v2).

Implements the losses from FINAL_PROPOSAL.md §Proposed Method:

  * masked CVaR-over-set   (no 1[.] zero-contamination)
  * logmeanexp             (resolution-invariant confidence lock)
  * ROI-BCE                (no background mass outside the ROI)
  * L_loss  = Phase-1 per-insert composite loss
  * L_rec   = Phase-2 suppression + low-confidence + L_stale
  * L_stale = 3-bin KL(Q || P_u)   with `L_stale_margin` as fallback
  * L_fid   = augmented-Lagrangian insert-LPIPS penalty + seam ΔE
  * Lagrangian multiplier update rule.

All functions are pure torch (plus a tiny numpy helper for DeltaE colour
conversion); every function is differentiable end-to-end so PGD can
backprop into (δ, ν) pixels. Shapes are documented per function.

Math anchors (verbatim from FINAL_PROPOSAL §Loss):

  L_loss = (1/K_ins) · sum_k [ BCE_ROI(g_{ins_k}, 1[D_{ins_k}]) +
                α · softplus(CVaR_0.5({g_{ins_k}(p) : p ∈ C_{ins_k}}) + m) ]

  L_rec  = (1/|U|) · sum_{u∈U} [ α_supp · CVaR_0.5({g_u(p):p∈C_u})^+
                + α_conf · softplus(logmeanexp(g_u) − τ_conf) ]
           + β · L_stale

  L_stale = (1/|V|) · sum_{u∈V} KL(Q || P_u)   where  P_u = [A_ins, A_recent, A_other]
          ≈  L_stale_margin = softplus(γ + A_recent − A_ins) + λ · A_other   (fallback)

  L_fid  = μ_ν · (LPIPS(x_ins, f_prev) − 0.10)^+  +  μ_s · ΔE_seam

Notation: `g_u` is a logit map (pre-sigmoid) of shape [H, W]; `C_u` is a
binary foreground mask [H, W]; `P_u` is a [3]-tensor sum-to-1; `Q` is a
[3]-tensor sum-to-1. The dominant caller is `optimize_unified_v2`
(Chunk 5), which assembles these into per-step gradient updates.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def masked_cvar_over_set(
    values: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    min_count: int = 8,
) -> torch.Tensor:
    """CVaR_α over the SET `{v(p) : p ∈ mask}` (no zero-padding contamination).

    For α = 0.5, this is the mean of the top 50% of values *inside the
    mask*, disregarding all pixels outside. This differs from the legacy
    "indicator-CVaR" formulation `CVaR(values * mask)` which contaminates
    the set with zeros for mask==0 pixels, biasing the statistic.

    Args:
        values: [H, W] or [B, H, W] logit / score tensor. Must be float.
        mask:   [H, W] or [B, H, W] binary mask (0 or 1; can be float
                but is thresholded at > 0.5). Same spatial shape as
                values.
        alpha:  Fraction of the top tail to average (default 0.5).
                0 < alpha <= 1.
        min_count: If the mask has fewer than this many active pixels,
                returns 0 with a graph connection preserved (via
                `0 * values.sum()`) so the scaler stays differentiable
                without blowing up.

    Returns:
        Scalar tensor (per-batch mean if batched).

    Differentiability: uses torch.topk which propagates gradients through
    exactly the selected indices. No sort-based branches.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if values.shape != mask.shape:
        raise ValueError(
            f"values {tuple(values.shape)} and mask {tuple(mask.shape)} "
            "must have identical shape"
        )

    # Batched path: recurse per batch element for clarity and identical
    # semantics regardless of per-sample active-pixel count.
    if values.ndim == 3:
        out = values.new_zeros(values.shape[0])
        for b in range(values.shape[0]):
            out[b] = masked_cvar_over_set(values[b], mask[b], alpha, min_count)
        return out

    mask_bool = mask > 0.5
    n = int(mask_bool.sum().item())
    if n < min_count:
        # Keep graph connected: multiplying by 0 preserves autograd wiring.
        return (values.sum() * 0.0)

    active = values[mask_bool]                   # 1-D, length n
    k = max(1, int(math.ceil(alpha * n)))
    top_k, _ = torch.topk(active, k=k, largest=True, sorted=False)
    return top_k.mean()


def logmeanexp_2d(logits: torch.Tensor) -> torch.Tensor:
    """Resolution-invariant `log(mean(exp(logits)))` on a [H, W] map.

    Equals `logsumexp(logits.flatten()) − log(H*W)`. The key property is
    that it does NOT scale with H*W (unlike logsumexp), so the threshold
    τ_conf can be set once and reused across resolutions.

    Args:
        logits: [H, W] or [B, H, W] logit tensor.

    Returns:
        Scalar tensor (per-batch if batched).
    """
    if logits.ndim == 2:
        HW = logits.numel()
        return torch.logsumexp(logits.reshape(-1), dim=0) - math.log(HW)
    if logits.ndim == 3:
        B, H, W = logits.shape
        return torch.logsumexp(logits.reshape(B, -1), dim=1) - math.log(H * W)
    raise ValueError(f"expected 2D or 3D logits, got shape {tuple(logits.shape)}")


def roi_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary-cross-entropy restricted to ROI.

    Pixels outside `roi_mask` contribute 0 to the loss (no gradient). The
    final value is normalized by the ROI pixel count so it does not scale
    with image resolution.

    Args:
        logits:   [H, W] logit map (pre-sigmoid).
        target:   [H, W] float {0, 1} target.
        roi_mask: [H, W] float {0, 1} ROI mask.
        eps: minimum ROI-pixel count guard.

    Returns:
        Scalar tensor.
    """
    if logits.shape != target.shape or logits.shape != roi_mask.shape:
        raise ValueError("logits, target, roi_mask must share shape")

    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    masked = per_pixel * roi_mask
    denom = roi_mask.sum().clamp_min(eps)
    return masked.sum() / denom


# ---------------------------------------------------------------------------
# Phase 1 — L_loss (inserts)
# ---------------------------------------------------------------------------


def l_loss_insert(
    insert_logits: Sequence[torch.Tensor],
    decoy_mask: Sequence[torch.Tensor],
    true_mask: Sequence[torch.Tensor],
    roi_mask: Sequence[torch.Tensor],
    alpha: float = 1.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Phase-1 composite loss over the K_ins inserts.

    L_loss = (1/K) · Σ_k [ BCE_ROI(g_k, 1[D_k])
                          + α · softplus( CVaR_0.5({g_k(p) : p∈C_k}) + m ) ]

    The BCE term pulls the logit toward 1 inside the DECOY region and
    toward 0 inside the rest of the ROI — creating a memory entry that
    points at the decoy location.

    The CVaR term suppresses top-half logits inside the TRUE object
    region, pushing them below −m. `softplus(... + m)` keeps the gradient
    alive even when the CVaR already dropped below 0.

    Args:
        insert_logits: length-K list of [H, W] logit tensors (one per insert).
        decoy_mask:    length-K list of [H, W] binary masks marking the
                       decoy paste region in each insert.
        true_mask:     length-K list of [H, W] binary masks marking the
                       true target position in each insert.
        roi_mask:      length-K list of [H, W] binary masks of ROI
                       (decoy_mask ∪ true_mask) dilated 10 px.
        alpha: weight on the CVaR suppress term (default 1.0).
        margin: additive margin inside softplus (default 0.0).

    Returns:
        Scalar tensor.
    """
    K = len(insert_logits)
    if K == 0:
        raise ValueError("l_loss_insert got zero inserts")
    if not (len(decoy_mask) == K and len(true_mask) == K and len(roi_mask) == K):
        raise ValueError("all per-insert lists must have length K_ins")

    total = insert_logits[0].new_zeros(())
    for g, Dk, Ck, roi in zip(insert_logits, decoy_mask, true_mask, roi_mask):
        bce = roi_bce(g, Dk.to(g.dtype), roi.to(g.dtype))
        cvar = masked_cvar_over_set(g, Ck, alpha=0.5)
        total = total + bce + alpha * F.softplus(cvar + margin)
    return total / K


# ---------------------------------------------------------------------------
# Phase 2 — L_rec (suppression + low-conf + L_stale)
# ---------------------------------------------------------------------------


def _positive_part(x: torch.Tensor) -> torch.Tensor:
    """Elementwise `max(x, 0)` with gradients (== relu)."""
    return F.relu(x)


def l_stale_kl(
    P_u_list: Sequence[Optional[torch.Tensor]],
    Q: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """3-bin categorical KL(Q || P_u), averaged over valid frames.

    P_u is a length-3 distribution `[A_ins, A_recent, A_other]` coming from
    `memshield.mem_attn_probe.MemAttnProbe`. Entries of `P_u_list` that
    are None (e.g. frames where the probe did not fire) are skipped. If
    no valid entries exist the function returns 0.

    Args:
        P_u_list: list of tensors of shape [3] (or None).
        Q: tensor of shape [3], sum to 1.
        eps: clamp minimum for log stability.

    Returns:
        Scalar tensor (zero-of-Q.dtype if no valid entries).
    """
    if Q.numel() != 3:
        raise ValueError(f"Q must be length-3, got {tuple(Q.shape)}")
    valid = [P for P in P_u_list if P is not None]
    if len(valid) == 0:
        return Q.new_zeros(())

    Q_safe = Q.clamp_min(eps)
    kls = []
    for P in valid:
        if P.numel() != 3:
            raise ValueError(f"P must be length-3, got {tuple(P.shape)}")
        # Renormalize P for numerical safety (caller should already do so).
        P_safe = P.clamp_min(eps)
        P_safe = P_safe / P_safe.sum()
        # KL(Q || P) = sum_i Q_i · (log Q_i − log P_i); only Q_i > 0 bins count.
        q_nz = (Q > 0)
        kl = (Q[q_nz] * (Q_safe[q_nz].log() - P_safe[q_nz].log())).sum()
        kls.append(kl)
    return torch.stack(kls).mean()


def l_stale_margin(
    P_u_list: Sequence[Optional[torch.Tensor]],
    gamma: float = 0.4,
    lambda_other: float = 0.2,
) -> torch.Tensor:
    """Fallback margin form — triggered by F4 in failure-mode plan.

    L_margin = (1/|V|) · Σ_u [ softplus(γ + A_u^recent − A_u^ins)
                              + λ · A_u^other ]

    Args:
        P_u_list: list of tensors [A_ins, A_recent, A_other]; Nones skipped.
        gamma: separation margin (A_ins should exceed A_recent by at least γ).
        lambda_other: weight on A_other (discourages "other" collapse).

    Returns:
        Scalar tensor.
    """
    valid = [P for P in P_u_list if P is not None]
    if len(valid) == 0:
        device = next((p.device for p in P_u_list if p is not None),
                      torch.device("cpu"))
        return torch.zeros((), device=device)

    vals = []
    for P in valid:
        a_ins, a_rec, a_oth = P[0], P[1], P[2]
        vals.append(F.softplus(gamma + a_rec - a_ins) + lambda_other * a_oth)
    return torch.stack(vals).mean()


def l_rec(
    eval_logits: Sequence[torch.Tensor],
    true_mask: Sequence[torch.Tensor],
    alpha_supp: float = 1.0,
    alpha_conf: float = 1.0,
    tau_conf: float = 0.0,
    P_u_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    Q: Optional[torch.Tensor] = None,
    beta: float = 0.3,
    use_margin: bool = False,
    margin_gamma: float = 0.4,
    margin_lambda: float = 0.2,
) -> torch.Tensor:
    """Phase-2 recovery-prevention loss on clean post-prefix eval frames.

    L_rec = (1/|U|) · Σ_u [ α_supp · CVaR_0.5({g_u(p):p∈C_u})^+
                           + α_conf · softplus(logmeanexp(g_u) − τ_conf) ]
            + β · L_stale

    Args:
        eval_logits: length-|U| list of [H, W] logit maps.
        true_mask:   length-|U| list of [H, W] binary masks (C_u).
        alpha_supp, alpha_conf: per-term weights.
        tau_conf: confidence-lock threshold (logmeanexp space).
        P_u_list: optional attention-distribution list aligned with V
                  (first 3 clean post-last-insert frames). Passed to
                  `l_stale_kl` or `l_stale_margin`.
        Q: reference distribution for KL; required iff P_u_list given
           and use_margin is False.
        beta: weight on L_stale term (default 0.3).
        use_margin: if True, use the margin fallback form for L_stale.

    Returns:
        Scalar tensor.
    """
    U = len(eval_logits)
    if U == 0:
        raise ValueError("l_rec got zero eval logits")
    if len(true_mask) != U:
        raise ValueError("eval_logits and true_mask must have same length")

    supp = eval_logits[0].new_zeros(())
    conf = eval_logits[0].new_zeros(())
    for g, C in zip(eval_logits, true_mask):
        cvar = masked_cvar_over_set(g, C, alpha=0.5)
        supp = supp + _positive_part(cvar)
        conf = conf + F.softplus(logmeanexp_2d(g) - tau_conf)
    supp = supp / U
    conf = conf / U
    total = alpha_supp * supp + alpha_conf * conf

    if P_u_list is not None:
        if use_margin:
            stale = l_stale_margin(P_u_list, gamma=margin_gamma,
                                   lambda_other=margin_lambda)
        else:
            if Q is None:
                raise ValueError("l_rec with P_u_list requires Q (KL mode)")
            stale = l_stale_kl(P_u_list, Q)
        total = total + beta * stale

    return total


# ---------------------------------------------------------------------------
# Fidelity — L_fid (augmented Lagrangian)
# ---------------------------------------------------------------------------


def l_fid_augmented(
    lpips_values: Sequence[torch.Tensor],
    mu_nu: float = 10.0,
    lpips_budget: float = 0.10,
    seam_dE: Optional[torch.Tensor] = None,
    mu_s: float = 0.0,
) -> torch.Tensor:
    """Augmented-Lagrangian fidelity penalty on insert LPIPS + seam ΔE.

    L_fid = μ_ν · Σ_k (LPIPS(x_ins_k, f_prev_k) − budget)^+  +  μ_s · ΔE_seam

    The (·)^+ makes the penalty zero when under budget and linear above
    it; the augmented-Lagrangian multiplier μ_ν is updated exogenously by
    `update_lagrange_mu` as training progresses.

    Args:
        lpips_values: length-K list of scalar LPIPS values (one per insert).
        mu_nu: current Lagrangian multiplier on LPIPS constraint.
        lpips_budget: target LPIPS ≤ budget (default 0.10 per proposal).
        seam_dE: optional scalar ΔE penalty on the seam band. None to skip.
        mu_s: weight on seam ΔE (default 0.0 — seam term off by default).

    Returns:
        Scalar tensor.
    """
    if len(lpips_values) == 0:
        raise ValueError("l_fid_augmented got empty lpips_values")

    total = lpips_values[0].new_zeros(())
    for lp in lpips_values:
        total = total + mu_nu * _positive_part(lp - lpips_budget)
    if seam_dE is not None and mu_s > 0.0:
        total = total + mu_s * seam_dE
    return total


def update_lagrange_mu(
    mu: float,
    observed: float,
    target: float,
    grow: float = 1.5,
    shrink: float = 1.0,
    mu_min: float = 0.1,
    mu_max: float = 1e4,
) -> float:
    """Exponential multiplier update for the augmented Lagrangian.

    If `observed > target`, grow by `grow` (tighten the constraint);
    otherwise keep at `shrink` (default 1.0 = hold). Clamped to [mu_min,
    mu_max] to avoid blowup.

    Args:
        mu: current multiplier.
        observed: current realization of the constrained quantity
            (e.g. the max LPIPS across K inserts).
        target: the budget (e.g. 0.10).
        grow: multiplicative growth factor when above budget.
        shrink: multiplicative shrink factor when under budget. Default 1
            keeps μ constant; set < 1.0 to let μ decay.
        mu_min, mu_max: clamp bounds.

    Returns:
        Updated multiplier (Python float).
    """
    if observed > target:
        mu = mu * grow
    else:
        mu = mu * shrink
    return float(np.clip(mu, mu_min, mu_max))


# ---------------------------------------------------------------------------
# Seam ΔE (CIE Lab colour difference on a narrow band around the paste)
# ---------------------------------------------------------------------------


def _rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB-uint8-range (values 0..1) → CIE Lab D65.

    Args:
        rgb: [..., 3] tensor with values in [0, 1].

    Returns:
        Tensor of same leading shape in Lab space (L ∈ [0, 100]).

    Implementation follows the standard sRGB → XYZ → Lab chain. This is a
    minimal reimplementation so we don't depend on skimage / kornia at
    optimization time, and so gradients flow cleanly through the whole
    conversion.
    """
    # sRGB companding (linearize).
    thresh = 0.04045
    rgb_linear = torch.where(
        rgb <= thresh,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )
    # sRGB D65 → XYZ matrix (row-major).
    M = rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_linear @ M.T
    # Normalize by D65 white point.
    white = rgb.new_tensor([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    # XYZ → Lab
    eps_lab = (6.0 / 29.0) ** 3
    kappa = 1.0 / 3.0 * (29.0 / 6.0) ** 2
    f = torch.where(
        xyz > eps_lab,
        xyz.clamp_min(0.0).pow(1.0 / 3.0),
        kappa * xyz + 4.0 / 29.0,
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def seam_delta_e(
    insert_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    seam_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CIE76 ΔE averaged over `seam_mask` pixels.

    ΔE76 = √( (L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2 ) — the original 1976
    definition; fine for seam-quality regularization because we only
    need a differentiable penalty, not perceptual fidelity.

    Args:
        insert_rgb: [H, W, 3] tensor in [0, 1] (the current iterate x_ins).
        base_rgb:   [H, W, 3] tensor in [0, 1] (the ProPainter base b_k).
        seam_mask:  [H, W] binary mask of the seam-band (typically a
                    5-px ring around the paste region).

    Returns:
        Scalar tensor. Zero if mask is empty.
    """
    if insert_rgb.shape != base_rgb.shape or insert_rgb.shape[:-1] != seam_mask.shape:
        raise ValueError("insert_rgb / base_rgb / seam_mask shape mismatch")
    lab_ins = _rgb_to_lab_torch(insert_rgb)
    lab_base = _rgb_to_lab_torch(base_rgb)
    diff = lab_ins - lab_base
    de = torch.sqrt((diff ** 2).sum(dim=-1) + eps)
    denom = seam_mask.to(de.dtype).sum().clamp_min(1.0)
    return (de * seam_mask.to(de.dtype)).sum() / denom


# ---------------------------------------------------------------------------
# Convenience: fully-assembled PGD loss (per-step)
# ---------------------------------------------------------------------------


def total_loss(
    L_loss_val: torch.Tensor,
    L_rec_val: torch.Tensor,
    L_fid_val: torch.Tensor,
    lambda_r: float = 1.0,
    lambda_f: float = 1.0,
    gate_loss: float = 1.0,
    gate_rec: float = 1.0,
) -> torch.Tensor:
    """L(ν, δ) = γ_loss · L_loss + γ_rec · λ_r · L_rec + λ_f · L_fid.

    `gate_loss` and `gate_rec` toggle terms on/off across PGD stages:
      Stage 1 (ν-only, L_loss):  gate_loss=1, gate_rec=0
      Stage 2 (δ-only, L_rec):   gate_loss=0, gate_rec=1
      Stage 3 (joint):           gate_loss=1, gate_rec=1

    Args:
        L_loss_val, L_rec_val, L_fid_val: scalar tensors.
        lambda_r, lambda_f: global weights on rec / fid blocks.
        gate_loss, gate_rec: per-stage gates.

    Returns:
        Scalar tensor.
    """
    return (
        gate_loss * L_loss_val
        + gate_rec * lambda_r * L_rec_val
        + lambda_f * L_fid_val
    )


# ---------------------------------------------------------------------------
# Smoke tests (executable via `python -m memshield.losses_v2`)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    torch.manual_seed(0)
    H, W = 16, 24

    # 1. masked_cvar_over_set — top 50% of values inside mask
    vals = torch.arange(H * W, dtype=torch.float32).reshape(H, W)
    mask = torch.zeros_like(vals)
    mask[4:8, 4:12] = 1.0   # 4*8 = 32 pixels
    cv = masked_cvar_over_set(vals, mask, alpha=0.5)
    # Pixels in mask: indices 4*W+4 .. 7*W+11; values = 4W+4..4W+11, 5W+4..5W+11, 6W+4..6W+11, 7W+4..7W+11
    inside = vals[mask > 0.5].sort(descending=True).values
    expected = inside[:16].mean()      # ceil(0.5*32)=16
    assert torch.allclose(cv, expected), (cv, expected)

    # 2. logmeanexp_2d
    x = torch.randn(H, W)
    lme = logmeanexp_2d(x).item()
    expected = (x.logsumexp(0).logsumexp(0) - math.log(H * W)).item()
    assert abs(lme - expected) < 1e-5

    # 3. roi_bce with full-ones ROI equals unreduced BCE mean
    g = torch.randn(H, W, requires_grad=True)
    t = (torch.rand(H, W) > 0.5).float()
    roi = torch.ones_like(t)
    bce_mine = roi_bce(g, t, roi)
    bce_ref = F.binary_cross_entropy_with_logits(g, t)
    assert torch.allclose(bce_mine, bce_ref, atol=1e-6)

    # 4. roi_bce with ROI subset
    roi2 = torch.zeros_like(t)
    roi2[4:8, 4:12] = 1.0
    bce2 = roi_bce(g, t, roi2)
    sub = F.binary_cross_entropy_with_logits(g[4:8, 4:12], t[4:8, 4:12])
    assert torch.allclose(bce2, sub, atol=1e-6), (bce2, sub)

    # 5. l_loss_insert: positive, gradient flows
    K = 3
    logits = [torch.randn(H, W, requires_grad=True) for _ in range(K)]
    D = [torch.zeros(H, W) for _ in range(K)]
    for m in D:
        m[6:10, 10:16] = 1.0
    C = [torch.zeros(H, W) for _ in range(K)]
    for m in C:
        m[6:10, 4:10] = 1.0
    ROI = [torch.clamp(D_k + C_k, 0, 1) for D_k, C_k in zip(D, C)]
    L1 = l_loss_insert(logits, D, C, ROI, alpha=1.0, margin=0.2)
    assert L1.requires_grad
    L1.backward()
    assert all(L.grad is not None and torch.isfinite(L.grad).all() for L in logits)

    # 6. l_stale_kl: KL(Q||P); zero when P == Q, positive otherwise
    Q = torch.tensor([0.6, 0.2, 0.2])
    P_equal = [Q.clone().requires_grad_(True)]
    P_off = [torch.tensor([0.2, 0.6, 0.2], requires_grad=True)]
    kl_eq = l_stale_kl(P_equal, Q)
    kl_off = l_stale_kl(P_off, Q)
    assert kl_eq.abs().item() < 1e-5, kl_eq
    assert kl_off.item() > kl_eq.item(), (kl_eq, kl_off)

    # 7. l_stale_margin: shrinks when A_ins grows (sensible gradient sign)
    P_lo = [torch.tensor([0.2, 0.6, 0.2], requires_grad=True)]
    P_hi = [torch.tensor([0.8, 0.1, 0.1], requires_grad=True)]
    lo = l_stale_margin(P_lo, gamma=0.4, lambda_other=0.2)
    hi = l_stale_margin(P_hi, gamma=0.4, lambda_other=0.2)
    assert lo.item() > hi.item(), (lo, hi)

    # 8. l_rec with L_stale term in both KL and margin modes
    Us = 4
    evl = [torch.randn(H, W, requires_grad=True) for _ in range(Us)]
    Cs = [(torch.rand(H, W) > 0.7).float() for _ in range(Us)]
    P_list = [torch.tensor([0.5, 0.3, 0.2], requires_grad=True) for _ in range(3)]
    rec_kl = l_rec(evl, Cs, alpha_supp=1.0, alpha_conf=1.0, tau_conf=-1.0,
                    P_u_list=P_list, Q=Q, beta=0.3)
    rec_mg = l_rec(evl, Cs, alpha_supp=1.0, alpha_conf=1.0, tau_conf=-1.0,
                    P_u_list=P_list, beta=0.3, use_margin=True,
                    margin_gamma=0.4, margin_lambda=0.2)
    assert rec_kl.requires_grad and rec_mg.requires_grad
    assert rec_kl.item() > 0 and rec_mg.item() > 0

    # 9. l_fid_augmented: zero below budget, positive above, gradient
    lp_under = [torch.tensor(0.05, requires_grad=True) for _ in range(3)]
    lp_over = [torch.tensor(0.15, requires_grad=True) for _ in range(3)]
    fid_u = l_fid_augmented(lp_under, mu_nu=10.0, lpips_budget=0.10)
    fid_o = l_fid_augmented(lp_over, mu_nu=10.0, lpips_budget=0.10)
    assert fid_u.item() == 0.0, fid_u
    # mu*3*(0.15-0.10) = 10 * 3 * 0.05 = 1.5
    assert abs(fid_o.item() - 1.5) < 1e-5, fid_o
    fid_o.backward()
    assert all(torch.isfinite(lp.grad) and lp.grad.item() > 0 for lp in lp_over)

    # 10. Lagrangian update rule: grows above target, holds below
    m0 = 1.0
    m1 = update_lagrange_mu(m0, observed=0.15, target=0.10, grow=1.5)
    assert abs(m1 - 1.5) < 1e-6, m1
    m2 = update_lagrange_mu(m1, observed=0.05, target=0.10, grow=1.5)
    assert m2 == m1

    # 11. seam_delta_e: zero for identical images, positive for shifted
    img_a = torch.rand(H, W, 3, requires_grad=True)
    img_b = img_a.detach().clone()
    seam = torch.zeros(H, W)
    seam[6:10, 10:14] = 1.0
    de_same = seam_delta_e(img_a, img_b, seam)
    assert de_same.item() < 0.1, de_same
    img_c = (img_a.detach() + 0.3).clamp(0, 1).requires_grad_(True)
    de_diff = seam_delta_e(img_a, img_c, seam)
    assert de_diff.item() > 1.0, de_diff

    # 12. total_loss stage gating
    t_s1 = total_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0),
                       lambda_r=1.0, lambda_f=1.0, gate_loss=1, gate_rec=0)
    assert t_s1.item() == 4.0, t_s1
    t_s2 = total_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0),
                       lambda_r=1.0, lambda_f=1.0, gate_loss=0, gate_rec=1)
    assert t_s2.item() == 5.0, t_s2
    t_s3 = total_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0),
                       lambda_r=1.0, lambda_f=1.0, gate_loss=1, gate_rec=1)
    assert t_s3.item() == 6.0, t_s3

    print("  all 12 loss-module invariants OK")


if __name__ == "__main__":
    _smoke()
