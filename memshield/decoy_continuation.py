"""Joint Trajectory-Consistent Decoy Attack helpers (codex Loop 3 R4 design,
locked 2026-04-25).

Replaces ε∞-PGD δ on bridge originals with semantic editing: each bridge
original t carries a feathered, low-α duplicate-object overlay from its own
clean content + a small decoy-directed translation warp on the true-object
ROI. Goal: bridge originals BECOME plausible continuation frames of the
inserted decoy's false trajectory, rather than weakly nudged real frames.

Locked design (codex final answer):
  - Overlay source: per-bridge-frame duplicate built from frame t's own clean
    content via build_duplicate_object_decoy_frame (NOT replay of insert seed).
  - Overlay strength: learnable α_t = α_max · sigmoid(a_t), α_max ∈ [0.25, 0.35].
  - Overlay mask: softened, slightly dilated decoy region mask (σ = 2-3 px).
  - Warp: translation-only, s_t·u_k + r_t·u_k_perp (u_k = unit decoy direction).
    Hard cap |d_t| ≤ 3 px (default ≤ 2 px). Initialize in decoy direction,
    regularize orthogonal motion (r_t).
  - Optimization: Adam on α and warp params (low-dim smooth controls).
  - Threat model: drop ε∞ on non-prompt frames. Keep LPIPS ≤ 0.20.

Pure torch (no SAM2). Run `python -m memshield.decoy_continuation` for self-tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------------


def soften_decoy_mask(
    mask: Tensor, *, dilate_px: int = 2, feather_sigma: float = 2.5,
) -> Tensor:
    """Dilate a binary/soft mask by `dilate_px` then Gaussian-feather with
    `feather_sigma`.

    mask:           [H, W] float in [0, 1] or [1, H, W]
    Returns:        [H, W] float in [0, 1]
    """
    if mask.dim() == 3 and mask.shape[0] == 1:
        m = mask[0]
    elif mask.dim() == 2:
        m = mask
    else:
        raise ValueError(
            f"soften_decoy_mask: unexpected mask shape {tuple(mask.shape)}")
    m = m.clamp(0.0, 1.0).float()
    H, W = m.shape

    # Dilate via max-pool (kernel = 2*dilate_px + 1).
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m4 = m.view(1, 1, H, W)
        m_dilated = F.max_pool2d(m4, kernel_size=k, stride=1, padding=k // 2)
        m = m_dilated.view(H, W)

    # Gaussian feather. Build a 1-D Gaussian kernel, separable.
    if feather_sigma > 0:
        kernel_radius = max(1, int(round(3 * feather_sigma)))
        ks = 2 * kernel_radius + 1
        coords = torch.arange(ks, dtype=m.dtype, device=m.device) - kernel_radius
        gauss = torch.exp(-0.5 * (coords / feather_sigma) ** 2)
        gauss = gauss / gauss.sum()
        # Apply 1-D conv along width then height.
        m4 = m.view(1, 1, H, W)
        gauss_h = gauss.view(1, 1, 1, ks)
        gauss_v = gauss.view(1, 1, ks, 1)
        m4 = F.conv2d(m4, gauss_h, padding=(0, kernel_radius))
        m4 = F.conv2d(m4, gauss_v, padding=(kernel_radius, 0))
        m = m4.view(H, W)

    return m.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Continuation overlay (semantic edit on bridge original)
# ---------------------------------------------------------------------------


def apply_continuation_overlay(
    x_clean_t: Tensor,                  # [H, W, 3] in [0, 1]
    duplicate_t: Tensor,                # [H, W, 3] in [0, 1] (already-built duplicate-object frame)
    softened_decoy_mask: Tensor,        # [H, W] in [0, 1]
    alpha: Tensor,                      # scalar in [0, alpha_max] (differentiable)
) -> Tensor:
    """Blend a duplicate-object frame onto a clean original via a softened
    decoy-region mask, with learnable strength α.

    Output formula at each pixel:
        out = (1 - α · mask) * x_clean + (α · mask) * duplicate

    Returns [H, W, 3] in [0, 1].

    Why this form: when α=0, output = x_clean exactly. When α=α_max, the
    decoy region fully replaces clean; the surround stays clean. The α·mask
    factor is the effective "duplicate fraction" per pixel — perfect for
    LPIPS budget control (small α ⇒ small perceptual change).
    """
    if x_clean_t.shape != duplicate_t.shape:
        raise ValueError(
            f"shape mismatch: x_clean={tuple(x_clean_t.shape)} vs "
            f"duplicate={tuple(duplicate_t.shape)}")
    H, W, C = x_clean_t.shape
    if int(softened_decoy_mask.shape[0]) != H or \
            int(softened_decoy_mask.shape[1]) != W:
        raise ValueError(
            f"mask shape {tuple(softened_decoy_mask.shape)} != [H,W]={H},{W}")
    weight = (alpha * softened_decoy_mask).clamp(0.0, 1.0)  # [H, W]
    weight3 = weight.unsqueeze(-1)                          # [H, W, 1]
    blended = (1.0 - weight3) * x_clean_t + weight3 * duplicate_t
    return blended.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Translation warp on true-object ROI (1-3 px decoy-directed shift)
# ---------------------------------------------------------------------------


def apply_translation_warp_roi(
    x: Tensor,                          # [H, W, 3] in [0, 1]
    true_mask_softened: Tensor,         # [H, W] in [0, 1] — true-object region
    displacement_px: Tensor,            # [2] in pixels (dy, dx) — differentiable
) -> Tensor:
    """Warp the true-object ROI of `x` by a small translation, blending with
    the un-warped surround via the softened mask.

    Algorithm:
      1. Build a global grid_sample translation map from `displacement_px`.
      2. Warp the entire frame.
      3. Blend warped (inside mask) with original (outside mask) via the
         softened true_mask.

    This shifts the apparent object by `displacement_px` while keeping the
    background fixed. For small displacements (1-3 px) the seam is invisible
    after feathering.

    displacement_px convention: positive dy = down, positive dx = right.

    Returns [H, W, 3] in [0, 1].
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError(
            f"apply_translation_warp_roi: x must be [H,W,3], got "
            f"{tuple(x.shape)}")
    H, W, C = x.shape
    device = x.device
    dtype = x.dtype

    # Build affine grid for translation. grid_sample uses normalized coords
    # in [-1, 1]; sampling at (gx, gy) reads source pixel (x + gx*W/2, y + gy*H/2).
    # To shift output by +dy down, we sample from y - dy → grid offset = -dy/(H/2).
    # PyTorch grid_sample: output[oy, ox] = input[gy, gx]; we want
    # output[oy, ox] = input[oy - dy_px, ox - dx_px]. So gy = oy - dy_px, etc.
    # Normalized offset = -2 * dy_px / H along the y-axis (similar for x).
    dy = displacement_px[0]
    dx = displacement_px[1]
    norm_dy = -2.0 * dy / H
    norm_dx = -2.0 * dx / W

    # Identity affine + translation.
    theta = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    theta[0, 0, 0] = 1.0
    theta[0, 1, 1] = 1.0
    theta[0, 0, 2] = norm_dx
    theta[0, 1, 2] = norm_dy

    grid = F.affine_grid(
        theta, size=(1, C, H, W), align_corners=False,
    )  # [1, H, W, 2]
    x_chw = x.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    warped = F.grid_sample(
        x_chw, grid, mode='bilinear', padding_mode='reflection',
        align_corners=False,
    )  # [1, C, H, W]
    warped_hwc = warped.squeeze(0).permute(1, 2, 0)  # [H, W, C]

    # Blend warped (where mask is high) with original (where mask is low).
    if int(true_mask_softened.shape[0]) != H or \
            int(true_mask_softened.shape[1]) != W:
        raise ValueError(
            f"true_mask shape {tuple(true_mask_softened.shape)} != [H,W]")
    mask3 = true_mask_softened.clamp(0.0, 1.0).unsqueeze(-1)  # [H, W, 1]
    blended = mask3 * warped_hwc + (1.0 - mask3) * x
    return blended.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Learnable parameters wrapper
# ---------------------------------------------------------------------------


@dataclass
class BridgeEditParams:
    """All learnable params for a single insert k's bridge frames.

    For K inserts and bridge_length L per insert, total params:
      - alpha_logits:  [K, L]   → α_t = α_max · sigmoid(logit)
      - warp_s:        [K, L]   → main displacement along decoy direction (px)
      - warp_r:        [K, L]   → orthogonal displacement (px)

    All initialized to give:
      - α_t with mild decay (0.30, 0.22, 0.16 for L=3)
      - warp aligned with decoy direction at small magnitude (s_init = 1 px)
      - r_init = 0
    """
    alpha_logits: Tensor       # [K, L] requires_grad
    warp_s: Tensor             # [K, L] requires_grad
    warp_r: Tensor             # [K, L] requires_grad


def init_bridge_edit_params(
    K: int, L: int, *,
    alpha_init_decay: Sequence[float] = (0.30, 0.22, 0.16, 0.12),
    alpha_max: float = 0.35,
    s_init_px: float = 1.0,
    r_init_px: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> BridgeEditParams:
    """Initialize bridge-edit parameters.

    alpha_logits chosen so that sigmoid(logit) * alpha_max equals the
    desired alpha_init for each bridge step.
    """
    device = device or torch.device("cpu")
    if L > len(alpha_init_decay):
        # Extend with last value if L exceeds the decay schedule length.
        alpha_init_decay = list(alpha_init_decay) + \
            [alpha_init_decay[-1]] * (L - len(alpha_init_decay))
    alpha_init = torch.tensor(
        alpha_init_decay[:L], dtype=dtype, device=device)  # [L]
    alpha_init = alpha_init.unsqueeze(0).expand(K, L).contiguous()  # [K, L]
    # Solve sigmoid(logit) = alpha / alpha_max for logit.
    eps = 1e-6
    p = (alpha_init / alpha_max).clamp(eps, 1.0 - eps)
    alpha_logits = torch.log(p / (1.0 - p))                # [K, L]
    alpha_logits = alpha_logits.detach().clone().requires_grad_(True)

    warp_s = torch.full(
        (K, L), float(s_init_px), dtype=dtype, device=device,
    ).requires_grad_(True)
    warp_r = torch.full(
        (K, L), float(r_init_px), dtype=dtype, device=device,
    ).requires_grad_(True)
    return BridgeEditParams(
        alpha_logits=alpha_logits, warp_s=warp_s, warp_r=warp_r,
    )


def alpha_from_logits(
    alpha_logits: Tensor, alpha_max: float,
) -> Tensor:
    """[K, L] logit → α via α_max · sigmoid()."""
    return alpha_max * torch.sigmoid(alpha_logits)


def displacement_from_warp(
    s: Tensor, r: Tensor, u_dir: Tensor, max_disp_px: float = 3.0,
) -> Tensor:
    """Build per-bridge-frame displacement vector (dy, dx) in pixels.

    s:        [K, L]   main shift along decoy direction
    r:        [K, L]   orthogonal correction
    u_dir:    [K, 2]   unit vectors (dy, dx) of each insert's decoy direction
    max_disp_px: hard cap on |d|

    Returns: [K, L, 2] (dy, dx)
    """
    if u_dir.dim() != 2 or u_dir.shape[1] != 2:
        raise ValueError(
            f"u_dir must be [K, 2], got {tuple(u_dir.shape)}")
    # u_perp: rotate u by 90° → (-dx, dy)
    u_perp = torch.stack([-u_dir[:, 1], u_dir[:, 0]], dim=-1)  # [K, 2]
    K_, L_ = s.shape
    # d = s · u + r · u_perp, shape [K, L, 2]
    d = s.unsqueeze(-1) * u_dir.unsqueeze(1) + \
        r.unsqueeze(-1) * u_perp.unsqueeze(1)
    # Hard cap on magnitude.
    norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)        # [K, L, 1]
    scale = (max_disp_px / norm).clamp(max=1.0)                # ≤ 1 means clip
    return d * scale


# ---------------------------------------------------------------------------
# Loss helpers (positive objectness, area preservation, regularizers)
# ---------------------------------------------------------------------------


def positive_objectness_loss(
    object_score_logits: Tensor, *, threshold: float = 0.5,
) -> Tensor:
    """Softplus(threshold - logit) summed over frames. Forces logits to
    stay above `threshold` (default +0.5) so SAM2 doesn't collapse to
    no-object — that's suppression, not decoy.

    object_score_logits: [N] (one scalar per bridge frame)
    Returns scalar.
    """
    if object_score_logits.numel() == 0:
        return torch.zeros((), dtype=torch.float32,
                           device=object_score_logits.device)
    return F.softplus(threshold - object_score_logits).mean()


def area_preservation_loss(
    pred_logits_by_t: Dict[int, Tensor],   # {t → logits [H, W] or [1, 1, H, W]}
    true_mask_by_t: Dict[int, Tensor],     # {t → [H, W] in [0, 1]}
    *,
    area_min: float = 0.6, area_max: float = 1.4, eps: float = 1e-4,
) -> Tuple[Tensor, Dict[int, float]]:
    """Penalize predicted-area / true-mask-area ratios outside [area_min,
    area_max]. Prevents collapse to empty mask (suppression-like) and
    runaway expansion.

    Returns (loss, per_frame_area_ratio).
    """
    if not pred_logits_by_t:
        return torch.zeros(()), {}
    losses: List[Tensor] = []
    ratios: Dict[int, float] = {}
    for t, logits in pred_logits_by_t.items():
        if t not in true_mask_by_t:
            continue
        m_true = true_mask_by_t[t].float()
        denom = m_true.sum().clamp_min(eps)
        # Squeeze logits to 2D.
        l = logits
        while l.dim() > 2:
            if l.shape[0] == 1:
                l = l[0]
            else:
                l = l.mean(dim=0)
        pred_area = torch.sigmoid(l).sum()
        ratio = pred_area / denom
        ratios[int(t)] = float(ratio.detach().item())
        loss_t = (
            F.relu(area_min - ratio) + F.relu(ratio - area_max)
        )
        losses.append(loss_t)
    if not losses:
        return torch.zeros(()), {}
    return torch.stack(losses).mean(), ratios


def alpha_regularizer(
    alphas: Tensor,                      # [K, L]
    *, l1_weight: float = 1.0, smoothness_weight: float = 1.0,
) -> Tensor:
    """L1 penalty on α (encourage smaller blends) + temporal-smoothness
    along the L axis.

    Returns scalar.
    """
    L1 = alphas.abs().mean()
    if alphas.shape[-1] >= 2:
        diff = alphas[:, 1:] - alphas[:, :-1]
        smooth = diff.abs().mean()
    else:
        smooth = torch.zeros((), dtype=alphas.dtype, device=alphas.device)
    return l1_weight * L1 + smoothness_weight * smooth


def warp_regularizer(
    s: Tensor, r: Tensor,                # [K, L]
    *, l2_weight: float = 1.0,
    orthogonal_weight: float = 1.0,
    smoothness_weight: float = 1.0,
) -> Tensor:
    """L2 penalty on |d|² + orthogonal-component penalty + smoothness.

    Returns scalar.
    """
    L2_d = (s ** 2 + r ** 2).mean()
    orth = r.abs().mean()
    if s.shape[-1] >= 2:
        ds = s[:, 1:] - s[:, :-1]
        dr = r[:, 1:] - r[:, :-1]
        smooth = (ds ** 2 + dr ** 2).mean()
    else:
        smooth = torch.zeros((), dtype=s.dtype, device=s.device)
    return l2_weight * L2_d + orthogonal_weight * orth + \
        smoothness_weight * smooth


# ---------------------------------------------------------------------------
# Decoy-direction utilities
# ---------------------------------------------------------------------------


def unit_decoy_direction(
    decoy_offsets: Sequence[Tuple[int, int]],
) -> Tensor:
    """Normalize a list of (dy, dx) decoy offsets to unit vectors.

    Returns: [K, 2] (dy, dx) unit. Zero offsets get a default (0, 1) so
    warp logic doesn't div-by-zero.
    """
    out = torch.zeros(len(decoy_offsets), 2, dtype=torch.float32)
    for k, (dy, dx) in enumerate(decoy_offsets):
        v = torch.tensor([float(dy), float(dx)])
        n = v.norm()
        if float(n) < 1e-6:
            out[k] = torch.tensor([0.0, 1.0])
        else:
            out[k] = v / n
    return out


# ---------------------------------------------------------------------------
# Bridge frame selection (re-export for convenience)
# ---------------------------------------------------------------------------


def select_bridge_frames(
    W_attacked: Sequence[int], T_proc: int, bridge_length: int,
) -> Dict[int, List[int]]:
    """Same shape as memshield.state_continuation.select_bridge_frames.
    Re-exported here so this module is self-contained.
    """
    if bridge_length <= 0:
        raise ValueError(f"bridge_length must be >= 1, got {bridge_length}")
    W_sorted = sorted(int(w) for w in W_attacked)
    out: Dict[int, List[int]] = {}
    for k, w in enumerate(W_sorted):
        next_w = W_sorted[k + 1] if k + 1 < len(W_sorted) else T_proc
        end = min(w + int(bridge_length), next_w - 1, T_proc - 1)
        out[k] = list(range(w + 1, end + 1))
    return out


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_soften_decoy_mask() -> None:
    m = torch.zeros(20, 20)
    m[8:12, 8:12] = 1.0
    soft = soften_decoy_mask(m, dilate_px=2, feather_sigma=2.0)
    assert soft.shape == (20, 20)
    assert soft.max().item() <= 1.0 + 1e-6
    assert soft[10, 10].item() > 0.9, soft[10, 10]
    # Edge pixels should be partial.
    assert 0.0 < soft[8, 8].item() <= 1.0
    print("  soften_decoy_mask OK")


def _test_apply_continuation_overlay() -> None:
    H, W = 16, 16
    x = torch.zeros(H, W, 3); x[:, :, 0] = 1.0   # red clean frame
    dup = torch.zeros(H, W, 3); dup[:, :, 1] = 1.0  # green duplicate
    mask = torch.zeros(H, W); mask[4:12, 4:12] = 1.0
    soft = soften_decoy_mask(mask, dilate_px=0, feather_sigma=0.0)
    # alpha=0 → pure clean
    out0 = apply_continuation_overlay(x, dup, soft, torch.tensor(0.0))
    assert torch.allclose(out0, x), "alpha=0 should yield clean"
    # alpha=1 → pure duplicate where mask=1
    out1 = apply_continuation_overlay(x, dup, soft, torch.tensor(1.0))
    inside = out1[8, 8]
    outside = out1[0, 0]
    assert torch.allclose(inside, dup[8, 8]), f"inside mask: {inside}"
    assert torch.allclose(outside, x[0, 0]), f"outside mask: {outside}"
    # Gradient flow check — use non-symmetric loss so the channel-sum mean
    # doesn't vanish (red→green has opposing signs across channels that
    # cancel in mean()). Use an asymmetric loss like sum(out[..., 1]).
    a = torch.tensor(0.5, requires_grad=True)
    out = apply_continuation_overlay(x, dup, soft, a)
    loss = out[..., 1].sum()       # only the green channel
    loss.backward()
    assert a.grad is not None and float(a.grad.item()) != 0.0, a.grad
    print("  apply_continuation_overlay OK")


def _test_apply_translation_warp_roi() -> None:
    H, W = 16, 16
    x = torch.zeros(H, W, 3)
    x[6:10, 6:10, 0] = 1.0  # red square at center
    mask = torch.zeros(H, W); mask[6:10, 6:10] = 1.0
    soft = soften_decoy_mask(mask, dilate_px=1, feather_sigma=1.0)
    # Translation by (2, 0) should shift the red square down by 2 px.
    disp = torch.tensor([2.0, 0.0], requires_grad=True)
    out = apply_translation_warp_roi(x, soft, disp)
    # The center should still be red (mask is high there).
    # The pixel at (12, 8) should now have some red (shifted from (10,8)).
    # Just sanity: red mass should be roughly conserved.
    red_in = x[:, :, 0].sum()
    red_out = out[:, :, 0].sum()
    assert abs(float(red_out - red_in)) < 5.0, f"red mass: {red_in} → {red_out}"
    # Gradient flow.
    out.sum().backward()
    assert disp.grad is not None and disp.grad.abs().sum() > 0
    print("  apply_translation_warp_roi OK")


def _test_init_bridge_edit_params() -> None:
    K, L = 3, 3
    p = init_bridge_edit_params(K, L, alpha_max=0.35)
    assert p.alpha_logits.shape == (K, L)
    assert p.warp_s.shape == (K, L)
    assert p.warp_r.shape == (K, L)
    a = alpha_from_logits(p.alpha_logits, alpha_max=0.35)
    # Default decay (0.30, 0.22, 0.16)
    assert abs(float(a[0, 0]) - 0.30) < 1e-3
    assert abs(float(a[0, 1]) - 0.22) < 1e-3
    assert abs(float(a[0, 2]) - 0.16) < 1e-3
    print("  init_bridge_edit_params OK")


def _test_displacement_from_warp() -> None:
    K, L = 2, 3
    s = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    r = torch.zeros(K, L)
    u = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # k=0 right, k=1 down
    d = displacement_from_warp(s, r, u, max_disp_px=3.0)
    # k=0, l=0: s=1 along u=(0,1) → d=(0,1)
    assert torch.allclose(d[0, 0], torch.tensor([0.0, 1.0]))
    # k=1, l=0: s=2 along u=(1,0) → d=(2,0)
    assert torch.allclose(d[1, 0], torch.tensor([2.0, 0.0]))
    # Cap test: s=10 → norm 10, scaled to 3.
    s_big = torch.tensor([[10.0]])
    r_big = torch.zeros(1, 1)
    u_big = torch.tensor([[1.0, 0.0]])
    d_big = displacement_from_warp(s_big, r_big, u_big, max_disp_px=3.0)
    assert torch.allclose(d_big[0, 0], torch.tensor([3.0, 0.0])), d_big
    print("  displacement_from_warp OK")


def _test_positive_objectness_loss() -> None:
    # Logits well above 0.5: loss small.
    L = positive_objectness_loss(torch.tensor([2.0, 3.0, 1.5]), threshold=0.5)
    assert float(L) < 0.2, L
    # Logits below 0.5: loss > 0.
    L = positive_objectness_loss(torch.tensor([-0.5, 0.0, 0.2]), threshold=0.5)
    assert float(L) > 0.5, L
    print("  positive_objectness_loss OK")


def _test_area_preservation_loss() -> None:
    # Pred area ≈ true area: loss ≈ 0.
    pred = {0: torch.full((10, 10), 5.0)}      # sigmoid(5) ≈ 0.99 → area ≈ 99
    true = {0: torch.ones(10, 10)}             # area = 100
    L, ratios = area_preservation_loss(pred, true, area_min=0.6, area_max=1.4)
    assert float(L) < 0.1, (L, ratios)
    # Pred mostly empty: loss > 0.
    pred = {0: torch.full((10, 10), -5.0)}
    L, ratios = area_preservation_loss(pred, true)
    assert float(L) > 0.5, (L, ratios)
    print("  area_preservation_loss OK")


def _test_select_bridge_frames() -> None:
    out = select_bridge_frames([10, 20, 30], 40, 3)
    assert out == {0: [11, 12, 13], 1: [21, 22, 23], 2: [31, 32, 33]}, out
    print("  select_bridge_frames (re-export) OK")


if __name__ == "__main__":
    print("memshield.decoy_continuation self-tests:")
    _test_soften_decoy_mask()
    _test_apply_continuation_overlay()
    _test_apply_translation_warp_roi()
    _test_init_bridge_edit_params()
    _test_displacement_from_warp()
    _test_positive_objectness_loss()
    _test_area_preservation_loss()
    _test_select_bridge_frames()
    print("memshield.decoy_continuation: all self-tests PASSED")
