"""
Loss functions for MemoryShield v2: margin-based multi-term loss.

Per GPT-5.4 Round 2 review:
  - Don't optimize clamped pred_masks (saturates at -1024)
  - Use object_score_logits with negative margin
  - Use pre-clamp pred_masks_high_res for mask loss
  - Add fake uint8 quantization for transport robustness
"""
import numpy as np
import torch
import torch.nn.functional as F


def mean_logit_loss(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Mean logit in GT region. Minimize → suppress object."""
    gt = gt_masks.float()
    area = gt.sum() + 1e-6
    return (pred_logits * gt).sum() / area


def object_score_margin_loss(
    score_logits: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Push object_score_logits robustly below -margin.

    Uses softplus to provide gradient even when score is already negative.
    Minimize → score robustly negative → object deemed absent.
    """
    # softplus(score + margin) → 0 when score << -margin, → (score+margin) when score >> -margin
    return F.softplus(score_logits + margin).mean()


def fake_uint8_quantize(x: torch.Tensor) -> torch.Tensor:
    """Straight-through fake uint8 quantization.

    Makes PGD aware of quantization during optimization.
    """
    x_q = torch.round((x * 255.0).clamp(0, 255)) / 255.0
    return x + (x_q - x).detach()  # Straight-through estimator


def differentiable_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Differentiable SSIM between [B, C, H, W] tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = x.shape

    k = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    w1d = torch.exp(-k ** 2 / (2 * 1.5 ** 2))
    w1d = w1d / w1d.sum()
    w2d = w1d.unsqueeze(1) * w1d.unsqueeze(0)
    window = w2d.view(1, 1, window_size, window_size).expand(C, 1, -1, -1)

    pad = window_size // 2

    def blur(t):
        return F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), window, groups=C)

    mu_x, mu_y = blur(x), blur(y)
    sig_x2 = blur(x * x) - mu_x * mu_x
    sig_y2 = blur(y * y) - mu_y * mu_y
    sig_xy = blur(x * y) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2)
    return (num / (den + 1e-8)).mean()


def decoy_target_loss(
    pred_logits: torch.Tensor,
    target_dict: dict,
) -> torch.Tensor:
    """Push SAM2 to predict the decoy pseudo-target.

    Uses 3 non-overlapping binary masks with role-specific weights:
      - core: keep predicting object here (BCE target=1, weight=core_w)
      - bridge: extend prediction toward decoy (BCE target=1, weight=bridge_w)
      - decoy: predict object at wrong location (BCE target=1, weight=decoy_w)
      - background: everything else (BCE target=0, weight=1.0)

    Args:
        pred_logits: [1, 1, H, W] mask logits from SAM2.
        target_dict: dict with keys 'core', 'bridge', 'decoy' (each [1,1,H,W] binary)
                     and 'core_w', 'bridge_w', 'decoy_w' (float weights).
    """
    core = target_dict["core"]
    bridge = target_dict["bridge"]
    decoy = target_dict["decoy"]
    core_w = target_dict["core_w"]
    bridge_w = target_dict["bridge_w"]
    decoy_w = target_dict["decoy_w"]

    # Build binary target: 1 where we want SAM2 to predict "object"
    target = torch.zeros_like(pred_logits)
    target[core > 0.5] = 1.0
    target[bridge > 0.5] = 1.0
    target[decoy > 0.5] = 1.0

    # Build per-pixel weight map
    # Background gets LOW weight (0.1) so it doesn't dominate
    # Important regions get HIGH weight relative to background
    bg_w = 0.1
    weight = torch.full_like(pred_logits, bg_w)
    weight[core > 0.5] = core_w
    weight[bridge > 0.5] = bridge_w
    weight[decoy > 0.5] = decoy_w

    # Unreduced BCE, then weight-normalized mean
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none",
    )
    loss = (bce * weight).sum() / (weight.sum() + 1e-6)
    return loss


def memory_drift_loss(
    clean_features: torch.Tensor,
    adv_features: torch.Tensor,
) -> torch.Tensor:
    """Maximize divergence between clean and adversarial memory features.

    Ensures the inserted frame writes a DIFFERENT memory than a clean frame
    at the same position would.
    """
    # Negative cosine similarity (minimize = maximize divergence)
    clean_flat = clean_features.flatten()
    adv_flat = adv_features.flatten()
    cos_sim = F.cosine_similarity(clean_flat.unsqueeze(0), adv_flat.unsqueeze(0))
    return cos_sim.mean()  # Minimize → features diverge


def compute_attack_loss(
    all_frame_outs: list,
    gt_masks_uint8: list,
    eval_mod_indices: list,
    eval_orig_indices: list,
    device: torch.device,
    persistence_weighting: bool = True,
    lambda_obj: float = 1.0,
    lambda_mask: float = 1.0,
    obj_margin: float = 2.0,
) -> torch.Tensor:
    """Multi-term margin loss on future clean frames.

    Combines:
      - object_score_logits margin loss (push robustly negative)
      - mean_logit_loss on high-res masks (push logits down with margin)

    Uses pre-clamp masks to avoid -1024 saturation.
    """
    loss = torch.tensor(0.0, device=device)
    weight_sum = 0.0

    for rank, (mod_idx, orig_idx) in enumerate(zip(eval_mod_indices, eval_orig_indices)):
        if mod_idx >= len(all_frame_outs):
            break
        if orig_idx >= len(gt_masks_uint8):
            break

        frame_out = all_frame_outs[mod_idx]
        w = float(rank + 1) if persistence_weighting else 1.0

        # Mask loss: use high-res pre-clamp masks if available
        hi_masks = frame_out.get("pred_masks_high_res")
        logits = frame_out.get("logits_orig_hw")
        if hi_masks is not None:
            gt_np = gt_masks_uint8[orig_idx]
            gt = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if gt.shape[-2:] != hi_masks.shape[-2:]:
                gt_hi = F.interpolate(gt, size=hi_masks.shape[-2:], mode="nearest")
            else:
                gt_hi = gt
            loss_mask = mean_logit_loss(hi_masks, gt_hi)
        elif logits is not None:
            gt_np = gt_masks_uint8[orig_idx]
            gt = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if gt.shape[-2:] != logits.shape[-2:]:
                gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
            loss_mask = mean_logit_loss(logits, gt)
        else:
            loss_mask = torch.tensor(0.0, device=device)

        # Object score loss: push robustly below -margin
        obj_score = frame_out.get("object_score_logits")
        if obj_score is not None:
            loss_obj = object_score_margin_loss(obj_score, margin=obj_margin)
        else:
            loss_obj = torch.tensor(0.0, device=device)

        loss = loss + w * (lambda_mask * loss_mask + lambda_obj * loss_obj)
        weight_sum += w

    if weight_sum > 0:
        loss = loss / weight_sum
    return loss
