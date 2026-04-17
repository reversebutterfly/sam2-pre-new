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


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over a binary mask."""
    area = mask.sum()
    if area.item() <= 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (x * mask).sum() / (area + 1e-6)


def _masked_softplus_pos(
    logits: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Encourage logits in the mask to exceed +margin."""
    area = mask.sum()
    if area.item() <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = F.softplus(margin - logits)
    return (loss * mask).sum() / (area + 1e-6)


def _masked_softplus_neg(
    logits: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Encourage logits in the mask to stay below -margin."""
    area = mask.sum()
    if area.item() <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = F.softplus(logits + margin)
    return (loss * mask).sum() / (area + 1e-6)


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


def object_score_positive_loss(
    score_logits: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Push object_score_logits robustly above +margin."""
    return F.softplus(margin - score_logits).mean()


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
    """Transfer-oriented relocation loss.

    Union-mask BCE makes it too easy to keep the true object active while adding
    a weak secondary blob at the decoy. For relocation, the decoy must beat the
    true location. This objective therefore combines:
      - positive pressure on support / bridge / decoy regions
      - negative pressure on the true location
      - a ranking term that enforces decoy logits > true logits
    """
    support = target_dict["core"]
    bridge = target_dict["bridge"]
    decoy = target_dict["decoy"]
    suppress = target_dict.get("suppress", torch.zeros_like(pred_logits))

    support_w = float(target_dict.get("core_w", 0.0))
    bridge_w = float(target_dict.get("bridge_w", 0.0))
    decoy_w = float(target_dict.get("decoy_w", 0.0))
    suppress_w = float(target_dict.get("suppress_w", 0.0))
    rank_w = float(target_dict.get("rank_w", 0.0))
    bg_w = float(target_dict.get("bg_w", 0.0))

    support_margin = float(target_dict.get("support_margin", 0.0))
    bridge_margin = float(target_dict.get("bridge_margin", 0.25))
    decoy_margin = float(target_dict.get("decoy_margin", 0.75))
    suppress_margin = float(target_dict.get("suppress_margin", 0.5))
    rank_margin = float(target_dict.get("rank_margin", 0.75))

    loss = torch.zeros((), device=pred_logits.device, dtype=pred_logits.dtype)
    denom = 0.0

    if support_w > 0.0:
        loss = loss + support_w * _masked_softplus_pos(pred_logits, support, support_margin)
        denom += support_w
    if bridge_w > 0.0:
        loss = loss + bridge_w * _masked_softplus_pos(pred_logits, bridge, bridge_margin)
        denom += bridge_w
    if decoy_w > 0.0:
        loss = loss + decoy_w * _masked_softplus_pos(pred_logits, decoy, decoy_margin)
        denom += decoy_w
    if suppress_w > 0.0:
        loss = loss + suppress_w * _masked_softplus_neg(pred_logits, suppress, suppress_margin)
        denom += suppress_w
    if rank_w > 0.0 and decoy.sum().item() > 0 and suppress.sum().item() > 0:
        true_mean = _masked_mean(pred_logits, suppress)
        decoy_mean = _masked_mean(pred_logits, decoy)
        loss = loss + rank_w * F.softplus(true_mean - decoy_mean + rank_margin)
        denom += rank_w
    if bg_w > 0.0:
        occupied = ((support + bridge + decoy + suppress) > 0.5).float()
        bg = 1.0 - occupied
        loss = loss + bg_w * _masked_softplus_neg(pred_logits, bg, 0.0)
        denom += bg_w

    if denom <= 0.0:
        return torch.zeros((), device=pred_logits.device, dtype=pred_logits.dtype)
    return loss / denom


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


def memory_teacher_loss(
    adv_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Align adversarial memory features WITH decoy teacher features.

    Unlike memory_drift_loss (which just says "be different from clean"),
    this says "be the same as what SAM2 would write if it were tracking
    the object at the decoy location." This is the key difference between
    suppression (diverge from truth) and decoy (converge to specific lie).

    Uses a mixed loss:
      - Channel-wise cosine similarity (align feature channel distributions)
      - Spatial smooth-L1 on L2-normalized features (align spatial activation pattern)

    The spatial term preserves the structure that tells SAM2 WHERE the object
    is, not just WHAT it looks like globally.
    """
    if adv_features is None or teacher_features is None:
        dev = adv_features.device if adv_features is not None else teacher_features.device
        return torch.zeros((), device=dev)

    teacher_det = teacher_features.detach()

    # Handle varying tensor shapes: [B, C, H, W] or [B, HW, C] or flat
    if adv_features.dim() >= 3:
        # Spatial features: use mixed channel cosine + spatial smooth-L1
        # Reshape to [B, C, spatial...] if needed
        adv = adv_features
        tch = teacher_det
        if adv.dim() == 3:
            # [B, HW, C] → [B, C, HW]
            adv = adv.permute(0, 2, 1)
            tch = tch.permute(0, 2, 1)

        # Channel cosine: average cosine sim across spatial positions
        # [B, C, H, W] → flatten spatial → [B, C, N]
        B = adv.shape[0]
        C = adv.shape[1]
        adv_flat = adv.reshape(B, C, -1)   # [B, C, N]
        tch_flat = tch.reshape(B, C, -1)   # [B, C, N]
        # Cosine per spatial position
        cos_per_pos = F.cosine_similarity(adv_flat, tch_flat, dim=1)  # [B, N]
        channel_cos_loss = 1.0 - cos_per_pos.mean()

        # Spatial smooth-L1: normalize features then compare spatially
        adv_norm = F.normalize(adv_flat, dim=1)   # L2 normalize along channels
        tch_norm = F.normalize(tch_flat, dim=1)
        spatial_loss = F.smooth_l1_loss(adv_norm, tch_norm)

        return 0.6 * channel_cos_loss + 0.4 * spatial_loss
    else:
        # Fallback: global cosine for flat/1D features
        adv_flat = adv_features.flatten()
        teacher_flat = teacher_det.flatten()
        cos_sim = F.cosine_similarity(adv_flat.unsqueeze(0), teacher_flat.unsqueeze(0))
        return 1.0 - cos_sim.mean()


def anti_anchor_loss(
    adv_features: torch.Tensor,
    clean_anchor_features: torch.Tensor,
) -> torch.Tensor:
    """Push adversarial memory features AWAY from the clean true-location anchor.

    Used on f0 and pre-insert originals to weaken the correct memory anchor
    so that the poisoned decoy memories have more influence during cross-attention.

    Uses cosine similarity: minimize → features diverge from clean anchor.
    """
    if adv_features is None or clean_anchor_features is None:
        return torch.zeros((), device=adv_features.device if adv_features is not None
                           else clean_anchor_features.device)
    adv_flat = adv_features.flatten()
    anchor_flat = clean_anchor_features.flatten().detach()
    cos_sim = F.cosine_similarity(adv_flat.unsqueeze(0), anchor_flat.unsqueeze(0))
    return cos_sim.mean()  # Minimize → diverge from correct anchor


def obj_ptr_teacher_loss(
    adv_ptr: torch.Tensor,
    teacher_ptr: torch.Tensor,
) -> torch.Tensor:
    """Align adversarial obj_ptr with decoy teacher obj_ptr.

    obj_ptr encodes object identity. Matching the teacher ptr ensures SAM2
    associates the same object identity with the decoy location.
    """
    if adv_ptr is None or teacher_ptr is None:
        return torch.zeros((), device=adv_ptr.device if adv_ptr is not None
                           else teacher_ptr.device)
    adv_flat = adv_ptr.flatten()
    teacher_flat = teacher_ptr.flatten().detach()
    return F.mse_loss(adv_flat, teacher_flat)


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
