"""Hiera feature-steering δ utilities for VADI-v5 (codex Loop 3 R2 design #1).

Codex Round 2 verdict (2026-04-25): boundary-bridge δ failed (+0.0006);
the surviving δ branch is "perceptual-budgeted feature/carrier δ as a
second-order co-mechanism". Specifically: drive post-insert frames'
Hiera (backbone) features toward a synthetic-decoy teacher in addition
to the existing decoy-mask margin loss.

## Why feature-steering (vs boundary-bridge)

Boundary-bridge attacked the **mask-decision contour** with tiny
amplitude. SAM2's mask decoder is trained to be edge-robust, so a
4/255 perturbation on a 5-pixel-wide ring couldn't shift the decision.

Feature-steering attacks the **upstream Hiera representation** —
the current-frame backbone tokens that feed memory_attention. SAM2's
Hiera was NOT trained to be invariant to small input perturbations
in textured regions. A teacher-aligned δ pushes Hiera tokens at post-
insert frames toward "looks like the decoy is here", which then
biases the memory-attention readout toward the decoy mask
geometrically.

## v0 simplifications (this file)

- One teacher per insert k: a synthetic decoy frame built from
  `x_clean[c_k]` via `build_duplicate_object_decoy_frame`. Use the
  same teacher Hiera for all post-insert frames in insert k's window.
- Uniform ε=4/255 ℓ∞ clamp (NO gain-map yet; codex acknowledged this
  is a follow-up).
- L2 distance loss in Hiera token space, summed across polish frames.

Future v1 extensions (deferred until v0 shows signal):
- Per-frame teacher (not per-insert) — target Hiera of "what frame t
  would look like with object at decoy position".
- Gain-map δ parameterization with bisection shrink to LPIPS≤0.20.
- Outside-mask identity preservation + seam ΔE/TV penalties.

## Self-test

`python -m memshield.hiera_features` → synthetic-input checks for
teacher construction + Hiera distance + edge cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


# =============================================================================
# Teacher frame construction
# =============================================================================


def build_decoy_teacher_frames(
    x_clean: Tensor,                  # [T, H, W, 3] in [0, 1]
    pseudo_masks: Sequence[np.ndarray],
    W_clean_positions: Sequence[int],
    decoy_offsets: Sequence[Tuple[int, int]],
    *,
    feather_radius: int = 5,
    feather_sigma: float = 2.0,
) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Build per-insert synthetic decoy teacher frames.

    For each insert k at clean-space anchor `c_k = W_clean_positions[k]`:
      teacher_k = duplicate_object_decoy(x_clean[c_k], pseudo_mask[c_k],
                                         decoy_offsets[k])

    The duplicate places a copy of the tracked object at
    `original_position + decoy_offsets[k]` (with feathered alpha) while
    keeping the original. Codex insight: "this gives Hiera a concrete
    decoy-shaped feature signal we can teach the post-insert pred
    toward."

    Codex R3 critical fix (2026-04-25): build per-insert directly with
    the caller's recorded offsets. Previously delegated to
    `build_decoy_insert_seeds` with `decoy_offset=None` which would
    recompute offsets fresh — causing teacher to be spatially
    misaligned with the v5 driver's actual decoy-mask construction.

    Returns:
      teachers: `[K, H, W, 3]` float on x_clean's device
      offsets_used: list of `(dy, dx)` per k (echoed back from input)
    """
    from memshield.decoy_seed import build_duplicate_object_decoy_frame
    K = len(W_clean_positions)
    if K == 0:
        return x_clean.new_zeros((0, *x_clean.shape[1:])), []
    if len(decoy_offsets) != K:
        raise ValueError(
            f"len(decoy_offsets)={len(decoy_offsets)} != "
            f"len(W_clean_positions)={K}")
    seeds: List[Tensor] = []
    offsets_echo: List[Tuple[int, int]] = []
    for k, c_k in enumerate(W_clean_positions):
        c_k = int(c_k)
        if not (0 <= c_k < x_clean.shape[0]):
            raise ValueError(
                f"insert anchor c_k={c_k} out of [0, {x_clean.shape[0]})")
        x_ref = x_clean[c_k]
        mask_np = np.asarray(pseudo_masks[c_k], dtype=np.float32)
        mask_t = torch.from_numpy(mask_np).to(
            x_clean.device).to(x_clean.dtype)
        dy, dx = int(decoy_offsets[k][0]), int(decoy_offsets[k][1])
        seed = build_duplicate_object_decoy_frame(
            x_ref, mask_t, (dy, dx),
            feather_radius=feather_radius,
            feather_sigma=feather_sigma,
        )
        seeds.append(seed)
        offsets_echo.append((dy, dx))
    return torch.stack(seeds, dim=0), offsets_echo


# =============================================================================
# Hiera token extraction (with gradient through processed input)
# =============================================================================


def extract_hiera_tokens_with_grad(
    predictor,
    images_norm: Tensor,             # [B, 3, S, S] already SAM2-normalized
    *,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    use_gradient_checkpointing: bool = True,
    fpn_level: int = -1,             # which FPN level to extract; -1 = deepest
) -> Tensor:
    """Run `predictor.forward_image` and return the requested FPN level.

    Differentiable wrt `images_norm`. Stays inside the same autocast and
    gradient-checkpointing regime as the main VADI forward to maintain
    bf16 + memory-friendly behavior.

    Args:
        predictor: SAM2VideoPredictor instance.
        images_norm: `[B, 3, S, S]` SAM2-normalized images. B≥1; the
            function returns a single tensor concatenated over B.
        autocast_dtype: bf16 default for Pro 6000 Blackwell parity.
        use_gradient_checkpointing: True → wraps forward_image with
            `torch.utils.checkpoint.checkpoint` (use_reentrant=False).
        fpn_level: which `backbone_fpn` index to return. -1 = deepest
            (the level used by vulnerability_scorer).

    Returns:
        Hiera tokens at the requested FPN level, shape
        `[B, C, H_feat, W_feat]` for the chosen level, on `images_norm`'s
        device, fp32 (cast back from bf16 for stable downstream losses).
    """
    if autocast_dtype is not None and images_norm.device.type == "cuda":
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        if use_gradient_checkpointing and images_norm.requires_grad:
            from torch.utils.checkpoint import checkpoint as _ckpt
            backbone_out = _ckpt(
                predictor.forward_image, images_norm,
                use_reentrant=False,
            )
        else:
            backbone_out = predictor.forward_image(images_norm)
    fpn = backbone_out["backbone_fpn"]
    token = fpn[int(fpn_level)]
    return token.float()             # cast to fp32 for stable L2 / cosine


@torch.no_grad()
def extract_hiera_teacher_tokens(
    predictor,
    teacher_frames: Tensor,          # [K, H, W, 3] in [0, 1]
    *,
    image_size: int = 1024,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    fpn_level: int = -1,
) -> List[Tensor]:
    """Run clean SAM2 on each teacher frame (no grad) → extract Hiera tokens.

    Used ONCE per clip at setup, before the PGD loop starts. The
    returned tokens stay on GPU as the targets (.detach().clone() to
    decouple from any compute graph).

    Returns: list of K Hiera tokens, each shape `[1, C, h, w]`.
    """
    from memshield.vadi_sam2_wiring import _to_sam2_input

    K = teacher_frames.shape[0]
    teachers: List[Tensor] = []
    if autocast_dtype is not None and teacher_frames.device.type == "cuda":
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    img_mean_t = torch.tensor(img_mean, device=teacher_frames.device,
                              dtype=teacher_frames.dtype).view(1, 3, 1, 1)
    img_std_t = torch.tensor(img_std, device=teacher_frames.device,
                             dtype=teacher_frames.dtype).view(1, 3, 1, 1)

    with autocast_ctx:
        for k in range(K):
            frame = teacher_frames[k:k + 1]                     # [1, H, W, 3]
            img_norm = _to_sam2_input(frame, image_size, img_mean_t, img_std_t)
            backbone_out = predictor.forward_image(img_norm)
            tok = backbone_out["backbone_fpn"][int(fpn_level)]
            teachers.append(tok.detach().clone().float())
    return teachers


# =============================================================================
# Hiera feature loss
# =============================================================================


def hiera_feature_l2_loss(
    student_tokens_by_t: Dict[int, Tensor],   # t → [1, C, h, w] (with grad)
    teacher_tokens_by_k: List[Tensor],         # k → [1, C, h, w] (no grad)
    polish_to_insert_k: Dict[int, int],        # t → most-recent-insert k
    *,
    eps: float = 1e-8,
    normalize: bool = True,
) -> Tensor:
    """L2 distance between student and teacher Hiera tokens.

    For each polish frame t, look up `k = polish_to_insert_k[t]` and
    compute `||student_tokens_by_t[t] - teacher_tokens_by_k[k]||²`,
    optionally normalized by token norm. Sum across polish frames.

    Returns scalar tensor, with grad to student tokens.
    """
    if not student_tokens_by_t:
        # Caller responsibility: this should not happen during normal polish.
        # Return device-neutral zero.
        return torch.zeros((), dtype=torch.float32)
    losses: List[Tensor] = []
    for t, st_tok in student_tokens_by_t.items():
        k = polish_to_insert_k.get(int(t))
        if k is None:
            continue
        if k < 0 or k >= len(teacher_tokens_by_k):
            continue
        te_tok = teacher_tokens_by_k[k].to(st_tok.device).to(st_tok.dtype)
        diff = (st_tok - te_tok)                  # [1, C, h, w]
        if normalize:
            denom = te_tok.pow(2).mean().clamp(min=eps).sqrt()
            losses.append(diff.pow(2).mean() / denom)
        else:
            losses.append(diff.pow(2).mean())
    if not losses:
        return torch.zeros(
            (), dtype=next(iter(student_tokens_by_t.values())).dtype,
            device=next(iter(student_tokens_by_t.values())).device)
    return torch.stack(losses).sum()


def hiera_feature_cosine_loss(
    student_tokens_by_t: Dict[int, Tensor],
    teacher_tokens_by_k: List[Tensor],
    polish_to_insert_k: Dict[int, int],
    *,
    eps: float = 1e-8,
) -> Tensor:
    """1 - cosine_similarity between student and teacher per polish frame.

    Cosine is rotation-invariant on the token magnitude — sometimes more
    robust than L2 if Hiera tokens have variable scale across frames.
    Sum across polish frames.
    """
    if not student_tokens_by_t:
        return torch.zeros((), dtype=torch.float32)
    losses: List[Tensor] = []
    for t, st_tok in student_tokens_by_t.items():
        k = polish_to_insert_k.get(int(t))
        if k is None or k < 0 or k >= len(teacher_tokens_by_k):
            continue
        te_tok = teacher_tokens_by_k[k].to(st_tok.device).to(st_tok.dtype)
        st_flat = st_tok.flatten(1)                  # [1, C·h·w]
        te_flat = te_tok.flatten(1)
        cos = torch.nn.functional.cosine_similarity(
            st_flat, te_flat, dim=-1, eps=eps)
        losses.append((1.0 - cos).mean())
    if not losses:
        return torch.zeros(
            (), dtype=next(iter(student_tokens_by_t.values())).dtype,
            device=next(iter(student_tokens_by_t.values())).device)
    return torch.stack(losses).sum()


# =============================================================================
# Helper: polish-frame → most-recent-insert-k mapping (in processed space)
# =============================================================================


def build_polish_to_insert_k_map(
    polish_frame_ids_proc: Sequence[int],
    W_attacked: Sequence[int],
) -> Dict[int, int]:
    """For each polish frame `t` (processed-space), return `k` such that
    `W_attacked[k]` is the most recent insert ≤ t.

    Insert positions themselves get `k = (their position in W_attacked sorted)`.
    Pre-first-insert frames get `k = -1` (caller should skip).
    """
    W_sorted = sorted(int(w) for w in W_attacked)
    out: Dict[int, int] = {}
    for t in polish_frame_ids_proc:
        t = int(t)
        k_cover = -1
        for k, w in enumerate(W_sorted):
            if w <= t:
                k_cover = k
            else:
                break
        out[t] = k_cover
    return out


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    np.random.seed(0); torch.manual_seed(0)

    # -- build_polish_to_insert_k_map: simple cases
    W = [5, 10, 15]
    polish = [4, 6, 11, 14, 16, 20]
    m = build_polish_to_insert_k_map(polish, W)
    assert m[4] == -1                       # before any insert
    assert m[6] == 0                        # after W[0]=5
    assert m[11] == 1                       # after W[1]=10
    assert m[14] == 1                       # still after W[1] (before W[2]=15)
    assert m[16] == 2                       # after W[2]
    assert m[20] == 2

    # -- hiera_feature_l2_loss: synthetic tokens
    # Create dummy teacher and student tokens with known L2 distances.
    K = 2
    C, h, w = 4, 3, 3
    teacher_tokens = [
        torch.zeros(1, C, h, w),            # k=0: zero
        torch.ones(1, C, h, w) * 2.0,       # k=1: all 2s
    ]
    # Polish frames {0, 1, 2, 3} with map t→k (built from W_attacked=[5, 10] and
    # polish containing [5, 6, 10, 11], say).
    student = {
        0: torch.zeros(1, C, h, w, requires_grad=True),    # match teacher 0
        1: torch.ones(1, C, h, w, requires_grad=True),     # diff 1 from teacher 0
        2: torch.ones(1, C, h, w) * 2.0,                   # match teacher 1
        2: torch.ones(1, C, h, w, requires_grad=True) * 2.0,
        3: torch.zeros(1, C, h, w, requires_grad=True),    # diff 2 from teacher 1
    }
    p2k = {0: 0, 1: 0, 2: 1, 3: 1}
    L = hiera_feature_l2_loss(student, teacher_tokens, p2k, normalize=False)
    # Frame 0: diff = 0 → 0 mean. Frame 1: diff = 1 → 1 mean. Frame 2: 0. Frame 3: diff=2 → 4 mean.
    # Total = 0 + 1 + 0 + 4 = 5.
    assert abs(L.item() - 5.0) < 1e-5, f"L2 sum mismatch: got {L.item()}"

    # -- gradient flow
    L.backward()
    assert student[1].grad is not None and student[1].grad.abs().sum() > 0
    assert student[3].grad is not None

    # -- normalized variant
    student2 = {0: torch.ones(1, C, h, w, requires_grad=True)}
    teacher2 = [torch.ones(1, C, h, w) * 2.0]
    p2k2 = {0: 0}
    Ln = hiera_feature_l2_loss(student2, teacher2, p2k2, normalize=True)
    # diff = 1, diff² mean = 1; teacher norm rms = 2; normalized = 1/2 = 0.5
    assert abs(Ln.item() - 0.5) < 1e-4

    # -- cosine loss
    student3 = {0: torch.ones(1, C, h, w, requires_grad=True),
                1: torch.full((1, C, h, w), -1.0, requires_grad=True)}
    teacher3 = [torch.ones(1, C, h, w)]
    p2k3 = {0: 0, 1: 0}
    Lc = hiera_feature_cosine_loss(student3, teacher3, p2k3)
    # Frame 0: cos = 1 → loss 0. Frame 1: cos = -1 → loss 2. Sum = 2.
    assert abs(Lc.item() - 2.0) < 1e-4

    # -- Out-of-range k_cover (pre-first-insert)
    p2k_bad = {0: -1}
    L_skip = hiera_feature_l2_loss(student2, teacher2, p2k_bad)
    assert L_skip.item() == 0.0     # k=-1 skipped

    # -- empty student tokens
    L_empty = hiera_feature_l2_loss({}, teacher_tokens, {})
    assert L_empty.item() == 0.0

    print("memshield.hiera_features: all self-tests PASSED "
          "(build_polish_to_insert_k_map, hiera_feature_l2_loss "
          "exact + normalized, hiera_feature_cosine_loss, gradient flow, "
          "edge cases: k=-1 skipped, empty input)")


if __name__ == "__main__":
    _self_test()
