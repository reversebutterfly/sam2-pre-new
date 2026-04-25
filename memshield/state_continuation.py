"""Decoy State Continuation helpers — Loop 3 Round 3 design (codex 2026-04-25).

The mechanism: SAM2 propagates state across frames via two recurrent channels
written by `_encode_new_memory` and read by `memory_attention`:

  1. `maskmem_features` — encoded spatial memory (per-frame [B, C, h, w])
  2. `obj_ptr`         — low-dim object identity token (per-frame [B, D])

Past delta designs (boundary-bridge, hiera-steering) attacked transient
features (current Hiera, contour mask) which DO NOT persist across frames.
This module attacks the persistent recurrent state directly.

Pipeline (called from polish PGD, run_vadi_v5.py Stage 12):
  - During A0 (frozen ν*, δ=0), forward through the SAM2 wrapper to cache
    per-insert `M̄_k = maskmem_features[w_k]` and `p̄_k = obj_ptr[w_k]` as
    "teacher" states.
  - During polish PGD (frozen ν, δ on bridge originals only), forward to get
    student states `M_t = maskmem_features[t]` and `p_t = obj_ptr[t]` at
    bridge frames B_k = {w_k+1 ... w_k+L}.
  - Aggregate loss:
        L_M = mean_{k,t∈B_k}  1 - cos(Pool_d(M_t), Pool_d(M̄_k))
        L_P = mean_{k,t∈B_k}  1 - cos(p_t, p̄_k)
  - Pool is masked-average over the decoy region downsampled to memory res.
  - Falsification log: track per-step mean cos(M, M̄) and cos(p, p̄) lift
    over the warm-start baseline (delta=0 case). Codex pre-committed:
    if state alignment up >= 0.15 BUT mean J-drop < +0.02 → cut δ permanently.

Pure torch. No SAM2-internal calls. Run `python -m memshield.state_continuation`
for self-tests.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _normalize_mask_2d(m: Tensor) -> Tensor:
    """Squeeze a soft mask down to 2-D [h, w] of float in [0, 1].

    Accepts shapes [h, w], [1, h, w], [h, w, 1], [1, 1, h, w], etc.
    Clamps values to [0, 1].
    """
    while m.dim() > 2:
        # Squeeze leading or trailing singletons in priority.
        if m.shape[0] == 1:
            m = m.squeeze(0)
        elif m.shape[-1] == 1:
            m = m.squeeze(-1)
        else:
            # Sum over remaining batch/channel dims as a last resort.
            m = m.mean(dim=0)
    return m.clamp(0.0, 1.0).float()


def downsample_decoy_mask(
    decoy_mask: Tensor, target_h: int, target_w: int,
) -> Tensor:
    """Downsample a soft decoy mask [H, W] (or 3D/4D) to memory resolution
    [target_h, target_w] via adaptive avg-pool. Returns [target_h, target_w]
    in [0, 1].
    """
    m2 = _normalize_mask_2d(decoy_mask)               # [H, W]
    if int(m2.shape[0]) == int(target_h) and \
            int(m2.shape[1]) == int(target_w):
        return m2
    m4 = m2.unsqueeze(0).unsqueeze(0)                 # [1, 1, H, W]
    pooled = F.adaptive_avg_pool2d(m4, (int(target_h), int(target_w)))
    return pooled.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def pool_masked_avg(
    feat: Tensor, mask_2d: Tensor, *, eps: float = 1e-6,
) -> Tensor:
    """Masked-average pool of feature `feat` over a 2-D `mask_2d`.

    feat:    [B, C, h, w] or [HW, B, C] (SAM2 memory layout) or [C, h, w]
    mask_2d: [h, w] in [0, 1]. Will be broadcast across B/C.

    Returns:
      For 4-D [B, C, h, w] input: [B, C]
      For 3-D [HW, B, C] input  : [B, C]   (HW = h*w; mask flattened)
      For 3-D [C, h, w] input   : [C]
    """
    if feat.dim() == 4:
        B, C, h, w = feat.shape
        if int(mask_2d.shape[0]) != int(h) or \
                int(mask_2d.shape[1]) != int(w):
            raise ValueError(
                f"pool_masked_avg: feat={tuple(feat.shape)} vs mask="
                f"{tuple(mask_2d.shape)}")
        m = mask_2d.to(feat.dtype).to(feat.device)
        m4 = m.view(1, 1, h, w)
        weighted = (feat * m4).sum(dim=(-2, -1))      # [B, C]
        denom = m4.sum().clamp_min(eps)
        return weighted / denom
    if feat.dim() == 3 and feat.shape[0] != mask_2d.numel():
        # [C, h, w]
        C, h, w = feat.shape
        if int(mask_2d.shape[0]) != int(h) or \
                int(mask_2d.shape[1]) != int(w):
            raise ValueError(
                f"pool_masked_avg: feat={tuple(feat.shape)} vs mask="
                f"{tuple(mask_2d.shape)}")
        m = mask_2d.to(feat.dtype).to(feat.device)
        weighted = (feat * m.view(1, h, w)).sum(dim=(-2, -1))
        denom = m.sum().clamp_min(eps)
        return weighted / denom
    if feat.dim() == 3:
        # [HW, B, C] (SAM2 token layout). Flatten mask to [HW].
        HW, B, C = feat.shape
        if int(mask_2d.numel()) != int(HW):
            raise ValueError(
                f"pool_masked_avg HW-layout: HW={HW} but mask has "
                f"{mask_2d.numel()} elements")
        m = mask_2d.to(feat.dtype).to(feat.device).flatten()  # [HW]
        weighted = (feat * m.view(HW, 1, 1)).sum(dim=0)        # [B, C]
        denom = m.sum().clamp_min(eps)
        return weighted / denom
    raise ValueError(
        f"pool_masked_avg: unsupported feat shape {tuple(feat.shape)}")


# ---------------------------------------------------------------------------
# Cosine loss helpers
# ---------------------------------------------------------------------------


def cosine_distance(
    x: Tensor, y: Tensor, *, eps: float = 1e-8, dim: int = -1,
) -> Tensor:
    """1 - cos(x, y) along `dim`. Detaches neither side.

    Both inputs must broadcast. Returns scalar if x/y are 1-D, else a
    tensor of one fewer dimension than `x`.
    """
    x_n = F.normalize(x, p=2, dim=dim, eps=eps)
    y_n = F.normalize(y, p=2, dim=dim, eps=eps)
    cos = (x_n * y_n).sum(dim=dim)
    return 1.0 - cos


# ---------------------------------------------------------------------------
# Bridge-frame selection
# ---------------------------------------------------------------------------


def select_bridge_frames(
    W_attacked: Sequence[int], T_proc: int, bridge_length: int,
) -> Dict[int, List[int]]:
    """For each insert k (in W-sorted order), return the bridge frames
    B_k = {w_k+1, ..., w_k+L}, clipped to next insert and to T_proc-1.

    Returns: {k → list[int]} keyed by insert index in W_sorted order.
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


def build_bridge_to_insert_k(
    bridge_frames_by_k: Dict[int, List[int]],
) -> Dict[int, int]:
    """Flatten {k → [t...]} to {t → k}."""
    out: Dict[int, int] = {}
    for k, ts in bridge_frames_by_k.items():
        for t in ts:
            out[int(t)] = int(k)
    return out


# ---------------------------------------------------------------------------
# Decoy mask projection (per insert / per bridge frame)
# ---------------------------------------------------------------------------


def build_decoy_region_at_mem_res(
    decoy_mask_full: Tensor,
    target_h: int, target_w: int,
) -> Tensor:
    """Convenience wrapper: downsample a single decoy mask to memory res.

    Returns: [target_h, target_w] in [0, 1] of float.
    """
    return downsample_decoy_mask(decoy_mask_full, target_h, target_w)


# ---------------------------------------------------------------------------
# State-continuation loss (aggregate)
# ---------------------------------------------------------------------------


def state_continuation_loss(
    student_M_by_t: Dict[int, Tensor],
    student_p_by_t: Dict[int, Tensor],
    teacher_M_by_k: Dict[int, Tensor],
    teacher_p_by_k: Dict[int, Tensor],
    student_decoy_mask_by_t: Dict[int, Tensor],
    teacher_decoy_mask_by_k: Dict[int, Tensor],
    bridge_to_insert_k: Dict[int, int],
    *,
    lambda_M: float = 1.0,
    lambda_P: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Aggregate state-continuation loss over bridge frames.

    Mathematical formulation (codex Round 3, R3-fix per bridge mask):
        L_M = mean_{t in bridge}  1 - cos(Pool_{d̂_t}(M_t),
                                          Pool_{d̂_{w_{k(t)}}}(M̄_{k(t)}))
        L_P = mean_{t in bridge}  1 - cos(p_t, p̄_{k(t)})

    Where k(t) = bridge_to_insert_k[t] gives the parent insert of bridge
    frame t. Critically: the STUDENT pool uses the bridge frame's own
    decoy region mask (d̂_t), while the TEACHER pool uses the insert
    frame's mask (d̂_{w_k}). On moving clips the decoy region tracks
    the duplicate object, so per-bridge-frame masks are required for
    spatial correctness (codex Loop 3 R3 review fix, 2026-04-25).

    Args:
      student_M_by_t: {t → M_t tensor with grad}
      student_p_by_t: {t → p_t tensor with grad}
      teacher_M_by_k: {k → M̄_k tensor (no grad)}
      teacher_p_by_k: {k → p̄_k tensor (no grad)}
      student_decoy_mask_by_t: {t → 2-D mask at memory resolution} (bridge-keyed)
      teacher_decoy_mask_by_k: {k → 2-D mask at memory resolution} (insert-keyed)
      bridge_to_insert_k: {t → k}
      lambda_M, lambda_P: relative weights (combined into returned scalar
        L = lambda_M * L_M + lambda_P * L_P).

    Returns:
      (L_combined, log_dict) where log_dict carries per-component scalars
      for the falsification tripwire.
    """
    if len(bridge_to_insert_k) == 0:
        zero = torch.zeros((), device=next(iter(
            student_M_by_t.values())).device, requires_grad=False) \
            if student_M_by_t else torch.zeros(())
        return zero, {
            "L_M": 0.0, "L_P": 0.0, "n_bridge": 0,
            "mean_cos_M": 0.0, "mean_cos_P": 0.0,
        }

    pieces_M: List[Tensor] = []
    pieces_P: List[Tensor] = []
    cos_M_list: List[float] = []
    cos_P_list: List[float] = []

    for t, k in bridge_to_insert_k.items():
        if t not in student_M_by_t or t not in student_p_by_t:
            continue
        if k not in teacher_M_by_k or k not in teacher_p_by_k:
            continue
        if t not in student_decoy_mask_by_t:
            continue
        if k not in teacher_decoy_mask_by_k:
            continue
        M_t = student_M_by_t[t]
        p_t = student_p_by_t[t]
        M_bar = teacher_M_by_k[k]
        p_bar = teacher_p_by_k[k]
        student_mask = student_decoy_mask_by_t[t].to(M_t.device)
        teacher_mask = teacher_decoy_mask_by_k[k].to(M_t.device)

        # Maskmem: pool student over its frame's decoy region, teacher
        # over the insert frame's decoy region. Per-bridge masks fix
        # spatial mis-specification on moving clips.
        pooled_t = pool_masked_avg(M_t, student_mask)
        pooled_bar = pool_masked_avg(M_bar, teacher_mask).detach()
        # Flatten to a feature vector for cosine. After pool_masked_avg:
        #   4-D input → [B, C]; 3-D HW input → [B, C]; 3-D [C, h, w] → [C].
        v_t = pooled_t.flatten()
        v_bar = pooled_bar.flatten()
        L_M_t = cosine_distance(v_t, v_bar)
        pieces_M.append(L_M_t)
        cos_M_list.append(float((1.0 - L_M_t).detach().item()))

        # obj_ptr: 1 - cos(p_t, p̄_k). Flatten in case of [B, D] shape.
        v_pt = p_t.flatten()
        v_pbar = p_bar.detach().flatten()
        L_P_t = cosine_distance(v_pt, v_pbar)
        pieces_P.append(L_P_t)
        cos_P_list.append(float((1.0 - L_P_t).detach().item()))

    if not pieces_M:
        zero = torch.zeros((), device=next(iter(
            student_M_by_t.values())).device, requires_grad=False) \
            if student_M_by_t else torch.zeros(())
        return zero, {
            "L_M": 0.0, "L_P": 0.0, "n_bridge": 0,
            "mean_cos_M": 0.0, "mean_cos_P": 0.0,
        }

    L_M = torch.stack(pieces_M).mean()
    L_P = torch.stack(pieces_P).mean()
    L = lambda_M * L_M + lambda_P * L_P

    log = {
        "L_M": float(L_M.detach().item()),
        "L_P": float(L_P.detach().item()),
        "n_bridge": len(pieces_M),
        "mean_cos_M": float(sum(cos_M_list) / max(1, len(cos_M_list))),
        "mean_cos_P": float(sum(cos_P_list) / max(1, len(cos_P_list))),
    }
    return L, log


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_pool_masked_avg() -> None:
    feat_4d = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ])  # [1, 2, 2, 2]
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    out = pool_masked_avg(feat_4d, mask)
    expected = torch.tensor([[1.0, 5.0]])  # only top-left pixel kept
    assert torch.allclose(out, expected), f"4D pool failed: {out}"

    mask_uniform = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    out = pool_masked_avg(feat_4d, mask_uniform)
    expected = torch.tensor([[2.5, 6.5]])  # plain mean
    assert torch.allclose(out, expected), f"uniform-mask pool failed: {out}"

    feat_hw = torch.tensor([
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [7.0], [8.0]],
    ])  # [HW=4, B=2, C=1]  — wait no, this needs [HW, B, C]
    # Build [HW=4, B=1, C=2] explicit
    feat_hw = torch.zeros(4, 1, 2)
    feat_hw[0, 0] = torch.tensor([1.0, 5.0])
    feat_hw[1, 0] = torch.tensor([2.0, 6.0])
    feat_hw[2, 0] = torch.tensor([3.0, 7.0])
    feat_hw[3, 0] = torch.tensor([4.0, 8.0])
    out = pool_masked_avg(feat_hw, mask)
    expected = torch.tensor([[1.0, 5.0]])
    assert torch.allclose(out, expected), f"HW-layout pool failed: {out}"
    print("  pool_masked_avg OK")


def _test_downsample_decoy_mask() -> None:
    m = torch.zeros(8, 8)
    m[2:6, 2:6] = 1.0
    out = downsample_decoy_mask(m, 4, 4)
    assert out.shape == (4, 4)
    # Center 2x2 of the 4x4 should be exactly 1.0; corners should be 0
    # or partial. With adaptive_avg_pool2d the center-most cells are full
    # and edges are partial.
    # Simpler check: total mass should be (4*4)*0.25*16/(4*4) = 4? Actually
    # avg-pool over 8x8 → 4x4 means each output cell averages a 2x2 patch.
    # Patch coverage of [2:6, 2:6] varies; sum of all output pixels equals
    # sum(input) / (8*8/4*4) = 16/4 = 4.
    assert torch.allclose(out.sum(), torch.tensor(4.0)), \
        f"downsample sum mismatch: {out.sum()}"
    print("  downsample_decoy_mask OK")


def _test_select_bridge_frames() -> None:
    out = select_bridge_frames(W_attacked=[10, 20, 30], T_proc=40,
                               bridge_length=3)
    assert out == {0: [11, 12, 13], 1: [21, 22, 23], 2: [31, 32, 33]}, out
    out = select_bridge_frames(W_attacked=[10, 12], T_proc=40,
                               bridge_length=5)
    # k=0: w=10, next_w=12, end = min(10+5, 12-1, 39) = 11; range = [11]
    assert out == {0: [11], 1: [13, 14, 15, 16, 17]}, out
    out = select_bridge_frames(W_attacked=[37], T_proc=40, bridge_length=5)
    # k=0: w=37, next_w=40, end = min(42, 39, 39) = 39; range = [38, 39]
    assert out == {0: [38, 39]}, out
    print("  select_bridge_frames OK")


def _test_state_continuation_loss() -> None:
    # Identity case: student == teacher → L = 0
    M = torch.randn(1, 4, 4, 4, requires_grad=True)
    p = torch.randn(1, 8, requires_grad=True)
    M_bar = M.detach().clone()
    p_bar = p.detach().clone()
    mask = torch.ones(4, 4)
    student_M = {1: M}
    student_p = {1: p}
    teacher_M = {0: M_bar}
    teacher_p = {0: p_bar}
    student_mask_by_t = {1: mask}
    teacher_mask_by_k = {0: mask}
    bridge = {1: 0}
    L, log = state_continuation_loss(
        student_M, student_p, teacher_M, teacher_p,
        student_mask_by_t, teacher_mask_by_k, bridge)
    assert abs(float(L.item())) < 1e-5, f"identity L != 0: {L}"
    assert abs(log["mean_cos_M"] - 1.0) < 1e-5, log
    assert abs(log["mean_cos_P"] - 1.0) < 1e-5, log

    # Orthogonal case: student in (1,0,...,0), teacher in (0,1,0,...,0) → cos = 0 → L = 1
    M2 = torch.zeros(1, 4, 4, 4, requires_grad=True)
    M2.data[0, 0, 0, 0] = 1.0
    p2 = torch.zeros(1, 8, requires_grad=True)
    p2.data[0, 0] = 1.0
    M2_bar = torch.zeros(1, 4, 4, 4)
    M2_bar[0, 1, 0, 0] = 1.0
    p2_bar = torch.zeros(1, 8)
    p2_bar[0, 1] = 1.0
    mask2 = torch.zeros(4, 4)
    mask2[0, 0] = 1.0
    L, log = state_continuation_loss(
        {1: M2}, {1: p2}, {0: M2_bar}, {0: p2_bar},
        {1: mask2}, {0: mask2}, {1: 0})
    assert abs(float(L.item()) - 2.0) < 1e-5, f"orthogonal L != 2: {L}"

    # Gradient flow check
    M3 = torch.randn(1, 4, 4, 4, requires_grad=True)
    p3 = torch.randn(1, 8, requires_grad=True)
    M3_bar = torch.randn(1, 4, 4, 4)
    p3_bar = torch.randn(1, 8)
    mask3_t = torch.rand(4, 4)
    mask3_k = torch.rand(4, 4)
    L, _ = state_continuation_loss(
        {1: M3}, {1: p3}, {0: M3_bar}, {0: p3_bar},
        {1: mask3_t}, {0: mask3_k}, {1: 0})
    L.backward()
    assert M3.grad is not None and M3.grad.abs().sum() > 0
    assert p3.grad is not None and p3.grad.abs().sum() > 0

    # Different masks: student vs teacher use different decoy regions.
    M4 = torch.zeros(1, 4, 4, 4, requires_grad=True)
    M4.data[0, 0, 0, 0] = 1.0    # student concentrated at (0,0)
    p4 = torch.zeros(1, 8, requires_grad=True)
    p4.data[0, 0] = 1.0
    M4_bar = torch.zeros(1, 4, 4, 4)
    M4_bar[0, 0, 1, 1] = 1.0     # teacher concentrated at (1,1)
    p4_bar = torch.zeros(1, 8)
    p4_bar[0, 0] = 1.0           # same p as student
    mask4_student = torch.zeros(4, 4); mask4_student[0, 0] = 1.0
    mask4_teacher = torch.zeros(4, 4); mask4_teacher[1, 1] = 1.0
    L, log4 = state_continuation_loss(
        {1: M4}, {1: p4}, {0: M4_bar}, {0: p4_bar},
        {1: mask4_student}, {0: mask4_teacher}, {1: 0})
    # student pool at (0,0) → channel 0 = 1; teacher pool at (1,1) → channel 0 = 1
    # → cos(M) = 1 → L_M = 0; cos(p) = 1 → L_P = 0.
    assert abs(float(L.item())) < 1e-5, \
        f"different-mask aligned-channels L != 0: {L}"
    print(f"  different-mask test: L={float(L.item()):.6f} cos_M={log4['mean_cos_M']:.4f}")

    # Empty bridge case
    L_empty, log_empty = state_continuation_loss(
        {}, {}, {0: M3_bar}, {0: p3_bar}, {}, {0: mask3_k}, {})
    assert float(L_empty.item()) == 0.0, log_empty
    assert log_empty["n_bridge"] == 0
    print("  state_continuation_loss OK")


def _test_build_bridge_to_insert_k() -> None:
    bf = {0: [11, 12, 13], 1: [21, 22], 2: [31]}
    flat = build_bridge_to_insert_k(bf)
    assert flat == {11: 0, 12: 0, 13: 0, 21: 1, 22: 1, 31: 2}, flat
    print("  build_bridge_to_insert_k OK")


if __name__ == "__main__":
    print("memshield.state_continuation self-tests:")
    _test_pool_masked_avg()
    _test_downsample_decoy_mask()
    _test_select_bridge_frames()
    _test_build_bridge_to_insert_k()
    _test_state_continuation_loss()
    print("memshield.state_continuation: all self-tests PASSED")
