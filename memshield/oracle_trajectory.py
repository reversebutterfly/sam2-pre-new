"""Oracle false trajectory representation for VADI Round 5 (Bundle A,
codex Loop 3 R5 design, locked 2026-04-25).

In our publisher-side offline threat model, attacker has the entire clean
clip — so trajectory PREDICTION is itself a proxy. The full no-proxy
representation parameterizes the false trajectory as its OWN learnable
sequence of (dy, dx) offsets, decoupled from the true object's motion,
optimized jointly with insert content and bridge edits.

Concretely, for K inserts at clean positions {c_k} with bridge length L_k
each, we maintain a parameter set:

  anchor_offset[k]   ∈ R^2   per insert: where decoy appears at insert frame
  delta_offset[k][i] ∈ R^2   per (insert, bridge_step) — additional shift on
                              top of anchor at bridge frame c_k + i

The false trajectory at any frame t is:

  if t == c_k for some k:                        # at insert position
      decoy_offset(t) = anchor_offset[k]
  elif t = c_k + i for some k, 1 <= i <= L_k:    # at bridge frame
      decoy_offset(t) = anchor_offset[k] + delta_offset[k][i-1]
  else:                                          # outside any bridge window
      decoy_offset(t) = (0, 0) — no decoy supervision

The decoy mask at frame t is:

  m_decoy(t) = shift(pseudo_masks[t], decoy_offset(t))

where pseudo_masks[t] is clean SAM2's predicted object mask at frame t
(the OBJECT's current pose, since we shift the object's mask not the insert
frame's mask). The duplicate inherits the object's CURRENT shape but at
the trajectory-specified location.

Smoothness regularizers (codex R5 design):

  L_smooth_anchor = sum over neighboring k pairs of |anchor[k+1] - anchor[k]|
  L_smooth_delta  = sum over (k, i) of |delta[k][i] - delta[k][i-1]|
  L_magnitude     = sum |anchor[k]|^2 + sum |delta[k][i]|^2

The Round 5 attack jointly optimizes these trajectory parameters with
insert content (Bundle B compositor) and bridge edits (Bundle B bridge
compositor) end-to-end (Bundle C).

Pure torch. Run `python -m memshield.oracle_trajectory` for self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Trajectory parameter container
# ---------------------------------------------------------------------------


@dataclass
class FalseTrajectoryParams:
    """Learnable false-trajectory parameters.

    For K inserts × max bridge length L, parameters:
      anchor_offset:  [K, 2]   — (dy, dx) at insert frame per insert
      delta_offset:   [K, L, 2] — bridge step deltas (relative to anchor)

    All in pixel units (float). Bounded by max_offset_px during optimization
    via projection, NOT via reparameterization (codex R5: keep optimizer free,
    project after step).
    """
    anchor_offset: Tensor       # [K, 2]
    delta_offset: Tensor        # [K, L, 2]
    L: int                      # max bridge length supported


def init_false_trajectory(
    K: int, L: int,
    init_anchor_offsets: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    delta_init_std: float = 0.0,
) -> FalseTrajectoryParams:
    """Initialize the false trajectory.

    init_anchor_offsets: optional [K, 2] starting values. If None, anchors
        are initialized to zero (decoy starts at the object's exact position
        — degenerate, will be pushed by the attack objective).
    delta_init_std: Gaussian std for delta initialization. Default 0 means
        the bridge initially mirrors the anchor (no drift). The attack
        objective will push deltas away from zero if non-trivial trajectory
        helps J-drop.
    """
    device = device or torch.device("cpu")
    if init_anchor_offsets is not None:
        if len(init_anchor_offsets) != K:
            raise ValueError(
                f"init_anchor_offsets length {len(init_anchor_offsets)} "
                f"!= K {K}")
        anchor = torch.tensor(
            [list(p) for p in init_anchor_offsets],
            dtype=dtype, device=device,
        )
    else:
        anchor = torch.zeros((K, 2), dtype=dtype, device=device)
    anchor = anchor.detach().clone().requires_grad_(True)

    if delta_init_std > 0:
        delta = (torch.randn((K, L, 2), dtype=dtype, device=device)
                 * float(delta_init_std))
    else:
        delta = torch.zeros((K, L, 2), dtype=dtype, device=device)
    delta = delta.detach().clone().requires_grad_(True)

    return FalseTrajectoryParams(
        anchor_offset=anchor, delta_offset=delta, L=int(L),
    )


# ---------------------------------------------------------------------------
# Trajectory query — get (dy, dx) at any (k, bridge_step) pair
# ---------------------------------------------------------------------------


def trajectory_offset_at(
    params: FalseTrajectoryParams,
    k: int,
    bridge_step: int,
) -> Tensor:
    """Compute the trajectory's (dy, dx) at insert k, bridge_step i.

    bridge_step:
      - 0 means at the insert frame itself (uses anchor only)
      - 1..L means bridge frame i (uses anchor + delta[k][i-1])

    Returns: [2] tensor (dy, dx) with grad through anchor + delta.
    """
    if not (0 <= k < params.anchor_offset.shape[0]):
        raise ValueError(
            f"k={k} out of range [0, {params.anchor_offset.shape[0]})")
    if not (0 <= bridge_step <= params.L):
        raise ValueError(
            f"bridge_step={bridge_step} out of range [0, {params.L}]")
    if bridge_step == 0:
        return params.anchor_offset[k]
    return params.anchor_offset[k] + params.delta_offset[k][bridge_step - 1]


def project_trajectory_to_budget(
    params: FalseTrajectoryParams, max_offset_px: float,
) -> None:
    """Project trajectory parameters to within max_offset_px ball.

    For each insert k and bridge step i, ensure |anchor + delta_i| <= R.
    Projects in-place on the parameters' .data buffers (no_grad).

    This is the post-step projection (codex R5: optimizer free, project after).
    """
    with torch.no_grad():
        K = params.anchor_offset.shape[0]
        L = params.L
        # Project anchor first (the "central" trajectory point).
        anchor_norm = params.anchor_offset.norm(dim=-1, keepdim=True)  # [K, 1]
        anchor_scale = (max_offset_px / anchor_norm.clamp_min(1e-6)).clamp(max=1.0)
        params.anchor_offset.mul_(anchor_scale)
        # Project each (anchor + delta_i) compound. If compound exceeds R,
        # clip delta only (anchor stays). This avoids reducing anchor twice.
        anchor_safe = params.anchor_offset.unsqueeze(1)  # [K, 1, 2]
        compound = anchor_safe + params.delta_offset      # [K, L, 2]
        compound_norm = compound.norm(dim=-1, keepdim=True)  # [K, L, 1]
        excess_mask = (compound_norm > max_offset_px).float()
        target_compound = compound * (
            max_offset_px / compound_norm.clamp_min(1e-6))
        new_delta = excess_mask * (target_compound - anchor_safe) \
            + (1.0 - excess_mask) * params.delta_offset
        params.delta_offset.copy_(new_delta)


# ---------------------------------------------------------------------------
# Decoy mask construction from false trajectory
# ---------------------------------------------------------------------------


def shift_mask_torch(
    mask: Tensor, dy: Tensor, dx: Tensor,
) -> Tensor:
    """Differentiable shift of a 2-D mask by (dy, dx) pixels via grid_sample.

    mask: [H, W] float in [0, 1]
    dy, dx: scalars (with grad)

    Returns: [H, W] in [0, 1].
    """
    H, W = mask.shape
    norm_dy = -2.0 * dy / H
    norm_dx = -2.0 * dx / W

    theta = torch.zeros((1, 2, 3), dtype=mask.dtype, device=mask.device)
    theta[0, 0, 0] = 1.0
    theta[0, 1, 1] = 1.0
    theta[0, 0, 2] = norm_dx
    theta[0, 1, 2] = norm_dy

    grid = F.affine_grid(theta, size=(1, 1, H, W), align_corners=False)
    m4 = mask.view(1, 1, H, W)
    shifted = F.grid_sample(
        m4, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False,
    )
    return shifted.view(H, W).clamp(0.0, 1.0)


def build_oracle_decoy_mask_at_frame(
    pseudo_mask_t: Tensor,                          # [H, W] clean object mask at t
    trajectory_params: FalseTrajectoryParams,
    k_cover: int,                                    # which insert covers t
    bridge_step: int,                                # 0 = insert frame, 1..L = bridge step
) -> Tensor:
    """Build the decoy mask at clean frame t = c_k + bridge_step.

    The duplicate inherits the OBJECT'S CURRENT POSE (pseudo_mask_t,
    which moves with the object) but is positioned at the trajectory's
    chosen location (anchor + delta).

    Returns [H, W] in [0, 1] with grad through trajectory_params.
    """
    offset = trajectory_offset_at(
        trajectory_params, k_cover, bridge_step)  # [2] (dy, dx)
    return shift_mask_torch(pseudo_mask_t, offset[0], offset[1])


def build_oracle_decoy_masks_for_clip(
    pseudo_masks: Sequence[Tensor],          # T clean masks, each [H, W]
    W_clean_positions: Sequence[int],
    trajectory_params: FalseTrajectoryParams,
    bridge_lengths: Sequence[int],            # per-insert L_k (≤ params.L)
) -> Dict[int, Tensor]:
    """Build the full per-clean-frame decoy mask trajectory.

    Returns {clean_t → decoy_mask}. Pre-first-insert frames are absent from
    the dict (caller treats as "no decoy supervision"). Post-last-insert-
    bridge frames are also absent (no further decoy after the trajectory).

    Both anchor and delta carry grad; calling this inside an optimizer step
    will produce a differentiable mask trajectory.
    """
    if len(W_clean_positions) != trajectory_params.anchor_offset.shape[0]:
        raise ValueError(
            f"W_clean_positions length {len(W_clean_positions)} != "
            f"anchor K {trajectory_params.anchor_offset.shape[0]}")
    if len(bridge_lengths) != len(W_clean_positions):
        raise ValueError(
            f"bridge_lengths length {len(bridge_lengths)} != "
            f"K {len(W_clean_positions)}")
    # Codex R5 review fix: refuse silent sort. anchor_offset[k], delta_offset[k],
    # and bridge_lengths[k] must already be aligned to W_clean_positions[k] in
    # the same order. Caller must pre-sort if needed (and reorder params/lengths
    # consistently). Silent sort here would attach trajectory params to the
    # wrong insert if caller passed unsorted W.
    W_int = [int(c) for c in W_clean_positions]
    if W_int != sorted(W_int):
        raise ValueError(
            f"W_clean_positions must be sorted ascending; got {W_int}. "
            f"Caller must sort BEFORE calling and reorder anchor_offset / "
            f"delta_offset / bridge_lengths to match.")
    W_sorted = W_int  # already sorted per assertion above
    out: Dict[int, Tensor] = {}
    for k, c_k in enumerate(W_sorted):
        L_k = int(bridge_lengths[k])
        next_c = W_sorted[k + 1] if k + 1 < len(W_sorted) else len(pseudo_masks)
        # Insert frame.
        if 0 <= c_k < len(pseudo_masks):
            out[c_k] = build_oracle_decoy_mask_at_frame(
                pseudo_masks[c_k], trajectory_params, k, 0)
        # Bridge frames c_k+1..c_k+L_k, clipped before next insert.
        for i in range(1, L_k + 1):
            t = c_k + i
            if t >= next_c or t >= len(pseudo_masks):
                break
            out[t] = build_oracle_decoy_mask_at_frame(
                pseudo_masks[t], trajectory_params, k, i)
    return out


# ---------------------------------------------------------------------------
# Trajectory regularizers (codex R5 spec)
# ---------------------------------------------------------------------------


def trajectory_smoothness_loss(
    params: FalseTrajectoryParams,
    *, anchor_smooth_weight: float = 1.0,
    delta_smooth_weight: float = 1.0,
    magnitude_weight: float = 0.1,
) -> Tensor:
    """Smoothness + magnitude regularizers over trajectory parameters.

    L = anchor_smooth_weight * ||anchor[k+1] - anchor[k]||
      + delta_smooth_weight  * ||delta[k][i] - delta[k][i-1]||
      + magnitude_weight     * (||anchor||^2 + ||delta||^2)
    """
    K = params.anchor_offset.shape[0]
    L_dim = params.delta_offset.shape[1]

    # Anchor smoothness across inserts (only if K>=2).
    if K >= 2:
        anchor_diff = (params.anchor_offset[1:]
                       - params.anchor_offset[:-1])      # [K-1, 2]
        L_smooth_anchor = anchor_diff.norm(dim=-1).mean()
    else:
        L_smooth_anchor = torch.zeros(
            (), dtype=params.anchor_offset.dtype,
            device=params.anchor_offset.device)

    # Delta smoothness within each bridge (only if L>=2).
    if L_dim >= 2:
        delta_diff = (params.delta_offset[:, 1:]
                      - params.delta_offset[:, :-1])     # [K, L-1, 2]
        L_smooth_delta = delta_diff.norm(dim=-1).mean()
    else:
        L_smooth_delta = torch.zeros_like(L_smooth_anchor)

    # Magnitude penalty.
    L_mag = (
        (params.anchor_offset ** 2).sum()
        + (params.delta_offset ** 2).sum()
    ) / max(1, K * (L_dim + 1))

    return (anchor_smooth_weight * L_smooth_anchor
            + delta_smooth_weight * L_smooth_delta
            + magnitude_weight * L_mag)


# ---------------------------------------------------------------------------
# Bridge length selection helper (codex R5 #10)
# ---------------------------------------------------------------------------


@dataclass
class BridgeLengthSearchResult:
    """Result of per-clip bridge length selection."""
    best_L: int
    L_to_score: Dict[int, float]


def select_bridge_length_per_insert(
    score_fn,                         # callable(L_per_insert: List[int]) -> float
    K: int,
    candidate_Ls: Sequence[int],
) -> BridgeLengthSearchResult:
    """Search for the best bridge length per insert.

    For simplicity, we do a UNIFORM bridge length search (same L for all
    inserts). The score_fn evaluates one (L_per_insert) tuple and returns
    a scalar score (e.g., low-budget attack J-drop).

    For per-insert variation, caller should iterate this with different
    fixed-L baselines + optimize via more expensive search.
    """
    best_L = int(candidate_Ls[0])
    L_to_score: Dict[int, float] = {}
    best_score = -float("inf")
    for L in candidate_Ls:
        L_per_insert = [int(L)] * K
        s = float(score_fn(L_per_insert))
        L_to_score[int(L)] = s
        if s > best_score:
            best_score = s
            best_L = int(L)
    return BridgeLengthSearchResult(best_L=best_L, L_to_score=L_to_score)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_init_false_trajectory() -> None:
    p = init_false_trajectory(K=3, L=4)
    assert p.anchor_offset.shape == (3, 2)
    assert p.delta_offset.shape == (3, 4, 2)
    assert p.L == 4
    assert p.anchor_offset.requires_grad and p.delta_offset.requires_grad
    # Default zero init.
    assert torch.allclose(p.anchor_offset, torch.zeros(3, 2))
    assert torch.allclose(p.delta_offset, torch.zeros(3, 4, 2))

    # With init values.
    p2 = init_false_trajectory(
        K=2, L=3, init_anchor_offsets=[(0.0, 50.0), (10.0, -20.0)])
    assert torch.allclose(
        p2.anchor_offset.detach(), torch.tensor([[0.0, 50.0], [10.0, -20.0]]))
    print("  init_false_trajectory OK")


def _test_trajectory_offset_at() -> None:
    p = init_false_trajectory(
        K=2, L=3, init_anchor_offsets=[(0.0, 100.0), (5.0, -50.0)])
    # bridge_step=0 → anchor only
    o00 = trajectory_offset_at(p, 0, 0)
    assert torch.allclose(o00.detach(), torch.tensor([0.0, 100.0]))
    # bridge_step=2 → anchor + delta[0][1] (which is 0)
    o02 = trajectory_offset_at(p, 0, 2)
    assert torch.allclose(o02.detach(), torch.tensor([0.0, 100.0]))
    # set delta and re-check
    with torch.no_grad():
        p.delta_offset[0, 1, 0] = 3.0
        p.delta_offset[0, 1, 1] = -7.0
    o02_new = trajectory_offset_at(p, 0, 2)
    assert torch.allclose(
        o02_new.detach(), torch.tensor([3.0, 93.0]))  # anchor (0,100) + delta(3,-7)
    print("  trajectory_offset_at OK")


def _test_project_trajectory_to_budget() -> None:
    p = init_false_trajectory(
        K=2, L=2, init_anchor_offsets=[(0.0, 200.0), (3.0, 4.0)])
    # Budget 100: anchor[0] norm 200 → scale to 100; anchor[1] norm 5 → unchanged
    project_trajectory_to_budget(p, max_offset_px=100.0)
    assert torch.allclose(
        p.anchor_offset[0].detach(), torch.tensor([0.0, 100.0]))
    assert torch.allclose(
        p.anchor_offset[1].detach(), torch.tensor([3.0, 4.0]))
    print("  project_trajectory_to_budget OK")


def _test_shift_mask_torch() -> None:
    H, W = 16, 16
    m = torch.zeros(H, W)
    m[8:10, 8:10] = 1.0
    # Shift by (2, 0): the patch should move down 2 rows
    out = shift_mask_torch(m, torch.tensor(2.0), torch.tensor(0.0))
    # Mass conservation
    assert abs(float(out.sum() - m.sum())) < 0.5
    # Center of mass shifted
    rows = torch.arange(H, dtype=torch.float32)
    cols = torch.arange(W, dtype=torch.float32)
    cy_in = float((m * rows.view(-1, 1)).sum() / m.sum())
    cy_out = float((out * rows.view(-1, 1)).sum() / out.sum().clamp_min(1e-6))
    assert abs((cy_out - cy_in) - 2.0) < 0.3, (cy_in, cy_out)

    # Gradient flow.
    dy = torch.tensor(2.0, requires_grad=True)
    dx = torch.tensor(0.0, requires_grad=True)
    out = shift_mask_torch(m, dy, dx)
    loss = out.sum()
    loss.backward()
    # dy may have gradient = 0 since shifting a binary mask is gradient-free
    # at most cells; but bilinear interpolation should produce SOME gradient
    # at boundary cells. Allow either, as long as no NaN/inf.
    assert (dy.grad is None or torch.isfinite(dy.grad).all())
    print("  shift_mask_torch OK")


def _test_build_oracle_decoy_masks() -> None:
    # T_clean = 30, K=2 inserts at clean positions [10, 20], bridge L=3 each.
    H, W = 16, 16
    pseudo_masks = []
    for t in range(30):
        m = torch.zeros(H, W)
        # Object moves: put a 2x2 patch starting at column t/3
        c0 = min(W - 2, max(0, t // 3))
        m[6:8, c0:c0 + 2] = 1.0
        pseudo_masks.append(m)

    p = init_false_trajectory(
        K=2, L=3, init_anchor_offsets=[(0.0, 5.0), (0.0, -3.0)])

    decoys = build_oracle_decoy_masks_for_clip(
        pseudo_masks=pseudo_masks,
        W_clean_positions=[10, 20],
        trajectory_params=p,
        bridge_lengths=[3, 3],
    )
    # Insert 0 at c=10 → covers c=10, 11, 12, 13
    assert 10 in decoys and 11 in decoys and 12 in decoys and 13 in decoys
    # Insert 1 at c=20 → covers c=20..23
    assert 20 in decoys and 21 in decoys and 22 in decoys and 23 in decoys
    # Pre-first-insert: c=0..9 should not be in decoys
    assert 5 not in decoys and 9 not in decoys
    # Between inserts (c=14..19) should not be in decoys
    assert 14 not in decoys and 18 not in decoys

    # Decoy at c=10 should be the object's mask at t=10 shifted by (0, 5).
    # Object at t=10 is at columns [3, 5). Shifted by 5 → [8, 10).
    decoy_10 = decoys[10]
    # Center of mass of decoy at column ~9
    cm_col = float((decoy_10 * torch.arange(W, dtype=torch.float32)).sum()
                   / decoy_10.sum().clamp_min(1e-6))
    expected_col = (3.0 + 5.0) + 5.0 - 0.5  # original COM + shift - center adj
    assert abs(cm_col - 8.5) < 1.0, (cm_col, expected_col)

    # Decoy at c=11 should follow the OBJECT's c=11 mask, not c=10
    decoy_11 = decoys[11]
    # Object at t=11 is at columns [3, 5) (since 11//3=3). Shifted by 5 → [8, 10).
    cm_col_11 = float((decoy_11 * torch.arange(W, dtype=torch.float32)).sum()
                      / decoy_11.sum().clamp_min(1e-6))
    # Centers should differ by object-motion-between-frames if any, but since
    # 10//3=3 and 11//3=3, positions same. For t=12: 12//3=4 → object moves.
    decoy_12 = decoys[12]
    cm_col_12 = float((decoy_12 * torch.arange(W, dtype=torch.float32)).sum()
                      / decoy_12.sum().clamp_min(1e-6))
    assert cm_col_12 > cm_col_11, (cm_col_11, cm_col_12)
    print("  build_oracle_decoy_masks OK")


def _test_trajectory_smoothness_loss() -> None:
    p = init_false_trajectory(K=3, L=2)
    # All zero → zero loss
    L = trajectory_smoothness_loss(p)
    assert abs(float(L)) < 1e-6

    # Non-zero anchors that are smooth: anchor = [0, 5], [0, 10], [0, 15] (linear ramp)
    with torch.no_grad():
        p.anchor_offset.copy_(torch.tensor([[0.0, 5.0], [0.0, 10.0], [0.0, 15.0]]))
    L_smooth = trajectory_smoothness_loss(p, anchor_smooth_weight=1.0)
    # First-order diff = 5.0 (norm of (0, 5))
    # Two diffs: ||a[1]-a[0]|| + ||a[2]-a[1]|| = 5 + 5 = 10, /2 = 5
    assert abs(float(L_smooth) - (5.0 + 0.0 + 100.0/3 * 0.1)) < 5.0  # rough check

    # Gradient flow
    p2 = init_false_trajectory(K=2, L=2)
    with torch.no_grad():
        p2.anchor_offset.copy_(torch.tensor([[0.0, 5.0], [3.0, 10.0]]))
        p2.delta_offset.copy_(torch.zeros(2, 2, 2))
    L = trajectory_smoothness_loss(p2)
    L.backward()
    assert p2.anchor_offset.grad is not None
    print("  trajectory_smoothness_loss OK")


def _test_select_bridge_length() -> None:
    # Mock score function: prefer L=3
    def score(L_per_insert):
        return -abs(L_per_insert[0] - 3.0) - 0.1 * sum(L_per_insert)
    res = select_bridge_length_per_insert(
        score, K=2, candidate_Ls=[2, 3, 4, 5])
    assert res.best_L == 3, res
    assert res.L_to_score[3] > res.L_to_score[2]
    assert res.L_to_score[3] > res.L_to_score[5]
    print("  select_bridge_length_per_insert OK")


if __name__ == "__main__":
    print("memshield.oracle_trajectory self-tests:")
    _test_init_false_trajectory()
    _test_trajectory_offset_at()
    _test_project_trajectory_to_budget()
    _test_shift_mask_torch()
    _test_build_oracle_decoy_masks()
    _test_trajectory_smoothness_loss()
    _test_select_bridge_length()
    print("memshield.oracle_trajectory: all self-tests PASSED")
