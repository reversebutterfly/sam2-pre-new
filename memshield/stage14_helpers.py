"""Stage 14 forward-loss + W-dependent state helpers.

Extracted from scripts/run_vadi_v5.py:_run_oracle_trajectory_pgd to enable:

* Joint placement-perturbation curriculum search (memshield.joint_placement_-
  search), which rebuilds attack state per W candidate while sharing
  perturbation parameters (traj / edit_params / R / nu) across schedules.
* The existing fixed-W path keeps bit-equivalent semantics by packing its
  pre-computed args into AttackState via `assemble_attack_state` and
  routing through `stage14_forward_loss`.

Auto-review-loop Round 6 R3 GO design (codex thread 019dc51a-c71a-7971-bece-
116a592de2f5).

Pure-torch state builders (no SAM2 forward inside the builders themselves;
the SAM2 forward is invoked inside `stage14_forward_loss` via the caller-
supplied `forward_fn`).

Run `python -m memshield.stage14_helpers` for self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from memshield.decoy_continuation import (
    alpha_from_logits,
    apply_continuation_overlay,
    apply_translation_warp_roi,
    area_preservation_loss,
    alpha_regularizer,
    displacement_from_warp,
    positive_objectness_loss,
    select_bridge_frames,
    soften_decoy_mask,
    warp_regularizer,
)
from memshield.decoy_seed import (
    build_decoy_insert_seeds,
    build_duplicate_object_decoy_frame,
    compute_decoy_offset_from_mask,
    shift_mask_np,
)
from memshield.oracle_trajectory import (
    FalseTrajectoryParams,
    build_oracle_decoy_masks_for_clip,
    trajectory_offset_at,
    trajectory_smoothness_loss,
)
from memshield.semantic_compositor import (
    apply_masked_residual,
    compose_decoy_alpha_paste,
)
from memshield.vadi_loss import (
    aggregate_margin_loss,
    decoy_margin_per_frame,
    lpips_cap_hinge,
    tv_hinge,
)
from memshield.vadi_optimize import attacked_to_clean, build_processed


# ---------------------------------------------------------------------------
# AttackState: all W-dependent state for one Stage 14 forward
# ---------------------------------------------------------------------------


@dataclass
class AttackState:
    """All W-dependent state needed by stage14_forward_loss.

    Built by `build_attack_state_from_W` (rebuilt per schedule in joint
    search), or by `assemble_attack_state` (packs the existing fixed-W path's
    pre-computed args without recomputation).

    Note: trajectory / edit_params / R / nu are NOT in here -- those are
    perturbation parameters that stay shared across schedules in joint
    search.
    """

    # Index conventions
    W_clean_sorted: List[int]                           # sorted clean c_k
    W_attacked: List[int]                                # sorted attacked w_k
    K: int
    T_clean: int
    T_proc: int

    # Insert content
    decoy_seeds: Tensor                                  # [K, H, W, 3]
    decoy_offsets_init: List[Tuple[int, int]]            # init anchors

    # Bridge structure
    bridge_lengths_list: List[int]                       # per-insert L_k
    L_max: int
    bridge_frames_by_k: Dict[int, List[int]]             # k -> attacked-space t list
    bridge_t_list: List[Tuple[int, int, int]]            # (t_proc, k, l_idx)
    bridge_polish: List[int]                             # sorted union of t
    insert_polish: List[int]                             # sorted W_attacked
    margin_query_frames: List[int]                       # union(insert, bridge)

    # Mask supervision
    pseudo_masks_torch: List[Tensor]                     # T_clean x [H, W]
    m_true_by_t: Dict[int, Tensor]                       # t_proc -> [H, W]
    m_decoy_by_t: Dict[int, Tensor]


# ---------------------------------------------------------------------------
# Internal: clean -> processed-space mask remap (inlined from
# scripts.run_vadi.remap_masks_to_processed_space to avoid a script-level
# cross-import that risks circularity with run_vadi_v5.py)
# ---------------------------------------------------------------------------


def _remap_masks_to_processed_space(
    clean_masks: Sequence[np.ndarray], W: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Map clean-space {c -> mask} to processed-space {t -> mask}.

    Mirrors `scripts.run_vadi.remap_masks_to_processed_space` exactly:
    non-insert frames inherit the clean mask via `attacked_to_clean`; insert
    positions get the midframe average `0.5*clean[c_k-1] + 0.5*clean[c_k]`.
    Caller is responsible for any subsequent insert-position override.
    """
    T_clean = len(clean_masks)
    W_sorted = sorted(int(w) for w in W)
    K = len(W_sorted)
    T_proc = T_clean + K
    out: Dict[int, np.ndarray] = {}
    for t in range(T_proc):
        if t in W_sorted:
            k = W_sorted.index(t)
            c_k = W_sorted[k] - k
            if not (1 <= c_k < T_clean):
                raise ValueError(
                    f"insert k={k} at W={W_sorted[k]} -> c_k={c_k} out of "
                    f"range [1, {T_clean}).")
            out[t] = 0.5 * clean_masks[c_k - 1] + 0.5 * clean_masks[c_k]
        else:
            c = attacked_to_clean(t, W_sorted)
            out[t] = clean_masks[c]
    return out


def _build_supervision_masks(
    pseudo_masks_clean_np: List[np.ndarray],
    W_clean_sorted: List[int],
    decoy_offsets: Sequence[Tuple[int, int]],
    W_attacked: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Reproduce the m_true_by_t / m_decoy_by_t logic from run_v5_for_clip.

    Per-frame logic (clean-space):
      m_true_clean[t]  = pseudo_masks[t]
      m_decoy_clean[t] = shift(pseudo_masks[t], decoy_offset[k_cover])
        with k_cover = most recent insert k where c_k <= t (-1 = pre-insert).
        Pre-first-insert: zero mask.

    Then remap to processed-space and override insert positions:
      m_true_by_t[w_k]  = pseudo_masks[c_k]
      m_decoy_by_t[w_k] = shift(pseudo_masks[c_k], decoy_offsets[k])

    This replicates the logic in run_v5_for_clip lines ~2819-2868 exactly so
    that `build_attack_state_from_W` produces the same supervision masks
    that the legacy code path computes.
    """
    T_clean = len(pseudo_masks_clean_np)
    K = len(W_clean_sorted)
    if K != len(decoy_offsets):
        raise ValueError(
            f"len(decoy_offsets)={len(decoy_offsets)} != K={K}")

    m_true_clean: List[np.ndarray] = list(pseudo_masks_clean_np)
    m_decoy_clean: List[np.ndarray] = []
    for t in range(T_clean):
        k_cover = -1
        for k, c_k in enumerate(W_clean_sorted):
            if c_k <= t:
                k_cover = k
            else:
                break
        if k_cover == -1:
            m_decoy_clean.append(np.zeros_like(m_true_clean[t]))
        else:
            dy, dx = decoy_offsets[k_cover]
            m_decoy_clean.append(
                shift_mask_np(m_true_clean[t], int(dy), int(dx)))

    m_true_by_t_np = _remap_masks_to_processed_space(m_true_clean, W_attacked)
    m_decoy_by_t_np = _remap_masks_to_processed_space(
        m_decoy_clean, W_attacked)

    W_sorted = sorted(int(w) for w in W_attacked)
    for k, w in enumerate(W_sorted):
        c_k = int(W_clean_sorted[k])
        if not (0 <= c_k < T_clean):
            continue
        m_true_by_t_np[w] = m_true_clean[c_k]
        dy, dx = decoy_offsets[k]
        m_decoy_by_t_np[w] = shift_mask_np(
            m_true_clean[c_k], int(dy), int(dx))

    return m_true_by_t_np, m_decoy_by_t_np


# ---------------------------------------------------------------------------
# Public builder (joint search rebuilds per schedule)
# ---------------------------------------------------------------------------


def build_attack_state_from_W(
    W_clean: Sequence[int],
    x_clean: Tensor,
    pseudo_masks_clean: Sequence[Any],
    config: Any,
    *,
    bridge_length: Optional[int] = None,
    insert_base_mode: Optional[str] = None,
) -> AttackState:
    """Build all W-dependent Stage 14 state from W_clean + clip context.

    Args:
      W_clean: insert positions in clean-space. Will be sorted; entries must
        be unique and in [1, T_clean) (need neighbors for decoy seed).
      x_clean: [T_clean, H, W, 3] in [0, 1].
      pseudo_masks_clean: T_clean clean-SAM2 pseudo-masks (np or torch).
      config: VADIv5Config (or anything duck-typed with the same fields).
      bridge_length: per-insert bridge length L. None -> use
        config.oracle_traj_bridge_length.
      insert_base_mode: "duplicate_seed" | "midframe" | "poisson_hifi" |
        "propainter". None -> use config.insert_base_mode. The
        ghost-free modes (poisson_hifi / propainter) were added in the
        2026-04-28 codex round 5 wiring; they dispatch through
        memshield.decoy_seed.build_decoy_insert_seeds_via_strategy.
        (Joint search v1 only exercises "duplicate_seed".)

    Returns: an AttackState whose fields exactly match what
    `run_v5_for_clip` would have built for the same W_clean.
    """
    # Local import to keep memshield/* free of `scripts/*` imports.
    from memshield.vadi_optimize import build_base_inserts as _build_midframe

    # --- input validation + sort ---
    W_clean_sorted = sorted(int(c) for c in W_clean)
    K = len(W_clean_sorted)
    if K == 0:
        raise ValueError("W_clean must contain at least 1 insert position")
    T_clean = int(x_clean.shape[0])
    for c in W_clean_sorted:
        if not (1 <= c < T_clean):
            raise ValueError(
                f"W_clean entry {c} out of [1, {T_clean}); decoy seed "
                "needs x_clean[c] to have neighbors")
    if len(set(W_clean_sorted)) != K:
        raise ValueError(f"W_clean contains duplicates: {W_clean_sorted}")

    # --- attacked-space + length ---
    W_attacked = sorted([c + k for k, c in enumerate(W_clean_sorted)])
    T_proc = T_clean + K

    # --- decoy seeds + offsets (insert_base_mode dependent) ---
    mode = insert_base_mode or config.insert_base_mode
    if mode == "duplicate_seed":
        decoy_seeds, decoy_offsets = build_decoy_insert_seeds(
            x_clean, pseudo_masks_clean, W_clean_sorted,
            feather_radius=config.feather_radius,
            feather_sigma=config.feather_sigma,
        )
        decoy_seeds = decoy_seeds.to(x_clean.device)
        decoy_offsets = [(int(dy), int(dx)) for dy, dx in decoy_offsets]
    elif mode == "midframe":
        decoy_seeds = _build_midframe(x_clean, W_attacked)
        decoy_offsets = [
            tuple(int(v) for v in compute_decoy_offset_from_mask(
                np.asarray(pseudo_masks_clean[c], dtype=np.float32)))
            for c in W_clean_sorted
        ]
    elif mode in ("poisson_hifi", "propainter"):
        # Codex round 5 ghost-fix wiring (2026-04-28): dispatch through
        # the strategy wrapper to get ghost-free insert bases.
        from memshield.decoy_seed import build_decoy_insert_seeds_via_strategy
        decoy_seeds, decoy_offsets = build_decoy_insert_seeds_via_strategy(
            mode,
            x_clean, pseudo_masks_clean, W_clean_sorted,
        )
        decoy_seeds = decoy_seeds.to(x_clean.device)
        decoy_offsets = [(int(dy), int(dx)) for dy, dx in decoy_offsets]
    else:
        raise ValueError(f"unknown insert_base_mode {mode!r}")

    # --- bridge frames ---
    L = int(bridge_length if bridge_length is not None
            else config.oracle_traj_bridge_length)
    bridge_frames_by_k = select_bridge_frames(
        W_attacked, T_proc, bridge_length=L)
    bridge_lengths_list = [
        len(bridge_frames_by_k.get(k, [])) for k in range(K)]
    L_max = max(bridge_lengths_list) if bridge_lengths_list else L
    if L_max < 1:
        L_max = L

    bridge_t_list: List[Tuple[int, int, int]] = []
    for k, t_list in bridge_frames_by_k.items():
        for l_idx, t in enumerate(t_list):
            bridge_t_list.append((int(t), int(k), int(l_idx)))
    bridge_polish = sorted({t for t, _, _ in bridge_t_list})
    insert_polish = sorted(int(w) for w in W_attacked)
    margin_query_frames = sorted(set(insert_polish + bridge_polish))

    # --- pseudo_masks -> torch (clean-space; matches run_v5_for_clip
    # input handling for stage 14) ---
    pseudo_masks_torch: List[Tensor] = []
    for m in pseudo_masks_clean:
        if isinstance(m, np.ndarray):
            pseudo_masks_torch.append(torch.as_tensor(
                m, dtype=x_clean.dtype, device=x_clean.device))
        else:
            pseudo_masks_torch.append(
                m.to(x_clean.device).to(x_clean.dtype))

    # --- supervision masks (m_true_by_t, m_decoy_by_t) ---
    pseudo_masks_clean_np = [
        np.asarray(m, dtype=np.float32) for m in pseudo_masks_clean]
    m_true_by_t_np, m_decoy_by_t_np = _build_supervision_masks(
        pseudo_masks_clean_np, W_clean_sorted, decoy_offsets, W_attacked)
    device = x_clean.device
    m_true_by_t = {
        int(t): torch.from_numpy(m).float().to(device)
        for t, m in m_true_by_t_np.items()}
    m_decoy_by_t = {
        int(t): torch.from_numpy(m).float().to(device)
        for t, m in m_decoy_by_t_np.items()}

    return AttackState(
        W_clean_sorted=W_clean_sorted,
        W_attacked=list(W_attacked),
        K=K, T_clean=T_clean, T_proc=T_proc,
        decoy_seeds=decoy_seeds,
        decoy_offsets_init=list(decoy_offsets),
        bridge_lengths_list=bridge_lengths_list,
        L_max=int(L_max),
        bridge_frames_by_k={int(k): [int(t) for t in v]
                            for k, v in bridge_frames_by_k.items()},
        bridge_t_list=bridge_t_list,
        bridge_polish=bridge_polish,
        insert_polish=insert_polish,
        margin_query_frames=margin_query_frames,
        pseudo_masks_torch=pseudo_masks_torch,
        m_true_by_t=m_true_by_t,
        m_decoy_by_t=m_decoy_by_t,
    )


# ---------------------------------------------------------------------------
# Pack-only assembler for the legacy fixed-W path (no recompute)
# ---------------------------------------------------------------------------


def assemble_attack_state(
    *,
    W_clean_sorted: Sequence[int],
    W_attacked: Sequence[int],
    decoy_seeds: Tensor,
    decoy_offsets_init: Sequence[Tuple[int, int]],
    bridge_frames_by_k: Dict[int, List[int]],
    bridge_lengths: Sequence[int],
    pseudo_masks_clean: Sequence[Any],
    m_true_by_t: Dict[int, Tensor],
    m_decoy_by_t: Dict[int, Tensor],
    x_clean: Tensor,
) -> AttackState:
    """Pack pre-computed fixed-W args into an AttackState (no recompute).

    Used by the legacy `_run_oracle_trajectory_pgd` so that its byte-level
    behavior is unchanged: the existing run_v5_for_clip already builds
    `decoy_seeds`, `m_true_by_t`, `m_decoy_by_t`, etc. for use by stages
    11-13 too, so there is no benefit to recomputing them for stage 14.

    The fields packed here MUST match what `build_attack_state_from_W`
    would have produced for the same W (within numerical equivalence). The
    fixed-W parity test verifies this.
    """
    K = len(list(W_clean_sorted))
    T_clean = int(x_clean.shape[0])
    T_proc = T_clean + K

    bridge_lengths_list = [int(bl) for bl in bridge_lengths]
    L_max = max(bridge_lengths_list) if bridge_lengths_list else 0
    bridge_t_list: List[Tuple[int, int, int]] = []
    for k, t_list in bridge_frames_by_k.items():
        for l_idx, t in enumerate(t_list):
            bridge_t_list.append((int(t), int(k), int(l_idx)))
    bridge_polish = sorted({t for t, _, _ in bridge_t_list})
    insert_polish = sorted(int(w) for w in W_attacked)
    margin_query_frames = sorted(set(insert_polish + bridge_polish))

    pseudo_masks_torch: List[Tensor] = []
    for m in pseudo_masks_clean:
        if isinstance(m, np.ndarray):
            pseudo_masks_torch.append(torch.as_tensor(
                m, dtype=x_clean.dtype, device=x_clean.device))
        else:
            pseudo_masks_torch.append(
                m.to(x_clean.device).to(x_clean.dtype))

    return AttackState(
        W_clean_sorted=sorted(int(c) for c in W_clean_sorted),
        W_attacked=sorted(int(w) for w in W_attacked),
        K=K, T_clean=T_clean, T_proc=T_proc,
        decoy_seeds=decoy_seeds,
        decoy_offsets_init=[
            (int(dy), int(dx)) for dy, dx in decoy_offsets_init],
        bridge_lengths_list=bridge_lengths_list,
        L_max=int(L_max),
        bridge_frames_by_k={int(k): [int(t) for t in v]
                            for k, v in bridge_frames_by_k.items()},
        bridge_t_list=bridge_t_list,
        bridge_polish=bridge_polish,
        insert_polish=insert_polish,
        margin_query_frames=margin_query_frames,
        pseudo_masks_torch=pseudo_masks_torch,
        m_true_by_t={int(t): v for t, v in m_true_by_t.items()},
        m_decoy_by_t={int(t): v for t, v in m_decoy_by_t.items()},
    )


# ---------------------------------------------------------------------------
# Stage 14 forward + loss diagnostics
# ---------------------------------------------------------------------------


@dataclass
class Stage14LossDiagnostics:
    """All diagnostic values produced by one Stage 14 forward step."""

    # Loss components (scalar Tensors; .item() for log).
    L_margin: Tensor
    L_obj: Tensor
    L_area: Tensor
    L_fid_bridge: Tensor
    L_fid_ins: Tensor
    L_fid_TV: Tensor
    L_alpha: Tensor
    L_warp: Tensor
    L_traj: Tensor
    L_R_tv: Tensor
    L_suffix: Tensor    # v3 (codex Coverage-Constrained spec): soft-IoU
                         # of true-mask retention at sparse probe frames
                         # spanning the full attacked-suffix horizon.
                         # Zero tensor when suffix probes are not used
                         # (phase 1-2 of v3 curriculum, or v2 / v2.1
                         # backwards-compat paths).
    # v4.x (Anchored Stage 14, codex spec 2026-04-26 / 2026-04-27 v4.1):
    # no-regression vs A0 baseline (frozen-nu, no bridge polish) plus
    # explicit suffix gain. All zero when teacher=None (v3 / v2 backwards-
    # compat path).
    L_keep_margin: Tensor    # mean_t relu(margin_loss_cur - margin_loss_A0)
                              # (penalize when current margin loss exceeds
                              # A0 -- i.e. attack got weaker than baseline)
                              # over insert + bridge_polish frames.
    L_keep_full: Tensor      # v4.1 hot-fix (codex thread 019dc51a 2026-
                              # 04-27): MEAN over the DENSE non-insert
                              # suffix of relu(u_cur(t) - u_A0(t)).
                              # Previous v4.0 used `L_keep_suffix` over
                              # only 6 sparse probes (sum) — that surro-
                              # gate let optimization regress on un-
                              # monitored frames (dog/bmx-trees revert
                              # rate 50% on dev-4). v4.1 enforces no-
                              # regression on the FULL suffix.
    L_gain_suffix: Tensor    # mean_t u_cur(t) over a SPARSE 6-probe set
                              # (the explicit attack-improvement term;
                              # v4.1 changed sum -> mean for clip-length
                              # invariance).

    # Detached / floated diagnostic scalars.
    mean_decoy_overlap: float
    mean_true_overlap: float
    delta_overlap: float
    mean_obj_score: float
    wrong_but_present_count: int
    feasible: bool
    alpha_mean: float
    alpha_max_step: float
    warp_disp_max: float
    n_bridge: int
    n_suffix_probes: int


# ---------------------------------------------------------------------------
# Stage 14 v4: Anchored Stage 14 teacher signals (A0 baseline, frozen-nu,
# no bridge polish). Used as no-regression anchor in stage14_forward_loss.
# Codex thread 019dc51a-c71a-7971-bece-116a592de2f5 round 6 R3 GO design v4.
# ---------------------------------------------------------------------------


@dataclass
class Stage14TeacherSignals:
    """A0-baseline (no Stage 14 polish) per-frame signals for v4.x anchoring.

    Fields:
      margin_by_t: per-frame margin_loss tensor at each margin_query_frame.
        Detached. Used as no-regression anchor in L_keep_margin.
      suffix_iou_by_t: per-frame soft-IoU(p_t, m_true_t) under A0 at each
        frame in (keep_suffix_frames ∪ gain_suffix_frames). Detached. Used
        as no-regression anchor in L_keep_full and as A0 reference for
        diagnostic of L_gain_suffix.
      keep_suffix_frames: v4.1 DENSE no-regression set — all non-insert
        attacked-space frames in (w_first+1, T_proc). The fix to v4.0's
        sparse-probe failure mode (codex thread 019dc51a 2026-04-27).
      gain_suffix_frames: v4.1 SPARSE gain set — `n_probes` evenly-spaced
        non-insert frames (typically 6). Provides the directional improve-
        ment signal without distracting the optimizer with full-suffix
        improvement noise.
    """

    margin_by_t: Dict[int, Tensor]
    suffix_iou_by_t: Dict[int, Tensor]
    keep_suffix_frames: List[int]
    gain_suffix_frames: List[int]


def build_suffix_probe_frames(
    W_attacked: Sequence[int],
    T_proc: int,
    *,
    n_probes: int,
) -> List[int]:
    """Choose `n_probes` evenly-spaced attacked-space frame indices in
    [w_first+1, T_proc-1], EXCLUDING the inserts themselves (W_attacked).

    v4.1 NOTE: this builds the SPARSE GAIN probe set (formerly used for
    both keep + gain in v4.0). For the DENSE no-regression set, use
    `build_keep_suffix_frames`.

    Returns sorted attacked-space ints. Duplicates are removed; the function
    is robust to small T_proc / large n_probes (gracefully clipped to all
    available non-insert suffix frames).
    """
    if n_probes <= 0:
        return []
    W_set = set(int(w) for w in W_attacked)
    if not W_attacked:
        return []
    w_first = min(int(w) for w in W_attacked)
    candidates = [t for t in range(w_first + 1, T_proc) if t not in W_set]
    if not candidates:
        return []
    if len(candidates) <= n_probes:
        return sorted(candidates)
    if n_probes == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (n_probes - 1)
    selected = sorted({
        candidates[int(round(i * step))] for i in range(n_probes)
    })
    return selected


def build_keep_suffix_frames(
    W_attacked: Sequence[int],
    T_proc: int,
) -> List[int]:
    """v4.1: build the DENSE no-regression set — every non-insert attacked-
    space frame in [w_first+1, T_proc-1].

    Codex hot-fix 2026-04-27 (thread 019dc51a). v4.0 used only 6 sparse
    probes for the no-regression term, which let optimization regress on
    the unmonitored 67-77% of suffix frames (dog/bmx-trees dev-4 revert
    rate = 50%). v4.1 enforces no-regression on the FULL suffix.

    Returns sorted attacked-space ints, excluding W_attacked.
    """
    if not W_attacked:
        return []
    W_set = set(int(w) for w in W_attacked)
    w_first = min(int(w) for w in W_attacked)
    return [t for t in range(w_first + 1, T_proc) if t not in W_set]


def build_stage14_teacher_signals(
    state: "AttackState",
    *,
    x_clean: Tensor,
    nu_teacher: Tensor,
    forward_fn: Any,
    config: Any,
    keep_suffix_frames: Sequence[int],
    gain_suffix_frames: Sequence[int],
) -> Stage14TeacherSignals:
    """Compute A0 baseline teacher signals via no_grad SAM2 forward.

    A0 baseline = decoy_seeds + nu_teacher inserted at W_attacked, NO bridge
    polish (no alpha_paste, warp, R, traj-driven content edits). This is the
    "K3 insert-only" reference whose mean J-drop ≈ 0.537 set the v4 no-
    regression target.

    v4.1: takes BOTH keep_suffix_frames (dense) and gain_suffix_frames
    (sparse). The teacher forward computes IoU on the union; the loss
    splits the IoU dict by the two frame lists.

    Returns Stage14TeacherSignals with detached per-frame margin tensors
    and detached per-frame suffix IoU tensors over the union.
    """
    device = x_clean.device
    with torch.no_grad():
        inserts_t = (state.decoy_seeds + nu_teacher).clamp(0.0, 1.0)
        x_full_t = x_clean
        if config.train_ste_quantize:
            from memshield.losses import fake_uint8_quantize
            x_full_t = fake_uint8_quantize(x_full_t)
            inserts_t = fake_uint8_quantize(inserts_t)
        processed_t = build_processed(
            x_full_t, inserts_t, state.W_attacked)

        suffix_union = sorted(set(
            int(t) for t in list(keep_suffix_frames) + list(gain_suffix_frames)
            if 0 <= int(t) < state.T_proc
        ))
        return_at_set = set(int(t) for t in state.margin_query_frames)
        for t in suffix_union:
            return_at_set.add(int(t))
        return_at_list = sorted(return_at_set)
        logits_by_t_t, _ = forward_fn.forward_with_objectness(
            processed_t, return_at=return_at_list,
            objectness_at=state.bridge_polish,
        )

        margins_by_t: Dict[int, Tensor] = {}
        for t in state.margin_query_frames:
            t_int = int(t)
            if (t_int not in state.m_true_by_t
                    or t_int not in state.m_decoy_by_t
                    or t_int not in logits_by_t_t):
                continue
            fmo = decoy_margin_per_frame(
                logits_by_t_t[t_int],
                state.m_true_by_t[t_int], state.m_decoy_by_t[t_int],
                margin=config.margin_threshold,
            )
            margins_by_t[t_int] = fmo.margin_loss.detach()

        suffix_iou_by_t: Dict[int, Tensor] = {}
        for t in suffix_union:
            t_int = int(t)
            if (t_int not in state.m_true_by_t
                    or t_int not in logits_by_t_t):
                continue
            p_t = torch.sigmoid(logits_by_t_t[t_int]).flatten()
            m_t = state.m_true_by_t[t_int].flatten().float()
            inter = (p_t * m_t).sum()
            union = (p_t + m_t - p_t * m_t).sum()
            iou = inter / union.clamp_min(1e-6)
            suffix_iou_by_t[t_int] = iou.detach()

    return Stage14TeacherSignals(
        margin_by_t=margins_by_t,
        suffix_iou_by_t=suffix_iou_by_t,
        keep_suffix_frames=sorted(int(t) for t in keep_suffix_frames),
        gain_suffix_frames=sorted(int(t) for t in gain_suffix_frames),
    )


def stage14_forward_loss(
    state: AttackState,
    *,
    x_clean: Tensor,
    traj: FalseTrajectoryParams,
    edit_params: Any,
    R: Optional[Tensor],
    nu: Tensor,
    forward_fn: Any,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    config: Any,
    lambda_fid_val: float,
    R_active: bool = False,
    suffix_probe_frames: Optional[Sequence[int]] = None,
    lambda_suffix: float = 0.0,
    teacher: Optional[Stage14TeacherSignals] = None,
    lambda_keep_margin: float = 0.0,
    keep_suffix_frames: Optional[Sequence[int]] = None,
    gain_suffix_frames: Optional[Sequence[int]] = None,
    lambda_keep_full: float = 0.0,
    lambda_gain_suffix: float = 0.0,
    margin_loss_scale: float = 1.0,
) -> Tuple[Tensor, Stage14LossDiagnostics, Tensor]:
    """One Stage 14 forward + loss computation.

    Returns:
      L_total: scalar Tensor ready to .backward().
      diag: Stage14LossDiagnostics.
      x_edited_full: [T_clean, H, W, 3] = clean-space edited frames (caller
        may snapshot for export at best step).

    NOTE: this function does NOT step optimizers, project trajectory, or
    update nu / R. The caller (legacy wrapper or joint search) is
    responsible for those state mutations between calls.

    v3 (Coverage-Constrained Joint Suffix Optimization):
      `suffix_probe_frames` (attacked-space frame indices, sparse, e.g.
      6 evenly spaced in [w_0+1, T_proc-1]) trigger soft-IoU computation
      of `sigmoid(logits_t) ∩ m_true_t` over those probes. The sum is
      added to L_total weighted by `lambda_suffix`. The probes are
      automatically merged into `forward_fn.return_at` so no extra
      forward call is needed. Defaults (`suffix_probe_frames=None`,
      `lambda_suffix=0.0`) reproduce the v2 / v2.1 behavior exactly.

    v4 (Anchored Stage 14, codex 2026-04-26):
      `teacher` (precomputed via `build_stage14_teacher_signals`) anchors
      the optimization to the A0 baseline (frozen-nu, no bridge polish).

    v4.1 hot-fix (codex 2026-04-27, thread 019dc51a, dev-4 revert
    diagnosis): SPLIT the suffix term into a DENSE no-regression
    constraint and a SPARSE gain term. v4.0 used 6 sparse probes for
    BOTH no-regression and gain — the sparsity allowed optimization to
    regress on the unmonitored ~70% of suffix frames (dog/bmx-trees
    50% revert rate on dev-4).
        L_keep_margin = mean_t relu(margin_loss_cur(t) - margin_loss_A0(t))
                        -- attacked-window no-regression (insert + bridge_polish).
        L_keep_full   = mean_t relu(u_cur(t) - u_A0(t))                       (v4.1)
                        -- DENSE suffix no-regression. mean over
                           `keep_suffix_frames` (typically all non-insert
                           suffix frames).
        L_gain_suffix = mean_t u_cur(t)                                       (v4.1)
                        -- SPARSE gain. mean over `gain_suffix_frames`
                           (typically 6 evenly-spaced probes).
      `margin_loss_scale` rescales the existing aggregate L_margin term.
      Codex v4.1 recommends margin_scale=0.05 (down from v4.0's 0.10) to
      further reduce local-surrogate dominance vs. the dense no-regression.
      Defaults (`teacher=None`, all v4 kwargs zero) reproduce v3 / v2
      behavior exactly.
    """
    device = x_clean.device
    dtype = x_clean.dtype
    T_clean = state.T_clean

    # 1. Oracle decoy masks (differentiable in anchor + delta).
    oracle_decoy_masks_clean = build_oracle_decoy_masks_for_clip(
        state.pseudo_masks_torch, state.W_clean_sorted, traj,
        state.bridge_lengths_list,
    )

    # 2. Per-step unit warp direction (detached).
    with torch.no_grad():
        anchor_norm = traj.anchor_offset.norm(dim=-1, keepdim=True)
        anchor_norm = anchor_norm.clamp_min(1.0)
        unit_dir = (traj.anchor_offset / anchor_norm).detach()

    alphas = alpha_from_logits(
        edit_params.alpha_logits,
        alpha_max=config.oracle_traj_alpha_max,
    )
    d_xy = displacement_from_warp(
        edit_params.warp_s, edit_params.warp_r,
        u_dir=unit_dir, max_disp_px=config.oracle_traj_max_disp_px,
    )

    # 3. Build edited bridge frames (per t, k, l_idx).
    _compositors = {
        "alpha_paste": compose_decoy_alpha_paste,
        "poisson": build_duplicate_object_decoy_frame,
    }
    if config.oracle_traj_compositor not in _compositors:
        raise ValueError(
            f"oracle_traj_compositor must be in {sorted(_compositors)}; "
            f"got {config.oracle_traj_compositor!r}")
    _build_dup = _compositors[config.oracle_traj_compositor]

    edited_by_c: Dict[int, Tensor] = {}
    for t, k_, l_idx in state.bridge_t_list:
        c_t = attacked_to_clean(int(t), state.W_attacked)
        if not (0 <= c_t < T_clean):
            continue
        if c_t not in oracle_decoy_masks_clean:
            continue
        x_t = x_clean[c_t]

        decoy_mask_diff = oracle_decoy_masks_clean[c_t]
        soft_decoy = soften_decoy_mask(
            decoy_mask_diff,
            dilate_px=config.oracle_traj_overlay_dilate_px,
            feather_sigma=config.oracle_traj_overlay_feather_sigma,
        )
        true_mask_c = state.pseudo_masks_torch[c_t]
        soft_true = soften_decoy_mask(
            true_mask_c, dilate_px=1,
            feather_sigma=config.oracle_traj_true_mask_feather_sigma,
        )

        # NOTE: l_idx is 0-based over bridge frames returned by
        # `select_bridge_frames` (t = w_k+1 -> l_idx=0). This aligns with
        # `build_oracle_decoy_masks_for_clip`, which uses bridge_step=0 for
        # the first post-insert clean frame c_k, bridge_step=1 for c_k+1,
        # etc. Using l_idx+1 would misalign duplicate placement vs the
        # softened decoy mask by one step.
        offset_detached = trajectory_offset_at(traj, k_, l_idx).detach()
        dup_offset = (
            int(round(float(offset_detached[0].item()))),
            int(round(float(offset_detached[1].item()))),
        )
        duplicate = _build_dup(
            x_ref=x_clean[c_t], object_mask=true_mask_c,
            decoy_offset=dup_offset,
            feather_radius=config.feather_radius,
            feather_sigma=config.feather_sigma,
        ).to(device)

        d_yx = d_xy[k_, l_idx]
        x_warped = apply_translation_warp_roi(x_t, soft_true, d_yx)
        x_edited = apply_continuation_overlay(
            x_warped, duplicate, soft_decoy, alphas[k_, l_idx],
        )
        if config.oracle_traj_use_residual and R is not None:
            support_R = soften_decoy_mask(
                decoy_mask_diff,
                dilate_px=config.oracle_traj_residual_dilate_px,
                feather_sigma=config.oracle_traj_residual_feather_sigma,
            ).detach()
            x_edited = apply_masked_residual(
                x_edited, R[k_, l_idx], support_R)
        edited_by_c[int(c_t)] = x_edited

    # 4. Stack to [T_clean, H, W, 3].
    frames: List[Tensor] = []
    for c in range(T_clean):
        if c in edited_by_c:
            frames.append(edited_by_c[c])
        else:
            frames.append(x_clean[c])
    x_edited_full = torch.stack(frames, dim=0)

    inserts = (state.decoy_seeds + nu).clamp(0.0, 1.0)
    if config.train_ste_quantize:
        from memshield.losses import fake_uint8_quantize
        x_edited_full = fake_uint8_quantize(x_edited_full)
        inserts = fake_uint8_quantize(inserts)

    processed = build_processed(x_edited_full, inserts, state.W_attacked)

    # 5. Forward + losses.
    # v3: extend return_at with suffix probe frames so we get logits for
    # soft-IoU at suffix-aware probes WITHOUT a second forward call.
    # v4.1: also include keep_suffix_frames (dense) and gain_suffix_frames
    # (sparse) in the return_at set so the L_keep_full / L_gain_suffix
    # terms can score IoU on those frames in the same forward.
    return_at_set = set(int(t) for t in state.margin_query_frames)
    suffix_probes_clean: List[int] = []
    if suffix_probe_frames:
        for t in suffix_probe_frames:
            t_int = int(t)
            if 0 <= t_int < state.T_proc:
                suffix_probes_clean.append(t_int)
                return_at_set.add(t_int)
    if keep_suffix_frames:
        for t in keep_suffix_frames:
            t_int = int(t)
            if 0 <= t_int < state.T_proc:
                return_at_set.add(t_int)
    if gain_suffix_frames:
        for t in gain_suffix_frames:
            t_int = int(t)
            if 0 <= t_int < state.T_proc:
                return_at_set.add(t_int)
    return_at_list = sorted(return_at_set)
    logits_by_t, obj_score_by_t = forward_fn.forward_with_objectness(
        processed, return_at=return_at_list,
        objectness_at=state.bridge_polish,
    )

    margins_by_t = {}
    for t in state.margin_query_frames:
        if (int(t) not in state.m_true_by_t
                or int(t) not in state.m_decoy_by_t):
            continue
        margins_by_t[int(t)] = decoy_margin_per_frame(
            logits_by_t[int(t)],
            state.m_true_by_t[int(t)], state.m_decoy_by_t[int(t)],
            margin=config.margin_threshold,
        )
    agg = aggregate_margin_loss(
        margins_by_t,
        insert_ids=state.insert_polish,
        neighbor_ids=state.bridge_polish,
        neighbor_weight=config.margin_neighbor_weight,
    )
    L_margin = agg.L_margin

    obj_logits_stack = torch.stack(
        [obj_score_by_t[t].flatten().mean()
         for t in state.bridge_polish],
        dim=0,
    ) if state.bridge_polish else torch.zeros(0, device=device)
    L_obj = positive_objectness_loss(
        obj_logits_stack, threshold=config.oracle_traj_obj_threshold,
    )

    bridge_logits_2d: Dict[int, Tensor] = {}
    bridge_true_2d: Dict[int, Tensor] = {}
    for t in state.bridge_polish:
        if int(t) not in state.m_true_by_t:
            continue
        bridge_logits_2d[int(t)] = logits_by_t[int(t)]
        bridge_true_2d[int(t)] = state.m_true_by_t[int(t)]
    L_area, area_ratios = area_preservation_loss(
        bridge_logits_2d, bridge_true_2d,
        area_min=config.oracle_traj_area_min,
        area_max=config.oracle_traj_area_max,
    )

    # Fidelity hinges (LPIPS + TV).
    L_fid_bridge = torch.zeros((), dtype=dtype, device=device)
    for c in sorted(edited_by_c.keys()):
        lp = lpips_fn(x_edited_full[c], x_clean[c])
        L_fid_bridge = L_fid_bridge + lpips_cap_hinge(
            lp, config.lpips_orig_cap)
    L_fid_ins = torch.zeros_like(L_fid_bridge)
    L_fid_TV = torch.zeros_like(L_fid_bridge)
    for k_, w_ in enumerate(state.W_attacked):
        c_k_ = int(w_ - k_)
        if not (0 <= c_k_ < T_clean):
            continue
        x_ref = x_clean[c_k_]
        lp = lpips_fn(inserts[k_], x_ref)
        L_fid_ins = L_fid_ins + lpips_cap_hinge(
            lp, config.lpips_insert_cap)
        ins_chw = inserts[k_].permute(2, 0, 1)
        ref_chw = x_ref.permute(2, 0, 1)
        L_fid_TV = L_fid_TV + tv_hinge(
            ins_chw, ref_chw, multiplier=config.tv_multiplier)
    L_fid_total = L_fid_bridge + L_fid_ins + L_fid_TV

    L_alpha = alpha_regularizer(
        alphas, l1_weight=1.0, smoothness_weight=1.0)
    L_warp = warp_regularizer(
        edit_params.warp_s, edit_params.warp_r,
        l2_weight=1.0, orthogonal_weight=1.0, smoothness_weight=1.0)

    L_traj = trajectory_smoothness_loss(
        traj,
        anchor_smooth_weight=config.oracle_traj_lambda_traj_anchor_smooth,
        delta_smooth_weight=config.oracle_traj_lambda_traj_delta_smooth,
        magnitude_weight=config.oracle_traj_lambda_traj_magnitude,
    )

    if config.oracle_traj_use_residual and R_active and R is not None:
        dh = (R[:, :, 1:, :, :] - R[:, :, :-1, :, :]).abs()
        dw = (R[:, :, :, 1:, :] - R[:, :, :, :-1, :]).abs()
        L_R_tv = (dh.mean() + dw.mean())
    else:
        L_R_tv = torch.zeros((), dtype=dtype, device=device)

    # v3 (Coverage-Constrained Joint Suffix Optimization, codex spec):
    # soft-IoU of true-mask retention at sparse probe frames spanning
    # the attacked-suffix horizon. The inserted-perturbation pipeline
    # tries to MINIMIZE this term (low IoU = SAM2 lost the true object
    # at that probe = good attack at that horizon point). The suffix
    # term aligns the placement gradient with the global mean-J-drop
    # objective, which the local L_margin surrogate alone misses on
    # clips with sharp local fragility (e.g. blackswan).
    # v4.1: compute soft-IoU on UNION of all suffix-related frames in a
    # single pass (covers v3's `suffix_probes_clean`, v4.1's
    # `keep_suffix_frames` (dense), and `gain_suffix_frames` (sparse)).
    # logits_by_t already contains all of these per the return_at_set
    # accumulation earlier in this fn.
    suffix_iou_cur_by_t: Dict[int, Tensor] = {}
    keep_clean = sorted(set(int(t) for t in (keep_suffix_frames or [])
                            if 0 <= int(t) < state.T_proc))
    gain_clean = sorted(set(int(t) for t in (gain_suffix_frames or [])
                            if 0 <= int(t) < state.T_proc))
    iou_union = sorted(set(suffix_probes_clean) | set(keep_clean)
                       | set(gain_clean))
    for t in iou_union:
        t_int = int(t)
        if t_int not in state.m_true_by_t:
            continue
        if t_int not in logits_by_t:
            continue
        p_t = torch.sigmoid(logits_by_t[t_int]).flatten()
        m_t = state.m_true_by_t[t_int].flatten().float()
        inter = (p_t * m_t).sum()
        union = (p_t + m_t - p_t * m_t).sum()
        iou = inter / union.clamp_min(1e-6)
        suffix_iou_cur_by_t[t_int] = iou

    # v3 (Coverage-Constrained Joint Suffix Optimization, codex spec):
    # soft-IoU of true-mask retention at sparse probe frames spanning
    # the attacked-suffix horizon. Backwards-compat path used by joint
    # placement search.
    if suffix_probes_clean and lambda_suffix > 0.0:
        v3_terms = [suffix_iou_cur_by_t[t] for t in suffix_probes_clean
                    if t in suffix_iou_cur_by_t]
        if v3_terms:
            L_suffix = torch.stack(v3_terms, dim=0).sum()
        else:
            L_suffix = torch.zeros((), dtype=dtype, device=device)
    else:
        L_suffix = torch.zeros((), dtype=dtype, device=device)

    # v4.x (Anchored Stage 14): no-regression vs A0 + explicit suffix gain.
    # Active only when teacher is supplied AND the corresponding lambda is
    # > 0. Each term is computed as a separate scalar tensor so it shows up
    # individually in diagnostics for trace plots.
    #
    # v4.1 hot-fix (codex 2026-04-27): L_keep_full is computed as the MEAN
    # over the DENSE keep_suffix_frames (replaces v4.0's sum over 6 sparse
    # probes). L_gain_suffix is now MEAN over the SPARSE gain_suffix_frames.
    L_keep_margin = torch.zeros((), dtype=dtype, device=device)
    L_keep_full = torch.zeros((), dtype=dtype, device=device)
    L_gain_suffix = torch.zeros((), dtype=dtype, device=device)
    if teacher is not None:
        if lambda_keep_margin > 0.0 and teacher.margin_by_t:
            keep_terms: List[Tensor] = []
            for t in state.margin_query_frames:
                t_int = int(t)
                if t_int not in margins_by_t:
                    continue
                if t_int not in teacher.margin_by_t:
                    continue
                # margin_loss_cur(t) - margin_loss_A0(t) > 0 means current
                # attack is WEAKER than A0 (loss is HIGHER). Penalize.
                m_cur = margins_by_t[t_int].margin_loss
                m_A0 = teacher.margin_by_t[t_int]
                keep_terms.append(torch.relu(m_cur - m_A0))
            if keep_terms:
                L_keep_margin = torch.stack(keep_terms, dim=0).mean()
        if lambda_keep_full > 0.0 and teacher.suffix_iou_by_t and keep_clean:
            kf_terms: List[Tensor] = []
            for t in keep_clean:
                t_int = int(t)
                if t_int not in suffix_iou_cur_by_t:
                    continue
                if t_int not in teacher.suffix_iou_by_t:
                    continue
                u_cur = suffix_iou_cur_by_t[t_int]
                u_A0 = teacher.suffix_iou_by_t[t_int]
                kf_terms.append(torch.relu(u_cur - u_A0))
            if kf_terms:
                L_keep_full = torch.stack(kf_terms, dim=0).mean()
        if lambda_gain_suffix > 0.0 and gain_clean:
            gain_terms = [suffix_iou_cur_by_t[t] for t in gain_clean
                          if t in suffix_iou_cur_by_t]
            if gain_terms:
                L_gain_suffix = torch.stack(gain_terms, dim=0).mean()

    L = (
        float(margin_loss_scale) * config.oracle_traj_lambda_margin * L_margin
        + config.oracle_traj_lambda_obj * L_obj
        + config.oracle_traj_lambda_area * L_area
        + lambda_fid_val * L_fid_total
        + config.oracle_traj_lambda_alpha * L_alpha
        + config.oracle_traj_lambda_warp * L_warp
        + config.oracle_traj_lambda_residual_tv * L_R_tv
        + L_traj
        + float(lambda_suffix) * L_suffix
        + float(lambda_keep_margin) * L_keep_margin
        + float(lambda_keep_full) * L_keep_full
        + float(lambda_gain_suffix) * L_gain_suffix
    )

    # Diagnostics.
    with torch.no_grad():
        decoy_overlap_list: List[float] = []
        true_overlap_list: List[float] = []
        obj_score_list: List[float] = []
        valid_t_list: List[int] = []
        for t in state.bridge_polish:
            if (int(t) not in state.m_decoy_by_t
                    or int(t) not in state.m_true_by_t):
                continue
            pred = torch.sigmoid(logits_by_t[int(t)]).flatten()
            m_d = state.m_decoy_by_t[int(t)].flatten().float()
            m_tr = state.m_true_by_t[int(t)].flatten().float()
            d_ov = float(((pred * m_d).sum()
                          / m_d.sum().clamp_min(1e-4)).item())
            t_ov = float(((pred * m_tr).sum()
                          / m_tr.sum().clamp_min(1e-4)).item())
            decoy_overlap_list.append(d_ov)
            true_overlap_list.append(t_ov)
            obj_score_list.append(float(
                obj_score_by_t[int(t)].flatten().mean().item()))
            valid_t_list.append(int(t))
        mean_decoy_overlap = (
            sum(decoy_overlap_list) / max(1, len(decoy_overlap_list)))
        mean_true_overlap = (
            sum(true_overlap_list) / max(1, len(true_overlap_list)))
        mean_obj_score = (
            float(obj_logits_stack.mean().item())
            if obj_logits_stack.numel() > 0 else 0.0)
        wrong_but_present = sum(
            1 for d, tr, ar, obj in zip(
                decoy_overlap_list, true_overlap_list,
                [area_ratios.get(t, 1.0) for t in valid_t_list],
                obj_score_list)
            if d > tr and ar > 0.5 and obj > 0.0
        )

    tol = 1e-6
    feas = (
        float(L_fid_bridge.detach().item()) <= tol
        and float(L_fid_ins.detach().item()) <= tol
        and float(L_fid_TV.detach().item()) <= tol
    )

    diag = Stage14LossDiagnostics(
        L_margin=L_margin, L_obj=L_obj, L_area=L_area,
        L_fid_bridge=L_fid_bridge, L_fid_ins=L_fid_ins, L_fid_TV=L_fid_TV,
        L_alpha=L_alpha, L_warp=L_warp, L_traj=L_traj, L_R_tv=L_R_tv,
        L_suffix=L_suffix,
        L_keep_margin=L_keep_margin,
        L_keep_full=L_keep_full,
        L_gain_suffix=L_gain_suffix,
        mean_decoy_overlap=float(mean_decoy_overlap),
        mean_true_overlap=float(mean_true_overlap),
        delta_overlap=float(mean_decoy_overlap - mean_true_overlap),
        mean_obj_score=mean_obj_score,
        wrong_but_present_count=int(wrong_but_present),
        feasible=feas,
        alpha_mean=float(alphas.mean().detach().item()),
        alpha_max_step=float(alphas.max().detach().item()),
        warp_disp_max=float(d_xy.norm(dim=-1).max().detach().item()),
        n_bridge=len(state.bridge_polish),
        n_suffix_probes=len(suffix_probes_clean),
    )
    return L, diag, x_edited_full


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


class _DummyConfig:
    """Minimal duck-type stand-in for VADIv5Config used only in self-tests."""

    insert_base_mode = "duplicate_seed"
    feather_radius = 5
    feather_sigma = 2.0
    oracle_traj_bridge_length = 4
    oracle_traj_compositor = "alpha_paste"
    oracle_traj_alpha_max = 0.35
    oracle_traj_max_disp_px = 3.0
    oracle_traj_overlay_dilate_px = 2
    oracle_traj_overlay_feather_sigma = 2.5
    oracle_traj_true_mask_feather_sigma = 2.0
    oracle_traj_residual_dilate_px = 4
    oracle_traj_residual_feather_sigma = 3.0
    oracle_traj_use_residual = False
    oracle_traj_obj_threshold = 0.0
    oracle_traj_area_min = 0.5
    oracle_traj_area_max = 1.5
    oracle_traj_lambda_margin = 1.0
    oracle_traj_lambda_obj = 0.1
    oracle_traj_lambda_area = 0.1
    oracle_traj_lambda_alpha = 0.01
    oracle_traj_lambda_warp = 0.01
    oracle_traj_lambda_residual_tv = 0.001
    oracle_traj_lambda_traj_anchor_smooth = 1.0
    oracle_traj_lambda_traj_delta_smooth = 1.0
    oracle_traj_lambda_traj_magnitude = 0.1
    margin_threshold = 1.0
    margin_neighbor_weight = 0.5
    lpips_orig_cap = 0.20
    lpips_insert_cap = 0.35
    tv_multiplier = 1.2
    train_ste_quantize = False


def _test_attack_state_invariants() -> None:
    """build_attack_state_from_W produces well-formed AttackState."""
    torch.manual_seed(0)
    np.random.seed(0)
    T_clean, H, W = 30, 32, 32
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c = min(W - 4, max(0, t // 3))
        m[12:20, c:c + 4] = 1.0
        pseudo_masks.append(m)

    cfg = _DummyConfig()
    W_clean = [5, 12, 20]
    state = build_attack_state_from_W(W_clean, x_clean, pseudo_masks, cfg)

    assert state.W_clean_sorted == [5, 12, 20]
    assert state.W_attacked == [5, 13, 22]    # +0, +1, +2
    assert state.K == 3
    assert state.T_clean == T_clean
    assert state.T_proc == T_clean + 3
    assert state.decoy_seeds.shape == (3, H, W, 3)
    assert len(state.decoy_offsets_init) == 3
    assert state.bridge_lengths_list == [4, 4, 4]
    assert state.L_max == 4
    # Bridge frames per insert (length<=4 with ordering).
    for k in range(3):
        assert k in state.bridge_frames_by_k
        bs = state.bridge_frames_by_k[k]
        assert len(bs) <= 4
        for tt in bs:
            assert tt > state.W_attacked[k]
            assert tt < state.T_proc
    assert state.insert_polish == sorted(state.W_attacked)
    # Mask invariants: all insert positions present in m_true_by_t /
    # m_decoy_by_t with the override (NOT midframe average).
    for k, w in enumerate(state.W_attacked):
        c_k = state.W_clean_sorted[k]
        assert int(w) in state.m_true_by_t
        m_true_at_w = state.m_true_by_t[int(w)].cpu().numpy()
        # Override: m_true_by_t[w] == clean[c_k] (NOT 0.5*clean[c_k-1] + 0.5*clean[c_k]).
        assert np.allclose(m_true_at_w, pseudo_masks[c_k])
    print("  attack_state invariants OK")


def _test_assemble_attack_state_matches_builder() -> None:
    """assemble_attack_state(...) must produce same AttackState as
    build_attack_state_from_W(...) given the same W (within numerical
    equivalence). This is the foundation of the parity test."""
    torch.manual_seed(0)
    np.random.seed(0)
    T_clean, H, W = 25, 32, 32
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c = min(W - 4, max(0, 4 + t // 3))
        m[12:20, c:c + 4] = 1.0
        pseudo_masks.append(m)

    cfg = _DummyConfig()
    W_clean = [4, 11, 18]
    state_built = build_attack_state_from_W(
        W_clean, x_clean, pseudo_masks, cfg)

    # Assemble using the same fields that the legacy path would have built.
    state_packed = assemble_attack_state(
        W_clean_sorted=state_built.W_clean_sorted,
        W_attacked=state_built.W_attacked,
        decoy_seeds=state_built.decoy_seeds,
        decoy_offsets_init=state_built.decoy_offsets_init,
        bridge_frames_by_k=state_built.bridge_frames_by_k,
        bridge_lengths=state_built.bridge_lengths_list,
        pseudo_masks_clean=pseudo_masks,
        m_true_by_t=state_built.m_true_by_t,
        m_decoy_by_t=state_built.m_decoy_by_t,
        x_clean=x_clean,
    )

    # Field-by-field equality.
    assert state_built.W_clean_sorted == state_packed.W_clean_sorted
    assert state_built.W_attacked == state_packed.W_attacked
    assert state_built.K == state_packed.K
    assert state_built.T_clean == state_packed.T_clean
    assert state_built.T_proc == state_packed.T_proc
    assert torch.allclose(state_built.decoy_seeds, state_packed.decoy_seeds)
    assert (state_built.decoy_offsets_init
            == state_packed.decoy_offsets_init)
    assert (state_built.bridge_lengths_list
            == state_packed.bridge_lengths_list)
    assert state_built.L_max == state_packed.L_max
    assert state_built.bridge_frames_by_k == state_packed.bridge_frames_by_k
    assert state_built.bridge_t_list == state_packed.bridge_t_list
    assert state_built.bridge_polish == state_packed.bridge_polish
    assert state_built.insert_polish == state_packed.insert_polish
    assert (state_built.margin_query_frames
            == state_packed.margin_query_frames)
    # m_true_by_t / m_decoy_by_t dicts (same keys + tensor allclose).
    for t in state_built.m_true_by_t:
        assert torch.allclose(
            state_built.m_true_by_t[t], state_packed.m_true_by_t[t])
    for t in state_built.m_decoy_by_t:
        assert torch.allclose(
            state_built.m_decoy_by_t[t], state_packed.m_decoy_by_t[t])
    print("  assemble == build (field-by-field) OK")


def _test_attack_state_unsorted_W_handled() -> None:
    """Builder accepts unsorted W_clean and sorts internally; preserves
    insert ordering for downstream traj_anchor/decoy_offsets alignment."""
    T_clean, H, W = 20, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        m[6:10, 4:8] = 1.0
        pseudo_masks.append(m)
    cfg = _DummyConfig()

    # Unsorted input.
    state_unsorted = build_attack_state_from_W(
        [12, 4, 8], x_clean, pseudo_masks, cfg)
    state_sorted = build_attack_state_from_W(
        [4, 8, 12], x_clean, pseudo_masks, cfg)
    assert state_unsorted.W_clean_sorted == state_sorted.W_clean_sorted
    assert state_unsorted.W_attacked == state_sorted.W_attacked
    assert (state_unsorted.bridge_frames_by_k
            == state_sorted.bridge_frames_by_k)
    print("  unsorted W handled OK")


def _test_attack_state_input_validation() -> None:
    """Builder rejects degenerate W (out-of-range, duplicates, empty)."""
    T_clean, H, W = 20, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = [np.zeros((H, W), dtype=np.float32) for _ in range(T_clean)]
    cfg = _DummyConfig()

    # Empty.
    try:
        build_attack_state_from_W([], x_clean, pseudo_masks, cfg)
        assert False, "expected ValueError on empty W_clean"
    except ValueError as e:
        assert "at least 1" in str(e)
    # Duplicate.
    try:
        build_attack_state_from_W([5, 5, 8], x_clean, pseudo_masks, cfg)
        assert False, "expected ValueError on duplicate W_clean"
    except ValueError as e:
        assert "duplicates" in str(e)
    # Out-of-range (c=0 not allowed because seed needs neighbors).
    try:
        build_attack_state_from_W([0, 5, 10], x_clean, pseudo_masks, cfg)
        assert False, "expected ValueError on c=0"
    except ValueError as e:
        assert "out of [1," in str(e)
    # Out-of-range (c >= T_clean).
    try:
        build_attack_state_from_W(
            [5, 10, T_clean], x_clean, pseudo_masks, cfg)
        assert False, "expected ValueError on c=T_clean"
    except ValueError as e:
        assert "out of [1," in str(e)
    print("  input validation OK")


def _test_remap_masks_processed_space_inline() -> None:
    """The inlined `_remap_masks_to_processed_space` matches the contract
    of the legacy `scripts.run_vadi.remap_masks_to_processed_space`."""
    T_clean = 10
    H, W = 8, 8
    masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        m[2:6, t % W:(t % W) + 2] = 1.0
        masks.append(m)
    W_attacked = [3, 6]                # K=2 inserts
    out = _remap_masks_to_processed_space(masks, W_attacked)
    # Total length T_proc.
    assert len(out) == T_clean + len(W_attacked)
    # Insert positions: midframe average.
    # w=3 -> c_k=3, expect 0.5*masks[2] + 0.5*masks[3]
    assert np.allclose(out[3], 0.5 * masks[2] + 0.5 * masks[3])
    # w=6 -> c_k=5 (=6-1), expect 0.5*masks[4] + 0.5*masks[5]
    assert np.allclose(out[6], 0.5 * masks[4] + 0.5 * masks[5])
    # Non-insert positions: pass-through via attacked_to_clean.
    # w=0 -> c=0 (no insert before it)
    assert np.allclose(out[0], masks[0])
    # w=4 -> c=3 (one insert at w=3 before, so subtract 1)
    assert np.allclose(out[4], masks[3])
    # w=7 -> c=5 (two inserts before, so subtract 2)
    assert np.allclose(out[7], masks[5])
    print("  _remap_masks_to_processed_space OK")


def _test_supervision_masks_insert_override() -> None:
    """Insert positions in m_true_by_t / m_decoy_by_t are overridden to
    seed-aligned masks (NOT the midframe average produced by the remap).

    This is the codex R1 high-severity fix in run_v5_for_clip; the helper
    must replicate it exactly."""
    T_clean = 12
    H, W = 8, 8
    masks_np = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c0 = min(W - 2, max(0, t // 3))
        m[3:5, c0:c0 + 2] = 1.0
        masks_np.append(m)

    W_clean_sorted = [3, 8]
    W_attacked = [3, 9]                 # +0, +1
    decoy_offsets = [(0, 2), (1, -1)]
    m_true, m_decoy = _build_supervision_masks(
        masks_np, W_clean_sorted, decoy_offsets, W_attacked)
    # m_true at w=3 == masks[3] (NOT 0.5*masks[2] + 0.5*masks[3])
    assert np.allclose(m_true[3], masks_np[3])
    # m_decoy at w=3 == shift(masks[3], 0, 2)
    assert np.allclose(
        m_decoy[3], shift_mask_np(masks_np[3], 0, 2))
    # m_true at w=9 == masks[8] (c_k=9-1=8)
    assert np.allclose(m_true[9], masks_np[8])
    # m_decoy at w=9 == shift(masks[8], 1, -1)
    assert np.allclose(
        m_decoy[9], shift_mask_np(masks_np[8], 1, -1))
    # Pre-first-insert frame (t=0 in clean = t=0 in attacked) should
    # resolve to a non-insert pass-through with zero decoy mass.
    assert m_decoy[0].sum() == 0.0
    print("  supervision-mask insert-override OK")


def _test_v4_suffix_probe_builder() -> None:
    """build_suffix_probe_frames excludes W_attacked, evenly samples,
    handles edge cases."""
    # Standard case: K=3 W=[5,13,22], T_proc=33, n_probes=6.
    W_attacked = [5, 13, 22]
    T_proc = 33
    probes = build_suffix_probe_frames(W_attacked, T_proc, n_probes=6)
    assert len(probes) == 6, f"expected 6 probes, got {len(probes)}"
    assert all(p not in W_attacked for p in probes), (
        f"probes overlap with inserts: {probes} vs {W_attacked}")
    assert all(min(W_attacked) < p < T_proc for p in probes), (
        f"probes out of suffix range: {probes}")
    assert probes == sorted(probes), f"probes not sorted: {probes}"
    assert len(set(probes)) == len(probes), f"duplicates: {probes}"
    # Exhausted candidates (n_probes > suffix length).
    short_probes = build_suffix_probe_frames([5, 6], T_proc=8, n_probes=6)
    assert short_probes == [7], (
        f"expected [7] (only candidate), got {short_probes}")
    # n_probes=0 -> empty.
    assert build_suffix_probe_frames([5, 13, 22], T_proc, n_probes=0) == []
    # Empty W_attacked -> empty.
    assert build_suffix_probe_frames([], T_proc, n_probes=6) == []
    # n_probes=1 -> one mid-suffix frame (no overlap with inserts).
    one_probe = build_suffix_probe_frames(W_attacked, T_proc, n_probes=1)
    assert len(one_probe) == 1
    assert one_probe[0] not in W_attacked
    print("  v4 suffix probe builder OK")


class _StubForwardFn:
    """Deterministic stub for SAM2 forward used in v4 self-tests.

    `logits_per_t`: dict t -> [H, W] tensor (will be returned for any call).
    `obj_per_t`: dict t -> scalar (objectness logit).

    Both A0 and current-step calls receive the SAME `logits_per_t` so that
    L_keep_margin / L_keep_suffix evaluate to exactly zero (pure relu(0)),
    confirming the no-regression sign convention.
    """

    def __init__(self, logits_per_t: Dict[int, Tensor],
                 obj_per_t: Dict[int, float]):
        self._logits = logits_per_t
        self._obj = obj_per_t

    def forward_with_objectness(
        self, processed: Tensor, *, return_at: List[int],
        objectness_at: List[int],
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        H, W = next(iter(self._logits.values())).shape
        device = processed.device
        dtype = processed.dtype
        out_logits: Dict[int, Tensor] = {}
        for t in return_at:
            if t in self._logits:
                out_logits[int(t)] = self._logits[int(t)].to(device).to(dtype)
            else:
                out_logits[int(t)] = torch.zeros(
                    (H, W), device=device, dtype=dtype)
        out_obj: Dict[int, Tensor] = {}
        for t in objectness_at:
            v = self._obj.get(int(t), 0.0)
            out_obj[int(t)] = torch.tensor(
                [v], device=device, dtype=dtype)
        return out_logits, out_obj


def _test_v4_teacher_zero_regression() -> None:
    """When current logits == A0 logits (stubbed), L_keep_margin and
    L_keep_full must be exactly zero (sign sanity), and L_gain_suffix
    equals MEAN_t soft-IoU(A0) over gain probes (v4.1 mean, not sum)."""
    torch.manual_seed(123)
    np.random.seed(123)
    T_clean, H, W = 18, 24, 24
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks: List[np.ndarray] = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c = min(W - 4, max(0, t // 2))
        m[8:14, c:c + 4] = 1.0
        pseudo_masks.append(m)

    cfg = _DummyConfig()
    W_clean = [5, 9, 14]
    state = build_attack_state_from_W(W_clean, x_clean, pseudo_masks, cfg)

    # v4.1: keep set = DENSE non-insert suffix; gain set = sparse 4 probes.
    keep_frames = build_keep_suffix_frames(state.W_attacked, state.T_proc)
    gain_probes = build_suffix_probe_frames(
        state.W_attacked, state.T_proc, n_probes=4)
    assert len(keep_frames) >= len(gain_probes), (
        "v4.1 dense keep should cover at least as many frames as sparse gain")

    logits_per_t: Dict[int, Tensor] = {}
    obj_per_t: Dict[int, float] = {}
    all_t = sorted(set(state.margin_query_frames) | set(state.bridge_polish)
                   | set(keep_frames) | set(gain_probes))
    for t in all_t:
        logits_per_t[t] = torch.randn(H, W) * 0.5
        obj_per_t[t] = 0.5

    forward_fn = _StubForwardFn(logits_per_t, obj_per_t)

    nu = torch.zeros_like(state.decoy_seeds)
    teacher = build_stage14_teacher_signals(
        state, x_clean=x_clean, nu_teacher=nu, forward_fn=forward_fn,
        config=cfg,
        keep_suffix_frames=keep_frames,
        gain_suffix_frames=gain_probes,
    )
    assert len(teacher.suffix_iou_by_t) > 0, (
        "teacher suffix IoUs not populated")
    assert len(teacher.margin_by_t) > 0, (
        "teacher margin tensors not populated")
    assert teacher.keep_suffix_frames == sorted(keep_frames)
    assert teacher.gain_suffix_frames == sorted(gain_probes)

    K = state.K
    L_max = state.L_max
    traj = FalseTrajectoryParams(
        anchor_offset=torch.zeros(K, 2),
        delta_offset=torch.zeros(K, L_max, 2),
        L=int(L_max),
    )

    @dataclass
    class _EditParams:
        alpha_logits: Tensor
        warp_s: Tensor
        warp_r: Tensor

    edit = _EditParams(
        alpha_logits=torch.full((K, L_max), -3.0),
        warp_s=torch.zeros(K, L_max),
        warp_r=torch.zeros(K, L_max),
    )

    def lpips_stub(a: Tensor, b: Tensor) -> Tensor:
        return torch.zeros((), dtype=x_clean.dtype, device=x_clean.device)

    L, diag, _ = stage14_forward_loss(
        state, x_clean=x_clean, traj=traj, edit_params=edit, R=None, nu=nu,
        forward_fn=forward_fn, lpips_fn=lpips_stub, config=cfg,
        lambda_fid_val=1.0,
        suffix_probe_frames=None, lambda_suffix=0.0,
        teacher=teacher,
        keep_suffix_frames=keep_frames, gain_suffix_frames=gain_probes,
        lambda_keep_margin=1.0,
        lambda_keep_full=25.0,
        lambda_gain_suffix=2.0,
        margin_loss_scale=0.05,
    )

    L_keep_margin_val = float(diag.L_keep_margin.detach().item())
    L_keep_full_val = float(diag.L_keep_full.detach().item())
    L_gain_suffix_val = float(diag.L_gain_suffix.detach().item())
    assert L_keep_margin_val < 1e-5, (
        f"L_keep_margin should be 0 (cur==A0), got {L_keep_margin_val}")
    assert L_keep_full_val < 1e-5, (
        f"L_keep_full should be 0 (cur==A0), got {L_keep_full_val}")
    # v4.1: L_gain_suffix == MEAN over gain probe IoUs from teacher.
    gain_iou_vals = [
        teacher.suffix_iou_by_t[t].item()
        for t in teacher.gain_suffix_frames
        if t in teacher.suffix_iou_by_t
    ]
    expected_gain = sum(gain_iou_vals) / max(1, len(gain_iou_vals))
    assert abs(L_gain_suffix_val - expected_gain) < 1e-3, (
        f"L_gain_suffix={L_gain_suffix_val} != expected mean={expected_gain}")
    print("  v4.1 teacher zero-regression OK "
          f"(keep_full=0, gain_suffix mean={L_gain_suffix_val:.4f})")


def _test_v4_keep_margin_sign_sanity() -> None:
    """L_keep_margin > 0 when current margin_loss > A0 margin_loss
    (current attack is WEAKER than A0). This validates the sign convention
    and rules out the inverse-direction bug."""
    torch.manual_seed(7)
    np.random.seed(7)
    T_clean, H, W = 14, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks: List[np.ndarray] = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        m[6:10, 4:8] = 1.0
        pseudo_masks.append(m)

    cfg = _DummyConfig()
    W_clean = [3, 7, 11]
    state = build_attack_state_from_W(W_clean, x_clean, pseudo_masks, cfg)

    # A0 logits: strongly favor decoy region (low margin_loss = strong A0).
    # Current logits: strongly favor true region (high margin_loss = weak).
    logits_A0: Dict[int, Tensor] = {}
    logits_cur: Dict[int, Tensor] = {}
    obj = {t: 0.5 for t in state.bridge_polish}
    all_t = sorted(set(state.margin_query_frames) | set(state.bridge_polish))
    for t in all_t:
        # A0: -2 on true region, +2 on decoy region.
        l_a = torch.full((H, W), -2.0)
        l_a[6:10, 4:8] = -3.0
        l_a[6:10, 8:12] = 3.0     # decoy shift +4 in x.
        # Current: opposite (true region wins, weaker attack).
        l_c = torch.full((H, W), -2.0)
        l_c[6:10, 4:8] = 3.0
        l_c[6:10, 8:12] = -3.0
        logits_A0[t] = l_a
        logits_cur[t] = l_c

    nu = torch.zeros_like(state.decoy_seeds)
    teacher = build_stage14_teacher_signals(
        state, x_clean=x_clean, nu_teacher=nu,
        forward_fn=_StubForwardFn(logits_A0, obj), config=cfg,
        keep_suffix_frames=[], gain_suffix_frames=[],
    )

    # Confirm teacher per-frame margin_loss is small (strong A0).
    teacher_margin_means = (
        sum(float(v.item()) for v in teacher.margin_by_t.values())
        / max(1, len(teacher.margin_by_t)))

    # Run forward with current (weaker) logits.
    K = state.K
    L_max = state.L_max
    traj = FalseTrajectoryParams(
        anchor_offset=torch.zeros(K, 2),
        delta_offset=torch.zeros(K, L_max, 2),
        L=int(L_max),
    )

    @dataclass
    class _EditParams:
        alpha_logits: Tensor
        warp_s: Tensor
        warp_r: Tensor

    edit = _EditParams(
        alpha_logits=torch.full((K, L_max), -3.0),
        warp_s=torch.zeros(K, L_max),
        warp_r=torch.zeros(K, L_max),
    )

    def lpips_stub(a: Tensor, b: Tensor) -> Tensor:
        return torch.zeros((), dtype=x_clean.dtype, device=x_clean.device)

    L, diag, _ = stage14_forward_loss(
        state, x_clean=x_clean, traj=traj, edit_params=edit, R=None, nu=nu,
        forward_fn=_StubForwardFn(logits_cur, obj), lpips_fn=lpips_stub,
        config=cfg, lambda_fid_val=1.0,
        suffix_probe_frames=None, lambda_suffix=0.0,
        teacher=teacher,
        keep_suffix_frames=[], gain_suffix_frames=[],
        lambda_keep_margin=1.0, lambda_keep_full=0.0,
        lambda_gain_suffix=0.0, margin_loss_scale=0.05,
    )

    keep_margin = float(diag.L_keep_margin.detach().item())
    assert keep_margin > 0.0, (
        f"L_keep_margin should be > 0 when current is weaker than A0; "
        f"got {keep_margin} (teacher margin mean = {teacher_margin_means})")
    print(f"  v4.1 L_keep_margin sign sanity OK "
          f"(current_weaker -> L_keep_margin = {keep_margin:.4f})")


def _test_v41_dense_keep_frames() -> None:
    """v4.1 build_keep_suffix_frames returns ALL non-insert suffix frames.

    Critical invariant: dense keep set is a SUPERSET of any sparse gain
    probes drawn from the same suffix region (so the teacher's IoU dict
    populated for keep frames covers all gain frames too).
    """
    W_attacked = [5, 13, 22]
    T_proc = 33
    keep = build_keep_suffix_frames(W_attacked, T_proc)
    # Expect every t in (5+1, T_proc) excluding {5, 13, 22}
    expected = [t for t in range(6, T_proc) if t not in W_attacked]
    assert keep == expected, f"keep frames mismatch: got {keep} expected {expected}"
    assert all(t not in W_attacked for t in keep)
    assert all(min(W_attacked) < t < T_proc for t in keep)
    # Sparse gain must be a subset of dense keep (they share the same
    # candidate filter: non-insert + suffix range).
    gain = build_suffix_probe_frames(W_attacked, T_proc, n_probes=6)
    assert set(gain).issubset(set(keep)), (
        f"sparse gain probes {gain} must be subset of dense keep {keep}")
    print(f"  v4.1 dense keep_suffix_frames OK "
          f"(|keep|={len(keep)}, |gain|={len(gain)}, gain subset of keep)")


def _test_v41_keep_full_dense_semantics() -> None:
    """v4.1 L_keep_full activates per-frame across the dense suffix.

    Setup: A0 logits favor decoy on ALL suffix frames, current logits favor
    true mask on a subset of suffix frames (mimicking 'optimization made
    keep frames worse than A0 on a subset'). L_keep_full should be > 0
    proportional to the regressing-frame fraction.

    Compare against v4.0 sparse-probe behavior: if gain set is sparse and
    the regressing frames happen to fall outside the sparse probes, v4.0
    L_keep_suffix would be ZERO. v4.1 L_keep_full sees the regression.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    T_clean, H, W = 16, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        m[6:10, 4:8] = 1.0
        pseudo_masks.append(m)

    cfg = _DummyConfig()
    W_clean = [3, 7, 12]
    state = build_attack_state_from_W(W_clean, x_clean, pseudo_masks, cfg)

    keep_frames = build_keep_suffix_frames(state.W_attacked, state.T_proc)
    gain_probes = build_suffix_probe_frames(
        state.W_attacked, state.T_proc, n_probes=2)
    obj = {t: 0.5 for t in state.bridge_polish}

    # A0 logits: strong attack everywhere (decoy wins on all suffix).
    logits_A0 = {}
    for t in sorted(set(state.margin_query_frames) | set(keep_frames)
                    | set(gain_probes)):
        l = torch.full((H, W), -3.0)
        l[6:10, 8:12] = 3.0   # decoy wins
        logits_A0[t] = l

    # Current logits: regress on keep frames NOT in gain probe set.
    logits_cur = dict(logits_A0)
    regressing = [t for t in keep_frames if t not in gain_probes]
    assert len(regressing) > 0, "test setup needs regressing-only frames"
    for t in regressing:
        l = torch.full((H, W), -3.0)
        l[6:10, 4:8] = 3.0    # true mask wins -> attack regressed
        l[6:10, 8:12] = -3.0
        logits_cur[t] = l

    nu = torch.zeros_like(state.decoy_seeds)
    teacher = build_stage14_teacher_signals(
        state, x_clean=x_clean, nu_teacher=nu,
        forward_fn=_StubForwardFn(logits_A0, obj), config=cfg,
        keep_suffix_frames=keep_frames,
        gain_suffix_frames=gain_probes,
    )

    K = state.K
    L_max = state.L_max
    traj = FalseTrajectoryParams(
        anchor_offset=torch.zeros(K, 2),
        delta_offset=torch.zeros(K, L_max, 2),
        L=int(L_max),
    )

    @dataclass
    class _EditParams:
        alpha_logits: Tensor
        warp_s: Tensor
        warp_r: Tensor

    edit = _EditParams(
        alpha_logits=torch.full((K, L_max), -3.0),
        warp_s=torch.zeros(K, L_max),
        warp_r=torch.zeros(K, L_max),
    )

    def lpips_stub(a, b):
        return torch.zeros((), dtype=x_clean.dtype, device=x_clean.device)

    L, diag, _ = stage14_forward_loss(
        state, x_clean=x_clean, traj=traj, edit_params=edit, R=None, nu=nu,
        forward_fn=_StubForwardFn(logits_cur, obj), lpips_fn=lpips_stub,
        config=cfg, lambda_fid_val=1.0,
        suffix_probe_frames=None, lambda_suffix=0.0,
        teacher=teacher,
        keep_suffix_frames=keep_frames,
        gain_suffix_frames=gain_probes,
        lambda_keep_margin=0.0,    # isolate L_keep_full effect
        lambda_keep_full=25.0,
        lambda_gain_suffix=0.0,
        margin_loss_scale=0.0,
    )

    keep_full = float(diag.L_keep_full.detach().item())
    # Expect L_keep_full > 0: regressing frames have IoU(cur) >> IoU(A0).
    # Specifically, fraction of frames regressing = len(regressing)/len(keep).
    frac = len(regressing) / max(1, len(keep_frames))
    assert keep_full > 0.05, (
        f"L_keep_full should detect regression on {len(regressing)}"
        f"/{len(keep_frames)} frames ({frac:.2%}); got {keep_full:.4f}")
    print(f"  v4.1 L_keep_full dense-coverage OK "
          f"(regressing={len(regressing)}/{len(keep_frames)}, "
          f"L_keep_full={keep_full:.4f})")


def _self_test() -> None:
    print("memshield.stage14_helpers self-tests:")
    _test_attack_state_invariants()
    _test_assemble_attack_state_matches_builder()
    _test_attack_state_unsorted_W_handled()
    _test_attack_state_input_validation()
    _test_remap_masks_processed_space_inline()
    _test_supervision_masks_insert_override()
    _test_v4_suffix_probe_builder()
    _test_v41_dense_keep_frames()
    _test_v4_teacher_zero_regression()
    _test_v4_keep_margin_sign_sanity()
    _test_v41_keep_full_dense_semantics()
    print("memshield.stage14_helpers: all self-tests PASSED")


if __name__ == "__main__":
    _self_test()
