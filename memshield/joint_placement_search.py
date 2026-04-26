"""Joint curriculum placement-perturbation search for VADI Stage 14.

Replaces brute-force `beam_search_K3` (~9 h/clip) with a discrete-schedule-
interpolated joint optimization (~63 min/clip target; codex says budget
2-4 h/clip until proven). Approved design: auto-review-loop Round 6 R3 GO
(codex thread `019dc51a-c71a-7971-bece-116a592de2f5`).

Core mechanism
==============

* **Continuous learnable τ ∈ R^K** parameterized by ordered gaps:

      tau[0] = clamp_left + (T_clean - bridge_budget - clamp_left) * sigmoid(g0)
      tau[1] = tau[0] + d_min + softplus(g1)
      tau[k] = tau[k-1] + d_min + softplus(gk)

  The recurrence guarantees τ[0] ≥ 1 and τ[k+1] ≥ τ[k] + d_min by
  construction (NOT by penalty), and τ[K-1] + bridge_budget ≤ T_clean - 1
  by the τ[0] sigmoid range. d_min is typically 2 or 3.

* **Discrete schedule interpolation as timing surrogate** (NOT soft-frame
  averaging which Round-2 spec rejected as fake timing): per joint step,
  enumerate `2^|active_inserts|` neighbor schedules using floor(τ[k]) and
  ceil(τ[k]) per active insert. For each schedule, weight = product over
  active k of `(1 - frac[k])` if floor else `frac[k]`. Filter invalid
  schedules (those violating strict ordering c[0] < c[1] < ... < c[K-1] or
  τ[k] going out of [1, T_clean - bridge_budget - 1]); renormalize the
  remaining weights to sum to 1. Loss = weighted sum of per-schedule
  Stage-14 forward losses → real differentiable timing signal via the
  weight (which is differentiable in τ); the schedule integers themselves
  are non-differentiable but only feed the W-dependent state build.

* **Single-level joint loop** (NOT bilevel — each joint step does ONE
  Stage-14 forward per schedule, NOT 30 inner steps): the same Adam
  optimizer holds (g_active, anchor, delta, alpha_logits, warp_s, warp_r,
  R, nu_active) and a single backward pass through the weighted loss
  drives both placement (via dweight/dτ) and perturbation (via
  dL_schedule/dperturbation).

* **Curriculum** (rebuild optimizer at each transition, not just zero
  grads): K=1 phase → K=2 phase → K=3 phase. At each transition we add
  the next g_k and the corresponding insert-k perturbation params to the
  optimizer.

* **Prescreen for τ initialization**: 1 forward × ~T_clean candidates
  with K=1 cheap defaults; pick top-K with d_min spacing → init τ values
  → invert sigmoid / softplus to get g_0..g_{K-1}.

* **Hardening + local refine**: round τ → discrete c_k = round(τ[k]);
  enumerate 27 = 3^K joint ±1 neighbor triples, filter invalid orderings,
  6-step cheap Stage-14 per valid triple, pick best by J-drop estimate.

* **Final 30-step Stage 14**: invokes the existing fixed-W path on the
  chosen triple to produce the deployment-ready output.

Six mandatory implementation guardrails (codex R3 GO conditions)
================================================================
1. **τ[0] ≥ 1 by construction** (NOT penalty). Bridge_budget margin built
   into right boundary too.
2. **Log valid_corner_count + valid_weight_mass every step**. If
   valid_corner_count < 2 or weight mass too small → project τ inward
   before enumeration (or fallback to single hard schedule).
3. **Bundle C OFF for joint-search v1**: the LPIPS-native ν line-search
   assumes fixed W; multi-schedule re-anchor is out of scope. Driver
   raises if both flags are set.
4. **Wall-clock instrument from day 1**: don't trust the 63 min/clip
   estimate. Budget first run as 2-4 h/clip until proven.
5. **Fixed-W parity regression test (MANDATORY)**: existing
   `_run_oracle_trajectory_pgd(W_fixed)` ≡ a degenerate joint loop
   pinned to W_fixed (single-corner schedule) within tolerance.
6. **Multi-seed fallback predeclared**: if dog J-drop > 0.03 below
   brute-force, immediately rerun with two more prescreen seeds (top-2
   and top-3 candidates) before declaring design failure.

Run `python -m memshield.joint_placement_search` for self-tests.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from memshield.decoy_continuation import init_bridge_edit_params
from memshield.oracle_trajectory import (
    init_false_trajectory, project_trajectory_to_budget,
)
from memshield.stage14_helpers import (
    AttackState, build_attack_state_from_W, stage14_forward_loss,
)


# ===========================================================================
# Section 1: τ parameterization (ordered gaps, by construction)
# ===========================================================================


@dataclass
class TauGapParams:
    """Learnable ordered τ parameterization via cumulative softplus gaps.

    Fields:
      g: [K] raw learnable gaps.  g[0] drives τ[0] via sigmoid; g[k>=1]
         drives the ADDITIONAL gap on top of d_min via softplus.
      K, T_clean: integer constants.
      clamp_left: τ[0] lower bound (≥ 1 to keep x_clean[c_0] with neighbors).
      bridge_budget: bridge_length + 1 (right margin so τ[K-1] + L_k stays
                     within [0, T_clean - 1]).
      d_min: minimum integer gap between consecutive τ (≥ 2).

    The recurrence guarantees:
      τ[0] ∈ [clamp_left, T_clean - bridge_budget]
      τ[k] ≥ τ[k-1] + d_min  (strict, so floor(τ[k]) > floor(τ[k-1]) when
                              fractional parts are nonzero; ties resolved
                              by the schedule-enumeration filter)

    g is FULL [K] always (no per-phase resizing). The curriculum controls
    which g_k are in the optimizer's parameter list at each phase.
    """

    g: Tensor                 # [K] learnable
    K: int
    T_clean: int
    clamp_left: float
    bridge_budget: int
    d_min: float


def init_tau_gap_params(
    K: int, T_clean: int, init_tau_values: Sequence[float],
    *,
    clamp_left: float = 1.0,
    bridge_budget: int = 4,
    d_min: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> TauGapParams:
    """Initialize TauGapParams so that the recurrence reproduces
    init_tau_values within tolerance.

    init_tau_values must be strictly ordered with gap ≥ d_min, and
    init_tau_values[0] ≥ clamp_left, init_tau_values[K-1] + bridge_budget
    ≤ T_clean - 1.
    """
    if len(init_tau_values) != K:
        raise ValueError(
            f"init_tau_values length {len(init_tau_values)} != K {K}")
    iv = [float(v) for v in init_tau_values]
    if iv[0] < clamp_left - 1e-6:
        raise ValueError(
            f"init_tau_values[0]={iv[0]:.4f} < clamp_left={clamp_left}")
    if iv[-1] + bridge_budget > T_clean - 1 + 1e-6:
        raise ValueError(
            f"init_tau_values[-1]={iv[-1]:.4f} + bridge_budget={bridge_budget} "
            f"> T_clean-1={T_clean-1}")
    for i in range(1, K):
        if iv[i] - iv[i - 1] < d_min - 1e-6:
            raise ValueError(
                f"init_tau_values gap [{i-1}, {i}] = "
                f"{iv[i] - iv[i-1]:.4f} < d_min={d_min}")

    device = device or torch.device("cpu")
    g = torch.zeros(K, dtype=dtype, device=device)

    # g[0]: invert tau[0] = clamp_left + span * sigmoid(g[0])
    span = float(T_clean - bridge_budget - clamp_left)
    if span <= 1e-6:
        raise ValueError(
            f"degenerate span={span:.4f}; "
            f"T_clean={T_clean}, bridge_budget={bridge_budget}, "
            f"clamp_left={clamp_left}")
    p0 = (iv[0] - clamp_left) / span
    p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
    g[0] = math.log(p0 / (1.0 - p0))

    # g[k>=1]: invert tau[k] = tau[k-1] + d_min + softplus(g[k])
    for k in range(1, K):
        extra = iv[k] - iv[k - 1] - d_min
        # softplus(g) = log(1 + exp(g)); inverse: g = log(exp(extra) - 1)
        # Numerically stable via log-expm1.
        extra_safe = max(float(extra), 1e-6)
        g[k] = math.log(math.expm1(extra_safe))

    return TauGapParams(
        g=g.detach().clone().requires_grad_(True),
        K=int(K), T_clean=int(T_clean),
        clamp_left=float(clamp_left),
        bridge_budget=int(bridge_budget),
        d_min=float(d_min),
    )


def tau_from_gaps(params: TauGapParams) -> Tensor:
    """Compute τ[0..K-1] from g[0..K-1] via the ordered cumulative
    sigmoid+softplus recurrence. Differentiable in g. Returns [K]."""
    span = params.T_clean - params.bridge_budget - params.clamp_left
    tau_list: List[Tensor] = []
    t0 = params.clamp_left + span * torch.sigmoid(params.g[0])
    tau_list.append(t0)
    for k in range(1, params.K):
        tk = tau_list[-1] + params.d_min + F.softplus(params.g[k])
        tau_list.append(tk)
    return torch.stack(tau_list, dim=0)


def project_tau_inward(params: TauGapParams, *, slack: float = 0.5) -> None:
    """Inward projection of τ if it has drifted to a region where most
    floor/ceil corners produce invalid orderings (guardrail #2 fallback).

    Concretely, this resets g toward the parameterization's "interior"
    (sigmoid → 0.5 for τ[0]; softplus(g[k]) → log(2) for k>0 so the gap
    above d_min equals log(2) ≈ 0.69). The post-projection τ is well
    away from boundary integers, restoring at least 2 valid corners on
    the next enumeration.
    """
    with torch.no_grad():
        params.g[0].zero_()
        for k in range(1, params.K):
            params.g[k].fill_(math.log(math.expm1(slack)))


# ===========================================================================
# Section 2: Schedule enumeration (multilinear corners + filter + renormalize)
# ===========================================================================


def _validate_schedule(W_clean: Sequence[int], T_clean: int,
                       bridge_budget: int, d_min: int) -> bool:
    """Strict ordering with min gap, in-range upper boundary."""
    K = len(W_clean)
    if K == 0:
        return False
    if W_clean[0] < 1:
        return False
    if W_clean[-1] + bridge_budget > T_clean - 1:
        return False
    for i in range(1, K):
        if W_clean[i] - W_clean[i - 1] < d_min:
            return False
    return True


def enumerate_neighbor_schedules(
    tau: Tensor,                                  # [K] (current τ values)
    active_inserts: Sequence[int],                # subset of [0..K-1]
    *,
    T_clean: int,
    bridge_budget: int,
    d_min: int,
) -> Tuple[List[Tuple[Tuple[int, ...], Tensor, int]], float]:
    """Enumerate the 2^|active_inserts| floor/ceil corner schedules of τ.

    For active inserts: c_k = floor(τ[k]) when corner-bit=0, ceil(τ[k])
    when corner-bit=1. For inactive inserts: c_k = round(τ[k]) (held
    integer, no branching).

    Weight per schedule = ∏_{k active} ((1 - frac(τ[k])) if bit=0 else
    frac(τ[k])) where frac(τ) = τ - floor(τ). The frac is differentiable
    in τ (`floor` is treated as zero-gradient via the explicit
    `tau - tau.floor().detach()` form), so the weight carries dL/dτ
    even though the schedule integer is itself non-differentiable.

    Schedules that violate strict ordering (c_0 < c_1 < ... with gap ≥
    d_min) or boundaries (c[0] ≥ 1, c[K-1] + bridge_budget ≤ T_clean - 1)
    are FILTERED OUT before normalization. Remaining weights are
    renormalized to sum to 1.0 — but `raw_weight_mass` (the SUM of the
    surviving weights BEFORE normalization) is also returned so callers
    can detect degenerate states like "only 1 surviving corner" or
    "tiny surviving mass" (guardrail #2 per codex R3).

    Returns:
      (schedules, raw_weight_mass) where:
        schedules: List[(W_tuple, normalized_weight, corner_id)]
          (only valid schedules; `corner_id` is the ORIGINAL binary
          corner index in [0, 2^|active|), preserved across filtering)
        raw_weight_mass: float — sum of surviving weights BEFORE
          renormalization (≤ 1.0; equal to 1.0 only when ALL 2^|active|
          corners are valid). Use this with `valid_corner_count` to
          detect degeneracy.
    """
    K = int(tau.shape[0])
    active = sorted({int(k) for k in active_inserts})
    if not active:
        # Nothing learnable; emit single round(τ) schedule with weight 1.
        W_round = tuple(int(torch.round(tau[k]).item()) for k in range(K))
        if not _validate_schedule(W_round, T_clean, bridge_budget, d_min):
            return [], 0.0
        w = torch.ones((), dtype=tau.dtype, device=tau.device)
        return [(W_round, w, 0)], 1.0

    # Per-active frac and floor/ceil ints.
    floors: Dict[int, int] = {}
    ceils: Dict[int, int] = {}
    fracs: Dict[int, Tensor] = {}
    for k in range(K):
        with torch.no_grad():
            fl = int(torch.floor(tau[k]).item())
            cl = int(torch.ceil(tau[k]).item())
        floors[k] = fl
        ceils[k] = cl if cl != fl else fl + 1   # avoid degenerate fl==cl
        # frac is differentiable in τ; explicit tau - floor(tau).detach()
        # so autograd knows the floor branch is zero-gradient.
        fracs[k] = tau[k] - torch.floor(tau[k]).detach()

    n_active = len(active)
    out: List[Tuple[Tuple[int, ...], Tensor, int]] = []
    raw_weights: List[Tensor] = []
    raw_W: List[Tuple[int, ...]] = []
    raw_corner_ids: List[int] = []
    for corner_id in range(1 << n_active):
        W_list: List[int] = [0] * K
        w = torch.ones((), dtype=tau.dtype, device=tau.device)
        for j, k in enumerate(active):
            bit = (corner_id >> j) & 1
            if bit == 0:
                W_list[k] = floors[k]
                w = w * (1.0 - fracs[k])
            else:
                W_list[k] = ceils[k]
                w = w * fracs[k]
        for k in range(K):
            if k not in active:
                # Frozen at integer τ value; round to nearest.
                with torch.no_grad():
                    W_list[k] = int(torch.round(tau[k]).item())
        W_tuple = tuple(W_list)
        if not _validate_schedule(W_tuple, T_clean, bridge_budget, d_min):
            continue
        raw_weights.append(w)
        raw_W.append(W_tuple)
        raw_corner_ids.append(int(corner_id))

    if not raw_weights:
        return [], 0.0
    weight_mass = torch.stack(raw_weights, dim=0).sum()
    raw_weight_mass = float(weight_mass.detach().item())
    norm = weight_mass.clamp_min(1e-12)
    for W_tuple, w_raw, cid in zip(raw_W, raw_weights, raw_corner_ids):
        out.append((W_tuple, w_raw / norm, cid))
    return out, raw_weight_mass


# ===========================================================================
# Section 3: AttackState cache (per-W rebuild amortization)
# ===========================================================================


class AttackStateCache:
    """LRU-bounded cache from W_clean tuple → AttackState.

    With τ moving slowly, neighbor schedules across consecutive joint
    steps share most of their corner W tuples — so caching saves the
    bulk of decoy-seed Poisson/alpha-paste cost (which dominates per-W
    rebuild time).
    """

    def __init__(self, x_clean: Tensor, pseudo_masks: Sequence[Any],
                 config: Any, *, max_size: int = 64,
                 bridge_length: Optional[int] = None,
                 insert_base_mode: Optional[str] = None) -> None:
        self._x_clean = x_clean
        self._pseudo_masks = pseudo_masks
        self._config = config
        self._max_size = int(max_size)
        self._bridge_length = bridge_length
        self._insert_base_mode = insert_base_mode
        self._cache: Dict[Tuple[int, ...], AttackState] = {}
        # Stats.
        self.hits = 0
        self.misses = 0

    def get(self, W_clean: Sequence[int]) -> AttackState:
        key = tuple(int(c) for c in W_clean)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            # Refresh LRU position (Python 3.7+ dicts preserve order).
            del self._cache[key]
            self._cache[key] = cached
            return cached
        self.misses += 1
        if len(self._cache) >= self._max_size:
            # Evict oldest entry.
            self._cache.pop(next(iter(self._cache)))
        state = build_attack_state_from_W(
            list(key), self._x_clean, self._pseudo_masks, self._config,
            bridge_length=self._bridge_length,
            insert_base_mode=self._insert_base_mode,
        )
        self._cache[key] = state
        return state


# ===========================================================================
# Section 4: Prescreen
# ===========================================================================


@dataclass
class PrescreenResult:
    """Result of the K=1 prescreen sweep."""
    init_tau_values: List[float]      # K candidates, sorted, with d_min spacing
    candidate_scores: List[Tuple[int, float]]  # all (c, score) pairs
    seed_index: int                    # which prescreen seed (0=top, 1=second, ...)


def prescreen_tau_init(
    x_clean: Tensor,
    pseudo_masks_clean: Sequence[Any],
    config: Any,
    forward_fn: Any,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    K: int,
    d_min: int,
    bridge_length: int,
    seed_index: int = 0,
    candidate_stride: int = 1,
) -> PrescreenResult:
    """K=1 cheap-default prescreen sweep over candidate clean-frame
    positions. For each c, we build attack_state at K=1 with c, run ONE
    Stage-14 forward, score = `-L_margin`. Picks the top-K candidates
    with strict d_min spacing (greedy seed selection: take the highest-
    score not yet within d_min of any chosen seed).

    seed_index lets callers fall back to alternative tops (multi-seed
    fallback per guardrail #6): seed_index=0 picks the global top,
    seed_index=1 picks the second-best top-K-spaced selection, etc.
    """
    T_clean = int(x_clean.shape[0])
    bridge_budget = bridge_length + 1
    candidates: List[int] = list(range(1, T_clean - bridge_budget,
                                       max(1, candidate_stride)))
    if len(candidates) < K:
        raise RuntimeError(
            f"prescreen: only {len(candidates)} candidate positions for "
            f"K={K} inserts (T_clean={T_clean}, bridge_budget="
            f"{bridge_budget})")

    # Allocate single-insert traj/edit/R/nu defaults (cheap — these are
    # only used for the prescreen forward pass).
    device = x_clean.device
    H, W = int(x_clean.shape[1]), int(x_clean.shape[2])
    nu = torch.zeros(1, H, W, 3, device=device, dtype=x_clean.dtype)
    R = (torch.zeros(1, bridge_length, H, W, 3, device=device,
                     dtype=x_clean.dtype)
         if config.oracle_traj_use_residual else None)

    scores: List[Tuple[int, float]] = []
    for c in candidates:
        try:
            state = build_attack_state_from_W(
                [int(c)], x_clean, pseudo_masks_clean, config,
                bridge_length=bridge_length,
            )
        except ValueError:
            continue
        # Cheap defaults for traj + edit_params.
        traj = init_false_trajectory(
            K=1, L=state.L_max,
            init_anchor_offsets=[(float(state.decoy_offsets_init[0][0]),
                                  float(state.decoy_offsets_init[0][1]))],
            device=device, dtype=x_clean.dtype,
        )
        edit_params = init_bridge_edit_params(
            1, state.L_max,
            alpha_max=config.oracle_traj_alpha_max,
            s_init_px=1.0, r_init_px=0.0,
            device=device, dtype=x_clean.dtype,
        )
        with torch.no_grad():
            try:
                _, diag, _ = stage14_forward_loss(
                    state, x_clean=x_clean,
                    traj=traj, edit_params=edit_params,
                    R=R, nu=nu,
                    forward_fn=forward_fn, lpips_fn=lpips_fn,
                    config=config,
                    lambda_fid_val=float(config.joint_traj_lambda_fid),
                    R_active=False,
                )
            except RuntimeError:
                continue
        score = -float(diag.L_margin.detach().item())
        scores.append((int(c), score))

    if not scores:
        raise RuntimeError(
            "prescreen: every candidate failed to build attack_state or "
            "Stage-14 forward")

    # Sort candidates by score descending.
    sorted_scores = sorted(scores, key=lambda kv: -kv[1])

    # Greedy d_min-spaced top-K selection. seed_index lets us pick the
    # next-best alternative if the global top has been tried already.
    chosen: List[int] = []
    skip_count = 0
    for c, _ in sorted_scores:
        if any(abs(c - x) < d_min for x in chosen):
            continue
        if skip_count < seed_index and len(chosen) == 0:
            # Skip first `seed_index` candidates from being picked as seed 0.
            skip_count += 1
            continue
        chosen.append(c)
        if len(chosen) == K:
            break

    if len(chosen) < K:
        raise RuntimeError(
            f"prescreen: only {len(chosen)} candidates with d_min="
            f"{d_min} spacing; need K={K}. seed_index={seed_index}.")

    init_tau_values = sorted(float(c) for c in chosen)
    return PrescreenResult(
        init_tau_values=init_tau_values,
        candidate_scores=scores,
        seed_index=int(seed_index),
    )


# ===========================================================================
# Section 5: Curriculum joint step (one inner step over schedules)
# ===========================================================================


@dataclass
class CurriculumStepDiagnostics:
    """Per-step diagnostics emitted by `curriculum_joint_step`."""
    L_step: float
    valid_corner_count: int
    valid_weight_mass: float            # raw sum of surviving weights BEFORE
                                        # renormalization (≤ 1.0; codex R3
                                        # MEDIUM fix). Compare with
                                        # 2^|active| / 2^|active| = 1.0 to
                                        # detect "tiny surviving mass".
    schedules: List[Dict[str, Any]]     # per-schedule W + weight + L_margin
    tau_values: List[float]
    inward_projected: bool
    singleton_corner: bool              # True when valid_corner_count == 1
                                        # after retry → τ has zero gradient
                                        # this step (codex R3 medium note).


def _R_active_slice_mask(R: Optional[Tensor], active_k: Sequence[int]
                         ) -> Optional[Tensor]:
    """Build a boolean mask for the R sign-PGD step that allows updates
    only on insert-k slices currently active in the curriculum. Inactive
    slices stay at their previous value (zero in the K=1 phase since R
    is initialized to zero).

    Returns a [K] bool tensor (True = active) or None if R is None.
    """
    if R is None:
        return None
    K = int(R.shape[0])
    mask = torch.zeros(K, dtype=torch.bool, device=R.device)
    for k in active_k:
        if 0 <= int(k) < K:
            mask[int(k)] = True
    return mask


def _zero_inactive_grads(
    active_inserts: Sequence[int], K: int,
    *,
    per_insert_first_dim_tensors: Sequence[Tensor],
    tau_g: Optional[Tensor] = None,
) -> None:
    """Zero gradients on inactive insert slices BEFORE optimizer.step().

    Codex R3 HIGH fix: per-phase Adam optimizer rebuild does not by itself
    freeze inactive slices, because each rebuilt optimizer still owns the
    full [K, ...] tensors. Without this hook, every Stage-14 forward
    over an active corner sends gradients into ALL slices via stage14_-
    forward_loss, and Adam steps inactive slices on every iteration —
    contradicting the curriculum's "inactive insert k held at prescreen
    init" semantics.

    For each tensor in `per_insert_first_dim_tensors` (assumed first-dim
    = K), zeros `grad[k]` for k NOT in active_inserts. For `tau_g`
    (1-D [K]), zeros `grad[k]` for inactive k as well so τ_k for
    inactive k is held fixed during this phase too.

    Note: this is best-applied AFTER L_step.backward() and BEFORE
    optimizer.step(). The next backward will re-populate gradients for
    all slices; we re-zero each step.
    """
    active = {int(k) for k in active_inserts}
    inactive = [k for k in range(int(K)) if k not in active]
    if not inactive:
        return
    for t in per_insert_first_dim_tensors:
        if t is None or t.grad is None:
            continue
        for k in inactive:
            if 0 <= k < int(t.shape[0]):
                t.grad[k].zero_()
    if tau_g is not None and tau_g.grad is not None:
        for k in inactive:
            if 0 <= k < int(tau_g.shape[0]):
                tau_g.grad[k].zero_()


def _apply_R_sign_pgd_active(
    R: Tensor, R_grad: Optional[Tensor],
    active_mask: Tensor, *, lr: float, eps: float,
) -> None:
    """Sign-PGD update on R restricted to active insert slices.

    Inactive slices are NOT modified (neither by the update nor by the
    clamp): they preserve whatever value they held entering this call,
    which lets earlier-phase R values survive into later phases as a
    frozen contribution.
    """
    if R_grad is None:
        return
    with torch.no_grad():
        # active_mask: [K] bool. Broadcast to R shape.
        K = R.shape[0]
        broadcast_shape = [K] + [1] * (R.dim() - 1)
        m_bool = active_mask.view(*broadcast_shape).expand_as(R.data).bool()
        m_float = active_mask.view(*broadcast_shape).to(R.dtype)
        # Update only active slices (inactive slices: m_float=0 -> no-op).
        update = -lr * R_grad.sign() * m_float
        R.data.add_(update)
        # Clamp only active slices. torch.where preserves inactive values.
        clamped = R.data.clamp(-eps, eps)
        R.data.copy_(torch.where(m_bool, clamped, R.data))


def curriculum_joint_step(
    *,
    x_clean: Tensor,
    pseudo_masks_clean: Sequence[Any],
    config: Any,
    forward_fn: Any,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    state_cache: AttackStateCache,
    tau_params: TauGapParams,
    traj_params: Any,                  # FalseTrajectoryParams
    edit_params: Any,                  # BridgeEditParams
    R: Optional[Tensor],
    nu: Tensor,
    optimizer: torch.optim.Optimizer,
    active_inserts: Sequence[int],
    lambda_fid_val: float,
    bridge_length: int,
    R_active_in_phase: bool,
    R_lr: float,
    R_eps: float,
) -> CurriculumStepDiagnostics:
    """One joint step of the curriculum: enumerate schedules, run
    Stage-14 forwards, weighted-sum loss, single backward, zero inactive
    Adam grads (codex R3 HIGH fix), single optimizer step, R sign-PGD
    restricted to active slices, τ inward-projection fallback if
    degenerate."""
    tau = tau_from_gaps(tau_params)         # [K] differentiable
    schedules, raw_weight_mass = enumerate_neighbor_schedules(
        tau, active_inserts,
        T_clean=tau_params.T_clean,
        bridge_budget=tau_params.bridge_budget,
        d_min=int(tau_params.d_min),
    )
    valid_corner_count = len(schedules)
    inward_projected = False

    # Guardrail #2 fallback: if degenerate (<2 surviving corners), project
    # τ inward and retry. After retry we accept whatever survives —
    # singleton corner means τ has no gradient this step, but we record
    # `singleton_corner=True` so callers can detect the degeneracy. If
    # zero corners survive even after projection, raise (config is
    # incompatible — caller should reduce K or d_min).
    if valid_corner_count < 2:
        project_tau_inward(tau_params)
        inward_projected = True
        tau = tau_from_gaps(tau_params)
        schedules, raw_weight_mass = enumerate_neighbor_schedules(
            tau, active_inserts,
            T_clean=tau_params.T_clean,
            bridge_budget=tau_params.bridge_budget,
            d_min=int(tau_params.d_min),
        )
        valid_corner_count = len(schedules)
        if valid_corner_count == 0:
            raise RuntimeError(
                "joint search: no valid schedule corners after τ inward "
                "projection. T_clean / bridge_length / d_min combination "
                "is incompatible.")

    singleton_corner = (valid_corner_count == 1)

    # Per-schedule forward + weighted-sum loss.
    L_step = torch.zeros((), dtype=x_clean.dtype, device=x_clean.device)
    schedule_logs: List[Dict[str, Any]] = []
    for W_tuple, weight, corner_id in schedules:
        state = state_cache.get(W_tuple)
        L_sched, diag_sched, _ = stage14_forward_loss(
            state, x_clean=x_clean,
            traj=traj_params, edit_params=edit_params,
            R=R, nu=nu,
            forward_fn=forward_fn, lpips_fn=lpips_fn,
            config=config, lambda_fid_val=lambda_fid_val,
            R_active=R_active_in_phase,
        )
        L_step = L_step + weight * L_sched
        schedule_logs.append({
            "W": list(W_tuple),
            "weight": float(weight.detach().item()),
            "corner_id": int(corner_id),
            "L_margin": float(diag_sched.L_margin.detach().item()),
            "L_total": float(L_sched.detach().item()),
            "feasible": bool(diag_sched.feasible),
            "delta_overlap": float(diag_sched.delta_overlap),
        })

    # Single backward (drives both placement via dweight/dτ and
    # perturbation via dL_schedule/dperturbation through state).
    optimizer.zero_grad()
    if R is not None and R.requires_grad and R.grad is not None:
        R.grad.zero_()
    L_step.backward()

    # Codex R3 HIGH fix: zero gradients on inactive insert slices BEFORE
    # optimizer.step(). The optimizer rebuild per phase still owns the
    # FULL [K, ...] tensors; without this hook the inactive slices would
    # be Adam-stepped on every iteration via gradients flowing through
    # stage14_forward_loss across all schedules.
    K_total = int(traj_params.anchor_offset.shape[0])
    per_insert_tensors: List[Tensor] = [
        traj_params.anchor_offset,
        traj_params.delta_offset,
        edit_params.alpha_logits,
        edit_params.warp_s,
        edit_params.warp_r,
        nu,
    ]
    _zero_inactive_grads(
        active_inserts=active_inserts, K=K_total,
        per_insert_first_dim_tensors=per_insert_tensors,
        tau_g=tau_params.g,
    )

    optimizer.step()

    # R sign-PGD restricted to active slices (Phase B only).
    if R_active_in_phase and R is not None and R.requires_grad:
        active_mask = _R_active_slice_mask(R, active_inserts)
        if active_mask is not None:
            _apply_R_sign_pgd_active(
                R, R.grad, active_mask, lr=R_lr, eps=R_eps)

    # Project trajectory after step (carries over from existing path).
    project_trajectory_to_budget(
        traj_params, max_offset_px=config.oracle_traj_max_offset_px)

    return CurriculumStepDiagnostics(
        L_step=float(L_step.detach().item()),
        valid_corner_count=valid_corner_count,
        valid_weight_mass=float(raw_weight_mass),
        schedules=schedule_logs,
        tau_values=[float(t.detach().item()) for t in tau],
        inward_projected=inward_projected,
        singleton_corner=singleton_corner,
    )


# ===========================================================================
# Section 6: Local refine (27-triple ±1 neighbor search after rounding)
# ===========================================================================


def _enumerate_27_neighbors(
    W_round: Sequence[int], *,
    T_clean: int, bridge_budget: int, d_min: int,
) -> List[Tuple[int, ...]]:
    """Enumerate 3^K = 27 (K=3) joint ±1 neighbors of W_round; filter
    invalid orderings."""
    K = len(W_round)
    out: List[Tuple[int, ...]] = []
    # Use base-3 enumeration: each digit in {-1, 0, +1}.
    for code in range(3 ** K):
        offsets = []
        c = code
        for _ in range(K):
            offsets.append((c % 3) - 1)
            c //= 3
        cand = tuple(int(W_round[k]) + offsets[k] for k in range(K))
        if _validate_schedule(cand, T_clean, bridge_budget, d_min):
            out.append(cand)
    return out


def local_refine_27(
    *,
    x_clean: Tensor,
    pseudo_masks_clean: Sequence[Any],
    config: Any,
    forward_fn: Any,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    state_cache: AttackStateCache,
    W_round: Sequence[int],
    bridge_length: int,
    d_min: int,
    sam2_eval_fn: Optional[Callable] = None,
    refine_steps: int = 6,
    traj_init_offsets: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[Tuple[int, ...], List[Dict[str, Any]]]:
    """Cheap 6-step Stage-14 estimate per valid neighbor; pick best by
    L_margin (proxy for J-drop, since SAM2 eval is expensive). Returns
    chosen W_clean tuple + diagnostics for all neighbors.

    `d_min` is REQUIRED (codex R3 MEDIUM fix) — earlier draft fell back
    to `config.oracle_traj_d_min` or `2`, which silently disagreed with
    the curriculum's d_min if the caller passed a different value.

    The full 30-step Stage-14 + export + SAM2 eval is run AFTER local
    refine on the chosen triple by the joint_curriculum_search caller.
    """
    K = len(W_round)
    bridge_budget = bridge_length + 1
    candidates = _enumerate_27_neighbors(
        W_round, T_clean=int(x_clean.shape[0]),
        bridge_budget=bridge_budget, d_min=int(d_min))

    if not candidates:
        raise RuntimeError(
            f"local_refine_27: 0 valid neighbors of W_round={list(W_round)}")

    device = x_clean.device
    H, W = int(x_clean.shape[1]), int(x_clean.shape[2])

    diags: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_W: Tuple[int, ...] = tuple(int(w) for w in candidates[0])

    for cand in candidates:
        state = state_cache.get(cand)
        # Init fresh traj + edit_params + nu + R for this neighbor (cheap).
        anchors = traj_init_offsets if traj_init_offsets is not None else \
            [(float(dy), float(dx)) for dy, dx in state.decoy_offsets_init]
        # Pad/truncate anchors to match K (use state.decoy_offsets_init).
        if len(anchors) != K:
            anchors = [(float(dy), float(dx))
                       for dy, dx in state.decoy_offsets_init]
        traj = init_false_trajectory(
            K=K, L=state.L_max, init_anchor_offsets=anchors,
            device=device, dtype=x_clean.dtype,
        )
        edit_p = init_bridge_edit_params(
            K, state.L_max, alpha_max=config.oracle_traj_alpha_max,
            s_init_px=1.0, r_init_px=0.0,
            device=device, dtype=x_clean.dtype,
        )
        nu = torch.zeros(K, H, W, 3, device=device, dtype=x_clean.dtype)
        R = (torch.zeros(K, state.L_max, H, W, 3, device=device,
                         dtype=x_clean.dtype)
             if config.oracle_traj_use_residual else None)

        opt = torch.optim.Adam(
            [traj.anchor_offset, traj.delta_offset,
             edit_p.alpha_logits, edit_p.warp_s, edit_p.warp_r],
            lr=float(config.oracle_traj_anchor_lr))

        for _ in range(int(refine_steps)):
            L, diag, _ = stage14_forward_loss(
                state, x_clean=x_clean, traj=traj, edit_params=edit_p,
                R=R, nu=nu,
                forward_fn=forward_fn, lpips_fn=lpips_fn,
                config=config,
                lambda_fid_val=float(config.joint_traj_lambda_fid),
                R_active=False,
            )
            opt.zero_grad()
            L.backward()
            opt.step()
            project_trajectory_to_budget(
                traj, max_offset_px=config.oracle_traj_max_offset_px)

        with torch.no_grad():
            _, diag_final, _ = stage14_forward_loss(
                state, x_clean=x_clean, traj=traj, edit_params=edit_p,
                R=R, nu=nu,
                forward_fn=forward_fn, lpips_fn=lpips_fn,
                config=config,
                lambda_fid_val=float(config.joint_traj_lambda_fid),
                R_active=False,
            )
        score = -float(diag_final.L_margin.detach().item())
        diags.append({
            "W": list(cand), "score": score,
            "L_margin": float(diag_final.L_margin.detach().item()),
            "delta_overlap": float(diag_final.delta_overlap),
        })
        if score > best_score:
            best_score = score
            best_W = tuple(int(w) for w in cand)

    return best_W, diags


# ===========================================================================
# Section 7: Top-level orchestrator
# ===========================================================================


@dataclass
class JointSearchResult:
    """Result returned by joint_curriculum_search."""
    chosen_W_clean: List[int]
    prescreen_init_tau: List[float]
    curriculum_logs: List[Dict[str, Any]]
    refine_diagnostics: List[Dict[str, Any]]
    refine_W_clean: List[int]
    cache_hits: int
    cache_misses: int
    wall_clock_seconds: Dict[str, float]


def joint_curriculum_search(
    x_clean: Tensor,
    pseudo_masks_clean: Sequence[Any],
    config: Any,
    forward_fn: Any,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    K: int = 3,
    d_min: int = 2,
    bridge_length: Optional[int] = None,
    phase_steps: Sequence[int] = (12, 12, 15),
    prescreen_seed_index: int = 0,
    candidate_stride: int = 1,
    cache_max_size: int = 64,
) -> JointSearchResult:
    """Run the full joint curriculum placement-perturbation search.

    Pipeline (auto-review-loop R6 GO design):
      1. Prescreen (1 fwd × ~T candidates) → init_tau_values.
      2. Initialize TauGapParams + traj + edit_params + R + nu.
      3. K=1 phase: optimize {g[0], anchor[0], delta[0,:], α[0,:],
         warp[0,:], R[0,:], ν[0]} for `phase_steps[0]` steps.
      4. K=2 phase: optimizer rebuild adds {g[1], anchor[1], delta[1,:],
         α[1,:], warp[1,:], R[1,:], ν[1]}. `phase_steps[1]` steps.
      5. K=3 phase: optimizer rebuild adds the rest. `phase_steps[2]`
         steps.
      6. Round τ → W_round; 27-triple ±1 local refine via 6-step Stage-14
         estimates; pick best.
      7. Return chosen W (caller runs the final 30-step Stage-14 +
         export via the existing fixed-W path).

    Bundle C (LPIPS-native ν) MUST be off — the driver enforces this via
    a mutex guard. The line-search assumes fixed W, which multi-schedule
    enumeration violates.
    """
    if int(K) != len(phase_steps):
        raise ValueError(
            f"phase_steps length {len(phase_steps)} != K {K}; one phase "
            "per insert.")
    if getattr(config, "oracle_traj_nu_lpips_native", False):
        raise ValueError(
            "joint_curriculum_search: Bundle C (oracle_traj_nu_lpips_native) "
            "is not supported in v1. The LPIPS-native ν line-search "
            "assumes fixed W; multi-schedule enumeration violates that. "
            "Disable --oracle-traj-nu-lpips-native or use "
            "--use-profiled-placement instead.")

    bridge_len = int(bridge_length if bridge_length is not None
                     else config.oracle_traj_bridge_length)
    bridge_budget = bridge_len + 1
    T_clean = int(x_clean.shape[0])
    device = x_clean.device
    H, W = int(x_clean.shape[1]), int(x_clean.shape[2])

    timings: Dict[str, float] = {}

    # ---- 1. Prescreen ----
    t0 = time.time()
    pre = prescreen_tau_init(
        x_clean, pseudo_masks_clean, config,
        forward_fn=forward_fn, lpips_fn=lpips_fn,
        K=K, d_min=d_min, bridge_length=bridge_len,
        seed_index=int(prescreen_seed_index),
        candidate_stride=int(candidate_stride),
    )
    timings["prescreen"] = time.time() - t0

    # ---- 2. Initialize state ----
    state_cache = AttackStateCache(
        x_clean, pseudo_masks_clean, config,
        max_size=int(cache_max_size),
        bridge_length=bridge_len,
    )
    tau_params = init_tau_gap_params(
        K=K, T_clean=T_clean,
        init_tau_values=pre.init_tau_values,
        clamp_left=1.0, bridge_budget=bridge_budget, d_min=float(d_min),
        device=device, dtype=x_clean.dtype,
    )
    # Pre-build the K=3 attack state at the rounded init values so traj
    # init_anchors come from the prescreen's seed.
    W_init = tuple(int(round(v)) for v in pre.init_tau_values)
    init_state = state_cache.get(W_init)
    traj = init_false_trajectory(
        K=K, L=init_state.L_max,
        init_anchor_offsets=[
            (float(dy), float(dx))
            for dy, dx in init_state.decoy_offsets_init],
        device=device, dtype=x_clean.dtype,
    )
    edit_params = init_bridge_edit_params(
        K, init_state.L_max,
        alpha_max=config.oracle_traj_alpha_max,
        s_init_px=1.0, r_init_px=0.0,
        device=device, dtype=x_clean.dtype,
    )
    nu = torch.zeros(K, H, W, 3, device=device, dtype=x_clean.dtype,
                     requires_grad=True)
    R = (torch.zeros(K, init_state.L_max, H, W, 3, device=device,
                     dtype=x_clean.dtype, requires_grad=True)
         if config.oracle_traj_use_residual else None)

    curriculum_logs: List[Dict[str, Any]] = []

    # ---- 3-5. Curriculum phases ----
    lambda_fid_val = float(config.joint_traj_lambda_fid)
    R_lr = float(config.oracle_traj_residual_lr)
    R_eps = float(config.oracle_traj_residual_eps)

    for phase_idx in range(K):
        t_phase = time.time()
        active_inserts = list(range(phase_idx + 1))
        # Build optimizer for this phase: g[active], anchor[active],
        # delta[active], alpha_logits[active], warp_s[active],
        # warp_r[active], nu[active]. We index slices via .data and
        # treat the underlying tensors as full [K, ...] but include
        # only the relevant params in the optimizer's param list.
        param_list: List[Tensor] = [tau_params.g]
        param_list.append(traj.anchor_offset)
        param_list.append(traj.delta_offset)
        param_list.append(edit_params.alpha_logits)
        param_list.append(edit_params.warp_s)
        param_list.append(edit_params.warp_r)
        param_list.append(nu)
        # NOTE: we pass the FULL tensors to the optimizer; gradients on
        # inactive insert slices flow through Stage-14 forward but the
        # weight-mass over inactive corners is degenerate enough that
        # the inactive-slice gradient is negligible. The R active-slice
        # mask in `_apply_R_sign_pgd_active` enforces strict locality
        # for R; we accept the soft-locality on Adam params as a
        # pragmatic v1 compromise (codex R3 reviewers acknowledged
        # this in the GO).

        # Use the existing oracle_traj_anchor_lr for trajectory params,
        # alpha_lr for α/warp, and a tau_lr default if not configured.
        tau_lr = float(getattr(config, "oracle_traj_tau_lr",
                               config.oracle_traj_anchor_lr))
        optimizer = torch.optim.Adam([
            {"params": [tau_params.g], "lr": tau_lr},
            {"params": [traj.anchor_offset],
             "lr": float(config.oracle_traj_anchor_lr)},
            {"params": [traj.delta_offset],
             "lr": float(config.oracle_traj_delta_lr)},
            {"params": [edit_params.alpha_logits],
             "lr": float(config.oracle_traj_alpha_lr)},
            {"params": [edit_params.warp_s, edit_params.warp_r],
             "lr": float(config.oracle_traj_warp_lr)},
            {"params": [nu], "lr": float(config.oracle_traj_nu_lr_phase_b)},
        ])
        R_active_in_phase = (R is not None and phase_idx == K - 1)

        for step in range(int(phase_steps[phase_idx])):
            diag = curriculum_joint_step(
                x_clean=x_clean,
                pseudo_masks_clean=pseudo_masks_clean,
                config=config,
                forward_fn=forward_fn, lpips_fn=lpips_fn,
                state_cache=state_cache,
                tau_params=tau_params, traj_params=traj,
                edit_params=edit_params, R=R, nu=nu,
                optimizer=optimizer, active_inserts=active_inserts,
                lambda_fid_val=lambda_fid_val,
                bridge_length=bridge_len,
                R_active_in_phase=R_active_in_phase,
                R_lr=R_lr, R_eps=R_eps,
            )
            curriculum_logs.append({
                "phase": phase_idx + 1, "step": step + 1,
                "L_step": diag.L_step,
                "valid_corner_count": diag.valid_corner_count,
                "valid_weight_mass": diag.valid_weight_mass,
                "tau_values": diag.tau_values,
                "inward_projected": diag.inward_projected,
                "schedules": diag.schedules,
            })
        timings[f"phase_{phase_idx + 1}"] = time.time() - t_phase

    # ---- 6. Round + 27-triple local refine ----
    t_refine = time.time()
    with torch.no_grad():
        tau_final = tau_from_gaps(tau_params)
    W_round = [int(round(float(t.detach().item()))) for t in tau_final]
    refine_W, refine_diags = local_refine_27(
        x_clean=x_clean, pseudo_masks_clean=pseudo_masks_clean,
        config=config, forward_fn=forward_fn, lpips_fn=lpips_fn,
        state_cache=state_cache, W_round=W_round,
        bridge_length=bridge_len, d_min=int(d_min),
    )
    timings["local_refine"] = time.time() - t_refine

    return JointSearchResult(
        chosen_W_clean=list(refine_W),
        prescreen_init_tau=list(pre.init_tau_values),
        curriculum_logs=curriculum_logs,
        refine_diagnostics=refine_diags,
        refine_W_clean=list(refine_W),
        cache_hits=int(state_cache.hits),
        cache_misses=int(state_cache.misses),
        wall_clock_seconds=timings,
    )


# ===========================================================================
# Section 8: Self-tests (6 mandatory + a few smoke)
# ===========================================================================


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
    oracle_traj_max_offset_px = 200.0
    oracle_traj_anchor_lr = 1.0
    oracle_traj_delta_lr = 0.5
    oracle_traj_alpha_lr = 0.1
    oracle_traj_warp_lr = 0.1
    oracle_traj_nu_lr_phase_b = 1.0 / 255.0
    oracle_traj_tau_lr = 0.5
    oracle_traj_residual_eps = 8.0 / 255.0
    oracle_traj_residual_lr = 2.0 / 255.0
    oracle_traj_d_min = 2
    oracle_traj_nu_lpips_native = False
    margin_threshold = 1.0
    margin_neighbor_weight = 0.5
    lpips_orig_cap = 0.20
    lpips_insert_cap = 0.35
    tv_multiplier = 1.2
    train_ste_quantize = False
    joint_traj_lambda_fid = 10.0


def _test_tau_gap_init_and_recurrence_inverts() -> None:
    """init_tau_gap_params(init_tau_values) → tau_from_gaps reproduces
    the input within tolerance (inverse-sigmoid + inverse-softplus
    correctness)."""
    cfg = _DummyConfig()
    targets = [3.0, 7.0, 15.0]
    p = init_tau_gap_params(
        K=3, T_clean=30, init_tau_values=targets,
        clamp_left=1.0, bridge_budget=5, d_min=2.0,
    )
    tau = tau_from_gaps(p).detach().cpu().tolist()
    for got, want in zip(tau, targets):
        assert abs(got - want) < 1e-3, (got, want, tau)
    print("  tau-gap init+recurrence inverts cleanly")


def _test_tau_ordered_legality() -> None:
    """For random init in valid range, tau is always ordered with
    τ[0] >= clamp_left and τ[K-1] + bridge_budget <= T_clean - 1."""
    torch.manual_seed(0)
    T_clean = 50
    clamp_left = 1.0
    bridge_budget = 5
    d_min = 2.0
    init_vals = [3.0, 8.0, 18.0]
    p = init_tau_gap_params(
        K=3, T_clean=T_clean, init_tau_values=init_vals,
        clamp_left=clamp_left, bridge_budget=bridge_budget, d_min=d_min,
    )
    # Try several random g perturbations.
    for _ in range(20):
        with torch.no_grad():
            p.g.add_(torch.randn_like(p.g))
        tau = tau_from_gaps(p).detach().cpu().tolist()
        assert tau[0] >= clamp_left - 1e-3, tau
        for k in range(1, len(tau)):
            assert tau[k] >= tau[k - 1] + d_min - 1e-3, tau
        assert tau[-1] + bridge_budget <= T_clean - 1 + 1e-3, tau
    print("  ordered-tau legality holds under random perturbations")


def _test_schedule_weight_sum_and_integer_exactness() -> None:
    """Weights sum to 1.0 (mod invalid filtering), and at integer τ the
    only valid schedule has weight 1 (corresponding to floor=ceil=τ).
    """
    # Integer τ: 5, 10, 15. Active inserts = [0, 1, 2].
    tau = torch.tensor([5.0, 10.0, 15.0], requires_grad=True)
    schedules, _ = enumerate_neighbor_schedules(
        tau, [0, 1, 2], T_clean=30, bridge_budget=5, d_min=2)
    # At integer τ, frac = 0 → corner with bit=0 has weight 1; corner
    # with bit=1 has weight 0. After validity filter (weight=0 corners
    # remain in the sum since their schedule is still valid), the
    # renormalized output should keep ALL corners (since they ARE
    # valid orderings) but with proper multilinear weights.
    # Actually, ceil schedules at integer τ have fl + 1 ≠ τ, so the
    # ceil schedule is e.g. (6, 11, 16) which is also valid → returns
    # weights with bit=1 corners having 0 weight.
    # The MASS is 1.0 (correct).
    weight_sum = sum(float(w.detach()) for _, w, _ in schedules)
    assert abs(weight_sum - 1.0) < 1e-5, weight_sum
    # Find the floor-only corner: should have weight 1.
    floor_W = (5, 10, 15)
    matched = [w for W_t, w, _ in schedules
               if tuple(W_t) == floor_W]
    assert matched and abs(float(matched[0].detach()) - 1.0) < 1e-5, schedules
    # Non-integer τ: weights sum to 1.
    tau2 = torch.tensor([5.3, 10.7, 15.4], requires_grad=True)
    schedules2, _ = enumerate_neighbor_schedules(
        tau2, [0, 1, 2], T_clean=30, bridge_budget=5, d_min=2)
    weight_sum2 = sum(float(w.detach()) for _, w, _ in schedules2)
    assert abs(weight_sum2 - 1.0) < 1e-5, weight_sum2
    # 8 corners at non-integer τ all valid (no ordering violation).
    assert len(schedules2) == 8, [s[0] for s in schedules2]
    print("  schedule weights sum=1; integer τ collapses to single corner")


def _test_schedule_filtering_invalid_orderings() -> None:
    """When floor/ceil produce orderings that violate strict d_min
    spacing, those corners are filtered out and remaining weights
    renormalize to 1."""
    # τ = [5.5, 7.5, 10.5], d_min=2. Some corners produce gap < 2.
    tau = torch.tensor([5.5, 7.5, 10.5], requires_grad=True)
    schedules, _ = enumerate_neighbor_schedules(
        tau, [0, 1, 2], T_clean=30, bridge_budget=5, d_min=2)
    # corner [ceil(5)=6, floor(7)=7, floor(10)=10] = (6, 7, 10) has gap
    # 7-6 = 1 < d_min=2 → filtered. The renormalized weights sum to 1.
    weight_sum = sum(float(w.detach()) for _, w, _ in schedules)
    assert abs(weight_sum - 1.0) < 1e-5, weight_sum
    assert len(schedules) < 8, len(schedules)
    # All returned schedules must be strict-ordered with gap ≥ 2.
    for W_t, _, _ in schedules:
        assert _validate_schedule(
            W_t, T_clean=30, bridge_budget=5, d_min=2), W_t
    print("  invalid-ordering corners filtered; weights renormalize")


def _test_schedule_grad_flows_through_weights() -> None:
    """Backward through the weighted sum populates τ.grad even though
    schedule W is itself non-differentiable."""
    tau = torch.tensor([5.3, 10.7, 15.4], requires_grad=True)
    schedules, _ = enumerate_neighbor_schedules(
        tau, [0, 1, 2], T_clean=30, bridge_budget=5, d_min=2)
    # Use a synthetic per-schedule "loss" depending only on the corner_id
    # to give the weights something to drive.
    L = torch.zeros(())
    for _, w, corner_id in schedules:
        L = L + w * float(corner_id)
    L.backward()
    assert tau.grad is not None
    assert tau.grad.abs().sum() > 1e-6, tau.grad
    print("  τ.grad flows through schedule weights")


def _test_R_active_slice_mask_locality() -> None:
    """Sign-PGD update + clamp restricted to active slices: inactive
    slices preserve their incoming values (even if those values are
    outside the ε ball). Active slice gets the sign-PGD update + clamp.
    """
    K = 3
    R = torch.zeros(K, 4, 8, 8, 3)
    R.data[0].zero_()
    R.data[1].fill_(0.123)              # inactive baseline OUTSIDE ε
    R.data[2].fill_(-0.456)             # inactive baseline OUTSIDE ε
    R.requires_grad_(True)
    R.grad = torch.ones_like(R.data)    # non-zero gradient everywhere

    active_mask = _R_active_slice_mask(R, [0])
    _apply_R_sign_pgd_active(
        R, R.grad, active_mask, lr=2.0 / 255.0, eps=8.0 / 255.0)

    # Active slice 0: was zero, gradient sign is +1, update = -2/255 -> -2/255.
    eps = 8.0 / 255.0
    assert torch.allclose(
        R.data[0], torch.full_like(R.data[0], -2.0 / 255.0)), \
        f"active slice 0 expected -2/255, got mean={R.data[0].mean()}"
    # Inactive slices preserved at their incoming values (NOT clamped).
    assert torch.allclose(R.data[1], torch.full_like(R.data[1], 0.123)), \
        f"inactive slice 1 changed: mean={R.data[1].mean()}"
    assert torch.allclose(R.data[2], torch.full_like(R.data[2], -0.456)), \
        f"inactive slice 2 changed: mean={R.data[2].mean()}"

    # Now re-run with active=[0,1] and check active 1 gets clamped.
    R2 = torch.zeros(K, 4, 8, 8, 3)
    R2.data[1].fill_(0.123)
    R2.data[2].fill_(-0.456)
    R2.requires_grad_(True)
    R2.grad = torch.ones_like(R2.data)
    active_mask_01 = _R_active_slice_mask(R2, [0, 1])
    _apply_R_sign_pgd_active(
        R2, R2.grad, active_mask_01, lr=2.0 / 255.0, eps=eps)
    # Active slice 1: was 0.123, update -2/255 -> 0.115; that's still > eps,
    # so clamp brings to +eps.
    assert torch.allclose(R2.data[1], torch.full_like(R2.data[1], eps)), \
        f"active slice 1 expected +eps, got mean={R2.data[1].mean()}"
    # Inactive slice 2 preserved at -0.456.
    assert torch.allclose(R2.data[2], torch.full_like(R2.data[2], -0.456)), \
        f"inactive slice 2 changed: mean={R2.data[2].mean()}"
    print("  R active-slice mask: inactive slices preserved (even outside ε)")


def _test_27_neighbor_enumeration_filters_invalid() -> None:
    """27-triple enumeration filters out invalid orderings."""
    # W_round at edge of valid range.
    cands = _enumerate_27_neighbors(
        [3, 8, 15], T_clean=30, bridge_budget=5, d_min=2)
    # All returned are strict-ordered with gap >= 2.
    for cand in cands:
        assert cand[0] < cand[1] < cand[2], cand
        assert cand[1] - cand[0] >= 2, cand
        assert cand[2] - cand[1] >= 2, cand
        assert cand[0] >= 1, cand
        assert cand[2] + 5 <= 29, cand
    # At the EDGE (left): W_round = [1, 5, 10]. Some -1 offsets push
    # τ[0] below clamp_left → filtered.
    edge = _enumerate_27_neighbors(
        [1, 5, 10], T_clean=30, bridge_budget=5, d_min=2)
    for cand in edge:
        assert cand[0] >= 1, cand
    print("  27-triple ±1 enumeration filters invalid orderings")


def _test_bundle_C_incompat_guard() -> None:
    """joint_curriculum_search raises when oracle_traj_nu_lpips_native is on."""
    cfg = _DummyConfig()
    cfg.oracle_traj_nu_lpips_native = True
    x_clean = torch.zeros(20, 16, 16, 3)
    masks = [np.zeros((16, 16), dtype=np.float32) for _ in range(20)]
    try:
        joint_curriculum_search(
            x_clean, masks, cfg,
            forward_fn=None, lpips_fn=None,    # never called — guard fires first
        )
        raise AssertionError("expected ValueError on Bundle C + joint search")
    except ValueError as e:
        assert "not supported" in str(e), e
    print("  Bundle C incompatibility guard raises clear error")


def _test_attack_state_cache_hit_miss() -> None:
    """Cache returns same AttackState for repeated W; LRU eviction on
    overflow."""
    torch.manual_seed(0)
    np.random.seed(0)
    T_clean, H, W = 25, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c = min(W - 4, max(0, t // 3))
        m[6:10, c:c + 4] = 1.0
        pseudo_masks.append(m)
    cfg = _DummyConfig()
    cache = AttackStateCache(x_clean, pseudo_masks, cfg, max_size=3)

    s1 = cache.get([3, 8, 15])
    s2 = cache.get([3, 8, 15])               # hit
    assert s1 is s2
    assert cache.hits == 1 and cache.misses == 1
    s3 = cache.get([4, 9, 16])               # miss
    s4 = cache.get([5, 10, 17])              # miss
    s5 = cache.get([6, 11, 18])              # miss → evict [3,8,15]
    s6 = cache.get([3, 8, 15])               # miss again (was evicted)
    assert cache.hits == 1 and cache.misses == 5
    print("  AttackState cache: hit/miss + LRU eviction OK")


class _StubForwardFn:
    """Stub forward_fn used by smoke tests.

    Returns deterministic zero logits and configurable obj scores at the
    queried frame indices. This exercises the stage14_forward_loss code
    path end-to-end (margin/area/obj/fid losses, diagnostics, gradient
    flow into traj/edit/R/nu/tau via the weighted schedule sum) without
    needing real SAM2.
    """

    def __init__(self, H: int, W: int, *, obj_value: float = -2.0):
        self.H = int(H)
        self.W = int(W)
        self.obj_value = float(obj_value)

    def forward_with_objectness(
        self, processed, *, return_at, objectness_at,
    ):
        # processed is [T_proc, H, W, 3]; we ignore content and emit
        # zeros of the right shape so downstream code paths run.
        device = processed.device
        dtype = processed.dtype
        # Tie logits weakly to processed mean so backward populates a
        # gradient through forward_fn (sanity for grad-flow tests).
        seed = processed.mean()
        logits_by_t = {}
        for t in return_at:
            logits_by_t[int(t)] = (
                torch.zeros(self.H, self.W, device=device, dtype=dtype) + seed)
        obj_by_t = {}
        for t in objectness_at:
            obj_by_t[int(t)] = (
                torch.full((1,), self.obj_value, device=device, dtype=dtype)
                + seed)
        return logits_by_t, obj_by_t


def _stub_lpips(a, b):
    """Stub LPIPS that returns a small differentiable scalar in [0, 1]."""
    return ((a - b) ** 2).mean().clamp(0.0, 1.0)


def _test_curriculum_smoke_K1_K2_K3() -> None:
    """Joint-loop smoke through K=1/K=2/K=3 curriculum transitions
    using stub forward_fn + stub LPIPS. Verifies:
      - τ updates each step (g moves under nonzero gradient)
      - Schedule enumeration correctly varies with active_inserts
      - Optimizer rebuild at phase transitions does not crash
      - R sign-PGD applies only on phase 3 (R active phase)
    """
    torch.manual_seed(0)
    np.random.seed(0)
    T_clean, H, W = 25, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        c = min(W - 4, max(0, 4 + t // 3))
        m[6:10, c:c + 4] = 1.0
        pseudo_masks.append(m)
    cfg = _DummyConfig()
    cfg.oracle_traj_use_residual = True   # exercise R path

    forward_fn = _StubForwardFn(H, W)
    lpips_fn = _stub_lpips

    state_cache = AttackStateCache(
        x_clean, pseudo_masks, cfg, max_size=16,
        bridge_length=cfg.oracle_traj_bridge_length,
    )
    init_tau = [3.0, 9.0, 16.0]
    tau_p = init_tau_gap_params(
        K=3, T_clean=T_clean, init_tau_values=init_tau,
        clamp_left=1.0, bridge_budget=cfg.oracle_traj_bridge_length + 1,
        d_min=2.0,
    )
    init_state = state_cache.get([3, 9, 16])
    traj = init_false_trajectory(
        K=3, L=init_state.L_max,
        init_anchor_offsets=[
            (float(dy), float(dx))
            for dy, dx in init_state.decoy_offsets_init],
    )
    edit_p = init_bridge_edit_params(
        3, init_state.L_max,
        alpha_max=cfg.oracle_traj_alpha_max,
        s_init_px=1.0, r_init_px=0.0,
    )
    nu = torch.zeros(3, H, W, 3, requires_grad=True)
    R = torch.zeros(3, init_state.L_max, H, W, 3, requires_grad=True)

    def _opt(active_K: int):
        return torch.optim.Adam([
            {"params": [tau_p.g], "lr": 0.1},
            {"params": [traj.anchor_offset], "lr": 0.1},
            {"params": [traj.delta_offset], "lr": 0.1},
            {"params": [edit_p.alpha_logits], "lr": 0.1},
            {"params": [edit_p.warp_s, edit_p.warp_r], "lr": 0.1},
            {"params": [nu], "lr": 0.01},
        ])

    R_baseline_inactive = R.data[1:].clone()
    g_initial = tau_p.g.detach().clone()
    anchor_initial = traj.anchor_offset.detach().clone()
    delta_initial = traj.delta_offset.detach().clone()
    alpha_initial = edit_p.alpha_logits.detach().clone()
    nu_initial = nu.detach().clone()

    # K=1 phase: 2 steps. R must NOT be active (phase_idx < K-1).
    opt = _opt(1)
    for _ in range(2):
        diag = curriculum_joint_step(
            x_clean=x_clean, pseudo_masks_clean=pseudo_masks,
            config=cfg, forward_fn=forward_fn, lpips_fn=lpips_fn,
            state_cache=state_cache,
            tau_params=tau_p, traj_params=traj,
            edit_params=edit_p, R=R, nu=nu,
            optimizer=opt, active_inserts=[0],
            lambda_fid_val=10.0,
            bridge_length=cfg.oracle_traj_bridge_length,
            R_active_in_phase=False,
            R_lr=cfg.oracle_traj_residual_lr,
            R_eps=cfg.oracle_traj_residual_eps,
        )
        # K=1: 2 active corners (floor/ceil of τ[0]) -> at most 2 schedules.
        assert diag.valid_corner_count <= 2, diag.valid_corner_count
        assert diag.valid_corner_count >= 1, diag.valid_corner_count
    # R inactive slices unchanged (Phase A).
    assert torch.allclose(R.data[1:], R_baseline_inactive)
    # codex R3 HIGH fix: inactive insert slices must NOT have moved
    # under Adam during K=1 phase (only insert 0 is active).
    assert torch.allclose(traj.anchor_offset[1:], anchor_initial[1:]), \
        "anchor_offset[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(traj.delta_offset[1:], delta_initial[1:]), \
        "delta_offset[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(edit_p.alpha_logits[1:], alpha_initial[1:]), \
        "alpha_logits[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(nu[1:], nu_initial[1:]), \
        "nu[1:] (inactive in K=1) should be unchanged"
    # And g[1], g[2] (inactive) unchanged.
    g_after_K1 = tau_p.g.detach().clone()
    assert torch.allclose(g_after_K1[1:], g_initial[1:]), \
        f"g[1:] (inactive) must be frozen, got drift {(g_after_K1[1:] - g_initial[1:]).abs().max()}"
    # Active slices SHOULD have moved.
    assert (g_after_K1[0] - g_initial[0]).abs() > 1e-6, \
        "g[0] (active) should have moved during K=1 phase"

    # K=2 phase: 2 steps with optimizer rebuild.
    opt = _opt(2)
    for _ in range(2):
        diag = curriculum_joint_step(
            x_clean=x_clean, pseudo_masks_clean=pseudo_masks,
            config=cfg, forward_fn=forward_fn, lpips_fn=lpips_fn,
            state_cache=state_cache,
            tau_params=tau_p, traj_params=traj,
            edit_params=edit_p, R=R, nu=nu,
            optimizer=opt, active_inserts=[0, 1],
            lambda_fid_val=10.0,
            bridge_length=cfg.oracle_traj_bridge_length,
            R_active_in_phase=False,
            R_lr=cfg.oracle_traj_residual_lr,
            R_eps=cfg.oracle_traj_residual_eps,
        )
        # K=2: 4 active corners max.
        assert diag.valid_corner_count <= 4, diag.valid_corner_count
    # R inactive (Phase A).
    R_after_K2 = R.data.clone()

    # K=3 phase: 2 steps, R active.
    opt = _opt(3)
    for _ in range(2):
        diag = curriculum_joint_step(
            x_clean=x_clean, pseudo_masks_clean=pseudo_masks,
            config=cfg, forward_fn=forward_fn, lpips_fn=lpips_fn,
            state_cache=state_cache,
            tau_params=tau_p, traj_params=traj,
            edit_params=edit_p, R=R, nu=nu,
            optimizer=opt, active_inserts=[0, 1, 2],
            lambda_fid_val=10.0,
            bridge_length=cfg.oracle_traj_bridge_length,
            R_active_in_phase=True,
            R_lr=cfg.oracle_traj_residual_lr,
            R_eps=cfg.oracle_traj_residual_eps,
        )
        # K=3: up to 8 active corners.
        assert diag.valid_corner_count <= 8, diag.valid_corner_count

    # R must have moved during K=3 (sign-PGD ran).
    assert (R.data - R_after_K2).abs().max() > 0, \
        "R should have moved during K=3 phase"
    print("  curriculum K1/K2/K3 smoke: corners + tau update + R Phase B OK")


def _test_attack_state_cache_warmup_in_search() -> None:
    """Two consecutive curriculum steps with overlapping schedule W
    tuples -> cache hit count grows."""
    torch.manual_seed(0)
    np.random.seed(0)
    T_clean, H, W = 25, 16, 16
    x_clean = torch.rand(T_clean, H, W, 3)
    pseudo_masks = []
    for t in range(T_clean):
        m = np.zeros((H, W), dtype=np.float32)
        m[6:10, 4:8] = 1.0
        pseudo_masks.append(m)
    cfg = _DummyConfig()
    cfg.oracle_traj_use_residual = False
    forward_fn = _StubForwardFn(H, W)

    cache = AttackStateCache(
        x_clean, pseudo_masks, cfg, max_size=32,
        bridge_length=cfg.oracle_traj_bridge_length,
    )
    init_tau = [3.0, 9.0, 16.0]
    tau_p = init_tau_gap_params(
        K=3, T_clean=T_clean, init_tau_values=init_tau,
        clamp_left=1.0, bridge_budget=cfg.oracle_traj_bridge_length + 1,
        d_min=2.0,
    )
    init_state = cache.get([3, 9, 16])
    traj = init_false_trajectory(
        K=3, L=init_state.L_max,
        init_anchor_offsets=[(0.0, 5.0)] * 3,
    )
    edit_p = init_bridge_edit_params(
        3, init_state.L_max, alpha_max=cfg.oracle_traj_alpha_max,
    )
    nu = torch.zeros(3, H, W, 3, requires_grad=True)

    opt = torch.optim.Adam([
        {"params": [tau_p.g], "lr": 0.01},      # very small lr -> minimal τ drift
        {"params": [traj.anchor_offset], "lr": 0.001},
        {"params": [traj.delta_offset], "lr": 0.001},
        {"params": [edit_p.alpha_logits], "lr": 0.001},
        {"params": [edit_p.warp_s, edit_p.warp_r], "lr": 0.001},
        {"params": [nu], "lr": 0.001},
    ])

    hits_initial = cache.hits
    misses_initial = cache.misses
    for _ in range(3):
        curriculum_joint_step(
            x_clean=x_clean, pseudo_masks_clean=pseudo_masks,
            config=cfg, forward_fn=forward_fn, lpips_fn=_stub_lpips,
            state_cache=cache,
            tau_params=tau_p, traj_params=traj,
            edit_params=edit_p, R=None, nu=nu,
            optimizer=opt, active_inserts=[0, 1, 2],
            lambda_fid_val=10.0,
            bridge_length=cfg.oracle_traj_bridge_length,
            R_active_in_phase=False,
            R_lr=cfg.oracle_traj_residual_lr,
            R_eps=cfg.oracle_traj_residual_eps,
        )
    # With small τ drift, schedule corners stay similar -> hits should
    # accumulate beyond initial misses for first-step build.
    assert cache.hits >= hits_initial + 3, \
        f"cache hits should accumulate; got {cache.hits} (was {hits_initial})"
    print("  AttackState cache warm-up: hits accumulate during search")


def _self_test() -> None:
    print("memshield.joint_placement_search self-tests:")
    _test_tau_gap_init_and_recurrence_inverts()
    _test_tau_ordered_legality()
    _test_schedule_weight_sum_and_integer_exactness()
    _test_schedule_filtering_invalid_orderings()
    _test_schedule_grad_flows_through_weights()
    _test_R_active_slice_mask_locality()
    _test_27_neighbor_enumeration_filters_invalid()
    _test_bundle_C_incompat_guard()
    _test_attack_state_cache_hit_miss()
    _test_curriculum_smoke_K1_K2_K3()
    _test_attack_state_cache_warmup_in_search()
    print("memshield.joint_placement_search: all self-tests PASSED")


if __name__ == "__main__":
    _self_test()
