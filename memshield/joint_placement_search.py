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
class TauPhaseParams:
    """Phase-local τ parameterization via simplex slack allocation (v2).

    Replaces the v1 cumulative-softplus parameterization, which had a
    saturation pathology: when prescreen produced a `d_min`-spaced
    cluster, the inverse-softplus init landed `g[k]` at ≈ -13.8 where
    `softplus'(g) ≈ 1e-6`, killing all gradient signal to gap variables
    and locking the curriculum at the cluster (codex R6 R3 GO missed
    this; ss7-v1 dog parity test J-drop=0.326 was the empirical
    falsification).

    The v2 design (codex R6 follow-up): per phase, allocate the usable
    "slack" budget as a probability simplex over (m+1) bins, where `m`
    is the number of *active* inserts in this phase. Each bin maps to
    a slack `s_j = U_m * p_j` that augments the minimum-gap structure:

        τ_0 = clamp_left + s_0
        τ_i = clamp_left + i * d_min + Σ_{j=0..i} s_j     (i = 1..m-1)
        τ_i = tau_fixed_tail[i - m]                         (i = m..K-1)

    where `U_m = B_m - clamp_left - (m - 1) * d_min` is the usable
    slack and `B_m` is the phase right boundary
    (B_m = tau_fixed_full[m] - d_min  if m < K
    B_m = T_clean - 1 - bridge_budget if m == K).

    Mathematical guarantees by construction:
      * τ_0 ≥ clamp_left
      * τ_i - τ_{i-1} = d_min + s_i ≥ d_min
      * τ_{m-1} ≤ B_m  (so the next-fixed insert is not violated)
      * if m == K: τ_{K-1} + bridge_budget ≤ T_clean - 1

    Backward sanity (the property v1 lacked):
      ∂τ_i/∂p_j = U_m   if j ≤ i
      ∂τ_i/∂p_j = 0     if j > i
    The Jacobian is **constant** in the interior — no saturation at
    any feasible point. With T_clean≈58, K=3 ⇒ U_3 ≈ 50, so a 0.02
    simplex-mass shift produces a ~1-frame τ shift; Adam lr=0.05 over
    10 steps moves placement by ~5 frames easily.

    Curriculum semantics (different from v1):
      * Inactive τ values are *fixed constants* during this phase
        (not "frozen but coupled via recurrence"). At each phase
        transition the wrapper rebuilds TauPhaseParams from the
        current full τ.
      * `_zero_inactive_grads` no longer touches tau (fixed-tail
        already enforces inactive-slice freezing); it still gates
        traj/edit/R/nu inactive slices.
    """

    p_raw: Tensor                # [m+1] learnable, projected onto simplex
    active_K: int                # m, number of active inserts in this phase
    full_K: int                  # K, total inserts (K=3)
    T_clean: int
    clamp_left: float
    bridge_budget: int
    d_min: int                   # integer in v2 (was float in v1)
    tau_fixed_tail: Tensor       # [K - m], detached constants for inactive τ


def _phase_boundary_and_slack(
    T_clean: int, bridge_budget: int, clamp_left: float,
    d_min: int, m: int,
    tau_fixed_full: Optional[Sequence[float]],
) -> Tuple[float, float]:
    """Compute (B_m, U_m) — phase right boundary and usable slack."""
    if m < 1:
        raise ValueError(f"phase active_K m={m} must be ≥ 1")
    if m < int(len(tau_fixed_full or [])):
        # m < K case: B_m reserves room for the next fixed insert.
        B_m = float(tau_fixed_full[m]) - float(d_min)
    else:
        # m == K case: B_m is clip end minus right margin.
        B_m = float(T_clean - 1 - bridge_budget)
    U_m = B_m - float(clamp_left) - float(m - 1) * float(d_min)
    if U_m <= 1e-6:
        raise ValueError(
            f"degenerate U_m={U_m:.4f} for phase m={m}, T_clean={T_clean}, "
            f"bridge_budget={bridge_budget}, clamp_left={clamp_left}, "
            f"d_min={d_min}, tau_fixed_full={tau_fixed_full}. Clip too "
            "short for K inserts at this d_min.")
    return B_m, U_m


def project_simplex_inplace(x: Tensor) -> None:
    """Project ``x`` onto the probability simplex {z ≥ 0, Σz = 1}.

    Standard Duchi et al. (2008) projection. Operates in-place on
    ``x.data`` under no_grad. After projection, ``x`` is non-negative
    and sums to 1.0 within float tolerance.
    """
    with torch.no_grad():
        v = x.detach().reshape(-1).clone()
        n = int(v.numel())
        if n == 0:
            return
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(
            1, n + 1, device=v.device, dtype=v.dtype)
        cond = (u - cssv / ind) > 0
        nz = torch.nonzero(cond, as_tuple=False)
        if nz.numel() == 0:
            # Degenerate: all entries pull below zero. Fall back to uniform.
            x.copy_(torch.full_like(x, 1.0 / n))
            return
        rho = int(nz[-1].item())
        theta = cssv[rho] / float(rho + 1)
        w = torch.clamp(v - theta, min=0.0)
        x.copy_(w.view_as(x))


def init_tau_phase_params(
    active_K: int, full_K: int, T_clean: int,
    init_tau_values_full: Sequence[float],
    *,
    clamp_left: float = 1.0,
    bridge_budget: int = 4,
    d_min: int = 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> TauPhaseParams:
    """Initialize TauPhaseParams from a full valid τ vector.

    `active_K` (m) is the number of inserts active in this phase;
    `full_K` is K=3. The first m entries of `init_tau_values_full` are
    used as targets for the simplex slacks; the remaining (K-m) become
    the detached tail constants.

    Recovers the simplex point that exactly produces `init_tau_values
    _full[:m]` under the forward formula, then renormalizes for float
    safety.
    """
    K = int(full_K)
    m = int(active_K)
    if m < 1 or m > K:
        raise ValueError(f"active_K={m} not in [1, full_K={K}]")
    if len(init_tau_values_full) != K:
        raise ValueError(
            f"init_tau_values_full length {len(init_tau_values_full)} "
            f"!= full_K {K}")
    iv = [float(v) for v in init_tau_values_full]
    # Validate ordered + boundaries.
    if iv[0] < float(clamp_left) - 1e-6:
        raise ValueError(
            f"iv[0]={iv[0]:.4f} < clamp_left={clamp_left}")
    if iv[-1] + bridge_budget > T_clean - 1 + 1e-6:
        raise ValueError(
            f"iv[-1]={iv[-1]:.4f} + bridge_budget={bridge_budget} > "
            f"T_clean-1={T_clean-1}")
    for i in range(1, K):
        if iv[i] - iv[i - 1] < d_min - 1e-6:
            raise ValueError(
                f"iv gap [{i-1},{i}]={iv[i] - iv[i-1]:.4f} < d_min={d_min}")

    B_m, U_m = _phase_boundary_and_slack(
        T_clean, bridge_budget, clamp_left, d_min, m,
        tau_fixed_full=iv,
    )

    # Recover slacks from active iv prefix.
    s = [0.0] * (m + 1)
    s[0] = iv[0] - float(clamp_left)
    for i in range(1, m):
        s[i] = iv[i] - iv[i - 1] - float(d_min)
    s[m] = B_m - iv[m - 1]
    # Clamp tiny float-noise negatives.
    s = [max(0.0, v) for v in s]
    p = [v / U_m for v in s]
    p_total = float(sum(p))
    if p_total <= 1e-6:
        # Degenerate: fall back to uniform.
        p = [1.0 / (m + 1)] * (m + 1)
    else:
        p = [v / p_total for v in p]

    device = device or torch.device("cpu")
    p_tensor = torch.tensor(
        p, dtype=dtype, device=device).detach().clone().requires_grad_(True)

    tail = iv[m:]
    if tail:
        tau_fixed_tail = torch.tensor(tail, dtype=dtype, device=device)
    else:
        tau_fixed_tail = torch.zeros(0, dtype=dtype, device=device)

    return TauPhaseParams(
        p_raw=p_tensor,
        active_K=m, full_K=K, T_clean=int(T_clean),
        clamp_left=float(clamp_left), bridge_budget=int(bridge_budget),
        d_min=int(d_min),
        tau_fixed_tail=tau_fixed_tail,
    )


def tau_from_phase_params(params: TauPhaseParams) -> Tensor:
    """Compute full τ[0..K-1] from phase-local p_raw + tau_fixed_tail.

    The first m entries are differentiable in p_raw; the trailing K-m
    entries are detached constants (the "fixed tail" enforced by
    curriculum semantics — codex R6 follow-up v2 spec).
    """
    m = params.active_K
    K = params.full_K
    if m < 1:
        raise ValueError(f"active_K={m} must be ≥ 1")
    fixed_tail_list = params.tau_fixed_tail.tolist() if K > m else []
    B_m, U_m = _phase_boundary_and_slack(
        params.T_clean, params.bridge_budget, params.clamp_left,
        params.d_min, m,
        tau_fixed_full=([0.0] * m) + fixed_tail_list,
    )
    # Active τ via cumulative sum of slacks.
    p = params.p_raw                                        # [m+1]
    if int(p.shape[0]) != m + 1:
        raise ValueError(
            f"p_raw length {p.shape[0]} != m+1 {m+1}")
    prefix = torch.cumsum(p[:-1], dim=0)                    # [m]
    arange_m = torch.arange(
        m, device=p.device, dtype=p.dtype)
    tau_active = (
        params.clamp_left
        + params.d_min * arange_m
        + U_m * prefix
    )                                                       # [m]
    if K > m:
        tau_full = torch.cat([tau_active, params.tau_fixed_tail], dim=0)
    else:
        tau_full = tau_active
    return tau_full


def blend_simplex_with_uniform(p_raw: Tensor, *, weight: float = 0.10) -> None:
    """Degeneracy-recovery primitive: blend p_raw toward uniform.

    Codex R6 v2 spec: when curriculum_joint_step finds <2 valid
    schedule corners, blend p_raw with the uniform simplex
    (90% original, 10% uniform) and re-project. Replaces the v1
    `project_tau_inward` fallback (which moved softplus gaps to log(2),
    a near-saturation point).

    Operates in-place on p_raw.data; caller should re-project to
    simplex afterward via project_simplex_inplace.
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight={weight} must be in [0, 1]")
    with torch.no_grad():
        n = int(p_raw.numel())
        if n == 0:
            return
        uniform_val = 1.0 / float(n)
        p_raw.data.mul_(1.0 - weight).add_(weight * uniform_val)
    project_simplex_inplace(p_raw)


# ===========================================================================
# Section 2: Schedule enumeration (multilinear corners + filter + renormalize)
# ===========================================================================


def _validate_schedule(
    W_clean: Sequence[int], T_clean: int,
    bridge_budget: int, d_min: int,
    *,
    W0: Optional[Sequence[int]] = None,
    trust_radius: Optional[int] = None,
    span_ratio: Optional[float] = None,
    gap_ratio: Optional[float] = None,
) -> bool:
    """Strict ordering + d_min spacing + boundaries; optional trust region.

    v3 trust region (codex Coverage-Constrained spec): when W0 is given
    AND any of (trust_radius, span_ratio, gap_ratio) is set, additionally
    require:

      |W_k - W0_k| ≤ trust_radius                 # bounded movement
      span(W) ≥ span_ratio · span(W0)             # preserve coverage
      min_gap(W) ≥ max(d_min, gap_ratio · min_gap(W0))   # forbid cluster

    These hard constraints catch the v2 cluster-collapse pathology
    (e.g. W0=[4,13,21] → curriculum cluster [7,9,21] would have span=14
    < 0.85·17=14.45 → REJECTED, gap_min=2 < 0.5·8=4 → REJECTED).
    """
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
    # v3 trust region (only when caller passes W0 + at least one bound).
    if W0 is not None and (
            trust_radius is not None
            or span_ratio is not None
            or gap_ratio is not None):
        W0_int = [int(c) for c in W0]
        if len(W0_int) != K:
            return False
        if trust_radius is not None and trust_radius > 0:
            for i in range(K):
                if abs(int(W_clean[i]) - W0_int[i]) > int(trust_radius):
                    return False
        if span_ratio is not None and span_ratio > 0:
            span_W = float(int(W_clean[-1]) - int(W_clean[0]))
            span_W0 = float(int(W0_int[-1]) - int(W0_int[0]))
            # Strict float comparison (no int-truncation leniency).
            if span_W < float(span_ratio) * span_W0 - 1e-6:
                return False
        if gap_ratio is not None and gap_ratio > 0:
            gaps_W0 = [int(W0_int[i + 1] - W0_int[i]) for i in range(K - 1)]
            min_gap_W0 = float(min(gaps_W0)) if gaps_W0 else float(d_min)
            min_gap_required = max(float(d_min),
                                   float(gap_ratio) * min_gap_W0)
            for i in range(1, K):
                gap = float(int(W_clean[i]) - int(W_clean[i - 1]))
                if gap < min_gap_required - 1e-6:
                    return False
    return True


def enumerate_neighbor_schedules(
    tau: Tensor,                                  # [K] (current τ values)
    active_inserts: Sequence[int],                # subset of [0..K-1]
    *,
    T_clean: int,
    bridge_budget: int,
    d_min: int,
    W0: Optional[Sequence[int]] = None,
    trust_radius: Optional[int] = None,
    span_ratio: Optional[float] = None,
    gap_ratio: Optional[float] = None,
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
    # v3 trust-region kwargs forwarded to _validate_schedule.
    trust_kwargs: Dict[str, Any] = {}
    if W0 is not None and (
            trust_radius is not None
            or span_ratio is not None
            or gap_ratio is not None):
        trust_kwargs = {
            "W0": list(W0),
            "trust_radius": trust_radius,
            "span_ratio": span_ratio,
            "gap_ratio": gap_ratio,
        }
    if not active:
        # Nothing learnable; emit single round(τ) schedule with weight 1.
        W_round = tuple(int(torch.round(tau[k]).item()) for k in range(K))
        if not _validate_schedule(
                W_round, T_clean, bridge_budget, d_min, **trust_kwargs):
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
        if not _validate_schedule(
                W_tuple, T_clean, bridge_budget, d_min, **trust_kwargs):
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
    """Result of the v2 coverage-aware prescreen sweep."""
    init_tau_values: List[float]                 # K candidates, sorted, d_min-spaced
    candidate_raw_scores: List[Tuple[int, float]]   # (c, v_raw=-L_margin)
    candidate_relevance: List[Tuple[int, float]]    # (c, r=v_rank * h)
    seed_index: int


def _coverage_aware_select(
    candidates_relevance: Sequence[Tuple[int, float]],
    *,
    K: int,
    d_min: int,
    target_gap: float,
    seed_index: int = 0,
) -> List[int]:
    """v2 selection: greedy MMR with hard d_min and capped linear
    diversity (codex R6 follow-up spec).

    Args:
      candidates_relevance: sorted [(c, r(c))] descending by r.
      target_gap: usable_span / K — the saturation point for diversity.
      seed_index: 0 picks top by r; 1 / 2 are fallback restarts.

    Algorithm:
      1. First pick: candidate at position `seed_index` in the
         relevance-sorted list.
      2. Subsequent picks: argmax over candidates with hard d_min
         spacing of `0.7 * r(c) + 0.3 * div(c, S)` where
         `div(c, S) = min(1.0, min_{s ∈ S} |c-s| / target_gap)`.
      3. Stop at K.
      4. Tie-break: smaller c wins.
    """
    if len(candidates_relevance) < K:
        raise RuntimeError(
            f"prescreen: only {len(candidates_relevance)} candidates for "
            f"K={K}")
    sorted_by_r = list(candidates_relevance)  # already r-desc
    # First pick: seed_index-th highest-r.
    if seed_index >= len(sorted_by_r):
        raise RuntimeError(
            f"seed_index={seed_index} out of range for "
            f"{len(sorted_by_r)} candidates")
    chosen: List[int] = [int(sorted_by_r[seed_index][0])]
    relevance: Dict[int, float] = {
        int(c): float(r) for c, r in sorted_by_r}

    while len(chosen) < K:
        best_c = None
        best_score = -float("inf")
        for c, _ in sorted_by_r:
            c_int = int(c)
            if c_int in chosen:
                continue
            if any(abs(c_int - s) < d_min for s in chosen):
                continue
            min_dist = min(abs(c_int - s) for s in chosen)
            div = min(1.0, float(min_dist) / max(1e-6, float(target_gap)))
            score = 0.7 * relevance[c_int] + 0.3 * div
            if (score > best_score
                    or (score == best_score
                        and (best_c is None or c_int < best_c))):
                best_score = score
                best_c = c_int
        if best_c is None:
            raise RuntimeError(
                f"prescreen MMR: cannot find {K - len(chosen)} more "
                f"candidates with d_min={d_min} from chosen={chosen}")
        chosen.append(best_c)
    return sorted(chosen)


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
    prescreen_horizon: int = 12,
) -> PrescreenResult:
    """v2 coverage-aware prescreen (codex R6 follow-up).

    Replaces the v1 "K=1 cheap defaults + greedy d_min-spaced top-K"
    procedure that produced two failure modes on dog (ss7-v1):
      1. Score function (-L_margin from K=1 with bridge_length=3) had
         a sharp peak at video end where SAM2 has accumulated true-
         object memory; brief K=1 attack disrupts it locally but has
         no whole-video J-drop leverage (Failure #1).
      2. Greedy d_min-spaced top-K is a cluster machine: when scores
         concentrate in a window, picks all K at d_min spacing inside
         (Failure #3).

    v2 fixes:
      * Score = `v_rank(c) * h(c)` where:
          - v_rank(c): rank-percentile of `-L_margin` (more robust
            than raw value across clips with different score scales)
          - h(c): remaining-horizon factor `(T_clean - bridge_budget
            - c) / (T_clean - bridge_budget - clamp_left)`, clipped to
            [0, 1]. Multiplicative — late-frame spikes get hard-
            discounted because they cannot influence whole-video mean
            J-drop.
      * Bridge horizon for K=1 forward = `prescreen_horizon=12` (was
        `bridge_length=3`); makes the proxy more future-aware.
      * MMR with `target_gap = usable_span / K` and capped linear
        diversity `min(1, min_dist / target_gap)`. Saturates once
        spacing is "good enough" — does not reward absurdly far
        extreme placements.

    Multi-seed fallback (R3 guardrail #6) operates by changing the
    first pick: `seed_index = 0` uses the global top by r(c);
    `seed_index = 1, 2` use the next-best by r. Subsequent picks are
    deterministic given the first.
    """
    T_clean = int(x_clean.shape[0])
    bridge_budget = bridge_length + 1
    clamp_left = 1
    H_pre = int(prescreen_horizon)
    if H_pre < 1:
        H_pre = bridge_length

    # Candidate range matches forward feasibility: c ∈ [clamp_left,
    # T_clean - bridge_budget). The end of the range is exclusive so
    # that c + bridge_budget ≤ T_clean - 1.
    candidates: List[int] = list(range(
        int(clamp_left), int(T_clean - bridge_budget),
        max(1, int(candidate_stride))))
    if len(candidates) < K:
        raise RuntimeError(
            f"prescreen: only {len(candidates)} candidate positions for "
            f"K={K} inserts (T_clean={T_clean}, bridge_budget="
            f"{bridge_budget})")

    device = x_clean.device
    H, W = int(x_clean.shape[1]), int(x_clean.shape[2])
    nu = torch.zeros(1, H, W, 3, device=device, dtype=x_clean.dtype)
    R = (torch.zeros(1, H_pre, H, W, 3, device=device,
                     dtype=x_clean.dtype)
         if config.oracle_traj_use_residual else None)

    raw_scores: List[Tuple[int, float]] = []
    for c in candidates:
        try:
            state = build_attack_state_from_W(
                [int(c)], x_clean, pseudo_masks_clean, config,
                bridge_length=H_pre,
            )
        except ValueError:
            continue
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
        v_raw = -float(diag.L_margin.detach().item())
        raw_scores.append((int(c), v_raw))

    if not raw_scores:
        raise RuntimeError(
            "prescreen: every candidate failed to build attack_state or "
            "Stage-14 forward")

    # Convert v_raw -> v_rank ∈ [0, 1] (rank percentile). Use simple
    # rank/(n-1) so the worst maps to 0 and the best maps to 1.
    sorted_by_v = sorted(raw_scores, key=lambda kv: kv[1])
    n = len(sorted_by_v)
    v_rank: Dict[int, float] = {}
    for rank, (c, _) in enumerate(sorted_by_v):
        v_rank[int(c)] = float(rank) / float(max(1, n - 1))

    # Coverage factor h(c) — multiplicative penalty for late frames.
    span_denom = max(1.0, float(T_clean - bridge_budget - clamp_left))
    relevance: List[Tuple[int, float]] = []
    for c, _ in raw_scores:
        h_c = max(0.0, min(1.0,
                           float(T_clean - bridge_budget - c) / span_denom))
        r_c = v_rank[int(c)] * h_c
        relevance.append((int(c), r_c))
    # Sort relevance descending by r for the selector.
    relevance.sort(key=lambda kv: -kv[1])

    usable_span = float(T_clean - 1 - bridge_budget) - float(clamp_left)
    target_gap = max(1.0, usable_span / float(K))

    chosen = _coverage_aware_select(
        relevance, K=K, d_min=int(d_min),
        target_gap=target_gap, seed_index=int(seed_index),
    )

    init_tau_values = sorted(float(c) for c in chosen)
    return PrescreenResult(
        init_tau_values=init_tau_values,
        candidate_raw_scores=raw_scores,
        candidate_relevance=relevance,
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
    suffix_loss_weighted: float = 0.0   # v3: weighted-mean L_suffix across
                                        # schedules this step (raw IoU-sum
                                        # value, not multiplied by lambda).
                                        # 0.0 when phase doesn't use suffix.
    trust_region_active: bool = False   # v3: True when phase passed W0 +
                                        # trust ratios for schedule
                                        # filtering. Phase 3 only.


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
    tau_params: TauPhaseParams,
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
    # v3 (Coverage-Constrained Joint Suffix Optimization, codex spec).
    # When `W0_for_trust` is provided AND any of the trust ratios is
    # set, schedule enumeration applies the trust region (forbid cluster
    # collapse). When `suffix_probe_frames` + `lambda_suffix > 0`, the
    # forward loss includes the suffix-IoU term. Both are passed only in
    # phase 3 by `joint_curriculum_search`; phase 1-2 keeps v2.2 freeze
    # behavior with these all None / 0 (perturbation warmup).
    W0_for_trust: Optional[Sequence[int]] = None,
    trust_radius: Optional[int] = None,
    span_ratio: Optional[float] = None,
    gap_ratio: Optional[float] = None,
    suffix_probe_frames: Optional[Sequence[int]] = None,
    lambda_suffix: float = 0.0,
) -> CurriculumStepDiagnostics:
    """One joint step of the v2 curriculum.

    Differences from v1:
      * `tau_params` is `TauPhaseParams` (phase-local simplex), not
        `TauGapParams` (cumulative softplus). `tau_from_phase_params`
        returns a [K] tensor where the first m entries are
        differentiable in `p_raw` and the trailing K-m entries are
        detached fixed constants from the prior phase.
      * Degeneracy fallback (`<2 valid corners`) blends `p_raw` with
        the uniform simplex (90/10) instead of resetting g to a
        softplus interior — this avoids the dead-zone retry trap
        that v1 had at the d_min lower bound.
      * After `optimizer.step()`, `project_simplex_inplace` ensures
        `p_raw` stays on the (m+1)-simplex.
      * `_zero_inactive_grads` no longer touches τ params — fixed-
        tail constants in `TauPhaseParams` enforce inactive-τ
        freezing structurally, so the gradient hook is unneeded for
        τ. It still guards traj/edit/R/nu inactive slices.
    """
    tau = tau_from_phase_params(tau_params)         # [K] differentiable
    schedules, raw_weight_mass = enumerate_neighbor_schedules(
        tau, active_inserts,
        T_clean=tau_params.T_clean,
        bridge_budget=tau_params.bridge_budget,
        d_min=int(tau_params.d_min),
        W0=W0_for_trust, trust_radius=trust_radius,
        span_ratio=span_ratio, gap_ratio=gap_ratio,
    )
    valid_corner_count = len(schedules)
    inward_projected = False

    # Guardrail #2 fallback (v2): if degenerate (<2 surviving corners),
    # blend p_raw with the uniform simplex and retry once. If zero
    # corners survive even after the blend, raise (config is
    # incompatible — caller should reduce K or d_min).
    if valid_corner_count < 2:
        blend_simplex_with_uniform(tau_params.p_raw, weight=0.10)
        inward_projected = True
        tau = tau_from_phase_params(tau_params)
        schedules, raw_weight_mass = enumerate_neighbor_schedules(
            tau, active_inserts,
            T_clean=tau_params.T_clean,
            bridge_budget=tau_params.bridge_budget,
            d_min=int(tau_params.d_min),
            W0=W0_for_trust, trust_radius=trust_radius,
            span_ratio=span_ratio, gap_ratio=gap_ratio,
        )
        valid_corner_count = len(schedules)
        if valid_corner_count == 0:
            raise RuntimeError(
                "joint search: no valid schedule corners after simplex "
                "uniform-blend retry. T_clean / bridge_length / d_min "
                "(or v3 trust region) combination is incompatible.")

    singleton_corner = (valid_corner_count == 1)

    # Per-schedule forward + per-schedule backward (memory fix, 2026-04-26
    # 19:55). The original implementation built one big graph
    # `L_step = Σ weight_i * L_sched_i` and called `L_step.backward()`
    # ONCE — which kept ALL 2^|active| schedule activations alive in
    # memory simultaneously. On dog (T_proc≈53, H=W=480) with K=3 phase
    # (8 schedules), this exceeded the Pro 6000's 95 GB.
    #
    # Mathematically `∂(Σ_i wᵢ Lᵢ)/∂θ = Σ_i ∂(wᵢ Lᵢ)/∂θ`, so per-schedule
    # `(weight * L_sched).backward()` accumulates into the SAME `θ.grad`
    # buffer (`optimizer.zero_grad()` is called BEFORE the loop). The
    # only subtlety: weights share a graph rooted at `tau`, so we must
    # `retain_graph=True` for all but the LAST schedule (otherwise the
    # second iteration's `weight` would have a freed graph).
    #
    # Memory: only ONE schedule's SAM2 activations are alive at a time
    # (≈8× reduction at K=3). CLAUDE.md categorizes OOM as engineering-
    # fix territory, so this patch is in scope without method redesign.
    optimizer.zero_grad()
    if R is not None and R.requires_grad and R.grad is not None:
        R.grad.zero_()

    L_step_value = 0.0
    L_suffix_step = 0.0       # diagnostic: avg suffix loss across schedules
    schedule_logs: List[Dict[str, Any]] = []
    n_schedules = len(schedules)
    for sched_idx, (W_tuple, weight, corner_id) in enumerate(schedules):
        state = state_cache.get(W_tuple)
        L_sched, diag_sched, _ = stage14_forward_loss(
            state, x_clean=x_clean,
            traj=traj_params, edit_params=edit_params,
            R=R, nu=nu,
            forward_fn=forward_fn, lpips_fn=lpips_fn,
            config=config, lambda_fid_val=lambda_fid_val,
            R_active=R_active_in_phase,
            suffix_probe_frames=suffix_probe_frames,
            lambda_suffix=lambda_suffix,
        )
        weighted = weight * L_sched
        # Last schedule frees the shared tau→weight graph; earlier
        # schedules retain it so subsequent backward calls can still
        # reach tau through their own weight terms.
        weighted.backward(retain_graph=(sched_idx < n_schedules - 1))
        L_step_value += float(weighted.detach().item())
        L_suffix_step += float(weight.detach().item()) * float(
            diag_sched.L_suffix.detach().item())
        schedule_logs.append({
            "W": list(W_tuple),
            "weight": float(weight.detach().item()),
            "corner_id": int(corner_id),
            "L_margin": float(diag_sched.L_margin.detach().item()),
            "L_suffix": float(diag_sched.L_suffix.detach().item()),
            "L_total": float(L_sched.detach().item()),
            "feasible": bool(diag_sched.feasible),
            "delta_overlap": float(diag_sched.delta_overlap),
            "n_suffix_probes": int(diag_sched.n_suffix_probes),
        })
        # Free per-schedule references that the autograd graph held onto
        # so the next iteration starts with maximum free memory. The
        # graph itself is released by .backward() above (last iter) or
        # by Python ref-cycle GC (earlier iters' state is no longer
        # referenced after the loop body advances).
        del L_sched, diag_sched, weighted

    # Codex R3 HIGH fix: zero gradients on inactive insert slices BEFORE
    # optimizer.step(). The optimizer rebuild per phase still owns the
    # FULL [K, ...] tensors; without this hook the inactive slices would
    # be Adam-stepped on every iteration via gradients flowing through
    # stage14_forward_loss across all schedules.
    #
    # v2: tau is no longer in this hook (TauPhaseParams uses fixed-tail
    # constants for inactive τ; the differentiable prefix is exactly
    # the active inserts). Only traj/edit/R/nu still need the gating.
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
        tau_g=None,                                      # v2: no tau gating
    )

    optimizer.step()

    # v2: simplex-project p_raw after the Adam step.
    project_simplex_inplace(tau_params.p_raw)

    # R sign-PGD restricted to active slices (Phase B only).
    if R_active_in_phase and R is not None and R.requires_grad:
        active_mask = _R_active_slice_mask(R, active_inserts)
        if active_mask is not None:
            _apply_R_sign_pgd_active(
                R, R.grad, active_mask, lr=R_lr, eps=R_eps)

    # Project trajectory after step (carries over from existing path).
    project_trajectory_to_budget(
        traj_params, max_offset_px=config.oracle_traj_max_offset_px)

    # Codex v2 review fix (2026-04-26 21:00): return POST-step τ.
    # The earlier `tau` was computed before optimizer.step() + simplex
    # projection, so using its values for `tau_values` would drop the
    # last update of every phase — `joint_curriculum_search` carries
    # `tau_full_state = diag.tau_values` forward, so a stale value
    # means the next phase initializes from one step behind and the
    # final W_round is also stale.
    with torch.no_grad():
        tau_post = tau_from_phase_params(tau_params)

    return CurriculumStepDiagnostics(
        L_step=float(L_step_value),
        valid_corner_count=valid_corner_count,
        valid_weight_mass=float(raw_weight_mass),
        schedules=schedule_logs,
        tau_values=[float(t.detach().item()) for t in tau_post],
        inward_projected=inward_projected,
        singleton_corner=singleton_corner,
        suffix_loss_weighted=float(L_suffix_step),
        trust_region_active=bool(W0_for_trust is not None and (
            trust_radius is not None
            or span_ratio is not None
            or gap_ratio is not None)),
    )


# ===========================================================================
# Section 6: Local refine (27-triple ±1 neighbor search after rounding)
# ===========================================================================


def _enumerate_27_neighbors(
    W_round: Sequence[int], *,
    T_clean: int, bridge_budget: int, d_min: int,
    W0: Optional[Sequence[int]] = None,
    trust_radius: Optional[int] = None,
    span_ratio: Optional[float] = None,
    gap_ratio: Optional[float] = None,
) -> List[Tuple[int, ...]]:
    """Enumerate 3^K = 27 (K=3) joint ±1 neighbors of W_round; filter
    invalid orderings + optional v3 trust region (codex Coverage-
    Constrained spec)."""
    K = len(W_round)
    out: List[Tuple[int, ...]] = []
    for code in range(3 ** K):
        offsets = []
        c = code
        for _ in range(K):
            offsets.append((c % 3) - 1)
            c //= 3
        cand = tuple(int(W_round[k]) + offsets[k] for k in range(K))
        if _validate_schedule(
                cand, T_clean, bridge_budget, d_min,
                W0=W0, trust_radius=trust_radius,
                span_ratio=span_ratio, gap_ratio=gap_ratio):
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
    W0_for_trust: Optional[Sequence[int]] = None,
    trust_radius: Optional[int] = None,
    span_ratio: Optional[float] = None,
    gap_ratio: Optional[float] = None,
    # v3 codex HIGH fix: refine score must use the same placement-
    # relevant objective as phase 3, otherwise the final discrete
    # selector can undo the v3 trust+suffix benefit by picking the
    # best neighbor under the OLD `-L_margin` surrogate.
    lambda_suffix_for_score: float = 0.0,
    suffix_probe_frames_for_score: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, ...], List[Dict[str, Any]]]:
    """Cheap 6-step Stage-14 estimate per valid neighbor; pick best by
    L_margin (proxy for J-drop, since SAM2 eval is expensive). Returns
    chosen W_clean tuple + diagnostics for all neighbors.

    `d_min` is REQUIRED (codex R3 MEDIUM fix) — earlier draft fell back
    to `config.oracle_traj_d_min` or `2`, which silently disagreed with
    the curriculum's d_min if the caller passed a different value.

    v3 (codex Coverage-Constrained spec): when `W0_for_trust` + any
    trust ratio are passed, the 27-neighbor enumeration applies the
    same trust region used during phase-3 schedule enumeration. This
    prevents local refine from undoing the trust-region benefit.

    The full 30-step Stage-14 + export + SAM2 eval is run AFTER local
    refine on the chosen triple by the joint_curriculum_search caller.
    """
    K = len(W_round)
    bridge_budget = bridge_length + 1
    candidates = _enumerate_27_neighbors(
        W_round, T_clean=int(x_clean.shape[0]),
        bridge_budget=bridge_budget, d_min=int(d_min),
        W0=W0_for_trust, trust_radius=trust_radius,
        span_ratio=span_ratio, gap_ratio=gap_ratio)

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
                suffix_probe_frames=suffix_probe_frames_for_score,
                lambda_suffix=float(lambda_suffix_for_score),
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
                suffix_probe_frames=suffix_probe_frames_for_score,
                lambda_suffix=float(lambda_suffix_for_score),
            )
        # v3 codex HIGH fix: score uses placement-relevant objective
        # (margin + suffix), not just L_margin. This keeps the discrete
        # selector aligned with phase-3's joint objective so it cannot
        # undo the trust-region + suffix-aware benefit.
        L_margin_val = float(diag_final.L_margin.detach().item())
        L_suffix_val = float(diag_final.L_suffix.detach().item())
        lambda_margin = float(config.oracle_traj_lambda_margin)
        score = -(lambda_margin * L_margin_val
                  + float(lambda_suffix_for_score) * L_suffix_val)
        diags.append({
            "W": list(cand), "score": score,
            "L_margin": L_margin_val,
            "L_suffix": L_suffix_val,
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
    """Run the full v2 joint curriculum placement-perturbation search.

    Pipeline (codex R6 follow-up v2):
      1. Coverage-aware prescreen (1 fwd × T candidates with horizon
         H_pre=12, score = v_rank * h(c) for late-frame discount,
         MMR 0.7/0.3 selector with capped linear diversity).
      2. Initialize traj + edit_params + R + ν from prescreen W. The
         **TauPhaseParams is rebuilt at each curriculum phase** from
         the current full τ — phase-local simplex slack allocation
         replaces the v1 global cumulative softplus that saturated at
         the d_min lower bound.
      3-5. Curriculum K=1, K=2, K=3 with `phase_steps` per phase. At
         each transition the optimizer + tau_params are rebuilt;
         inactive τ tail is held as detached constants (structural,
         not gradient-zeroing).
      6. Round τ → W_round; 27-triple ±1 local refine via 6-step
         cheap Stage-14 estimates.
      7. Return chosen W (caller runs final 30-step Stage-14 + export).

    Bundle C (LPIPS-native ν) is unsupported (multi-schedule violates
    fixed-W line-search assumption). Driver mutex enforces this.
    """
    if int(K) != len(phase_steps):
        raise ValueError(
            f"phase_steps length {len(phase_steps)} != K {K}; one phase "
            "per insert.")
    if getattr(config, "oracle_traj_nu_lpips_native", False):
        raise ValueError(
            "joint_curriculum_search: Bundle C (oracle_traj_nu_lpips_native) "
            "is not supported in v1/v2. The LPIPS-native ν line-search "
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

    # ---- 1. Coverage-aware prescreen ----
    t0 = time.time()
    pre = prescreen_tau_init(
        x_clean, pseudo_masks_clean, config,
        forward_fn=forward_fn, lpips_fn=lpips_fn,
        K=K, d_min=int(d_min), bridge_length=bridge_len,
        seed_index=int(prescreen_seed_index),
        candidate_stride=int(candidate_stride),
        prescreen_horizon=int(getattr(
            config, "oracle_traj_prescreen_horizon", 12)),
    )
    timings["prescreen"] = time.time() - t0

    # ---- 2. Initialize state cache + traj/edit/ν/R from prescreen W ----
    state_cache = AttackStateCache(
        x_clean, pseudo_masks_clean, config,
        max_size=int(cache_max_size),
        bridge_length=bridge_len,
    )
    # Track the FULL τ (length K) across phases. Initially set to the
    # prescreen output; updated at each phase transition from the
    # phase-local TauPhaseParams' produced full τ.
    tau_full_state: List[float] = [float(v) for v in pre.init_tau_values]

    # Pre-build the K-insert attack state at the prescreen-rounded W so
    # traj init_anchors come from the right offsets.
    W_init = tuple(int(round(v)) for v in tau_full_state)
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

    # ---- 3-5. Curriculum phases (TauPhaseParams rebuilt per phase) ----
    lambda_fid_val = float(config.joint_traj_lambda_fid)
    R_lr = float(config.oracle_traj_residual_lr)
    R_eps = float(config.oracle_traj_residual_eps)
    # v3 (Coverage-Constrained Joint Suffix Optimization, codex spec
    # 2026-04-26): preserve "joint placement-perturbation" claim by
    # restoring phase-3 τ motion, but constrain it via:
    #   1. Trust region around prescreen W0 (forbids cluster collapse)
    #   2. Suffix-probe loss term in phase 3 (aligns surrogate with
    #      whole-video J-drop instead of just attacked-window L_margin)
    #
    # Default phases-1-2 stay frozen (v2.2 perturbation warmup behavior
    # which was reliable on dog/camel) — only phase 3 unfreezes for
    # the joint-search refinement.
    #
    # Backwards compat:
    # - tau_freeze_phases=[1, 2, 3]                    → v2.2 (full freeze)
    # - tau_freeze_phases=[1, 2] + lr_phase3=0.002     → v2.1 (no v3 trust + suffix)
    # - tau_freeze_phases=[] + tau_lr=0.05 + force_legacy → v2 curriculum-on
    tau_lr_phase3 = float(getattr(
        config, "oracle_traj_tau_lr_phase3", 0.002))
    tau_freeze_phases: Sequence[int] = list(getattr(
        config, "oracle_traj_tau_freeze_phases", [1, 2]))
    # v3 toggles (default ON for new runs).
    v3_lambda_suffix = float(getattr(
        config, "oracle_traj_v3_lambda_suffix", 2.0))
    v3_n_probes = int(getattr(
        config, "oracle_traj_v3_n_probes", 6))
    v3_trust_radius = int(getattr(
        config, "oracle_traj_v3_trust_radius", 6))
    v3_span_ratio = float(getattr(
        config, "oracle_traj_v3_span_ratio", 0.85))
    v3_gap_ratio = float(getattr(
        config, "oracle_traj_v3_gap_ratio", 0.5))
    # Disable suffix-probe + trust region by setting these to 0 / None
    # (gives v2.1 behavior).
    v3_enable = (v3_lambda_suffix > 0.0
                 and v3_n_probes > 0
                 and v3_trust_radius > 0)
    # Legacy v2 single-LR knob (used only if user explicitly sets
    # oracle_traj_tau_lr; otherwise the v2.1 phase-freeze takes over).
    tau_lr_legacy: Optional[float] = (
        float(config.oracle_traj_tau_lr)
        if hasattr(config, "oracle_traj_tau_lr")
        and getattr(config, "oracle_traj_tau_lr_force_legacy", False)
        else None)

    for phase_idx in range(K):
        t_phase = time.time()
        active_inserts = list(range(phase_idx + 1))
        m = phase_idx + 1

        # v2: rebuild TauPhaseParams from the current full τ. The
        # active prefix is whatever the previous phase produced; the
        # tail (K-m entries) stays as fixed constants from prescreen
        # initialization (or from prior phases that touched them).
        tau_params = init_tau_phase_params(
            active_K=m, full_K=K, T_clean=T_clean,
            init_tau_values_full=tau_full_state,
            clamp_left=1.0, bridge_budget=bridge_budget,
            d_min=int(d_min),
            device=device, dtype=x_clean.dtype,
        )

        # v2.1: phase-conditional τ inclusion. Phases listed in
        # tau_freeze_phases keep τ frozen at prescreen init by NOT
        # adding p_raw to the optimizer. Phase 3 uses tiny lr.
        if tau_lr_legacy is not None:
            tau_lr_this_phase: Optional[float] = tau_lr_legacy
        elif (phase_idx + 1) in tau_freeze_phases:
            tau_lr_this_phase = None  # frozen
        else:
            tau_lr_this_phase = tau_lr_phase3

        param_groups: List[Dict[str, Any]] = [
            {"params": [traj.anchor_offset],
             "lr": float(config.oracle_traj_anchor_lr)},
            {"params": [traj.delta_offset],
             "lr": float(config.oracle_traj_delta_lr)},
            {"params": [edit_params.alpha_logits],
             "lr": float(config.oracle_traj_alpha_lr)},
            {"params": [edit_params.warp_s, edit_params.warp_r],
             "lr": float(config.oracle_traj_warp_lr)},
            {"params": [nu], "lr": float(config.oracle_traj_nu_lr_phase_b)},
        ]
        if tau_lr_this_phase is not None:
            # Insert at front (matches v2 ordering for Adam state
            # determinism if user A/B-tests with/without tau_freeze).
            param_groups.insert(
                0, {"params": [tau_params.p_raw],
                    "lr": float(tau_lr_this_phase)})
        optimizer = torch.optim.Adam(param_groups)
        R_active_in_phase = (R is not None and phase_idx == K - 1)

        # v3 phase-3 plumbing: only the LAST phase (= K-1 index) gets the
        # suffix-probe loss + trust region. Earlier phases keep v2.2
        # frozen-τ behavior (perturbation warmup).
        is_last_phase = (phase_idx == K - 1)
        v3_suffix_probe_frames: Optional[List[int]] = None
        v3_lambda_suffix_this: float = 0.0
        v3_W0_for_trust: Optional[List[int]] = None
        v3_trust_radius_this: Optional[int] = None
        v3_span_ratio_this: Optional[float] = None
        v3_gap_ratio_this: Optional[float] = None
        if is_last_phase and v3_enable:
            # Probes: M evenly spaced in attacked-space [t0, T_proc-1]
            # where t0 = W_attacked[0] + 1 ≈ first post-insert frame.
            # codex MEDIUM fix: EXCLUDE attacked insert positions from
            # probes — at insert frames, m_true_by_t was overridden to
            # the seed/decoy mask (NOT the original true mask) so the
            # IoU there measures attacked-frame retention which would
            # bleed back into local L_margin territory and defeat the
            # whole-suffix purpose. Backfill with nearest non-insert
            # post-suffix frames to maintain M=6 probes.
            W_init_attacked = sorted(int(round(v)) + i
                                     for i, v in enumerate(
                                         sorted(tau_full_state)))
            insert_set = set(int(w) for w in W_init_attacked)
            t0 = max(1, W_init_attacked[0] + 1)
            T_proc_v3 = T_clean + K
            probe_end = max(t0, T_proc_v3 - 1)
            # Initial arithmetic progression.
            if probe_end > t0:
                step_size = max(
                    1, (probe_end - t0) // max(1, v3_n_probes - 1))
                candidates_init = list(range(
                    t0, probe_end + 1, step_size))[:v3_n_probes]
            else:
                candidates_init = [t0]
            # Filter out insert positions.
            chosen_probes = [t for t in candidates_init
                             if int(t) not in insert_set]
            # Backfill from nearby non-insert frames if filter removed too many.
            if len(chosen_probes) < v3_n_probes:
                chosen_set = set(chosen_probes)
                for t in range(t0, probe_end + 1):
                    if int(t) in insert_set or int(t) in chosen_set:
                        continue
                    chosen_probes.append(int(t))
                    chosen_set.add(int(t))
                    if len(chosen_probes) >= v3_n_probes:
                        break
                chosen_probes.sort()
            v3_suffix_probe_frames = chosen_probes[:v3_n_probes]
            v3_lambda_suffix_this = float(v3_lambda_suffix)
            # Trust region anchored at prescreen W0 (clean-space), not
            # the current tau (which already drifted from prescreen due
            # to phase-1/2 perturbation warmup with frozen τ — drift is
            # 0 there, so W0 = prescreen output).
            v3_W0_for_trust = [int(round(float(c)))
                               for c in pre.init_tau_values]
            v3_trust_radius_this = int(v3_trust_radius)
            v3_span_ratio_this = float(v3_span_ratio)
            v3_gap_ratio_this = float(v3_gap_ratio)

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
                W0_for_trust=v3_W0_for_trust,
                trust_radius=v3_trust_radius_this,
                span_ratio=v3_span_ratio_this,
                gap_ratio=v3_gap_ratio_this,
                suffix_probe_frames=v3_suffix_probe_frames,
                lambda_suffix=v3_lambda_suffix_this,
            )
            curriculum_logs.append({
                "phase": phase_idx + 1, "step": step + 1,
                "L_step": diag.L_step,
                "valid_corner_count": diag.valid_corner_count,
                "valid_weight_mass": diag.valid_weight_mass,
                "tau_values": diag.tau_values,
                "inward_projected": diag.inward_projected,
                "singleton_corner": diag.singleton_corner,
                "schedules": diag.schedules,
                "suffix_loss_weighted": diag.suffix_loss_weighted,
                "trust_region_active": diag.trust_region_active,
            })
            # Update full τ state with the active prefix produced this
            # step; the tail stays at its stored constants.
            tau_full_state = list(diag.tau_values)
        timings[f"phase_{phase_idx + 1}"] = time.time() - t_phase

    # ---- 6. Round + 27-triple local refine ----
    # v3: apply same trust region in local refine to prevent it from
    # undoing the phase-3 trust-region benefit.
    t_refine = time.time()
    W_round = [int(round(float(v))) for v in tau_full_state]
    refine_W0 = [int(round(float(c))) for c in pre.init_tau_values] \
        if v3_enable else None
    # v3 codex HIGH fix: local refine MUST score with the same
    # placement-relevant objective as phase 3 (margin + suffix),
    # otherwise the discrete selector can pick a neighbor that the
    # OLD `-L_margin` surrogate likes but that v3 phase 3 was trying
    # to avoid. Recompute probes here using the FINAL τ (post-phase-3)
    # so the refine score reflects the actual chosen placement region.
    refine_suffix_probes: Optional[List[int]] = None
    refine_lambda_suffix = 0.0
    if v3_enable:
        W_round_attacked = sorted(int(round(v)) + i
                                  for i, v in enumerate(
                                      sorted(tau_full_state)))
        insert_set_refine = set(int(w) for w in W_round_attacked)
        t0_r = max(1, W_round_attacked[0] + 1)
        T_proc_r = T_clean + K
        probe_end_r = max(t0_r, T_proc_r - 1)
        if probe_end_r > t0_r:
            step_r = max(1,
                         (probe_end_r - t0_r) // max(1, v3_n_probes - 1))
            cand_r = list(range(t0_r, probe_end_r + 1, step_r))[:v3_n_probes]
        else:
            cand_r = [t0_r]
        refine_probes = [t for t in cand_r if int(t) not in insert_set_refine]
        if len(refine_probes) < v3_n_probes:
            done = set(refine_probes)
            for t in range(t0_r, probe_end_r + 1):
                if int(t) in insert_set_refine or int(t) in done:
                    continue
                refine_probes.append(int(t))
                done.add(int(t))
                if len(refine_probes) >= v3_n_probes:
                    break
            refine_probes.sort()
        refine_suffix_probes = refine_probes[:v3_n_probes]
        refine_lambda_suffix = float(v3_lambda_suffix)

    refine_W, refine_diags = local_refine_27(
        x_clean=x_clean, pseudo_masks_clean=pseudo_masks_clean,
        config=config, forward_fn=forward_fn, lpips_fn=lpips_fn,
        state_cache=state_cache, W_round=W_round,
        bridge_length=bridge_len, d_min=int(d_min),
        W0_for_trust=refine_W0,
        trust_radius=(int(v3_trust_radius) if v3_enable else None),
        span_ratio=(float(v3_span_ratio) if v3_enable else None),
        gap_ratio=(float(v3_gap_ratio) if v3_enable else None),
        lambda_suffix_for_score=refine_lambda_suffix,
        suffix_probe_frames_for_score=refine_suffix_probes,
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


def _test_phase_simplex_inversion_exact() -> None:
    """init_tau_phase_params(init_tau_values_full) → tau_from_phase_params
    reproduces the input within tolerance for both clustered and spread
    configurations (codex v2 spec mandatory test)."""
    for tag, full_tau in [("clustered", [13.0, 15.0, 17.0]),
                          ("spread", [8.0, 18.0, 32.0])]:
        for m in [1, 2, 3]:
            p = init_tau_phase_params(
                active_K=m, full_K=3, T_clean=58,
                init_tau_values_full=full_tau,
                clamp_left=1.0, bridge_budget=4, d_min=2,
            )
            tau = tau_from_phase_params(p).detach().cpu().tolist()
            for got, want in zip(tau, full_tau):
                assert abs(got - want) < 1e-5, (
                    tag, m, got, want, tau, full_tau)
    print("  phase-simplex inversion exact (clustered + spread)")


def _test_phase_simplex_feasibility_under_random_updates() -> None:
    """Random simplex perturbations stay legal: τ[0] ≥ clamp_left,
    τ[i] - τ[i-1] ≥ d_min, τ[K-1] + bridge_budget ≤ T_clean - 1
    (codex v2 spec mandatory test)."""
    torch.manual_seed(0)
    T_clean = 58
    clamp_left = 1.0
    bridge_budget = 4
    d_min = 2
    init_full = [8.0, 18.0, 32.0]
    p = init_tau_phase_params(
        active_K=3, full_K=3, T_clean=T_clean,
        init_tau_values_full=init_full,
        clamp_left=clamp_left, bridge_budget=bridge_budget, d_min=d_min,
    )
    for _ in range(30):
        with torch.no_grad():
            p.p_raw.add_(torch.randn_like(p.p_raw) * 0.5)
        project_simplex_inplace(p.p_raw)
        tau = tau_from_phase_params(p).detach().cpu().tolist()
        assert tau[0] >= clamp_left - 1e-3, tau
        for i in range(1, len(tau)):
            assert tau[i] >= tau[i - 1] + d_min - 1e-3, tau
        assert tau[-1] + bridge_budget <= T_clean - 1 + 1e-3, tau
        # And p_raw is on the simplex.
        assert (p.p_raw.detach() >= -1e-6).all()
        assert abs(float(p.p_raw.detach().sum()) - 1.0) < 1e-5
    print("  phase-simplex feasibility under random updates")


def _test_phase_simplex_linear_jacobian() -> None:
    """The Jacobian dτ/dp is constant: shifting probability mass from
    p[3] to p[1] increases τ[1] and τ[2] by the same amount (≈ U_m·ε).
    This is the property v1 lacked — saturation in cumulative softplus
    made dτ/dg ≈ 0 at clustered d_min init (codex v2 spec mandatory)."""
    T_clean = 58
    bridge_budget = 4
    d_min = 2
    clustered = [13.0, 15.0, 17.0]
    p = init_tau_phase_params(
        active_K=3, full_K=3, T_clean=T_clean,
        init_tau_values_full=clustered,
        clamp_left=1.0, bridge_budget=bridge_budget, d_min=d_min,
    )
    # B_m = T_clean - 1 - bridge_budget = 53; U_m = 53 - 1 - 2*2 = 48.
    _, U_m = _phase_boundary_and_slack(
        T_clean, bridge_budget, 1.0, d_min, 3,
        tau_fixed_full=clustered,
    )
    eps = 0.02
    tau_before = tau_from_phase_params(p).detach().clone()
    with torch.no_grad():
        p.p_raw.data[1] += eps
        p.p_raw.data[3] -= eps
    project_simplex_inplace(p.p_raw)
    tau_after = tau_from_phase_params(p).detach().clone()
    delta = (tau_after - tau_before).cpu().tolist()
    # τ[0] depends only on p[0] → unchanged.
    # τ[1] depends on p[0]+p[1] → +U_m*eps.
    # τ[2] depends on p[0]+p[1]+p[2] → +U_m*eps (p[2] unchanged).
    expected = [0.0, U_m * eps, U_m * eps]
    for k in range(3):
        assert abs(delta[k] - expected[k]) < 0.05, (
            k, delta, expected, U_m, eps)
    print("  phase-simplex linear Jacobian: dτ/dp = U_m (no saturation)")


def _test_phase_simplex_optimizer_can_escape_cluster() -> None:
    """THE missing test from R3: from clustered d_min init, can Adam
    actually move τ? In v1, cumulative softplus saturated at g≈-13.8
    → 12 Adam steps moved softplus from 0 to ~4e-4 (effectively 0).
    v2 simplex parameterization should escape easily.

    Synthetic objective L = -(τ[1] + τ[2]). 10 Adam steps with lr=0.05
    should push τ[2] up by ≥ 2 frames (a measurable escape from the
    d_min lower bound)."""
    T_clean = 58
    bridge_budget = 4
    d_min = 2
    clustered = [13.0, 15.0, 17.0]
    p = init_tau_phase_params(
        active_K=3, full_K=3, T_clean=T_clean,
        init_tau_values_full=clustered,
        clamp_left=1.0, bridge_budget=bridge_budget, d_min=d_min,
    )
    tau_init = tau_from_phase_params(p).detach().clone()
    opt = torch.optim.Adam([p.p_raw], lr=0.05)
    for _ in range(10):
        opt.zero_grad()
        tau = tau_from_phase_params(p)
        L = -(tau[1] + tau[2])
        L.backward()
        opt.step()
        project_simplex_inplace(p.p_raw)
    tau_final = tau_from_phase_params(p).detach().clone()
    delta_2 = float(tau_final[2] - tau_init[2])
    delta_1 = float(tau_final[1] - tau_init[1])
    assert delta_2 >= 2.0, (
        f"τ[2] must move ≥ 2 frames from clustered init; got Δτ[2]="
        f"{delta_2:.3f}, tau_init={tau_init.tolist()}, "
        f"tau_final={tau_final.tolist()}. THE saturation bug is back.")
    assert delta_1 >= 1.0, (
        f"τ[1] must move ≥ 1 frame; got Δτ[1]={delta_1:.3f}")
    print(f"  cluster-escape OK (Δτ[1]={delta_1:.2f}, Δτ[2]={delta_2:.2f})")


def _test_simplex_projection_basic() -> None:
    """project_simplex_inplace produces non-negative entries summing to 1."""
    torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn(5)
        original_x = x.clone()
        project_simplex_inplace(x)
        assert (x >= -1e-6).all(), x
        assert abs(float(x.sum()) - 1.0) < 1e-5, x
        # Idempotence: re-projection is a no-op.
        x_proj = x.clone()
        project_simplex_inplace(x_proj)
        assert torch.allclose(x_proj, x, atol=1e-6), (x, x_proj)
    print("  simplex projection: non-negative, sums to 1, idempotent")


def _test_blend_simplex_with_uniform() -> None:
    """blend_simplex_with_uniform pulls a degenerate one-hot toward
    uniform, restoring viable corner enumeration after singleton
    collapse (codex v2 spec degeneracy recovery)."""
    p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    p.requires_grad_(True)
    blend_simplex_with_uniform(p, weight=0.10)
    # 90% original + 10% uniform of size 4 = 0.025 each
    expected = torch.tensor([0.925, 0.025, 0.025, 0.025])
    assert torch.allclose(p.detach(), expected, atol=1e-5), p
    # Sums to 1, all non-negative.
    assert abs(float(p.detach().sum()) - 1.0) < 1e-5
    assert (p.detach() >= -1e-6).all()
    print("  uniform blend: pulls one-hot toward uniform")


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
    # v2: tau_full_state tracks the full K-tau across phases. The
    # TauPhaseParams is rebuilt per phase from this state.
    tau_full_state: List[float] = [3.0, 9.0, 16.0]
    bridge_budget = cfg.oracle_traj_bridge_length + 1
    init_state = state_cache.get(
        tuple(int(round(v)) for v in tau_full_state))
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

    R_baseline_inactive = R.data[1:].clone()
    anchor_initial = traj.anchor_offset.detach().clone()
    delta_initial = traj.delta_offset.detach().clone()
    alpha_initial = edit_p.alpha_logits.detach().clone()
    nu_initial = nu.detach().clone()
    tau_initial = list(tau_full_state)

    def _build_phase(active_K: int) -> Tuple[TauPhaseParams,
                                              torch.optim.Optimizer]:
        tau_p = init_tau_phase_params(
            active_K=active_K, full_K=3, T_clean=T_clean,
            init_tau_values_full=tau_full_state,
            clamp_left=1.0, bridge_budget=bridge_budget, d_min=2,
        )
        opt = torch.optim.Adam([
            {"params": [tau_p.p_raw], "lr": 0.05},
            {"params": [traj.anchor_offset], "lr": 0.1},
            {"params": [traj.delta_offset], "lr": 0.1},
            {"params": [edit_p.alpha_logits], "lr": 0.1},
            {"params": [edit_p.warp_s, edit_p.warp_r], "lr": 0.1},
            {"params": [nu], "lr": 0.01},
        ])
        return tau_p, opt

    # K=1 phase: 2 steps. R must NOT be active (phase_idx < K-1).
    tau_p, opt = _build_phase(1)
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
        assert diag.valid_corner_count <= 2, diag.valid_corner_count
        assert diag.valid_corner_count >= 1, diag.valid_corner_count
        tau_full_state = list(diag.tau_values)

    # R inactive slices unchanged (Phase A).
    assert torch.allclose(R.data[1:], R_baseline_inactive)
    # v2: TauPhaseParams' fixed-tail enforces τ[1], τ[2] don't move
    # in K=1 phase (structural, not gradient gating).
    assert abs(tau_full_state[1] - tau_initial[1]) < 1e-3, \
        f"τ[1] (inactive in K=1, fixed-tail) drifted: {tau_full_state[1]} vs {tau_initial[1]}"
    assert abs(tau_full_state[2] - tau_initial[2]) < 1e-3, \
        f"τ[2] (inactive in K=1, fixed-tail) drifted: {tau_full_state[2]} vs {tau_initial[2]}"
    # codex R3 HIGH fix: inactive insert Adam-params must NOT have moved
    # under K=1 phase (only insert 0 is active).
    assert torch.allclose(traj.anchor_offset[1:], anchor_initial[1:]), \
        "anchor_offset[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(traj.delta_offset[1:], delta_initial[1:]), \
        "delta_offset[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(edit_p.alpha_logits[1:], alpha_initial[1:]), \
        "alpha_logits[1:] (inactive in K=1) should be unchanged"
    assert torch.allclose(nu[1:], nu_initial[1:]), \
        "nu[1:] (inactive in K=1) should be unchanged"

    # K=2 phase: 2 steps with optimizer rebuild.
    tau_p, opt = _build_phase(2)
    tau_2_before = tau_full_state[2]
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
        assert diag.valid_corner_count <= 4, diag.valid_corner_count
        tau_full_state = list(diag.tau_values)
    # τ[2] (still inactive in K=2) MUST not move under fixed-tail.
    assert abs(tau_full_state[2] - tau_2_before) < 1e-3, \
        f"τ[2] (inactive in K=2) drifted: {tau_full_state[2]} vs {tau_2_before}"
    R_after_K2 = R.data.clone()

    # K=3 phase: 2 steps, R active.
    tau_p, opt = _build_phase(3)
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
        assert diag.valid_corner_count <= 8, diag.valid_corner_count

    # R must have moved during K=3 (sign-PGD ran).
    assert (R.data - R_after_K2).abs().max() > 0, \
        "R should have moved during K=3 phase"
    print("  curriculum K1/K2/K3 smoke (v2): fixed-tail freeze + R Phase B OK")


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
    tau_p = init_tau_phase_params(
        active_K=3, full_K=3, T_clean=T_clean,
        init_tau_values_full=init_tau,
        clamp_left=1.0, bridge_budget=cfg.oracle_traj_bridge_length + 1,
        d_min=2,
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
        {"params": [tau_p.p_raw], "lr": 0.001},   # small lr -> minimal τ drift
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


def _test_prescreen_rejects_late_cluster_spike() -> None:
    """Coverage-aware selector rejects late-frame cluster even when raw
    relevance peaks there (codex v2 spec mandatory test).

    Construct a synthetic candidate table where late frames 50-54 have
    the highest raw vulnerability, but earlier frames 8/18/32 also
    score well. The h(c) factor should pull the late spikes down so
    MMR picks an early-mid-late spread."""
    T_clean = 58
    bridge_budget = 4
    clamp_left = 1.0
    d_min = 2

    # Synthetic raw scores with late cluster + scattered earlier signal.
    # Higher v_raw = better attack at K=1.
    raw_scores: List[Tuple[int, float]] = []
    for c in range(int(clamp_left), T_clean - bridge_budget):
        if c in (50, 51, 52, 53, 54):
            v = 5.0 + 0.1 * (c - 50)        # late cluster, biggest peak
        elif c in (8, 9, 10):
            v = 4.0
        elif c in (18, 19, 20):
            v = 3.5
        elif c in (32, 33, 34):
            v = 3.0
        else:
            v = 1.0 + 0.01 * c              # generic small noise
        raw_scores.append((c, v))

    # Apply v_rank * h(c) (mirrors prescreen_tau_init internal pipeline).
    sorted_by_v = sorted(raw_scores, key=lambda kv: kv[1])
    n = len(sorted_by_v)
    v_rank = {c: float(rank) / float(max(1, n - 1))
              for rank, (c, _) in enumerate(sorted_by_v)}
    span_denom = max(1.0, float(T_clean - bridge_budget - clamp_left))
    relevance: List[Tuple[int, float]] = []
    for c, _ in raw_scores:
        h_c = max(0.0, min(1.0,
                           float(T_clean - bridge_budget - c) / span_denom))
        relevance.append((c, v_rank[c] * h_c))
    relevance.sort(key=lambda kv: -kv[1])

    usable_span = float(T_clean - 1 - bridge_budget) - float(clamp_left)
    target_gap = max(1.0, usable_span / 3.0)
    chosen = _coverage_aware_select(
        relevance, K=3, d_min=d_min, target_gap=target_gap, seed_index=0)

    # PRIMARY assertion: late cluster {50..53} must be REJECTED
    # (this is the actual dog-failure pattern we're protecting against).
    assert max(chosen) < 45, (
        f"prescreen must reject late cluster; got max={max(chosen)} in "
        f"chosen={chosen}")
    # SECONDARY: chosen should not be a tight d_min=2 cluster (the v1
    # failure mode). Any spread > 4 (i.e., not all 3 within d_min=2)
    # qualifies — MMR can legitimately pick one tight pair and one
    # spread point under 0.7/0.3 weighting.
    span = max(chosen) - min(chosen)
    assert span > 4, (
        f"chosen must not be a d_min cluster; got span={span} in "
        f"chosen={chosen} (would be the dog v1 failure pattern)")
    # No two chosen frames may be d_min-cluster (gap=2) on EVERY pair.
    gaps = [chosen[i + 1] - chosen[i] for i in range(len(chosen) - 1)]
    assert max(gaps) >= 5, (
        f"chosen has all-d_min gaps {gaps}; would replicate dog cluster.")
    print(f"  prescreen rejects late-cluster spike: chosen={chosen}")


def _test_v3_trust_region_rejects_blackswan_collapse() -> None:
    """v3 trust region must REJECT the v2 blackswan failure mode where
    prescreen W0=[4,13,21] (gap_min=8) was collapsed to [7,9,21]
    (gap_min=2). With v3 default ratios (gap_ratio=0.5, span_ratio=0.85,
    trust_radius=6), the bad cluster fails the filter.
    """
    W0 = [4, 13, 21]
    bad_cluster = (7, 9, 21)
    # span: 14 vs 17, ratio 0.82 < 0.85 → REJECT
    # gap_min: 2 vs 8·0.5=4 → REJECT
    assert _validate_schedule(
        bad_cluster, T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5,
    ) is False, "v3 trust region should reject [7,9,21] under W0=[4,13,21]"
    # The prescreen output ITSELF should pass (identity).
    assert _validate_schedule(
        tuple(W0), T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5,
    ) is True, "W0 must pass its own trust region"
    # A small perturbation [5,13,21] should pass (radius=1, span=16/17=0.94 OK).
    assert _validate_schedule(
        (5, 13, 21), T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5,
    ) is True, "small ±1 perturbation should pass"
    # A large perturbation outside trust_radius=6: c_0 → c_0+10 = 14
    assert _validate_schedule(
        (14, 16, 21), T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5,
    ) is False, "perturbation > trust_radius=6 should fail"
    print("  v3 trust region: rejects blackswan collapse [7,9,21], "
          "accepts W0 + small perturbations")


def _test_v3_27_neighbor_with_trust() -> None:
    """27-neighbor enumeration with trust region prunes invalid neighbors.

    Two scenarios:
    (A) W_round = W0 (good spread): trust filter is mostly inert (all ±1
        of healthy spread stay healthy).
    (B) W_round drifted to a collapse-style cluster: trust filter REJECTS
        all neighbors because base gap is too tight.
    """
    # Scenario A: W_round identical to W0 (clean spread)
    W0 = [4, 13, 21]
    cands_A = _enumerate_27_neighbors(
        W0, T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5)
    # Every kept neighbor should satisfy trust constraints.
    for cand in cands_A:
        gaps = [cand[i + 1] - cand[i] for i in range(len(cand) - 1)]
        assert min(gaps) >= 4, (cand, gaps)
        span = cand[-1] - cand[0]
        assert span >= 15, (cand, span)
        for k in range(len(cand)):
            assert abs(cand[k] - W0[k]) <= 6, (cand, k)
    assert tuple(W0) in cands_A, "W0 itself must pass filter"

    # Scenario B: W_round = bad cluster (the v2 blackswan collapse).
    # Most ±1 neighbors of the bad cluster have gap_min ≤ 3 < 4 (the
    # required floor) and are REJECTED. A few neighbors that *move
    # toward* W0 may still pass (e.g. (6, 10, 21) has gap=4 boundary +
    # span=15 boundary). Important: the bad cluster ITSELF must fail.
    W_bad = (7, 9, 21)
    cands_B = _enumerate_27_neighbors(
        list(W_bad), T_clean=50, bridge_budget=4, d_min=2,
        W0=W0, trust_radius=6, span_ratio=0.85, gap_ratio=0.5)
    cands_B_no_trust = _enumerate_27_neighbors(
        list(W_bad), T_clean=50, bridge_budget=4, d_min=2)
    # Without trust: many valid (gap≥d_min=2 is permissive)
    assert len(cands_B_no_trust) > 0, "without trust some should pass"
    # With trust: most filtered out. Tight assertion: bad cluster itself
    # MUST NOT be in candidates.
    assert W_bad not in cands_B, (
        f"trust region must reject bad cluster {W_bad}; got it in {cands_B}")
    # Filter ratio: at least 70% of no-trust candidates must be filtered.
    filter_ratio = 1.0 - len(cands_B) / max(1, len(cands_B_no_trust))
    assert filter_ratio >= 0.70, (
        f"trust filter too lenient: kept {len(cands_B)} of "
        f"{len(cands_B_no_trust)} = {(1-filter_ratio):.0%}")
    # Every kept neighbor must satisfy trust constraints.
    for cand in cands_B:
        gaps = [cand[i + 1] - cand[i] for i in range(len(cand) - 1)]
        assert min(gaps) >= 4, (cand, gaps)
        span = cand[-1] - cand[0]
        assert span >= 0.85 * 17 - 1e-6, (cand, span)
    print(f"  27-neighbor + trust region: scenario A kept {len(cands_A)}, "
          f"scenario B (bad cluster) kept {len(cands_B)} of "
          f"{len(cands_B_no_trust)} (filter rate {filter_ratio:.0%})")


def _self_test() -> None:
    print("memshield.joint_placement_search self-tests:")
    _test_simplex_projection_basic()
    _test_blend_simplex_with_uniform()
    _test_phase_simplex_inversion_exact()
    _test_phase_simplex_feasibility_under_random_updates()
    _test_phase_simplex_linear_jacobian()
    _test_phase_simplex_optimizer_can_escape_cluster()
    _test_schedule_weight_sum_and_integer_exactness()
    _test_schedule_filtering_invalid_orderings()
    _test_schedule_grad_flows_through_weights()
    _test_R_active_slice_mask_locality()
    _test_27_neighbor_enumeration_filters_invalid()
    _test_bundle_C_incompat_guard()
    _test_attack_state_cache_hit_miss()
    _test_prescreen_rejects_late_cluster_spike()
    _test_v3_trust_region_rejects_blackswan_collapse()
    _test_v3_27_neighbor_with_trust()
    _test_curriculum_smoke_K1_K2_K3()
    _test_attack_state_cache_warmup_in_search()
    print("memshield.joint_placement_search: all v3 self-tests PASSED")


if __name__ == "__main__":
    _self_test()
