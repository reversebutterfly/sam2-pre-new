"""VADI per-video PGD driver (3-stage, local-δ + ν, LPIPS-TV-bound).

Per `refine-logs/FINAL_PROPOSAL.md` §"Per-video PGD". Composes primitives
from `memshield/vadi_loss.py` and `memshield/losses.py` (fake_uint8_quantize,
differentiable_ssim) into the main attack loop.

Trainable tensors (2):
  * δ : [T_clean, H_vid, W_vid, 3]  — per-clean-frame perturbation; nonzero
        only at positions in S_δ_clean; two-tier ε (f0: 2/255, others: 4/255).
  * ν : [K, H_vid, W_vid, 3]        — per-insert perturbation vs the temporal-
        midframe base; no ε bound (constrained by LPIPS + TV hinges).

Stages (100 steps total):
  N_1 = 30 — attack-only (λ = 0)
  N_2 = 40 — fidelity regularization; λ init=10, grow ×2 per 10 steps
             when any fidelity hinge violated
  N_3 = 30 — Pareto-best tracking; η halved

Feasibility and artifact acceptance:
  - fake_uint8_quantize (STE) is applied every step, so internal metrics ARE
    the exported-uint8 metrics (no separate JPEG round-trip in this version).
  - `step_feasible = all(hinges <= 1e-6)` (tolerance for bf16/fp16 noise).
  - Running-best (δ*, ν*) = feasible step with maximum surrogate_J_drop.
  - If no feasible step: clip = INFEASIBLE (primary-denominator failure).
  - Host-RAM optimization: `(δ, ν)` snapshots are taken only on FEASIBLE
    steps (infeasible `StepLog.delta_snapshot` is None), capping memory at
    O(|feasible steps|) rather than O(total steps).

`forward_fn` contract:
    forward_fn(x_processed: Tensor[T_proc, H, W, 3], return_logits_at: Iterable[int])
      -> Dict[int, Tensor[H_vid, W_vid]]  # per-frame pred_logits

Callers wire this to `SAM2VideoAdapter` in `scripts/run_vadi.py`. The self-
test here uses a differentiable stub forward so PGD dynamics are validated
without SAM2.

Run `python -m memshield.vadi_optimize` for self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from memshield.losses import fake_uint8_quantize
from memshield.vadi_loss import (
    AggregatedMarginLoss,
    MarginTrace,
    aggregate_margin_loss,
    decoy_margin_per_frame,
    lpips_cap_hinge,
    ssim_floor_hinge,
    total_variation,
    tv_hinge,
)


# =============================================================================
# Config
# =============================================================================


@dataclass
class VADIConfig:
    """All numerical settings for VADI PGD. Defaults mirror FINAL_PROPOSAL.md."""

    # Stage lengths (sum = total PGD steps, proposal = 100).
    N_1: int = 30
    N_2: int = 40
    N_3: int = 30

    # Step size (sign-grad). η halves in stage 3.
    eta: float = 2.0 / 255.0

    # Contrastive margin hyperparams.
    margin: float = 0.75
    neighbor_weight: float = 0.5

    # Two-tier epsilon on δ. Both in pixel units ∈ [0, 1].
    eps_delta_f0: float = 2.0 / 255.0
    eps_delta_other: float = 4.0 / 255.0

    # Fidelity budgets. LPIPS caps, TV multiplier, SSIM floor.
    lpips_orig_cap: float = 0.20
    lpips_insert_cap: float = 0.35
    tv_multiplier: float = 1.2
    f0_ssim_floor: float = 0.98

    # ν initialization.
    nu_init_std: float = 0.02 / 255.0

    # λ schedule.
    lambda_init: float = 10.0
    lambda_growth_factor: float = 2.0
    lambda_growth_period: int = 10        # grow every N steps within stage 2
    lambda_f0: float = 100.0              # always strong on the prompt-frame hinge

    # Sanity.
    seed: int = 0


# =============================================================================
# Geometric utilities (frame-index bookkeeping)
# =============================================================================


def nbr_set(m: int, T: int, radius: int = 2) -> List[int]:
    """NbrSet(m) = {m±1, m±2} ∩ [0, T-1]. Does NOT include m itself."""
    return sorted({m + d for d in range(-radius, radius + 1)
                   if d != 0 and 0 <= m + d < T})


def build_support_sets(
    W: Sequence[int], T_processed: int,
    f0_processed_id: int = 0, radius: int = 2,
) -> Tuple[List[int], List[int]]:
    """Return (S_delta_processed, neighbor_ids_processed) in attacked-space.

    S_δ = (∪_k NbrSet(W_k)) ∪ {f0}  — where δ's two-tier clip applies
    neighbor_ids = (∪_k NbrSet(W_k)) \\ W  — margin-loss neighbor set

    Note both are attacked-space (post-insertion) indices. Caller maps these
    to clean-space via `attacked_to_clean()` when building δ masks.
    """
    nbr_union: set = set()
    for w in W:
        nbr_union |= set(nbr_set(int(w), T_processed, radius))
    S_delta = nbr_union | {int(f0_processed_id)}
    neighbor_ids = nbr_union - set(int(w) for w in W)
    return sorted(S_delta), sorted(neighbor_ids)


def attacked_to_clean(attacked_idx: int, W: Sequence[int]) -> int:
    """Map an attacked-space processing index to its clean-space frame index.

    Raises ValueError if `attacked_idx` itself is an insert position.

    For W=[2,5] at attacked_idx=3: one prior insert (at W_0=2) → clean_idx=2.
    For attacked_idx=7 with W=[2,5]: two prior inserts → clean_idx=5.
    """
    W_sorted = sorted(int(w) for w in W)
    if attacked_idx in W_sorted:
        raise ValueError(
            f"attacked_idx={attacked_idx} IS an insert; no clean mapping")
    k_before = sum(1 for w in W_sorted if w < attacked_idx)
    return attacked_idx - k_before


def build_base_inserts(x_clean: Tensor, W: Sequence[int]) -> Tensor:
    """Temporal-midframe bases, one per insert.

    base_insert_k = 0.5 · clean[c_k - 1] + 0.5 · clean[c_k],
    where c_k = W_k - k is the clean-frame index at the "right" of the
    gap where insert k will sit. Returns [K, H, W, 3].
    """
    W_sorted = sorted(int(w) for w in W)
    T_clean = x_clean.shape[0]
    bases = []
    for k, w in enumerate(W_sorted):
        c_k = w - k
        if c_k < 1 or c_k >= T_clean:
            raise ValueError(
                f"insert {k} at W_k={w} maps to clean index c_k={c_k}; "
                f"need 1 ≤ c_k < {T_clean}")
        bases.append(0.5 * x_clean[c_k - 1] + 0.5 * x_clean[c_k])
    return torch.stack(bases, dim=0)


def build_processed(
    x_prime: Tensor, inserts: Tensor, W: Sequence[int],
) -> Tensor:
    """Interleave δ-applied clean frames with ν-applied inserts.

    x_prime:  [T_clean, H, W, 3]
    inserts:  [K, H, W, 3]
    W:        ATTACKED-space positions of the K inserts (will be sorted).
    Returns:  [T_clean + K, H, W, 3]
    """
    W_sorted = sorted(int(w) for w in W)
    K = len(W_sorted)
    if inserts.shape[0] != K:
        raise ValueError(f"inserts.shape[0]={inserts.shape[0]} != len(W)={K}")
    T_clean = x_prime.shape[0]
    T_proc = T_clean + K
    for w in W_sorted:
        if not 0 <= w < T_proc:
            raise ValueError(f"W={w} out of range [0, {T_proc})")

    # Build via torch.stack (differentiable). `torch.empty() + item_assign`
    # would look equivalent but breaks autograd from `out` back to `x_prime`
    # / `inserts` — severing the margin-loss gradient path to δ and ν.
    insert_pos_to_k = {w: k for k, w in enumerate(W_sorted)}
    frames: List[Tensor] = []
    clean_i = 0
    for t in range(T_proc):
        if t in insert_pos_to_k:
            frames.append(inserts[insert_pos_to_k[t]])
        else:
            frames.append(x_prime[clean_i])
            clean_i += 1
    return torch.stack(frames, dim=0)


# =============================================================================
# State, logs, result
# =============================================================================


@dataclass
class VADIState:
    """The two trainable tensors. Both are leaf with requires_grad=True."""

    delta: Tensor    # [T_clean, H, W, 3] — nonzero only on S_δ_clean
    nu: Tensor       # [K, H, W, 3]
    step: int = 0


@dataclass
class VADIInputs:
    """Everything the PGD loop needs that's fixed across steps."""

    x_clean: Tensor                                # [T_clean, H, W, 3] in [0,1]
    base_inserts: Tensor                           # [K, H, W, 3] in [0,1]
    W: List[int]                                   # attacked-space, sorted
    S_delta_clean: List[int]                       # clean-space δ support (incl. f0)
    insert_ids_processed: List[int]                # = W
    neighbor_ids_processed: List[int]              # NbrSet \\ W in attacked-space
    m_hat_true_by_t: Dict[int, Tensor]             # processed-idx → [H_vid, W_vid]
    m_hat_decoy_by_t: Dict[int, Tensor]
    f0_clean_id: int                               # typically 0
    # Caller's responsibility (upstream of this driver):
    #   - m_hat_true_by_t / m_hat_decoy_by_t must cover every t in
    #     `insert_ids_processed ∪ neighbor_ids_processed`. Build via
    #     clean-SAM2 pseudo-mask + decoy shift, then REMAP from clean-space
    #     time indices to attacked-space by accounting for inserts.
    #   - `forward_fn` must NOT wrap SAM2 in torch.no_grad(); freeze weights
    #     via requires_grad_(False) instead and keep the input graph live.


@dataclass
class StepLog:
    """Per-step diagnostic record.

    `delta_snapshot` / `nu_snapshot` are populated only on FEASIBLE steps
    (all hinges ≤ tol); on infeasible steps both are `None` to cap host
    RAM. Both snapshots, when present, correspond to the (δ, ν) state the
    metrics on this log were computed against — i.e. the PRE-update state
    of the PGD iteration that produced this log.
    """

    step: int
    stage: int                                     # 1, 2, 3
    loss: float
    L_margin: float
    L_margin_insert: float
    L_margin_neighbor: float
    L_fid_orig: float
    L_fid_insert: float
    L_fid_TV: float
    L_fid_f0: float
    lambda_val: float
    eta: float
    ssim_f0: float
    per_frame_lpips_orig: Dict[int, float]         # clean-space idx → float
    per_insert_lpips: Dict[int, float]             # k → float
    per_insert_tv_excess: Dict[int, float]         # k → float (0 when below budget)
    mu_true_trace: Dict[int, float]
    mu_decoy_trace: Dict[int, float]
    surrogate_J_drop: float
    feasible: bool                                 # all budgets satisfied at step
    delta_snapshot: Optional[Tensor] = None
    nu_snapshot: Optional[Tensor] = None


@dataclass
class VADIResult:
    delta_star: Optional[Tensor]                   # None if INFEASIBLE
    nu_star: Optional[Tensor]
    best_step: int                                 # -1 if INFEASIBLE
    best_surrogate_J_drop: float                   # -1.0 if INFEASIBLE
    step_logs: List[StepLog]
    infeasible: bool


# =============================================================================
# Per-step helpers
# =============================================================================


def _apply_delta(x_clean: Tensor, delta: Tensor) -> Tensor:
    """x' = fake_uint8_quantize(clamp(x_clean + δ, 0, 1)). Per-frame."""
    return fake_uint8_quantize(torch.clamp(x_clean + delta, 0.0, 1.0))


def _apply_nu(base_inserts: Tensor, nu: Tensor) -> Tensor:
    """insert = fake_uint8_quantize(clamp(base + ν, 0, 1))."""
    return fake_uint8_quantize(torch.clamp(base_inserts + nu, 0.0, 1.0))


def _clip_delta_two_tier(
    delta: Tensor,
    S_delta_clean: Sequence[int],
    f0_clean_id: int,
    eps_f0: float,
    eps_other: float,
) -> None:
    """In-place two-tier ε clip + zero-out of non-S_δ frames.

    - Positions NOT in S_δ: δ := 0
    - f0: clamp to ±eps_f0
    - Other S_δ positions: clamp to ±eps_other
    """
    S = set(int(s) for s in S_delta_clean)
    T_clean = delta.shape[0]
    with torch.no_grad():
        for t in range(T_clean):
            if t not in S:
                delta[t].zero_()
            elif t == f0_clean_id:
                delta[t].clamp_(-eps_f0, eps_f0)
            else:
                delta[t].clamp_(-eps_other, eps_other)


def _mask_gradient_to_support(delta: Tensor, S_delta_clean: Sequence[int]) -> None:
    """Zero out δ's gradient at frames not in S_δ. Called before step()."""
    if delta.grad is None:
        return
    S = set(int(s) for s in S_delta_clean)
    T_clean = delta.shape[0]
    for t in range(T_clean):
        if t not in S:
            delta.grad[t].zero_()


def _lambda_at_step(
    step_idx_in_loop: int, config: VADIConfig,
    stage2_lambda_state: Dict[str, Any],
) -> Tuple[float, int]:
    """Return (λ for this step, stage ∈ {1,2,3}).

    Stage 1: λ = 0. Stage 2: λ starts at `lambda_init`; every
    `lambda_growth_period` steps (within stage 2), if any hinge was violated
    during the PRIOR step, multiply λ by `lambda_growth_factor`. Stage 3:
    λ is frozen at whatever stage-2 left it.
    """
    N1, N2 = config.N_1, config.N_2
    if step_idx_in_loop < N1:
        return 0.0, 1
    if step_idx_in_loop < N1 + N2:
        return stage2_lambda_state["lambda_val"], 2
    return stage2_lambda_state["lambda_val"], 3


def _eta_at_step(step_idx_in_loop: int, config: VADIConfig) -> float:
    """η is halved in stage 3 only."""
    if step_idx_in_loop < config.N_1 + config.N_2:
        return config.eta
    return 0.5 * config.eta


# =============================================================================
# The step
# =============================================================================


def vadi_step(
    state: VADIState,
    inputs: VADIInputs,
    forward_fn: Callable[[Tensor, Iterable[int]], Dict[int, Tensor]],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    config: VADIConfig,
    lambda_val: float,
    eta: float,
    stage: int,
) -> StepLog:
    """One PGD iteration. Mutates `state.delta` and `state.nu` in-place via
    sign-grad + two-tier ε clip. Returns a StepLog with detached diagnostics.
    """
    # -- Build the processed sequence under current δ, ν -----------------
    x_prime = _apply_delta(inputs.x_clean, state.delta)          # [T_clean, H, W, 3]
    inserts = _apply_nu(inputs.base_inserts, state.nu)           # [K, H, W, 3]
    processed = build_processed(x_prime, inserts, inputs.W)      # [T_proc, H, W, 3]

    # -- Forward SAM2 (stub-swappable) --------------------------------------
    return_at = set(inputs.insert_ids_processed) \
        | set(inputs.neighbor_ids_processed)
    pred_logits_by_t = forward_fn(processed, return_at)

    # -- Margin term ---------------------------------------------------------
    margins_by_t = {}
    for t in return_at:
        if t not in pred_logits_by_t:
            raise KeyError(
                f"forward_fn did not return logits for processed idx {t}")
        margins_by_t[t] = decoy_margin_per_frame(
            pred_logits_by_t[t],
            inputs.m_hat_true_by_t[t],
            inputs.m_hat_decoy_by_t[t],
            margin=config.margin,
        )
    agg = aggregate_margin_loss(
        margins_by_t,
        insert_ids=inputs.insert_ids_processed,
        neighbor_ids=inputs.neighbor_ids_processed,
        neighbor_weight=config.neighbor_weight,
    )

    # -- Fidelity hinges -----------------------------------------------------
    # LPIPS per original frame in S_δ \ {f0} (clean-space).
    per_frame_lpips_orig: Dict[int, float] = {}
    L_fid_orig = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
    for c in inputs.S_delta_clean:
        if c == inputs.f0_clean_id:
            continue
        lp = lpips_fn(x_prime[c], inputs.x_clean[c])
        per_frame_lpips_orig[c] = float(lp.detach().item())
        L_fid_orig = L_fid_orig + lpips_cap_hinge(lp, config.lpips_orig_cap)

    # LPIPS + TV per insert.
    per_insert_lpips: Dict[int, float] = {}
    per_insert_tv_excess: Dict[int, float] = {}
    L_fid_ins = torch.zeros_like(L_fid_orig)
    L_fid_TV = torch.zeros_like(L_fid_orig)
    for k in range(inserts.shape[0]):
        lp = lpips_fn(inserts[k], inputs.base_inserts[k])
        per_insert_lpips[k] = float(lp.detach().item())
        L_fid_ins = L_fid_ins + lpips_cap_hinge(lp, config.lpips_insert_cap)
        # Per-k TV (use 3-D, returns scalar).
        ins_chw = inserts[k].permute(2, 0, 1)
        base_chw = inputs.base_inserts[k].permute(2, 0, 1)
        tv_h = tv_hinge(ins_chw, base_chw, multiplier=config.tv_multiplier)
        per_insert_tv_excess[k] = float(tv_h.detach().item())
        L_fid_TV = L_fid_TV + tv_h

    # SSIM on f0. `ssim_fn` expects [B, C, H, W].
    f0c = inputs.f0_clean_id
    f0_x = x_prime[f0c].permute(2, 0, 1).unsqueeze(0)            # [1, 3, H, W]
    f0_y = inputs.x_clean[f0c].permute(2, 0, 1).unsqueeze(0)
    ssim_val = ssim_fn(f0_x, f0_y).squeeze()
    L_fid_f0 = ssim_floor_hinge(ssim_val, floor=config.f0_ssim_floor)

    # -- Composite loss ------------------------------------------------------
    L = (
        agg.L_margin
        + lambda_val * (L_fid_orig + L_fid_ins + L_fid_TV)
        + config.lambda_f0 * L_fid_f0
    )

    # -- Feasibility decision (on hinges we just computed; pre-mutation) ----
    _FEAS_TOL = 1e-6
    feasible = (
        L_fid_orig.detach().item() <= _FEAS_TOL
        and L_fid_ins.detach().item() <= _FEAS_TOL
        and L_fid_TV.detach().item() <= _FEAS_TOL
        and L_fid_f0.detach().item() <= _FEAS_TOL
    )

    # -- Snapshot ONLY on feasible steps (host-RAM cap O(|feasible|) rather
    # than O(total_steps)). When present, snapshots correspond to the (δ, ν)
    # state BEFORE the upcoming in-place mutation — matching the metrics on
    # this log. Detached and moved to CPU so the running-best payload
    # doesn't hold GPU memory.
    if feasible:
        delta_snapshot: Optional[Tensor] = (
            state.delta.detach().to("cpu").clone())
        nu_snapshot: Optional[Tensor] = state.nu.detach().to("cpu").clone()
    else:
        delta_snapshot = None
        nu_snapshot = None

    # -- Backward + sign-grad step ------------------------------------------
    if state.delta.grad is not None:
        state.delta.grad.zero_()
    if state.nu.grad is not None:
        state.nu.grad.zero_()
    L.backward()
    _mask_gradient_to_support(state.delta, inputs.S_delta_clean)

    with torch.no_grad():
        if state.delta.grad is not None:
            state.delta.add_(-eta * state.delta.grad.sign())
        if state.nu.grad is not None:
            state.nu.add_(-eta * state.nu.grad.sign())
        _clip_delta_two_tier(
            state.delta, inputs.S_delta_clean,
            inputs.f0_clean_id,
            config.eps_delta_f0, config.eps_delta_other,
        )
    state.step += 1

    # -- Surrogate J-drop ---------------------------------------------------
    with torch.no_grad():
        Js: List[float] = []
        for t, logits in pred_logits_by_t.items():
            pred_bin = (torch.sigmoid(logits) > 0.5).float()
            true_bin = (inputs.m_hat_true_by_t[t] > 0.5).float()
            inter = (pred_bin * true_bin).sum()
            union = torch.clamp(pred_bin + true_bin, max=1.0).sum()
            Js.append(
                float((inter / union).item()) if union.item() > 0 else 1.0)
        surrogate_J = float(sum(Js) / len(Js)) if Js else 1.0
        surrogate_J_drop = 1.0 - surrogate_J

    return StepLog(
        step=state.step, stage=stage,
        loss=float(L.detach().item()),
        L_margin=float(agg.L_margin.detach().item()),
        L_margin_insert=float(agg.L_insert.detach().item()),
        L_margin_neighbor=float(agg.L_neighbor.detach().item()),
        L_fid_orig=float(L_fid_orig.detach().item()),
        L_fid_insert=float(L_fid_ins.detach().item()),
        L_fid_TV=float(L_fid_TV.detach().item()),
        L_fid_f0=float(L_fid_f0.detach().item()),
        lambda_val=float(lambda_val),
        eta=float(eta),
        ssim_f0=float(ssim_val.detach().item()),
        per_frame_lpips_orig=per_frame_lpips_orig,
        per_insert_lpips=per_insert_lpips,
        per_insert_tv_excess=per_insert_tv_excess,
        mu_true_trace={t: float(m.mu_true.detach().item())
                       for t, m in margins_by_t.items()},
        mu_decoy_trace={t: float(m.mu_decoy.detach().item())
                        for t, m in margins_by_t.items()},
        surrogate_J_drop=surrogate_J_drop,
        feasible=feasible,
        delta_snapshot=delta_snapshot,
        nu_snapshot=nu_snapshot,
    )


# =============================================================================
# The PGD loop
# =============================================================================


def _init_state(
    inputs: VADIInputs, config: VADIConfig, device: torch.device,
) -> VADIState:
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    delta = torch.zeros_like(inputs.x_clean, device=device)
    delta.requires_grad_(True)
    K = inputs.base_inserts.shape[0]
    nu = (config.nu_init_std
          * torch.randn(K, *inputs.base_inserts.shape[1:],
                        generator=g, dtype=inputs.base_inserts.dtype)
          .to(device))
    nu.requires_grad_(True)
    return VADIState(delta=delta, nu=nu, step=0)


def run_vadi_pgd(
    inputs: VADIInputs,
    forward_fn: Callable[[Tensor, Iterable[int]], Dict[int, Tensor]],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    config: Optional[VADIConfig] = None,
    device: Optional[torch.device] = None,
) -> VADIResult:
    """Run the 100-step (N_1+N_2+N_3) PGD. Returns the running-best feasible
    (δ*, ν*) or INFEASIBLE if no step satisfies every hinge simultaneously.
    """
    config = config or VADIConfig()
    device = device or inputs.x_clean.device
    state = _init_state(inputs, config, device)

    total_steps = config.N_1 + config.N_2 + config.N_3
    stage2_lambda_state: Dict[str, Any] = {
        "lambda_val": config.lambda_init,
        "steps_since_growth": 0,
        "hinge_violated_window": False,
    }

    logs: List[StepLog] = []
    best_step = -1
    best_J_drop = float("-inf")
    best_delta: Optional[Tensor] = None
    best_nu: Optional[Tensor] = None

    for step_i in range(total_steps):
        lam, stage = _lambda_at_step(step_i, config, stage2_lambda_state)
        eta = _eta_at_step(step_i, config)
        log = vadi_step(
            state=state, inputs=inputs,
            forward_fn=forward_fn, lpips_fn=lpips_fn, ssim_fn=ssim_fn,
            config=config, lambda_val=lam, eta=eta, stage=stage,
        )
        logs.append(log)

        # λ growth trigger excludes L_fid_f0 — f0 is governed by config.lambda_f0
        # (a fixed strong coefficient), so growing `lambda_val` has no corrective
        # effect on f0 and would pollute the trigger.
        any_violated = (log.L_fid_orig > 0.0 or log.L_fid_insert > 0.0
                        or log.L_fid_TV > 0.0)
        if stage == 2:
            if any_violated:
                stage2_lambda_state["hinge_violated_window"] = True
            stage2_lambda_state["steps_since_growth"] += 1
            if stage2_lambda_state["steps_since_growth"] >= config.lambda_growth_period:
                if stage2_lambda_state["hinge_violated_window"]:
                    stage2_lambda_state["lambda_val"] *= config.lambda_growth_factor
                stage2_lambda_state["steps_since_growth"] = 0
                stage2_lambda_state["hinge_violated_window"] = False

        # Running-best: use the (δ, ν) snapshot attached to the log, which
        # corresponds to the state the feasibility/J-drop metrics were
        # computed on — NOT the post-update state living in `state` now.
        if log.feasible and log.surrogate_J_drop > best_J_drop:
            best_J_drop = log.surrogate_J_drop
            best_step = log.step
            best_delta = log.delta_snapshot           # already on CPU
            best_nu = log.nu_snapshot

    infeasible = best_delta is None
    return VADIResult(
        delta_star=best_delta, nu_star=best_nu,
        best_step=best_step if not infeasible else -1,
        best_surrogate_J_drop=best_J_drop if not infeasible else -1.0,
        step_logs=logs, infeasible=infeasible,
    )


# =============================================================================
# Self-tests (differentiable stub forward + stub LPIPS/SSIM)
# =============================================================================


def _self_test() -> None:
    torch.manual_seed(0)

    # --- nbr_set, build_support_sets, attacked_to_clean --------------------
    assert nbr_set(5, 10) == [3, 4, 6, 7]
    assert nbr_set(0, 10) == [1, 2]                                    # left edge
    assert nbr_set(9, 10) == [7, 8]                                    # right edge
    S_d, nbr = build_support_sets([3, 7], T_processed=10, f0_processed_id=0)
    # NbrSet(3) = {1,2,4,5}; NbrSet(7) = {5,6,8,9}. Union = {1,2,4,5,6,8,9}.
    # S_δ = that ∪ {0} = {0,1,2,4,5,6,8,9}.
    assert S_d == [0, 1, 2, 4, 5, 6, 8, 9]
    # neighbor_ids = union \\ W = {1,2,4,5,6,8,9} \\ {3,7} = same (no overlap).
    assert nbr == [1, 2, 4, 5, 6, 8, 9]
    # attacked_to_clean: W=[2, 5], attacked=3 → 1 prior insert → clean=2.
    assert attacked_to_clean(3, [2, 5]) == 2
    # attacked=7 → 2 prior inserts → clean=5.
    assert attacked_to_clean(7, [2, 5]) == 5
    # attacked IS an insert → raises.
    try:
        attacked_to_clean(2, [2, 5])
        raise AssertionError("insert index must raise")
    except ValueError:
        pass

    # --- build_base_inserts, build_processed -------------------------------
    T_clean, H, W, C = 5, 6, 6, 3
    x_clean = torch.linspace(0.0, 1.0, T_clean * H * W * C) \
        .reshape(T_clean, H, W, C)
    W_att = [2, 5]                                                     # K=2
    base = build_base_inserts(x_clean, W_att)
    assert base.shape == (2, H, W, C)
    # c_0 = 2, c_1 = 4. base_0 = 0.5*clean[1] + 0.5*clean[2].
    assert torch.allclose(base[0], 0.5 * x_clean[1] + 0.5 * x_clean[2])
    assert torch.allclose(base[1], 0.5 * x_clean[3] + 0.5 * x_clean[4])
    # build_processed: with W=[2,5], T_proc=7.
    fake_inserts = torch.full((2, H, W, C), 0.5)
    proc = build_processed(x_clean, fake_inserts, W_att)
    assert proc.shape == (T_clean + 2, H, W, C)
    # Position 2 and 5 are inserts.
    assert torch.allclose(proc[2], fake_inserts[0])
    assert torch.allclose(proc[5], fake_inserts[1])
    # Clean frames fill positions [0,1,3,4,6] in order.
    assert torch.allclose(proc[0], x_clean[0])
    assert torch.allclose(proc[1], x_clean[1])
    assert torch.allclose(proc[3], x_clean[2])
    assert torch.allclose(proc[4], x_clean[3])
    assert torch.allclose(proc[6], x_clean[4])
    # Gradient MUST flow through build_processed back to inputs — this is
    # the core regression check for the torch.empty()+item-assign bug that
    # would sever autograd from margin loss to δ/ν.
    xg = x_clean.clone().requires_grad_(True)
    ig = fake_inserts.clone().requires_grad_(True)
    pg = build_processed(xg, ig, W_att)
    (pg[3].sum() + pg[2].sum()).backward()
    assert xg.grad is not None and xg.grad[2].abs().sum().item() > 0, \
        "build_processed severed autograd from clean frames"
    assert ig.grad is not None and ig.grad[0].abs().sum().item() > 0, \
        "build_processed severed autograd from inserts"

    # --- _clip_delta_two_tier + support gradient mask ----------------------
    delta = torch.zeros(5, 2, 2, 3)
    # Tier f0: 0.01 (below 2/255 ≈ 0.00784 — exceeds). Clamp to ±0.00784.
    delta[0] = 0.01
    delta[2] = 0.02                                                    # other S_δ
    delta[4] = 0.5                                                     # not in S_δ → zero
    S_delta_clean = [0, 2]
    _clip_delta_two_tier(delta, S_delta_clean, f0_clean_id=0,
                         eps_f0=2.0 / 255.0, eps_other=4.0 / 255.0)
    assert delta[4].abs().max().item() == 0.0
    assert delta[0].abs().max().item() <= 2.0 / 255.0 + 1e-9
    assert delta[2].abs().max().item() <= 4.0 / 255.0 + 1e-9
    # t=1 not in S_δ either → zero.
    assert delta[1].abs().max().item() == 0.0

    # Gradient masking: only S_δ positions retain grad.
    delta2 = torch.zeros(5, 2, 2, 3, requires_grad=True)
    (delta2.sum()).backward()
    _mask_gradient_to_support(delta2, [0, 2])
    assert delta2.grad[0].abs().sum().item() > 0                       # kept
    assert delta2.grad[2].abs().sum().item() > 0                       # kept
    assert delta2.grad[1].abs().sum().item() == 0.0                    # zeroed
    assert delta2.grad[3].abs().sum().item() == 0.0
    assert delta2.grad[4].abs().sum().item() == 0.0

    # --- λ / η schedules ---------------------------------------------------
    cfg = VADIConfig(N_1=3, N_2=4, N_3=2, lambda_init=2.0,
                     lambda_growth_factor=2.0, lambda_growth_period=2)
    s = {"lambda_val": cfg.lambda_init, "steps_since_growth": 0,
         "hinge_violated_window": False}
    # Stage 1: λ=0.
    assert _lambda_at_step(0, cfg, s) == (0.0, 1)
    assert _lambda_at_step(2, cfg, s) == (0.0, 1)
    # Stage 2: λ starts at init.
    assert _lambda_at_step(3, cfg, s) == (2.0, 2)
    # Stage 3: pass through whatever stage 2 left.
    s["lambda_val"] = 8.0
    assert _lambda_at_step(7, cfg, s) == (8.0, 3)
    # η: halved in stage 3 only.
    assert _eta_at_step(0, cfg) == cfg.eta
    assert _eta_at_step(6, cfg) == cfg.eta
    assert _eta_at_step(7, cfg) == 0.5 * cfg.eta

    # =========================================================================
    # End-to-end PGD loop with differentiable stub forward / LPIPS / SSIM.
    # =========================================================================
    torch.manual_seed(0)
    T_clean, Hv, Wv = 4, 8, 8
    K = 1
    x_clean = torch.rand(T_clean, Hv, Wv, 3)

    W_att = [2]
    T_proc = T_clean + K
    S_d, nbr_ids = build_support_sets(W_att, T_proc, f0_processed_id=0)
    # Map S_δ from attacked-space to clean-space.
    S_delta_clean = []
    for a in S_d:
        if a in W_att:
            continue
        S_delta_clean.append(attacked_to_clean(a, W_att))
    # f0=0 is never an insert (VADI constraint); it's in S_d and maps to clean 0.
    f0_clean = 0

    # Pseudo-masks: true in lower-left, decoy in upper-right.
    m_hat_true = torch.zeros(Hv, Wv); m_hat_true[4:, :4] = 1.0
    m_hat_decoy = torch.zeros(Hv, Wv); m_hat_decoy[:4, 4:] = 1.0
    m_hat_true_by_t = {t: m_hat_true for t in set(S_d) | set(W_att)}
    m_hat_decoy_by_t = {t: m_hat_decoy for t in set(S_d) | set(W_att)}

    base_inserts = build_base_inserts(x_clean, W_att)
    inputs = VADIInputs(
        x_clean=x_clean,
        base_inserts=base_inserts,
        W=W_att,
        S_delta_clean=sorted(set(S_delta_clean + [f0_clean])),
        insert_ids_processed=W_att,
        neighbor_ids_processed=nbr_ids,
        m_hat_true_by_t=m_hat_true_by_t,
        m_hat_decoy_by_t=m_hat_decoy_by_t,
        f0_clean_id=f0_clean,
    )

    # Stub forward: pred_logits_t = α·(processed[t].mean_over_channels)
    # biased toward the decoy mask region. Differentiable through processed[t].
    def forward_stub(processed: Tensor, return_at: Iterable[int]) -> Dict[int, Tensor]:
        # Scalar-per-pixel activation map per frame; shape [H, W].
        out = {}
        for t in return_at:
            # Mean over channels, then add a small bias tuned so decoy region
            # scores higher when insert/δ brightens decoy pixels.
            gray = processed[t].mean(dim=-1)                           # [H, W]
            out[t] = 3.0 * (gray - 0.5)                                # logits in roughly [-1.5, 1.5]
        return out

    def lpips_stub(x: Tensor, y: Tensor) -> Tensor:
        # Proxy: mean abs diff (in [0, 1], ~0 when identical).
        return (x - y).abs().mean()

    def ssim_stub(x: Tensor, y: Tensor) -> Tensor:
        # Proxy: 1 - MSE. Returns scalar in [0, 1].
        return 1.0 - (x - y).pow(2).mean()

    cfg = VADIConfig(N_1=2, N_2=3, N_3=2,                              # small for test
                     lambda_init=1.0, lambda_growth_factor=2.0,
                     lambda_growth_period=2)
    result = run_vadi_pgd(inputs, forward_stub, lpips_stub, ssim_stub,
                          config=cfg)

    # Sanity: total step count = sum of stage lengths.
    assert len(result.step_logs) == cfg.N_1 + cfg.N_2 + cfg.N_3 == 7
    # Each log populated.
    for log in result.step_logs:
        assert log.loss == log.loss                                    # not NaN
        assert log.stage in (1, 2, 3)
    # Stage assignments: 2,3,4 → stage 2 boundary, last → stage 3.
    stages = [log.stage for log in result.step_logs]
    assert stages[:cfg.N_1] == [1] * cfg.N_1
    assert stages[cfg.N_1:cfg.N_1 + cfg.N_2] == [2] * cfg.N_2
    assert stages[-cfg.N_3:] == [3] * cfg.N_3
    # η halved in stage 3.
    assert result.step_logs[0].eta == cfg.eta
    assert result.step_logs[-1].eta == 0.5 * cfg.eta

    # δ confined to S_δ_clean after all steps.
    for t in range(T_clean):
        if t not in set(inputs.S_delta_clean):
            # Internal state isn't in result directly; re-construct by inspecting
            # the last feasible snapshot OR the running state via `delta_star`.
            if result.delta_star is not None:
                assert result.delta_star[t].abs().max().item() == 0.0, \
                    f"δ leaked to non-S_δ frame {t}"

    # Two-tier ε respected.
    if result.delta_star is not None:
        assert result.delta_star[f0_clean].abs().max().item() \
            <= cfg.eps_delta_f0 + 1e-9
        for c in inputs.S_delta_clean:
            if c == f0_clean:
                continue
            assert result.delta_star[c].abs().max().item() \
                <= cfg.eps_delta_other + 1e-9

    # At least one step should be internally feasible (stub hinges are gentle).
    feasible_count = sum(1 for log in result.step_logs if log.feasible)
    assert feasible_count > 0, "expected at least one feasible step in stub run"
    # best_step corresponds to one of the feasible ones.
    assert result.best_step > 0

    # Snapshot correspondence (Blocker 2 regression):
    # `delta_star` must equal the `delta_snapshot` at best_step, not the
    # post-update tensor.
    best_log = next(log for log in result.step_logs
                    if log.step == result.best_step)
    assert best_log.feasible
    assert torch.equal(result.delta_star, best_log.delta_snapshot), \
        "delta_star must match feasible log's pre-update snapshot"
    assert torch.equal(result.nu_star, best_log.nu_snapshot)
    # Snapshots must live on CPU (memory guarantee).
    assert result.delta_star.device.type == "cpu"
    assert result.nu_star.device.type == "cpu"

    # Feasible StepLogs carry CPU snapshots; infeasible ones are None (saves
    # host RAM — O(|feasible|) rather than O(total_steps)).
    for log in result.step_logs:
        if log.feasible:
            assert log.delta_snapshot is not None
            assert log.nu_snapshot is not None
            assert log.delta_snapshot.device.type == "cpu"
        else:
            assert log.delta_snapshot is None
            assert log.nu_snapshot is None

    # End-to-end margin-gradient sanity: a single `vadi_step` on a fresh
    # state must produce nonzero δ.grad on S_δ frames (Blocker 1 regression).
    fresh_state = _init_state(inputs, cfg, device=torch.device("cpu"))
    _ = vadi_step(
        fresh_state, inputs, forward_stub, lpips_stub, ssim_stub,
        cfg, lambda_val=0.0, eta=cfg.eta, stage=1,
    )
    # After one step δ on S_δ moved off zero (at least one S_δ frame changed).
    moved_counts = sum(1 for c in inputs.S_delta_clean
                       if fresh_state.delta[c].abs().max().item() > 0)
    assert moved_counts > 0, \
        "margin gradient did not propagate to δ (build_processed may have " \
        "broken autograd)"

    print("memshield.vadi_optimize: all self-tests PASSED "
          "(nbr_set, support sets, attacked_to_clean, base inserts, "
          "interleave, clip, grad mask, λ/η schedule, PGD loop)")


if __name__ == "__main__":
    _self_test()
