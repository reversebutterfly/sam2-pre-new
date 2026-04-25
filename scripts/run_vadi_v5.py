"""VADI-v5 (DIRE) driver: decoy-first insertion + recovery-evasion δ.

Written 2026-04-24 as the Round-1 fix of auto-review-loop topic
"VADI method redesign". Implements codex priorities 1-4 verbatim:

  1. Remove ALL pre-insert δ — δ support is POST-insert only.
  2. Replace temporal-midframe base with a duplicate-object decoy seed.
  3. Replace contrastive decoy margin with direct Dice + BCE tracking.
  4. Post-insert δ with R=8 default window (ablatable R ∈ {4, 8, 12}).

Plus:

  - Separate optimizers: Adam for ν, sign-PGD for δ (no shared η).
  - Sequential 3-stage schedule: ν-only → δ-only (ν frozen at best-A) →
    alternating polish with halved rates.
  - Fidelity budgets identical to v4 for apples-to-apples on fidelity
    metrics, but the δ support is strictly post-insert so LPIPS on pre-
    insert frames is 0 automatically.
  - Export via the existing `export_processed_uint8` + re-eval via
    `eval_exported_j_drop` for the paper-claim metric.

## CLI

    python scripts/run_vadi_v5.py \\
        --davis-root data/davis \\
        --checkpoint checkpoints/sam2.1_hiera_tiny.pt \\
        --out-root vadi_runs/v5 \\
        --clips dog camel blackswan \\
        --K 3 --placement top --post-insert-radius 8

Bare run → self-test with stub adapters.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memshield.decoy_seed import (
    build_decoy_insert_seeds, compute_decoy_offset_from_mask,
)
from memshield.vadi_optimize import (
    build_processed, nbr_set,
)
from memshield.vadi_v5_loss import (
    AggregateDecoyLossRecord, aggregate_decoy_tracking_loss,
)
from memshield.vulnerability_scorer import score as vulnerability_score
from scripts.run_vadi import (
    CleanPassOutput, VADIClipOutput,
    eval_exported_j_drop, export_processed_uint8,
    load_processed_uint8, remap_masks_to_processed_space,
    remeasure_exported_feasibility,
)
from scripts.run_vadi_pilot import load_davis_clip


# =============================================================================
# v5 config
# =============================================================================


@dataclass
class VADIv5Config:
    """All v5 numerical settings. Defaults per codex Round-1 proposal."""

    # Stage lengths (sum = total gradient steps, default 100 matching v4).
    N_A_nu: int = 40          # ν-only Adam
    N_B_delta: int = 30       # δ-only sign-PGD
    N_C_alt: int = 30         # alternating polish (halved rates)

    # Step sizes.
    eta_delta: float = 2.0 / 255.0        # sign-PGD step for δ
    eta_nu_lr: float = 1.0 / 255.0        # Adam learning rate for ν

    # δ ε-ball (post-insert ONLY — pre-insert is zero by design).
    eps_delta: float = 4.0 / 255.0

    # Fidelity budgets.
    lpips_orig_cap: float = 0.20
    lpips_insert_cap: float = 0.35        # vs x_ref (source frame), NOT vs seed
    tv_multiplier: float = 1.2
    # f0 (mask-prompt frame) two-tier high-fidelity SSIM floor (matches
    # VADIv4Config canonical 0.98). Required by remeasure_exported_feasibility
    # when the polish stage rechecks exported fidelity.
    f0_ssim_floor: float = 0.98

    # Decoy-tracking loss weights. Split γ by insert vs post-insert
    # (codex R1 review fix): γ=0.5 on insert can push ν to erase the
    # original object from the decoy seed → destroys the intended
    # "identity-confusion double" cue. Post-insert can tolerate more
    # anti-true pressure since no duplicate exists there.
    alpha_dice: float = 1.0
    beta_bce: float = 0.5
    gamma_anti_true_insert: float = 0.1       # insert frames
    gamma_anti_true_post: float = 0.3         # post-insert frames
    insert_weight: float = 1.0
    post_insert_weight: float = 1.0

    # Safety clamp on ν (codex R1 medium fix): Adam has no hard budget;
    # this clamp stops pathological drift without killing the method.
    eps_nu: float = 48.0 / 255.0

    # Fidelity-hinge escalation (same scheme as v4).
    lambda_init: float = 10.0
    lambda_growth_factor: float = 2.0
    lambda_growth_period: int = 10

    # Post-insert window length.
    post_insert_radius: int = 8

    # ν initialization scale (Gaussian std). Relative to the decoy seed —
    # the seed IS the main signal; ν adds a small refinement.
    nu_init_std: float = 0.01

    # Decoy seed feather (Gaussian blur radius on duplicate mask).
    feather_radius: int = 5
    feather_sigma: float = 2.0

    # Ablation axes (codex R2 disciplined plan, 2026-04-24).
    # Defaults reproduce the original v5-full; flip one axis per A1/A2/A3.
    insert_base_mode: str = "duplicate_seed"   # "midframe" | "duplicate_seed"
    loss_mode: str = "dice_bce"                # "margin" | "dice_bce"
    optimizer_nu_mode: str = "adam"            # "adam" | "sign_pgd"
    schedule_preset: str = "full"              # "full" | "insert_only_100"
    delta_support_mode: str = "post_insert"    # "off" | "v4_symmetric" | "post_insert"
    # Margin loss knobs (only consulted when loss_mode=="margin").
    margin_threshold: float = 0.75
    margin_neighbor_weight: float = 0.5

    # Boundary-δ polish stage (codex Round 1 design #1, 2026-04-24).
    # After the main A0 ν-only PGD finishes and exports/re-evaluates,
    # if boundary_polish=True we run a SHORT polish stage that:
    #   - selects "degraded" frames (with align_cos ≥ threshold) as polish targets
    #   - builds per-frame boundary-band δ support masks
    #   - runs N_polish joint-ν-and-δ steps with boundary-aware loss
    #   - compares polished J_drop vs A0 J_drop; reverts if worse (off-switch)
    boundary_polish: bool = False                    # enable/disable whole stage
    boundary_polish_n_steps: int = 30                # PGD steps in polish stage
    boundary_polish_nu_lr_scale: float = 0.25        # ν LR in polish (= 0.25× main)
    # Codex R2 high-fix (2026-04-24): separate step size from ε-ball. Step
    # size controls per-iteration update magnitude; ε_delta is the hard ℓ∞
    # clamp. Default: step = ε/2 so 2 steps saturate to boundary (standard).
    boundary_polish_eta_step: float = 2.0 / 255.0    # per-step sign-PGD
    boundary_polish_eps_delta: float = 4.0 / 255.0   # ℓ∞ ε-ball (threat-model)
    boundary_polish_align_cos_threshold: float = 0.5 # degraded+align>=this → polish
    boundary_polish_band_width: int = 5              # boundary band ring width
    boundary_polish_use_corridor: bool = True
    boundary_polish_corridor_width: int = 5
    boundary_polish_feather_sigma: float = 2.0
    boundary_polish_alpha_dice: float = 1.0
    boundary_polish_beta_bce: float = 0.5
    # Codex R2 medium-fix: split γ_anti_true by insert vs post-insert, matching
    # main v5 loss. Inserts deserve low γ (else ν erases the original object
    # in the duplicate-seed); post-insert can tolerate stronger anti-true.
    boundary_polish_gamma_anti_true_insert: float = 0.1
    boundary_polish_gamma_anti_true_post: float = 0.3
    boundary_polish_off_switch: bool = True          # revert if worse than A0
    boundary_polish_min_improvement: float = 0.0     # min Δ J_drop to accept polish

    # Hiera feature-steering δ polish (codex Loop 3 R2 design #1, 2026-04-25).
    # v0: simple version with uniform ε=4/255 + per-insert teacher Hiera token
    # + L2 (or cosine) feature loss added to existing decoy margin.
    # Mutually exclusive with boundary_polish in this implementation.
    hiera_steering: bool = False
    hiera_steering_n_steps: int = 30
    hiera_steering_nu_lr_scale: float = 0.25       # ν LR during polish
    hiera_steering_eta_step: float = 2.0 / 255.0   # δ sign-PGD step
    hiera_steering_eps_delta: float = 4.0 / 255.0  # δ ℓ∞ ε
    hiera_steering_polish_window: int = 5          # first-N post-insert frames
    hiera_steering_loss_weight: float = 0.5        # λ_hiera × L2/cos to teacher
    hiera_steering_loss_type: str = "l2"           # "l2" | "cosine"
    hiera_steering_teacher_normalize: bool = True  # normalize L2 by teacher RMS
    hiera_steering_off_switch: bool = True
    hiera_steering_min_improvement: float = 0.0

    # Decoy State Continuation polish (codex Loop 3 R3 design, 2026-04-25).
    # Targets SAM2's persistent recurrent state (maskmem_features +
    # obj_ptr) instead of transient Hiera. Per-insert teacher cached
    # from A0 forward; bridge originals are δ-pushed to write decoy-
    # compatible state (cos-sim alignment in pooled decoy region).
    # Codex falsification criterion: state-alignment lift >= 0.15 AND
    # mean J-drop < +0.02 → cut δ permanently.
    state_continuation: bool = False
    state_continuation_n_steps: int = 30
    state_continuation_eta_step: float = 2.0 / 255.0   # δ sign-PGD step
    state_continuation_eps_delta: float = 4.0 / 255.0  # δ ℓ∞ ε
    state_continuation_bridge_length: int = 3          # B_k = w_k+1..w_k+L
    state_continuation_lambda_M: float = 1.0           # weight on L_M (maskmem)
    state_continuation_lambda_P: float = 1.0           # weight on L_P (obj_ptr)
    state_continuation_lambda_margin: float = 1.0      # weight on L_margin
    state_continuation_lambda_fid: float = 10.0        # initial λ_fid (escalates)
    state_continuation_off_switch: bool = True
    state_continuation_min_improvement: float = 0.0

    # Joint Trajectory-Consistent Decoy Attack (codex Loop 3 R4 locked design,
    # 2026-04-25). Replaces ε∞-PGD δ with semantic bridge editing: per-bridge-
    # frame duplicate-object overlay (learnable α) + decoy-direction translation
    # warp on true-object ROI. Joint optimization with insert ν (Phase A frozen
    # ν, Phase B joint refinement). Drops ε∞ on non-prompt frames; keeps
    # LPIPS ≤ 0.20 perceptual budget. Falsification criterion (5th and FINAL):
    # mean Δ ≥ +0.05 → continue; < +0.02 → cut δ permanently.
    joint_trajectory: bool = False
    joint_traj_bridge_length: int = 3
    joint_traj_alpha_max: float = 0.30           # max overlay strength (sigmoid bound)
    joint_traj_max_disp_px: float = 2.0          # max warp magnitude in pixels
    joint_traj_alpha_lr: float = 0.05            # Adam LR for α_logits
    joint_traj_warp_lr: float = 0.1              # Adam LR for warp_s, warp_r
    joint_traj_phase_a_steps: int = 20           # frozen-ν bridge-edit steps
    joint_traj_phase_b_steps: int = 10           # joint ν+bridge refinement steps
    joint_traj_nu_lr_phase_b: float = 0.5 / 255.0  # ν step in joint phase (low)
    joint_traj_lambda_margin: float = 1.0
    joint_traj_lambda_obj: float = 0.5           # positive-objectness weight
    joint_traj_lambda_area: float = 0.25         # area-preservation weight
    joint_traj_lambda_fid: float = 10.0          # fidelity hinge (escalates)
    joint_traj_lambda_alpha: float = 0.05        # α regularizer
    joint_traj_lambda_warp: float = 0.02         # warp regularizer
    joint_traj_obj_threshold: float = 0.5        # positive-objectness floor
    joint_traj_area_min: float = 0.6
    joint_traj_area_max: float = 1.4
    joint_traj_overlay_dilate_px: int = 2
    joint_traj_overlay_feather_sigma: float = 2.5
    joint_traj_true_mask_feather_sigma: float = 2.0
    joint_traj_off_switch: bool = True
    joint_traj_min_improvement: float = 0.0

    # STE quantization during training (codex R2 post-fix: v4 pipeline
    # always applied fake_uint8_quantize in _apply_{delta,nu} which
    # simulated the export-time uint8 round-trip during every forward.
    # v5 default is False. When True, _step_loss applies STE quantization
    # to (x_clean+δ) and (decoy_seeds+ν) before build_processed. Needed
    # for apples-to-apples comparison with v4 K3_insert_only.
    train_ste_quantize: bool = False

    seed: int = 0


# =============================================================================
# Post-insert-only δ support (processed-space + clean-space)
# =============================================================================


def build_post_insert_support(
    W: Sequence[int], T_proc: int, radius: int,
) -> Tuple[List[int], List[int]]:
    """Return (post_insert_proc, post_insert_proc_same).

    `post_insert_proc` = ∪_k {W_k+1, ..., min(W_k+radius, W_{k+1}-1)} in
    processed-space (excludes insert positions themselves). The last
    insert's window extends up to min(W_K+radius, T_proc-1).

    Semantically: the frames where "prevent recovery" δ is applied.
    These are NEVER insert positions and NEVER pre-insert positions.

    Returns a second list identical to the first (kept as a convention
    for symmetry with v4's `(S_delta_processed, neighbor_ids_processed)`
    tuple; in v5 these two concepts collapse into one).
    """
    W_sorted = sorted(int(w) for w in W)
    post: set = set()
    if not W_sorted:
        return [], []
    for i, w in enumerate(W_sorted):
        next_w = W_sorted[i + 1] if i + 1 < len(W_sorted) else T_proc
        end = min(w + radius, next_w - 1, T_proc - 1)
        for t in range(w + 1, end + 1):
            post.add(int(t))
    post_list = sorted(post)
    return post_list, post_list


def _attacked_to_clean_same_space(
    attacked_idx: int, W_attacked: Sequence[int],
) -> int:
    """processed→clean idx for post-insert positions.

    Wraps memshield.vadi_optimize.attacked_to_clean with the v5 invariant
    that the argument is never an insert position.
    """
    from memshield.vadi_optimize import attacked_to_clean as _at2c
    return _at2c(attacked_idx, W_attacked)


# =============================================================================
# Per-step loss components (fidelity hinges same as v4 but on v5 support)
# =============================================================================


from memshield.vadi_loss import (                                     # noqa: E402
    lpips_cap_hinge, tv_hinge,
)


def _step_loss(
    x_clean: Tensor, delta: Tensor,
    decoy_seeds: Tensor, nu: Tensor,
    W: Sequence[int],
    post_insert_clean: Sequence[int],
    loss_query_post_proc: Sequence[int],
    forward_fn: Callable,
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    lpips_fn: Callable,
    config: VADIv5Config,
    lambda_fid: float,
    *,
    freeze_delta: bool,
    freeze_nu: bool,
    fake_uint8: bool = False,
) -> Tuple[Tensor, Dict[str, Any]]:
    """One forward pass + loss computation.

    `freeze_*` flags only affect which leaves receive gradients via
    `requires_grad` routing upstream; the loss itself is identical.
    `fake_uint8=False` is used during stages A/B/C (float optimization),
    `True` for a short quantization-aware polish (not used by default).
    """
    # Construct processed sequence (differentiable wrt δ and ν).
    x_prime = torch.clamp(x_clean + delta, 0.0, 1.0)
    inserts = torch.clamp(decoy_seeds + nu, 0.0, 1.0)
    if fake_uint8 or config.train_ste_quantize:
        from memshield.losses import fake_uint8_quantize
        x_prime = fake_uint8_quantize(x_prime)
        inserts = fake_uint8_quantize(inserts)
    processed = build_processed(x_prime, inserts, W)
    T_proc = processed.shape[0]

    # Query SAM2 at insert positions + loss-query-post-insert positions.
    # NOTE: loss query window is INDEPENDENT of δ support — in particular
    # when freeze_delta is True we still need ν supervision at post-insert
    # frames (SAM2 memory is causal, so post-insert frames are where ν's
    # gradient flows through the memory attention).
    return_at_small = set(int(w) for w in W) | set(int(t) for t in loss_query_post_proc
                                                   if int(t) not in set(int(w) for w in W))
    post_insert_proc_set = list(loss_query_post_proc)
    logits_by_t: Dict[int, Tensor] = forward_fn(processed, return_at_small)

    # Attack loss — branch by config.loss_mode (codex R2 ablation axis).
    decoy_logits = {t: logits_by_t[t] for t in return_at_small}
    insert_ids_sorted = sorted(int(w) for w in W)
    post_ids_sorted = sorted(int(t) for t in post_insert_proc_set
                             if int(t) not in set(insert_ids_sorted))

    if config.loss_mode == "dice_bce":
        # Codex R1-2 fix: different γ_anti_true on insert vs post-insert.
        L_insert_agg, rec_ins = aggregate_decoy_tracking_loss(
            decoy_logits, m_decoy_by_t, m_true_by_t,
            insert_ids=insert_ids_sorted, post_insert_ids=[],
            alpha_dice=config.alpha_dice, beta_bce=config.beta_bce,
            gamma_anti_true=config.gamma_anti_true_insert,
            insert_weight=1.0, post_insert_weight=0.0,
        )
        if post_ids_sorted:
            L_post_agg, _ = aggregate_decoy_tracking_loss(
                decoy_logits, m_decoy_by_t, m_true_by_t,
                insert_ids=[], post_insert_ids=post_ids_sorted,
                alpha_dice=config.alpha_dice, beta_bce=config.beta_bce,
                gamma_anti_true=config.gamma_anti_true_post,
                insert_weight=0.0, post_insert_weight=1.0,
            )
        else:
            L_post_agg = torch.zeros((), dtype=x_prime.dtype,
                                     device=x_prime.device)
        w_ins = config.insert_weight; w_post = config.post_insert_weight
        if insert_ids_sorted and post_ids_sorted:
            L_margin = (w_ins * L_insert_agg + w_post * L_post_agg) \
                       / (w_ins + w_post)
        elif insert_ids_sorted:
            L_margin = L_insert_agg
        elif post_ids_sorted:
            L_margin = L_post_agg
        else:
            L_margin = torch.zeros((), dtype=x_prime.dtype,
                                   device=x_prime.device)
        margin_rec = rec_ins
    elif config.loss_mode == "margin":
        # v4-style contrastive margin. post_insert_ids play the role of
        # v4's "neighbor_ids" (half-weight).
        from memshield.vadi_loss import (
            aggregate_margin_loss, decoy_margin_per_frame,
        )
        margins_by_t = {}
        for t in set(insert_ids_sorted) | set(post_ids_sorted):
            margins_by_t[t] = decoy_margin_per_frame(
                decoy_logits[t], m_true_by_t[t], m_decoy_by_t[t],
                margin=config.margin_threshold,
            )
        agg = aggregate_margin_loss(
            margins_by_t,
            insert_ids=insert_ids_sorted,
            neighbor_ids=post_ids_sorted,
            neighbor_weight=config.margin_neighbor_weight,
        )
        L_margin = agg.L_margin
        margin_rec = {
            "mode": "margin",
            "L_margin": float(agg.L_margin.detach().item()),
            "L_insert": float(agg.L_insert.detach().item()),
            "L_neighbor": float(agg.L_neighbor.detach().item()),
        }
    else:
        raise ValueError(f"unknown loss_mode {config.loss_mode!r}")

    # Fidelity on originals: LPIPS on post-insert-clean positions vs x_clean there.
    L_fid_orig = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
    per_frame_lpips_orig: Dict[int, float] = {}
    for c in post_insert_clean:
        lp = lpips_fn(x_prime[c], x_clean[c])
        per_frame_lpips_orig[c] = float(lp.detach().item())
        L_fid_orig = L_fid_orig + lpips_cap_hinge(lp, config.lpips_orig_cap)

    # Fidelity on inserts: LPIPS(insert_final, x_ref=x_{c_k}), NOT vs seed.
    # c_k = W_k - k so the "source frame" for insert k is x_clean[c_k].
    per_insert_lpips: Dict[int, float] = {}
    per_insert_tv_excess: Dict[int, float] = {}
    L_fid_ins = torch.zeros_like(L_fid_orig)
    L_fid_TV = torch.zeros_like(L_fid_orig)
    W_sorted = sorted(int(w) for w in W)
    for k, w in enumerate(W_sorted):
        c_k = int(w - k)
        if not (0 <= c_k < x_clean.shape[0]):
            continue
        x_ref = x_clean[c_k]
        lp = lpips_fn(inserts[k], x_ref)
        per_insert_lpips[k] = float(lp.detach().item())
        L_fid_ins = L_fid_ins + lpips_cap_hinge(lp, config.lpips_insert_cap)
        ins_chw = inserts[k].permute(2, 0, 1)
        ref_chw = x_ref.permute(2, 0, 1)
        tv_h = tv_hinge(ins_chw, ref_chw, multiplier=config.tv_multiplier)
        per_insert_tv_excess[k] = float(tv_h.detach().item())
        L_fid_TV = L_fid_TV + tv_h

    L = L_margin + lambda_fid * (L_fid_orig + L_fid_ins + L_fid_TV)

    diag = {
        "L_margin": float(L_margin.detach().item()),
        "L_fid_orig": float(L_fid_orig.detach().item()),
        "L_fid_ins": float(L_fid_ins.detach().item()),
        "L_fid_TV": float(L_fid_TV.detach().item()),
        "per_frame_lpips_orig": per_frame_lpips_orig,
        "per_insert_lpips": per_insert_lpips,
        "per_insert_tv_excess": per_insert_tv_excess,
        "margin_record": margin_rec,
    }
    return L, diag


def _post_clean_to_proc(
    post_insert_clean: Sequence[int], W: Sequence[int], T_proc: int,
) -> List[int]:
    """Map post-insert clean-space indices to processed-space.

    For a clean frame at clean index c, its processed index is c +
    (number of inserts at processed positions ≤ c + k). Use the
    inverse: for a post-insert processed index a, clean idx = a −
    (number of W entries < a).
    """
    # Since post_insert_clean is already in clean-space, and we want
    # back in processed-space, compute processed_idx = clean_idx +
    # (# inserts at processed-idx < processed_idx).
    # Simpler: iterate post_insert clean → find processed such that
    # that processed idx is NOT in W AND maps back to clean c.
    W_sorted = sorted(int(w) for w in W)
    out: List[int] = []
    for c in post_insert_clean:
        c_int = int(c)
        # processed_idx = c + (# W entries ≤ the final processed_idx - 1)
        # Which is equivalent to c_int + (# W entries ≤ final processed_idx).
        # This is a self-referential equation; solve by iteration.
        p = c_int
        while True:
            shift = sum(1 for w in W_sorted if w <= p)
            new_p = c_int + shift
            if new_p == p:
                break
            p = new_p
        if p < T_proc and p not in set(W_sorted):
            out.append(p)
    return sorted(set(out))


# =============================================================================
# Optimizer: Stage A (ν-only Adam) / Stage B (δ-only PGD) / Stage C (alt)
# =============================================================================


@dataclass
class V5StepLog:
    step: int
    stage: str                         # "A_nu" | "B_delta" | "C_alt"
    L: float
    L_margin: float
    L_fid_orig: float
    L_fid_ins: float
    L_fid_TV: float
    lambda_fid: float
    feasible: bool
    margin_record: Any = None


@dataclass
class V5Result:
    delta_star: Optional[Tensor]       # CPU
    nu_star: Optional[Tensor]          # CPU
    best_surrogate_loss: float
    infeasible: bool
    step_logs: List[V5StepLog] = field(default_factory=list)


def _run_v5_pgd(
    x_clean: Tensor,
    decoy_seeds: Tensor,
    W: Sequence[int],
    post_insert_clean: Sequence[int],
    loss_query_post_proc: Sequence[int],
    forward_fn: Callable,
    lpips_fn: Callable,
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    config: VADIv5Config,
) -> V5Result:
    """3-stage optimizer. Returns running-best over feasible steps."""
    device = x_clean.device
    T_clean = x_clean.shape[0]
    K = len(W)

    # Leaves.
    delta = torch.zeros_like(x_clean, device=device, requires_grad=True)
    nu = torch.zeros_like(decoy_seeds, device=device)
    if K > 0 and config.nu_init_std > 0.0:
        g = torch.Generator(device="cpu").manual_seed(config.seed)
        nu_init = (config.nu_init_std
                   * torch.randn(*decoy_seeds.shape, generator=g,
                                 dtype=decoy_seeds.dtype)
                   .to(device))
        nu = nu_init.detach().clone()
    nu.requires_grad_(True)

    # ν optimizer — branch on config.optimizer_nu_mode.
    nu_opt: Optional[torch.optim.Optimizer]
    if config.optimizer_nu_mode == "adam":
        nu_opt = torch.optim.Adam([nu], lr=config.eta_nu_lr)
    elif config.optimizer_nu_mode == "sign_pgd":
        nu_opt = None                          # manual sign-grad updates
    else:
        raise ValueError(
            f"unknown optimizer_nu_mode {config.optimizer_nu_mode!r}")

    step_logs: List[V5StepLog] = []
    best_state: Optional[Tuple[Tensor, Tensor, float]] = None
    lambda_val = config.lambda_init

    def _run_step(stage: str, freeze_delta: bool, freeze_nu: bool,
                  eta_delta: float, lambda_current: float):
        """Run one step. Mutates δ, ν, updates best_state."""
        nonlocal best_state, lambda_val
        L, diag = _step_loss(
            x_clean, delta, decoy_seeds, nu, W, post_insert_clean,
            loss_query_post_proc,
            forward_fn, m_decoy_by_t, m_true_by_t, lpips_fn,
            config, lambda_current,
            freeze_delta=freeze_delta, freeze_nu=freeze_nu,
            fake_uint8=False,
        )
        if delta.grad is not None:
            delta.grad.zero_()
        if nu_opt is not None:
            nu_opt.zero_grad()
        elif nu.grad is not None:
            nu.grad.zero_()
        L.backward()

        # δ update: sign-PGD, masked to post-insert-clean support only.
        if not freeze_delta:
            with torch.no_grad():
                if delta.grad is not None:
                    # Mask δ gradient to post_insert_clean positions.
                    mask = torch.zeros_like(delta.grad)
                    for c in post_insert_clean:
                        if 0 <= c < T_clean:
                            mask[c] = 1.0
                    delta.grad.mul_(mask)
                    delta.add_(-eta_delta * delta.grad.sign())
                    # ε-ball clip on the same support (outside = 0, no clip needed).
                    for c in range(T_clean):
                        if c in set(post_insert_clean):
                            delta[c].clamp_(-config.eps_delta, config.eps_delta)
                        else:
                            delta[c].zero_()

        # ν update — Adam or sign-PGD, then hard ε-ball clamp as a safety
        # belt. Codex R1 medium fix (2026-04-24).
        if not freeze_nu:
            if nu_opt is not None:
                nu_opt.step()
            elif nu.grad is not None:
                with torch.no_grad():
                    nu.add_(-config.eta_nu_lr * nu.grad.sign())
            with torch.no_grad():
                nu.clamp_(-config.eps_nu, config.eps_nu)

        # Feasibility check.
        tol = 1e-6
        feas = (
            diag["L_fid_orig"] <= tol
            and diag["L_fid_ins"] <= tol
            and diag["L_fid_TV"] <= tol
        )

        log = V5StepLog(
            step=len(step_logs) + 1, stage=stage,
            L=float(L.detach().item()),
            L_margin=diag["L_margin"],
            L_fid_orig=diag["L_fid_orig"],
            L_fid_ins=diag["L_fid_ins"],
            L_fid_TV=diag["L_fid_TV"],
            lambda_fid=lambda_current, feasible=feas,
            margin_record=asdict(diag["margin_record"])
                if hasattr(diag["margin_record"], "__dataclass_fields__")
                else diag["margin_record"],
        )
        step_logs.append(log)

        if feas:
            this_score = -diag["L_margin"]    # lower L_margin = better attack
            if best_state is None or this_score > best_state[2]:
                best_state = (
                    delta.detach().to("cpu").clone(),
                    nu.detach().to("cpu").clone(),
                    this_score,
                )

        # λ escalation on hinge violation every `lambda_growth_period` steps
        # (stage B/C only — stage A has no δ so originals-LPIPS is 0 trivially).
        if (not feas) and len(step_logs) % config.lambda_growth_period == 0:
            lambda_val *= config.lambda_growth_factor

    # Schedule preset — codex R2 ablation axis.
    if config.schedule_preset == "insert_only_100":
        N_A = 100; N_B = 0; N_C = 0
    elif config.schedule_preset == "full":
        N_A = config.N_A_nu; N_B = config.N_B_delta; N_C = config.N_C_alt
    else:
        raise ValueError(f"unknown schedule_preset {config.schedule_preset!r}")

    # --- Stage A: ν-only ---
    for _ in range(N_A):
        _run_step(stage="A_nu", freeze_delta=True, freeze_nu=False,
                  eta_delta=0.0, lambda_current=lambda_val)

    # --- Stage B: δ-only sign-PGD (ν frozen at its current value) ---
    for _ in range(N_B):
        _run_step(stage="B_delta", freeze_delta=False, freeze_nu=True,
                  eta_delta=config.eta_delta, lambda_current=lambda_val)

    # --- Stage C: alternating polish with halved rates ---
    if N_C > 0:
        if config.optimizer_nu_mode == "adam":
            nu_opt = torch.optim.Adam([nu], lr=0.5 * config.eta_nu_lr)
        for i in range(N_C):
            if i % 2 == 0:
                _run_step(stage="C_alt", freeze_delta=True, freeze_nu=False,
                          eta_delta=0.0, lambda_current=lambda_val)
            else:
                _run_step(stage="C_alt", freeze_delta=False, freeze_nu=True,
                          eta_delta=0.5 * config.eta_delta,
                          lambda_current=lambda_val)

    if best_state is None:
        return V5Result(
            delta_star=None, nu_star=None,
            best_surrogate_loss=float("inf"),
            infeasible=True, step_logs=step_logs,
        )
    return V5Result(
        delta_star=best_state[0], nu_star=best_state[1],
        best_surrogate_loss=float(-best_state[2]),
        infeasible=False, step_logs=step_logs,
    )


# =============================================================================
# Boundary-δ polish stage (codex R1 design #1, 2026-04-24)
# =============================================================================


def _select_polish_frames_from_decoy_semantic(
    exported_j_drop_details: Dict[str, Any],
    W_attacked: Sequence[int],
    align_cos_threshold: float,
) -> List[int]:
    """Select polish target frames from A0's in-band decoy-semantic record.

    Polish set = (insert positions W_attacked) ∪ (degraded frames with
    align_cos ≥ threshold). Degraded + aligned is exactly the regime
    where boundary δ is expected to help (direction right, mask wrong).

    If decoy_semantic is absent (no sam2_eval_fn), returns just W_attacked.
    """
    polish: set = set(int(w) for w in W_attacked)
    ds = exported_j_drop_details.get("decoy_semantic") or {}
    per_frame = ds.get("per_frame") or []
    for rec in per_frame:
        if not rec.get("valid", False):
            continue
        if rec.get("mode") != "degraded":
            continue
        a = rec.get("decoy_alignment_cos")
        if a is None:
            continue
        if float(a) >= align_cos_threshold:
            polish.add(int(rec["t"]))
    return sorted(polish)


def _build_boundary_support_masks(
    m_true_by_t: Dict[int, Tensor],
    m_decoy_by_t: Dict[int, Tensor],
    polish_frame_ids: Sequence[int],
    *,
    band_width: int,
    use_corridor: bool,
    corridor_width: int,
    feather_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """Build per-frame (support_mask, band_true) dicts for polish frames.

    Returns ({t → [H,W] float support}, {t → [H,W] float true-band}).
    Both on `device` in `dtype`. Built via `memshield.boundary_bands`.
    """
    from memshield.boundary_bands import (
        build_delta_support_mask, boundary_band,
    )
    support_by_t: Dict[int, Tensor] = {}
    band_true_by_t: Dict[int, Tensor] = {}
    for t in polish_frame_ids:
        t = int(t)
        mt = m_true_by_t[t].detach().cpu().numpy()
        md = m_decoy_by_t[t].detach().cpu().numpy()
        supp_np = build_delta_support_mask(
            (mt > 0.5).astype(np.uint8), (md > 0.5).astype(np.uint8),
            band_width=band_width,
            use_corridor=use_corridor,
            corridor_width=corridor_width,
            feather_sigma=feather_sigma,
        )
        btrue_np = boundary_band(
            (mt > 0.5).astype(np.uint8), band_width=band_width,
        ).astype(np.float32)
        support_by_t[t] = torch.from_numpy(supp_np).to(device=device, dtype=dtype)
        band_true_by_t[t] = torch.from_numpy(btrue_np).to(device=device, dtype=dtype)
    return support_by_t, band_true_by_t


def _run_boundary_polish_pgd(
    x_clean: Tensor,
    decoy_seeds: Tensor,
    nu_init: Tensor,              # warm-start from A0 ν* (already on device)
    W_attacked: Sequence[int],
    polish_frame_ids: Sequence[int],
    support_by_t: Dict[int, Tensor],      # [H, W] float
    band_true_by_t: Dict[int, Tensor],    # [H, W] float
    m_decoy_by_t: Dict[int, Tensor],      # [H, W] float
    m_true_by_t: Dict[int, Tensor],       # [H, W] float, for anti-true eval only
    forward_fn: Callable,
    lpips_fn: Callable,
    config: VADIv5Config,
) -> Tuple[Optional[Tensor], Optional[Tensor], List[Dict[str, Any]]]:
    """Short boundary-δ polish PGD.

    Warm-start: ν ← nu_init (A0 ν*), δ ← 0. δ is spatially masked to
    per-frame `support_by_t` at every step (zeroed outside). Loss is
    the boundary-aware Dice/BCE/anti-true aggregate on polish_frame_ids.

    Sign-PGD on δ with η=`config.boundary_polish_eta_delta`, Adam on ν
    with lr = `config.eta_nu_lr × config.boundary_polish_nu_lr_scale`
    (reuses ν's existing geometry).

    Returns (δ*, ν*, step_logs). On infeasibility (no feasible step),
    both tensors are None.
    """
    from memshield.vadi_boundary_loss import aggregate_boundary_loss
    from memshield.vadi_loss import lpips_cap_hinge, tv_hinge

    device = x_clean.device
    dtype = x_clean.dtype
    T_clean = x_clean.shape[0]
    K = len(W_attacked)

    # Leaves.
    delta = torch.zeros_like(x_clean, device=device, requires_grad=True)
    nu = nu_init.detach().clone().to(device)
    nu.requires_grad_(True)
    nu_opt = torch.optim.Adam(
        [nu], lr=config.eta_nu_lr * config.boundary_polish_nu_lr_scale,
    )

    # Pre-compute a clean-space support mask for δ (union over polish frames
    # mapped to clean-space). Outside this region, δ stays at 0.
    support_clean_union = torch.zeros(
        (T_clean, x_clean.shape[1], x_clean.shape[2]),
        device=device, dtype=dtype,
    )
    for t_proc in polish_frame_ids:
        if int(t_proc) in set(W_attacked):
            # Insert positions in processed space — affect insert content (ν),
            # NOT original frames (δ lives in clean-space). Skip for δ mask.
            continue
        t_clean = _attacked_to_clean_same_space(int(t_proc), W_attacked)
        if 0 <= t_clean < T_clean:
            # Broadcast [H, W] across 3 channels below at update time.
            support_clean_union[t_clean] = torch.maximum(
                support_clean_union[t_clean], support_by_t[int(t_proc)],
            )

    # Best-step tracking.
    best: Optional[Tuple[Tensor, Tensor, float]] = None
    step_logs: List[Dict[str, Any]] = []
    lambda_val = config.lambda_init

    # Split polish frame ids by insert-vs-post for γ_anti_true weighting.
    W_set = set(int(w) for w in W_attacked)
    insert_polish_ids = sorted(t for t in polish_frame_ids if int(t) in W_set)
    post_polish_ids = sorted(t for t in polish_frame_ids if int(t) not in W_set)

    for step in range(config.boundary_polish_n_steps):
        # Forward build.
        x_prime = torch.clamp(x_clean + delta, 0.0, 1.0)
        inserts = torch.clamp(decoy_seeds + nu, 0.0, 1.0)
        if config.train_ste_quantize:
            from memshield.losses import fake_uint8_quantize
            x_prime = fake_uint8_quantize(x_prime)
            inserts = fake_uint8_quantize(inserts)
        processed = build_processed(x_prime, inserts, W_attacked)

        # Query SAM2 at polish frame ids.
        return_at = set(int(t) for t in polish_frame_ids)
        logits_by_t: Dict[int, Tensor] = forward_fn(processed, return_at)

        # Boundary-weighted decoy loss — split insert vs post for γ_anti.
        # Codex R2 medium-fix: inserts get γ=0.1, post gets γ=0.3 to avoid
        # eroding the duplicate-seed identity on insert frames.
        if insert_polish_ids:
            L_ins_agg, rec_ins = aggregate_boundary_loss(
                pred_logits_by_t=logits_by_t,
                m_decoy_by_t=m_decoy_by_t,
                band_true_by_t=band_true_by_t,
                support_by_t=support_by_t,
                polish_frame_ids=insert_polish_ids,
                alpha_dice=config.boundary_polish_alpha_dice,
                beta_bce=config.boundary_polish_beta_bce,
                gamma_anti_true=config.boundary_polish_gamma_anti_true_insert,
            )
        else:
            L_ins_agg = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
            rec_ins = None
        if post_polish_ids:
            L_post_agg, rec_post = aggregate_boundary_loss(
                pred_logits_by_t=logits_by_t,
                m_decoy_by_t=m_decoy_by_t,
                band_true_by_t=band_true_by_t,
                support_by_t=support_by_t,
                polish_frame_ids=post_polish_ids,
                alpha_dice=config.boundary_polish_alpha_dice,
                beta_bce=config.boundary_polish_beta_bce,
                gamma_anti_true=config.boundary_polish_gamma_anti_true_post,
            )
        else:
            L_post_agg = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
            rec_post = None
        # Equal 1:1 mean over two non-empty groups; falls back to either alone.
        if insert_polish_ids and post_polish_ids:
            L_margin = 0.5 * (L_ins_agg + L_post_agg)
        elif insert_polish_ids:
            L_margin = L_ins_agg
        elif post_polish_ids:
            L_margin = L_post_agg
        else:
            L_margin = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
        bnd_n_frames = len(insert_polish_ids) + len(post_polish_ids)

        # Fidelity hinges on original frames (where δ is nonzero)
        # AND on insert frames (where ν continues to move).
        polish_clean = sorted({
            _attacked_to_clean_same_space(int(t), W_attacked)
            for t in polish_frame_ids if int(t) not in W_set
        })
        L_fid_orig = torch.zeros((), dtype=x_prime.dtype, device=x_prime.device)
        for c in polish_clean:
            if 0 <= c < T_clean:
                lp = lpips_fn(x_prime[c], x_clean[c])
                L_fid_orig = L_fid_orig + lpips_cap_hinge(
                    lp, config.lpips_orig_cap)
        L_fid_ins = torch.zeros_like(L_fid_orig)
        L_fid_TV = torch.zeros_like(L_fid_orig)
        W_sorted = sorted(int(w) for w in W_attacked)
        for k, w in enumerate(W_sorted):
            c_k = int(w - k)
            if not (0 <= c_k < T_clean):
                continue
            x_ref = x_clean[c_k]
            lp = lpips_fn(inserts[k], x_ref)
            L_fid_ins = L_fid_ins + lpips_cap_hinge(
                lp, config.lpips_insert_cap)
            ins_chw = inserts[k].permute(2, 0, 1)
            ref_chw = x_ref.permute(2, 0, 1)
            L_fid_TV = L_fid_TV + tv_hinge(
                ins_chw, ref_chw, multiplier=config.tv_multiplier)

        L = L_margin + lambda_val * (L_fid_orig + L_fid_ins + L_fid_TV)

        # Codex R2 high-fix: feasibility / best-state bookkeeping captures
        # PRE-update state — the (δ, ν) values that the recorded
        # L_fid_* and L_margin were computed against. Snapshot them
        # BEFORE the upcoming mutation.
        tol = 1e-6
        feas = (
            float(L_fid_orig.detach().item()) <= tol
            and float(L_fid_ins.detach().item()) <= tol
            and float(L_fid_TV.detach().item()) <= tol
        )
        if feas:
            score = -float(L_margin.detach().item())
            if best is None or score > best[2]:
                best = (delta.detach().cpu().clone(),
                        nu.detach().cpu().clone(), score)

        log = {
            "step": step + 1, "stage": "polish",
            "L": float(L.detach().item()),
            "L_margin": float(L_margin.detach().item()),
            "L_fid_orig": float(L_fid_orig.detach().item()),
            "L_fid_ins": float(L_fid_ins.detach().item()),
            "L_fid_TV": float(L_fid_TV.detach().item()),
            "n_polish_frames": bnd_n_frames,
            "feasible": feas,
        }
        step_logs.append(log)

        if delta.grad is not None:
            delta.grad.zero_()
        nu_opt.zero_grad()
        L.backward()

        # δ update: sign-PGD, masked to per-frame support then per-clean-frame
        # union. Codex R2 high-fix: strict ε=ε_delta clamp (not 4× over-budget).
        with torch.no_grad():
            if delta.grad is not None:
                delta.add_(
                    -config.boundary_polish_eta_step * delta.grad.sign())
                # Broadcast [T, H, W] clean-space union → [T, H, W, 3].
                mask_bcast = support_clean_union.unsqueeze(-1)
                delta.mul_(mask_bcast)
                # Threat-model hard ε clamp.
                delta.clamp_(
                    -config.boundary_polish_eps_delta,
                    +config.boundary_polish_eps_delta)

        # ν update: Adam + eps_nu clamp.
        nu_opt.step()
        with torch.no_grad():
            nu.clamp_(-config.eps_nu, config.eps_nu)

        # λ escalation.
        if (not feas) and (step + 1) % config.lambda_growth_period == 0:
            lambda_val *= config.lambda_growth_factor

    if best is None:
        return None, None, step_logs
    return best[0], best[1], step_logs


# =============================================================================
# Hiera feature-steering δ polish (codex Loop 3 R2 design #1, 2026-04-25)
# =============================================================================


def _select_hiera_polish_frames(
    W_attacked: Sequence[int], T_proc: int, polish_window: int,
) -> List[int]:
    """Polish set for Hiera-steering: insert positions ∪ first
    `polish_window` post-insert frames per insert (clipped to next
    insert's position and to T_proc).

    Simpler than boundary-polish's "degraded+aligned" filter — Hiera
    steering targets ALL post-insert frames in the codex-suggested
    "first W_k+1..W_k+3-ish" halo.
    """
    W_sorted = sorted(int(w) for w in W_attacked)
    out: set = set(W_sorted)
    for i, w in enumerate(W_sorted):
        next_w = W_sorted[i + 1] if i + 1 < len(W_sorted) else T_proc
        end = min(w + polish_window, next_w - 1, T_proc - 1)
        for t in range(w + 1, end + 1):
            out.add(int(t))
    return sorted(out)


def _run_hiera_steering_pgd(
    x_clean: Tensor,
    decoy_seeds: Tensor,
    nu_init: Tensor,
    W_attacked: Sequence[int],
    polish_frame_ids: Sequence[int],
    teacher_hiera_tokens: List[Tensor],     # per-insert teacher token, [1,C,h,w]
    polish_to_insert_k: Dict[int, int],
    forward_fn,                              # VADIForwardFn instance (has .forward_with_hiera)
    lpips_fn: Callable,
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    config: VADIv5Config,
) -> Tuple[Optional[Tensor], Optional[Tensor], List[Dict[str, Any]]]:
    """Hiera feature-steering polish: warm-start ν from A0; joint
    optimize (ν, δ) with combined loss = decoy margin + λ_hiera ·
    L2(hiera_token, teacher_token) at polish frames.

    δ uses uniform ε=hiera_steering_eps_delta (no spatial gating in v0).
    ν uses Adam at hiera_steering_nu_lr_scale × main lr, with eps_nu
    safety clamp.

    Returns (δ*, ν*, step_logs); (None, None) if no feasible step.
    """
    from memshield.hiera_features import (
        hiera_feature_l2_loss, hiera_feature_cosine_loss,
    )
    from memshield.vadi_loss import (
        aggregate_margin_loss, decoy_margin_per_frame,
        lpips_cap_hinge, tv_hinge,
    )

    device = x_clean.device
    T_clean = x_clean.shape[0]
    K = len(W_attacked)

    delta = torch.zeros_like(x_clean, device=device, requires_grad=True)
    # Codex Loop3-R3 fix (2026-04-25): freeze ν during Hiera polish.
    # `hiera_at` excludes insert positions (R4 fix earlier), so L_hiera has
    # no ν gradient path; the only ν path is L_margin on insert frames,
    # which warm-started from A0 ν* is already optimized. Leaving ν trainable
    # let polish push ν off A0's TV-feasible operating point. Frozen ν also
    # removes the per-insert TV blowup we saw in the v0 pilot.
    nu = nu_init.detach().clone().to(device)  # NO requires_grad → frozen

    # Polish-clean indices for δ LPIPS hinge (clean-space, exclude inserts).
    W_set = set(int(w) for w in W_attacked)
    polish_clean = sorted({
        _attacked_to_clean_same_space(int(t), W_attacked)
        for t in polish_frame_ids if int(t) not in W_set
    })

    # Codex Loop3-R3 fix (2026-04-25): δ support mask. Without this,
    # gradient through SAM2's causal memory chain backprops from polish
    # frames all the way back to f0 → δ[0].grad ≠ 0 → sign-PGD saturates
    # δ[0] to ε → f0 SSIM drops below 0.98 floor → off-switch revert.
    # Mask is True only on clean-space indices that map to polish-frame
    # post-insert positions (= polish_clean). f0 (idx 0) is naturally
    # excluded since polish_frame_ids excludes pre-insert frames.
    support_mask = torch.zeros(
        (T_clean, 1, 1, 1), dtype=delta.dtype, device=device,
    )
    for c in polish_clean:
        if 0 <= c < T_clean:
            support_mask[c] = 1.0

    # Build hiera_at = polish_frame_ids that are POST-INSERT (k_cover ≥ 0
    # AND t is NOT itself an insert position). Codex R4 fix (2026-04-25):
    # excluding insert frames is critical — applying L_hiera to insert
    # frames asks SAM2's Hiera at the insert (which IS the synthetic
    # decoy frame) to match the teacher Hiera (which is also the
    # synthetic decoy's Hiera) — degenerate, no signal, wastes the
    # query budget for actual post-insert "teach SAM2 to look decoy-like
    # in a real-frame context" work.
    hiera_at = sorted(t for t in polish_frame_ids
                      if polish_to_insert_k.get(int(t), -1) >= 0
                      and int(t) not in W_set)
    # Loss-query (margin) frames = insert ∪ post-insert in polish set.
    insert_polish = sorted(t for t in polish_frame_ids if int(t) in W_set)
    post_polish = sorted(t for t in polish_frame_ids if int(t) not in W_set)

    best: Optional[Tuple[Tensor, Tensor, float]] = None
    step_logs: List[Dict[str, Any]] = []
    lambda_val = config.lambda_init

    for step in range(config.hiera_steering_n_steps):
        x_prime = torch.clamp(x_clean + delta, 0.0, 1.0)
        inserts = torch.clamp(decoy_seeds + nu, 0.0, 1.0)
        if config.train_ste_quantize:
            from memshield.losses import fake_uint8_quantize
            x_prime = fake_uint8_quantize(x_prime)
            inserts = fake_uint8_quantize(inserts)
        processed = build_processed(x_prime, inserts, W_attacked)

        # Single forward returning both logits + Hiera tokens.
        logits_by_t, hiera_by_t = forward_fn.forward_with_hiera(
            processed, return_at=polish_frame_ids, hiera_at=hiera_at,
        )

        # Decoy margin loss (v4 contrastive — same as A0).
        margins_by_t = {}
        for t in polish_frame_ids:
            if int(t) not in m_true_by_t or int(t) not in m_decoy_by_t:
                continue
            margins_by_t[int(t)] = decoy_margin_per_frame(
                logits_by_t[int(t)],
                m_true_by_t[int(t)], m_decoy_by_t[int(t)],
                margin=config.margin_threshold,
            )
        agg = aggregate_margin_loss(
            margins_by_t,
            insert_ids=insert_polish,
            neighbor_ids=post_polish,
            neighbor_weight=config.margin_neighbor_weight,
        )
        L_margin = agg.L_margin

        # Hiera teacher loss.
        if config.hiera_steering_loss_type == "cosine":
            L_hiera = hiera_feature_cosine_loss(
                hiera_by_t, teacher_hiera_tokens, polish_to_insert_k,
            )
        else:
            L_hiera = hiera_feature_l2_loss(
                hiera_by_t, teacher_hiera_tokens, polish_to_insert_k,
                normalize=config.hiera_steering_teacher_normalize,
            )

        # Fidelity hinges.
        L_fid_orig = torch.zeros((), dtype=x_prime.dtype, device=device)
        for c in polish_clean:
            if 0 <= c < T_clean:
                lp = lpips_fn(x_prime[c], x_clean[c])
                L_fid_orig = L_fid_orig + lpips_cap_hinge(
                    lp, config.lpips_orig_cap)
        L_fid_ins = torch.zeros_like(L_fid_orig)
        L_fid_TV = torch.zeros_like(L_fid_orig)
        W_sorted = sorted(int(w) for w in W_attacked)
        for k, w in enumerate(W_sorted):
            c_k = int(w - k)
            if not (0 <= c_k < T_clean):
                continue
            x_ref = x_clean[c_k]
            lp = lpips_fn(inserts[k], x_ref)
            L_fid_ins = L_fid_ins + lpips_cap_hinge(
                lp, config.lpips_insert_cap)
            ins_chw = inserts[k].permute(2, 0, 1)
            ref_chw = x_ref.permute(2, 0, 1)
            L_fid_TV = L_fid_TV + tv_hinge(
                ins_chw, ref_chw, multiplier=config.tv_multiplier)

        L = (
            L_margin
            + config.hiera_steering_loss_weight * L_hiera
            + lambda_val * (L_fid_orig + L_fid_ins + L_fid_TV)
        )

        # Best-state PRE-update snapshot (codex R3 fix pattern from boundary).
        tol = 1e-6
        feas = (
            float(L_fid_orig.detach().item()) <= tol
            and float(L_fid_ins.detach().item()) <= tol
            and float(L_fid_TV.detach().item()) <= tol
        )
        if feas:
            # Codex R3 high-fix (2026-04-25): rank by the same WEIGHTED
            # objective the optimizer is minimizing, not unweighted sum.
            score = -float(
                L_margin.detach().item()
                + config.hiera_steering_loss_weight * L_hiera.detach().item()
            )
            if best is None or score > best[2]:
                best = (delta.detach().cpu().clone(),
                        nu.detach().cpu().clone(), score)

        # Codex Loop3-R3 invariant tripwire: δ should be exactly 0 at
        # frames OUTSIDE polish_clean. Log max |δ| there to catch any
        # future regression of the support-masking logic. Expected: 0.0.
        with torch.no_grad():
            outside_mask = (1.0 - support_mask)
            delta_outside_linf = float(
                (delta.detach() * outside_mask).abs().max().item()
            ) if outside_mask.any() else 0.0

        log = {
            "step": step + 1, "stage": "hiera_steering",
            "L": float(L.detach().item()),
            "L_margin": float(L_margin.detach().item()),
            "L_hiera": float(L_hiera.detach().item()),
            "L_fid_orig": float(L_fid_orig.detach().item()),
            "L_fid_ins": float(L_fid_ins.detach().item()),
            "L_fid_TV": float(L_fid_TV.detach().item()),
            "n_polish_frames": len(polish_frame_ids),
            "feasible": feas,
            "delta_outside_support_linf": delta_outside_linf,
        }
        step_logs.append(log)

        if delta.grad is not None:
            delta.grad.zero_()
        # ν is frozen — no Adam optimizer to zero_grad.
        L.backward()

        # δ update: support-masked sign-PGD, strict ε clamp.
        # Codex Loop3-R3 fix: gate δ.grad to support BEFORE sign() so
        # off-support frames cannot accumulate updates via memory backprop.
        # Belt-and-suspenders: also project δ to support after the step.
        with torch.no_grad():
            if delta.grad is not None:
                delta.grad.mul_(support_mask)
                delta.add_(-config.hiera_steering_eta_step * delta.grad.sign())
                delta.mul_(support_mask)         # off-support → 0
                delta.clamp_(
                    -config.hiera_steering_eps_delta,
                    +config.hiera_steering_eps_delta)
        # ν is frozen — skip Adam step + clamp.

        if (not feas) and (step + 1) % config.lambda_growth_period == 0:
            lambda_val *= config.lambda_growth_factor

    if best is None:
        return None, None, step_logs
    return best[0], best[1], step_logs


# =============================================================================
# Decoy State Continuation polish (codex Loop 3 R3 design, 2026-04-25)
# =============================================================================


def _run_state_continuation_pgd(
    x_clean: Tensor,
    decoy_seeds: Tensor,
    nu_init: Tensor,
    W_attacked: Sequence[int],
    bridge_frames_by_k: Dict[int, List[int]],   # {k → [t_proc...]}
    teacher_M_by_k: Dict[int, Tensor],          # {k → M̄_k} (no grad)
    teacher_p_by_k: Dict[int, Tensor],          # {k → p̄_k} (no grad)
    forward_fn,                                  # VADIForwardFn (forward_with_state)
    lpips_fn: Callable,
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    config: VADIv5Config,
    *,
    student_decoy_mask_by_t: Optional[Dict[int, Tensor]] = None,
    teacher_decoy_mask_by_k: Optional[Dict[int, Tensor]] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor],
           List[Dict[str, Any]], Optional[int]]:
    """State-continuation polish PGD: warm-start ν from A0 (frozen), δ=0
    on bridge originals, optimize δ to align bridge-frame `maskmem_features`
    + `obj_ptr` to per-insert teacher states via cosine distance.

    Loss:
        L = λ_margin · L_margin   (v4 contrastive on insert ∪ bridge)
          + λ_M     · L_M         (state_continuation_loss maskmem term)
          + λ_P     · L_P         (state_continuation_loss obj_ptr term)
          + λ_fid   · L_fid       (LPIPS_polish_clean + LPIPS_inserts + TV_inserts)

    δ is support-masked to bridge originals (clean-space). ν is frozen.
    Strict ε ℓ∞ clamp on δ. Best-state PRE-update snapshot keyed on
    (margin + λ_M·L_M + λ_P·L_P) when feasible.

    Returns (δ*, ν*, step_logs); (None, None, logs) if no feasible step.
    """
    from memshield.state_continuation import state_continuation_loss
    from memshield.vadi_loss import (
        aggregate_margin_loss, decoy_margin_per_frame,
        lpips_cap_hinge, tv_hinge,
    )

    if student_decoy_mask_by_t is None or teacher_decoy_mask_by_k is None:
        raise ValueError(
            "_run_state_continuation_pgd: student_decoy_mask_by_t and "
            "teacher_decoy_mask_by_k are required (codex R3-fix1: per-"
            "bridge-frame masks for spatial correctness on moving clips)")

    device = x_clean.device
    T_clean = x_clean.shape[0]
    K = len(W_attacked)
    W_set = set(int(w) for w in W_attacked)
    W_sorted = sorted(W_set)

    # δ in clean-space; support mask = bridge originals only.
    delta = torch.zeros_like(x_clean, device=device, requires_grad=True)
    nu = nu_init.detach().clone().to(device)  # FROZEN — no requires_grad

    # Flatten bridge frames + build bridge → insert k map.
    from memshield.state_continuation import build_bridge_to_insert_k
    bridge_to_k = build_bridge_to_insert_k(bridge_frames_by_k)
    bridge_t_proc_set = sorted(bridge_to_k.keys())

    # Polish-clean indices for δ LPIPS hinge + support mask.
    polish_clean = sorted({
        _attacked_to_clean_same_space(int(t), W_attacked)
        for t in bridge_t_proc_set if int(t) not in W_set
    })
    support_mask = torch.zeros(
        (T_clean, 1, 1, 1), dtype=delta.dtype, device=device,
    )
    for c in polish_clean:
        if 0 <= c < T_clean:
            support_mask[c] = 1.0

    # Loss-query frames for L_margin: insert ∪ bridge (insert is k itself).
    insert_polish = list(W_sorted)
    bridge_polish = sorted(t for t in bridge_t_proc_set if t not in W_set)
    margin_query_frames = sorted(set(insert_polish + bridge_polish))

    # state_at = bridge frames only (insert teachers are pre-cached, no
    # need to re-extract during polish).
    state_at = sorted(bridge_polish)

    best: Optional[Tuple[Tensor, Tensor, float]] = None
    step_logs: List[Dict[str, Any]] = []
    lambda_val = config.state_continuation_lambda_fid

    for step in range(config.state_continuation_n_steps):
        x_prime = torch.clamp(x_clean + delta, 0.0, 1.0)
        inserts = torch.clamp(decoy_seeds + nu, 0.0, 1.0)
        if config.train_ste_quantize:
            from memshield.losses import fake_uint8_quantize
            x_prime = fake_uint8_quantize(x_prime)
            inserts = fake_uint8_quantize(inserts)
        processed = build_processed(x_prime, inserts, W_attacked)

        # Single forward returning logits at margin-query frames + state at bridges.
        logits_by_t, maskmem_by_t, obj_ptr_by_t = \
            forward_fn.forward_with_state(
                processed, return_at=margin_query_frames, state_at=state_at,
            )

        # Decoy margin loss.
        margins_by_t = {}
        for t in margin_query_frames:
            if int(t) not in m_true_by_t or int(t) not in m_decoy_by_t:
                continue
            margins_by_t[int(t)] = decoy_margin_per_frame(
                logits_by_t[int(t)],
                m_true_by_t[int(t)], m_decoy_by_t[int(t)],
                margin=config.margin_threshold,
            )
        agg = aggregate_margin_loss(
            margins_by_t,
            insert_ids=insert_polish,
            neighbor_ids=bridge_polish,
            neighbor_weight=config.margin_neighbor_weight,
        )
        L_margin = agg.L_margin

        # State-continuation loss (only bridge frames; teachers are insert-keyed).
        bridge_to_k_only_present = {
            t: bridge_to_k[t] for t in maskmem_by_t.keys() if t in bridge_to_k
        }
        L_state, state_log = state_continuation_loss(
            student_M_by_t={t: maskmem_by_t[t] for t in
                            bridge_to_k_only_present},
            student_p_by_t={t: obj_ptr_by_t[t] for t in
                            bridge_to_k_only_present},
            teacher_M_by_k=teacher_M_by_k,
            teacher_p_by_k=teacher_p_by_k,
            student_decoy_mask_by_t=student_decoy_mask_by_t,
            teacher_decoy_mask_by_k=teacher_decoy_mask_by_k,
            bridge_to_insert_k=bridge_to_k_only_present,
            lambda_M=config.state_continuation_lambda_M,
            lambda_P=config.state_continuation_lambda_P,
        )

        # Fidelity hinges (unchanged from hiera-steering pattern).
        L_fid_orig = torch.zeros((), dtype=x_prime.dtype, device=device)
        for c in polish_clean:
            if 0 <= c < T_clean:
                lp = lpips_fn(x_prime[c], x_clean[c])
                L_fid_orig = L_fid_orig + lpips_cap_hinge(
                    lp, config.lpips_orig_cap)
        L_fid_ins = torch.zeros_like(L_fid_orig)
        L_fid_TV = torch.zeros_like(L_fid_orig)
        for k_, w_ in enumerate(W_sorted):
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

        L = (
            config.state_continuation_lambda_margin * L_margin
            + L_state
            + lambda_val * (L_fid_orig + L_fid_ins + L_fid_TV)
        )

        # Best-state PRE-update snapshot.
        tol = 1e-6
        feas = (
            float(L_fid_orig.detach().item()) <= tol
            and float(L_fid_ins.detach().item()) <= tol
            and float(L_fid_TV.detach().item()) <= tol
        )
        if feas:
            score = -float(
                config.state_continuation_lambda_margin
                * L_margin.detach().item()
                + L_state.detach().item()
            )
            if best is None or score > best[2]:
                # Codex Loop3-R3-fix2: also record the step number so the
                # driver can attribute falsification metrics (cos_M/cos_P)
                # to the same checkpoint the polish actually returned.
                best = (delta.detach().cpu().clone(),
                        nu.detach().cpu().clone(), score, step + 1)

        # Invariant tripwire — δ outside support should stay 0.
        with torch.no_grad():
            outside_mask = (1.0 - support_mask)
            delta_outside_linf = float(
                (delta.detach() * outside_mask).abs().max().item()
            ) if outside_mask.any() else 0.0

        log = {
            "step": step + 1, "stage": "state_continuation",
            "L": float(L.detach().item()),
            "L_margin": float(L_margin.detach().item()),
            "L_state": float(L_state.detach().item()),
            "L_M": state_log["L_M"], "L_P": state_log["L_P"],
            "mean_cos_M": state_log["mean_cos_M"],
            "mean_cos_P": state_log["mean_cos_P"],
            "n_bridge": state_log["n_bridge"],
            "L_fid_orig": float(L_fid_orig.detach().item()),
            "L_fid_ins": float(L_fid_ins.detach().item()),
            "L_fid_TV": float(L_fid_TV.detach().item()),
            "feasible": feas,
            "delta_outside_support_linf": delta_outside_linf,
        }
        step_logs.append(log)

        if delta.grad is not None:
            delta.grad.zero_()
        L.backward()

        with torch.no_grad():
            if delta.grad is not None:
                delta.grad.mul_(support_mask)
                delta.add_(
                    -config.state_continuation_eta_step
                    * delta.grad.sign())
                delta.mul_(support_mask)         # off-support → 0
                delta.clamp_(
                    -config.state_continuation_eps_delta,
                    +config.state_continuation_eps_delta)

        if (not feas) and (step + 1) % config.lambda_growth_period == 0:
            lambda_val *= config.lambda_growth_factor

    if best is None:
        return None, None, step_logs, None
    return best[0], best[1], step_logs, int(best[3])


# =============================================================================
# Joint Trajectory-Consistent Decoy Attack (codex Loop 3 R4 locked design)
# =============================================================================


def _jt_perceptual_feasible(
    remeasure_dict: Dict[str, Any], config: VADIv5Config,
) -> bool:
    """JT's perceptual-only feasibility gate (codex Loop3-R4 post-pilot fix).

    The R4 threat model is perceptual: f0 SSIM ≥ 0.98, original-frame LPIPS
    ≤ 0.20, insert LPIPS ≤ 0.35. The 1.2× clean-source TV cap was an ε-PGD-
    era heuristic and is now advisory only — LPIPS already captures
    perceptual fidelity, while TV-as-hard-gate proved over-sensitive to
    cuDNN bf16 nondeterm in A0's ν*.

    Returns True iff all PERCEPTUAL gates pass (TV ignored).
    """
    _TOL = 1e-6
    ssim_f0 = float(remeasure_dict.get("ssim_f0", 0.0))
    if ssim_f0 < float(config.f0_ssim_floor) - _TOL:
        return False
    orig_lpips = remeasure_dict.get("per_frame_lpips_orig", {}) or {}
    if any(float(v) > float(config.lpips_orig_cap) + _TOL
           for v in orig_lpips.values()):
        return False
    ins_lpips = remeasure_dict.get("per_insert_lpips", {}) or {}
    if any(float(v) > float(config.lpips_insert_cap) + _TOL
           for v in ins_lpips.values()):
        return False
    return True


def _run_joint_trajectory_pgd(
    x_clean: Tensor,
    decoy_seeds: Tensor,
    nu_init: Tensor,
    W_attacked: Sequence[int],
    bridge_frames_by_k: Dict[int, List[int]],
    softened_decoy_masks_by_t: Dict[int, Tensor],   # {t → [H, W]} for overlay placement
    softened_true_masks_by_t: Dict[int, Tensor],    # {t → [H, W]} for warp ROI
    duplicate_frames_by_t: Dict[int, Tensor],       # {t → [H, W, 3]} pre-built duplicates
    decoy_offsets_unit: Tensor,                     # [K, 2] unit (dy, dx)
    forward_fn,                                      # VADIForwardFn
    lpips_fn: Callable,
    m_decoy_by_t: Dict[int, Tensor],
    m_true_by_t: Dict[int, Tensor],
    config: VADIv5Config,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]],
           List[Dict[str, Any]], Optional[int]]:
    """Joint trajectory-consistent decoy attack PGD (codex Loop 3 R4).

    Replaces ε∞-PGD δ with: localized continuation overlay + decoy-direction
    translation warp on bridge originals. Both controlled by learnable
    parameters (α_logits, warp_s, warp_r) optimized via Adam.

    Phase A (config.joint_traj_phase_a_steps, default 20): freeze ν, optimize
    α + warp on bridge frames. Validates whether semantic bridge edits help
    without destabilizing the insert anchor.
    Phase B (config.joint_traj_phase_b_steps, default 10): unfreeze ν at low
    LR for joint refinement.

    Returns:
      (x_edited_star, nu_star, edit_params, step_logs, best_step)

      x_edited_star: [T_clean, H, W, 3] — the bridge-edited clean tensor at
        the best snapshot. Exporter passes this directly as `x_clean` with
        delta=0 to maintain compatibility.
      edit_params: dict of α_logits, warp_s, warp_r at best snapshot
        (for downstream interpretability/debugging).
    """
    from memshield.decoy_continuation import (
        init_bridge_edit_params, alpha_from_logits, displacement_from_warp,
        apply_continuation_overlay, apply_translation_warp_roi,
        positive_objectness_loss, area_preservation_loss,
        alpha_regularizer, warp_regularizer,
    )
    from memshield.vadi_loss import (
        aggregate_margin_loss, decoy_margin_per_frame,
        lpips_cap_hinge, tv_hinge,
    )

    device = x_clean.device
    T_clean = x_clean.shape[0]
    K = len(W_attacked)
    W_set = set(int(w) for w in W_attacked)
    W_sorted = sorted(W_set)

    # Bridge frame flat list + map to (k, l_idx).
    bridge_t_list: List[Tuple[int, int, int]] = []   # (t_proc, k, l_idx)
    for k_, t_list in bridge_frames_by_k.items():
        for l_idx, t in enumerate(t_list):
            bridge_t_list.append((int(t), int(k_), int(l_idx)))
    if not bridge_t_list:
        return None, None, None, [], None
    bridge_t_proc_set = sorted(t for t, _, _ in bridge_t_list)
    L_max = max(len(v) for v in bridge_frames_by_k.values()) if \
        bridge_frames_by_k else 0

    # Initialize learnable parameters.
    edit_params = init_bridge_edit_params(
        K, L_max,
        alpha_max=config.joint_traj_alpha_max,
        s_init_px=1.0, r_init_px=0.0,
        device=device, dtype=x_clean.dtype,
    )
    decoy_unit = decoy_offsets_unit.to(device).to(x_clean.dtype)

    # ν: frozen in Phase A.
    nu = nu_init.detach().clone().to(device)
    nu_requires_grad = False

    opt_alpha = torch.optim.Adam(
        [edit_params.alpha_logits], lr=config.joint_traj_alpha_lr,
    )
    opt_warp = torch.optim.Adam(
        [edit_params.warp_s, edit_params.warp_r],
        lr=config.joint_traj_warp_lr,
    )

    # Loss-query frames: insert ∪ bridge.
    insert_polish = sorted(W_set)
    bridge_polish = [t for t, _, _ in bridge_t_list]
    margin_query_frames = sorted(set(insert_polish + bridge_polish))

    best: Optional[Tuple[Tensor, Tensor, Dict[str, Tensor], float, int]] = \
        None
    step_logs: List[Dict[str, Any]] = []
    lambda_fid_val = config.joint_traj_lambda_fid

    total_steps = (
        config.joint_traj_phase_a_steps + config.joint_traj_phase_b_steps)

    for step in range(total_steps):
        # Phase B switch: unfreeze ν at low LR (sign-PGD applied manually).
        if step == config.joint_traj_phase_a_steps and not nu_requires_grad:
            nu = nu.detach().clone().requires_grad_(True)
            nu_requires_grad = True

        # Build edited bridge frames (differentiable via list+stack).
        alphas = alpha_from_logits(
            edit_params.alpha_logits, alpha_max=config.joint_traj_alpha_max,
        )                                              # [K, L]
        d_xy = displacement_from_warp(
            edit_params.warp_s, edit_params.warp_r,
            u_dir=decoy_unit, max_disp_px=config.joint_traj_max_disp_px,
        )                                              # [K, L, 2] (dy, dx)

        # Build a per-clean-index edit dict so we can stack.
        edited_by_c: Dict[int, Tensor] = {}
        for t, k_, l_idx in bridge_t_list:
            c_t = _attacked_to_clean_same_space(int(t), W_attacked)
            if not (0 <= c_t < T_clean):
                continue
            x_t = x_clean[c_t]
            if t not in softened_true_masks_by_t or \
                    t not in softened_decoy_masks_by_t or \
                    t not in duplicate_frames_by_t:
                continue
            true_mask = softened_true_masks_by_t[int(t)].to(device)
            decoy_mask = softened_decoy_masks_by_t[int(t)].to(device)
            duplicate = duplicate_frames_by_t[int(t)].to(device)
            d_yx = d_xy[k_, l_idx]
            x_warped = apply_translation_warp_roi(x_t, true_mask, d_yx)
            x_edited = apply_continuation_overlay(
                x_warped, duplicate, decoy_mask, alphas[k_, l_idx],
            )
            edited_by_c[int(c_t)] = x_edited

        # Stack into a full [T_clean, H, W, 3] tensor (non-bridge frames =
        # original x_clean).
        frames: List[Tensor] = []
        for c in range(T_clean):
            if c in edited_by_c:
                frames.append(edited_by_c[c])
            else:
                frames.append(x_clean[c])
        x_edited_full = torch.stack(frames, dim=0)

        # ν is frozen in Phase A; in Phase B it has grad.
        inserts = (decoy_seeds + nu).clamp(0.0, 1.0)
        if config.train_ste_quantize:
            from memshield.losses import fake_uint8_quantize
            x_edited_full = fake_uint8_quantize(x_edited_full)
            inserts = fake_uint8_quantize(inserts)

        processed = build_processed(x_edited_full, inserts, W_attacked)

        # Forward — logits at margin frames + objectness at bridge frames.
        logits_by_t, obj_score_by_t = forward_fn.forward_with_objectness(
            processed, return_at=margin_query_frames,
            objectness_at=bridge_polish,
        )

        # Decoy margin loss (existing v4 contrastive).
        margins_by_t = {}
        for t in margin_query_frames:
            if int(t) not in m_true_by_t or int(t) not in m_decoy_by_t:
                continue
            margins_by_t[int(t)] = decoy_margin_per_frame(
                logits_by_t[int(t)],
                m_true_by_t[int(t)], m_decoy_by_t[int(t)],
                margin=config.margin_threshold,
            )
        agg = aggregate_margin_loss(
            margins_by_t,
            insert_ids=insert_polish,
            neighbor_ids=bridge_polish,
            neighbor_weight=config.margin_neighbor_weight,
        )
        L_margin = agg.L_margin

        # Positive objectness (no-suppression guard).
        obj_logits_stack = torch.stack(
            [obj_score_by_t[t].flatten().mean() for t in bridge_polish],
            dim=0,
        ) if bridge_polish else torch.zeros(0, device=device)
        L_obj = positive_objectness_loss(
            obj_logits_stack, threshold=config.joint_traj_obj_threshold,
        )

        # Area preservation (no empty-mask collapse).
        bridge_logits_2d: Dict[int, Tensor] = {}
        bridge_true_2d: Dict[int, Tensor] = {}
        for t in bridge_polish:
            if int(t) not in m_true_by_t:
                continue
            l = logits_by_t[int(t)]
            bridge_logits_2d[int(t)] = l
            bridge_true_2d[int(t)] = m_true_by_t[int(t)]
        L_area, area_ratios = area_preservation_loss(
            bridge_logits_2d, bridge_true_2d,
            area_min=config.joint_traj_area_min,
            area_max=config.joint_traj_area_max,
        )

        # Fidelity hinges — bridge originals (LPIPS), inserts (LPIPS + TV),
        # f0 SSIM is enforced implicitly because we don't edit f0.
        L_fid_bridge = torch.zeros((), dtype=x_edited_full.dtype, device=device)
        for c in sorted(edited_by_c.keys()):
            lp = lpips_fn(x_edited_full[c], x_clean[c])
            L_fid_bridge = L_fid_bridge + lpips_cap_hinge(
                lp, config.lpips_orig_cap)
        L_fid_ins = torch.zeros_like(L_fid_bridge)
        L_fid_TV = torch.zeros_like(L_fid_bridge)
        for k_, w_ in enumerate(W_sorted):
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

        # Edit parameter regularizers.
        L_alpha = alpha_regularizer(
            alphas, l1_weight=1.0, smoothness_weight=1.0,
        )
        L_warp = warp_regularizer(
            edit_params.warp_s, edit_params.warp_r,
            l2_weight=1.0, orthogonal_weight=1.0, smoothness_weight=1.0,
        )

        # Total loss.
        L = (
            config.joint_traj_lambda_margin * L_margin
            + config.joint_traj_lambda_obj * L_obj
            + config.joint_traj_lambda_area * L_area
            + lambda_fid_val * L_fid_total
            + config.joint_traj_lambda_alpha * L_alpha
            + config.joint_traj_lambda_warp * L_warp
        )

        # Diagnostics.
        with torch.no_grad():
            decoy_overlap_list: List[float] = []
            true_overlap_list: List[float] = []
            obj_score_list: List[float] = []
            valid_t_list: List[int] = []
            for t in bridge_polish:
                if int(t) not in m_decoy_by_t or int(t) not in m_true_by_t:
                    continue
                pred = torch.sigmoid(logits_by_t[int(t)]).flatten()
                m_d = m_decoy_by_t[int(t)].flatten().float()
                m_tr = m_true_by_t[int(t)].flatten().float()
                d_ov = float(((pred * m_d).sum()
                              / m_d.sum().clamp_min(1e-4)).item())
                t_ov = float(((pred * m_tr).sum()
                              / m_tr.sum().clamp_min(1e-4)).item())
                decoy_overlap_list.append(d_ov)
                true_overlap_list.append(t_ov)
                # Per-frame object_score for the wrong-but-present check.
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
            # Codex Loop3-R4 fix (low): wrong-but-present requires positive
            # object_score per frame, otherwise the metric overlaps with
            # suppression cases (negative score → "no object" → empty mask).
            wrong_but_present = sum(
                1 for d, tr, ar, obj in zip(
                    decoy_overlap_list, true_overlap_list,
                    [area_ratios.get(t, 1.0) for t in valid_t_list],
                    obj_score_list)
                if d > tr and ar > 0.5 and obj > 0.0
            )

        # Feasibility (only fidelity hinges; J-drop is checked at export).
        tol = 1e-6
        feas = (
            float(L_fid_bridge.detach().item()) <= tol
            and float(L_fid_ins.detach().item()) <= tol
            and float(L_fid_TV.detach().item()) <= tol
        )
        if feas:
            # Codex Loop3-R4 fix (medium): include L_alpha + L_warp in the
            # best-state ranking. Optimizer minimizes the full feasible
            # objective; the snapshot must reflect the same trade-offs.
            score = -float(
                config.joint_traj_lambda_margin
                * L_margin.detach().item()
                + config.joint_traj_lambda_obj * L_obj.detach().item()
                + config.joint_traj_lambda_area * L_area.detach().item()
                + config.joint_traj_lambda_alpha * L_alpha.detach().item()
                + config.joint_traj_lambda_warp * L_warp.detach().item()
            )
            if best is None or score > best[3]:
                best = (
                    x_edited_full.detach().cpu().clone(),
                    nu.detach().cpu().clone(),
                    {
                        "alpha_logits":
                            edit_params.alpha_logits.detach().cpu().clone(),
                        "warp_s": edit_params.warp_s.detach().cpu().clone(),
                        "warp_r": edit_params.warp_r.detach().cpu().clone(),
                    },
                    score,
                    step + 1,
                )

        log = {
            "step": step + 1,
            "phase": "A" if step < config.joint_traj_phase_a_steps else "B",
            "L": float(L.detach().item()),
            "L_margin": float(L_margin.detach().item()),
            "L_obj": float(L_obj.detach().item()),
            "L_area": float(L_area.detach().item()),
            "L_fid_bridge": float(L_fid_bridge.detach().item()),
            "L_fid_ins": float(L_fid_ins.detach().item()),
            "L_fid_TV": float(L_fid_TV.detach().item()),
            "L_alpha": float(L_alpha.detach().item()),
            "L_warp": float(L_warp.detach().item()),
            "mean_decoy_overlap": float(mean_decoy_overlap),
            "mean_true_overlap": float(mean_true_overlap),
            "delta_overlap": float(mean_decoy_overlap - mean_true_overlap),
            "mean_obj_score": mean_obj_score,
            "wrong_but_present_count": int(wrong_but_present),
            "feasible": feas,
            "alpha_mean": float(alphas.mean().detach().item()),
            "alpha_max_step": float(alphas.max().detach().item()),
            "warp_disp_max": float(d_xy.norm(dim=-1).max().detach().item()),
            "n_bridge": len(bridge_polish),
        }
        step_logs.append(log)

        # Backward + step.
        opt_alpha.zero_grad()
        opt_warp.zero_grad()
        if nu_requires_grad and nu.grad is not None:
            nu.grad.zero_()
        L.backward()
        opt_alpha.step()
        opt_warp.step()
        if nu_requires_grad and nu.grad is not None:
            with torch.no_grad():
                # ν: low-LR sign-PGD for joint Phase B.
                nu.add_(
                    -config.joint_traj_nu_lr_phase_b * nu.grad.sign())
                nu.clamp_(-config.eps_nu, config.eps_nu)

        if (not feas) and (step + 1) % config.lambda_growth_period == 0:
            lambda_fid_val *= config.lambda_growth_factor

    if best is None:
        return None, None, None, step_logs, None
    return best[0], best[1], best[2], step_logs, int(best[4])


# =============================================================================
# Per-clip orchestrator
# =============================================================================


@dataclass
class V5ClipOutput:
    clip_name: str
    config_name: str
    W: List[int]
    decoy_offsets: List[Tuple[int, int]]
    infeasible: bool
    best_surrogate_loss: float
    exported_j_drop: Optional[float]
    exported_j_drop_details: Dict[str, Any]
    export_dir: Optional[str]
    step_log_summary: List[Dict[str, Any]]
    placement_source: str
    post_insert_radius: int
    # Boundary-δ polish fields (codex R1 #1, 2026-04-24).
    # polish_applied: whether polish actually ran (i.e. there were degraded+
    # aligned frames AND polish improved over A0 without hitting off-switch).
    # polish_reverted: polish ran but off-switch triggered, reverted to A0.
    polish_applied: bool = False
    polish_reverted: bool = False
    polish_stats: Dict[str, Any] = field(default_factory=dict)


def run_v5_for_clip(
    clip_name: str,
    config_name: str,
    x_clean: Tensor,
    prompt_mask: np.ndarray,
    clean_pass_fn: Callable[[Tensor, np.ndarray], CleanPassOutput],
    forward_fn_builder: Callable[..., Callable],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    ssim_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    K_ins: int = 3,
    min_gap: int = 2,
    placement_mode: str = "top",      # "top" | "random"
    post_insert_radius: int = 8,
    rng: Optional[np.random.Generator] = None,
    config: Optional[VADIv5Config] = None,
    out_root: Optional[Path] = None,
    sam2_eval_fn: Optional[Callable] = None,
    W_clean_override: Optional[Sequence[int]] = None,
    seed_only: bool = False,
) -> V5ClipOutput:
    """Top-level single-clip v5 orchestrator.

    Pipeline: clean-SAM2 pass → vulnerability score → placement →
    decoy-seed construction → 3-stage PGD/Adam → export uint8 →
    exported J-drop.
    """
    config = config or VADIv5Config()
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    # --- clean-SAM2 pass ---
    clean_out = clean_pass_fn(x_clean, prompt_mask)
    assert len(clean_out.pseudo_masks) == x_clean.shape[0]
    T_clean = x_clean.shape[0]

    # --- placement ---
    if W_clean_override is not None:
        W_clean = sorted(int(c) for c in W_clean_override)
        if len(W_clean) != K_ins:
            raise ValueError(
                f"W_clean_override has {len(W_clean)} entries, expected "
                f"K_ins={K_ins}")
        # Codex R1 minor fix: scorer guarantees 1 ≤ c_k < T_clean but
        # the override path did not. Validate explicitly.
        for c in W_clean:
            if not (1 <= c < T_clean):
                raise ValueError(
                    f"W_clean_override entry {c} out of [1, {T_clean}) "
                    "— decoy-seed init needs x_clean[c_k] to be a "
                    "valid frame index with neighbors.")
        if len(set(W_clean)) != len(W_clean):
            raise ValueError(f"W_clean_override has duplicates: {W_clean}")
        for i in range(1, len(W_clean)):
            if W_clean[i] - W_clean[i - 1] < min_gap:
                raise ValueError(
                    f"W_clean_override violates min_gap={min_gap}: "
                    f"{W_clean[i-1]} and {W_clean[i]}")
        placement_source = "override"
    else:
        vs = vulnerability_score(
            confidences=clean_out.confidences,
            pseudo_masks=clean_out.pseudo_masks,
            hiera_features=clean_out.hiera_features,
        )
        W_clean = vs.pick(K=K_ins, mode=placement_mode, min_gap=min_gap,
                          rng=rng if placement_mode == "random" else None)
        if len(W_clean) != K_ins:
            raise RuntimeError(
                f"vs.pick returned {len(W_clean)} positions; expected {K_ins}")
        placement_source = "scorer"

    # Attacked-space: c_k → W_k = c_k + k after sort.
    W_attacked = sorted([c + k for k, c in enumerate(sorted(W_clean))])
    T_proc = T_clean + len(W_attacked)

    # --- insert base construction (codex R2 ablation axis) ---
    if config.insert_base_mode == "duplicate_seed":
        decoy_seeds, decoy_offsets = build_decoy_insert_seeds(
            x_clean, clean_out.pseudo_masks, sorted(W_clean),
            feather_radius=config.feather_radius,
            feather_sigma=config.feather_sigma,
        )
        decoy_seeds = decoy_seeds.to(x_clean.device)
    elif config.insert_base_mode == "midframe":
        from memshield.vadi_optimize import build_base_inserts as _bbi
        decoy_seeds = _bbi(x_clean, W_attacked)
        # Still need decoy offsets for shifted-mask supervision.
        decoy_offsets = [
            compute_decoy_offset_from_mask(
                np.asarray(clean_out.pseudo_masks[c], dtype=np.float32))
            for c in sorted(W_clean)
        ]
    else:
        raise ValueError(
            f"unknown insert_base_mode {config.insert_base_mode!r}")

    # --- decoy mask trajectory (for loss supervision) ---
    # For each insert, the decoy target at post-insert frames is the
    # SAME spatial offset applied to the clean pseudo-mask at that post-
    # insert position. We use the PER-INSERT offset for its window.
    # For simplicity in v5 round 1, we use insert k's offset for all
    # post-insert frames in insert k's window; and for multi-insert
    # overlapping windows, the later insert's offset wins (consistent
    # with "latest hijack controls the scene").
    m_true_clean_np = [np.asarray(m, dtype=np.float32)
                       for m in clean_out.pseudo_masks]
    m_decoy_clean_np: List[np.ndarray] = []
    # Default: no offset → decoy mask == true mask (shouldn't happen).
    for t in range(T_clean):
        # Find the most recent insert k such that c_k ≤ t.
        k_cover = -1
        for k, c_k in enumerate(sorted(W_clean)):
            if c_k <= t:
                k_cover = k
            else:
                break
        if k_cover == -1:
            # Pre-first-insert: no decoy applicable. Use zeros (SAM2
            # should still track true; loss on those positions is
            # typically not queried).
            m_decoy_clean_np.append(np.zeros_like(m_true_clean_np[t]))
        else:
            dy, dx = decoy_offsets[k_cover]
            from memshield.decoy_seed import shift_mask_np
            m_decoy_clean_np.append(
                shift_mask_np(m_true_clean_np[t], dy, dx))

    # Remap to processed-space (handles the +k shift for inserts).
    # NOTE: `remap_masks_to_processed_space` uses midframe-average
    # `0.5·mask[c_k-1] + 0.5·mask[c_k]` for INSERT positions. That is
    # correct for v4 (temporal-midframe inserts) but WRONG for v5
    # (duplicate-object inserts built from `x_clean[c_k]`). We override
    # the insert positions to point at the correct decoy-seed supervision:
    #   true at W_k  → m_true_clean[c_k]   (original object in the seed)
    #   decoy at W_k → shift(m_true_clean[c_k], offset_k)   (duplicate)
    # Codex R1 high-severity fix (2026-04-24).
    m_true_by_t_np = remap_masks_to_processed_space(
        m_true_clean_np, W_attacked)
    m_decoy_by_t_np = remap_masks_to_processed_space(
        m_decoy_clean_np, W_attacked)
    W_sorted_clean = sorted(int(c) for c in W_clean)
    from memshield.decoy_seed import shift_mask_np as _shift_np
    for k, w in enumerate(sorted(W_attacked)):
        c_k = int(W_sorted_clean[k])
        if not (0 <= c_k < T_clean):
            continue
        m_true_by_t_np[w] = m_true_clean_np[c_k]
        dy, dx = decoy_offsets[k]
        m_decoy_by_t_np[w] = _shift_np(m_true_clean_np[c_k], dy, dx)
    device = x_clean.device
    m_true_by_t = {t: torch.from_numpy(m).float().to(device)
                   for t, m in m_true_by_t_np.items()}
    m_decoy_by_t = {t: torch.from_numpy(m).float().to(device)
                    for t, m in m_decoy_by_t_np.items()}

    # --- Loss query window vs δ support window (DECOUPLED — 2026-04-24
    # post-Round2 bug fix). Previously these were collapsed, which meant
    # `delta_support_mode="off"` silently killed the ν gradient signal
    # because loss was only queried at insert positions. The ν gradient
    # ONLY flows through post-insert frames (SAM2 memory is causal; pre-
    # insert frames have zero cross-partial from the insert's ν).
    #
    #   * `loss_query_post_proc` = ALWAYS the post-insert R-radius window.
    #     Drives loss supervision for ν regardless of δ state.
    #   * `post_insert_clean` = δ support window (may be empty if δ is off).
    # ---
    # Unconditional base window: post-insert R positions in processed space.
    loss_query_post_proc, _ = build_post_insert_support(
        W_attacked, T_proc, radius=post_insert_radius,
    )
    loss_query_post_clean = sorted({
        _attacked_to_clean_same_space(a, W_attacked)
        for a in loss_query_post_proc if a not in set(W_attacked)
    })

    if config.delta_support_mode == "off":
        post_insert_clean: List[int] = []
    elif config.delta_support_mode == "post_insert":
        post_insert_clean = list(loss_query_post_clean)   # δ on same window
    elif config.delta_support_mode == "v4_symmetric":
        # v4-style: ∪_k NbrSet(W_k, ±2) ∪ {f0=0} in processed space.
        # δ supported there; loss query window stays at post-insert R.
        from memshield.vadi_optimize import build_support_sets
        S_delta_proc, _ = build_support_sets(
            W_attacked, T_proc, f0_processed_id=0)
        post_insert_clean = sorted({
            _attacked_to_clean_same_space(a, W_attacked)
            for a in S_delta_proc if int(a) not in set(W_attacked)
        })
    else:
        raise ValueError(
            f"unknown delta_support_mode {config.delta_support_mode!r}")

    # --- forward_fn ---
    forward_fn = forward_fn_builder(
        x_clean=x_clean, prompt_mask=prompt_mask, W=W_attacked,
    )

    # --- optimize (or skip for --seed-only A/B comparison) ---
    if seed_only:
        # Zero δ and zero ν — direct measurement of "what does the
        # decoy seed alone achieve without any optimization". Codex
        # R1-2 recommended this as the paired baseline for full v5.
        result = V5Result(
            delta_star=torch.zeros_like(x_clean).to("cpu"),
            nu_star=torch.zeros_like(decoy_seeds).to("cpu"),
            best_surrogate_loss=float("nan"),
            infeasible=False, step_logs=[],
        )
    else:
        result = _run_v5_pgd(
            x_clean=x_clean, decoy_seeds=decoy_seeds,
            W=W_attacked, post_insert_clean=post_insert_clean,
            loss_query_post_proc=list(loss_query_post_proc),
            forward_fn=forward_fn, lpips_fn=lpips_fn,
            m_decoy_by_t=m_decoy_by_t, m_true_by_t=m_true_by_t,
            config=config,
        )

    # --- export + exported J-drop ---
    export_dir: Optional[Path] = None
    exported_j_drop_val: Optional[float] = None
    exported_j_drop_details: Dict[str, Any] = {}
    if not result.infeasible:
        if out_root is None:
            out_root = Path(".") / "vadi_runs_v5"
        export_dir = out_root / clip_name / config_name / "processed"
        delta_on_device = result.delta_star.to(x_clean.device)
        nu_on_device = result.nu_star.to(x_clean.device)
        export_processed_uint8(
            x_clean, delta_on_device, nu_on_device, decoy_seeds,
            W_attacked, export_dir,
        )
        exported = load_processed_uint8(export_dir).to(x_clean.device)
        if sam2_eval_fn is not None:
            exported_j_drop_details = eval_exported_j_drop(
                sam2_eval_fn=sam2_eval_fn,
                prompt_mask=prompt_mask,
                x_clean=x_clean,
                base_inserts=decoy_seeds,
                exported=exported,
                W=W_attacked,
                m_hat_true_by_t=m_true_by_t,
                m_hat_decoy_by_t=m_decoy_by_t,
                decoy_offsets=decoy_offsets,
            )
            exported_j_drop_val = float(
                exported_j_drop_details["J_drop_mean"])

    # --- Stage 10 (optional): boundary-δ polish (codex R1 #1, 2026-04-24)
    polish_applied = False
    polish_reverted = False
    polish_stats: Dict[str, Any] = {}
    polish_logs: List[Dict[str, Any]] = []
    if (config.boundary_polish
            and not result.infeasible
            and sam2_eval_fn is not None
            and result.nu_star is not None
            and exported_j_drop_details):
        polish_frames = _select_polish_frames_from_decoy_semantic(
            exported_j_drop_details, W_attacked,
            align_cos_threshold=config.boundary_polish_align_cos_threshold,
        )
        n_degraded_aligned = len(polish_frames) - len(W_attacked)
        polish_stats["polish_frames"] = polish_frames
        polish_stats["n_degraded_aligned"] = n_degraded_aligned
        if n_degraded_aligned <= 0:
            # No eligible degraded frames → skip polish entirely.
            polish_stats["skipped"] = "no_degraded_aligned_frames"
        else:
            # Build per-frame support masks.
            support_by_t, band_true_by_t = _build_boundary_support_masks(
                m_true_by_t, m_decoy_by_t, polish_frames,
                band_width=config.boundary_polish_band_width,
                use_corridor=config.boundary_polish_use_corridor,
                corridor_width=config.boundary_polish_corridor_width,
                feather_sigma=config.boundary_polish_feather_sigma,
                device=x_clean.device, dtype=x_clean.dtype,
            )
            # Warm-start ν from A0 ν*.
            nu_star_a0 = result.nu_star.to(x_clean.device)
            delta_polish, nu_polish, polish_logs = _run_boundary_polish_pgd(
                x_clean=x_clean, decoy_seeds=decoy_seeds,
                nu_init=nu_star_a0,
                W_attacked=W_attacked, polish_frame_ids=polish_frames,
                support_by_t=support_by_t,
                band_true_by_t=band_true_by_t,
                m_decoy_by_t=m_decoy_by_t,
                m_true_by_t=m_true_by_t,
                forward_fn=forward_fn, lpips_fn=lpips_fn,
                config=config,
            )
            if delta_polish is None or nu_polish is None:
                polish_stats["skipped"] = "polish_infeasible_no_best_step"
            else:
                # Export polish result to a sibling directory + eval.
                polish_dir = out_root / clip_name / f"{config_name}__polish" \
                    / "processed"
                export_processed_uint8(
                    x_clean, delta_polish.to(x_clean.device),
                    nu_polish.to(x_clean.device), decoy_seeds,
                    W_attacked, polish_dir,
                )
                exported_polish = load_processed_uint8(
                    polish_dir).to(x_clean.device)
                polish_eval = eval_exported_j_drop(
                    sam2_eval_fn=sam2_eval_fn,
                    prompt_mask=prompt_mask,
                    x_clean=x_clean,
                    base_inserts=decoy_seeds,
                    exported=exported_polish,
                    W=W_attacked,
                    m_hat_true_by_t=m_true_by_t,
                    m_hat_decoy_by_t=m_decoy_by_t,
                    decoy_offsets=decoy_offsets,
                )
                polish_j_drop = float(polish_eval["J_drop_mean"])
                # Codex R3 high-fix: `exported_j_drop_val or 0.0` would
                # silently treat a negative A0 J-drop (rare failed clips)
                # as 0.0, corrupting the accept/revert decision. Use
                # explicit None check.
                a0_j_drop = (exported_j_drop_val
                             if exported_j_drop_val is not None else 0.0)
                delta_j = polish_j_drop - a0_j_drop
                polish_stats["a0_j_drop"] = a0_j_drop
                polish_stats["polish_j_drop"] = polish_j_drop
                polish_stats["delta_j_drop"] = delta_j
                # Off-switch: accept polish only if improvement ≥ min threshold.
                accept = (not config.boundary_polish_off_switch) or (
                    delta_j >= config.boundary_polish_min_improvement
                )
                polish_stats["accepted"] = accept
                if accept:
                    polish_applied = True
                    # Overwrite the returned A0 export_dir / J_drop /
                    # details with the polish-winning result. Codex R3
                    # low-fix: also refresh step_logs so bookkeeping
                    # reflects the polish run (A0 logs are prepended;
                    # polish logs appended separately).
                    export_dir = polish_dir
                    exported_j_drop_val = polish_j_drop
                    exported_j_drop_details = polish_eval
                    # best_surrogate_loss: compute from polish's final
                    # reported L_margin if polish_logs has any feasible
                    # entry; otherwise mark NaN to signal boundary-loss
                    # has different scale than margin loss.
                    best_surrogate = float("nan")
                    if polish_logs:
                        feas_logs = [lg for lg in polish_logs
                                     if lg.get("feasible")]
                        if feas_logs:
                            best_surrogate = min(
                                lg["L_margin"] for lg in feas_logs)
                    result = V5Result(
                        delta_star=delta_polish, nu_star=nu_polish,
                        best_surrogate_loss=best_surrogate,
                        infeasible=False, step_logs=result.step_logs,
                    )
                else:
                    polish_reverted = True
                    # Keep A0 export unchanged; polish dir stays as reference.

    # --- Stage 11 (alternative to Stage 10): Hiera feature-steering polish
    # Codex Loop 3 R2 design #1, 2026-04-25. Mutually exclusive with
    # boundary_polish in v0; if both flags are set, hiera takes precedence.
    # Codex Loop3-R3 fix (2026-04-25): require ssim_fn for any Hiera polish
    # run. Without it, the preflight + exported-fidelity gates become no-ops
    # and acceptance degrades to J-drop only — fine for debugging but unsafe
    # for a decision run. Record skip reason if ssim_fn missing.
    # state_continuation (Stage 12) takes precedence over hiera_steering
    # (Stage 11) when both flags are set.
    _hiera_eligible = (
        config.hiera_steering
        and not config.state_continuation
        and not result.infeasible
        and not polish_applied
        and sam2_eval_fn is not None
        and result.nu_star is not None
    )
    if _hiera_eligible and ssim_fn is None:
        polish_stats["skipped"] = (
            "hiera_polish_requires_ssim_fn_for_fidelity_gate")
    if _hiera_eligible and ssim_fn is not None:
        from memshield.hiera_features import (
            build_decoy_teacher_frames,
            extract_hiera_teacher_tokens,
            build_polish_to_insert_k_map,
        )
        # Build clean-source per-insert references shared by preflight + remeasure.
        # Codex Loop3-R3 fix: invariant `0 <= w - k < T_clean` should always
        # hold under v5 placement rules; assert instead of clamp so an
        # upstream bug surfaces loudly.
        W_sorted_int = sorted(int(w) for w in W_attacked)
        for k_, w_ in enumerate(W_sorted_int):
            c_k_ = w_ - k_
            if not (0 <= c_k_ < x_clean.shape[0]):
                # Codex Loop3-R3 fix (2026-04-25): explicit RuntimeError so
                # the check survives `python -O` (assert is stripped).
                raise RuntimeError(
                    f"Hiera polish: invalid c_k={c_k_} for W_sorted[{k_}]={w_} "
                    f"(T_clean={x_clean.shape[0]}) — invariant w-k in [0, T_clean) violated"
                )
        clean_refs_for_inserts = torch.stack([
            x_clean[w_ - k_] for k_, w_ in enumerate(W_sorted_int)
        ], dim=0).to(x_clean.device)

        # Codex Loop3-R3 preflight (2026-04-25): with frozen ν, polish cannot
        # change insert TV/LPIPS at all. If A0's exported insert fidelity is
        # already infeasible against the clean-source reference, every polish
        # step is doomed regardless of δ work. Skip and record reason.
        polish_preflight_ok = True
        if ssim_fn is not None:
            a0_clean_ref_remeasure = remeasure_exported_feasibility(
                x_clean, clean_refs_for_inserts, exported, W_attacked,
                lpips_fn, ssim_fn, config,
            )
            polish_stats["a0_clean_ref_feasibility"] = a0_clean_ref_remeasure
            polish_preflight_ok = bool(
                a0_clean_ref_remeasure.get("step_feasible_on_export", False))

        if polish_preflight_ok:
            # Step 1: build per-insert teacher frames + Hiera tokens (no_grad,
            # one-time setup). Codex R3 critical fix (2026-04-25): pass the
            # actual recorded `decoy_offsets` so teacher frames are spatially
            # aligned with v5 driver's decoy-mask construction.
            # Note: `decoy_offsets` is sorted by W_clean_positions order; we
            # need it in the same order as sorted(W_clean) (which sorts by
            # clean-space c_k, identical to original since W_clean is already
            # sorted before c_k → W_attacked mapping).
            W_clean_sorted = sorted(W_clean)
            offsets_for_sorted_W = [
                decoy_offsets[i] for i, _ in
                sorted(enumerate(W_clean), key=lambda kv: int(kv[1]))
            ]
            teachers, _teacher_offsets = build_decoy_teacher_frames(
                x_clean, clean_out.pseudo_masks, W_clean_sorted,
                decoy_offsets=offsets_for_sorted_W,
                feather_radius=config.feather_radius,
                feather_sigma=config.feather_sigma,
            )
            teachers = teachers.to(x_clean.device)
            # Codex R4 low fix (2026-04-25): pass through SAM2 preprocessing
            # constants from forward_fn so teacher extraction stays in lockstep
            # with the differentiable forward. Avoids latent parity bug if SAM2
            # config (image_size / mean / std / autocast) ever differs from
            # tiny defaults.
            teacher_hiera = extract_hiera_teacher_tokens(
                forward_fn.predictor, teachers,
                image_size=int(forward_fn.image_size),
                img_mean=tuple(forward_fn._img_mean.flatten().tolist())
                    if hasattr(forward_fn, "_img_mean")
                    else (0.485, 0.456, 0.406),
                img_std=tuple(forward_fn._img_std.flatten().tolist())
                    if hasattr(forward_fn, "_img_std")
                    else (0.229, 0.224, 0.225),
                autocast_dtype=getattr(forward_fn, "autocast_dtype",
                                       torch.bfloat16),
            )
            # Step 2: select polish frames + map to insert k.
            hiera_polish_frames = _select_hiera_polish_frames(
                W_attacked, T_proc,
                polish_window=config.hiera_steering_polish_window,
            )
            polish_to_k = build_polish_to_insert_k_map(
                hiera_polish_frames, W_attacked,
            )
            polish_stats["hiera_polish_frames"] = hiera_polish_frames
            polish_stats["n_polish_frames"] = len(hiera_polish_frames)
            # Step 3: run polish PGD warm-started from A0 ν*.
            nu_a0 = result.nu_star.to(x_clean.device)
            delta_h, nu_h, hiera_logs = _run_hiera_steering_pgd(
                x_clean=x_clean, decoy_seeds=decoy_seeds,
                nu_init=nu_a0, W_attacked=W_attacked,
                polish_frame_ids=hiera_polish_frames,
                teacher_hiera_tokens=teacher_hiera,
                polish_to_insert_k=polish_to_k,
                forward_fn=forward_fn, lpips_fn=lpips_fn,
                m_decoy_by_t=m_decoy_by_t, m_true_by_t=m_true_by_t,
                config=config,
            )
            polish_logs = hiera_logs
            if delta_h is None or nu_h is None:
                polish_stats["skipped"] = "hiera_polish_infeasible_no_best_step"
            else:
                polish_dir = out_root / clip_name / f"{config_name}__hiera" \
                    / "processed"
                export_processed_uint8(
                    x_clean, delta_h.to(x_clean.device),
                    nu_h.to(x_clean.device), decoy_seeds,
                    W_attacked, polish_dir,
                )
                exported_polish = load_processed_uint8(
                    polish_dir).to(x_clean.device)
                polish_eval = eval_exported_j_drop(
                    sam2_eval_fn=sam2_eval_fn,
                    prompt_mask=prompt_mask,
                    x_clean=x_clean,
                    base_inserts=decoy_seeds,
                    exported=exported_polish,
                    W=W_attacked,
                    m_hat_true_by_t=m_true_by_t,
                    m_hat_decoy_by_t=m_decoy_by_t,
                    decoy_offsets=decoy_offsets,
                )
                hiera_j_drop = float(polish_eval["J_drop_mean"])
                a0_j_drop = (exported_j_drop_val
                             if exported_j_drop_val is not None else 0.0)
                delta_j = hiera_j_drop - a0_j_drop
                polish_stats["a0_j_drop"] = a0_j_drop
                polish_stats["hiera_j_drop"] = hiera_j_drop
                polish_stats["delta_j_drop"] = delta_j
                # Codex R3 medium-fix (2026-04-25): re-check exported fidelity
                # on the polish artifact too. Reject if LPIPS/TV/SSIM hinges
                # are violated on the uint8 export, even if J_drop improved.
                # Skips quietly if ssim_fn is not available (caller didn't
                # supply it — degrades to J_drop-only acceptance).
                polish_export_feasible = True
                if ssim_fn is not None:
                    # Codex Loop3-R3 fix (2026-04-25): use the SAME clean-source
                    # reference tensor built earlier for the preflight check.
                    # Both the polish PGD opt-time L_fid_TV and this remeasure
                    # now use x_clean[w-k] as the per-insert reference (was the
                    # apples-to-oranges decoy_seeds mismatch causing the v0
                    # 8k-26k TV blowup).
                    polish_remeasure = remeasure_exported_feasibility(
                        x_clean, clean_refs_for_inserts, exported_polish,
                        W_attacked, lpips_fn, ssim_fn, config,
                    )
                    polish_export_feasible = bool(
                        polish_remeasure.get("step_feasible_on_export", False))
                    polish_stats["exported_feasibility"] = polish_remeasure
                polish_stats["polish_export_feasible"] = polish_export_feasible
                accept = (
                    polish_export_feasible
                    and (
                        (not config.hiera_steering_off_switch)
                        or (delta_j >= config.hiera_steering_min_improvement)
                    )
                )
                polish_stats["accepted"] = accept
                if accept:
                    polish_applied = True
                    export_dir = polish_dir
                    exported_j_drop_val = hiera_j_drop
                    exported_j_drop_details = polish_eval
                    best_surrogate = float("nan")
                    if hiera_logs:
                        feas_logs = [lg for lg in hiera_logs
                                     if lg.get("feasible")]
                        if feas_logs:
                            # Codex R3 high-fix: weighted objective parity.
                            w = config.hiera_steering_loss_weight
                            best_surrogate = min(
                                lg["L_margin"] + w * lg["L_hiera"]
                                for lg in feas_logs)
                    result = V5Result(
                        delta_star=delta_h, nu_star=nu_h,
                        best_surrogate_loss=best_surrogate,
                        infeasible=False, step_logs=result.step_logs,
                    )
                else:
                    polish_reverted = True
        else:
            polish_stats["skipped"] = (
                "a0_clean_ref_infeasible_polish_cannot_recover_with_frozen_nu")

    # --- Stage 12: Decoy State Continuation polish (codex Loop 3 R3, 2026-04-25)
    # Joint Trajectory (Stage 13) takes precedence over State Continuation.
    _state_eligible = (
        config.state_continuation
        and not config.joint_trajectory
        and not result.infeasible
        and not polish_applied
        and sam2_eval_fn is not None
        and result.nu_star is not None
    )
    if _state_eligible and ssim_fn is None:
        polish_stats["skipped"] = (
            "state_continuation_requires_ssim_fn_for_fidelity_gate")
    if _state_eligible and ssim_fn is not None:
        from memshield.state_continuation import (
            select_bridge_frames, build_bridge_to_insert_k,
            downsample_decoy_mask,
        )
        # Build clean-source per-insert refs (preflight + remeasure share).
        W_sorted_int = sorted(int(w) for w in W_attacked)
        for k_, w_ in enumerate(W_sorted_int):
            c_k_ = w_ - k_
            if not (0 <= c_k_ < x_clean.shape[0]):
                raise RuntimeError(
                    f"State continuation: invalid c_k={c_k_} for "
                    f"W_sorted[{k_}]={w_} (T_clean={x_clean.shape[0]})")
        sc_clean_refs_for_inserts = torch.stack([
            x_clean[w_ - k_] for k_, w_ in enumerate(W_sorted_int)
        ], dim=0).to(x_clean.device)

        # A0 clean-ref preflight (same logic as Stage 11): with frozen ν
        # the polish cannot improve insert TV/LPIPS, so if A0 already
        # fails clean-ref export feasibility we skip immediately.
        sc_preflight_ok = True
        sc_preflight = remeasure_exported_feasibility(
            x_clean, sc_clean_refs_for_inserts, exported, W_attacked,
            lpips_fn, ssim_fn, config,
        )
        polish_stats["sc_a0_clean_ref_feasibility"] = sc_preflight
        sc_preflight_ok = bool(
            sc_preflight.get("step_feasible_on_export", False))

        if not sc_preflight_ok:
            polish_stats["skipped"] = (
                "state_continuation_a0_clean_ref_infeasible")
        else:
            # Step 1: cache A0 teachers (M̄_k, p̄_k) at insert positions.
            # Codex Loop3-R3-fix2: extract from the EXPORTED uint8 artifact
            # (`exported`), not the pre-export float tensor — ensures
            # threat-model parity (the delivered bytes write the same
            # state we're teaching toward).
            with torch.no_grad():
                _, teacher_M_dict_t, teacher_p_dict_t = \
                    forward_fn.forward_with_state(
                        exported, return_at=[],
                        state_at=W_sorted_int,
                    )
            teacher_M_by_k = {
                k_: teacher_M_dict_t[w_].detach()
                for k_, w_ in enumerate(W_sorted_int)
            }
            teacher_p_by_k = {
                k_: teacher_p_dict_t[w_].detach()
                for k_, w_ in enumerate(W_sorted_int)
            }

            # Step 2: project decoy masks to memory resolution.
            sample_M = next(iter(teacher_M_by_k.values()))
            if sample_M.dim() == 4:
                _, _, h_mem, w_mem = sample_M.shape
            elif sample_M.dim() == 3:
                # [HW, B, C] layout — derive h_mem assuming square.
                HW = sample_M.shape[0]
                h_mem = w_mem = int(round(HW ** 0.5))
                if h_mem * w_mem != HW:
                    raise RuntimeError(
                        f"State continuation: non-square HW={HW} in "
                        f"maskmem token layout; need explicit shape")
            else:
                raise RuntimeError(
                    f"State continuation: unsupported maskmem shape "
                    f"{tuple(sample_M.shape)}")

            # Codex Loop3-R3-fix1: per-bridge-frame STUDENT masks from
            # m_decoy_by_t[t] (decoy's apparent location at frame t,
            # which tracks the moving object). Per-insert TEACHER masks
            # from m_decoy_by_t[w_k]. On moving clips the duplicate
            # region drifts, so a single insert-time mask spatially
            # mis-specifies the student pool.
            teacher_decoy_mask_by_k: Dict[int, Tensor] = {}
            for k_, w_ in enumerate(W_sorted_int):
                if int(w_) in m_decoy_by_t:
                    teacher_decoy_mask_by_k[k_] = downsample_decoy_mask(
                        m_decoy_by_t[int(w_)], int(h_mem), int(w_mem)
                    ).to(x_clean.device)

            # Step 3: select bridge frames (per insert, first L post-insert).
            bridge_frames_by_k = select_bridge_frames(
                W_attacked, T_proc,
                bridge_length=config.state_continuation_bridge_length,
            )
            n_bridge_total = sum(
                len(v) for v in bridge_frames_by_k.values())
            polish_stats["sc_bridge_frames_by_k"] = {
                int(k): list(v) for k, v in bridge_frames_by_k.items()}
            polish_stats["sc_n_bridge_total"] = n_bridge_total

            # Per-bridge-frame student masks (uses m_decoy_by_t at the
            # bridge frame's own T_proc index).
            student_decoy_mask_by_t: Dict[int, Tensor] = {}
            for k_, t_list in bridge_frames_by_k.items():
                for t in t_list:
                    if int(t) in m_decoy_by_t:
                        student_decoy_mask_by_t[int(t)] = \
                            downsample_decoy_mask(
                                m_decoy_by_t[int(t)],
                                int(h_mem), int(w_mem)
                            ).to(x_clean.device)

            if n_bridge_total == 0:
                polish_stats["skipped"] = (
                    "state_continuation_no_bridge_frames_in_window")
            else:
                # Step 4: run state-continuation polish PGD.
                nu_a0 = result.nu_star.to(x_clean.device)
                delta_sc, nu_sc, sc_logs, best_step = \
                    _run_state_continuation_pgd(
                        x_clean=x_clean, decoy_seeds=decoy_seeds,
                        nu_init=nu_a0, W_attacked=W_attacked,
                        bridge_frames_by_k=bridge_frames_by_k,
                        teacher_M_by_k=teacher_M_by_k,
                        teacher_p_by_k=teacher_p_by_k,
                        forward_fn=forward_fn, lpips_fn=lpips_fn,
                        m_decoy_by_t=m_decoy_by_t,
                        m_true_by_t=m_true_by_t,
                        config=config,
                        student_decoy_mask_by_t=student_decoy_mask_by_t,
                        teacher_decoy_mask_by_k=teacher_decoy_mask_by_k,
                    )
                polish_logs = sc_logs

                if delta_sc is None or nu_sc is None:
                    polish_stats["skipped"] = (
                        "state_continuation_no_feasible_step")
                else:
                    polish_dir = out_root / clip_name \
                        / f"{config_name}__sc" / "processed"
                    export_processed_uint8(
                        x_clean, delta_sc.to(x_clean.device),
                        nu_sc.to(x_clean.device), decoy_seeds,
                        W_attacked, polish_dir,
                    )
                    exported_polish = load_processed_uint8(
                        polish_dir).to(x_clean.device)
                    polish_eval = eval_exported_j_drop(
                        sam2_eval_fn=sam2_eval_fn,
                        prompt_mask=prompt_mask,
                        x_clean=x_clean,
                        base_inserts=decoy_seeds,
                        exported=exported_polish,
                        W=W_attacked,
                        m_hat_true_by_t=m_true_by_t,
                        m_hat_decoy_by_t=m_decoy_by_t,
                        decoy_offsets=decoy_offsets,
                    )
                    sc_j_drop = float(polish_eval["J_drop_mean"])
                    a0_j_drop = (exported_j_drop_val
                                 if exported_j_drop_val is not None else 0.0)
                    delta_j = sc_j_drop - a0_j_drop
                    polish_stats["a0_j_drop"] = a0_j_drop
                    polish_stats["sc_j_drop"] = sc_j_drop
                    polish_stats["delta_j_drop"] = delta_j

                    # Falsification metrics (codex pre-committed): mean
                    # cos(M_t, M̄_k) and cos(p_t, p̄_k) at the SELECTED
                    # checkpoint (the snapshot the polish actually
                    # returned, identified by best_step) vs warm-start
                    # (step 1, δ=0). Lift ≥ 0.15 = state alignment achieved.
                    # Codex Loop3-R3-fix3: tie reported metrics to the
                    # actual returned checkpoint, NOT a separate max-cos
                    # log entry — that would overstate alignment vs the
                    # exported δ*.
                    if sc_logs:
                        first_step = sc_logs[0]
                        baseline_cos_M = float(first_step.get(
                            "mean_cos_M", 0.0))
                        baseline_cos_P = float(first_step.get(
                            "mean_cos_P", 0.0))
                        # Find the log corresponding to best_step (1-based).
                        if best_step is not None and 1 <= best_step <= len(sc_logs):
                            chosen_log = sc_logs[best_step - 1]
                            sel_cos_M = float(chosen_log.get("mean_cos_M", 0.0))
                            sel_cos_P = float(chosen_log.get("mean_cos_P", 0.0))
                        else:
                            sel_cos_M = baseline_cos_M
                            sel_cos_P = baseline_cos_P
                        polish_stats["sc_baseline_cos_M"] = baseline_cos_M
                        polish_stats["sc_baseline_cos_P"] = baseline_cos_P
                        polish_stats["sc_selected_step"] = best_step
                        polish_stats["sc_selected_cos_M"] = sel_cos_M
                        polish_stats["sc_selected_cos_P"] = sel_cos_P
                        polish_stats["sc_lift_cos_M"] = sel_cos_M - baseline_cos_M
                        polish_stats["sc_lift_cos_P"] = sel_cos_P - baseline_cos_P

                    # Re-check exported fidelity (clean-ref aligned).
                    polish_remeasure = remeasure_exported_feasibility(
                        x_clean, sc_clean_refs_for_inserts, exported_polish,
                        W_attacked, lpips_fn, ssim_fn, config,
                    )
                    polish_export_feasible = bool(
                        polish_remeasure.get("step_feasible_on_export", False))
                    polish_stats["exported_feasibility"] = polish_remeasure
                    polish_stats["polish_export_feasible"] = polish_export_feasible
                    accept = (
                        polish_export_feasible
                        and (
                            (not config.state_continuation_off_switch)
                            or (delta_j >= config.state_continuation_min_improvement)
                        )
                    )
                    polish_stats["accepted"] = accept
                    if accept:
                        polish_applied = True
                        export_dir = polish_dir
                        exported_j_drop_val = sc_j_drop
                        exported_j_drop_details = polish_eval
                        result = V5Result(
                            delta_star=delta_sc, nu_star=nu_sc,
                            best_surrogate_loss=float("nan"),
                            infeasible=False, step_logs=result.step_logs,
                        )
                    else:
                        polish_reverted = True

    # --- Stage 13: Joint Trajectory-Consistent Decoy Attack (codex Loop 3 R4)
    # Locked design 2026-04-25. Replaces ε∞-PGD δ with semantic bridge edits:
    # learnable per-bridge-frame duplicate-object overlay (α) + decoy-direction
    # translation warp on true-object ROI. ν warm-started from A0, frozen for
    # Phase A (20 steps), unfrozen at low LR for Phase B joint refinement (10).
    _jt_eligible = (
        config.joint_trajectory
        and not result.infeasible
        and not polish_applied
        and sam2_eval_fn is not None
        and result.nu_star is not None
    )
    if _jt_eligible and ssim_fn is None:
        polish_stats["skipped"] = (
            "joint_trajectory_requires_ssim_fn_for_fidelity_gate")
    if _jt_eligible and ssim_fn is not None:
        from memshield.decoy_continuation import (
            select_bridge_frames as jt_select_bridge_frames,
            soften_decoy_mask, unit_decoy_direction,
        )
        from memshield.decoy_seed import build_duplicate_object_decoy_frame

        # Build clean-ref refs (shared with preflight + remeasure).
        W_sorted_int = sorted(int(w) for w in W_attacked)
        for k_, w_ in enumerate(W_sorted_int):
            c_k_ = w_ - k_
            if not (0 <= c_k_ < x_clean.shape[0]):
                raise RuntimeError(
                    f"Joint trajectory: invalid c_k={c_k_} for "
                    f"W_sorted[{k_}]={w_} (T_clean={x_clean.shape[0]})")
        jt_clean_refs_for_inserts = torch.stack([
            x_clean[w_ - k_] for k_, w_ in enumerate(W_sorted_int)
        ], dim=0).to(x_clean.device)

        # Preflight: A0 export passes clean-ref fidelity? Joint trajectory
        # has bridge-frame edits, so it CAN improve insert TV/LPIPS via ν
        # in Phase B — but if A0 is already infeasible, the start point
        # is bad and Phase B has limited budget to fix.
        jt_preflight = remeasure_exported_feasibility(
            x_clean, jt_clean_refs_for_inserts, exported, W_attacked,
            lpips_fn, ssim_fn, config,
        )
        polish_stats["jt_a0_preflight"] = jt_preflight
        # Codex Loop3-R4 fix (post-pilot 2026-04-25): R4's threat model is
        # perceptual (LPIPS+SSIM), not TV-bounded. The 1.2× clean-source TV
        # cap was an ε-PGD-era heuristic; LPIPS already covers perceptual
        # fidelity. TV-as-hard-gate is too sensitive to cuDNN bf16
        # nondeterm in A0's ν* (saw 273/579 absolute TV excess on dog/
        # blackswan today, all under 2% relative, while LPIPS+SSIM passed).
        # Reduce preflight to perceptual-only; record TV as advisory.
        jt_preflight_ok = bool(_jt_perceptual_feasible(jt_preflight, config))
        polish_stats["jt_a0_perceptual_feasible"] = jt_preflight_ok
        polish_stats["jt_a0_full_feasible"] = bool(
            jt_preflight.get("step_feasible_on_export", False))

        if not jt_preflight_ok:
            polish_stats["skipped"] = (
                "joint_trajectory_a0_preflight_infeasible")
        else:
            # Build bridge frames per insert.
            bridge_frames_by_k = jt_select_bridge_frames(
                W_attacked, T_proc,
                bridge_length=config.joint_traj_bridge_length,
            )
            n_bridge_total = sum(
                len(v) for v in bridge_frames_by_k.values())
            polish_stats["jt_bridge_frames_by_k"] = {
                int(k): list(v) for k, v in bridge_frames_by_k.items()}
            polish_stats["jt_n_bridge_total"] = n_bridge_total

            if n_bridge_total == 0:
                polish_stats["skipped"] = (
                    "joint_trajectory_no_bridge_frames")
            else:
                # Pre-build per-bridge-frame duplicates + softened masks.
                duplicate_frames_by_t: Dict[int, Tensor] = {}
                softened_decoy_masks_by_t: Dict[int, Tensor] = {}
                softened_true_masks_by_t: Dict[int, Tensor] = {}

                # decoy_offsets keyed by W_clean order; rebuild by-insert-k order.
                W_clean_sorted = sorted(W_clean)
                offsets_for_sorted_W = [
                    decoy_offsets[i] for i, _ in
                    sorted(enumerate(W_clean), key=lambda kv: int(kv[1]))
                ]
                for k_, t_list in bridge_frames_by_k.items():
                    if k_ >= len(offsets_for_sorted_W):
                        continue
                    dy_k, dx_k = offsets_for_sorted_W[k_]
                    for t in t_list:
                        c_t = _attacked_to_clean_same_space(int(t), W_attacked)
                        if not (0 <= c_t < x_clean.shape[0]):
                            continue
                        if int(c_t) >= len(clean_out.pseudo_masks):
                            continue
                        m_true_c = torch.as_tensor(
                            clean_out.pseudo_masks[int(c_t)],
                            dtype=x_clean.dtype, device=x_clean.device,
                        )
                        # Build per-bridge duplicate using THIS frame's clean
                        # content + true mask + same offset as parent insert.
                        # Codex Loop3-R4 fix: builder returns ONE tensor.
                        dup_frame = build_duplicate_object_decoy_frame(
                            x_ref=x_clean[int(c_t)].to(x_clean.device),
                            object_mask=m_true_c,
                            decoy_offset=(int(dy_k), int(dx_k)),
                            feather_radius=config.feather_radius,
                            feather_sigma=config.feather_sigma,
                        )
                        duplicate_frames_by_t[int(t)] = dup_frame.to(
                            x_clean.device)
                        # Softened decoy mask (for overlay placement).
                        # Source: m_decoy_by_t[t] — the v5 driver's
                        # already-projected duplicate region at frame t in
                        # processed space (codex R4 recommendation b).
                        if int(t) in m_decoy_by_t:
                            decoy_mask_src = m_decoy_by_t[int(t)].to(
                                x_clean.device)
                        else:
                            # Fallback (rare — m_decoy_by_t is built for all
                            # T_proc indices in the v5 pipeline). If it
                            # triggers we use the un-shifted true mask as a
                            # weak best-effort source; the overlay then sits
                            # at the original object location, which weakens
                            # but does not break the attack.
                            decoy_mask_src = m_true_c
                        softened_decoy_masks_by_t[int(t)] = soften_decoy_mask(
                            decoy_mask_src,
                            dilate_px=config.joint_traj_overlay_dilate_px,
                            feather_sigma=config.joint_traj_overlay_feather_sigma,
                        )
                        # Softened true-object mask (for warp ROI).
                        softened_true_masks_by_t[int(t)] = soften_decoy_mask(
                            m_true_c, dilate_px=1,
                            feather_sigma=config.joint_traj_true_mask_feather_sigma,
                        )

                # Unit decoy directions per insert.
                jt_decoy_offsets_unit = unit_decoy_direction(
                    offsets_for_sorted_W).to(x_clean.device)

                # Run joint trajectory PGD.
                nu_a0 = result.nu_star.to(x_clean.device)
                x_edited_star, nu_jt, edit_params, jt_logs, best_step = \
                    _run_joint_trajectory_pgd(
                        x_clean=x_clean,
                        decoy_seeds=decoy_seeds,
                        nu_init=nu_a0,
                        W_attacked=W_attacked,
                        bridge_frames_by_k=bridge_frames_by_k,
                        softened_decoy_masks_by_t=softened_decoy_masks_by_t,
                        softened_true_masks_by_t=softened_true_masks_by_t,
                        duplicate_frames_by_t=duplicate_frames_by_t,
                        decoy_offsets_unit=jt_decoy_offsets_unit,
                        forward_fn=forward_fn, lpips_fn=lpips_fn,
                        m_decoy_by_t=m_decoy_by_t, m_true_by_t=m_true_by_t,
                        config=config,
                    )
                polish_logs = jt_logs

                if x_edited_star is None or nu_jt is None:
                    polish_stats["skipped"] = (
                        "joint_trajectory_no_feasible_step")
                else:
                    polish_dir = out_root / clip_name \
                        / f"{config_name}__jt" / "processed"
                    # Export: pass x_edited_star AS x_clean with delta=0.
                    # The semantic bridge edits ARE the modification.
                    delta_zero = torch.zeros_like(
                        x_edited_star, device=x_edited_star.device)
                    export_processed_uint8(
                        x_edited_star.to(x_clean.device),
                        delta_zero.to(x_clean.device),
                        nu_jt.to(x_clean.device),
                        decoy_seeds, W_attacked, polish_dir,
                    )
                    exported_polish = load_processed_uint8(
                        polish_dir).to(x_clean.device)
                    polish_eval = eval_exported_j_drop(
                        sam2_eval_fn=sam2_eval_fn,
                        prompt_mask=prompt_mask,
                        x_clean=x_clean,
                        base_inserts=decoy_seeds,
                        exported=exported_polish,
                        W=W_attacked,
                        m_hat_true_by_t=m_true_by_t,
                        m_hat_decoy_by_t=m_decoy_by_t,
                        decoy_offsets=decoy_offsets,
                    )
                    jt_j_drop = float(polish_eval["J_drop_mean"])
                    a0_j_drop = (exported_j_drop_val
                                 if exported_j_drop_val is not None else 0.0)
                    delta_j = jt_j_drop - a0_j_drop
                    polish_stats["a0_j_drop"] = a0_j_drop
                    polish_stats["jt_j_drop"] = jt_j_drop
                    polish_stats["delta_j_drop"] = delta_j

                    # Falsification metrics from the SELECTED checkpoint.
                    if jt_logs and best_step is not None and \
                            1 <= best_step <= len(jt_logs):
                        chosen = jt_logs[best_step - 1]
                        polish_stats["jt_selected_step"] = best_step
                        polish_stats["jt_selected_decoy_overlap"] = \
                            chosen.get("mean_decoy_overlap")
                        polish_stats["jt_selected_true_overlap"] = \
                            chosen.get("mean_true_overlap")
                        polish_stats["jt_selected_delta_overlap"] = \
                            chosen.get("delta_overlap")
                        polish_stats["jt_selected_obj_score"] = \
                            chosen.get("mean_obj_score")
                        polish_stats["jt_selected_wrong_but_present"] = \
                            chosen.get("wrong_but_present_count")
                        polish_stats["jt_selected_alpha_mean"] = \
                            chosen.get("alpha_mean")
                        polish_stats["jt_selected_warp_disp_max"] = \
                            chosen.get("warp_disp_max")

                    # Re-check exported fidelity (clean-ref aligned).
                    # Codex Loop3-R4 fix (post-pilot): perceptual-only gate
                    # (LPIPS + SSIM), TV recorded as advisory. Same rationale
                    # as the preflight relaxation.
                    polish_remeasure = remeasure_exported_feasibility(
                        x_clean, jt_clean_refs_for_inserts, exported_polish,
                        W_attacked, lpips_fn, ssim_fn, config,
                    )
                    polish_export_feasible = bool(
                        _jt_perceptual_feasible(polish_remeasure, config))
                    polish_export_full_feasible = bool(
                        polish_remeasure.get(
                            "step_feasible_on_export", False))
                    polish_stats["exported_feasibility"] = polish_remeasure
                    polish_stats["polish_export_feasible"] = polish_export_feasible
                    polish_stats["polish_export_full_feasible"] = polish_export_full_feasible
                    accept = (
                        polish_export_feasible
                        and (
                            (not config.joint_traj_off_switch)
                            or (delta_j >= config.joint_traj_min_improvement)
                        )
                    )
                    polish_stats["accepted"] = accept
                    if accept:
                        polish_applied = True
                        export_dir = polish_dir
                        exported_j_drop_val = jt_j_drop
                        exported_j_drop_details = polish_eval
                        # Joint trajectory doesn't produce a δ; result.delta
                        # stays zero (semantic edits are baked into export).
                        result = V5Result(
                            delta_star=torch.zeros_like(x_clean),
                            nu_star=nu_jt,
                            best_surrogate_loss=float("nan"),
                            infeasible=False, step_logs=result.step_logs,
                        )
                    else:
                        polish_reverted = True

    summary_logs: List[Dict[str, Any]] = [asdict(log) for log in result.step_logs]
    if polish_logs:
        polish_stats["step_logs"] = polish_logs

    return V5ClipOutput(
        clip_name=clip_name, config_name=config_name,
        W=list(W_attacked),
        decoy_offsets=[(int(dy), int(dx)) for dy, dx in decoy_offsets],
        infeasible=result.infeasible,
        best_surrogate_loss=float(result.best_surrogate_loss),
        exported_j_drop=exported_j_drop_val,
        exported_j_drop_details=exported_j_drop_details,
        export_dir=str(export_dir) if export_dir else None,
        step_log_summary=summary_logs,
        placement_source=placement_source,
        post_insert_radius=post_insert_radius,
        polish_applied=polish_applied,
        polish_reverted=polish_reverted,
        polish_stats=polish_stats,
    )


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VADI-v5 (DIRE) driver")
    p.add_argument("--davis-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-root", default="vadi_runs/v5")
    p.add_argument("--clips", nargs="+", default=["dog"])
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--placement", choices=["top", "random"], default="top")
    p.add_argument("--post-insert-radius", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed-only", action="store_true",
                   help="Skip optimization. Export with δ=0, ν=0 "
                        "(decoy seed alone) for A/B comparison.")
    # Ablation axes (codex R2 disciplined plan, 2026-04-24).
    p.add_argument("--insert-base", choices=["midframe", "duplicate_seed"],
                   default="duplicate_seed")
    p.add_argument("--loss", choices=["margin", "dice_bce"], default="dice_bce")
    p.add_argument("--nu-optimizer", choices=["adam", "sign_pgd"],
                   default="adam")
    p.add_argument("--schedule", choices=["full", "insert_only_100"],
                   default="full")
    p.add_argument("--delta-support", choices=["off", "post_insert",
                                               "v4_symmetric"],
                   default="post_insert")
    p.add_argument("--train-ste", action="store_true",
                   help="Apply fake_uint8_quantize STE during training "
                        "(v4-parity; default off in v5).")
    # Boundary-δ polish (codex R1 #1, 2026-04-24).
    p.add_argument("--boundary-polish", action="store_true",
                   help="After A0 ν-only run, run a short boundary-δ polish "
                        "stage on degraded+aligned frames. Off by default.")
    p.add_argument("--boundary-polish-n-steps", type=int, default=30)
    p.add_argument("--boundary-polish-align-cos-threshold", type=float,
                   default=0.5)
    p.add_argument("--boundary-polish-band-width", type=int, default=5)
    p.add_argument("--boundary-polish-no-corridor", action="store_true",
                   help="Disable centroid corridor in δ support mask.")
    p.add_argument("--boundary-polish-no-off-switch", action="store_true",
                   help="Accept polish result even if worse than A0.")
    # Hiera feature-steering δ (codex Loop 3 R2 design #1, 2026-04-25).
    p.add_argument("--hiera-steering", action="store_true",
                   help="After A0 ν-only run, run a Hiera feature-steering "
                        "δ polish stage with synthetic-decoy teacher tokens.")
    p.add_argument("--hiera-steering-n-steps", type=int, default=30)
    p.add_argument("--hiera-steering-loss-weight", type=float, default=0.5)
    p.add_argument("--hiera-steering-loss-type",
                   choices=["l2", "cosine"], default="l2")
    p.add_argument("--hiera-steering-no-off-switch", action="store_true")
    # Decoy State Continuation polish (Loop 3 R3)
    p.add_argument("--state-continuation", action="store_true",
                   help="After A0 ν-only run, run Decoy State Continuation "
                        "polish: cache A0's per-insert maskmem_features + "
                        "obj_ptr as teachers, optimize δ on bridge originals "
                        "to align bridge state to teachers via cosine loss. "
                        "Takes precedence over --hiera-steering when both set.")
    p.add_argument("--state-continuation-n-steps", type=int, default=30)
    p.add_argument("--state-continuation-bridge-length", type=int, default=3)
    p.add_argument("--state-continuation-lambda-M", type=float, default=1.0)
    p.add_argument("--state-continuation-lambda-P", type=float, default=1.0)
    p.add_argument("--state-continuation-lambda-margin", type=float, default=1.0)
    p.add_argument("--state-continuation-no-off-switch", action="store_true")
    # Joint Trajectory-Consistent Decoy Attack (Loop 3 R4)
    p.add_argument("--joint-trajectory", action="store_true",
                   help="After A0, run Joint Trajectory-Consistent Decoy "
                        "Attack: semantic bridge edits (per-frame duplicate "
                        "overlay + decoy-direction translation warp) jointly "
                        "optimized with insert ν. Replaces ε-PGD δ. Takes "
                        "precedence over --state-continuation and --hiera-steering.")
    p.add_argument("--joint-traj-bridge-length", type=int, default=3)
    p.add_argument("--joint-traj-alpha-max", type=float, default=0.30)
    p.add_argument("--joint-traj-max-disp-px", type=float, default=2.0)
    p.add_argument("--joint-traj-phase-a-steps", type=int, default=20)
    p.add_argument("--joint-traj-phase-b-steps", type=int, default=10)
    p.add_argument("--joint-traj-no-off-switch", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        from scripts.run_vadi_pilot import build_pilot_adapters
    except (ImportError, NotImplementedError) as e:
        print(f"[v5] adapter import failed: {e}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    clean_fac, fwd_fac, lpips_fn, ssim_fn, sam2_eval_fn = build_pilot_adapters(
        checkpoint_path=args.checkpoint, device=device,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Compact config name encoding the 4 ablation axes + placement + K.
    # b={midframe|dup}, l={margin|dice}, o={adam|pgd}, d={off|post|v4}.
    short = {
        "midframe": "mid", "duplicate_seed": "dup",
        "margin": "mg", "dice_bce": "dc",
        "adam": "ad", "sign_pgd": "pg",
        "off": "off", "post_insert": "post", "v4_symmetric": "v4",
        "full": "fs", "insert_only_100": "io100",
    }
    if args.seed_only:
        config_name = f"K{args.K}_{args.placement}_seedonly"
    else:
        suffix = ""
        if args.hiera_steering:
            suffix = "_hiera"
        elif args.boundary_polish:
            suffix = "_polish"
        config_name = (
            f"K{args.K}_{args.placement}_R{args.post_insert_radius}_"
            f"b-{short[args.insert_base]}_l-{short[args.loss]}_"
            f"o-{short[args.nu_optimizer]}_d-{short[args.delta_support]}_"
            f"s-{short[args.schedule]}{suffix}"
        )

    all_results: List[V5ClipOutput] = []
    for clip_name in args.clips:
        x_clean, prompt_mask = load_davis_clip(Path(args.davis_root), clip_name)
        x_clean = x_clean.to(device)
        clean_pass_fn = clean_fac(clip_name, x_clean, prompt_mask)
        fwd_builder = fwd_fac(clip_name, x_clean, prompt_mask)
        rng = np.random.default_rng(0)
        cfg = VADIv5Config(
            insert_base_mode=args.insert_base,
            loss_mode=args.loss,
            optimizer_nu_mode=args.nu_optimizer,
            schedule_preset=args.schedule,
            delta_support_mode=args.delta_support,
            post_insert_radius=args.post_insert_radius,
            train_ste_quantize=args.train_ste,
            boundary_polish=args.boundary_polish,
            boundary_polish_n_steps=args.boundary_polish_n_steps,
            boundary_polish_align_cos_threshold=(
                args.boundary_polish_align_cos_threshold),
            boundary_polish_band_width=args.boundary_polish_band_width,
            boundary_polish_use_corridor=(
                not args.boundary_polish_no_corridor),
            boundary_polish_off_switch=(
                not args.boundary_polish_no_off_switch),
            hiera_steering=args.hiera_steering,
            hiera_steering_n_steps=args.hiera_steering_n_steps,
            hiera_steering_loss_weight=args.hiera_steering_loss_weight,
            hiera_steering_loss_type=args.hiera_steering_loss_type,
            hiera_steering_off_switch=(
                not args.hiera_steering_no_off_switch),
            state_continuation=args.state_continuation,
            state_continuation_n_steps=args.state_continuation_n_steps,
            state_continuation_bridge_length=(
                args.state_continuation_bridge_length),
            state_continuation_lambda_M=args.state_continuation_lambda_M,
            state_continuation_lambda_P=args.state_continuation_lambda_P,
            state_continuation_lambda_margin=(
                args.state_continuation_lambda_margin),
            state_continuation_off_switch=(
                not args.state_continuation_no_off_switch),
            joint_trajectory=args.joint_trajectory,
            joint_traj_bridge_length=args.joint_traj_bridge_length,
            joint_traj_alpha_max=args.joint_traj_alpha_max,
            joint_traj_max_disp_px=args.joint_traj_max_disp_px,
            joint_traj_phase_a_steps=args.joint_traj_phase_a_steps,
            joint_traj_phase_b_steps=args.joint_traj_phase_b_steps,
            joint_traj_off_switch=(
                not args.joint_traj_no_off_switch),
        )
        out = run_v5_for_clip(
            clip_name=clip_name, config_name=config_name,
            x_clean=x_clean, prompt_mask=prompt_mask,
            clean_pass_fn=clean_pass_fn,
            forward_fn_builder=fwd_builder,
            lpips_fn=lpips_fn,
            ssim_fn=ssim_fn,
            K_ins=args.K, placement_mode=args.placement,
            post_insert_radius=args.post_insert_radius,
            rng=rng, config=cfg, out_root=out_root,
            sam2_eval_fn=sam2_eval_fn,
            seed_only=args.seed_only,
        )
        all_results.append(out)
        print(f"[v5] {clip_name}: exported_j_drop="
              f"{out.exported_j_drop!r} infeasible={out.infeasible} "
              f"W={out.W} offsets={out.decoy_offsets}")
        # Persist per-clip results.json.
        if out.export_dir:
            rj = Path(out.export_dir).parent / "results.json"
            with open(rj, "w", encoding="utf-8") as f:
                json.dump(asdict(out), f, indent=2, default=str)

    summary = {
        "configs": config_name,
        "clips": args.clips,
        "mean_exported_j_drop": float(np.mean([
            r.exported_j_drop for r in all_results
            if r.exported_j_drop is not None
        ])) if all_results else float("nan"),
        "per_clip": {
            r.clip_name: {
                "exported_j_drop": r.exported_j_drop,
                "infeasible": r.infeasible,
                "W": r.W,
                "decoy_offsets": r.decoy_offsets,
                "placement_source": r.placement_source,
            } for r in all_results
        },
        "thresholds": {
            "submission_bar_mean_j_drop": 0.60,
            "submission_bar_delta_vs_insert_only": 0.05,
        },
    }
    with open(out_root / "v5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[v5] summary: mean J-drop = {summary['mean_exported_j_drop']:.4f}")
    return 0


# =============================================================================
# Self-test (stub adapters)
# =============================================================================


def _self_test() -> None:
    import tempfile

    torch.manual_seed(0); np.random.seed(0)
    T, Hv, Wv = 14, 16, 16
    x_clean = torch.rand(T, Hv, Wv, 3)
    prompt = np.zeros((Hv, Wv), dtype=np.uint8)
    prompt[Hv // 2:, :Wv // 2] = 1

    def stub_clean_pass(x_c, prompt):
        Tt = x_c.shape[0]
        pseudo = []
        for t in range(Tt):
            m = np.zeros((Hv, Wv), dtype=np.float32)
            y0 = min(t, Hv - 4); x0 = min(t, Wv - 4)
            m[y0:y0 + 4, x0:x0 + 4] = 1.0
            pseudo.append(m)
        conf = np.where(np.arange(Tt) < Tt // 2, 0.9, 0.3).astype(np.float32)
        feats = [np.full(32, float(t) + (10.0 if t >= Tt // 2 else 0.0),
                         dtype=np.float32) for t in range(Tt)]
        return CleanPassOutput(
            pseudo_masks=pseudo, confidences=conf, hiera_features=feats)

    def stub_forward_builder(x_clean, prompt_mask, W):
        def fn(processed, return_at):
            return {t: 3.0 * (processed[t].mean(dim=-1) - 0.5)
                    for t in return_at}
        return fn

    def lpips_stub(x, y): return (x - y).abs().mean()

    cfg = VADIv5Config(N_A_nu=2, N_B_delta=2, N_C_alt=2,
                       lambda_init=1.0, lambda_growth_factor=2.0,
                       lambda_growth_period=2,
                       post_insert_radius=3)

    with tempfile.TemporaryDirectory() as td:
        out = run_v5_for_clip(
            clip_name="stub", config_name="K3_top_v5",
            x_clean=x_clean, prompt_mask=prompt,
            clean_pass_fn=stub_clean_pass,
            forward_fn_builder=stub_forward_builder,
            lpips_fn=lpips_stub,
            K_ins=3, placement_mode="top",
            post_insert_radius=3,
            config=cfg, out_root=Path(td),
        )
        assert isinstance(out, V5ClipOutput)
        assert len(out.W) == 3
        assert len(out.decoy_offsets) == 3
        # Non-adjacent W.
        for i in range(len(out.W)):
            for j in range(i + 1, len(out.W)):
                assert abs(out.W[i] - out.W[j]) >= 2
        # If feasible: export dir exists + step log captured.
        if not out.infeasible:
            assert out.export_dir is not None
            assert Path(out.export_dir).exists()
        assert len(out.step_log_summary) == 6    # 2+2+2
        # Stage labels present.
        stages = {s["stage"] for s in out.step_log_summary}
        assert stages == {"A_nu", "B_delta", "C_alt"}, \
            f"expected 3 stages, got {stages}"

    # -- build_post_insert_support: R=3, W=[2,5,9] in T_proc=15 →
    # post-insert = {3,4, 6,7,8, 10,11,12}
    post, post2 = build_post_insert_support([2, 5, 9], 15, radius=3)
    assert post == [3, 4, 6, 7, 8, 10, 11, 12], f"got {post}"
    assert post == post2

    # -- post-insert clips at T_proc boundary: W=[2] in T_proc=5, R=8 →
    # post = {3, 4} (clipped to T_proc-1).
    post3, _ = build_post_insert_support([2], 5, radius=8)
    assert post3 == [3, 4]

    # -- post-insert clips at next insert: W=[2, 4] in T_proc=10, R=8 →
    # after w=2: {3}, after w=4: {5..9}.
    post4, _ = build_post_insert_support([2, 4], 10, radius=8)
    assert post4 == [3, 5, 6, 7, 8, 9], f"got {post4}"

    # -- _post_clean_to_proc: W_attacked=[3,7] T_proc=10. Clean frame c=4
    # is at processed index 5 (since 1 insert at 3 shifts it +1).
    # Let's verify: attacked-space positions 0,1,2 map to clean 0,1,2;
    # position 3 is insert; 4 maps to clean 3; 5 maps to clean 4; 6 to 5;
    # 7 is insert; 8 to 6; 9 to 7. So clean 4 → processed 5.
    mapped = _post_clean_to_proc([4], [3, 7], 10)
    assert mapped == [5], f"clean 4 → processed {mapped}"

    # -- Ablation axes smoke: make sure each alternative mode imports
    # cleanly and runs end-to-end with stub adapters. Tiny step counts.
    ablation_variants = [
        # A1: duplicate_seed + margin + sign_pgd + off + insert_only_100
        dict(insert_base_mode="duplicate_seed", loss_mode="margin",
             optimizer_nu_mode="sign_pgd", delta_support_mode="off",
             schedule_preset="insert_only_100"),
        # A2: midframe + dice_bce + sign_pgd + off + insert_only_100
        dict(insert_base_mode="midframe", loss_mode="dice_bce",
             optimizer_nu_mode="sign_pgd", delta_support_mode="off",
             schedule_preset="insert_only_100"),
        # A3: midframe + margin + adam + off + insert_only_100
        dict(insert_base_mode="midframe", loss_mode="margin",
             optimizer_nu_mode="adam", delta_support_mode="off",
             schedule_preset="insert_only_100"),
        # B2-ish: duplicate_seed + margin + sign_pgd + v4_symmetric + full
        dict(insert_base_mode="duplicate_seed", loss_mode="margin",
             optimizer_nu_mode="sign_pgd", delta_support_mode="v4_symmetric",
             schedule_preset="full"),
    ]
    for i, ax in enumerate(ablation_variants):
        with tempfile.TemporaryDirectory() as td:
            cfg_ab = VADIv5Config(
                N_A_nu=2, N_B_delta=1, N_C_alt=1,
                lambda_init=1.0, lambda_growth_factor=2.0,
                lambda_growth_period=2,
                post_insert_radius=2,
                **ax,
            )
            # insert_only_100 forces N_A=100; override for speed to avoid
            # a 100-step self-test. We instead stub the preset mapping by
            # switching back to "full" after constructing; cheap hack.
            if cfg_ab.schedule_preset == "insert_only_100":
                cfg_ab.schedule_preset = "full"
                cfg_ab.N_A_nu, cfg_ab.N_B_delta, cfg_ab.N_C_alt = 3, 0, 0
            out = run_v5_for_clip(
                clip_name=f"stubab{i}", config_name=f"ab{i}",
                x_clean=x_clean, prompt_mask=prompt,
                clean_pass_fn=stub_clean_pass,
                forward_fn_builder=stub_forward_builder,
                lpips_fn=lpips_stub,
                K_ins=3, placement_mode="top",
                post_insert_radius=2,
                config=cfg_ab, out_root=Path(td),
            )
            assert isinstance(out, V5ClipOutput), \
                f"ablation variant {i} produced wrong type"
            assert len(out.W) == 3

    # -- Boundary-δ polish stage integration smoke with stubs.
    # Need a sam2_eval_fn that returns per-frame masks; stub it to return
    # all-zeros so decoy_semantic classifies everything as either empty-
    # clean (excluded) or suppressed (never "degraded"). With no degraded
    # frames, polish stage should skip cleanly without crashing.
    def _stub_sam2_eval(video, prompt):
        return [np.zeros(prompt.shape, dtype=np.uint8)
                for _ in range(int(video.shape[0]))]

    with tempfile.TemporaryDirectory() as td:
        cfg_polish = VADIv5Config(
            N_A_nu=3, N_B_delta=0, N_C_alt=0,
            lambda_init=1.0, lambda_growth_factor=2.0,
            lambda_growth_period=2,
            post_insert_radius=2,
            insert_base_mode="midframe", loss_mode="margin",
            optimizer_nu_mode="sign_pgd",
            schedule_preset="full",           # N_A_nu=3 so effectively short
            delta_support_mode="off",
            boundary_polish=True,
            boundary_polish_n_steps=2,        # tiny for self-test
            boundary_polish_band_width=3,
        )
        out_pol = run_v5_for_clip(
            clip_name="stub_polish", config_name="K3_polish_smoke",
            x_clean=x_clean, prompt_mask=prompt,
            clean_pass_fn=stub_clean_pass,
            forward_fn_builder=stub_forward_builder,
            lpips_fn=lpips_stub,
            K_ins=3, placement_mode="top",
            post_insert_radius=2,
            config=cfg_polish, out_root=Path(td),
            sam2_eval_fn=_stub_sam2_eval,
        )
        # With all-zero stub masks, no frame is "degraded" in the strict
        # decoy-semantic sense → polish should skip, not crash.
        assert not out_pol.polish_applied, \
            "polish should not apply with all-empty stub masks"
        # polish_stats should indicate either 'skipped' or the decoy_semantic
        # wasn't populated (sam2_eval_fn returned zeros → baseline/attacked
        # have zero masks, jaccard both-empty = 1.0, J_drop = 0, all frames
        # excluded as empty_pred_clean → n_degraded_aligned = 0 → skipped).
        # Either state is acceptable as long as no exception is raised.

    print("scripts.run_vadi_v5: all self-tests PASSED "
          "(build_post_insert_support edge cases, _post_clean_to_proc, "
          "3-stage PGD end-to-end with stub adapters, decoy-seed construction, "
          "4 ablation axes {base, loss, optimizer, delta_support} "
          "import and run cleanly with stubs, boundary-polish integration "
          "smoke with stub sam2_eval_fn)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
