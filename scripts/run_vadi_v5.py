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
    W: Sequence[int], post_insert_clean: Sequence[int],
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
    if fake_uint8:
        from memshield.losses import fake_uint8_quantize
        x_prime = fake_uint8_quantize(x_prime)
        inserts = fake_uint8_quantize(inserts)
    processed = build_processed(x_prime, inserts, W)
    T_proc = processed.shape[0]

    # Query SAM2 at insert positions + post-insert positions.
    return_at = set(int(w) for w in W) | set(int(t) for t in range(T_proc)
                                             if t not in set(int(w) for w in W))
    # Filter return_at to only what we need for loss (saves compute): insert + post-insert.
    # post-insert is already in processed-space.
    return_at_small = set(int(w) for w in W)
    post_insert_proc_set = _post_clean_to_proc(post_insert_clean, W, T_proc)
    return_at_small |= set(post_insert_proc_set)
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


def run_v5_for_clip(
    clip_name: str,
    config_name: str,
    x_clean: Tensor,
    prompt_mask: np.ndarray,
    clean_pass_fn: Callable[[Tensor, np.ndarray], CleanPassOutput],
    forward_fn_builder: Callable[..., Callable],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    *,
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

    # --- δ support (codex R2 ablation axis) ---
    if config.delta_support_mode == "off":
        post_insert_proc: List[int] = []
        post_insert_clean: List[int] = []
    elif config.delta_support_mode == "post_insert":
        post_insert_proc, _ = build_post_insert_support(
            W_attacked, T_proc, radius=post_insert_radius,
        )
        post_insert_clean = sorted({
            _attacked_to_clean_same_space(a, W_attacked)
            for a in post_insert_proc if a not in set(W_attacked)
        })
    elif config.delta_support_mode == "v4_symmetric":
        # v4-style: ∪_k NbrSet(W_k, ±2) ∪ {f0_processed=0} in attacked space,
        # map to clean-space (excluding insert positions themselves).
        from memshield.vadi_optimize import build_support_sets
        S_delta_proc, _ = build_support_sets(
            W_attacked, T_proc, f0_processed_id=0)
        post_insert_proc = sorted({
            int(a) for a in S_delta_proc if int(a) not in set(W_attacked)
        })
        post_insert_clean = sorted({
            _attacked_to_clean_same_space(a, W_attacked)
            for a in post_insert_proc
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
            )
            exported_j_drop_val = float(
                exported_j_drop_details["J_drop_mean"])

    summary_logs: List[Dict[str, Any]] = [asdict(log) for log in result.step_logs]

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
        config_name = (
            f"K{args.K}_{args.placement}_R{args.post_insert_radius}_"
            f"b-{short[args.insert_base]}_l-{short[args.loss]}_"
            f"o-{short[args.nu_optimizer]}_d-{short[args.delta_support]}_"
            f"s-{short[args.schedule]}"
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
        )
        out = run_v5_for_clip(
            clip_name=clip_name, config_name=config_name,
            x_clean=x_clean, prompt_mask=prompt_mask,
            clean_pass_fn=clean_pass_fn,
            forward_fn_builder=fwd_builder,
            lpips_fn=lpips_fn,
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

    print("scripts.run_vadi_v5: all self-tests PASSED "
          "(build_post_insert_support edge cases, _post_clean_to_proc, "
          "3-stage PGD end-to-end with stub adapters, decoy-seed construction, "
          "4 ablation axes {base, loss, optimizer, delta_support} "
          "import and run cleanly with stubs)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
