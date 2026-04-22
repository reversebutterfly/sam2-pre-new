"""
MemoryShield Chunk 5a: per-video PGD optimizer (orchestration skeleton).

Wires scheduler (Chunk 2) + insert-base generator (Chunk 3) + loss stack
(Chunk 4) + memory-attention probe (Chunk 1) into the 3-stage PGD loop
specified in FINAL_PROPOSAL.md §Training Plan:

    Stage 1 (steps 1-40):      ν-only  + L_loss
    Stage 2 (steps 41-80):     δ-only  + L_rec   (inserts frozen)
    Stage 3 (steps 81-200):    joint 2:1 δ:ν + full L

Chunk 5a is SAM2-model-agnostic: it drives the optimization loop against
an abstract forward callable `Sam2ForwardFn` that must produce:
    * per-insert logit maps g_{ins_k} (for L_loss)
    * per-eval-frame logit maps g_u    (for L_rec suppress + low-conf)
    * per-stale-frame P_u              (for L_stale — Chunk 1 probe output)

Chunk 5b will supply the real SAM2-backed implementation + runtime
provenance binding. A `dummy_sam2_forward_fn` is provided here so that
5a can be smoke-tested end-to-end without a GPU or a checkpoint.

Design contracts (exactly one clear contract across the three chunks)
---------------------------------------------------------------------
All tensors live on a single torch device (cfg.device) in float32.
Pixel tensors are in [0, 1]. uint8 frames come in from numpy and are
converted at the boundary. The optimizer owns ν and δ as leaf tensors
with `requires_grad_(True)`. At each PGD step, the loss is evaluated on
a freshly assembled `modified_video = build_modified_video(state)`
tensor that is a differentiable function of ν, δ, and the frozen
insert bases / original frames.

L∞ budget enforcement on δ is done by hard clamping per step (classic
PGD). ν has no L∞ constraint — it is only region-masked (to the
edit_mask) and LPIPS-budgeted via the augmented-Lagrangian `l_fid`
term; the Lagrange multiplier μ_ν grows exponentially when LPIPS
exceeds the budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .scheduler import (
    ScheduleV2,
    build_index_maps_v2,
    compute_schedule_v2,
)
from . import losses_v2 as LV


# ---------------------------------------------------------------------------
# Config / state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptimizeConfig:
    """All hyper-parameters for one per-video PGD run. Values come from
    FINAL_PROPOSAL.md defaults unless otherwise noted."""

    # -- schedule / inserts --
    K_ins: int = 3
    num_maskmem: int = 7
    schedule_variant: str = "canonical"
    schedule_offset: int = 0
    schedule_custom_m: Optional[Sequence[int]] = None

    # -- insert base realization --
    insert_base_strategy: str = "propainter"   # alt: "poisson_hifi"
    seam_dilate_px: int = 5
    insert_safety_margin: int = 12

    # -- PGD budgets / ranges --
    eps_f0: float = 2.0 / 255.0            # f0 conditioning slot
    eps_other: float = 4.0 / 255.0         # f1..f_{T-1} prefix
    lpips_budget: float = 0.10

    # -- stages --
    n_steps: int = 200
    stage1_end: int = 40                    # 1..40 = Stage 1 (ν-only)
    stage2_end: int = 80                    # 41..80 = Stage 2 (δ-only)
    # 81..n_steps = Stage 3 (joint 2:1 δ:ν)

    # -- step sizes --
    lr_nu: float = 4.0 / 255.0
    lr_delta: float = 1.0 / 255.0
    # In Stage 3 we do `stage3_delta_per_nu_ratio` δ-steps per ν-step.
    stage3_delta_per_nu_ratio: int = 2

    # -- loss weights / multipliers (FINAL_PROPOSAL §Loss) --
    alpha_loss_cvar: float = 1.0
    margin_loss: float = 0.2
    alpha_supp_rec: float = 1.0
    alpha_conf_rec: float = 1.0
    tau_conf: float = -1.0
    beta_stale: float = 0.3
    lambda_rec: float = 1.0
    lambda_fid: float = 1.0
    mu_nu_initial: float = 10.0
    mu_nu_grow: float = 1.5
    mu_s_seam: float = 0.0                 # seam ΔE off by default

    # -- L_stale target + fallback --
    Q: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    use_margin_stale: bool = False
    margin_gamma: float = 0.4
    margin_lambda: float = 0.2

    # -- windows --
    T_prefix_orig: int = 15
    eval_window_size: int = 7               # |U|  — f15..f21
    stale_window_size: int = 3              # |V|  — first 3 after last insert

    # -- runtime --
    device: str = "cuda:0"
    dtype: str = "float32"
    seed: int = 42
    log_every: int = 10
    lagrange_update_every: int = 10


@dataclass
class VideoBundle:
    """Frozen per-video inputs resolved before the PGD loop starts.

    Numpy arrays below are the "source of truth" inputs from upstream
    (decoy search, ProPainter, clean SAM2 forward). The `_cache_*` fields
    are populated lazily by `_ensure_device_cache()` with torch tensors
    on `cfg.device` so the hot inner loop does not re-materialize from
    numpy every PGD step (Codex R5 IMPORTANT #4).

    `frames_orig` is the FULL clean video (prefix + eval suffix), length
    `cfg.T_prefix_orig + cfg.eval_window_size`. The optimizer only
    perturbs the first `cfg.T_prefix_orig` frames via `state.delta`; the
    suffix is read-only and is used by the SAM2 forward impl to evaluate
    recovery (Codex R6 MINOR on unified contract).
    """
    frames_orig: np.ndarray                # [T_full, H, W, 3] uint8
    masks_gt: np.ndarray                   # [T_full, H, W]   uint8 {0,1}
    schedule: ScheduleV2
    insert_bases: List[np.ndarray]         # K × [H, W, 3] uint8
    edit_masks: List[np.ndarray]           # K × [H, W] uint8
    decoy_offset: Tuple[int, int]          # (dy, dx)
    # Per-insert semantic masks (filled downstream):
    D_ins: List[np.ndarray] = field(default_factory=list)   # decoy paste in insert k
    C_ins: List[np.ndarray] = field(default_factory=list)   # true-obj location in insert k
    ROI_ins: List[np.ndarray] = field(default_factory=list) # dilated(D ∪ C)
    # Per-eval-frame foreground-query regions (clean SAM2 prediction, eroded):
    C_u: List[np.ndarray] = field(default_factory=list)     # |U| × [H, W] uint8

    # Device-tensor caches — populated once by _ensure_device_cache(),
    # reused across all PGD steps.
    _cache_frames: Optional[torch.Tensor] = None    # [T, H, W, 3] float
    _cache_bases: Optional[torch.Tensor] = None     # [K, H, W, 3] float
    _cache_edit: Optional[torch.Tensor] = None      # [K, H, W, 1] float
    _cache_device: Optional[torch.device] = None
    _cache_dtype: Optional[torch.dtype] = None


@dataclass
class PGDState:
    """Leaf tensors (ν, δ) + bookkeeping."""
    nu: torch.Tensor        # [K, H, W, 3] float, requires_grad
    delta: torch.Tensor     # [T_orig, H, W, 3] float, requires_grad
    mu_nu: float
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SAM2 forward contract (Chunk 5b implements the real one)
# ---------------------------------------------------------------------------


class Sam2ForwardFn(Protocol):
    """Protocol for SAM2 forward callable used by the optimizer.

    Called per PGD step with `mode="attack"` and once at setup with
    `mode="clean"`.

    Input contract
    --------------
    `modified_video` is the MODIFIED PREFIX only, shape
    `[T_prefix_mod, H, W, 3]` in `[0, 1]` float. The implementation is
    responsible for appending the clean suffix `bundle.frames_orig[
    cfg.T_prefix_orig : cfg.T_prefix_orig + cfg.eval_window_size]` before
    running SAM2 — the optimizer does not modify the suffix.

    Output contract
    ---------------
        {
            "insert_logits": List[Tensor[H, W]],  # len == cfg.K_ins
                # Logit (pre-sigmoid) maps, one per inserted frame, in
                # schedule.slots order. For Stage 1 L_loss.
            "eval_logits":   List[Tensor[H, W]],  # len == cfg.eval_window_size
                # One per eval frame u ∈ U, in original-time order
                # (u = T_prefix_orig .. T_prefix_orig + eval_window_size - 1).
                # For Stage 2/3 L_rec.
            "P_u_list":      List[Optional[Tensor[3]]],  # len == cfg.stale_window_size
                # One per stale frame v ∈ V (first stale_window_size
                # frames after the last insert). Shape [3] = [A_ins,
                # A_recent, A_other]. None = probe did not fire for that
                # frame. For Stage 2/3 L_stale.
            "pred_masks":    Optional[List[Tensor[H, W]]],
                # Only populated when mode="clean"; caller uses this to
                # build `bundle.C_u`.
        }

    All returned tensors must be on the same device as `modified_video`
    and remain differentiable w.r.t. `modified_video` where the caller
    will backprop (insert and eval logits; P_u).
    """
    def __call__(
        self,
        modified_video: torch.Tensor,
        mode: str,
        cfg: OptimizeConfig,
        bundle: VideoBundle,
    ) -> Dict[str, object]: ...


def _compute_lpips_per_insert(
    state: PGDState,
    bundle: VideoBundle,
    cfg: OptimizeConfig,
    lpips_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> List[torch.Tensor]:
    """Fresh LPIPS evaluation. MUST be called inside each forward/backward
    pair — graphs are freed on backward, so reusing an old LPIPS tensor
    across Stage-3 substeps causes a "backward through freed graph" error.

    Contract for `lpips_fn`:
        Input: two tensors each of shape `[H, W, 3]` in `[0, 1]`.
        Output: scalar torch.Tensor (LPIPS distance).
    If callers have a standard `[B, 3, H, W]`-in-`[-1, 1]` LPIPS model,
    they should wrap it with a small adapter before passing here.

    `f_prev` uses the CLEAN preceding original (not `frames_orig + δ`):
    keeps the fidelity constraint honest (Codex R5 IMPORTANT #1).
    Raises if a schedule slot has `o_after == -1` (should be unreachable
    given Chunk 2 enforcement, but we guard explicitly).
    """
    if lpips_fn is None:
        return []
    device = state.nu.device
    dtype = state.nu.dtype
    _ensure_device_cache(bundle, device, dtype)

    bases = bundle._cache_bases          # [K, H, W, 3]
    edit = bundle._cache_edit            # [K, H, W, 1]
    frames_t = bundle._cache_frames      # [T_orig, H, W, 3]

    vals: List[torch.Tensor] = []
    for k, slot in enumerate(bundle.schedule.slots):
        if slot.o_after < 0:
            raise RuntimeError(
                f"slot {k} has o_after=-1; first-position inserts not "
                "supported by optimize_unified_v2 LPIPS path."
            )
        # CLEAN f_prev — no δ term, so LPIPS measures drift from the
        # original prefix frame, not a moving attacked one.
        f_prev = frames_t[slot.o_after]
        x_ins = (bases[k] + state.nu[k] * edit[k]).clamp(0.0, 1.0)
        vals.append(lpips_fn(x_ins, f_prev))
    return vals


# ---------------------------------------------------------------------------
# Frame assembly (differentiable)
# ---------------------------------------------------------------------------


def _to_device_tensor(arr: np.ndarray, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
    """numpy uint8 [..., 3] -> float tensor [..., 3] in [0,1]."""
    t = torch.from_numpy(arr).to(device=device, dtype=dtype) / 255.0
    return t


def _ensure_device_cache(bundle: VideoBundle, device: torch.device,
                         dtype: torch.dtype) -> None:
    """Populate bundle._cache_* with device tensors (once per bundle).

    Called lazily on first forward so the hot inner loop does not
    re-materialize tensors from numpy every step. If device or dtype
    changes, the cache is rebuilt.
    """
    if (bundle._cache_frames is not None
            and bundle._cache_device == device
            and bundle._cache_dtype == dtype):
        return
    bundle._cache_frames = _to_device_tensor(bundle.frames_orig, device, dtype)
    bundle._cache_bases = torch.stack([
        _to_device_tensor(b, device, dtype) for b in bundle.insert_bases
    ], dim=0)
    bundle._cache_edit = torch.stack([
        torch.from_numpy(m).to(device=device, dtype=dtype) for m in bundle.edit_masks
    ], dim=0).unsqueeze(-1)
    bundle._cache_device = device
    bundle._cache_dtype = dtype


def build_modified_video(
    bundle: VideoBundle,
    state: PGDState,
    cfg: OptimizeConfig,
) -> torch.Tensor:
    """Assemble the modified prefix video as a differentiable tensor.

    Returns:
        [T_mod, H, W, 3] float tensor in [0, 1] (clamped). Differentiable
        w.r.t. state.nu and state.delta.

    Layout: for each modified-index m in 0..T_mod-1:
        * if m is an insert (m ∈ {w_k}):   modified[m] = insert_base[k] + ν[k]
        * else (m is an original o):        modified[m] = frames_orig[o] + δ[o]

    ν is masked to the edit region (outside = 0). δ is L∞-clamped upstream
    by `clamp_delta_`. Device tensors are cached on bundle for reuse
    across PGD steps.
    """
    device = state.nu.device
    dtype = state.nu.dtype
    _ensure_device_cache(bundle, device, dtype)

    frames_t = bundle._cache_frames                                    # [T, H, W, 3]
    insert_bases_t = bundle._cache_bases                               # [K, H, W, 3]
    edit_masks_t = bundle._cache_edit                                  # [K, H, W, 1]

    nu_masked = state.nu * edit_masks_t                                # gradient lives on edit region
    attacked_inserts = (insert_bases_t + nu_masked).clamp(0.0, 1.0)    # [K, H, W, 3]
    attacked_prefix  = (frames_t + state.delta).clamp(0.0, 1.0)        # [T, H, W, 3]

    T_mod = bundle.schedule.T_prefix_mod
    H, W = frames_t.shape[1], frames_t.shape[2]
    # Build row-by-row; unavoidable with dynamic insert positions.
    # Index maps are cached on the bundle — they depend only on the
    # schedule which is frozen per-video (Codex R6 MINOR).
    if getattr(bundle, "_cache_idx_maps", None) is None:
        bundle._cache_idx_maps = build_index_maps_v2(bundle.schedule)      # type: ignore[attr-defined]
        bundle._cache_mod_to_k = {                                         # type: ignore[attr-defined]
            s.m_k: k for k, s in enumerate(bundle.schedule.slots)
        }
        bundle._cache_insert_mods = set(                                   # type: ignore[attr-defined]
            bundle._cache_idx_maps["insert_mod_indices"]
        )
    out_rows: List[torch.Tensor] = []
    idx_maps = bundle._cache_idx_maps                                      # type: ignore[attr-defined]
    mod_to_orig = idx_maps["mod_to_orig"]
    insert_mods = bundle._cache_insert_mods                                # type: ignore[attr-defined]
    mod_to_k = bundle._cache_mod_to_k                                      # type: ignore[attr-defined]

    for m in range(T_mod):
        if m in insert_mods:
            k = mod_to_k[m]
            out_rows.append(attacked_inserts[k])
        else:
            o = mod_to_orig[m]
            if o < 0:
                raise RuntimeError(
                    f"Internal: mod_to_orig[{m}] = -1 but m not in insert_mods")
            out_rows.append(attacked_prefix[o])

    return torch.stack(out_rows, dim=0)


# ---------------------------------------------------------------------------
# Clamping / projection
# ---------------------------------------------------------------------------


def clamp_delta_(state: PGDState, cfg: OptimizeConfig) -> None:
    """Hard-clamp state.delta to per-frame L∞ budget. In-place."""
    with torch.no_grad():
        d = state.delta
        # Frame 0 (conditioning): tighter ε_f0. Rest: ε_other.
        d[0].clamp_(-cfg.eps_f0, cfg.eps_f0)
        if d.shape[0] > 1:
            d[1:].clamp_(-cfg.eps_other, cfg.eps_other)


# ---------------------------------------------------------------------------
# Per-stage PGD steps
# ---------------------------------------------------------------------------


def _compute_stage1_loss(
    fwd: Dict[str, object], bundle: VideoBundle, state: PGDState,
    cfg: OptimizeConfig,
    lpips_values: Optional[List[torch.Tensor]] = None,
    seam_dE: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_loss (inserts) + L_fid. Stage 1 gate: gate_loss=1, gate_rec=0."""
    insert_logits = fwd["insert_logits"]           # type: ignore[index]
    device = insert_logits[0].device

    D = [torch.from_numpy(m).to(device=device, dtype=insert_logits[0].dtype)
         for m in bundle.D_ins]
    C = [torch.from_numpy(m).to(device=device, dtype=insert_logits[0].dtype)
         for m in bundle.C_ins]
    ROI = [torch.from_numpy(m).to(device=device, dtype=insert_logits[0].dtype)
           for m in bundle.ROI_ins]

    L_loss = LV.l_loss_insert(
        insert_logits=list(insert_logits), decoy_mask=D, true_mask=C, roi_mask=ROI,
        alpha=cfg.alpha_loss_cvar, margin=cfg.margin_loss,
    )
    L_rec = insert_logits[0].new_zeros(())
    L_fid = insert_logits[0].new_zeros(())
    if lpips_values is not None and len(lpips_values) > 0:
        L_fid = LV.l_fid_augmented(
            lpips_values, mu_nu=state.mu_nu, lpips_budget=cfg.lpips_budget,
            seam_dE=seam_dE, mu_s=cfg.mu_s_seam,
        )
    total = LV.total_loss(
        L_loss, L_rec, L_fid,
        lambda_r=cfg.lambda_rec, lambda_f=cfg.lambda_fid,
        gate_loss=1.0, gate_rec=0.0,
    )
    diag = {"L_loss": float(L_loss.detach()), "L_fid": float(L_fid.detach()),
            "total": float(total.detach()), "mu_nu": state.mu_nu}
    return total, diag


def _compute_stage2_3_loss(
    fwd: Dict[str, object], bundle: VideoBundle, state: PGDState,
    cfg: OptimizeConfig, gate_loss: float, gate_rec: float,
    lpips_values: Optional[List[torch.Tensor]] = None,
    seam_dE: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_rec (+ L_loss when gate_loss=1) + L_fid. Stages 2 and 3."""
    insert_logits: List[torch.Tensor] = fwd["insert_logits"]    # type: ignore[assignment]
    eval_logits:   List[torch.Tensor] = fwd["eval_logits"]      # type: ignore[assignment]
    P_u_list:      List[Optional[torch.Tensor]] = fwd["P_u_list"]  # type: ignore[assignment]
    device = eval_logits[0].device

    # L_loss (Stage 3 only — Stage 2 passes gate_loss=0).
    if gate_loss > 0.0:
        D = [torch.from_numpy(m).to(device=device, dtype=eval_logits[0].dtype)
             for m in bundle.D_ins]
        C_k = [torch.from_numpy(m).to(device=device, dtype=eval_logits[0].dtype)
               for m in bundle.C_ins]
        ROI = [torch.from_numpy(m).to(device=device, dtype=eval_logits[0].dtype)
               for m in bundle.ROI_ins]
        L_loss = LV.l_loss_insert(
            insert_logits=list(insert_logits), decoy_mask=D, true_mask=C_k, roi_mask=ROI,
            alpha=cfg.alpha_loss_cvar, margin=cfg.margin_loss,
        )
    else:
        L_loss = eval_logits[0].new_zeros(())

    C_u = [torch.from_numpy(m).to(device=device, dtype=eval_logits[0].dtype)
           for m in bundle.C_u]
    Q_t = torch.tensor(cfg.Q, dtype=eval_logits[0].dtype, device=device)
    L_rec = LV.l_rec(
        eval_logits=list(eval_logits), true_mask=C_u,
        alpha_supp=cfg.alpha_supp_rec, alpha_conf=cfg.alpha_conf_rec,
        tau_conf=cfg.tau_conf,
        P_u_list=list(P_u_list) if P_u_list is not None else None,
        Q=Q_t, beta=cfg.beta_stale,
        use_margin=cfg.use_margin_stale,
        margin_gamma=cfg.margin_gamma, margin_lambda=cfg.margin_lambda,
    )

    L_fid = eval_logits[0].new_zeros(())
    if lpips_values is not None and len(lpips_values) > 0:
        L_fid = LV.l_fid_augmented(
            lpips_values, mu_nu=state.mu_nu, lpips_budget=cfg.lpips_budget,
            seam_dE=seam_dE, mu_s=cfg.mu_s_seam,
        )

    total = LV.total_loss(
        L_loss, L_rec, L_fid,
        lambda_r=cfg.lambda_rec, lambda_f=cfg.lambda_fid,
        gate_loss=gate_loss, gate_rec=gate_rec,
    )
    diag = {
        "L_loss": float(L_loss.detach()), "L_rec": float(L_rec.detach()),
        "L_fid": float(L_fid.detach()), "total": float(total.detach()),
        "mu_nu": state.mu_nu,
    }
    return total, diag


def _step_nu(state: PGDState, total: torch.Tensor, cfg: OptimizeConfig) -> None:
    """Sign-based PGD step on ν only. Zeros δ.grad if present."""
    if state.delta.grad is not None:
        state.delta.grad.zero_()
    if state.nu.grad is not None:
        state.nu.grad.zero_()
    total.backward(retain_graph=False)
    with torch.no_grad():
        grad = state.nu.grad
        if grad is None:
            return
        # Classic sign-PGD on ν. ν has no L∞ clamp (LPIPS-budgeted via L_fid).
        state.nu.data.add_(-cfg.lr_nu * grad.sign())


def _step_delta(state: PGDState, total: torch.Tensor, cfg: OptimizeConfig) -> None:
    """Sign-based PGD step on δ only. Applies hard L∞ clamp."""
    if state.delta.grad is not None:
        state.delta.grad.zero_()
    if state.nu.grad is not None:
        state.nu.grad.zero_()
    total.backward(retain_graph=False)
    with torch.no_grad():
        grad = state.delta.grad
        if grad is None:
            return
        state.delta.data.add_(-cfg.lr_delta * grad.sign())
    clamp_delta_(state, cfg)


def _step_joint(state: PGDState, total: torch.Tensor, cfg: OptimizeConfig,
                do_nu: bool, do_delta: bool) -> None:
    """Sign-based PGD step on ν and/or δ, followed by δ-clamp."""
    if state.delta.grad is not None:
        state.delta.grad.zero_()
    if state.nu.grad is not None:
        state.nu.grad.zero_()
    total.backward(retain_graph=False)
    with torch.no_grad():
        if do_nu and state.nu.grad is not None:
            state.nu.data.add_(-cfg.lr_nu * state.nu.grad.sign())
        if do_delta and state.delta.grad is not None:
            state.delta.data.add_(-cfg.lr_delta * state.delta.grad.sign())
    clamp_delta_(state, cfg)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _default_edit_mask(bundle: VideoBundle) -> List[np.ndarray]:
    """Fallback edit mask = full-frame ones if not supplied elsewhere."""
    H, W = bundle.insert_bases[0].shape[:2]
    return [np.ones((H, W), dtype=np.uint8) for _ in bundle.insert_bases]


def optimize_unified_v2(
    frames_orig: np.ndarray,
    masks_gt: np.ndarray,
    sam2_forward_fn: Sam2ForwardFn,
    cfg: OptimizeConfig,
    insert_bases: Optional[List[np.ndarray]] = None,
    edit_masks: Optional[List[np.ndarray]] = None,
    decoy_offset: Optional[Tuple[int, int]] = None,
    D_ins: Optional[List[np.ndarray]] = None,
    C_ins: Optional[List[np.ndarray]] = None,
    ROI_ins: Optional[List[np.ndarray]] = None,
    C_u: Optional[List[np.ndarray]] = None,
    lpips_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run the 3-stage per-video PGD loop.

    Required inputs:
        frames_orig: [T, H, W, 3] uint8. T == cfg.T_prefix_orig (or the
            eval-window tail beyond T is handled by sam2_forward_fn).
        masks_gt:    [T, H, W] uint8 (0/1). First-frame mask of the
            target object.
        sam2_forward_fn: callable matching the `Sam2ForwardFn` protocol.
        cfg: OptimizeConfig.

    Optional per-video inputs. If None, the caller is expected to
    supply them from upstream pipeline stages (decoy-search,
    ProPainter base generation, clean-SAM2 forward for C_u):
        insert_bases: K × [H, W, 3] uint8
        edit_masks:   K × [H, W] uint8
        decoy_offset: (dy, dx)
        D_ins, C_ins, ROI_ins: K × [H, W] uint8 semantic masks
        C_u:          |U| × [H, W] uint8 eval-frame foreground
        lpips_fn:     (tensor, tensor) -> scalar Tensor, called as
            lpips_fn(insert_k_rgb, prev_k_rgb) per insert per step.

    Returns:
        (modified_video_uint8[T_mod, H, W, 3], diagnostics)
        Diagnostics is a dict with keys:
            'history': list of per-step loss dicts
            'final_nu': [K, H, W, 3] numpy
            'final_delta': [T, H, W, 3] numpy
            'mu_nu_final': float
            'stage_boundaries': (stage1_end, stage2_end)
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

    # 1. Schedule.
    schedule = compute_schedule_v2(
        T_prefix_orig=cfg.T_prefix_orig,
        num_maskmem=cfg.num_maskmem, K_ins=cfg.K_ins,
        variant=cfg.schedule_variant, offset=cfg.schedule_offset,
        custom_m=cfg.schedule_custom_m,
    )

    # 2. Bundle.
    if insert_bases is None or edit_masks is None or decoy_offset is None:
        raise ValueError(
            "optimize_unified_v2 requires insert_bases, edit_masks, and "
            "decoy_offset. Upstream pipeline (Chunk 3 + decoy search + "
            "ProPainter) should produce them before calling this function."
        )
    bundle = VideoBundle(
        frames_orig=frames_orig, masks_gt=masks_gt, schedule=schedule,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=D_ins or [], C_ins=C_ins or [], ROI_ins=ROI_ins or [],
        C_u=C_u or [],
    )
    if not bundle.D_ins or not bundle.C_ins or not bundle.ROI_ins:
        raise ValueError(
            "D_ins, C_ins, ROI_ins must be provided (K each). These are "
            "built upstream from the decoy-offset search + GT masks.")
    if not bundle.C_u:
        raise ValueError("C_u (foreground queries per eval frame) required.")

    # 3. PGD state init.
    T_orig, H, W, _ = frames_orig.shape
    nu = torch.zeros(cfg.K_ins, H, W, 3, device=device, dtype=dtype,
                     requires_grad=True)
    delta = torch.zeros(T_orig, H, W, 3, device=device, dtype=dtype,
                        requires_grad=True)
    state = PGDState(nu=nu, delta=delta, mu_nu=cfg.mu_nu_initial, step=0)

    # 4. Loop. LPIPS is computed fresh inside each forward/backward pair
    # (Codex R5 CRITICAL #1) so the autograd graph is never backpropped
    # twice. We also bookkeep a detached `last_max_lpips` across substeps
    # for the Lagrange update at step boundaries.
    last_max_lpips: Optional[float] = None
    mode = "attack"

    def _one_forward_backward(
        gate_loss=None, gate_rec=None, do_nu=False, do_delta=False,
        is_stage1=False, is_stage2=False,
    ) -> Dict[str, float]:
        nonlocal last_max_lpips
        modified_video = build_modified_video(bundle, state, cfg)
        fwd_local = sam2_forward_fn(modified_video=modified_video, mode=mode,
                                    cfg=cfg, bundle=bundle)
        lpips_vals = _compute_lpips_per_insert(state, bundle, cfg, lpips_fn)

        if is_stage1:
            total_local, diag_local = _compute_stage1_loss(
                fwd_local, bundle, state, cfg,
                lpips_values=lpips_vals or None,
            )
            _step_nu(state, total_local, cfg)
        elif is_stage2:
            total_local, diag_local = _compute_stage2_3_loss(
                fwd_local, bundle, state, cfg, gate_loss=0.0, gate_rec=1.0,
                lpips_values=lpips_vals or None,
            )
            _step_delta(state, total_local, cfg)
        else:
            total_local, diag_local = _compute_stage2_3_loss(
                fwd_local, bundle, state, cfg,
                gate_loss=gate_loss, gate_rec=gate_rec,
                lpips_values=lpips_vals or None,
            )
            _step_joint(state, total_local, cfg, do_nu=do_nu, do_delta=do_delta)

        if lpips_vals:
            last_max_lpips = max(float(lp.detach()) for lp in lpips_vals)
        return diag_local

    for step in range(1, cfg.n_steps + 1):
        state.step = step

        if step <= cfg.stage1_end:
            diag = _one_forward_backward(is_stage1=True)
            stage = 1
        elif step <= cfg.stage2_end:
            diag = _one_forward_backward(is_stage2=True)
            stage = 2
        else:
            stage = 3
            for _sub in range(cfg.stage3_delta_per_nu_ratio):
                diag = _one_forward_backward(
                    gate_loss=1.0, gate_rec=1.0,
                    do_nu=False, do_delta=True,
                )
            diag = _one_forward_backward(
                gate_loss=1.0, gate_rec=1.0,
                do_nu=True, do_delta=True,
            )

        diag["stage"] = stage
        diag["step"] = step
        state.history.append(diag)

        # Periodic Lagrange update using most recent max LPIPS (detached).
        if (step % cfg.lagrange_update_every == 0
                and last_max_lpips is not None):
            state.mu_nu = LV.update_lagrange_mu(
                state.mu_nu, observed=last_max_lpips, target=cfg.lpips_budget,
                grow=cfg.mu_nu_grow,
            )

        if step % cfg.log_every == 0:
            pass   # hook for future tqdm / wandb integration

    # 5. Assemble final modified video as uint8.
    with torch.no_grad():
        final_mod = build_modified_video(bundle, state, cfg)
        final_np = (final_mod.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)

    diagnostics = {
        "history": state.history,
        "final_nu": state.nu.detach().cpu().numpy(),
        "final_delta": state.delta.detach().cpu().numpy(),
        "mu_nu_final": state.mu_nu,
        "stage_boundaries": (cfg.stage1_end, cfg.stage2_end),
        "schedule": {
            "variant": schedule.variant,
            "w_positions": schedule.w_positions,
            "T_mod": schedule.T_prefix_mod,
        },
    }
    return final_np, diagnostics


# ---------------------------------------------------------------------------
# Dummy forward (for Chunk 5a smoke tests — NOT a real model)
# ---------------------------------------------------------------------------


def dummy_sam2_forward_fn(
    modified_video: torch.Tensor,
    mode: str,
    cfg: OptimizeConfig,
    bundle: VideoBundle,
) -> Dict[str, object]:
    """Returns plausibly-shaped logits + P_u with connection to `modified_video`.

    Each insert's logit map = mean of the corresponding insert frame's RGB
    channels (so backprop reaches ν). Each eval frame's logit map is the
    frame mean. P_u is a softmax over learned-less-than-trivial features
    of the modified-video prefix, so ν and δ BOTH influence it. This is
    only for pipeline smoke-testing; replace with Chunk 5b's real SAM2.
    """
    T_mod, H, W, _ = modified_video.shape
    device = modified_video.device
    dtype = modified_video.dtype

    insert_mods = set(build_index_maps_v2(bundle.schedule)["insert_mod_indices"])
    mod_to_k = {s.m_k: k for k, s in enumerate(bundle.schedule.slots)}

    # Per-insert logit: mean channel of that frame * 4 - 2 (centred around 0)
    insert_logits: List[torch.Tensor] = [None] * cfg.K_ins   # type: ignore[list-item]
    for m in range(T_mod):
        if m in insert_mods:
            k = mod_to_k[m]
            g = modified_video[m].mean(dim=-1) * 4.0 - 2.0
            insert_logits[k] = g

    # Eval logits: first eval_window_size frames past T_prefix_mod. But the
    # input video only has the prefix; we fake eval logits from the last
    # eval_window_size frames of the modified prefix instead (smoke-test only).
    U = cfg.eval_window_size
    eval_logits: List[torch.Tensor] = []
    for i in range(min(U, T_mod)):
        frame = modified_video[T_mod - 1 - i]
        eval_logits.append(frame.mean(dim=-1) * 4.0 - 2.0)

    # P_u: for each stale frame v, compute a 3-vector that depends on
    # modified_video (so gradients reach ν and δ). We use mean pixel intensity
    # in three disjoint bands and softmax.
    V = cfg.stale_window_size
    P_u_list: List[Optional[torch.Tensor]] = []
    for i in range(V):
        if i < T_mod:
            frame = modified_video[T_mod - 1 - i]
            r, g, b = frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()
            logits3 = torch.stack([r, g, b])
            P = torch.softmax(logits3, dim=0)
            P_u_list.append(P)
        else:
            P_u_list.append(None)

    return {
        "insert_logits": insert_logits,
        "eval_logits":   eval_logits,
        "P_u_list":      P_u_list,
        "pred_masks":    None,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Run a tiny end-to-end smoke test of the orchestration skeleton.

    Uses dummy_sam2_forward_fn so no SAM2 weights needed. Verifies that:
      * the schedule + bundle + state plumbing works
      * 3 stages execute in order
      * ν and δ both move away from zero during the run
      * diagnostics are populated
    """
    torch.manual_seed(0)
    T_prefix, eval_len, H, W, K = 15, 7, 48, 64, 3
    T_full = T_prefix + eval_len
    frames = np.random.randint(0, 256, (T_full, H, W, 3), dtype=np.uint8)
    masks = np.zeros((T_full, H, W), dtype=np.uint8)
    masks[:, 10:20, 20:40] = 1

    # Mock insert bases + edit masks + semantic regions.
    insert_bases = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                    for _ in range(K)]
    edit_masks = [np.ones((H, W), dtype=np.uint8) for _ in range(K)]
    D_ins = [np.zeros((H, W), dtype=np.uint8) for _ in range(K)]
    C_ins = [np.zeros((H, W), dtype=np.uint8) for _ in range(K)]
    for m in D_ins:
        m[10:20, 44:60] = 1        # decoy at right side
    for m in C_ins:
        m[10:20, 20:40] = 1        # true obj
    ROI_ins = [((D + C) > 0).astype(np.uint8) for D, C in zip(D_ins, C_ins)]
    C_u = [(masks[10 + i]).astype(np.uint8) for i in range(3)]
    # pad to eval_window_size=7 by repeating last element
    while len(C_u) < 7:
        C_u.append(C_u[-1])

    # Short run so the smoke test is fast.
    cfg = OptimizeConfig(
        n_steps=12, stage1_end=4, stage2_end=8, log_every=1,
        lagrange_update_every=4, T_prefix_orig=T_prefix, K_ins=K,
        device="cpu", stage3_delta_per_nu_ratio=1,
        eval_window_size=eval_len, stale_window_size=3,
    )

    final, diag = optimize_unified_v2(
        frames_orig=frames, masks_gt=masks,
        sam2_forward_fn=dummy_sam2_forward_fn, cfg=cfg,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=(0, 24), D_ins=D_ins, C_ins=C_ins, ROI_ins=ROI_ins,
        C_u=C_u, lpips_fn=None,
    )

    assert final.shape == (cfg.T_prefix_orig + cfg.K_ins, H, W, 3), final.shape
    assert final.dtype == np.uint8
    history = diag["history"]
    assert len(history) == cfg.n_steps, (len(history), cfg.n_steps)
    stages = [h["stage"] for h in history]
    assert stages[:4] == [1, 1, 1, 1]
    assert stages[4:8] == [2, 2, 2, 2]
    assert stages[8:] == [3, 3, 3, 3]

    # ν and δ should have moved (dummy forward gives nonzero gradients).
    nu_norm = float(np.abs(diag["final_nu"]).mean())
    delta_norm = float(np.abs(diag["final_delta"]).mean())
    assert nu_norm > 0.0, nu_norm
    assert delta_norm > 0.0, delta_norm
    # δ respects the L∞ clamp.
    d = diag["final_delta"]
    assert np.abs(d[0]).max() <= cfg.eps_f0 + 1e-6
    if d.shape[0] > 1:
        assert np.abs(d[1:]).max() <= cfg.eps_other + 1e-6

    print(f"  12-step smoke PASS  nu_L1={nu_norm:.4f}  delta_L1={delta_norm:.4f}")
    print(f"  schedule: {diag['schedule']}")

    # -- LPIPS-enabled smoke (Codex R5 MINOR #5) --
    # Exercises Stage-3 multiple-backward path with a differentiable
    # surrogate LPIPS (squared L2). Would have caught the "backward
    # through freed graph" bug in the original Chunk 5a.
    torch.manual_seed(1)
    def dummy_lpips(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b each [H, W, 3] in [0, 1] per the protocol.
        return ((a - b) ** 2).mean()

    frames2 = np.random.randint(0, 256, (T_full, H, W, 3), dtype=np.uint8)
    masks2 = masks.copy()
    final2, diag2 = optimize_unified_v2(
        frames_orig=frames2, masks_gt=masks2,
        sam2_forward_fn=dummy_sam2_forward_fn, cfg=cfg,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=(0, 24), D_ins=D_ins, C_ins=C_ins, ROI_ins=ROI_ins,
        C_u=C_u, lpips_fn=dummy_lpips,
    )
    assert final2.shape == final.shape
    assert len(diag2["history"]) == cfg.n_steps
    # Lagrange should have updated at step 4 and 8.
    assert diag2["mu_nu_final"] == cfg.mu_nu_initial or \
        diag2["mu_nu_final"] >= cfg.mu_nu_initial   # grew or held
    # Stage 3 took 3 forward/backward per logical step (2 substeps + 1
    # joint) and none of them crashed with "backward through freed graph".
    s3 = [h for h in diag2["history"] if h["stage"] == 3]
    assert len(s3) == cfg.n_steps - cfg.stage2_end
    print(f"  LPIPS-enabled smoke PASS  mu_nu_final={diag2['mu_nu_final']:.2f}")

    # -- Stage 3 ratio=2 smoke (Codex R6 MINOR #2) --
    # Exercises the proposal's default "2:1 delta:nu" Stage-3 schedule.
    torch.manual_seed(2)
    cfg3 = OptimizeConfig(
        n_steps=10, stage1_end=2, stage2_end=5, log_every=1,
        lagrange_update_every=4, T_prefix_orig=T_prefix, K_ins=K,
        device="cpu", stage3_delta_per_nu_ratio=2,
        eval_window_size=eval_len, stale_window_size=3,
    )
    final3, diag3 = optimize_unified_v2(
        frames_orig=frames, masks_gt=masks,
        sam2_forward_fn=dummy_sam2_forward_fn, cfg=cfg3,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=(0, 24), D_ins=D_ins, C_ins=C_ins, ROI_ins=ROI_ins,
        C_u=C_u, lpips_fn=dummy_lpips,
    )
    # Each Stage-3 logical step with ratio=2 does 3 forward/backward pairs.
    # n_steps=10, stage1=2, stage2=3, stage3=5 logical -> 2 + 3 + 5*3 = 20
    # but `history` has one entry per logical step, so len == n_steps.
    assert len(diag3["history"]) == cfg3.n_steps
    s3_count = sum(1 for h in diag3["history"] if h["stage"] == 3)
    assert s3_count == cfg3.n_steps - cfg3.stage2_end, s3_count
    print(f"  Stage3 ratio=2 smoke PASS  mu_nu_final={diag3['mu_nu_final']:.2f}")


if __name__ == "__main__":
    _smoke()
