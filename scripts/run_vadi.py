"""VADI CLI driver: clean-SAM2 → vulnerability scoring → PGD → export uint8.

End-to-end single-clip pipeline (File 5 of 8 per HANDOFF_VADI_PILOT.md):

  1. Load clean video x_clean + prompt mask m_0 from DAVIS.
  2. Clean-SAM2 pass: collect per-frame pseudo-masks m̂_true_t, confidences,
     and Hiera features H_t.
  3. Vulnerability scoring (memshield.vulnerability_scorer):
     pick W via rank-sum over 3 signals, with mode ∈ {top, random, bottom}
     and K ∈ {1, 2, 3}.
  4. Decoy construction: find a geometric decoy offset from m̂_true_0 via
     memshield.decoy.find_decoy_region; shift every m̂_true_t to get
     m̂_decoy_t.
  5. Remap clean-space pseudo-masks into attacked-space (insert positions
     take the temporal midframe average of the two surrounding clean masks).
  6. Build VADIInputs + a differentiable forward_fn wrapping the SAM2
     adapter (caller-supplied; stubbable for tests).
  7. Run memshield.vadi_optimize.run_vadi_pgd.
  8. Export processed video as uint8 PNGs; re-measure LPIPS/SSIM/TV on the
     exported artifact to confirm the internal feasibility claim matches
     the delivered bytes.
  9. Persist results.json + per-step diagnostics.

The heavy SAM2 + LPIPS wiring lives in `make_sam2_forward_fn(...)` and
`make_clean_pass_fn(...)`, which the caller injects. The pilot / main-table
drivers (`run_vadi_pilot.py`, `run_vadi_davis10.py`) import and reuse the
pure functions here.

Run `python -m scripts.run_vadi --help` for CLI args; `python scripts/run_vadi.py`
for self-tests with stub adapters (no SAM2/LPIPS required).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

# Allow `python scripts/run_vadi.py` from repo root without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memshield.decoy import find_decoy_region, shift_mask
from memshield.vadi_loss import confidence_weight
from memshield.vadi_optimize import (
    VADIConfig,
    VADIInputs,
    VADIResult,
    attacked_to_clean,
    build_base_inserts,
    build_processed,
    build_support_sets,
    run_vadi_pgd,
)
from memshield.vulnerability_scorer import (
    VulnerabilityScores,
    score as vulnerability_score,
)


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class CleanPassOutput:
    """Per-frame artifacts from the clean-SAM2 pass.

    Indexed by CLEAN-video frame index (0..T_clean-1).
    """

    pseudo_masks: List[np.ndarray]      # [T_clean] of [H_vid, W_vid] float in [0,1]
    confidences: np.ndarray             # [T_clean] scalar confidences
    hiera_features: List[Any]           # [T_clean] per-frame features (tensors or arrays)
    pred_masks_binary: List[np.ndarray] = field(default_factory=list)  # optional hard masks


@dataclass
class VADIClipOutput:
    """Final artifact returned by `run_vadi_for_clip`.

    Two distinct J-drop fields (post codex R1 Fix 2, 2026-04-23):

      best_surrogate_J_drop: INTERNAL to PGD, computed during optimization
        on differentiable `pred_logits` at `insert_ids ∪ neighbor_ids` only,
        against the pseudo-masks (clean_SAM2 self-consistency). Used by the
        optimizer for running-best selection. NOT a paper-claim metric.

      exported_j_drop: EXTERNAL, computed after export. Loads the uint8
        PNG sequence, re-runs SAM2 on both the exported attacked video AND
        a "clean-processed" baseline (x_clean interleaved with base_inserts
        at W, no δ or ν), computes whole-processed-video Jaccard against
        pseudo-GT. This is the metric the pilot gate + main-table claims
        reference. `None` if clip was infeasible on export.
    """

    clip_name: str
    config_name: str                    # "K1_top", "K3_random_seed42", ...
    W: List[int]                        # attacked-space insert positions
    infeasible: bool
    best_surrogate_J_drop: float        # see class docstring — internal PGD metric
    exported_j_drop: Optional[float]    # whole-video J-drop on exported artifact (paper metric)
    exported_j_drop_details: Dict[str, Any]  # per-frame, originals-only, inserts-only
    export_dir: Optional[str]           # PNG directory if exported; None if infeasible
    vulnerability_scores: Dict[str, Any]
    exported_feasibility: Dict[str, Any]
    step_log_summary: List[Dict[str, Any]]   # stripped-down per-step dicts
    decoy_offset: Tuple[int, int]
    placement_source: str = "scorer"         # "scorer" | "override"


# =============================================================================
# Decoy + mask remap utilities
# =============================================================================


def compute_decoy_offset(
    m_hat_true_0: np.ndarray,
    frame_0_uint8: np.ndarray,
) -> Tuple[int, int]:
    """Pick the (dy, dx) decoy shift for this clip.

    Uses `memshield.decoy.find_decoy_region` on the frame-0 hard mask; returns
    just the offset (ignore the decoy_mask + distractor flag, which we
    don't need here — we re-compute the decoy for every frame via `shift_mask`).
    """
    hard0 = (m_hat_true_0 > 0.5).astype(np.uint8)
    if hard0.sum() == 0:
        return (0, 0)
    _, (dy, dx), _ = find_decoy_region(hard0, frame_0_uint8)
    return int(dy), int(dx)


def build_decoy_trajectory(
    pseudo_masks: Sequence[np.ndarray], offset: Tuple[int, int],
) -> List[np.ndarray]:
    """Apply `shift_mask(pseudo_masks[t], dy, dx)` to every frame.

    Since `shift_mask` expects a BINARY uint8 input but our pseudo-masks
    are soft, we quantize to [0, 1] soft via shift of a scaled uint8 proxy,
    then rescale. Equivalently: shift on the float tensor domain directly.
    """
    dy, dx = offset
    out: List[np.ndarray] = []
    for m in pseudo_masks:
        H, W = m.shape
        shifted = np.zeros_like(m, dtype=np.float32)
        sy0, sy1 = max(0, -dy), min(H, H - dy)
        sx0, sx1 = max(0, -dx), min(W, W - dx)
        dy0, dy1 = max(0, dy), min(H, H + dy)
        dx0, dx1 = max(0, dx), min(W, W + dx)
        hl = min(sy1 - sy0, dy1 - dy0)
        wl = min(sx1 - sx0, dx1 - dx0)
        if hl > 0 and wl > 0:
            shifted[dy0:dy0 + hl, dx0:dx0 + wl] = \
                m[sy0:sy0 + hl, sx0:sx0 + wl]
        out.append(shifted)
    return out


def remap_masks_to_processed_space(
    clean_masks: Sequence[np.ndarray], W: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Map clean-space {c → mask} to processed-space {t → mask}.

    Non-insert frames: m[t] = clean_masks[attacked_to_clean(t, W)].
    Insert positions: m[W_k] = midframe average = 0.5·clean[c_k-1] + 0.5·clean[c_k],
      where c_k = W_k - k is the clean-side "right-edge" frame of the gap.
    """
    T_clean = len(clean_masks)
    W_sorted = sorted(int(w) for w in W)
    K = len(W_sorted)
    T_proc = T_clean + K
    out: Dict[int, np.ndarray] = {}
    for t in range(T_proc):
        if t in W_sorted:
            k = W_sorted.index(t)
            c_k = W_sorted[k] - k                             # clean index right-of-gap
            if not (1 <= c_k < T_clean):
                raise ValueError(
                    f"insert k={k} at W={W_sorted[k]} → c_k={c_k} out of range "
                    f"[1, {T_clean}).")
            out[t] = 0.5 * clean_masks[c_k - 1] + 0.5 * clean_masks[c_k]
        else:
            c = attacked_to_clean(t, W_sorted)
            out[t] = clean_masks[c]
    return out


# =============================================================================
# Export + re-measure
# =============================================================================


def export_processed_uint8(
    x_clean: Tensor,              # [T_clean, H, W, 3] in [0, 1]
    delta_star: Tensor,            # [T_clean, H, W, 3]
    nu_star: Tensor,               # [K, H, W, 3]
    base_inserts: Tensor,          # [K, H, W, 3]
    W: Sequence[int],
    out_dir: Path,
) -> List[Path]:
    """Clamp + quantize to uint8 and write PNG per processed frame.

    No JPEG round-trip (PNG is lossless). This is the "delivered bytes"
    the threat model assumes — hence feasibility must hold here, not just
    during optimization.

    Returns the list of PNG paths (processed-order ascending).
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)

    x_prime = (x_clean + delta_star).clamp(0.0, 1.0)
    inserts = (base_inserts + nu_star).clamp(0.0, 1.0)
    processed = build_processed(x_prime, inserts, W)         # [T_proc, H, W, 3]
    u8 = (processed.detach().cpu().numpy() * 255.0 \
          + 0.5).clip(0, 255).astype(np.uint8)

    paths: List[Path] = []
    for t in range(u8.shape[0]):
        p = out_dir / f"frame_{t:04d}.png"
        Image.fromarray(u8[t]).save(p)
        paths.append(p)
    return paths


def load_processed_uint8(png_dir: Path) -> Tensor:
    """Re-read the exported PNG sequence as a float [T_proc, H, W, 3] in [0, 1]."""
    from PIL import Image
    paths = sorted(png_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.png in {png_dir}")
    frames = [np.asarray(Image.open(p), dtype=np.uint8) for p in paths]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Hard-binary Jaccard IoU between two `[H, W]` uint8 masks in `{0, 1}`.
    Both-empty → 1.0 (matches `eval_v2.jaccard`)."""
    a_ = (a > 0).astype(np.uint8)
    b_ = (b > 0).astype(np.uint8)
    if a_.shape != b_.shape:
        raise ValueError(f"shape mismatch: {a_.shape} vs {b_.shape}")
    inter = int(np.logical_and(a_, b_).sum())
    union = int(np.logical_or(a_, b_).sum())
    if union == 0:
        return 1.0
    return inter / union


def eval_exported_j_drop(
    sam2_eval_fn: Callable[[Tensor, np.ndarray], List[np.ndarray]],
    prompt_mask: np.ndarray,
    x_clean: Tensor,
    base_inserts: Tensor,
    exported: Tensor,              # [T_proc, H, W, 3] reloaded from PNG
    W: Sequence[int],
    m_hat_true_by_t: Dict[int, Tensor],  # processed-space float soft masks
) -> Dict[str, Any]:
    """Re-evaluate attack effectiveness on the EXPORTED uint8 artifact.

    This is the codex-R1-mandated paper-claim metric (Fix 2, 2026-04-23).
    Unlike `best_surrogate_J_drop` (which is computed during PGD on
    differentiable logits at insert+neighbor positions against
    pseudo-masks), this function:
      1. Runs clean SAM2 on a "clean-processed" baseline
         (x_clean interleaved with base_inserts at W, no δ, no ν) →
         `masks_clean[t]` — what SAM2 outputs for insert-placement alone.
      2. Runs clean SAM2 on the exported attacked video → `masks_attacked[t]`.
      3. Computes per-frame Jaccard against pseudo-GT
         (binarized `m_hat_true_by_t`) for BOTH runs on the WHOLE processed
         video.
      4. J_drop_exported = J_baseline_mean − J_attacked_mean.

    Why the baseline is "clean-processed" (not "clean-only"): the point of
    the J-drop number is to isolate what the OPTIMIZED (δ, ν) add on top
    of the insert-placement intervention. If we used clean-only as the
    baseline, J_drop would include the effect of simply inserting extra
    frames at W, which is architectural, not attack-specific. The midframe
    baseline gets us "δ + ν optimization contribution on delivered bytes".

    Args:
        sam2_eval_fn(video, prompt) → List[np.ndarray[H, W] uint8 binary]
            per-frame hard masks. Injected from the pilot driver so this
            function stays SAM2-free at import time.
        prompt_mask: `[H, W]` uint8 binary first-frame mask.
        x_clean: `[T_clean, H, W, 3]` float in `[0, 1]`.
        base_inserts: `[K, H, W, 3]` float — temporal-midframe insert bases.
        exported: `[T_proc, H, W, 3]` reloaded uint8 PNG as float in `[0, 1]`.
        W: attacked-space insert positions (length K).
        m_hat_true_by_t: processed-space pseudo-GT (output of
            `remap_masks_to_processed_space`), float in `[0, 1]`; binarized
            here at >0.5 for Jaccard.

    Returns:
        Dict with:
          'J_baseline_mean': float — mean J over whole processed video
          'J_attacked_mean': float
          'J_drop_mean': float — paper claim metric (baseline − attacked)
          'per_frame': {t: {'J_baseline': ..., 'J_attacked': ...,
                            'J_drop': ..., 'is_insert': bool}} for all t
          'J_drop_originals_only': float — mean J_drop across t NOT in W
          'J_drop_inserts_only': float — mean J_drop across t IN W
          'n_originals': int, 'n_inserts': int
    """
    W_sorted = sorted(int(w) for w in W)
    W_set = set(W_sorted)

    # Clean-processed baseline: same interleave structure, zero perturbation.
    # Codex R2 Fix: the attacked video went through uint8 PNG round-trip
    # before `sam2_eval_fn` sees it, but processed_clean would otherwise be
    # a plain float tensor. For measurement purity we apply the SAME uint8
    # quantization (round to integer, scale back to [0, 1]) to processed_clean
    # so both videos are evaluated on the identical numerical grid. Without
    # this, `J_baseline - J_attacked` would mix "quantization damage" into
    # the reported attack effect.
    processed_clean = build_processed(x_clean, base_inserts, W_sorted)
    processed_clean_u8rt = (
        (processed_clean * 255.0 + 0.5).clamp(0.0, 255.0)
        .to(torch.uint8).to(processed_clean.dtype) / 255.0
    )

    # Per-frame binary masks from both videos.
    masks_clean = sam2_eval_fn(processed_clean_u8rt, prompt_mask)
    masks_attacked = sam2_eval_fn(exported, prompt_mask)

    T_proc = processed_clean_u8rt.shape[0]
    if len(masks_clean) != T_proc or len(masks_attacked) != T_proc:
        raise RuntimeError(
            f"sam2_eval_fn returned inconsistent lengths: "
            f"clean={len(masks_clean)}, attacked={len(masks_attacked)}, "
            f"expected {T_proc}")

    per_frame: Dict[int, Dict[str, Any]] = {}
    Js_baseline: List[float] = []
    Js_attacked: List[float] = []
    for t in range(T_proc):
        # pseudo-GT at t (soft → hard).
        if t not in m_hat_true_by_t:
            raise KeyError(
                f"m_hat_true_by_t missing entry for t={t}; "
                f"expected all of range({T_proc})")
        gt_soft = m_hat_true_by_t[t]
        if isinstance(gt_soft, torch.Tensor):
            gt_np = gt_soft.detach().cpu().numpy()
        else:
            gt_np = np.asarray(gt_soft)
        gt_hard = (gt_np > 0.5).astype(np.uint8)

        j_b = _jaccard_binary(masks_clean[t], gt_hard)
        j_a = _jaccard_binary(masks_attacked[t], gt_hard)
        Js_baseline.append(j_b)
        Js_attacked.append(j_a)
        per_frame[t] = {
            "J_baseline": j_b, "J_attacked": j_a, "J_drop": j_b - j_a,
            "is_insert": (t in W_set),
        }

    J_baseline_mean = float(np.mean(Js_baseline))
    J_attacked_mean = float(np.mean(Js_attacked))
    J_drop_mean = J_baseline_mean - J_attacked_mean

    J_orig_drops = [per_frame[t]["J_drop"] for t in range(T_proc)
                    if t not in W_set]
    J_ins_drops = [per_frame[t]["J_drop"] for t in range(T_proc)
                   if t in W_set]

    return {
        "J_baseline_mean": J_baseline_mean,
        "J_attacked_mean": J_attacked_mean,
        "J_drop_mean": J_drop_mean,
        "per_frame": per_frame,
        "J_drop_originals_only": (
            float(np.mean(J_orig_drops)) if J_orig_drops else float("nan")),
        "J_drop_inserts_only": (
            float(np.mean(J_ins_drops)) if J_ins_drops else float("nan")),
        "n_originals": len(J_orig_drops),
        "n_inserts": len(J_ins_drops),
    }


def remeasure_exported_feasibility(
    x_clean: Tensor,
    base_inserts: Tensor,
    exported: Tensor,              # [T_proc, H, W, 3] reloaded from PNG
    W: Sequence[int],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    config: VADIConfig,
) -> Dict[str, Any]:
    """Recompute LPIPS/SSIM/TV on the EXPORTED artifact vs the originals.

    Returns a dict with per-frame values and a `step_feasible` bool. Used
    as a HARD acceptance gate: if PNG-space metrics exceed budget, the clip
    is counted INFEASIBLE in the primary denominator (per FINAL_PROPOSAL.md
    spec, F16).
    """
    from memshield.vadi_loss import lpips_cap_hinge, ssim_floor_hinge, tv_hinge

    T_clean = x_clean.shape[0]
    W_sorted = sorted(int(w) for w in W)

    per_frame_lpips_orig: Dict[int, float] = {}
    per_insert_lpips: Dict[int, float] = {}
    per_insert_tv_excess: Dict[int, float] = {}

    # Re-split exported back into per-clean + per-insert frames via W mapping.
    exported_clean = []
    exported_inserts = []
    clean_i = 0
    for t in range(exported.shape[0]):
        if t in W_sorted:
            exported_inserts.append(exported[t])
        else:
            exported_clean.append(exported[t])
            clean_i += 1
    exported_clean_stack = torch.stack(exported_clean, dim=0)   # [T_clean, H, W, 3]
    exported_inserts_stack = torch.stack(exported_inserts, dim=0)

    for c in range(T_clean):
        if c == 0:
            continue                                            # f0 uses SSIM
        lp = float(lpips_fn(exported_clean_stack[c], x_clean[c]).item())
        per_frame_lpips_orig[c] = lp

    for k in range(exported_inserts_stack.shape[0]):
        lp = float(lpips_fn(exported_inserts_stack[k], base_inserts[k]).item())
        per_insert_lpips[k] = lp
        ins_chw = exported_inserts_stack[k].permute(2, 0, 1)
        base_chw = base_inserts[k].permute(2, 0, 1)
        per_insert_tv_excess[k] = float(tv_hinge(
            ins_chw, base_chw, multiplier=config.tv_multiplier).item())

    # f0 SSIM.
    f0_x = exported_clean_stack[0].permute(2, 0, 1).unsqueeze(0)
    f0_y = x_clean[0].permute(2, 0, 1).unsqueeze(0)
    ssim_f0 = float(ssim_fn(f0_x, f0_y).squeeze().item())
    ssim_hinge = float(ssim_floor_hinge(
        torch.tensor(ssim_f0), floor=config.f0_ssim_floor).item())

    # Consistent tolerance-based checks. PNG round-trip + fp arithmetic can
    # leak ~1e-6 noise; strict equality would spuriously flag infeasibility.
    _EXP_TOL = 1e-6
    orig_ok = all(v <= config.lpips_orig_cap + _EXP_TOL
                  for v in per_frame_lpips_orig.values())
    ins_lpips_ok = all(v <= config.lpips_insert_cap + _EXP_TOL
                       for v in per_insert_lpips.values())
    ins_tv_ok = all(v <= _EXP_TOL for v in per_insert_tv_excess.values())
    f0_ok = ssim_hinge <= _EXP_TOL
    step_feasible = bool(orig_ok and ins_lpips_ok and ins_tv_ok and f0_ok)

    return {
        "per_frame_lpips_orig": per_frame_lpips_orig,
        "per_insert_lpips": per_insert_lpips,
        "per_insert_tv_excess": per_insert_tv_excess,
        "ssim_f0": ssim_f0,
        "step_feasible_on_export": step_feasible,
        "budgets": {
            "lpips_orig_cap": config.lpips_orig_cap,
            "lpips_insert_cap": config.lpips_insert_cap,
            "tv_multiplier": config.tv_multiplier,
            "f0_ssim_floor": config.f0_ssim_floor,
        },
    }


# =============================================================================
# Main pipeline (orchestrator — caller supplies SAM2 + LPIPS + SSIM closures)
# =============================================================================


def run_vadi_for_clip(
    clip_name: str,
    config_name: str,
    x_clean: Tensor,               # [T_clean, H, W, 3] in [0, 1]
    prompt_mask: np.ndarray,       # [H, W] uint8 binary
    clean_pass_fn: Callable[[Tensor, np.ndarray], CleanPassOutput],
    forward_fn_builder: Callable[..., Callable],  # closes over x_clean
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    vulnerability_mode: str = "top",
    K_ins: int = 1,
    min_gap: int = 2,
    rng: Optional[np.random.Generator] = None,
    config: Optional[VADIConfig] = None,
    out_root: Optional[Path] = None,
    W_clean_override: Optional[Sequence[int]] = None,
    sam2_eval_fn: Optional[
        Callable[[Tensor, np.ndarray], List[np.ndarray]]
    ] = None,
) -> VADIClipOutput:
    """Top-level single-clip orchestrator.

    See module docstring for the 9-stage pipeline. `clean_pass_fn` and
    `forward_fn_builder` are injected to decouple the real SAM2 dependencies
    from the orchestration; stubs in `_self_test` exercise the full flow.
    """
    config = config or VADIConfig()
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    # Stage 2: clean-SAM2 pass.
    clean_out = clean_pass_fn(x_clean, prompt_mask)
    assert len(clean_out.pseudo_masks) == x_clean.shape[0]

    # Stage 3: vulnerability scoring.
    vs = vulnerability_score(
        confidences=clean_out.confidences,
        pseudo_masks=clean_out.pseudo_masks,
        hiera_features=clean_out.hiera_features,
    )
    # `pick` returns CLEAN-space indices. Convert to ATTACKED-space (W) by
    # inserting k at clean-idx c_k → attacked-idx W_k = c_k + k AFTER sort.
    # Callers driving paired baselines (e.g., `run_vadi_pilot.py` matching
    # a random-5-draw to a top placement) can skip the scorer and supply
    # `W_clean_override` directly.
    if W_clean_override is not None:
        W_clean = sorted(int(c) for c in W_clean_override)
        if len(W_clean) != K_ins:
            raise ValueError(
                f"W_clean_override has {len(W_clean)} entries; K_ins={K_ins}")
        # Validate: uniqueness, range [1, T_clean-1], min_gap.
        T_clean = x_clean.shape[0]
        if len(set(W_clean)) != K_ins:
            raise ValueError(
                f"W_clean_override has duplicate entries: {W_clean_override}")
        for c in W_clean:
            if not (1 <= c < T_clean):
                raise ValueError(
                    f"W_clean_override entry {c} out of [1, {T_clean}) "
                    f"— clean-space insert points need 1 ≤ c_k < T_clean.")
        for i in range(1, len(W_clean)):
            if W_clean[i] - W_clean[i - 1] < min_gap:
                raise ValueError(
                    f"W_clean_override violates min_gap={min_gap}: "
                    f"{W_clean[i - 1]} and {W_clean[i]}")
        placement_source = "override"
    else:
        W_clean = vs.pick(K=K_ins, mode=vulnerability_mode, min_gap=min_gap,
                          rng=rng if vulnerability_mode == "random" else None)
        placement_source = "scorer"
    W_attacked = sorted([c + k for k, c in enumerate(sorted(W_clean))])

    # Stage 4: decoy offset + per-frame decoy trajectory.
    frame_0_u8 = (x_clean[0].cpu().numpy() * 255.0 + 0.5).clip(0, 255) \
        .astype(np.uint8)
    decoy_offset = compute_decoy_offset(clean_out.pseudo_masks[0], frame_0_u8)
    decoy_masks = build_decoy_trajectory(clean_out.pseudo_masks, decoy_offset)

    # Stage 5: remap to processed-space.
    m_hat_true_by_t = remap_masks_to_processed_space(
        clean_out.pseudo_masks, W_attacked)
    m_hat_decoy_by_t = remap_masks_to_processed_space(
        decoy_masks, W_attacked)
    m_hat_true_by_t_t = {t: torch.from_numpy(m).float()
                         for t, m in m_hat_true_by_t.items()}
    m_hat_decoy_by_t_t = {t: torch.from_numpy(m).float()
                          for t, m in m_hat_decoy_by_t.items()}

    # Stage 6: VADIInputs.
    T_proc = x_clean.shape[0] + len(W_attacked)
    S_delta_proc, neighbor_ids = build_support_sets(
        W_attacked, T_proc, f0_processed_id=0)
    # S_δ in attacked-space → clean-space (excluding insert positions).
    S_delta_clean_list = sorted({attacked_to_clean(a, W_attacked)
                                 for a in S_delta_proc if a not in W_attacked})
    base_inserts = build_base_inserts(x_clean, W_attacked)

    inputs = VADIInputs(
        x_clean=x_clean,
        base_inserts=base_inserts,
        W=W_attacked,
        S_delta_clean=S_delta_clean_list,
        insert_ids_processed=list(W_attacked),
        neighbor_ids_processed=neighbor_ids,
        m_hat_true_by_t=m_hat_true_by_t_t,
        m_hat_decoy_by_t=m_hat_decoy_by_t_t,
        f0_clean_id=0,
    )

    # Stage 7: build forward_fn + run PGD.
    forward_fn = forward_fn_builder(x_clean=x_clean, prompt_mask=prompt_mask,
                                    W=W_attacked)
    result: VADIResult = run_vadi_pgd(
        inputs, forward_fn, lpips_fn, ssim_fn, config=config,
    )

    # Stage 8: export + re-measure (only if we have a feasible step).
    export_dir: Optional[Path] = None
    remeasure: Dict[str, Any] = {}
    infeasible_on_export = False
    exported_j_drop_val: Optional[float] = None
    exported_j_drop_details: Dict[str, Any] = {}
    if not result.infeasible:
        if out_root is None:
            out_root = Path(".") / "vadi_runs"
        export_dir = out_root / clip_name / config_name / "processed"
        delta_on_device = result.delta_star.to(x_clean.device)
        nu_on_device = result.nu_star.to(x_clean.device)
        export_processed_uint8(
            x_clean, delta_on_device, nu_on_device, base_inserts,
            W_attacked, export_dir,
        )
        exported = load_processed_uint8(export_dir).to(x_clean.device)
        remeasure = remeasure_exported_feasibility(
            x_clean, base_inserts, exported, W_attacked,
            lpips_fn, ssim_fn, config,
        )
        # HARD gate: exported-artifact infeasibility flips the top-level
        # `infeasible` flag. The primary denominator in the 10-clip table
        # counts this as a failure (spec F16), regardless of internal PGD
        # success.
        if not remeasure.get("step_feasible_on_export", False):
            infeasible_on_export = True

        # Stage 8.5: exported-artifact SAM2 re-eval — the paper-claim
        # J-drop metric (codex R1 Fix 2, 2026-04-23). Only runs if
        # caller supplied `sam2_eval_fn`; otherwise the field stays None
        # and downstream consumers (pilot gate, main table) must treat
        # this clip as "metric not measured" rather than "metric zero".
        # NOTE: we run the eval even on exported-infeasible clips so the
        # main-table denominator sees the number; infeasibility is a
        # separate failure mode tracked by `infeasible`.
        if sam2_eval_fn is not None:
            exported_j_drop_details = eval_exported_j_drop(
                sam2_eval_fn=sam2_eval_fn,
                prompt_mask=prompt_mask,
                x_clean=x_clean,
                base_inserts=base_inserts,
                exported=exported,
                W=W_attacked,
                m_hat_true_by_t=m_hat_true_by_t_t,
            )
            exported_j_drop_val = float(
                exported_j_drop_details["J_drop_mean"])

    # Stage 9: build output. Strip snapshot tensors from step logs.
    summary_logs: List[Dict[str, Any]] = []
    for log in result.step_logs:
        d = asdict(log)
        d.pop("delta_snapshot", None)
        d.pop("nu_snapshot", None)
        summary_logs.append(d)

    out = VADIClipOutput(
        clip_name=clip_name,
        config_name=config_name,
        W=list(W_attacked),
        infeasible=result.infeasible or infeasible_on_export,
        best_surrogate_J_drop=float(result.best_surrogate_J_drop),
        exported_j_drop=exported_j_drop_val,
        exported_j_drop_details=exported_j_drop_details,
        export_dir=str(export_dir) if export_dir else None,
        vulnerability_scores=vs.to_dict(),
        exported_feasibility=remeasure,
        step_log_summary=summary_logs,
        decoy_offset=decoy_offset,
        placement_source=placement_source,
    )

    # Persist results.json next to the export dir.
    if export_dir is not None:
        results_json = export_dir.parent / "results.json"
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(asdict(out), f, indent=2, default=str)

    return out


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VADI: vulnerability-aware decoy insertion attack on SAM2.")
    p.add_argument("--clip", required=True, help="DAVIS clip name (e.g. dog).")
    p.add_argument("--davis-root", required=True,
                   help="Path to DAVIS root (JPEGImages/480p inside).")
    p.add_argument("--checkpoint", required=True,
                   help="SAM2.1 tiny checkpoint path.")
    p.add_argument("--out-root", default="vadi_runs",
                   help="Directory under which results are written.")
    p.add_argument("--mode", choices=["top", "random", "bottom"], default="top",
                   help="Vulnerability placement mode.")
    p.add_argument("--K", type=int, default=1, help="K_ins ∈ {1, 2, 3}.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-name", default=None,
                   help="Output sub-directory label; defaults to K{K}_{mode}.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse args + build pipeline but do not run SAM2/PGD.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    config_name = args.config_name or f"K{args.K}_{args.mode}"

    if args.dry_run:
        print(f"[vadi] dry-run: clip={args.clip} config={config_name} "
              f"mode={args.mode} K={args.K} seed={args.seed}")
        return 0

    # Lazy imports for the real pipeline so `--dry-run` and `--help` don't
    # require torch-gpu / lpips / sam2 to be installed on a doc-only host.
    print("[vadi] real run not wired to this import-light CLI by design; "
          "use scripts/run_vadi_pilot.py or run_vadi_davis10.py as the "
          "entrypoints that instantiate SAM2VideoAdapter + LPIPS + SSIM.",
          file=sys.stderr)
    return 2


# =============================================================================
# Self-tests (stub adapters — no SAM2 / no LPIPS installation required)
# =============================================================================


def _self_test() -> None:
    import tempfile
    torch.manual_seed(0)
    np.random.seed(0)

    # -- compute_decoy_offset + shift_mask round-trip
    H, W = 16, 16
    m0 = np.zeros((H, W), dtype=np.float32); m0[4:8, 4:8] = 1.0
    frame0 = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    dy, dx = compute_decoy_offset(m0, frame0)
    assert isinstance(dy, int) and isinstance(dx, int)
    # Zero-mask edge case.
    empty = np.zeros((H, W), dtype=np.float32)
    assert compute_decoy_offset(empty, frame0) == (0, 0)

    # -- build_decoy_trajectory: shape + basic sanity
    masks = [m0.copy() for _ in range(5)]
    decoy = build_decoy_trajectory(masks, (3, 2))
    assert len(decoy) == 5
    assert decoy[0].shape == m0.shape
    # Content shifted: decoy's occupied rows shifted by dy=3.
    assert decoy[0][7:11, 6:10].sum() > 0                    # shifted region

    # -- remap_masks_to_processed_space
    T_clean = 5
    W_att = [2, 5]                                            # K=2 → T_proc=7
    clean_masks = [np.full((H, W), float(i)) for i in range(T_clean)]  # synthetic
    rp = remap_masks_to_processed_space(clean_masks, W_att)
    assert set(rp.keys()) == set(range(T_clean + 2))
    # attacked_to_clean(3) = 2, so rp[3] == clean_masks[2].
    assert np.allclose(rp[3], clean_masks[2])
    # Insert positions get midframe average.
    # For W=[2,5]: c_0 = 2, c_1 = 4. rp[2] = 0.5*clean[1] + 0.5*clean[2].
    assert np.allclose(rp[2], 0.5 * clean_masks[1] + 0.5 * clean_masks[2])
    assert np.allclose(rp[5], 0.5 * clean_masks[3] + 0.5 * clean_masks[4])

    # -- Export + reload uint8 round-trip.
    torch.manual_seed(1)
    T_clean, Hv, Wv = 4, 8, 8
    x_clean = torch.rand(T_clean, Hv, Wv, 3)
    W_att = [2]
    delta_star = torch.zeros_like(x_clean)
    # Put a small nonzero δ on a non-f0 frame to verify round-trip.
    delta_star[1] = 3.0 / 255.0
    base_ins = build_base_inserts(x_clean, W_att)
    nu_star = torch.zeros_like(base_ins)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "proc"
        paths = export_processed_uint8(
            x_clean, delta_star, nu_star, base_ins, W_att, out)
        assert len(paths) == T_clean + 1
        reloaded = load_processed_uint8(out)
        # Max pixel error should be bounded by 1/255 + f.p. epsilon.
        processed_float = build_processed(
            (x_clean + delta_star).clamp(0, 1),
            (base_ins + nu_star).clamp(0, 1),
            W_att)
        err = (reloaded - processed_float).abs().max().item()
        assert err < 2.0 / 255.0, \
            f"export/reload diverged by {err} (expected < 2/255)"

    # -- End-to-end via stub adapters --------------------------------------
    # Stub 1: clean_pass_fn returns fake pseudo-masks + conf + Hiera.
    def stub_clean_pass(x_clean_t: Tensor, prompt: np.ndarray) -> CleanPassOutput:
        T = x_clean_t.shape[0]
        Hv, Wv = x_clean_t.shape[1], x_clean_t.shape[2]
        # True mask follows a slow drift — synthetic but plausible.
        pseudo = []
        for t in range(T):
            m = np.zeros((Hv, Wv), dtype=np.float32)
            m[Hv // 2:, :Wv // 2] = 1.0                      # object in lower-left
            pseudo.append(m)
        conf = np.linspace(0.9, 0.3, T).astype(np.float32)
        feats = [np.random.randn(32).astype(np.float32) for _ in range(T)]
        return CleanPassOutput(
            pseudo_masks=pseudo, confidences=conf, hiera_features=feats)

    # Stub 2: forward_fn that's differentiable through x_processed.
    def stub_forward_builder(x_clean, prompt_mask, W):
        def fn(processed: Tensor, return_at: Iterable[int]) -> Dict[int, Tensor]:
            out = {}
            for t in return_at:
                gray = processed[t].mean(dim=-1)              # [H, W]
                out[t] = 3.0 * (gray - 0.5)
            return out
        return fn

    # Stub 3/4: LPIPS / SSIM proxies.
    def lpips_stub(x, y):
        return (x - y).abs().mean()

    def ssim_stub(x, y):
        return 1.0 - (x - y).pow(2).mean()

    cfg = VADIConfig(N_1=2, N_2=3, N_3=2,
                     lambda_init=1.0, lambda_growth_factor=2.0,
                     lambda_growth_period=2)
    prompt_np = np.zeros((Hv, Wv), dtype=np.uint8)
    prompt_np[Hv // 2:, :Wv // 2] = 1

    with tempfile.TemporaryDirectory() as td:
        result = run_vadi_for_clip(
            clip_name="stub", config_name="K1_top",
            x_clean=x_clean, prompt_mask=prompt_np,
            clean_pass_fn=stub_clean_pass,
            forward_fn_builder=stub_forward_builder,
            lpips_fn=lpips_stub, ssim_fn=ssim_stub,
            vulnerability_mode="top", K_ins=1,
            config=cfg, out_root=Path(td),
        )
        assert isinstance(result, VADIClipOutput)
        assert result.clip_name == "stub"
        assert len(result.W) == 1
        # T_clean=4, K=1 → T_proc=5; vulnerability scorer picks from
        # clean m=1..3; after shift (K=1 sort shift 0), W ∈ [1, 3].
        assert 1 <= result.W[0] <= 3
        # At least one feasibility field populated.
        if not result.infeasible:
            assert "step_feasible_on_export" in result.exported_feasibility
            assert result.export_dir is not None
            assert Path(result.export_dir).exists()
            # Sibling results.json persisted.
            assert (Path(result.export_dir).parent / "results.json").exists()
        # step_log_summary stripped of tensors (JSON-serializable).
        json.dumps(result.step_log_summary, default=str)

    # -- K=3 with time-varying masks: exercises the `+k` shift and the
    # non-adjacent placement logic for a non-degenerate pick.
    T_clean_big = 12
    x_big = torch.rand(T_clean_big, Hv, Wv, 3)

    def stub_clean_pass_tv(x_clean_t: Tensor, prompt: np.ndarray) -> CleanPassOutput:
        T = x_clean_t.shape[0]
        Hv_, Wv_ = x_clean_t.shape[1], x_clean_t.shape[2]
        # Time-varying masks: object moves diagonally each frame (creates
        # nonzero r_mask signal, breaks rank ties).
        pseudo = []
        for t in range(T):
            m = np.zeros((Hv_, Wv_), dtype=np.float32)
            y0 = min(t, Hv_ - 4)
            x0 = min(t, Wv_ - 4)
            m[y0:y0 + 4, x0:x0 + 4] = 1.0
            pseudo.append(m)
        # Confidences with a sharp drop midway (creates r_conf peak).
        conf = np.where(np.arange(T) < T // 2, 0.9, 0.3).astype(np.float32)
        # Hiera-like features with a jump at the same point.
        feats = []
        for t in range(T):
            v = np.full(32, float(t), dtype=np.float32)
            if t >= T // 2:
                v += 10.0
            feats.append(v)
        return CleanPassOutput(
            pseudo_masks=pseudo, confidences=conf, hiera_features=feats)

    with tempfile.TemporaryDirectory() as td:
        result_k3 = run_vadi_for_clip(
            clip_name="stub3", config_name="K3_top",
            x_clean=x_big, prompt_mask=prompt_np,
            clean_pass_fn=stub_clean_pass_tv,
            forward_fn_builder=stub_forward_builder,
            lpips_fn=lpips_stub, ssim_fn=ssim_stub,
            vulnerability_mode="top", K_ins=3, min_gap=2,
            config=cfg, out_root=Path(td),
        )
        assert len(result_k3.W) == 3
        # Non-adjacent in attacked-space: pairwise |W_i - W_j| ≥ 2.
        for i in range(len(result_k3.W)):
            for j in range(i + 1, len(result_k3.W)):
                assert abs(result_k3.W[i] - result_k3.W[j]) >= 2

        # Test W_clean_override path: bypass scorer with explicit W_clean=[2,6,10].
        result_override = run_vadi_for_clip(
            clip_name="stub3_override", config_name="K3_override",
            x_clean=x_big, prompt_mask=prompt_np,
            clean_pass_fn=stub_clean_pass_tv,
            forward_fn_builder=stub_forward_builder,
            lpips_fn=lpips_stub, ssim_fn=ssim_stub,
            vulnerability_mode="top", K_ins=3, min_gap=2,
            config=cfg, out_root=Path(td),
            W_clean_override=[2, 6, 10],
        )
        # Verify attacked-space shift: c=[2,6,10] → W=[2, 7, 12].
        assert result_override.W == [2, 7, 12], \
            f"W_clean_override shift incorrect: got {result_override.W}"
        assert result_override.placement_source == "override"
        assert result_k3.placement_source == "scorer"

        # Override validation: mismatched K / duplicates / out-of-range / gap.
        for bad, expected in [
            ([2, 6], "wrong length"),
            ([2, 2, 6], "duplicates"),
            ([0, 6, 10], "out of range (0 reserved for f0)"),
            ([2, 3, 6], "min_gap violation"),
        ]:
            try:
                run_vadi_for_clip(
                    clip_name="x", config_name="x",
                    x_clean=x_big, prompt_mask=prompt_np,
                    clean_pass_fn=stub_clean_pass_tv,
                    forward_fn_builder=stub_forward_builder,
                    lpips_fn=lpips_stub, ssim_fn=ssim_stub,
                    K_ins=3, config=cfg, out_root=Path(td),
                    W_clean_override=bad,
                )
                raise AssertionError(
                    f"W_clean_override={bad} must raise ({expected})")
            except ValueError:
                pass

    # -- CLI argparser --------------------------------------------------------
    args = build_argparser().parse_args([
        "--clip", "dog", "--davis-root", "/tmp",
        "--checkpoint", "/tmp/ckpt.pt", "--dry-run",
    ])
    assert args.clip == "dog"
    assert args.dry_run is True
    # main() with --dry-run exits 0.
    assert main([
        "--clip", "dog", "--davis-root", "/tmp",
        "--checkpoint", "/tmp/ckpt.pt", "--dry-run",
    ]) == 0

    print("scripts.run_vadi: all self-tests PASSED "
          "(decoy offset, mask remap, export/reload round-trip, "
          "end-to-end stub pipeline, CLI --dry-run)")


if __name__ == "__main__":
    # If invoked with any CLI args → run main(); else → self-test.
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
