#!/usr/bin/env python3
"""
Two Memory-Poisoning Regimes: Unified Experiment Runner.

One shared PGD loop, pluggable loss. Only the loss function and insert base
differ between regimes. Everything else is strictly matched:
  - Same frame schedule (insertions after f1, f7)
  - Same perturbation bounds (f0: 2/255, attacked originals: 4/255, inserts: 8/2/255)
  - Same 3-stage PGD (20% insert-only -> 60% joint -> 20% stabilize)
  - Same surrogate (track_step-based)
  - Same fake uint8 quantization
  - Same eval window (f10:f14, disjoint from attack)
  - Same DAVIS prompt extraction

Fixes from code review:
  1. CRITICAL: Both regimes share one PGD loop (no separate implementations)
  2. CRITICAL: Signature extraction uses surrogate object_score_logits + DecoyHitRate/CentroidShift
  3. CRITICAL: Boundary F-measure uses DAVIS-standard seg2bmap + distance_transform_edt
  4. CRITICAL: Per-object mask loading (primary object, not union foreground)

Design choice: Evaluation writes JPEG Q100 (near-lossless) — this tests transport
robustness. PGD uses fake uint8 quantization which covers the dominant quantization
effect. If needed, add JPEG simulation to PGD loop for full codec awareness.

Usage:
  # Block 1: Core comparison (20 videos)
  python run_two_regimes.py --block core --device cuda:0

  # Block 2: Mechanism isolation (5 pilot videos)
  python run_two_regimes.py --block isolation --device cuda:0

  # Single regime test
  python run_two_regimes.py --regime suppression --videos bear,dog --device cuda:0
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from memshield.config import MemShieldConfig
from memshield.decoy import shift_mask
from memshield.generator import (
    build_role_targets, build_synthetic_decoy_video, select_perturb_originals,
)
from memshield.losses import (
    anti_anchor_loss,
    decoy_target_loss,
    differentiable_ssim,
    fake_uint8_quantize,
    mean_logit_loss,
    memory_teacher_loss,
    obj_ptr_teacher_loss,
    object_score_margin_loss,
    object_score_positive_loss,
)
from memshield.scheduler import (
    InsertionSlot,
    build_modified_index_map,
    compute_resonance_schedule,
)
from memshield.surrogate import SAM2Surrogate, get_interior_prompt


# ── Constants ────────────────────────────────────────────────────────────────

# Official DAVIS 2017 validation set (30 videos)
DAVIS_VAL = [
    "bike-packing", "blackswan", "bmx-trees", "breakdance", "camel",
    "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog",
    "dogs-jump", "drift-chicane", "drift-straight", "goat", "gold-fish",
    "horsejump-high", "india", "judo", "kite-surf", "lab-coat",
    "libby", "loading", "mbike-trick", "motocross-jump",
    "paragliding-launch", "parkour", "pigs", "scooter-black",
    "shooting", "soapbox",
]

# Legacy alias (kept for backwards compat with existing results)
DAVIS_20 = DAVIS_VAL

DAVIS_PILOT = ["blackswan", "car-shadow", "dog", "cows", "gold-fish"]

ATTACK_PREFIX_SUPP = 15   # Suppression: 15-frame prefix (already persistent)
ATTACK_PREFIX_DECOY = 15  # Decoy: same 15f prefix, but cover_prefix adds 3rd insert at f13
EVAL_START = 10           # Disjoint eval window start (inclusive)
EVAL_END = 15             # Legacy: short-horizon eval end (for backwards compat)
# Round-1 universal redesign: extend PGD supervision horizon to cover post-prefix
# clean frames (attacks FIFO self-healing directly via memory-attention gradient).
# `min(T, EVAL_START + EVAL_HORIZON)` is used in optimize_unified.
EVAL_HORIZON = 7          # 7-frame supervision window (f10..f16) — covers 2
                          # frames past 15-frame attack prefix without blowing up
                          # rollout memory. Bump to 10 if VRAM allows.


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1: Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_video(davis_root: str, vid: str, max_frames: int = 0,
               target_obj: Optional[int] = None):
    """Load DAVIS 2017 video frames and per-object annotation masks.

    Args:
        max_frames: 0 = load all frames (default). >0 = truncate.

    DAVIS 2017 has multi-object sequences. We select a single target object
    for evaluation (default: smallest positive label = primary object).
    """
    img_dir = Path(davis_root) / "JPEGImages" / "480p" / vid
    anno_dir = Path(davis_root) / "Annotations" / "480p" / vid
    stems = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if max_frames > 0:
        stems = stems[:max_frames]
    frames, masks = [], []
    for i, stem in enumerate(stems):
        frames.append(np.array(Image.open(img_dir / f"{stem}.jpg").convert("RGB")))
        anno = np.array(Image.open(anno_dir / f"{stem}.png"))
        # On first frame, auto-detect primary object if not specified
        if i == 0 and target_obj is None:
            obj_ids = sorted(set(anno.flat) - {0})
            target_obj = obj_ids[0] if obj_ids else 1
        masks.append((anno == target_obj).astype(np.uint8))
    return frames, masks


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2: Insert Base Construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_suppression_bases(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
) -> Dict[int, np.ndarray]:
    """Suppression bases: inpainted frames with object region removed.

    Gives the optimizer a head start at convincing SAM2 the object is absent.
    """
    T = len(frames_uint8)
    bases = {}
    for si, slot in enumerate(schedule):
        pos = slot.after_original_idx
        frame_after = frames_uint8[min(pos + 1, T - 1)]
        mask_after = masks_uint8[min(pos + 1, T - 1)]
        mask_bin = (mask_after > 0).astype(np.uint8)
        if mask_bin.sum() == 0:
            bases[si] = frame_after.copy()
            continue
        inpaint_mask = (mask_bin * 255).astype(np.uint8)
        bgr = cv2.cvtColor(frame_after, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
        bases[si] = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return bases


def _build_decoy_bases_and_targets(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
    perturb_set: Set[int],
    device: torch.device,
    surrogate=None,
    high_fidelity_insert: bool = False,
) -> Tuple[Dict[int, np.ndarray], dict, Tuple[int, int], Optional[List[dict]]]:
    """Decoy bases (relocated object) + role targets + teacher trajectory.

    When surrogate is provided, generates a synthetic decoy video and runs
    SAM2 on it to get teacher memory features. These features represent
    "what SAM2's memory looks like when tracking the object at the decoy
    location" — the target for memory-level cooperation loss.
    """
    role_data = build_role_targets(
        masks_uint8, frames_uint8, schedule, perturb_set, device,
        high_fidelity_insert=high_fidelity_insert,
    )
    offset = role_data["decoy_offset"]

    teacher_features = None
    if surrogate is not None:
        # Build teacher on MODIFIED timeline (with inserts at same positions)
        synth_frames, decoy_mask_f0, teacher_mod_to_orig = \
            build_synthetic_decoy_video(
                frames_uint8, masks_uint8, offset, schedule=schedule)
        if decoy_mask_f0.sum() > 50:
            teacher_features = surrogate.generate_teacher_trajectory(
                synth_frames, decoy_mask_f0)
            print(f"    [teacher] Generated {len(teacher_features)} features "
                  f"on modified timeline ({len(synth_frames)} frames, "
                  f"offset=({offset[0]},{offset[1]}))")
        else:
            print(f"    [teacher] Decoy mask too small at f0, skipping teacher")

    return role_data["insert_bases"], role_data, offset, teacher_features


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3: Regime-Specific Loss Functions
# ══════════════════════════════════════════════════════════════════════════════

def _to_gt(mask_np: np.ndarray, spatial_size: Tuple[int, int],
           device: torch.device) -> torch.Tensor:
    """[H,W] uint8 mask -> [1,1,Hs,Ws] float tensor, resized if needed."""
    gt = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if gt.shape[-2:] != spatial_size:
        gt = F.interpolate(gt, size=spatial_size, mode="nearest")
    return gt


def _resize_td(td: dict, sz: Tuple[int, int]) -> dict:
    """Resize spatial masks in a decoy target_dict to match logits size."""
    out = dict(td)
    for k in ("core", "bridge", "decoy", "suppress"):
        if k in out and out[k].shape[-2:] != sz:
            out[k] = F.interpolate(out[k], size=sz, mode="nearest")
    return out


# ── Suppression losses ───────────────────────────────────────────────────────

def _supp_write(all_outs, masks_uint8, idx_map, perturb_set, schedule, device,
                obj_margin=2.0):
    """Suppression write-path: push attacked frames toward 'object absent'.

    Weight structure:
      f0: 0.5 (light touch on conditioning)
      post-insert originals: 1.5 (reinforce poisoned memory)
      inserts: 1.5x multiplier (matches decoy)
      other: 1.0
    """
    T = len(masks_uint8)
    loss = torch.tensor(0.0, device=device)
    n = 0

    for oi in sorted(perturb_set):
        mod_idx = idx_map["orig_to_mod"][oi]
        if mod_idx >= len(all_outs):
            continue
        out = all_outs[mod_idx]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        gt = _to_gt(masks_uint8[min(oi, T - 1)], logits.shape[-2:], device)
        fl = mean_logit_loss(logits, gt)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + object_score_margin_loss(sc, margin=obj_margin)
        # Role-based weighting
        is_post = any(oi in (s.after_original_idx + 1, s.after_original_idx + 2)
                      for s in schedule)
        w = 0.5 if oi == 0 else (1.5 if is_post else 1.0)
        loss = loss + w * fl
        n += 1

    # Loss on inserted frames
    ins_indices = idx_map["insert_mod_indices"]
    for cursor in range(len(schedule)):
        if cursor >= len(ins_indices):
            break
        mod_idx = ins_indices[cursor]
        if mod_idx >= len(all_outs):
            continue
        out = all_outs[mod_idx]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        pos = schedule[cursor].after_original_idx
        gt = _to_gt(masks_uint8[min(pos + 1, T - 1)], logits.shape[-2:], device)
        fl = mean_logit_loss(logits, gt)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + object_score_margin_loss(sc, margin=obj_margin)
        loss = loss + 1.5 * fl
        n += 1

    return loss / max(n, 1)


def _supp_read(all_outs, masks_uint8, eval_mod, eval_orig, device,
               obj_margin=2.0):
    """Suppression read-path: verify eval frames have suppressed predictions."""
    loss = torch.tensor(0.0, device=device)
    n = 0
    for rank, (mi, oi) in enumerate(zip(eval_mod, eval_orig)):
        if mi >= len(all_outs) or oi >= len(masks_uint8):
            break
        out = all_outs[mi]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        gt = _to_gt(masks_uint8[oi], logits.shape[-2:], device)
        w = max(0.5, 2.5 - rank * 0.3)  # Front-loaded weighting
        fl = mean_logit_loss(logits, gt)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + object_score_margin_loss(sc, margin=obj_margin)
        loss = loss + w * fl
        n += 1
    return loss / max(n, 1)


# ── Decoy losses (v4 + round-1 universal redesign) ─────────────────────────
#
# Design principle: perturbations and inserts have DIFFERENT attack roles.
#   Perturbations: SUPPRESS — weaken true-location memories (independently useful)
#   Inserts: REDIRECT — write confident decoy memories (independently useful)
#   Combined: suppression creates vulnerability, decoy fills the gap (synergy)
#
# Round-1 redesign:
#   - Split true support into `core` (eroded GT) and `ring` (annulus inside GT)
#     to prevent the "smear across GT" cheat where full-region mean rank is
#     satisfied by diffuse activation over the whole true object.
#   - Top-k mean rank (20%) instead of full-region mean: forces sharp decoy
#     peak above sharp ring peak, not just average elevation.


def _masked_pos(logits, mask, margin):
    """Hinge penalty: push logits in masked region UP above `margin`.

    Uses softplus(margin - logits*mask), normalized by mask area.
    """
    s = mask.sum()
    if s.item() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.softplus(margin - logits * mask).sum() / (s + 1e-6)


def _masked_neg(logits, mask, margin=0.0):
    """Hinge penalty: push logits in masked region DOWN below `-margin`.

    Uses softplus(logits*mask + margin), normalized by mask area.
    """
    s = mask.sum()
    if s.item() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.softplus(logits * mask + margin).sum() / (s + 1e-6)


def _topk_mean(logits, mask, k_frac=0.2):
    """Top-k mean of logits inside `mask` (binary). k = k_frac * region size.

    Forces rank/margin terms to compare SHARP peaks, not diffuse averages.

    DEPRECATED for rank loss: hard top-k makes the active support jump
    across texture patches and PGD steps, causing "sharp but brittle" peaks
    that don't persist under memory refresh. Use `_soft_cvar_mean` instead.
    Retained here for A/B comparison and legacy call sites.
    """
    bmask = mask > 0.5
    values = logits[bmask]
    if values.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    k = max(1, int(values.numel() * k_frac))
    topk = values.topk(k).values
    return topk.mean()


def _soft_cvar_mean(logits, mask, q=0.5, tau=None):
    """Soft CVaR: smooth mean of logits above the q-quantile inside `mask`.

    Round-2 fix for hard top-k brittleness (Stage 2 regression on cows/dog).
    Uses a sigmoid gate centred on the DETACHED q-quantile so gradient flows
    through values only, not through the quantile selector. As q→1 this
    approaches max; q=0.5 gives mean-of-top-half (CVaR_0.5), which is a
    softer peak estimator than top-20% mean.

    Args:
        q: quantile threshold in [0,1]. Default 0.5 = mean of top 50%.
        tau: sigmoid temperature. If None, auto-scale via in-mask stdev
             (clamped ≥0.05) so the gate width tracks the logit range.
    """
    bmask = mask > 0.5
    values = logits[bmask]
    if values.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    with torch.no_grad():
        qv = torch.quantile(values.detach().float(), q).to(values.dtype)
        if tau is None:
            tau_val = (values.detach().std() * 0.5).clamp(min=0.05)
        else:
            tau_val = torch.tensor(tau, device=values.device, dtype=values.dtype)
    weights = torch.sigmoid((values - qv) / tau_val)
    return (values * weights).sum() / (weights.sum() + 1e-6)

def _decoy_write(all_outs, targets, idx_map, perturb_set, schedule, device,
                 masks_uint8=None, insert_strength=1.0,
                 teacher_features=None, clean_features=None):
    """Decoy write-path: suppress originals + redirect inserts.

    Originals: softened suppression (reduce true-region logits, band object_score
    near zero — weak enough for decoy to dominate, not so strong as to kill object).

    Inserts: decoy activation with configurable strength.
      insert_strength=1.0: v4 baseline (margin=0.9, gentle)
      insert_strength=2.0: v4.1 aggressive (margin=2.0, may collapse)
    """
    loss_orig = torch.tensor(0.0, device=device)
    loss_ins = torch.tensor(0.0, device=device)
    n_orig, n_ins = 0, 0
    T_masks = len(masks_uint8) if masks_uint8 is not None else 1

    # Band parameters for object_score on originals
    z_lo, z_hi = -1.0, 0.5  # Keep score in [-1, 0.5] — weak but not dead

    for oi in sorted(perturb_set):
        mi = idx_map["orig_to_mod"][oi]
        if mi >= len(all_outs):
            continue
        out = all_outs[mi]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue

        # Get suppress mask from targets, or fallback to GT mask
        key = ("orig", oi)
        suppress_mask = None
        if key in targets:
            td = _resize_td(targets[key], logits.shape[-2:])
            suppress_mask = td.get("suppress")
        if (suppress_mask is None or suppress_mask.sum() == 0) and masks_uint8 is not None:
            gt = _to_gt(masks_uint8[min(oi, T_masks - 1)], logits.shape[-2:], device)
            suppress_mask = gt

        # === SOFTENED SUPPRESSION on originals ===
        # Push down logits in true region (independently useful)
        if suppress_mask is not None and suppress_mask.sum() > 0:
            fl = F.softplus(logits * suppress_mask).sum() / (suppress_mask.sum() + 1e-6)
        else:
            fl = torch.tensor(0.0, device=device)

        # Object score band: keep in [z_lo, z_hi], not deeply negative
        sc = out.get("object_score_logits")
        if sc is not None:
            # Push toward the band — penalize if too positive OR too negative
            fl = fl + 0.5 * F.relu(sc - z_hi) + 0.3 * F.relu(z_lo - sc)

        # f0 gets lighter touch (privileged conditioning memory)
        w = 0.5 if oi == 0 else 1.0
        loss_orig = loss_orig + w * fl
        n_orig += 1

    # === DECOY ACTIVATION on inserted frames (round-1 annulus + top-k rank) ===
    # Split true support into `core` (eroded GT, deep interior) and `ring`
    # (annulus inside GT outside core, excluding decoy overlap). Ring is pushed
    # down harder than core to prevent the "smear across GT" solution.
    # Top-k mean (20%) for rank comparison forces sharp decoy peak above
    # sharp ring peak.
    s = insert_strength
    decoy_m = 0.9 + (s - 1.0) * 1.1     # 0.9 → 2.0
    rank_m = 0.8 + (s - 1.0) * 1.2       # 0.8 → 2.0
    score_m = 1.0 + (s - 1.0) * 1.0      # 1.0 → 2.0
    bridge_m = 0.2                        # light pressure, keeps region contiguous
    ring_m = 0.5                          # stronger than core push-down
    core_m = 0.5

    # Weights (round-2 tuning; reviewer-prescribed soft-CVaR rank).
    # w_ring lowered 1.5 → 1.0 to reduce "over-sharpening" pressure that
    # combined with hard top-k caused Stage-2 regression on cows/dog.
    w_decoy_base = 1.0
    w_ring_base = 1.0
    w_core_base = 0.5
    w_bridge_base = 0.05
    w_rank_base = 1.0
    w_score_base = 0.4
    # Round-3 teacher memory cooperation (v3 resurrected on modified timeline).
    # Adds memory-level supervision on inserts + anti-anchor on originals so
    # SAM2 writes decoy memories and simultaneously weakens the true anchor.
    # Active only when teacher_features is provided via --use_teacher.
    w_teacher_mem = 0.6
    w_teacher_ptr = 0.2
    w_anti_anchor = 0.3
    # Scale with insert_strength but gently
    w_decoy = w_decoy_base * s
    w_ring = w_ring_base * (0.5 + 0.5 * s)
    w_core = w_core_base
    w_bridge = w_bridge_base
    w_rank = w_rank_base * s
    w_score = w_score_base * s

    ins_indices = idx_map["insert_mod_indices"]
    for cursor in range(len(schedule)):
        key = ("insert", cursor)
        if key not in targets or cursor >= len(ins_indices):
            continue
        mi = ins_indices[cursor]
        if mi >= len(all_outs):
            continue
        out = all_outs[mi]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        td = _resize_td(targets[key], logits.shape[-2:])

        core_mask = td.get("core")
        bridge_mask = td.get("bridge")
        decoy_mask = td.get("decoy")
        suppress_mask = td.get("suppress")
        fl = torch.tensor(0.0, device=device)

        # Compute ring = suppress & ~core (annulus inside GT, excluding core).
        # suppress = GT\decoy, core = eroded GT \ decoy, so ring = GT \ core \ decoy.
        if core_mask is not None and suppress_mask is not None:
            ring_mask = suppress_mask * (1.0 - core_mask)
        else:
            ring_mask = suppress_mask

        # Push UP at decoy region
        if decoy_mask is not None:
            fl = fl + w_decoy * _masked_pos(logits, decoy_mask, decoy_m)

        # Push UP at bridge (light — keeps decoy region visually connected)
        if bridge_mask is not None:
            fl = fl + w_bridge * _masked_pos(logits, bridge_mask, bridge_m)

        # Push DOWN at ring (hard — prevents smear into GT annulus)
        if ring_mask is not None:
            fl = fl + w_ring * _masked_neg(logits, ring_mask, ring_m)

        # Push DOWN at core (mild — core already deep interior, weaker pressure)
        if core_mask is not None:
            fl = fl + w_core * _masked_neg(logits, core_mask, core_m)

        # Rank: soft CVaR_0.5 (mean of top 50%) of decoy logits must beat
        # CVaR_0.5 of ring logits + margin. Replaces hard top-20% (Round 1).
        # Soft aggregator keeps the active support stable across PGD steps,
        # giving persistent decoy mass instead of brittle spiky peaks.
        if (decoy_mask is not None and ring_mask is not None
                and decoy_mask.sum() > 0 and ring_mask.sum() > 0):
            ring_agg = _soft_cvar_mean(logits, ring_mask, q=0.5)
            decoy_agg = _soft_cvar_mean(logits, decoy_mask, q=0.5)
            fl = fl + w_rank * F.softplus(ring_agg - decoy_agg + rank_m)

        # Object score: positive (confident wrong, not absent)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + w_score * object_score_positive_loss(sc, margin=score_m)

        # Round-3: teacher memory cooperation on inserts (v3 resurrected).
        # Pulls maskmem_features + obj_ptr toward the synthetic-teacher-video
        # targets generated by surrogate.generate_teacher_trajectory(). This
        # is the memory-level analog of the output-level annulus/CVaR terms:
        # we teach SAM2's FIFO bank to store what SAM2 WOULD HAVE STORED if
        # it were tracking the object at the decoy position.
        if teacher_features is not None and mi < len(teacher_features):
            tf = teacher_features[mi]
            adv_mem = out.get("maskmem_features")
            adv_ptr = out.get("obj_ptr")
            t_mem = tf.get("maskmem_features") if isinstance(tf, dict) else None
            t_ptr = tf.get("obj_ptr") if isinstance(tf, dict) else None
            if adv_mem is not None and t_mem is not None:
                fl = fl + w_teacher_mem * memory_teacher_loss(adv_mem, t_mem)
            if adv_ptr is not None and t_ptr is not None:
                fl = fl + w_teacher_ptr * obj_ptr_teacher_loss(adv_ptr, t_ptr)

        loss_ins = loss_ins + fl
        n_ins += 1

    # Normalize and combine
    if n_orig > 0:
        loss_orig = loss_orig / n_orig
    if n_ins > 0:
        loss_ins = loss_ins / n_ins
    return loss_orig + 1.5 * loss_ins


def _decoy_read(all_outs, masks_uint8, eval_mod, eval_orig, decoy_offset,
                device, teacher_features=None):
    """Decoy read-path: decoy-vs-true margin on rollout frames.

    On clean future frames, require:
      1. object_score > 0 (object still "present" — not suppression)
      2. logit(decoy) > logit(true) + margin (tracking at wrong location)

    This gives gradient signal back through the memory bank to both
    perturbation and insert parameters.
    """
    dy, dx = decoy_offset
    T = len(masks_uint8)
    H, W = masks_uint8[0].shape
    loss = torch.tensor(0.0, device=device)
    n = 0

    for rank, (mi, oi) in enumerate(zip(eval_mod, eval_orig)):
        if mi >= len(all_outs):
            break
        out = all_outs[mi]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        mask = masks_uint8[min(oi, T - 1)]
        if mask.sum() == 0:
            continue

        d_mask = shift_mask(mask, dy, dx)
        true_region = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        decoy_region = torch.from_numpy(((d_mask > 0) & (mask == 0)).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        if true_region.shape[-2:] != logits.shape[-2:]:
            true_region = F.interpolate(true_region, size=logits.shape[-2:], mode="nearest")
            decoy_region = F.interpolate(decoy_region, size=logits.shape[-2:], mode="nearest")

        fl = torch.tensor(0.0, device=device)

        # Decoy-vs-true margin: decoy logits must exceed true logits
        if true_region.sum() > 0 and decoy_region.sum() > 0:
            true_mean = (logits * true_region).sum() / (true_region.sum() + 1e-6)
            decoy_mean = (logits * decoy_region).sum() / (decoy_region.sum() + 1e-6)
            margin = max(0.5, 1.0 - rank * 0.05)  # Decay margin with distance
            fl = fl + F.softplus(true_mean - decoy_mean + margin)

        # Positive objectness: object must still be "present" (not suppression)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + 0.4 * object_score_positive_loss(sc, margin=0.5)

        # Round-3: teacher memory match on FIRST FEW post-insert clean frames
        # (rank < 3). After an insert poisons the FIFO, the first 2-3 clean
        # reads should also write decoy-like memories. This is what makes
        # the poison "live" in the memory bank beyond the attack prefix.
        if teacher_features is not None and rank < 3 and mi < len(teacher_features):
            tf = teacher_features[mi]
            adv_mem = out.get("maskmem_features")
            t_mem = tf.get("maskmem_features") if isinstance(tf, dict) else None
            if adv_mem is not None and t_mem is not None:
                # Lighter weight than write-path: 0.3 (vs 0.6 on inserts)
                fl = fl + 0.3 * memory_teacher_loss(adv_mem, t_mem)

        # Front-loaded weighting (closer frames matter more).
        # Round-1: floor raised 0.3 → 0.5 and decay slowed 0.2 → 0.15 so that
        # post-prefix frames (f15+) carry non-trivial weight under EVAL_HORIZON=10.
        w = max(0.5, 2.0 - rank * 0.15)
        loss = loss + w * fl
        n += 1

    return loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4: Unified PGD Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def optimize_unified(
    surrogate: SAM2Surrogate,
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
    regime: str,
    no_teacher: bool = False,
    ablation: str = "combined",
    insert_strength: float = 1.0,
    use_teacher: bool = False,
    high_fidelity_insert: bool = False,
    hi_fi_gated: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict]:
    """Unified PGD for both regimes. Only loss + insert bases differ.

    Shared: frame schedule, perturbation bounds, 3-stage PGD, fake quant,
            step sizes, eval window, quality constraints.
    """
    device = surrogate.device
    T = len(frames_uint8)
    H, W = frames_uint8[0].shape[:2]

    frames_t = [
        torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        for f in frames_uint8
    ]

    idx_map = build_modified_index_map(T, schedule)
    perturb_set = select_perturb_originals(schedule, T)

    # ── Regime-specific setup ────────────────────────────────────────────
    high_fidelity_insert_effective = False
    insert_edit_masks_np = {}
    if regime == "suppression":
        insert_bases_np = _build_suppression_bases(frames_uint8, masks_uint8, schedule)
        decoy_targets = None
        decoy_offset = None
    else:
        # Round 3: optionally build teacher trajectory on modified timeline.
        # When --use_teacher is set, surrogate generates "what SAM2 would
        # write if it were tracking the object at the decoy position" and
        # _decoy_write pulls inserts' maskmem_features + obj_ptr toward
        # those teacher features (memory-level cooperation, v3 resurrected).
        teacher_surr = surrogate if (use_teacher and not no_teacher) else None
        # Probe is_natural_distractor cheaply for gated mode (reviewer Round 3).
        effective_hifi = high_fidelity_insert
        if hi_fi_gated and high_fidelity_insert:
            from memshield.decoy import find_decoy_region
            T_probe = len(masks_uint8)
            ref_idx = min(1, T_probe - 1)
            if masks_uint8[ref_idx].sum() < 100 and masks_uint8[0].sum() > 100:
                ref_idx = 0
            _, _, is_nd = find_decoy_region(
                masks_uint8[ref_idx], frames_uint8[ref_idx], 0.75)
            effective_hifi = not is_nd
            print(f"    [gated hi-fi] is_natural_distractor={is_nd} "
                  f"→ hi-fi={'ON' if effective_hifi else 'OFF'}")
        insert_bases_np, role_data, decoy_offset, teacher_features = \
            _build_decoy_bases_and_targets(
                frames_uint8, masks_uint8, schedule, perturb_set, device,
                surrogate=teacher_surr,
                high_fidelity_insert=effective_hifi)
        decoy_targets = role_data["targets"]
        insert_edit_masks_np = role_data.get("insert_edit_masks", {})
        # When gating disabled hi-fi for this clip, the downstream spatial-eps
        # branch must also be inactive. Flip the flag used later.
        high_fidelity_insert_effective = effective_hifi

    # ── Deltas (identical init for both regimes) ─────────────────────────
    insert_deltas, insert_eps, insert_bases_t = {}, {}, {}
    insert_eps_map = {}  # Per-pixel ε budget when hi-fi is enabled.
    for si, slot in enumerate(schedule):
        base_np = insert_bases_np[si]
        base_t = torch.from_numpy(base_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        insert_bases_t[si] = base_t.detach()
        insert_deltas[si] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[si] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    # Hi-fi decoy (Pilot C, 2026-04-21): spatial ε budget. Inside the dilated
    # paste region keep full ε_ins (8/255); outside shrink to eps_outside
    # (2/255) so the global adversarial field is preserved but bounded small.
    # Hard-clamp (Pilot A/B) regressed the attack because the signal is
    # distributed across the whole frame, not only inside the paste region.
    eps_outside = 2.0 / 255
    # Use the per-clip effective flag (flipped off by gated mode on distractor
    # videos) so both the base construction and the epsilon map stay consistent.
    _hifi_eff = (high_fidelity_insert_effective
                 if regime == "decoy" else False)
    if regime == "decoy" and _hifi_eff:
        for si in insert_deltas:
            em_np = insert_edit_masks_np.get(si)
            if em_np is None:
                continue
            em_t = torch.from_numpy(em_np.astype(np.float32))
            em_t = em_t.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            eps_in = insert_eps[si]
            eps_map = em_t * eps_in + (1.0 - em_t) * eps_outside
            # Broadcast to [1,3,H,W] for elementwise clamp.
            eps_map = eps_map.expand(1, 3, H, W).contiguous()
            insert_eps_map[si] = eps_map.detach()

    orig_deltas, orig_eps = {}, {}
    for oi in perturb_set:
        orig_deltas[oi] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[oi] = 2.0 / 255 if oi == 0 else 4.0 / 255

    # ── Eval indices ─────────────────────────────────────────────────────
    # Use EVAL_HORIZON (post-prefix coverage) for decoy, legacy EVAL_END for suppression
    eval_end_effective = (min(T, EVAL_START + EVAL_HORIZON)
                          if regime == "decoy" else min(T, EVAL_END))
    eval_orig = [i for i in range(EVAL_START, eval_end_effective)]
    eval_mod = [idx_map["orig_to_mod"][j] for j in eval_orig]

    # ── PGD params ────────────────────────────────────────────────────────
    n_steps = cfg.n_steps_strong

    # Regime-specific PGD stages
    if regime == "decoy":
        # Suppress+redirect: perturb warmup → insert warmup → joint rollout
        stage1_end = int(n_steps * 0.25)   # perturb-only
        stage2_end = int(n_steps * 0.50)   # insert-only
        # stage3: joint (all params)
    else:
        # Suppression: inserts first → joint → stabilize (original schedule)
        stage1_end = int(n_steps * 0.2)
        stage2_end = int(n_steps * 0.8)

    # Per-slot step sizes proportional to epsilon
    alpha_ins = {si: max(eps / max(n_steps // 3, 1), eps * 0.1)
                 for si, eps in insert_eps.items()}
    alpha_orig = max(4.0 / 255 / max(n_steps // 3, 1), 0.5 / 255)

    best_loss = float("inf")
    best_id, best_od = {}, {}

    # Build insert_after map: orig_idx -> [slot_indices]
    insert_after = {}
    for si, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(si)

    for step in range(n_steps):
        # ── Active params per stage ──────────────────────────────────
        if regime == "decoy":
            if step < stage1_end:
                # Stage 1 (decoy): perturb-only warmup
                a_ins = set()
                a_orig = set(orig_deltas.keys())
            elif step < stage2_end:
                # Stage 2 (decoy): insert-only warmup
                a_ins = set(insert_deltas.keys())
                a_orig = set()
            else:
                # Stage 3 (decoy): joint rollout optimization (all params)
                a_ins = set(insert_deltas.keys())
                a_orig = set(orig_deltas.keys())
        else:
            if step < stage1_end:
                # Stage 1 (suppression): inserts only
                a_ins = set(insert_deltas.keys())
                a_orig = set()
            elif step < stage2_end:
                # Stage 2 (suppression): all params
                a_ins = set(insert_deltas.keys())
                a_orig = set(orig_deltas.keys())
            else:
                # Stage 3 (suppression): freeze first insert + f0
                a_ins = {si for si in insert_deltas if si > 0}
                a_orig = {oi for oi in orig_deltas if oi > 0}

        # Ablation override: force one component off
        if ablation == "perturb_only":
            a_ins = set()
        elif ablation == "insert_only":
            a_orig = set()

        for si in a_ins:
            if insert_deltas[si].grad is not None:
                insert_deltas[si].grad.zero_()
        for oi in a_orig:
            if orig_deltas[oi].grad is not None:
                orig_deltas[oi].grad.zero_()

        # ── Build modified video ─────────────────────────────────────
        mod_frames = []
        for oi in range(T):
            if oi in orig_deltas and oi in a_orig:
                f = fake_uint8_quantize((frames_t[oi] + orig_deltas[oi]).clamp(0, 1))
            elif oi in orig_deltas:
                f = fake_uint8_quantize(
                    (frames_t[oi] + orig_deltas[oi].detach()).clamp(0, 1))
            else:
                f = frames_t[oi].detach()
            mod_frames.append(f)

            if oi in insert_after:
                for si in insert_after[oi]:
                    d = insert_deltas[si] if si in a_ins else insert_deltas[si].detach()
                    adv = fake_uint8_quantize((insert_bases_t[si] + d).clamp(0, 1))
                    mod_frames.append(adv)

        # ── Forward ──────────────────────────────────────────────────
        all_outs = surrogate.forward_video(mod_frames, masks_uint8[0])

        # ── Regime-specific loss ─────────────────────────────────────
        if regime == "suppression":
            lw = _supp_write(all_outs, masks_uint8, idx_map, perturb_set,
                             schedule, device)
            lr = _supp_read(all_outs, masks_uint8, eval_mod, eval_orig, device)
        else:
            lw = _decoy_write(all_outs, decoy_targets, idx_map, perturb_set,
                              schedule, device, masks_uint8=masks_uint8,
                              insert_strength=insert_strength,
                              teacher_features=teacher_features)
            lr = _decoy_read(all_outs, masks_uint8, eval_mod, eval_orig,
                             decoy_offset, device,
                             teacher_features=teacher_features)

        # ── Quality loss (shared) ────────────────────────────────────
        lq = torch.tensor(0.0, device=device)
        ssim_vals = []
        nq = 0
        for si in insert_deltas:
            adv = (insert_bases_t[si] + insert_deltas[si]).clamp(0, 1)
            sv = differentiable_ssim(insert_bases_t[si], adv)
            ssim_vals.append(sv.item())
            thresh = (cfg.ssim_threshold_strong if schedule[si].frame_type == "strong"
                      else cfg.ssim_threshold_weak)
            lq = lq + F.relu(thresh - sv)
            nq += 1
        for oi in orig_deltas:
            adv = (frames_t[oi] + orig_deltas[oi]).clamp(0, 1)
            sv = differentiable_ssim(frames_t[oi], adv)
            ssim_vals.append(sv.item())
            lq = lq + F.relu(0.97 - sv)
            nq += 1
        if nq > 0:
            lq = lq / nq

        # ── Total loss ───────────────────────────────────────────────
        loss = lw + 1.3 * lr + cfg.lambda_quality * lq

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_id = {si: d.detach().clone() for si, d in insert_deltas.items()}
            best_od = {oi: d.detach().clone() for oi, d in orig_deltas.items()}

        loss.backward()

        # ── PGD step ─────────────────────────────────────────────────
        with torch.no_grad():
            for si in a_ins:
                g = insert_deltas[si].grad
                if g is not None:
                    insert_deltas[si].data -= alpha_ins[si] * g.sign()
                    if si in insert_eps_map:
                        # Spatial ε budget: elementwise clamp to [-eps_map, eps_map].
                        em = insert_eps_map[si]
                        insert_deltas[si].data = torch.clamp(
                            insert_deltas[si].data, -em, em)
                    else:
                        insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])
            for oi in a_orig:
                g = orig_deltas[oi].grad
                if g is not None:
                    orig_deltas[oi].data -= alpha_orig * g.sign()
                    orig_deltas[oi].data.clamp_(-orig_eps[oi], orig_eps[oi])

        if regime == "decoy":
            stage = "Pert" if step < stage1_end else ("Ins" if step < stage2_end else "Joint")
        else:
            stage = "S1" if step < stage1_end else ("S2" if step < stage2_end else "S3")
        ssim_min = min(ssim_vals) if ssim_vals else 0.0
        if step % 10 == 0 or step == n_steps - 1:
            print(f"    [{regime[:4]}|{stage}] step {step:3d}  "
                  f"Lw={lw.item():.4f}  Lr={lr.item():.4f}  SSIM={ssim_min:.4f}")

    # ── Final eval of last-step deltas (may beat best from before step) ──
    with torch.no_grad():
        mod_frames_final = []
        for oi in range(T):
            if oi in orig_deltas:
                f = fake_uint8_quantize((frames_t[oi] + orig_deltas[oi]).clamp(0, 1))
            else:
                f = frames_t[oi]
            mod_frames_final.append(f)
            if oi in insert_after:
                for si in insert_after[oi]:
                    adv = fake_uint8_quantize(
                        (insert_bases_t[si] + insert_deltas[si]).clamp(0, 1))
                    mod_frames_final.append(adv)
        all_outs_final = surrogate.forward_video(mod_frames_final, masks_uint8[0])
        if regime == "suppression":
            lw_f = _supp_write(all_outs_final, masks_uint8, idx_map, perturb_set,
                               schedule, device)
            lr_f = _supp_read(all_outs_final, masks_uint8, eval_mod, eval_orig,
                              device)
        else:
            lw_f = _decoy_write(all_outs_final, decoy_targets, idx_map, perturb_set,
                                schedule, device, masks_uint8=masks_uint8,
                                insert_strength=insert_strength,
                                teacher_features=teacher_features)
            lr_f = _decoy_read(all_outs_final, masks_uint8, eval_mod, eval_orig,
                               decoy_offset, device,
                               teacher_features=teacher_features)
        loss_final = (lw_f + 1.3 * lr_f).item()
        if loss_final < best_loss:
            best_loss = loss_final
            best_id = {si: d.detach().clone() for si, d in insert_deltas.items()}
            best_od = {oi: d.detach().clone() for oi, d in orig_deltas.items()}

    # ── Export ───────────────────────────────────────────────────────────
    ins_u8 = {}
    for si, delta in best_id.items():
        arr = (insert_bases_t[si] + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        ins_u8[si] = (arr.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    orig_u8 = {}
    for oi, delta in best_od.items():
        arr = (frames_t[oi] + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        orig_u8[oi] = (arr.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)

    return ins_u8, orig_u8, {
        "best_loss": best_loss, "regime": regime, "n_steps": n_steps,
        "n_inserted": len(schedule), "n_perturbed": len(perturb_set),
        "decoy_offset": decoy_offset,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5: DAVIS-Standard Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """DAVIS-standard boundary map: pad + find_boundaries(mode='thick').

    Matches the official DAVIS 2017 evaluation toolkit (seg2bmap):
    1. Pad by 1 pixel so image-edge boundaries are detected.
    2. Find thick boundaries (a pixel is boundary if ANY 4-connected
       neighbor has a different value).
    3. Unpad.
    """
    try:
        from skimage.segmentation import find_boundaries
        seg = seg.astype(bool)
        h, w = seg.shape
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = seg
        contour = find_boundaries(padded, mode="thick")
        return contour[1:-1, 1:-1].astype(np.float64)
    except ImportError:
        # Fallback: thick 4-connected boundary with padding
        seg = seg.astype(bool)
        h, w = seg.shape
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = seg
        e = np.zeros_like(padded, dtype=bool)
        e[1:, :] |= padded[1:, :] != padded[:-1, :]   # differs from above
        e[:-1, :] |= padded[:-1, :] != padded[1:, :]   # differs from below
        e[:, 1:] |= padded[:, 1:] != padded[:, :-1]     # differs from left
        e[:, :-1] |= padded[:, :-1] != padded[:, 1:]     # differs from right
        return e[1:-1, 1:-1].astype(np.float64)


def compute_boundary_f(pred: np.ndarray, gt: np.ndarray,
                       bound_th: float = 0.008) -> float:
    """DAVIS-standard boundary F-measure.

    Uses seg2bmap for boundary extraction and distance_transform_edt for
    distance-based precision/recall matching.

    Args:
        pred: [H, W] bool/uint8 predicted mask.
        gt: [H, W] bool/uint8 ground truth mask.
        bound_th: boundary tolerance as fraction of image diagonal
                  (DAVIS default: 0.008).
    """
    h, w = pred.shape[-2:]
    bound_pix = max(1, int(np.ceil(bound_th * np.sqrt(h ** 2 + w ** 2))))

    fg_bd = _seg2bmap(pred.astype(bool))
    gt_bd = _seg2bmap(gt.astype(bool))

    if fg_bd.sum() == 0 and gt_bd.sum() == 0:
        return 1.0
    if fg_bd.sum() == 0 or gt_bd.sum() == 0:
        return 0.0

    # Distance from each pixel to the nearest boundary pixel
    fg_dist = distance_transform_edt(1.0 - fg_bd)
    gt_dist = distance_transform_edt(1.0 - gt_bd)

    # Precision: predicted boundary pixels within threshold of GT boundary
    prec = float((fg_bd * (gt_dist <= bound_pix)).sum()) / float(fg_bd.sum())
    # Recall: GT boundary pixels within threshold of predicted boundary
    rec = float((gt_bd * (fg_dist <= bound_pix)).sum()) / float(gt_bd.sum())

    if prec + rec < 1e-9:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6: Evaluation & Signatures
# ══════════════════════════════════════════════════════════════════════════════

def build_protected_video(
    frames: List[np.ndarray],
    ins_u8: Dict[int, np.ndarray],
    orig_u8: Dict[int, np.ndarray],
    schedule: List[InsertionSlot],
) -> List[np.ndarray]:
    """Assemble protected video from optimized inserts + perturbed originals."""
    insert_after = {}
    for si, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(si)
    protected = []
    for oi in range(len(frames)):
        protected.append(orig_u8.get(oi, frames[oi]))
        if oi in insert_after:
            for si in insert_after[oi]:
                if si in ins_u8:
                    protected.append(ins_u8[si])
    return protected


def build_full_protected_video(
    frames_all: List[np.ndarray],
    ins_u8: Dict[int, np.ndarray],
    orig_u8: Dict[int, np.ndarray],
    schedule: List[InsertionSlot],
    attack_prefix: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """Build full-length video: attacked prefix + clean tail.

    PGD optimizes only the first `attack_prefix` frames. The rest of the
    video is appended unchanged. Returns (full_video, mod_to_orig).
    """
    # Build attacked prefix (with inserts)
    prefix_frames = frames_all[:attack_prefix]
    prefix = build_protected_video(prefix_frames, ins_u8, orig_u8, schedule)

    # Append clean tail
    full = prefix + list(frames_all[attack_prefix:])

    # Build full mod_to_orig mapping
    prefix_map = build_modified_index_map(attack_prefix, schedule)
    mod_to_orig = list(prefix_map["mod_to_orig"])
    for i in range(attack_prefix, len(frames_all)):
        mod_to_orig.append(i)

    return full, mod_to_orig


def compute_horizon_metrics(
    per_frame: Dict[int, Tuple[float, float]],
    T_full: int,
) -> dict:
    """Compute J/F/J&F for multiple time horizons.

    Args:
        per_frame: {orig_idx: (j_score, f_score)} for evaluated frames.
        T_full: total number of original frames.

    Returns dict with keys: short, mid, long, all_future, each containing
    mean_j, mean_f, mean_jf, n_frames.
    """
    horizons = {
        "short": range(EVAL_START, min(EVAL_START + 5, T_full)),
        "mid": range(EVAL_START + 5, min(EVAL_START + 20, T_full)),
        "long": range(EVAL_START + 20, T_full),
        "all_future": range(EVAL_START, T_full),
    }
    results = {}
    for name, frame_range in horizons.items():
        js = [per_frame[i][0] for i in frame_range if i in per_frame]
        fs = [per_frame[i][1] for i in frame_range if i in per_frame]
        if js:
            mj = float(np.mean(js))
            mf = float(np.mean(fs))
            mjf = 0.5 * (mj + mf)
        else:
            mj = mf = mjf = float("nan")  # No frames in this horizon
        results[name] = {
            "mean_j": mj, "mean_f": mf,
            "mean_jf": mjf,
            "n_frames": len(js),
        }
    return results


def evaluate_official(
    protected_frames: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    mod_to_orig: List[int],
    eval_range: set,
    checkpoint: str,
    config: str,
    device_str: str,
) -> dict:
    """J/F/J&F on eval window using official SAM2 VideoPredictor."""
    import shutil
    import tempfile
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(device_str)
    tmpdir = tempfile.mkdtemp(prefix="regime_eval_")
    try:
        for i, frame in enumerate(protected_frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmpdir)
            # SAM2Long compat: set defaults so SAM2Long code path works
            # (num_pathway=1 makes it behave like standard SAM2)
            state.setdefault("num_pathway", 1)
            state.setdefault("iou_thre", 0.0)
            state.setdefault("uncertainty", 0)
            coords, labels = get_interior_prompt(masks_uint8[0])
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=coords, labels=labels)
            preds = {}
            result = predictor.propagate_in_video(state)
            if isinstance(result, tuple):
                # SAM2Long returns (obj_ids, mask_list)
                _, mask_list = result
                for fi in range(len(mask_list)):
                    preds[fi] = (mask_list[fi][0] > 0.0).cpu().numpy().squeeze()
            else:
                # Standard SAM2 yields (frame_idx, obj_ids, masks_out)
                for fi, _, masks_out in result:
                    preds[fi] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        j_scores, f_scores = [], []
        per_frame = {}  # {orig_idx: (j, f)} for multi-horizon slicing
        for mi in range(len(protected_frames)):
            oi = mod_to_orig[mi]
            if oi < 0 or oi not in eval_range:
                continue
            if mi not in preds or oi >= len(masks_uint8):
                continue
            pred = preds[mi].astype(bool)
            gt = masks_uint8[oi].astype(bool)
            inter = float((pred & gt).sum())
            union = float((pred | gt).sum())
            j = inter / max(union, 1e-9) if union > 0 else 1.0
            f = compute_boundary_f(pred, gt)
            j_scores.append(j)
            f_scores.append(f)
            per_frame[oi] = (j, f)

        mj = float(np.mean(j_scores)) if j_scores else 0.0
        mf = float(np.mean(f_scores)) if f_scores else 0.0
        return {"mean_j": mj, "mean_f": mf, "mean_jf": 0.5 * (mj + mf),
                "j_scores": j_scores, "f_scores": f_scores,
                "per_frame": per_frame,
                "n_eval_frames": len(j_scores)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def extract_signatures(
    surrogate: SAM2Surrogate,
    protected_frames: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    mod_to_orig: List[int],
    eval_range: set,
    decoy_offset: Optional[Tuple[int, int]] = None,
) -> dict:
    """Regime signatures via surrogate forward pass.

    Returns:
        neg_score_rate: fraction of eval frames with object_score < 0
        pos_score_rate: fraction of eval frames with object_score > 0
        collapse_rate: fraction with pred area < 1% of GT area
        decoy_hit_rate: fraction where IoU(pred, decoy_GT) > IoU(pred, GT)
        centroid_shift: mean normalized centroid displacement toward decoy
    """
    device = surrogate.device
    frames_t = [
        torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        for f in protected_frames
    ]
    with torch.no_grad():
        all_outs = surrogate.forward_video(frames_t, masks_uint8[0], use_amp=True)

    neg, pos, col, dhit = 0, 0, 0, 0
    cshifts = []
    total = 0

    for mi in range(len(protected_frames)):
        oi = mod_to_orig[mi]
        if oi < 0 or oi not in eval_range:
            continue
        if mi >= len(all_outs) or oi >= len(masks_uint8):
            continue
        total += 1
        out = all_outs[mi]
        gt = masks_uint8[oi].astype(bool)
        gt_area = float(gt.sum())

        # Object score from surrogate (the real score, not an area proxy)
        sc = out.get("object_score_logits")
        if sc is not None:
            if sc.item() < 0:
                neg += 1
            else:
                pos += 1

        # Predicted mask for area / IoU metrics
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        pred = (logits > 0).cpu().numpy().squeeze().astype(bool)
        pred_area = float(pred.sum())

        # Collapse rate (skip if GT is empty — object genuinely absent)
        if gt_area > 0 and pred_area < 0.01 * gt_area:
            col += 1

        # Decoy-specific metrics (only meaningful when GT is non-empty)
        if decoy_offset is not None and gt_area > 0:
            dy, dx = decoy_offset
            decoy_gt = shift_mask(masks_uint8[oi], dy, dx).astype(bool)
            iou_true = float((pred & gt).sum()) / max(float((pred | gt).sum()), 1e-9)
            iou_decoy = float((pred & decoy_gt).sum()) / max(
                float((pred | decoy_gt).sum()), 1e-9)
            if iou_decoy > iou_true:
                dhit += 1

            # Centroid shift: project pred centroid onto GT->decoy axis
            if pred.sum() > 0 and gt.sum() > 0:
                ys_p, xs_p = np.where(pred)
                cy_p, cx_p = float(ys_p.mean()), float(xs_p.mean())
                ys_g, xs_g = np.where(gt)
                cy_g, cx_g = float(ys_g.mean()), float(xs_g.mean())
                if decoy_gt.sum() > 0:
                    ys_d, xs_d = np.where(decoy_gt)
                    cy_d, cx_d = float(np.mean(ys_d)), float(np.mean(xs_d))
                else:
                    cy_d, cx_d = cy_g + dy, cx_g + dx
                dist_gd = math.sqrt((cy_d - cy_g) ** 2 + (cx_d - cx_g) ** 2)
                if dist_gd > 1:
                    vec_gd = np.array([cy_d - cy_g, cx_d - cx_g])
                    vec_gp = np.array([cy_p - cy_g, cx_p - cx_g])
                    shift = float(np.dot(vec_gp, vec_gd) / (dist_gd ** 2))
                    cshifts.append(np.clip(shift, -0.5, 1.5))

    t = max(total, 1)
    sigs = {
        "neg_score_rate": neg / t,
        "pos_score_rate": pos / t,
        "collapse_rate": col / t,
        "n_eval": total,
    }
    if decoy_offset is not None:
        sigs["decoy_hit_rate"] = dhit / t
        sigs["centroid_shift"] = float(np.mean(cshifts)) if cshifts else 0.0
    return sigs


def compute_ssim_attacked(
    original_frames: List[np.ndarray],
    protected_frames: List[np.ndarray],
    idx_map: dict,
    perturb_set: Set[int],
) -> float:
    """Mean SSIM between original and perturbed attacked frames."""
    vals = []
    for oi in sorted(perturb_set):
        mi = idx_map["orig_to_mod"][oi]
        if mi >= len(protected_frames):
            continue
        o = torch.from_numpy(original_frames[oi]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        p = torch.from_numpy(protected_frames[mi]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            vals.append(differentiable_ssim(o, p).item())
    return float(np.mean(vals)) if vals else 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Section 7: Main Runner
# ══════════════════════════════════════════════════════════════════════════════

def save_protected_video(protected_frames, output_dir, vid, regime):
    """Save protected video as JPEG sequence for SAM2Long reuse."""
    vid_dir = os.path.join(output_dir, "videos", f"{vid}_{regime}")
    os.makedirs(vid_dir, exist_ok=True)
    for i, frame in enumerate(protected_frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(vid_dir, f"{i:05d}.jpg"), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])


def main():
    parser = argparse.ArgumentParser(description="Two Memory-Poisoning Regimes")
    parser.add_argument("--block", choices=["core", "isolation", "single"],
                        default="single")
    parser.add_argument("--regime", choices=["suppression", "decoy", "both"],
                        default="both")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video names")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="0=full video (default), >0=truncate")
    parser.add_argument("--attack_prefix_supp", type=int,
                        default=ATTACK_PREFIX_SUPP,
                        help="Attack prefix for suppression (default 15)")
    parser.add_argument("--attack_prefix_decoy", type=int,
                        default=ATTACK_PREFIX_DECOY,
                        help="Attack prefix for decoy (default 30)")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible PGD")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--davis_root",
                        default=os.path.join(ROOT, "data", "davis"))
    parser.add_argument("--checkpoint",
                        default=os.path.join(ROOT, "checkpoints",
                                             "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2_config",
                        default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--output_dir",
                        default=os.path.join(ROOT, "results_regimes"))
    parser.add_argument("--save_videos", action="store_true",
                        help="Save protected videos for SAM2Long reuse")
    parser.add_argument("--no_teacher", action="store_true",
                        help="Disable teacher cooperation (output-only decoy baseline)")
    parser.add_argument("--use_teacher", action="store_true",
                        help="Round-3: enable teacher memory cooperation (v3 "
                             "resurrected: maskmem + obj_ptr match on inserts + "
                             "first 2-3 post-insert clean frames).")
    parser.add_argument("--ablation", choices=["combined", "perturb_only", "insert_only"],
                        default="combined",
                        help="Ablation mode: combined (default), perturb_only, insert_only")
    parser.add_argument("--insert_strength", type=float, default=1.0,
                        help="Insert loss strength: 1.0=v4 gentle, 2.0=v4.1 aggressive")
    parser.add_argument("--high_fidelity_insert", action="store_true",
                        help="Hi-fi insert pipeline: f_prev identity anchor, "
                             "border-safe Poisson clone (no alpha fallback), "
                             "and hard-clamp PGD delta to dilated paste region. "
                             "Reviewer Pilot A: minimum fidelity fix.")
    parser.add_argument("--seam_dilate_px", type=int, default=5,
                        help="Seam ring width in pixels for edit support mask.")
    parser.add_argument("--hi_fi_gated", action="store_true",
                        help="Gate hi-fi mode: use hi-fi only when the clip "
                             "is NOT a natural-distractor video. Reviewer "
                             "Round 3 (2026-04-21): regime-aware Pareto.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.videos:
        videos = args.videos.split(",")
    elif args.block == "core":
        videos = DAVIS_20
    elif args.block == "isolation":
        videos = DAVIS_PILOT
    else:
        videos = DAVIS_PILOT

    for ap_name, ap_val in [("supp", args.attack_prefix_supp),
                             ("decoy", args.attack_prefix_decoy)]:
        if ap_val < EVAL_START + 5:
            raise ValueError(
                f"--attack_prefix_{ap_name} ({ap_val}) must be >= "
                f"EVAL_START+5 ({EVAL_START + 5})")

    regimes = (["suppression", "decoy"] if args.regime == "both"
               else [args.regime])
    cfg = MemShieldConfig(
        epsilon_strong=8.0 / 255,
        n_steps_strong=args.n_steps,
        device=args.device,
    )

    print("=" * 70)
    print("  Two Memory-Poisoning Regimes - Unified Runner")
    print("=" * 70)
    print(f"  Videos:        {len(videos)}")
    print(f"  Regimes:       {regimes}")
    print(f"  Attack prefix: supp=f0-f{args.attack_prefix_supp-1}, "
          f"decoy=f0-f{args.attack_prefix_decoy-1}")
    print(f"  Eval start:    f{EVAL_START} to end of video")
    print(f"  PGD steps:     {args.n_steps}  seed={args.seed}")
    print(f"  Budget:        f0=2/255  orig=4/255  "
          f"ins_s=8/255  ins_w={cfg.epsilon_weak * 255:.0f}/255")
    print("=" * 70)

    device = torch.device(args.device)
    surrogate = SAM2Surrogate(args.checkpoint, args.sam2_config, device)
    all_results = {}

    for vid in videos:
        print(f"\n{'#' * 60}")
        print(f"  {vid}")
        print(f"{'#' * 60}")

        frames, masks = load_video(args.davis_root, vid, args.max_frames)
        T_full = len(frames)
        min_prefix = min(args.attack_prefix_supp, args.attack_prefix_decoy)
        if T_full < min_prefix:
            print(f"  [skip] {T_full} frames < min attack_prefix {min_prefix}")
            continue

        vid_results = {"T_full": T_full}

        # ── Clean baseline (full video) ──────────────────────────────
        eval_all_future = set(range(EVAL_START, T_full))
        print(f"  [clean] evaluating ({T_full} frames, eval f{EVAL_START}-f{T_full-1})...")
        clean_eval = evaluate_official(
            frames, masks, list(range(T_full)), eval_all_future,
            args.checkpoint, args.sam2_config, args.device)
        clean_horizons = compute_horizon_metrics(
            clean_eval.get("per_frame", {}), T_full)
        vid_results["clean"] = {**clean_eval, "horizons": clean_horizons}
        print(f"  [clean] J&F={clean_eval['mean_jf']:.4f}  "
              f"(short={clean_horizons['short']['mean_jf']:.3f}  "
              f"mid={clean_horizons['mid']['mean_jf']:.3f}  "
              f"long={clean_horizons['long']['mean_jf']:.3f})")

        # ── Run each regime ──────────────────────────────────────────
        for regime in regimes:
            # Regime-specific attack prefix
            ap = (args.attack_prefix_decoy if regime == "decoy"
                  else args.attack_prefix_supp)
            ap = min(ap, T_full)  # Don't exceed video length

            # Decoy: probe distractor mode to select schedule
            if regime == "decoy":
                from memshield.decoy import find_decoy_region
                ref_idx = min(1, ap - 1)
                if masks[ref_idx].sum() < 100 and masks[0].sum() > 100:
                    ref_idx = 0
                _, _, is_distractor = find_decoy_region(
                    masks[ref_idx], frames[ref_idx])
                if is_distractor:
                    # Natural distractor: use conservative schedule (like old)
                    schedule = compute_resonance_schedule(
                        min(15, ap), cfg.fifo_window, cfg.max_insertion_ratio)
                    decoy_mode = "distractor"
                else:
                    # Background decoy: use cover_prefix for persistence
                    schedule = compute_resonance_schedule(
                        ap, cfg.fifo_window, cfg.max_insertion_ratio,
                        cover_prefix=True)
                    decoy_mode = "background"
            else:
                schedule = compute_resonance_schedule(
                    ap, cfg.fifo_window, cfg.max_insertion_ratio)
                decoy_mode = "n/a"
            perturb_set = select_perturb_originals(schedule, ap)

            # Fixed seed for reproducibility
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            n_ins = len(schedule)
            mode_str = f" mode={decoy_mode}" if regime == "decoy" else ""
            print(f"  [{regime}] optimizing ({args.n_steps} steps on f0-f{ap-1}, "
                  f"{n_ins} inserts{mode_str})...")
            try:
                t0 = time.time()
                # PGD on attack prefix only
                ins_u8, orig_u8, opt_met = optimize_unified(
                    surrogate, frames[:ap], masks[:ap], schedule, cfg, regime,
                    no_teacher=args.no_teacher, ablation=args.ablation,
                    insert_strength=args.insert_strength,
                    use_teacher=args.use_teacher,
                    high_fidelity_insert=args.high_fidelity_insert,
                    hi_fi_gated=args.hi_fi_gated)
                opt_time = time.time() - t0

                # Build full video: attacked prefix + clean tail
                protected, mod_to_orig = build_full_protected_video(
                    frames, ins_u8, orig_u8, schedule, ap)

                if args.save_videos:
                    save_protected_video(
                        protected, args.output_dir, vid, regime)

                print(f"  [{regime}] evaluating ({len(protected)} frames)...")
                ev = evaluate_official(
                    protected, masks, mod_to_orig, eval_all_future,
                    args.checkpoint, args.sam2_config, args.device)

                # Multi-horizon metrics
                atk_horizons = compute_horizon_metrics(
                    ev.get("per_frame", {}), T_full)

                # Signatures (on short horizon only, uses surrogate)
                prefix_map = build_modified_index_map(ap, schedule)
                short_range = set(range(EVAL_START, min(EVAL_START + 5, T_full)))
                prefix_protected = build_protected_video(
                    frames[:ap], ins_u8, orig_u8, schedule)
                sigs = extract_signatures(
                    surrogate, prefix_protected, masks[:ap],
                    prefix_map["mod_to_orig"],
                    short_range, opt_met.get("decoy_offset"))

                ssim_atk = compute_ssim_attacked(
                    frames[:ap], prefix_protected, prefix_map, perturb_set)

                # Compute drops for each horizon (NaN if no frames)
                drops = {}
                for h in ["short", "mid", "long", "all_future"]:
                    c = clean_horizons[h]["mean_jf"]
                    a = atk_horizons[h]["mean_jf"]
                    if math.isnan(c) or math.isnan(a):
                        drops[h] = float("nan")
                    else:
                        drops[h] = c - a

                vid_results[regime] = {
                    **ev,
                    "horizons": atk_horizons,
                    "drops": drops,
                    "jf_drop": drops["all_future"],
                    "ssim_attacked": ssim_atk,
                    "opt_time": opt_time,
                    "signatures": sigs,
                    "opt_metrics": opt_met,
                }
                print(f"  [{regime}] J&F={ev['mean_jf']:.4f}  "
                      f"drop(all)={drops['all_future']:.4f}  "
                      f"drop(short)={drops['short']:.4f}  "
                      f"drop(mid)={drops.get('mid', 0):.4f}  "
                      f"drop(long)={drops.get('long', 0):.4f}  "
                      f"SSIM={ssim_atk:.4f}")
                print(f"            Sigs: {sigs}")

            except Exception as e:
                traceback.print_exc()
                vid_results[regime] = {"error": str(e)}

        all_results[vid] = vid_results

        # Save incrementally
        out_path = os.path.join(args.output_dir, "regimes_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary Table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS — Multi-Horizon (all videos, primary=all_future)")
    print("=" * 70)

    hdr = f"  {'Video':<20s} {'T':>3s} {'J&F_c':>7s}"
    for r in regimes:
        hdr += f"  {'dAll_' + r[:4]:>11s} {'dShort':>7s} {'dMid':>6s} {'dLong':>6s}"
    print(hdr)
    print("-" * 90)

    for vid, vr in all_results.items():
        jfc = vr.get("clean", {}).get("mean_jf", 0)
        tf = vr.get("T_full", 0)
        row = f"  {vid:<20s} {tf:>3d} {jfc:>7.4f}"
        for r in regimes:
            drops = vr.get(r, {}).get("drops", {})
            da = drops.get("all_future", float("nan"))
            ds = drops.get("short", float("nan"))
            dm = drops.get("mid", float("nan"))
            dl = drops.get("long", float("nan"))
            row += f"  {da:>11.4f} {ds:>7.4f} {dm:>6.3f} {dl:>6.3f}"
        print(row)

    # Aggregate: ALL videos (primary)
    print(f"\n  PRIMARY (all {len(all_results)} videos):")
    for r in regimes:
        drops = [vr[r]["jf_drop"] for vr in all_results.values()
                 if isinstance(vr.get(r), dict) and "jf_drop" in vr[r]]
        if drops:
            print(f"    {r}: mean dJF(all_future)={np.mean(drops):.4f}  "
                  f"median={np.median(drops):.4f}  n={len(drops)}")

    # Aggregate: trackable-clean subset (secondary, tau=0.60)
    for tau in [0.50, 0.60, 0.70]:
        n_elig = sum(1 for vr in all_results.values()
                     if vr.get("clean", {}).get("mean_jf", 0) >= tau)
        print(f"\n  SECONDARY (clean J&F >= {tau:.2f}: {n_elig}/{len(all_results)}):")
        for r in regimes:
            drops = [vr[r]["jf_drop"] for vr in all_results.values()
                     if vr.get("clean", {}).get("mean_jf", 0) >= tau
                     and isinstance(vr.get(r), dict) and "jf_drop" in vr[r]]
            if drops:
                print(f"    {r}: mean dJF={np.mean(drops):.4f}  "
                      f"median={np.median(drops):.4f}  n={len(drops)}")

    # Signature comparison (all videos)
    print("\n  Signatures (all videos):")
    for r in regimes:
        all_sigs = [vr[r]["signatures"] for vr in all_results.values()
                    if isinstance(vr.get(r), dict) and "signatures" in vr[r]]
        if all_sigs:
            neg_r = np.mean([s["neg_score_rate"] for s in all_sigs])
            pos_r = np.mean([s["pos_score_rate"] for s in all_sigs])
            col_r = np.mean([s["collapse_rate"] for s in all_sigs])
            line = f"    {r}: NegScore={neg_r:.2f}  PosScore={pos_r:.2f}  Collapse={col_r:.2f}"
            if "decoy_hit_rate" in all_sigs[0]:
                dhr = np.mean([s["decoy_hit_rate"] for s in all_sigs])
                cs = np.mean([s["centroid_shift"] for s in all_sigs])
                line += f"  DecoyHit={dhr:.2f}  CentShift={cs:.2f}"
            print(line)

    print(f"\nResults: {os.path.join(args.output_dir, 'regimes_results.json')}")


if __name__ == "__main__":
    main()
