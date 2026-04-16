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
from memshield.generator import build_role_targets, select_perturb_originals
from memshield.losses import (
    decoy_target_loss,
    differentiable_ssim,
    fake_uint8_quantize,
    mean_logit_loss,
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

DAVIS_20 = [
    "bear", "bike-packing", "blackswan", "bmx-bumps", "bmx-trees",
    "boat", "breakdance", "breakdance-flare", "bus", "car-roundabout",
    "car-shadow", "car-turn", "cat-girl", "classic-car", "color-run",
    "cows", "crossing", "dance-jump", "dance-twirl", "dog",
]

DAVIS_PILOT = ["bear", "car-shadow", "dance-jump", "dog", "cows"]

EVAL_START = 10  # Disjoint eval window start (inclusive)
EVAL_END = 15    # Exclusive


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1: Dataset Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_video(davis_root: str, vid: str, max_frames: int = 15,
               target_obj: Optional[int] = None):
    """Load DAVIS 2017 video frames and per-object annotation masks.

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
) -> Tuple[Dict[int, np.ndarray], dict, Tuple[int, int]]:
    """Decoy bases (relocated object) + role targets from generator.py."""
    role_data = build_role_targets(
        masks_uint8, frames_uint8, schedule, perturb_set, device,
    )
    return role_data["insert_bases"], role_data, role_data["decoy_offset"]


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


# ── Decoy losses ─────────────────────────────────────────────────────────────

def _decoy_write(all_outs, targets, idx_map, perturb_set, schedule, device):
    """Decoy write-path: relocate object in attack-window frames."""
    loss = torch.tensor(0.0, device=device)
    n = 0

    for oi in sorted(perturb_set):
        key = ("orig", oi)
        if key not in targets:
            continue
        mi = idx_map["orig_to_mod"][oi]
        if mi >= len(all_outs):
            continue
        out = all_outs[mi]
        logits = out.get("logits_orig_hw")
        if logits is None:
            continue
        td = _resize_td(targets[key], logits.shape[-2:])
        fl = decoy_target_loss(logits, td)
        sc = out.get("object_score_logits")
        if sc is not None and float(td.get("score_w", 0)) > 0:
            fl = fl + float(td["score_w"]) * object_score_positive_loss(
                sc, margin=float(td.get("score_margin", 0.5)))
        loss = loss + fl
        n += 1

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
        fl = decoy_target_loss(logits, td)
        sc = out.get("object_score_logits")
        if sc is not None and float(td.get("score_w", 0)) > 0:
            fl = fl + float(td["score_w"]) * object_score_positive_loss(
                sc, margin=float(td.get("score_margin", 0.5)))
        loss = loss + 1.5 * fl
        n += 1

    return loss / max(n, 1)


def _decoy_read(all_outs, masks_uint8, eval_mod, eval_orig, decoy_offset,
                device):
    """Decoy read-path: verify eval frames show relocated object."""
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
        alpha = max(0.55, 1.25 - rank * 0.08)
        w = max(0.5, 2.5 - rank * 0.3)
        mask = masks_uint8[min(oi, T - 1)]
        d_mask = shift_mask(mask, dy, dx)
        suppress = ((mask > 0) & (d_mask == 0)).astype(np.float32)
        decoy_only = ((d_mask > 0) & (mask == 0)).astype(np.float32)
        td = {
            "core": torch.zeros(1, 1, H, W, device=device),
            "bridge": torch.zeros(1, 1, H, W, device=device),
            "decoy": torch.from_numpy(decoy_only).unsqueeze(0).unsqueeze(0).to(device),
            "suppress": torch.from_numpy(suppress).unsqueeze(0).unsqueeze(0).to(device),
            "core_w": 0.0, "bridge_w": 0.0,
            "decoy_w": alpha,
            "suppress_w": min(1.6, alpha + 0.40),
            "rank_w": min(1.3, alpha + 0.20),
            "bg_w": 0.0,
            "score_margin": 0.5, "support_margin": 0.0, "bridge_margin": 0.0,
            "decoy_margin": 0.9, "suppress_margin": 0.6, "rank_margin": 0.8,
        }
        td = _resize_td(td, logits.shape[-2:])
        fl = decoy_target_loss(logits, td)
        sc = out.get("object_score_logits")
        if sc is not None:
            fl = fl + 0.30 * object_score_positive_loss(sc, margin=0.5)
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
    if regime == "suppression":
        insert_bases_np = _build_suppression_bases(frames_uint8, masks_uint8, schedule)
        decoy_targets = None
        decoy_offset = None
    else:
        insert_bases_np, role_data, decoy_offset = _build_decoy_bases_and_targets(
            frames_uint8, masks_uint8, schedule, perturb_set, device)
        decoy_targets = role_data["targets"]

    # ── Deltas (identical init for both regimes) ─────────────────────────
    insert_deltas, insert_eps, insert_bases_t = {}, {}, {}
    for si, slot in enumerate(schedule):
        base_np = insert_bases_np[si]
        base_t = torch.from_numpy(base_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        insert_bases_t[si] = base_t.detach()
        insert_deltas[si] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[si] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    orig_deltas, orig_eps = {}, {}
    for oi in perturb_set:
        orig_deltas[oi] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[oi] = 2.0 / 255 if oi == 0 else 4.0 / 255

    # ── Eval indices ─────────────────────────────────────────────────────
    eval_orig = [i for i in range(EVAL_START, min(T, EVAL_END))]
    eval_mod = [idx_map["orig_to_mod"][j] for j in eval_orig]

    # ── PGD params (matched for both regimes) ────────────────────────────
    n_steps = cfg.n_steps_strong
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
        if step < stage1_end:
            # Stage 1: inserts only
            a_ins = set(insert_deltas.keys())
            a_orig = set()
        elif step < stage2_end:
            # Stage 2: all params
            a_ins = set(insert_deltas.keys())
            a_orig = set(orig_deltas.keys())
        else:
            # Stage 3: freeze first insert + f0, optimize rest
            a_ins = {si for si in insert_deltas if si > 0}
            a_orig = {oi for oi in orig_deltas if oi > 0}

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
                              schedule, device)
            lr = _decoy_read(all_outs, masks_uint8, eval_mod, eval_orig,
                             decoy_offset, device)

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
                    insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])
            for oi in a_orig:
                g = orig_deltas[oi].grad
                if g is not None:
                    orig_deltas[oi].data -= alpha_orig * g.sign()
                    orig_deltas[oi].data.clamp_(-orig_eps[oi], orig_eps[oi])

        stage = "S1" if step < stage1_end else ("S2" if step < stage2_end else "S3")
        ssim_min = min(ssim_vals) if ssim_vals else 0.0
        if step % 10 == 0 or step == n_steps - 1:
            print(f"    [{regime[:4]}|{stage}] step {step:3d}  "
                  f"Lw={lw.item():.4f}  Lr={lr.item():.4f}  SSIM={ssim_min:.4f}")

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
            coords, labels = get_interior_prompt(masks_uint8[0])
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=coords, labels=labels)
            preds = {}
            for fi, _, masks_out in predictor.propagate_in_video(state):
                preds[fi] = (masks_out[0] > 0.0).cpu().numpy().squeeze()

        j_scores, f_scores = [], []
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
            j_scores.append(inter / max(union, 1e-9) if union > 0 else 1.0)
            f_scores.append(compute_boundary_f(pred, gt))

        mj = float(np.mean(j_scores)) if j_scores else 0.0
        mf = float(np.mean(f_scores)) if f_scores else 0.0
        return {"mean_j": mj, "mean_f": mf, "mean_jf": 0.5 * (mj + mf),
                "j_scores": j_scores, "f_scores": f_scores,
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

        # Collapse rate
        if pred_area < 0.01 * max(gt_area, 1):
            col += 1

        # Decoy-specific metrics
        if decoy_offset is not None:
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
    parser.add_argument("--max_frames", type=int, default=15)
    parser.add_argument("--n_steps", type=int, default=50)
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

    regimes = (["suppression", "decoy"] if args.regime == "both"
               else [args.regime])
    cfg = MemShieldConfig(
        epsilon_strong=8.0 / 255,
        n_steps_strong=args.n_steps,
        device=args.device,
    )
    eval_range = set(range(EVAL_START, min(args.max_frames, EVAL_END)))

    print("=" * 70)
    print("  Two Memory-Poisoning Regimes - Unified Runner")
    print("=" * 70)
    print(f"  Videos:      {len(videos)}")
    print(f"  Regimes:     {regimes}")
    print(f"  Eval window: f{EVAL_START}-f{EVAL_END - 1}")
    print(f"  PGD steps:   {args.n_steps}")
    print(f"  Budget:      f0=2/255  orig=4/255  "
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
        if len(frames) < 15:
            print(f"  [skip] {len(frames)} frames < 15")
            continue

        vid_results = {}
        T = len(frames)

        # ── Clean baseline ───────────────────────────────────────────
        print("  [clean] evaluating...")
        clean_eval = evaluate_official(
            frames, masks, list(range(T)), eval_range,
            args.checkpoint, args.sam2_config, args.device)
        vid_results["clean"] = clean_eval
        print(f"  [clean] J={clean_eval['mean_j']:.4f}  "
              f"F={clean_eval['mean_f']:.4f}  "
              f"J&F={clean_eval['mean_jf']:.4f}")

        schedule = compute_resonance_schedule(
            T, cfg.fifo_window, cfg.max_insertion_ratio)
        perturb_set = select_perturb_originals(schedule, T)
        idx_map = build_modified_index_map(T, schedule)

        # ── Run each regime ──────────────────────────────────────────
        for regime in regimes:
            print(f"  [{regime}] optimizing ({args.n_steps} steps)...")
            try:
                t0 = time.time()
                ins_u8, orig_u8, opt_met = optimize_unified(
                    surrogate, frames, masks, schedule, cfg, regime)
                opt_time = time.time() - t0

                protected = build_protected_video(
                    frames, ins_u8, orig_u8, schedule)

                if args.save_videos:
                    save_protected_video(
                        protected, args.output_dir, vid, regime)

                print(f"  [{regime}] evaluating ({len(protected)} frames)...")
                ev = evaluate_official(
                    protected, masks, idx_map["mod_to_orig"], eval_range,
                    args.checkpoint, args.sam2_config, args.device)

                sigs = extract_signatures(
                    surrogate, protected, masks, idx_map["mod_to_orig"],
                    eval_range, opt_met.get("decoy_offset"))

                ssim_atk = compute_ssim_attacked(
                    frames, protected, idx_map, perturb_set)

                djf = clean_eval["mean_jf"] - ev["mean_jf"]
                dj = clean_eval["mean_j"] - ev["mean_j"]
                df = clean_eval["mean_f"] - ev["mean_f"]

                vid_results[regime] = {
                    **ev,
                    "jf_drop": djf, "j_drop": dj, "f_drop": df,
                    "ssim_attacked": ssim_atk,
                    "opt_time": opt_time,
                    "signatures": sigs,
                    "opt_metrics": opt_met,
                }
                print(f"  [{regime}] J={ev['mean_j']:.4f}  "
                      f"F={ev['mean_f']:.4f}  "
                      f"J&F={ev['mean_jf']:.4f}  "
                      f"drop(J&F)={djf:.4f}  SSIM={ssim_atk:.4f}")
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
    print("  RESULTS (eval on f10-f14)")
    print("=" * 70)

    hdr = f"  {'Video':<18s} {'J&F_c':>7s}"
    for r in regimes:
        hdr += f"  {'dJF_' + r:>13s}  {'dJ_' + r:>11s}"
    print(hdr)
    print("-" * 70)

    eligible = 0
    for vid, vr in all_results.items():
        jfc = vr.get("clean", {}).get("mean_jf", 0)
        elig = jfc >= 0.60
        if elig:
            eligible += 1
        row = f"  {vid:<18s} {jfc:>7.4f}"
        for r in regimes:
            djf = vr.get(r, {}).get("jf_drop", float("nan"))
            dj = vr.get(r, {}).get("j_drop", float("nan"))
            row += f"  {djf:>13.4f}  {dj:>11.4f}"
        print(row + (" *" if elig else ""))

    print(f"\n  Eligible (J&F >= 0.60): {eligible}/{len(all_results)}")

    # Aggregate stats on eligible subset
    for r in regimes:
        drops = [
            vr[r]["jf_drop"]
            for vr in all_results.values()
            if vr.get("clean", {}).get("mean_jf", 0) >= 0.60
            and isinstance(vr.get(r), dict) and "jf_drop" in vr[r]
        ]
        if drops:
            print(f"  {r} eligible: mean dJF={np.mean(drops):.4f}  "
                  f"median={np.median(drops):.4f}  n={len(drops)}")

    # Signature comparison
    print("\n  Signature Comparison (eligible):")
    for r in regimes:
        all_sigs = [
            vr[r]["signatures"]
            for vr in all_results.values()
            if vr.get("clean", {}).get("mean_jf", 0) >= 0.60
            and isinstance(vr.get(r), dict) and "signatures" in vr[r]
        ]
        if all_sigs:
            neg_r = np.mean([s["neg_score_rate"] for s in all_sigs])
            pos_r = np.mean([s["pos_score_rate"] for s in all_sigs])
            col_r = np.mean([s["collapse_rate"] for s in all_sigs])
            line = (f"  {r}: NegScore={neg_r:.2f}  PosScore={pos_r:.2f}  "
                    f"Collapse={col_r:.2f}")
            if "decoy_hit_rate" in all_sigs[0]:
                dhr = np.mean([s["decoy_hit_rate"] for s in all_sigs])
                cs = np.mean([s["centroid_shift"] for s in all_sigs])
                line += f"  DecoyHit={dhr:.2f}  CentShift={cs:.2f}"
            print(line)

    print(f"\nResults: {os.path.join(args.output_dir, 'regimes_results.json')}")


if __name__ == "__main__":
    main()
