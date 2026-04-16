"""
Cooperative Decoy Generator v3: Synergistic insert + perturb.

Key design (from GPT-5.4 cooperative scheme):
  - All frames share ONE decoy trajectory (same offset direction)
  - Each frame has a ROLE-SPECIFIC pseudo-target
  - Originals "open the door" for decoy; inserts "plant wrong memory"
  - 3-stage PGD: (1) inserts alone → (2) joint → (3) stabilization
  - Object-score KEPT positive (confident wrong, not absent)
"""
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import MemShieldConfig
from .decoy import find_decoy_region, create_bridge_mask, create_decoy_base_frame, shift_mask
from .losses import (
    decoy_target_loss, fake_uint8_quantize,
    differentiable_ssim,
)
from .scheduler import InsertionSlot, build_modified_index_map
from .surrogate import SAM2Surrogate


def select_perturb_originals(
    schedule: List[InsertionSlot], T: int,
) -> Set[int]:
    """Select which original frames to perturb."""
    perturb_set = {0}
    for slot in schedule:
        pos = slot.after_original_idx
        if pos >= 0:
            perturb_set.add(pos)
        if pos + 1 < T:
            perturb_set.add(pos + 1)
        if pos + 2 < T:
            perturb_set.add(pos + 2)
    return perturb_set


def build_role_targets(
    masks_uint8: List[np.ndarray],
    frames_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
    perturb_set: Set[int],
    device: torch.device,
    offset_ratio: float = 0.5,
) -> Dict:
    """Build per-frame pseudo-targets sharing ONE decoy direction.

    Returns dict with:
      'decoy_offset': (dy, dx) used for all frames
      'targets': {frame_or_slot_key: [1,1,H,W] float tensor}
      'decoy_masks': {frame_idx: np.ndarray} for visualization
      'insert_bases': {slot_i: [H,W,3] uint8} sharp base frames
    """
    import cv2

    T = len(masks_uint8)
    H, W = masks_uint8[0].shape

    # Use first available mask with good area for decoy computation
    ref_mask = masks_uint8[min(1, T - 1)]
    if ref_mask.sum() < 100 and masks_uint8[0].sum() > 100:
        ref_mask = masks_uint8[0]
    ref_frame = frames_uint8[min(1, T - 1)]

    # Find ONE shared decoy direction
    decoy_mask, offset = find_decoy_region(ref_mask, ref_frame, offset_ratio)
    dy, dx = offset

    targets = {}
    decoy_masks_np = {}
    insert_bases = {}

    def erode_mask(mask, iters=2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded = cv2.erode(mask, kernel, iterations=iters)
        return eroded if eroded.sum() > 50 else mask

    def make_target_dict(mask, core_w, bridge_w, decoy_w):
        """Build 3 NON-OVERLAPPING binary masks + weights for weighted BCE.

        Returns dict for decoy_target_loss with keys:
          core, bridge, decoy (each [1,1,H,W] float tensor)
          core_w, bridge_w, decoy_w (float weights)
        """
        core = erode_mask(mask)
        d_mask = shift_mask(mask, dy, dx)
        bridge_full = create_bridge_mask(mask, d_mask, bridge_width=12)

        # Make non-overlapping: priority decoy > bridge > core
        core_only = (core > 0).astype(np.float32)
        decoy_only = (d_mask > 0).astype(np.float32)
        bridge_only = ((bridge_full > 0) & (core == 0) & (d_mask == 0)).astype(np.float32)
        # Remove overlap: decoy wins over core where they overlap
        core_only[(d_mask > 0)] = 0.0

        return {
            "core": torch.from_numpy(core_only).unsqueeze(0).unsqueeze(0).to(device),
            "bridge": torch.from_numpy(bridge_only).unsqueeze(0).unsqueeze(0).to(device),
            "decoy": torch.from_numpy(decoy_only).unsqueeze(0).unsqueeze(0).to(device),
            "core_w": core_w,
            "bridge_w": bridge_w,
            "decoy_w": decoy_w,
        }, d_mask

    # Build insertion-position lookup
    insert_positions = {s.after_original_idx: s for s in schedule}

    # ── Frame 0: loosen conditioning memory ──────────────────────────────
    if 0 in perturb_set:
        td, dm = make_target_dict(masks_uint8[0], core_w=1.0, bridge_w=0.05, decoy_w=0.0)
        targets[("orig", 0)] = td

    # ── Per-frame targets based on role ──────────────────────────────────
    for orig_idx in sorted(perturb_set):
        if orig_idx == 0:
            continue
        mask = masks_uint8[min(orig_idx, T - 1)]

        is_pre_insert = any(orig_idx == s.after_original_idx for s in schedule)
        is_post_insert = any(orig_idx == s.after_original_idx + 1 for s in schedule)
        is_post_insert2 = any(orig_idx == s.after_original_idx + 2 for s in schedule)

        if is_pre_insert:
            td, dm = make_target_dict(mask, core_w=1.0, bridge_w=0.15, decoy_w=0.10)
        elif is_post_insert:
            td, dm = make_target_dict(mask, core_w=1.0, bridge_w=0.25, decoy_w=0.25)
        elif is_post_insert2:
            td, dm = make_target_dict(mask, core_w=1.0, bridge_w=0.15, decoy_w=0.15)
        else:
            td, dm = make_target_dict(mask, core_w=1.0, bridge_w=0.05, decoy_w=0.05)

        targets[("orig", orig_idx)] = td
        decoy_masks_np[orig_idx] = dm

    # ── Insert targets (use shared offset, consistent mask_after) ────────
    for slot_i, slot in enumerate(schedule):
        pos = slot.after_original_idx
        # Fix M2: use mask_after consistently (matches frame_after base)
        frame_after = frames_uint8[min(pos + 1, T - 1)]
        mask_after = masks_uint8[min(pos + 1, T - 1)]

        if slot.frame_type == "strong":
            td, dm = make_target_dict(mask_after, core_w=0.30, bridge_w=0.70, decoy_w=1.00)
        else:
            td, dm = make_target_dict(mask_after, core_w=0.25, bridge_w=0.60, decoy_w=0.85)

        # Fix C1: pass shared offset to base frame constructor
        base = create_decoy_base_frame(frame_after, mask_after, offset)

        targets[("insert", slot_i)] = td
        insert_bases[slot_i] = base
        decoy_masks_np[f"insert_{slot_i}"] = dm

    return {
        "decoy_offset": offset,
        "targets": targets,
        "decoy_masks": decoy_masks_np,
        "insert_bases": insert_bases,
    }


def optimize_cooperative(
    surrogate: SAM2Surrogate,
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict]:
    """Cooperative Decoy PGD: 3-stage optimization with role-specific targets.

    Stage 1 (20%): Optimize inserts only → establish decoy attractor
    Stage 2 (60%): Joint optimization → originals support decoy
    Stage 3 (20%): Freeze insert1+f0, optimize rest → stabilize

    Returns: (inserted_uint8, perturbed_orig_uint8, metrics)
    """
    device = surrogate.device
    T = len(frames_uint8)
    H, W = frames_uint8[0].shape[:2]

    frames_t = [torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                for f in frames_uint8]

    idx_map = build_modified_index_map(T, schedule)
    perturb_set = select_perturb_originals(schedule, T)

    # Build role targets
    role_data = build_role_targets(
        masks_uint8, frames_uint8, schedule, perturb_set, device,
    )
    targets = role_data["targets"]
    insert_bases_np = role_data["insert_bases"]

    # ── Create deltas ────────────────────────────────────────────────────
    insert_deltas = {}
    insert_eps = {}
    insert_bases_t = {}
    for slot_i, slot in enumerate(schedule):
        base_np = insert_bases_np[slot_i]
        base_t = torch.from_numpy(base_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        insert_bases_t[slot_i] = base_t.detach()
        insert_deltas[slot_i] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[slot_i] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    orig_deltas = {}
    orig_eps = {}
    for oi in perturb_set:
        orig_deltas[oi] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[oi] = 2.0 / 255 if oi == 0 else 4.0 / 255

    # ── Eval setup ───────────────────────────────────────────────────────
    first_insert = schedule[0].after_original_idx if schedule else 1
    eval_orig_start = first_insert + 1
    eval_orig_end = min(T, eval_orig_start + cfg.future_horizon)
    eval_orig_indices = list(range(eval_orig_start, eval_orig_end))
    eval_mod_indices = [idx_map["orig_to_mod"][j] for j in eval_orig_indices]

    # Build future targets (decoy-aware, decaying weight)
    future_targets = {}
    dy, dx = role_data["decoy_offset"]
    for rank, oi in enumerate(eval_orig_indices):
        alpha = max(0.05, 0.25 - rank * 0.02)
        mask = masks_uint8[min(oi, T - 1)]
        d_mask = shift_mask(mask, dy, dx)
        core = mask.astype(np.float32)
        decoy_only = ((d_mask > 0) & (mask == 0)).astype(np.float32)
        future_targets[oi] = {
            "core": torch.from_numpy(core).unsqueeze(0).unsqueeze(0).to(device),
            "bridge": torch.zeros(1, 1, H, W, device=device),
            "decoy": torch.from_numpy(decoy_only).unsqueeze(0).unsqueeze(0).to(device),
            "core_w": 1.0,
            "bridge_w": 0.0,
            "decoy_w": alpha,
        }

    # ── PGD setup ────────────────────────────────────────────────────────
    n_steps = cfg.n_steps_strong
    stage1_end = int(n_steps * 0.2)
    stage2_end = int(n_steps * 0.8)
    # stage3: rest

    alpha_ins = max(cfg.epsilon_strong / max(n_steps // 3, 1), 1.0 / 255)
    alpha_orig = max(4.0 / 255 / max(n_steps // 3, 1), 0.5 / 255)

    best_loss = float("inf")
    best_insert_deltas = {}
    best_orig_deltas = {}
    history = {"steps": [], "loss": []}

    insert_after = {}
    for slot_i, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(slot_i)

    for step in range(n_steps):
        # ── Determine active parameters for this stage ───────────────
        if step < stage1_end:
            # Stage 1: inserts only
            active_insert = set(insert_deltas.keys())
            active_orig = set()
        elif step < stage2_end:
            # Stage 2: all
            active_insert = set(insert_deltas.keys())
            active_orig = set(orig_deltas.keys())
        else:
            # Stage 3: freeze insert[0] and f0, optimize rest
            active_insert = {si for si in insert_deltas if si > 0}
            active_orig = {oi for oi in orig_deltas if oi > 0}

        # Zero grads for active params only
        for si in active_insert:
            if insert_deltas[si].grad is not None:
                insert_deltas[si].grad.zero_()
        for oi in active_orig:
            if orig_deltas[oi].grad is not None:
                orig_deltas[oi].grad.zero_()

        # ── Build modified video ─────────────────────────────────────
        mod_frames = []
        for orig_idx in range(T):
            if orig_idx in orig_deltas and orig_idx in active_orig:
                frame = (frames_t[orig_idx] + orig_deltas[orig_idx]).clamp(0.0, 1.0)
                frame = fake_uint8_quantize(frame)
            elif orig_idx in orig_deltas:
                frame = (frames_t[orig_idx] + orig_deltas[orig_idx].detach()).clamp(0.0, 1.0)
                frame = fake_uint8_quantize(frame)
            else:
                frame = frames_t[orig_idx].detach()
            mod_frames.append(frame)

            if orig_idx in insert_after:
                for si in insert_after[orig_idx]:
                    if si in active_insert:
                        adv = (insert_bases_t[si] + insert_deltas[si]).clamp(0.0, 1.0)
                    else:
                        adv = (insert_bases_t[si] + insert_deltas[si].detach()).clamp(0.0, 1.0)
                    adv = fake_uint8_quantize(adv)
                    mod_frames.append(adv)

        # ── Forward ──────────────────────────────────────────────────
        all_outs = surrogate.forward_video(mod_frames, masks_uint8[0])

        # ── Write-path loss (on attack-window frames) ────────────────
        loss_write = torch.tensor(0.0, device=device)
        n_write = 0

        def _resize_target_dict(td, target_size):
            """Resize all masks in a target_dict to match logits spatial size."""
            resized = dict(td)
            for k in ("core", "bridge", "decoy"):
                if resized[k].shape[-2:] != target_size:
                    resized[k] = F.interpolate(resized[k], size=target_size, mode="nearest")
            return resized

        # Loss on perturbed originals
        for oi in perturb_set:
            key = ("orig", oi)
            if key in targets:
                mod_idx = idx_map["orig_to_mod"][oi]
                if mod_idx < len(all_outs):
                    out = all_outs[mod_idx]
                    logits = out.get("logits_orig_hw") if isinstance(out, dict) else out
                    if logits is not None:
                        td = _resize_target_dict(targets[key], logits.shape[-2:])
                        loss_write = loss_write + decoy_target_loss(logits, td)
                        n_write += 1

        # Loss on inserted frames
        for si in insert_deltas:
            key = ("insert", si)
            if key in targets:
                insert_indices = idx_map["insert_mod_indices"]
                cursor = 0
                for check_si in range(len(schedule)):
                    if check_si == si:
                        break
                    cursor += 1
                if cursor < len(insert_indices):
                    mod_idx = insert_indices[cursor]
                    if mod_idx < len(all_outs):
                        out = all_outs[mod_idx]
                        logits = out.get("logits_orig_hw") if isinstance(out, dict) else out
                        if logits is not None:
                            td = _resize_target_dict(targets[key], logits.shape[-2:])
                            loss_write = loss_write + 1.5 * decoy_target_loss(logits, td)
                            n_write += 1

        if n_write > 0:
            loss_write = loss_write / n_write

        # ── Read-path loss (on future clean frames) ──────────────────
        loss_read = torch.tensor(0.0, device=device)
        n_read = 0
        for rank, (mod_idx, oi) in enumerate(zip(eval_mod_indices, eval_orig_indices)):
            if mod_idx >= len(all_outs):
                break
            out = all_outs[mod_idx]
            logits = out.get("logits_orig_hw") if isinstance(out, dict) else out
            if logits is None:
                continue
            # Front-loaded weighting
            w = max(0.5, 2.5 - rank * 0.3)
            if oi in future_targets:
                td = _resize_target_dict(future_targets[oi], logits.shape[-2:])
                loss_read = loss_read + w * decoy_target_loss(logits, td)
            n_read += 1
        if n_read > 0:
            loss_read = loss_read / n_read

        # ── Quality loss ─────────────────────────────────────────────
        loss_quality = torch.tensor(0.0, device=device)
        ssim_vals = []
        n_qual = 0
        for si in insert_deltas:
            adv = (insert_bases_t[si] + insert_deltas[si]).clamp(0.0, 1.0)
            sv = differentiable_ssim(insert_bases_t[si], adv)
            ssim_vals.append(sv.item())
            loss_quality = loss_quality + F.relu(cfg.ssim_threshold_strong - sv)
            n_qual += 1
        for oi in orig_deltas:
            adv = (frames_t[oi] + orig_deltas[oi]).clamp(0.0, 1.0)
            sv = differentiable_ssim(frames_t[oi], adv)
            ssim_vals.append(sv.item())
            loss_quality = loss_quality + F.relu(0.97 - sv)
            n_qual += 1
        if n_qual > 0:
            loss_quality = loss_quality / n_qual

        # ── Total loss ───────────────────────────────────────────────
        loss_total = loss_write + 0.7 * loss_read + cfg.lambda_quality * loss_quality

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_insert_deltas = {si: insert_deltas[si].detach().clone() for si in insert_deltas}
            best_orig_deltas = {oi: orig_deltas[oi].detach().clone() for oi in orig_deltas}

        loss_total.backward()

        # ── PGD step (only active params) ────────────────────────────
        with torch.no_grad():
            for si in active_insert:
                if insert_deltas[si].grad is not None:
                    insert_deltas[si].data -= alpha_ins * insert_deltas[si].grad.sign()
                    insert_deltas[si].data.clamp_(-insert_eps[si], insert_eps[si])
            for oi in active_orig:
                if orig_deltas[oi].grad is not None:
                    orig_deltas[oi].data -= alpha_orig * orig_deltas[oi].grad.sign()
                    orig_deltas[oi].data.clamp_(-orig_eps[oi], orig_eps[oi])

        ssim_min = min(ssim_vals) if ssim_vals else 0.0
        stage = "S1" if step < stage1_end else ("S2" if step < stage2_end else "S3")
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  [{stage}] step {step:3d}  L_write={loss_write.item():.4f}  "
                  f"L_read={loss_read.item():.4f}  SSIM={ssim_min:.4f}")

        history["steps"].append(step)
        history["loss"].append(loss_total.item())

    # ── Export ────────────────────────────────────────────────────────
    inserted_uint8 = {}
    for si in best_insert_deltas:
        adv = (insert_bases_t[si] + best_insert_deltas[si]).clamp(0.0, 1.0)
        arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        inserted_uint8[si] = np.rint(arr).clip(0, 255).astype(np.uint8)

    perturbed_orig_uint8 = {}
    for oi in best_orig_deltas:
        adv = (frames_t[oi] + best_orig_deltas[oi]).clamp(0.0, 1.0)
        arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        perturbed_orig_uint8[oi] = np.rint(arr).clip(0, 255).astype(np.uint8)

    metrics = {
        "best_loss": best_loss,
        "n_steps": n_steps,
        "n_inserted": len(schedule),
        "n_perturbed_originals": len(perturb_set),
        "decoy_offset": role_data["decoy_offset"],
        "history": history,
    }
    return inserted_uint8, perturbed_orig_uint8, metrics
