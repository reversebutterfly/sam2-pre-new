"""
Cooperative Decoy Generator v3: Synergistic insert + perturb.

Key design (from GPT-5.4 cooperative scheme):
  - All frames share ONE decoy trajectory (same offset direction)
  - Each frame has a ROLE-SPECIFIC pseudo-target
  - Originals "open the door" for decoy; inserts "plant wrong memory"
  - 3-stage PGD: (1) inserts alone → (2) joint → (3) stabilization
  - Object-score KEPT positive (confident wrong, not absent)
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import MemShieldConfig
from .decoy import (
    find_decoy_region, find_decoy_candidates, create_bridge_mask,
    create_decoy_base_frame, create_decoy_base_frame_hifi, shift_mask,
)
from .losses import (
    decoy_target_loss, fake_uint8_quantize,
    differentiable_ssim,
    object_score_positive_loss,
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
    offset_ratio: float = 0.75,
    high_fidelity_insert: bool = False,
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
) -> Dict:
    """Build per-frame pseudo-targets sharing ONE decoy direction.

    Returns dict with:
      'decoy_offset': (dy, dx) used for all frames
      'targets': {frame_or_slot_key: [1,1,H,W] float tensor}
      'decoy_masks': {frame_idx: np.ndarray} for visualization
      'insert_bases': {slot_i: [H,W,3] uint8} sharp base frames
      'insert_edit_masks': {slot_i: [H,W] uint8} (only when high_fidelity_insert)
    """
    import cv2

    T = len(masks_uint8)
    # Use one reference index for BOTH mask and frame to avoid mismatch
    ref_idx = min(1, T - 1)
    if masks_uint8[ref_idx].sum() < 100 and masks_uint8[0].sum() > 100:
        ref_idx = 0
    ref_mask = masks_uint8[ref_idx]
    ref_frame = frames_uint8[ref_idx]

    if high_fidelity_insert:
        candidates, is_natural_distractor = find_decoy_candidates(
            ref_mask, ref_frame, offset_ratio, top_k=6)
        offset = candidates[0][0]
    else:
        _, offset, is_natural_distractor = find_decoy_region(
            ref_mask, ref_frame, offset_ratio)
        candidates = [(offset, 0.0)]
    dy, dx = offset

    targets = {}
    decoy_masks_np = {}
    insert_bases = {}
    insert_edit_masks = {}

    def erode_mask(mask, iters=2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded = cv2.erode(mask, kernel, iterations=iters)
        return eroded if eroded.sum() > 50 else mask

    def to_tensor(mask_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    def make_target_dict(
        mask,
        core_w,
        bridge_w,
        decoy_w,
        suppress_w,
        rank_w,
        score_w,
        decoy_margin=0.9,
        suppress_margin=0.5,
        rank_margin=0.75,
    ):
        """Build non-overlapping relocation masks + weights."""
        mask_bin = (mask > 0).astype(np.uint8)
        core = erode_mask(mask_bin)
        d_mask = shift_mask(mask_bin, dy, dx)
        bridge_full = create_bridge_mask(mask_bin, d_mask, bridge_width=12)

        core_only = ((core > 0) & (d_mask == 0)).astype(np.float32)
        decoy_only = ((d_mask > 0) & (mask_bin == 0)).astype(np.float32)
        bridge_only = ((bridge_full > 0) & (core == 0) & (d_mask == 0) & (mask_bin == 0)).astype(np.float32)
        suppress_only = ((mask_bin > 0) & (d_mask == 0)).astype(np.float32)

        return {
            "core": to_tensor(core_only),
            "bridge": to_tensor(bridge_only),
            "decoy": to_tensor(decoy_only),
            "suppress": to_tensor(suppress_only),
            "core_w": core_w,
            "bridge_w": bridge_w,
            "decoy_w": decoy_w,
            "suppress_w": suppress_w,
            "rank_w": rank_w,
            "score_w": score_w,
            "score_margin": 0.5,
            "support_margin": 0.0,
            "bridge_margin": 0.25,
            "decoy_margin": decoy_margin,
            "suppress_margin": suppress_margin,
            "rank_margin": rank_margin,
        }, d_mask

    # ── Frame 0: conditioning memory treatment ───────────────────────────
    # Frame 0 is privileged conditioning memory. For natural distractor
    # videos (e.g. cows), leave f0 clean so the correct anchor doesn't
    # interfere. For synthetic background decoys, add weak decoy pressure
    # to weaken the recovery anchor.
    if 0 in perturb_set:
        if is_natural_distractor:
            # Distractor mode: only loosen, don't add decoy term to f0
            td, dm = make_target_dict(
                masks_uint8[0],
                core_w=1.0, bridge_w=0.0, decoy_w=0.0,
                suppress_w=0.0, rank_w=0.0, score_w=0.0,
            )
        else:
            # Background mode: weak decoy on f0 to weaken recovery anchor
            td, dm = make_target_dict(
                masks_uint8[0],
                core_w=0.5, bridge_w=0.10, decoy_w=0.15,
                suppress_w=0.10, rank_w=0.10, score_w=0.0,
                decoy_margin=0.3, suppress_margin=0.15, rank_margin=0.2,
            )
        targets[("orig", 0)] = td
        decoy_masks_np[0] = dm

    # ── Per-frame targets based on role ──────────────────────────────────
    for orig_idx in sorted(perturb_set):
        if orig_idx == 0:
            continue
        mask = masks_uint8[min(orig_idx, T - 1)]

        is_pre_insert = any(orig_idx == s.after_original_idx for s in schedule)
        is_post_insert = any(orig_idx == s.after_original_idx + 1 for s in schedule)
        is_post_insert2 = any(orig_idx == s.after_original_idx + 2 for s in schedule)

        if is_pre_insert:
            td, dm = make_target_dict(
                mask,
                core_w=0.35,
                bridge_w=0.15,
                decoy_w=0.15,
                suppress_w=0.10,
                rank_w=0.10,
                score_w=0.15,
                decoy_margin=0.6,
                suppress_margin=0.25,
                rank_margin=0.35,
            )
        elif is_post_insert:
            td, dm = make_target_dict(
                mask,
                core_w=0.0,
                bridge_w=0.20,
                decoy_w=0.50,
                suppress_w=1.00,
                rank_w=1.00,
                score_w=0.30,
                suppress_margin=0.7,
                rank_margin=0.9,
            )
        elif is_post_insert2:
            td, dm = make_target_dict(
                mask,
                core_w=0.0,
                bridge_w=0.10,
                decoy_w=0.35,
                suppress_w=0.80,
                rank_w=0.80,
                score_w=0.25,
                suppress_margin=0.6,
                rank_margin=0.75,
            )
        else:
            td, dm = make_target_dict(
                mask,
                core_w=0.25,
                bridge_w=0.05,
                decoy_w=0.10,
                suppress_w=0.10,
                rank_w=0.05,
                score_w=0.10,
                decoy_margin=0.5,
                suppress_margin=0.2,
                rank_margin=0.25,
            )

        targets[("orig", orig_idx)] = td
        decoy_masks_np[orig_idx] = dm

    # ── Insert targets (use shared offset, consistent mask_after) ────────
    for slot_i, slot in enumerate(schedule):
        pos = slot.after_original_idx
        # Fix M2: use mask_after consistently (matches frame_after base)
        frame_after = frames_uint8[min(pos + 1, T - 1)]
        mask_after = masks_uint8[min(pos + 1, T - 1)]

        if slot.frame_type == "strong":
            td, dm = make_target_dict(
                mask_after,
                core_w=0.0,
                bridge_w=0.30,
                decoy_w=1.10,
                suppress_w=1.30,
                rank_w=1.40,
                score_w=0.50,
                decoy_margin=1.1,
                suppress_margin=1.0,
                rank_margin=1.25,
            )
        else:
            td, dm = make_target_dict(
                mask_after,
                core_w=0.0,
                bridge_w=0.20,
                decoy_w=1.00,
                suppress_w=1.00,
                rank_w=1.10,
                score_w=0.45,
                decoy_margin=1.0,
                suppress_margin=0.8,
                rank_margin=1.0,
            )

        # Fix C1: pass shared offset to base frame constructor
        if high_fidelity_insert:
            # Identity anchor = frame_prev (clean frame just BEFORE insertion
            # point). LPIPS reference in measure_fidelity.py also uses
            # frame_prev, so this keeps outside-edit pixels at LPIPS = 0.
            # Pilot B: also inpaint the true-object region from frame_prev so
            # SAM2 sees "object deleted at prev location + reappeared at decoy"
            # (retains memory-redirect signal).
            frame_prev = frames_uint8[pos]
            mask_prev = masks_uint8[pos]
            base = None
            edit_mask_np = None
            chosen_offset = offset
            for cand_off, _ in candidates:
                res = create_decoy_base_frame_hifi(
                    frame_prev, frame_after, mask_after, cand_off,
                    seam_dilate_px=seam_dilate_px,
                    safety_margin=safety_margin,
                    mask_prev=mask_prev,
                    inpaint_true_region=False,
                )
                if res is not None:
                    base, edit_mask_np = res
                    chosen_offset = cand_off
                    break
            if base is None:
                # Last-resort fallback: old pipeline (alpha-blend is possible).
                base = create_decoy_base_frame(frame_after, mask_after, offset)
                paste_region = shift_mask((mask_after > 0).astype(np.uint8),
                                          offset[0], offset[1])
                if seam_dilate_px > 0:
                    ker = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * seam_dilate_px + 1, 2 * seam_dilate_px + 1))
                    edit_mask_np = cv2.dilate(paste_region, ker, iterations=1)
                else:
                    edit_mask_np = paste_region
                edit_mask_np = (edit_mask_np > 0).astype(np.uint8)
            insert_edit_masks[slot_i] = edit_mask_np
        else:
            base = create_decoy_base_frame(frame_after, mask_after, offset)

        targets[("insert", slot_i)] = td
        insert_bases[slot_i] = base
        decoy_masks_np[f"insert_{slot_i}"] = dm

    return {
        "decoy_offset": offset,
        "is_natural_distractor": is_natural_distractor,
        "targets": targets,
        "decoy_masks": decoy_masks_np,
        "insert_bases": insert_bases,
        "insert_edit_masks": insert_edit_masks,
    }


def _relocate_single_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    dy: int, dx: int,
) -> np.ndarray:
    """Relocate the object in a single frame: inpaint true region, paste at decoy."""
    import cv2

    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return frame.copy()

    H, W = frame.shape[:2]
    inpaint_mask = (mask_bin * 255).astype(np.uint8)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    base = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB).astype(np.float32)

    obj_alpha = cv2.GaussianBlur(
        mask_bin.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    obj_alpha = np.clip(obj_alpha, 0.0, 1.0)
    obj_layer = frame.astype(np.float32) * obj_alpha[..., None]

    affine = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_alpha = cv2.warpAffine(
        obj_alpha, affine, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted_layer = cv2.warpAffine(
        obj_layer, affine, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    paste_alpha = shifted_alpha[..., None]
    synth = base * (1.0 - paste_alpha) + shifted_layer
    return np.clip(synth, 0, 255).astype(np.uint8)


def build_synthetic_decoy_video(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    decoy_offset: Tuple[int, int],
    schedule: Optional[List] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """Build a synthetic decoy video on the MODIFIED timeline.

    When schedule is provided, the teacher video includes synthetic inserted
    frames at the same positions as the attack schedule. This ensures the
    teacher memory features correspond 1:1 with the modified-video indices.

    Returns:
        (synth_frames, decoy_mask_f0, mod_to_orig):
          synth_frames: synthetic video matching modified timeline
          decoy_mask_f0: shifted f0 mask for teacher prompt
          mod_to_orig: mapping from modified index to original (-1 for inserts)
    """
    dy, dx = decoy_offset
    T = len(frames_uint8)

    # Build per-original-frame synthetic versions
    synth_originals = []
    for i in range(T):
        synth_originals.append(
            _relocate_single_frame(frames_uint8[i], masks_uint8[i], dy, dx))

    # Assemble on modified timeline (with inserts if schedule provided)
    if schedule is not None:
        insert_after = {}
        for si, slot in enumerate(schedule):
            insert_after.setdefault(slot.after_original_idx, []).append(si)

        synth_frames = []
        mod_to_orig = []
        for oi in range(T):
            synth_frames.append(synth_originals[oi])
            mod_to_orig.append(oi)
            if oi in insert_after:
                for si in insert_after[oi]:
                    # Insert: use synth version of frame_after
                    ref = min(oi + 1, T - 1)
                    synth_frames.append(synth_originals[ref])
                    mod_to_orig.append(-1)  # -1 = insert
    else:
        synth_frames = synth_originals
        mod_to_orig = list(range(T))

    decoy_mask_f0 = shift_mask(masks_uint8[0], dy, dx)
    return synth_frames, decoy_mask_f0, mod_to_orig


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
        alpha = max(0.55, 1.25 - rank * 0.08)
        mask = masks_uint8[min(oi, T - 1)]
        d_mask = shift_mask(mask, dy, dx)
        suppress = ((mask > 0) & (d_mask == 0)).astype(np.float32)
        decoy_only = ((d_mask > 0) & (mask == 0)).astype(np.float32)
        future_targets[oi] = {
            "core": torch.zeros(1, 1, H, W, device=device),
            "bridge": torch.zeros(1, 1, H, W, device=device),
            "decoy": torch.from_numpy(decoy_only).unsqueeze(0).unsqueeze(0).to(device),
            "suppress": torch.from_numpy(suppress).unsqueeze(0).unsqueeze(0).to(device),
            "core_w": 0.0,
            "bridge_w": 0.0,
            "decoy_w": alpha,
            "suppress_w": min(1.6, alpha + 0.40),
            "rank_w": min(1.3, alpha + 0.20),
            "score_w": 0.30,
            "score_margin": 0.5,
            "support_margin": 0.0,
            "bridge_margin": 0.0,
            "decoy_margin": 0.9,
            "suppress_margin": 0.6,
            "rank_margin": 0.8,
        }

    # ── PGD setup ────────────────────────────────────────────────────────
    n_steps = cfg.n_steps_strong
    stage1_end = int(n_steps * 0.2)
    stage2_end = int(n_steps * 0.8)
    # stage3: rest

    # Per-slot step sizes (floor proportional to epsilon, not absolute)
    alpha_ins = {}
    for si in insert_deltas:
        eps = insert_eps[si]
        alpha_ins[si] = max(eps / max(n_steps // 3, 1), eps * 0.1)
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
            for k in ("core", "bridge", "decoy", "suppress"):
                if k not in resized:
                    continue
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
                        frame_loss = decoy_target_loss(logits, td)
                        score_logits = out.get("object_score_logits") if isinstance(out, dict) else None
                        if score_logits is not None and float(td.get("score_w", 0.0)) > 0.0:
                            frame_loss = frame_loss + float(td["score_w"]) * object_score_positive_loss(
                                score_logits, margin=float(td.get("score_margin", 0.5)),
                            )
                        loss_write = loss_write + frame_loss
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
                            frame_loss = decoy_target_loss(logits, td)
                            score_logits = out.get("object_score_logits") if isinstance(out, dict) else None
                            if score_logits is not None and float(td.get("score_w", 0.0)) > 0.0:
                                frame_loss = frame_loss + float(td["score_w"]) * object_score_positive_loss(
                                    score_logits, margin=float(td.get("score_margin", 0.5)),
                                )
                            loss_write = loss_write + 1.5 * frame_loss
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
                frame_loss = decoy_target_loss(logits, td)
                score_logits = out.get("object_score_logits") if isinstance(out, dict) else None
                if score_logits is not None and float(td.get("score_w", 0.0)) > 0.0:
                    frame_loss = frame_loss + float(td["score_w"]) * object_score_positive_loss(
                        score_logits, margin=float(td.get("score_margin", 0.5)),
                    )
                loss_read = loss_read + w * frame_loss
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
            ssim_thresh = (cfg.ssim_threshold_strong if schedule[si].frame_type == "strong"
                           else cfg.ssim_threshold_weak)
            loss_quality = loss_quality + F.relu(ssim_thresh - sv)
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
        loss_total = loss_write + 1.3 * loss_read + cfg.lambda_quality * loss_quality

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_insert_deltas = {si: insert_deltas[si].detach().clone() for si in insert_deltas}
            best_orig_deltas = {oi: orig_deltas[oi].detach().clone() for oi in orig_deltas}

        loss_total.backward()

        # ── PGD step (only active params) ────────────────────────────
        with torch.no_grad():
            for si in active_insert:
                if insert_deltas[si].grad is not None:
                    insert_deltas[si].data -= alpha_ins[si] * insert_deltas[si].grad.sign()
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
