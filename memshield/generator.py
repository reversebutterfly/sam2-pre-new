"""
Adversarial Frame Generator v2: hybrid insert + perturb existing frames.

Key change from v1 (per GPT-5.4 review):
  - Also perturbs key ORIGINAL frames alongside inserted frames
  - Frame 0 (prompt frame) gets small perturbation (ε=2/255)
  - Frames adjacent to insertions get medium perturbation (ε=4/255)
  - Inserted frames keep full budget (ε=8/255 strong, 2/255 weak)
"""
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import MemShieldConfig
from .losses import compute_attack_loss, differentiable_ssim, fake_uint8_quantize
from .scheduler import InsertionSlot, build_modified_index_map
from .surrogate import SAM2Surrogate


def interpolate_base_frame(
    frames_x01: List[torch.Tensor],
    after_idx: int,
) -> torch.Tensor:
    """Create a base frame by blending neighbors."""
    idx_before = after_idx
    idx_after = min(after_idx + 1, len(frames_x01) - 1)
    return 0.5 * (frames_x01[idx_before] + frames_x01[idx_after])


def select_perturb_originals(
    schedule: List[InsertionSlot],
    T: int,
) -> Set[int]:
    """Select which original frames to perturb (hybrid approach).

    Strategy: perturb frame 0 (prompt) + frames adjacent to each insertion.
    """
    perturb_set = {0}  # Always perturb prompt frame
    for slot in schedule:
        pos = slot.after_original_idx
        # Frame before insertion
        if pos >= 0:
            perturb_set.add(pos)
        # Frame after insertion
        if pos + 1 < T:
            perturb_set.add(pos + 1)
        # One more frame after (first memory write after insertion)
        if pos + 2 < T:
            perturb_set.add(pos + 2)
    return perturb_set


def optimize_hybrid(
    surrogate: SAM2Surrogate,
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    schedule: List[InsertionSlot],
    cfg: MemShieldConfig,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict]:
    """Optimize inserted frames AND key original frames jointly via PGD.

    Returns:
        (inserted_frames_uint8: {slot_idx -> [H,W,3] uint8},
         perturbed_originals_uint8: {orig_idx -> [H,W,3] uint8},
         metrics: optimization history)
    """
    device = surrogate.device
    T = len(frames_uint8)
    H, W = frames_uint8[0].shape[:2]

    # Convert original frames to tensors
    frames_t = []
    for f in frames_uint8:
        t = torch.from_numpy(f).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frames_t.append(t.to(device))

    # Build index map
    idx_map = build_modified_index_map(T, schedule)

    # ── Insertion deltas ─────────────────────────────────────────────────────
    insert_base = {}
    insert_deltas = {}
    insert_eps = {}

    for slot_i, slot in enumerate(schedule):
        base = interpolate_base_frame(frames_t, slot.after_original_idx)
        insert_base[slot_i] = base.detach()
        insert_deltas[slot_i] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        insert_eps[slot_i] = cfg.epsilon_strong if slot.frame_type == "strong" else cfg.epsilon_weak

    # ── Original frame deltas (hybrid) ───────────────────────────────────────
    perturb_set = select_perturb_originals(schedule, T)
    orig_deltas = {}
    orig_eps = {}

    # Epsilon budget for original frames (smaller than inserted)
    eps_prompt = 2.0 / 255   # Frame 0: very small
    eps_adjacent = 4.0 / 255  # Adjacent frames: medium

    for orig_idx in perturb_set:
        orig_deltas[orig_idx] = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
        orig_eps[orig_idx] = eps_prompt if orig_idx == 0 else eps_adjacent

    # All trainable parameters
    all_params = list(insert_deltas.values()) + list(orig_deltas.values())
    if not all_params:
        return {}, {}, {"error": "no parameters to optimize"}

    # ── Evaluation setup ─────────────────────────────────────────────────────
    first_insert_orig = schedule[0].after_original_idx if schedule else 1
    eval_orig_start = first_insert_orig + 1
    eval_orig_end = min(T, eval_orig_start + cfg.future_horizon)
    eval_orig_indices = list(range(eval_orig_start, eval_orig_end))
    eval_mod_indices = [idx_map["orig_to_mod"][j] for j in eval_orig_indices]

    # ── PGD setup ────────────────────────────────────────────────────────────
    n_steps = cfg.n_steps_strong
    pgd_alpha_insert = max(cfg.epsilon_strong / max(n_steps // 3, 1), 1.0 / 255)
    pgd_alpha_orig = max(eps_adjacent / max(n_steps // 3, 1), 0.5 / 255)

    best_loss = float("inf")
    best_insert_deltas = {}
    best_orig_deltas = {}
    history = {"steps": [], "loss_attack": [], "loss_quality": [], "ssim_min": []}

    # Insertion-after lookup
    insert_after = {}
    for slot_i, slot in enumerate(schedule):
        insert_after.setdefault(slot.after_original_idx, []).append(slot_i)

    for step in range(n_steps):
        # Zero gradients
        for d in all_params:
            if d.grad is not None:
                d.grad.zero_()

        # Build modified video sequence with fake quantization
        mod_frames = []
        for orig_idx in range(T):
            if orig_idx in orig_deltas:
                frame = (frames_t[orig_idx] + orig_deltas[orig_idx]).clamp(0.0, 1.0)
                frame = fake_uint8_quantize(frame)  # Transport-aware
            else:
                frame = frames_t[orig_idx].detach()
            mod_frames.append(frame)

            if orig_idx in insert_after:
                for slot_i in insert_after[orig_idx]:
                    adv = (insert_base[slot_i] + insert_deltas[slot_i]).clamp(0.0, 1.0)
                    adv = fake_uint8_quantize(adv)  # Transport-aware
                    mod_frames.append(adv)

        # Forward through SAM2
        all_logits = surrogate.forward_video(mod_frames, masks_uint8[0])

        # Attack loss
        loss_attack = compute_attack_loss(
            all_logits, masks_uint8, eval_mod_indices, eval_orig_indices,
            device, cfg.persistence_weighting,
        )

        # Quality loss on ALL perturbed frames
        ssim_vals = []
        loss_quality = torch.tensor(0.0, device=device)
        n_qual = 0

        # Quality for inserted frames
        for slot_i in insert_deltas:
            base = insert_base[slot_i]
            adv = (base + insert_deltas[slot_i]).clamp(0.0, 1.0)
            threshold = (cfg.ssim_threshold_strong if schedule[slot_i].frame_type == "strong"
                         else cfg.ssim_threshold_weak)
            ssim_val = differentiable_ssim(base, adv)
            ssim_vals.append(ssim_val.item())
            loss_quality = loss_quality + F.relu(threshold - ssim_val)
            n_qual += 1

        # Quality for perturbed originals
        for orig_idx in orig_deltas:
            clean = frames_t[orig_idx]
            adv = (clean + orig_deltas[orig_idx]).clamp(0.0, 1.0)
            ssim_val = differentiable_ssim(clean, adv)
            ssim_vals.append(ssim_val.item())
            loss_quality = loss_quality + F.relu(0.97 - ssim_val)  # Tight quality for originals
            n_qual += 1

        if n_qual > 0:
            loss_quality = loss_quality / n_qual

        loss_total = loss_attack + cfg.lambda_quality * loss_quality

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_insert_deltas = {si: insert_deltas[si].detach().clone() for si in insert_deltas}
            best_orig_deltas = {oi: orig_deltas[oi].detach().clone() for oi in orig_deltas}

        loss_total.backward()

        # Sign-based PGD step
        with torch.no_grad():
            for slot_i in insert_deltas:
                if insert_deltas[slot_i].grad is not None:
                    insert_deltas[slot_i].data -= pgd_alpha_insert * insert_deltas[slot_i].grad.sign()
                    insert_deltas[slot_i].data.clamp_(-insert_eps[slot_i], insert_eps[slot_i])

            for orig_idx in orig_deltas:
                if orig_deltas[orig_idx].grad is not None:
                    orig_deltas[orig_idx].data -= pgd_alpha_orig * orig_deltas[orig_idx].grad.sign()
                    orig_deltas[orig_idx].data.clamp_(-orig_eps[orig_idx], orig_eps[orig_idx])

        ssim_min = min(ssim_vals) if ssim_vals else 0.0
        if step % 10 == 0 or step == n_steps - 1:
            d_ins = max(insert_deltas[si].abs().max().item() for si in insert_deltas) * 255 if insert_deltas else 0
            d_orig = max(orig_deltas[oi].abs().max().item() for oi in orig_deltas) * 255 if orig_deltas else 0
            print(f"  step {step:4d}  L_atk={loss_attack.item():.4f}  "
                  f"L_qual={loss_quality.item():.4f}  SSIM={ssim_min:.4f}  "
                  f"δ_ins={d_ins:.1f}  δ_orig={d_orig:.1f}/255")

        history["steps"].append(step)
        history["loss_attack"].append(loss_attack.item())
        history["loss_quality"].append(loss_quality.item())
        history["ssim_min"].append(ssim_min)

    # Convert to uint8 with proper rounding (not truncation!)
    inserted_uint8 = {}
    for slot_i in best_insert_deltas:
        adv = (insert_base[slot_i] + best_insert_deltas[slot_i]).clamp(0.0, 1.0)
        arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        inserted_uint8[slot_i] = np.rint(arr).clip(0, 255).astype(np.uint8)

    perturbed_orig_uint8 = {}
    for orig_idx in best_orig_deltas:
        adv = (frames_t[orig_idx] + best_orig_deltas[orig_idx]).clamp(0.0, 1.0)
        arr = adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        perturbed_orig_uint8[orig_idx] = np.rint(arr).clip(0, 255).astype(np.uint8)

    metrics = {
        "best_loss": best_loss,
        "final_loss_attack": history["loss_attack"][-1] if history["loss_attack"] else None,
        "final_ssim_min": history["ssim_min"][-1] if history["ssim_min"] else None,
        "n_steps": n_steps,
        "n_inserted": len(schedule),
        "n_perturbed_originals": len(perturb_set),
        "perturbed_original_indices": sorted(perturb_set),
        "schedule": [(s.after_original_idx, s.frame_type, s.reason) for s in schedule],
        "history": history,
    }
    return inserted_uint8, perturbed_orig_uint8, metrics
