"""R002 gate run (EXPERIMENT_TRACKER): Full, dog, K_ins=1, 200 steps, LPIPS ON.

R002 differs from R001 only in:
    * n_steps: 50 → 200
    * stages: 10/20/20 → 40/40/120 (per EXPERIMENT_PLAN Block 1 default)
    * lpips_fn: None → LPIPS-alex with budget 0.10 (FINAL_PROPOSAL target)
    * output: runs/r001/ → runs/r002/

Gate criterion (per refine-logs): attack achieves target J-drop while
staying within the fidelity budget (insert LPIPS ≤ 0.10). This pilot
replaces R001's sanity-only verdict with a proper fidelity-constrained
run that can be referenced in the paper's main table (Block 1).

Shares all clean-SAM2 / ProPainter-base logic with run_pilot_r001 — we
import those helpers directly so this file is a thin driver.

Usage (Pro 6000):
    conda activate memshield
    cd ~/sam2-pre-new
    CUDA_VISIBLE_DEVICES=1 python -m memshield.run_pilot_r002 \\
        --clip dog --n_steps 200 --stage1_end 40 --stage2_end 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from memshield.optimize_v2 import OptimizeConfig, VideoBundle, optimize_unified_v2
from memshield.scheduler import compute_schedule_v2
from memshield.sam2_forward_adapter import SAM2VideoAdapter
from memshield.propainter_base import create_insert_base, is_propainter_available
from memshield.run_pilot_r001 import (
    load_clip, clean_sam2_forward, shift_mask, dilate_u8,
    REPO_ROOT, DAVIS_ROOT, CHECKPOINT, CONFIG,
)


# -----------------------------------------------------------------------------
# LPIPS adapter
# -----------------------------------------------------------------------------


def build_lpips_fn(device: str):
    """Wrap `lpips.LPIPS(net='alex')` to match optimize_v2's contract:
    f(x_ins [H,W,3] ∈ [0,1], f_prev [H,W,3] ∈ [0,1]) -> scalar Tensor.

    The LPIPS library expects `[B, 3, H, W]` in `[-1, 1]`. We keep the
    model in eval mode with frozen params; LPIPS passes gradients through
    its feature extractors even in eval mode.
    """
    import lpips as _lpips
    model = _lpips.LPIPS(net="alex", verbose=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def lpips_fn(x_ins: torch.Tensor, f_prev: torch.Tensor) -> torch.Tensor:
        x = x_ins.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        y = f_prev.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        return model(x, y).squeeze()

    return lpips_fn


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="dog")
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--stage1_end", type=int, default=40)
    parser.add_argument("--stage2_end", type=int, default=80)
    parser.add_argument("--lpips_budget", type=float, default=0.10)
    parser.add_argument("--K_ins", type=int, default=1)
    parser.add_argument("--T_prefix", type=int, default=15)
    parser.add_argument("--eval_window", type=int, default=7)
    parser.add_argument("--decoy_dy", type=int, default=0)
    parser.add_argument("--decoy_dx", type=int, default=80)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "runs" / "r002"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[r002] device={device} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    assert CHECKPOINT.exists(), f"missing checkpoint: {CHECKPOINT}"
    assert DAVIS_ROOT.exists(), f"missing DAVIS: {DAVIS_ROOT}"

    T_full = args.T_prefix + args.eval_window
    frames, mask0 = load_clip(args.clip, n_frames=T_full)
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    print(f"[r002] loaded {args.clip}: frames {frames.shape} mask0 {mask0.shape}")

    # --- SAM2 predictor + clean forward --------------------------------------
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device)
    predictor.eval()

    print("[r002] clean SAM2 forward ...")
    t0 = time.time()
    clean_masks = clean_sam2_forward(predictor, frames, mask0, device)
    print(f"[r002] clean forward done in {time.time()-t0:.1f}s; "
          f"fg area mask0={int(mask0.sum())} "
          f"mask[-1]={int(clean_masks[-1].sum())}")

    # --- schedule + ProPainter insert base -----------------------------------
    schedule = compute_schedule_v2(
        T_prefix_orig=args.T_prefix, num_maskmem=7, K_ins=args.K_ins,
        variant="canonical",
    )
    print(f"[r002] schedule: w_positions={schedule.w_positions} "
          f"T_mod={schedule.T_prefix_mod}")

    assert is_propainter_available(), "ProPainter not installed on this machine"
    decoy_offset = (args.decoy_dy, args.decoy_dx)
    insert_bases:  List[np.ndarray] = []
    edit_masks:    List[np.ndarray] = []
    D_ins_list:    List[np.ndarray] = []
    C_ins_list:    List[np.ndarray] = []
    ROI_ins_list:  List[np.ndarray] = []

    for k, slot in enumerate(schedule.slots):
        o_after = slot.o_after
        frame_prev = frames[o_after]
        frame_after = frames[o_after + 1]
        mask_prev = clean_masks[o_after]
        mask_after = clean_masks[o_after + 1]
        print(f"[r002] ProPainter slot k={k}: o_after={o_after}, "
              f"fg_prev={int(mask_prev.sum())}, fg_after={int(mask_after.sum())}")
        result = create_insert_base(
            strategy="propainter",
            frame_prev=frame_prev, frame_after=frame_after,
            mask_prev=mask_prev, mask_after=mask_after,
            decoy_offset=decoy_offset,
            seam_dilate_px=5, safety_margin=8, feather_px=3,
        )
        if result is None:
            raise RuntimeError(
                f"ProPainter insert-base returned None at slot {k}")
        base, edit_mask = result
        insert_bases.append(base)
        edit_masks.append(edit_mask)
        C = (mask_after > 0).astype(np.uint8)
        D = shift_mask(C, *decoy_offset)
        ROI = dilate_u8(((C | D) > 0).astype(np.uint8), radius=5)
        D_ins_list.append(D)
        C_ins_list.append(C)
        ROI_ins_list.append(ROI)

    C_u = [clean_masks[args.T_prefix + i] for i in range(args.eval_window)]
    masks_gt = np.stack(clean_masks[:T_full], axis=0)

    # --- LPIPS adapter -------------------------------------------------------
    print(f"[r002] loading LPIPS(alex) on {device} ...")
    lpips_fn = build_lpips_fn(device)
    # Smoke-check the adapter on a trivial pair so we fail fast on shape/dtype
    # mismatches BEFORE the 200-step loop.
    with torch.no_grad():
        probe_x = torch.rand(H_vid, W_vid, 3, device=device)
        probe_y = probe_x.clone()
        val = float(lpips_fn(probe_x, probe_y).item())
        assert np.isfinite(val), f"LPIPS self-pair not finite: {val}"
        print(f"[r002] lpips_fn sanity: identical inputs → {val:.4f} (should be ~0)")

    # --- cfg / adapter -------------------------------------------------------
    cfg = OptimizeConfig(
        K_ins=args.K_ins, num_maskmem=7,
        T_prefix_orig=args.T_prefix,
        eval_window_size=args.eval_window, stale_window_size=3,
        n_steps=args.n_steps,
        stage1_end=args.stage1_end, stage2_end=args.stage2_end,
        stage3_delta_per_nu_ratio=2,
        lpips_budget=args.lpips_budget,
        lagrange_update_every=10, log_every=10, device=device,
    )

    adapter = SAM2VideoAdapter(
        predictor=predictor, cfg=cfg,
        first_frame_mask_video_res=mask0,
        video_H=H_vid, video_W=W_vid,
    )
    prep_bundle = VideoBundle(
        frames_orig=frames, masks_gt=masks_gt, schedule=schedule,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=D_ins_list, C_ins=C_ins_list, ROI_ins=ROI_ins_list, C_u=C_u,
    )
    adapter.prepare_from_clean(prep_bundle)
    print(f"[r002] adapter ready; HW_mem={adapter.HW_mem}")

    # --- run optimize --------------------------------------------------------
    print(f"[r002] optimize_unified_v2 for {args.n_steps} steps "
          f"(stages {args.stage1_end}/{args.stage2_end-args.stage1_end}/"
          f"{args.n_steps-args.stage2_end}, LPIPS budget={args.lpips_budget})")
    t0 = time.time()
    final, diag = optimize_unified_v2(
        frames_orig=frames, masks_gt=masks_gt,
        sam2_forward_fn=adapter, cfg=cfg,
        insert_bases=insert_bases, edit_masks=edit_masks,
        decoy_offset=decoy_offset,
        D_ins=D_ins_list, C_ins=C_ins_list, ROI_ins=ROI_ins_list, C_u=C_u,
        lpips_fn=lpips_fn,
    )
    elapsed = time.time() - t0

    assert final.dtype == np.uint8
    history = diag["history"]
    assert len(history) == args.n_steps
    finite = all(
        all(np.isfinite(v) for k, v in h.items() if isinstance(v, (int, float)))
        for h in history
    )
    if not finite:
        offenders = [
            (h["step"], k) for h in history for k, v in h.items()
            if isinstance(v, (int, float)) and not np.isfinite(v)
        ]
        raise RuntimeError(f"non-finite loss values: {offenders[:5]}")

    # --- save artifacts ------------------------------------------------------
    np.save(out_dir / "modified_video.npy", final)
    np.save(out_dir / "final_nu.npy", diag["final_nu"])
    np.save(out_dir / "final_delta.npy", diag["final_delta"])
    # (final LPIPS measurement deferred until after save so we can include it)

    # Measure final insert LPIPS directly (history does not record it).
    final_nu_t = torch.from_numpy(diag["final_nu"]).to(device)
    final_lpips_per_k: List[float] = []
    with torch.no_grad():
        for k, slot in enumerate(schedule.slots):
            base_t = torch.from_numpy(insert_bases[k]).to(device).float() / 255.0
            edit_t = torch.from_numpy(edit_masks[k]).to(device).float().unsqueeze(-1)
            x_ins = (base_t + final_nu_t[k] * edit_t).clamp(0.0, 1.0)
            f_prev = torch.from_numpy(frames[slot.o_after]).to(device).float() / 255.0
            final_lpips_per_k.append(float(lpips_fn(x_ins, f_prev).item()))
    final_lpips_max = max(final_lpips_per_k) if final_lpips_per_k else float("nan")

    stages = [h["stage"] for h in history]
    nu_L1 = float(np.abs(diag["final_nu"]).mean())
    delta_L1 = float(np.abs(diag["final_delta"]).mean())

    json.dump(
        {
            "clip": args.clip, "n_steps": args.n_steps,
            "K_ins": args.K_ins, "seed": args.seed,
            "decoy_offset": list(decoy_offset),
            "lpips_budget": args.lpips_budget,
            "schedule": diag["schedule"],
            "stage_boundaries": list(diag["stage_boundaries"]),
            "mu_nu_final": diag["mu_nu_final"],
            "final_lpips_per_k": final_lpips_per_k,
            "final_lpips_max": final_lpips_max,
            "final_nu_L1": nu_L1,
            "final_delta_L1": delta_L1,
            "elapsed_sec": elapsed,
            "history": history,
        },
        open(out_dir / "diagnostics.json", "w"), indent=2, default=float,
    )

    print(f"[r002] DONE in {elapsed:.1f}s  "
          f"nu_L1={nu_L1:.4f} delta_L1={delta_L1:.4f} "
          f"mu_nu_final={diag['mu_nu_final']:.3f} "
          f"final_lpips_max={final_lpips_max:.4f} (budget={args.lpips_budget})")
    print(f"[r002] final_lpips_per_k={[f'{x:.4f}' for x in final_lpips_per_k]}")
    print(f"[r002] stages summary: "
          f"stage1={stages.count(1)} stage2={stages.count(2)} stage3={stages.count(3)}")
    print(f"[r002] artifacts → {out_dir}")
    print("[r002] RUN COMPLETE — next step: scripts/eval_memshield_v2.py "
          f"--run_dir {out_dir}")


if __name__ == "__main__":
    main()
