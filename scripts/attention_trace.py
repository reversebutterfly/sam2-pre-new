"""D1: Attention trace diagnostic (auto-review Round 2 fix).

Addresses Codex's weakness #4 ("mechanism unproven"). Loads a run's
`modified_video.npy`, runs the SAM2VideoAdapter in no-grad forward mode,
and dumps `probe.P_u_by_frame` — the 3-bin softmax decomposition
[A_insert, A_recent, A_other] of memory-attention mass per eval frame
per memory-attention layer.

Interpretation guide:
  * A_insert ≈ 1: eval frame's query attends almost entirely to
    poisoned insert slots. Attack should be devastating.
  * A_insert ≈ 0 and A_recent ≈ 1: the FIFO has already evicted
    inserts and/or the query prefers the most recent (clean) memory.
    Self-heal is happening.
  * A_insert > 0 but J-drop ≈ 0: inserts ARE attended but their
    content does not produce a mis-segmentation — hypothesis (C) from
    the review, "inserts are too OOD to be useful even when attended".
  * A_insert ≈ 0 AND J-drop ≈ 0: mechanism doesn't fire — the bank
    poisoning doesn't land. The proposal's causal thesis is wrong.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/attention_trace.py \\
        --run_dir runs/r003 --output_json runs/r003/attention_trace.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memshield.optimize_v2 import OptimizeConfig, VideoBundle
from memshield.sam2_forward_adapter import SAM2VideoAdapter
from memshield.scheduler import compute_schedule_v2


def load_davis_clip(davis_root: Path, clip: str, n_frames: int):
    jpg_dir = davis_root / "JPEGImages" / "480p" / clip
    ann_dir = davis_root / "Annotations" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    anns = sorted(ann_dir.glob("*.png"))[:n_frames]
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])
    masks = [(np.array(Image.open(p)) > 0).astype(np.uint8) for p in anns]
    return frames, masks


def clean_sam2_forward_simple(predictor, frames, mask0, device):
    import tempfile
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(Path(td) / f"{i:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=td)
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1,
            mask=torch.from_numpy(mask0).to(device).bool(),
        )
        masks = [None] * len(frames)
        for fid, _, vrm in predictor.propagate_in_video(state):
            m = (vrm[0, 0].float().sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            masks[fid] = m
    for i in range(len(masks)):
        if masks[i] is None:
            masks[i] = np.zeros((H_vid, W_vid), dtype=np.uint8)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--davis_root", default=str(REPO_ROOT / "data" / "davis"))
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt",
                    default=str(REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval_window", type=int, default=7)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    diag = json.loads((run_dir / "diagnostics.json").read_text())
    clip = diag["clip"]
    K_ins = int(diag["K_ins"])
    T_mod = int(diag["schedule"]["T_mod"])
    T_prefix_orig = T_mod - K_ins
    eval_window = args.eval_window
    T_full_orig = T_prefix_orig + eval_window
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_json) if args.output_json else run_dir / "attention_trace.json"

    print(f"[attn] run_dir={run_dir} clip={clip} K_ins={K_ins} "
          f"T_mod={T_mod} T_prefix_orig={T_prefix_orig}")

    # Load modified video
    mod_np = np.load(run_dir / "modified_video.npy")                         # [T_mod, H, W, 3] uint8
    assert mod_np.shape[0] == T_mod

    # DAVIS + clean
    davis_root = Path(args.davis_root)
    frames, gt_masks = load_davis_clip(davis_root, clip, n_frames=T_full_orig)
    H_vid, W_vid = frames.shape[1], frames.shape[2]
    mask0 = gt_masks[0]

    # Build predictor + clean SAM2 forward → masks_gt / C_u
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device)
    predictor.eval()

    print("[attn] running clean SAM2 forward (for C_u) ...")
    clean_masks = clean_sam2_forward_simple(predictor, frames, mask0, device)
    masks_gt = np.stack(clean_masks[:T_full_orig], axis=0)
    C_u = [clean_masks[T_prefix_orig + i] for i in range(eval_window)]

    # Schedule
    schedule = compute_schedule_v2(
        T_prefix_orig=T_prefix_orig, num_maskmem=7, K_ins=K_ins,
        variant=diag["schedule"]["variant"],
    )
    assert schedule.T_prefix_mod == T_mod
    assert schedule.w_positions == diag["schedule"]["w_positions"]

    # Bundle — adapter only needs frames_orig, schedule, C_u, masks_gt.
    # insert_bases/edit_masks/D_ins/C_ins/ROI_ins are only used by
    # build_modified_video and l_loss — irrelevant in a forward-only probe.
    # Use zeros of correct shape so VideoBundle's dataclass validation passes.
    H_nat, W_nat = mask0.shape
    zeros_img = np.zeros((H_nat, W_nat, 3), dtype=np.uint8)
    zeros_mask = np.zeros((H_nat, W_nat), dtype=np.uint8)
    bundle = VideoBundle(
        frames_orig=frames, masks_gt=masks_gt, schedule=schedule,
        insert_bases=[zeros_img] * K_ins,
        edit_masks=[zeros_mask] * K_ins,
        decoy_offset=tuple(diag["decoy_offset"]),
        D_ins=[zeros_mask] * K_ins,
        C_ins=[zeros_mask] * K_ins,
        ROI_ins=[zeros_mask] * K_ins,
        C_u=C_u,
    )

    cfg = OptimizeConfig(
        K_ins=K_ins, num_maskmem=7,
        T_prefix_orig=T_prefix_orig,
        eval_window_size=eval_window,
        stale_window_size=eval_window,   # capture P_u_list for ALL eval frames
        n_steps=1,  # unused in forward-only
        device=device,
    )

    adapter = SAM2VideoAdapter(
        predictor=predictor, cfg=cfg,
        first_frame_mask_video_res=mask0,
        video_H=H_vid, video_W=W_vid,
    )
    adapter.prepare_from_clean(bundle)
    print(f"[attn] adapter ready; probe.layer_indices={adapter.probe.layer_indices}")

    # Move modified video to device as [T_mod, H, W, 3] in [0,1]
    mod_t = torch.from_numpy(mod_np).to(device).float() / 255.0

    print("[attn] running probe-enabled forward on modified_video ...")
    t0 = time.time()
    with torch.no_grad():
        out = adapter(mod_t, mode="clean", cfg=cfg, bundle=bundle)
    t_fwd = time.time() - t0
    print(f"[attn] forward done in {t_fwd:.1f}s")

    # Extract per-frame P_u across layers
    result = {
        "run_dir": str(run_dir),
        "clip": clip,
        "K_ins": K_ins,
        "T_mod": T_mod,
        "T_prefix_orig": T_prefix_orig,
        "eval_frames_mod_time": list(range(T_mod, T_mod + eval_window)),
        "w_positions": schedule.w_positions,
        "layer_indices": list(adapter.probe.layer_indices),
        "P_u_by_eval_frame": {},
    }
    last_layer = adapter.probe.layer_indices[-1]
    print("\n=== attention trace (per eval frame, last memory-attention layer) ===")
    print(f"{'mod_fid':>8s} {'A_ins':>8s} {'A_rec':>8s} {'A_other':>8s}  all-layer mean")
    for fid in range(T_mod, T_mod + eval_window):
        layer_dict = adapter.probe.P_u_by_frame.get(fid, {})
        per_layer = {}
        vals_last = None
        vals_mean = []
        for li in adapter.probe.layer_indices:
            P = layer_dict.get(li)
            if P is None:
                per_layer[li] = None
                continue
            P_np = [float(P[0]), float(P[1]), float(P[2])]
            per_layer[li] = P_np
            vals_mean.append(P_np)
            if li == last_layer:
                vals_last = P_np
        result["P_u_by_eval_frame"][fid] = per_layer
        if vals_last is not None:
            mean_vec = [float(np.mean([v[i] for v in vals_mean])) for i in range(3)]
            print(f"{fid:>8d} {vals_last[0]:>8.4f} {vals_last[1]:>8.4f} {vals_last[2]:>8.4f}  "
                  f"mean=[{mean_vec[0]:.4f}, {mean_vec[1]:.4f}, {mean_vec[2]:.4f}]")
        else:
            print(f"{fid:>8d}  no probe fire (frame not in fg_mask_by_frame?)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[attn] wrote {out_path}")


if __name__ == "__main__":
    main()
