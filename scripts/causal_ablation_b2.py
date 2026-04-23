"""B2 causal ablation (auto-review Round 4): drop non-cond bank at eval.

Runs 4 configurations on one clip and reports per-frame J for the eval
window (T_prefix .. T_prefix + eval_window - 1, orig-time indices).

    C1  clean,    normal   (baseline)
    C2  clean,    ablated  (non-cond bank dropped on eval frames)
    C3  attacked, normal   (matches R003 attacked eval; sanity check)
    C4  attacked, ablated  (tests whether inserts route harm via bank)

Key diagnostic signal (Codex R3 / B2): compare C1 vs C2. If J is
near-identical (delta < ~0.02 per frame, mean drop < 0.05), the
non-conditioning memory bank is a minor input to SAM2's segmentation
on this clip. Consequence: poisoning the bank via decoy-insert attacks
cannot meaningfully damage J — the attack surface is wrong.

Secondary signal: compare C3 vs C4. With bank dropped, inserts have no
routing path; if C3 ~ C4, the observed (R003) zero J-drop is because
the inserts were never behaviorally effective anyway.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/causal_ablation_b2.py \\
        --run_dir runs/r003 --output_json runs/r003/b2_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memshield.ablation_hook import DropNonCondBankHook
from memshield.eval_v2 import jaccard


def load_davis_clip(davis_root: Path, clip: str, n_frames: int
                    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    jpg_dir = davis_root / "JPEGImages" / "480p" / clip
    ann_dir = davis_root / "Annotations" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    anns = sorted(ann_dir.glob("*.png"))[:n_frames]
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])
    masks = [(np.array(Image.open(p)) > 0).astype(np.uint8) for p in anns]
    return frames, masks, masks[0]


def stage_frames(frames: np.ndarray, td: Path) -> None:
    td.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(td / f"{i:05d}.jpg", quality=95)


def propagate_with_optional_ablation(predictor, frames: np.ndarray,
                                     mask0: np.ndarray, device: str,
                                     ablate_frame_ids: Set[int]
                                     ) -> List[np.ndarray]:
    """Run propagate_in_video. If ablate_frame_ids is non-empty, wrap the
    run with DropNonCondBankHook so those frames see empty non-cond bank."""
    H, W = frames.shape[1], frames.shape[2]
    with tempfile.TemporaryDirectory() as td:
        stage_frames(frames, Path(td))
        state = predictor.init_state(video_path=td)
        predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1,
            mask=torch.from_numpy(mask0).to(device).bool(),
        )
        masks: List[np.ndarray] = [None] * len(frames)
        if ablate_frame_ids:
            hook_ctx = DropNonCondBankHook(predictor, ablate_frame_ids)
        else:
            from contextlib import nullcontext
            hook_ctx = nullcontext()
        with hook_ctx as hook:
            for fid, _ids, vrm in predictor.propagate_in_video(state):
                m = (vrm[0, 0].float().sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                masks[fid] = m
            fires = getattr(hook, "fires", None) if hook is not None else None
    for i in range(len(masks)):
        if masks[i] is None:
            masks[i] = np.zeros((H, W), dtype=np.uint8)
    return masks, fires


def j_per_frame(pred: List[np.ndarray], gt: List[np.ndarray],
                eval_start: int, eval_end: int) -> List[float]:
    out = []
    for t in range(eval_start, eval_end):
        p = pred[t]
        g = gt[t]
        if p.shape != g.shape:
            p_img = Image.fromarray((p > 0).astype(np.uint8) * 255)
            p_img = p_img.resize((g.shape[1], g.shape[0]), Image.NEAREST)
            p = (np.array(p_img) > 0).astype(np.uint8)
        out.append(float(jaccard(p, g)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Dir with modified_video.npy + diagnostics.json "
                         "(used for C3/C4 attacked configs).")
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
    n_needed = T_prefix_orig + eval_window
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_json) if args.output_json else run_dir / "b2_ablation.json"

    print(f"[b2] run_dir={run_dir} clip={clip} K_ins={K_ins} "
          f"T_mod={T_mod} T_prefix_orig={T_prefix_orig} device={device}")

    # Load DAVIS clean + GT
    davis_root = Path(args.davis_root)
    frames_clean, gt_masks, mask0 = load_davis_clip(davis_root, clip, n_needed)

    # Load attacked (modified prefix + clean suffix)
    mod_prefix = np.load(run_dir / "modified_video.npy")
    assert mod_prefix.shape[0] == T_mod
    assert mod_prefix.dtype == np.uint8
    attacked = np.concatenate(
        [mod_prefix, frames_clean[T_prefix_orig:T_prefix_orig + eval_window]],
        axis=0,
    )
    T_atk = attacked.shape[0]

    # Predictor
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, args.sam2_ckpt, device=device)
    predictor.eval()

    # Ablation target ids:
    #   - clean: eval window is orig frames [T_prefix_orig .. T_prefix_orig+U)
    #   - attacked: eval window is mod frames [T_mod .. T_mod+U)
    clean_eval_ids = set(range(T_prefix_orig, T_prefix_orig + eval_window))
    atk_eval_ids = set(range(T_mod, T_mod + eval_window))

    configs = [
        ("C1_clean_normal",    frames_clean, set(),           T_prefix_orig),
        ("C2_clean_ablated",   frames_clean, clean_eval_ids,  T_prefix_orig),
        ("C3_attacked_normal", attacked,     set(),           T_mod),
        ("C4_attacked_ablated",attacked,     atk_eval_ids,    T_mod),
    ]

    results = {
        "run_dir": str(run_dir), "clip": clip, "K_ins": K_ins,
        "T_mod": T_mod, "T_prefix_orig": T_prefix_orig,
        "eval_window": eval_window, "configs": {},
    }

    for name, frames, abl_ids, eval_start in configs:
        print(f"\n[b2] === {name} === ablate_ids={sorted(abl_ids) or 'none'} "
              f"eval=[{eval_start}, {eval_start+eval_window})")
        t0 = time.time()
        preds, fires = propagate_with_optional_ablation(
            predictor, frames, mask0, device, abl_ids)
        t_prop = time.time() - t0

        # Eval GT for THIS config's timeline:
        # - For clean frames, gt_masks (orig) aligns directly.
        # - For attacked timeline, the eval frames at mod indices
        #   [T_mod..T_mod+U) correspond to clean orig frames
        #   [T_prefix_orig..T_prefix_orig+U). Build padded gt aligned
        #   to attacked timeline.
        if frames is frames_clean:
            gt_aligned = gt_masks
        else:
            H_gt, W_gt = gt_masks[0].shape
            gt_aligned = [np.zeros((H_gt, W_gt), dtype=np.uint8)] * T_mod + \
                         gt_masks[T_prefix_orig:T_prefix_orig + eval_window]
            assert len(gt_aligned) == T_atk

        j_vals = j_per_frame(
            preds, gt_aligned, eval_start, eval_start + eval_window)
        mean_j = float(np.mean(j_vals))
        print(f"[b2]   propagate {t_prop:.1f}s; ablation fires={fires}; "
              f"mean J={mean_j:.4f} per-frame={[f'{x:.3f}' for x in j_vals]}")

        results["configs"][name] = {
            "ablate_ids": sorted(abl_ids),
            "eval_start": eval_start,
            "eval_end": eval_start + eval_window,
            "j_per_frame": j_vals,
            "mean_j": mean_j,
            "propagate_sec": t_prop,
            "hook_fires": fires,
        }

    # Summary deltas
    c = results["configs"]
    delta_clean = c["C1_clean_normal"]["mean_j"] - c["C2_clean_ablated"]["mean_j"]
    delta_atk   = c["C3_attacked_normal"]["mean_j"] - c["C4_attacked_ablated"]["mean_j"]
    results["summary"] = {
        "j_drop_from_bank_ablation_on_clean":    delta_clean,
        "j_drop_from_bank_ablation_on_attacked": delta_atk,
        "interpretation_hint":
            "delta_clean < 0.05 → bank is behaviorally marginal; "
            "decoy-insert attacks that poison this bank cannot meaningfully "
            "damage J regardless of attention weight routing.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== B2 summary ===")
    print(f"  C1 clean,    normal   mean_J = {c['C1_clean_normal']['mean_j']:.4f}")
    print(f"  C2 clean,    ablated  mean_J = {c['C2_clean_ablated']['mean_j']:.4f}  "
          f"(delta {delta_clean:+.4f})")
    print(f"  C3 attacked, normal   mean_J = {c['C3_attacked_normal']['mean_j']:.4f}")
    print(f"  C4 attacked, ablated  mean_J = {c['C4_attacked_ablated']['mean_j']:.4f}  "
          f"(delta {delta_atk:+.4f})")
    print(f"\n[b2] wrote {out_path}")
    if abs(delta_clean) < 0.05:
        print("[b2] VERDICT: non-cond bank is behaviorally marginal on clean "
              f"(|delta|={abs(delta_clean):.4f} < 0.05). Decoy attacks that "
              "poison this bank cannot meaningfully damage J.")
    else:
        print(f"[b2] VERDICT: non-cond bank matters on clean "
              f"(|delta|={abs(delta_clean):.4f} ≥ 0.05). Decoy has a "
              "theoretical attack surface; current inserts' feature content "
              "is the next thing to audit.")


if __name__ == "__main__":
    main()
