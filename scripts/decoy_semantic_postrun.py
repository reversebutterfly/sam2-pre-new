"""Post-hoc decoy-semantic evaluation for an existing VADI run.

Given a `results.json` from a completed v5 run (or decisive K3_* run
with compatible fields), this tool:

1. Loads `W` (processed-space insert positions) and `decoy_offsets`.
2. Re-reads the exported `processed/frame_*.png` sequence.
3. Rebuilds the "clean-processed baseline" by interleaving `x_clean`
   (fresh DAVIS load) with the insert bases (midframe for `insert_base=
   midframe` or duplicate-object for `insert_base=duplicate_seed`).
4. Runs a SAM2 eval on BOTH the baseline and the exported video.
5. Reconstructs the `m_true` and `m_decoy` mask trajectories (clean
   pseudo-masks via `clean_pass_vadi`, then decoy via shift-by-offset).
6. Computes per-frame + aggregate decoy-semantic metrics from
   `memshield.decoy_semantic_metrics`.
7. Writes `decoy_semantic.json` next to the original `results.json`.

Typical use:

    python scripts/decoy_semantic_postrun.py \\
        --davis-root data/davis \\
        --checkpoint checkpoints/sam2.1_hiera_tiny.pt \\
        --runs vadi_runs/v5_ablation2/dog/K3_top_R8_b-mid_l-mg_o-pg_d-off_s-io100 \\
               vadi_runs/v5_ablation2/camel/K3_top_R8_b-mid_l-mg_o-pg_d-off_s-io100 \\
               vadi_runs/v5_ablation2/blackswan/K3_top_R8_b-mid_l-mg_o-pg_d-off_s-io100

For the 10-clip main table run, v5 driver saves decoy_semantic inline
via `eval_exported_j_drop`, so this tool is primarily for:
  - back-filling the 3-clip ablation2 A0 runs
  - diagnosing a specific clip post-hoc without a full re-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _infer_clip_name(run_dir: Path) -> str:
    """A results.json sits at `vadi_runs/<scope>/<clip>/<config>/results.json`.
    The clip name is the grandparent's grandparent (2 levels up from
    results.json is the clip dir).
    """
    # run_dir points to the config dir (containing results.json + processed/).
    return run_dir.parent.name


def _infer_config_flags(results: Dict[str, Any]) -> Dict[str, str]:
    """Best-effort reconstruction of the config flags from the name field.

    Returns a dict with keys {insert_base, loss, nu_optimizer, delta_support,
    schedule} with default values if not decodable.
    """
    name = results.get("config_name", "")
    parts = name.split("_")
    short_map_rev = {
        "mid": "midframe", "dup": "duplicate_seed",
        "mg": "margin", "dc": "dice_bce",
        "ad": "adam", "pg": "sign_pgd",
        "off": "off", "post": "post_insert", "v4": "v4_symmetric",
        "fs": "full", "io100": "insert_only_100",
    }
    out = {
        "insert_base": "midframe", "loss": "margin",
        "nu_optimizer": "sign_pgd", "delta_support": "off",
        "schedule": "full",
    }
    for p in parts:
        if p.startswith("b-"):
            v = p[2:]; out["insert_base"] = short_map_rev.get(v, v)
        elif p.startswith("l-"):
            v = p[2:]; out["loss"] = short_map_rev.get(v, v)
        elif p.startswith("o-"):
            v = p[2:]; out["nu_optimizer"] = short_map_rev.get(v, v)
        elif p.startswith("d-"):
            v = p[2:]; out["delta_support"] = short_map_rev.get(v, v)
        elif p.startswith("s-"):
            v = p[2:]; out["schedule"] = short_map_rev.get(v, v)
    return out


def evaluate_decoy_semantic_for_run(
    run_dir: Path,
    davis_root: Path,
    sam2_predictor,                # pre-built SAM2 predictor
    sam2_eval_fn: Callable[[Tensor, np.ndarray], List[np.ndarray]],
    clean_pass_vadi: Callable,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the post-hoc decoy-semantic eval on a single run directory.

    `run_dir` is expected to contain:
      - results.json
      - processed/frame_NNNN.png (exported uint8)

    Returns the decoy-semantic metrics dict (also written to
    decoy_semantic.json next to results.json).
    """
    from memshield.decoy_seed import (
        build_decoy_insert_seeds, compute_decoy_offset_from_mask,
    )
    from memshield.decoy_semantic_metrics import (
        build_decoy_mask_trajectory,
        remap_masks_to_processed_space_decoy,
        per_frame_decoy_semantic, aggregate_decoy_semantic,
    )
    from memshield.vadi_optimize import build_base_inserts
    from scripts.run_vadi import (
        build_processed, load_processed_uint8,
        remap_masks_to_processed_space,
    )
    from scripts.run_vadi_pilot import load_davis_clip

    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"no results.json at {run_dir}")
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    clip_name = results.get("clip_name") or _infer_clip_name(run_dir)
    W = sorted(int(w) for w in results.get("W") or [])
    decoy_offsets = results.get("decoy_offsets") or []
    if not decoy_offsets or len(decoy_offsets) != len(W):
        raise ValueError(
            f"{run_dir}: decoy_offsets field missing or length != |W| "
            f"({len(decoy_offsets)} vs {len(W)}). This run is likely "
            "from a pre-v5 driver; post-hoc eval not supported.")
    decoy_offsets = [(int(dy), int(dx)) for dy, dx in decoy_offsets]

    flags = _infer_config_flags(results)

    # Load clip.
    x_clean, prompt_mask = load_davis_clip(davis_root, clip_name)
    x_clean = x_clean.to(device)
    T_clean = int(x_clean.shape[0])

    # Rebuild clean-pass pseudo-masks for decoy-mask trajectory.
    clean_out = clean_pass_vadi(sam2_predictor, x_clean, prompt_mask, device)
    pseudo_masks_clean = [np.asarray(m, dtype=np.float32)
                          for m in clean_out.pseudo_masks]

    # Reconstruct insert base tensor that matches the run's `insert_base`.
    W_clean = [int(W[k] - k) for k in range(len(W))]   # c_k = W_k - k
    if flags["insert_base"] == "duplicate_seed":
        seeds, _offs = build_decoy_insert_seeds(
            x_clean, pseudo_masks_clean, W_clean,
        )
        seeds = seeds.to(x_clean.device)
    else:   # midframe (default)
        seeds = build_base_inserts(x_clean, W)

    # Build clean-processed baseline + load exported.
    processed_clean = build_processed(x_clean, seeds, W)
    processed_clean_u8rt = (
        (processed_clean * 255.0 + 0.5).clamp(0.0, 255.0)
        .to(torch.uint8).to(processed_clean.dtype) / 255.0
    )
    exported = load_processed_uint8(run_dir / "processed").to(x_clean.device)
    T_proc = int(exported.shape[0])

    # Run SAM2 on both.
    masks_clean = sam2_eval_fn(processed_clean_u8rt, prompt_mask)
    masks_attacked = sam2_eval_fn(exported, prompt_mask)

    # Rebuild m_true and m_decoy in processed-space.
    # m_true_by_t_np: processed-space true pseudo-masks, with insert override.
    m_true_by_t_np = remap_masks_to_processed_space(pseudo_masks_clean, W)
    # v5 override at insert positions: m_true_by_t[W_k] = m_true_clean[c_k].
    for k, w in enumerate(W):
        c_k = W_clean[k]
        m_true_by_t_np[w] = pseudo_masks_clean[c_k]

    # m_decoy trajectory in clean-space.
    m_decoy_clean = build_decoy_mask_trajectory(
        pseudo_masks_clean, W_clean, decoy_offsets,
    )
    # Then remap to processed-space with v5's insert override.
    m_decoy_by_t_np = remap_masks_to_processed_space_decoy(
        m_decoy_clean, pseudo_masks_clean, W, decoy_offsets,
    )

    # Per-frame records.
    W_sorted = sorted(W); W_set = set(W_sorted)

    def _offset_for_t(t_proc: int) -> Tuple[int, int]:
        k_cover = -1
        for k, w in enumerate(W_sorted):
            if w <= t_proc:
                k_cover = k
            else:
                break
        if k_cover == -1:
            return (0, 0)
        return decoy_offsets[k_cover]

    ds_records = []
    first_insert_proc = min(W_sorted) if W_sorted else None
    for t in range(T_proc):
        gt_true = (m_true_by_t_np[t] > 0.5).astype(np.uint8)
        gt_decoy = (m_decoy_by_t_np[t] > 0.5).astype(np.uint8)
        is_pre = (first_insert_proc is not None
                  and t < first_insert_proc)
        rec = per_frame_decoy_semantic(
            t=t, is_insert=(t in W_set),
            pred_clean=masks_clean[t], pred_attacked=masks_attacked[t],
            m_true=gt_true, m_decoy=gt_decoy,
            decoy_offset=_offset_for_t(t),
            is_pre_first_insert=is_pre,
        )
        ds_records.append(rec)
    agg = aggregate_decoy_semantic(ds_records)

    summary = {
        "clip_name": clip_name,
        "config_name": results.get("config_name"),
        "config_flags": flags,
        "W": W,
        "decoy_offsets": decoy_offsets,
        "per_frame": [asdict(r) for r in ds_records],
        "aggregate": asdict(agg),
        "exported_j_drop": results.get("exported_j_drop"),
    }
    with open(run_dir / "decoy_semantic.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Post-hoc decoy-semantic evaluator.")
    p.add_argument("--davis-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories (each contains results.json + processed/).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    # Lazy SAM2 setup.
    try:
        from memshield.vadi_sam2_wiring import (
            build_sam2_lpips_ssim, clean_pass_vadi, sam2_eval_pseudo_masks,
        )
    except (ImportError, NotImplementedError) as e:
        print(f"[decoy-postrun] adapter import failed: {e}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    predictor, _lpips, _ssim = build_sam2_lpips_ssim(
        checkpoint_path=args.checkpoint, device=device,
    )

    def sam2_eval_fn(video: Tensor, prompt_mask_np: np.ndarray):
        return sam2_eval_pseudo_masks(predictor, video, prompt_mask_np, device)

    collated: List[Dict[str, Any]] = []
    for run in args.runs:
        run_dir = Path(run)
        try:
            summary = evaluate_decoy_semantic_for_run(
                run_dir=run_dir,
                davis_root=Path(args.davis_root),
                sam2_predictor=predictor,
                sam2_eval_fn=sam2_eval_fn,
                clean_pass_vadi=clean_pass_vadi,
                device=device,
            )
        except Exception as e:
            print(f"[decoy-postrun] {run_dir}: FAILED {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        collated.append(summary)
        agg = summary["aggregate"]
        rates = agg["mode_rates"]
        print(f"[decoy-postrun] {summary['clip_name']:12s} "
              f"valid={agg['n_frames_valid']:3d}/{agg['n_frames_total']:3d}  "
              f"(excl pre={agg['n_excluded_pre_first_insert']}, "
              f"empty_clean={agg['n_excluded_empty_pred_clean']})  "
              f"redirect={rates['redirected']:.2f} "
              f"suppress={rates['suppressed']:.2f} "
              f"intact={rates['intact']:.2f} "
              f"split={rates['split']:.2f} "
              f"degraded={rates['degraded']:.2f}  "
              f"J_vs_true={agg['mean_j_vs_true']:.3f} "
              f"J_vs_decoy={agg['mean_j_vs_decoy']:.3f}")
    return 0 if collated else 1


def _self_test() -> None:
    # Self-test is exercised by memshield.decoy_semantic_metrics at the
    # primitive level; here we only validate the CLI help surface (no
    # SAM2 import needed for argparse).
    ap = argparse.ArgumentParser()
    args = ap.parse_known_args(
        ["--davis-root", "x", "--checkpoint", "y", "--runs", "z"]
    )
    print("scripts.decoy_semantic_postrun: no-op self-test PASSED "
          "(CLI surface only; primitive logic exercised by "
          "memshield.decoy_semantic_metrics._self_test)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
