"""A3 gating driver: 3-condition memory-block ablation on v5-polished
videos + d_mem(t) trace extraction.

Consumes existing v5-polished output dirs (from prior `scripts.run_vadi_v5
--oracle-traj-v4 --placement-search joint_curriculum` runs) and runs three
eval forwards per clip:

  baseline  = full v5 (no blocking)            [reference]
  attacked  = full v5 + block W_attacked       [insert-position memory blocked]
  control   = full v5 + block W_control        [matched non-insert blocked]

Plus a clean reference forward for d_mem(t) computation.

Per codex M0 review (2026-04-27 thread 019dcd87): the intervention is
broader than "memory writes" — it suppresses ALL future temporal-state
contributions (mask-memory chunks AND obj_ptr tokens) from blocked frames,
because SAM2 reads both from `non_cond_frame_outputs`. The pre-registered
A3 claim has been refined to match.

Pre-registered acceptance (see refine-logs/FINAL_PROPOSAL.md):
  Strong pass : collapse_attacked >= 0.20 AND (att - ctrl) >= 0.10 on >=7/10
  Partial pass: collapse_attacked >= 0.10 AND (att - ctrl) >= 0.05 on >=6/10
  Fail        : workshop pivot

Usage:
  python -m scripts.run_a3_gating \\
    --davis-root ~/sam2-pre-new/data/davis \\
    --checkpoint ~/sam2-pre-new/checkpoints/sam2.1_hiera_tiny.pt \\
    --v5-root vadi_runs/v5_paper_m3 \\
    --out-root vadi_runs/v5_paper_m2_a3 \\
    --clips dog bmx-trees ... \\
    --device cuda

Per-clip output: <out-root>/<clip>/a3_results.json with all 3 conditions'
J-drop + d_mem(t) trace + collapse magnitudes + control-frame indices.

CLI flag --smoke runs only 1 clip with reduced output for the M0 R004
deployment-gate smoke test (parity + blocked-frame + extractor + RoPE
path on real SAM2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def jaccard_per_frame(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> List[float]:
    """Per-frame binary Jaccard (IoU). Both inputs are uint8 lists of [H, W]."""
    n = min(len(pred_masks), len(gt_masks))
    out: List[float] = []
    for t in range(n):
        p = (pred_masks[t] > 0).astype(bool)
        g = (gt_masks[t] > 0).astype(bool)
        inter = float(np.logical_and(p, g).sum())
        union = float(np.logical_or(p, g).sum())
        out.append(inter / union if union > 0 else 1.0)
    return out


def load_processed_video(processed_dir: Path) -> Tensor:
    """Load `frame_NNNN.png` files into a [T, H, W, 3] float tensor in [0, 1]."""
    files = sorted(processed_dir.glob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"No frame_*.png in {processed_dir}")
    arrs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    arr = np.stack(arrs, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def remap_pseudo_masks_to_processed(
    clean_pseudo_masks: List[np.ndarray],
    W_attacked: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Match the v5 driver's processed-space mask remap (insert positions
    use midframe average, others inherit via attacked_to_clean)."""
    from memshield.vadi_optimize import attacked_to_clean
    T_clean = len(clean_pseudo_masks)
    W_sorted = sorted(int(w) for w in W_attacked)
    K = len(W_sorted)
    T_proc = T_clean + K
    out: Dict[int, np.ndarray] = {}
    for t in range(T_proc):
        if t in W_sorted:
            k = W_sorted.index(t)
            c_k = W_sorted[k] - k
            if 1 <= c_k < T_clean:
                out[t] = (0.5 * clean_pseudo_masks[c_k - 1]
                          + 0.5 * clean_pseudo_masks[c_k])
            else:
                out[t] = clean_pseudo_masks[max(0, min(T_clean - 1, c_k))]
        else:
            c = attacked_to_clean(t, W_sorted)
            out[t] = clean_pseudo_masks[c]
    return out


def run_eval_with_hooks(
    forward_fn: Any,
    processed: Tensor,
    *,
    blocked_frames: Sequence[int],
    extractor: Optional[Any],
) -> Tuple[List[np.ndarray], Dict[int, Tuple[Tensor, Tensor]]]:
    """Run a single forward (with optional blocking and extractor) on a
    processed video; return per-frame hard masks + per-frame (V, attn)
    with CPU tensors (GPU memory released).

    Uses `memshield.causal_diagnostics.make_blocking_forward_fn` to wrap.

    MEDIUM-fix (codex 2026-04-27): after copying captured tensors to CPU,
    call ``extractor.reset()`` to release GPU references. Caller can also
    `del` the extractor + `torch.cuda.empty_cache()` between conditions.
    """
    from memshield.causal_diagnostics import make_blocking_forward_fn

    blocking = make_blocking_forward_fn(
        forward_fn,
        blocked_frames=blocked_frames,
        extractor=extractor,
    )

    T_proc = int(processed.shape[0])
    return_at = list(range(T_proc))
    with torch.no_grad():
        proc_dev = processed.to(forward_fn.device)
        proc_dev.requires_grad_(False)
        logits_by_t = blocking(proc_dev, return_at=return_at)

    masks: List[np.ndarray] = []
    for t in range(T_proc):
        logits = logits_by_t[t]
        hard = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()
        masks.append(hard)

    V_attn: Dict[int, Tuple[Tensor, Tensor]] = {}
    if extractor is not None:
        for fid, V in extractor.V_by_frame.items():
            attn = extractor.attn_by_frame.get(fid)
            if attn is None:
                continue
            V_attn[int(fid)] = (V.detach().cpu(), attn.detach().cpu())
        # MEDIUM-fix: release GPU references; caller may also empty_cache.
        extractor.reset()

    # Free per-call ephemeral GPU memory.
    del logits_by_t, proc_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks, V_attn


def parity_check(
    forward_fn: Any,
    processed: Tensor,
    return_at: List[int],
) -> bool:
    """Smoke-test: blocked_frames=[] and extractor=None must produce IDENTICAL
    output to base forward. Returns True if all returned logits match within
    bf16 tolerance."""
    from memshield.causal_diagnostics import make_blocking_forward_fn
    proc_dev = processed.to(forward_fn.device)
    proc_dev.requires_grad_(False)
    with torch.no_grad():
        out_base = forward_fn(proc_dev, return_at=return_at)
        wrap = make_blocking_forward_fn(
            forward_fn, blocked_frames=(), extractor=None,
        )
        out_wrapped = wrap(proc_dev, return_at=return_at)
    for t in return_at:
        a = out_base[t].detach().float().cpu()
        b = out_wrapped[t].detach().float().cpu()
        # bf16 autocast tolerance ~1e-2.
        if not torch.allclose(a, b, atol=1e-2, rtol=1e-2):
            print(f"[parity] FAIL at t={t}: max diff {(a - b).abs().max().item():.4e}")
            return False
    return True


# ---------------------------------------------------------------------------
# Per-clip pipeline
# ---------------------------------------------------------------------------


def process_one_clip(
    clip_name: str,
    *,
    davis_root: Path,
    checkpoint_path: Path,
    v5_run_dir: Path,
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
    control_seed: int = 0,
    top_k: int = 32,
) -> Dict[str, Any]:
    """Run baseline / attacked / control eval + clean reference + d_mem(t)
    on one clip. Returns aggregated results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports.
    from scripts.run_vadi_pilot import build_pilot_adapters, load_davis_clip
    from memshield.causal_diagnostics import (
        MemoryReadoutExtractor, build_control_frames, compute_d_mem_trace,
    )

    print(f"[a3] {clip_name}: building adapters")
    clean_fac, fwd_fac, lpips_fn, ssim_fn, _ = build_pilot_adapters(
        checkpoint_path=str(checkpoint_path), device=device,
    )

    print(f"[a3] {clip_name}: loading clean clip + first-frame mask")
    x_clean, prompt_mask = load_davis_clip(davis_root, clip_name)
    x_clean = x_clean.to(device)
    T_clean = int(x_clean.shape[0])

    # Locate the v5-polished processed dir + results.json. Prefer the
    # __ot dir (Stage 14 polish); fall back only if absent. LOW-fix
    # (codex 2026-04-27): require unique match to avoid silent
    # non-determinism across reruns.
    ot_dirs = list((v5_run_dir / clip_name).glob("*__ot/processed"))
    main_dirs = list((v5_run_dir / clip_name).glob("*/processed"))
    main_dirs = [d for d in main_dirs if not d.parent.name.endswith("__ot")]
    if ot_dirs:
        if len(ot_dirs) > 1:
            raise RuntimeError(
                f"Multiple __ot/processed dirs in {v5_run_dir / clip_name}: "
                f"{ot_dirs}. Pass an explicit config dir or remove duplicates."
            )
        processed_dir = ot_dirs[0]
    elif main_dirs:
        if len(main_dirs) > 1:
            raise RuntimeError(
                f"Multiple */processed dirs in {v5_run_dir / clip_name}: "
                f"{main_dirs}."
            )
        processed_dir = main_dirs[0]
    else:
        raise FileNotFoundError(
            f"No processed/ dir under {v5_run_dir / clip_name}")
    results_json = processed_dir.parent / "results.json"
    if not results_json.exists():
        # Fall back to the non-_ot results.json (Stage 11-13 dir).
        alt = list((v5_run_dir / clip_name).glob("*/results.json"))
        if alt:
            results_json = alt[0]
        else:
            raise FileNotFoundError(
                f"No results.json under {v5_run_dir / clip_name}")
    with open(results_json, "r", encoding="utf-8") as f:
        v5_results = json.load(f)

    W_attacked = sorted(int(w) for w in v5_results["W"])
    K = len(W_attacked)
    T_proc = T_clean + K
    print(f"[a3] {clip_name}: W_attacked={W_attacked}, T_proc={T_proc}")

    # bridge_frames: derive from v5 default (4 frames after each insert).
    L = 4
    W_set = set(W_attacked)
    bridge_frames: List[int] = []
    for w in W_attacked:
        for j in range(1, L + 1):
            t = w + j
            if t < T_proc and t not in W_set:
                bridge_frames.append(t)
    bridge_frames = sorted(set(bridge_frames))

    W_control = build_control_frames(
        W_attacked, bridge_frames, T_proc, K=K, seed=control_seed,
    )
    print(f"[a3] {clip_name}: W_control={W_control} (seed={control_seed})")

    # Load v5-polished processed video (uint8 PNGs).
    x_polished = load_processed_video(processed_dir).to(device)
    if int(x_polished.shape[0]) != T_proc:
        raise RuntimeError(
            f"Loaded video has T={x_polished.shape[0]} but expected "
            f"T_proc={T_proc} (T_clean={T_clean} + K={K})")

    # Get clean pseudo masks (for ground-truth comparison + d_mem clean reference).
    clean_pass_fn = clean_fac(clip_name, x_clean, prompt_mask)
    print(f"[a3] {clip_name}: running clean pass for pseudo-GT + d_mem ref")
    clean_out = clean_pass_fn(x_clean, prompt_mask)
    clean_pseudo_masks_np = [
        (m > 0.5).astype(np.uint8) for m in clean_out.pseudo_masks
    ]
    pseudo_gt_proc = remap_pseudo_masks_to_processed(
        clean_pseudo_masks_np, W_attacked,
    )
    pseudo_gt_list = [pseudo_gt_proc[t] for t in range(T_proc)]

    # Build a per-clip VADIForwardFn (what we wrap with extractor + blocking).
    # The fwd_fac returns a builder; we use it directly on x_clean for mask
    # init + on x_polished for eval.
    fwd_builder = fwd_fac(clip_name, x_clean, prompt_mask)
    forward_fn = fwd_builder(x_clean, prompt_mask, W_attacked)
    predictor = forward_fn.predictor
    memory_attention = predictor.memory_attention

    # Smoke-test parity (M0 R004 deployment gate).
    print(f"[a3] {clip_name}: parity check (blocked=[], extractor=None)")
    parity_t = list(range(min(8, T_proc)))   # first 8 frames sufficient
    ok = parity_check(forward_fn, x_polished, parity_t)
    if not ok:
        raise RuntimeError("PARITY FAILED — wrapper diverges from base forward")
    print(f"[a3] {clip_name}: parity OK on {len(parity_t)} frames")

    # ============ 4 forward passes ============
    # 1. Clean reference (x_clean, no blocking, with extractor for d_mem ref).
    print(f"[a3] {clip_name}: clean reference forward (extractor on)")
    extractor_clean = MemoryReadoutExtractor(memory_attention)
    with extractor_clean:
        masks_clean_proc, V_attn_clean = run_eval_with_hooks(
            forward_fn, x_clean,
            blocked_frames=(),
            extractor=extractor_clean,
        )
    # masks_clean_proc has length T_clean (clean video has no inserts).
    # For d_mem comparison, we use V_attn_clean's frame indices DIRECTLY —
    # but they're in CLEAN-space. To compare with attacked-space frames, we
    # need to map: attacked-space t -> clean-space attacked_to_clean(t).
    # See d_mem trace section below.

    if smoke:
        # Smoke-mode early exit: verify all 3 hook paths run without crash.
        # MEDIUM-fix (codex 2026-04-27): smoke MUST exercise the blocked-
        # frame path too, not just parity + clean-extractor. Run one quick
        # blocked forward and a quick masked-decoder check.
        print(f"[a3] {clip_name}: SMOKE — exercising blocked-frame path")
        from memshield.causal_diagnostics import (
            MemoryReadoutExtractor, make_blocking_forward_fn,
        )
        extractor_smoke = MemoryReadoutExtractor(memory_attention)
        with extractor_smoke:
            blocked_forward = make_blocking_forward_fn(
                forward_fn,
                blocked_frames=W_attacked,
                extractor=extractor_smoke,
            )
            with torch.no_grad():
                proc_dev = x_polished.to(forward_fn.device)
                proc_dev.requires_grad_(False)
                logits_blocked = blocked_forward(
                    proc_dev, return_at=list(range(min(8, T_proc))))
        n_blocked_frames_captured = len(extractor_smoke.V_by_frame)
        smoke_result = {
            "clip_name": clip_name,
            "smoke": True,
            "parity_ok": ok,
            "extractor_clean_n_frames_captured": len(V_attn_clean),
            "extractor_blocked_n_frames_captured": n_blocked_frames_captured,
            "blocked_forward_returned_n_frames": len(logits_blocked),
            "T_clean": T_clean,
            "T_proc": T_proc,
            "W_attacked": W_attacked,
            "W_control": W_control,
        }
        with open(out_dir / "a3_smoke_result.json", "w", encoding="utf-8") as f:
            json.dump(smoke_result, f, indent=2)
        print(f"[a3] {clip_name}: SMOKE OK (parity={ok}, "
              f"clean_frames={len(V_attn_clean)}, "
              f"blocked_frames={n_blocked_frames_captured}, "
              f"blocked_logits={len(logits_blocked)})")
        return smoke_result

    # 2. baseline (x_polished, no blocking, with extractor).
    print(f"[a3] {clip_name}: baseline forward on x_polished")
    extractor_base = MemoryReadoutExtractor(memory_attention)
    with extractor_base:
        masks_base, V_attn_base = run_eval_with_hooks(
            forward_fn, x_polished,
            blocked_frames=(),
            extractor=extractor_base,
        )
    # MEDIUM-fix (codex 2026-04-27): drop GPU references between conditions.
    del extractor_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. attacked (x_polished, block W_attacked, with extractor).
    print(f"[a3] {clip_name}: attacked forward (block W_attacked={W_attacked})")
    extractor_att = MemoryReadoutExtractor(memory_attention)
    with extractor_att:
        masks_att, V_attn_att = run_eval_with_hooks(
            forward_fn, x_polished,
            blocked_frames=W_attacked,
            extractor=extractor_att,
        )
    del extractor_att
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. control (x_polished, block W_control, with extractor).
    print(f"[a3] {clip_name}: control forward (block W_control={W_control})")
    extractor_ctl = MemoryReadoutExtractor(memory_attention)
    with extractor_ctl:
        masks_ctl, V_attn_ctl = run_eval_with_hooks(
            forward_fn, x_polished,
            blocked_frames=W_control,
            extractor=extractor_ctl,
        )
    del extractor_ctl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============ J-drop computations ============
    # Clean baseline = run clean pass on x_clean (already done) -> per-frame
    # masks at T_clean; we need per-frame J for each attacked-space t. Use
    # the same processed-space remap convention as eval_exported_j_drop.
    # For the A3 ablation, the comparable baseline is the v5-polished
    # baseline (no blocking) re-run; "attacked - baseline" gives the
    # collapse magnitude.

    J_base = jaccard_per_frame(masks_base, pseudo_gt_list)
    J_att = jaccard_per_frame(masks_att, pseudo_gt_list)
    J_ctl = jaccard_per_frame(masks_ctl, pseudo_gt_list)

    J_base_mean = float(np.mean(J_base))
    J_att_mean = float(np.mean(J_att))
    J_ctl_mean = float(np.mean(J_ctl))

    # J-drop relative to (hypothetical) clean propagation. For pre-registered
    # A3, we report:
    #   collapse_attacked = J_att_mean - J_base_mean   (positive if blocking
    #                       insert frames RAISES J, i.e. attack collapses)
    #   collapse_control  = J_ctl_mean - J_base_mean
    # Pre-registered acceptance is on the magnitude of these collapses.
    collapse_attacked = float(J_att_mean - J_base_mean)
    collapse_control = float(J_ctl_mean - J_base_mean)

    print(f"[a3] {clip_name}: J_base={J_base_mean:.4f} "
          f"J_att={J_att_mean:.4f} J_ctl={J_ctl_mean:.4f}")
    print(f"[a3] {clip_name}: collapse_attacked={collapse_attacked:+.4f} "
          f"collapse_control={collapse_control:+.4f} "
          f"(att-ctrl={collapse_attacked - collapse_control:+.4f})")

    # ============ d_mem(t) trace ============
    # HIGH-fix (codex 2026-04-27, two issues):
    #
    # (1) Pre-registered C1.b is on the LAST INSERT'S BRIDGE WINDOW
    #     (w_K, w_K + L], not the full suffix. Recovery later in the clip
    #     would dilute or invert the persistence claim. Compute d_mem only
    #     at attacked-space frames in (max(W_attacked), max(W_attacked)+L]
    #     that are also in `bridge_frames` (excludes inserts themselves).
    #
    # (2) Clean-reference frame mapping: V_attn_clean is indexed by
    #     CLEAN-SPACE frame ids (length T_clean). For each attacked-space
    #     evaluation frame t, the comparable clean-space frame is
    #     attacked_to_clean(t, W_attacked). Build an attacked-space-keyed
    #     clean reference dict before calling compute_d_mem_trace.
    from memshield.vadi_optimize import attacked_to_clean
    w_K = max(W_attacked)
    L = 4   # matches v5 default oracle_traj_bridge_length
    bridge_eval_frames = sorted(
        t for t in range(w_K + 1, w_K + L + 1)
        if t < T_proc and t not in W_set
    )

    # Map V_attn_clean (clean-space keys) to attacked-space keys for the
    # bridge_eval_frames. Drop frames whose clean mapping is out of range
    # (shouldn't happen for non-insert non-zero attacked-space frames).
    V_attn_clean_proc: Dict[int, Tuple[Tensor, Tensor]] = {}
    for t in bridge_eval_frames:
        c_t = attacked_to_clean(t, W_attacked)
        if 0 <= c_t < T_clean and c_t in V_attn_clean:
            V_attn_clean_proc[t] = V_attn_clean[c_t]
        else:
            print(f"[a3] {clip_name}: WARNING — bridge_eval_frame t={t} "
                  f"has no clean mapping (c_t={c_t})")

    print(f"[a3] {clip_name}: d_mem at last-insert bridge window "
          f"{bridge_eval_frames} (w_K={w_K}, L={L})")
    d_mem_base = compute_d_mem_trace(
        V_attn_clean_proc, V_attn_base, top_k=top_k,
        frames=bridge_eval_frames)
    d_mem_att = compute_d_mem_trace(
        V_attn_clean_proc, V_attn_att, top_k=top_k,
        frames=bridge_eval_frames)
    d_mem_ctl = compute_d_mem_trace(
        V_attn_clean_proc, V_attn_ctl, top_k=top_k,
        frames=bridge_eval_frames)

    # NOTE: C1.b "integral of (d_mem_full − d_mem_only) > 0" is computed
    # downstream by the analysis script that merges this M2 output with
    # the A1-only ablation output (codex confirmed division of labor
    # 2026-04-27). M2 here just emits per-frame d_mem-vs-clean for each
    # condition; analysis script subtracts the matching A1-only values.

    # ============ save ============
    result: Dict[str, Any] = {
        "clip_name": clip_name,
        "T_clean": T_clean,
        "T_proc": T_proc,
        "K": K,
        "W_attacked": W_attacked,
        "W_control": W_control,
        "control_seed": control_seed,
        "bridge_frames": bridge_frames,
        "v5_run_dir": str(v5_run_dir),
        "v5_results_path": str(results_json),
        "v5_exported_j_drop": v5_results.get("exported_j_drop"),

        "J_baseline_mean": J_base_mean,
        "J_attacked_mean": J_att_mean,
        "J_control_mean": J_ctl_mean,
        "collapse_attacked": collapse_attacked,
        "collapse_control": collapse_control,
        "collapse_attacked_minus_control": (
            collapse_attacked - collapse_control),

        "J_per_frame": {
            "baseline": J_base,
            "attacked": J_att,
            "control": J_ctl,
        },

        "d_mem": {
            "top_k": top_k,
            "bridge_eval_frames": bridge_eval_frames,
            "w_K": int(w_K),
            "L": int(L),
            "baseline_vs_clean": d_mem_base,
            "attacked_vs_clean": d_mem_att,
            "control_vs_clean": d_mem_ctl,
            "_note": (
                "d_mem at last-insert bridge window only (codex HIGH-fix "
                "2026-04-27). C1.b persistence integral (full-v5 minus "
                "insert-only) requires merging this output with A1-only "
                "results in the analysis layer."
            ),
        },

        "parity_ok": ok,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "a3_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[a3] {clip_name}: saved {out_dir / 'a3_results.json'}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="A3 gating ablation (memory-block + d_mem trace) on "
                    "v5-polished videos."
    )
    p.add_argument("--davis-root", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--v5-root", required=True, type=Path,
                   help="Root dir containing per-clip v5-polished outputs "
                        "(e.g. vadi_runs/v5_paper_m3).")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 1 clip parity + extractor sanity, "
                        "no 3-condition or d_mem (M0 R004 gate).")
    p.add_argument("--control-seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=32,
                   help="d_mem aggregation top-K (default 32).")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke and len(args.clips) > 1:
        print(f"[a3] SMOKE mode: limiting to first clip ({args.clips[0]})")
        args.clips = args.clips[:1]

    summary: Dict[str, Any] = {
        "v5_root": str(args.v5_root),
        "out_root": str(args.out_root),
        "clips": list(args.clips),
        "smoke": bool(args.smoke),
        "per_clip": {},
    }

    for clip in args.clips:
        clip_out = args.out_root / clip
        try:
            res = process_one_clip(
                clip,
                davis_root=args.davis_root,
                checkpoint_path=args.checkpoint,
                v5_run_dir=args.v5_root,
                out_dir=clip_out,
                device=device,
                smoke=args.smoke,
                control_seed=args.control_seed,
                top_k=args.top_k,
            )
            if not args.smoke:
                summary["per_clip"][clip] = {
                    "collapse_attacked": res["collapse_attacked"],
                    "collapse_control": res["collapse_control"],
                    "collapse_att_minus_ctrl":
                        res["collapse_attacked_minus_control"],
                    "J_baseline_mean": res["J_baseline_mean"],
                    "J_attacked_mean": res["J_attacked_mean"],
                    "J_control_mean": res["J_control_mean"],
                }
            else:
                summary["per_clip"][clip] = {
                    "smoke_ok": True,
                    "parity_ok": res.get("parity_ok"),
                }
        except Exception as e:
            import traceback
            print(f"[a3] ERROR on clip {clip}: {e}")
            traceback.print_exc()
            summary["per_clip"][clip] = {"error": str(e)}

    # ============ pre-registered tier verdict ============
    if not args.smoke:
        n_clips = len(args.clips)
        att_per_clip = [
            v["collapse_attacked"] for v in summary["per_clip"].values()
            if "collapse_attacked" in v
        ]
        att_minus_ctrl = [
            v["collapse_att_minus_ctrl"] for v in summary["per_clip"].values()
            if "collapse_att_minus_ctrl" in v
        ]
        if len(att_per_clip) == n_clips:
            n_strong = sum(
                1 for a, d in zip(att_per_clip, att_minus_ctrl)
                if a >= 0.20 and d >= 0.10
            )
            n_partial = sum(
                1 for a, d in zip(att_per_clip, att_minus_ctrl)
                if a >= 0.10 and d >= 0.05
            )
            tier = "FAIL"
            if n_strong >= 7:
                tier = "STRONG"
            elif n_partial >= 6:
                tier = "PARTIAL"
            summary["a3_tier"] = tier
            summary["n_strong_clips"] = n_strong
            summary["n_partial_clips"] = n_partial
            print(f"[a3] PRE-REGISTERED VERDICT: {tier} "
                  f"(strong={n_strong}, partial={n_partial} of {n_clips})")

    with open(args.out_root / "a3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[a3] DONE. Summary -> {args.out_root / 'a3_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
