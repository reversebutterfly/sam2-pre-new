"""Placement profiling preprocessing driver — VADI Round 5 Bundle A
sub-session 2 (2026-04-25).

Pre-computes per-clip K=3 placement subsets via beam search using
actual A0 attack J-drop as the score (replacing the falsified
3-signal vulnerability_scorer heuristic which was anti-correlated
with attack effectiveness — decisive-round result, 0.488 ranked
vs 0.534 random K=3 mean J-drop on 10 DAVIS clips).

The output `profile.json` per clip is consumed by Stage 14
(`scripts/run_vadi_v5.py --use-profiled-placement`) to seed the
oracle false trajectory's anchor positions.

## Codex Q1 verdict (c) — two-pass budget

  K=1 / K=2 layers: 12-step ν-only PGD (cheap surrogate, only used
                    for beam pruning where rank stability is less
                    critical because beam expansion forgives minor
                    noise).
  K=3 final layer:  full 100-step ν-only PGD (fidelity-true scoring
                    of the cached "best" subset Stage 14 will load).

Random K=3 baseline (n=10) is profiled at the same FULL budget for
fair comparison with beam K=3 best — confirming our beam search
beats random selection (or, if it doesn't, signaling a deeper
problem).

## Codex Q2 verdict (c) — full cache

profile.json stores:
  - best:                top K=3 subset (full SubsetScore + metadata)
  - top_k1, top_k2, top_k3:  beam survivors per layer
  - raw_k1_scores:       ALL ~N single-frame K=1 scores (used to
                         replay the heuristic-vs-random decisive
                         comparison without re-running)
  - random_k3_scores:    n=10 random K=3 baseline scores
  - run_config:          {step counts, beam_width, min_gap, seeds,
                         decoy_offsets per subset}

## Cost estimate (per clip)

  N_candidates ≈ 50, beam_width = 8 (default), min_gap = 2

  Phase A (full K=1): 50 cheap evals    ≈  10 min
  Phase B (beam K=2): ~320 cheap evals  ≈  64 min  (rough)
  Phase C (beam K=3): ~304 full evals   ≈ 304 min  (≈ 5 hours)
  Phase D (rand K=3):  10 full evals    ≈  17 min

  Total ≈ 6 hours/clip on Pro 6000 (cheap=12s, full=100s assumed).
  3 clips ≈ 18h overnight (too long); recommend beam_width=4 or fewer
  clips per run if budget tight. CLI flags expose dials.

## CLI

    python scripts/run_placement_profile.py \\
        --davis-root data/davis \\
        --checkpoint checkpoints/sam2.1_hiera_tiny.pt \\
        --clips dog camel blackswan \\
        --beam-width 8 \\
        --cheap-n-steps 12 \\
        --full-n-steps 100 \\
        --random-baseline-n 10 \\
        --out-root vadi_runs/v5_placement_profile

Heartbeat-friendly: prints subset / score / wall-clock per eval so a
tail-F + grep monitor produces frequent stdout activity (per CLAUDE.md
Monitor Heartbeat rule).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memshield.placement_profiler import (
    BeamSearchResult, SubsetScore, beam_search_K3,
    make_cached_scorer, random_K3_subsets, serialize_result,
)
from memshield.v5_score_fn import (
    ScoreFnContext, ScoreFnPolicy, build_score_fn, subset_tag,
)
from scripts.run_vadi_pilot import load_davis_clip
from scripts.run_vadi_v5 import VADIv5Config, run_v5_for_clip


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VADI Round 5 placement profiling preprocessing")
    p.add_argument("--davis-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--out-root", default="vadi_runs/v5_placement_profile")
    p.add_argument("--beam-width", type=int, default=8,
                   help="Beam survivors per layer (Codex spec: 8). "
                        "Decrease to 4 if budget is tight.")
    p.add_argument("--min-gap", type=int, default=2,
                   help="Minimum frame distance between any two inserts.")
    p.add_argument("--cheap-n-steps", type=int, default=12,
                   help="ν-only PGD steps for K=1/K=2 cheap surrogate.")
    p.add_argument("--full-n-steps", type=int, default=100,
                   help="ν-only PGD steps for K=3 fidelity-true scoring.")
    p.add_argument("--cheap-threshold-K", type=int, default=3,
                   help="K at which we switch to full N (Codex Q1c). "
                        "K >= threshold → full; K < threshold → cheap.")
    p.add_argument("--candidate-min-frame", type=int, default=1,
                   help="Smallest valid candidate frame index. Frames "
                        "0..MIN-1 are excluded (f0 is the prompt; "
                        "decoy seed builder needs c_k-1 to be valid).")
    p.add_argument("--candidate-max-frame", type=int, default=None,
                   help="Largest valid candidate frame index (default: "
                        "T_clean-1).")
    p.add_argument("--random-baseline-n", type=int, default=10,
                   help="Number of random K=3 subsets to profile as "
                        "baseline for the beam result.")
    p.add_argument("--random-baseline-seed", type=int, default=42)
    p.add_argument("--config-seed", type=int, default=0,
                   help="Seed passed into VADIv5Config for ν init.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip clips with existing profile.json.")
    p.add_argument("--insert-base",
                   choices=["midframe", "duplicate_seed",
                            "poisson_hifi", "propainter"],
                   default="duplicate_seed",
                   help="Insert base mode for v5 PGD. Should match what "
                        "Stage 14 will use. (codex round 6 2026-04-28: "
                        "added poisson_hifi / propainter for ghost-free "
                        "synthesis.)")
    return p


# ---------------------------------------------------------------------------
# Per-clip profiling
# ---------------------------------------------------------------------------


def profile_clip(
    *,
    clip_name: str,
    x_clean: torch.Tensor,
    prompt_mask: np.ndarray,
    clean_pass_fn: Callable,
    forward_fn_builder: Callable,
    lpips_fn: Callable,
    ssim_fn: Optional[Callable],
    sam2_eval_fn: Callable,
    base_config: VADIv5Config,
    out_root: Path,
    candidates: Sequence[int],
    beam_width: int,
    min_gap: int,
    policy: ScoreFnPolicy,
    random_baseline_n: int,
    random_baseline_seed: int,
) -> Dict[str, Any]:
    """Run the 4-phase profiling for a single clip; return profile dict."""
    clip_out_root = out_root / clip_name
    clip_out_root.mkdir(parents=True, exist_ok=True)
    # Per-subset exports go to a tmp/ sub-dir which v5_score_fn cleans
    # incrementally — keeps disk usage bounded.
    eval_tmp_root = clip_out_root / "tmp_evals"
    eval_tmp_root.mkdir(parents=True, exist_ok=True)

    # Caller-owned metadata sink: per-subset (tag → metadata dict).
    metadata_sink: Dict[str, Dict[str, Any]] = {}

    ctx = ScoreFnContext(
        clip_name=clip_name,
        x_clean=x_clean,
        prompt_mask=prompt_mask,
        clean_pass_fn=clean_pass_fn,
        forward_fn_builder=forward_fn_builder,
        lpips_fn=lpips_fn,
        ssim_fn=ssim_fn,
        sam2_eval_fn=sam2_eval_fn,
        base_config=base_config,
        out_root=eval_tmp_root,
        rng=np.random.default_rng(0),
    )
    raw_score_fn = build_score_fn(
        ctx, policy, run_v5_for_clip=run_v5_for_clip,
        cleanup_export_after_scoring=True,
        metadata_sink=metadata_sink,
        config_name_prefix=f"profile_{clip_name}",
    )
    # Cache subset → score so beam-layer expansion overlaps don't redo work.
    score_fn = make_cached_scorer(raw_score_fn)

    notes: List[str] = []

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[profile {clip_name} {ts}] {msg}", flush=True)
        notes.append(msg)

    _log(f"start: {len(candidates)} candidates, beam={beam_width} "
         f"cheap={policy.cheap_n_steps} full={policy.full_n_steps}")
    t0 = time.time()

    # --- Phase A: explicit raw K=1 scan over ALL candidates.
    # Required by Codex Q2c (cache full K=1 layer). Beam search will reuse
    # these via the make_cached_scorer wrapper.
    raw_k1_scores: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        t_eval = time.time()
        s = score_fn((int(c),))
        elapsed = time.time() - t_eval
        raw_k1_scores.append({
            "subset": [int(c)],
            "score": float(s),
            "wallclock_s": float(elapsed),
        })
        if (i + 1) % 5 == 0 or i == 0:
            _log(f"  raw K=1 [{i+1}/{len(candidates)}] c={int(c)} "
                 f"score={s:.4f} wc={elapsed:.1f}s")
    _log(f"phase A done: {len(candidates)} K=1 evals in "
         f"{time.time() - t0:.1f}s")

    # --- Phase B & C: beam search K=3 (K=1 layer mostly cache-hit).
    t1 = time.time()
    beam_result = beam_search_K3(
        candidates=candidates, score_fn=score_fn,
        beam_width=beam_width, min_gap=min_gap,
        log=_log,
    )
    _log(f"phase B+C done in {time.time() - t1:.1f}s; "
         f"total beam evals = {beam_result.total_evals}")

    # Attach metadata to each layer's SubsetScore.
    def _attach_meta(layer: List[SubsetScore]) -> None:
        for ss in layer:
            tag = subset_tag(ss.subset)
            if tag in metadata_sink:
                ss.metadata = dict(metadata_sink[tag])

    _attach_meta([beam_result.best])
    _attach_meta(beam_result.top_k1)
    _attach_meta(beam_result.top_k2)
    _attach_meta(beam_result.top_k3)

    # --- Phase D: random K=3 baseline at FULL budget.
    t2 = time.time()
    rand_subsets = random_K3_subsets(
        candidates, n=random_baseline_n, min_gap=min_gap,
        seed=random_baseline_seed, strict=False,
    )
    random_k3_scores: List[Dict[str, Any]] = []
    for j, subset in enumerate(rand_subsets):
        t_eval = time.time()
        s = score_fn(tuple(subset))
        elapsed = time.time() - t_eval
        meta = metadata_sink.get(subset_tag(subset), {})
        random_k3_scores.append({
            "subset": list(subset),
            "score": float(s),
            "wallclock_s": float(elapsed),
            "metadata": meta,
        })
        _log(f"  random K=3 [{j+1}/{len(rand_subsets)}] "
             f"{subset} score={s:.4f}")
    _log(f"phase D done in {time.time() - t2:.1f}s; "
         f"{len(rand_subsets)} random subsets")

    # --- Aggregate stats.
    valid_random = [r["score"] for r in random_k3_scores
                    if r["score"] > -1.0]
    aggregate = {
        "best_beam_score": float(beam_result.best.score),
        "best_beam_subset": list(beam_result.best.subset),
        "random_n": len(rand_subsets),
        "random_mean": (float(np.mean(valid_random))
                        if valid_random else float("nan")),
        "random_max": (float(np.max(valid_random))
                       if valid_random else float("nan")),
        "beam_minus_random_max": (
            float(beam_result.best.score - np.max(valid_random))
            if valid_random else float("nan")),
    }

    # --- Build profile.json structure.
    profile = {
        "clip_name": clip_name,
        "best": {
            "subset": list(beam_result.best.subset),
            "score": float(beam_result.best.score),
            "metadata": dict(beam_result.best.metadata),
        },
        "top_k1": [
            {
                "subset": list(s.subset),
                "score": float(s.score),
                "metadata": dict(s.metadata),
            } for s in beam_result.top_k1
        ],
        "top_k2": [
            {
                "subset": list(s.subset),
                "score": float(s.score),
                "metadata": dict(s.metadata),
            } for s in beam_result.top_k2
        ],
        "top_k3": [
            {
                "subset": list(s.subset),
                "score": float(s.score),
                "metadata": dict(s.metadata),
            } for s in beam_result.top_k3
        ],
        "raw_k1_scores": raw_k1_scores,
        "random_k3_scores": random_k3_scores,
        "aggregate": aggregate,
        "run_config": {
            "candidates": [int(c) for c in candidates],
            "beam_width": int(beam_width),
            "min_gap": int(min_gap),
            "cheap_n_steps": int(policy.cheap_n_steps),
            "full_n_steps": int(policy.full_n_steps),
            "cheap_threshold_K": int(policy.cheap_threshold_K),
            "random_baseline_seed": int(random_baseline_seed),
            "config_seed": int(base_config.seed),
            "schedule_preset": str(base_config.schedule_preset),
            "delta_support_mode": str(base_config.delta_support_mode),
            "insert_base_mode": str(base_config.insert_base_mode),
            "total_beam_evals": int(beam_result.total_evals),
            "total_wallclock_s": float(time.time() - t0),
        },
        "notes": notes,
    }
    return profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    try:
        from scripts.run_vadi_pilot import build_pilot_adapters
    except (ImportError, NotImplementedError) as e:
        print(f"[profile] adapter import failed: {e}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    clean_fac, fwd_fac, lpips_fn, ssim_fn, sam2_eval_fn = build_pilot_adapters(
        checkpoint_path=args.checkpoint, device=device,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    policy = ScoreFnPolicy(
        cheap_n_steps=args.cheap_n_steps,
        full_n_steps=args.full_n_steps,
        cheap_threshold_K=args.cheap_threshold_K,
    )

    summaries: List[Dict[str, Any]] = []

    for clip_name in args.clips:
        profile_path = out_root / clip_name / "profile.json"
        if args.skip_existing and profile_path.exists():
            print(f"[profile] {clip_name}: profile.json exists — skipping",
                  flush=True)
            continue

        print(f"[profile] === clip: {clip_name} ===", flush=True)
        try:
            x_clean, prompt_mask = load_davis_clip(
                Path(args.davis_root), clip_name)
        except Exception as e:
            print(f"[profile] {clip_name}: load failed: {type(e).__name__}: {e}",
                  flush=True)
            continue

        x_clean = x_clean.to(device)
        T_clean = int(x_clean.shape[0])
        clean_pass_fn = clean_fac(clip_name, x_clean, prompt_mask)
        fwd_builder = fwd_fac(clip_name, x_clean, prompt_mask)

        # Candidate frame pool. v5 driver requires 1 ≤ c < T_clean (decoy-
        # seed init reads x_clean[c-1]).
        c_min = max(1, int(args.candidate_min_frame))
        c_max = (T_clean - 1 if args.candidate_max_frame is None
                 else min(T_clean - 1, int(args.candidate_max_frame)))
        if c_max < c_min:
            print(f"[profile] {clip_name}: empty candidate pool "
                  f"[{c_min}, {c_max}] — skipping", flush=True)
            continue
        candidates = list(range(c_min, c_max + 1))

        # Base config for profiling — A0 ν-only insert-only, no polish.
        base_config = VADIv5Config(
            seed=args.config_seed,
            insert_base_mode=args.insert_base,
        )

        try:
            profile = profile_clip(
                clip_name=clip_name,
                x_clean=x_clean,
                prompt_mask=prompt_mask,
                clean_pass_fn=clean_pass_fn,
                forward_fn_builder=fwd_builder,
                lpips_fn=lpips_fn,
                ssim_fn=ssim_fn,
                sam2_eval_fn=sam2_eval_fn,
                base_config=base_config,
                out_root=out_root,
                candidates=candidates,
                beam_width=int(args.beam_width),
                min_gap=int(args.min_gap),
                policy=policy,
                random_baseline_n=int(args.random_baseline_n),
                random_baseline_seed=int(args.random_baseline_seed),
            )
        except Exception as e:
            print(f"[profile] {clip_name}: profiling failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        # Persist.
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"[profile] {clip_name}: wrote {profile_path}", flush=True)
        print(f"[profile] {clip_name}: best K=3 = "
              f"{profile['best']['subset']} "
              f"score={profile['best']['score']:.4f} "
              f"random_max={profile['aggregate']['random_max']:.4f} "
              f"beam-random={profile['aggregate']['beam_minus_random_max']:+.4f}",
              flush=True)
        summaries.append({
            "clip": clip_name,
            "best_subset": profile["best"]["subset"],
            "best_score": profile["best"]["score"],
            "random_max": profile["aggregate"]["random_max"],
            "random_mean": profile["aggregate"]["random_mean"],
            "beam_minus_random_max":
                profile["aggregate"]["beam_minus_random_max"],
            "wallclock_s":
                profile["run_config"]["total_wallclock_s"],
        })

    # Top-level summary across all profiled clips.
    summary_path = out_root / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_args": vars(args),
            "per_clip": summaries,
        }, f, indent=2, default=str)
    print(f"[profile] summary written: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
