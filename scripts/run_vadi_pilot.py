"""VADI 3-clip × 4-config pilot gate with pre-committed GO/NO-GO decision.

File 6 of 8 per HANDOFF_VADI_PILOT.md. Runs the pre-committed pilot scope
(dog, cows, bmx-trees × K=1 top / K=1 random / K=3 top / δ-only-local-random)
and evaluates the GO conditions:

  GO condition 1: J-drop(K=1 top) − J-drop(K=1 random) ≥ 0.05 on ≥ 2/3 clips.
  GO condition 2: J-drop(K=3 top) ≥ 0.20 on ≥ 2/3 clips.
  BOTH must hold for GO.

Diagnostic (not blocking GO, flagged at pilot):
  Δmu_decoy > 0 AND Δmu_decoy ≥ 2·max(0, -Δmu_true) on ≥ 2/3 clips.

NO-GO → pivot paper to "architecture-aware attack-surface analysis of SAM2"
using restoration + vulnerability scoring as primary content (honest fallback).

Configs 1-3 go through the standard `run_vadi_for_clip` path. Config 4
(phantom-δ-only) needs a K_ins=0 variant of the PGD driver; it's left as
an explicit TODO — it's a main-table row (7/8) that doesn't block GO.

SAM2 / LPIPS / SSIM wiring: real adapters are built lazily by
`build_pilot_adapters(...)` on the remote Pro 6000; self-tests here use
stubs so `--dry-run` and the GO/NO-GO aggregation logic can be validated
locally without any heavyweight imports.

Run `python scripts/run_vadi_pilot.py --help` for CLI; bare run →
self-tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memshield.vadi_optimize import VADIConfig
from scripts.run_vadi import VADIClipOutput, run_vadi_for_clip


# =============================================================================
# Pilot scope + GO/NO-GO thresholds (FROZEN — do not edit without updating
# REFINEMENT_REPORT.md and FINAL_PROPOSAL.md in parallel)
# =============================================================================


PILOT_CLIPS: Tuple[str, ...] = ("dog", "cows", "bmx-trees")

# Configs evaluated by the GO gate + the main pilot table.
# NOTE: handoff spec lists 4 configs per clip (3 GO + config4 δ-only-local-
# random). Config4 requires K_ins=0 phantom-mode support in
# `run_vadi_for_clip`, which is not landed. The 3 configs below are
# sufficient for GO/NO-GO; config4 (main-table row 7/8) is TODO.
PILOT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "K1_top":    {"K_ins": 1, "vulnerability_mode": "top",    "seed": 0},
    "K1_random": {"K_ins": 1, "vulnerability_mode": "random", "seed": 0},
    "K3_top":    {"K_ins": 3, "vulnerability_mode": "top",    "seed": 0},
    # "delta_only_local_random": {"K_ins": 0, ...}   # TODO: K_ins=0 path
}
N_GO_CONFIGS: int = len(PILOT_CONFIGS)                 # currently 3

GO_COND1_MIN_DELTA: float = 0.05      # top − random
GO_COND2_MIN_J_DROP: float = 0.20     # K=3 top absolute
GO_MIN_CLIPS_PASS: int = 2            # out of 3 clips

DIAGNOSTIC_DELTA_MU_RATIO: float = 2.0   # Δmu_decoy ≥ ratio · max(0, -Δmu_true)


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class ClipConfigRecord:
    """A single (clip, config) outcome plus the diagnostic Δmu decomposition."""

    clip: str
    config: str
    infeasible: bool
    best_surrogate_J_drop: float
    delta_mu_decoy: float      # mean_t(mu_decoy_best_step − mu_decoy_clean_proxy)
    delta_mu_true: float       # mean_t(mu_true_best_step − mu_true_clean_proxy)
    W_attacked: List[int]

    @classmethod
    def from_vadi_output(cls, clip: str, out: VADIClipOutput) -> "ClipConfigRecord":
        # Δmu proxy: last_feasible − first_logged_step. The first logged
        # step is already post-forward of a (near-)zero-δ state AND post-
        # λ=0 attack-only move, so this is NOT truly μ_clean — it's a
        # "from-initial-PGD-step" shift. For the pilot diagnostic flag
        # this is acceptable (FINAL_PROPOSAL.md marks the check as "flag
        # at pilot, not blocking GO"). For the paper-ready decomposition
        # in the DAVIS-10 main run, the caller should pass μ_clean from a
        # dedicated clean-SAM2 forward (a TODO extension, not needed here).
        if not out.step_log_summary:
            return cls(clip=clip, config=out.config_name,
                       infeasible=out.infeasible,
                       best_surrogate_J_drop=out.best_surrogate_J_drop,
                       delta_mu_decoy=0.0, delta_mu_true=0.0,
                       W_attacked=out.W)
        first = out.step_log_summary[0]
        last_feasible = next(
            (log for log in reversed(out.step_log_summary) if log["feasible"]),
            out.step_log_summary[-1],
        )

        def _mean(d: Dict[Any, float]) -> float:
            if not d:
                return 0.0
            return float(sum(d.values()) / len(d))

        d_mu_decoy = _mean(last_feasible["mu_decoy_trace"]) \
            - _mean(first["mu_decoy_trace"])
        d_mu_true = _mean(last_feasible["mu_true_trace"]) \
            - _mean(first["mu_true_trace"])
        return cls(
            clip=clip, config=out.config_name,
            infeasible=out.infeasible,
            best_surrogate_J_drop=out.best_surrogate_J_drop,
            delta_mu_decoy=d_mu_decoy,
            delta_mu_true=d_mu_true,
            W_attacked=out.W,
        )


@dataclass
class PilotDecision:
    go: bool
    cond1_pass: bool                    # top − random ≥ 0.05 on ≥2/3 clips
    cond2_pass: bool                    # K3_top ≥ 0.20 on ≥2/3 clips
    cond1_hits: int
    cond2_hits: int
    diagnostic_pass: bool               # Δmu_decoy dominates, ≥2/3 clips
    diagnostic_hits: int
    per_clip: Dict[str, Dict[str, Any]]
    records: List[ClipConfigRecord]
    n_clips: int


# =============================================================================
# GO/NO-GO decision logic
# =============================================================================


def decide_go_no_go(
    records: Sequence[ClipConfigRecord],
    clips: Sequence[str] = PILOT_CLIPS,
    min_clips_pass: int = GO_MIN_CLIPS_PASS,
) -> PilotDecision:
    """Evaluate the pre-committed GO conditions on a set of clip records.

    Infeasibility on a primary config = failure of that condition on that
    clip (per F16 primary-denominator policy). Missing configs are treated
    as failure.
    """
    # Guard against duplicate (clip, config) tuples — accidentally merging
    # records across reruns would silently overwrite earlier entries.
    seen: set = set()
    for r in records:
        key = (r.clip, r.config)
        if key in seen:
            raise ValueError(
                f"duplicate record for (clip={r.clip!r}, config={r.config!r}); "
                f"upstream rerun concatenation bug?")
        seen.add(key)
    by_key: Dict[Tuple[str, str], ClipConfigRecord] = {
        (r.clip, r.config): r for r in records
    }

    cond1_hits = 0
    cond2_hits = 0
    diagnostic_hits = 0
    per_clip: Dict[str, Dict[str, Any]] = {}

    for clip in clips:
        k1_top = by_key.get((clip, "K1_top"))
        k1_rand = by_key.get((clip, "K1_random"))
        k3_top = by_key.get((clip, "K3_top"))

        # Cond 1: top - random ≥ 0.05 (both must be feasible).
        delta_j: Optional[float] = None
        c1 = False
        if k1_top and k1_rand and not k1_top.infeasible and not k1_rand.infeasible:
            delta_j = (k1_top.best_surrogate_J_drop
                       - k1_rand.best_surrogate_J_drop)
            c1 = delta_j >= GO_COND1_MIN_DELTA

        # Cond 2: K3_top ≥ 0.20 (must be feasible).
        c2 = False
        if k3_top and not k3_top.infeasible:
            c2 = k3_top.best_surrogate_J_drop >= GO_COND2_MIN_J_DROP

        # Diagnostic (on K=3 top): decoy mean up AND dominates any true mean dip.
        d_pass = False
        if k3_top and not k3_top.infeasible:
            dm_d = k3_top.delta_mu_decoy
            dm_t = k3_top.delta_mu_true
            d_pass = (dm_d > 0.0
                      and dm_d >= DIAGNOSTIC_DELTA_MU_RATIO * max(0.0, -dm_t))

        if c1:
            cond1_hits += 1
        if c2:
            cond2_hits += 1
        if d_pass:
            diagnostic_hits += 1

        per_clip[clip] = {
            "cond1_top_minus_random_delta": delta_j,
            "cond1_pass": c1,
            "cond2_K3_top_J_drop": (
                k3_top.best_surrogate_J_drop if k3_top else None),
            "cond2_pass": c2,
            "diagnostic_delta_mu_decoy": (
                k3_top.delta_mu_decoy if k3_top else None),
            "diagnostic_delta_mu_true": (
                k3_top.delta_mu_true if k3_top else None),
            "diagnostic_pass": d_pass,
        }

    cond1_pass = cond1_hits >= min_clips_pass
    cond2_pass = cond2_hits >= min_clips_pass
    diagnostic_pass = diagnostic_hits >= min_clips_pass
    go = cond1_pass and cond2_pass

    return PilotDecision(
        go=go,
        cond1_pass=cond1_pass, cond2_pass=cond2_pass,
        cond1_hits=cond1_hits, cond2_hits=cond2_hits,
        diagnostic_pass=diagnostic_pass, diagnostic_hits=diagnostic_hits,
        per_clip=per_clip,
        records=list(records),
        n_clips=len(clips),
    )


# =============================================================================
# DAVIS loader (real mode)
# =============================================================================


def load_davis_clip(
    davis_root: Path, clip_name: str, device: Optional[torch.device] = None,
) -> Tuple[Tensor, np.ndarray]:
    """Load DAVIS 480p clip as `(x_clean [T,H,W,3] in [0,1], prompt_mask [H,W] uint8)`.

    Prompt is the frame-0 annotation binarized (any non-zero id → foreground).
    """
    from PIL import Image                                 # noqa: WPS433
    davis_root = Path(davis_root)
    img_dir = davis_root / "JPEGImages" / "480p" / clip_name
    ann_path = davis_root / "Annotations" / "480p" / clip_name / "00000.png"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"DAVIS clip not found at {img_dir}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"DAVIS annotation missing at {ann_path}")

    frame_files = sorted(img_dir.glob("*.jpg"))
    frames = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
              for f in frame_files]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    x = torch.from_numpy(arr)
    if device is not None:
        x = x.to(device)

    ann = np.asarray(Image.open(ann_path), dtype=np.uint8)
    prompt_mask = (ann > 0).astype(np.uint8)
    return x, prompt_mask


# =============================================================================
# Main pilot runner (injectable adapters — callers supply SAM2/LPIPS/SSIM)
# =============================================================================


def run_pilot(
    clips: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    davis_root: Path,
    out_root: Path,
    clean_pass_fn_factory: Callable[[str, Tensor, np.ndarray], Callable],
    forward_fn_builder_factory: Callable[[str, Tensor, np.ndarray], Callable],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    clip_loader: Callable[[Path, str], Tuple[Tensor, np.ndarray]] = load_davis_clip,
    config_builder: Callable[[], VADIConfig] = VADIConfig,
    device: Optional[torch.device] = None,
) -> PilotDecision:
    """Run every (clip, config) pair, build records, and decide GO/NO-GO.

    `clean_pass_fn_factory(clip_name, x_clean, prompt_mask)` returns the
    per-clip closure `run_vadi_for_clip` expects; same for
    `forward_fn_builder_factory`. Factories exist to let the runner build
    per-clip adapters without sharing state across clips.
    """
    records: List[ClipConfigRecord] = []
    for clip_name in clips:
        x_clean, prompt_mask = clip_loader(davis_root, clip_name)
        if device is not None:
            x_clean = x_clean.to(device)
        clean_pass_fn = clean_pass_fn_factory(clip_name, x_clean, prompt_mask)
        forward_fn_builder = forward_fn_builder_factory(
            clip_name, x_clean, prompt_mask)

        for config_name, kwargs in configs.items():
            vadi_cfg = config_builder()
            rng = np.random.default_rng(int(kwargs.get("seed", 0)))
            out: VADIClipOutput = run_vadi_for_clip(
                clip_name=clip_name, config_name=config_name,
                x_clean=x_clean, prompt_mask=prompt_mask,
                clean_pass_fn=clean_pass_fn,
                forward_fn_builder=forward_fn_builder,
                lpips_fn=lpips_fn, ssim_fn=ssim_fn,
                vulnerability_mode=kwargs["vulnerability_mode"],
                K_ins=int(kwargs["K_ins"]),
                min_gap=int(kwargs.get("min_gap", 2)),
                rng=rng,
                config=vadi_cfg,
                out_root=out_root,
            )
            records.append(ClipConfigRecord.from_vadi_output(clip_name, out))

    decision = decide_go_no_go(records, clips=tuple(clips))
    # Persist the pilot summary for paper reproducibility.
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "pilot_decision.json", "w", encoding="utf-8") as f:
        json.dump({
            "go": decision.go,
            "cond1_pass": decision.cond1_pass,
            "cond2_pass": decision.cond2_pass,
            "cond1_hits": decision.cond1_hits,
            "cond2_hits": decision.cond2_hits,
            "diagnostic_pass": decision.diagnostic_pass,
            "diagnostic_hits": decision.diagnostic_hits,
            "per_clip": decision.per_clip,
            "thresholds": {
                "GO_COND1_MIN_DELTA": GO_COND1_MIN_DELTA,
                "GO_COND2_MIN_J_DROP": GO_COND2_MIN_J_DROP,
                "GO_MIN_CLIPS_PASS": GO_MIN_CLIPS_PASS,
                "DIAGNOSTIC_DELTA_MU_RATIO": DIAGNOSTIC_DELTA_MU_RATIO,
            },
            "records": [asdict(r) for r in decision.records],
            "n_clips": decision.n_clips,
        }, f, indent=2, default=str)
    return decision


# =============================================================================
# Real SAM2 wiring (lazy-imported so --dry-run works without sam2/lpips)
# =============================================================================


def build_pilot_adapters(
    checkpoint_path: str, device: torch.device,
) -> Tuple[Callable, Callable, Callable, Callable]:
    """Build `(clean_pass_fn_factory, forward_fn_builder_factory, lpips,
    ssim)` for real SAM2.1 + LPIPS(alex) + SSIM on Pro 6000.

    Delegates the heavy SAM2 / LPIPS setup to
    `memshield.vadi_sam2_wiring.build_sam2_lpips_ssim`; the factories
    returned here are thin per-clip closures that:
      - cache the `CleanPassOutput` so all configs for a clip share ONE
        SAM2 clean pass (3 configs × 3 clips → 3 clean passes, not 9);
      - build a fresh `VADIForwardFn` per (clip, config) pairing (cheap —
        only copies prompt mask + normalization constants). The predictor
        is reused across everything.

    Autograd + device contract
    --------------------------
    - Predictor parameters frozen via `requires_grad_(False)` in
      `build_sam2_lpips_ssim`; gradient flows only through
      `x_processed` → Hiera → memory_attention → mask_decoder.
    - bf16 autocast is enabled inside VADIForwardFn; outputs are cast
      back to fp32 before returning so `vadi_loss`'s margin / LPIPS /
      SSIM terms see stable precision.
    - `x_clean` is moved to `device` inside the clean-pass helper if
      needed; PGD leaves (`delta`, `nu`) are allocated on whatever device
      `x_clean` ends up on after `run_pilot(...)` preps it.

    Pilot-config fanout
    -------------------
    This pilot runs 3 clips × 3 configs (K1_top / K1_random / K3_top).
    The `δ_only_local_random` config requires a `K_ins=0` phantom path
    in `run_vadi_for_clip` that is not yet landed — it's a main-table
    row (7/8), NOT a GO/NO-GO blocker. See `PILOT_CONFIGS` above.
    """
    # Lazy imports — not executed on --dry-run or self-test paths.
    from memshield.vadi_sam2_wiring import (                          # noqa: WPS433
        VADIForwardFn, build_sam2_lpips_ssim, clean_pass_vadi,
    )

    predictor, lpips_fn, ssim_fn = build_sam2_lpips_ssim(
        checkpoint_path=str(checkpoint_path), device=device,
    )

    def clean_pass_fn_factory(
        clip_name: str, x_clean: Tensor, prompt_mask: np.ndarray,
    ) -> Callable:
        cache: Dict[str, Any] = {"out": None}

        def clean_pass(x: Tensor, prompt: np.ndarray):
            if cache["out"] is None:
                cache["out"] = clean_pass_vadi(
                    predictor, x, prompt, device,
                )
            return cache["out"]

        return clean_pass

    def forward_fn_builder_factory(
        clip_name: str, x_clean: Tensor, prompt_mask: np.ndarray,
    ) -> Callable:
        H_vid = int(x_clean.shape[1])
        W_vid = int(x_clean.shape[2])

        def forward_fn_builder(
            x_clean: Tensor, prompt_mask: np.ndarray, W: Sequence[int],
        ) -> Callable:
            return VADIForwardFn(
                predictor=predictor, prompt_mask=prompt_mask,
                video_H=H_vid, video_W=W_vid, device=device,
            )

        return forward_fn_builder

    return clean_pass_fn_factory, forward_fn_builder_factory, lpips_fn, ssim_fn


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VADI pilot gate (3 clips × 4 configs → GO/NO-GO).")
    p.add_argument("--davis-root", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="SAM2.1 tiny checkpoint path.")
    p.add_argument("--out-root", default="vadi_runs/pilot")
    p.add_argument("--clips", nargs="+", default=list(PILOT_CLIPS),
                   help=f"DAVIS clip names. Default: {PILOT_CLIPS}.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run and exit.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("[pilot] dry-run scope:")
        print(f"  clips = {args.clips}")
        for cfg_name, cfg in PILOT_CONFIGS.items():
            print(f"  config {cfg_name}: {cfg}")
        print(f"  out_root = {out_root}")
        print(f"  checkpoint = {args.checkpoint}")
        print("[pilot] GO thresholds (pre-committed):")
        print(f"  cond1: top - random >= {GO_COND1_MIN_DELTA}, "
              f">= {GO_MIN_CLIPS_PASS}/3 clips")
        print(f"  cond2: K3_top J-drop >= {GO_COND2_MIN_J_DROP}, "
              f">= {GO_MIN_CLIPS_PASS}/3 clips")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        clean_fac, fwd_fac, lp, ss = build_pilot_adapters(
            args.checkpoint, device)
    except (NotImplementedError, ImportError) as e:
        # NotImplementedError: adapter stub (pre-wiring).
        # ImportError: running on a host without `sam2` / `lpips` installed
        # (e.g., a Windows dev machine) — pipeline cannot execute and we
        # return 2 so self-tests and CI stay deterministic.
        print(f"[pilot] cannot run real pipeline: {e}", file=sys.stderr)
        return 2

    decision = run_pilot(
        clips=args.clips, configs=PILOT_CONFIGS,
        davis_root=Path(args.davis_root), out_root=out_root,
        clean_pass_fn_factory=clean_fac,
        forward_fn_builder_factory=fwd_fac,
        lpips_fn=lp, ssim_fn=ss,
        device=device,
    )
    n = decision.n_clips
    print(f"[pilot] GO = {decision.go}")
    print(f"[pilot]   cond1 (top-vs-random) pass: {decision.cond1_pass} "
          f"({decision.cond1_hits}/{n} clips)")
    print(f"[pilot]   cond2 (K3_top absolute) pass: {decision.cond2_pass} "
          f"({decision.cond2_hits}/{n} clips)")
    print(f"[pilot]   diagnostic (d_mu_decoy dominates) pass: "
          f"{decision.diagnostic_pass} ({decision.diagnostic_hits}/{n} clips)")
    return 0 if decision.go else 1


# =============================================================================
# Self-test (stub adapters; no SAM2/LPIPS deps)
# =============================================================================


def _self_test() -> None:
    import tempfile
    torch.manual_seed(0)
    np.random.seed(0)

    # -- decide_go_no_go: GO case (both conds pass on 3/3 clips) ----------
    def rec(clip, config, J, feas=True, dmd=0.5, dmt=-0.1):
        return ClipConfigRecord(
            clip=clip, config=config, infeasible=not feas,
            best_surrogate_J_drop=J,
            delta_mu_decoy=dmd, delta_mu_true=dmt,
            W_attacked=[5],
        )

    strong = [
        rec("dog", "K1_top", 0.30), rec("dog", "K1_random", 0.10),
        rec("dog", "K3_top", 0.40),
        rec("cows", "K1_top", 0.25), rec("cows", "K1_random", 0.10),
        rec("cows", "K3_top", 0.35),
        rec("bmx-trees", "K1_top", 0.20), rec("bmx-trees", "K1_random", 0.10),
        rec("bmx-trees", "K3_top", 0.30),
    ]
    d = decide_go_no_go(strong, clips=("dog", "cows", "bmx-trees"))
    assert d.go is True
    assert d.cond1_hits == 3
    assert d.cond2_hits == 3
    assert d.diagnostic_hits == 3

    # -- NO-GO: cond1 fails (delta < 0.05) on all clips --------------------
    weak_gap = [
        rec("dog", "K1_top", 0.12), rec("dog", "K1_random", 0.10),
        rec("dog", "K3_top", 0.35),
        rec("cows", "K1_top", 0.12), rec("cows", "K1_random", 0.10),
        rec("cows", "K3_top", 0.35),
        rec("bmx-trees", "K1_top", 0.12), rec("bmx-trees", "K1_random", 0.10),
        rec("bmx-trees", "K3_top", 0.35),
    ]
    d_nogo1 = decide_go_no_go(weak_gap, clips=("dog", "cows", "bmx-trees"))
    assert d_nogo1.go is False
    assert d_nogo1.cond1_pass is False
    assert d_nogo1.cond2_pass is True                   # unaffected

    # -- NO-GO: cond2 fails (K3_top < 0.20) on majority -----------------
    weak_abs = [
        rec("dog", "K1_top", 0.30), rec("dog", "K1_random", 0.10),
        rec("dog", "K3_top", 0.15),
        rec("cows", "K1_top", 0.30), rec("cows", "K1_random", 0.10),
        rec("cows", "K3_top", 0.18),
        rec("bmx-trees", "K1_top", 0.30), rec("bmx-trees", "K1_random", 0.10),
        rec("bmx-trees", "K3_top", 0.25),
    ]
    d_nogo2 = decide_go_no_go(weak_abs, clips=("dog", "cows", "bmx-trees"))
    assert d_nogo2.go is False
    assert d_nogo2.cond1_pass is True
    assert d_nogo2.cond2_pass is False                  # 1/3 passes only

    # -- Edge: infeasibility on a primary config counts as failure ---------
    with_infeasible = [
        rec("dog", "K1_top", 0.30, feas=False),         # infeasible → cond1 fail
        rec("dog", "K1_random", 0.10), rec("dog", "K3_top", 0.35),
        rec("cows", "K1_top", 0.30), rec("cows", "K1_random", 0.10),
        rec("cows", "K3_top", 0.35),
        rec("bmx-trees", "K1_top", 0.30), rec("bmx-trees", "K1_random", 0.10),
        rec("bmx-trees", "K3_top", 0.35),
    ]
    d_inf = decide_go_no_go(with_infeasible, clips=("dog", "cows", "bmx-trees"))
    assert d_inf.cond1_hits == 2                        # dog failed
    assert d_inf.go is True                             # still ≥ 2/3

    # -- Missing config in records counts as failure -----------------------
    partial = [
        rec("dog", "K1_top", 0.30), rec("dog", "K1_random", 0.10),
        # no K3_top for dog
        rec("cows", "K1_top", 0.30), rec("cows", "K1_random", 0.10),
        rec("cows", "K3_top", 0.35),
        rec("bmx-trees", "K1_top", 0.30), rec("bmx-trees", "K1_random", 0.10),
        rec("bmx-trees", "K3_top", 0.35),
    ]
    d_partial = decide_go_no_go(partial, clips=("dog", "cows", "bmx-trees"))
    assert d_partial.cond2_hits == 2                    # dog missing → fail
    assert d_partial.go is True                         # still 2/3

    # -- Diagnostic: decoy up, but true dropped HARD → ratio violated ------
    # Δmu_decoy = 0.1, Δmu_true = -0.1 → 2 · max(0, 0.1) = 0.2; 0.1 < 0.2 fail.
    diag_fail = [
        rec("dog", "K3_top", 0.35, dmd=0.10, dmt=-0.10),
        rec("cows", "K3_top", 0.35, dmd=0.10, dmt=-0.10),
        rec("bmx-trees", "K3_top", 0.35, dmd=0.10, dmt=-0.10),
    ]
    d_diag = decide_go_no_go(diag_fail, clips=("dog", "cows", "bmx-trees"))
    assert d_diag.diagnostic_hits == 0
    assert d_diag.diagnostic_pass is False

    # -- Diagnostic: decoy dominates → pass --------------------------------
    diag_ok = [
        rec("dog", "K3_top", 0.35, dmd=0.50, dmt=0.0),
        rec("cows", "K3_top", 0.35, dmd=0.50, dmt=0.0),
        rec("bmx-trees", "K3_top", 0.35, dmd=0.50, dmt=0.0),
    ]
    d_diag_ok = decide_go_no_go(diag_ok, clips=("dog", "cows", "bmx-trees"))
    assert d_diag_ok.diagnostic_hits == 3

    # -- Duplicate (clip, config) tuples must raise -----------------------
    try:
        decide_go_no_go(
            strong + [rec("dog", "K1_top", 0.99)],             # duplicate
            clips=("dog", "cows", "bmx-trees"),
        )
        raise AssertionError("duplicate (clip,config) must raise")
    except ValueError:
        pass

    # -- End-to-end run_pilot with stub adapters --------------------------
    # Factories return closures matching the `run_vadi.py` contract.
    Hv, Wv = 8, 8
    T_clean = 6

    def stub_clip_loader(davis_root: Path, clip_name: str) -> Tuple[Tensor, np.ndarray]:
        torch.manual_seed(hash(clip_name) % (2 ** 31))
        x = torch.rand(T_clean, Hv, Wv, 3)
        m = np.zeros((Hv, Wv), dtype=np.uint8); m[Hv // 2:, :Wv // 2] = 1
        return x, m

    def stub_clean_fac(clip: str, x_clean: Tensor, prompt: np.ndarray):
        def fn(x_clean_t: Tensor, prompt_np: np.ndarray):
            from scripts.run_vadi import CleanPassOutput
            T = x_clean_t.shape[0]
            pseudo = []
            for t in range(T):
                m = np.zeros((Hv, Wv), dtype=np.float32)
                # Shift a tiny amount each frame so ranks are non-degenerate.
                y0 = min(t, Hv - 4); x0 = min(t, Wv - 4)
                m[y0:y0 + 4, x0:x0 + 4] = 1.0
                pseudo.append(m)
            conf = np.linspace(0.9, 0.3, T).astype(np.float32)
            feats = [np.random.randn(16).astype(np.float32) for _ in range(T)]
            return CleanPassOutput(
                pseudo_masks=pseudo, confidences=conf, hiera_features=feats)
        return fn

    def stub_forward_fac(clip: str, x_clean: Tensor, prompt: np.ndarray):
        def builder(x_clean, prompt_mask, W):
            def fn(processed: Tensor, return_at):
                return {t: 3.0 * (processed[t].mean(dim=-1) - 0.5)
                        for t in return_at}
            return fn
        return builder

    def lpips_stub(x, y): return (x - y).abs().mean()

    def ssim_stub(x, y): return 1.0 - (x - y).pow(2).mean()

    small_cfg = lambda: VADIConfig(N_1=1, N_2=1, N_3=1,
                                    lambda_init=1.0,
                                    lambda_growth_factor=2.0,
                                    lambda_growth_period=1)

    with tempfile.TemporaryDirectory() as td:
        decision = run_pilot(
            clips=("clipA", "clipB", "clipC"),        # 3 stub clips
            configs=PILOT_CONFIGS,
            davis_root=Path(td), out_root=Path(td) / "pilot",
            clean_pass_fn_factory=stub_clean_fac,
            forward_fn_builder_factory=stub_forward_fac,
            lpips_fn=lpips_stub, ssim_fn=ssim_stub,
            clip_loader=stub_clip_loader,
            config_builder=small_cfg,
        )
        # 3 clips × 3 configs = 9 records.
        assert len(decision.records) == 9
        # pilot_decision.json was written.
        assert (Path(td) / "pilot" / "pilot_decision.json").exists()
        # GO value is a bool (value itself depends on the stub's signal —
        # the test just asserts the aggregation pipeline ran without
        # exceptions).
        assert isinstance(decision.go, bool)

    # -- CLI --dry-run ----------------------------------------------------
    rv = main(["--davis-root", "/tmp", "--checkpoint", "/tmp/ckpt.pt",
               "--dry-run"])
    assert rv == 0

    # -- CLI real mode fails gracefully without SAM2 wiring ---------------
    # On a dev host WITHOUT `sam2`/`lpips` installed, `build_pilot_adapters`
    # raises ImportError → main() returns 2. On Pro 6000 (or any host with
    # SAM2 installed), the path reaches real predictor construction and
    # would fail with FileNotFoundError on the bogus checkpoint path — a
    # loud failure is correct for real runs. We only assert rv==2 on hosts
    # that can't run the real pipeline; hosts with SAM2 get a skip message.
    try:
        import sam2  # noqa: F401, WPS433
        sam2_available = True
    except ImportError:
        sam2_available = False
    if not sam2_available:
        rv = main(["--davis-root", "/tmp", "--checkpoint", "/tmp/ckpt.pt"])
        assert rv == 2, (
            f"expected rv=2 on host without SAM2 (ImportError path); "
            f"got {rv}")
    else:
        print("  [self-test skip] SAM2 installed — rv==2 check only "
              "meaningful on dev hosts without sam2/lpips")

    print("scripts.run_vadi_pilot: all self-tests PASSED "
          "(GO/NO-GO logic: GO, fail-cond1, fail-cond2, infeasibility, "
          "missing-config, diagnostic fail/pass, stub end-to-end, CLI)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
