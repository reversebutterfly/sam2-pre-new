"""VADI decisive round — 10 clips × 4 configs, post-pilot triage.

Written 2026-04-24 after pilot NO-GO. Pilot cond1 (K1_top vs K1_random)
failed 0/3 and diagnostic (Δμ_decoy dominates) failed 0/3; but cond2
(K3_top absolute) passed 3/3 with mean exported J-drop 0.377.

Per post-pilot research-review (codex xhigh, thread 019dbd43-...,
`REVIEW_POST_PILOT_2026-04-24.md`), the correct next step is a hard-gated
decisive round to resolve whether the narrower "attack-works for data
protection" claim is defensible:

    1. `K3_top`                — positive control (the method itself).
    2. `K3_random`             — placement necessity at K=3 (untested by pilot).
    3. `K3_delta_only_top`     — insert necessity: pick top-3 vulnerability
                                  positions as PHANTOM centers, do NOT insert,
                                  run δ only on their ±2 neighborhoods.
    4. `K3_insert_only_top`    — δ necessity: normal K=3 inserts BUT freeze
                                  δ at 0; only ν optimizes insert content.

Decisive decision (all on exported_j_drop, paper-claim metric):

    PLACEMENT_WIN_i    := K3_top[i] - K3_random[i]        >= 0.05
    INSERT_WIN_i       := K3_top[i] - K3_delta_only[i]    >= 0.10
    DELTA_WIN_i        := K3_top[i] - K3_insert_only[i]   >= 0.05

    NARROWED_PROCEED if  #PLACEMENT_WIN >= 7/10  AND  #INSERT_WIN >= 7/10
    AUDIT_PIVOT       otherwise

`DELTA_WIN` is reported but is informational: the narrowed paper can still
proceed if δ is not the key contributor (the paper becomes "insert-based
poisoning" instead of "insert + δ"), but if δ does not help we drop the
δ component from the method entirely.

Compute budget: 10 clips × 4 configs × ~5 min (K=3, 100-step PGD,
gradient checkpointing) ≈ 3.3 GPU-hours on Pro 6000 GPU1.

Bare run → self-tests with stub adapters. Real run is `build_pilot_adapters()`
from `scripts.run_vadi_pilot` (same SAM2 / LPIPS / SSIM wiring).
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
from scripts.run_vadi_pilot import ClipConfigRecord


# =============================================================================
# Scope + decisive thresholds (FROZEN — do not edit without updating the
# post-pilot review doc in parallel)
# =============================================================================


DAVIS10_CLIPS_DEFAULT: Tuple[str, ...] = (
    "dog", "cows", "bmx-trees",                              # pilot clips
    "blackswan", "camel", "motocross-jump",
    "car-roundabout", "dance-twirl", "drift-straight", "soapbox",
)

# Four configs per clip. The keys match `PILOT_CONFIGS` convention so the
# `build_pilot_adapters` wiring carries over unchanged.
DECISIVE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "K3_top": {
        "K_ins": 3, "K_phantom": 0,
        "vulnerability_mode": "top", "seed": 0, "freeze_delta": False,
    },
    "K3_random": {
        "K_ins": 3, "K_phantom": 0,
        "vulnerability_mode": "random", "seed": 0, "freeze_delta": False,
    },
    "K3_delta_only_top": {
        "K_ins": 0, "K_phantom": 3,
        "vulnerability_mode": "top", "seed": 0, "freeze_delta": False,
    },
    "K3_insert_only_top": {
        "K_ins": 3, "K_phantom": 0,
        "vulnerability_mode": "top", "seed": 0, "freeze_delta": True,
    },
}

# Decisive win thresholds (per-clip, paired comparison).
PLACEMENT_WIN_MIN_DELTA: float = 0.05     # K3_top - K3_random
INSERT_WIN_MIN_DELTA:    float = 0.10     # K3_top - K3_delta_only
DELTA_WIN_MIN_DELTA:     float = 0.05     # K3_top - K3_insert_only  (informational)
# Pass fraction — 7/10 = 70% by default. Scaled to n_clips so custom subsets
# get a proportional bar instead of the fixed "7" from the 10-clip plan.
DECISIVE_MIN_FRACTION: float = 0.70
DECISIVE_PLANNED_N_CLIPS: int = 10        # the pre-committed scope


def _required_hits(n_clips: int) -> int:
    """Scale the win-threshold to the actual clip count.

    The pre-committed 10-clip × 4-config plan calls for ≥ 7/10 wins. For
    any custom subset we keep the same fraction (70%) rather than the
    absolute "7" — running 3 clips and reporting `NARROWED_PROCEED` on
    1-2 wins would be scientifically meaningless.
    """
    if n_clips <= 0:
        return 1
    import math
    return max(1, int(math.ceil(DECISIVE_MIN_FRACTION * n_clips)))


# =============================================================================
# Decision container
# =============================================================================


@dataclass
class DecisiveDecision:
    proceed: bool                          # NARROWED_PROCEED vs AUDIT_PIVOT
    decisive_result: str                   # "NARROWED_PROCEED" | "AUDIT_PIVOT"
    placement_pass: bool
    insert_pass: bool
    delta_informational: bool              # δ actually helps (>= threshold)
    placement_win_count: int
    insert_win_count: int
    delta_win_count: int
    n_clips: int
    per_clip: Dict[str, Dict[str, Any]]
    mean_j_drop: Dict[str, float]          # config → 10-clip mean
    records: List[ClipConfigRecord]


def decide_decisive(
    records: List[ClipConfigRecord], clips: Sequence[str],
    *,
    allow_surrogate_gate: bool = False,
) -> DecisiveDecision:
    """Aggregate paired per-clip deltas over the 4 configs and apply the
    decisive thresholds. Missing / infeasible metrics fail the check for
    that clip (do not silently impute 0).

    `allow_surrogate_gate` MUST stay False on the real decisive run — it
    is only exposed to let self-tests exercise the threshold logic with
    stub `ClipConfigRecord`s whose `exported_j_drop` is None.
    """
    # Index records by (clip, config) — DISALLOW duplicates to prevent
    # silent last-write-wins if logs from multiple attempts got merged.
    by_clip_cfg: Dict[Tuple[str, str], ClipConfigRecord] = {}
    for r in records:
        key = (r.clip, r.config)
        if key in by_clip_cfg:
            raise ValueError(
                f"decide_decisive: duplicate record for {key!r}. "
                "The decisive paired comparison requires exactly one "
                "record per (clip, config); concatenated re-runs must "
                "be de-duplicated upstream.")
        by_clip_cfg[key] = r

    def _m(clip: str, cfg: str) -> Optional[float]:
        rec = by_clip_cfg.get((clip, cfg))
        if rec is None or rec.infeasible:
            return None
        # Hard-assert that feasible decisive records use the paper-claim
        # metric, not the surrogate (the very gap pilot exposed). If a
        # real run lands here with gate_source=="surrogate", we refuse to
        # produce a decisive verdict.
        src = rec.gate_source()
        if src != "exported_j_drop" and not allow_surrogate_gate:
            raise RuntimeError(
                f"decide_decisive: record for {(clip, cfg)!r} reports "
                f"gate_source={src!r}; decisive verdict requires "
                "exported_j_drop on every feasible record. Pass "
                "allow_surrogate_gate=True only from self-tests.")
        return rec.gate_metric()

    per_clip: Dict[str, Dict[str, Any]] = {}
    placement_hits = 0
    insert_hits = 0
    delta_hits = 0
    for clip in clips:
        j_top = _m(clip, "K3_top")
        j_rand = _m(clip, "K3_random")
        j_dlt = _m(clip, "K3_delta_only_top")
        j_ins = _m(clip, "K3_insert_only_top")

        placement_delta = (
            (j_top - j_rand) if (j_top is not None and j_rand is not None)
            else None
        )
        insert_delta = (
            (j_top - j_dlt) if (j_top is not None and j_dlt is not None)
            else None
        )
        delta_delta = (
            (j_top - j_ins) if (j_top is not None and j_ins is not None)
            else None
        )

        placement_pass = bool(
            placement_delta is not None
            and placement_delta >= PLACEMENT_WIN_MIN_DELTA
        )
        insert_pass = bool(
            insert_delta is not None
            and insert_delta >= INSERT_WIN_MIN_DELTA
        )
        delta_pass = bool(
            delta_delta is not None
            and delta_delta >= DELTA_WIN_MIN_DELTA
        )

        if placement_pass:
            placement_hits += 1
        if insert_pass:
            insert_hits += 1
        if delta_pass:
            delta_hits += 1

        per_clip[clip] = {
            "J_K3_top":        j_top,
            "J_K3_random":     j_rand,
            "J_K3_delta_only": j_dlt,
            "J_K3_insert_only": j_ins,
            "placement_delta": placement_delta,
            "insert_delta":    insert_delta,
            "delta_delta":     delta_delta,
            "placement_pass":  placement_pass,
            "insert_pass":     insert_pass,
            "delta_pass":      delta_pass,
        }

    required = _required_hits(len(clips))
    placement_pass_total = placement_hits >= required
    insert_pass_total = insert_hits >= required
    delta_informational = delta_hits >= required

    proceed = bool(placement_pass_total and insert_pass_total)

    # Mean J-drop over ALL 10 clips per config (NaN-excluding).
    mean_j_drop: Dict[str, float] = {}
    for cfg in DECISIVE_CONFIGS.keys():
        vals = [_m(clip, cfg) for clip in clips]
        vals = [v for v in vals if v is not None]
        mean_j_drop[cfg] = float(np.mean(vals)) if vals else float("nan")

    return DecisiveDecision(
        proceed=proceed,
        decisive_result="NARROWED_PROCEED" if proceed else "AUDIT_PIVOT",
        placement_pass=placement_pass_total,
        insert_pass=insert_pass_total,
        delta_informational=delta_informational,
        placement_win_count=placement_hits,
        insert_win_count=insert_hits,
        delta_win_count=delta_hits,
        n_clips=len(clips),
        per_clip=per_clip,
        mean_j_drop=mean_j_drop,
        records=records,
    )


# =============================================================================
# Runner
# =============================================================================


def run_decisive(
    clips: Sequence[str],
    clip_loader: Callable[[Path, str], Tuple[Tensor, np.ndarray]],
    davis_root: Path,
    clean_pass_fn_factory: Callable[[str, Tensor, np.ndarray], Callable],
    forward_fn_builder_factory: Callable[[str, Tensor, np.ndarray], Callable],
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    sam2_eval_fn: Optional[Callable[[Tensor, np.ndarray], List[np.ndarray]]],
    out_root: Path,
    device: Optional[torch.device] = None,
    config_builder: Callable[[], VADIConfig] = VADIConfig,
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
    allow_surrogate_gate: bool = False,
) -> DecisiveDecision:
    """Run the 4-config decisive round on `clips` and return the decision.

    `sam2_eval_fn` must be supplied unless `allow_surrogate_gate=True`
    (matches `run_pilot` policy — codex R2 purity).
    """
    if configs is None:
        configs = DECISIVE_CONFIGS
    if sam2_eval_fn is None and not allow_surrogate_gate:
        raise RuntimeError(
            "run_decisive: sam2_eval_fn is None and allow_surrogate_gate=False. "
            "The decisive round MUST use exported_j_drop — surrogate is the "
            "very thing pilot exposed as a ~60pp overestimate. Pass "
            "allow_surrogate_gate=True only from self-tests.")

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
            if bool(kwargs.get("freeze_delta", False)):
                vadi_cfg.freeze_delta = True
            rng = np.random.default_rng(int(kwargs.get("seed", 0)))
            out: VADIClipOutput = run_vadi_for_clip(
                clip_name=clip_name, config_name=config_name,
                x_clean=x_clean, prompt_mask=prompt_mask,
                clean_pass_fn=clean_pass_fn,
                forward_fn_builder=forward_fn_builder,
                lpips_fn=lpips_fn, ssim_fn=ssim_fn,
                vulnerability_mode=kwargs["vulnerability_mode"],
                K_ins=int(kwargs.get("K_ins", 0)),
                K_phantom=int(kwargs.get("K_phantom", 0)),
                min_gap=int(kwargs.get("min_gap", 2)),
                rng=rng,
                config=vadi_cfg,
                out_root=out_root,
                sam2_eval_fn=sam2_eval_fn,
            )
            records.append(ClipConfigRecord.from_vadi_output(clip_name, out))

    decision = decide_decisive(records, clips=tuple(clips),
                               allow_surrogate_gate=allow_surrogate_gate)
    out_root.mkdir(parents=True, exist_ok=True)
    gate_sources = sorted({r.gate_source() for r in decision.records})
    with open(out_root / "decisive_decision.json", "w", encoding="utf-8") as f:
        json.dump({
            "decisive_result": decision.decisive_result,
            "proceed": decision.proceed,
            "gate_metric_sources_observed": gate_sources,
            "placement_pass": decision.placement_pass,
            "insert_pass": decision.insert_pass,
            "delta_informational": decision.delta_informational,
            "placement_win_count": decision.placement_win_count,
            "insert_win_count": decision.insert_win_count,
            "delta_win_count": decision.delta_win_count,
            "n_clips": decision.n_clips,
            "mean_j_drop": decision.mean_j_drop,
            "per_clip": decision.per_clip,
            "thresholds": {
                "PLACEMENT_WIN_MIN_DELTA": PLACEMENT_WIN_MIN_DELTA,
                "INSERT_WIN_MIN_DELTA": INSERT_WIN_MIN_DELTA,
                "DELTA_WIN_MIN_DELTA": DELTA_WIN_MIN_DELTA,
                "DECISIVE_MIN_FRACTION": DECISIVE_MIN_FRACTION,
                "required_hits_for_this_n_clips": _required_hits(
                    decision.n_clips),
                "DECISIVE_PLANNED_N_CLIPS": DECISIVE_PLANNED_N_CLIPS,
            },
            "records": [asdict(r) for r in decision.records],
        }, f, indent=2, default=str)
    return decision


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VADI decisive round (10 clips × 4 configs).")
    p.add_argument("--davis-root", required=True,
                   help="Path to DAVIS root (contains JPEGImages/480p).")
    p.add_argument("--checkpoint", required=True,
                   help="SAM2.1 tiny checkpoint path.")
    p.add_argument("--out-root", default="vadi_runs/decisive",
                   help="Directory for outputs + decisive_decision.json.")
    p.add_argument("--clips", nargs="+",
                   default=list(DAVIS10_CLIPS_DEFAULT),
                   help=f"Clip names (default = {DAVIS10_CLIPS_DEFAULT}).")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Optional subset of config names "
                        "(default = all 4 from DECISIVE_CONFIGS).")
    p.add_argument("--device", default="cuda",
                   help="cuda | cpu (default cuda).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        from scripts.run_vadi_pilot import (
            build_pilot_adapters, load_davis_clip,
        )
    except (ImportError, NotImplementedError) as e:
        print(f"[decisive] adapter import failed: {e}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    (clean_fac, fwd_fac, lpips_fn, ssim_fn, sam2_eval_fn) = \
        build_pilot_adapters(
            checkpoint_path=args.checkpoint, device=device,
        )

    subset: Optional[Dict[str, Dict[str, Any]]] = None
    if args.configs is not None:
        bad = set(args.configs) - set(DECISIVE_CONFIGS.keys())
        if bad:
            print(f"[decisive] unknown configs: {sorted(bad)}",
                  file=sys.stderr)
            return 2
        subset = {k: DECISIVE_CONFIGS[k] for k in args.configs}

    decision = run_decisive(
        clips=args.clips,
        clip_loader=load_davis_clip,
        davis_root=Path(args.davis_root),
        clean_pass_fn_factory=clean_fac,
        forward_fn_builder_factory=fwd_fac,
        lpips_fn=lpips_fn, ssim_fn=ssim_fn,
        sam2_eval_fn=sam2_eval_fn,
        out_root=Path(args.out_root),
        device=device,
        configs=subset,
    )

    print(f"[decisive] {decision.decisive_result}")
    print(f"[decisive]   placement_pass: {decision.placement_pass} "
          f"({decision.placement_win_count}/{decision.n_clips} clips)")
    print(f"[decisive]   insert_pass:    {decision.insert_pass} "
          f"({decision.insert_win_count}/{decision.n_clips} clips)")
    print(f"[decisive]   delta info:     {decision.delta_informational} "
          f"({decision.delta_win_count}/{decision.n_clips} clips)")
    print(f"[decisive]   mean J-drop:    {decision.mean_j_drop}")
    return 0 if decision.proceed else 1


# =============================================================================
# Self-test (stub adapters — exercises the 4-config loop + decision logic)
# =============================================================================


def _self_test() -> None:
    import tempfile
    from scripts.run_vadi import CleanPassOutput

    torch.manual_seed(0)
    np.random.seed(0)
    T, Hv, Wv = 14, 8, 8

    def stub_clip_loader(davis_root: Path, clip: str) -> Tuple[Tensor, np.ndarray]:
        x = torch.rand(T, Hv, Wv, 3)
        m = np.zeros((Hv, Wv), dtype=np.uint8)
        m[Hv // 2:, :Wv // 2] = 1
        return x, m

    def _stub_clean_pass(x_clean_t, prompt):
        T_local = x_clean_t.shape[0]
        Hv_, Wv_ = x_clean_t.shape[1], x_clean_t.shape[2]
        pseudo = []
        for t in range(T_local):
            m = np.zeros((Hv_, Wv_), dtype=np.float32)
            y0 = min(t, Hv_ - 4); x0 = min(t, Wv_ - 4)
            m[y0:y0 + 4, x0:x0 + 4] = 1.0
            pseudo.append(m)
        conf = np.where(np.arange(T_local) < T_local // 2,
                        0.9, 0.3).astype(np.float32)
        feats = [np.full(32, float(t) + (10.0 if t >= T_local // 2 else 0.0),
                         dtype=np.float32)
                 for t in range(T_local)]
        return CleanPassOutput(
            pseudo_masks=pseudo, confidences=conf, hiera_features=feats)

    def _stub_forward_fn(processed, return_at):
        out = {}
        for t in return_at:
            gray = processed[t].mean(dim=-1)
            out[t] = 3.0 * (gray - 0.5)
        return out

    def clean_fac(clip, x, m):
        return _stub_clean_pass

    def fwd_fac(clip, x, m):
        def builder(x_clean, prompt_mask, W):
            return _stub_forward_fn
        return builder

    def lpips_stub(x, y): return (x - y).abs().mean()
    def ssim_stub(x, y): return 1.0 - (x - y).pow(2).mean()

    # Stub sam2_eval_fn: returns `T_proc` all-zero masks so J_baseline == J_attacked
    # and gate_metric() returns 0 across the board (decision exercise).
    def stub_sam2_eval(video, prompt):
        return [np.zeros(prompt.shape, dtype=np.uint8)
                for _ in range(int(video.shape[0]))]

    def cfg_builder():
        return VADIConfig(
            N_1=1, N_2=1, N_3=1,
            lambda_init=1.0, lambda_growth_factor=2.0, lambda_growth_period=2,
        )

    with tempfile.TemporaryDirectory() as td:
        decision = run_decisive(
            clips=["clipA", "clipB"],
            clip_loader=stub_clip_loader,
            davis_root=Path(td),
            clean_pass_fn_factory=clean_fac,
            forward_fn_builder_factory=fwd_fac,
            lpips_fn=lpips_stub, ssim_fn=ssim_stub,
            sam2_eval_fn=stub_sam2_eval,
            out_root=Path(td) / "decisive_out",
            config_builder=cfg_builder,
        )
        # Sanity: the 4 configs × 2 clips produced 8 records.
        assert len(decision.records) == 8
        assert decision.decisive_result in {"NARROWED_PROCEED", "AUDIT_PIVOT"}
        # With stub sam2_eval_fn always returning zeros, all J_drops = 0 → no
        # wins on any axis → AUDIT_PIVOT.
        assert decision.decisive_result == "AUDIT_PIVOT"
        # decisive_decision.json persisted + parseable.
        path = Path(td) / "decisive_out" / "decisive_decision.json"
        assert path.exists()
        js = json.loads(path.read_text())
        assert "decisive_result" in js and "mean_j_drop" in js
        assert set(js["mean_j_drop"].keys()) == set(DECISIVE_CONFIGS.keys())

    # -- decide_decisive pure-logic test --------------------------------------
    def _rec(clip, cfg, j):
        return ClipConfigRecord(
            clip=clip, config=cfg, infeasible=False,
            best_surrogate_J_drop=0.0, exported_j_drop=j,
            delta_mu_decoy=0.0, delta_mu_true=0.0, W_attacked=[],
        )

    # Construct a deliberately-passing scenario: K3_top beats the other 3
    # baselines by ≥ threshold on 8/10 clips, fails on 2/10.
    clips10 = [f"c{i}" for i in range(10)]
    recs: List[ClipConfigRecord] = []
    for i, c in enumerate(clips10):
        if i < 8:
            recs.append(_rec(c, "K3_top",            0.50))
            recs.append(_rec(c, "K3_random",         0.20))   # Δ = 0.30 ≥ 0.05 ✓
            recs.append(_rec(c, "K3_delta_only_top", 0.30))   # Δ = 0.20 ≥ 0.10 ✓
            recs.append(_rec(c, "K3_insert_only_top", 0.40))  # Δ = 0.10 ≥ 0.05 ✓
        else:
            recs.append(_rec(c, "K3_top",            0.30))
            recs.append(_rec(c, "K3_random",         0.28))   # Δ = 0.02 ✗
            recs.append(_rec(c, "K3_delta_only_top", 0.25))   # Δ = 0.05 ✗ (<0.10)
            recs.append(_rec(c, "K3_insert_only_top", 0.29))  # Δ = 0.01 ✗
    dec = decide_decisive(recs, clips10)
    assert dec.proceed, \
        f"8/10 wins across all axes must proceed; got {dec.decisive_result}"
    assert dec.placement_win_count == 8
    assert dec.insert_win_count == 8
    assert dec.delta_win_count == 8

    # Counter-example: placement wins only 6/10 → fail (< 7/10).
    recs2 = []
    for i, c in enumerate(clips10):
        if i < 6:
            recs2.append(_rec(c, "K3_top",            0.50))
            recs2.append(_rec(c, "K3_random",         0.20))
            recs2.append(_rec(c, "K3_delta_only_top", 0.30))
            recs2.append(_rec(c, "K3_insert_only_top", 0.40))
        else:
            recs2.append(_rec(c, "K3_top",            0.50))
            recs2.append(_rec(c, "K3_random",         0.49))   # Δ = 0.01 ✗
            recs2.append(_rec(c, "K3_delta_only_top", 0.30))
            recs2.append(_rec(c, "K3_insert_only_top", 0.40))
    dec2 = decide_decisive(recs2, clips10)
    assert not dec2.proceed, \
        "placement wins 6/10 < 7 must not proceed"
    assert dec2.placement_win_count == 6
    assert dec2.insert_win_count == 10

    # Missing-record / infeasible handling: if K3_random record is absent for
    # a clip, that clip cannot contribute a placement_win; insert_win may
    # still pass independently.
    recs3 = []
    for c in clips10:
        recs3.append(_rec(c, "K3_top",            0.50))
        recs3.append(_rec(c, "K3_delta_only_top", 0.30))
        recs3.append(_rec(c, "K3_insert_only_top", 0.40))
    dec3 = decide_decisive(recs3, clips10)
    assert dec3.placement_win_count == 0, \
        "missing K3_random records must NOT impute a win"
    assert dec3.insert_win_count == 10
    assert not dec3.proceed

    # -- Threshold scales with n_clips (codex R3 High-1) ---------------------
    # 3 clips → required = ceil(0.7 * 3) = 3. Winning 2/3 must NOT pass.
    clips3 = ["c0", "c1", "c2"]
    recs_2of3 = []
    for i, c in enumerate(clips3):
        if i < 2:
            recs_2of3.append(_rec(c, "K3_top",            0.50))
            recs_2of3.append(_rec(c, "K3_random",         0.20))
            recs_2of3.append(_rec(c, "K3_delta_only_top", 0.30))
            recs_2of3.append(_rec(c, "K3_insert_only_top", 0.40))
        else:
            recs_2of3.append(_rec(c, "K3_top",            0.30))
            recs_2of3.append(_rec(c, "K3_random",         0.29))
            recs_2of3.append(_rec(c, "K3_delta_only_top", 0.29))
            recs_2of3.append(_rec(c, "K3_insert_only_top", 0.29))
    dec4 = decide_decisive(recs_2of3, clips3)
    assert _required_hits(3) == 3, \
        f"required hits for n=3 should be 3 (ceil(0.7*3)); got {_required_hits(3)}"
    assert not dec4.proceed, \
        "2/3 wins below required=3 must NOT proceed"
    # 3/3 does pass.
    for r in recs_2of3:
        if r.clip == "c2":
            if r.config == "K3_top": r.exported_j_drop = 0.50
            elif r.config == "K3_random": r.exported_j_drop = 0.20
            elif r.config == "K3_delta_only_top": r.exported_j_drop = 0.30
            elif r.config == "K3_insert_only_top": r.exported_j_drop = 0.40
    dec5 = decide_decisive(recs_2of3, clips3)
    assert dec5.proceed, "3/3 wins must proceed"

    # -- Duplicate (clip, config) records raise (codex R3 Medium) -----------
    recs_dup = [_rec("c0", "K3_top", 0.5), _rec("c0", "K3_top", 0.4)]
    try:
        decide_decisive(recs_dup, ["c0"])
        raise AssertionError("duplicate (clip, config) must raise")
    except ValueError:
        pass

    # -- Surrogate-only record raises in real run, passes with override -----
    surrogate_rec = ClipConfigRecord(
        clip="c0", config="K3_top", infeasible=False,
        best_surrogate_J_drop=0.9,
        exported_j_drop=None,                   # → gate_source() == "surrogate_*"
        delta_mu_decoy=0.0, delta_mu_true=0.0, W_attacked=[],
    )
    try:
        decide_decisive([surrogate_rec], ["c0"])
        raise AssertionError("surrogate gate must raise without override")
    except RuntimeError:
        pass
    # With override (self-test path): no raise, falls back to surrogate metric.
    dec_surr = decide_decisive([surrogate_rec], ["c0"],
                               allow_surrogate_gate=True)
    assert not dec_surr.proceed   # 1 record can't hit required=1 without K3_random counterpart

    print("scripts.run_vadi_decisive: all self-tests PASSED "
          "(4-config runner, decide_decisive threshold logic, "
          "paired-comparison edge cases, stub end-to-end, "
          "n-clip-scaled threshold, duplicate-record rejection, "
          "surrogate-gate hard-assert)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
