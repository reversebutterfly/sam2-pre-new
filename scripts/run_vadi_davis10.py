"""VADI DAVIS-10 main table orchestrator — runs ONLY after pilot = GO.

File 7 of 8 per HANDOFF_VADI_PILOT.md. Implements the 10-clip main table
and the pre-committed 8-claim success bar (§"Success Criteria",
FINAL_PROPOSAL.md).

Primary denominator = all 10 DAVIS clips. Infeasible = failure. ≥ 7/10
clips must satisfy each claim for the claim to be declared SUPPORTED.

Configs run (primary table rows 1-6 + 10):
  1. clean            (no PGD)
  2. K1_top           (centerpiece — standard)
  3. K3_top           (stronger variant — standard)
  4. K1_random × 5    (paired bootstrap, 5 seeds)
  5. K3_random × 5    (paired bootstrap, 5 seeds)
  6. K3_bottom        (placement causality 2)
 10. canonical        (W_clean=[6,12,14] — legacy comparison)

Configs NOT yet run (rows 7-9 require phantom-mode / ν-frozen paths that
are not landed in `run_vadi_for_clip`):
  7. top-δ-only K=0          → insert necessity
  8. random-δ-only K=0       → insert necessity placement-matched
  9. top-base-insert+δ (ν=0) → ν optimization necessity

Claims 1, 2, 3, 6 are evaluated; claims 4 (insert necessity), 5 (ν
optimization necessity), 7 (R2 restoration), 8 (R3 restoration) are
reported as "pending" — they need configs 7/9 above + the restoration
script (File 8).

Run `python scripts/run_vadi_davis10.py --help` for CLI; bare run →
self-tests with stub adapters.
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
from scripts.run_vadi_pilot import (
    ClipConfigRecord,
    build_pilot_adapters,
    load_davis_clip,
)


# =============================================================================
# DAVIS-10 scope (FROZEN)
# =============================================================================


# Composed of: 3 pilot clips + 7 well-studied DAVIS-val clips with varying
# motion / object-class / occlusion profiles. Frozen for reproducibility.
DAVIS10_CLIPS: Tuple[str, ...] = (
    "dog", "cows", "bmx-trees",                              # pilot
    "blackswan", "camel", "motocross-jump",
    "car-roundabout", "dance-twirl", "drift-straight", "soapbox",
)
N_RANDOM_DRAWS: int = 5                    # paired bootstrap draws per random row

# Success-bar thresholds (pre-committed — see FINAL_PROPOSAL.md §"Success Criteria").
CLAIM_J_DROP_MIN: float = 0.35             # claim 1
CLAIM_VS_RANDOM_MULT: float = 2.0          # claim 2: ours ≥ max(M·random, random+Δ)
CLAIM_VS_RANDOM_DELTA: float = 0.05
CLAIM_VS_BOTTOM_MULT: float = 3.0          # claim 3
CLAIM_VS_BOTTOM_DELTA: float = 0.05
CLAIM_DECOY_RATIO: float = 2.0             # claim 6: Δμ_decoy ≥ ratio·max(0,-Δμ_true)
CLAIM_PASS_MIN_CLIPS: int = 7              # 7/10 clips must satisfy each claim


# =============================================================================
# Config expansion
# =============================================================================


def build_main_table_configs(
    n_random_draws: int = N_RANDOM_DRAWS,
    canonical_W_clean: Sequence[int] = (6, 12, 14),
) -> Dict[str, Dict[str, Any]]:
    """Expand the main table into a flat {config_name → kwargs} dict.

    The "clean" row is handled separately (no PGD). TODO rows 7-9 are omitted.
    """
    configs: Dict[str, Dict[str, Any]] = {
        "K1_top":    {"K_ins": 1, "vulnerability_mode": "top",    "seed": 0},
        "K3_top":    {"K_ins": 3, "vulnerability_mode": "top",    "seed": 0},
        "K3_bottom": {"K_ins": 3, "vulnerability_mode": "bottom", "seed": 0},
        "canonical": {"K_ins": 3, "vulnerability_mode": "top",    "seed": 0,
                      "W_clean_override": list(canonical_W_clean)},
    }
    for s in range(n_random_draws):
        configs[f"K1_random_seed{s}"] = {
            "K_ins": 1, "vulnerability_mode": "random", "seed": s}
        configs[f"K3_random_seed{s}"] = {
            "K_ins": 3, "vulnerability_mode": "random", "seed": s}
    return configs


# =============================================================================
# Per-clip aggregation
# =============================================================================


@dataclass
class ClipAggregation:
    """All (config → record) results for one clip, plus derived aggregates."""

    clip: str
    records: Dict[str, ClipConfigRecord] = field(default_factory=dict)

    def mean_J_drop(self, configs: Sequence[str]) -> float:
        """Mean J-drop over matching feasible records. NaN if none feasible.

        Infeasible configs are EXCLUDED from the mean (not 0-imputed). The
        per-clip caller re-checks feasibility count separately.
        """
        vals = [r.best_surrogate_J_drop for name, r in self.records.items()
                if name in configs and not r.infeasible]
        return float(np.mean(vals)) if vals else float("nan")

    def any_feasible(self, configs: Sequence[str]) -> bool:
        return any((name in self.records and not self.records[name].infeasible)
                   for name in configs)


@dataclass
class ClaimResult:
    name: str
    hits: int                                  # clips satisfying
    total: int                                 # clips evaluated
    supported: bool                            # hits ≥ CLAIM_PASS_MIN_CLIPS
    per_clip_detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAVIS10Result:
    aggregations: Dict[str, ClipAggregation]
    claims: Dict[str, ClaimResult]             # "J_drop_035", "vs_random", etc.
    pending_claims: List[str]                  # not yet evaluated (configs not landed)


# =============================================================================
# 8-claim success bar evaluation
# =============================================================================


def _random_draws(
    aggs: ClipAggregation, K: int, n_draws: int = N_RANDOM_DRAWS,
) -> List[float]:
    """Collect paired-bootstrap random J-drops for K=K, feasible only.

    Honors the CLI's `--n-random-draws` by taking `n_draws` instead of the
    hardcoded module constant.
    """
    out: List[float] = []
    for s in range(n_draws):
        r = aggs.records.get(f"K{K}_random_seed{s}")
        if r is not None and not r.infeasible:
            out.append(r.best_surrogate_J_drop)
    return out


def evaluate_claims(
    aggregations: Dict[str, ClipAggregation],
    min_pass: int = CLAIM_PASS_MIN_CLIPS,
    n_random_draws: int = N_RANDOM_DRAWS,
) -> Tuple[Dict[str, ClaimResult], List[str]]:
    """Evaluate claims 1, 2, 3, 6 on DAVIS-10 aggregations.

    Per-clip interpretation (from handoff): "primary denominator = 10 clips,
    ≥ 7/10 must satisfy" each claim. For claim 1, "mean on exported artifact"
    refers to J-drop being a mean over the eval-window frames FOR THAT CLIP
    (so J-drop is a scalar per clip, compared to 0.35 per clip, counted
    across 10 clips). The aggregate mean J-drop across the 10 clips is a
    separate sanity diagnostic — not the claim-1 test — but is exposed in
    `claim1.per_clip_detail['aggregate_mean']` for paper tables.

    Returns (supported_claims, pending_claims). Claims 4, 5 (insert / ν
    necessity) and 7, 8 (R2/R3 restoration) are pending until configs 7-9
    land and File 8 runs.
    """
    claim_hits = {"J_drop_035": 0, "vs_random": 0, "vs_bottom": 0,
                  "decoy_dominates": 0}
    per_clip_detail: Dict[str, Dict[str, Any]] = {}

    clips = sorted(aggregations.keys())
    for clip in clips:
        agg = aggregations[clip]
        ours = agg.records.get("K3_top")            # centerpiece per Row 3
        detail: Dict[str, Any] = {}

        # Claim 1: J-drop(ours) ≥ 0.35
        if ours and not ours.infeasible:
            ours_J = ours.best_surrogate_J_drop
            if ours_J >= CLAIM_J_DROP_MIN:
                claim_hits["J_drop_035"] += 1
            detail["claim1_J_drop"] = ours_J
        else:
            detail["claim1_J_drop"] = None
            ours_J = None

        # Claim 2: ours ≥ max(M·random_K3, random_K3 + Δ)
        random_draws = _random_draws(agg, K=3, n_draws=n_random_draws)
        if ours and not ours.infeasible and random_draws:
            rand_mean = float(np.mean(random_draws))
            threshold = max(CLAIM_VS_RANDOM_MULT * rand_mean,
                            rand_mean + CLAIM_VS_RANDOM_DELTA)
            if ours_J >= threshold:
                claim_hits["vs_random"] += 1
            detail["claim2_random_mean"] = rand_mean
            detail["claim2_threshold"] = threshold
        else:
            detail["claim2_random_mean"] = None

        # Claim 3: ours ≥ max(M·bottom, bottom + Δ)
        bot = agg.records.get("K3_bottom")
        if ours and not ours.infeasible and bot and not bot.infeasible:
            bot_J = bot.best_surrogate_J_drop
            threshold = max(CLAIM_VS_BOTTOM_MULT * bot_J,
                            bot_J + CLAIM_VS_BOTTOM_DELTA)
            if ours_J >= threshold:
                claim_hits["vs_bottom"] += 1
            detail["claim3_bottom"] = bot_J
            detail["claim3_threshold"] = threshold
        else:
            detail["claim3_bottom"] = None

        # Claim 6: Δμ_decoy > 0 AND Δμ_decoy ≥ 2·max(0, -Δμ_true)
        if ours and not ours.infeasible:
            dmd = ours.delta_mu_decoy
            dmt = ours.delta_mu_true
            if dmd > 0 and dmd >= CLAIM_DECOY_RATIO * max(0.0, -dmt):
                claim_hits["decoy_dominates"] += 1
            detail["claim6_delta_mu_decoy"] = dmd
            detail["claim6_delta_mu_true"] = dmt

        per_clip_detail[clip] = detail

    total = len(clips)
    # Aggregate-mean diagnostic for claim 1 (paper table convenience —
    # not the claim-1 test itself, which is per-clip + count-based).
    J_drops = [per_clip_detail[c].get("claim1_J_drop")
               for c in per_clip_detail]
    finite_J = [v for v in J_drops if v is not None]
    aggregate_mean_J_drop = float(np.mean(finite_J)) if finite_J else float("nan")

    out: Dict[str, ClaimResult] = {}
    for name, hits in claim_hits.items():
        detail_per_clip = {c: {k: v for k, v in per_clip_detail[c].items()
                               if k.startswith(_claim_prefix(name))}
                           for c in per_clip_detail}
        if name == "J_drop_035":
            # Inject the aggregate diagnostic under a well-known key so the
            # paper table can pull it directly.
            for c in detail_per_clip:
                detail_per_clip[c]["aggregate_mean_J_drop_across_clips"] = \
                    aggregate_mean_J_drop
        out[name] = ClaimResult(
            name=name, hits=hits, total=total,
            supported=(hits >= min_pass),
            per_clip_detail=detail_per_clip,
        )
    pending = [
        "insert_necessity (claim 4, needs config 7 top-δ-only K=0)",
        "nu_necessity (claim 5, needs config 9 top-base-insert+δ)",
        "R2_restoration (claim 7, needs run_vadi_restoration.py)",
        "R3_restoration (claim 8, needs run_vadi_restoration.py)",
    ]
    return out, pending


def _claim_prefix(name: str) -> str:
    return {
        "J_drop_035": "claim1_",
        "vs_random": "claim2_",
        "vs_bottom": "claim3_",
        "decoy_dominates": "claim6_",
    }[name]


# =============================================================================
# Orchestrator
# =============================================================================


def run_davis10_main(
    clips: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    davis_root: Path,
    out_root: Path,
    clean_pass_fn_factory: Callable,
    forward_fn_builder_factory: Callable,
    lpips_fn: Callable[[Tensor, Tensor], Tensor],
    ssim_fn: Callable[[Tensor, Tensor], Tensor],
    clip_loader: Callable = load_davis_clip,
    config_builder: Callable[[], VADIConfig] = VADIConfig,
    device: Optional[torch.device] = None,
) -> DAVIS10Result:
    """Run every (clip, config) in the main table and evaluate the claims.

    Each clip's clean-SAM2 pass is run ONCE (inside the factory closure
    when the caller wires it) and shared across the clip's configs, per
    the performance tip in `run_vadi_pilot.build_pilot_adapters`.
    """
    aggregations: Dict[str, ClipAggregation] = {}
    for clip_name in clips:
        x_clean, prompt = clip_loader(davis_root, clip_name)
        if device is not None:
            x_clean = x_clean.to(device)
        clean_pass_fn = clean_pass_fn_factory(clip_name, x_clean, prompt)
        forward_fn_builder = forward_fn_builder_factory(
            clip_name, x_clean, prompt)

        clip_agg = ClipAggregation(clip=clip_name)
        for cfg_name, kwargs in configs.items():
            vadi_cfg = config_builder()
            rng = np.random.default_rng(int(kwargs.get("seed", 0)))
            override = kwargs.get("W_clean_override")
            out: VADIClipOutput = run_vadi_for_clip(
                clip_name=clip_name, config_name=cfg_name,
                x_clean=x_clean, prompt_mask=prompt,
                clean_pass_fn=clean_pass_fn,
                forward_fn_builder=forward_fn_builder,
                lpips_fn=lpips_fn, ssim_fn=ssim_fn,
                vulnerability_mode=kwargs["vulnerability_mode"],
                K_ins=int(kwargs["K_ins"]),
                min_gap=int(kwargs.get("min_gap", 2)),
                rng=rng, config=vadi_cfg, out_root=out_root,
                W_clean_override=override,
            )
            clip_agg.records[cfg_name] = ClipConfigRecord.from_vadi_output(
                clip_name, out)
        aggregations[clip_name] = clip_agg

    n_random_draws = sum(1 for name in configs if name.startswith("K3_random"))
    claims, pending = evaluate_claims(
        aggregations, n_random_draws=n_random_draws or N_RANDOM_DRAWS)
    result = DAVIS10Result(
        aggregations=aggregations, claims=claims, pending_claims=pending)

    # Persist for paper reproducibility.
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "clips": list(clips),
        "n_configs": len(configs),
        "claims": {name: asdict(cr) for name, cr in claims.items()},
        "pending_claims": pending,
        "thresholds": {
            "CLAIM_J_DROP_MIN": CLAIM_J_DROP_MIN,
            "CLAIM_VS_RANDOM_MULT": CLAIM_VS_RANDOM_MULT,
            "CLAIM_VS_RANDOM_DELTA": CLAIM_VS_RANDOM_DELTA,
            "CLAIM_VS_BOTTOM_MULT": CLAIM_VS_BOTTOM_MULT,
            "CLAIM_VS_BOTTOM_DELTA": CLAIM_VS_BOTTOM_DELTA,
            "CLAIM_DECOY_RATIO": CLAIM_DECOY_RATIO,
            "CLAIM_PASS_MIN_CLIPS": CLAIM_PASS_MIN_CLIPS,
            "N_DAVIS_CLIPS": len(clips),
        },
        "per_clip_records": {
            clip: {cfg: asdict(rec) for cfg, rec in agg.records.items()}
            for clip, agg in aggregations.items()
        },
    }
    with open(out_root / "davis10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return result


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VADI DAVIS-10 main-table runner. Run AFTER pilot = GO.")
    p.add_argument("--davis-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-root", default="vadi_runs/davis10")
    p.add_argument("--clips", nargs="+", default=list(DAVIS10_CLIPS))
    p.add_argument("--n-random-draws", type=int, default=N_RANDOM_DRAWS)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    configs = build_main_table_configs(n_random_draws=args.n_random_draws)

    if args.dry_run:
        print(f"[davis10] clips = {args.clips}")
        print(f"[davis10] n_configs = {len(configs)} "
              f"(incl. {2 * args.n_random_draws} random-draw runs)")
        for name in configs:
            print(f"  {name}")
        print(f"[davis10] out_root = {out_root}")
        print("[davis10] claims to evaluate (primary denominator = "
              f"{len(args.clips)}; >= {CLAIM_PASS_MIN_CLIPS} clips to SUPPORT):")
        for c in ["J_drop_035", "vs_random", "vs_bottom", "decoy_dominates"]:
            print(f"  {c}")
        print("[davis10] pending (configs 7-9 / restoration not landed):")
        for c in ["insert_necessity", "nu_necessity", "R2", "R3"]:
            print(f"  {c}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        clean_fac, fwd_fac, lp, ss = build_pilot_adapters(
            args.checkpoint, device)
    except NotImplementedError as e:
        print(f"[davis10] cannot run real pipeline: {e}", file=sys.stderr)
        return 2

    result = run_davis10_main(
        clips=args.clips, configs=configs,
        davis_root=Path(args.davis_root), out_root=out_root,
        clean_pass_fn_factory=clean_fac,
        forward_fn_builder_factory=fwd_fac,
        lpips_fn=lp, ssim_fn=ss, device=device,
    )
    print("[davis10] claim evaluation:")
    for name, cr in result.claims.items():
        tag = "SUPPORTED" if cr.supported else "NOT_SUPPORTED"
        print(f"  {name}: {cr.hits}/{cr.total} clips -> {tag}")
    print("[davis10] pending claims (need more code):")
    for p in result.pending_claims:
        print(f"  - {p}")
    return 0


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    # -- build_main_table_configs: 4 + 2×N_RANDOM = 4 + 10 = 14
    configs = build_main_table_configs()
    expected_N = 4 + 2 * N_RANDOM_DRAWS
    assert len(configs) == expected_N, f"expected {expected_N}, got {len(configs)}"
    assert "K1_top" in configs and "K3_top" in configs
    assert "K3_bottom" in configs and "canonical" in configs
    assert configs["canonical"]["W_clean_override"] == [6, 12, 14]
    for s in range(N_RANDOM_DRAWS):
        assert f"K1_random_seed{s}" in configs
        assert configs[f"K1_random_seed{s}"]["seed"] == s

    # -- evaluate_claims with synthetic data --------------------------------
    def rec(J, feas=True, dmd=0.5, dmt=-0.1):
        return ClipConfigRecord(
            clip="x", config="x", infeasible=not feas,
            best_surrogate_J_drop=J,
            delta_mu_decoy=dmd, delta_mu_true=dmt, W_attacked=[5],
        )

    def build_agg(clip, J_top_k3, J_bottom_k3, J_random_k3_list):
        agg = ClipAggregation(clip=clip)
        agg.records["K3_top"] = rec(J_top_k3)
        agg.records["K3_bottom"] = rec(J_bottom_k3)
        for s, jr in enumerate(J_random_k3_list):
            agg.records[f"K3_random_seed{s}"] = rec(jr)
        return agg

    # Scenario 1: strong attack on 10 clips — all claims supported.
    aggs_strong = {
        f"clip{i:02d}": build_agg(
            f"clip{i:02d}", J_top_k3=0.50, J_bottom_k3=0.10,
            J_random_k3_list=[0.15] * 5,
        ) for i in range(10)
    }
    claims, pending = evaluate_claims(aggs_strong)
    assert claims["J_drop_035"].supported      # 0.50 ≥ 0.35
    assert claims["J_drop_035"].hits == 10
    assert claims["vs_random"].supported       # 0.50 ≥ max(2×0.15, 0.15+0.05)=0.30
    assert claims["vs_random"].hits == 10
    assert claims["vs_bottom"].supported       # 0.50 ≥ max(3×0.10, 0.10+0.05)=0.30
    assert claims["vs_bottom"].hits == 10
    assert claims["decoy_dominates"].supported # dmd=0.5 ≥ 2·max(0, 0.1)=0.2
    assert len(pending) == 4                   # 4 pending claims noted

    # Scenario 2: borderline — 7/10 exactly meet claim 1 (J_drop >= 0.35).
    aggs_border = {}
    for i in range(10):
        J = 0.50 if i < 7 else 0.20
        aggs_border[f"clip{i:02d}"] = build_agg(
            f"clip{i:02d}", J_top_k3=J, J_bottom_k3=0.10,
            J_random_k3_list=[0.15] * 5,
        )
    claims_b, _ = evaluate_claims(aggs_border)
    assert claims_b["J_drop_035"].hits == 7
    assert claims_b["J_drop_035"].supported    # exactly 7 → SUPPORTED

    # Scenario 3: 6/10 meet → NOT supported.
    aggs_short = {}
    for i in range(10):
        J = 0.50 if i < 6 else 0.20
        aggs_short[f"clip{i:02d}"] = build_agg(
            f"clip{i:02d}", J_top_k3=J, J_bottom_k3=0.10,
            J_random_k3_list=[0.15] * 5,
        )
    claims_s, _ = evaluate_claims(aggs_short)
    assert claims_s["J_drop_035"].hits == 6
    assert not claims_s["J_drop_035"].supported

    # Scenario 4: infeasible K3_top → claim 1 fails for that clip.
    aggs_inf = {
        "clip00": ClipAggregation(clip="clip00", records={
            "K3_top": rec(0.50, feas=False),
            "K3_bottom": rec(0.10),
            **{f"K3_random_seed{s}": rec(0.15) for s in range(5)},
        }),
        **{f"clip{i:02d}": build_agg(
              f"clip{i:02d}", 0.50, 0.10, [0.15] * 5)
           for i in range(1, 10)},
    }
    claims_inf, _ = evaluate_claims(aggs_inf)
    assert claims_inf["J_drop_035"].hits == 9   # clip00 lost
    assert claims_inf["J_drop_035"].supported  # 9/10 ≥ 7

    # -- Decoy-vs-suppression claim 6: dmd=0.1 not enough vs dmt=-0.5
    # → 2·max(0, 0.5) = 1.0, 0.1 < 1.0 → fail.
    aggs_decoy_weak = {
        f"clip{i:02d}": build_agg(
            f"clip{i:02d}", 0.50, 0.10, [0.15] * 5)
        for i in range(10)
    }
    for agg in aggs_decoy_weak.values():
        agg.records["K3_top"] = rec(0.50, dmd=0.10, dmt=-0.50)
    claims_d, _ = evaluate_claims(aggs_decoy_weak)
    assert claims_d["decoy_dominates"].hits == 0
    assert not claims_d["decoy_dominates"].supported

    # -- evaluate_claims honors n_random_draws (regression for LOW bug):
    # Build a clip with 10 K3_random seeds (0..9); a test with n_random_draws=3
    # must ignore seeds 3..9. Seeds 0..2 have J=0.30 (high); seeds 3..9 have
    # J=0.05. n_random=3 → rand_mean=0.30; n_random=10 → rand_mean ≈ 0.125.
    agg_many = ClipAggregation(clip="c0", records={
        "K3_top": rec(0.50),
        "K3_bottom": rec(0.10),
        **{f"K3_random_seed{s}": rec(0.30 if s < 3 else 0.05)
           for s in range(10)},
    })
    _, _ = evaluate_claims({"c0": agg_many}, n_random_draws=3)
    # With rand_mean=0.30, threshold = max(2×0.30, 0.30+0.05)=0.60; 0.50 < 0.60 → fail.
    claims_r3, _ = evaluate_claims({"c0": agg_many}, n_random_draws=3)
    assert claims_r3["vs_random"].hits == 0
    # With rand_mean=0.125, threshold = max(2×0.125, 0.125+0.05)=0.25; 0.50 ≥ 0.25 → pass.
    claims_r10, _ = evaluate_claims({"c0": agg_many}, n_random_draws=10)
    assert claims_r10["vs_random"].hits == 1

    # -- Aggregate-mean diagnostic exposed under claim 1.
    aggs_diag = {f"clip{i:02d}": build_agg(f"clip{i:02d}", 0.40, 0.10, [0.15] * 5)
                 for i in range(10)}
    claims_diag, _ = evaluate_claims(aggs_diag)
    first_clip = next(iter(claims_diag["J_drop_035"].per_clip_detail))
    assert "aggregate_mean_J_drop_across_clips" in \
        claims_diag["J_drop_035"].per_clip_detail[first_clip]
    assert abs(
        claims_diag["J_drop_035"].per_clip_detail[first_clip][
            "aggregate_mean_J_drop_across_clips"] - 0.40
    ) < 1e-6

    # -- ClipAggregation.mean_J_drop: excludes infeasible
    agg = ClipAggregation(clip="x", records={
        "K3_random_seed0": rec(0.20),
        "K3_random_seed1": rec(0.30),
        "K3_random_seed2": rec(0.99, feas=False),       # excluded
    })
    mean = agg.mean_J_drop([f"K3_random_seed{s}" for s in range(5)])
    assert abs(mean - 0.25) < 1e-6, f"expected 0.25, got {mean}"
    # All-infeasible → NaN.
    agg_inf = ClipAggregation(clip="y", records={
        "K3_random_seed0": rec(0.20, feas=False),
    })
    assert np.isnan(agg_inf.mean_J_drop(["K3_random_seed0"]))

    # -- CLI --dry-run.
    rv = main(["--davis-root", "/tmp", "--checkpoint", "/tmp/x.pt", "--dry-run"])
    assert rv == 0

    # -- CLI real-mode fails gracefully (SAM2 not wired).
    rv = main(["--davis-root", "/tmp", "--checkpoint", "/tmp/x.pt"])
    assert rv == 2

    print("scripts.run_vadi_davis10: all self-tests PASSED "
          "(config expansion, evaluate_claims: all 4 supported / borderline "
          "7/10 / below-threshold / infeasibility / decoy-weak, "
          "mean_J_drop with infeasible, CLI)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _self_test()
