"""Option C aggregator (codex round 16, 2026-04-29).

Reads per-(seed, method, clip) results.json from
``vadi_runs/v5_optionc/seed${r}/${method}/${clip}/K*__ot/results.json``
and produces:

  • backend per-clip table  : mean ± std of J_drop across replicates
  • frontend aggregate table: mean ± std over clips of per-clip means
  • paired Wilcoxon signed-rank test on per-clip means (target vs each
    baseline) — reports W, p, n
  • paired bootstrap 95% CI on the per-clip mean difference
  • ghost-fallback pass rate per (method) by scanning the run.log for
    ``[ghost-fallback]`` warnings.

Outputs:
  • <out_root>/aggregate.json           — full machine-readable record
  • <out_root>/aggregate_main_table.txt — frontend aggregate table
  • <out_root>/aggregate_appendix.txt   — per-clip table for appendix
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# scipy is mandatory for the AAAI main table. Fail fast — codex round 17.
try:
    from scipy import stats as _scipy_stats  # type: ignore
except ImportError as _e:  # noqa: BLE001
    sys.stderr.write(
        f"FATAL: scipy is required for paired Wilcoxon ({_e}). "
        "Install scipy in the active env before running the aggregator.\n"
    )
    raise SystemExit(2)


# Result discovery: prefer the canonical __ot eval directory; fall back to
# bare K* only as a last resort. The launcher pre-cleans cell_out so we
# never expect more than one matching dir per (seed, method, clip) — but
# we still hard-fail if multiple sibling configs co-exist (codex round
# 17 CRITICAL #2).
CLIP_RESULTS_GLOBS = ("K*__ot/results.json", "K*/results.json")


def _read_jdrop(clip_dir: pathlib.Path) -> Optional[float]:
    """Return the per-clip J-drop, hard-failing on ambiguous dir state.

    Prefers ``K*__ot/results.json`` (canonical eval) over bare ``K*``.
    If multiple ``__ot`` directories co-exist (e.g. left-over from a
    previous method's run that wasn't pre-cleaned), raises — silent
    selection of "first sorted" was a CRITICAL bug in round 17.
    """
    if not clip_dir.is_dir():
        return None
    for pat in CLIP_RESULTS_GLOBS:
        cands = sorted(clip_dir.glob(pat))
        if not cands:
            continue
        # Within a single glob pattern, multiple matches mean stale
        # outputs from another method/run. Refuse to silently pick.
        if len(cands) > 1:
            raise RuntimeError(
                f"ambiguous results in {clip_dir}: {len(cands)} matches "
                f"for {pat!r}: {[str(p) for p in cands]}"
            )
        raw = cands[0].read_text(encoding="utf-8").replace("NaN", "null")
        d = json.loads(raw)
        v = d.get("exported_j_drop")
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(v):
            return None
        return v
    return None


def _scan_ghost_log(out_root: pathlib.Path) -> Dict[Tuple[int, str, str], int]:
    """Return ghost-fallback occurrence count keyed by (seed, method, clip).

    The v5 driver does NOT print a "=== clip <name>" start marker — it only
    prints ``[v5] using profiled placement for <clip>: ...`` (skipped if
    profile fallback) and ``[v5] <clip>: exported_j_drop=...`` at the end
    of each clip. ``[ghost-fallback]`` warnings are emitted during the
    decoy-seed build phase, which sits BETWEEN those two markers.

    Strategy: forward-fill. Buffer ``[ghost-fallback]`` count between
    consecutive clip-finish lines and flush onto the clip name carried by
    the finish line. Reset buffer on every new CELL header. (Codex round
    17 CRITICAL #1 fix.)
    """
    log_path = out_root / "run.log"
    counts: Dict[Tuple[int, str, str], int] = {}
    if not log_path.is_file():
        return counts
    cell_re = re.compile(r"=== CELL seed=(\d+) method=([a-zA-Z0-9_+\-]+) ===")
    finish_re = re.compile(r"\[v5\]\s+(\S+):\s*exported_j_drop=")
    cur_seed: Optional[int] = None
    cur_method: Optional[str] = None
    pending: int = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = cell_re.search(line)
        if m:
            cur_seed = int(m.group(1))
            cur_method = m.group(2)
            pending = 0
            continue
        if "[ghost-fallback]" in line and cur_seed is not None and cur_method:
            pending += 1
            continue
        m = finish_re.search(line)
        if m and cur_seed is not None and cur_method:
            clip = m.group(1)
            if pending > 0:
                key = (cur_seed, cur_method, clip)
                counts[key] = counts.get(key, 0) + pending
            pending = 0
    return counts


def _per_clip_means(
    raw: Dict[Tuple[int, str, str], Optional[float]],
    methods: Sequence[str],
    clips: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """Aggregate raw J-drops to per-clip mean ± std (sample) across replicates.

    Returns: {method: {clip: (mean, std, n_finite)}}
    Cells with no result return ``mean=NaN`` and ``n_finite=0``.
    """
    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for m in methods:
        out[m] = {}
        for c in clips:
            vals: List[float] = []
            for s in seeds:
                v = raw.get((s, m, c))
                if v is not None and not math.isnan(v):
                    vals.append(v)
            if not vals:
                out[m][c] = (float("nan"), float("nan"), 0)
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            out[m][c] = (mean, std, len(vals))
    return out


def _aggregate_over_clips(
    per_clip: Dict[str, Dict[str, Tuple[float, float, int]]],
    methods: Sequence[str],
    clips: Sequence[str],
) -> Dict[str, Tuple[float, float, int]]:
    """For each method, mean ± std (across-clip std) of per-clip means."""
    out: Dict[str, Tuple[float, float, int]] = {}
    for m in methods:
        means = [per_clip[m][c][0] for c in clips
                 if not math.isnan(per_clip[m][c][0])]
        if not means:
            out[m] = (float("nan"), float("nan"), 0)
            continue
        avg = statistics.fmean(means)
        std = statistics.stdev(means) if len(means) >= 2 else 0.0
        out[m] = (avg, std, len(means))
    return out


def _paired_deltas(
    per_clip: Dict[str, Dict[str, Tuple[float, float, int]]],
    target: str,
    baseline: str,
    clips: Sequence[str],
) -> Tuple[List[str], List[float]]:
    used_clips: List[str] = []
    deltas: List[float] = []
    for c in clips:
        a = per_clip[target][c][0]
        b = per_clip[baseline][c][0]
        if math.isnan(a) or math.isnan(b):
            continue
        used_clips.append(c)
        deltas.append(a - b)
    return used_clips, deltas


def _wilcoxon(deltas: Sequence[float]) -> Optional[Dict[str, object]]:
    """Paired one-sided Wilcoxon (target > baseline).

    Uses ``method='auto'`` so SciPy itself picks exact vs. asymptotic
    based on sample size and ties — codex round 17 caveat #6: exact-mode
    is only truly exact when there are no ties or zeros, and
    ``zero_method='wilcox'`` drops zeros before ranking.
    """
    if not deltas:
        return None
    nonzero = [d for d in deltas if d != 0.0]
    if len(nonzero) < 1:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0,
                "method_used": "trivial-all-zero",
                "alternative": "greater"}
    try:
        res = _scipy_stats.wilcoxon(
            deltas, zero_method="wilcox",
            alternative="greater", method="auto",
        )
    except Exception as e:  # noqa: BLE001
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_nonzero": len(nonzero), "error": str(e),
                "alternative": "greater"}
    # SciPy's WilcoxonResult exposes the chosen method as either
    # ``res.method`` (newer) or ``res._method`` (older) — fall back to
    # "auto" if neither exists, since the exact branch requires no ties.
    method_used = getattr(res, "method", None) or getattr(res, "_method", "auto")
    return {"statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "n_nonzero": len(nonzero),
            "alternative": "greater",
            "zero_method": "wilcox",
            "method_used": str(method_used)}


def _paired_bootstrap_ci(
    deltas: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 1234,
) -> Optional[Tuple[float, float, float]]:
    """Resample clips with replacement, recompute mean(delta).

    Returns: (mean, lo, hi) for the requested CI.
    """
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = means[int(math.floor(alpha * n_boot))]
    hi_idx = int(math.ceil((1.0 - alpha) * n_boot)) - 1
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    hi = means[hi_idx]
    return (sum(deltas) / n, lo, hi)


def _win_rate(
    per_clip: Dict[str, Dict[str, Tuple[float, float, int]]],
    target: str,
    baseline: str,
    clips: Sequence[str],
) -> Tuple[int, int]:
    wins = 0
    total = 0
    for c in clips:
        a = per_clip[target][c][0]
        b = per_clip[baseline][c][0]
        if math.isnan(a) or math.isnan(b):
            continue
        total += 1
        if a > b:
            wins += 1
    return wins, total


def _ghost_pass_rate(
    ghost_counts: Dict[Tuple[int, str, str], int],
    methods: Sequence[str],
    clips: Sequence[str],
    seeds: Sequence[int],
    raw_results: Dict[Tuple[int, str, str], Optional[float]],
) -> Dict[str, Tuple[int, int]]:
    """Return (ghost_free_cells, total_cells_with_result) per method."""
    out: Dict[str, Tuple[int, int]] = {}
    for m in methods:
        ghost_free = 0
        total = 0
        for s in seeds:
            for c in clips:
                if raw_results.get((s, m, c)) is None:
                    continue
                total += 1
                if ghost_counts.get((s, m, c), 0) == 0:
                    ghost_free += 1
        out[m] = (ghost_free, total)
    return out


def _format_main_table(
    aggregate: Dict[str, Tuple[float, float, int]],
    win_rates: Dict[str, Tuple[int, int]],
    ghost_rates: Dict[str, Tuple[int, int]],
    methods: Sequence[str],
    target: str,
) -> str:
    """Frontend table. Per codex round 17 #5, the wins column shows the
    number of clips where the *target* beat the baseline on its row, so
    label it ``"<target> > row"``."""
    win_header = f"{target} > row (W/n)"
    lines = [
        "Frontend aggregate (DAVIS-13 / SAM2.1-Tiny):",
        "",
        f"{'Method':<12} {'mean DeltaJ':>12} {'+/- std':>10} "
        f"{'n_clips':>8} {win_header:>20} {'ghost-free':>12}",
        "-" * 80,
    ]
    for m in methods:
        mu, sd, n = aggregate.get(m, (float("nan"), float("nan"), 0))
        if m == target:
            wins_str = "—"
        else:
            w, t = win_rates.get(m, (0, 0))
            wins_str = f"{w}/{t}"
        gf, gt = ghost_rates.get(m, (0, 0))
        gf_str = f"{gf}/{gt}" if gt else "—"
        lines.append(
            f"{m:<12} {mu:>12.4f} {sd:>10.4f} {n:>8d} "
            f"{wins_str:>20} {gf_str:>12}"
        )
    return "\n".join(lines)


def _format_appendix_table(
    per_clip: Dict[str, Dict[str, Tuple[float, float, int]]],
    methods: Sequence[str],
    clips: Sequence[str],
) -> str:
    header = ["clip"] + [f"{m}_mean" for m in methods] + [f"{m}_std" for m in methods] + [f"{m}_n" for m in methods]
    lines = ["Per-clip backend (mean ± std across replicates):", "",
             "\t".join(header)]
    for c in clips:
        row: List[str] = [c]
        for m in methods:
            row.append(f"{per_clip[m][c][0]:.4f}")
        for m in methods:
            row.append(f"{per_clip[m][c][1]:.4f}")
        for m in methods:
            row.append(str(per_clip[m][c][2]))
        lines.append("\t".join(row))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True, type=pathlib.Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--target", required=True,
                   help="method label whose superiority we test")
    p.add_argument("--baseline", action="append", required=True,
                   help="baseline method label (can be repeated)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--strict", action="store_true", default=True,
                   help="hard-fail if any (seed, method, clip) cell is "
                        "missing a result (default: true)")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="permit missing cells (manual partial-run mode)")
    args = p.parse_args()

    if args.target not in args.methods:
        p.error(f"--target {args.target!r} not in --methods {args.methods}")
    for b in args.baseline:
        if b not in args.methods:
            p.error(f"--baseline {b!r} not in --methods {args.methods}")
        if b == args.target:
            p.error(f"--baseline {b!r} must differ from --target")

    out_root: pathlib.Path = args.out_root
    raw: Dict[Tuple[int, str, str], Optional[float]] = {}
    missing_cells: List[Tuple[int, str, str]] = []
    for s in args.seeds:
        for m in args.methods:
            for c in args.clips:
                cell_dir = out_root / f"seed{s}" / m / c
                v = _read_jdrop(cell_dir) if cell_dir.is_dir() else None
                raw[(s, m, c)] = v
                if v is None:
                    missing_cells.append((s, m, c))
    if missing_cells:
        msg = (
            f"missing results for {len(missing_cells)}/"
            f"{len(args.seeds)*len(args.methods)*len(args.clips)} cells: "
            + ", ".join(f"seed{s}/{m}/{c}" for s, m, c in missing_cells[:8])
            + (" ..." if len(missing_cells) > 8 else "")
        )
        if args.strict:
            sys.stderr.write(f"FATAL: {msg}\n")
            raise SystemExit(3)
        sys.stderr.write(f"WARN: {msg}\n")

    per_clip = _per_clip_means(raw, args.methods, args.clips, args.seeds)
    aggregate = _aggregate_over_clips(per_clip, args.methods, args.clips)

    win_rates: Dict[str, Tuple[int, int]] = {}
    for m in args.methods:
        if m == args.target:
            continue
        win_rates[m] = _win_rate(per_clip, args.target, m, args.clips)

    paired_stats: Dict[str, Dict[str, object]] = {}
    for b in args.baseline:
        used_clips, deltas = _paired_deltas(
            per_clip, args.target, b, args.clips)
        wilc = _wilcoxon(deltas)
        boot = _paired_bootstrap_ci(deltas, n_boot=args.n_boot, ci=args.ci)
        paired_stats[b] = {
            "n_clips_paired": len(used_clips),
            "clips_used": used_clips,
            "deltas": deltas,
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else float("nan"),
            "wilcoxon": wilc,
            "bootstrap_ci": (
                {"mean": boot[0], "lo": boot[1], "hi": boot[2], "level": args.ci}
                if boot else None
            ),
        }

    ghost_counts = _scan_ghost_log(out_root)
    ghost_rates = _ghost_pass_rate(
        ghost_counts, args.methods, args.clips, args.seeds, raw)

    # Reproducibility metadata snapshot, written by the launcher before the
    # first cell. Optional — keeps the aggregator usable as a stand-alone CLI.
    metadata_path = out_root / "metadata.json"
    metadata: Dict[str, object] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"WARN: failed to parse metadata.json ({e}); skipping.\n"
            )

    record = {
        "out_root": str(out_root),
        "seeds": list(args.seeds),
        "methods": list(args.methods),
        "clips": list(args.clips),
        "target": args.target,
        "baselines": list(args.baseline),
        "metadata_snapshot": metadata,
        "raw_jdrop": {
            f"seed{s}/{m}/{c}": raw[(s, m, c)]
            for s in args.seeds for m in args.methods for c in args.clips
        },
        "per_clip_mean_std_n": {
            m: {c: list(per_clip[m][c]) for c in args.clips}
            for m in args.methods
        },
        "aggregate_over_clips": {
            m: list(aggregate[m]) for m in args.methods
        },
        "win_rates_vs_target": {
            m: {"wins": w, "n_paired": n} for m, (w, n) in win_rates.items()
        },
        "paired_stats": paired_stats,
        "ghost_fallback_counts": {
            f"seed{s}/{m}/{c}": v for (s, m, c), v in ghost_counts.items()
        },
        "ghost_free_pass_rate": {
            m: {"ghost_free": gf, "total": gt}
            for m, (gf, gt) in ghost_rates.items()
        },
    }

    (out_root / "aggregate.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")

    main_tbl = _format_main_table(
        aggregate, win_rates, ghost_rates, args.methods, args.target)
    (out_root / "aggregate_main_table.txt").write_text(main_tbl, encoding="utf-8")

    appendix_tbl = _format_appendix_table(per_clip, args.methods, args.clips)
    (out_root / "aggregate_appendix.txt").write_text(appendix_tbl, encoding="utf-8")

    print(main_tbl)
    print()
    print("=== Paired stats ===")
    for b, s in paired_stats.items():
        print(f"\n{args.target} vs {b}: n_paired={s['n_clips_paired']} "
              f"mean_delta={s['mean_delta']:.4f}")
        wilc = s.get("wilcoxon")
        if wilc:
            stat = wilc.get("statistic")
            p = wilc.get("p_value")
            stat_str = f"{stat:.4g}" if isinstance(stat, (int, float)) else str(stat)
            p_str = f"{p:.4g}" if isinstance(p, (int, float)) else str(p)
            print(f"  Wilcoxon (method_used={wilc.get('method_used','?')}, "
                  f"alt={wilc.get('alternative','?')}): "
                  f"W={stat_str}, p={p_str}, "
                  f"n_nonzero={wilc.get('n_nonzero')}")
        boot = s.get("bootstrap_ci")
        if boot:
            print(f"  Paired bootstrap {int(args.ci*100)}% CI on mean delta: "
                  f"{boot['mean']:.4f}  [{boot['lo']:.4f}, {boot['hi']:.4f}]")

    print()
    print(f"Wrote {out_root/'aggregate.json'}, "
          f"{out_root/'aggregate_main_table.txt'}, "
          f"{out_root/'aggregate_appendix.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
