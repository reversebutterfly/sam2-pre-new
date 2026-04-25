"""K=3 subset placement profiling via beam search (codex Loop 3 R5 Bundle A,
locked 2026-04-25).

The current vulnerability scorer (memshield/vulnerability_scorer.py) ranks
frames by a 3-signal rank-sum heuristic (confidence delta + mask IoU +
Hiera distance). The decisive 10-clip ablation (auto-review-loop, Decisive
Round) showed random-K BEATS top-K by mean 0.534 vs 0.488 — i.e., the
heuristic is empirically anti-correlated with attack effectiveness.

Codex's no-proxy alternative: rank candidate K=3 SUBSETS by their actual
attack J-drop. A naive enumeration costs C(N, 3) ≈ 18000 subsets per clip,
each requiring ~10s for a low-budget attack surrogate → ~50 hours per clip,
infeasible.

This module implements **beam search**:
  Step 1: profile every single-frame K=1 attack    (N evals)
  Step 2: top-B K=1 → expand to K=2 (B × N evals)
  Step 3: top-B K=2 → expand to K=3 (B × N evals)

For B=8 and N=50: ~50 + 400 + 400 = ~850 attack evals per clip. At 10s
each, ~140 min per clip — feasible to run as overnight preprocessing on
Pro 6000.

The "score function" for each subset is supplied by the caller (the v5
driver runs a low-budget A0 attack at the candidate W, exports, evals
J-drop). This module is pure algorithm — it does not depend on SAM2 or
torch internals. Only thing this module knows: how to rank by a callable,
how to expand subsets respecting min_gap.

Pure Python + numpy. Run `python -m memshield.placement_profiler` for tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubsetScore:
    """A scored placement subset."""
    subset: Tuple[int, ...]              # the chosen frame indices, sorted
    score: float                          # higher = better (e.g., J-drop)
    metadata: Dict = field(default_factory=dict)  # optional aux info


@dataclass
class BeamSearchResult:
    """Output of beam_search_K3."""
    best: SubsetScore
    top_k1: List[SubsetScore]            # K=1 layer survivors
    top_k2: List[SubsetScore]            # K=2 layer survivors
    top_k3: List[SubsetScore]            # K=3 layer survivors (full beam)
    total_evals: int
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subset expansion utilities
# ---------------------------------------------------------------------------


def feasible_candidates(
    candidates: Sequence[int], existing: Sequence[int], min_gap: int,
) -> List[int]:
    """Filter candidates by min_gap to existing positions.

    A candidate c is feasible iff for every e in existing, |c - e| >= min_gap.
    """
    existing_set = set(int(e) for e in existing)
    return [
        int(c) for c in candidates
        if int(c) not in existing_set
        and all(abs(int(c) - int(e)) >= min_gap for e in existing)
    ]


def expand_beam(
    parents: Sequence[SubsetScore],
    candidates: Sequence[int],
    min_gap: int,
) -> List[Tuple[int, ...]]:
    """For each parent subset, expand by one candidate satisfying min_gap.

    Returns the deduplicated list of expanded subsets (each as sorted tuple).
    """
    out: set = set()
    for p in parents:
        exist = list(p.subset)
        for c in feasible_candidates(candidates, exist, min_gap):
            new_subset = tuple(sorted(exist + [int(c)]))
            out.add(new_subset)
    return sorted(out)


# ---------------------------------------------------------------------------
# Beam search (the main entry point)
# ---------------------------------------------------------------------------


def beam_search_K3(
    candidates: Sequence[int],
    score_fn: Callable[[Tuple[int, ...]], float],
    *,
    beam_width: int = 8,
    min_gap: int = 2,
    log: Optional[Callable[[str], None]] = None,
) -> BeamSearchResult:
    """Beam search for the best K=3 placement subset.

    Args:
      candidates: list of candidate frame indices (the full pool, e.g.,
        all valid frames in the clip excluding f0).
      score_fn: callable(subset_tuple) → float scalar score (higher = better).
        For our use, this runs a low-budget attack at the subset and
        returns exported J-drop.
      beam_width: how many top-scoring partial subsets to retain at each
        layer. Default 8 (codex R5 spec).
      min_gap: minimum frame-distance between any two inserts in a subset.
        Default 2 (consistent with v5 placement).
      log: optional logging callback (for progress tracking).

    Returns: BeamSearchResult with best K=3 subset + intermediate layers.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if not candidates:
        raise ValueError("candidates list is empty")

    notes: List[str] = []
    total_evals = 0

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        notes.append(msg)

    # --- Layer 1: K=1 ---
    _log(f"[beam] K=1 layer: profiling {len(candidates)} single-frame inserts")
    k1_scores: List[SubsetScore] = []
    for c in candidates:
        subset = (int(c),)
        s = float(score_fn(subset))
        total_evals += 1
        k1_scores.append(SubsetScore(subset=subset, score=s))
    k1_scores.sort(key=lambda x: -x.score)
    top_k1 = k1_scores[:beam_width]
    if not top_k1:
        raise ValueError("beam_search_K3: K=1 layer produced no scores")
    _log(f"[beam] K=1 best: {top_k1[0].subset} score={top_k1[0].score:.4f}")

    # --- Layer 2: K=2 expansion ---
    expanded_k2 = expand_beam(top_k1, candidates, min_gap=min_gap)
    if not expanded_k2:
        raise ValueError(
            f"beam_search_K3: no feasible K=2 subset under min_gap={min_gap} "
            f"from K=1 beam (candidate pool size={len(candidates)})")
    _log(f"[beam] K=2 layer: profiling {len(expanded_k2)} pairs")
    k2_scores: List[SubsetScore] = []
    for subset in expanded_k2:
        s = float(score_fn(subset))
        total_evals += 1
        k2_scores.append(SubsetScore(subset=subset, score=s))
    k2_scores.sort(key=lambda x: -x.score)
    top_k2 = k2_scores[:beam_width]
    _log(f"[beam] K=2 best: {top_k2[0].subset} score={top_k2[0].score:.4f}")

    # --- Layer 3: K=3 expansion ---
    expanded_k3 = expand_beam(top_k2, candidates, min_gap=min_gap)
    if not expanded_k3:
        raise ValueError(
            f"beam_search_K3: no feasible K=3 subset under min_gap={min_gap} "
            f"from K=2 beam (candidate pool size={len(candidates)})")
    _log(f"[beam] K=3 layer: profiling {len(expanded_k3)} triples")
    k3_scores: List[SubsetScore] = []
    for subset in expanded_k3:
        s = float(score_fn(subset))
        total_evals += 1
        k3_scores.append(SubsetScore(subset=subset, score=s))
    k3_scores.sort(key=lambda x: -x.score)
    top_k3 = k3_scores[:beam_width]
    _log(f"[beam] K=3 best: {top_k3[0].subset} score={top_k3[0].score:.4f}")

    return BeamSearchResult(
        best=top_k3[0],
        top_k1=top_k1, top_k2=top_k2, top_k3=top_k3,
        total_evals=total_evals, notes=notes,
    )


# ---------------------------------------------------------------------------
# Cached score function wrapper (avoids re-profiling identical subsets across
# layer-2 / layer-3 expansion when overlap occurs)
# ---------------------------------------------------------------------------


def make_cached_scorer(
    raw_score_fn: Callable[[Tuple[int, ...]], float],
) -> Callable[[Tuple[int, ...]], float]:
    """Wrap a score function with subset-tuple caching.

    Useful for beam search since two parents in layer L may expand to the
    same child subset in layer L+1 (e.g., {a, b} expanding by c and {a, c}
    expanding by b both yield {a, b, c}).
    """
    cache: Dict[Tuple[int, ...], float] = {}

    def _scored(subset: Tuple[int, ...]) -> float:
        key = tuple(sorted(int(x) for x in subset))
        if key in cache:
            return cache[key]
        s = float(raw_score_fn(key))
        cache[key] = s
        return s

    return _scored


# ---------------------------------------------------------------------------
# Result serialization (for offline preprocessing → cached files)
# ---------------------------------------------------------------------------


def _ss_to_dict(s: SubsetScore) -> Dict:
    """Codex R5 review fix: preserve metadata across all serialized layers
    (was previously dropped on top_k1/k2/k3, only kept on best). Survivor
    metadata (e.g., low-budget attack diagnostics per subset) is useful
    for downstream ablation / debugging."""
    return {
        "subset": list(s.subset),
        "score": s.score,
        "metadata": dict(s.metadata),
    }


def _dict_to_ss(item: Dict) -> SubsetScore:
    return SubsetScore(
        subset=tuple(item["subset"]), score=float(item["score"]),
        metadata=item.get("metadata", {}),
    )


def serialize_result(result: BeamSearchResult) -> Dict:
    """Convert a BeamSearchResult to JSON-friendly dict."""
    return {
        "best": _ss_to_dict(result.best),
        "top_k1": [_ss_to_dict(s) for s in result.top_k1],
        "top_k2": [_ss_to_dict(s) for s in result.top_k2],
        "top_k3": [_ss_to_dict(s) for s in result.top_k3],
        "total_evals": result.total_evals,
        "notes": result.notes,
    }


def deserialize_result(d: Dict) -> BeamSearchResult:
    """Inverse of serialize_result. Useful for loading cached preprocessing."""
    return BeamSearchResult(
        best=_dict_to_ss(d["best"]),
        top_k1=[_dict_to_ss(item) for item in d["top_k1"]],
        top_k2=[_dict_to_ss(item) for item in d["top_k2"]],
        top_k3=[_dict_to_ss(item) for item in d["top_k3"]],
        total_evals=int(d.get("total_evals", 0)),
        notes=list(d.get("notes", [])),
    )


# ---------------------------------------------------------------------------
# Random subset baseline (sanity check / control)
# ---------------------------------------------------------------------------


def random_K3_subsets(
    candidates: Sequence[int], n: int, *,
    min_gap: int = 2, seed: Optional[int] = None,
    strict: bool = True,
) -> List[Tuple[int, ...]]:
    """Sample n random feasible K=3 subsets.

    Uses rejection sampling: pick 3 distinct candidates, check min_gap.

    Codex R5 review fix: explicit handling when fewer than n feasible
    subsets exist. With `strict=True` (default), raises ValueError if
    we can't find n subsets within max_tries. With `strict=False`,
    returns however many were found (possibly fewer than n).
    """
    rng = np.random.default_rng(seed)
    cands = sorted(int(c) for c in candidates)
    if len(cands) < 3:
        if strict:
            raise ValueError(
                f"random_K3_subsets: fewer than 3 candidates "
                f"({len(cands)}); cannot form any K=3 subset")
        return []
    out: set = set()
    max_tries = max(20 * n, 1000)
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        idx = rng.choice(len(cands), size=3, replace=False)
        triple = tuple(sorted(int(cands[i]) for i in idx))
        a, b, c = triple
        if b - a >= min_gap and c - b >= min_gap:
            out.add(triple)
    if strict and len(out) < n:
        raise ValueError(
            f"random_K3_subsets: only found {len(out)} feasible subsets "
            f"under min_gap={min_gap} after {tries} tries (requested n={n}). "
            f"Either reduce n, lower min_gap, or set strict=False.")
    return sorted(out)[:n]


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_feasible_candidates() -> None:
    cands = list(range(20))
    out = feasible_candidates(cands, existing=[5, 10], min_gap=2)
    # Should exclude {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} that are within gap of either
    # |c - 5| >= 2: c ∉ {4, 5, 6}; |c - 10| >= 2: c ∉ {9, 10, 11}
    forbidden = {4, 5, 6, 9, 10, 11}
    expected = [c for c in cands if c not in forbidden]
    assert out == expected, (out, expected)
    print("  feasible_candidates OK")


def _test_expand_beam() -> None:
    parents = [
        SubsetScore(subset=(5,), score=0.5),
        SubsetScore(subset=(10,), score=0.4),
    ]
    cands = list(range(15))
    expanded = expand_beam(parents, cands, min_gap=2)
    # parent (5,) expands to (c, 5) for c ∈ cands \ {3, 4, 5, 6, 7}
    # parent (10,) expands to (c, 10) for c ∈ cands \ {8, 9, 10, 11, 12}
    # All sorted as tuples.
    assert (0, 5) in expanded
    # (5, 10): parent (5,) expanding by 10, gap 5 ≥ 2 → IS in expanded
    assert (5, 10) in expanded, expanded
    # (3, 5): parent (5,) expanding by 3, gap 2 ≥ 2 → in
    assert (3, 5) in expanded
    # (4, 5): parent (5,) expanding by 4, gap 1 < 2 → NOT in
    assert (4, 5) not in expanded
    # (5, 7): parent (5,) expanding by 7, gap 2 ≥ 2 → in
    assert (5, 7) in expanded
    print("  expand_beam OK")


def _test_beam_search_K3() -> None:
    # Mock scorer: prefer subsets with sum near 30, with non-additive interaction
    # (penalize subsets that include both 5 and 10 simultaneously).
    def score_fn(subset):
        s = sum(subset)
        target = 30
        base = -abs(s - target)
        # Non-additive: penalize specific bad pairings
        if 5 in subset and 10 in subset:
            base -= 5
        return float(base)

    candidates = list(range(20))
    res = beam_search_K3(
        candidates, score_fn, beam_width=4, min_gap=2,
    )
    # Best should sum near 30 and avoid {5, 10} pair
    a, b, c = res.best.subset
    assert a < b < c
    assert b - a >= 2 and c - b >= 2
    s = a + b + c
    assert abs(s - 30) <= 4, res.best
    # Should avoid the {5, 10} pair if possible
    assert not (5 in res.best.subset and 10 in res.best.subset)
    # Total evals = N + B*expanded_pairs + B*expanded_triples; bounded by exhaustive
    assert 0 < res.total_evals < 10000
    print(f"  beam_search_K3 OK — best={res.best.subset} score={res.best.score:.2f} evals={res.total_evals}")


def _test_make_cached_scorer() -> None:
    call_count = [0]
    def raw(subset):
        call_count[0] += 1
        return float(sum(subset))
    cached = make_cached_scorer(raw)
    s1 = cached((1, 2, 3))
    s2 = cached((1, 2, 3))   # same key
    s3 = cached((3, 2, 1))   # same after sort
    assert s1 == s2 == s3 == 6.0
    assert call_count[0] == 1, call_count
    s4 = cached((1, 2, 4))
    assert call_count[0] == 2
    assert s4 == 7.0
    print("  make_cached_scorer OK")


def _test_serialize_deserialize() -> None:
    res = BeamSearchResult(
        best=SubsetScore(subset=(5, 10, 15), score=0.5, metadata={"a": 1}),
        top_k1=[SubsetScore(subset=(5,), score=0.4)],
        top_k2=[SubsetScore(subset=(5, 10), score=0.45)],
        top_k3=[SubsetScore(subset=(5, 10, 15), score=0.5)],
        total_evals=100, notes=["test"],
    )
    d = serialize_result(res)
    s = json.dumps(d)
    d2 = json.loads(s)
    res2 = deserialize_result(d2)
    assert res2.best.subset == res.best.subset
    assert res2.best.score == res.best.score
    assert res2.total_evals == 100
    print("  serialize_deserialize OK")


def _test_random_K3_subsets() -> None:
    cands = list(range(50))
    triples = random_K3_subsets(cands, n=20, min_gap=2, seed=42)
    assert len(triples) == 20
    for triple in triples:
        a, b, c = triple
        assert a < b < c
        assert b - a >= 2 and c - b >= 2
    # Reproducibility with same seed
    triples2 = random_K3_subsets(cands, n=20, min_gap=2, seed=42)
    assert triples == triples2
    print("  random_K3_subsets OK")


if __name__ == "__main__":
    print("memshield.placement_profiler self-tests:")
    _test_feasible_candidates()
    _test_expand_beam()
    _test_beam_search_K3()
    _test_make_cached_scorer()
    _test_serialize_deserialize()
    _test_random_K3_subsets()
    print("memshield.placement_profiler: all self-tests PASSED")
