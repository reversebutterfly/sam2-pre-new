"""Score-function wrapper for placement profiling beam search.

Round 5 Bundle A sub-session 2 (2026-04-25). Wraps the v5 driver's
`run_v5_for_clip` into a `score_fn(subset_tuple) → J_drop` callable that
beam_search_K3 can invoke. Implements Codex Q1 verdict (c) — the
"two-pass" budget policy:

  * For |subset| ∈ {1, 2}: 12-step ν-only PGD (cheap surrogate, used
    only for layer pruning in beam search).
  * For |subset| == 3: full A0 PGD (100 ν-only steps, no δ, no α/warp);
    fidelity-true scoring of the final K=3 decision.

The score function:
  1. Clones the base VADIv5Config with the appropriate n_steps,
     `delta_support_mode="off"` (insert-only A0), and exports under a
     subset-tagged sub-directory so concurrent / sequential calls do
     not clobber each other.
  2. Calls `run_v5_for_clip` with `W_clean_override=sorted(subset)`
     and `K_ins=len(subset)`, which exercises the same orchestration
     path as the production driver (clean pass → decoy-seed
     construction → PGD → export → eval).
  3. Returns the exported J-drop (mean over post-insert frames) as
     reported by `eval_exported_j_drop`. If the run was infeasible
     (no PGD step satisfied feasibility), returns
     `infeasible_score` (default -1.0) so beam search ranks it
     below any feasible result.

The reason we do NOT cache `clean_pass_fn` outputs across subsets here:
clean SAM2 inference is invariant of the subset, so callers are
expected to wrap it with `functools.lru_cache` or precompute
`pseudo_masks` once per clip in their context. (Memoization on
caller's side keeps THIS module pure.)

Caching identical subsets across beam-layer expansions is handled by
`memshield.placement_profiler.make_cached_scorer` — wrapping the
score function returned here is the recommended pattern.

Tests: `python -m memshield.v5_score_fn` runs unit tests with mocked
`run_v5_for_clip` (no SAM2 dependency).
"""
from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Context + policy dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoreFnContext:
    """Pre-loaded clip context shared across all subset evaluations.

    Holds tensors and callables that are subset-invariant: the clip
    pixels, the SAM2 forward fn builder, the loss / eval helpers, and
    the base config template.
    """
    clip_name: str
    x_clean: Any                                # Tensor [T, H, W, 3]
    prompt_mask: np.ndarray
    clean_pass_fn: Callable
    forward_fn_builder: Callable
    lpips_fn: Callable
    sam2_eval_fn: Callable
    base_config: Any                            # VADIv5Config
    out_root: Path                              # subset exports go here
    ssim_fn: Optional[Callable] = None
    rng: Optional[np.random.Generator] = None
    extra_run_kwargs: dict = field(default_factory=dict)


@dataclass
class ScoreFnPolicy:
    """Two-pass budget policy for placement profiling beam search.

    Codex Q1 verdict (c): cheap (12-step ν-only) for K=1/K=2 layers,
    full (100-step) for the K=3 final decision layer.

    `cheap_threshold_K` is the K at which we switch to full budget;
    K >= cheap_threshold_K uses full_n_steps, K < uses cheap_n_steps.
    """
    cheap_n_steps: int = 12
    full_n_steps: int = 100
    cheap_threshold_K: int = 3


# ---------------------------------------------------------------------------
# Config cloning helpers
# ---------------------------------------------------------------------------


def clone_config_for_n_steps(base_config: Any, n_steps: int) -> Any:
    """Clone a VADIv5Config with custom A-stage step count.

    The score function uses ν-only insert-only PGD: schedule_preset
    = "full" with N_A_nu=n_steps, N_B_delta=0, N_C_alt=0 produces the
    cleanest A0 surrogate. delta_support_mode is forced to "off" so
    no δ contamination of the score (placement profiling is about
    where to insert, not about δ vs no-δ).

    Note we deepcopy to avoid sharing mutable fields with caller's
    base_config.
    """
    cfg = deepcopy(base_config)
    # Force insert-only ν-only schedule.
    cfg.schedule_preset = "full"
    cfg.N_A_nu = int(n_steps)
    cfg.N_B_delta = 0
    cfg.N_C_alt = 0
    cfg.delta_support_mode = "off"
    # Disable any polish stages — placement profiling measures pure A0.
    cfg.boundary_polish = False
    cfg.hiera_steering = False
    cfg.state_continuation = False
    cfg.joint_trajectory = False
    return cfg


def subset_tag(subset: Sequence[int]) -> str:
    """Stable directory tag for a subset (sorted, x-separated)."""
    return "x".join(str(int(c)) for c in sorted(subset))


# ---------------------------------------------------------------------------
# Score function builder
# ---------------------------------------------------------------------------


def build_score_fn(
    ctx: ScoreFnContext,
    policy: ScoreFnPolicy,
    run_v5_for_clip: Callable,
    *,
    infeasible_score: float = -1.0,
    config_name_prefix: str = "profile",
    cleanup_export_after_scoring: bool = True,
    metadata_sink: Optional[dict] = None,
) -> Callable[[Tuple[int, ...]], float]:
    """Build a callable score_fn(subset) → J_drop.

    The returned function is suitable for direct use with
    `placement_profiler.beam_search_K3` (and benefits from wrapping
    via `make_cached_scorer` to avoid re-evaluating duplicate subsets
    across beam-layer expansions).

    Args:
      ctx: clip-level context (immutable across subsets).
      policy: two-pass budget policy.
      run_v5_for_clip: the driver's orchestrator function (passed
        in to avoid an import cycle between memshield/ and scripts/).
      infeasible_score: returned when v5 run reports infeasible. Should
        be lower than any plausible J-drop so beam search ranks it last.
      config_name_prefix: prepended to the subset_tag for export dir
        naming (e.g. "profile" → "profile__5x10x15").
      cleanup_export_after_scoring: if True (default) delete the per-
        subset export directory after extracting J-drop. Important for
        beam profiling (~850 evals/clip × ~40MB/export ≈ 35GB/clip
        without cleanup).
      metadata_sink: optional caller-owned dict; if provided, score_fn
        writes per-subset metadata under key `subset_tag(subset)` —
        contains (decoy_offsets, infeasible, exported_j_drop). The
        caller can attach this metadata to the corresponding
        SubsetScore via beam_search post-processing.

    Returns:
      score_fn(subset) → float (J_drop_mean, or infeasible_score on
      infeasible run).
    """
    if not callable(run_v5_for_clip):
        raise ValueError("run_v5_for_clip must be callable")

    def _score_fn(subset: Tuple[int, ...]) -> float:
        K = len(subset)
        if K < 1:
            raise ValueError(f"score_fn called with empty subset: {subset!r}")
        n_steps = (
            policy.full_n_steps
            if K >= policy.cheap_threshold_K
            else policy.cheap_n_steps
        )
        cfg = clone_config_for_n_steps(ctx.base_config, n_steps)
        config_name = f"{config_name_prefix}__{subset_tag(subset)}__K{K}__N{n_steps}"

        try:
            output = run_v5_for_clip(
                clip_name=ctx.clip_name,
                config_name=config_name,
                x_clean=ctx.x_clean,
                prompt_mask=ctx.prompt_mask,
                clean_pass_fn=ctx.clean_pass_fn,
                forward_fn_builder=ctx.forward_fn_builder,
                lpips_fn=ctx.lpips_fn,
                ssim_fn=ctx.ssim_fn,
                K_ins=K,
                min_gap=2,
                placement_mode="top",          # ignored when override is set
                post_insert_radius=8,
                rng=ctx.rng,
                config=cfg,
                out_root=ctx.out_root,
                sam2_eval_fn=ctx.sam2_eval_fn,
                W_clean_override=sorted(int(c) for c in subset),
                **ctx.extra_run_kwargs,
            )
        except Exception as e:
            # Surface the failure but do not abort the entire profiling
            # run; record as infeasible_score so the caller's beam search
            # can continue with other subsets.
            print(f"[v5_score_fn] subset={subset} EXCEPTION: "
                  f"{type(e).__name__}: {e}")
            return float(infeasible_score)

        # Record metadata before cleanup so we keep the decoy_offsets etc.
        if metadata_sink is not None:
            tag = subset_tag(subset)
            metadata_sink[tag] = {
                "subset": sorted(int(c) for c in subset),
                "K": int(K),
                "n_steps": int(n_steps),
                "infeasible": bool(getattr(output, "infeasible", True)),
                "exported_j_drop": getattr(output, "exported_j_drop", None),
                "decoy_offsets": (
                    [list(o) for o in getattr(output, "decoy_offsets", [])]
                    if getattr(output, "decoy_offsets", None) is not None
                    else None),
                "W_attacked": list(getattr(output, "W", [])),
                "config_name": config_name,
            }

        # Clean up export directory to bound disk usage during profiling.
        if cleanup_export_after_scoring:
            export_dir = getattr(output, "export_dir", None)
            if export_dir:
                # export_dir is the .../processed/ leaf — clean its parent
                # (the config_name dir) which is unique per subset.
                config_dir = Path(export_dir).parent
                if config_dir.exists():
                    try:
                        shutil.rmtree(config_dir)
                    except OSError as e:
                        print(f"[v5_score_fn] cleanup of {config_dir} failed: {e}")

        if output is None or output.infeasible:
            return float(infeasible_score)
        if output.exported_j_drop is None:
            return float(infeasible_score)
        return float(output.exported_j_drop)

    return _score_fn


# ---------------------------------------------------------------------------
# Self-tests (mocked run_v5_for_clip — no SAM2 dependency)
# ---------------------------------------------------------------------------


def _make_mock_config():
    """Build a minimal mock of VADIv5Config so deepcopy + field set works."""
    class _MockConfig:
        schedule_preset = "full"
        N_A_nu = 100
        N_B_delta = 0
        N_C_alt = 0
        delta_support_mode = "post_insert"
        boundary_polish = False
        hiera_steering = False
        state_continuation = False
        joint_trajectory = False
    return _MockConfig()


def _make_mock_v5_output(j_drop: Optional[float], infeasible: bool):
    class _MockOutput:
        pass
    o = _MockOutput()
    o.exported_j_drop = j_drop
    o.infeasible = infeasible
    return o


def _test_clone_config_for_n_steps() -> None:
    base = _make_mock_config()
    base.boundary_polish = True
    base.delta_support_mode = "v4_symmetric"
    base.N_A_nu = 100
    cfg = clone_config_for_n_steps(base, n_steps=12)
    assert cfg.N_A_nu == 12
    assert cfg.N_B_delta == 0
    assert cfg.N_C_alt == 0
    assert cfg.delta_support_mode == "off"
    assert cfg.boundary_polish is False
    assert cfg.schedule_preset == "full"
    # Ensure base_config is NOT mutated.
    assert base.N_A_nu == 100
    assert base.delta_support_mode == "v4_symmetric"
    assert base.boundary_polish is True
    print("  clone_config_for_n_steps OK")


def _test_subset_tag() -> None:
    assert subset_tag([5, 10, 15]) == "5x10x15"
    assert subset_tag([15, 5, 10]) == "5x10x15"      # sort
    assert subset_tag([3]) == "3"
    print("  subset_tag OK")


def _test_score_fn_routing() -> None:
    """Score function picks cheap n_steps for K<3, full for K==3."""
    base = _make_mock_config()
    captured = []

    def mock_run(*, K_ins, W_clean_override, config, **kwargs):
        captured.append({
            "K_ins": K_ins,
            "W_clean_override": list(W_clean_override),
            "N_A_nu": config.N_A_nu,
        })
        return _make_mock_v5_output(j_drop=0.5 + 0.01 * K_ins, infeasible=False)

    ctx = ScoreFnContext(
        clip_name="dog",
        x_clean=None,
        prompt_mask=np.zeros((4, 4)),
        clean_pass_fn=lambda *a, **k: None,
        forward_fn_builder=lambda *a, **k: None,
        lpips_fn=lambda *a, **k: None,
        sam2_eval_fn=lambda *a, **k: None,
        base_config=base,
        out_root=Path("/tmp/_v5_score_fn_test"),
    )
    policy = ScoreFnPolicy(cheap_n_steps=12, full_n_steps=100,
                           cheap_threshold_K=3)
    score = build_score_fn(ctx, policy, run_v5_for_clip=mock_run)

    s1 = score((5,))
    s2 = score((5, 10))
    s3 = score((5, 10, 15))

    assert abs(s1 - 0.51) < 1e-6
    assert abs(s2 - 0.52) < 1e-6
    assert abs(s3 - 0.53) < 1e-6

    assert captured[0]["K_ins"] == 1
    assert captured[0]["N_A_nu"] == 12
    assert captured[1]["K_ins"] == 2
    assert captured[1]["N_A_nu"] == 12
    assert captured[2]["K_ins"] == 3
    assert captured[2]["N_A_nu"] == 100
    print("  score_fn_routing OK")


def _test_infeasible_returns_low() -> None:
    base = _make_mock_config()

    def mock_run(*, K_ins, W_clean_override, config, **kwargs):
        return _make_mock_v5_output(j_drop=None, infeasible=True)

    ctx = ScoreFnContext(
        clip_name="dog", x_clean=None,
        prompt_mask=np.zeros((4, 4)),
        clean_pass_fn=lambda *a, **k: None,
        forward_fn_builder=lambda *a, **k: None,
        lpips_fn=lambda *a, **k: None,
        sam2_eval_fn=lambda *a, **k: None,
        base_config=base,
        out_root=Path("/tmp/_v5_score_fn_test"),
    )
    policy = ScoreFnPolicy()
    score = build_score_fn(ctx, policy, run_v5_for_clip=mock_run,
                           infeasible_score=-99.0)
    assert score((5,)) == -99.0
    print("  infeasible_returns_low OK")


def _test_exception_returns_infeasible() -> None:
    base = _make_mock_config()

    def mock_run(*, K_ins, W_clean_override, config, **kwargs):
        raise RuntimeError("synthetic failure")

    ctx = ScoreFnContext(
        clip_name="dog", x_clean=None,
        prompt_mask=np.zeros((4, 4)),
        clean_pass_fn=lambda *a, **k: None,
        forward_fn_builder=lambda *a, **k: None,
        lpips_fn=lambda *a, **k: None,
        sam2_eval_fn=lambda *a, **k: None,
        base_config=base,
        out_root=Path("/tmp/_v5_score_fn_test"),
    )
    policy = ScoreFnPolicy()
    score = build_score_fn(ctx, policy, run_v5_for_clip=mock_run,
                           infeasible_score=-50.0)
    # Exception is logged, not raised.
    s = score((5,))
    assert s == -50.0
    print("  exception_returns_infeasible OK")


def _test_empty_subset_raises() -> None:
    base = _make_mock_config()
    ctx = ScoreFnContext(
        clip_name="x", x_clean=None,
        prompt_mask=np.zeros((1, 1)),
        clean_pass_fn=lambda *a, **k: None,
        forward_fn_builder=lambda *a, **k: None,
        lpips_fn=lambda *a, **k: None,
        sam2_eval_fn=lambda *a, **k: None,
        base_config=base,
        out_root=Path("/tmp/_v5_score_fn_test"),
    )
    score = build_score_fn(
        ctx, ScoreFnPolicy(),
        run_v5_for_clip=lambda **k: _make_mock_v5_output(0.5, False))
    try:
        score(())
    except ValueError as e:
        assert "empty subset" in str(e)
    else:
        raise AssertionError("expected ValueError on empty subset")
    print("  empty_subset_raises OK")


if __name__ == "__main__":
    print("memshield.v5_score_fn self-tests:")
    _test_clone_config_for_n_steps()
    _test_subset_tag()
    _test_score_fn_routing()
    _test_infeasible_returns_low()
    _test_exception_returns_infeasible()
    _test_empty_subset_raises()
    print("memshield.v5_score_fn: all self-tests PASSED")
