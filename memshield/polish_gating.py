"""Shared perceptual-gated polish-stage helper for VADI v5 driver.

Round 5 Bundle A sub-session 2 (2026-04-25). Extracts the preflight +
remeasure + accept/revert pattern that Stages 10-13 in
`scripts/run_vadi_v5.py` repeat with subtle variations, into one
parameterized helper that:

  1. Remeasures A0 export feasibility against `clean_refs_for_inserts`.
  2. Applies a `perceptual_check_fn` to decide whether A0 is good
     enough to be the starting point for polish.
  3. Calls a `polish_fn` callback that runs the actual stage-specific
     PGD; expects unified return:
        (x_for_export, delta_for_export, nu_for_export,
         step_logs, best_step)
  4. Exports the polish artifact to `polish_dir`, loads, evaluates
     J-drop via `eval_exported_j_drop`, and remeasures perceptual
     feasibility on the polish export.
  5. Applies the off-switch accept rule:
        accept iff (perceptual_feasible AND
                    (not off_switch OR delta_j >= min_improvement))
  6. Returns a `PolishGatingResult` dataclass; the caller is responsible
     for writing stage-specific polish_stats keys (so we don't break
     Stage 13's existing key naming during migration).

The helper is intentionally side-effect-light: it does not write into
any caller dict directly. Callers compose the result into their own
`polish_stats` mapping under the keys they prefer (e.g., Stage 13 uses
`jt_a0_preflight` while a future Stage 14 might use `ot_a0_preflight`).

Stage adapter pattern: each stage's existing PGD function returns its
native tuple (e.g., Stage 13 returns
`(x_edited_star, nu_jt, edit_params, logs, best_step)`); the caller
wraps it into a `polish_fn` lambda that adapts to this helper's
required `(x_for_export, delta_for_export, nu_for_export, logs,
best_step)` signature. δ-style stages (Stages 10-12) pass
`x_clean` for `x_for_export` and `delta_polish` for `delta_for_export`;
edited-frame stages (Stage 13/14) pass `x_edited_star` for
`x_for_export` and `torch.zeros_like(...)` for `delta_for_export`.

Tests: `python -m memshield.polish_gating` runs unit + parity self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PolishGatingResult:
    """Structured outcome of a perceptual-gated polish stage call.

    The helper populates these fields from helper-internal events; the
    caller picks the subset relevant to its stage and writes them under
    stage-specific polish_stats keys.

    Lifecycle states:
      A. Preflight infeasible
         → a0_preflight_remeasure populated
         → a0_perceptual_feasible = False
         → skip_reason = preflight_skip_reason
         → all polish_* fields are None / False
         → accept = False
      B. Polish returned None (no feasible step)
         → a0_preflight_remeasure populated, perceptual ok
         → polish_step_logs populated (may be empty)
         → polish_skipped_reason explains
         → all polish_export_* fields None
         → accept = False
      C. Polish ran, exported, evaluated
         → all polish_* fields populated
         → accept = perceptual gate AND off-switch rule
    """
    # --- Preflight (always populated when helper runs) ---
    a0_preflight_remeasure: Optional[Dict[str, Any]] = None
    a0_perceptual_feasible: bool = False
    a0_full_feasible: bool = False
    preflight_skip_reason: Optional[str] = None

    # --- Polish run (populated if preflight passed) ---
    polish_skipped_reason: Optional[str] = None
    polish_step_logs: List[Dict[str, Any]] = field(default_factory=list)
    polish_best_step: Optional[int] = None
    # Polish-stage outputs that the caller may want to keep for stats:
    polish_x_for_export: Optional[Tensor] = None
    polish_delta_for_export: Optional[Tensor] = None
    polish_nu_for_export: Optional[Tensor] = None

    # --- Eval (populated if polish ran successfully) ---
    polish_j_drop: Optional[float] = None
    a0_j_drop: Optional[float] = None
    delta_j: Optional[float] = None
    polish_eval_details: Optional[Dict[str, Any]] = None

    # --- Export remeasure (populated if polish exported) ---
    polish_export_remeasure: Optional[Dict[str, Any]] = None
    polish_perceptual_feasible: bool = False
    polish_full_feasible: bool = False
    polish_dir: Optional[Path] = None

    # --- Final decision ---
    accept: bool = False


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def build_clean_refs_for_inserts(
    x_clean: Tensor,
    W_attacked: Sequence[int],
    *,
    error_label: str = "polish",
) -> Tensor:
    """Stack per-insert clean reference frames `x_clean[w_k - k]`.

    The k-th insert at processed position w_k corresponds to clean frame
    `w_k - k` (the frame "underneath" insertion offset). This invariant
    must hold under v5 placement rules; we assert it loudly so any
    upstream bug surfaces.

    Codex Loop3-R3 fix preserved: explicit RuntimeError so the check
    survives `python -O` (assert is stripped under -O).
    """
    W_sorted_int = sorted(int(w) for w in W_attacked)
    T_clean = int(x_clean.shape[0])
    refs = []
    for k_, w_ in enumerate(W_sorted_int):
        c_k_ = w_ - k_
        if not (0 <= c_k_ < T_clean):
            raise RuntimeError(
                f"{error_label}: invalid c_k={c_k_} for "
                f"W_sorted[{k_}]={w_} (T_clean={T_clean}) — invariant "
                f"w-k in [0, T_clean) violated")
        refs.append(x_clean[c_k_])
    return torch.stack(refs, dim=0).to(x_clean.device)


def run_perceptual_gated_polish(
    *,
    # Scene state
    x_clean: Tensor,
    decoy_seeds: Tensor,
    a0_export: Tensor,
    a0_j_drop_val: Optional[float],
    W_attacked: Sequence[int],
    decoy_offsets: Sequence[Tuple[int, int]],

    # Pre-built clean refs (caller may rebuild differently per stage)
    clean_refs_for_inserts: Tensor,

    # Perceptual feasibility checker (e.g., _jt_perceptual_feasible)
    perceptual_check_fn: Callable[[Dict[str, Any], Any], bool],
    config: Any,

    # Polish driver — caller adapts native return to unified signature
    # `(x_for_export, delta_for_export, nu_for_export, step_logs,
    #   best_step)`. None for x/delta/nu means "polish failed".
    polish_fn: Callable[
        [],
        Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
              List[Dict[str, Any]], Optional[int]],
    ],

    # Eval helpers
    sam2_eval_fn: Callable,
    prompt_mask: Tensor,
    lpips_fn: Callable,
    ssim_fn: Optional[Callable],
    m_true_by_t: Dict[int, Tensor],
    m_decoy_by_t: Dict[int, Tensor],

    # Imports passed in to avoid circular dep on driver-side helpers
    export_processed_uint8: Callable,
    load_processed_uint8: Callable,
    eval_exported_j_drop: Callable,
    remeasure_exported_feasibility: Callable,

    # Output
    polish_dir: Path,

    # Off-switch
    off_switch: bool,
    min_improvement: float,

    # Skip-reason naming (e.g., "joint_trajectory_a0_preflight_infeasible")
    skip_reason_prefix: str,
) -> PolishGatingResult:
    """Run preflight → polish → eval → remeasure → accept/revert.

    See module docstring for lifecycle states A/B/C.
    """
    result = PolishGatingResult()
    result.polish_dir = Path(polish_dir)

    # --- Stage A: A0 export preflight remeasure ---
    a0_remeasure = remeasure_exported_feasibility(
        x_clean, clean_refs_for_inserts, a0_export, W_attacked,
        lpips_fn, ssim_fn, config,
    )
    result.a0_preflight_remeasure = a0_remeasure
    result.a0_perceptual_feasible = bool(perceptual_check_fn(a0_remeasure, config))
    result.a0_full_feasible = bool(
        a0_remeasure.get("step_feasible_on_export", False))

    if not result.a0_perceptual_feasible:
        result.preflight_skip_reason = (
            f"{skip_reason_prefix}_a0_preflight_infeasible")
        return result  # State A: preflight infeasible

    # --- Stage B: run polish ---
    x_for_export, delta_for_export, nu_for_export, step_logs, best_step = (
        polish_fn())
    result.polish_step_logs = step_logs
    result.polish_best_step = best_step

    if x_for_export is None or nu_for_export is None:
        result.polish_skipped_reason = f"{skip_reason_prefix}_no_feasible_step"
        return result  # State B: polish failed

    # delta_for_export may be None if caller's polish_fn returns None
    # for δ-zero stages — coerce to zeros tensor on caller's behalf so
    # downstream export call has consistent inputs.
    if delta_for_export is None:
        delta_for_export = torch.zeros_like(x_for_export)

    result.polish_x_for_export = x_for_export
    result.polish_delta_for_export = delta_for_export
    result.polish_nu_for_export = nu_for_export

    # --- Stage C: export + eval + remeasure ---
    export_processed_uint8(
        x_for_export.to(x_clean.device),
        delta_for_export.to(x_clean.device),
        nu_for_export.to(x_clean.device),
        decoy_seeds, W_attacked, polish_dir,
    )
    exported_polish = load_processed_uint8(polish_dir).to(x_clean.device)

    polish_eval = eval_exported_j_drop(
        sam2_eval_fn=sam2_eval_fn,
        prompt_mask=prompt_mask,
        x_clean=x_clean,
        base_inserts=decoy_seeds,
        exported=exported_polish,
        W=W_attacked,
        m_hat_true_by_t=m_true_by_t,
        m_hat_decoy_by_t=m_decoy_by_t,
        decoy_offsets=decoy_offsets,
    )
    result.polish_eval_details = polish_eval
    result.polish_j_drop = float(polish_eval["J_drop_mean"])

    # Codex R3 high-fix preserved: `exported_j_drop_val or 0.0` would
    # silently treat a negative A0 J-drop (rare failed clips) as 0.0,
    # corrupting the accept/revert decision. Use explicit None check.
    a0_j_drop = (
        a0_j_drop_val if a0_j_drop_val is not None else 0.0)
    result.a0_j_drop = a0_j_drop
    result.delta_j = result.polish_j_drop - a0_j_drop

    # Polish-export remeasure (perceptual-only gate per Codex Loop3-R4).
    polish_remeasure = remeasure_exported_feasibility(
        x_clean, clean_refs_for_inserts, exported_polish, W_attacked,
        lpips_fn, ssim_fn, config,
    )
    result.polish_export_remeasure = polish_remeasure
    result.polish_perceptual_feasible = bool(
        perceptual_check_fn(polish_remeasure, config))
    result.polish_full_feasible = bool(
        polish_remeasure.get("step_feasible_on_export", False))

    # --- Final accept/revert decision ---
    result.accept = bool(
        result.polish_perceptual_feasible
        and (
            (not off_switch)
            or (result.delta_j >= min_improvement)
        )
    )
    return result


# ---------------------------------------------------------------------------
# Self-tests (mock-based, no SAM2 dependency)
# ---------------------------------------------------------------------------


def _mock_perceptual_check(remeasure: Dict[str, Any], _config: Any) -> bool:
    """Mock perceptual check — passes when remeasure dict says ok."""
    return bool(remeasure.get("perceptual_ok", True))


def _mock_remeasure_factory(
    a0_perceptual_ok: bool, polish_perceptual_ok: bool,
    a0_full_ok: bool = True, polish_full_ok: bool = True,
):
    """Build a mock remeasure_exported_feasibility that returns canned dicts.

    Distinguishes A0 preflight from polish remeasure by tensor identity
    on the `exported` arg.
    """
    state = {"call_count": 0}

    def remeasure(_x_clean, _clean_refs, exported, _W, _lpips, _ssim, _cfg):
        state["call_count"] += 1
        # First call = A0 preflight; subsequent = polish remeasure
        if state["call_count"] == 1:
            return {
                "perceptual_ok": a0_perceptual_ok,
                "step_feasible_on_export": a0_full_ok,
                "tag": "a0",
            }
        return {
            "perceptual_ok": polish_perceptual_ok,
            "step_feasible_on_export": polish_full_ok,
            "tag": "polish",
        }
    return remeasure


def _mock_eval_factory(j_drop_value: float):
    def evaluator(*, sam2_eval_fn, prompt_mask, x_clean, base_inserts,
                  exported, W, m_hat_true_by_t, m_hat_decoy_by_t,
                  decoy_offsets):
        return {
            "J_drop_mean": j_drop_value,
            "frames": list(range(len(W))),
        }
    return evaluator


def _mock_export_factory():
    state = {"calls": 0}

    def export(_x, _delta, _nu, _seeds, _W, polish_dir):
        Path(polish_dir).mkdir(parents=True, exist_ok=True)
        state["calls"] += 1
        return None

    def loader(_polish_dir):
        return torch.zeros((1, 4, 4, 3))

    return export, loader, state


def _test_state_A_preflight_infeasible() -> None:
    """A0 preflight fails → skip with preflight_skip_reason populated."""
    x_clean = torch.zeros((10, 8, 8, 3))
    a0 = torch.zeros((1, 8, 8, 3))
    refs = torch.zeros((3, 8, 8, 3))
    polish_called = {"value": False}

    def polish_fn():
        polish_called["value"] = True
        return (None, None, None, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 8, 8, 3)),
        a0_export=a0, a0_j_drop_val=0.5,
        W_attacked=[5, 7, 9],
        decoy_offsets=[(0, 50), (0, 50), (0, 50)],
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((8, 8)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.7),
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=False, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_A"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    assert result.a0_perceptual_feasible is False
    assert result.preflight_skip_reason == "test_a0_preflight_infeasible"
    assert result.accept is False
    assert result.polish_j_drop is None
    assert polish_called["value"] is False, "polish_fn should not be called"
    print("  state_A_preflight_infeasible OK")


def _test_state_B_polish_failed() -> None:
    """Preflight passes, polish_fn returns None → no_feasible_step."""
    x_clean = torch.zeros((10, 8, 8, 3))
    a0 = torch.zeros((1, 8, 8, 3))
    refs = torch.zeros((3, 8, 8, 3))

    def polish_fn():
        return (None, None, None, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 8, 8, 3)),
        a0_export=a0, a0_j_drop_val=0.5,
        W_attacked=[5, 7, 9],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((8, 8)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.7),
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_B"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    assert result.a0_perceptual_feasible is True
    assert result.polish_skipped_reason == "test_no_feasible_step"
    assert result.accept is False
    assert result.polish_j_drop is None
    print("  state_B_polish_failed OK")


def _test_state_C_accept() -> None:
    """All gates pass, off-switch satisfied → accept=True with delta_j."""
    x_clean = torch.zeros((10, 4, 4, 3))
    a0 = torch.zeros((1, 4, 4, 3))
    refs = torch.zeros((3, 4, 4, 3))
    x_edit = torch.ones_like(x_clean) * 0.5
    nu = torch.zeros((3, 4, 4, 3))

    def polish_fn():
        return (x_edit, None, nu, [{"step": 0}], 1)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 4, 4, 3)),
        a0_export=a0, a0_j_drop_val=0.5,
        W_attacked=[2, 4, 6],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((4, 4)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.65),
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_C"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    assert result.accept is True
    assert abs(result.polish_j_drop - 0.65) < 1e-6
    assert abs(result.a0_j_drop - 0.5) < 1e-6
    assert abs(result.delta_j - 0.15) < 1e-6
    assert result.polish_perceptual_feasible is True
    assert result.polish_delta_for_export is not None
    assert torch.allclose(result.polish_delta_for_export, torch.zeros_like(x_edit))
    print("  state_C_accept OK")


def _test_state_C_revert_off_switch() -> None:
    """Off-switch with delta_j < min_improvement → revert."""
    x_clean = torch.zeros((10, 4, 4, 3))
    a0 = torch.zeros((1, 4, 4, 3))
    refs = torch.zeros((3, 4, 4, 3))
    x_edit = torch.ones_like(x_clean) * 0.5
    nu = torch.zeros((3, 4, 4, 3))

    def polish_fn():
        return (x_edit, None, nu, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 4, 4, 3)),
        a0_export=a0, a0_j_drop_val=0.7,
        W_attacked=[2, 4, 6],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((4, 4)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.65),  # less than a0=0.7
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_revert"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    # Polish is feasible but delta_j = 0.65 - 0.7 = -0.05 < 0.0 min → revert
    assert result.accept is False
    assert result.polish_perceptual_feasible is True
    assert abs(result.delta_j - (-0.05)) < 1e-6
    print("  state_C_revert_off_switch OK")


def _test_state_C_revert_perceptual() -> None:
    """Polish-export perceptual fail → revert (overrides positive delta_j)."""
    x_clean = torch.zeros((10, 4, 4, 3))
    a0 = torch.zeros((1, 4, 4, 3))
    refs = torch.zeros((3, 4, 4, 3))
    x_edit = torch.ones_like(x_clean) * 0.5
    nu = torch.zeros((3, 4, 4, 3))

    def polish_fn():
        return (x_edit, None, nu, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 4, 4, 3)),
        a0_export=a0, a0_j_drop_val=0.5,
        W_attacked=[2, 4, 6],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((4, 4)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.95),  # huge delta_j=+0.45
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=False),
        polish_dir=Path("/tmp/_polish_gating_test_revert_perc"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    assert result.polish_perceptual_feasible is False
    assert result.accept is False, (
        "perceptual gate must override even a strongly positive delta_j")
    assert abs(result.delta_j - 0.45) < 1e-6
    print("  state_C_revert_perceptual OK")


def _test_off_switch_disabled() -> None:
    """off_switch=False accepts negative delta_j as long as perceptual ok."""
    x_clean = torch.zeros((10, 4, 4, 3))
    a0 = torch.zeros((1, 4, 4, 3))
    refs = torch.zeros((3, 4, 4, 3))
    x_edit = torch.ones_like(x_clean) * 0.5
    nu = torch.zeros((3, 4, 4, 3))

    def polish_fn():
        return (x_edit, None, nu, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 4, 4, 3)),
        a0_export=a0, a0_j_drop_val=0.7,
        W_attacked=[2, 4, 6],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((4, 4)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.65),
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_no_off"),
        off_switch=False, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    # off_switch disabled → accept regardless of delta_j sign
    assert result.accept is True
    assert result.delta_j < 0
    print("  off_switch_disabled OK")


def _test_a0_j_drop_none() -> None:
    """a0_j_drop_val=None → treated as 0.0, NOT silently skipping accept logic."""
    x_clean = torch.zeros((10, 4, 4, 3))
    a0 = torch.zeros((1, 4, 4, 3))
    refs = torch.zeros((3, 4, 4, 3))
    x_edit = torch.ones_like(x_clean) * 0.5
    nu = torch.zeros((3, 4, 4, 3))

    def polish_fn():
        return (x_edit, None, nu, [], None)

    export, loader, _ = _mock_export_factory()
    result = run_perceptual_gated_polish(
        x_clean=x_clean, decoy_seeds=torch.zeros((3, 4, 4, 3)),
        a0_export=a0, a0_j_drop_val=None,
        W_attacked=[2, 4, 6],
        decoy_offsets=[(0, 50)] * 3,
        clean_refs_for_inserts=refs,
        perceptual_check_fn=_mock_perceptual_check,
        config=None,
        polish_fn=polish_fn,
        sam2_eval_fn=lambda **_: None,
        prompt_mask=torch.zeros((4, 4)),
        lpips_fn=None, ssim_fn=None,
        m_true_by_t={}, m_decoy_by_t={},
        export_processed_uint8=export,
        load_processed_uint8=loader,
        eval_exported_j_drop=_mock_eval_factory(0.20),
        remeasure_exported_feasibility=_mock_remeasure_factory(
            a0_perceptual_ok=True, polish_perceptual_ok=True),
        polish_dir=Path("/tmp/_polish_gating_test_anone"),
        off_switch=True, min_improvement=0.0,
        skip_reason_prefix="test",
    )
    assert abs(result.a0_j_drop - 0.0) < 1e-6
    assert abs(result.delta_j - 0.20) < 1e-6
    assert result.accept is True
    print("  a0_j_drop_none OK")


def _test_build_clean_refs_for_inserts() -> None:
    x_clean = torch.arange(20 * 4 * 4 * 3, dtype=torch.float32).reshape(
        20, 4, 4, 3)
    # W = [3, 7, 11] → c_k = [3-0, 7-1, 11-2] = [3, 6, 9]
    refs = build_clean_refs_for_inserts(
        x_clean, W_attacked=[3, 7, 11], error_label="test")
    assert refs.shape == (3, 4, 4, 3)
    assert torch.allclose(refs[0], x_clean[3])
    assert torch.allclose(refs[1], x_clean[6])
    assert torch.allclose(refs[2], x_clean[9])
    # Invariant violation should raise
    try:
        build_clean_refs_for_inserts(
            x_clean, W_attacked=[1, 0, -5], error_label="test")
    except RuntimeError as e:
        assert "invariant" in str(e)
    else:
        raise AssertionError("expected RuntimeError on negative c_k")
    print("  build_clean_refs_for_inserts OK")


if __name__ == "__main__":
    print("memshield.polish_gating self-tests:")
    _test_build_clean_refs_for_inserts()
    _test_state_A_preflight_infeasible()
    _test_state_B_polish_failed()
    _test_state_C_accept()
    _test_state_C_revert_off_switch()
    _test_state_C_revert_perceptual()
    _test_off_switch_disabled()
    _test_a0_j_drop_none()
    print("memshield.polish_gating: all self-tests PASSED")
