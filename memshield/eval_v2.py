"""Evaluation primitives for MemoryShield v2 (Chunk 6.1).

Pure functions over binary mask arrays. No torch / SAM2 dependencies —
consume numpy output from the attack pipeline.

Primitives:
  * `jaccard(pred, gt)` — IoU of two binary masks. Both-empty → 1.0
    (matches `scripts/sam2long_eval.py::db_eval_iou`).
  * `j_trajectory(pred_masks, gt_masks, eval_start, eval_end)` — per-frame
    Jaccard over the eval window. Handles shape mismatch via nearest-
    neighbor resize to GT resolution.
  * `post_loss_auc(j_traj)` — normalized area = mean(J_u), bounds [0, 1].
  * `rebound_at_k(j_traj, threshold)` — first u ≥ 1 where
    J_u ≥ max(J_0, threshold); −1 if never. u=0 is excluded because it
    trivially satisfies the self-reference.

An end-to-end `evaluate_run` wrapper composes all three primitives and
returns a `RunMetrics` dataclass that JSON-serializes via `.to_dict()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of two binary masks. Returns 1.0 if both are empty."""
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def j_trajectory(pred_masks: Sequence[np.ndarray],
                 gt_masks: Sequence[np.ndarray],
                 eval_start: int,
                 eval_end: int) -> np.ndarray:
    """Per-frame Jaccard on the eval window [eval_start, eval_end).

    Args:
        pred_masks: length-T sequence of [H, W] uint8/bool masks (or None).
        gt_masks:   length-T sequence of [H, W] uint8/bool masks.
        eval_start: inclusive index (typically T_prefix).
        eval_end:   exclusive index (typically T_prefix + eval_window).

    Returns:
        float32 array of shape (eval_end - eval_start,).
    """
    if eval_end <= eval_start:
        raise ValueError(f"eval_end {eval_end} <= eval_start {eval_start}")
    if eval_end > len(gt_masks):
        raise ValueError(
            f"eval_end {eval_end} > len(gt_masks) {len(gt_masks)}")
    if eval_end > len(pred_masks):
        raise ValueError(
            f"eval_end {eval_end} > len(pred_masks) {len(pred_masks)}")
    scores = np.empty(eval_end - eval_start, dtype=np.float32)
    for i, t in enumerate(range(eval_start, eval_end)):
        pred = pred_masks[t]
        gt = gt_masks[t]
        if pred is None:
            scores[i] = 0.0
            continue
        if pred.shape != gt.shape:
            from PIL import Image as _Im
            p_img = _Im.fromarray((pred > 0).astype(np.uint8) * 255)
            p_img = p_img.resize((gt.shape[1], gt.shape[0]), _Im.NEAREST)
            pred = (np.array(p_img) > 0).astype(np.uint8)
        scores[i] = jaccard(pred, gt)
    return scores


def post_loss_auc(j_traj: np.ndarray) -> float:
    """Normalized AUC of the J trajectory. Equal to mean(J_u)."""
    if len(j_traj) == 0:
        return float("nan")
    return float(np.mean(j_traj))


def rebound_at_k(j_traj: np.ndarray, threshold: float = 0.5) -> int:
    """First u ≥ 1 where J_u ≥ max(J_0, threshold); −1 if never recovers.

    J_0 is the attack-time dip reference. Recovery means returning to at
    least that level, bounded below by the absolute `threshold`.
    """
    if len(j_traj) < 2:
        return -1
    ref = max(float(j_traj[0]), float(threshold))
    for u in range(1, len(j_traj)):
        if float(j_traj[u]) >= ref:
            return u
    return -1


@dataclass
class RunMetrics:
    j_per_eval_frame: List[float]
    auc: float
    rebound_at_1: int
    mean_j: float
    eval_start: int
    eval_end: int
    threshold: float

    def to_dict(self) -> dict:
        return {
            "j_per_eval_frame": [float(x) for x in self.j_per_eval_frame],
            "auc": float(self.auc),
            "rebound_at_1": int(self.rebound_at_1),
            "mean_j": float(self.mean_j),
            "eval_start": int(self.eval_start),
            "eval_end": int(self.eval_end),
            "threshold": float(self.threshold),
        }


def evaluate_run(pred_masks: Sequence[np.ndarray],
                 gt_masks: Sequence[np.ndarray],
                 T_prefix: int,
                 eval_window_size: int,
                 threshold: float = 0.5) -> RunMetrics:
    """Compose j_trajectory + post_loss_auc + rebound_at_k for one run."""
    eval_start = T_prefix
    eval_end = T_prefix + eval_window_size
    j_traj = j_trajectory(pred_masks, gt_masks, eval_start, eval_end)
    mean_j = float(np.mean(j_traj)) if len(j_traj) > 0 else float("nan")
    return RunMetrics(
        j_per_eval_frame=[float(x) for x in j_traj],
        auc=post_loss_auc(j_traj),
        rebound_at_1=rebound_at_k(j_traj, threshold=threshold),
        mean_j=mean_j,
        eval_start=eval_start,
        eval_end=eval_end,
        threshold=threshold,
    )


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------


def _self_test() -> None:
    # --- jaccard ---
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert abs(jaccard(a, b) - 0.5) < 1e-6
    zero = np.zeros((2, 2), dtype=np.uint8)
    assert jaccard(zero, zero) == 1.0, "both-empty convention"
    assert jaccard(a, a) == 1.0
    assert jaccard(a, zero) == 0.0

    # --- j_trajectory ---
    T, H, W = 5, 10, 10
    gt = [np.ones((H, W), dtype=np.uint8) for _ in range(T)]
    pred_perfect = [np.ones((H, W), dtype=np.uint8) for _ in range(T)]
    jt = j_trajectory(pred_perfect, gt, 2, 5)
    assert jt.shape == (3,)
    assert np.allclose(jt, 1.0)

    pred_empty = [np.zeros((H, W), dtype=np.uint8) for _ in range(T)]
    jt_empty = j_trajectory(pred_empty, gt, 2, 5)
    assert np.allclose(jt_empty, 0.0)

    # None entry → 0
    pred_with_none = [None] * T
    jt_none = j_trajectory(pred_with_none, gt, 2, 5)
    assert np.allclose(jt_none, 0.0)

    # bounds errors
    try:
        j_trajectory(pred_perfect, gt, 3, 3)
        raise AssertionError("eval_end <= eval_start must raise")
    except ValueError:
        pass
    try:
        j_trajectory(pred_perfect, gt, 0, T + 1)
        raise AssertionError("eval_end > T must raise")
    except ValueError:
        pass

    # shape-mismatch resize path
    pred_small = [np.ones((H // 2, W // 2), dtype=np.uint8) for _ in range(T)]
    jt_resize = j_trajectory(pred_small, gt, 2, 5)
    assert np.allclose(jt_resize, 1.0)

    # --- post_loss_auc ---
    assert abs(post_loss_auc(np.array([1.0, 1.0, 1.0])) - 1.0) < 1e-6
    assert abs(post_loss_auc(np.array([0.0, 0.0, 0.0])) - 0.0) < 1e-6
    assert abs(post_loss_auc(np.array([0.2, 0.4, 0.6])) - 0.4) < 1e-6
    assert np.isnan(post_loss_auc(np.array([])))

    # --- rebound_at_k ---
    # Monotone drop, J_0=0.3, thr=0.5, ref=0.5 → never ≥ 0.5 → -1
    assert rebound_at_k(np.array([0.3, 0.25, 0.2, 0.1]), threshold=0.5) == -1
    # Recovery at u=2: J_0=0.2, thr=0.5, ref=0.5. 0.6 first qualifies at u=2.
    assert rebound_at_k(np.array([0.2, 0.3, 0.6, 0.9]), threshold=0.5) == 2
    # J_0=0.8 sets high bar. [0.8, 0.3, 0.9, 0.2] → 0.9 at u=2 qualifies.
    assert rebound_at_k(np.array([0.8, 0.3, 0.9, 0.2]), threshold=0.5) == 2
    # Trivial 1-elt → -1 (no u ≥ 1 exists).
    assert rebound_at_k(np.array([0.5]), threshold=0.5) == -1
    # Flat high trajectory → u=1.
    assert rebound_at_k(np.array([0.6, 0.6, 0.6]), threshold=0.5) == 1
    # threshold > J_0 case: J_0=0.1, thr=0.5. [0.1, 0.2, 0.51] → u=2.
    assert rebound_at_k(np.array([0.1, 0.2, 0.51]), threshold=0.5) == 2
    assert rebound_at_k(np.array([0.1, 0.2, 0.49]), threshold=0.5) == -1

    # --- evaluate_run end-to-end ---
    metrics = evaluate_run(pred_perfect, gt, T_prefix=2, eval_window_size=3,
                           threshold=0.5)
    assert metrics.auc == 1.0
    assert metrics.rebound_at_1 == 1  # u=1 first satisfies J ≥ max(1.0, 0.5).
    assert metrics.mean_j == 1.0
    assert len(metrics.j_per_eval_frame) == 3
    d = metrics.to_dict()
    assert set(d.keys()) == {
        "j_per_eval_frame", "auc", "rebound_at_1", "mean_j",
        "eval_start", "eval_end", "threshold",
    }
    import json as _json
    _json.dumps(d)  # must be JSON-serializable

    print("memshield.eval_v2: all self-tests PASSED "
          "(jaccard, j_trajectory, post_loss_auc, rebound_at_k, evaluate_run)")


if __name__ == "__main__":
    _self_test()
