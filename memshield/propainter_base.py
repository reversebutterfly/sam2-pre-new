"""
Insert-base generator dispatcher, with ProPainter as the primary realization.

Purpose
-------
The MemoryShield inserts are single synthetic frames spliced into a clean
video. Their LPIPS budget is tight (<= 0.10 target, <= 0.15 fallback), so
the realization method matters: per-frame Poisson blending hits a ~0.13-0.14
floor (Pilot A/B/C, 2026-04-21) because it has no temporal coherence.

ProPainter (Zhou et al., ICCV 2023) is the chosen frontier primitive for
the insert base because it does flow-guided recurrent video inpainting
with temporal consistency, which should plausibly break the Poisson floor.
This module wraps ProPainter behind a strategy dispatcher so we can:

(a) R001 sanity:   strategy = "poisson_hifi"  (existing Pilot-B code path)
(b) R002 gate:     strategy = "propainter"    (this module)
(c) All downstream runs: strategy decided by R002 outcome per
    EXPERIMENT_PLAN.md gate logic.

Current state (2026-04-22)
--------------------------
ProPainter is NOT yet installed on the Pro 6000 memshield env. This module
loads fine without ProPainter (lazy import) and `is_propainter_available()`
returns False. Any call to `create_insert_base(strategy="propainter", ...)`
in the meantime raises an InstallationError with install instructions.

Installation
------------
Once sign-off is given, run on the Pro 6000:

    cd ~/  # home NAS
    git clone https://github.com/sczhou/ProPainter
    cd ~/ProPainter
    bash scripts/download_weights.sh       # ~500 MB of .pth checkpoints
    conda activate memshield
    # Do NOT run `pip install -r requirements.txt` verbatim because that
    # pins torch/cuda versions that would conflict with our 2.8.0/cu128
    # Blackwell build. Install only the non-torch runtime deps manually:
    pip install av einops imageio imageio-ffmpeg scipy scikit-image \
                opencv-contrib-python matplotlib

Set the env var or config path so `create_insert_base_propainter` can
locate the repo:

    export PROPAINTER_ROOT=~/ProPainter

or pass it explicitly via `ProPainterConfig.root`.

API
---
Primary entry:

    create_insert_base(strategy, frame_prev, frame_after, mask_prev,
                       mask_after, decoy_offset, **kwargs) -> Optional[(base, edit)]

Strategies:
    "propainter"    -- flow-guided recurrent inpainting (not yet installed).
    "poisson_hifi"  -- existing Pilot-B code (decoy.create_decoy_base_frame_hifi).
    "poisson_basic" -- existing baseline (decoy.create_decoy_base_frame).

Return contract (identical across strategies):
    (base_frame_uint8[H,W,3], edit_mask_uint8[H,W]) on success;
    None on failure (e.g. border-safety violation, inpaint error).

The edit_mask marks the union of regions PGD is allowed to modify
(seam + paste region + optionally true-object position for memory-redirect).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from . import decoy as _decoy  # Poisson fallbacks already implemented here


class InstallationError(RuntimeError):
    """Raised when a required external dependency is not installed."""


# ---------------------------------------------------------------------------
# Availability / configuration
# ---------------------------------------------------------------------------


@dataclass
class ProPainterConfig:
    """Paths and runtime options for ProPainter.

    Attributes:
        root: Filesystem path to a cloned `sczhou/ProPainter` repo. Defaults
            to the PROPAINTER_ROOT env var or ~/ProPainter.
        checkpoint: Path to the ProPainter main checkpoint (.pth). If None,
            resolved to `<root>/weights/ProPainter.pth`.
        raft_checkpoint: Path to the RAFT-Things checkpoint used by
            ProPainter's flow completion branch. Defaults to
            `<root>/weights/raft-things.pth`.
        recurrent_flow_checkpoint: Defaults to
            `<root>/weights/recurrent_flow_completion.pth`.
        device: Torch device string, default "cuda:0".
        neighbor_length: ProPainter's `neighbor_length` arg (default 10).
        ref_stride: Temporal reference frame stride (default 10).
        half_precision: Run the model in fp16 when True (saves memory).
    """
    root: Optional[str] = None
    checkpoint: Optional[str] = None
    raft_checkpoint: Optional[str] = None
    recurrent_flow_checkpoint: Optional[str] = None
    device: str = "cuda:0"
    neighbor_length: int = 10
    ref_stride: int = 10
    half_precision: bool = True

    def resolve(self) -> "ProPainterConfig":
        """Fill in defaults from env / home dir and return a copy."""
        root = self.root or os.environ.get("PROPAINTER_ROOT") \
            or os.path.expanduser("~/ProPainter")
        root = os.path.abspath(root)
        weights = os.path.join(root, "weights")
        return ProPainterConfig(
            root=root,
            checkpoint=self.checkpoint or os.path.join(weights, "ProPainter.pth"),
            raft_checkpoint=self.raft_checkpoint
                or os.path.join(weights, "raft-things.pth"),
            recurrent_flow_checkpoint=self.recurrent_flow_checkpoint
                or os.path.join(weights, "recurrent_flow_completion.pth"),
            device=self.device,
            neighbor_length=self.neighbor_length,
            ref_stride=self.ref_stride,
            half_precision=self.half_precision,
        )


def is_propainter_available(config: Optional[ProPainterConfig] = None) -> bool:
    """Return True if ProPainter's repo and main checkpoint both exist.

    Does NOT import ProPainter (that would trigger heavy GPU deps at import
    time and fail on missing mmcv/basicsr even when we only want to probe).
    """
    cfg = (config or ProPainterConfig()).resolve()
    if not os.path.isdir(cfg.root):
        return False
    if not os.path.isfile(cfg.checkpoint):
        return False
    # Raft weight is strictly required; recurrent-flow is nice-to-have but
    # ProPainter's default inference path will also use it.
    if not os.path.isfile(cfg.raft_checkpoint):
        return False
    # The core module must at least be importable as a directory of py files.
    core_dir = os.path.join(cfg.root, "model")
    if not os.path.isdir(core_dir):
        return False
    return True


def _install_instructions(cfg: ProPainterConfig) -> str:
    return (
        "ProPainter is not installed. Expected to find:\n"
        f"    repo:       {cfg.root}\n"
        f"    checkpoint: {cfg.checkpoint}\n"
        f"    raft:       {cfg.raft_checkpoint}\n"
        "\n"
        "Install (on Pro 6000, once user has approved):\n"
        "    cd ~\n"
        "    git clone https://github.com/sczhou/ProPainter\n"
        "    cd ~/ProPainter && bash scripts/download_weights.sh\n"
        "    conda activate memshield\n"
        "    pip install av einops imageio imageio-ffmpeg scipy \\\n"
        "                scikit-image opencv-contrib-python matplotlib\n"
        "    export PROPAINTER_ROOT=~/ProPainter\n"
        "Do NOT run `pip install -r ProPainter/requirements.txt` verbatim:\n"
        "that would pin torch/cuda versions incompatible with our 2.8.0/cu128.\n"
    )


# ---------------------------------------------------------------------------
# Strategy: "propainter"
# ---------------------------------------------------------------------------


# Lazy-loaded singletons so the model is built exactly once per process.
_PROPAINTER_STATE: Dict[str, object] = {"inpainter": None, "raft": None,
                                         "flow_complete": None}


def _load_propainter(cfg: ProPainterConfig) -> None:
    """Import and instantiate ProPainter. Raises InstallationError if absent.

    This is deferred to first-call so the rest of the codebase can import
    memshield.propainter_base without triggering the import cascade.
    """
    if _PROPAINTER_STATE["inpainter"] is not None:
        return

    if not is_propainter_available(cfg):
        raise InstallationError(_install_instructions(cfg))

    import sys
    import torch  # type: ignore

    # Ensure the ProPainter repo is on sys.path so `from model.propainter
    # import InpaintGenerator` works. ProPainter doesn't ship a pip package.
    if cfg.root not in sys.path:
        sys.path.insert(0, cfg.root)

    # The following imports are placeholders for the real module layout
    # inside sczhou/ProPainter. Actual classes:
    #     model.propainter.InpaintGenerator      (main inpainter)
    #     RAFT                                   (flow estimator, repo-local)
    #     model.recurrent_flow_completion.RecurrentFlowCompleteNet
    try:
        from model.propainter import InpaintGenerator                 # type: ignore
        from model.modules.flow_comp_raft import RAFT_bi              # type: ignore
        from model.recurrent_flow_completion import (                 # type: ignore
            RecurrentFlowCompleteNet,
        )
    except Exception as exc:  # pragma: no cover — only hit when installed
        raise InstallationError(
            f"ProPainter repo at {cfg.root} found but imports failed "
            f"({exc}). Check that you installed the non-torch runtime deps "
            "(av, einops, imageio, scipy, scikit-image)."
        ) from exc

    device = torch.device(cfg.device)
    dtype = torch.float16 if cfg.half_precision else torch.float32

    raft = RAFT_bi(cfg.raft_checkpoint, device)
    flow_complete = RecurrentFlowCompleteNet(cfg.recurrent_flow_checkpoint)
    flow_complete = flow_complete.to(device).eval()
    inpainter = InpaintGenerator(model_path=cfg.checkpoint).to(device).eval()
    if cfg.half_precision:
        flow_complete = flow_complete.half()
        inpainter = inpainter.half()
    for p in list(raft.parameters()) + list(flow_complete.parameters()) \
            + list(inpainter.parameters()):
        p.requires_grad_(False)

    _PROPAINTER_STATE["raft"] = raft
    _PROPAINTER_STATE["flow_complete"] = flow_complete
    _PROPAINTER_STATE["inpainter"] = inpainter


def _propainter_fill(
    frame_prev: np.ndarray,
    frame_after: np.ndarray,
    mask_fill: np.ndarray,
    cfg: ProPainterConfig,
) -> np.ndarray:
    """Use ProPainter to synthesize the missing content between two keyframes.

    We feed a two-frame "video" [frame_prev, frame_after] with `mask_fill`
    marking the region to inpaint on both frames, letting the model produce
    a temporally coherent reconstruction; we return the predicted frame_prev
    with the mask-region filled as the insert base.

    NOTE: This is a specification stub. The actual wiring to ProPainter's
    InpaintGenerator.forward signature depends on tensor layout and the
    neighbor/ref-stride protocol their inference loop uses; we will bind
    it concretely once ProPainter is installed and we can inspect the
    module's .forward. Until then this raises InstallationError on first
    call; `create_insert_base(strategy="propainter", ...)` will therefore
    also raise, as desired.
    """
    _load_propainter(cfg)  # raises InstallationError if not installed
    raise NotImplementedError(
        "ProPainter forward-wiring is specified but not yet implemented; "
        "this is pending Chunk 3b (actual Pro-6000 installation + smoke "
        "test). The rest of the call chain from create_insert_base() is "
        "complete and already raises a clear install error upstream."
    )


def create_insert_base_propainter(
    frame_prev: np.ndarray,
    frame_after: np.ndarray,
    mask_prev: Optional[np.ndarray],
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
    config: Optional[ProPainterConfig] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ProPainter-based insert base.

    Pipeline (same spirit as create_decoy_base_frame_hifi but with
    flow-guided inpainting instead of per-frame Poisson blending):

    1. Compute paste region = shifted copy of mask_after by decoy_offset.
       If paste region violates border safety, return None (caller tries
       another offset).
    2. Build a two-frame video [frame_prev, frame_after] and a fill mask
       covering the union of:
           - the true object region in frame_prev (mask_prev, dilated)
             so ProPainter removes the true object ("object deletion"
             signal for memory redirect)
           - the paste region in frame_prev (so ProPainter also fills in
             the decoy location with temporally-coherent content)
    3. Run ProPainter to get a temporally-consistent fill for frame_prev.
    4. Composite the object crop (from frame_after) onto the inpainted
       background at the decoy location. Use ProPainter's output INSTEAD
       of cv2.seamlessClone to avoid Poisson-blending artifacts.
    5. Edit support mask = dilated(paste_region) ∪ dilated(mask_prev).

    Args match `decoy.create_decoy_base_frame_hifi` so it can serve as a
    drop-in replacement. The only extra arg is `config`; default reads
    PROPAINTER_ROOT env or ~/ProPainter.

    Returns:
        (base_frame_uint8, edit_mask_uint8) on success; None on
        border-safety / install / model failures.

    Raises:
        InstallationError if ProPainter is selected but not installed.
    """
    cfg = (config or ProPainterConfig()).resolve()
    H, W = frame_prev.shape[:2]
    mask_ref = (mask_after > 0).astype(np.uint8)

    ys, xs = np.where(mask_ref > 0)
    if len(ys) == 0:
        return frame_prev.copy(), np.zeros((H, W), dtype=np.uint8)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dy, dx = decoy_offset
    cy_obj = (y0 + y1) // 2 + dy
    cx_obj = (x0 + x1) // 2 + dx

    if not _decoy._is_border_safe(
        (y0, y1, x0, x1), (cy_obj, cx_obj), H, W, safety_margin
    ):
        return None

    # Region to ask ProPainter to fill: true-obj (prev) + paste (decoy loc).
    prev_mask = mask_prev if mask_prev is not None else mask_after
    prev_mask_bin = (prev_mask > 0).astype(np.uint8)
    paste_region = _decoy.shift_mask(mask_ref, dy, dx)
    fill_mask = ((prev_mask_bin > 0) | (paste_region > 0)).astype(np.uint8)

    # Inpaint frame_prev (gives a clean, temporally-consistent background).
    filled_prev = _propainter_fill(frame_prev, frame_after, fill_mask, cfg)

    # Composite object crop onto inpainted background at decoy location.
    obj_crop = frame_after[y0:y1, x0:x1]
    mask_crop = mask_ref[y0:y1, x0:x1]
    paste_h, paste_w = mask_crop.shape
    half_h, half_w = paste_h // 2, paste_w // 2
    py0, py1 = cy_obj - half_h, cy_obj - half_h + paste_h
    px0, px1 = cx_obj - half_w, cx_obj - half_w + paste_w
    base = filled_prev.copy()
    # Soft alpha composite using mask_crop (same dtype as result).
    alpha = mask_crop.astype(np.float32)[..., None]
    base[py0:py1, px0:px1] = (
        alpha * obj_crop.astype(np.float32)
        + (1.0 - alpha) * base[py0:py1, px0:px1].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Edit support mask
    import cv2  # local to keep module-top clean
    if seam_dilate_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * seam_dilate_px + 1, 2 * seam_dilate_px + 1)
        )
        paste_dil = cv2.dilate(paste_region, ker, iterations=1)
        prev_dil = cv2.dilate(prev_mask_bin, ker, iterations=1)
    else:
        paste_dil = paste_region
        prev_dil = prev_mask_bin
    edit_mask = ((paste_dil > 0) | (prev_dil > 0)).astype(np.uint8)

    return base, edit_mask


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

_STRATEGIES: Dict[str, Callable[..., Optional[Tuple[np.ndarray, np.ndarray]]]] = {
    "propainter":    create_insert_base_propainter,
    "poisson_hifi":  _decoy.create_decoy_base_frame_hifi,
    "poisson_basic": _decoy.create_decoy_base_frame,
}


def list_strategies() -> Dict[str, str]:
    """Return {name: one-line description} for introspection / CLI help."""
    return {
        "propainter":    "Flow-guided recurrent inpainting (ICCV'23). "
                         "Primary choice for LPIPS <= 0.10 target.",
        "poisson_hifi":  "Pilot-B Poisson clone + true-region inpaint; "
                         "LPIPS ~0.13-0.14 floor (Pilot A/B/C 2026-04-21).",
        "poisson_basic": "Original Poisson clone only; used by pre-hifi runs.",
    }


def create_insert_base(
    strategy: str,
    frame_prev: np.ndarray,
    frame_after: np.ndarray,
    mask_prev: Optional[np.ndarray],
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
    **kwargs,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Dispatch to the selected insert-base strategy.

    The strategy argument controls which realization is used; all
    realizations share the `(base, edit_mask)` contract. Extra kwargs are
    passed through to the strategy function (e.g. `seam_dilate_px`,
    `safety_margin`, `config` for propainter).

    Raises:
        ValueError on unknown strategy.
        InstallationError when strategy="propainter" but ProPainter is
            not installed on this machine.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown insert-base strategy '{strategy}'. "
            f"Available: {list(_STRATEGIES)}"
        )

    # poisson_hifi signature differs slightly: mask_prev is a kwarg.
    if strategy == "poisson_hifi":
        return _STRATEGIES[strategy](
            frame_prev=frame_prev,
            frame_after=frame_after,
            mask_after=mask_after,
            decoy_offset=decoy_offset,
            mask_prev=mask_prev,
            **kwargs,
        )
    if strategy == "poisson_basic":
        # poisson_basic has no frame_prev / mask_prev; drop them.
        return _STRATEGIES[strategy](
            frame_after=frame_after,
            mask_after=mask_after,
            decoy_offset=decoy_offset,
            **kwargs,
        )
    return _STRATEGIES[strategy](
        frame_prev=frame_prev,
        frame_after=frame_after,
        mask_prev=mask_prev,
        mask_after=mask_after,
        decoy_offset=decoy_offset,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test: strategy table, availability probe, and Poisson fallback
    # dispatch work without any external deps.
    print("Strategies available in the dispatcher:")
    for name, desc in list_strategies().items():
        print(f"  [{name}] {desc}")

    cfg = ProPainterConfig().resolve()
    print(f"\nProPainter root (resolved): {cfg.root}")
    print(f"  available: {is_propainter_available(cfg)}")

    if not is_propainter_available(cfg):
        print("\nInstallation instructions:")
        print(_install_instructions(cfg))
