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
    half_precision: bool = False  # opt in after one clean comparison run


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
    """Return True if ProPainter's repo and ALL three checkpoints exist.

    Does NOT import ProPainter (that would trigger heavy GPU deps at import
    time and fail on missing mmcv/basicsr even when we only want to probe).
    We require all three checkpoints (main / RAFT / recurrent-flow) because
    `_load_propainter` unconditionally instantiates all three; reporting
    True without the recurrent-flow weight would let the first real call
    die with an opaque checkpoint-load error.
    """
    cfg = (config or ProPainterConfig()).resolve()
    if not os.path.isdir(cfg.root):
        return False
    if not os.path.isfile(cfg.checkpoint):
        return False
    if not os.path.isfile(cfg.raft_checkpoint):
        return False
    if not os.path.isfile(cfg.recurrent_flow_checkpoint):
        return False
    if not os.path.isdir(os.path.join(cfg.root, "model")):
        return False
    return True


# Pinned ProPainter source for reproducibility. Update these together if
# we ever re-install: commit, checkpoint URLs, and SHA256 go in 3b log.
PROPAINTER_COMMIT = "main@e870e79"  # Verified 2026-04-22: `v0.1.0` tag is
#   an empty scaffold; actual code/weights live on main at HEAD=e870e79.
#   The weight URLs use the `v0.1.0` GitHub release as a CDN path, but the
#   code must be at main; we thus pin the code commit explicitly.
PROPAINTER_WEIGHT_SHAS: Dict[str, str] = {
    # Verified 2026-04-22 after download via sha256sum.
    "ProPainter.pth":
        "12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591",
    "raft-things.pth":
        "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    "recurrent_flow_completion.pth":
        "22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283",
}


def _install_instructions(cfg: ProPainterConfig) -> str:
    return (
        "ProPainter is not installed. Expected to find:\n"
        f"    repo:       {cfg.root}\n"
        f"    checkpoint: {cfg.checkpoint}\n"
        f"    raft:       {cfg.raft_checkpoint}\n"
        f"    rec. flow:  {cfg.recurrent_flow_checkpoint}\n"
        "\n"
        "Install (on Pro 6000, once user has approved). Pinned to "
        f"{PROPAINTER_COMMIT} for reproducibility:\n"
        "    cd ~\n"
        "    git clone https://github.com/sczhou/ProPainter\n"
        f"    cd ~/ProPainter && git checkout {PROPAINTER_COMMIT}\n"
        "    bash scripts/download_weights.sh\n"
        "    # After download, record SHA256 of each .pth and update\n"
        "    # PROPAINTER_WEIGHT_SHAS in memshield/propainter_base.py:\n"
        "    sha256sum weights/*.pth\n"
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


# Lazy-loaded cache keyed by resolved config so (a) a different device /
# precision / root builds a fresh instance instead of silently reusing the
# first one, and (b) a single config re-used across calls costs only one
# load. Key is a tuple of the load-relevant fields.
_PROPAINTER_CACHE: Dict[Tuple, Dict[str, object]] = {}


def _cfg_key(cfg: ProPainterConfig) -> Tuple:
    """Tuple of config fields that affect how the models are built."""
    return (
        cfg.root, cfg.checkpoint, cfg.raft_checkpoint,
        cfg.recurrent_flow_checkpoint, cfg.device, cfg.half_precision,
    )


def _load_propainter(cfg: ProPainterConfig) -> Dict[str, object]:
    """Import and instantiate ProPainter. Raises InstallationError if absent.

    Returns a dict `{"raft":..., "flow_complete":..., "inpainter":...}`.
    Cached by `_cfg_key(cfg)` so repeated calls with the same config are
    cheap and different configs build their own instances.
    """
    key = _cfg_key(cfg)
    cached = _PROPAINTER_CACHE.get(key)
    if cached is not None:
        return cached

    if not is_propainter_available(cfg):
        raise InstallationError(_install_instructions(cfg))

    import sys
    import torch  # type: ignore

    if cfg.root not in sys.path:
        sys.path.insert(0, cfg.root)

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

    bundle = {"raft": raft, "flow_complete": flow_complete,
              "inpainter": inpainter, "cfg": cfg}
    _PROPAINTER_CACHE[key] = bundle
    return bundle


def _pad_to_multiple_of_8(frames: np.ndarray, masks: np.ndarray):
    """Pad frames/masks so H,W are both multiples of 8 (ProPainter constraint).

    Returns (padded_frames, padded_masks, (top, bottom, left, right)) where
    the offsets let the caller crop back to the original size.
    """
    T, H, W, C = frames.shape
    pad_h = (-H) % 8
    pad_w = (-W) % 8
    if pad_h == 0 and pad_w == 0:
        return frames, masks, (0, 0, 0, 0)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    frames_p = np.pad(frames, ((0, 0), (top, bottom), (left, right), (0, 0)),
                      mode="edge")
    masks_p = np.pad(masks, ((0, 0), (top, bottom), (left, right)),
                     mode="constant", constant_values=0)
    return frames_p, masks_p, (top, bottom, left, right)


def _propainter_fill(
    frames,
    masks,
    target_idx: int,
    cfg: ProPainterConfig,
    raft_iters: int = 20,
) -> np.ndarray:
    """Run ProPainter on a video clip and return the filled target frame.

    Sequence-based contract that mirrors ProPainter's intended inference
    API (see inference_propainter.py in sczhou/ProPainter @ main).

    Args:
        frames: Length-T sequence of `[H, W, 3]` uint8 RGB frames. Accepts
            np.ndarray of shape (T, H, W, 3) or a list of (H, W, 3) arrays.
        masks: Length-T sequence of `[H, W]` uint8 masks (1 = fill, 0 =
            keep). Accepts np.ndarray of shape (T, H, W) or a list.
        target_idx: Index into `frames` whose filled version is returned.
        cfg: Resolved ProPainterConfig.
        raft_iters: RAFT iteration count (ProPainter default = 20).

    Returns:
        Filled version of `frames[target_idx]` as `[H, W, 3]` uint8.

    Raises:
        InstallationError: if ProPainter is not installed.
        ValueError: on shape / target_idx / length mismatches.

    Pipeline (from sczhou/ProPainter inference_propainter.py main loop):
      1. RAFT bidirectional flows on [B, T, 3, H, W] in [-1, 1].
      2. flow_complete.forward_bidirect_flow + .combine_flow to fill flows
         under the mask.
      3. model.img_propagation to produce masked-region warps via nearest-
         neighbor flow sampling.
      4. model(updated_frames, pred_flows_bi, masks, updated_masks, T) to
         run the transformer refinement over all T frames.
      5. Return target frame, mapped back to uint8 [H, W, 3].

    H and W must be multiples of 8 for ProPainter. We pad with edge/zeros
    and crop back before returning.
    """
    import torch
    import torch.nn.functional as F  # noqa: F401  (may be used by callers)

    frames_arr = np.asarray(frames)
    masks_arr = np.asarray(masks)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (T, H, W, 3), got {frames_arr.shape}")
    if masks_arr.ndim != 3:
        raise ValueError(
            f"masks must have shape (T, H, W), got {masks_arr.shape}")
    T, H, W, _ = frames_arr.shape
    if masks_arr.shape != (T, H, W):
        raise ValueError(
            f"masks shape {masks_arr.shape} inconsistent with frames "
            f"(expected {(T, H, W)})")
    if not (0 <= target_idx < T):
        raise ValueError(f"target_idx={target_idx} out of range for T={T}")

    bundle = _load_propainter(cfg)   # may raise InstallationError
    raft, flow_complete, inpainter = (
        bundle["raft"], bundle["flow_complete"], bundle["inpainter"])
    device = torch.device(cfg.device)

    padded_frames, padded_masks, (top, bottom, left, right) = \
        _pad_to_multiple_of_8(frames_arr, masks_arr)
    Hp, Wp = padded_frames.shape[1:3]

    frames_t = torch.from_numpy(padded_frames).to(device).float() / 255.0
    frames_t = frames_t.permute(0, 3, 1, 2).unsqueeze(0)        # [1, T, 3, H, W]
    frames_t = frames_t * 2.0 - 1.0                              # to [-1, 1]
    masks_t = torch.from_numpy((padded_masks > 0).astype(np.float32)).to(device)
    masks_t = masks_t.unsqueeze(0).unsqueeze(2)                  # [1, T, 1, H, W]

    if cfg.half_precision:
        frames_t = frames_t.half()
        masks_t = masks_t.half()

    with torch.no_grad():
        flows_f, flows_b = raft(frames_t, iters=raft_iters)
        gt_flows_bi = (flows_f, flows_b)
        if cfg.half_precision:
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
        pred_flows_bi, _ = flow_complete.forward_bidirect_flow(
            gt_flows_bi, masks_t)
        pred_flows_bi = flow_complete.combine_flow(
            gt_flows_bi, pred_flows_bi, masks_t)

        masked = frames_t * (1 - masks_t)
        prop, upd = inpainter.img_propagation(
            masked, pred_flows_bi, masks_t, "nearest")
        b, t, _, _, _ = masks_t.size()
        upd_frames = frames_t * (1 - masks_t) + prop.view(b, t, 3, Hp, Wp) * masks_t
        upd_masks = upd.view(b, t, 1, Hp, Wp)

        pred_img = inpainter(upd_frames, pred_flows_bi, masks_t, upd_masks, T)
        # pred_img: [1, T, 3, Hp, Wp] in roughly [-1, 1]
        pred_img = (pred_img + 1.0) / 2.0
        pred_img = pred_img.clamp(0.0, 1.0)

    frame = pred_img[0, target_idx].permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255.0).round().astype(np.uint8)

    # Crop back to original H, W
    if top or bottom or left or right:
        frame = frame[top:Hp - bottom, left:Wp - right, :]

    # Composite: keep original pixels where mask == 0, use predicted pixels
    # where mask == 1. This matches inference_propainter.py's final step and
    # avoids ProPainter's tendency to slightly shift unmasked pixels.
    orig = frames_arr[target_idx]
    mk = (masks_arr[target_idx] > 0)[..., None].astype(np.uint8)
    frame = frame * mk + orig * (1 - mk)
    return frame.astype(np.uint8)


def create_insert_base_propainter(
    frame_prev: np.ndarray,
    frame_after: np.ndarray,
    mask_prev: Optional[np.ndarray],
    mask_after: np.ndarray,
    decoy_offset: Tuple[int, int],
    seam_dilate_px: int = 5,
    safety_margin: int = 8,
    feather_px: int = 3,
    config: Optional[ProPainterConfig] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ProPainter-based insert base (3-frame middle-slot design).

    Pipeline (3-frame clip formulation per Codex R3 recommendation):

    1. Compute paste region = shifted copy of mask_after by decoy_offset.
       If paste region violates border safety, return None (caller tries
       another offset).
    2. Build a length-3 "video" `[frame_prev, placeholder, frame_after]`
       with per-frame masks `[0, 1, 0]` — ProPainter sees two clean
       neighbors and is asked to fill exactly the middle slot.
    3. Call _propainter_fill(frames, masks, target_idx=1, cfg) to get an
       inpainted middle frame. This is what ProPainter is built for
       (flow-guided recurrent fill from a neighbor window).
    4. On the returned middle frame, alpha-composite the object crop
       (from frame_after[y0:y1, x0:x1]) at the decoy center with a
       `feather_px`-pixel soft alpha to avoid a hard seam.
    5. If the user wants additional true-object suppression signal in
       frame_prev's position, we also modify the crop's edge blending
       using `prev_mask_bin` as support in the edit_mask return.
    6. edit_mask = dilated(paste_region) ∪ dilated(mask_prev). This is
       the region PGD is allowed to perturb downstream.

    Args match `decoy.create_decoy_base_frame_hifi` so it can serve as a
    drop-in replacement; extra kwargs: `feather_px`, `config`.

    Returns:
        (base_frame_uint8, edit_mask_uint8) on success; None on
        border-safety failure.

    Raises:
        InstallationError: when ProPainter is not installed.
        NotImplementedError: until Chunk 3b wires the actual forward.
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

    prev_mask = mask_prev if mask_prev is not None else mask_after
    prev_mask_bin = (prev_mask > 0).astype(np.uint8)
    paste_region = _decoy.shift_mask(mask_ref, dy, dx)
    # Middle-frame fill region: true-obj (at prev position) + paste location.
    middle_mask = ((prev_mask_bin > 0) | (paste_region > 0)).astype(np.uint8)

    # Build 3-frame clip; placeholder can be the average of neighbors — it
    # gets masked out completely and never influences the output.
    placeholder = ((frame_prev.astype(np.int32) + frame_after.astype(np.int32))
                   // 2).astype(np.uint8)
    frames = np.stack([frame_prev, placeholder, frame_after], axis=0)
    masks = np.stack([
        np.zeros((H, W), dtype=np.uint8),
        middle_mask,
        np.zeros((H, W), dtype=np.uint8),
    ], axis=0)

    filled_middle = _propainter_fill(frames, masks, target_idx=1, cfg=cfg)

    # Alpha-composite object crop onto inpainted middle frame with feathered
    # edge so there's no hard seam where the crop meets the fill.
    import cv2  # local import — module-top stays light
    obj_crop = frame_after[y0:y1, x0:x1]
    mask_crop = mask_ref[y0:y1, x0:x1].astype(np.float32)
    if feather_px > 0:
        ker_f = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * feather_px + 1, 2 * feather_px + 1)
        )
        mask_crop_soft = cv2.GaussianBlur(
            cv2.dilate(mask_crop, ker_f, iterations=1),
            (2 * feather_px + 1, 2 * feather_px + 1), 0.0,
        )
        mask_crop_soft = np.clip(mask_crop_soft, 0.0, 1.0)
    else:
        mask_crop_soft = mask_crop

    paste_h, paste_w = mask_crop.shape
    half_h, half_w = paste_h // 2, paste_w // 2
    py0, py1 = cy_obj - half_h, cy_obj - half_h + paste_h
    px0, px1 = cx_obj - half_w, cx_obj - half_w + paste_w
    base = filled_middle.copy()
    alpha = mask_crop_soft[..., None]
    base[py0:py1, px0:px1] = (
        alpha * obj_crop.astype(np.float32)
        + (1.0 - alpha) * base[py0:py1, px0:px1].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    if seam_dilate_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * seam_dilate_px + 1, 2 * seam_dilate_px + 1)
        )
        paste_dil = cv2.dilate(paste_region, ker, iterations=1)
        prev_dil = cv2.dilate(prev_mask_bin, ker, iterations=1)
    else:
        paste_dil, prev_dil = paste_region, prev_mask_bin
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
