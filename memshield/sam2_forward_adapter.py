"""
MemoryShield Chunk 5b-ii: full SAM2 video-predictor adapter.

Implements the `Sam2ForwardFn` protocol from `memshield.optimize_v2` using the
real SAM2.1 VideoPredictor. Couples with Chunk 5b-i's `RuntimeProvenanceHook`
and Chunk 1's `MemAttnProbe` to extract `P_u` on eval / stale frames.

Why a custom propagation loop (not `predictor.propagate_in_video`)
-----------------------------------------------------------------
`SAM2VideoPredictor.propagate_in_video` (and `init_state`, `add_new_mask`,
`_run_single_frame_inference`) are all decorated with `@torch.inference_mode()`.
Inference mode is stricter than `no_grad` — tensors produced inside it are
"inference tensors" that CANNOT participate in autograd afterwards. For PGD we
need gradient flow from the loss on insert / eval logits and `P_u` all the way
back to `state.nu` and `state.delta` through memory_attention + Hiera. So we
bypass those decorated entry points and re-implement a minimal per-frame loop
using un-decorated model-level methods (`forward_image`, `track_step`).

Hiera cache strategy
--------------------
The eval suffix frames are NEVER modified by the optimizer; their pixels are
identical across all PGD steps. So we pre-compute the Hiera backbone output
for the suffix once at setup (mode="clean") and reuse the cached features
during attack steps. The prefix Hiera is always re-run under grad.

Memory conditioning frame
-------------------------
For SAM2.1 (tiny / small / base+ / large) `use_mask_input_as_output_without_sam`
defaults to True — when we feed a mask prompt to an init-cond frame, SAM2 skips
`_prepare_memory_conditioned_features` (no previous memory exists anyway) and
uses the mask directly to build the conditioning frame's output / memory. The
memory encoder still runs when `run_mem_encoder=True`, so the cond frame
contributes a `maskmem_features` entry that future frames can attend to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .mem_attn_probe import MemAttnProbe
from .optimize_v2 import OptimizeConfig, VideoBundle
from .sam2_forward_v2 import RuntimeProvenanceHook


# ---------------------------------------------------------------------------
# ImageNet normalization (SAM2 default, see sam2/utils/misc.py)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Small utility dataclass for the suffix Hiera cache
# ---------------------------------------------------------------------------


@dataclass
class _SuffixCacheEntry:
    """One frame's precomputed backbone_fpn + vision_pos_enc.

    Stored as detached tensors — the suffix pixels do not change across PGD
    steps, so gradients through suffix Hiera are unnecessary. Each tensor is
    kept on the same device as the model so the hot inner loop does not move
    data every step.
    """
    backbone_fpn: List[Tensor]
    vision_pos_enc: List[Tensor]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SAM2VideoAdapter:
    """`Sam2ForwardFn` implementation backed by a real SAM2 VideoPredictor.

    Usage:
        predictor = build_sam2_video_predictor(...)       # as in SAM2 docs
        adapter = SAM2VideoAdapter(predictor, cfg, first_frame_mask_vid_res)
        adapter.prepare_from_clean(bundle)                # pre-fills suffix cache
        # then pass `adapter` (it's callable) to `optimize_unified_v2` as
        # `sam2_forward_fn=adapter`.

    The conditioning-frame mask prompt is fixed for the whole PGD run — it's
    the caller-provided first-frame GT mask. We binarize once and cache the
    resized mask at the model's image_size.
    """

    def __init__(
        self,
        predictor,
        cfg: OptimizeConfig,
        first_frame_mask_video_res: np.ndarray,
        video_H: int,
        video_W: int,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        """
        Args:
            predictor: a `SAM2VideoPredictor` instance already loaded with
                weights. This adapter uses the inner `predictor.model`-style
                methods directly (the predictor IS a SAM2Base subclass).
            cfg: OptimizeConfig.
            first_frame_mask_video_res: [H, W] uint8/bool numpy. The first-
                frame GT mask at video resolution.
            video_H, video_W: original video resolution; needed for the final
                logit upsample back to mask-space.
        """
        self.sam2 = predictor
        self.cfg = cfg
        self.video_H = int(video_H)
        self.video_W = int(video_W)
        self.device = predictor.device
        self.image_size = int(predictor.image_size)
        # bf16 autocast keeps the 22-frame Hiera + memory-attention graph
        # within Blackwell's 96 GB at 1024-res. Set None to force fp32.
        self.autocast_dtype = autocast_dtype

        # Backbone feature grid (memory bank resolution). For SAM2.1 @1024 with
        # stride 16 this is 64x64 → HW_mem = 4096. We read it lazily from the
        # first Hiera forward so we don't hardcode.
        self._H_feat: Optional[int] = None
        self._W_feat: Optional[int] = None

        # Precomputed suffix Hiera cache.
        self._suffix_cache: Dict[int, _SuffixCacheEntry] = {}

        # Cached first-frame mask at image_size, ready for track_step.
        self._mask_inputs_frame0 = self._prepare_first_frame_mask(
            first_frame_mask_video_res,
        )

        # Cached ImageNet mean / std on device (avoid per-step .to()).
        self._img_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32,
                                      device=self.device).view(1, 3, 1, 1)
        self._img_std = torch.tensor(IMAGENET_STD, dtype=torch.float32,
                                     device=self.device).view(1, 3, 1, 1)

        # Probe — layer index defaults to last memory-attention layer.
        self.probe = MemAttnProbe(
            self.sam2.memory_attention,
            layer_indices=None,        # = [last]
            dtype_softmax=torch.float32,
        )

    # -- setup --------------------------------------------------------------

    @torch.no_grad()
    def prepare_from_clean(self, bundle: VideoBundle) -> None:
        """Precompute Hiera for the clean eval suffix.

        Must be called once per video (or per bundle) BEFORE any __call__.
        Uses bundle.frames_orig[T_prefix_orig : T_prefix_orig + eval_window].
        Stores post-conv_s0/conv_s1 backbone_fpn + vision_pos_enc tensors
        detached on self.device.
        """
        cfg = self.cfg
        start = cfg.T_prefix_orig
        stop = cfg.T_prefix_orig + cfg.eval_window_size
        suffix_np = bundle.frames_orig[start:stop]                        # uint8
        if suffix_np.shape[0] != cfg.eval_window_size:
            raise ValueError(
                f"bundle.frames_orig has {bundle.frames_orig.shape[0]} "
                f"frames; need at least {stop} for the eval suffix."
            )
        suffix_t = torch.from_numpy(suffix_np).to(self.device).float() / 255.0

        T_mod = bundle.schedule.T_prefix_mod
        self._suffix_cache = {}
        for i in range(cfg.eval_window_size):
            frame_id_mod = T_mod + i
            img_norm = self._to_sam2_input(suffix_t[i:i + 1])             # [1, 3, image_size, image_size]
            backbone_out = self.sam2.forward_image(img_norm)
            # Detach + clone so the entries survive beyond any outer grad context.
            fpn = [f.detach().clone() for f in backbone_out["backbone_fpn"]]
            pos = [p.detach().clone() for p in backbone_out["vision_pos_enc"]]
            self._suffix_cache[frame_id_mod] = _SuffixCacheEntry(
                backbone_fpn=fpn, vision_pos_enc=pos,
            )

        # Seed feature-grid size from the last level of the first cached entry.
        last_pos = self._suffix_cache[T_mod].vision_pos_enc[-1]
        self._H_feat = int(last_pos.shape[-2])
        self._W_feat = int(last_pos.shape[-1])

    @property
    def HW_mem(self) -> int:
        """Memory-bank slot resolution (= H_feat * W_feat). Set by
        `prepare_from_clean`; must not be queried before."""
        if self._H_feat is None or self._W_feat is None:
            raise RuntimeError(
                "HW_mem queried before prepare_from_clean() ran. Call "
                "adapter.prepare_from_clean(bundle) first.")
        return self._H_feat * self._W_feat

    # -- the Sam2ForwardFn entry point --------------------------------------

    def __call__(
        self,
        modified_video: torch.Tensor,
        mode: str,
        cfg: OptimizeConfig,
        bundle: VideoBundle,
    ) -> Dict[str, object]:
        """Run SAM2 on (modified prefix + clean suffix) and extract the
        outputs the optimizer needs. See `Sam2ForwardFn` docstring.

        `modified_video` is `[T_prefix_mod, H_vid, W_vid, 3]` in [0, 1]. The
        eval suffix (original-time indices T_prefix_orig : T_prefix_orig +
        eval_window_size) is appended internally from bundle.frames_orig, so
        the full modified video processed by SAM2 has length
        `T_prefix_mod + eval_window_size`.
        """
        if mode not in ("attack", "clean"):
            raise ValueError(f"mode must be 'attack' or 'clean', got {mode!r}")
        if self._H_feat is None:
            raise RuntimeError(
                "adapter not set up — call prepare_from_clean(bundle) first.")

        device = modified_video.device
        T_mod = bundle.schedule.T_prefix_mod
        T_suffix = cfg.eval_window_size
        T_full = T_mod + T_suffix

        # Assemble the full video in mod-time: [T_full, H_vid, W_vid, 3].
        suffix_t = self._suffix_as_tensor(bundle).to(device)
        full_video = torch.cat([modified_video, suffix_t], dim=0)

        # Index bookkeeping in mod-time.
        insert_mods_set = set(bundle.schedule.w_positions)
        mod_to_k = {s.m_k: k for k, s in enumerate(bundle.schedule.slots)}
        eval_start = T_mod
        eval_frame_ids = list(range(eval_start, eval_start + T_suffix))
        eval_set = set(eval_frame_ids)
        stale_start = eval_start
        stale_stop = stale_start + cfg.stale_window_size
        stale_set = set(range(stale_start, stale_stop))

        # Build fg_mask_by_frame at feature-grid resolution for the eval &
        # stale frames only (probe returns None for frames not in this map).
        fg_mask_by_frame = self._build_fg_mask_by_frame(
            bundle, eval_frame_ids, device,
        )

        # Per-object output dict (single-object tracking).
        obj_output_dict: Dict[str, Dict[int, Dict]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        insert_logits: List[Optional[Tensor]] = [None] * cfg.K_ins
        eval_logits:   List[Optional[Tensor]] = [None] * cfg.eval_window_size
        P_u_list:      List[Optional[Tensor]] = [None] * cfg.stale_window_size
        pred_masks_list: Optional[List[Tensor]] = [] if mode == "clean" else None

        last_layer_idx = self.probe.layer_indices[-1]

        # Hook is a no-op on is_init_cond_frame (SAM2.1 bypasses the memory
        # path there), but wrapping the whole loop keeps setup / teardown
        # symmetric and lets us log cond_frame_outputs for unit checks.
        self.probe.reset()
        # bf16 autocast — SAM2 was trained in bf16; keeps 22-frame graph in
        # GPU memory. Probe's softmax side-channel is already float32 for
        # numerical stability.
        if self.autocast_dtype is not None:
            autocast_ctx = torch.amp.autocast(
                device_type="cuda", dtype=self.autocast_dtype,
            )
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        hook_ctx = RuntimeProvenanceHook(
            sam2_base=self.sam2,
            insert_frame_ids=insert_mods_set,
            probe=self.probe,
            fg_mask_by_frame=fg_mask_by_frame,
            HW_mem=self.HW_mem,
            has_obj_ptrs=bool(getattr(self.sam2, "use_obj_ptrs_in_encoder", False)),
        )
        with autocast_ctx, hook_ctx, self.probe:
            # --- conditioning frame (fid = 0) ---
            current_out_0 = self._track_single_frame(
                full_video=full_video,
                frame_idx=0,
                is_init_cond_frame=True,
                mask_inputs=self._mask_inputs_frame0,
                obj_output_dict=obj_output_dict,
                num_frames=T_full,
                run_mem_encoder=True,
            )
            obj_output_dict["cond_frame_outputs"][0] = current_out_0
            if mode == "clean":
                pred_masks_list.append(
                    self._low_res_to_video_res(current_out_0["pred_masks"]))

            # --- non-cond frames (fid = 1 .. T_full - 1) ---
            for fid in range(1, T_full):
                # Inform probe of the current frame BEFORE track_step runs
                # memory_attention; hook's _wrapped will overwrite this with
                # the same fid (idempotent) but setting here is defensive.
                self.probe.set_frame(fid)

                current_out = self._track_single_frame(
                    full_video=full_video,
                    frame_idx=fid,
                    is_init_cond_frame=False,
                    mask_inputs=None,
                    obj_output_dict=obj_output_dict,
                    num_frames=T_full,
                    run_mem_encoder=True,
                )
                obj_output_dict["non_cond_frame_outputs"][fid] = current_out

                # Upsample low-res logits to video resolution; [H_vid, W_vid].
                pred_vid = self._low_res_to_video_res(current_out["pred_masks"])

                if fid in insert_mods_set:
                    k = mod_to_k[fid]
                    insert_logits[k] = pred_vid
                if fid in eval_set:
                    u_idx = fid - eval_start
                    eval_logits[u_idx] = pred_vid
                if fid in stale_set:
                    v_idx = fid - stale_start
                    P = self.probe.P_u_by_frame.get(fid, {}).get(last_layer_idx)
                    P_u_list[v_idx] = P
                if mode == "clean":
                    pred_masks_list.append(pred_vid)

        # Sanity: all protocol slots must have been filled (except P_u which
        # is explicitly `Optional` per protocol — None means probe didn't fire
        # for that frame, e.g. empty foreground).
        if any(x is None for x in insert_logits):
            raise RuntimeError(
                f"Not all insert slots filled: "
                f"{[i for i, v in enumerate(insert_logits) if v is None]}")
        if any(x is None for x in eval_logits):
            raise RuntimeError(
                f"Not all eval slots filled: "
                f"{[i for i, v in enumerate(eval_logits) if v is None]}")

        return {
            "insert_logits": insert_logits,
            "eval_logits":   eval_logits,
            "P_u_list":      P_u_list,
            "pred_masks":    pred_masks_list,
        }

    # -- per-frame forward (replaces the decorated _run_single_frame_inference)

    def _track_single_frame(
        self,
        full_video: Tensor,
        frame_idx: int,
        is_init_cond_frame: bool,
        mask_inputs: Optional[Tensor],
        obj_output_dict: Dict,
        num_frames: int,
        run_mem_encoder: bool,
    ) -> Dict[str, Tensor]:
        """One frame's forward — Hiera (or cache) → track_step."""
        vision_feats, vision_pos_embeds, feat_sizes = self._compute_features(
            full_video, frame_idx,
        )
        current_out = self.sam2.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict=obj_output_dict,
            num_frames=num_frames,
            track_in_reverse=False,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=None,
        )
        return current_out

    # -- feature extraction (cached suffix vs. live prefix) -----------------

    def _compute_features(
        self, full_video: Tensor, frame_idx: int,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[int, int]]]:
        """Return (vision_feats [HWxBxC flattened], vision_pos_embeds,
        feat_sizes) for one frame. Uses the suffix cache if available."""
        entry = self._suffix_cache.get(frame_idx)
        if entry is not None:
            backbone_out = {
                "backbone_fpn":   entry.backbone_fpn,
                "vision_pos_enc": entry.vision_pos_enc,
            }
        else:
            img_norm = self._to_sam2_input(full_video[frame_idx:frame_idx + 1])
            backbone_out = self.sam2.forward_image(img_norm)

        # _prepare_backbone_features returns (backbone_out_copy, vision_feats,
        # vision_pos_embeds, feat_sizes). We discard the first.
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.sam2._prepare_backbone_features(backbone_out)
        return vision_feats, vision_pos_embeds, feat_sizes

    # -- input preprocessing -------------------------------------------------

    def _to_sam2_input(self, frame_hwc_01: Tensor) -> Tensor:
        """Convert a [N, H_vid, W_vid, 3] in [0, 1] batch to SAM2's expected
        [N, 3, image_size, image_size] ImageNet-normalized tensor."""
        if frame_hwc_01.dim() != 4 or frame_hwc_01.shape[-1] != 3:
            raise ValueError(
                f"expected [N, H, W, 3], got {tuple(frame_hwc_01.shape)}")
        img = frame_hwc_01.permute(0, 3, 1, 2).contiguous()               # [N, 3, H, W]
        img = F.interpolate(img, size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False)
        img = (img - self._img_mean) / self._img_std
        return img

    def _prepare_first_frame_mask(self, mask_video_res: np.ndarray) -> Tensor:
        """Binarize + resize to [1, 1, image_size, image_size] float
        following `add_new_mask`'s path (antialias bilinear + 0.5 threshold)."""
        if mask_video_res.ndim != 2:
            raise ValueError(
                f"first-frame mask must be 2D [H, W], got "
                f"{mask_video_res.shape}")
        m = torch.from_numpy(mask_video_res).to(self.device).float()
        m = m.unsqueeze(0).unsqueeze(0)                                   # [1, 1, H, W]
        if m.shape[-2] != self.image_size or m.shape[-1] != self.image_size:
            m = F.interpolate(
                m, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            m = (m >= 0.5).float()
        else:
            m = (m >= 0.5).float()
        return m

    # -- helpers -------------------------------------------------------------

    def _suffix_as_tensor(self, bundle: VideoBundle) -> Tensor:
        """Clean eval suffix as [T_suffix, H_vid, W_vid, 3] float in [0, 1]."""
        start = self.cfg.T_prefix_orig
        stop = start + self.cfg.eval_window_size
        arr = bundle.frames_orig[start:stop]
        return torch.from_numpy(arr).to(self.device).float() / 255.0

    def _build_fg_mask_by_frame(
        self, bundle: VideoBundle, eval_frame_ids: List[int], device,
    ) -> Dict[int, Tensor]:
        """Map eval-frame mod-id → 1-D bool tensor of length HW_mem.

        bundle.C_u is a list of [H_vid, W_vid] uint8 masks, one per eval
        frame in original-time order (same order as eval_frame_ids in mod-
        time since we just stream past T_prefix_mod). We downsample with
        nearest interpolation to the feature grid.
        """
        if len(bundle.C_u) < len(eval_frame_ids):
            raise ValueError(
                f"bundle.C_u has {len(bundle.C_u)} entries; need "
                f"{len(eval_frame_ids)} for the eval window.")
        H_f, W_f = self._H_feat, self._W_feat
        out: Dict[int, Tensor] = {}
        for i, fid in enumerate(eval_frame_ids):
            m = bundle.C_u[i]
            if m.ndim != 2:
                raise ValueError(
                    f"C_u[{i}] must be 2D [H, W], got {m.shape}")
            t = torch.from_numpy(m).to(device).float().unsqueeze(0).unsqueeze(0)
            t_ds = F.interpolate(t, size=(H_f, W_f), mode="nearest")
            out[fid] = (t_ds.view(-1) > 0.5)
        return out

    def _low_res_to_video_res(self, pred_masks_low: Tensor) -> Tensor:
        """SAM2 returns `pred_masks` as [B, 1, H_low, W_low] (typically
        256x256). The optimizer's loss stack operates on [H_vid, W_vid]
        logits aligned with `bundle.D_ins / C_ins / C_u`. Bilinear upsample
        back to video resolution; squeeze singleton batch and channel dims."""
        if pred_masks_low.dim() != 4:
            raise ValueError(
                f"pred_masks must be [B, 1, H, W], got "
                f"{tuple(pred_masks_low.shape)}")
        up = F.interpolate(
            pred_masks_low, size=(self.video_H, self.video_W),
            mode="bilinear", align_corners=False,
        )
        return up[0, 0]                                                  # [H_vid, W_vid]


# ---------------------------------------------------------------------------
# Convenience constructor + simple self-check (does not require real weights)
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    """Lightweight static self-check: verify imports and signature shape.

    Does NOT run SAM2. The real smoke test lives in
    `scripts/smoke_5b_ii.py` and needs a GPU + checkpoint.
    """
    # Imports resolve, types exist, public API is reachable.
    assert SAM2VideoAdapter.__call__ is not None
    assert hasattr(SAM2VideoAdapter, "prepare_from_clean")
    assert hasattr(SAM2VideoAdapter, "HW_mem")
    print("  sam2_forward_adapter imports OK")
    print("  (real SAM2 smoke test: python scripts/smoke_5b_ii.py)")


if __name__ == "__main__":
    _selfcheck()
