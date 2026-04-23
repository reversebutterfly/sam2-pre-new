"""Real SAM2.1 + LPIPS + SSIM wiring for VADI on Pro 6000.

Provides non-stub implementations of the `clean_pass_fn` and `forward_fn`
contracts declared in `scripts/run_vadi_pilot.build_pilot_adapters`'s
docstring:

    clean_pass_fn(x_clean, prompt_mask) -> CleanPassOutput
        pseudo_masks[t]    = sigmoid(pred_logits_t) at video resolution, soft [0,1]
        confidences[t]     = sigmoid(object_score_logits_t) * mean(pseudo_mask_t > 0.5)
        hiera_features[t]  = detached last-level backbone_fpn tensor (per-frame)

    forward_fn(x_processed, return_at) -> {int: Tensor[H_vid, W_vid]}
        Differentiable SAM2 forward over `x_processed` (REQUIRES_GRAD). Returns
        pred_logits at video resolution for every frame id in `return_at`.

Design notes
------------
1. We reuse SAM2.1's `forward_image` + `_prepare_backbone_features` +
   `track_step` directly (same pattern as `memshield/sam2_forward_adapter.py`),
   bypassing `SAM2VideoPredictor.propagate_in_video` + `init_state` because
   those are decorated with `@torch.inference_mode()` which would block
   autograd from reaching `x_processed` during PGD.
2. VADI does NOT use the prefix/eval-window split that `SAM2VideoAdapter`
   was built for (v2 bank-poisoning regime); we process every frame in
   `[0, T_proc)` identically and return logits at arbitrary requested ids.
3. The SAM2 predictor is built ONCE by `build_pilot_adapters` and shared
   across all (clip, config) pairs in a pilot run — its parameters are
   frozen (`requires_grad_(False)`) so only the input perturbations carry
   grad.
4. bf16 autocast is kept ON (SAM2 was trained in bf16; fp32 inference
   costs ~30% more VRAM on Blackwell with no accuracy change for our
   purpose). Forward outputs are cast back to float32 before returning so
   downstream losses (LPIPS, SSIM, margin) see stable precision.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ImageNet normalization constants (SAM2 default — matches
# `memshield/sam2_forward_adapter.py` and `sam2/utils/misc.py`).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared per-frame preprocessing helpers
# ---------------------------------------------------------------------------


def _to_sam2_input(
    frame_hwc_01: Tensor,
    image_size: int,
    img_mean: Tensor,
    img_std: Tensor,
) -> Tensor:
    """Convert `[N, H_vid, W_vid, 3]` in `[0, 1]` to SAM2's
    `[N, 3, image_size, image_size]` ImageNet-normalized input."""
    if frame_hwc_01.dim() != 4 or frame_hwc_01.shape[-1] != 3:
        raise ValueError(
            f"expected [N,H,W,3]; got {tuple(frame_hwc_01.shape)}")
    img = frame_hwc_01.permute(0, 3, 1, 2).contiguous()
    img = F.interpolate(
        img, size=(image_size, image_size),
        mode="bilinear", align_corners=False,
    )
    return (img - img_mean) / img_std


def _prepare_first_frame_mask(
    mask_video_res: np.ndarray,
    image_size: int,
    device: torch.device,
) -> Tensor:
    """Binarize + resize a video-res mask to
    `[1, 1, image_size, image_size]` float (antialias bilinear + 0.5
    threshold) matching SAM2's `add_new_mask` path."""
    if mask_video_res.ndim != 2:
        raise ValueError(
            f"first-frame mask must be 2D [H, W]; got {mask_video_res.shape}")
    m = torch.from_numpy(mask_video_res).to(device).float()
    m = m.unsqueeze(0).unsqueeze(0)
    if m.shape[-2] != image_size or m.shape[-1] != image_size:
        m = F.interpolate(
            m, size=(image_size, image_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
    return (m >= 0.5).float()


def _low_res_to_video_res(
    pred_masks_low: Tensor, video_H: int, video_W: int,
) -> Tensor:
    """SAM2 returns pred_masks at ~256x256 (its mask-decoder resolution).
    Upsample back to `[H_vid, W_vid]`, squeezing batch + channel dims."""
    if pred_masks_low.dim() != 4:
        raise ValueError(
            f"pred_masks must be [B, 1, H, W]; got "
            f"{tuple(pred_masks_low.shape)}")
    up = F.interpolate(
        pred_masks_low, size=(video_H, video_W),
        mode="bilinear", align_corners=False,
    )
    return up[0, 0]


# ---------------------------------------------------------------------------
# Clean pass — builds CleanPassOutput for the vulnerability scorer + decoy
# ---------------------------------------------------------------------------


@torch.no_grad()
def clean_pass_vadi(
    predictor,
    x_clean: Tensor,
    prompt_mask: np.ndarray,
    device: torch.device,
    *,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    """Run clean SAM2.1 on `x_clean` and collect the per-frame artifacts
    VADI's vulnerability scorer + decoy-mask builder need.

    Args:
        predictor: a `SAM2VideoPredictor` loaded with weights and set to
            `.eval()`; parameters should be frozen (`requires_grad_(False)`)
            though this function itself runs under `@torch.no_grad()`.
        x_clean: `[T, H_vid, W_vid, 3]` float in `[0, 1]`. Moved to
            `device` internally (no-op if already there).
        prompt_mask: `[H_vid, W_vid]` uint8 binary first-frame mask.
        device: torch device the predictor lives on.
        autocast_dtype: bf16 by default; pass `None` to force fp32.

    Returns:
        `scripts.run_vadi.CleanPassOutput` with:
          - pseudo_masks: `List[np.ndarray[H_vid, W_vid] float32]` — soft.
          - confidences: `np.ndarray[T] float32` — per-frame object score
            gate, = `sigmoid(object_score_logits) * mean(pseudo_mask > 0.5)`.
            For object-score-missing configs, falls back to the foreground
            area fraction alone (object_score term = 1.0).
          - hiera_features: `List[Tensor]` of detached last-level
            `backbone_fpn` tensors (kept on GPU; vulnerability scorer
            duck-types torch tensors via `detach().cpu().numpy()`).
    """
    # Lazy import — this module is safe to import on hosts without the
    # scripts/ package available.
    from scripts.run_vadi import CleanPassOutput

    if x_clean.dim() != 4 or x_clean.shape[-1] != 3:
        raise ValueError(
            f"x_clean must be [T, H, W, 3]; got {tuple(x_clean.shape)}")
    T = int(x_clean.shape[0])
    H_vid = int(x_clean.shape[1])
    W_vid = int(x_clean.shape[2])
    image_size = int(predictor.image_size)

    img_mean = torch.tensor(
        IMAGENET_MEAN, device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    img_std = torch.tensor(
        IMAGENET_STD, device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)

    mask_inputs_f0 = _prepare_first_frame_mask(prompt_mask, image_size, device)
    x_dev = x_clean.to(device) if x_clean.device != device else x_clean

    obj_output_dict: Dict[str, Dict[int, Dict]] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    pseudo_masks: List[np.ndarray] = []
    confidences: List[float] = []
    hiera_features: List[Tensor] = []

    if autocast_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype,
        )
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        for fid in range(T):
            frame = x_dev[fid:fid + 1]                              # [1, H, W, 3]
            img_norm = _to_sam2_input(frame, image_size, img_mean, img_std)
            backbone_out = predictor.forward_image(img_norm)
            _, vision_feats, vision_pos, feat_sizes = \
                predictor._prepare_backbone_features(backbone_out)

            is_init = (fid == 0)
            current_out = predictor.track_step(
                frame_idx=fid,
                is_init_cond_frame=is_init,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=mask_inputs_f0 if is_init else None,
                output_dict=obj_output_dict,
                num_frames=T,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            if is_init:
                obj_output_dict["cond_frame_outputs"][fid] = current_out
            else:
                obj_output_dict["non_cond_frame_outputs"][fid] = current_out

            # Pseudo-mask: upsample low-res logits → video-res, sigmoid, detach.
            pred_vid_logits = _low_res_to_video_res(
                current_out["pred_masks"], H_vid, W_vid,
            ).float()
            soft = pred_vid_logits.sigmoid()
            pseudo_masks.append(
                soft.detach().cpu().numpy().astype(np.float32))

            # Confidence gate.
            obj_logits = current_out.get("object_score_logits")
            if obj_logits is not None:
                obj_prob = float(
                    torch.sigmoid(obj_logits.float()).mean().item())
            else:
                obj_prob = 1.0
            fg_frac = float((soft > 0.5).float().mean().item())
            confidences.append(obj_prob * fg_frac)

            # Hiera feature (last level = deepest / lowest spatial resolution).
            # Detached for the vulnerability scorer; kept on GPU since the
            # scorer duck-types torch tensors and GPU->CPU happens there.
            hiera_last = backbone_out["backbone_fpn"][-1].detach()
            hiera_features.append(hiera_last)

    return CleanPassOutput(
        pseudo_masks=pseudo_masks,
        confidences=np.asarray(confidences, dtype=np.float32),
        hiera_features=hiera_features,
    )


@torch.no_grad()
def sam2_eval_pseudo_masks(
    predictor,
    video: Tensor,
    prompt_mask: np.ndarray,
    device: torch.device,
    *,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> List[np.ndarray]:
    """Lightweight no-grad SAM2 forward for EVALUATION of already-processed
    video (clean-processed baseline OR exported uint8 attack artifact).

    Differs from `clean_pass_vadi` in that it only returns per-frame
    HARD-thresholded binary masks at video resolution — no confidence,
    no hiera features, no CleanPassOutput. Used by
    `eval_exported_j_drop` in `scripts/run_vadi.py` to re-measure attack
    effectiveness on delivered bytes (codex R1 Fix 2, 2026-04-23).

    Args:
        predictor: SAM2 VideoPredictor (same one used elsewhere).
        video: `[T, H_vid, W_vid, 3]` float in `[0, 1]`. Already
            interleaved — contains whatever frames the caller wants SAM2
            to track (originals + inserts, or just originals, or exported
            uint8 reloaded).
        prompt_mask: `[H_vid, W_vid]` uint8 binary first-frame prompt.
        device: predictor's device.

    Returns:
        `List[np.ndarray]` of length `T`, each element `[H_vid, W_vid]`
        uint8 binary (foreground = 1) — suitable for jaccard() against
        a pseudo-ground-truth mask sequence.
    """
    if video.dim() != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"video must be [T, H, W, 3]; got {tuple(video.shape)}")
    T, H_vid, W_vid = (
        int(video.shape[0]), int(video.shape[1]), int(video.shape[2]),
    )
    image_size = int(predictor.image_size)

    img_mean = torch.tensor(
        IMAGENET_MEAN, device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    img_std = torch.tensor(
        IMAGENET_STD, device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)

    mask_inputs_f0 = _prepare_first_frame_mask(prompt_mask, image_size, device)
    v_dev = video.to(device) if video.device != device else video

    obj_output_dict: Dict[str, Dict[int, Dict]] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    if autocast_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype,
        )
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    out: List[np.ndarray] = []
    with autocast_ctx:
        for fid in range(T):
            frame = v_dev[fid:fid + 1]
            img_norm = _to_sam2_input(frame, image_size, img_mean, img_std)
            backbone_out = predictor.forward_image(img_norm)
            _, vision_feats, vision_pos, feat_sizes = \
                predictor._prepare_backbone_features(backbone_out)

            is_init = (fid == 0)
            current_out = predictor.track_step(
                frame_idx=fid,
                is_init_cond_frame=is_init,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=mask_inputs_f0 if is_init else None,
                output_dict=obj_output_dict,
                num_frames=T,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            if is_init:
                obj_output_dict["cond_frame_outputs"][fid] = current_out
            else:
                obj_output_dict["non_cond_frame_outputs"][fid] = current_out

            pred_vid_logits = _low_res_to_video_res(
                current_out["pred_masks"], H_vid, W_vid,
            ).float()
            hard = (pred_vid_logits.sigmoid() > 0.5).to(torch.uint8)
            out.append(hard.detach().cpu().numpy())
    return out


# ---------------------------------------------------------------------------
# Differentiable forward — the PGD-time entry point
# ---------------------------------------------------------------------------


class VADIForwardFn:
    """Per-clip differentiable SAM2 forward that `run_vadi_pgd` calls on
    every optimization step with a fresh `processed` tensor.

    Constructed once per (clip, W) pairing by
    `forward_fn_builder_factory(...)(...)`. Holds only stateless per-clip
    metadata (prompt mask, video resolution, preprocessing constants); the
    SAM2 predictor is passed in by reference and shared across all
    forward-fn instances in the run.

    Autograd contract:
      - `processed` MUST have `requires_grad=True` (caller manages leaves).
      - Predictor weights MUST be frozen by caller
        (`for p in predictor.parameters(): p.requires_grad_(False)`)
        otherwise the graph will retain per-parameter gradients, wasting
        ~1GB VRAM for Tiny over 22 frames.
      - Returns float32 tensors even under bf16 autocast — downstream
        losses expect fp32 stability.

    Frame-0 caveat (SAM2.1 mask-input shortcut):
      SAM2.1 tiny's config sets `use_mask_input_as_output_without_sam=True`.
      For the conditioning frame (fid=0, `is_init_cond_frame=True`), this
      makes `current_out["pred_masks"]` a DIRECT function of the prompt
      mask input — not the perturbed pixels. So `out[0]` (if fid=0 ∈
      `return_at`) carries no gradient w.r.t. `processed[0]`.

      VADI's PGD still optimizes `δ_0` correctly because the attack signal
      for δ_0 flows through a DIFFERENT pathway: processed[0] → Hiera
      backbone → memory encoder → `cond_frame_outputs[0]` → cross-attention
      at frames t≥1 → `out[1..T]`. So the margin loss aggregated over
      `t ≥ 1` yields gradient for δ_0. The frame-0 margin term itself is
      effectively zero-gradient w.r.t. δ_0 and saturates (prompt mask ≈
      pseudo-mask by construction → mu_true≫mu_decoy → softplus ≈ 0). This
      is acceptable (no signal corruption) but worth knowing if debugging
      why a frame-0 margin log-line stays flat across PGD steps.
    """

    def __init__(
        self,
        predictor,
        prompt_mask: np.ndarray,
        video_H: int,
        video_W: int,
        device: torch.device,
        *,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        # Video / mask resolution must match — if `prompt_mask` and later
        # `processed` disagree on H×W we would silently bilinear-resize into
        # SAM2's image_size space and return logits at a mismatched spatial
        # shape. `vadi_loss` enforces strict shape equality with
        # `m_hat_true_by_t` so a mismatch would blow up at loss time, but a
        # clear error here is easier to diagnose.
        if prompt_mask.ndim != 2:
            raise ValueError(
                f"prompt_mask must be 2D [H, W]; got {prompt_mask.shape}")
        if prompt_mask.shape != (int(video_H), int(video_W)):
            raise ValueError(
                f"prompt_mask shape {prompt_mask.shape} does not match "
                f"video resolution ({video_H}, {video_W}). The pilot's "
                f"clip_loader emits both from the same source — a mismatch "
                f"here indicates the prompt mask came from a different clip "
                f"or an aspect-ratio-changing transform.")
        self.predictor = predictor
        self.device = device
        self.video_H = int(video_H)
        self.video_W = int(video_W)
        self.image_size = int(predictor.image_size)
        self.autocast_dtype = autocast_dtype
        self._img_mean = torch.tensor(
            IMAGENET_MEAN, device=device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._img_std = torch.tensor(
            IMAGENET_STD, device=device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._mask_inputs_f0 = _prepare_first_frame_mask(
            prompt_mask, self.image_size, device,
        )

    def __call__(
        self,
        processed: Tensor,
        return_at: Iterable[int],
    ) -> Dict[int, Tensor]:
        """Forward SAM2 frame-by-frame on `processed`; return pred_logits
        at video resolution for every id in `return_at`.

        Args:
            processed: `[T_proc, H_vid, W_vid, 3]` float in `[0, 1]`. Typical
                call from `run_vadi_pgd` has `processed.requires_grad=True`
                (leaves δ and ν are routed through `build_processed`).
            return_at: iterable of ints in `[0, T_proc)`. Duplicates and
                out-of-order are accepted; indices outside the range raise.

        Returns:
            `{t: pred_logits[H_vid, W_vid]}` float32 tensor per requested id.
            Autograd flows from each returned logit back to `processed[t']`
            for all `t' ≤ t` through SAM2's Hiera + memory_attention stack.
        """
        if processed.dim() != 4 or processed.shape[-1] != 3:
            raise ValueError(
                f"processed must be [T, H, W, 3]; got "
                f"{tuple(processed.shape)}")
        if int(processed.shape[1]) != self.video_H \
                or int(processed.shape[2]) != self.video_W:
            raise ValueError(
                f"processed spatial shape ({processed.shape[1]}, "
                f"{processed.shape[2]}) != VADIForwardFn init "
                f"({self.video_H}, {self.video_W})")
        T_proc = int(processed.shape[0])
        return_set = {int(t) for t in return_at}
        bad = [t for t in return_set if not (0 <= t < T_proc)]
        if bad:
            raise ValueError(
                f"return_at ids out of [0, {T_proc}): {sorted(bad)}")

        obj_output_dict: Dict[str, Dict[int, Dict]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        out: Dict[int, Tensor] = {}

        if self.autocast_dtype is not None and self.device.type == "cuda":
            autocast_ctx = torch.amp.autocast(
                device_type="cuda", dtype=self.autocast_dtype,
            )
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            for fid in range(T_proc):
                frame = processed[fid:fid + 1]                      # [1, H, W, 3]
                img_norm = _to_sam2_input(
                    frame, self.image_size, self._img_mean, self._img_std,
                )
                backbone_out = self.predictor.forward_image(img_norm)
                _, vision_feats, vision_pos, feat_sizes = \
                    self.predictor._prepare_backbone_features(backbone_out)

                is_init = (fid == 0)
                current_out = self.predictor.track_step(
                    frame_idx=fid,
                    is_init_cond_frame=is_init,
                    current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=self._mask_inputs_f0 if is_init else None,
                    output_dict=obj_output_dict,
                    num_frames=T_proc,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=None,
                )
                if is_init:
                    obj_output_dict["cond_frame_outputs"][fid] = current_out
                else:
                    obj_output_dict["non_cond_frame_outputs"][fid] = current_out

                if fid in return_set:
                    pred_vid = _low_res_to_video_res(
                        current_out["pred_masks"],
                        self.video_H, self.video_W,
                    )
                    out[fid] = pred_vid.float()

        missing = return_set - set(out.keys())
        if missing:
            raise RuntimeError(
                f"VADIForwardFn: failed to fill return_at slots {sorted(missing)}")
        return out


# ---------------------------------------------------------------------------
# High-level factory for build_pilot_adapters / build_restoration_adapters
# ---------------------------------------------------------------------------


def build_sam2_lpips_ssim(
    checkpoint_path: str,
    device: torch.device,
    *,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
):
    """Build a shared (predictor, lpips_fn, ssim_fn) trio.

    Predictor is loaded once per pilot / main-table run and parameters are
    frozen. LPIPS comes from `memshield.run_pilot_r002.build_lpips_fn`
    (alex net, ImageNet-init) and SSIM from
    `memshield.losses.differentiable_ssim`.

    Contracts:
      lpips_fn(x[H, W, 3] ∈ [0,1], y[H, W, 3] ∈ [0,1]) -> scalar Tensor.
      ssim_fn(x[1, 3, H, W] ∈ [0,1], y[1, 3, H, W] ∈ [0,1]) -> scalar Tensor.
    """
    from sam2.build_sam import build_sam2_video_predictor              # noqa: WPS433
    from memshield.run_pilot_r002 import build_lpips_fn                # noqa: WPS433
    from memshield.losses import differentiable_ssim                   # noqa: WPS433

    predictor = build_sam2_video_predictor(
        sam2_config, str(checkpoint_path), device=str(device),
    )
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    lpips_fn = build_lpips_fn(str(device))

    def ssim_fn(x: Tensor, y: Tensor) -> Tensor:
        return differentiable_ssim(x, y)

    return predictor, lpips_fn, ssim_fn


# ---------------------------------------------------------------------------
# Self-check (import-only; real SAM2 smoke test lives on Pro 6000)
# ---------------------------------------------------------------------------


def _selfcheck() -> None:
    """Lightweight self-test: verify imports + preprocessing helpers without
    needing SAM2 weights or LPIPS. Real end-to-end validation happens on
    Pro 6000 via `python scripts/run_vadi_pilot.py --dry-run` (dry-run
    skips real adapters) followed by the actual pilot launch.
    """
    # Preprocessing helpers are pure-tensor and do not depend on SAM2.
    device = torch.device("cpu")
    img_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    img_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    # _to_sam2_input: [N, H, W, 3] → [N, 3, image_size, image_size].
    frame = torch.rand(1, 480, 854, 3)
    out = _to_sam2_input(frame, image_size=1024,
                         img_mean=img_mean, img_std=img_std)
    assert out.shape == (1, 3, 1024, 1024), f"got {out.shape}"

    # _prepare_first_frame_mask: [H, W] u8 → [1, 1, image_size, image_size] binarized.
    mask = np.random.randint(0, 2, size=(480, 854), dtype=np.uint8)
    m = _prepare_first_frame_mask(mask, image_size=1024, device=device)
    assert m.shape == (1, 1, 1024, 1024), f"got {m.shape}"
    assert torch.all((m == 0) | (m == 1)), "mask should be binary"

    # _low_res_to_video_res: [1, 1, 256, 256] → [480, 854].
    low = torch.randn(1, 1, 256, 256)
    vid = _low_res_to_video_res(low, video_H=480, video_W=854)
    assert vid.shape == (480, 854), f"got {vid.shape}"

    # VADIForwardFn / clean_pass_vadi both require a real SAM2 predictor
    # (and lpips package) — not runnable here. The API-check below just
    # confirms symbols are importable.
    assert callable(clean_pass_vadi)
    assert hasattr(VADIForwardFn, "__call__")
    assert callable(build_sam2_lpips_ssim)

    print("  vadi_sam2_wiring imports OK (preprocessing helpers validated)")
    print("  (real end-to-end smoke: python scripts/run_vadi_pilot.py "
          "--davis-root .../davis --checkpoint .../sam2.1_hiera_tiny.pt)")


if __name__ == "__main__":
    _selfcheck()
