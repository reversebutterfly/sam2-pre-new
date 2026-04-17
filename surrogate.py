"""
SAM2 Differentiable Surrogate — v2: calls track_step directly.

Key insight from GPT-5.4 review (Round 1, score 4/10):
  - v1 manually reconstructed the tracking path and got it wrong
  - The official SAM2Base.track_step() is NOT wrapped in inference_mode()
  - We can call it directly with a manually managed output_dict
  - This gives us the EXACT official memory pipeline with gradient flow

Changes from v1:
  1. Removed prior_mask propagation (only for same-frame correction)
  2. Separate cond_frame_outputs / non_cond_frame_outputs
  3. Memory encoded from RAW features (inside track_step)
  4. Uses _forward_sam_heads with proper multimask + object scores
  5. No more manual memory_encoder / memory_attention calls
"""
from contextlib import nullcontext
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_interior_prompt(mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the most interior point of a mask via distance transform."""
    mask_u8 = mask_np.astype(np.uint8)
    if mask_u8.sum() == 0:
        h, w = mask_u8.shape
        return (np.array([[w // 2, h // 2]], dtype=np.float32),
                np.array([1], dtype=np.int32))
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (np.array([[float(x), float(y)]], dtype=np.float32),
            np.array([1], dtype=np.int32))


class SAM2Surrogate:
    """SAM2 surrogate v2: thin wrapper around official track_step.

    Instead of manually calling prompt_encoder / mask_decoder / memory_encoder,
    we call SAM2Base.track_step() directly, which uses the exact official
    memory pipeline. We just manage the output_dict state ourselves.
    """

    INPUT_SIZE = 1024

    def __init__(self, checkpoint: str, config: str, device: torch.device):
        from sam2.build_sam import build_sam2
        self.sam2 = build_sam2(config, checkpoint, device=device)
        self.sam2.eval()
        for p in self.sam2.parameters():
            p.requires_grad_(False)
        self.device = device
        self.num_maskmem = getattr(self.sam2, "num_maskmem", 7)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def _encode_backbone(self, x01: torch.Tensor):
        """Encode [1, 3, 1024, 1024] image in [0, 1] to backbone features."""
        x_norm = (x01 - self._mean) / self._std
        backbone_out = self.sam2.forward_image(x_norm)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.sam2._prepare_backbone_features(backbone_out)
        )
        return vision_feats, vision_pos_embeds, feat_sizes

    def _build_point_inputs(
        self, coords_np: np.ndarray, labels_np: np.ndarray, orig_hw: Tuple[int, int],
    ) -> dict:
        """Build point_inputs dict matching SAM2's expected format."""
        H, W = orig_hw
        pts = torch.from_numpy(coords_np).float().to(self.device)
        pts[:, 0] *= self.INPUT_SIZE / W
        pts[:, 1] *= self.INPUT_SIZE / H
        pts = pts.unsqueeze(0)
        lbls = torch.from_numpy(labels_np).int().to(self.device).unsqueeze(0)
        return {"point_coords": pts, "point_labels": lbls}

    def forward_video(
        self,
        frames_x01: List[torch.Tensor],
        first_mask_np: np.ndarray,
        use_amp: bool = True,
    ) -> List[torch.Tensor]:
        """Forward video through SAM2 using the official track_step path.

        Args:
            frames_x01: [1, 3, H, W] tensors in [0, 1]. Adversarial frames have grad_fn.
            first_mask_np: [H, W] uint8 GT mask for frame 0 prompt.
            use_amp: use mixed precision to reduce memory.

        Returns:
            all_logits: list of [1, 1, H_orig, W_orig] mask logits per frame.
        """
        coords_np, labels_np = get_interior_prompt(first_mask_np)
        orig_hw = first_mask_np.shape[:2]
        point_inputs = self._build_point_inputs(coords_np, labels_np, orig_hw)
        num_frames = len(frames_x01)

        # output_dict mimics SAM2's inference_state structure
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        all_logits = []
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

        for i, x01 in enumerate(frames_x01):
            # Resize to SAM2 input size
            if x01.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
                x_in = F.interpolate(
                    x01, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )
            else:
                x_in = x01

            # Memory opt: no_grad for non-adversarial frames
            needs_grad = x01.grad_fn is not None or x01.requires_grad

            if needs_grad:
                with amp_ctx:
                    vision_feats, vision_pos, feat_sizes = self._encode_backbone(x_in)
            else:
                with torch.no_grad(), amp_ctx:
                    vision_feats, vision_pos, feat_sizes = self._encode_backbone(x_in)
                vision_feats = [f.detach() for f in vision_feats]
                vision_pos = [p.detach() for p in vision_pos]

            with amp_ctx:
                if i == 0:
                    # Frame 0: initial conditioning frame with point prompt
                    current_out = self.sam2.track_step(
                        frame_idx=i,
                        is_init_cond_frame=True,
                        current_vision_feats=vision_feats,
                        current_vision_pos_embeds=vision_pos,
                        feat_sizes=feat_sizes,
                        point_inputs=point_inputs,
                        mask_inputs=None,
                        output_dict=output_dict,
                        num_frames=num_frames,
                        track_in_reverse=False,
                        run_mem_encoder=True,
                        prev_sam_mask_logits=None,
                    )
                    # Store as conditioning frame (privileged memory)
                    output_dict["cond_frame_outputs"][i] = current_out
                else:
                    # Frame 1+: tracking with memory (no point/mask inputs)
                    current_out = self.sam2.track_step(
                        frame_idx=i,
                        is_init_cond_frame=False,
                        current_vision_feats=vision_feats,
                        current_vision_pos_embeds=vision_pos,
                        feat_sizes=feat_sizes,
                        point_inputs=None,
                        mask_inputs=None,
                        output_dict=output_dict,
                        num_frames=num_frames,
                        track_in_reverse=False,
                        run_mem_encoder=True,
                        prev_sam_mask_logits=None,
                    )
                    # Store as non-conditioning frame
                    # NOTE: Do NOT detach memory features — gradients must flow
                    # through intermediate clean frames for long-range read loss.
                    # Memory will be higher but gradient chain stays intact.
                    output_dict["non_cond_frame_outputs"][i] = current_out

            # Extract rich outputs for loss computation
            frame_out = {
                "pred_masks": current_out.get("pred_masks"),
                "pred_masks_high_res": current_out.get("pred_masks_high_res"),
                "object_score_logits": current_out.get("object_score_logits"),
                # Memory features: what SAM2 WRITES to the FIFO bank
                "maskmem_features": current_out.get("maskmem_features"),
                "obj_ptr": current_out.get("obj_ptr"),
            }

            # Interpolate to orig_hw for attack loss
            hi_masks = frame_out.get("pred_masks_high_res")
            if hi_masks is not None:
                logits = F.interpolate(
                    hi_masks.float(), size=orig_hw,
                    mode="bilinear", align_corners=False,
                )
            elif frame_out["pred_masks"] is not None:
                logits = F.interpolate(
                    frame_out["pred_masks"].float(), size=orig_hw,
                    mode="bilinear", align_corners=False,
                )
            else:
                logits = torch.zeros(1, 1, *orig_hw, device=self.device)

            frame_out["logits_orig_hw"] = logits
            all_logits.append(frame_out)

        return all_logits

    def generate_teacher_trajectory(
        self,
        synth_frames_uint8: List[np.ndarray],
        decoy_mask_np: np.ndarray,
    ) -> List[dict]:
        """Run SAM2 on synthetic decoy video to get teacher memory features.

        Args:
            synth_frames_uint8: Synthetic video with object at decoy location.
            decoy_mask_np: [H, W] uint8 mask at decoy location for frame 0 prompt.

        Returns:
            List of {maskmem_features, obj_ptr} per frame (detached).
        """
        coords_np, labels_np = get_interior_prompt(decoy_mask_np)
        orig_hw = decoy_mask_np.shape[:2]
        point_inputs = self._build_point_inputs(coords_np, labels_np, orig_hw)
        num_frames = len(synth_frames_uint8)

        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        teacher_features = []

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for i, frame_np in enumerate(synth_frames_uint8):
                x01 = torch.from_numpy(frame_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                if x01.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
                    x_in = F.interpolate(
                        x01, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                        mode="bilinear", align_corners=False,
                    )
                else:
                    x_in = x01

                vision_feats, vision_pos, feat_sizes = self._encode_backbone(x_in)

                current_out = self.sam2.track_step(
                    frame_idx=i,
                    is_init_cond_frame=(i == 0),
                    current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs if i == 0 else None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=None,
                )

                if i == 0:
                    output_dict["cond_frame_outputs"][i] = current_out
                else:
                    output_dict["non_cond_frame_outputs"][i] = current_out

                teacher_features.append({
                    "maskmem_features": current_out.get("maskmem_features"),
                    "obj_ptr": current_out.get("obj_ptr"),
                })

        return teacher_features
