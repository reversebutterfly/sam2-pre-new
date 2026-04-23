"""Causal-ablation hook for SAM2's non-cond memory bank (auto-review R4 / B2).

Extends the runtime hook pattern from `memshield/sam2_forward_v2.py`. Where
`RuntimeProvenanceHook` tags slots for the probe, `DropNonCondBankHook`
drops them entirely: for any frame in `target_frame_ids`, the wrapped
`_prepare_memory_conditioned_features` receives an `output_dict` whose
`non_cond_frame_outputs` is empty, so the query sees only conditioning
(f0) memory + current-frame features.

Purpose: tests whether the non-cond FIFO bank is a behaviorally
meaningful input to SAM2's segmentation. If clean J is near-identical
with and without the bank, decoy-insert attacks that poison this bank
cannot affect segmentation — regardless of attention weight routing.

Usage:
    ids_to_ablate = set(range(T_prefix, T_prefix + eval_window))
    with DropNonCondBankHook(predictor.model, ids_to_ablate):
        state = predictor.init_state(video_path=...)
        predictor.add_new_mask(...)
        for t, _, masks in predictor.propagate_in_video(state):
            ...
"""
from __future__ import annotations

from typing import Callable, Optional, Set


class DropNonCondBankHook:
    """Monkey-patches SAM2Base._prepare_memory_conditioned_features so that
    target frames see an empty non-cond bank.

    The patch is strictly additive: for frames NOT in `target_frame_ids`,
    the original function is called unmodified. For target frames, the
    original is called with a shallow-copied output_dict whose
    `non_cond_frame_outputs` field is replaced by an empty dict. This
    does not mutate the caller's output_dict; subsequent frames still
    see their real non-cond writebacks.
    """

    def __init__(self, sam2_base, target_frame_ids: Set[int]) -> None:
        self.sam2_base = sam2_base
        self.target_frame_ids = set(int(t) for t in target_frame_ids)
        self._original: Optional[Callable] = None
        self.fires: int = 0  # count of ablated calls (for sanity checks)

    def __enter__(self) -> "DropNonCondBankHook":
        self._original = self.sam2_base._prepare_memory_conditioned_features
        self.sam2_base._prepare_memory_conditioned_features = self._wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._original is not None:
            self.sam2_base._prepare_memory_conditioned_features = self._original
            self._original = None

    def _wrapped(self, frame_idx, is_init_cond_frame, current_vision_feats,
                 current_vision_pos_embeds, feat_sizes, output_dict,
                 num_frames, track_in_reverse=False):
        if (frame_idx in self.target_frame_ids
                and not is_init_cond_frame
                and self.sam2_base.num_maskmem > 0):
            patched = dict(output_dict)
            patched["non_cond_frame_outputs"] = {}
            self.fires += 1
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, patched,
                num_frames, track_in_reverse,
            )
        return self._original(
            frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, output_dict,
            num_frames, track_in_reverse,
        )
