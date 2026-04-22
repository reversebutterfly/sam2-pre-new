"""
MemoryShield Chunk 5b-i: runtime provenance binding for SAM2 memory bank.

This module closes the Chunk 1 R2 IMPORTANT — "runtime provenance
binding to SAM2's actual `_prepare_memory_conditioned_features`". The
purpose is to compute, *at the moment each non-init frame is being
memory-conditioned*, a `slot_tag` tensor of length `Nk` that maps each
memory-bank token to one of three provenance classes:

    0 = insert           (memory came from a MemoryShield-inserted frame)
    1 = recent (clean)   (memory from a clean prefix frame currently in FIFO)
    2 = other            (conditioning frames beyond the FIFO window, or
                          object-pointer tokens)

`MemAttnProbe` (Chunk 1) uses this tag to compute P_u = [A_ins, A_recent,
A_other] on each eval frame via `probe.set_targets(...)`.

Why this module exists separately from the SAM2 forward glue (Chunk 5b-ii):
- The monkey-patch site (`_prepare_memory_conditioned_features` in
  `sam2.modeling.sam2_base.SAM2Base`) is well-defined; the hard part is
  computing slot_tag from SAM2's runtime state. This module does only
  that.
- The full SAM2 video-predictor glue (init_state / add_new_points /
  propagate_in_video + embedding cache + logit collection at specific
  frames) is a distinct concern handled by 5b-ii.
- Keeping the two concerns apart makes review tractable — the slot-tag
  math can be unit-tested without SAM2 actually running.

Memory layout in SAM2 (verified against sam2_repo/sam2/modeling/sam2_base.py
lines 497–680, pinned commit on Pro 6000):

  to_cat_memory = [
      <selected cond frame 0 maskmem>,   # shape [HW_mem, B, C_mem]
      <selected cond frame 1 maskmem>,   # shape [HW_mem, B, C_mem]
      ...
      <t_pos=1 frame's maskmem>,         # FIFO recent frame
      ...
      <t_pos=N-1 frame's maskmem>,       # FIFO oldest recent frame
      <object pointer tokens>,           # num_obj_ptr_tokens items
  ]
  memory = torch.cat(to_cat_memory, dim=0)   # [Nk, B, C_mem]

Nk decomposition:
  Nk = HW_mem * (num_selected_cond + num_active_recent_frames)
       + num_obj_ptr_tokens

where the cond and recent chunks may overlap in source frame (an
unselected cond that happens to fall in the FIFO window is materialized
only once per concat).

Slot-tag rule for MemoryShield:
- Frames in `insert_frame_ids` (provided by caller) -> 0
- Frames in `recent_clean_ids` (currently-resident prefix frames, i.e.
  the FIFO ids at this moment MINUS the insert ids) -> 1
- Everything else (older cond frames, obj_ptrs) -> 2

The caller provides the set of insert frame indices (which are known
from the Chunk 2 schedule). Recent vs other is derived from SAM2's
runtime selection inside the patched method.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch


SLOT_INSERT = 0
SLOT_RECENT = 1
SLOT_OTHER = 2


@dataclass
class SlotTagResult:
    """Per-frame outputs of one `_prepare_memory_conditioned_features` call."""
    slot_tag: torch.Tensor             # long tensor [Nk]
    source_frame_ids: List[int]        # len = num_chunks; one frame id per chunk
    chunk_lengths: List[int]           # corresponding lengths (HW_mem or obj_ptr count)


class RuntimeProvenanceHook:
    """Monkey-patches SAM2Base._prepare_memory_conditioned_features.

    Usage (typical, inside the PGD step):

        hook = RuntimeProvenanceHook(
            sam2_base=predictor.model,
            insert_frame_ids={6, 12, 17},   # from Chunk 2 schedule
            probe=mem_attn_probe,
            fg_mask_by_frame={f: ... for f in eval_frames},
            HW_mem=64 * 64,
        )
        with hook:
            # ... run SAM2 forward ...
            # hook populates probe.slot_tag_by_frame automatically.
            # probe's patched cross-attn reads it and emits P_u.

    The class is careful about:
      * exactly-once restore of the original method, even on exception
      * not mutating SAM2 tensors — tag building reads frame ids only
      * supporting `track_in_reverse=False` only (the eval pass); the
        few places SAM2 uses reverse order are orthogonal to our flow
    """

    def __init__(
        self,
        sam2_base,
        insert_frame_ids: Set[int],
        probe,
        fg_mask_by_frame: Dict[int, torch.Tensor],
        HW_mem: int,
        has_obj_ptrs: bool = True,
    ) -> None:
        self.sam2_base = sam2_base
        self.insert_frame_ids = set(insert_frame_ids)
        self.probe = probe
        self.fg_mask_by_frame = fg_mask_by_frame
        self.HW_mem = HW_mem
        self.has_obj_ptrs = has_obj_ptrs

        self._original: Optional[Callable] = None
        # Side-channel records for debugging / tests.
        self.slot_tag_by_frame: Dict[int, torch.Tensor] = {}
        self.source_chunks_by_frame: Dict[int, List[Tuple[int, int]]] = {}

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "RuntimeProvenanceHook":
        self._original = self.sam2_base._prepare_memory_conditioned_features
        self.sam2_base._prepare_memory_conditioned_features = self._wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._original is not None:
            self.sam2_base._prepare_memory_conditioned_features = self._original
            self._original = None

    # -- the patched method --------------------------------------------------

    def _wrapped(self, frame_idx, is_init_cond_frame, current_vision_feats,
                 current_vision_pos_embeds, feat_sizes, output_dict,
                 num_frames, track_in_reverse=False):
        """Call the original with slot-tag side effect.

        Fails closed on `track_in_reverse=True` — MemoryShield is forward-
        only (Codex R7 MINOR #1). The `_build_slot_tag` reverse math is
        intentionally still present for future support; we only refuse to
        fire the hook in that mode.
        """
        if track_in_reverse:
            raise NotImplementedError(
                "RuntimeProvenanceHook does not support track_in_reverse "
                "(MemoryShield is forward-only). Remove the hook before "
                "any reverse-tracking SAM2 calls."
            )
        # Init-cond frames have no memory concat (SAM2 returns a fused
        # single-token path); skip tagging for them.
        if is_init_cond_frame or self.sam2_base.num_maskmem == 0:
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )

        # Build slot tag BEFORE the original call so we can pass it into
        # the probe. We recompute the same chunk selection SAM2 does.
        slot_tag, source_chunks = self._build_slot_tag(
            frame_idx=frame_idx, output_dict=output_dict,
            num_frames=num_frames, track_in_reverse=track_in_reverse,
        )
        self.slot_tag_by_frame[frame_idx] = slot_tag
        self.source_chunks_by_frame[frame_idx] = source_chunks

        # Propagate into the probe so its patched cross-attn finds it.
        # We pass the union of previously-set tags so multiple frames can
        # fire without clobbering each other.
        self.probe.set_targets(
            fg_mask_by_frame=self.fg_mask_by_frame,
            slot_tag_by_frame=self.slot_tag_by_frame,
        )
        self.probe.set_frame(frame_idx)

        return self._original(
            frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, output_dict,
            num_frames, track_in_reverse,
        )

    # -- slot-tag math (unit-testable without SAM2) --------------------------

    def _build_slot_tag(
        self, frame_idx: int, output_dict: Dict,
        num_frames: int, track_in_reverse: bool,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Replicate SAM2's chunk selection to build the per-slot provenance
        tag. Returns (slot_tag [Nk] long, source_chunks [(frame_id, len)]).
        """
        base = self.sam2_base
        num_maskmem = base.num_maskmem
        HW_mem = self.HW_mem

        # --- 1. conditioning chunks (t_pos = 0) ---
        from sam2.modeling.sam2_utils import select_closest_cond_frames   # type: ignore
        cond_outputs = output_dict["cond_frame_outputs"]
        selected_cond, unselected_cond = select_closest_cond_frames(
            frame_idx, cond_outputs, base.max_cond_frames_in_attn,
        )
        cond_frame_ids = list(selected_cond.keys())

        # --- 2. FIFO recent chunks (t_pos = 1..num_maskmem-1) ---
        stride = 1 if base.training else base.memory_temporal_stride_for_eval
        recent_frame_ids: List[Optional[int]] = []
        for t_pos in range(1, num_maskmem):
            t_rel = num_maskmem - t_pos
            if t_rel == 1:
                prev_idx = (frame_idx - t_rel) if not track_in_reverse \
                    else (frame_idx + t_rel)
            else:
                if not track_in_reverse:
                    prev_idx = ((frame_idx - 2) // stride) * stride \
                        - (t_rel - 2) * stride
                else:
                    prev_idx = -(-(frame_idx + 2) // stride) * stride \
                        + (t_rel - 2) * stride
            # Check whether this frame actually has a memory entry; if not
            # (padding), we don't include it in the concat.
            out = output_dict["non_cond_frame_outputs"].get(prev_idx, None)
            if out is None:
                out = unselected_cond.get(prev_idx, None)
            if out is not None:
                recent_frame_ids.append(prev_idx)

        # --- 3. object pointer chunks (if enabled) ---
        num_obj_ptr_tokens = 0
        if self.has_obj_ptrs and base.use_obj_ptrs_in_encoder:
            # SAM2 appends up to max_obj_ptrs_in_encoder pointer chunks;
            # each chunk contributes (C // mem_dim) tokens when mem_dim < C.
            # We don't need the exact tokens-per-pointer count for the tag
            # as long as we mark them all "other" — the probe is only
            # interested in foreground-query attention summed per bin.
            # For the chunk-length bookkeeping we count pointers and
            # multiply by tokens-per-pointer.
            tokens_per_ptr = 1
            if base.mem_dim < base.hidden_dim:
                tokens_per_ptr = base.hidden_dim // base.mem_dim
            # Count potentially-added pointers: cond frames (filtered by
            # reverse / past-only), then up to max_obj_ptrs_in_encoder - 1
            # recent non-cond frames.
            if (not base.training
                    and base.only_obj_ptrs_in_the_past_for_eval):
                ptr_cond = {t: out for t, out in selected_cond.items()
                            if (t >= frame_idx if track_in_reverse
                                else t <= frame_idx)}
            else:
                ptr_cond = selected_cond
            num_ptrs = len(ptr_cond)
            max_n = min(num_frames, base.max_obj_ptrs_in_encoder)
            for t_diff in range(1, max_n):
                t = (frame_idx + t_diff) if track_in_reverse \
                    else (frame_idx - t_diff)
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                if (output_dict["non_cond_frame_outputs"].get(t, None)
                        is not None
                        or unselected_cond.get(t, None) is not None):
                    num_ptrs += 1
            num_obj_ptr_tokens = num_ptrs * tokens_per_ptr

        # Delegate to build_slot_tag_direct for the canonical rule. We
        # also record per-chunk source ids for debugging/tests.
        slot_tag = build_slot_tag_direct(
            insert_frame_ids=self.insert_frame_ids,
            cond_frame_ids=cond_frame_ids,
            recent_frame_ids=recent_frame_ids,
            HW_mem=HW_mem,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        source_chunks: List[Tuple[int, int]] = []
        for fid in cond_frame_ids:
            source_chunks.append((fid, HW_mem))
        for fid in recent_frame_ids:
            source_chunks.append((fid, HW_mem))
        if num_obj_ptr_tokens > 0:
            source_chunks.append((-1, num_obj_ptr_tokens))
        return slot_tag, source_chunks


# ---------------------------------------------------------------------------
# Unit-testable helpers (no SAM2 required)
# ---------------------------------------------------------------------------


def build_slot_tag_direct(
    insert_frame_ids: Set[int],
    cond_frame_ids: List[int],
    recent_frame_ids: List[int],
    HW_mem: int,
    num_obj_ptr_tokens: int,
) -> torch.Tensor:
    """Direct slot_tag construction given pre-resolved chunk membership.

    Used by both the `RuntimeProvenanceHook._build_slot_tag` path (after
    it replicates SAM2's chunk selection) and by standalone unit tests
    that don't need a SAM2 model loaded.

    Rule:
        frame_id ∈ insert_frame_ids           -> SLOT_INSERT
        frame_id ∈ FIFO window (recent chunk) -> SLOT_RECENT
        frame_id in cond chunk but not FIFO   -> SLOT_OTHER
        obj_ptr tokens                         -> SLOT_OTHER
    """
    fifo_set = set(recent_frame_ids)
    parts: List[torch.Tensor] = []
    for fid in cond_frame_ids:
        if fid in insert_frame_ids:
            tag = SLOT_INSERT
        elif fid in fifo_set:
            tag = SLOT_RECENT
        else:
            tag = SLOT_OTHER
        parts.append(torch.full((HW_mem,), tag, dtype=torch.long))
    for fid in recent_frame_ids:
        if fid in insert_frame_ids:
            tag = SLOT_INSERT
        else:
            tag = SLOT_RECENT
        parts.append(torch.full((HW_mem,), tag, dtype=torch.long))
    if num_obj_ptr_tokens > 0:
        parts.append(torch.full((num_obj_ptr_tokens,), SLOT_OTHER,
                                dtype=torch.long))
    if not parts:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Smoke tests (no SAM2 required)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    # 1. Simple: 3 inserts {6, 12, 17}. FIFO window at current frame 18 is
    #    t_pos=1..6 -> prev indices 17, 16, 15, 14, 13, 12 (in SAM2's
    #    order; we'll just say the recent chunk is [17, 16, 15, 14, 13, 12]).
    #    Cond frame is {0} (first-frame prompt). No obj ptrs.
    HW = 16  # tiny for test
    tag = build_slot_tag_direct(
        insert_frame_ids={6, 12, 17},
        cond_frame_ids=[0],
        recent_frame_ids=[17, 16, 15, 14, 13, 12],
        HW_mem=HW,
        num_obj_ptr_tokens=0,
    )
    # Total length = (1 cond + 6 recent) * 16 = 112
    assert tag.shape == (112,), tag.shape
    # Cond chunk [0..16): frame 0 not in inserts, not in FIFO -> OTHER
    assert (tag[:HW] == SLOT_OTHER).all()
    # Recent chunk position-by-position:
    # idx 17: INSERT; idx 16: RECENT; idx 15: RECENT; idx 14: RECENT;
    # idx 13: RECENT; idx 12: INSERT
    expected_recent_tags = [SLOT_INSERT, SLOT_RECENT, SLOT_RECENT,
                             SLOT_RECENT, SLOT_RECENT, SLOT_INSERT]
    for i, exp in enumerate(expected_recent_tags):
        start = HW + i * HW
        stop = start + HW
        assert (tag[start:stop] == exp).all(), (i, tag[start:stop].unique())
    print("  slot-tag basic smoke PASS")

    # 2. Cond frame itself is an insert (edge case: first insert is the
    #    initial conditioning frame — impossible per current Chunk 2 schedule
    #    but we test the rule anyway).
    tag2 = build_slot_tag_direct(
        insert_frame_ids={0},
        cond_frame_ids=[0],
        recent_frame_ids=[3, 2, 1],
        HW_mem=HW,
        num_obj_ptr_tokens=0,
    )
    assert (tag2[:HW] == SLOT_INSERT).all()
    print("  cond-is-insert edge case PASS")

    # 3. obj_ptr tokens always OTHER
    tag3 = build_slot_tag_direct(
        insert_frame_ids={2},
        cond_frame_ids=[0],
        recent_frame_ids=[2, 1],
        HW_mem=HW,
        num_obj_ptr_tokens=8,
    )
    # length = (1 + 2) * 16 + 8 = 56
    assert tag3.shape == (56,), tag3.shape
    assert (tag3[-8:] == SLOT_OTHER).all()
    print("  obj_ptr tag PASS")

    # 4. Empty case
    tag4 = build_slot_tag_direct(
        insert_frame_ids=set(), cond_frame_ids=[], recent_frame_ids=[],
        HW_mem=HW, num_obj_ptr_tokens=0,
    )
    assert tag4.shape == (0,) and tag4.dtype == torch.long
    print("  empty-memory PASS")

    # 5. PRECEDENCE UNIT TEST (Codex R7 MINOR #2): this scenario —
    #    cond_frame_ids and recent_frame_ids sharing a frame id — does
    #    NOT arise under real SAM2 runtime, because `selected_cond` and
    #    `unselected_cond` are disjoint by construction; an unselected
    #    cond can appear in the recent chunk via the fallback, but not
    #    in the cond chunk at the same time. This test exists only to
    #    verify the precedence rule in the pure `build_slot_tag_direct`
    #    helper (insert > FIFO-resident > cond-beyond-FIFO).
    tag5 = build_slot_tag_direct(
        insert_frame_ids={12},
        cond_frame_ids=[0, 10],
        recent_frame_ids=[13, 12, 11, 10],
        HW_mem=HW,
        num_obj_ptr_tokens=0,
    )
    # Cond chunk idx 0 -> OTHER; cond chunk idx 10 -> RECENT (in FIFO)
    assert (tag5[:HW] == SLOT_OTHER).all()
    assert (tag5[HW:2 * HW] == SLOT_RECENT).all()
    # Recent chunk: 13=RECENT, 12=INSERT, 11=RECENT, 10=RECENT
    exp_recent = [SLOT_RECENT, SLOT_INSERT, SLOT_RECENT, SLOT_RECENT]
    for i, exp in enumerate(exp_recent):
        start = 2 * HW + i * HW
        stop = start + HW
        assert (tag5[start:stop] == exp).all()
    print("  cond-in-FIFO promotion PASS")

    print("  all 5 provenance-tag invariants OK")


if __name__ == "__main__":
    _smoke()
