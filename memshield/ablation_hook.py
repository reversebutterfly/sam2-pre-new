"""Causal-ablation + restoration hooks for SAM2 memory pathways.

Four context-manager hooks that monkey-patch `SAM2Base` methods so a caller
can isolate WHICH input pathway a given J-drop depends on:

  * `DropNonCondBankHook`         — drops the FIFO non-cond bank on eval
                                    frames (B2 causal-ablation from R4).
  * `SwapHieraFeaturesHook`       — swaps current-frame Hiera output
                                    (vision_feats / pos_embeds / feat_sizes)
                                    at target frames with cached clean
                                    counterparts. Used for VADI R2/R2b
                                    restoration: "damage lives in the
                                    current-frame pathway at inserts."
  * `SwapBankHook`                — swaps entries in
                                    `output_dict['non_cond_frame_outputs']`
                                    with clean cached entries. Used for
                                    VADI R3 restoration ("bank not causal").
  * `SwapF0MemoryHook`            — swaps the init-cond entry
                                    `output_dict['cond_frame_outputs'][f0_id]`
                                    with a clean cached entry. Isolates
                                    whether f0 prompt memory is where the
                                    damage lives.

All hooks are strictly additive: non-target / init-cond calls pass through
unmodified, and shallow copies are used so the caller's output_dict is
never mutated.

Patch points:
  * SAM2Base._prepare_memory_conditioned_features — Drop / SwapBank /
    SwapF0Memory (reads output_dict here).
  * SAM2Base.track_step — SwapHieraFeatures (vision feats enter here).

Call via kwargs: our `SAM2VideoAdapter` invokes `sam2.track_step(**kwargs)`
with all named arguments, so the kwargs-only wrapper in SwapHieraFeatures
is compatible.

Run `python -m memshield.ablation_hook` for API-check self-tests.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Keys that SAM2's `_prepare_memory_conditioned_features` reads from each
# per-frame entry in `output_dict['cond_frame_outputs' / 'non_cond_frame_outputs']`.
# A clean swap entry must at minimum contain these (may contain more).
_REQUIRED_MEM_ENTRY_KEYS = ("maskmem_features", "maskmem_pos_enc", "obj_ptr")


def _validate_mem_entry(tag: str, entry: Any) -> Dict[str, Any]:
    """Ensure a clean memory entry has the keys SAM2 will read from it.

    We validate at construction time (not per-call) so config errors
    surface before any hook fires, not as a cryptic SAM2 crash deep in
    memory attention.
    """
    if not isinstance(entry, dict):
        raise TypeError(
            f"{tag}: expected a dict (SAM2 per-frame current_out), got "
            f"{type(entry).__name__}")
    missing = [k for k in _REQUIRED_MEM_ENTRY_KEYS if k not in entry]
    if missing:
        raise KeyError(
            f"{tag}: clean entry missing required key(s) {missing}. "
            f"A swap entry must be a complete SAM2 current_out dict; "
            f"partial entries will crash _prepare_memory_conditioned_features.")
    return entry

# -----------------------------------------------------------------------------
# B2: drop non-cond bank
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# VADI R2 / R2b: swap current-frame Hiera features
# -----------------------------------------------------------------------------


class SwapHieraFeaturesHook:
    """Monkey-patches `SAM2Base.track_step` so that at target frames the
    current-frame Hiera output (`current_vision_feats`,
    `current_vision_pos_embeds`, `feat_sizes`) is replaced with a clean
    cached counterpart BEFORE SAM2 runs memory attention + mask decoder +
    memory encoder.

    Caller provides `clean_features_by_frame_id` mapping:
        { frame_idx (int) : {
              'vision_feats':      List[Tensor],   # SAM2's flattened HWxBxC
              'vision_pos_embeds': List[Tensor],   # same
              'feat_sizes':        List[Tuple[int, int]],  # (H, W) per level
          } }

    Frame index convention: keys are ATTACKED-video processing indices
    (post-insertion), not clean-video indices. VADI inserts shift the
    index space; the caller is responsible for mapping clean-video
    frames to their attacked-video processing positions before building
    this map.

    Frames NOT in the map are processed with their original (attacked)
    features. Init-cond calls (where `is_init_cond_frame=True`, SAM2.1
    bypasses the memory path) are ALWAYS passed through unmodified —
    swapping f0's Hiera output would change the prompt-encoding regime
    and is out of scope for R2/R2b. Downstream reads of the f0-written
    memory can still be isolated via `SwapF0MemoryHook`.

    Only the `track_step` invocation receives the swapped tensors — the
    caller's adapter state (e.g. `_suffix_cache`) is not mutated. The
    swap entry dict itself is treated as READ-ONLY by this hook; if
    SAM2 performs in-place mutation on the tensors (it does not in
    practice, but defensively), the caller should pass a deep copy.

    Requires `sam2.track_step` to be called KWARGS-only. Our
    `SAM2VideoAdapter._track_single_frame` already does this.
    """

    def __init__(
        self,
        sam2_base,
        clean_features_by_frame_id: Dict[int, Dict[str, Any]],
    ) -> None:
        self.sam2_base = sam2_base
        self.clean_by_id: Dict[int, Dict[str, Any]] = {
            int(k): self._validate_entry(k, v)
            for k, v in clean_features_by_frame_id.items()
        }
        self._original: Optional[Callable] = None
        self.fires: int = 0

    @staticmethod
    def _validate_entry(k: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("vision_feats", "vision_pos_embeds", "feat_sizes"):
            if key not in entry:
                raise KeyError(
                    f"clean_features_by_frame_id[{k}] missing '{key}'")
        return entry

    def __enter__(self) -> "SwapHieraFeaturesHook":
        self._original = self.sam2_base.track_step
        self.sam2_base.track_step = self._wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._original is not None:
            self.sam2_base.track_step = self._original
            self._original = None

    def _wrapped(self, *args, **kwargs):
        if args:
            raise NotImplementedError(
                "SwapHieraFeaturesHook expects track_step called via kwargs "
                "(SAM2VideoAdapter does so); got positional args.")
        frame_idx = kwargs.get("frame_idx")
        if frame_idx is None:
            raise KeyError("track_step called without 'frame_idx' kwarg")
        # Init-cond (f0) processing is the prompt-encoding regime; SAM2.1
        # bypasses memory there. Swapping Hiera here would change the
        # prompt encoding rather than isolate the current-frame pathway,
        # so we always pass through.
        if kwargs.get("is_init_cond_frame", False):
            return self._original(**kwargs)
        if int(frame_idx) in self.clean_by_id:
            swap = self.clean_by_id[int(frame_idx)]
            kwargs["current_vision_feats"] = swap["vision_feats"]
            kwargs["current_vision_pos_embeds"] = swap["vision_pos_embeds"]
            kwargs["feat_sizes"] = swap["feat_sizes"]
            self.fires += 1
        return self._original(**kwargs)


# -----------------------------------------------------------------------------
# VADI R3: swap non-cond bank entries
# -----------------------------------------------------------------------------


class SwapBankHook:
    """Monkey-patches `SAM2Base._prepare_memory_conditioned_features` so
    that, for every non-init-cond call, any entry in
    `output_dict['non_cond_frame_outputs']` whose key appears in
    `clean_bank_entries` is replaced with the clean counterpart.

    Unlike `DropNonCondBankHook` which empties the bank on a per-target-
    frame basis, this hook always re-routes to the clean cache whenever
    a matching key is present. This matches the R3 restoration semantics:
    "what if the attacked frames had read a clean bank instead?"

    `clean_bank_entries`: `{ frame_id (int) : current_out dict }`,
    where `current_out` is a clean-SAM2 track_step return value
    containing at minimum `maskmem_features`, `maskmem_pos_enc`,
    `obj_ptr` (validated at construction time).

    Frame index convention: keys are the bank-side frame IDs (same
    space SAM2 uses in `output_dict['non_cond_frame_outputs']`) — i.e.
    the ATTACKED-video processing indices at which those entries were
    WRITTEN during the attacked forward. Caller must align to that
    space when building the clean cache.

    The caller's output_dict is NOT mutated — the container maps are
    shallow-copied before assignment. The swap entry dicts themselves
    are treated as read-only by this hook (SAM2 does not in practice
    mutate them, but defensively the caller should pass copies if
    sharing the cache across multiple hook invocations).
    """

    def __init__(
        self,
        sam2_base,
        clean_bank_entries: Dict[int, Dict[str, Any]],
    ) -> None:
        self.sam2_base = sam2_base
        self.clean_bank_entries: Dict[int, Dict[str, Any]] = {
            int(k): _validate_mem_entry(
                f"SwapBankHook.clean_bank_entries[{k}]", v)
            for k, v in clean_bank_entries.items()
        }
        self._original: Optional[Callable] = None
        self.fires: int = 0

    def __enter__(self) -> "SwapBankHook":
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
        if is_init_cond_frame or self.sam2_base.num_maskmem <= 0:
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        current_nc = output_dict.get("non_cond_frame_outputs", {})
        overlap = [fid for fid in self.clean_bank_entries if fid in current_nc]
        if not overlap:
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        patched = dict(output_dict)
        new_nc = dict(current_nc)
        for fid in overlap:
            new_nc[fid] = self.clean_bank_entries[fid]
        patched["non_cond_frame_outputs"] = new_nc
        self.fires += 1
        return self._original(
            frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, patched,
            num_frames, track_in_reverse,
        )


# -----------------------------------------------------------------------------
# F0 restoration: swap the init-cond (prompt) memory entry
# -----------------------------------------------------------------------------


class SwapF0MemoryHook:
    """Monkey-patches `SAM2Base._prepare_memory_conditioned_features` so
    that, for every non-init-cond call, the entry
    `output_dict['cond_frame_outputs'][f0_frame_id]` is replaced with a
    clean cached counterpart (if present in `output_dict`).

    Isolates whether f0 (prompt-frame) memory is the pathway where the
    attack's damage lives. If clean-f0 restoration recovers J, the damage
    is in f0 propagation; if not, the f0 memory is not causally important
    for the observed J-drop.

    `clean_f0_entry`: the clean `current_out` dict for the prompt frame
    (typically produced by running clean-SAM2 with `is_init_cond_frame=
    True`). Must contain `maskmem_features`, `maskmem_pos_enc`,
    `obj_ptr` (validated at construction time).

    Frame index convention: `f0_frame_id` is the attacked-video
    processing index of the prompt frame (VADI uses f0=0). Same read-
    only caveat on the entry as `SwapBankHook`.
    """

    def __init__(
        self,
        sam2_base,
        clean_f0_entry: Dict[str, Any],
        f0_frame_id: int = 0,
    ) -> None:
        self.sam2_base = sam2_base
        self.clean_f0_entry = _validate_mem_entry(
            "SwapF0MemoryHook.clean_f0_entry", clean_f0_entry)
        self.f0_frame_id = int(f0_frame_id)
        self._original: Optional[Callable] = None
        self.fires: int = 0

    def __enter__(self) -> "SwapF0MemoryHook":
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
        if is_init_cond_frame:
            # The init-cond call is where f0's entry gets WRITTEN to
            # cond_frame_outputs by the caller. We don't intercept that —
            # we intercept downstream reads by non-init-cond frames.
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        current_cond = output_dict.get("cond_frame_outputs", {})
        if self.f0_frame_id not in current_cond:
            return self._original(
                frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        patched = dict(output_dict)
        new_cond = dict(current_cond)
        new_cond[self.f0_frame_id] = self.clean_f0_entry
        patched["cond_frame_outputs"] = new_cond
        self.fires += 1
        return self._original(
            frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, patched,
            num_frames, track_in_reverse,
        )


# -----------------------------------------------------------------------------
# Self-tests (fake SAM2Base — no torch / SAM2 deps required)
# -----------------------------------------------------------------------------


class _FakeSam2Base:
    """Minimal SAM2Base stand-in for hook API tests.

    Records every call to `track_step` and `_prepare_memory_conditioned_features`
    so tests can assert what the hook passed through to the ORIGINAL.
    """

    num_maskmem: int = 16

    def __init__(self) -> None:
        self.track_step_calls: List[Dict[str, Any]] = []
        self.prep_mem_calls: List[Dict[str, Any]] = []

    def track_step(self, **kwargs) -> Dict[str, Any]:
        # Record a shallow copy so downstream mutations don't bleed in.
        self.track_step_calls.append(dict(kwargs))
        return {"sentinel": "track_step_result", "frame_idx": kwargs["frame_idx"]}

    def _prepare_memory_conditioned_features(
        self, frame_idx, is_init_cond_frame, current_vision_feats,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False,
    ) -> str:
        self.prep_mem_calls.append({
            "frame_idx": frame_idx,
            "is_init_cond_frame": is_init_cond_frame,
            "output_dict_cond": dict(output_dict.get("cond_frame_outputs", {})),
            "output_dict_noncond": dict(
                output_dict.get("non_cond_frame_outputs", {})),
        })
        return f"prep_mem_{frame_idx}"


def _self_test() -> None:
    # Factory for a valid clean memory entry — all 3 required keys present.
    def _mk_mem_entry(tag: str) -> Dict[str, Any]:
        return {
            "maskmem_features": f"{tag}_feats",
            "maskmem_pos_enc":  f"{tag}_pos",
            "obj_ptr":          f"{tag}_ptr",
        }

    # =========================================================================
    # _validate_mem_entry: TypeError + every partial-dict shape
    # =========================================================================
    try:
        _validate_mem_entry("tag", "not-a-dict")
        raise AssertionError("non-dict must raise TypeError")
    except TypeError:
        pass
    # Missing exactly 1 key.
    entry_missing_one = _mk_mem_entry("x")
    del entry_missing_one["obj_ptr"]
    try:
        _validate_mem_entry("tag", entry_missing_one)
        raise AssertionError("missing 1 key must raise KeyError")
    except KeyError as e:
        assert "obj_ptr" in str(e)
    # Missing 2 keys.
    try:
        _validate_mem_entry("tag", {"maskmem_features": "x"})
        raise AssertionError("missing 2 keys must raise KeyError")
    except KeyError as e:
        assert "maskmem_pos_enc" in str(e) and "obj_ptr" in str(e)
    # Missing all 3 keys.
    try:
        _validate_mem_entry("tag", {})
        raise AssertionError("missing all keys must raise KeyError")
    except KeyError:
        pass
    # Extra keys are allowed (SAM2 current_out has many).
    extra = _mk_mem_entry("y")
    extra["pred_masks"] = "bonus"
    assert _validate_mem_entry("tag", extra) is extra

    # =========================================================================
    # DropNonCondBankHook
    # =========================================================================
    sam2 = _FakeSam2Base()
    # Simulate SAM2's _prepare_mem being invoked for frames 0..5; 0 is init-cond.
    # Target frames 3, 4. Bank has a fake entry at frame 1.
    target = {3, 4}
    bank = {1: _mk_mem_entry("ORIG_t1")}
    with DropNonCondBankHook(sam2, target) as h:
        captured_out_dicts = []
        for fid in range(6):
            out_dict = {
                "cond_frame_outputs": {0: _mk_mem_entry("f0")},
                "non_cond_frame_outputs": dict(bank),
            }
            captured_out_dicts.append(out_dict)
            sam2._prepare_memory_conditioned_features(
                fid, fid == 0, None, None, None, out_dict, 6,
            )
        # Hook fires only on target & non-init & num_maskmem > 0.
        assert h.fires == 2, f"expected 2 drops, got {h.fires}"
        # Caller's output_dict must not be mutated (even for target frames).
        for od in captured_out_dicts:
            assert od["non_cond_frame_outputs"] == bank, \
                "DropNonCondBankHook must not mutate caller's output_dict"
    # After exit, patch restored.
    assert sam2._prepare_memory_conditioned_features.__func__ is \
        _FakeSam2Base._prepare_memory_conditioned_features
    # Target frames saw empty non-cond bank; non-targets saw original.
    dropped_calls = [c for c in sam2.prep_mem_calls if c["frame_idx"] in target]
    passthrough_calls = [c for c in sam2.prep_mem_calls
                         if c["frame_idx"] not in target]
    assert all(c["output_dict_noncond"] == {} for c in dropped_calls)
    assert all(c["output_dict_noncond"] == bank for c in passthrough_calls
               if c["frame_idx"] != 0)
    # Init-cond (frame 0) always passes through regardless of target.
    f0_call = next(c for c in sam2.prep_mem_calls if c["frame_idx"] == 0)
    assert f0_call["output_dict_noncond"] == bank

    # num_maskmem == 0 disables drop.
    sam2_nomem = _FakeSam2Base()
    sam2_nomem.num_maskmem = 0
    with DropNonCondBankHook(sam2_nomem, {2}) as h:
        out_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {1: "x"}}
        sam2_nomem._prepare_memory_conditioned_features(
            2, False, None, None, None, out_dict, 3,
        )
        assert h.fires == 0

    # =========================================================================
    # SwapHieraFeaturesHook
    # =========================================================================
    sam2 = _FakeSam2Base()
    clean_feats = {
        3: {"vision_feats": ["CLEAN_F3"], "vision_pos_embeds": ["CLEAN_P3"],
            "feat_sizes": [(8, 8)]},
        5: {"vision_feats": ["CLEAN_F5"], "vision_pos_embeds": ["CLEAN_P5"],
            "feat_sizes": [(8, 8)]},
    }
    with SwapHieraFeaturesHook(sam2, clean_feats) as h:
        for fid in range(6):
            sam2.track_step(
                frame_idx=fid,
                is_init_cond_frame=(fid == 0),
                current_vision_feats=[f"ATTACKED_F{fid}"],
                current_vision_pos_embeds=[f"ATTACKED_P{fid}"],
                feat_sizes=[(4, 4)],
                point_inputs=None,
                mask_inputs=None,
                output_dict={},
                num_frames=6,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
        assert h.fires == 2, f"expected 2 Hiera swaps, got {h.fires}"
    # track_step restored after exit.
    assert sam2.track_step.__func__ is _FakeSam2Base.track_step
    # Swapped frames saw clean feats; others saw attacked.
    for call in sam2.track_step_calls:
        fid = call["frame_idx"]
        if fid in clean_feats:
            assert call["current_vision_feats"] == [f"CLEAN_F{fid}"]
            assert call["current_vision_pos_embeds"] == [f"CLEAN_P{fid}"]
            assert call["feat_sizes"] == [(8, 8)]
        else:
            assert call["current_vision_feats"] == [f"ATTACKED_F{fid}"]
            assert call["current_vision_pos_embeds"] == [f"ATTACKED_P{fid}"]
            assert call["feat_sizes"] == [(4, 4)]

    # Positional track_step must raise.
    sam2_pos = _FakeSam2Base()
    with SwapHieraFeaturesHook(sam2_pos, {1: clean_feats[3]}):
        try:
            sam2_pos.track_step(0, False, [], [], [(4, 4)])
            raise AssertionError("positional track_step must raise")
        except NotImplementedError:
            pass

    # Missing required key in clean entry raises at construction time.
    try:
        SwapHieraFeaturesHook(sam2, {0: {"vision_feats": []}})
        raise AssertionError("missing feat_sizes must raise")
    except KeyError:
        pass

    # Init-cond passthrough: even if frame_idx is in the swap map, an
    # init-cond call must NOT trigger a swap.
    sam2 = _FakeSam2Base()
    with SwapHieraFeaturesHook(sam2, {0: clean_feats[3]}) as h:
        sam2.track_step(
            frame_idx=0, is_init_cond_frame=True,
            current_vision_feats=["ATTACKED_F0"],
            current_vision_pos_embeds=["ATTACKED_P0"],
            feat_sizes=[(4, 4)], point_inputs=None, mask_inputs=None,
            output_dict={}, num_frames=3, track_in_reverse=False,
            run_mem_encoder=True, prev_sam_mask_logits=None,
        )
        assert h.fires == 0, "init-cond frame must skip Hiera swap"
        assert sam2.track_step_calls[-1]["current_vision_feats"] == ["ATTACKED_F0"]

    # Exception inside `with` restores track_step.
    sam2 = _FakeSam2Base()
    try:
        with SwapHieraFeaturesHook(sam2, {0: clean_feats[3]}):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert sam2.track_step.__func__ is _FakeSam2Base.track_step

    # =========================================================================
    # SwapBankHook
    # =========================================================================
    sam2 = _FakeSam2Base()
    clean_bank = {
        1: _mk_mem_entry("CLEAN_t1"),
        2: _mk_mem_entry("CLEAN_t2"),
    }
    att_bank = {
        1: _mk_mem_entry("ATT_t1"),
        2: _mk_mem_entry("ATT_t2"),
        9: _mk_mem_entry("ATT_t9"),
    }
    caller_out_3 = {
        "cond_frame_outputs": {0: _mk_mem_entry("f0_ATT")},
        "non_cond_frame_outputs": dict(att_bank),
    }
    with SwapBankHook(sam2, clean_bank) as h:
        # Frame 3 processing: bank has t=1,2 matching (both swapped) + t=9
        # untouched.
        sam2._prepare_memory_conditioned_features(
            3, False, None, None, None, caller_out_3, 10)
        # Frame 4 processing: bank only has t=9 (no overlap).
        caller_out_4 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {9: _mk_mem_entry("ATT_t9")},
        }
        sam2._prepare_memory_conditioned_features(
            4, False, None, None, None, caller_out_4, 10)
        # Init-cond frame: must pass through.
        caller_out_0 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": dict(clean_bank),
        }
        sam2._prepare_memory_conditioned_features(
            0, True, None, None, None, caller_out_0, 10)
        assert h.fires == 1, f"expected 1 bank swap (frame 3), got {h.fires}"
    # Frame 3 saw clean bank entries swapped in; frame 9 preserved.
    call3 = sam2.prep_mem_calls[0]
    assert call3["output_dict_noncond"][1]["maskmem_features"] == "CLEAN_t1_feats"
    assert call3["output_dict_noncond"][2]["maskmem_features"] == "CLEAN_t2_feats"
    assert call3["output_dict_noncond"][9]["maskmem_features"] == "ATT_t9_feats"
    # cond_frame_outputs untouched by SwapBankHook.
    assert call3["output_dict_cond"][0]["maskmem_features"] == "f0_ATT_feats"
    # Frame 4 saw no swap.
    call4 = sam2.prep_mem_calls[1]
    assert call4["output_dict_noncond"][9]["maskmem_features"] == "ATT_t9_feats"
    # Init-cond frame 0 passed through untouched.
    call0 = sam2.prep_mem_calls[2]
    assert call0["is_init_cond_frame"] is True
    # Caller's output_dict not mutated — still has ATT entries at t=1,2.
    assert caller_out_3["non_cond_frame_outputs"][1]["maskmem_features"] \
        == "ATT_t1_feats"
    assert caller_out_3["non_cond_frame_outputs"][2]["maskmem_features"] \
        == "ATT_t2_feats"

    # Partial entry rejected at construction time.
    try:
        SwapBankHook(sam2, {1: {"maskmem_features": "x"}})
        raise AssertionError("partial bank entry must raise")
    except KeyError:
        pass

    # Exception-safety.
    sam2 = _FakeSam2Base()
    try:
        with SwapBankHook(sam2, clean_bank):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert sam2._prepare_memory_conditioned_features.__func__ is \
        _FakeSam2Base._prepare_memory_conditioned_features

    # =========================================================================
    # SwapF0MemoryHook
    # =========================================================================
    sam2 = _FakeSam2Base()
    clean_f0 = _mk_mem_entry("CLEAN_F0")
    caller_out_by_fid: List[Dict[str, Any]] = []
    with SwapF0MemoryHook(sam2, clean_f0, f0_frame_id=0) as h:
        # Non-init-cond frames with cond_frame_outputs[0] present → swap.
        for fid in (1, 2, 3):
            out = {
                "cond_frame_outputs": {0: _mk_mem_entry("ATT_F0")},
                "non_cond_frame_outputs": {},
            }
            caller_out_by_fid.append(out)
            sam2._prepare_memory_conditioned_features(
                fid, False, None, None, None, out, 4)
        # Init-cond call → pass-through (no swap even though key present).
        sam2._prepare_memory_conditioned_features(
            0, True, None, None, None,
            {"cond_frame_outputs": {0: _mk_mem_entry("ATT_F0")},
             "non_cond_frame_outputs": {}}, 4)
        # Frame with no cond_frame_outputs[0] entry → pass-through.
        sam2._prepare_memory_conditioned_features(
            5, False, None, None, None,
            {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}, 4)
        assert h.fires == 3, f"expected 3 f0 swaps, got {h.fires}"
    # 3 swapped calls: cond dict has clean entry.
    swap_calls = [c for c in sam2.prep_mem_calls
                  if c["frame_idx"] in (1, 2, 3)]
    for c in swap_calls:
        assert c["output_dict_cond"][0]["maskmem_features"] == "CLEAN_F0_feats"
    # Init-cond call preserved original.
    init_call = next(c for c in sam2.prep_mem_calls if c["frame_idx"] == 0)
    assert init_call["output_dict_cond"][0]["maskmem_features"] == "ATT_F0_feats"
    # Caller's output_dicts still hold ATT entries (no mutation).
    for od in caller_out_by_fid:
        assert od["cond_frame_outputs"][0]["maskmem_features"] == "ATT_F0_feats"

    # Non-default f0_frame_id.
    sam2 = _FakeSam2Base()
    with SwapF0MemoryHook(sam2, clean_f0, f0_frame_id=2) as h:
        sam2._prepare_memory_conditioned_features(
            3, False, None, None, None,
            {"cond_frame_outputs": {2: _mk_mem_entry("ATT_F2")},
             "non_cond_frame_outputs": {}}, 4)
        assert h.fires == 1

    # Partial entry rejected at construction time.
    try:
        SwapF0MemoryHook(sam2, {"obj_ptr": "x"})
        raise AssertionError("partial f0 entry must raise")
    except KeyError:
        pass

    # Exception-safety.
    sam2 = _FakeSam2Base()
    try:
        with SwapF0MemoryHook(sam2, clean_f0):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert sam2._prepare_memory_conditioned_features.__func__ is \
        _FakeSam2Base._prepare_memory_conditioned_features

    # =========================================================================
    # Nested use: SwapHiera + SwapBank simultaneously on the same SAM2Base
    # Each patches a different method so they coexist.
    # =========================================================================
    sam2 = _FakeSam2Base()
    with SwapHieraFeaturesHook(sam2, {2: clean_feats[3]}) as h_hiera, \
            SwapBankHook(sam2, clean_bank) as h_bank:
        sam2.track_step(
            frame_idx=2,
            is_init_cond_frame=False,
            current_vision_feats=["ATT"], current_vision_pos_embeds=["ATT_P"],
            feat_sizes=[(4, 4)], point_inputs=None, mask_inputs=None,
            output_dict={}, num_frames=3, track_in_reverse=False,
            run_mem_encoder=True, prev_sam_mask_logits=None,
        )
        sam2._prepare_memory_conditioned_features(
            2, False, None, None, None,
            {"cond_frame_outputs": {}, "non_cond_frame_outputs":
                {1: {"maskmem_features": "ATT_t1"}}}, 3)
        assert h_hiera.fires == 1
        assert h_bank.fires == 1
    # Both restored.
    assert sam2.track_step.__func__ is _FakeSam2Base.track_step
    assert sam2._prepare_memory_conditioned_features.__func__ is \
        _FakeSam2Base._prepare_memory_conditioned_features

    print("memshield.ablation_hook: all self-tests PASSED "
          "(DropNonCondBank, SwapHieraFeatures, SwapBank, SwapF0Memory: "
          "isolation + passthrough + nesting + exception-safety)")


if __name__ == "__main__":
    _self_test()
