"""
FIFO Resonance Scheduler: architecture-aware insertion timing for MemoryShield.

Two generations of API co-exist in this module:

(1) LEGACY (pre-2026-04-22) -- `compute_resonance_schedule`,
    `merge_event_triggers`, `build_modified_index_map`, `InsertionSlot`.
    Kept unchanged for callers in eval_sam2long.py, run_ablation.py,
    run_isolation.py, memshield/generator*.py, memshield/shield.py, and
    run_two_regimes.py.

(2) NEW three-clock (Chunk 2, FINAL_PROPOSAL 2026-04-22):
    `compute_schedule_v2`, `ScheduleV2`, `ClockPositions`,
    `build_index_maps_v2`. These are the canonical interface for the new
    MemoryShield pipeline (Chunk 5 `optimize_unified_v2` and downstream).

Three-clock formalization
-------------------------
    Clock O -- original-video frame index                    o = 0 .. T_orig-1
    Clock M -- modified-sequence frame index                 m = 0 .. T_mod-1
    Clock W -- memory-write index                            w = 0 .. T_mod-1

    Under SAM2's streaming protocol every modified prefix frame produces
    exactly one memory write, so in the prefix M == W. We keep them as
    separate fields to make downstream logic (off-resonance control,
    boundary-forcing) explicit.

Write-aligned seed-plus-boundary schedule
-----------------------------------------
    Period  N - 1 (N = num_maskmem, default 7)
    Seeds   w_k = (N - 1) * k        for k = 1 .. K_ins - 1
    Bound.  w_{K_ins} = T_prefix_mod - 1     (forced adjacent to eval start)

Variants
--------
    "canonical"      -- the above; claim 4 target.
    "off_resonance"  -- seeds at period N - 3 (matched recency on boundary).
    "offset_shift"   -- seeds at (N-1)*k + offset; boundary preserved.
    "custom"         -- caller supplies explicit m-positions.

See refine-logs/FINAL_PROPOSAL.md (§Schedule) and EXPERIMENT_PLAN.md (B2)
for the rationale.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class InsertionSlot:
    """One planned frame insertion."""
    after_original_idx: int   # Insert AFTER this original-video frame index
    frame_type: str           # "strong" | "weak"
    reason: str               # Why this insertion was scheduled


def compute_resonance_schedule(
    n_original: int,
    fifo_window: int = 7,
    max_ratio: float = 0.15,
    first_strong_pos: int = 1,
    cover_prefix: bool = False,
) -> List[InsertionSlot]:
    """Compute FIFO-resonant insertion positions.

    The strong frame is inserted early (after frame `first_strong_pos`).
    Weak frames follow at intervals of (fifo_window - 1), timed so that
    a poisoned entry is always present in the FIFO bank.

    Args:
        n_original: Number of original video frames.
        fifo_window: SAM2 FIFO bank size (num_maskmem, typically 7).
        max_ratio: Maximum inserted frames / original frames.
        first_strong_pos: Insert the first strong frame after this index.
        cover_prefix: If True, ignore max_ratio and insert at every
            resonance period until n_original - 5 (for persistent decoy).

    Returns:
        Sorted list of InsertionSlot.
    """
    if cover_prefix:
        max_inserts = n_original  # No cap — cover the whole prefix
    else:
        max_inserts = max(1, int(n_original * max_ratio))
    period = max(2, fifo_window - 1)

    slots: List[InsertionSlot] = []

    # Strong frame
    strong_pos = min(first_strong_pos, n_original - 2)
    slots.append(InsertionSlot(
        after_original_idx=strong_pos,
        frame_type="strong",
        reason="resonance_anchor",
    ))

    # Weak frames at resonance intervals
    pos = strong_pos + period
    while pos < n_original - 1 and len(slots) < max_inserts:
        slots.append(InsertionSlot(
            after_original_idx=pos,
            frame_type="weak",
            reason="resonance_sustain",
        ))
        pos += period

    return slots


def merge_event_triggers(
    base_schedule: List[InsertionSlot],
    event_positions: List[int],
    n_original: int,
    max_ratio: float = 0.15,
    min_gap: int = 3,
) -> List[InsertionSlot]:
    """Merge event-triggered insertions (occlusion, topology) into the schedule.

    Event triggers can upgrade a nearby weak frame to strong, or insert a new
    strong frame if no scheduled insertion is close.

    Args:
        base_schedule: Output of compute_resonance_schedule.
        event_positions: Original-video frame indices where events occur.
        n_original: Number of original frames.
        max_ratio: Maximum insertion ratio.
        min_gap: Minimum gap between consecutive insertions.

    Returns:
        Updated schedule with event-triggered insertions merged.
    """
    max_inserts = max(1, int(n_original * max_ratio))
    existing_positions = {s.after_original_idx for s in base_schedule}
    merged = list(base_schedule)

    for evt_pos in event_positions:
        if len(merged) >= max_inserts:
            break
        if evt_pos < 1 or evt_pos >= n_original - 1:
            continue

        # Check if there's already an insertion within min_gap
        close = [s for s in merged
                 if abs(s.after_original_idx - evt_pos) <= min_gap]
        if close:
            # Upgrade the closest weak to strong
            closest = min(close, key=lambda s: abs(s.after_original_idx - evt_pos))
            if closest.frame_type == "weak":
                closest.frame_type = "strong"
                closest.reason = "event_upgrade"
        else:
            # Insert a new strong frame
            if evt_pos not in existing_positions:
                merged.append(InsertionSlot(
                    after_original_idx=evt_pos,
                    frame_type="strong",
                    reason="event_trigger",
                ))
                existing_positions.add(evt_pos)

    # Sort by position and enforce max budget
    merged.sort(key=lambda s: s.after_original_idx)
    return merged[:max_inserts]


def build_modified_index_map(
    n_original: int,
    schedule: List[InsertionSlot],
) -> dict:
    """Build a mapping from modified-video indices to original indices.

    Returns:
        dict with keys:
            'mod_to_orig': list where mod_to_orig[i] = original index or -1 if inserted
            'orig_to_mod': list where orig_to_mod[j] = modified index
            'insert_mod_indices': list of modified-video indices that are inserted frames
            'n_modified': total length of modified video
    """
    mod_to_orig = []
    orig_to_mod = [0] * n_original
    insert_mod_indices = []

    # Build insertion lookup: after_original_idx -> list of slots
    insert_after = {}
    for slot in schedule:
        insert_after.setdefault(slot.after_original_idx, []).append(slot)

    mod_idx = 0
    for orig_idx in range(n_original):
        orig_to_mod[orig_idx] = mod_idx
        mod_to_orig.append(orig_idx)
        mod_idx += 1

        if orig_idx in insert_after:
            for slot in insert_after[orig_idx]:
                insert_mod_indices.append(mod_idx)
                mod_to_orig.append(-1)  # -1 = inserted frame
                mod_idx += 1

    return {
        "mod_to_orig": mod_to_orig,
        "orig_to_mod": orig_to_mod,
        "insert_mod_indices": insert_mod_indices,
        "n_modified": mod_idx,
    }


# ============================================================================
# NEW (Chunk 2, 2026-04-22): three-clock write-aligned seed-plus-boundary
# ============================================================================


@dataclass
class ClockPositions:
    """Three-clock position of a single insert in the modified prefix.

    w_k -- memory-write index in the modified prefix (Clock W).
    m_k -- modified-sequence index (Clock M). Equals w_k because every
           modified prefix frame produces exactly one SAM2 write.
    o_after -- the insert is placed immediately AFTER original-frame index
           `o_after` (Clock O). Range: [-1, T_prefix_orig - 1], where -1
           means "before original frame 0" (possible only when the first
           scheduled w is 0).
    role -- "seed" for the K_ins-1 resonance seeds, "boundary" for the
           forced adjacent-to-eval insert.
    """
    w_k: int
    m_k: int
    o_after: int
    role: str


@dataclass
class ScheduleV2:
    """A fully-specified schedule. `slots` is ordered by w_k ascending."""
    T_prefix_orig: int
    T_prefix_mod: int
    num_maskmem: int
    K_ins: int
    variant: str
    slots: List[ClockPositions]

    @property
    def w_positions(self) -> List[int]:
        return [s.w_k for s in self.slots]

    @property
    def m_positions(self) -> List[int]:
        return [s.m_k for s in self.slots]


def _canonical_seed_w(num_maskmem: int, K_ins: int) -> List[int]:
    """Canonical seeds: w_k = (N - 1) * k for k = 1 .. K_ins - 1."""
    period = num_maskmem - 1
    return [period * k for k in range(1, K_ins)]


def _off_resonance_seed_w(num_maskmem: int, K_ins: int) -> List[int]:
    """Off-resonance seeds: w_k = (N - 3) * k (period-2 off canonical).

    Boundary is still forced to T_mod - 1 by the caller, which matches the
    canonical boundary — this preserves recency and isolates the schedule
    alignment effect from a recency confound (EXPERIMENT_PLAN B2).
    """
    period = max(2, num_maskmem - 3)
    return [period * k for k in range(1, K_ins)]


def _offset_shift_seed_w(num_maskmem: int, K_ins: int, offset: int) -> List[int]:
    """Offset-shifted seeds: w_k = (N - 1) * k + offset."""
    period = num_maskmem - 1
    return [period * k + offset for k in range(1, K_ins)]


def compute_schedule_v2(
    T_prefix_orig: int,
    num_maskmem: int = 7,
    K_ins: int = 3,
    variant: str = "canonical",
    offset: int = 0,
    custom_m: Optional[Sequence[int]] = None,
) -> ScheduleV2:
    """Compute a write-aligned seed-plus-boundary insertion schedule.

    Args:
        T_prefix_orig: Number of ORIGINAL-video frames in the attack prefix
            (the eval window starts at original index T_prefix_orig).
        num_maskmem: SAM2 FIFO-memory-bank size N (default 7).
        K_ins: Number of inserts. Must be >= 1.
        variant: "canonical" | "off_resonance" | "offset_shift" | "custom".
        offset: Used only when variant == "offset_shift". Shift applied to
            seed positions; boundary is unaffected.
        custom_m: Used only when variant == "custom". Sequence of modified-
            index positions. Length must equal K_ins. Last element is the
            boundary; preceding entries are seeds. Must be strictly
            increasing and within [0, T_prefix_mod - 1].

    Returns:
        ScheduleV2 with `slots` sorted by w_k ascending.

    Raises:
        ValueError: on inconsistent configuration (bad K_ins, out-of-range
            positions, non-increasing positions, insufficient prefix room).
    """
    if K_ins < 1:
        raise ValueError(f"K_ins must be >= 1, got {K_ins}")
    if num_maskmem < 2:
        raise ValueError(f"num_maskmem must be >= 2, got {num_maskmem}")
    if T_prefix_orig < 1:
        raise ValueError(f"T_prefix_orig must be >= 1, got {T_prefix_orig}")

    T_prefix_mod = T_prefix_orig + K_ins
    w_boundary = T_prefix_mod - 1     # forced adjacent to eval start

    if variant == "canonical":
        seed_w = _canonical_seed_w(num_maskmem, K_ins)
    elif variant == "off_resonance":
        seed_w = _off_resonance_seed_w(num_maskmem, K_ins)
    elif variant == "offset_shift":
        seed_w = _offset_shift_seed_w(num_maskmem, K_ins, offset)
    elif variant == "custom":
        if custom_m is None or len(custom_m) != K_ins:
            raise ValueError(
                f"variant='custom' requires custom_m of length K_ins={K_ins}, "
                f"got {None if custom_m is None else len(custom_m)}"
            )
        cm = list(custom_m)
        for i in range(len(cm) - 1):
            if cm[i + 1] <= cm[i]:
                raise ValueError(f"custom_m must be strictly increasing, got {cm}")
        seed_w = cm[:-1]
        w_boundary = cm[-1]
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected one of: "
            "canonical, off_resonance, offset_shift, custom."
        )

    w_positions = list(seed_w) + [w_boundary]

    for i, w in enumerate(w_positions):
        if not (0 <= w < T_prefix_mod):
            raise ValueError(
                f"w position {w} (idx {i}) out of range [0, {T_prefix_mod - 1}] "
                f"for variant='{variant}'. T_prefix_orig={T_prefix_orig}, "
                f"K_ins={K_ins}, num_maskmem={num_maskmem}, offset={offset}. "
                "Consider smaller K_ins, smaller |offset|, or larger prefix."
            )
    for i in range(len(w_positions) - 1):
        if w_positions[i + 1] <= w_positions[i]:
            raise ValueError(
                f"Schedule positions not strictly increasing: {w_positions}. "
                f"variant='{variant}', offset={offset}. "
                "Seeds collide with boundary or with each other."
            )

    slots: List[ClockPositions] = []
    for one_idx, w in enumerate(w_positions, start=1):
        # With k inserts preceding (k-1 of them before this one), the number
        # of originals before position w is w - (k-1). The insert sits right
        # after the last such original, whose index is w - k.
        o_after = w - one_idx
        if not (-1 <= o_after < T_prefix_orig):
            raise ValueError(
                f"Computed o_after={o_after} for w={w} (k={one_idx}) out of "
                f"range [-1, {T_prefix_orig - 1}]. Check T_prefix_orig "
                f"vs K_ins={K_ins}."
            )
        role = "boundary" if one_idx == K_ins else "seed"
        slots.append(ClockPositions(w_k=w, m_k=w, o_after=o_after, role=role))

    return ScheduleV2(
        T_prefix_orig=T_prefix_orig,
        T_prefix_mod=T_prefix_mod,
        num_maskmem=num_maskmem,
        K_ins=K_ins,
        variant=variant,
        slots=slots,
    )


def build_index_maps_v2(schedule: ScheduleV2) -> dict:
    """Build bidirectional index maps from a ScheduleV2.

    Returns:
        dict with:
            'mod_to_orig' : list of length T_prefix_mod; -1 for inserts.
            'orig_to_mod' : list of length T_prefix_orig.
            'insert_mod_indices' : sorted list of insert mod indices.
            'n_modified' : T_prefix_mod.
    """
    T_orig = schedule.T_prefix_orig
    T_mod = schedule.T_prefix_mod

    insert_set = {s.m_k for s in schedule.slots}
    mod_to_orig: List[int] = [0] * T_mod
    orig_to_mod: List[int] = [0] * T_orig

    next_orig = 0
    for m in range(T_mod):
        if m in insert_set:
            mod_to_orig[m] = -1
        else:
            mod_to_orig[m] = next_orig
            orig_to_mod[next_orig] = m
            next_orig += 1

    if next_orig != T_orig:
        raise RuntimeError(
            f"Index-map inconsistency: placed {next_orig} originals, expected "
            f"{T_orig}. schedule={schedule}"
        )

    return {
        "mod_to_orig": mod_to_orig,
        "orig_to_mod": orig_to_mod,
        "insert_mod_indices": sorted(insert_set),
        "n_modified": T_mod,
    }


def to_legacy_slots(schedule: ScheduleV2) -> List[InsertionSlot]:
    """Adapter: ScheduleV2 -> legacy `List[InsertionSlot]`.

    Legacy `frame_type` has only {"strong","weak"}; both "seed" and "boundary"
    map to "strong". Downstream code that needs the finer distinction should
    inspect `schedule.slots[*].role` directly.
    """
    out: List[InsertionSlot] = []
    for s in schedule.slots:
        if s.o_after < 0:
            raise ValueError(
                "to_legacy_slots: o_after=-1 has no legacy representation "
                "(legacy inserts require after_original_idx >= 0)."
            )
        out.append(InsertionSlot(
            after_original_idx=s.o_after,
            frame_type="strong",
            reason=f"v2_{s.role}",
        ))
    return out


# ----------------------------------------------------------------------------
# Invariant-checking smoke test. Safe to run: only uses this module and numpy.
# ----------------------------------------------------------------------------

def _run_invariant_checks(verbose: bool = False) -> None:
    """Invariant checks. Executed by `python -m memshield.scheduler`."""

    def show(name, sched):
        if verbose:
            print(f"  [{name}] w={sched.w_positions}  "
                  f"o_after={[s.o_after for s in sched.slots]}  "
                  f"T_mod={sched.T_prefix_mod}")

    # 1. Canonical K_ins=3, N=7, T_orig=12 -> w = {6, 12, 14}, o = {5, 10, 11}
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3, variant="canonical")
    show("canonical K=3", s)
    assert s.w_positions == [6, 12, 14], s.w_positions
    assert [x.o_after for x in s.slots] == [5, 10, 11]
    assert s.slots[-1].role == "boundary"
    assert s.slots[0].role == "seed" and s.slots[1].role == "seed"

    # 2. Canonical K_ins=1 -> only boundary at T_mod-1; no seeds
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=1, variant="canonical")
    show("canonical K=1", s)
    assert len(s.slots) == 1 and s.slots[0].role == "boundary"
    assert s.w_positions == [12]
    assert s.slots[0].o_after == 11

    # 3. Off-resonance K_ins=3 -> w = {4, 8, 14}, o = {3, 6, 11}
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3, variant="off_resonance")
    show("off_resonance K=3", s)
    assert s.w_positions == [4, 8, 14]
    assert [x.o_after for x in s.slots] == [3, 6, 11]
    # Boundary preserved for matched recency:
    assert s.slots[-1].w_k == 14

    # 4. Offset shifts -> {5,11,14} and {7,13,14}
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3,
                             variant="offset_shift", offset=-1)
    show("offset_shift -1", s)
    assert s.w_positions == [5, 11, 14]
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3,
                             variant="offset_shift", offset=+1)
    show("offset_shift +1", s)
    assert s.w_positions == [7, 13, 14]

    # 5. Custom matches canonical when given canonical positions
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3,
                             variant="custom", custom_m=[6, 12, 14])
    assert s.w_positions == [6, 12, 14]

    # 6. Invalid custom length
    try:
        compute_schedule_v2(T_prefix_orig=12, K_ins=3, variant="custom", custom_m=[6, 14])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length-mismatched custom_m")

    # 7. Non-increasing custom
    try:
        compute_schedule_v2(T_prefix_orig=12, K_ins=3, variant="custom",
                             custom_m=[6, 6, 14])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-increasing custom_m")

    # 8. Seed collides with boundary -> strict-increase error
    #    offset=+2: seeds=(8,14), boundary=14 -> 14<=14 invalid
    try:
        compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3,
                             variant="offset_shift", offset=+2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for seed==boundary collision")

    # 9. Prefix too small for canonical K_ins=3 (second seed would fall outside)
    try:
        compute_schedule_v2(T_prefix_orig=6, num_maskmem=7, K_ins=3, variant="canonical")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for tiny prefix canonical K=3")

    # 10. build_index_maps_v2 roundtrip for canonical K_ins=3, T_orig=12
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3, variant="canonical")
    maps = build_index_maps_v2(s)
    assert maps["n_modified"] == 15
    assert maps["insert_mod_indices"] == [6, 12, 14]
    assert len(maps["mod_to_orig"]) == 15
    assert len(maps["orig_to_mod"]) == 12
    # All originals present and in order
    placed = [m for m in maps["mod_to_orig"] if m != -1]
    assert placed == list(range(12)), placed
    # orig_to_mod inverts mod_to_orig for non-insert slots
    for o in range(12):
        m = maps["orig_to_mod"][o]
        assert maps["mod_to_orig"][m] == o

    # 11. Legacy adapter preserves o_after as after_original_idx
    legacy = to_legacy_slots(s)
    assert [sl.after_original_idx for sl in legacy] == [5, 10, 11]
    assert all(sl.frame_type == "strong" for sl in legacy)
    assert legacy[-1].reason == "v2_boundary"

    # 12. Off-resonance build_index_maps_v2 roundtrip
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=3,
                             variant="off_resonance")
    maps = build_index_maps_v2(s)
    assert maps["insert_mod_indices"] == [4, 8, 14]
    assert [m for m in maps["mod_to_orig"] if m != -1] == list(range(12))

    # 13. K_ins=1 index-map trivially correct
    s = compute_schedule_v2(T_prefix_orig=12, num_maskmem=7, K_ins=1, variant="canonical")
    maps = build_index_maps_v2(s)
    assert maps["n_modified"] == 13
    assert maps["insert_mod_indices"] == [12]
    assert maps["orig_to_mod"] == list(range(12))

    if verbose:
        print("  all invariants OK")


if __name__ == "__main__":
    _run_invariant_checks(verbose=True)
