"""
FIFO Resonance Scheduler: architecture-aware insertion timing for MemoryShield.

Core idea: SAM2 maintains a FIFO memory bank of size N (default 7).
We insert a strong adversarial frame, then weak frames at intervals of (N-1)
so that by the time one poisoned memory is evicted, the next has arrived.
This creates a "standing wave" of corruption in the memory bank.

Event triggers (occlusion, topology) can inject additional strong frames.
"""
from dataclasses import dataclass
from typing import List, Optional

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

    Returns:
        Sorted list of InsertionSlot.
    """
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
