"""
Content Analyzer: detect occlusion events, narrow topology, and scene changes.

Used by the scheduler to place event-triggered insertions at naturally
vulnerable moments in the video.
"""
from typing import List, Tuple

import cv2
import numpy as np


def compute_optical_flow_magnitudes(
    frames_uint8: List[np.ndarray],
) -> np.ndarray:
    """Compute per-frame mean optical flow magnitude.

    Args:
        frames_uint8: list of [H, W, 3] uint8 frames.

    Returns:
        flow_mags: [T-1] array of mean flow magnitudes between consecutive frames.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_uint8]
    mags = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
    return np.array(mags, dtype=np.float32)


def detect_occlusion_events(
    masks_uint8: List[np.ndarray],
    flow_mags: np.ndarray,
    flow_drop_threshold: float = 0.3,
    area_drop_threshold: float = 0.4,
) -> List[int]:
    """Detect frames where the target object is likely occluded or reappearing.

    Heuristic: large mask area drop + flow discontinuity → occlusion event.

    Args:
        masks_uint8: list of [H, W] binary masks.
        flow_mags: [T-1] per-frame flow magnitudes.
        flow_drop_threshold: relative flow drop to flag.
        area_drop_threshold: relative mask area drop to flag.

    Returns:
        List of frame indices where occlusion events are detected.
    """
    areas = [float(m.astype(bool).sum()) for m in masks_uint8]
    events = []

    for i in range(1, len(masks_uint8)):
        prev_area = max(areas[i - 1], 1.0)
        area_ratio = areas[i] / prev_area

        # Large area drop → object disappearing / occluded
        if area_ratio < (1.0 - area_drop_threshold):
            events.append(i)
            continue

        # Large area increase → object reappearing
        if areas[i] > 0 and area_ratio > (1.0 + area_drop_threshold):
            events.append(i)
            continue

        # Flow discontinuity near object
        if i - 1 < len(flow_mags) and i >= 2:
            flow_prev = flow_mags[i - 2] if i >= 2 else flow_mags[0]
            flow_curr = flow_mags[i - 1]
            if flow_prev > 1.0:
                flow_ratio = flow_curr / flow_prev
                if flow_ratio < (1.0 - flow_drop_threshold):
                    events.append(i)

    return events


def detect_narrow_topology(
    masks_uint8: List[np.ndarray],
    narrow_px: int = 8,
    min_narrow_fraction: float = 0.05,
) -> List[int]:
    """Detect frames where the mask has narrow / thin regions (topology attack targets).

    Uses distance transform: if the max inscribed radius is small relative to
    total area, the object has narrow parts.

    Args:
        masks_uint8: list of [H, W] binary masks.
        narrow_px: threshold for "narrow" (max distance transform value).
        min_narrow_fraction: minimum fraction of boundary pixels that are narrow.

    Returns:
        List of frame indices with narrow topological features.
    """
    events = []
    for i, mask in enumerate(masks_uint8):
        if mask.sum() < 100:
            continue
        mask_u8 = mask.astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        max_dist = dist.max()

        if max_dist < narrow_px:
            # Entire object is thin
            events.append(i)
            continue

        # Check fraction of mask pixels in narrow corridors (small distance)
        narrow_pixels = (dist > 0) & (dist < narrow_px // 2)
        mask_pixels = mask_u8.astype(bool).sum()
        if mask_pixels > 0:
            narrow_frac = float(narrow_pixels.sum()) / float(mask_pixels)
            if narrow_frac > 0.30:
                events.append(i)

    return events


def detect_scene_changes(
    frames_uint8: List[np.ndarray],
    hist_threshold: float = 0.5,
) -> List[int]:
    """Detect scene changes via histogram comparison.

    Args:
        frames_uint8: list of [H, W, 3] uint8 frames.
        hist_threshold: correlation drop threshold.

    Returns:
        List of frame indices at scene boundaries.
    """
    events = []
    prev_hist = None
    for i, frame in enumerate(frames_uint8):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < hist_threshold:
                events.append(i)

        prev_hist = hist

    return events


def analyze_video(
    frames_uint8: List[np.ndarray],
    masks_uint8: List[np.ndarray],
    enable_occlusion: bool = True,
    enable_topology: bool = True,
    occlusion_flow_threshold: float = 0.3,
    topology_narrow_px: int = 8,
) -> dict:
    """Run full content analysis on a video.

    Returns:
        dict with keys:
            'flow_mags': per-frame optical flow magnitudes
            'occlusion_events': frame indices of occlusion events
            'topology_events': frame indices with narrow topology
            'scene_changes': frame indices of scene changes
            'all_events': merged and deduplicated event list
    """
    flow_mags = compute_optical_flow_magnitudes(frames_uint8)

    occlusion_events = []
    if enable_occlusion:
        occlusion_events = detect_occlusion_events(
            masks_uint8, flow_mags,
            flow_drop_threshold=occlusion_flow_threshold,
        )

    topology_events = []
    if enable_topology:
        topology_events = detect_narrow_topology(
            masks_uint8, narrow_px=topology_narrow_px,
        )

    scene_changes = detect_scene_changes(frames_uint8)

    # Merge and deduplicate
    all_events = sorted(set(occlusion_events + topology_events + scene_changes))

    return {
        "flow_mags": flow_mags,
        "occlusion_events": occlusion_events,
        "topology_events": topology_events,
        "scene_changes": scene_changes,
        "all_events": all_events,
    }
