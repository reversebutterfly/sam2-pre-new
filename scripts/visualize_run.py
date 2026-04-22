"""Dump a MemoryShield v2 run as JPEGs (+ optional MP4) for eyeball QA.

Input:
    --run_dir runs/<run_id>/   containing
        modified_video.npy   [T_mod, H, W, 3] uint8
        final_delta.npy      [T_orig, H, W, 3] float (prefix perturbation)
        diagnostics.json     (clip + schedule)

Output (written to <run_dir>/viz/):
    modified/frame_{m:04d}.jpg   — each frame of modified_video.npy
                                    (filename has `_INSERT` suffix for slots
                                    that are inserts, so you can scroll and
                                    see whether ProPainter inserts look plausible)
    diff/frame_{o:04d}.jpg       — |modified[m] - orig[o]| × 20 for every
                                    non-insert slot, so you can see the δ
                                    distribution. Invisible at eps=4/255 means
                                    this should be very faint.
    modified.mp4                 — optional, only if ffmpeg is on PATH and
                                    --mp4 is passed.

This is a sanity visualizer, not a paper figure. It exists so you can scroll
through and catch obvious bugs (all-black inserts, white-noise δ, wrong
frame order).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_davis_frames(davis_root: Path, clip: str, n_frames: int) -> np.ndarray:
    jpg_dir = davis_root / "JPEGImages" / "480p" / clip
    jpgs = sorted(jpg_dir.glob("*.jpg"))[:n_frames]
    if len(jpgs) < n_frames:
        raise RuntimeError(f"{clip}: got {len(jpgs)} JPGs, need {n_frames}")
    return np.stack([np.array(Image.open(p).convert("RGB")) for p in jpgs])


def save_jpg(arr: np.ndarray, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=quality)


def ffmpeg_jpg_to_mp4(jpg_dir: Path, out_mp4: Path, fps: int = 8) -> bool:
    """Best-effort MP4 assembly. Returns False if ffmpeg not found."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Try CLAUDE.md local ffmpeg path as a fallback.
        for cand in [r"C:\ffmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg"]:
            if Path(cand).exists():
                ffmpeg = cand
                break
    if ffmpeg is None:
        print("[viz] ffmpeg not found on PATH; skipping MP4")
        return False
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(jpg_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # even dims for libx264
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[viz] ffmpeg failed: {e.stderr.decode(errors='ignore')[:200]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--davis_root", default=str(REPO_ROOT / "data" / "davis"))
    ap.add_argument("--diff_scale", type=float, default=20.0,
                    help="δ visualization gain (eps=4/255 becomes ~0.31 after ×20)")
    ap.add_argument("--mp4", action="store_true",
                    help="Also assemble modified.mp4 via ffmpeg if available")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out_dir", default=None,
                    help="Default: <run_dir>/viz")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    mod_path = run_dir / "modified_video.npy"
    diag_path = run_dir / "diagnostics.json"
    assert mod_path.exists(), f"missing {mod_path}"
    assert diag_path.exists(), f"missing {diag_path}"

    diag = json.loads(diag_path.read_text())
    clip = diag["clip"]
    K_ins = int(diag["K_ins"])
    w_positions: List[int] = list(diag["schedule"]["w_positions"])
    T_mod = int(diag["schedule"]["T_mod"])
    T_prefix_orig = T_mod - K_ins

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    mod_dir = out_dir / "modified"
    diff_dir = out_dir / "diff"
    seq_dir = out_dir / "_seq_for_mp4"
    # Clear stale output subdirs from prior runs so a rerun with smaller T_mod
    # (or a prior ffmpeg failure) can't leave stragglers that confuse the MP4
    # encoder or make diff/modified look out of sync.
    for d in (mod_dir, diff_dir, seq_dir):
        if d.exists():
            shutil.rmtree(d)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod_video = np.load(mod_path)
    assert mod_video.dtype == np.uint8
    assert mod_video.shape[0] == T_mod, (
        f"modified_video T={mod_video.shape[0]} != diagnostics.T_mod={T_mod}")
    H, W = mod_video.shape[1], mod_video.shape[2]
    print(f"[viz] clip={clip} T_mod={T_mod} K_ins={K_ins} HxW={H}x{W} "
          f"w_positions={w_positions}")

    # --- dump modified frames ------------------------------------------------
    insert_set = set(w_positions)
    for m in range(T_mod):
        tag = "INSERT" if m in insert_set else "orig"
        path = mod_dir / f"frame_{m:04d}_{tag}.jpg"
        save_jpg(mod_video[m], path)
    print(f"[viz] wrote {T_mod} modified JPGs to {mod_dir}")

    # Also copy plain-numbered files for ffmpeg's sequential reader.
    if args.mp4:
        seq_dir.mkdir(parents=True, exist_ok=True)
        for m in range(T_mod):
            Image.fromarray(mod_video[m]).save(
                seq_dir / f"frame_{m:04d}.jpg", quality=95)

    # --- dump δ visualization on original (non-insert) slots ----------------
    davis_root = Path(args.davis_root)
    if davis_root.exists():
        try:
            orig_frames = load_davis_frames(davis_root, clip, n_frames=T_prefix_orig)
        except RuntimeError as e:
            print(f"[viz] skip diff visualization: {e}")
            orig_frames = None
    else:
        print(f"[viz] davis_root not found ({davis_root}); skip diff visualization")
        orig_frames = None

    if orig_frames is not None:
        # Build mod_index -> orig_index for non-insert slots.
        o = 0
        n_diff = 0
        d_max_running = 0.0
        for m in range(T_mod):
            if m in insert_set:
                continue
            if o >= len(orig_frames):
                break
            mod_f = mod_video[m].astype(np.int32)
            orig_f = orig_frames[o].astype(np.int32)
            diff = np.abs(mod_f - orig_f).astype(np.float32)
            scaled = np.clip(diff * args.diff_scale, 0, 255).astype(np.uint8)
            save_jpg(scaled, diff_dir / f"frame_{o:04d}.jpg")
            d_max_running = max(d_max_running, float(diff.max()))
            n_diff += 1
            o += 1
        print(f"[viz] wrote {n_diff} diff JPGs to {diff_dir}; "
              f"max |δ| observed = {d_max_running:.1f} / 255  "
              f"(eps=4/255 threshold → max should be ≤ 4)")

    # --- optional MP4 --------------------------------------------------------
    if args.mp4:
        mp4_path = out_dir / "modified.mp4"
        ok = ffmpeg_jpg_to_mp4(seq_dir, mp4_path, fps=args.fps)
        if ok:
            print(f"[viz] wrote {mp4_path}")
        # keep seq_dir for debugging; remove only on success request
        if ok:
            shutil.rmtree(seq_dir)

    print(f"[viz] done → {out_dir}")


if __name__ == "__main__":
    main()
