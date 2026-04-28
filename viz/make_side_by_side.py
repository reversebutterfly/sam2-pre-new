"""
Side-by-side visualization of clean vs attacked video frames.
Top row: clean DAVIS original
Bottom row: VADI v4.1 processed (with K=3 inserted decoy frames + bridge δ)
Insert positions are highlighted with a red border + "DECOY" label.
Per-frame J_drop annotation overlays the bottom strip.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VIZ_DIR = Path(__file__).parent
CLIP = sys.argv[1] if len(sys.argv) > 1 else "dog"
ATTACKED_DIR = VIZ_DIR / CLIP
CLEAN_DIR = VIZ_DIR / f"{CLIP}_clean"
# results.json may contain literal NaN (Python JSON allows it)
_text = (ATTACKED_DIR / "results.json").read_text(encoding="utf-8")
RESULTS = json.loads(_text.replace("NaN", "null"))

W = sorted(RESULTS["W"])  # insert positions in PROCESSED stream
PER_FRAME = RESULTS["exported_j_drop_details"]["per_frame"]

# Build mapping processed_idx -> original_davis_idx (None for INSERT positions)
def build_index_map(W, n_proc):
    is_insert = [False] * n_proc
    for w in W:
        is_insert[w] = True
    proc_to_orig = []
    o = 0
    for p in range(n_proc):
        if is_insert[p]:
            proc_to_orig.append(None)  # synthetic
        else:
            proc_to_orig.append(o)
            o += 1
    return proc_to_orig, is_insert


# Font (fall back to default if Arial missing)
def get_font(size):
    for path in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",  # Chinese support
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_frame(p_idx, attacked_path, clean_path_or_none, is_insert, j_drop, w_label):
    pad = 20
    label_h = 40
    target_w = 480

    att_img = Image.open(attacked_path).convert("RGB")
    aw, ah = att_img.size
    scale = target_w / aw
    new_h = int(ah * scale)
    att_img = att_img.resize((target_w, new_h))

    if clean_path_or_none is not None:
        clean_img = Image.open(clean_path_or_none).convert("RGB")
        clean_img = clean_img.resize((target_w, new_h))
    else:
        # blank gray panel saying "(no clean equivalent — INSERTED frame)"
        clean_img = Image.new("RGB", (target_w, new_h), (40, 40, 40))
        d = ImageDraw.Draw(clean_img)
        f = get_font(18)
        d.text((20, new_h // 2 - 10),
               "(synthetic decoy — no original counterpart)",
               fill=(220, 220, 220), font=f)

    canvas_w = target_w + 2 * pad
    canvas_h = label_h + new_h + label_h + new_h + label_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(20)
    label_font = get_font(16)
    big_font = get_font(22)

    # top label
    draw.text((pad, pad // 2), "CLEAN ORIGINAL (DAVIS)", fill=(0, 0, 0), font=title_font)
    canvas.paste(clean_img, (pad, label_h))

    # bottom label
    btm_y = label_h + new_h + 5
    if is_insert:
        draw.text((pad, btm_y), "ATTACKED — INSERTED DECOY FRAME",
                  fill=(180, 0, 0), font=title_font)
    else:
        draw.text((pad, btm_y),
                  f"ATTACKED — bridge-δ region" if w_label else "ATTACKED",
                  fill=(0, 0, 0), font=title_font)
    canvas.paste(att_img, (pad, btm_y + label_h - 5))

    # red border on attacked panel if INSERT
    if is_insert:
        bw = 6
        x0 = pad - bw // 2
        y0 = btm_y + label_h - 5 - bw // 2
        x1 = pad + target_w + bw // 2
        y1 = btm_y + label_h - 5 + new_h + bw // 2
        draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 30), width=bw)

    # bottom annotation: per-frame J_drop, processed idx, mapping
    annot_y = btm_y + label_h - 5 + new_h + 8
    if is_insert:
        annot = f"proc[{p_idx:02d}] = INSERT (W={W})  | J_drop n/a"
    else:
        annot = (f"proc[{p_idx:02d}] = orig[{w_label:02d}]  "
                 f"| J_baseline={j_drop['J_baseline']:.3f}  "
                 f"J_attacked={j_drop['J_attacked']:.3f}  "
                 f"ΔJ={j_drop['J_drop']:+.3f}")
    color = (200, 0, 0) if (not is_insert and j_drop['J_drop'] > 0.3) else (0, 0, 0)
    draw.text((pad, annot_y), annot, fill=color, font=label_font)

    # global summary banner at top-right
    j_drop_mean = RESULTS["exported_j_drop_details"]["J_drop_mean"]
    summary = (f"clip={CLIP}  W={W}  ΔJ_mean={j_drop_mean:.3f}  "
               f"K={len(W)}  T_proc={len(PER_FRAME)}")
    bbox = draw.textbbox((0, 0), summary, font=label_font)
    sw = bbox[2] - bbox[0]
    draw.text((canvas_w - sw - pad, pad // 2), summary,
              fill=(80, 80, 80), font=label_font)

    return canvas


def main():
    n_proc = len(PER_FRAME)
    proc_to_orig, is_insert_arr = build_index_map(W, n_proc)
    out_dir = VIZ_DIR / f"{CLIP}_sbs_frames"
    out_dir.mkdir(exist_ok=True)
    for p in range(n_proc):
        att_p = ATTACKED_DIR / f"frame_{p:04d}.png"
        if not att_p.exists():
            print(f"missing attacked frame: {att_p}")
            continue
        orig_idx = proc_to_orig[p]
        is_ins = is_insert_arr[p]
        clean_p = None
        if orig_idx is not None:
            cp = CLEAN_DIR / f"{orig_idx:05d}.jpg"
            if cp.exists():
                clean_p = cp
        per_frame_key = str(p)
        j_drop = PER_FRAME.get(per_frame_key, {})
        frame = make_frame(p, att_p, clean_p, is_ins, j_drop, orig_idx)
        out_path = out_dir / f"sbs_{p:04d}.png"
        frame.save(out_path)
    print(f"wrote {n_proc} side-by-side frames to {out_dir}")


if __name__ == "__main__":
    main()
