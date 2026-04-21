#!/bin/bash
# Extract 24fps JPEG frames from SA-V test split to match UAPSAM paper convention.
# Paper/repo hardcode: target_image_dir = './data/sav_test/JPEGImages_24fps' (confirmed
# from CGCL-codes/UAP-SAM2 public uap_attack.py and local originals).
#
# Input:  data/sav_test_dl/sav_test.tar  (17.7 GB SA-V test split, contains .mp4 + .json)
# Output: data/sav_test/JPEGImages_24fps/{sav_XXXXXX}/{NNNNN}.jpg
#
# Only processes FEA_NUM videos; we only need fea_num=30 distractor folders.
# Extracts 5 frames per video at 5-frame stride (matches existing fake-sav dir layout).

set -euo pipefail

cd /IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2

FEA_NUM=${FEA_NUM:-35}       # grab 35 so we have 30+buffer; some may fail
STRIDE=${STRIDE:-5}
FRAMES_PER_VIDEO=${FRAMES_PER_VIDEO:-5}
SRC_TAR="data/sav_test_dl/sav_test.tar"
EXTRACT_DIR="data/sav_test_dl/extracted"
OUT_ROOT="data/sav_test/JPEGImages_24fps"
FFMPEG="/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/envs/memshield/bin/ffmpeg"

[ -f "$SRC_TAR" ] || { echo "ERROR: $SRC_TAR not found"; exit 1; }
[ -x "$FFMPEG" ] || { echo "ERROR: $FFMPEG not executable"; exit 1; }

mkdir -p "$EXTRACT_DIR" "$OUT_ROOT"

echo "[1/3] list tar contents and pick first $FEA_NUM mp4s"
tar -tf "$SRC_TAR" | grep '\.mp4$' | sort | head -n "$FEA_NUM" > /tmp/sav_mp4_list.txt
wc -l /tmp/sav_mp4_list.txt

echo "[2/3] extract only those mp4s from tar (fast, skips JSON)"
tar -xf "$SRC_TAR" -C "$EXTRACT_DIR" -T /tmp/sav_mp4_list.txt

echo "[3/3] ffmpeg -> JPEGImages_24fps/{id}/{NNNNN}.jpg"
count=0
while read -r mp4_path; do
    video_id=$(basename "$mp4_path" .mp4)
    src="$EXTRACT_DIR/$mp4_path"
    dst_dir="$OUT_ROOT/$video_id"
    if [ ! -f "$src" ]; then
        echo "  SKIP $video_id (file missing after extract)"
        continue
    fi
    mkdir -p "$dst_dir"
    # Extract frames at native fps, keep every STRIDE-th, cap at FRAMES_PER_VIDEO.
    # Frame names: 00000.jpg, 00005.jpg, ... matching fake-sav layout.
    "$FFMPEG" -hide_banner -loglevel error -i "$src" \
        -vf "select='not(mod(n\,$STRIDE))',setpts=N/FRAME_RATE/TB" \
        -vsync vfr -frames:v "$FRAMES_PER_VIDEO" \
        -q:v 2 \
        -start_number 0 \
        "$dst_dir/%05d.jpg"
    # Rename sequentially-numbered output to stride-numbered
    idx=0
    for f in $(ls "$dst_dir"/*.jpg 2>/dev/null | sort); do
        new_idx=$(printf "%05d" $((idx * STRIDE)))
        mv "$f" "$dst_dir/${new_idx}.jpg.tmp" 2>/dev/null
        idx=$((idx + 1))
    done
    # Remove .tmp suffix
    for f in "$dst_dir"/*.tmp; do [ -f "$f" ] && mv "$f" "${f%.tmp}"; done
    rm -f "$src"  # reclaim space as we go
    count=$((count + 1))
    if [ $((count % 5)) -eq 0 ]; then
        echo "  [$count/$FEA_NUM] $video_id -> $(ls "$dst_dir" | wc -l) frames"
    fi
done < /tmp/sav_mp4_list.txt

echo "=== DONE ==="
echo "Videos processed: $count"
echo "Dir: $OUT_ROOT"
ls "$OUT_ROOT" | head -5
echo "Total folders: $(ls "$OUT_ROOT" | wc -l)"
echo "Sample first-folder files:"
ls "$OUT_ROOT/$(ls "$OUT_ROOT" | head -1)" 2>/dev/null | head -10
echo ""
echo "Disk usage of $OUT_ROOT:"
du -sh "$OUT_ROOT"
echo ""
echo "Cleanup suggestion: rm -rf $EXTRACT_DIR  # after verification"
