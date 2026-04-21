"""Apply trained UAPSAM universal perturbation to DAVIS clips.

Reproduces UAPSAM's `uap_eval_heldout_jpeg.py` attack-side logic:
  X = transform_image(frame)             # resize to 1024x1024 + normalize
  benign = denormalize(X)                # back to [0,1]
  adv = clamp(benign + uap, 0, 1)        # add UAP
  save adv as JPEG at 1024x1024

Outputs `{output_dir}/videos/{vid}_uapsam/00000.jpg`... per clip so that the
downstream SAM2/SAM2Long eval pipelines can point to these dirs directly.

Expected: pure input-space transformation, no backward pass, CPU-friendly
(GPU just speeds up the transform).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UAP_SAM2_ROOT = Path("/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2")
sys.path.insert(0, str(UAP_SAM2_ROOT))

ROOT = Path(__file__).resolve().parent.parent

# SAM2 normalization constants (match SAM2AutomaticMaskGenerator / image_predictor)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SAM2_INPUT_SIZE = 1024


def transform_image(img_uint8, device):
    """Resize to 1024x1024 and normalize (ImageNet stats)."""
    import torch.nn.functional as F
    x = torch.from_numpy(img_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    x = F.interpolate(x, size=(SAM2_INPUT_SIZE, SAM2_INPUT_SIZE),
                      mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x


def denormalize(x, device):
    return x * IMAGENET_STD.to(device) + IMAGENET_MEAN.to(device)


def apply_uap_to_clip(uap, frame_dir, out_dir, device, max_frames=None):
    """Load frames from frame_dir, apply UAP, save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frame_dir.iterdir())
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    for i, p in enumerate(frame_paths):
        img = np.array(Image.open(p).convert("RGB"))
        x = transform_image(img, device)
        benign = denormalize(x, device)
        adv = torch.clamp(benign + uap, 0, 1)
        adv_np = (adv.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_path = out_dir / f"{i:05d}.jpg"
        Image.fromarray(adv_np).save(out_path, quality=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uap_path",
                    default=str(UAP_SAM2_ROOT / "uap_file" / "YOUTUBE.pth"))
    ap.add_argument("--davis_root", default=str(ROOT / "data" / "davis"))
    ap.add_argument("--output_dir", default=str(ROOT / "results_uapsam"))
    ap.add_argument("--videos", required=True,
                    help="Comma-separated DAVIS video names")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    uap = torch.load(args.uap_path, map_location=device, weights_only=True)
    assert uap.shape == (1, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE), f"Unexpected UAP shape {uap.shape}"
    print(f"Loaded UAP: {uap.shape}, |abs_max|={uap.abs().max().item():.4f}")

    out_videos_root = Path(args.output_dir) / "videos"
    out_videos_root.mkdir(parents=True, exist_ok=True)

    davis_jpeg = Path(args.davis_root) / "JPEGImages" / "480p"
    for vid in args.videos.split(","):
        frame_dir = davis_jpeg / vid
        out_dir = out_videos_root / f"{vid}_uapsam"
        print(f"Applying UAP to {vid}...")
        apply_uap_to_clip(uap, frame_dir, out_dir, device)
        n = len(list(out_dir.iterdir()))
        print(f"  saved {n} frames to {out_dir}")

    print(f"\nDone. Output at {out_videos_root}")


if __name__ == "__main__":
    main()
