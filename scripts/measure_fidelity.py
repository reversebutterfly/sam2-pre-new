"""Measure LPIPS + SSIM + PSNR on saved attacked JPEG sequences vs DAVIS clean.

Handles the standard 3-insert schedule (inserts after orig 3, 7, 11 of a
15-frame attack prefix). Reports:
  - ssim_orig / psnr_orig / lpips_orig: over ATTACKED ORIGINAL frames
    (paired 1:1 with clean DAVIS frames — direct fidelity measurement).
  - ssim_ins / psnr_ins / lpips_ins: over INSERTED frames (paired with
    nearest earlier original clean frame — measures visual coherence of
    the synthetic insert with surrounding scene content).

Usage:
  python scripts/measure_fidelity.py \
    --attacks_dir results_stage2_merged \
    --davis_root data/davis \
    --videos blackswan,bike-packing,...
    --regime decoy \
    --output_json results_stage2_merged/fidelity.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ATTACK_PREFIX = 15  # Must match run_two_regimes.py default
INSERT_MOD_INDICES_15 = [4, 9, 14]  # Inserts after orig 3, 7, 11


def load_jpegs(directory):
    paths = sorted(Path(directory).iterdir())
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def build_mod_to_orig(T_mod, T_orig, attack_prefix=ATTACK_PREFIX):
    """Return (m2o, insert_mod_indices).

    m2o[mi] = orig index, or -1 for insert. For the standard
    attack_prefix=15 + 3-insert schedule, inserts are at mod indices
    {4, 9, 14}. If the frame-count difference is not 3, falls back to
    even spacing over the prefix.
    """
    n_ins = T_mod - T_orig
    if n_ins == 3 and attack_prefix == 15:
        ins = list(INSERT_MOD_INDICES_15)
    else:
        # Even spacing
        ins = []
        step = max(1, (attack_prefix + n_ins) // (n_ins + 1))
        for i in range(n_ins):
            ins.append(min((i + 1) * step + i, attack_prefix + n_ins - 1))
    ins_set = set(ins)
    m2o = []
    oi = 0
    for mi in range(T_mod):
        if mi in ins_set:
            m2o.append(-1)
        else:
            m2o.append(oi)
            oi += 1
    return m2o, sorted(ins_set)


def ssim_torch(a, b, window=11):
    """Mean SSIM for NCHW tensors in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window // 2
    mu_a = F.avg_pool2d(a, window, stride=1, padding=pad)
    mu_b = F.avg_pool2d(b, window, stride=1, padding=pad)
    sig_a = F.avg_pool2d(a * a, window, stride=1, padding=pad) - mu_a ** 2
    sig_b = F.avg_pool2d(b * b, window, stride=1, padding=pad) - mu_b ** 2
    sig_ab = F.avg_pool2d(a * b, window, stride=1, padding=pad) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
    return float((num / den).mean().item())


def psnr(a, b):
    mse = float(((a - b) ** 2).mean().item())
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def to_nchw(img, device):
    """np.uint8 HxWx3 -> torch [1,3,H,W] float in [0,1]."""
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def match_shape(src_img_np, ref_shape):
    """Resize src to ref_shape if needed; returns np uint8."""
    if src_img_np.shape[:2] == ref_shape[:2]:
        return src_img_np
    pil = Image.fromarray(src_img_np).resize(
        (ref_shape[1], ref_shape[0]), Image.BICUBIC)
    return np.array(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks_dir", required=True,
                    help="Parent dir containing videos/{vid}_{regime}/")
    ap.add_argument("--davis_root", required=True)
    ap.add_argument("--videos", required=True, help="Comma-separated")
    ap.add_argument("--regime", default="decoy")
    ap.add_argument("--attack_prefix", type=int, default=ATTACK_PREFIX)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--no_lpips", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)

    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
            lpips_fn.eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print("[OK] LPIPS (alex) loaded")
        except Exception as e:
            print(f"[WARN] LPIPS unavailable: {e}")

    davis_jpeg_root = Path(args.davis_root) / "JPEGImages" / "480p"
    atk_root = Path(args.attacks_dir) / "videos"
    results = {}

    videos = args.videos.split(",")
    for vid in videos:
        clean_dir = davis_jpeg_root / vid
        atk_dir = atk_root / f"{vid}_{args.regime}"
        if not atk_dir.exists():
            print(f"[SKIP] {vid}: {atk_dir} missing")
            continue
        clean = load_jpegs(clean_dir)
        atk = load_jpegs(atk_dir)
        T_mod, T_orig = len(atk), len(clean)
        m2o, ins_mi = build_mod_to_orig(T_mod, T_orig, args.attack_prefix)

        # Make clean match attacked shape
        ref_shape = atk[0].shape
        clean = [match_shape(c, ref_shape) for c in clean]

        print(f"\n=== {vid} ===  T_mod={T_mod} T_orig={T_orig} "
              f"inserts@{ins_mi}")

        so, po, lo = [], [], []
        si, pi, li = [], [], []
        for mi in range(T_mod):
            a = to_nchw(atk[mi], device)
            if m2o[mi] >= 0:
                c = to_nchw(clean[m2o[mi]], device)
                so.append(ssim_torch(a, c))
                po.append(psnr(a, c))
                if lpips_fn is not None:
                    with torch.no_grad():
                        lp = float(lpips_fn(a * 2 - 1, c * 2 - 1).item())
                    lo.append(lp)
            else:
                prev_mi = mi - 1
                while prev_mi >= 0 and m2o[prev_mi] < 0:
                    prev_mi -= 1
                if prev_mi < 0:
                    continue
                c = to_nchw(clean[m2o[prev_mi]], device)
                si.append(ssim_torch(a, c))
                pi.append(psnr(a, c))
                if lpips_fn is not None:
                    with torch.no_grad():
                        lp = float(lpips_fn(a * 2 - 1, c * 2 - 1).item())
                    li.append(lp)

        def mean(xs):
            return float(np.mean(xs)) if xs else float("nan")

        entry = {
            "n_orig": len(so), "n_ins": len(si),
            "ssim_orig": mean(so), "psnr_orig": mean(po),
            "lpips_orig": mean(lo) if lpips_fn is not None else None,
            "ssim_ins": mean(si), "psnr_ins": mean(pi),
            "lpips_ins": mean(li) if lpips_fn is not None else None,
            "insert_mod_indices": ins_mi,
        }
        results[vid] = entry
        print(f"  orig(n={entry['n_orig']}): "
              f"SSIM={entry['ssim_orig']:.4f}  "
              f"PSNR={entry['psnr_orig']:.2f}  "
              f"LPIPS={entry['lpips_orig']}")
        print(f"  ins (n={entry['n_ins']}): "
              f"SSIM={entry['ssim_ins']:.4f}  "
              f"PSNR={entry['psnr_ins']:.2f}  "
              f"LPIPS={entry['lpips_ins']}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output_json, "w"), indent=2)
    print(f"\nWrote {args.output_json}")

    def agg(k):
        xs = [results[v][k] for v in results
              if results[v][k] is not None and not np.isnan(results[v][k])]
        return float(np.mean(xs)) if xs else float("nan")

    print("\nAGGREGATE over", len(results), "videos:")
    print(f"  attacked-originals:  SSIM={agg('ssim_orig'):.4f}  "
          f"PSNR={agg('psnr_orig'):.2f}  LPIPS={agg('lpips_orig'):.4f}")
    print(f"  inserted-frames:     SSIM={agg('ssim_ins'):.4f}  "
          f"PSNR={agg('psnr_ins'):.2f}  LPIPS={agg('lpips_ins'):.4f}")


if __name__ == "__main__":
    main()
