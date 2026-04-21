"""
uap_eval_heldout_jpeg.py  —  UAP-SAM2 Held-Out JPEG Baseline
=============================================================
命名：UAP-SAM2 held-out JPEG baseline
不是：strict reproduction / official default-protocol baseline

与官方 uap_atk_test.py 的唯一区别：
  1. --test_dataset YOUTUBE_VALID  →  使用 YouTube-VOS valid split（held-out）
     --test_dataset YOUTUBE        →  使用 train split（复现官方默认行为）
  2. 保存 eval video IDs，加载 train video IDs，计算并记录 overlap
  3. 明确日志标注：JPEG-eval baseline，非 lossless eval

保持不变：
  - UAP 方法本身（训练、损失、扰动预算）
  - JPEG save/reload 路径（有意保留）
  - mIoU 指标（含 clean<0.3 过滤，与官方一致）

使用方法（sanity）：
  python uap_eval_heldout_jpeg.py --test_dataset YOUTUBE_VALID \\
      --limit_img 5 --limit_frames 5 --test_prompts pt \\
      --checkpoints sam2-t --seed 30 --P_num 10 --prompts_num 256

使用方法（完整 held-out JPEG baseline）：
  python uap_eval_heldout_jpeg.py --test_dataset YOUTUBE_VALID \\
      --limit_img 100 --limit_frames 15 --test_prompts pt \\
      --checkpoints sam2-t --seed 30 --P_num 10 --prompts_num 256
"""
import json
import os
from argparse import ArgumentParser, Namespace
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sam2_util import (get_frame_index, save_image_only, choose_dataset,
                       choose_heldout_dataset, collate_fn,
                       get_video_to_indices, load_model, process_videos_test)
from attack_setting import seed_everything

# ── Train video IDs 文件路径（由 training 时写入）────────────────────────────
TRAIN_VIDEO_IDS_PATH = (
    Path(__file__).parent.parent /
    "sam2_privacy_preprocessor/refine-logs/train_video_ids.json"
    if (Path(__file__).parent.parent /
        "sam2_privacy_preprocessor/refine-logs/train_video_ids.json").exists()
    else Path("/IMBR_Data/Student-home/2025M_LvShaoting/"
              "sam2_privacy_preprocessor/refine-logs/train_video_ids.json")
)


def select_dataset(args):
    """Route --test_dataset to the correct split.

    YOUTUBE       →  train split  (官方默认行为，in-sample)
    YOUTUBE_VALID →  valid split  (held-out)
    """
    if args.test_dataset == "YOUTUBE_VALID":
        print("[dataset] Using YouTube-VOS VALID split (held-out)")
        return choose_heldout_dataset(args)
    else:
        print("[dataset] Using YouTube-VOS TRAIN split (in-sample, official default)")
        return choose_dataset(args)


def check_overlap(eval_video_ids: list, args) -> dict:
    """加载 train video IDs，计算 overlap，返回结果 dict。"""
    result = {
        "eval_split": args.test_dataset,
        "eval_video_count": len(eval_video_ids),
        "train_video_ids_file": str(TRAIN_VIDEO_IDS_PATH),
        "train_video_ids_found": False,
        "overlap": None,
        "overlap_pct": None,
    }
    if not TRAIN_VIDEO_IDS_PATH.exists():
        print(f"[overlap] WARNING: train_video_ids file not found at {TRAIN_VIDEO_IDS_PATH}")
        print("[overlap] Cannot verify overlap — skipping check")
        return result

    d = json.loads(TRAIN_VIDEO_IDS_PATH.read_text())
    train_ids = set(d["video_ids"])
    eval_ids = set(eval_video_ids)
    overlap = train_ids & eval_ids
    result.update({
        "train_video_ids_found": True,
        "train_video_count": len(train_ids),
        "overlap": len(overlap),
        "overlap_pct": round(len(overlap) / max(len(eval_ids), 1) * 100, 1),
        "overlap_ids": sorted(list(overlap))[:10],  # first 10 for log
    })
    return result


def run(args, custom_dataset):
    device = "cuda:1"
    sam_fwder, predictor = load_model(args, device=device)

    video_to_indices = get_video_to_indices(custom_dataset)
    total_miou_clean = 0.0
    total_miou_adv = 0.0
    iou_count = 0
    iou_count_adv = 0
    total_jf_clean = 0.0;  total_j_clean = 0.0;  total_f_clean = 0.0
    total_jf_adv   = 0.0;  total_j_adv   = 0.0;  total_f_adv   = 0.0
    jf_count_clean = 0;    jf_count_adv = 0
    video_count = 0
    printed_videos = set()
    denorm = lambda x: sam_fwder.denorm_image(x)

    total_frame_count = 0
    mask_gt_dict_all = {}
    start_P_dict_all = {}
    video_result_paths = {}
    video_result_cleans = {}

    # ── Overlap check (fail-fast for held-out mode) ───────────────────────────
    eval_video_ids = list(video_to_indices.keys())
    overlap_info = check_overlap(eval_video_ids, args)
    print("\n[overlap-check]")
    print(f"  eval split:        {overlap_info['eval_split']}")
    print(f"  eval video count:  {overlap_info['eval_video_count']}")
    if overlap_info["train_video_ids_found"]:
        print(f"  train video count: {overlap_info['train_video_count']}")
        print(f"  overlap:           {overlap_info['overlap']} / {overlap_info['eval_video_count']} "
              f"({overlap_info['overlap_pct']}%)")
        if args.test_dataset == "YOUTUBE_VALID" and overlap_info["overlap"] > 0:
            print(f"  [WARN] Non-zero overlap in held-out mode! Overlap IDs: {overlap_info['overlap_ids']}")
        elif overlap_info["overlap"] == 0:
            print("  [OK] overlap = 0  →  eval is genuinely held-out")
        else:
            print("  [INFO] in-sample eval  →  overlap is expected")
    print()

    # ── Phase 1: apply UAP and save frames as JPEG ────────────────────────────
    # NOTE: 有意保留 JPEG save/reload 路径（模拟发布链路压缩）
    # 这是 JPEG-eval baseline，不是 lossless eval。
    print("[format] Frame format: JPEG (lossy, simulates publish-path compression)")
    print("[format] This is JPEG-eval baseline — NOT lossless eval\n")

    adv_root = f"./adv_heldout_jpeg/{args.test_dataset}"
    clean_root = f"./clean_heldout_jpeg/{args.test_dataset}"

    for video_name, indices in video_to_indices.items():
        print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

        uap_path = "uap_file/YOUTUBE_fixed.pth"
        if not os.path.exists(uap_path):
            print(f"UAP file not found at {uap_path}, skipping {video_name}...")
            continue
        uap = torch.load(uap_path, map_location=device)

        video_subset = Subset(custom_dataset, indices)
        video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn)
        video_result_path = f"{adv_root}/{video_name}"
        video_result_clean = f"{clean_root}/{video_name}"

        mask_gt_dict = {}
        frame_counter = {}
        start_P_dict = {}
        start_frame_idx = None
        start_point = None

        for batch in tqdm(video_loader):
            total_frame_count += 1
            images, P_list, sample_ids, mask_gt_list, point = batch
            image = images[0]
            mask_gt = mask_gt_list[0]
            img_ID = sample_ids[0]
            vname = img_ID.split("/")[0]

            frame_idx = get_frame_index(img_ID)
            print(f"Processing frame {frame_idx} from video {vname}")

            if vname not in printed_videos:
                start_frame_idx = custom_dataset.get_start_frame_idx(vname)
                printed_videos.add(vname)

            if vname not in frame_counter:
                frame_counter[vname] = 0

            indexed_frame_idx = frame_counter[vname]
            if frame_idx == start_frame_idx:
                start_point = point[0]
                print("Load new video and initialize memory........................")

            if vname not in start_P_dict and start_point is not None:
                start_P_dict[vname] = start_point

            X = sam_fwder.transform_image(image).to(device)
            benign_img = denorm(X).to(device)
            adv_img = torch.clamp(benign_img + uap, 0, 1)

            adv_image = adv_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            adv_image = (adv_image * 255).astype("uint8")
            # JPEG 保存（有意保留，use_png=False 为默认值）
            save_image_only(adv_image, vname, frame_idx, video_result_path, use_png=False)

            im = benign_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            im = (im * 255).astype("uint8")
            save_image_only(im, vname, frame_idx, video_result_clean, use_png=False)

            frame_counter[vname] += 1
            mask_gt_dict[(vname, indexed_frame_idx)] = mask_gt

        mask_gt_dict_all.update(mask_gt_dict)
        start_P_dict_all.update(start_P_dict)
        video_result_paths[video_name] = video_result_path
        video_result_cleans[video_name] = video_result_clean
        video_count += 1

    # ── Phase 2: SAM2 inference and mIoU ─────────────────────────────────────
    for vname in video_result_paths.keys():
        miou_clean, iou_count_video, skipped_frames_video, jf_clean, j_clean, f_clean = process_videos_test(
            video_result_cleans[vname],
            clean_root,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="clean",
            args=args)

        miou_adv, iou_count_adv_video, _, jf_adv, j_adv, f_adv = process_videos_test(
            video_result_paths[vname],
            adv_root,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="adversarial",
            skipped_frames=skipped_frames_video,
            args=args)

        total_miou_clean += miou_clean * iou_count_video
        total_miou_adv += miou_adv * iou_count_adv_video
        iou_count += iou_count_video
        iou_count_adv += iou_count_adv_video
        total_jf_clean += jf_clean * iou_count_video
        total_j_clean  += j_clean  * iou_count_video
        total_f_clean  += f_clean  * iou_count_video
        jf_count_clean += iou_count_video
        total_jf_adv += jf_adv * iou_count_adv_video
        total_j_adv  += j_adv  * iou_count_adv_video
        total_f_adv  += f_adv  * iou_count_adv_video
        jf_count_adv += iou_count_adv_video

    avg_miou_clean = total_miou_clean / iou_count if iou_count > 0 else 0
    avg_miou_adv = total_miou_adv / iou_count_adv if iou_count_adv > 0 else 0
    avg_jf_clean = total_jf_clean / jf_count_clean if jf_count_clean > 0 else 0.0
    avg_j_clean  = total_j_clean  / jf_count_clean if jf_count_clean > 0 else 0.0
    avg_f_clean  = total_f_clean  / jf_count_clean if jf_count_clean > 0 else 0.0
    avg_jf_adv   = total_jf_adv   / jf_count_adv   if jf_count_adv   > 0 else 0.0
    avg_j_adv    = total_j_adv    / jf_count_adv   if jf_count_adv   > 0 else 0.0
    avg_f_adv    = total_f_adv    / jf_count_adv   if jf_count_adv   > 0 else 0.0
    print(f"iou_count: {iou_count}")
    print(f"[jf] clean: JF={avg_jf_clean*100:.2f}%  J={avg_j_clean*100:.2f}%  F={avg_f_clean*100:.2f}%")
    print(f"[jf]   adv: JF={avg_jf_adv*100:.2f}%  J={avg_j_adv*100:.2f}%  F={avg_f_adv*100:.2f}%")

    return (video_count, avg_miou_clean, avg_miou_adv,
            iou_count, iou_count_adv, total_frame_count,
            eval_video_ids, overlap_info,
            avg_jf_clean, avg_j_clean, avg_f_clean,
            avg_jf_adv,   avg_j_adv,   avg_f_adv)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="UAP-SAM2 held-out JPEG baseline eval  "
                    "(--test_dataset YOUTUBE_VALID = held-out, YOUTUBE = in-sample)"
    )
    parser.add_argument("--limit_img", default=100, type=int)
    parser.add_argument("--limit_frames", default=15, type=int)
    parser.add_argument("--train_dataset", default="YOUTUBE")
    parser.add_argument(
        "--test_dataset", default="YOUTUBE_VALID",
        choices=["YOUTUBE", "YOUTUBE_VALID"],
        help="YOUTUBE = train split (in-sample); YOUTUBE_VALID = valid split (held-out)")
    parser.add_argument("--point")
    parser.add_argument("--train_prompts", choices=["bx", "pt"], default="pt")
    parser.add_argument("--test_prompts", choices=["bx", "pt"], default="pt")
    parser.add_argument("--checkpoints", default="sam2-t")
    parser.add_argument("--fea_num", default=30, type=int)
    parser.add_argument("--weight_fea", default=0.000001, type=float)
    parser.add_argument("--seed", default=30, type=int)
    parser.add_argument("--eps", default=10, type=int)
    parser.add_argument("--P_num", default=10, type=int)
    parser.add_argument("--prompts_num", default=256, type=int)
    parser.add_argument("--save_img_with_mask", action="store_true")
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--loss_fea", action="store_true")
    parser.add_argument("--loss_t", action="store_true")
    parser.add_argument("--loss_diff", action="store_true")
    return parser


def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args


if __name__ == "__main__":
    parser = get_parser()
    args = get_args(parser)
    seed_everything(args.seed)

    print("=== UAP-SAM2 Held-Out JPEG Baseline Eval ===")
    print(f"  --test_dataset:  {args.test_dataset}")
    print(f"  --limit_img:     {args.limit_img}")
    print(f"  --limit_frames:  {args.limit_frames}")
    print(f"  frame format:    JPEG (lossy)  ← JPEG-eval baseline, NOT lossless eval")
    print(f"  mIoU:            filtered (clean<0.3 skipped) — same as official code")
    print()

    custom_dataset = select_dataset(args)

    (video_test, miouimg, miouadv,
     frames_clean, frames_adv, frames_train,
     eval_video_ids, overlap_info,
     jf_clean, j_clean, f_clean,
     jf_adv, j_adv, f_adv) = run(args, custom_dataset)

    # ── Save eval video IDs ───────────────────────────────────────────────────
    eval_ids_path = Path(
        "/IMBR_Data/Student-home/2025M_LvShaoting/"
        "sam2_privacy_preprocessor/refine-logs/eval_heldout_jpeg_video_ids.json"
    )
    eval_ids_path.parent.mkdir(parents=True, exist_ok=True)
    eval_ids_path.write_text(json.dumps({
        "split": args.test_dataset,
        "seed": args.seed,
        "limit_img": args.limit_img,
        "video_ids": sorted(eval_video_ids),
    }, indent=2))

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "baseline_name": "UAP-SAM2 held-out JPEG baseline",
        "test_dataset": args.test_dataset,
        "frame_format": "JPEG",
        "limit_img": args.limit_img,
        "limit_frames": args.limit_frames,
        "seed": args.seed,
        "miou_clean_pct": round(miouimg * 100, 2),
        "miou_adv_pct": round(miouadv * 100, 2),
        "jf_clean_pct":  round(jf_clean * 100, 2),
        "j_clean_pct":   round(j_clean  * 100, 2),
        "f_clean_pct":   round(f_clean  * 100, 2),
        "jf_adv_pct":    round(jf_adv   * 100, 2),
        "j_adv_pct":     round(j_adv    * 100, 2),
        "f_adv_pct":     round(f_adv    * 100, 2),
        "video_count": video_test,
        "frames_clean": frames_clean,
        "frames_adv": frames_adv,
        "overlap_with_train": overlap_info.get("overlap"),
        "overlap_pct": overlap_info.get("overlap_pct"),
    }
    results_path = Path(
        "/IMBR_Data/Student-home/2025M_LvShaoting/"
        "sam2_privacy_preprocessor/refine-logs/heldout_jpeg_results.json"
    )
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n[overlap-summary] overlap with train = "
          f"{overlap_info.get('overlap', 'N/A')} / {len(eval_video_ids)} "
          f"({overlap_info.get('overlap_pct', 'N/A')}%)")
    print(f"[format] JPEG-eval baseline  (NOT lossless eval)")
    print(f":: jf_clean: {jf_clean*100:.2f}%  j_clean: {j_clean*100:.2f}%  f_clean: {f_clean*100:.2f}%")
    print(f":: jf_adv:   {jf_adv*100:.2f}%  j_adv:   {j_adv*100:.2f}%  f_adv:   {f_adv*100:.2f}%  delta_jf: {(jf_adv-jf_clean)*100:.2f}pp")
    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %, "
          f"video_test: {video_test} , frame_clean_test:{frames_clean} , "
          f"frames_adv_test:{frames_adv},frame_train:{frames_train}")
    import sys; sys.stdout.flush()
