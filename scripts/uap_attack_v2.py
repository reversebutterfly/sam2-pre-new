import copy
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from imagecorruptions import corrupt
from torch.autograd import grad
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm, trange


from sam2_util import get_frame_index, collate_fn, choose_dataset, get_video_to_indices, load_model, \
    get_fused_prototype, infonce_loss
from attack_setting import SamForwarder, make_multi_prompts,seed_everything
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Your script description here")
    parser.add_argument('--limit_img', default=100, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--limit_frames', default=15, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--fea_num', default=30, type=int)
    parser.add_argument('--train_dataset', default='YOUTUBE')
    parser.add_argument('--test_dataset', default='YOUTUBE')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--checkpoints', default='sam2-t', help='model checkpoint')

    parser.add_argument('--seed', default=30, type=int, help='rand seed')
    parser.add_argument('--eps', default = 10 / 255, type=float)
    parser.add_argument('--alpha', default= 2 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)
    parser.add_argument('--weight_fea', default=0.000001, type=float)
    parser.add_argument('--loss_fea', action='store_true')
    parser.add_argument('--loss_diff', action='store_true')
    parser.add_argument('--loss_t', action='store_true')
    parser.add_argument('--target_image_dir', default='./data/sav_test/JPEGImages_24fps',
                        help='Real SA-V distractor dir (paper uses sav_test split; must not overlap with YT-VOS train/eval)')
    parser.add_argument('--out_suffix', default='v2',
                        help='Output file suffix: uap_file/{train_dataset}_{out_suffix}.pth')
    return parser

def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args

if __name__ == '__main__':
    parser = get_parser()
    args = get_args(parser)
    seed_everything(seed=args.seed)

    device = "cuda:1"

    sam_fwder, predictor = load_model(args, device=device)

    custom_dataset = choose_dataset(args)
    video_to_indices = get_video_to_indices(custom_dataset)
    data_loader = DataLoader(custom_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    denorm = lambda x: sam_fwder.denorm_image(x)
    weight_Y = -1

    loss_fn = F.mse_loss
    mse_loss = torch.nn.MSELoss()
    cosine_loss = F.cosine_similarity
    cosfn = torch.nn.CosineSimilarity(dim=-1)

    tensor_shape = (1, 3, 1024, 1024)
    shape_tensor = torch.empty(tensor_shape)

    feature_diff = 0
    loss_fea = 0
    loss_t = 0
    loss_ft = 0

    weight_loss_fea = 0
    weight_loss_diff = 0
    weight_loss_t = 0

    perturbation = torch.empty_like(shape_tensor).uniform_(-args.eps, args.eps).to(device)

    target_image_dir = args.target_image_dir
    assert os.path.isdir(target_image_dir), f"SA-V distractor dir not found: {target_image_dir}"

    # Strict SA-V whitelist: folder names must match ^sav_\d{6}$.
    # Fail hard on any mixed/suspicious content (Codex R2 feedback: loose prefix check
    # would let quarantined YT-VOS folders leak in if anyone re-added them).
    SAV_PATTERN = re.compile(r'^sav_\d{6}$')
    all_entries = sorted(os.listdir(target_image_dir))  # sort for reproducibility (same seed -> same sample)
    folders = [f for f in all_entries if os.path.isdir(os.path.join(target_image_dir, f))]
    bad = [f for f in folders if not SAV_PATTERN.match(f)]
    if bad:
        raise RuntimeError(
            f"[contamination-guard] {len(bad)}/{len(folders)} folders in {target_image_dir} "
            f"do not match strict SA-V pattern ^sav_\\d{{6}}$. "
            f"Bad examples: {bad[:5]}. Refusing to train on contaminated distractor pool.")
    if len(folders) < args.fea_num:
        raise RuntimeError(
            f"Only {len(folders)} SA-V folders available, need --fea_num={args.fea_num}")

    # Sample fea_num folders (seeded, deterministic due to sorted `folders`).
    selected_folders = random.sample(folders, args.fea_num)

    # Preflight: every selected folder must have ≥1 image. Empty folders would cause
    # silent video-skip in the main loop (Codex R2: makes training set nondeterministic
    # against disk state).
    for f in selected_folders:
        fp = os.path.join(target_image_dir, f)
        imgs = [x for x in os.listdir(fp) if x.endswith(('.png', '.jpg', '.jpeg'))]
        if not imgs:
            raise RuntimeError(f"Selected SA-V folder {f} has no images; preflight failed.")
    print(f"[target] dir={target_image_dir}  total_folders={len(folders)}  using={len(selected_folders)}")
    print(f"[target] preflight OK: every selected folder has ≥1 image")

    for step in range(args.P_num):
        ema_grad = None
        sign_flip_count = 0
        sign_total = 0

        for video_name, indices in video_to_indices.items():
            print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

            # FIX 2: reset prev_adv_feature at video boundary
            # Bug: prev_adv_feature persisted across videos, causing loss_diff on
            # the first frame of each video to measure cross-video feature distance.
            prev_adv_feature = None

            folder = random.choice(selected_folders)
            folder_path = os.path.join(target_image_dir, folder)
            # Sort for determinism: random.choice over an unsorted os.listdir() result
            # would give different picks across filesystems / remounts.
            image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
            if not image_files:
                # Preflight should have caught this; defensive re-check.
                raise RuntimeError(f"Selected folder {folder} lost its images between preflight and use.")
            image_path = random.choice(image_files)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((1024, 1024), Image.Resampling.BICUBIC)
            image = np.array(image)
            # target_feature is a constant distractor feature used only as InfoNCE
            # negative. Compute under no_grad to avoid keeping an autograd graph
            # alive for the whole video (~8 GB per graph on 1024x1024).
            with torch.no_grad():
                tgt = sam_fwder.transform_image(image).to(device)
                tgt = denorm(tgt)
                target_feature = sam_fwder.get_image_feature(tgt).detach()

            video_subset = Subset(custom_dataset, indices)
            video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

            pre_dict = None
            pre_dict_adv = None
            mask_pre = None
            mask_pre_adv = None
            start_frame_processed = False

            for images, P_list, img_ids, gt, point in tqdm(video_loader):
                img_ID, img, mask_gt, P_gt = img_ids[0], images[0], gt[0], P_list[0]
                video_name = img_ID.split('/')[0]
                frame_idx = get_frame_index(img_ID)

                X = sam_fwder.transform_image(img).to(device)
                benign_img = denorm(X)
                H, W, _ = img.shape
                Y = torch.ones([1, 1, H, W]).to(X.device, torch.float32) * weight_Y
                Y_bin = Y.bool()
                assert Y_bin.dtype in ['bool', bool, torch.bool]
                print(f"args.train_dataset: {args.train_dataset} ")

                transform1 = transforms.RandomRotation(degrees=15)
                transform2 = transforms.Lambda(lambda img: img + 0.03 * torch.rand_like(img))
                transform3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)

                # prototype_feature is the InfoNCE positive anchor built from augmented
                # benign views. It feeds loss_fea but we do NOT backprop into it (only
                # adv_feature carries grad). Wrap under no_grad to avoid multiplying
                # the per-frame memory footprint by 3 (three augmented forward passes).
                with torch.no_grad():
                    aug_img1 = transform1(benign_img)
                    aug_img2 = transform2(benign_img)
                    aug_img3 = transform3(benign_img)
                    img_list = [aug_img1, aug_img2, aug_img3]
                    prototype_feature = get_fused_prototype(img_list, sam_fwder, device).to(device).detach()

                if not start_frame_processed:
                    start_frame_idx = frame_idx
                    pre_dict = None
                    pre_dict_adv = None
                    mask_pre = None
                    mask_pre_adv = None
                    start_frame_processed = True
                    start_P = P_gt

                prompts = make_multi_prompts(args.point, (1024, 1024), args.prompts_num)
                P = sam_fwder.transform_prompts(*prompts)

                # Clean-branch logits: no grad needed (used only to build mask_pre for
                # the predictor's memory bank via get_current_out).
                with torch.no_grad():
                    logits_clean = sam_fwder.forward(benign_img, *P)
                    mask_clean = logits_clean > sam_fwder.mask_threshold
                    mask_pre = mask_clean.clone().detach()
                    # get_current_out re-runs the image encoder; wrap under no_grad too.
                    output_dict = sam_fwder.get_current_out(frame_idx, benign_img, mask_pre)
                pre_dict = output_dict

                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True

                # adv-branch logits: graph REQUIRED for grad(loss, adv_img).
                logits = sam_fwder.forward(adv_img, *P)
                mask = logits > sam_fwder.mask_threshold
                mask_pre_adv = mask.clone().detach()

                # Note: output_dict_adv is stored in predictor state for future frames but
                # gradients do NOT need to flow through it — the attack loss uses `logits`
                # and `adv_feature` directly. Wrap in no_grad so the second encoder forward
                # doesn't build a redundant 8GB graph (passing adv_img without .detach()
                # because SAM2 may track tensor identity for its memory-bank state).
                with torch.no_grad():
                    output_dict_adv = sam_fwder.get_current_out(frame_idx, adv_img, mask_pre_adv)
                pre_dict_adv = output_dict_adv

                # adv_feature: graph REQUIRED (used in loss_fea via infonce_loss).
                adv_feature = sam_fwder.get_image_feature(adv_img)


                if args.loss_t:
                    attacked = mask == Y_bin
                    output = attacked * logits
                    output_f = ~attacked * (1 - logits)
                    loss_t = F.binary_cross_entropy_with_logits(output, Y)
                    loss_ft = -F.binary_cross_entropy_with_logits(output_f, Y)

                    weight_loss_t = 1

                if args.loss_diff:
                    if prev_adv_feature is not None:
                        feature_diff = -cosine_loss(prev_adv_feature, adv_feature).mean()
                        weight_loss_diff = 1
                    else:
                        feature_diff = 0

                if args.loss_fea:
                    loss_fea = infonce_loss(adv_feature, prototype_feature, target_feature)

                loss = weight_loss_t * loss_t + 0.01 * loss_ft  +  weight_loss_diff * feature_diff + args.weight_fea*loss_fea

                # FIX: drop grad_outputs=loss (was multiplying gradient by scalar loss,
                # which distorted magnitudes when averaging raw gradients and could flip
                # sign when total loss went negative via loss_ft).
                g = grad(loss, adv_img)[0]

                # FIX: drop history-average-then-sign (sample_total_g / sample_step_count).
                # With eps=10/255 and alpha=2/255, sign-update saturates in ~5 frames;
                # averaging over 1500 frames per outer step locked in early sign patterns.
                # Use per-step EMA directly.
                beta = 0.95
                if ema_grad is None:
                    ema_grad = g.detach()
                else:
                    prev_sign = ema_grad.sign()
                    ema_grad = beta * ema_grad + (1 - beta) * g.detach()
                    new_sign = ema_grad.sign()
                    sign_flip_count += (prev_sign != new_sign).sum().item()
                    sign_total += prev_sign.numel()
                perturbation = (perturbation - args.alpha * ema_grad.sign()).clamp(-args.eps, args.eps).detach()
                prev_adv_feature = adv_feature.detach()

            # End-of-video: clear Python refs to predictor state so next video starts fresh.
            # Note: removed torch.cuda.empty_cache() and explicit `del` — prior attempt
            # triggered CUDA illegal memory access, likely because SAM2's memory bank held
            # internal references to tensors we'd eagerly freed. Rely on Python GC + no_grad.
            pre_dict = None
            pre_dict_adv = None
            mask_pre = None
            mask_pre_adv = None

        if sign_total > 0:
            print(f"[step {step}] EMA-sign flip rate: {sign_flip_count/sign_total:.4%} "
                  f"(low rate = saturation/frozen update; healthy rate ~1-20%)")

    os.makedirs("uap_file", exist_ok=True)  # Codex R2 nice-to-have: never fail at save
    uap_save_path = f"uap_file/{args.train_dataset}_{args.out_suffix}.pth"
    torch.save(perturbation.cpu(), uap_save_path)
    print(f"\n Global UAP saved to {uap_save_path}")
