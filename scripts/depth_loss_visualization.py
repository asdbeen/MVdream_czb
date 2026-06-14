import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


DEFAULT_POSE_PATH = (
    "customized_simple_dataset_tagVersion_simplified/data/0892/pose/001.txt"
)
DEFAULT_HULL_DEPTH_PATH = (
    "customized_simple_dataset_tagVersion_simplified/data/0892/depth_hull/001.png"
)
DEFAULT_BASE_CKPT = "checkpoints/pretrained/sd-v2.1-base-4view.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_from_category(pose_path: str, fallback: str = "object") -> str:
    sample_dir = os.path.dirname(os.path.dirname(pose_path))
    category_path = os.path.join(sample_dir, "category.json")
    if not os.path.exists(category_path):
        return fallback
    try:
        with open(category_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            fields = ["entity", "volume", "direction", "operation", "affect"]
            parts = []
            for key in fields:
                value = str(data.get(key, "")).strip()
                if value:
                    parts.append(f"{key}: {value}")
            return ", ".join(parts) if parts else fallback
        return str(data)
    except Exception:
        return fallback


def load_pose_as_camera(pose_path: str, device: str) -> torch.Tensor:
    values = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if line:
                values.extend(float(x) for x in line.split())

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 16:
        pose_4x4 = arr.reshape(4, 4)
    elif arr.size == 12:
        pose_3x4 = arr.reshape(3, 4)
        bottom = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        pose_4x4 = np.concatenate([pose_3x4, bottom], axis=0)
    else:
        raise ValueError(
            f"Pose file must contain 12 or 16 numbers, got {arr.size}: {pose_path}"
        )

    return torch.from_numpy(pose_4x4.reshape(1, 16)).to(device=device, dtype=torch.float32)


def load_hull_depth(path: str, image_size: int, device: str) -> torch.Tensor:
    im = Image.open(path)
    if im.mode in ("I", "I;16", "F"):
        arr = np.asarray(im).astype(np.float32)
        finite = np.isfinite(arr)
        if finite.any():
            valid = arr[finite]
            lo, hi = float(valid.min()), float(valid.max())
            arr = (arr - lo) / max(hi - lo, 1e-6)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        im = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    else:
        im = im.convert("L")
    im = im.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device)


def normalize_depth_per_image(depth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    depth_min = depth.amin(dim=(2, 3), keepdim=True)
    depth_max = depth.amax(dim=(2, 3), keepdim=True)
    return (depth - depth_min) / (depth_max - depth_min + eps)


def depth_exceed_loss_from_rgb(
    decoded_imgs: torch.Tensor,
    hull_depth: torch.Tensor,
    method: str = "inverse_luminance",
    direction: str = "pred_gt_hull",
    margin: float = 0.02,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_gray = decoded_imgs.mean(dim=1, keepdim=True)
    if method == "inverse_luminance":
        pred_depth = 1.0 - pred_gray
    elif method == "luminance":
        pred_depth = pred_gray
    else:
        raise ValueError(f"Unknown pred depth method: {method}")

    if hull_depth.shape[1] != 1:
        hull_depth = hull_depth.mean(dim=1, keepdim=True)
    hull_depth = hull_depth.float()

    if hull_depth.shape[-2:] != pred_depth.shape[-2:]:
        hull_depth = F.interpolate(
            hull_depth,
            size=pred_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if normalize:
        pred_depth = normalize_depth_per_image(pred_depth)
        hull_depth = normalize_depth_per_image(hull_depth)
    else:
        pred_depth = pred_depth.clamp(0.0, 1.0)
        hull_depth = hull_depth.clamp(0.0, 1.0)

    if direction == "pred_gt_hull":
        exceed = F.relu(pred_depth - hull_depth - float(margin))
    elif direction == "pred_lt_hull":
        exceed = F.relu(hull_depth - pred_depth - float(margin))
    else:
        raise ValueError(f"Unknown depth exceed direction: {direction}")

    return exceed.mean(), exceed, pred_depth, hull_depth


def tensor_to_gray_image(x: torch.Tensor) -> Image.Image:
    arr = x.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").convert("RGB")


def tensor_to_rgb_image(x: torch.Tensor) -> Image.Image:
    arr = x.detach().float().cpu().squeeze(0).clamp(0.0, 1.0).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_diff_image(pred_depth: torch.Tensor, hull_depth: torch.Tensor) -> Image.Image:
    diff = (pred_depth - hull_depth).detach().float().cpu().squeeze().numpy()
    pos = np.clip(diff, 0.0, None)
    neg = np.clip(-diff, 0.0, None)
    scale = max(float(pos.max()), float(neg.max()), 1e-6)
    pos = pos / scale
    neg = neg / scale
    rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255.0).astype(np.uint8)
    rgb[..., 2] = (neg * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def make_grid(images: list[tuple[str, Image.Image]], metrics: str) -> Image.Image:
    tile_w, tile_h = images[0][1].size
    label_h = 30
    metrics_h = 90
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    grid = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h) + metrics_h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, (label, img) in enumerate(images):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        grid.paste(img, (x, y + label_h))
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
    draw.text((8, rows * (tile_h + label_h) + 8), metrics, fill=(0, 0, 0))
    return grid


def sample_base_model(
    model,
    sampler,
    prompt: str,
    negative_prompt: str,
    camera: torch.Tensor,
    image_size: int,
    steps: int,
    guidance_scale: float,
    device: str,
    fp16: bool,
) -> torch.Tensor:
    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype == torch.float16
        else torch.no_grad()
    )
    with torch.no_grad(), amp_ctx:
        text_c = model.get_learned_conditioning([prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)
        cond = {
            "context": text_c,
            "camera": camera,
            "num_frames": 1,
        }
        uncond = {
            "context": uc_text,
            "camera": camera,
            "num_frames": 1,
        }
        samples, _ = sampler.sample(
            S=steps,
            conditioning=cond,
            batch_size=1,
            shape=[4, image_size // 8, image_size // 8],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond,
            eta=0.0,
            x_T=None,
        )
        decoded = model.decode_first_stage(samples)
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize train_lora_hull_depthLoss depth exceed loss on one pose/depth_hull pair."
    )
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--base_ckpt", type=str, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--pose_path", type=str, default=DEFAULT_POSE_PATH)
    parser.add_argument("--hull_depth_path", type=str, default=DEFAULT_HULL_DEPTH_PATH)
    parser.add_argument("--out_dir", type=str, default="outputs/depth_loss_visualization/0892_001")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--pred_depth_method", choices=["inverse_luminance", "luminance"], default="inverse_luminance")
    parser.add_argument("--depth_exceed_direction", choices=["pred_gt_hull", "pred_lt_hull"], default="pred_gt_hull")
    parser.add_argument("--depth_margin", type=float, default=0.02)
    parser.add_argument("--normalize_depth_loss", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    prompt = args.prompt or load_prompt_from_category(args.pose_path)
    set_seed(args.seed)

    print(f"Loading base model: {args.base_ckpt}")
    model = build_model(args.model_name, ckpt_path=args.base_ckpt)
    model.to(args.device)
    model.eval()
    sampler = DDIMSampler(model)

    camera = load_pose_as_camera(args.pose_path, args.device)
    hull_depth = load_hull_depth(args.hull_depth_path, args.size, args.device)
    decoded = sample_base_model(
        model=model,
        sampler=sampler,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        camera=camera,
        image_size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        fp16=args.fp16,
    )

    loss, exceed_map, pred_depth, hull_depth_for_loss = depth_exceed_loss_from_rgb(
        decoded_imgs=decoded,
        hull_depth=hull_depth,
        method=args.pred_depth_method,
        direction=args.depth_exceed_direction,
        margin=args.depth_margin,
        normalize=args.normalize_depth_loss,
    )

    exceed_ratio = (exceed_map > 0).float().mean().item()
    max_excess = exceed_map.max().item()
    mean_abs_diff = (pred_depth - hull_depth_for_loss).abs().mean().item()
    metrics = (
        f"prompt={prompt}\n"
        f"loss={loss.item():.8f}, exceed_ratio={exceed_ratio:.6f}, "
        f"max_excess={max_excess:.6f}, mean_abs_diff={mean_abs_diff:.6f}\n"
        f"method={args.pred_depth_method}, direction={args.depth_exceed_direction}, "
        f"margin={args.depth_margin}, normalize={args.normalize_depth_loss}"
    )

    generated_img = tensor_to_rgb_image(decoded)
    pred_depth_img = tensor_to_gray_image(pred_depth)
    hull_depth_img = tensor_to_gray_image(hull_depth_for_loss)
    exceed_img = tensor_to_gray_image(exceed_map / exceed_map.max().clamp_min(1e-6))
    diff_img = make_diff_image(pred_depth, hull_depth_for_loss)

    generated_img.save(os.path.join(args.out_dir, "generated_rgb.png"))
    pred_depth_img.save(os.path.join(args.out_dir, "pred_depth.png"))
    hull_depth_img.save(os.path.join(args.out_dir, "hull_depth.png"))
    exceed_img.save(os.path.join(args.out_dir, "depth_exceed.png"))
    diff_img.save(os.path.join(args.out_dir, "signed_diff_red_pred_blue_hull.png"))

    grid = make_grid(
        [
            ("generated RGB", generated_img),
            ("pred depth", pred_depth_img),
            ("hull depth", hull_depth_img),
            ("exceed map", exceed_img),
            ("signed diff", diff_img),
        ],
        metrics,
    )
    grid.save(os.path.join(args.out_dir, "depth_loss_visualization_grid.png"))
    with open(os.path.join(args.out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics + "\n")

    print(metrics)
    print(f"Saved visualization to: {args.out_dir}")


if __name__ == "__main__":
    main()
