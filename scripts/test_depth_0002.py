"""
Test script to generate 4-view images with depth predictions and compare with GT depth_hull.
Uses data from /home/chenzebin/MVdream_czb/customized_simple_dataset_tagVersion_simplified/data/0002
Views: 000, 006, 012, 018
"""

import argparse
import json
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.encoders.modules import ImageEmbedder
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pose_as_camera(pose_path: str, device: str) -> torch.Tensor:
    """Load pose from txt file and convert to camera format."""
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


def load_rgba_image(image_path: str, image_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load RGBA image and return RGB (white background) and alpha mask.
    
    Returns:
        rgb:   [3, H, W], float in [0, 1]
        alpha: [1, H, W], float in [0, 1]
    """
    im = Image.open(image_path).convert("RGBA")
    im = im.resize((image_size, image_size), Image.BILINEAR)
    
    arr = np.asarray(im).astype(np.float32) / 255.0
    
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    
    # Composite RGB over white background
    rgb_white_bg = rgb * alpha + (1.0 - alpha) * 1.0
    
    rgb_t = torch.from_numpy(rgb_white_bg.transpose(2, 0, 1)).float().to(device)
    alpha_t = torch.from_numpy(alpha.transpose(2, 0, 1)).float().to(device)
    
    # Threshold alpha
    alpha_t = (alpha_t > 0.05).float()
    
    return rgb_t, alpha_t


def load_hull_depth(path: str, image_size: int, device: str) -> torch.Tensor:
    """Load depth map from PNG file."""
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
    return torch.from_numpy(arr).unsqueeze(0).to(device=device)


def normalize_depth_per_image(depth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize depth map to [0, 1] range per image.
    
    Args:
        depth: [N, C, H, W] tensor
    """
    depth_min = depth.amin(dim=(2, 3), keepdim=True)
    depth_max = depth.amax(dim=(2, 3), keepdim=True)
    return (depth - depth_min) / (depth_max - depth_min + eps)


def depth_exceed_loss_from_rgb(
    decoded_imgs: torch.Tensor,  # [N, 3, H, W]
    hull_depth: torch.Tensor,     # [N, 1, H, W]
    method: str = "inverse_luminance",
    direction: str = "pred_gt_hull",
    margin: float = 0.02,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate depth exceed loss and return loss, exceed map, pred depth, and hull depth."""
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

    loss_per_view = exceed.mean(dim=(1, 2, 3))
    return loss_per_view, exceed, pred_depth, hull_depth


def tensor_to_gray_image(x: torch.Tensor) -> Image.Image:
    """Convert single channel tensor to gray image."""
    arr = x.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").convert("RGB")


def tensor_to_rgb_image(x: torch.Tensor) -> Image.Image:
    """Convert RGB tensor to PIL image."""
    arr = x.detach().float().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_diff_image(pred_depth: torch.Tensor, hull_depth: torch.Tensor) -> Image.Image:
    """Create signed difference image: red=pred>hull, blue=hull>pred."""
    diff = (pred_depth - hull_depth).detach().float().cpu().squeeze().numpy()
    pos = np.clip(diff, 0.0, None)
    neg = np.clip(-diff, 0.0, None)
    scale = max(float(pos.max()), float(neg.max()), 1e-6)
    pos = pos / scale
    neg = neg / scale
    rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255.0).astype(np.uint8)  # Red for pred > hull
    rgb[..., 2] = (neg * 255.0).astype(np.uint8)  # Blue for hull > pred
    return Image.fromarray(rgb, mode="RGB")


def make_grid(images: List[Tuple[str, Image.Image]], title: str = "") -> Image.Image:
    """Create a grid of images with labels."""
    if not images:
        return Image.new("RGB", (256, 256), "white")
    
    tile_w, tile_h = images[0][1].size
    label_h = 30
    title_h = 40 if title else 0
    
    # Calculate grid dimensions
    cols = 4  # 4 views
    rows = len(images) // cols + (1 if len(images) % cols else 0)
    
    grid_w = cols * tile_w
    grid_h = rows * (tile_h + label_h) + title_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0))
    
    for idx, (label, img) in enumerate(images):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h) + title_h
        grid.paste(img, (x, y + label_h))
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
    
    return grid


def load_adapter_weights(model, adapter_ckpt_path: str, lora_rank: int, lora_alpha: float, device: str):
    """Load LoRA adapter weights into model and return auxiliary modules."""
    replaced = inject_lora(model, r=lora_rank, alpha=lora_alpha)
    print(f"Injected LoRA into {replaced} modules for inference.")

    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    model_state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"Loaded adapter checkpoint: {adapter_ckpt_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    
    # Load cond_proj for projecting hull embeddings
    cond_proj = None
    cond_proj_state = ckpt.get("cond_proj_state")
    if cond_proj_state is not None:
        weight = cond_proj_state["weight"]
        out_features, in_features = weight.shape
        cond_proj = torch.nn.Linear(in_features, out_features)
        cond_proj.load_state_dict(cond_proj_state)
        cond_proj.to(device)
        cond_proj.eval()
        print(f"Loaded cond_proj: {in_features} -> {out_features}")
    else:
        print("cond_proj_state not found in checkpoint")
    
    # Load alpha mask encoder
    mask_embedder = None
    alpha_mask_encoder_state = ckpt.get("alpha_mask_encoder_state")
    if alpha_mask_encoder_state is not None:
        pos_embed = alpha_mask_encoder_state["pos_embed"]
        _, num_tokens, embed_dim = pos_embed.shape
        grid_size = int(round(num_tokens ** 0.5))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Invalid alpha mask token count in checkpoint: {num_tokens}")
        mask_embedder = AlphaMaskEmbedder(embed_dim=embed_dim, grid_size=grid_size)
        mask_embedder.load_state_dict(alpha_mask_encoder_state)
        mask_embedder.to(device)
        mask_embedder.eval()
        print(f"Loaded alpha_mask_encoder: tokens={num_tokens}, dim={embed_dim}")
    else:
        print("alpha_mask_encoder_state not found in checkpoint")
    
    # Load image encoder state
    image_encoder_state = ckpt.get("image_encoder_state")
    
    return cond_proj, mask_embedder, image_encoder_state


class AlphaMaskEmbedder(torch.nn.Module):
    """Encode alpha hull masks into cross-attention tokens."""

    def __init__(self, embed_dim: int, grid_size: int = 4):
        super().__init__()
        self.grid_size = int(grid_size)
        self.pool = torch.nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, embed_dim)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float().clamp(0.0, 1.0)
        pooled = self.pool(mask).flatten(2).transpose(1, 2)
        return self.proj(pooled) + self.pos_embed


def sample_with_hull_guidance(
    model,
    sampler,
    prompt: str,
    negative_prompt: str,
    cameras: torch.Tensor,  # [4, 16]
    hull_masks: torch.Tensor,  # [4, 1, H, W]
    hull_images: torch.Tensor,  # [4, 3, H, W]
    image_size: int,
    steps: int,
    guidance_scale: float,
    device: str,
    fp16: bool,
    image_encoder: torch.nn.Module,
    cond_proj: Optional[torch.nn.Module] = None,
    mask_embedder: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Sample 4 views with hull mask guidance."""
    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype == torch.float16
        else torch.no_grad()
    )
    
    num_frames = 4
    
    with torch.no_grad(), amp_ctx:
        # Get text conditioning [1, 77, text_dim]
        text_c = model.get_learned_conditioning([prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)
        text_dim = text_c.shape[-1]
        
        # Repeat for 4 views [4, 77, text_dim]
        c_text = text_c.repeat(num_frames, 1, 1)
        uc_text_batch = uc_text.repeat(num_frames, 1, 1)
        
        # Get hull image embeddings [4, 1, embed_dim]
        hull_embed = image_encoder.encode(hull_images).to(device)
        
        # Project hull embeddings to text dimension if needed
        if hull_embed.shape[-1] != text_dim:
            if cond_proj is not None:
                hull_embed = cond_proj(hull_embed)
            else:
                raise ValueError(
                    f"Hull embedding dim {hull_embed.shape[-1]} != text dim {text_dim}, "
                    "and no cond_proj available."
                )
        
        # Build context by concatenating embeddings
        if mask_embedder is not None:
            mask_emb = mask_embedder(hull_masks)  # [4, grid_size^2, embed_dim]
            context_cat = torch.cat([c_text, hull_embed, mask_emb], dim=1)
            uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_embed), torch.zeros_like(mask_emb)], dim=1)
        else:
            context_cat = torch.cat([c_text, hull_embed], dim=1)
            uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_embed)], dim=1)
        
        # Build conditioning
        cond = {
            "context": context_cat,
            "camera": cameras,
            "num_frames": num_frames,
        }
        uncond = {
            "context": uc_context_cat,
            "camera": cameras,
            "num_frames": num_frames,
        }
        
        # Sample
        samples, _ = sampler.sample(
            S=steps,
            conditioning=cond,
            batch_size=num_frames,
            shape=[4, image_size // 8, image_size // 8],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond,
            eta=0.0,
            x_T=None,
        )
        
        # Decode
        decoded = model.decode_first_stage(samples)
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test depth prediction on 0002 sample with 4 views (000, 006, 012, 018)."
    )
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--base_ckpt", type=str, default="checkpoints/pretrained/sd-v2.1-base-4view.pt")
    parser.add_argument("--adapter_ckpt", type=str, default="checkpoints/hull_mask/ckpt_epoch_99.pth")
    parser.add_argument("--data_root", type=str, default="customized_simple_dataset_tagVersion_simplified/data/0002")
    parser.add_argument("--out_dir", type=str, default="outputs/test_depth_0002")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--pred_depth_method", choices=["inverse_luminance", "luminance"], default="inverse_luminance")
    parser.add_argument("--depth_exceed_direction", choices=["pred_gt_hull", "pred_lt_hull"], default="pred_gt_hull")
    parser.add_argument("--depth_margin", type=float, default=0.02)
    parser.add_argument("--normalize_depth_loss", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Define views and prompt
    view_ids = ["000", "006", "012", "018"]
    prompt = "entity: single, volume: arch, direction: linear, operation: stretching, affect: porous"
    negative_prompt = ""
    
    print(f"Prompt: {prompt}")
    print(f"Loading base model: {args.base_ckpt}")
    
    # Load base model
    model = build_model(args.model_name, ckpt_path=args.base_ckpt)
    model.to(args.device)
    model.eval()
    
    # Load adapter
    print(f"Loading adapter: {args.adapter_ckpt}")
    cond_proj, mask_embedder, image_encoder_state = load_adapter_weights(
        model, args.adapter_ckpt, args.lora_rank, args.lora_alpha, args.device
    )
    
    # Create image encoder for hull RGB conditioning
    print("Creating image encoder...")
    image_encoder = ImageEmbedder(device=args.device, img_size=args.size)
    if image_encoder_state is not None:
        image_encoder.load_state_dict(image_encoder_state)
        print("Loaded image_encoder_state from checkpoint")
    else:
        print("[warn] image_encoder_state not found; using default weights")
    image_encoder.to(args.device)
    image_encoder.eval()
    
    # Create sampler
    sampler = DDIMSampler(model)
    
    # Load images, poses, and GT depths
    print("Loading input data...")
    cameras = []
    hull_masks = []
    gt_depths = []
    input_images = []
    
    for view_id in view_ids:
        # Load pose
        pose_path = os.path.join(args.data_root, "pose", f"{view_id}.txt")
        camera = load_pose_as_camera(pose_path, args.device)
        cameras.append(camera)
        
        # Load RGB and alpha mask
        rgb_path = os.path.join(args.data_root, "rgb_convexhull", f"{view_id}.png")
        rgb, alpha = load_rgba_image(rgb_path, args.size, args.device)
        hull_masks.append(alpha)
        input_images.append(rgb)
        
        # Load GT depth
        depth_path = os.path.join(args.data_root, "depth_hull", f"{view_id}.png")
        gt_depth = load_hull_depth(depth_path, args.size, args.device)
        gt_depths.append(gt_depth)
    
    cameras = torch.cat(cameras, dim=0)  # [4, 16]
    hull_masks = torch.stack(hull_masks, dim=0)  # [4, 1, H, W]
    gt_depths = torch.stack(gt_depths, dim=0)  # [4, 1, H, W]
    input_images = torch.stack(input_images, dim=0)  # [4, 3, H, W]
    
    print(f"Cameras: {cameras.shape}")
    print(f"Hull masks: {hull_masks.shape}")
    print(f"GT depths: {gt_depths.shape}")
    
    # Generate 4 views
    print("Generating 4 views...")
    decoded = sample_with_hull_guidance(
        model=model,
        sampler=sampler,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cameras=cameras,
        hull_masks=hull_masks,
        hull_images=input_images,
        image_size=args.size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        fp16=args.fp16,
        image_encoder=image_encoder,
        cond_proj=cond_proj,
        mask_embedder=mask_embedder,
    )
    
    print(f"Generated images: {decoded.shape}")
    
    # Calculate depth loss for each view
    print("Calculating depth losses...")
    
    # Calculate with normalization (for training-style loss)
    loss_per_view_norm, exceed_maps_norm, pred_depths_norm, hull_depths_norm = depth_exceed_loss_from_rgb(
        decoded_imgs=decoded,
        hull_depth=gt_depths,
        method=args.pred_depth_method,
        direction=args.depth_exceed_direction,
        margin=args.depth_margin,
        normalize=True,  # Normalized version
    )
    
    # Calculate without normalization (for absolute comparison)
    loss_per_view_raw, exceed_maps_raw, pred_depths_raw, hull_depths_raw = depth_exceed_loss_from_rgb(
        decoded_imgs=decoded,
        hull_depth=gt_depths,
        method=args.pred_depth_method,
        direction=args.depth_exceed_direction,
        margin=args.depth_margin,
        normalize=False,  # Raw version
    )
    
    # Save results
    print("Saving results...")
    all_images = []
    
    for i, view_id in enumerate(view_ids):
        # Save individual images
        view_dir = os.path.join(args.out_dir, f"view_{view_id}")
        os.makedirs(view_dir, exist_ok=True)
        
        input_img = tensor_to_rgb_image(input_images[i])
        generated_img = tensor_to_rgb_image(decoded[i])
        
        # Normalized versions
        pred_depth_norm_img = tensor_to_gray_image(pred_depths_norm[i])
        gt_depth_norm_img = tensor_to_gray_image(hull_depths_norm[i])
        exceed_norm_img = tensor_to_gray_image(exceed_maps_norm[i] / exceed_maps_norm[i].max().clamp_min(1e-6))
        diff_norm_img = make_diff_image(pred_depths_norm[i], hull_depths_norm[i])
        
        # Raw (unnormalized) versions
        pred_depth_raw_img = tensor_to_gray_image(pred_depths_raw[i])
        gt_depth_raw_img = tensor_to_gray_image(hull_depths_raw[i])
        exceed_raw_img = tensor_to_gray_image(exceed_maps_raw[i] / exceed_maps_raw[i].max().clamp_min(1e-6))
        diff_raw_img = make_diff_image(pred_depths_raw[i], hull_depths_raw[i])
        
        # Save all versions
        input_img.save(os.path.join(view_dir, "input_rgb.png"))
        generated_img.save(os.path.join(view_dir, "generated_rgb.png"))
        
        # Batch-normalized (all 4 views use same scale)
        pred_depth_norm_img.save(os.path.join(view_dir, "pred_depth_batch_norm.png"))
        gt_depth_norm_img.save(os.path.join(view_dir, "gt_depth_batch_norm.png"))
        exceed_norm_img.save(os.path.join(view_dir, "exceed_map_batch_norm.png"))
        diff_norm_img.save(os.path.join(view_dir, "signed_diff_batch_norm.png"))
        
        # Raw (no normalization)
        pred_depth_raw_img.save(os.path.join(view_dir, "pred_depth_raw.png"))
        gt_depth_raw_img.save(os.path.join(view_dir, "gt_depth_raw.png"))
        exceed_raw_img.save(os.path.join(view_dir, "exceed_map_raw.png"))
        diff_raw_img.save(os.path.join(view_dir, "signed_diff_raw.png"))
        
        # Collect for grid (use normalized versions)
        all_images.extend([
            (f"View {view_id} - Input", input_img),
            (f"View {view_id} - Gen", generated_img),
            (f"View {view_id} - Pred (Norm)", pred_depth_norm_img),
            (f"View {view_id} - GT (Norm)", gt_depth_norm_img),
        ])
        
        # Calculate metrics for both versions
        exceed_ratio_norm = (exceed_maps_norm[i] > 0).float().mean().item()
        max_excess_norm = exceed_maps_norm[i].max().item()
        mean_abs_diff_norm = (pred_depths_norm[i] - hull_depths_norm[i]).abs().mean().item()
        
        exceed_ratio_raw = (exceed_maps_raw[i] > 0).float().mean().item()
        max_excess_raw = exceed_maps_raw[i].max().item()
        mean_abs_diff_raw = (pred_depths_raw[i] - hull_depths_raw[i]).abs().mean().item()
        
        metrics = (
            f"View {view_id}:\n"
            f"\n=== Batch Normalized (all 4 views use same scale) ===\n"
            f"  Loss: {loss_per_view_norm[i].item():.8f}\n"
            f"  Exceed ratio: {exceed_ratio_norm:.6f}\n"
            f"  Max excess: {max_excess_norm:.6f}\n"
            f"  Mean abs diff: {mean_abs_diff_norm:.6f}\n"
            f"  Pred depth range (normalized): [{pred_depths_norm[i].min().item():.4f}, {pred_depths_norm[i].max().item():.4f}]\n"
            f"  GT depth range (normalized): [{hull_depths_norm[i].min().item():.4f}, {hull_depths_norm[i].max().item():.4f}]\n"
            f"\n=== Raw (no normalization) ===\n"
            f"  Loss: {loss_per_view_raw[i].item():.8f}\n"
            f"  Exceed ratio: {exceed_ratio_raw:.6f}\n"
            f"  Max excess: {max_excess_raw:.6f}\n"
            f"  Mean abs diff: {mean_abs_diff_raw:.6f}\n"
            f"  Pred depth range: [{pred_depths_raw[i].min().item():.4f}, {pred_depths_raw[i].max().item():.4f}]\n"
            f"  GT depth range: [{hull_depths_raw[i].min().item():.4f}, {hull_depths_raw[i].max().item():.4f}]\n"
        )
        print(metrics)
        
        with open(os.path.join(view_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(metrics)
    
    # Create overall grid
    grid = make_grid(all_images, f"4-View Depth Test - Sample 0002 (Batch Normalized)")
    grid.save(os.path.join(args.out_dir, "all_views_grid.png"))
    
    # Save summary metrics
    avg_loss_norm = loss_per_view_norm.mean().item()
    avg_loss_raw = loss_per_view_raw.mean().item()
    
    # Get global min/max for batch normalized version
    pred_global_min = pred_depths_norm.min().item()
    pred_global_max = pred_depths_norm.max().item()
    gt_global_min = hull_depths_norm.min().item()
    gt_global_max = hull_depths_norm.max().item()
    
    summary = (
        f"Prompt: {prompt}\n"
        f"\n=== Batch Normalized (all 4 views use same scale) ===\n"
        f"Average Loss: {avg_loss_norm:.8f}\n"
        f"Per-view losses: {[f'{l.item():.8f}' for l in loss_per_view_norm]}\n"
        f"Global pred depth range: [{pred_global_min:.4f}, {pred_global_max:.4f}]\n"
        f"Global GT depth range: [{gt_global_min:.4f}, {gt_global_max:.4f}]\n"
        f"\n=== Raw (no normalization) ===\n"
        f"Average Loss: {avg_loss_raw:.8f}\n"
        f"Per-view losses: {[f'{l.item():.8f}' for l in loss_per_view_raw]}\n"
        f"\n=== Settings ===\n"
        f"Method: {args.pred_depth_method}\n"
        f"Direction: {args.depth_exceed_direction}\n"
        f"Margin: {args.depth_margin}\n"
    )
    print("\n" + "="*60)
    print(summary)
    print("="*60)
    
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nAll results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
