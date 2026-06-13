import os
import sys
import argparse
import contextlib
import random
import inspect
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import tempfile
import shutil

# 后处理：将接近黑色的像素设为白色
def set_white_background(img: np.ndarray, threshold: int = 30) -> np.ndarray:
    img = img.copy()
    mask = np.all(img < threshold, axis=-1)
    img[mask] = [255, 255, 255]
    return img

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.encoders.modules import ImageEmbedder
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model


# =========================================================
# Hull-guided iterative denoise helpers
# =========================================================


def _load_rgba_with_alpha_any(image_input, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load an uploaded hull image the same way as the dataset snippet.

    Returns:
        rgb:   [3, H, W], float in [0, 1]
        alpha: [1, H, W], float in [0, 1]

    The uploaded image must be a real RGBA PNG with transparency. The RGB
    channel is composited over white to match the training dataset logic, while
    alpha is used as the strict hull boundary.
    """
    if image_input is None:
        raise ValueError("Image input is None.")

    path = image_input.name if hasattr(image_input, "name") else str(image_input)

    im = Image.open(path).convert("RGBA")
    im = im.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(im).astype(np.float32) / 255.0

    rgb = arr[..., :3]
    alpha = arr[..., 3:4]

    rgb_white_bg = rgb * alpha + (1.0 - alpha) * 1.0

    alpha_min = float(alpha.min())
    alpha_max = float(alpha.max())

    print(
        f"[alpha check] path={path} "
        f"min={alpha_min:.4f}, "
        f"max={alpha_max:.4f}"
    )

    if alpha_min > 0.999 and alpha_max > 0.999:
        raise ValueError(
            "PNG has no transparency alpha channel. "
            "Please upload REAL RGBA PNG exported with transparency."
        )

    rgb_t = torch.from_numpy(rgb_white_bg.transpose(2, 0, 1)).float()
    alpha_t = torch.from_numpy(alpha.transpose(2, 0, 1)).float()

    alpha_t = (alpha_t > 0.05).float()

    return rgb_t, alpha_t


def _prepare_hull_masks_from_images(
    images: List[object],
    image_size: int,
    device: str,
    dilate_px: int = 0,
) -> torch.Tensor:
    """Prepare hull masks from the uploaded image alpha channels.

    This matches the training dataset logic:
        hull_rgb, hull_alpha = self._load_rgba_with_alpha(hull_path)
        hull_masks.append(hull_alpha.unsqueeze(0))

    Returns:
        [V, 1, H, W], where 1 means allowed hull region and 0 means forbidden background.
    """
    masks = []
    for im in images:
        _, alpha = _load_rgba_with_alpha_any(im, image_size)
        masks.append(alpha)

    masks = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)

    if dilate_px and dilate_px > 1:
        pad = dilate_px // 2
        masks = torch.nn.functional.max_pool2d(masks, kernel_size=dilate_px, stride=1, padding=pad)

    return masks.clamp(0.0, 1.0)


def _hull_masks_to_preview_images(hull_mask: torch.Tensor) -> Tuple[List[np.ndarray], np.ndarray]:
    """Convert [V,1,H,W] mask tensor to displayable white/black images.

    White = allowed hull/generation area.
    Black = forbidden background area.
    """
    masks = hull_mask.detach().float().cpu().clamp(0.0, 1.0)
    arr = (masks[:, 0].numpy() * 255.0).astype(np.uint8)
    images = [np.stack([arr[i], arr[i], arr[i]], axis=-1) for i in range(arr.shape[0])]
    grid = np.concatenate(images, axis=1)
    return images, grid


@torch.no_grad()
def enforce_hull_boundary_on_latent(
    model,
    latent: torch.Tensor,
    hull_mask: torch.Tensor,
    white_bg: float = 1.0,
) -> torch.Tensor:
    """Project latent to the hull-valid image domain.

    latent:    [V, 4, h, w] latent from DDIM step.
    hull_mask: [V, 1, H, W], 1=inside hull, 0=outside hull.

    This decodes latent -> masks image outside hull to white -> encodes back.
    It is a strict 2D boundary constraint applied at every denoise step.
    """
    decoded = model.decode_first_stage(latent)
    decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    mask = hull_mask.to(device=decoded.device, dtype=decoded.dtype)
    if mask.shape[-2:] != decoded.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=decoded.shape[-2:], mode="nearest")

    decoded = decoded * mask + white_bg * (1.0 - mask)

    decoded = decoded * 2.0 - 1.0
    posterior = model.encode_first_stage(decoded)
    latent_projected = model.get_first_stage_encoding(posterior)
    return latent_projected


class HullGuidedDDIMSampler(DDIMSampler):
    """DDIM sampler that enforces hull masks after every denoise step.

    It keeps RGB hull conditioning unchanged. The alpha mask is only used as a strict
    generation boundary during sampling.
    """
    def __init__(self, model):
        super().__init__(model)
        self.hull_mask = None
        self.hull_start_step = 0
        self.hull_interval = 1
        self.hull_total_steps = 1
        self.hull_start_ratio = 0.4
        self.hull_strength = 0.3
        self._hull_step_counter = 0

    def set_hull_guidance(
        self,
        hull_mask: Optional[torch.Tensor],
        start_step: int = 0,
        interval: int = 1,
        total_steps: int = 1,
        start_ratio: float = 0.4,
        strength: float = 0.3,
    ):
        self.hull_mask = hull_mask
        self.hull_start_step = int(start_step)
        self.hull_interval = max(1, int(interval))
        self.hull_total_steps = max(1, int(total_steps))
        self.hull_start_ratio = max(0.0, min(1.0, float(start_ratio)))
        self.hull_strength = max(0.0, min(1.0, float(strength)))
        self._hull_step_counter = 0

    def _current_hull_blend_weight(self) -> float:
        if self.hull_strength <= 0.0:
            return 0.0
        denom = max(1, self.hull_total_steps - 1)
        step_ratio = max(0.0, min(1.0, self._hull_step_counter / denom))
        if step_ratio < self.hull_start_ratio:
            return 0.0
        ramp = (step_ratio - self.hull_start_ratio) / max(1e-6, 1.0 - self.hull_start_ratio)
        return self.hull_strength * float(ramp ** 2)

    @torch.no_grad()
    def p_sample_ddim(self, *args, **kwargs):
        out = super().p_sample_ddim(*args, **kwargs)

        # Common DDIMSampler returns (x_prev, pred_x0). Keep compatibility if extra values exist.
        if not isinstance(out, tuple) or len(out) < 2:
            return out

        x_prev = out[0]
        pred_x0 = out[1]

        do_project = (
            self.hull_mask is not None
            and self._hull_step_counter >= self.hull_start_step
            and ((self._hull_step_counter - self.hull_start_step) % self.hull_interval == 0)
        )

        if do_project:
            # Softly blend the mask-projected latent instead of hard replacing it.
            w = self._current_hull_blend_weight()
            if w > 0.0:
                x_prev_masked = enforce_hull_boundary_on_latent(self.model, x_prev, self.hull_mask)
                pred_x0_masked = enforce_hull_boundary_on_latent(self.model, pred_x0, self.hull_mask)
                x_prev = x_prev * (1.0 - w) + x_prev_masked * w
                pred_x0 = pred_x0 * (1.0 - w) + pred_x0_masked * w

        self._hull_step_counter += 1
        return (x_prev, pred_x0, *out[2:])

CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]


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


def _parse_pose_txt(pose_file) -> Optional[torch.Tensor]:
    """Parse uploaded pose txt into [N, 12] camera pose vectors.

    Supported formats:
    1. N lines, each line has 12 numbers.
    2. N lines, each line has 16 numbers, e.g. flattened 4x4 matrix; first 12 are used.
    3. A single long list of numbers that can be reshaped to [-1, 12] or [-1, 16].

    This is designed to reproduce the training-time modality where reference poses
    are provided as camera vectors rather than manually entered elevation/azimuth sliders.
    """
    if pose_file is None:
        return None

    path = pose_file.name if hasattr(pose_file, "name") else str(pose_file)
    values: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            values.extend(float(x) for x in line.split())

    if len(values) == 0:
        raise ValueError("Pose txt is empty.")

    arr = np.asarray(values, dtype=np.float32)

    if arr.size % 16 == 0:
        arr = arr.reshape(-1, 16)[:, :12]
    elif arr.size % 12 == 0:
        arr = arr.reshape(-1, 12)
    else:
        raise ValueError(
            f"Pose txt has {arr.size} numbers, which cannot be reshaped to Nx12 or Nx16."
        )

    return torch.from_numpy(arr)  # [N, 12]


def _parse_single_pose_txt(pose_file) -> Optional[torch.Tensor]:
    """Parse one uploaded pose txt into a single [12] pose vector.

    The file may contain either 12 numbers or a flattened 4x4 matrix with 16 numbers.
    If more than one pose is present, only the first pose is used.
    """
    poses = _parse_pose_txt(pose_file)
    if poses is None:
        return None
    if poses.shape[0] == 0:
        return None
    return poses[0]


def _collect_image_pose_pairs(
    images: List[Optional[object]],
    pose_files: List[Optional[object]],
    expected_num_refs: int = 4,
) -> Tuple[List[object], torch.Tensor]:
    """Collect valid (image, pose) pairs and expand them to expected_num_refs.

    image1 must pair with pose1, image2 with pose2, etc. A pair is valid only when
    both the image and corresponding pose txt are uploaded.
    """
    pairs = []
    missing = []
    for idx, (image, pose_file) in enumerate(zip(images, pose_files), start=1):
        if image is None and pose_file is None:
            continue
        if image is None or pose_file is None:
            missing.append(idx)
            continue
        pose = _parse_single_pose_txt(pose_file)
        if pose is None:
            raise ValueError(f"Pose {idx} txt is empty or invalid.")
        pairs.append((image, pose))

    if missing:
        raise ValueError(
            "Each uploaded image must have its corresponding pose txt. "
            f"Missing image/pose pair at slot(s): {missing}"
        )
    if len(pairs) == 0:
        raise ValueError("Please upload at least one image and its corresponding pose txt.")

    pairs = _expand_to_length(pairs, expected_num_refs)
    valid_images = [p[0] for p in pairs]
    poses = torch.stack([p[1] for p in pairs], dim=0)  # [expected_num_refs, 12]
    return valid_images, poses


def _expand_to_length(items: List, target_len: int) -> List:
    if len(items) == 0:
        return items
    if len(items) >= target_len:
        return items[:target_len]
    out = list(items)
    idx = 0
    while len(out) < target_len:
        out.append(items[idx % len(items)])
        idx += 1
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_adapter_weights(model, adapter_ckpt_path: str, lora_rank: int, lora_alpha: float, device: str):
    replaced = inject_lora(model, r=lora_rank, alpha=lora_alpha)
    print(f"Injected LoRA into {replaced} modules for inference.")

    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    model_state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"Loaded adapter checkpoint: {adapter_ckpt_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    cond_proj = None
    cond_proj_state = ckpt.get("cond_proj_state")
    if cond_proj_state is not None:
        weight = cond_proj_state["weight"]
        out_features, in_features = weight.shape
        cond_proj = torch.nn.Linear(in_features, out_features)
        cond_proj.load_state_dict(cond_proj_state)
        cond_proj.to(device)
        cond_proj.eval()
        print("Loaded cond_proj from adapter checkpoint.")
    else:
        print("cond_proj_state not found in checkpoint; using embedding dim auto-match.")

    ref_pose_proj = None
    ref_pose_proj_state = ckpt.get("ref_pose_proj_state")
    if ref_pose_proj_state is not None:
        weight = ref_pose_proj_state["weight"]
        out_features, in_features = weight.shape
        ref_pose_proj = torch.nn.Linear(in_features, out_features)
        ref_pose_proj.load_state_dict(ref_pose_proj_state)
        ref_pose_proj.to(device)
        ref_pose_proj.eval()
        print("Loaded ref_pose_proj from adapter checkpoint.")
    else:
        print("ref_pose_proj_state not found in checkpoint; reference poses will be ignored.")

    alpha_mask_encoder = None
    alpha_mask_encoder_state = ckpt.get("alpha_mask_encoder_state")
    if alpha_mask_encoder_state is not None:
        pos_embed = alpha_mask_encoder_state["pos_embed"]
        _, num_tokens, embed_dim = pos_embed.shape
        grid_size = int(round(num_tokens ** 0.5))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Invalid alpha mask token count in checkpoint: {num_tokens}")
        alpha_mask_encoder = AlphaMaskEmbedder(embed_dim=embed_dim, grid_size=grid_size)
        alpha_mask_encoder.load_state_dict(alpha_mask_encoder_state)
        alpha_mask_encoder.to(device)
        alpha_mask_encoder.eval()
        print(f"Loaded alpha_mask_encoder from adapter checkpoint: tokens={num_tokens}, dim={embed_dim}.")
    else:
        print("alpha_mask_encoder_state not found in checkpoint; alpha mask condition tokens will be disabled.")

    return cond_proj, ref_pose_proj, alpha_mask_encoder


def _load_image_encoder_weights(image_encoder: ImageEmbedder, adapter_ckpt_path: str, device: str) -> None:
    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    image_encoder_state = ckpt.get("image_encoder_state")
    if image_encoder_state is not None:
        image_encoder.load_state_dict(image_encoder_state)
        image_encoder.to(device)
        image_encoder.eval()
        print("Loaded image_encoder_state from adapter checkpoint.")
    else:
        print("[warn] image_encoder_state not found in checkpoint; hull/style conditioning may be weak with this adapter.")


def _prepare_hull_embed(
    image_encoder: ImageEmbedder,
    images: List[object],
    image_size: int,
    device: str,
    cond_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> torch.Tensor:
    """Encode the RGB hull images into conditioning tokens.

    This uses the RGB output from _load_rgba_with_alpha_any, while the alpha output
    is separately used as the strict denoise boundary mask.
    """
    if len(images) == 0:
        raise ValueError("Please upload at least one image.")

    embeds = []
    with torch.no_grad():
        for im in images:
            rgb, _ = _load_rgba_with_alpha_any(im, image_size)  # [3,H,W], [0,1]
            x = rgb.unsqueeze(0).to(device)
            emb = image_encoder.encode(x)  # (1, 1, D)
            embeds.append(emb)

    hull_embed = torch.cat(embeds, dim=0)  # (num_images, num_tokens, D)
    if hull_embed.shape[-1] != target_text_dim:
        if cond_proj is not None:
            hull_embed = cond_proj(hull_embed)
        else:
            raise ValueError(
                f"Hull embedding dim {hull_embed.shape[-1]} != text dim {target_text_dim}, "
                "and no cond_proj_state exists in checkpoint."
            )
    return hull_embed

def _prepare_reference_pose_tokens_from_poses(
    poses: torch.Tensor,
    device: str,
    ref_pose_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> Optional[torch.Tensor]:
    """Create reference pose tokens from paired poses.

    Args:
        poses: [4, 12], one pose for each reference image slot after expansion.

    Returns:
        pose_rep: [4, 1, target_text_dim]
    """
    if ref_pose_proj is None:
        print("[warn] ref_pose_proj_state not found in checkpoint; uploaded pose txt will be ignored.")
        return None

    poses = poses.to(device)                          # [4, 12]
    pose_rep = ref_pose_proj(poses).unsqueeze(1)       # [4, 1, target_text_dim]
    if pose_rep.shape[-1] != target_text_dim:
        raise ValueError(
            f"Reference pose dim {pose_rep.shape[-1]} != text dim {target_text_dim}."
        )
    return pose_rep


def _prepare_alpha_mask_tokens(
    alpha_mask_encoder: Optional[AlphaMaskEmbedder],
    hull_mask: torch.Tensor,
    target_text_dim: int,
) -> Optional[torch.Tensor]:
    """Create alpha mask condition tokens from [V,1,H,W] hull masks."""
    if alpha_mask_encoder is None:
        return None
    with torch.no_grad():
        mask_rep = alpha_mask_encoder(hull_mask)
    if mask_rep.shape[-1] != target_text_dim:
        raise ValueError(
            f"Alpha mask token dim {mask_rep.shape[-1]} != text dim {target_text_dim}."
        )
    return mask_rep


def _prepare_camera_tensor_from_poses(
    poses: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Create target camera tensor directly from paired poses.

    This matches the training code: the same pose vectors used for pose conditioning
    are also passed as the model camera condition.

    Returns:
        camera_tensor: [4, 16], flattened 4x4 camera matrices.
    """
    poses = poses.to(device)  # [4, 12]
    num_frames = poses.shape[0]
    poses_3x4 = poses.view(num_frames, 3, 4)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=poses_3x4.dtype)
    bottom = bottom.view(1, 1, 4).repeat(num_frames, 1, 1)
    poses_4x4 = torch.cat([poses_3x4, bottom], dim=1)  # [4, 4, 4]
    camera_tensor = poses_4x4.reshape(num_frames, 16)
    return camera_tensor

def _sample_multiview(
    model,
    sampler,
    prompt: str,
    negative_prompt: str,
    category_name: str,
    hull_embed: torch.Tensor,
    alpha_mask_embed: Optional[torch.Tensor],
    ref_pose_embed: Optional[torch.Tensor],
    target_camera: torch.Tensor,
    image_size: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    elevation: int,
    azimuth: int,
    num_frames: int,
    device: str,
    fp16: bool,
    hull_mask: Optional[torch.Tensor] = None,
    enforce_denoise_mask: bool = False,
    hull_start_ratio: float = 0.0,
    hull_interval: int = 1,
    hull_strength: float = 0.3,
    # strict_camera: bool,  # 已移除，始终 batch 生成
) -> Tuple[List[np.ndarray], np.ndarray]:
    set_seed(seed)
    dtype = torch.float16 if fp16 and device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype == torch.float16 else contextlib.nullcontext()


    with torch.no_grad(), amp_ctx:
        text_prompt = prompt.strip() if prompt.strip() else category_name
        text_c = model.get_learned_conditioning([text_prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)
        camera = target_camera.to(device)
        c_text = text_c.repeat(num_frames, 1, 1)
        uc_text_batch = uc_text.repeat(num_frames, 1, 1)
        hull_rep = hull_embed.to(device)
        if hull_rep.shape[0] != num_frames:
            raise ValueError(
                f"Hull embedding batch {hull_rep.shape[0]} != num_frames {num_frames}."
            )
        mask_rep = alpha_mask_embed.to(device) if alpha_mask_embed is not None else None
        if mask_rep is not None and mask_rep.shape[0] != num_frames:
            raise ValueError(
                f"Alpha mask token batch {mask_rep.shape[0]} != num_frames {num_frames}."
            )
        if ref_pose_embed is not None:
            pose_rep = ref_pose_embed.to(device)
            if pose_rep.shape[0] != num_frames:
                raise ValueError(
                    f"Reference pose token batch {pose_rep.shape[0]} != num_frames {num_frames}."
                )
            if mask_rep is not None:
                context_cat = torch.cat([c_text, hull_rep, mask_rep, pose_rep], dim=1)
                uc_context_cat = torch.cat(
                    [uc_text_batch, torch.zeros_like(hull_rep), torch.zeros_like(mask_rep), torch.zeros_like(pose_rep)],
                    dim=1,
                )
            else:
                context_cat = torch.cat([c_text, hull_rep, pose_rep], dim=1)
                uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)], dim=1)
        else:
            if mask_rep is not None:
                context_cat = torch.cat([c_text, hull_rep, mask_rep], dim=1)
                uc_context_cat = torch.cat(
                    [uc_text_batch, torch.zeros_like(hull_rep), torch.zeros_like(mask_rep)],
                    dim=1,
                )
            else:
                context_cat = torch.cat([c_text, hull_rep], dim=1)
                uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_rep)], dim=1)

        c_ = {
            "context": context_cat,
            "camera": camera,
            "num_frames": num_frames,
        }
        uc_ = {
            "context": uc_context_cat,
            "camera": camera,
            "num_frames": num_frames,
        }

        shape = [4, image_size // 8, image_size // 8]

        # Optional iterative hull-boundary projection during DDIM sampling.
        # This is a soft blend, not a hard replacement.
        if hasattr(sampler, "set_hull_guidance"):
            if enforce_denoise_mask:
                start_step = int(max(0.0, min(1.0, hull_start_ratio)) * int(steps))
                sampler.set_hull_guidance(
                    hull_mask=hull_mask,
                    start_step=start_step,
                    interval=max(1, int(hull_interval)),
                    total_steps=int(steps),
                    start_ratio=float(hull_start_ratio),
                    strength=float(hull_strength),
                )
            else:
                sampler.set_hull_guidance(hull_mask=None)

        samples, _ = sampler.sample(
            S=steps,
            conditioning=c_,
            batch_size=num_frames,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=None,
        )
        decoded = model.decode_first_stage(samples)
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

        arr = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        images = [arr[i] for i in range(arr.shape[0])]

    # 后处理：将黑色背景替换为白色
    images = [set_white_background(im) for im in images]
    grid = np.concatenate(images, axis=1)
    return images, grid


def build_app(args):
    print("Loading base model...")
    model = build_model(args.model_name, ckpt_path=args.base_ckpt)
    model.to(args.device)
    model.eval()

    cond_proj, ref_pose_proj, alpha_mask_encoder = _load_adapter_weights(
        model,
        adapter_ckpt_path=args.adapter_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )
    model.to(args.device)
    model.eval()

    sampler = HullGuidedDDIMSampler(model)
    image_encoder = ImageEmbedder(device=args.device, img_size=args.size)
    _load_image_encoder_weights(image_encoder, adapter_ckpt_path=args.adapter_ckpt, device=args.device)
    image_encoder.eval()

    def _save_mask_zip(mask_grid, mask_images):
        """Create a ZIP containing the generated hull alpha masks on every Generate click."""
        try:
            tmpdir = tempfile.mkdtemp(prefix="hull_masks_")

            if mask_grid is not None:
                Image.fromarray(mask_grid.astype(np.uint8)).save(
                    os.path.join(tmpdir, "hull_mask_grid.png")
                )

            if mask_images:
                for i, im in enumerate(mask_images):
                    if im is None:
                        continue
                    arr = im.astype(np.uint8) if isinstance(im, np.ndarray) else np.asarray(im).astype(np.uint8)
                    Image.fromarray(arr).save(
                        os.path.join(tmpdir, f"hull_mask_alpha_{i+1}.png")
                    )

            zip_base = os.path.join(tmpdir, "hull_masks")
            return shutil.make_archive(zip_base, "zip", root_dir=tmpdir)
        except Exception as e:
            print(f"[WARN] Failed to create hull mask zip: {e}")
            return None

    def infer(
        image1,
        pose1_txt,
        image2,
        pose2_txt,
        image3,
        pose3_txt,
        image4,
        pose4_txt,
        cat_entity,
        cat_volume,
        cat_direction,
        cat_operation,
        cat_affect,
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        seed,
        enforce_denoise_mask,
        hull_start_ratio,
        hull_interval,
        hull_strength,
        hull_dilate_px,
        # strict_camera,  # 已移除
    ):
        try:
            cat_values = {
                "entity": cat_entity.strip(),
                "volume": cat_volume.strip(),
                "direction": cat_direction.strip(),
                "operation": cat_operation.strip(),
                "affect": cat_affect.strip(),
            }
            cat_str = ", ".join(f"{k}: {v}" for k, v in cat_values.items() if v)
            category_name = cat_str if cat_str else "object"

            text_dim = model.get_learned_conditioning([category_name]).shape[-1]
            paired_images, paired_poses = _collect_image_pose_pairs(
                images=[image1, image2, image3, image4],
                pose_files=[pose1_txt, pose2_txt, pose3_txt, pose4_txt],
                expected_num_refs=4,
            )

            hull_embed = _prepare_hull_embed(
                image_encoder=image_encoder,
                images=paired_images,
                image_size=args.size,
                device=args.device,
                cond_proj=cond_proj,
                target_text_dim=text_dim,
            )
            ref_pose_embed = _prepare_reference_pose_tokens_from_poses(
                poses=paired_poses,
                device=args.device,
                ref_pose_proj=ref_pose_proj,
                target_text_dim=text_dim,
            )
            target_camera = _prepare_camera_tensor_from_poses(
                poses=paired_poses,
                device=args.device,
            )

            # Binary masks are derived from the same RGB hull reference images.
            # RGB hull image remains in conditioning; hull_mask is only a strict sampling boundary.
            hull_mask = _prepare_hull_masks_from_images(
                images=paired_images,
                image_size=args.size,
                device=args.device,
                dilate_px=int(hull_dilate_px),
            )
            mask_images, mask_grid = _hull_masks_to_preview_images(hull_mask)
            alpha_mask_embed = _prepare_alpha_mask_tokens(
                alpha_mask_encoder=alpha_mask_encoder,
                hull_mask=hull_mask,
                target_text_dim=text_dim,
            )

            images, grid = _sample_multiview(
                model=model,
                sampler=sampler,
                prompt=prompt,
                negative_prompt=negative_prompt,
                category_name=category_name,
                hull_embed=hull_embed,
                alpha_mask_embed=alpha_mask_embed,
                ref_pose_embed=ref_pose_embed,
                target_camera=target_camera,
                image_size=args.size,
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                elevation=0,
                azimuth=0,
                num_frames=4,
                device=args.device,
                fp16=args.fp16,
                hull_mask=hull_mask,
                enforce_denoise_mask=bool(enforce_denoise_mask),
                hull_start_ratio=float(hull_start_ratio),
                hull_interval=int(hull_interval),
                hull_strength=float(hull_strength),
                
            )
            mask_zip = _save_mask_zip(mask_grid, mask_images)
            status = (
                f"Done. category='{category_name}', paired refs={len(paired_images)}, "
                f"alpha_mask=True, mask_condition={alpha_mask_embed is not None}, "
                f"denoise_mask={bool(enforce_denoise_mask)}, denoise_strength={float(hull_strength):.2f}, "
                f"dilate_px={int(hull_dilate_px)}, mask_zip_ready={mask_zip is not None}"
            )
            return grid, images, mask_grid, mask_images, status, mask_zip
        except Exception as e:
            return None, None, None, None, f"Error: {e}", None

    with gr.Blocks(title="MVDream Adapter Inference") as demo:
        gr.Markdown("## MVDream Adapter Inference\nUpload 1-4 reference images with matching pose txt files, fill in category fields, and generate the corresponding views.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    image1 = gr.File(
                        label="Reference Image 1 RGBA PNG",
                        file_types=[".png"],
                    )
                    pose1_txt = gr.File(label="Pose 1 TXT (for Image 1)", file_types=[".txt"])
                with gr.Group():
                    image2 = gr.File(
                        label="Reference Image 2 RGBA PNG",
                        file_types=[".png"],
                    )
                    pose2_txt = gr.File(label="Pose 2 TXT (for Image 2)", file_types=[".txt"])
                with gr.Group():
                    image3 = gr.File(
                        label="Reference Image 3 RGBA PNG",
                        file_types=[".png"],
                    )
                    pose3_txt = gr.File(label="Pose 3 TXT (for Image 3)", file_types=[".txt"])
                with gr.Group():
                    image4 = gr.File(
                        label="Reference Image 4 RGBA PNG",
                        file_types=[".png"],
                    )
                    pose4_txt = gr.File(label="Pose 4 TXT (for Image 4)", file_types=[".txt"])
                gr.Markdown("### Category")
                cat_entity    = gr.Textbox(value="", label="entity",    placeholder="e.g. single")
                cat_volume    = gr.Textbox(value="", label="volume",    placeholder="e.g. cuboid")
                cat_direction = gr.Textbox(value="", label="direction", placeholder="e.g. planar")
                cat_operation = gr.Textbox(value="", label="operation", placeholder="e.g. perforating")
                cat_affect    = gr.Textbox(value="", label="affect",    placeholder="e.g. porous")
                prompt = gr.Textbox(value="", label="Prompt (optional)")
                negative_prompt = gr.Textbox(value="", label="Negative Prompt")
                steps = gr.Slider(10, 80, value=30, step=1, label="Sampling Steps")
                guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance Scale")
                seed = gr.Number(value=23, precision=0, label="Seed")
                gr.Markdown("### Strict Hull Boundary (uses uploaded PNG alpha as mask)")
                enforce_denoise_mask = gr.Checkbox(
                    value=False,
                    label="Force Mask During Denoise",
                )
                hull_start_ratio = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Start Enforcing After Ratio of Steps")
                hull_interval = gr.Slider(1, 5, value=1, step=1, label="Enforce Every N DDIM Steps")
                hull_strength = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Denoise Mask Blend Strength")
                hull_dilate_px = gr.Slider(0, 15, value=3, step=2, label="Mask Dilation Pixels")
                # strict_camera = gr.Checkbox(value=True, label="Strict Camera (sample views one-by-one)")  # 已移除
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                grid_out = gr.Image(type="numpy", label="4-View Grid")
                gallery_out = gr.Gallery(label="Generated Views", columns=4, object_fit="contain", height=220)
                mask_grid_out = gr.Image(type="numpy", label="Hull Mask Grid (white=allowed, black=forbidden)")
                mask_gallery_out = gr.Gallery(label="Hull Masks Per View", columns=4, object_fit="contain", height=180)
                status = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("Download Results")
                download_file = gr.File(label="Download Results")

        run_btn.click(
            infer,
            inputs=[image1, pose1_txt, image2, pose2_txt, image3, pose3_txt, image4, pose4_txt,
                    cat_entity, cat_volume, cat_direction, cat_operation, cat_affect,
                    prompt, negative_prompt, steps, guidance_scale, seed,
                    enforce_denoise_mask, hull_start_ratio, hull_interval, hull_strength, hull_dilate_px],
            outputs=[grid_out, gallery_out, mask_grid_out, mask_gallery_out, status, download_file],
        )

        def _make_download(grid, images, mask_grid, mask_images):
            """Save grid, individual views, and mask previews to a ZIP."""
            try:
                if grid is None:
                    return None
                tmpdir = tempfile.mkdtemp(prefix="mvdream_")

                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(tmpdir, "grid.png"))
                if mask_grid is not None:
                    Image.fromarray(mask_grid.astype(np.uint8)).save(os.path.join(tmpdir, "hull_mask_grid.png"))

                if images:
                    for i, im in enumerate(images):
                        if im is None:
                            continue
                        img = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
                        img.save(os.path.join(tmpdir, f"view_{i+1}.png"))

                if mask_images:
                    for i, im in enumerate(mask_images):
                        if im is None:
                            continue
                        img = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
                        img.save(os.path.join(tmpdir, f"hull_mask_{i+1}.png"))

                zip_base = os.path.join(tmpdir, "results")
                return shutil.make_archive(zip_base, 'zip', root_dir=tmpdir)
            except Exception:
                return None

        download_btn.click(_make_download, inputs=[grid_out, gallery_out, mask_grid_out, mask_gallery_out], outputs=[download_file])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Optional base model checkpoint path")
    parser.add_argument("--adapter_ckpt", type=str, default="checkpoints/ckpt_epoch_0.pth", help="Adapter-only checkpoint path")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args)
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "show_api": False,
    }
    launch_sig = inspect.signature(app.launch)
    launch_kwargs = {k: v for k, v in launch_kwargs.items() if k in launch_sig.parameters}
    try:
        app.launch(**launch_kwargs)
    except ValueError as e:
        # Some environments cannot access localhost during Gradio checks.
        if "localhost is not accessible" in str(e) and not args.share:
            print("[warn] localhost check failed; retrying with share=True")
            launch_kwargs["share"] = True
            app.launch(**launch_kwargs)
        else:
            raise
