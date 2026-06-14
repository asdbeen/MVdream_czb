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

CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]


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
    images: List[Optional[np.ndarray]],
    pose_files: List[Optional[object]],
    expected_num_refs: int = 4,
) -> Tuple[List[np.ndarray], torch.Tensor]:
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

    return cond_proj, ref_pose_proj


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
    images: List[np.ndarray],
    image_size: int,
    device: str,
    cond_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> torch.Tensor:
    """Encode already-paired reference images into hull/image tokens.

    images should already have length 4 after pair expansion.
    """
    if len(images) == 0:
        raise ValueError("Please upload at least one image.")

    to_tensor = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    embeds = []
    with torch.no_grad():
        for im in images:
            x = to_tensor(im).unsqueeze(0).to(device)
            emb = image_encoder.encode(x)  # (1, 1, D)
            embeds.append(emb)

    hull_embed = torch.cat(embeds, dim=0)  # (num_images, 1, D)
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
        if ref_pose_embed is not None:
            pose_rep = ref_pose_embed.to(device)
            if pose_rep.shape[0] != num_frames:
                raise ValueError(
                    f"Reference pose token batch {pose_rep.shape[0]} != num_frames {num_frames}."
                )
            context_cat = torch.cat([c_text, hull_rep, pose_rep], dim=1)
            uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)], dim=1)
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

    cond_proj, ref_pose_proj = _load_adapter_weights(
        model,
        adapter_ckpt_path=args.adapter_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )
    model.to(args.device)
    model.eval()

    sampler = DDIMSampler(model)
    image_encoder = ImageEmbedder(device=args.device, img_size=args.size)
    _load_image_encoder_weights(image_encoder, adapter_ckpt_path=args.adapter_ckpt, device=args.device)
    image_encoder.eval()

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

            images, grid = _sample_multiview(
                model=model,
                sampler=sampler,
                prompt=prompt,
                negative_prompt=negative_prompt,
                category_name=category_name,
                hull_embed=hull_embed,
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
                # strict_camera=bool(strict_camera),  # 已移除
            )
            status = f"Done. category='{category_name}', paired refs={len(paired_images)}"
            return grid, images, status
        except Exception as e:
            return None, None, f"Error: {e}"

    with gr.Blocks(title="MVDream Adapter Inference") as demo:
        gr.Markdown("## MVDream Adapter Inference\nUpload 1-4 reference images with matching pose txt files, fill in category fields, and generate the corresponding views.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    image1 = gr.Image(type="numpy", label="Reference Image 1 (required)")
                    pose1_txt = gr.File(label="Pose 1 TXT (for Image 1)", file_types=[".txt"])
                with gr.Group():
                    image2 = gr.Image(type="numpy", label="Reference Image 2 (optional)")
                    pose2_txt = gr.File(label="Pose 2 TXT (for Image 2)", file_types=[".txt"])
                with gr.Group():
                    image3 = gr.Image(type="numpy", label="Reference Image 3 (optional)")
                    pose3_txt = gr.File(label="Pose 3 TXT (for Image 3)", file_types=[".txt"])
                with gr.Group():
                    image4 = gr.Image(type="numpy", label="Reference Image 4 (optional)")
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
                # strict_camera = gr.Checkbox(value=True, label="Strict Camera (sample views one-by-one)")  # 已移除
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                grid_out = gr.Image(type="numpy", label="4-View Grid")
                gallery_out = gr.Gallery(label="Generated Views", columns=4, object_fit="contain", height=220)
                status = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("Download Results")
                download_file = gr.File(label="Download Results")

        run_btn.click(
            infer,
            inputs=[image1, pose1_txt, image2, pose2_txt, image3, pose3_txt, image4, pose4_txt,
                    cat_entity, cat_volume, cat_direction, cat_operation, cat_affect,
                    prompt, negative_prompt, steps, guidance_scale, seed],
            outputs=[grid_out, gallery_out, status],
        )

        def _make_download(grid, images):
            """Save grid image and individual views to a temporary ZIP or PNG and return file path (for Gradio compatibility)."""
            try:
                if grid is None:
                    return None
                tmpdir = tempfile.mkdtemp(prefix="mvdream_")
                # Save grid
                grid_img = Image.fromarray(grid.astype(np.uint8))
                grid_path = os.path.join(tmpdir, "grid.png")
                grid_img.save(grid_path)

                # Save individual views
                img_paths = []
                if images:
                    for i, im in enumerate(images):
                        if im is None:
                            continue
                        if isinstance(im, np.ndarray):
                            img = Image.fromarray(im.astype(np.uint8))
                        else:
                            img = im
                        p = os.path.join(tmpdir, f"view_{i+1}.png")
                        img.save(p)
                        img_paths.append(p)

                # 如果只有一张图片，直接返回该图片路径
                if len(img_paths) == 1:
                    return img_paths[0]

                # 否则打包为zip，返回zip路径
                zip_base = os.path.join(tmpdir, "results")
                zip_path = shutil.make_archive(zip_base, 'zip', root_dir=tmpdir)
                return zip_path
            except Exception:
                return None

        download_btn.click(_make_download, inputs=[grid_out, gallery_out], outputs=[download_file])

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
