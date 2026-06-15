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
    images: List[Optional[np.ndarray]],
    image_size: int,
    device: str,
    cond_proj: Optional[torch.nn.Module],
    target_text_dim: int,
    expected_num_refs: int = 4,
) -> torch.Tensor:
    valid = [im for im in images if im is not None]
    if len(valid) == 0:
        raise ValueError("Please upload at least one image.")
    valid = _expand_to_length(valid, expected_num_refs)

    to_tensor = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    embeds = []
    with torch.no_grad():
        for im in valid:
            x = to_tensor(im).unsqueeze(0).to(device)
            emb = image_encoder.encode(x)  # (1, 1, D)
            embeds.append(emb)

    hull_embed = torch.cat(embeds, dim=1)  # (1, num_images, D)
    if hull_embed.shape[-1] != target_text_dim:
        if cond_proj is not None:
            hull_embed = cond_proj(hull_embed)
        else:
            raise ValueError(
                f"Hull embedding dim {hull_embed.shape[-1]} != text dim {target_text_dim}, "
                "and no cond_proj_state exists in checkpoint."
            )
    return hull_embed


def _prepare_reference_pose_tokens(
    images: List[Optional[np.ndarray]],
    elevations: List[float],
    azimuths: List[float],
    device: str,
    ref_pose_proj: Optional[torch.nn.Module],
    target_text_dim: int,
    expected_num_refs: int = 4,
) -> Optional[torch.Tensor]:
    if ref_pose_proj is None:
        return None

    pose_tokens = []
    for image, elevation, azimuth in zip(images, elevations, azimuths):
        if image is None:
            continue
        pose = get_camera(1, elevation=float(elevation), azimuth_start=float(azimuth)).to(device)
        pose_tokens.append(pose[:, :12].unsqueeze(1))

    if len(pose_tokens) == 0:
        return None

    pose_tokens = _expand_to_length(pose_tokens, expected_num_refs)

    pose_tokens = torch.cat(pose_tokens, dim=1)
    pose_rep = ref_pose_proj(pose_tokens)
    if pose_rep.shape[-1] != target_text_dim:
        raise ValueError(
            f"Reference pose dim {pose_rep.shape[-1]} != text dim {target_text_dim}."
        )
    return pose_rep


def _sample_multiview(
    model,
    sampler,
    prompt: str,
    negative_prompt: str,
    category_name: str,
    hull_embed: torch.Tensor,
    ref_pose_embed: Optional[torch.Tensor],
    image_size: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    elevation: int,
    azimuth: int,
    num_frames: int,
    device: str,
    fp16: bool,
    strict_camera: bool,
) -> Tuple[List[np.ndarray], np.ndarray]:
    set_seed(seed)
    dtype = torch.float16 if fp16 and device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype == torch.float16 else contextlib.nullcontext()

    with torch.no_grad(), amp_ctx:
        text_prompt = prompt.strip() if prompt.strip() else category_name
        text_c = model.get_learned_conditioning([text_prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)

        if strict_camera:
            # Match training style more closely: one target camera per forward pass.
            image_list: List[np.ndarray] = []
            shape = [4, image_size // 8, image_size // 8]
            for frame_idx in range(num_frames):
                frame_azimuth = int((azimuth + frame_idx * 360.0 / num_frames) % 360)
                camera = get_camera(1, elevation=elevation, azimuth_start=frame_azimuth).to(device)

                c_text = text_c
                uc_text_single = uc_text
                hull_rep = hull_embed
                if ref_pose_embed is not None:
                    context_cat = torch.cat([c_text, hull_rep, ref_pose_embed], dim=1)
                    uc_context_cat = torch.cat(
                        [uc_text_single, torch.zeros_like(hull_rep), torch.zeros_like(ref_pose_embed)],
                        dim=1,
                    )
                else:
                    context_cat = torch.cat([c_text, hull_rep], dim=1)
                    uc_context_cat = torch.cat([uc_text_single, torch.zeros_like(hull_rep)], dim=1)

                c_ = {
                    "context": context_cat,
                    "camera": camera,
                    "num_frames": 1,
                }
                uc_ = {
                    "context": uc_context_cat,
                    "camera": camera,
                    "num_frames": 1,
                }

                sample, _ = sampler.sample(
                    S=steps,
                    conditioning=c_,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc_,
                    eta=0.0,
                    x_T=None,
                )
                decoded = model.decode_first_stage(sample)
                decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
                arr = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                image_list.append(arr[0])
            images = image_list
        else:
            camera = get_camera(num_frames, elevation=elevation, azimuth_start=azimuth).to(device)

            c_text = text_c.repeat(num_frames, 1, 1)
            uc_text_batch = uc_text.repeat(num_frames, 1, 1)
            hull_rep = hull_embed.repeat(num_frames, 1, 1)
            if ref_pose_embed is not None:
                pose_rep = ref_pose_embed.repeat(num_frames, 1, 1)
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
        image2,
        image3,
        image4,
        image1_elevation,
        image1_azimuth,
        image2_elevation,
        image2_azimuth,
        image3_elevation,
        image3_azimuth,
        image4_elevation,
        image4_azimuth,
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
        elevation,
        azimuth,
        strict_camera,
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
            hull_embed = _prepare_hull_embed(
                image_encoder=image_encoder,
                images=[image1, image2, image3, image4],
                image_size=args.size,
                device=args.device,
                cond_proj=cond_proj,
                target_text_dim=text_dim,
                expected_num_refs=4,
            )
            ref_pose_embed = _prepare_reference_pose_tokens(
                images=[image1, image2, image3, image4],
                elevations=[image1_elevation, image2_elevation, image3_elevation, image4_elevation],
                azimuths=[image1_azimuth, image2_azimuth, image3_azimuth, image4_azimuth],
                device=args.device,
                ref_pose_proj=ref_pose_proj,
                target_text_dim=text_dim,
                expected_num_refs=4,
            )

            images, grid = _sample_multiview(
                model=model,
                sampler=sampler,
                prompt=prompt,
                negative_prompt=negative_prompt,
                category_name=category_name,
                hull_embed=hull_embed,
                ref_pose_embed=ref_pose_embed,
                image_size=args.size,
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                elevation=int(elevation),
                azimuth=int(azimuth),
                num_frames=4,
                device=args.device,
                fp16=args.fp16,
                strict_camera=False,
            )
            status = f"Done. category='{category_name}', refs={sum(x is not None for x in [image1, image2, image3, image4])}"
            return grid, images, status
        except Exception as e:
            return None, None, f"Error: {e}"

    with gr.Blocks(title="MVDream Adapter Inference") as demo:
        gr.Markdown("## MVDream Adapter Inference\nUpload 1-4 reference images, fill in category fields, and generate 4 views.")
        with gr.Row():
            with gr.Column(scale=1):
                image1 = gr.Image(type="numpy", label="Reference Image 1 (required)")
                image1_elevation = gr.Slider(-90, 90, value=15, step=1, label="Image 1 Elevation")
                image1_azimuth = gr.Slider(0, 360, value=0, step=1, label="Image 1 Azimuth")
                image2 = gr.Image(type="numpy", label="Reference Image 2 (optional)")
                image2_elevation = gr.Slider(-90, 90, value=15, step=1, label="Image 2 Elevation")
                image2_azimuth = gr.Slider(0, 360, value=90, step=1, label="Image 2 Azimuth")
                image3 = gr.Image(type="numpy", label="Reference Image 3 (optional)")
                image3_elevation = gr.Slider(-90, 90, value=15, step=1, label="Image 3 Elevation")
                image3_azimuth = gr.Slider(0, 360, value=180, step=1, label="Image 3 Azimuth")
                image4 = gr.Image(type="numpy", label="Reference Image 4 (optional)")
                image4_elevation = gr.Slider(-90, 90, value=15, step=1, label="Image 4 Elevation")
                image4_azimuth = gr.Slider(0, 360, value=270, step=1, label="Image 4 Azimuth")
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
                elevation = gr.Slider(0, 30, value=15, step=1, label="Camera Elevation")
                azimuth = gr.Slider(0, 360, value=0, step=1, label="Camera Azimuth Start")
                strict_camera = gr.Checkbox(value=True, label="Strict Camera (sample views one-by-one)")
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                grid_out = gr.Image(type="numpy", label="4-View Grid")
                gallery_out = gr.Gallery(label="Generated Views", columns=4, object_fit="contain", height=220)
                status = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("Download Results")
                download_file = gr.File(label="Download Results")

        run_btn.click(
            infer,
            inputs=[image1, image2, image3, image4,
                image1_elevation, image1_azimuth,
                image2_elevation, image2_azimuth,
                image3_elevation, image3_azimuth,
                image4_elevation, image4_azimuth,
                    cat_entity, cat_volume, cat_direction, cat_operation, cat_affect,
                    prompt, negative_prompt, steps, guidance_scale, seed, elevation, azimuth, strict_camera],
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
