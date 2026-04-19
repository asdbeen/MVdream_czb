import os
import sys
import argparse
import contextlib
import random
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.encoders.modules import ImageEmbedder
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model

CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]


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


def _prepare_hull_embed(
    image_encoder: ImageEmbedder,
    images: List[Optional[np.ndarray]],
    image_size: int,
    device: str,
    cond_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> torch.Tensor:
    valid = [im for im in images if im is not None]
    if len(valid) == 0:
        raise ValueError("Please upload at least one image.")

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
) -> Tuple[List[np.ndarray], np.ndarray]:
    set_seed(seed)
    dtype = torch.float16 if fp16 and device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype == torch.float16 else contextlib.nullcontext()

    camera = get_camera(num_frames, elevation=elevation, azimuth_start=azimuth).to(device)

    with torch.no_grad(), amp_ctx:
        text_prompt = prompt.strip() if prompt.strip() else category_name
        text_c = model.get_learned_conditioning([text_prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)

        c_text = text_c.repeat(num_frames, 1, 1)
        uc_text = uc_text.repeat(num_frames, 1, 1)
        hull_rep = hull_embed.repeat(num_frames, 1, 1)
        if ref_pose_embed is not None:
            pose_rep = ref_pose_embed.repeat(num_frames, 1, 1)
            context_cat = torch.cat([c_text, hull_rep, pose_rep], dim=1)
            uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)], dim=1)
        else:
            context_cat = torch.cat([c_text, hull_rep], dim=1)
            uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep)], dim=1)

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
            )
            ref_pose_embed = _prepare_reference_pose_tokens(
                images=[image1, image2, image3, image4],
                elevations=[image1_elevation, image2_elevation, image3_elevation, image4_elevation],
                azimuths=[image1_azimuth, image2_azimuth, image3_azimuth, image4_azimuth],
                device=args.device,
                ref_pose_proj=ref_pose_proj,
                target_text_dim=text_dim,
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
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                grid_out = gr.Image(type="numpy", label="4-View Grid")
                gallery_out = gr.Gallery(label="Generated Views", columns=4, object_fit="contain", height=220)
                status = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            infer,
            inputs=[image1, image2, image3, image4,
                image1_elevation, image1_azimuth,
                image2_elevation, image2_azimuth,
                image3_elevation, image3_azimuth,
                image4_elevation, image4_azimuth,
                    cat_entity, cat_volume, cat_direction, cat_operation, cat_affect,
                    prompt, negative_prompt, steps, guidance_scale, seed, elevation, azimuth],
            outputs=[grid_out, gallery_out, status],
        )

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
