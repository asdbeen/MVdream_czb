import os
import sys
import argparse
import contextlib
import random
import inspect
from functools import partial
from typing import List, Optional

import gradio as gr
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_load_lora_adapter(model, adapter_ckpt_path: Optional[str], lora_rank: int, lora_alpha: float) -> None:
    if not adapter_ckpt_path:
        print("No adapter checkpoint provided. Running base model only.")
        return

    if not os.path.isfile(adapter_ckpt_path):
        raise FileNotFoundError(f"adapter_ckpt not found: {adapter_ckpt_path}")

    replaced = inject_lora(model, r=lora_rank, alpha=lora_alpha)
    print(f"Injected LoRA into {replaced} modules.")

    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    model_state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(
        f"Loaded adapter checkpoint: {adapter_ckpt_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


@torch.no_grad()
def t2i(
    model,
    image_size: int,
    prompt,
    uc,
    sampler,
    step: int = 20,
    scale: float = 7.5,
    batch_size: int = 8,
    ddim_eta: float = 0.0,
    dtype=torch.float32,
    device: str = "cuda",
    camera=None,
    num_frames: int = 1,
) -> List[np.ndarray]:
    if not isinstance(prompt, list):
        prompt = [prompt]

    use_autocast = (
        dtype == torch.float16
        and isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

    with amp_ctx:
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}

        if camera is not None:
            c_["camera"] = camera
            uc_["camera"] = camera
            c_["num_frames"] = num_frames
            uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def generate_images(args, model, sampler, text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera):
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    if use_camera:
        camera = get_camera(args.num_frames, elevation=elevation, azimuth_start=azimuth)
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
        num_frames = args.num_frames
    else:
        camera = None
        num_frames = 1

    prompt = text_input + args.suffix
    uc = model.get_learned_conditioning([uncond_text_input]).to(device)

    set_seed(int(seed))
    images = []
    for _ in range(2):
        img = t2i(
            model,
            args.size,
            prompt,
            uc,
            sampler,
            step=int(step),
            scale=float(guidance_scale),
            batch_size=batch_size,
            ddim_eta=0.0,
            dtype=dtype,
            device=device,
            camera=camera,
            num_frames=num_frames,
        )
        img = np.concatenate(img, axis=1)
        images.append(img)

    images = np.concatenate(images, axis=0)
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="Load pre-trained model from Hugging Face")
    parser.add_argument("--config_path", type=str, default=None, help="Load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to local base checkpoint")
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="Path to LoRA adapter checkpoint (adapter-only from training)")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank used during training")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha used during training")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        raise ValueError("config_path mode is not implemented in this launcher. Please use model_name/ckpt_path.")

    _maybe_load_lora_adapter(
        model,
        adapter_ckpt_path=args.adapter_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    model.to(args.device)
    model.eval()

    sampler = DDIMSampler(model)
    print("Model ready.")

    fn_with_model = partial(generate_images, args, model, sampler)

    with gr.Blocks() as demo:
        gr.Markdown("MVDream OG-style demo with optional LoRA adapter loading.")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(value="", label="prompt")
                uncond_text_input = gr.Textbox(value="", label="negative prompt")
                seed = gr.Number(value=23, label="seed", precision=0)
                guidance_scale = gr.Number(value=7.5, label="guidance_scale")
                step = gr.Number(value=25, label="sample steps", precision=0)
                elevation = gr.Slider(0, 30, value=15, label="Elevation", info="Choose between 0 and 30")
                azimuth = gr.Slider(0, 360, value=0, label="Azimuth", info="Choose between 0 and 360")
                use_camera = gr.Checkbox(value=True, label="Multi-view Mode", info="Multi-view mode or independent images")
                text_button = gr.Button("Generate Images")
            with gr.Column():
                image_output = gr.Image()

        inputs = [text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera]
        default_params = ["", 23, 7.5, 30, 15, 0, True]
        gr.Examples(
            [
                ["an astronaut riding a horse"] + default_params,
                ["an earth"] + default_params,
                ["a statue of a cute cat"] + default_params,
                ["Luffy from one piece, head, super detailed, best quality, 4K, HD"] + default_params,
                ["highly detailed, majestic royal tall ship, realistic painting"] + default_params,
            ],
            inputs,
            image_output,
            fn_with_model,
            cache_examples=True,
        )

        text_button.click(fn_with_model, inputs=inputs, outputs=image_output)

    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "show_api": False,
    }
    launch_sig = inspect.signature(demo.launch)
    launch_kwargs = {k: v for k, v in launch_kwargs.items() if k in launch_sig.parameters}
    demo.launch(**launch_kwargs)
