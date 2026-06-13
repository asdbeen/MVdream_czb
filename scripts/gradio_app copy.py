import os
import sys
import argparse
import contextlib
import random
import inspect
from typing import List, Optional

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.encoders.modules import ImageEmbedder
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model

def parse_pose_txt(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    pose_data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32).reshape(4, 4)
    return torch.from_numpy(pose_data[:3, :])

def build_app(args):
    print("Loading base model...")
    model = build_model(args.model_name, ckpt_path=args.base_ckpt)
    model.to(args.device)
    model.eval()

    cond_proj, ref_pose_proj = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha), None
    sampler = DDIMSampler(model)
    image_encoder = ImageEmbedder(device=args.device, img_size=args.size)
    image_encoder.eval()

    def infer(
        image1, pose1,
        image2, pose2,
        image3, pose3,
        image4, pose4,
        category,
        steps, guidance_scale, seed
    ):
        try:
            images = [image1, image2, image3, image4]
            poses = [pose1, pose2, pose3, pose4]
            # 处理图片
            to_tensor = T.Compose([
                T.ToPILImage() if isinstance(images[0], np.ndarray) else lambda x: x,
                T.Resize((args.size, args.size)),
                T.ToTensor(),
            ])
            embeds = []
            for im in images:
                if im is None:
                    raise ValueError("请上传4张图片")
                x = to_tensor(im).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    emb = image_encoder.encode(x)
                embeds.append(emb)
            hull_embed = torch.cat(embeds, dim=1)
            # 处理pose
            pose_tensors = []
            for p in poses:
                if p is None:
                    raise ValueError("请上传4个pose txt")
                # 解析txt
                if isinstance(p, str):
                    pose_tensor = parse_pose_txt(p)
                else:
                    raise ValueError("pose必须为txt文件")
                pose_tensors.append(pose_tensor)
            camera_tensor = torch.stack(pose_tensors, dim=0).reshape(4, -1).to(args.device)
            # 文本
            text_c = model.get_learned_conditioning([category]*4).to(args.device)
            # 采样
            c_ = {"context": text_c, "camera": camera_tensor, "num_frames": 4}
            uc_ = {"context": torch.zeros_like(text_c), "camera": camera_tensor, "num_frames": 4}
            shape = [4, args.size // 8, args.size // 8]
            with torch.no_grad():
                samples, _ = sampler.sample(
                    S=int(steps),
                    conditioning=c_,
                    batch_size=4,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=float(guidance_scale),
                    unconditional_conditioning=uc_,
                    eta=0.0,
                    x_T=None,
                )
                decoded = model.decode_first_stage(samples)
                decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
                arr = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                images_out = [arr[i] for i in range(arr.shape[0])]
            return images_out, "推理完成"
        except Exception as e:
            return [None]*4, f"Error: {e}"

    with gr.Blocks(title="MVDream 手动上传4视角图片+4pose") as demo:
        gr.Markdown("## MVDream 手动上传4视角图片和4个pose txt\n图片顺序和pose顺序需一一对应")
        with gr.Row():
            with gr.Column():
                image1 = gr.Image(type="numpy", label="视角1图片")
                pose1 = gr.File(label="视角1 pose.txt", file_types=[".txt"])
                image2 = gr.Image(type="numpy", label="视角2图片")
                pose2 = gr.File(label="视角2 pose.txt", file_types=[".txt"])
                image3 = gr.Image(type="numpy", label="视角3图片")
                pose3 = gr.File(label="视角3 pose.txt", file_types=[".txt"])
                image4 = gr.Image(type="numpy", label="视角4图片")
                pose4 = gr.File(label="视角4 pose.txt", file_types=[".txt"])
                category = gr.Textbox(label="类别描述", placeholder="如 car 或自定义描述")
                steps = gr.Slider(10, 80, value=30, step=1, label="采样步数")
                guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="引导系数")
                seed = gr.Number(value=23, precision=0, label="Seed")
                run_btn = gr.Button("生成4视角图片", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="生成的4视角图片", columns=4, object_fit="contain", height=220)
                status = gr.Textbox(label="状态", interactive=False)
        run_btn.click(
            infer,
            inputs=[image1, pose1, image2, pose2, image3, pose3, image4, pose4, category, steps, guidance_scale, seed],
            outputs=[gallery, status],
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
    launch_sig = inspect.signature(app.launch)
    launch_kwargs = {k: v for k, v in launch_kwargs.items() if k in launch_sig.parameters}
    try:
        app.launch(**launch_kwargs)
    except ValueError as e:
        if "localhost is not accessible" in str(e) and not args.share:
            print("[warn] localhost check failed; retrying with share=True")
            launch_kwargs["share"] = True
            app.launch(**launch_kwargs)
        else:
            raise
    app.launch()
