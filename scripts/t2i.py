import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from PIL import Image
import torchvision.transforms as T
from mvdream.ldm.modules.encoders.modules import ImageEmbedder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1, hull_embed=None, category_tensor=None):
    if type(prompt)!=list:
        prompt = [prompt]
        # text/context: if prompt empty and category provided, use category as text
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        if prompt is None or len(prompt) == 0:
            text_prompts = [str(category_tensor.item())] if category_tensor is not None and category_tensor.numel()==1 else [""]
        else:
            text_prompts = prompt

        text_c = model.get_learned_conditioning(text_prompts).to(device)
        # expand/prepare context and unconditional context
        c_text = text_c.repeat(batch_size,1,1)
        uc_text = model.get_learned_conditioning([""]).to(device).repeat(batch_size,1,1)

        # hull embedding: provided as (1,1,dim) or (batch,1,dim)
        if hull_embed is not None:
            if hull_embed.shape[0] == 1:
                hull_rep = hull_embed.repeat(batch_size,1,1)
            else:
                hull_rep = hull_embed
        else:
            hull_rep = torch.zeros((batch_size,1,c_text.shape[2]), device=device)

        # concatenate along sequence dim
        context_cat = torch.cat([c_text, hull_rep], dim=1)
        uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep)], dim=1)

        c_ = {"context": context_cat}
        uc_ = {"context": uc_context_cat}
        if category_tensor is not None:
            # ensure long tensor for class-conditional y
            y = category_tensor.long()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            if y.shape[0] == 1:
                y_rep = y.repeat(batch_size)
            else:
                y_rep = y
            c_["y"] = y_rep
            uc_["y"] = torch.zeros_like(y_rep)

        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--hull_path", type=str, default=None, help="path to hull (convexhull) image")
    parser.add_argument("--category", type=int, default=None, help="category id for conditioning")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("load t2i model done . ")

    # prepare hull encoder and load hull if provided
    image_encoder = ImageEmbedder(device=device, img_size=args.size)
    hull_embed = None
    category_tensor = None
    if args.hull_path is not None:
        img = Image.open(args.hull_path).convert('RGB')
        transform = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
        img_t = transform(img).unsqueeze(0).to(device)
        hull_embed = image_encoder.encode(img_t)
    if args.category is not None:
        # single category id repeated
        category_tensor = torch.tensor([args.category], device=device)

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames, elevation=args.camera_elev, 
                azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        camera = camera.repeat(batch_size//args.num_frames,1).to(device)
    else:
        camera = None
    
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    for j in range(3):
        img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0, 
            dtype=dtype, device=device, camera=camera, num_frames=args.num_frames, hull_embed=hull_embed, category_tensor=category_tensor)
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"sample.png")