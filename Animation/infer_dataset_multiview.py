import argparse
import contextlib
import json
import os
import random
import shutil
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.camera_utils import create_camera_to_world_matrix, convert_opengl_to_blender
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.ldm.modules.encoders.modules import ImageEmbedder
from mvdream.ldm.modules.lora import inject_lora
from mvdream.model_zoo import build_model

CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]
DEFAULT_INPUT_VIEW_IDS = ["001", "013", "007", "019"]
DEFAULT_OUTPUT_VIEW_IDS = ["001", "013", "007", "019"]

# DEFAULT_INPUT_VIEW_IDS = ["000", "007", "014", "021"]
# DEFAULT_OUTPUT_VIEW_IDS = ["000", "003", "007", "010", "014", "017", "021", "024"]


def set_white_background(img: np.ndarray, color_tolerance: int = 18, corner_patch: int = 12) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    patch = max(1, min(corner_patch, h // 4, w // 4))

    # Estimate the background color from the four corners, which are expected
    # to mostly contain background instead of the foreground object.
    corner_pixels = np.concatenate(
        [
            img[:patch, :patch].reshape(-1, 3),
            img[:patch, w - patch :].reshape(-1, 3),
            img[h - patch :, :patch].reshape(-1, 3),
            img[h - patch :, w - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(corner_pixels, axis=0)

    # Replace pixels close to the estimated background color with pure white.
    color_distance = np.linalg.norm(img.astype(np.float32) - bg_color.astype(np.float32), axis=-1)
    mask = color_distance <= float(color_tolerance)
    img[mask] = [255, 255, 255]
    return img


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_category_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_category_name(category_data: dict) -> str:
    values = []
    for field in CATEGORY_FIELDS:
        value = str(category_data.get(field, "")).strip()
        if value:
            values.append(f"{field}: {value}")
    return ", ".join(values) if values else "object"


def resolve_category_path(sample_dir: str, category_root: Optional[str], sample_id: str) -> str:
    if category_root:
        override_path = os.path.join(category_root, sample_id, "category.json")
        if os.path.exists(override_path):
            return override_path
        raise FileNotFoundError(f"Override category.json not found: {override_path}")
    return os.path.join(sample_dir, "category.json")


def format_sample_ids_in_range(start: int, end: int) -> List[str]:
    if start > end:
        raise ValueError(f"sample_id_start ({start}) cannot be greater than sample_id_end ({end})")
    width = max(4, len(str(end)))
    return [f"{sample_id:0{width}d}" for sample_id in range(start, end + 1)]


def view_id_to_azimuth(view_id: str, total_views: int = 40) -> float:
    return int(view_id) * (360.0 / total_views)


def build_camera_tensor_for_azimuths(
    azimuths: Sequence[float],
    elevation: float,
    device: str,
    blender_coord: bool = True,
) -> torch.Tensor:
    cameras = []
    for azimuth in azimuths:
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, axis=0)).float().to(device)


def load_pose_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pose_data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32).reshape(4, 4)
    return pose_data


def load_target_cameras(
    sample_dir: str,
    view_ids: Sequence[str],
    elevation: float,
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    pose_dir = os.path.join(sample_dir, "pose")
    cameras = []
    resolved_ids: List[str] = []
    missing = []

    for view_id in view_ids:
        pose_path = os.path.join(pose_dir, f"{view_id}.txt")
        if os.path.exists(pose_path):
            camera_matrix = load_pose_matrix(pose_path)
        else:
            image_path = os.path.join(sample_dir, "rgb_convexhull", f"{view_id}.png")
            if not os.path.exists(image_path):
                missing.append(view_id)
                continue
            azimuth = view_id_to_azimuth(view_id)
            camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
        resolved_ids.append(view_id)

    if missing:
        raise FileNotFoundError("Missing target views:\n" + "\n".join(missing))
    if not cameras:
        raise ValueError(f"No valid target cameras found in {sample_dir}")

    return torch.tensor(np.stack(cameras, axis=0)).float().to(device), resolved_ids


def load_reference_images_and_poses(
    sample_dir: str,
    view_ids: Sequence[str],
) -> Tuple[List[np.ndarray], torch.Tensor, List[str]]:
    image_dir = os.path.join(sample_dir, "rgb_convexhull")
    pose_dir = os.path.join(sample_dir, "pose")
    images: List[np.ndarray] = []
    pose_tokens = []
    resolved_ids: List[str] = []
    missing = []

    for view_id in view_ids:
        image_path = os.path.join(image_dir, f"{view_id}.png")
        pose_path = os.path.join(pose_dir, f"{view_id}.txt")
        if not os.path.exists(image_path) or not os.path.exists(pose_path):
            missing.append(f"{view_id}: image={os.path.exists(image_path)}, pose={os.path.exists(pose_path)}")
            continue
        image = Image.open(image_path).convert("RGB")
        pose_matrix = load_pose_matrix(pose_path)
        images.append(np.array(image))
        pose_tokens.append(pose_matrix[:3, :].reshape(-1))
        resolved_ids.append(view_id)

    if missing:
        raise FileNotFoundError("Missing reference inputs:\n" + "\n".join(missing))
    if not images:
        raise ValueError(f"No valid reference images found in {image_dir}")

    return images, torch.tensor(np.stack(pose_tokens, axis=0)).float(), resolved_ids


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
        print("[warn] image_encoder_state not found in checkpoint; hull/style conditioning may be weak.")


def _prepare_hull_embed(
    image_encoder: ImageEmbedder,
    images: Sequence[np.ndarray],
    image_size: int,
    device: str,
    cond_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> torch.Tensor:
    to_tensor = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    embeds = []
    with torch.no_grad():
        for image in images:
            x = to_tensor(image).unsqueeze(0).to(device)
            emb = image_encoder.encode(x)
            embeds.append(emb)

    hull_embed = torch.cat(embeds, dim=1)
    if hull_embed.shape[-1] != target_text_dim:
        if cond_proj is None:
            raise ValueError(
                f"Hull embedding dim {hull_embed.shape[-1]} != text dim {target_text_dim}, "
                "and no cond_proj_state exists in checkpoint."
            )
        hull_embed = cond_proj(hull_embed)
    return hull_embed


def _prepare_reference_pose_tokens(
    pose_tokens: torch.Tensor,
    device: str,
    ref_pose_proj: Optional[torch.nn.Module],
    target_text_dim: int,
) -> Optional[torch.Tensor]:
    if ref_pose_proj is None:
        return None

    pose_rep = ref_pose_proj(pose_tokens.unsqueeze(0).to(device))
    if pose_rep.shape[-1] != target_text_dim:
        raise ValueError(f"Reference pose dim {pose_rep.shape[-1]} != text dim {target_text_dim}.")
    return pose_rep


def sample_multiview(
    model,
    sampler,
    prompt: str,
    negative_prompt: str,
    category_name: str,
    hull_embed: torch.Tensor,
    ref_pose_embed: Optional[torch.Tensor],
    target_cameras: torch.Tensor,
    image_size: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    elevation: float,
    device: str,
    fp16: bool,
    strict_camera: bool,
) -> List[np.ndarray]:
    set_seed(seed)
    dtype = torch.float16 if fp16 and device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype == torch.float16 else contextlib.nullcontext()

    with torch.no_grad(), amp_ctx:
        text_prompt = prompt.strip() if prompt.strip() else category_name
        text_c = model.get_learned_conditioning([text_prompt]).to(device)
        uc_text = model.get_learned_conditioning([negative_prompt]).to(device)
        num_frames = target_cameras.shape[0]
        shape = [4, image_size // 8, image_size // 8]

        if strict_camera:
            images = []
            for frame_idx in range(num_frames):
                camera = target_cameras[frame_idx : frame_idx + 1]
                if ref_pose_embed is not None:
                    context_cat = torch.cat([text_c, hull_embed, ref_pose_embed], dim=1)
                    uc_context_cat = torch.cat(
                        [uc_text, torch.zeros_like(hull_embed), torch.zeros_like(ref_pose_embed)],
                        dim=1,
                    )
                else:
                    context_cat = torch.cat([text_c, hull_embed], dim=1)
                    uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_embed)], dim=1)

                c_ = {"context": context_cat, "camera": camera, "num_frames": 1}
                uc_ = {"context": uc_context_cat, "camera": camera, "num_frames": 1}
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
                images.append(set_white_background(arr[0]))
            return images

        c_text = text_c.repeat(num_frames, 1, 1)
        uc_text_batch = uc_text.repeat(num_frames, 1, 1)
        hull_rep = hull_embed.repeat(num_frames, 1, 1)
        if ref_pose_embed is not None:
            pose_rep = ref_pose_embed.repeat(num_frames, 1, 1)
            context_cat = torch.cat([c_text, hull_rep, pose_rep], dim=1)
            uc_context_cat = torch.cat(
                [uc_text_batch, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)],
                dim=1,
            )
        else:
            context_cat = torch.cat([c_text, hull_rep], dim=1)
            uc_context_cat = torch.cat([uc_text_batch, torch.zeros_like(hull_rep)], dim=1)

        c_ = {"context": context_cat, "camera": target_cameras, "num_frames": num_frames}
        uc_ = {"context": uc_context_cat, "camera": target_cameras, "num_frames": num_frames}
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
        return [set_white_background(arr[i]) for i in range(arr.shape[0])]


def save_outputs(
    output_dir: str,
    sample_dir: str,
    sample_id: str,
    output_view_ids: Sequence[str],
    input_view_ids: Sequence[str],
    images: Sequence[np.ndarray],
    category_name: str,
    prompt: str,
    category_path: str,
) -> None:
    sample_output_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    sample_input_dir = os.path.join(sample_output_dir, "input")
    os.makedirs(sample_input_dir, exist_ok=True)

    for view_id, image in zip(output_view_ids, images):
        Image.fromarray(image).save(os.path.join(sample_output_dir, f"{view_id}.png"))

    grid = np.concatenate(list(images), axis=1)
    Image.fromarray(grid).save(os.path.join(sample_output_dir, "grid.png"))

    # Save the source materials used for this inference run for traceability.
    if os.path.exists(category_path):
        shutil.copy2(category_path, os.path.join(sample_input_dir, "category.json"))

    src_convexhull_dir = os.path.join(sample_dir, "rgb_convexhull")
    dst_convexhull_dir = os.path.join(sample_input_dir, "rgb_convexhull")
    os.makedirs(dst_convexhull_dir, exist_ok=True)
    for view_id in input_view_ids:
        src_image_path = os.path.join(src_convexhull_dir, f"{view_id}.png")
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, os.path.join(dst_convexhull_dir, f"{view_id}.png"))
    src_pose_dir = os.path.join(sample_dir, "pose")
    dst_pose_dir = os.path.join(sample_input_dir, "pose")
    os.makedirs(dst_pose_dir, exist_ok=True)
    for view_id in input_view_ids:
        src_pose_path = os.path.join(src_pose_dir, f"{view_id}.txt")
        if os.path.exists(src_pose_path):
            shutil.copy2(src_pose_path, os.path.join(dst_pose_dir, f"{view_id}.txt"))

    meta = {
        "sample_id": sample_id,
        "input_view_ids": list(input_view_ids),
        "output_view_ids": list(output_view_ids),
        "category_name": category_name,
        "prompt": prompt,
        "category_path": category_path,
    }
    with open(os.path.join(sample_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/chenzebin/MVdream_czb/customized_simple_dataset_tagVersion_simplified/data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/chenzebin/MVdream_czb/Animation/output",
    )
    parser.add_argument(
        "--category_root",
        type=str,
        default=None,
        help="Optional override root containing <sample_id>/category.json, e.g. Animation/randomizecat/replace_1",
    )
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--base_ckpt", type=str, default=None)
    parser.add_argument("--adapter_ckpt", type=str, default="checkpoints/ckpt_epoch_0.pth")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--camera_elevation", type=float, default=15.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--input_view_ids",
        nargs="+",
        default=DEFAULT_INPUT_VIEW_IDS,
        help="Reference input view ids, e.g. 000 007 014 021",
    )
    parser.add_argument(
        "--output_view_ids",
        nargs="+",
        default=DEFAULT_OUTPUT_VIEW_IDS,
        help="Generated output view ids, e.g. 000 003 007 010 014 017 021 024",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="*",
        default=None,
        help="Optional subset of sample folder names under data_root",
    )
    parser.add_argument(
        "--sample_id_start",
        type=int,
        default=None,
        help="Optional inclusive numeric sample id start, e.g. 1",
    )
    parser.add_argument(
        "--sample_id_end",
        type=int,
        default=None,
        help="Optional inclusive numeric sample id end, e.g. 1001",
    )
    parser.add_argument(
        "--strict_camera",
        action="store_true",
        help="Sample one view at a time instead of one round batch inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    print("Loading base model...")
    model = build_model(args.model_name, ckpt_path=args.base_ckpt)
    model.to(args.device)
    model.eval()

    cond_proj, ref_pose_proj = _load_adapter_weights(
        model=model,
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

    if args.sample_ids and (args.sample_id_start is not None or args.sample_id_end is not None):
        raise ValueError("Use either --sample_ids or --sample_id_start/--sample_id_end, not both.")
    if (args.sample_id_start is None) != (args.sample_id_end is None):
        raise ValueError("Both --sample_id_start and --sample_id_end must be provided together.")

    if args.sample_ids:
        sample_ids = sorted(args.sample_ids)
    elif args.sample_id_start is not None and args.sample_id_end is not None:
        sample_ids = format_sample_ids_in_range(args.sample_id_start, args.sample_id_end)
    else:
        sample_ids = sorted(
            name for name in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, name))
        )
    print(f"Found {len(sample_ids)} samples.")

    success = 0
    failed = 0
    for sample_id in sample_ids:
        sample_dir = os.path.join(args.data_root, sample_id)
        category_path = resolve_category_path(sample_dir, args.category_root, sample_id)
        if not os.path.exists(category_path):
            print(f"[skip] {sample_id}: category.json not found")
            failed += 1
            continue

        try:
            category_data = load_category_json(category_path)
            category_name = build_category_name(category_data)
            ref_images, ref_pose_tokens, resolved_input_view_ids = load_reference_images_and_poses(
                sample_dir,
                args.input_view_ids,
            )
            target_cameras, resolved_output_view_ids = load_target_cameras(
                sample_dir=sample_dir,
                view_ids=args.output_view_ids,
                elevation=args.camera_elevation,
                device=args.device,
            )
            text_dim = model.get_learned_conditioning([category_name]).shape[-1]
            hull_embed = _prepare_hull_embed(
                image_encoder=image_encoder,
                images=ref_images,
                image_size=args.size,
                device=args.device,
                cond_proj=cond_proj,
                target_text_dim=text_dim,
            )
            ref_pose_embed = _prepare_reference_pose_tokens(
                pose_tokens=ref_pose_tokens,
                device=args.device,
                ref_pose_proj=ref_pose_proj,
                target_text_dim=text_dim,
            )
            images = sample_multiview(
                model=model,
                sampler=sampler,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                category_name=category_name,
                hull_embed=hull_embed,
                ref_pose_embed=ref_pose_embed,
                target_cameras=target_cameras,
                image_size=args.size,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                elevation=args.camera_elevation,
                device=args.device,
                fp16=args.fp16,
                strict_camera=args.strict_camera,
            )
            save_outputs(
                output_dir=args.output_root,
                sample_dir=sample_dir,
                sample_id=sample_id,
                output_view_ids=resolved_output_view_ids,
                input_view_ids=resolved_input_view_ids,
                images=images,
                category_name=category_name,
                prompt=args.prompt if args.prompt.strip() else category_name,
                category_path=category_path,
            )
            print(f"[ok] {sample_id}: saved {len(images)} views")
            success += 1
        except Exception as exc:
            print(f"[failed] {sample_id}: {exc}")
            failed += 1

    print(f"Finished. success={success}, failed={failed}, output_root={args.output_root}")


if __name__ == "__main__":
    main()
