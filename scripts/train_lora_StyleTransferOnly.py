###### 这个是最简单的实验。 测试是否能把数据集风格迁移到MVDream而已
###### checkpoint 输出在 StyleTransferOnly
###### 使用 gradio_app_OG_lora.py


import argparse
import contextlib
import glob
import os
import random
import sys
import threading
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.model_zoo import build_model
from mvdream.ldm.modules.lora import inject_lora


def _default_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _validate_dataset_layout(dataset_root: str, meta_path: str) -> None:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta_path not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"meta_path has no sample ids: {meta_path}")

    sample_uid = lines[0]
    sample_root = os.path.join(dataset_root, sample_uid)

    required_dirs = ["pose", "rgb_groundtruth"]
    missing = [d for d in required_dirs if not os.path.isdir(os.path.join(sample_root, d))]
    if missing:
        raise FileNotFoundError(
            "dataset sample is missing required folders: "
            f"sample={sample_uid}, missing={missing}, sample_root={sample_root}"
        )

    category_path = os.path.join(sample_root, "category.json")
    if not os.path.isfile(category_path):
        raise FileNotFoundError(f"category.json not found for first sample: {category_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        out = {}
        for key in elem:
            if isinstance(elem[key], str):
                out[key] = [d[key] for d in batch]
            else:
                out[key] = torch.utils.data.default_collate([d[key] for d in batch])
        return out
    return torch.utils.data.default_collate(batch)


def _build_camera_tensor_from_poses(poses: torch.Tensor, device: str) -> torch.Tensor:
    # poses: [N, 3, 4] -> [N, 16]
    n = poses.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=poses.dtype).view(1, 1, 4)
    poses4 = torch.cat([poses.to(device), bottom.repeat(n, 1, 1)], dim=1)
    return poses4.reshape(n, 16)


def _start_gpu_usage_logger(device: str, interval_sec: float):
    """Start a background logger that prints GPU memory usage periodically."""
    if not (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()):
        print("[gpu] CUDA not available or device is not CUDA; GPU usage logger disabled.")
        return None, None

    interval_sec = max(1.0, float(interval_sec))
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(interval_sec):
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(
                    "[gpu] "
                    f"used={used / 1024**3:.2f}GB, "
                    f"free={free / 1024**3:.2f}GB, "
                    f"total={total / 1024**3:.2f}GB, "
                    f"allocated={allocated / 1024**3:.2f}GB, "
                    f"reserved={reserved / 1024**3:.2f}GB"
                )
            except Exception as e:
                print(f"[gpu] failed to query GPU usage: {e}")

    thread = threading.Thread(target=_worker, name="gpu-usage-logger", daemon=True)
    thread.start()
    print(f"[gpu] usage logger started, interval={interval_sec:.1f}s")
    return stop_event, thread


def train_text_only(args) -> None:
    device = args.device
    use_amp = bool(args.fp16 and isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    set_seed(int(args.seed))

    print(f"fp16 mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(
        f"mixed timestep sampling: high_t_prob={args.high_t_prob}, "
        f"min_t_ratio={args.min_t_ratio}, use_camera_condition={args.use_camera_condition}"
    )

    print("loading model...")
    model = build_model(args.model_name, ckpt_path=args.ckpt)
    model.to(device)
    model.train()

    from mvdream.datasets.customized_dataset_dir1 import customizedDataset

    dataset = customizedDataset(
        args.dataset_root,
        args.meta_path,
        sample_side_views=args.num_views,
        source_image_res=args.size,
        use_value_json=False,
    )
    dl = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=_custom_collate,
    )

    # Freeze base model and train only injected LoRA parameters.
    for p in model.parameters():
        p.requires_grad = False
    n_replaced = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)
    model.train()
    print(f"Injected LoRA into {n_replaced} modules. Training text-only adapters.")

    trainable_param_names = {name for name, p in model.named_parameters() if p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    start_epoch = 0
    if args.resume_ckpt:
        resume = torch.load(args.resume_ckpt, map_location="cpu")
        model_state = resume.get("model_state", resume)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Resumed model adapter from {args.resume_ckpt}; missing={len(missing)}, unexpected={len(unexpected)}")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        start_epoch = int(resume.get("epoch", -1)) + 1
        print(f"Resume training from epoch {start_epoch} / target epochs {args.epochs}")

    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    for epoch in range(start_epoch, args.epochs):
        epoch_desc = f"epoch {epoch + 1}/{args.epochs}"
        epoch_loader = tqdm(dl, total=len(dl), desc=epoch_desc, leave=True) if tqdm is not None else dl

        for it, batch in enumerate(epoch_loader):
            gts = batch["render_image_groundtruth"].to(device)  # [B, V, 3, H, W]
            poses = batch["poses"].to(device)                  # [B, V, 3, 4]
            categories = batch["category"]                     # [B]

            bsz = gts.shape[0]
            num_views = gts.shape[1]

            gts = gts.view(bsz * num_views, gts.shape[2], gts.shape[3], gts.shape[4])
            poses = poses.view(bsz * num_views, 3, 4)
            categories = [str(c) for c in categories for _ in range(num_views)]

            text_c = model.get_learned_conditioning(categories).to(device)

            with torch.no_grad():
                enc_posterior = model.encode_first_stage(gts)
                z = model.get_first_stage_encoding(enc_posterior)

            high_t = torch.rand((), device=device) < float(args.high_t_prob)
            if high_t:
                t_low = int(model.num_timesteps * float(args.min_t_ratio))
                t_low = max(0, min(t_low, model.num_timesteps - 1))
            else:
                t_low = 0

            t_single = torch.randint(t_low, model.num_timesteps, (1,), device=device).long()
            t = t_single.repeat(bsz * num_views)
            noise = torch.randn_like(z)
            x_t = model.q_sample(z, t, noise=noise)

            cond: Dict[str, torch.Tensor] = {
                "context": text_c,
            }
            if args.use_camera_condition:
                cond["camera"] = _build_camera_tensor_from_poses(poses, device)
                cond["num_frames"] = num_views

            with amp_ctx():
                model_out = model.apply_model(x_t, t, cond)
                if model.parameterization == "v":
                    pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                else:
                    pred = model_out
                loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if it % args.log_steps == 0:
                print(f"epoch {epoch} iter {it} batch_loss {loss.item():.6f}")

            if hasattr(epoch_loader, "set_postfix"):
                epoch_loader.set_postfix(loss=f"{loss.item():.6f}")

        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pth")
        adapter_state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name in trainable_param_names
        }
        save_dict = {
            "epoch": epoch,
            "model_state": adapter_state,
            "model_state_type": "adapter_only",
            "optimizer": optimizer.state_dict(),
            "train_mode": "text_only",
            "use_camera_condition": bool(args.use_camera_condition),
        }
        torch.save(save_dict, ckpt_path)
        print(f"checkpoint saved (adapter-only): {ckpt_path}; model tensors: {len(adapter_state)}")

        if args.keep_latest_ckpt_only:
            ckpt_files = sorted(glob.glob(os.path.join(args.out_dir, "ckpt_epoch_*.pth")))
            for old_file in ckpt_files[:-1]:
                try:
                    os.remove(old_file)
                    print(f"Deleted old checkpoint: {old_file}")
                except Exception as e:
                    print(f"Failed to delete {old_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MVDream text-only LoRA (text + camera condition, GT as supervision only)."
    )

    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional base checkpoint path")

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=_default_path("customized_simple_dataset_tagVersion_simplified"),
        help="Dataset root. Meta entries are joined onto this path.",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=_default_path("customized_simple_dataset_tagVersion_simplified", "train.txt"),
        help="Meta file listing sample paths such as data/0001.",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=_default_path("checkpoints", "lora_mvdream_custom"))
    parser.add_argument("--log_steps", type=int, default=20)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fp16 mixed precision (use --no-fp16 to disable).",
    )
    parser.add_argument("--debug_timing", action="store_true")

    parser.add_argument("--save_pred_images", action="store_true")
    parser.add_argument("--save_pred_images_epoch", type=int, nargs="+", default=None)

    parser.add_argument("--high_t_prob", type=float, default=0.7)
    parser.add_argument("--min_t_ratio", type=float, default=0.7)
    parser.add_argument("--keep_latest_ckpt_only", action="store_true")
    parser.add_argument("--gpu_log_interval", type=float, default=10.0, help="Print GPU usage every N seconds (CUDA only)")
    parser.add_argument("--use_camera_condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume training from an adapter checkpoint saved by this script")

    args = parser.parse_args()

    if args.save_pred_images_epoch is not None and any(e < 1 for e in args.save_pred_images_epoch):
        raise ValueError("--save_pred_images_epoch values must be >= 1")
    if not 0.0 <= args.high_t_prob <= 1.0:
        raise ValueError("--high_t_prob must be in [0, 1]")
    if not 0.0 <= args.min_t_ratio < 1.0:
        raise ValueError("--min_t_ratio must be in [0, 1)")

    _validate_dataset_layout(args.dataset_root, args.meta_path)

    os.makedirs(args.out_dir, exist_ok=True)

    stop_event, monitor_thread = _start_gpu_usage_logger(args.device, args.gpu_log_interval)
    try:
        train_text_only(args)
    finally:
        if stop_event is not None:
            stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
