import argparse
import os
import sys
import threading
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_hull_finetune_test_new_mask_enhanced import train


def _default_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _validate_dataset_layout(dataset_root: str, meta_path: str, hull_source_dir: str) -> None:
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

    candidate_hull_dir = os.path.join(sample_root, hull_source_dir)
    if not os.path.isdir(candidate_hull_dir):
        print(
            f"[warn] hull_source_dir '{hull_source_dir}' not found for {sample_uid}; "
            "dataset loader will fallback to rgb_convexhull when available."
        )

    category_path = os.path.join(sample_root, "category.json")
    if not os.path.isfile(category_path):
        raise FileNotFoundError(f"category.json not found for first sample: {category_path}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MVDream LoRA using customized multi-view data (pose + views + GT + category)."
    )

    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--config", type=str, default=None)
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
    parser.add_argument(
        "--hull_source_dir",
        type=str,
        default="rgb_groundtruth",
        help="Folder used as hull/source image input. Use rgb_groundtruth to avoid convexhull.",
    )

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2)
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

    parser.add_argument("--lambda_hull", type=float, default=0.0)
    parser.add_argument("--mask_token_grid", type=int, default=4)
    parser.add_argument("--mask_loss_inside_weight", type=float, default=2.0)
    parser.add_argument("--mask_loss_outside_weight", type=float, default=0.5)

    parser.add_argument("--high_t_prob", type=float, default=0.7)
    parser.add_argument("--min_t_ratio", type=float, default=0.7    )
    parser.add_argument("--keep_latest_ckpt_only", action="store_true")
    parser.add_argument("--gpu_log_interval", type=float, default=10.0, help="Print GPU usage every N seconds (CUDA only)")

    args = parser.parse_args()

    if args.save_pred_images_epoch is not None and any(e < 1 for e in args.save_pred_images_epoch):
        raise ValueError("--save_pred_images_epoch values must be >= 1")
    if not 0.0 <= args.high_t_prob <= 1.0:
        raise ValueError("--high_t_prob must be in [0, 1]")
    if not 0.0 <= args.min_t_ratio < 1.0:
        raise ValueError("--min_t_ratio must be in [0, 1)")

    _validate_dataset_layout(args.dataset_root, args.meta_path, args.hull_source_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # Let dataset loader choose the source image directory.
    os.environ["MVDREAM_HULL_SOURCE_DIR"] = args.hull_source_dir

    # The reused trainer expects this field to exist even in customized dataset mode.
    args.csv = None

    stop_event, monitor_thread = _start_gpu_usage_logger(args.device, args.gpu_log_interval)
    try:
        train(args)
    finally:
        if stop_event is not None:
            stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
