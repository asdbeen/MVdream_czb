import os
import random
from typing import List
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]


def opposite_view(i):
    if 0 <= i <= 24:
        return (i + 12) % 24
    elif 27 <= i <= 39:
        return ((i - 27) + 6) % 12 + 27
    else:
        raise ValueError("Input number must be between 0-24 or 27-39.")


def get_random_views(rgba_dir, num_views=4):
    all_files = [f for f in os.listdir(rgba_dir) if f.endswith(".png")]
    view_numbers = sorted([int(os.path.splitext(f)[0]) for f in all_files])
    if len(view_numbers) < num_views:
        return np.array(view_numbers)
    selected_views = random.sample(view_numbers, num_views)
    return np.array(selected_views)


def get_4_cardinal_views(rgba_dir, front_view=None):
    """Return [front, back, left, right] sampled within the same ring.

    Ring-1: 0..23  (24 is duplicate of 0)
    Ring-2: 27..38 (39 is duplicate of 27)
    """
    all_files = [f for f in os.listdir(rgba_dir) if f.endswith(".png")]
    view_numbers = sorted([int(os.path.splitext(f)[0]) for f in all_files])

    # remove duplicated closing views
    ring1 = [v for v in view_numbers if 0 <= v <= 23]
    ring2 = [v for v in view_numbers if 27 <= v <= 38]

    if len(ring1) == 0 and len(ring2) == 0:
        return np.array([])

    ring = None

    if front_view is not None:
        fv = int(front_view)

        # map duplicated closing views back to their equivalent start views
        if fv == 24:
            fv = 0
        elif fv == 39:
            fv = 27

        if fv in ring1:
            ring = ring1
        elif fv in ring2:
            ring = ring2

        front_view = fv

    if ring is None:
        candidates = []
        if len(ring1) >= 4:
            candidates.append(ring1)
        if len(ring2) >= 4:
            candidates.append(ring2)

        if len(candidates) == 0:
            return get_random_views(rgba_dir, num_views=4)

        ring = random.choice(candidates)

    N = len(ring)

    if N < 4:
        return get_random_views(rgba_dir, num_views=4)

    if front_view is None:
        front = random.choice(ring)
    else:
        front = int(front_view)
        if front not in ring:
            front = random.choice(ring)

    idx = ring.index(front)

    half = N // 2
    quarter = N // 4

    back_idx = (idx + half) % N
    left_idx = (idx + quarter) % N
    right_idx = (idx - quarter) % N

    picks = [
        ring[idx],
        ring[back_idx],
        ring[left_idx],
        ring[right_idx],
    ]

    return np.array(picks)

class customizedDataset(Dataset):
    """Lightweight multi-view dataset loader compatible with the example structure.

    Expects each sample under `root_dir/uid/` containing folders:
    - pose/
    - rgb_convexhull/
    - rgb_groundtruth/
    - category.json
    - value.json
    """

    def __init__(
        self,
        root_dir: str,
        meta_path: str,
        sample_side_views: int = 4,
        source_image_res: int = 256,
        render_image_res_low: int = 256,
        render_image_res_high: int = 256,
        render_region_size: int = 256,
        normalize_camera: bool = False,
        normed_dist_to_center=None,
        use_value_json: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        with open(meta_path, "r") as f:
            self.uids = [l.strip() for l in f.readlines() if l.strip()]

        self.sample_side_views = sample_side_views
        self.source_image_res = source_image_res
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.use_value_json = use_value_json

    def __len__(self):
        return len(self.uids)

    def _load_rgba_with_alpha(self, path: str):
        """
        Returns:
            rgb_white_bg: [3,H,W] float in [0,1], alpha composited on white
            alpha:        [1,H,W] float in [0,1]
        """
        im = Image.open(path).convert("RGBA")
        arr = np.array(im).astype(np.float32) / 255.0  # [H,W,4]
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]

        # Composite on white background so the visible shape stays bright and background stays white.
        rgb_white_bg = rgb * alpha + (1.0 - alpha) * 1.0  # white background

        rgb_t = torch.from_numpy(rgb_white_bg.transpose(2, 0, 1)).float()  # [3,H,W]
        alpha_t = torch.from_numpy(alpha.transpose(2, 0, 1)).float()       # [1,H,W]
        return rgb_t, alpha_t

    def __getitem__(self, idx):
        uid = self.uids[idx]
        root = os.path.join(self.root_dir, uid)

        pose_dir = os.path.join(root, "pose")
        hull_source_dir = os.environ.get("MVDREAM_HULL_SOURCE_DIR", "rgb_convexhull").strip() or "rgb_convexhull"
        convex_dir = os.path.join(root, hull_source_dir)
        if not os.path.isdir(convex_dir):
            convex_dir = os.path.join(root, "rgb_convexhull")
        gt_dir = os.path.join(root, "rgb_groundtruth")

        category_path = os.path.join(root, "category.json")
        try:
            with open(category_path, "r", encoding="utf-8-sig") as f:
                cat = json.load(f)
                if isinstance(cat, dict):
                    parts = []
                    for key in CATEGORY_FIELDS:
                        value = str(cat.get(key, "")).strip()
                        if value:
                            parts.append(f"{key}: {value}")
                    category_text = ", ".join(parts) if parts else ""
                else:
                    category_text = str(cat)
        except Exception:
            category_text = ""

        if self.use_value_json:
            value_path = os.path.join(root, "value.json")
            try:
                with open(value_path, "r") as f:
                    val = json.load(f)
                value_tensor = torch.tensor(
                    [
                        float(val.get("linearity", 0.0)),
                        float(val.get("planarity", 0.0)),
                        float(val.get("sphericity", 0.0)),
                    ],
                    dtype=torch.float32,
                )
            except Exception:
                value_tensor = torch.zeros(3, dtype=torch.float32)
        else:
            value_tensor = torch.zeros(3, dtype=torch.float32)

        view_source_dir = convex_dir if os.path.isdir(convex_dir) else gt_dir
        sample_views = get_4_cardinal_views(view_source_dir)

        hulls = []
        hull_masks = []
        gts = []
        poses = []

        for v in sample_views:
            vstr = f"{int(v):03d}.png"
            hull_path = os.path.join(convex_dir, vstr)
            gt_path = os.path.join(gt_dir, vstr)

            try:
                hull_rgb, hull_alpha = self._load_rgba_with_alpha(hull_path)
            except Exception:
                hull_rgb = torch.zeros(3, self.source_image_res, self.source_image_res)
                hull_alpha = torch.zeros(1, self.source_image_res, self.source_image_res)

            try:
                gt_rgb, _ = self._load_rgba_with_alpha(gt_path)
            except Exception:
                gt_rgb = torch.zeros(3, self.source_image_res, self.source_image_res)

            hull_rgb = T.functional.resize(hull_rgb, [self.source_image_res, self.source_image_res])
            hull_alpha = T.functional.resize(hull_alpha, [self.source_image_res, self.source_image_res])
            gt_rgb = T.functional.resize(gt_rgb, [self.source_image_res, self.source_image_res])

            hulls.append(hull_rgb.unsqueeze(0))           # [1,3,H,W]
            hull_masks.append(hull_alpha.unsqueeze(0))    # [1,1,H,W]
            gts.append(gt_rgb.unsqueeze(0))               # [1,3,H,W]

            pose_path = os.path.join(pose_dir, f"{int(v):03d}.txt")
            if os.path.exists(pose_path):
                try:
                    with open(pose_path, "r") as f:
                        lines = f.readlines()
                    pose_data = np.array(
                        [list(map(float, line.split())) for line in lines],
                        dtype=np.float32,
                    ).reshape(4, 4)
                    poses.append(torch.from_numpy(pose_data)[:3, :])
                except Exception:
                    poses.append(torch.eye(3, 4))
            else:
                poses.append(torch.eye(3, 4))

        hulls = torch.cat(hulls, dim=0)              # [V,3,H,W]
        hull_masks = torch.cat(hull_masks, dim=0)    # [V,1,H,W]
        gts = torch.cat(gts, dim=0)                  # [V,3,H,W]
        poses = torch.stack(poses, dim=0)            # [V,3,4]

        sample = {
            "uid": uid,
            "poses": poses,
            "category": category_text,
            "value_tensor": value_tensor,
            "selected_view_ids": torch.tensor(sample_views, dtype=torch.long),

            "source_image_front": hulls[0],
            "source_image_back": hulls[1] if hulls.shape[0] > 1 else hulls[0],
            "source_image_left": hulls[2] if hulls.shape[0] > 2 else hulls[0],
            "source_image_right": hulls[3] if hulls.shape[0] > 3 else hulls[0],

            "source_mask": hull_masks[0],
            "source_mask_back": hull_masks[1] if hull_masks.shape[0] > 1 else hull_masks[0],
            "source_mask_left": hull_masks[2] if hull_masks.shape[0] > 2 else hull_masks[0],
            "source_mask_right": hull_masks[3] if hull_masks.shape[0] > 3 else hull_masks[0],

            "hulls": hulls,                   # [V,3,H,W]
            "hull_masks": hull_masks,               # [V,1,H,W]

            # "source_image_groundtruth": gts[0],
            "render_image_groundtruth": gts,
        }
        return sample