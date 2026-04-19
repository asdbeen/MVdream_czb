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
    all_files = [f for f in os.listdir(rgba_dir) if f.endswith('.png')]
    view_numbers = sorted([int(os.path.splitext(f)[0]) for f in all_files])
    if len(view_numbers) < num_views:
        return np.array(view_numbers)
    selected_views = random.sample(view_numbers, num_views)
    return np.array(selected_views)


def get_4_cardinal_views(rgba_dir, front_view=None):
    """Return [front, back, left, right] sampled within the same ring.

    Ring-1: 0..24, Ring-2: 27..39.
    """
    all_files = [f for f in os.listdir(rgba_dir) if f.endswith('.png')]
    view_numbers = sorted([int(os.path.splitext(f)[0]) for f in all_files])
    ring1 = [v for v in view_numbers if 0 <= v <= 24]
    ring2 = [v for v in view_numbers if 27 <= v <= 39]

    if len(ring1) == 0 and len(ring2) == 0:
        return np.array([])

    # choose ring by front_view when provided; otherwise randomly pick one available ring
    ring = None
    if front_view is not None:
        fv = int(front_view)
        if fv in ring1:
            ring = ring1
        elif fv in ring2:
            ring = ring2
    if ring is None:
        candidates = []
        if len(ring1) > 0:
            candidates.append(ring1)
        if len(ring2) > 0:
            candidates.append(ring2)
        ring = random.choice(candidates)

    N = len(ring)
    if N < 4:
        # not enough views in this ring, fallback to random views from all available files
        return get_random_views(rgba_dir, num_views=4)

    # pick front
    if front_view is None:
        front = random.choice(ring)
    else:
        front = int(front_view)
        if front not in ring:
            # fallback to random
            front = random.choice(ring)
    idx = ring.index(front)

    # compute cardinal positions
    half = N // 2
    quarter = max(1, N // 4)
    back_idx = (idx + half) % N
    left_idx = (idx + quarter) % N
    right_idx = (idx - quarter) % N
    picks = [ring[idx], ring[back_idx], ring[left_idx], ring[right_idx]]
    return np.array(picks)


class customizedDataset(Dataset):
    """Lightweight multi-view dataset loader compatible with the example structure.

    Expects each sample under `root_dir/uid/` containing folders: `pose/`,
    `rgb_convexhull/`, `rgb_groundtruth/`, and files `category.json`, `value.json`.
    A simple meta file lists all uids (one per line).
    """
    def __init__(self, root_dir: str, meta_path: str, sample_side_views: int = 4, source_image_res: int = 256,
                 render_image_res_low: int = 256, render_image_res_high: int = 256, render_region_size: int = 256,
                 normalize_camera: bool = False, normed_dist_to_center=None, use_value_json: bool = True):
        super().__init__()
        self.root_dir = root_dir
        with open(meta_path, 'r') as f:
            self.uids = [l.strip() for l in f.readlines() if l.strip()]
        self.sample_side_views = sample_side_views
        self.source_image_res = source_image_res
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.use_value_json = use_value_json
        self.transform = T.Compose([T.Resize((self.source_image_res, self.source_image_res)), T.ToTensor()])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        root = os.path.join(self.root_dir, uid)
        pose_dir = os.path.join(root, 'pose')
        convex_dir = os.path.join(root, 'rgb_convexhull')
        gt_dir = os.path.join(root, 'rgb_groundtruth')

        # load category
        category_path = os.path.join(root, 'category.json')
        category_text = ''
        try:
            with open(category_path, 'r') as f:
                cat = json.load(f)
                if isinstance(cat, dict):
                    parts = []
                    for key in CATEGORY_FIELDS:
                        value = str(cat.get(key, '')).strip()
                        if value:
                            parts.append(f"{key}: {value}")
                    category_text = ", ".join(parts) if parts else ''
                else:
                    category_text = str(cat)
        except Exception:
            category_text = ''

        # value
        if self.use_value_json:
            value_path = os.path.join(root, 'value.json')
            try:
                with open(value_path, 'r') as f:
                    val = json.load(f)
                    value_tensor = torch.tensor([float(val.get('linearity', 0.0)), float(val.get('planarity', 0.0)), float(val.get('sphericity', 0.0))], dtype=torch.float32)
            except Exception:
                value_tensor = torch.zeros(3, dtype=torch.float32)
        else:
            value_tensor = torch.zeros(3, dtype=torch.float32)

        # fixed 4-view sampling: [front, back, left, right] from the same ring
        sample_views = get_4_cardinal_views(convex_dir)

        hulls = []
        gts = []
        poses = []

        for v in sample_views:
            vstr = f"{int(v):03d}.png"
            hull_path = os.path.join(convex_dir, vstr)
            gt_path = os.path.join(gt_dir, vstr)
            # load image, allow RGBA
            def load_rgba(p):
                im = Image.open(p).convert('RGBA')
                arr = np.array(im).astype(np.float32)/255.0
                alpha = arr[...,3:4]
                rgb = arr[...,:3]*alpha + (1-alpha)
                return torch.from_numpy(rgb.transpose(2,0,1)).float()

            try:
                hull = load_rgba(hull_path)
            except Exception:
                hull = torch.zeros(3, self.source_image_res, self.source_image_res)
            try:
                gt = load_rgba(gt_path)
            except Exception:
                gt = torch.zeros(3, self.source_image_res, self.source_image_res)

            hull = T.functional.resize(hull, [self.source_image_res, self.source_image_res])
            gt = T.functional.resize(gt, [self.source_image_res, self.source_image_res])
            hulls.append(hull.unsqueeze(0))
            gts.append(gt.unsqueeze(0))

            pose_path = os.path.join(pose_dir, f"{int(v):03d}.txt")
            if os.path.exists(pose_path):
                try:
                    with open(pose_path, 'r') as f:
                        lines = f.readlines()
                    pose_data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32).reshape(4,4)
                    poses.append(torch.from_numpy(pose_data)[:3,:])
                except Exception:
                    poses.append(torch.eye(3,4))
            else:
                poses.append(torch.eye(3,4))

        hulls = torch.cat(hulls, dim=0)
        gts = torch.cat(gts, dim=0)
        poses = torch.stack(poses, dim=0)

        sample = {
            'uid': uid,
            'poses': poses,
            'category': category_text,
            'value_tensor': value_tensor,
            'selected_view_ids': torch.tensor(sample_views, dtype=torch.long),
            'source_image': hulls[0],
            'source_image_back': hulls[1] if hulls.shape[0]>1 else hulls[0],
            'source_image_left': hulls[2] if hulls.shape[0]>2 else hulls[0],
            'source_image_right': hulls[3] if hulls.shape[0]>3 else hulls[0],
            'render_image': hulls,
            'source_image_groundtruth': gts[0],
            'render_image_groundtruth': gts,
        }
        return sample
