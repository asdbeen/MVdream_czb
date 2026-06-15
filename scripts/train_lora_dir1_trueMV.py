# Joint 4-view LoRA training for MVDream: send all GT views of each object into diffusion together with num_frames=V.
import os
import argparse
import csv
import sys
import time
import contextlib
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as VT
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.ldm.util import instantiate_from_config
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler


# 後處理：將接近黑色的像素設為白色
def set_white_background(img: np.ndarray, threshold: int = 30) -> np.ndarray:
    img = img.copy()
    mask = np.all(img < threshold, axis=-1)
    img[mask] = [255, 255, 255]
    return img


# 設定隨機種子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dilate_mask(mask, kernel_size=5):
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size, stride=1, padding=pad)


class ConvexHullLoss(torch.nn.Module):
    """
    Penalize predicted content outside the convex-hull mask.

    pred_rgb:     [N, M, 3, H, W] in [0,1]
    render_image: [N, M, 3, H, W] in [0,1] or [-1,1]

    We do NOT compare colors. We only compare soft occupancy masks.
    """
    def __init__(
        self,
        hull_threshold=0.1,     # threshold for GT/render hull mask
        pred_threshold=0.5,     # threshold center for predicted soft mask
        pred_sharpness=20.0,    # larger = closer to hard threshold
        use_dilation=False,
        kernel_size=5,
        eps=1e-6,
    ):
        super().__init__()
        self.hull_threshold = hull_threshold
        self.pred_threshold = pred_threshold
        self.pred_sharpness = pred_sharpness
        self.use_dilation = use_dilation
        self.kernel_size = kernel_size
        self.eps = eps

    def _to_hull_mask(self, render_image):
        if render_image.dim() != 5:
            raise ValueError(f"render_image must be [N, M, C, H, W], got {tuple(render_image.shape)}")
        x = render_image
        # if input is in [-1,1], map to [0,1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        gray = x.mean(dim=2, keepdim=True)                      # [N,M,1,H,W]
        hull_mask = (gray > self.hull_threshold).to(x.dtype)   # hard GT mask
        return hull_mask

    def _to_pred_soft_mask(self, pred_x):
        if pred_x.dim() != 5:
            raise ValueError(f"pred_x must be [N,M,C,H,W], got {tuple(pred_x.shape)}")
        if pred_x.shape[2] == 1:
            pred_mask = pred_x.clamp(0, 1)
        else:
            gray = pred_x.mean(dim=2, keepdim=True)
            pred_mask = torch.sigmoid(self.pred_sharpness * (self.pred_threshold - gray))
        return pred_mask

    def forward(self, pred_rgb, render_image):
        if render_image.shape[2] == 1:
            hull_mask = render_image
        else:
            hull_mask = self._to_hull_mask(render_image)
            
        pred_mask = self._to_pred_soft_mask(pred_rgb)

        if hull_mask.shape[-2:] != pred_mask.shape[-2:]:
            n, m, c, h, w = hull_mask.shape
            target_h, target_w = pred_mask.shape[-2], pred_mask.shape[-1]
            hull_mask = F.interpolate(
                hull_mask.view(n * m, c, h, w),
                size=(target_h, target_w),
                mode='nearest',
            ).view(n, m, c, target_h, target_w)

        if self.use_dilation:
            n, m, _, h, w = hull_mask.shape
            hull_mask = dilate_mask(
                hull_mask.view(n * m, 1, h, w),
                self.kernel_size
            ).view(n, m, 1, h, w)

        outside = pred_mask * (1.0 - hull_mask)
        missing_inside = hull_mask * (1.0 - pred_mask)

        loss_outside = outside.mean()
        loss_inside = missing_inside.mean()

        loss = loss_outside + 0.1 * loss_inside

        return loss.mean(), outside, pred_mask, hull_mask



class HullDataset(Dataset):
    """CSV with columns: hull_path, gt_path, category (int), camera_path (optional)"""
    def __init__(self, csv_path, img_size=256):
        self.rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        hull_img = Image.open(r['hull_path']).convert('RGBA')
        rgb = hull_img.convert('RGB')
        alpha = hull_img.split()[-1]  # alpha channel
        hull_rgb = self.transform(rgb)
        hull_alpha = T.ToTensor()(alpha)  # [1,H,W] in [0,1]
        gt = Image.open(r['gt_path']).convert('RGB')
        gt_t = self.transform(gt)
        category = int(r.get('category', 0))
        sample = {'hull': hull_rgb, 'hull_mask': hull_alpha, 'gt': gt_t, 'category': category}
        # camera optional: store as flat 16 numbers if exists
        if 'camera_path' in r and r['camera_path']:
            try:
                import json
                cam = json.load(open(r['camera_path']))
                sample['camera'] = torch.tensor(cam).float()
            except Exception:
                pass
        return sample


def train(args):
    device = args.device
    use_amp = bool(args.fp16 and isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pred_image_dir = os.path.join(args.out_dir, 'pred_images')

    def _save_pred_image(latent, save_path):
        with torch.no_grad():
            decoded = model.decode_first_stage(latent.detach())
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            image = decoded[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255.0).astype('uint8')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(image).save(save_path)

    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    def _sync_if_cuda():
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _vram_str():
        """Return a short string with used/free VRAM on the current CUDA device."""
        if not (isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available()):
            return ""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return (f"  [VRAM used={used/1024**3:.2f}GB / total={total/1024**3:.2f}GB,"
                f" free={free/1024**3:.2f}GB]")

    print(f"fp16 mixed precision: {'enabled' if use_amp else 'disabled'}")

    print('loading model...')
    if args.config is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt)
    else:
        config = instantiate_from_config(args.config)
        model = config
    model.to(device)
    model.train()

    # support either a simple CSV-based dataset or the customized multi-view dataset
    def custom_collate(batch):
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

    if args.dataset_root is not None and args.meta_path is not None:
        from mvdream.datasets.customized_dataset_dir1 import customizedDataset
        dataset = customizedDataset(
            args.dataset_root,
            args.meta_path,
            sample_side_views=args.num_views,
            source_image_res=args.size,
            use_value_json=False,
        )
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate)
        
        # Validation dataset
        val_dl = None
        if args.val_meta_path is not None:
            val_dataset = customizedDataset(
                args.dataset_root,
                args.val_meta_path,
                sample_side_views=args.num_views,
                source_image_res=args.size,
                use_value_json=False,
            )
            val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate)
            print(f"Validation dataset: {len(val_dataset)} samples")
    else:
        if args.csv is None:
            raise ValueError("--csv is required when --dataset_root/--meta_path are not provided")
        dataset = HullDataset(args.csv, img_size=args.size)
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate)
        val_dl = None

    # inject LoRA adapters and only train LoRA params
    from mvdream.ldm.modules.lora import inject_lora
    # Freeze all pretrained parameters first; LoRA adapters and newly added heads remain trainable.
    for p in model.parameters():
        p.requires_grad = False
    n_replaced = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    # LoRA layers are created at injection time; move again to keep all params on target device.
    model.to(device)
    model.train()
    print(f"Injected LoRA into {n_replaced} modules. Training adapters only.")

    # image encoder for hulls
    from mvdream.ldm.modules.encoders.modules import ImageEmbedder
    image_encoder = ImageEmbedder(device=device, img_size=args.size)
    image_encoder.to(device)
    image_encoder.train()
    for p in image_encoder.backbone.parameters():
        p.requires_grad = False
    for p in image_encoder.proj.parameters():
        p.requires_grad = True
    image_encoder.backbone.eval()

    # cond_proj aligns hull embedding dim (e.g. 768) to text conditioning dim (e.g. 1024)
    # mask_head: 从 latent feature 预测 pred_alpha
    mask_head = None
    # cond_proj aligns hull embedding dim (e.g. 768) to text conditioning dim (e.g. 1024)
    cond_proj = None
    # ref_pose_proj maps reference camera poses (3x4 -> 12) into context tokens.
    ref_pose_proj = None

    # convex-hull loss (参数名修正)
    hull_criterion = ConvexHullLoss(hull_threshold=0.1, use_dilation=True, kernel_size=5).to(device)

    # collect trainable params (LoRA + image encoder projection).
    # mask_head/cond_proj/ref_pose_proj params are added lazily when created.
    trainable = [p for p in model.parameters() if p.requires_grad] + list(image_encoder.proj.parameters())
    # mask_head 参数稍后加入
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # Optional image-saving epochs (1-based). If unset, save on all epochs when enabled.
    target_save_epochs = set(args.save_pred_images_epoch) if args.save_pred_images_epoch is not None else None
    start_epoch = 0
    
    # Create validation log file
    if val_dl is not None:
        val_log_path = os.path.join(args.out_dir, 'validation_log.csv')
        if start_epoch == 0:  # Only write header if starting from scratch
            with open(val_log_path, 'w') as f:
                f.write('epoch,val_loss,val_hull_loss\n')
        print(f"Validation results will be saved to: {val_log_path}")
    
    # Create DDIM sampler for validation
    sampler = DDIMSampler(model)
    
    # Validation function
    def validate(val_loader, model, sampler, image_encoder, cond_proj, ref_pose_proj, mask_head, device, fp16, save_images=False, save_dir=None):
        """Run validation and return average loss"""
        model.eval()
        image_encoder.eval()
        if mask_head is not None:
            mask_head.eval()
        
        total_loss = 0.0
        total_hull_loss = 0.0
        num_batches = 0
        saved_samples = 0
        max_save_samples = 4  # Save 4 validation samples for visualization
        
        with torch.no_grad():
            for batch in val_loader:
                prompts = batch['prompt']
                images_rgba = batch['image'].to(device)
                images_rgb = images_rgba[:, :3, :, :]
                alpha = images_rgba[:, 3:4, :, :]
                hull_mask = batch['convex_hull_mask'].to(device)
                poses = batch['T'].to(device)
                
                # Encode text
                c_text = model.get_learned_conditioning(prompts)
                c_text = c_text.repeat(images_rgb.shape[0], 1, 1)
                
                # Encode image
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    emb_img = image_encoder(images_rgb)
                    if cond_proj is not None:
                        emb_img = cond_proj(emb_img)
                    c_emb = emb_img.unsqueeze(1)
                    
                    # Combine text and image conditioning
                    c = torch.cat([c_text, c_emb], dim=1)
                    
                    # Encode images to latent
                    x_0 = model.encode_first_stage(images_rgb.to(model.dtype))
                    x_0 = model.get_first_stage_encoding(x_0).detach()
                    
                    # Sample noise and timestep
                    noise = torch.randn_like(x_0)
                    t = torch.randint(0, model.num_timesteps, (x_0.shape[0],), device=device).long()
                    x_t = model.q_sample(x_start=x_0, t=t, noise=noise)
                    
                    # Predict noise
                    noise_pred = model.apply_model(x_t, t, c)
                    
                    # Diffusion loss
                    loss_diff = F.mse_loss(noise_pred, noise, reduction='mean')
                    
                    # Hull loss (if mask_head exists)
                    loss_hull = torch.tensor(0.0, device=device)
                    if mask_head is not None:
                        pred_mask = mask_head(x_0)
                        loss_hull = F.binary_cross_entropy(pred_mask, hull_mask)
                    
                    loss = loss_diff + args.lambda_hull * loss_hull
                
                total_loss += loss.item()
                total_hull_loss += loss_hull.item()
                num_batches += 1
                
                # Save validation images every N epochs
                if save_images and saved_samples < max_save_samples and save_dir is not None:
                    # Generate 4-view images for the first sample in batch
                    batch_idx = 0
                    if batch_idx < len(prompts):
                        sample_dir = os.path.join(save_dir, f'sample_{saved_samples:03d}')
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        # Get single sample data
                        single_prompt = prompts[batch_idx]
                        single_rgb = images_rgb[batch_idx:batch_idx+1].repeat(4, 1, 1, 1)  # [4, 3, H, W]
                        single_hull = hull_mask[batch_idx:batch_idx+1].repeat(4, 1, 1, 1)  # [4, 1, H, W]
                        single_pose = poses[batch_idx:batch_idx+1].repeat(4, 1)  # [4, 16]
                        
                        # Save input RGB
                        img_rgb = images_rgb[batch_idx].detach().cpu()
                        img_rgb = (img_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_rgb).save(os.path.join(sample_dir, 'input_rgb.png'))
                        
                        # Save GT hull mask
                        mask_gt = hull_mask[batch_idx].detach().cpu().squeeze().numpy()
                        mask_gt = (mask_gt * 255).astype(np.uint8)
                        Image.fromarray(mask_gt).save(os.path.join(sample_dir, 'gt_hull_mask.png'))
                        
                        # Generate 4-view images
                        try:
                            set_seed(42)  # Fixed seed for validation
                            with torch.no_grad():
                                # Prepare conditioning
                                text_c = model.get_learned_conditioning([single_prompt])
                                c_text = text_c.repeat(4, 1, 1)
                                
                                # Hull embedding
                                emb_img = image_encoder(single_rgb)
                                if cond_proj is not None:
                                    emb_img = cond_proj(emb_img)
                                hull_rep = emb_img.unsqueeze(1)  # [4, 1, dim]
                                
                                # Combine conditioning
                                context_cat = torch.cat([c_text, hull_rep], dim=1)
                                uc_text = model.get_learned_conditioning([""]).repeat(4, 1, 1)
                                uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep)], dim=1)
                                
                                camera = single_pose.to(device)
                                c_ = {
                                    "context": context_cat,
                                    "camera": camera,
                                    "num_frames": 4,
                                }
                                uc_ = {
                                    "context": uc_context_cat,
                                    "camera": camera,
                                    "num_frames": 4,
                                }
                                
                                # Sample
                                shape = [4, args.size // 8, args.size // 8]
                                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                                    samples, _ = sampler.sample(
                                        S=30,  # validation steps
                                        conditioning=c_,
                                        batch_size=4,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=7.5,
                                        unconditional_conditioning=uc_,
                                        eta=0.0,
                                        x_T=None,
                                    )
                                
                                # Decode
                                decoded = model.decode_first_stage(samples)
                                decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
                                arr = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                                
                                # Save 4 views
                                for view_idx in range(4):
                                    img = set_white_background(arr[view_idx])
                                    Image.fromarray(img).save(os.path.join(sample_dir, f'generated_view_{view_idx:03d}.png'))
                                
                                # Save grid
                                grid = np.concatenate([set_white_background(arr[i]) for i in range(4)], axis=1)
                                Image.fromarray(grid).save(os.path.join(sample_dir, 'generated_4views_grid.png'))
                        
                        except Exception as e:
                            print(f"Warning: Failed to generate 4-view images for sample {saved_samples}: {e}")
                        
                        saved_samples += 1
        
        model.train()
        image_encoder.train()
        if mask_head is not None:
            mask_head.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_hull_loss = total_hull_loss / max(num_batches, 1)
        return avg_loss, avg_hull_loss
    if args.resume_ckpt:
        resume = torch.load(args.resume_ckpt, map_location="cpu")
        model_state = resume.get("model_state", resume)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Resumed model adapter from {args.resume_ckpt}; missing={len(missing)}, unexpected={len(unexpected)}")
        if "image_encoder_state" in resume:
            image_encoder.load_state_dict(resume["image_encoder_state"])
            image_encoder.to(device)
            image_encoder.backbone.eval()
        if cond_proj is not None and "cond_proj_state" in resume:
            cond_proj.load_state_dict(resume["cond_proj_state"])
        if ref_pose_proj is not None and "ref_pose_proj_state" in resume:
            ref_pose_proj.load_state_dict(resume["ref_pose_proj_state"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        start_epoch = int(resume.get("epoch", -1)) + 1
    

    # mask_head: 定义在主循环外，输入 latent，输出 [N,1,H,W]
    class MaskHead(torch.nn.Module):
        def __init__(self, latent_dim):
            super().__init__()
            hidden = max(latent_dim // 2, 8)
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(latent_dim, hidden, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(hidden, 1, 1)
            )

        def forward(self, x):
            return torch.sigmoid(self.conv(x))

    mask_head = None

    for epoch in range(start_epoch, args.epochs):
        epoch_desc = f"epoch {epoch + 1}/{args.epochs}"
        epoch_loader = tqdm(dl, total=len(dl), desc=epoch_desc, leave=True) if tqdm is not None else dl
        for it, batch in enumerate(epoch_loader):
            iter_t0 = time.perf_counter()
            current_loss = None
            if args.debug_timing and it == 0:
                print(f"[timing] first batch fetched in {time.perf_counter() - iter_t0:.2f}s{_vram_str()}")
            # detect customized multi-view dataset by presence of render_image_groundtruth
            if 'render_image_groundtruth' in batch:
                # batch elements are collated: batch['render_image'] shape (bs, n_views, C, H, W)
                bs = len(batch['uid']) if 'uid' in batch else batch['render_image_groundtruth'].shape[0]
                total_loss = 0.0
                for bi in range(bs):
                    sample_t0 = time.perf_counter()
                    if args.debug_timing and it == 0:
                        print(f"[timing] start sample bi={bi}{_vram_str()}")
                    uid = batch['uid'][bi] if 'uid' in batch else f"sample_{bi}"
                    # gather per-sample fields
                    hulls = batch['hulls'][bi].to(device)  # (n_views, 3, H, W)
                    hull_masks = batch['hull_masks'][bi].to(device)  # (n_views, 1, H, W)
                    gts = batch['render_image_groundtruth'][bi].to(device)  # (n_views, C, H, W)
                    poses = batch['poses'][bi].to(device)  # (n_views, 3, 4)
                    category = batch['category'][bi]
                    selected_view_ids = batch['selected_view_ids'][bi] if 'selected_view_ids' in batch else None
                    # print ("batch['render_image_groundtruth'][bi].shape:", batch['render_image_groundtruth'][bi].shape)
                    # print ("category:", category)
                    # print ("str(category):", str(category))
                    # print ("poses:", poses)
                    # print ("asd".sad)

                    # encode hulls into embeddings
                    e_list = []
                    for k in range(hulls.shape[0]):
                        img = hulls[k].unsqueeze(0)
                        e = image_encoder.encode(img)
                        e_list.append(e)
                    hull_rep = torch.cat(e_list, dim=1)  # (1, n_views, D)
                    if args.debug_timing and it == 0 and bi == 0:
                        print(f"[timing] image encode: {time.perf_counter() - sample_t0:.2f}s{_vram_str()}")

                    # build text context from category (treat category as text)
                    text_c = model.get_learned_conditioning([str(category)]).to(device)
                    
                   
                    if hull_rep.shape[-1] != text_c.shape[-1]:
                        if cond_proj is None:
                            cond_proj = torch.nn.Linear(hull_rep.shape[-1], text_c.shape[-1]).to(device)
                            optimizer.add_param_group({'params': cond_proj.parameters()})
                        hull_rep = cond_proj(hull_rep)
                    pose_tokens = poses.reshape(1, poses.shape[0], -1)
                    if ref_pose_proj is None:
                        ref_pose_proj = torch.nn.Linear(pose_tokens.shape[-1], text_c.shape[-1]).to(device)
                        optimizer.add_param_group({'params': ref_pose_proj.parameters()})
                    pose_rep = ref_pose_proj(pose_tokens)
                    context_cat = torch.cat([text_c, hull_rep, pose_rep], dim=1)
                  
                    # Joint 4-view supervision.
                    # IMPORTANT:
                    # The old code looped over views and called apply_model(..., num_frames=1).
                    # That trains 4 independent single-view denoising steps.
                    # Here we encode all GT views together and call apply_model once with num_frames=V,
                    # so MVDream can use its cross-view attention.
                    num_views = gts.shape[0]

                    if args.debug_timing and it == 0:
                        if selected_view_ids is not None:
                            view_ids = [int(x.item()) for x in selected_view_ids]
                            print(f"[timing] joint view ids: {view_ids}")
                        else:
                            print(f"[timing] joint views: {num_views}")

                    # gts: [V,3,H,W], where V should normally be 4.
                    gt_imgs = gts

                    with torch.no_grad():
                        enc_posterior = model.encode_first_stage(gt_imgs)
                        z = model.get_first_stage_encoding(enc_posterior)  # [V,C,h,w]

                    # Use the same timestep for all views in one object.
                    # This is the common multi-view diffusion setting: one object, V synchronized noisy views.
                    t_one = torch.randint(0, model.num_timesteps, (1,), device=device).long()
                    t = t_one.repeat(num_views)
                    noise = torch.randn_like(z)
                    x_t = model.q_sample(z, t, noise=noise)

                    # Build camera tensor for all views: [V,16].
                    camera_list = []
                    for j in range(num_views):
                        p = poses[j]
                        if tuple(p.shape) == (3, 4):
                            bottom = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=p.dtype)
                            p4 = torch.cat([p, bottom], dim=0)
                        else:
                            p4 = p
                        camera_list.append(p4.reshape(-1))
                    camera_tensor = torch.stack(camera_list, dim=0).to(device)  # [V,16]

                    # Context contains text + all hull tokens + all pose tokens.
                    # Repeat to [V,T,D] so every view has access to the same global object condition.
                    context_for_model = context_cat.repeat(num_views, 1, 1)

                    with amp_ctx():
                        model_out = model.apply_model(
                            x_t,
                            t,
                            {
                                'context': context_for_model,
                                'camera': camera_tensor,
                                'num_frames': num_views,
                            }
                        )

                        if model.parameterization == 'v':
                            pred_x0 = model.predict_start_from_z_and_v(x_t, t, model_out)
                            pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                        else:
                            pred = model_out
                            pred_x0 = model.predict_start_from_noise(x_t, t, pred)

                        # 1. Joint diffusion noise-prediction loss over all views.
                        mse_loss_val = F.mse_loss(pred, noise)

                        # 2. Hull alpha loss over all views together.
                        latent = pred_x0  # [V,C,h,w]
                        if mask_head is None:
                            latent_dim = latent.shape[1]
                            mask_head = MaskHead(latent_dim).to(device)
                            optimizer.add_param_group({'params': mask_head.parameters()})

                        pred_alpha = mask_head(latent)  # [V,1,h,w]
                        if hull_masks.shape[-2:] != pred_alpha.shape[-2:]:
                            pred_alpha = F.interpolate(
                                pred_alpha,
                                size=hull_masks.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )

                        hull_loss_val, _, _, _ = hull_criterion(
                            pred_alpha.unsqueeze(0),  # [1,V,1,H,W]
                            hull_masks.unsqueeze(0),  # [1,V,1,H,W]
                        )

                        sample_loss = mse_loss_val + args.lambda_hull * hull_loss_val
                        print(
                            f"joint_loss: {sample_loss.item():.6f} "
                            f"(MSE: {mse_loss_val.item():.6f}, Hull: {hull_loss_val.item():.6f}, V: {num_views})"
                        )

                    should_save_pred = args.save_pred_images and (
                        target_save_epochs is None or (epoch + 1) in target_save_epochs
                    )
                    if should_save_pred:
                        for j in range(num_views):
                            if selected_view_ids is not None:
                                view_name = f"{int(selected_view_ids[j].item()):03d}"
                            else:
                                view_name = f"view{j:02d}"
                            save_path = os.path.join(
                                pred_image_dir,
                                f"epoch_{epoch:03d}",
                                f"iter_{it:05d}",
                                f"{uid}_{view_name}_pred.png",
                            )
                            _save_pred_image(pred_x0[j:j + 1], save_path)

                    if args.debug_timing and it == 0 and bi == 0:
                        print(f"[timing] joint 4-view step: {time.perf_counter() - sample_t0:.2f}s{_vram_str()}")

                    if args.debug_timing and it == 0:
                        _sync_if_cuda()
                        bw_t0 = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(sample_loss).backward()
                    else:
                        sample_loss.backward()
                    if args.debug_timing and it == 0:
                        _sync_if_cuda()
                        print(f"[timing] backward: {time.perf_counter() - bw_t0:.2f}s{_vram_str()}")
                        step_t0 = time.perf_counter()
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if args.debug_timing and it == 0:
                        _sync_if_cuda()
                        print(f"[timing] optimizer.step: {time.perf_counter() - step_t0:.2f}s{_vram_str()}")
                    total_loss += sample_loss.item()
                    if args.debug_timing and it == 0:
                        print(f"[timing] sample {bi} total: {time.perf_counter() - sample_t0:.2f}s{_vram_str()}")
                        print(f"[timing] finish sample bi={bi}")

                if it % args.log_steps == 0:
                    avg_loss = total_loss / float(bs)
                    current_loss = avg_loss
                    print(f"epoch {epoch} iter {it} avg_loss {avg_loss:.6f}")

            if current_loss is not None and hasattr(epoch_loader, 'set_postfix'):
                epoch_loader.set_postfix(loss=f"{current_loss:.6f}")

        # Validation
        if val_dl is not None:
            # Save images every 10 epochs
            save_val_images = (epoch + 1) % 10 == 0
            val_save_dir = None
            if save_val_images:
                val_save_dir = os.path.join(args.out_dir, f'validation_epoch_{epoch + 1}')
                os.makedirs(val_save_dir, exist_ok=True)
            
            val_loss, val_hull_loss = validate(val_dl, model, sampler, image_encoder, cond_proj, ref_pose_proj, mask_head, device, args.fp16, 
                                               save_images=save_val_images, save_dir=val_save_dir)
            print(f"Validation - Loss: {val_loss:.6f}, Hull Loss: {val_hull_loss:.6f}")
            
            # Log to CSV
            with open(val_log_path, 'a') as f:
                f.write(f'{epoch + 1},{val_loss:.8f},{val_hull_loss:.8f}\n')
            
            if save_val_images:
                print(f"Validation samples saved to: {val_save_dir}")

        # save checkpoint
        ckpt_path = os.path.join(args.out_dir, f'ckpt_epoch_{epoch}.pth')
        trainable_param_names = {name for name, p in model.named_parameters() if p.requires_grad}
        adapter_state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name in trainable_param_names
        }
        save_dict = {
            'epoch': epoch,
            'model_state': adapter_state,
            'model_state_type': 'adapter_only',
            'optimizer': optimizer.state_dict(),
            'image_encoder_state': image_encoder.state_dict(),
        }
        if mask_head is not None:
            save_dict['mask_head_state'] = mask_head.state_dict()
        if cond_proj is not None:
            save_dict['cond_proj_state'] = cond_proj.state_dict()
        if ref_pose_proj is not None:
            save_dict['ref_pose_proj_state'] = ref_pose_proj.state_dict()
        torch.save(save_dict, ckpt_path)
        print(f"checkpoint saved (adapter-only): {ckpt_path}; model tensors: {len(adapter_state)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='CSV path for single-image fallback dataset mode')
    parser.add_argument('--model_name', type=str, default='sd-v2.1-base-4view')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha scaling')
    parser.add_argument('--dataset_root', type=str, default=None, help='root dir for customized multi-view dataset')
    parser.add_argument('--meta_path', type=str, default=None, help='meta file listing uids for customized dataset')
    parser.add_argument('--val_meta_path', type=str, default=None, help='meta file for validation dataset (optional)')
    parser.add_argument('--num_views', type=int, default=4, help='number of side views to sample for customized dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (set 0 on Windows for stability)')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='enable fp16 mixed precision (use --no-fp16 to disable)')
    parser.add_argument('--debug_timing', action='store_true', help='print detailed timing for first batch')
    parser.add_argument('--save_pred_images', action='store_true', help='save decoded predicted images to out_dir/pred_images during training')
    parser.add_argument('--save_pred_images_epoch', type=int, nargs='+', default=None, help='1-based epoch index list to save predicted images (e.g. --save_pred_images_epoch 100 200); requires --save_pred_images')
    parser.add_argument('--lambda_hull', type=float, default=1, help='weight for convex-hull alpha loss')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='resume training from an adapter checkpoint saved by this script')
    args = parser.parse_args()

    if args.save_pred_images_epoch is not None and any(epoch_idx < 1 for epoch_idx in args.save_pred_images_epoch):
        raise ValueError('--save_pred_images_epoch values must be >= 1 (1-based epoch indices)')

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)