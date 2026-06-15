
# 導入必要的標準庫與第三方庫
import os
import argparse
import csv
import sys
import time
import contextlib
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as VT
# 進度條工具
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None



# 設定專案根目錄，確保可以正確 import 專案內部模組
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 導入專案內部的模型構建與工具
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



# 膨脹操作：對 mask 做 max pooling，擴大 mask 區域
def dilate_mask(mask, kernel_size=5):
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size, stride=1, padding=pad)


class AlphaMaskEmbedder(torch.nn.Module):
    """Encode alpha hull masks into cross-attention tokens."""

    def __init__(self, embed_dim: int, grid_size: int = 4):
        super().__init__()
        self.grid_size = int(grid_size)
        self.pool = torch.nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, embed_dim)
        )
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float().clamp(0.0, 1.0)
        pooled = self.pool(mask).flatten(2).transpose(1, 2)
        return self.proj(pooled) + self.pos_embed


def masked_diffusion_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    inside_weight: float = 2.0,
    outside_weight: float = 0.5,
) -> torch.Tensor:
    """MSE on diffusion target with mask-aware spatial weighting."""
    mask = mask.float().clamp(0.0, 1.0)
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode='nearest')
    weight = mask * float(inside_weight) + (1.0 - mask) * float(outside_weight)
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(1)
    loss = (pred.float() - target.float()).pow(2) * weight
    return loss.sum() / weight.expand_as(pred).sum().clamp_min(1.0)



# 凸包損失：懲罰預測內容超出凸包區域
class ConvexHullLoss(torch.nn.Module):
    """
    懲罰預測內容超出凸包區域。
    pred_rgb:     [N, M, 3, H, W]，像素值範圍 [0,1]
    render_image: [N, M, 3, H, W]，像素值範圍 [0,1] 或 [-1,1]
    只比較 occupancy mask，不比較顏色。
    """
    def __init__(
        self,
        hull_threshold=0.1,     # GT/render mask 的閾值
        pred_threshold=0.5,     # 預測 soft mask 的閾值
        pred_sharpness=20.0,    # 越大越接近 hard threshold
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




# 主訓練流程
import glob

def train(args):
    device = args.device
    # 是否啟用自動混合精度（fp16）
    use_amp = bool(args.fp16 and isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    

    # 儲存預測圖像到指定路徑
    pred_image_dir = os.path.join(args.out_dir, 'pred_images')
    def _save_pred_image(tensor, save_path):
        """
        tensor: [1, H, W], [H, W], [C, H, W] or [3, H, W] (float, 0~1)
        """
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)  # CHW to HWC
        arr = (arr * 255.0).clip(0, 255).astype('uint8')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(arr).save(save_path)

    # 混合精度上下文管理器
    def amp_ctx():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    # CUDA 同步，確保計時準確
    def _sync_if_cuda():
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()


    # 顯示 VRAM 使用情況
    def _vram_str():
        """回傳目前 CUDA 裝置的 VRAM 使用狀態字串"""
        if not (isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available()):
            return ""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return (f"  [VRAM used={used/1024**3:.2f}GB / total={total/1024**3:.2f}GB,"
                f" free={free/1024**3:.2f}GB]")


    print(f"fp16 mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(
        f"mixed timestep sampling: high_t_prob={args.high_t_prob}, "
        f"min_t_ratio={args.min_t_ratio}"
    )

    # 載入模型
    print('loading model...')
    if args.config is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt)
    else:
        config = instantiate_from_config(args.config)
        model = config
    model.to(device)
    model.train()

    # support either a simple CSV-based dataset or the customized multi-view dataset
    # 自訂 collate_fn，支援 dict 結構的 batch
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


    # 根據參數選擇資料集：自訂多視角或單一 CSV 模式
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
        
        # Create validation dataset if val_meta_path is provided
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
            print(f"Validation dataset created with {len(val_dataset)} samples")
 
    # 注入 LoRA 適配器，只訓練 LoRA 參數
    from mvdream.ldm.modules.lora import inject_lora
    # 先凍結所有預訓練參數，僅 LoRA 與新 head 可訓練
    for p in model.parameters():
        p.requires_grad = False
    n_replaced = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    # LoRA 層注入後需再次移動到目標裝置
    model.to(device)
    model.train()
    print(f"Injected LoRA into {n_replaced} modules. Training adapters only.")

    # hull 圖像編碼器，只訓練投影層
    from mvdream.ldm.modules.encoders.modules import ImageEmbedder
    image_encoder = ImageEmbedder(device=device, img_size=args.size)
    image_encoder.to(device)
    image_encoder.train()
    for p in image_encoder.backbone.parameters():
        p.requires_grad = False
    for p in image_encoder.proj.parameters():
        p.requires_grad = True
    image_encoder.backbone.eval()


    # 從 dataloader 取一個 batch 推斷維度
    dummy_batch = next(iter(dl))
    hulls = dummy_batch['hulls'][0].to(device)  # (n_views, 3, H, W)
    hull_masks = dummy_batch['hull_masks'][0].to(device)  # (n_views, 1, H, W)
    gts = dummy_batch['render_image_groundtruth'][0].to(device)  # (n_views, C, H, W)
    poses = dummy_batch['poses'][0].to(device)  # (n_views, 3, 4)
    category = dummy_batch['category'][0]
    n_views = gts.shape[0]
    text_c = model.get_learned_conditioning([str(category)] * n_views).to(device)
    hull_rep_list = []
    for k in range(hulls.shape[0]):
        img = hulls[k].unsqueeze(0)
        e = image_encoder.encode(img)
        hull_rep_list.append(e.squeeze(0))
    hull_rep = torch.stack(hull_rep_list, dim=0)
    # cond_proj
    if hull_rep.shape[-1] != text_c.shape[-1]:
        cond_proj = torch.nn.Linear(hull_rep.shape[-1], text_c.shape[-1]).to(device)
        hull_rep_proj = cond_proj(hull_rep)
    else:
        cond_proj = None
        hull_rep_proj = hull_rep
    # ref_pose_proj
    pose_tokens = poses.reshape(n_views, -1)
    ref_pose_proj = torch.nn.Linear(pose_tokens.shape[-1], text_c.shape[-1]).to(device)
    pose_rep = ref_pose_proj(pose_tokens).unsqueeze(1)
    alpha_mask_encoder = AlphaMaskEmbedder(
        embed_dim=text_c.shape[-1],
        grid_size=args.mask_token_grid,
    ).to(device)
    mask_rep = alpha_mask_encoder(hull_masks)
    print(
        f"Alpha mask condition enabled: mask_tokens={mask_rep.shape[1]}, "
        f"dim={mask_rep.shape[-1]}"
    )

    # ----------- 優化器收集所有參數 -----------
    trainable = [p for p in model.parameters() if p.requires_grad] + list(image_encoder.proj.parameters())
    if cond_proj is not None:
        trainable += list(cond_proj.parameters())
    trainable += list(ref_pose_proj.parameters())
    trainable += list(alpha_mask_encoder.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # 其他初始化
    hull_criterion = ConvexHullLoss(hull_threshold=0.1, use_dilation=True, kernel_size=5).to(device)
    target_save_epochs = set(args.save_pred_images_epoch) if args.save_pred_images_epoch is not None else None
    start_epoch = 0
    
    # Create DDIM sampler for validation
    sampler = DDIMSampler(model)
    
    # Create validation log file
    if val_dl is not None:
        val_log_path = os.path.join(args.out_dir, 'validation_log.csv')
        if start_epoch == 0:  # Only write header if starting from scratch
            with open(val_log_path, 'w') as f:
                f.write('epoch,val_loss\n')
        print(f"Validation results will be saved to: {val_log_path}")
    
    if args.resume_ckpt:
        resume = torch.load(args.resume_ckpt, map_location="cpu")
        model_state = resume.get("model_state", resume)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Resumed model adapter from {args.resume_ckpt}; missing={len(missing)}, unexpected={len(unexpected)}")
        if "image_encoder_state" in resume:
            image_encoder.load_state_dict(resume["image_encoder_state"])
            image_encoder.to(device)
            image_encoder.backbone.eval()
        if "alpha_mask_encoder_state" in resume:
            alpha_mask_encoder.load_state_dict(resume["alpha_mask_encoder_state"])
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
        print(f"Resume training from epoch {start_epoch} / target epochs {args.epochs}")

    # Validation function
    def validate(val_loader, model, sampler, image_encoder, alpha_mask_encoder, cond_proj, ref_pose_proj, device, fp16, save_images=False, save_dir=None):
        """Run validation and return average loss"""
        model.eval()
        image_encoder.eval()
        alpha_mask_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        saved_samples = 0
        max_save_samples = 4  # Save 4 validation samples for visualization
        
        with torch.no_grad():
            for batch in val_loader:
                if 'render_image_groundtruth' not in batch:
                    continue
                    
                bs = len(batch['uid']) if 'uid' in batch else batch['render_image_groundtruth'].shape[0]
                n_views = batch['render_image_groundtruth'][0].shape[0]
                
                # Prepare batch data
                hulls = batch['hulls'].to(device)
                hull_masks = batch['hull_masks'].to(device)
                gts = batch['render_image_groundtruth'].to(device)
                poses = batch['poses'].to(device)
                categories = batch['category']
                
                # Flatten batch and view dimensions
                hulls = hulls.view(bs * n_views, 3, hulls.shape[-2], hulls.shape[-1])
                hull_masks = hull_masks.view(bs * n_views, 1, hull_masks.shape[-2], hull_masks.shape[-1])
                gts = gts.view(bs * n_views, gts.shape[-2], gts.shape[-1], gts.shape[-1])
                poses_flat = poses.view(bs * n_views, 3, 4)
                
                # Encode conditioning
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    # Hull embedding
                    hull_emb = image_encoder(hulls)
                    if cond_proj is not None:
                        hull_emb = cond_proj(hull_emb)
                    
                    # Alpha mask embedding
                    alpha_mask_emb = alpha_mask_encoder(hull_masks.float())
                    
                    # Text conditioning
                    text_prompts = []
                    for i in range(bs):
                        cat = categories[i]
                        text_prompts.extend([str(cat)] * n_views)
                    c_text = model.get_learned_conditioning(text_prompts)
                    
                    # Pose embedding
                    pose_tokens = poses_flat.reshape(bs * n_views, -1)
                    pose_emb = ref_pose_proj(pose_tokens)
                    
                    # Combine conditioning
                    hull_rep = hull_emb.unsqueeze(1)
                    alpha_rep = alpha_mask_emb
                    pose_rep = pose_emb.unsqueeze(1)
                    c = torch.cat([c_text, hull_rep, alpha_rep, pose_rep], dim=1)
                    
                    # Encode images to latent
                    x_0 = model.encode_first_stage(gts.to(model.dtype))
                    x_0 = model.get_first_stage_encoding(x_0).detach()
                    
                    # Sample noise and timestep
                    noise = torch.randn_like(x_0)
                    t = torch.randint(0, model.num_timesteps, (x_0.shape[0],), device=device).long()
                    x_t = model.q_sample(x_start=x_0, t=t, noise=noise)
                    
                    # Predict noise
                    noise_pred = model.apply_model(x_t, t, c)
                    
                    # Diffusion loss
                    loss_diff = F.mse_loss(noise_pred, noise, reduction='mean')
                    loss = loss_diff
                
                total_loss += loss.item()
                num_batches += 1
                
                # Save validation images every N epochs (simplified version)
                if save_images and saved_samples < max_save_samples and save_dir is not None:
                    batch_idx = 0
                    if batch_idx < bs:
                        sample_dir = os.path.join(save_dir, f'sample_{saved_samples:03d}')
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        # Save first view of first sample
                        view_idx = 0
                        img_idx = batch_idx * n_views + view_idx
                        img_gt = gts[img_idx].detach().cpu()
                        if img_gt.shape[0] == 3:  # RGB
                            img_gt = (img_gt.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                        else:  # Single channel
                            img_gt = (img_gt.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
                            img_gt = np.stack([img_gt]*3, axis=-1)  # Convert to RGB for saving
                        Image.fromarray(img_gt).save(os.path.join(sample_dir, 'gt_view_000.png'))
                        
                        saved_samples += 1
        
        model.train()
        image_encoder.train()
        alpha_mask_encoder.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    for epoch in range(start_epoch, args.epochs):
        epoch_desc = f"epoch {epoch + 1}/{args.epochs}"
        epoch_loader = tqdm(dl, total=len(dl), desc=epoch_desc, leave=True) if tqdm is not None else dl
        for it, batch in enumerate(epoch_loader):
            iter_t0 = time.perf_counter()
            current_loss = None
            if args.debug_timing and it == 0:
                print(f"[timing] first batch fetched in {time.perf_counter() - iter_t0:.2f}s{_vram_str()}")
            if 'render_image_groundtruth' in batch:
                bs = len(batch['uid']) if 'uid' in batch else batch['render_image_groundtruth'].shape[0]
                B = bs  # B = batch_size
                n_views = batch['render_image_groundtruth'][0].shape[0]
                # 拼接 batch 維度
                hulls = batch['hulls'].to(device)
                hull_masks = batch['hull_masks'].to(device)
                gts = batch['render_image_groundtruth'].to(device)
                poses = batch['poses'].to(device)
                categories = batch['category']

                # 展平 batch 與 view 維度
                hulls = hulls.view(bs * n_views, 3, hulls.shape[-2], hulls.shape[-1])
                hull_masks = hull_masks.view(bs * n_views, 1, hull_masks.shape[-2], hull_masks.shape[-1])
                gts = gts.view(bs * n_views, gts.shape[2], gts.shape[-2], gts.shape[-1])
                poses = poses.view(bs * n_views, 3, 4)
                # category 展平
                categories = [str(c) for c in categories for _ in range(n_views)]

                # 條件編碼
                text_c = model.get_learned_conditioning(categories).to(device)  # [bs*n_views, 77, D]
                # hull 編碼
                hull_rep_list = []
                for k in range(hulls.shape[0]):
                    img = hulls[k].unsqueeze(0)
                    e = image_encoder.encode(img)
                    hull_rep_list.append(e.squeeze(0))
                hull_rep = torch.stack(hull_rep_list, dim=0)
                if cond_proj is not None:
                    hull_rep = cond_proj(hull_rep)
                # alpha mask condition
                mask_rep = alpha_mask_encoder(hull_masks)
                # pose 編碼
                pose_tokens = poses.reshape(bs * n_views, -1)
                pose_rep = ref_pose_proj(pose_tokens).unsqueeze(1)
                # 拼接 context
                context_cat = torch.cat([text_c, hull_rep, mask_rep, pose_rep], dim=1)

                # ground truth latent
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
                t = t_single.repeat(bs * n_views)
                noise = torch.randn_like(z)
                x_t = model.q_sample(z, t, noise=noise)

                # camera tensor
                camera_list = []
                for j in range(bs * n_views):
                    p = poses[j]
                    if p.shape == (3, 4):
                        bottom = torch.tensor([[0., 0., 0., 1.]], device=device)
                        p4 = torch.cat([p, bottom], dim=0)
                    else:
                        p4 = p
                    camera_list.append(p4.reshape(-1))
                camera_tensor = torch.stack(camera_list, dim=0).to(device)

                with amp_ctx():
                    model_out = model.apply_model(
                        x_t,
                        t,
                        {
                            'context': context_cat,
                            'camera': camera_tensor,
                            'num_frames': n_views,
                        }
                    )
                    if model.parameterization == 'v':
                        pred_x0 = model.predict_start_from_z_and_v(x_t, t, model_out)
                        pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                    else:
                        pred = model_out
                        pred_x0 = model.predict_start_from_noise(x_t, t, pred)
                    diffusion_loss = masked_diffusion_mse_loss(
                        pred,
                        noise,
                        hull_masks,
                        inside_weight=args.mask_loss_inside_weight,
                        outside_weight=args.mask_loss_outside_weight,
                    )
                    gt_mask = None  # hull_criterion 已弃用，gt_mask 仅为后续保存接口保留
                    # === 新增 pixel_loss、soft foreground mask、outside_loss，並保存 pred_soft_mask 圖像 ===
                    # =========================================================
                    # =========================================================
                    # =========================================================
                    # decode predicted x0
                    # decoded_imgs = model.decode_first_stage(pred_x0)
                    # decoded_imgs = torch.clamp((decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                    # # 處理 groundtruth 圖像
                    # gt_imgs = gts
                    # if gt_imgs.min() < 0:
                    #     gt_imgs = (gt_imgs + 1.0) / 2.0
                    # gt_imgs = torch.clamp(gt_imgs, 0.0, 1.0)
                    # # pixel supervision（已取消，不再加入loss）
                    # # pixel_loss = F.l1_loss(decoded_imgs, gt_imgs)
                    # hull_mask = hull_masks.float()
                    # # 方案1：从RGB推soft occupancy（pseudo alpha）
                    # gray = decoded_imgs.mean(dim=1, keepdim=True)
                    # # white background, gray/dark object
                    # pred_alpha = torch.sigmoid(30.0 * (0.95 - gray))

                    # # hull mask
                    # hull_mask = hull_masks.float()

                    # # 方案：从 denoised RGB 推 pseudo alpha
                    # gray = decoded_imgs.mean(dim=1, keepdim=True)

                    # # normalize each image separately，避免背景不是纯白导致 alpha 偏灰
                    # gray_min = gray.amin(dim=(2, 3), keepdim=True)
                    # gray_max = gray.amax(dim=(2, 3), keepdim=True)
                    # gray_norm = (gray - gray_min) / (gray_max - gray_min + 1e-8)

                    # # white background / gray-dark object:
                    # # 背景亮 -> 0，物体暗 -> 1
                    # pred_alpha = 1.0 - gray_norm

                    # # 压低背景灰度，让背景更接近黑色
                    # pred_alpha = pred_alpha.clamp(0.0, 1.0)
                    # pred_alpha = pred_alpha.pow(4)

                    # if hull_mask.shape[-2:] != pred_alpha.shape[-2:]:
                    #     hull_mask = F.interpolate(
                    #         hull_mask,
                    #         size=pred_alpha.shape[-2:],
                    #         mode='nearest'
                    #     )

                    # outside_loss = (pred_alpha * (1.0 - hull_mask)).mean()
                    # =========================================================
                    # =========================================================
                    # =========================================================


                    # =========================================================
                    # Randomly sample ONE view per object
                    # =========================================================

                    V = n_views

                    # reshape back to [B, V, C, H, W]
                    pred_x0 = pred_x0.view(B, V, *pred_x0.shape[1:])
                    hull_masks = hull_masks.view(B, V, *hull_masks.shape[1:])

                    # random select one view for each object
                    rand_views = torch.randint(
                        0,
                        V,
                        (B,),
                        device=pred_x0.device
                    )

                    batch_ids = torch.arange(B, device=pred_x0.device)

                    # [B, C, H, W]
                    pred_x0_small = pred_x0[batch_ids, rand_views]

                    # [B, 1, H, W]
                    hull_mask = hull_masks[batch_ids, rand_views].float()

                    # =========================================================
                    # Decode only sampled views
                    # =========================================================

                    with torch.cuda.amp.autocast():

                        decoded_imgs = model.decode_first_stage(pred_x0_small)

                        decoded_imgs = torch.clamp(
                            (decoded_imgs + 1.0) / 2.0,
                            0.0,
                            1.0
                        )

                        # -----------------------------------------------------
                        # pseudo alpha from RGB
                        # -----------------------------------------------------

                        gray = decoded_imgs.mean(dim=1, keepdim=True)

                        # normalize per-image
                        gray_min = gray.amin(dim=(2, 3), keepdim=True)
                        gray_max = gray.amax(dim=(2, 3), keepdim=True)

                        gray_norm = (gray - gray_min) / (
                            gray_max - gray_min + 1e-8
                        )

                        # white background -> 0
                        # dark object -> 1
                        pred_alpha = 1.0 - gray_norm

                        pred_alpha = pred_alpha.clamp(0.0, 1.0)

                        # sharpen alpha
                        pred_alpha = pred_alpha.pow(4)

                        # resize hull mask if needed
                        if hull_mask.shape[-2:] != pred_alpha.shape[-2:]:
                            hull_mask = F.interpolate(
                                hull_mask,
                                size=pred_alpha.shape[-2:],
                                mode='nearest'
                            )

                        # outside penalty
                        outside_loss = (
                            pred_alpha * (1.0 - hull_mask)
                        ).mean()

                    # 保存 pred_alpha
                    if args.save_pred_images and (target_save_epochs is None or (epoch + 1) in target_save_epochs):
                        max_save = min(8, pred_alpha.shape[0])
                        for idx in range(max_save):
                            pred_alpha_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_predalpha_{idx}.png')
                            _save_pred_image(pred_alpha[idx], pred_alpha_path)
                            print(f"[LOG] Saved predalpha: {pred_alpha_path}")
                    total_loss = diffusion_loss + args.lambda_hull * outside_loss
                    print(f"[LOG] epoch={epoch+1} iter={it} total_loss={total_loss.item():.6f} diffusion_loss={diffusion_loss.item():.6f} outside_loss={outside_loss.item():.6f}")

                    # === 儲存 pred_alpha、ground truth alpha、denoise 圖、rgb_convexhull、rgb_groundtruth ===
                    if args.save_pred_images and (target_save_epochs is None or (epoch + 1) in target_save_epochs):

                        V = n_views

                        # pred_x0 已经是 [B, V, C, H, W]
                        pred_x0_reshaped = pred_x0

                        # z 和 x_t 还没有 reshape，可以 reshape
                        z_reshaped = z.view(B, V, *z.shape[1:])
                        x_t_reshaped = x_t.view(B, V, *x_t.shape[1:])

                        rand_views = torch.randint(0, V, (B,), device=pred_x0.device)
                        batch_ids = torch.arange(B, device=pred_x0.device)

                        pred_x0_small = pred_x0_reshaped[batch_ids, rand_views]
                        z_small = z_reshaped[batch_ids, rand_views]
                        x_t_small = x_t_reshaped[batch_ids, rand_views]
                        with torch.no_grad():
                            decoded_imgs = model.decode_first_stage(pred_x0_small)
                            decoded_imgs = torch.clamp((decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                            # decode groundtruth latent (z) for comparison
                            gt_decoded_imgs = model.decode_first_stage(z_small)
                            gt_decoded_imgs = torch.clamp((gt_decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                            # decode noisy latent (x_t) for visualization
                            noisy_decoded_imgs = model.decode_first_stage(x_t_small)
                            noisy_decoded_imgs = torch.clamp((noisy_decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                        # 嘗試從 batch 取出原始路徑資訊
                        batch_uids = batch['uid'] if 'uid' in batch else None
                        batch_view_ids = batch['selected_view_ids'] if 'selected_view_ids' in batch else None
                        from mvdream.datasets.customized_dataset_dir1 import customizedDataset
                        dataset_for_load = None
                        if batch_uids is not None and batch_view_ids is not None and hasattr(args, 'dataset_root') and hasattr(args, 'meta_path'):
                            dataset_for_load = customizedDataset(args.dataset_root, args.meta_path)
                        for idx in range(B):
                            # 只保存 predalpha_*.png，不再保存 pred_*.png，避免重复
                            pred_alpha_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_predalpha_{idx}.png')
                            _save_pred_image(pred_alpha[idx], pred_alpha_path)
                            denoise_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_denoise_{idx}.png')
                            _save_pred_image(decoded_imgs[idx], denoise_path)
                            gt_decoded_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_gtdecoded_{idx}.png')
                            _save_pred_image(gt_decoded_imgs[idx], gt_decoded_path)
                            noisy_decoded_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_noisy_{idx}.png')
                            _save_pred_image(noisy_decoded_imgs[idx], noisy_decoded_path)

                            # 保存 groundtruth mask
                            if gt_mask is not None:
                                gt_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_gt_{idx}.png')
                                gt_img = gt_mask[idx]
                                if gt_img.dim() == 4:
                                    gt_img = gt_img[0]
                                _save_pred_image(gt_img, gt_path)

                            # 額外儲存 rgb_convexhull 和 rgb_groundtruth
                            if dataset_for_load is not None and batch_uids is not None and batch_view_ids is not None:
                                uid = batch_uids[idx]
                                # 取采样的view
                                view_id = rand_views[idx].item() if hasattr(rand_views[idx], 'item') else int(rand_views[idx])
                                convex_path = os.path.join(args.dataset_root, uid, 'rgb_convexhull', f'{view_id:03d}.png')
                                gtimg_path = os.path.join(args.dataset_root, uid, 'rgb_groundtruth', f'{view_id:03d}.png')
                                try:
                                    rgb_convex, _ = dataset_for_load._load_rgba_with_alpha(convex_path)
                                    rgb_gt, _ = dataset_for_load._load_rgba_with_alpha(gtimg_path)
                                    convex_save_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_convexhull_{idx}.png')
                                    gtimg_save_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_groundtruth_{idx}.png')
                                    _save_pred_image(rgb_convex, convex_save_path)
                                    _save_pred_image(rgb_gt, gtimg_save_path)
                                except Exception as e:
                                    print(f"[WARN] Failed to save rgb_convexhull/groundtruth for {uid} view {view_id}: {e}")

                if args.debug_timing and it == 0:
                    _sync_if_cuda()
                    bw_t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                if args.debug_timing and it == 0:
                    _sync_if_cuda()
                    print(f"[timing] backward+step: {time.perf_counter() - bw_t0:.2f}s{_vram_str()}")

                if it % args.log_steps == 0:
                    current_loss = total_loss.item()
                    print(f"epoch {epoch} iter {it} batch_loss {current_loss:.6f}")

            if current_loss is not None and hasattr(epoch_loader, 'set_postfix'):
                epoch_loader.set_postfix(loss=f"{current_loss:.6f}")

        # Run validation if validation dataset exists
        if val_dl is not None:
            save_val_images = (target_save_epochs is None or (epoch + 1) in target_save_epochs) and ((epoch + 1) % 10 == 0)
            val_save_dir = os.path.join(args.out_dir, f'validation_epoch_{epoch}') if save_val_images else None
            if val_save_dir:
                os.makedirs(val_save_dir, exist_ok=True)
            
            val_loss = validate(val_dl, model, sampler, image_encoder, alpha_mask_encoder, cond_proj, ref_pose_proj, device, args.fp16, 
                               save_images=save_val_images, save_dir=val_save_dir)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Log validation results to CSV
            val_log_path = os.path.join(args.out_dir, 'validation_log.csv')
            with open(val_log_path, 'a') as f:
                f.write(f'{epoch},{val_loss:.6f}\n')
        
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
            'alpha_mask_encoder_state': alpha_mask_encoder.state_dict(),
            'alpha_mask_encoder_grid_size': args.mask_token_grid,
        }
        if cond_proj is not None:
            save_dict['cond_proj_state'] = cond_proj.state_dict()
        if ref_pose_proj is not None:
            save_dict['ref_pose_proj_state'] = ref_pose_proj.state_dict()
        torch.save(save_dict, ckpt_path)
        print(f"checkpoint saved (adapter-only): {ckpt_path}; model tensors: {len(adapter_state)}")
        # Optionally keep only the latest checkpoint
        if getattr(args, 'keep_latest_ckpt_only', False):
            ckpt_files = sorted(glob.glob(os.path.join(args.out_dir, 'ckpt_epoch_*.pth')))
            for f in ckpt_files[:-1]:
                try:
                    os.remove(f)
                    print(f"Deleted old checkpoint: {f}")
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")




# 主程式入口：解析命令列參數，建立資料夾並啟動訓練
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_latest_ckpt_only', action='store_true', help='Only keep the latest checkpoint, delete all previous ones after each epoch')
    parser.add_argument('--csv', type=str, default=None, help='CSV path for single-image fallback dataset mode')
    parser.add_argument('--model_name', type=str, default='sd-v2.1-base-4view')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
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
    parser.add_argument('--mask_token_grid', type=int, default=4, help='alpha mask condition token grid size; 4 means 16 mask tokens')
    parser.add_argument('--mask_loss_inside_weight', type=float, default=2.0, help='diffusion MSE weight inside alpha hull mask')
    parser.add_argument('--mask_loss_outside_weight', type=float, default=0.5, help='diffusion MSE weight outside alpha hull mask')
    parser.add_argument('--high_t_prob', type=float, default=0.7, help='probability of sampling timestep from the high-noise range')
    parser.add_argument('--min_t_ratio', type=float, default=0.7, help='lower timestep ratio used when high-noise sampling is selected')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='resume training from an adapter checkpoint saved by this script')
    args = parser.parse_args()

    if args.save_pred_images_epoch is not None and any(epoch_idx < 1 for epoch_idx in args.save_pred_images_epoch):
        raise ValueError('--save_pred_images_epoch values must be >= 1 (1-based epoch indices)')
    if not 0.0 <= args.high_t_prob <= 1.0:
        raise ValueError('--high_t_prob must be in [0, 1]')
    if not 0.0 <= args.min_t_ratio < 1.0:
        raise ValueError('--min_t_ratio must be in [0, 1)')

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
