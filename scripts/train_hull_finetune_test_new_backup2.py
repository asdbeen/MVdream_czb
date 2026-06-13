
# 導入必要的標準庫與第三方庫
import os
import argparse
import csv
import sys
import time
import contextlib
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



# 膨脹操作：對 mask 做 max pooling，擴大 mask 區域
def dilate_mask(mask, kernel_size=5):
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size, stride=1, padding=pad)



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


    # ----------- 預先初始化 mask_head, cond_proj, ref_pose_proj -----------
    # class MaskHead(torch.nn.Module):
    #     def __init__(self, latent_dim):
    #         super().__init__()
    #         hidden = max(latent_dim // 2, 8)
    #         self.conv = torch.nn.Sequential(
    #             torch.nn.Conv2d(latent_dim, hidden, 3, padding=1),
    #             torch.nn.SiLU(),
    #             torch.nn.Conv2d(hidden, 1, 1)
    #         )
    #     def forward(self, x):
    #         return torch.sigmoid(self.conv(x))


    class MaskHead(torch.nn.Module):
        def __init__(self, latent_dim, sharpness=10.0):
            super().__init__()

            hidden = max(latent_dim // 2, 8)

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(latent_dim, hidden, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(hidden, 1, 1)
            )

            # sigmoid sharpen factor
            self.sharpness = sharpness

        def forward(self, x):

            logits = self.conv(x)

            # sharpen sigmoid
            alpha = torch.sigmoid(self.sharpness * logits)

            return alpha
    
    # 從 dataloader 取一個 batch 推斷維度
    dummy_batch = next(iter(dl))
    hulls = dummy_batch['hulls'][0].to(device)  # (n_views, 3, H, W)
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
    # mask_head
    with torch.no_grad():
        enc_posterior = model.encode_first_stage(gts)
        z = model.get_first_stage_encoding(enc_posterior)
    latent_dim = z.shape[1]
    mask_head = MaskHead(latent_dim).to(device)

    # ----------- 優化器收集所有參數 -----------
    trainable = [p for p in model.parameters() if p.requires_grad] + list(image_encoder.proj.parameters())
    if cond_proj is not None:
        trainable += list(cond_proj.parameters())
    trainable += list(ref_pose_proj.parameters())
    trainable += list(mask_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # 其他初始化
    hull_criterion = ConvexHullLoss(hull_threshold=0.1, use_dilation=True, kernel_size=5).to(device)
    target_save_epochs = set(args.save_pred_images_epoch) if args.save_pred_images_epoch is not None else None


    for epoch in range(args.epochs):
        epoch_desc = f"epoch {epoch + 1}/{args.epochs}"
        epoch_loader = tqdm(dl, total=len(dl), desc=epoch_desc, leave=True) if tqdm is not None else dl
        for it, batch in enumerate(epoch_loader):
            iter_t0 = time.perf_counter()
            current_loss = None
            if args.debug_timing and it == 0:
                print(f"[timing] first batch fetched in {time.perf_counter() - iter_t0:.2f}s{_vram_str()}")
            if 'render_image_groundtruth' in batch:
                bs = len(batch['uid']) if 'uid' in batch else batch['render_image_groundtruth'].shape[0]
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
                # pose 編碼
                pose_tokens = poses.reshape(bs * n_views, -1)
                pose_rep = ref_pose_proj(pose_tokens).unsqueeze(1)
                # 拼接 context
                context_cat = torch.cat([text_c, hull_rep, pose_rep], dim=1)

                # ground truth latent
                with torch.no_grad():
                    enc_posterior = model.encode_first_stage(gts)
                    z = model.get_first_stage_encoding(enc_posterior)

                t_single = torch.randint(0, model.num_timesteps, (1,), device=device).long()
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
                    diffusion_loss = F.mse_loss(pred, noise)
                    latent = pred_x0
                    pred_alpha = mask_head(latent)
                    hull_mask = hull_masks
                    if hull_mask.shape[-2:] != pred_alpha.shape[-2:]:
                        pred_alpha = F.interpolate(
                            pred_alpha,
                            size=hull_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    hull_loss_val, _, pred_mask, gt_mask = hull_criterion(
                        pred_alpha.unsqueeze(1),
                        hull_mask.unsqueeze(1)
                    )
                    total_loss = diffusion_loss + args.lambda_hull * hull_loss_val
                    print(f"[DEBUG] batch_loss: {total_loss.item():.6f} (diffusion: {diffusion_loss.item():.6f}, hull: {hull_loss_val.item():.6f})")

                    # === 儲存 pred_alpha、ground truth alpha、denoise 圖、rgb_convexhull、rgb_groundtruth ===
                    if args.save_pred_images and (target_save_epochs is None or (epoch + 1) in target_save_epochs):
                        # 只存第一個 batch 的前幾張
                        max_save = min(8, pred_alpha.shape[0])
                        with torch.no_grad():
                            decoded_imgs = model.decode_first_stage(pred_x0[:max_save].detach())
                            decoded_imgs = torch.clamp((decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                            # decode groundtruth latent (z) for comparison
                            gt_decoded_imgs = model.decode_first_stage(z[:max_save].detach())
                            gt_decoded_imgs = torch.clamp((gt_decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                            # decode noisy latent (x_t) for visualization
                            noisy_decoded_imgs = model.decode_first_stage(x_t[:max_save].detach())
                            noisy_decoded_imgs = torch.clamp((noisy_decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
                        # 嘗試從 batch 取出原始路徑資訊
                        batch_uids = batch['uid'] if 'uid' in batch else None
                        batch_view_ids = batch['selected_view_ids'] if 'selected_view_ids' in batch else None
                        from mvdream.datasets.customized_dataset_dir1 import customizedDataset
                        dataset_for_load = None
                        if batch_uids is not None and batch_view_ids is not None and hasattr(args, 'dataset_root') and hasattr(args, 'meta_path'):
                            dataset_for_load = customizedDataset(args.dataset_root, args.meta_path)
                        for idx in range(max_save):
                            pred_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_pred_{idx}.png')
                            gt_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_gt_{idx}.png')
                            denoise_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_denoise_{idx}.png')
                            _save_pred_image(pred_alpha[idx], pred_path)
                            gt_img = gt_mask[idx]
                            if gt_img.dim() == 4:
                                gt_img = gt_img[0]
                            _save_pred_image(gt_img, gt_path)
                            _save_pred_image(decoded_imgs[idx], denoise_path)
                            # save decoded groundtruth latent
                            gt_decoded_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_gtdecoded_{idx}.png')
                            _save_pred_image(gt_decoded_imgs[idx], gt_decoded_path)
                            # save decoded noisy latent (加噪音的)
                            noisy_decoded_path = os.path.join(pred_image_dir, f'epoch{epoch+1}_iter{it}_noisy_{idx}.png')
                            _save_pred_image(noisy_decoded_imgs[idx], noisy_decoded_path)

                            # 額外儲存 rgb_convexhull 和 rgb_groundtruth
                            if dataset_for_load is not None:
                                # idx 對應 batch 中的第幾個樣本
                                batch_idx = idx // n_views
                                view_idx = idx % n_views
                                uid = batch_uids[batch_idx]
                                view_id = batch_view_ids[batch_idx][view_idx].item() if hasattr(batch_view_ids[batch_idx][view_idx], 'item') else int(batch_view_ids[batch_idx][view_idx])
                                # 構造路徑
                                convex_path = os.path.join(args.dataset_root, uid, 'rgb_convexhull', f'{view_id:03d}.png')
                                gtimg_path = os.path.join(args.dataset_root, uid, 'rgb_groundtruth', f'{view_id:03d}.png')
                                # 用 _load_rgba_with_alpha 處理
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
    parser.add_argument('--num_views', type=int, default=4, help='number of side views to sample for customized dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (set 0 on Windows for stability)')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='enable fp16 mixed precision (use --no-fp16 to disable)')
    parser.add_argument('--debug_timing', action='store_true', help='print detailed timing for first batch')
    parser.add_argument('--save_pred_images', action='store_true', help='save decoded predicted images to out_dir/pred_images during training')
    parser.add_argument('--save_pred_images_epoch', type=int, nargs='+', default=None, help='1-based epoch index list to save predicted images (e.g. --save_pred_images_epoch 100 200); requires --save_pred_images')
    parser.add_argument('--lambda_hull', type=float, default=1, help='weight for convex-hull alpha loss')
    args = parser.parse_args()

    if args.save_pred_images_epoch is not None and any(epoch_idx < 1 for epoch_idx in args.save_pred_images_epoch):
        raise ValueError('--save_pred_images_epoch values must be >= 1 (1-based epoch indices)')

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
