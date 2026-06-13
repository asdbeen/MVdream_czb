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
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvdream.ldm.util import instantiate_from_config
from mvdream.model_zoo import build_model


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
        from mvdream.datasets.customized_dataset import customizedDataset
        dataset = customizedDataset(
            args.dataset_root,
            args.meta_path,
            sample_side_views=args.num_views,
            source_image_res=args.size,
            use_value_json=False,
        )
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate)
    else:
        if args.csv is None:
            raise ValueError("--csv is required when --dataset_root/--meta_path are not provided")
        dataset = HullDataset(args.csv, img_size=args.size)
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate)

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

    for epoch in range(args.epochs):
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
                  
                    # for each target view supervise
                    sample_loss = None
                    for j in range(gts.shape[0]):
                        view_t0 = time.perf_counter()
                        if args.debug_timing and it == 0:
                            if selected_view_ids is not None:
                                view_id = int(selected_view_ids[j].item())
                                view_path = os.path.join(args.dataset_root, uid, 'rgb_convexhull', f"{view_id:03d}.png")
                                print(f"[timing] view {j} path: {view_path}")
                            else:
                                print(f"[timing] view {j} path: unavailable")


                        gt_img = gts[j].unsqueeze(0)  # (1,3,H,W)
                        # 预测RGB

                        with torch.no_grad():
                            enc_posterior = model.encode_first_stage(gt_img)
                            z = model.get_first_stage_encoding(enc_posterior)
                        t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device).long()
                        noise = torch.randn_like(z)
                        x_t = model.q_sample(z, t, noise=noise)

                        # build camera 4x4 flattened (from 3x4)
                        p = poses[j]
                        if p.shape == (3, 4):
                            p4 = torch.cat([p, torch.tensor([[0.,0.,0.,1.]], device=device)], dim=0)
                        else:
                            p4 = p
                        camera_tensor = p4.reshape(1, -1).to(device)

                        with amp_ctx():
                            model_out = model.apply_model(x_t, t, {'context': context_cat, 'camera': camera_tensor, 'num_frames': 1})
                            if model.parameterization == 'v':
                                pred_x0 = model.predict_start_from_z_and_v(x_t, t, model_out)
                                pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                            else:
                                pred = model_out
                                pred_x0 = model.predict_start_from_noise(x_t, t, pred)

                            # 1. pred_rgb loss (MSE)
                            l = F.mse_loss(pred, noise)

                            # 2. mask_head: 用 latent 预测 pred_alpha
                            # 取 pred_x0 作为 latent，解码前
                            latent = pred_x0

                            if mask_head is None:
                                latent_dim = latent.shape[1]
                                mask_head = MaskHead(latent_dim).to(device)
                                optimizer.add_param_group({'params': mask_head.parameters()})

                            pred_alpha = mask_head(latent)  # [1,1,h,w]

                            hull_mask = hull_masks[j].unsqueeze(0)  # [1,1,H,W]
                            if hull_mask.shape[-2:] != pred_alpha.shape[-2:]:
                                pred_alpha = F.interpolate(
                                    pred_alpha,
                                    size=hull_mask.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                )

                            hull_loss_val, _, _, _ = hull_criterion(
                                pred_alpha.unsqueeze(1),   # [1,1,1,H,W]
                                hull_mask.unsqueeze(1)     # [1,1,1,H,W]
                            )

                            term_loss = l + args.lambda_hull * hull_loss_val
                            
                            # print ("term_loss",term_loss.item())
                            sample_loss = term_loss if sample_loss is None else (sample_loss + term_loss)
                            print (f"term_loss: {term_loss.item():.6f} (MSE: {l.item():.6f}, Hull: {hull_loss_val.item():.6f})")
                        should_save_pred = args.save_pred_images and (
                            target_save_epochs is None or (epoch + 1) in target_save_epochs
                        )
                        if should_save_pred:
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
                            _save_pred_image(pred_x0, save_path)
                        if args.debug_timing and it == 0 and bi == 0:
                            print(f"[timing] view {j} step: {time.perf_counter() - view_t0:.2f}s{_vram_str()}")

                    sample_loss = sample_loss / float(gts.shape[0])
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
