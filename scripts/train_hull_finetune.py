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
    Penalize predicted opacity outside the convex-hull mask from render images.

    Args:
        threshold: binarization threshold for render_image-derived hull mask.
        use_dilation: whether to dilate hull mask for tolerance.
        kernel_size: dilation kernel size.
        eps: denominator stability epsilon.
    """

    def __init__(self, threshold=0.1, use_dilation=False, kernel_size=5, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.use_dilation = use_dilation
        self.kernel_size = kernel_size
        self.eps = eps

    def _to_hull_mask(self, render_image):
        if render_image.dim() != 5:
            raise ValueError(f"render_image must be [N, M, C, H, W], got {tuple(render_image.shape)}")
        # Convert RGB render image to binary silhouette-like mask.
        gray = render_image.mean(dim=2, keepdim=True)
        hull_mask = (gray > self.threshold).to(render_image.dtype)
        return hull_mask

    def forward(self, pred_alpha, render_image):
        if pred_alpha.dim() != 5:
            raise ValueError(f"pred_alpha must be [N, M, 1, H, W], got {tuple(pred_alpha.shape)}")

        hull_mask = self._to_hull_mask(render_image)
        if hull_mask.shape[-2:] != pred_alpha.shape[-2:]:
            n, m, c, h, w = hull_mask.shape
            target_h, target_w = pred_alpha.shape[-2], pred_alpha.shape[-1]
            hull_mask = F.interpolate(
                hull_mask.view(n * m, c, h, w),
                size=(target_h, target_w),
                mode='nearest',
            ).view(n, m, c, target_h, target_w)
        pred_alpha = pred_alpha.clamp(0, 1)
        hull_mask = hull_mask.clamp(0, 1)

        if self.use_dilation:
            n, m, _, h, w = hull_mask.shape
            hull_mask = dilate_mask(hull_mask.view(n * m, 1, h, w), self.kernel_size).view(n, m, 1, h, w)

        outside = pred_alpha * (1.0 - hull_mask)

        outside_area = outside.sum(dim=(1, 2, 3, 4))
        pred_area = pred_alpha.sum(dim=(1, 2, 3, 4))

        loss = outside_area / (pred_area + self.eps)
        return loss.mean(), outside



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
        hull = Image.open(r['hull_path']).convert('RGB')
        gt = Image.open(r['gt_path']).convert('RGB')
        hull_t = self.transform(hull)
        gt_t = self.transform(gt)
        category = int(r.get('category', 0))
        sample = {'hull': hull_t, 'gt': gt_t, 'category': category}
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
    if args.dataset_root is not None and args.meta_path is not None:
        from mvdream.datasets.customized_dataset import customizedDataset
        dataset = customizedDataset(
            args.dataset_root,
            args.meta_path,
            sample_side_views=args.num_views,
            source_image_res=args.size,
            use_value_json=False,
        )
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False)
    else:
        if args.csv is None:
            raise ValueError("--csv is required when --dataset_root/--meta_path are not provided")
        dataset = HullDataset(args.csv, img_size=args.size)
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=False)

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
    image_encoder.eval()

    # mask_head will be created lazily to map latent channels -> alpha
    mask_head = None
    # cond_proj aligns hull embedding dim (e.g. 768) to text conditioning dim (e.g. 1024)
    cond_proj = None
    # ref_pose_proj maps reference camera poses (3x4 -> 12) into context tokens.
    ref_pose_proj = None

    # convex-hull loss
    hull_criterion = ConvexHullLoss(threshold=0.1, use_dilation=True, kernel_size=5).to(device)

    # collect trainable params (LoRA params). mask_head params will be added when created
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    

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
                    hulls = batch['render_image'][bi].to(device)  # (n_views, C, H, W)
                    gts = batch['render_image_groundtruth'][bi].to(device)  # (n_views, C, H, W)
                    poses = batch['poses'][bi].to(device)  # (n_views, 3, 4)
                    category = batch['category'][bi]
                    selected_view_ids = batch['selected_view_ids'][bi] if 'selected_view_ids' in batch else None

                    # encode hulls into embeddings
                    with torch.no_grad():
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
                    uc_text = model.get_learned_conditioning([""]).to(device)
                    uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)], dim=1)

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
                        gt_img = gts[j].unsqueeze(0)
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
                            l = F.mse_loss(pred, noise)

                            # hull-constrained alpha loss (compute when we can decode x0)
                            hull_loss_val = torch.tensor(0.0, device=device)
                            if model.parameterization != 'v':
                                # pred_x0 is latent (N, C, H, W) - apply mask_head on latent channels
                                if mask_head is None:
                                    in_ch = pred_x0.shape[1]
                                    mask_head = torch.nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0).to(device)
                                    optimizer.add_param_group({'params': mask_head.parameters()})
                                pred_alpha = torch.sigmoid(mask_head(pred_x0))  # [N,1,H,W]
                                pred_alpha = pred_alpha.unsqueeze(1)  # [N, M=1, 1, H, W]
                                hull_img = hulls[j].unsqueeze(0).unsqueeze(0)  # [N=1, M=1, C, H, W]
                                hull_loss_val, _ = hull_criterion(pred_alpha, hull_img)
                            term_loss = l + args.lambda_hull * hull_loss_val
                            sample_loss = term_loss if sample_loss is None else (sample_loss + term_loss)
                        if args.save_pred_images:
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

            else:
                # fallback: single-image CSV dataset
                hull = batch['hull'].to(device)
                gt = batch['gt'].to(device)
                cats = torch.tensor(batch['category'], device=device)

                # encode gt to latent
                with torch.no_grad():
                    enc_posterior = model.encode_first_stage(gt)
                    z = model.get_first_stage_encoding(enc_posterior)

                # sample timestep and noise
                t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device).long()
                noise = torch.randn_like(z)
                x_t = model.q_sample(z, t, noise=noise)

                # prepare conditioning: use category as text if no other text
                with torch.no_grad():
                    hull_embed = image_encoder.encode(hull)

                # build text context from category ints (treat category as text label)
                prompts = [str(int(c)) for c in batch['category']]
                text_c = model.get_learned_conditioning(prompts).to(device)
                c_text = text_c  # already batch-sized (len(prompts), seq, dim)

                # unconditional text context
                uc_text = model.get_learned_conditioning([""]).to(device).repeat(c_text.shape[0],1,1)

                # hull_rep: ensure batch first
                if hull_embed.shape[0] == 1:
                    hull_rep = hull_embed.repeat(c_text.shape[0],1,1)
                else:
                    hull_rep = hull_embed

                if hull_rep.shape[-1] != c_text.shape[-1]:
                    if cond_proj is None:
                        cond_proj = torch.nn.Linear(hull_rep.shape[-1], c_text.shape[-1]).to(device)
                        optimizer.add_param_group({'params': cond_proj.parameters()})
                    hull_rep = cond_proj(hull_rep)

                ref_camera = batch.get('camera', None)
                pose_rep = None
                if ref_camera is not None:
                    if not torch.is_tensor(ref_camera):
                        ref_camera = torch.tensor(ref_camera)
                    ref_camera = ref_camera.to(device=device, dtype=torch.float32)
                    if ref_camera.dim() == 1:
                        ref_camera = ref_camera.unsqueeze(0)
                    pose_tokens = ref_camera.view(ref_camera.shape[0], 1, -1)
                    if pose_tokens.shape[0] == 1 and c_text.shape[0] > 1:
                        pose_tokens = pose_tokens.repeat(c_text.shape[0], 1, 1)
                    if ref_pose_proj is None:
                        ref_pose_proj = torch.nn.Linear(pose_tokens.shape[-1], c_text.shape[-1]).to(device)
                        optimizer.add_param_group({'params': ref_pose_proj.parameters()})
                    pose_rep = ref_pose_proj(pose_tokens)

                if pose_rep is not None:
                    context_cat = torch.cat([c_text, hull_rep, pose_rep], dim=1)
                    uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep), torch.zeros_like(pose_rep)], dim=1)
                else:
                    context_cat = torch.cat([c_text, hull_rep], dim=1)
                    uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep)], dim=1)

                # class labels
                y = cats.long()
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                if y.shape[0] == 1:
                    y_rep = y.repeat(c_text.shape[0])
                else:
                    y_rep = y

                with amp_ctx():
                    # apply model -> predict noise
                    model_out = model.apply_model(x_t, t, {'context': context_cat, 'y': y_rep, 'camera': batch.get('camera', None)})

                    # predicted e_t
                    if model.parameterization == 'v':
                        # convert to eps; use model.predict_eps_from_z_and_v
                        pred_x0 = model.predict_start_from_z_and_v(x_t, t, model_out)
                        pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                    else:
                        pred = model_out
                        pred_x0 = model.predict_start_from_noise(x_t, t, pred)

                    loss = F.mse_loss(pred, noise)

                    # hull loss for CSV fallback (when we can decode)
                    hull_loss_val = torch.tensor(0.0, device=device)
                    if model.parameterization != 'v':
                        if mask_head is None:
                            in_ch = pred_x0.shape[1]
                            mask_head = torch.nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0).to(device)
                            optimizer.add_param_group({'params': mask_head.parameters()})
                        pred_alpha = torch.sigmoid(mask_head(pred_x0))  # [bs,1,H,W]
                        pred_alpha = pred_alpha.unsqueeze(1)  # [N, M=1, 1, H, W]
                        hull_img = hull.unsqueeze(1)  # [N, M=1, C, H, W]
                        hull_loss_val, _ = hull_criterion(pred_alpha, hull_img)

                    loss = F.mse_loss(pred, noise) + args.lambda_hull * hull_loss_val

                if args.save_pred_images:
                    for batch_idx in range(pred_x0.shape[0]):
                        save_path = os.path.join(
                            pred_image_dir,
                            f"epoch_{epoch:03d}",
                            f"iter_{it:05d}",
                            f"csv_sample_{batch_idx:03d}_pred.png",
                        )
                        _save_pred_image(pred_x0[batch_idx:batch_idx + 1], save_path)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if args.debug_timing and it == 0:
                    print(f"[timing] first csv-mode batch total: {time.perf_counter() - iter_t0:.2f}s{_vram_str()}")

                if it % args.log_steps == 0:
                    current_loss = loss.item()
                    print(f"epoch {epoch} iter {it} loss {loss.item():.6f}")

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
    parser.add_argument('--lambda_hull', type=float, default=1, help='weight for convex-hull alpha loss')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
