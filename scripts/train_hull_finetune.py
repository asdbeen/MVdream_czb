import os
import argparse
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as VT

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
        dataset = customizedDataset(args.dataset_root, args.meta_path, sample_side_views=args.num_views, source_image_res=args.size)
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    else:
        dataset = HullDataset(args.csv, img_size=args.size)
        dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

    # inject LoRA adapters and only train LoRA params
    from mvdream.ldm.modules.lora import inject_lora
    n_replaced = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
    print(f"Injected LoRA into {n_replaced} modules. Training adapters only.")

    # image encoder for hulls
    from mvdream.ldm.modules.encoders.modules import ImageEmbedder
    image_encoder = ImageEmbedder(device=device, img_size=args.size)
    image_encoder.to(device)
    image_encoder.eval()

    # mask_head will be created lazily to map latent channels -> alpha
    mask_head = None

    # convex-hull loss
    hull_criterion = ConvexHullLoss(threshold=0.1, use_dilation=True, kernel_size=5).to(device)

    # collect trainable params (LoRA params). mask_head params will be added when created
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    

    for epoch in range(args.epochs):
        for it, batch in enumerate(dl):
            # detect customized multi-view dataset by presence of render_image_groundtruth
            if 'render_image_groundtruth' in batch:
                # batch elements are collated: batch['render_image'] shape (bs, n_views, C, H, W)
                bs = len(batch['uid']) if 'uid' in batch else batch['render_image_groundtruth'].shape[0]
                total_loss = 0.0
                for bi in range(bs):
                    # gather per-sample fields
                    hulls = batch['render_image'][bi].to(device)  # (n_views, C, H, W)
                    gts = batch['render_image_groundtruth'][bi].to(device)  # (n_views, C, H, W)
                    poses = batch['poses'][bi].to(device)  # (n_views, 3, 4)
                    category = batch['category'][bi]

                    # encode hulls into embeddings
                    with torch.no_grad():
                        e_list = []
                        for k in range(hulls.shape[0]):
                            img = hulls[k].unsqueeze(0)
                            e = image_encoder.encode(img)
                            e_list.append(e)
                        hull_rep = torch.cat(e_list, dim=1)  # (1, n_views, D)

                    # build text context from category (treat category as text)
                    text_c = model.get_learned_conditioning([str(category)]).to(device)
                    context_cat = torch.cat([text_c, hull_rep], dim=1)
                    uc_text = model.get_learned_conditioning([""]).to(device)
                    uc_context_cat = torch.cat([uc_text, torch.zeros_like(hull_rep)], dim=1)

                    # for each target view supervise
                    sample_loss = 0.0
                    for j in range(gts.shape[0]):
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

                        model_out = model.apply_model(x_t, t, {'context': context_cat, 'camera': camera_tensor, 'num_frames': 1})
                        if model.parameterization == 'v':
                            pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                        else:
                            pred = model_out
                        l = F.mse_loss(pred, noise)

                        # hull-constrained alpha loss (compute when we can decode x0)
                        hull_loss_val = 0.0
                        if model.parameterization != 'v':
                            pred_x0 = model.predict_start_from_noise(x_t, t, pred)
                            # pred_x0 is latent (N, C, H, W) - apply mask_head on latent channels
                            if mask_head is None:
                                in_ch = pred_x0.shape[1]
                                mask_head = torch.nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0).to(device)
                                optimizer.add_param_group({'params': mask_head.parameters()})
                            pred_alpha = torch.sigmoid(mask_head(pred_x0))  # [N,1,H,W]
                            pred_alpha = pred_alpha.unsqueeze(1)  # [N, M=1, 1, H, W]
                            hull_img = hulls[j].unsqueeze(0).unsqueeze(0)  # [N=1, M=1, C, H, W]
                            hull_loss_val, _ = hull_criterion(pred_alpha, hull_img)
                        sample_loss += l + args.lambda_hull * hull_loss_val

                    sample_loss = sample_loss / float(gts.shape[0])
                    optimizer.zero_grad()
                    sample_loss.backward()
                    optimizer.step()
                    total_loss += sample_loss.item()

                if it % args.log_steps == 0:
                    avg_loss = total_loss / float(bs)
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

                # apply model -> predict noise
                model_out = model.apply_model(x_t, t, {'context': context_cat, 'y': y_rep, 'camera': batch.get('camera', None)})

                # predicted e_t
                if model.parameterization == 'v':
                    # convert to eps; use model.predict_eps_from_z_and_v
                    pred = model.predict_eps_from_z_and_v(x_t, t, model_out)
                else:
                    pred = model_out

                loss = F.mse_loss(pred, noise)

                # hull loss for CSV fallback (when we can decode)
                hull_loss_val = 0.0
                if model.parameterization != 'v':
                    pred_x0 = model.predict_start_from_noise(x_t, t, pred)
                    if mask_head is None:
                        in_ch = pred_x0.shape[1]
                        mask_head = torch.nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0).to(device)
                        optimizer.add_param_group({'params': mask_head.parameters()})
                    pred_alpha = torch.sigmoid(mask_head(pred_x0))  # [bs,1,H,W]
                    pred_alpha = pred_alpha.unsqueeze(1)  # [N, M=1, 1, H, W]
                    hull_img = hull.unsqueeze(1)  # [N, M=1, C, H, W]
                    hull_loss_val, _ = hull_criterion(pred_alpha, hull_img)

                loss = F.mse_loss(pred, noise) + args.lambda_hull * hull_loss_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % args.log_steps == 0:
                    print(f"epoch {epoch} iter {it} loss {loss.item():.6f}")

        # save checkpoint
        ckpt_path = os.path.join(args.out_dir, f'ckpt_epoch_{epoch}.pth')
        save_dict = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}
        if mask_head is not None:
            save_dict['mask_head_state'] = mask_head.state_dict()
        torch.save(save_dict, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
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
    parser.add_argument('--lambda_hull', type=float, default=0.5, help='weight for convex-hull alpha loss')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
