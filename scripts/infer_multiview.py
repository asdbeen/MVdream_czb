
# [infer_multiview.py (line 1)](/home/chenzebin/MVdream_czb/scripts/infer_multiview.py:1) 是一个 quick test / smoke test：
# 只加载 base model：checkpoints/pretrained/sd-v2.1-base-4view.pt
# 不加载 LoRA
# 不加载 mask_enhanced
# 不用完整 DDIM 多步采样
# 只是拿 category = "car" + 数据集里 0001 的 4 个 camera pose，跑一次模型 forward，然后 decode 出 4 张图



import os
import torch
from PIL import Image
from mvdream.model_zoo import build_model
import numpy as np

# 配置参数
ckpt_path = "checkpoints/pretrained/sd-v2.1-base-4view.pt"  # 原始MVDream权重
out_dir = "checkpoints/infer_multiview_demo"
os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# 读取0001样本的四个视角的真实相机参数，与customizedDataset一致
def get_4_cardinal_camera_tensor_from_sample(root_dir, uid="0001"):
    import glob
    from mvdream.datasets.customized_dataset_dir1 import get_4_cardinal_views
    pose_dir = os.path.join(root_dir, uid, "pose")
    convex_dir = os.path.join(root_dir, uid, "rgb_convexhull")
    # 获取四个视角编号
    view_ids = get_4_cardinal_views(convex_dir)
    pose_list = []
    for v in view_ids:
        pose_path = os.path.join(pose_dir, f"{int(v):03d}.txt")
        if os.path.exists(pose_path):
            with open(pose_path, "r") as f:
                lines = f.readlines()
            pose_data = np.array(
                [list(map(float, line.split())) for line in lines],
                dtype=np.float32,
            ).reshape(4, 4)
            pose_list.append(torch.from_numpy(pose_data[:3, :]))
        else:
            pose_list.append(torch.eye(3, 4))
    camera_tensor = torch.stack(pose_list, dim=0)  # [4,3,4]
    # 若模型需要展平成[4,12]或[4,16]，可根据实际需求调整
    camera_tensor = camera_tensor.reshape(4, -1)
    return camera_tensor, view_ids


# 输入文本类别
category = "car"  # 可自定义
num_views = 4
# 数据集根目录和样本uid
dataset_root = "customized_simple_dataset_tagVersion_simplified"
uid = "0001"

# 加载模型
print("Loading MVDream model...")
model = build_model("sd-v2.1-base-4view", ckpt_path=ckpt_path)
model.to(device)
model.eval()

# 构造条件

with torch.no_grad():
    text_c = model.get_learned_conditioning([category]*num_views).to(device)
    camera_tensor, view_ids = get_4_cardinal_camera_tensor_from_sample(dataset_root, uid)
    camera_tensor = camera_tensor.to(device)
    context_cat = text_c  # 不拼接hull/pose，仅文本
    # 随机采样初始噪声
    z_shape = (num_views, model.channels, model.image_size, model.image_size)
    z = torch.randn(z_shape, device=device)
    t = torch.full((num_views,), model.num_timesteps-1, device=device, dtype=torch.long)  # 采样终点
    # 采样过程（简化，实际可用DDIM/PLMS采样器）
    # 这里只做一次反向推理演示
    model_out = model.apply_model(
        z, t,
        {
            'context': context_cat,
            'camera': camera_tensor,
            'num_frames': num_views,
        }
    )
    if model.parameterization == 'v':
        pred_x0 = model.predict_start_from_z_and_v(z, t, model_out)
    else:
        pred_x0 = model.predict_start_from_noise(z, t, model_out)
    decoded_imgs = model.decode_first_stage(pred_x0)
    decoded_imgs = torch.clamp((decoded_imgs + 1.0) / 2.0, 0.0, 1.0)
    for i in range(num_views):
        arr = (decoded_imgs[i].detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, f"infer_view_{i}_viewid_{view_ids[i]}.png"))
        print(f"Saved: {os.path.join(out_dir, f'infer_view_{i}_viewid_{view_ids[i]}.png')}")

print("Done. 请检查 infer_multiview_demo 目录下的4张图片。")
