import numpy as np
from PIL import Image

# 路径
predalpha_path = "checkpoints/pred_images/epoch1_iter0_predalpha_0.png"
hull_mask_path = "checkpoints/pred_images/epoch1_iter0_gt_0.png"

# 读取图片并归一化到0~1
pred_alpha = np.array(Image.open(predalpha_path)).astype(np.float32) / 255.0
hull_mask = np.array(Image.open(hull_mask_path)).astype(np.float32) / 255.0

# 如果是彩色，取单通道
if pred_alpha.ndim == 3:
    pred_alpha = pred_alpha[..., 0]
if hull_mask.ndim == 3:
    hull_mask = hull_mask[..., 0]

# 计算 outside_loss
outside = pred_alpha * (1.0 - hull_mask)
outside_loss = np.mean(outside)
print("outside_loss =", outside_loss)

# 生成超出区域标红的图片
# 原图用灰度显示 pred_alpha，超出区域叠加红色
h, w = pred_alpha.shape
rgb = np.stack([pred_alpha, pred_alpha, pred_alpha], axis=-1)
# 红色叠加
mask = outside > 1e-3  # 只要有超出就标红
rgb[mask] = [1.0, 0.0, 0.0]
# 保存
Image.fromarray((rgb * 255).astype(np.uint8)).save("checkpoints/pred_images/epoch1_iter0_predalpha_0_outside_red.png")
print("已保存标红图片: checkpoints/pred_images/epoch1_iter0_predalpha_0_outside_red.png")