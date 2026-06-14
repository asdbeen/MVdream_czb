import sys
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import numpy as np

def sobel_edge(tensor):
    # tensor: (N, 1, H, W)
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=tensor.dtype, device=tensor.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=tensor.dtype, device=tensor.device).view(1,1,3,3)
    edge_x = F.conv2d(tensor, sobel_x, padding=1)
    edge_y = F.conv2d(tensor, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    return edge

def edge_align_loss(img1, img2):
    # img1, img2: (1, 1, H, W), float32, [0,1]
    edge1 = sobel_edge(img1)
    edge2 = sobel_edge(img2)
    return F.l1_loss(edge1, edge2).item()

def load_gray_tensor(path, size=256):
    img = Image.open(path).convert('L')
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    t = tf(img).unsqueeze(0)  # (1, 1, H, W)
    return t

def main():
    if len(sys.argv) != 3:
        print("Usage: python edge_loss_test.py img1_path img2_path")
        return
    img1_path, img2_path = sys.argv[1], sys.argv[2]
    t1 = load_gray_tensor(img1_path)
    t2 = load_gray_tensor(img2_path)
    loss = edge_align_loss(t1, t2)
    print(f"Edge L1 loss: {loss}")

if __name__ == "__main__":
    main()
