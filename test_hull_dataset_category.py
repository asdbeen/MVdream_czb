import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import csv


from mvdream.datasets.customized_dataset import customizedDataset


if __name__ == '__main__':
    dataset = customizedDataset(
        "customized_simple_dataset_tagVersion_simplified",
        "customized_simple_dataset_tagVersion_simplified/train.txt",
        sample_side_views=4,
        source_image_res=256,
        use_value_json=False,
    )
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i} category:", sample['category'])