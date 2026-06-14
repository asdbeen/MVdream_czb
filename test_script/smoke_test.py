import traceback
import os
import sys

try:
    from mvdream.datasets.customized_dataset import customizedDataset
    
    root_dir = "customized_simple_dataset_tagVersion_simplified"
    meta_path = os.path.join(root_dir, "train.txt")
    
    dataset = customizedDataset(
        root_dir=root_dir,
        meta_path=meta_path,
        sample_side_views=4,
        source_image_res=256
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    print(f"First uid line from train.txt: {first_line}")
    
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    for key in ["render_image", "render_image_groundtruth", "poses", "source_image"]:
        val = sample.get(key)
        shape = val.shape if hasattr(val, "shape") else "None"
        print(f"{key} shape: {shape}")

except Exception:
    traceback.print_exc()
