import traceback
import os
from mvdream.datasets.customized_dataset import customizedDataset

try:
    root_dir = 'customized_simple_dataset_tagVersion_simplified'
    meta_path = os.path.join(root_dir, 'train.txt')
    
    dataset = customizedDataset(
        root_dir=root_dir,
        meta_path=meta_path,
        sample_side_views=4,
        source_image_res=256
    )
    
    print(f'Dataset length: {len(dataset)}')
    if len(dataset) > 0:
        print(f'First uid repr: {repr(dataset.uids[0])}')
        sample = dataset[0]
        print(f"render_image shape: {sample['render_image'].shape}")
        print(f"render_image_groundtruth shape: {sample['render_image_groundtruth'].shape}")
        print(f"poses shape: {sample['poses'].shape}")
        print(f"source_image shape: {sample['source_image'].shape}")
    else:
        print('Dataset is empty.')

except Exception:
    traceback.print_exc()
