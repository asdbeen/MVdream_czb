import torch
import sys

print(f'Python version: {sys.version}')
print(f'Torch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    if hasattr(torch.version, 'cuda_arch_list'):
        print(f'CUDA Arch List: {torch.version.cuda_arch_list}')
    else:
        print('CUDA Arch List: Not available')
