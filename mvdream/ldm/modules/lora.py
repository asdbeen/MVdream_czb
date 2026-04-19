import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank adapter (LoRA).

    The original linear's weights are frozen; the adapter adds W + alpha/r * B A x.
    """
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = r
        self.alpha = alpha

        # keep original weight and bias (frozen)
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias
        if self.weight is not None:
            self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if r > 0:
            # A: (r, in), B: (out, r)
            param_device = self.weight.device if self.weight is not None else None
            param_dtype = self.weight.dtype if self.weight is not None else None
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=param_device, dtype=param_dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=param_device, dtype=param_dtype))
            # initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = self.alpha / max(1, self.r)
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor):
        # x: (..., in_features) or (batch, in_features) or (batch, seq, in_features)
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # adapter: x -> (.., r) via A, then -> (.., out) via B
            lora_A = self.lora_A
            lora_B = self.lora_B
            if lora_A.device != x.device or lora_A.dtype != x.dtype:
                lora_A = lora_A.to(device=x.device, dtype=x.dtype)
            if lora_B.device != x.device or lora_B.dtype != x.dtype:
                lora_B = lora_B.to(device=x.device, dtype=x.dtype)
            adapter = F.linear(x, lora_A)
            adapter = F.linear(adapter, lora_B)
            return out + self.scaling * adapter
        return out


def inject_lora(module: nn.Module, r: int = 4, alpha: float = 1.0, target_modules=(nn.Linear,)):
    """Recursively replace target linear modules with LoRA-wrapped versions.

    Returns number of replaced modules.
    """
    count = 0
    for name, child in list(module.named_children()):
        replaced = False
        if isinstance(child, target_modules):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
            replaced = True
        else:
            c = inject_lora(child, r=r, alpha=alpha, target_modules=target_modules)
            count += c
    return count
