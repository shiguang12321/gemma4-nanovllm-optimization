import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module): #实现 SwiGLU 激活

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
