"""Gemma elementwise reference functions and opt-in compiled CUDA dispatch.

Keep checkpoint parameters in the original modules.  These small functions may
be compiled independently; neither attention nor the model is compiled here.
The runner must warm every graph shape before entering CUDA graph capture.
Explicit casts retain the reference model's low-precision operation boundaries;
compiled floating-point reductions still require tolerance-based GPU validation.
"""

from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    value = x.float()
    value = value * torch.pow(value.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        value = value * weight.float()
    return value.to(x.dtype)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)


def qkv_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # value can alias the RAW key projection in attention_k_eq_v layers.  All
    # operations are out of place so value never sees normalized/rotated keys.
    query = rotary(rms_norm(query, query_weight, eps), cos, sin)
    key = rotary(rms_norm(key, key_weight, eps), cos, sin)
    value = rms_norm(value, None, eps)
    return query, key, value


def gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    activated = F.gelu(gate, approximate="tanh").to(gate.dtype)
    return activated * up


def post_norm_add_next_norm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    next_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = (residual + rms_norm(branch, post_weight, eps)).to(residual.dtype)
    return rms_norm(residual, next_weight, eps), residual


def post_norm_add_scalar(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden = (residual + rms_norm(branch, post_weight, eps)).to(residual.dtype)
    return hidden * layer_scalar


def moe_post_norm_add_scalar(
    dense_branch: torch.Tensor,
    expert_branch: torch.Tensor,
    residual: torch.Tensor,
    dense_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    post_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dense_branch = rms_norm(dense_branch, dense_weight, eps)
    expert_branch = rms_norm(expert_branch, expert_weight, eps)
    branch = (dense_branch + expert_branch).to(dense_branch.dtype)
    return post_norm_add_scalar(branch, residual, post_weight, layer_scalar, eps)


def router_norm_scale(
    hidden: torch.Tensor, scale: torch.Tensor, scalar_root_size: float, eps: float,
) -> torch.Tensor:
    hidden = rms_norm(hidden, None, eps)
    # Preserve the two multiplications and the reference dtype promotion.
    return (hidden * scale) * scalar_root_size


@lru_cache(maxsize=None)
def _compiled(function: Callable) -> Callable:
    # Dynamic leading dimensions reduce bucket recompiles. Batch size 1, head
    # widths, and dtype may still legitimately specialize: warm all graph buckets.
    # Outer CUDAGraph ownership belongs to ModelRunner, not torch.compile.
    return torch.compile(
        function, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False},
    )


def run_elementwise(function: Callable, *args, enabled: bool = False):
    """CPU/default always uses the readable reference implementation."""
    if enabled and args[0].is_cuda:
        return _compiled(function)(*args)
    return function(*args)
