"""Fixed-capacity, device-routed MoE for explicit CUDA Graph use.

The launch shapes depend on tensor shapes, never the number of selected experts.
Routes are sorted on the device and each expert's rows are padded to BLOCK_M.
For R routes and E experts, R + E * (BLOCK_M - 1) bounds the padded rows; no
E-by-R activation or copied-expert-weight tensor is materialized. Empty expert
tiles are skipped inside the kernels. This path needs CUDA numerical/capture
validation before its performance or deployment coverage can be claimed.
"""

import torch
import triton
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def _scatter_sorted_routes(
    sorted_order_ptr,
    flat_experts_ptr,
    offsets_ptr,
    padded_offsets_ptr,
    padded_route_ids_ptr,
    tile_experts_ptr,
    NUM_ROUTES: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rank = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rank < NUM_ROUTES
    route = tl.load(sorted_order_ptr + rank, mask=valid, other=0).to(tl.int32)
    expert = tl.load(flat_experts_ptr + route, mask=valid, other=0).to(tl.int32)
    start = tl.load(offsets_ptr + expert)
    padded_start = tl.load(padded_offsets_ptr + expert)
    local_rank = rank - start
    padded_rank = padded_start + local_rank
    tl.store(padded_route_ids_ptr + padded_rank, route, mask=valid)
    # Exactly one selected route owns the first row of every occupied tile.
    tl.store(
        tile_experts_ptr + padded_rank // TILE_M,
        expert,
        mask=valid & (local_rank % TILE_M == 0),
    )


@triton.jit
def _grouped_gate_up(
    hidden_ptr,
    weights_ptr,
    padded_route_ids_ptr,
    tile_experts_ptr,
    padded_ends_ptr,
    intermediate_ptr,
    hidden_stride_t: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    weight_stride_e: tl.constexpr,
    weight_stride_m: tl.constexpr,
    weight_stride_h: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile = tl.program_id(0)
    total_padded = tl.load(padded_ends_ptr + NUM_EXPERTS - 1)
    # This is a device-side branch. The host grid remains fixed during replay.
    if tile * TILE_M < total_padded:
        expert = tl.load(tile_experts_ptr + tile)
        rows = tile * TILE_M + tl.arange(0, TILE_M)
        route = tl.load(padded_route_ids_ptr + rows, mask=rows < CAPACITY, other=-1)
        valid_route = route >= 0
        tokens = tl.maximum(route, 0) // TOP_K
        cols = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
        ks = tl.arange(0, TILE_K)
        acc = tl.zeros((TILE_M, TILE_N), tl.float32)
        for start in range(0, HIDDEN_DIM, TILE_K):
            k = start + ks
            x = tl.load(
                hidden_ptr + tokens[:, None] * hidden_stride_t + k[None, :] * hidden_stride_h,
                mask=valid_route[:, None] & (k[None, :] < HIDDEN_DIM),
                other=0.0,
            )
            weight = tl.load(
                weights_ptr + expert * weight_stride_e
                + k[:, None] * weight_stride_h + cols[None, :] * weight_stride_m,
                mask=(k[:, None] < HIDDEN_DIM) & (cols[None, :] < 2 * INTERMEDIATE_DIM),
                other=0.0,
            )
            acc = tl.dot(x, weight, acc)
        tl.store(
            intermediate_ptr + rows[:, None] * (2 * INTERMEDIATE_DIM) + cols[None, :],
            acc,
            mask=valid_route[:, None] & (cols[None, :] < 2 * INTERMEDIATE_DIM),
        )


@triton.jit
def _grouped_down(
    intermediate_ptr,
    weights_ptr,
    route_weights_ptr,
    padded_route_ids_ptr,
    tile_experts_ptr,
    padded_ends_ptr,
    route_output_ptr,
    weight_stride_e: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_i: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile = tl.program_id(0)
    total_padded = tl.load(padded_ends_ptr + NUM_EXPERTS - 1)
    if tile * TILE_M < total_padded:
        expert = tl.load(tile_experts_ptr + tile)
        rows = tile * TILE_M + tl.arange(0, TILE_M)
        route = tl.load(padded_route_ids_ptr + rows, mask=rows < CAPACITY, other=-1)
        valid_route = route >= 0
        safe_route = tl.maximum(route, 0)
        cols = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
        ks = tl.arange(0, TILE_K)
        acc = tl.zeros((TILE_M, TILE_N), tl.float32)
        for start in range(0, INTERMEDIATE_DIM, TILE_K):
            k = start + ks
            valid = valid_route[:, None] & (k[None, :] < INTERMEDIATE_DIM)
            gate = tl.load(
                intermediate_ptr + rows[:, None] * (2 * INTERMEDIATE_DIM) + k[None, :],
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                intermediate_ptr + rows[:, None] * (2 * INTERMEDIATE_DIM)
                + INTERMEDIATE_DIM + k[None, :],
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            gelu = 0.5 * gate * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))
            # Match the eager activation's two low-precision rounding points:
            # F.gelu(gate), followed by multiplication with up.
            act_dtype = intermediate_ptr.dtype.element_ty
            gelu = gelu.to(act_dtype).to(tl.float32)
            activation = (gelu * up).to(act_dtype)
            weight = tl.load(
                weights_ptr + expert * weight_stride_e
                + k[:, None] * weight_stride_i + cols[None, :] * weight_stride_h,
                mask=(k[:, None] < INTERMEDIATE_DIM) & (cols[None, :] < HIDDEN_DIM),
                other=0.0,
            )
            acc = tl.dot(activation, weight, acc)
        # F.linear returns the input dtype before multiplication by router weights.
        acc = acc.to(intermediate_ptr.dtype.element_ty).to(tl.float32)
        route_weight = tl.load(route_weights_ptr + safe_route, mask=valid_route, other=0.0).to(tl.float32)
        tl.store(
            route_output_ptr + safe_route[:, None] * HIDDEN_DIM + cols[None, :],
            acc * route_weight[:, None],
            mask=valid_route[:, None] & (cols[None, :] < HIDDEN_DIM),
        )


@triton.jit
def _reduce_routes(
    route_output_ptr,
    output_ptr,
    HIDDEN_DIM: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_TOP_K: tl.constexpr,
):
    token = tl.program_id(0)
    hidden = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    top_k = tl.arange(0, BLOCK_TOP_K)
    values = tl.load(
        route_output_ptr + (token * TOP_K + top_k[:, None]) * HIDDEN_DIM + hidden[None, :],
        mask=(top_k[:, None] < TOP_K) & (hidden[None, :] < HIDDEN_DIM),
        other=0.0,
    )
    out = tl.sum(values, axis=0)
    tl.store(output_ptr + token * HIDDEN_DIM + hidden, out, mask=hidden < HIDDEN_DIM)


def grouped_moe(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj):
    """Compute routed MoE with fixed-shape CUDA workspaces.

    Shapes are [N,H], [N,K], [N,K], [E,2I,H], [E,H,I]. Weights and hidden
    states must share CUDA device and fp16/bf16 dtype; routes must be valid
    expert indices from the router. Experts receiving zero routes need no
    special host handling. N=0 returns an empty [0,H] output.

    Weighted expert outputs are accumulated in fp32 and cast once per token.
    This intentionally differs from repeated low-precision index_add_ in the
    legacy expert loop, so numerical comparison uses dtype-aware tolerances.
    """
    if hidden_states.ndim != 2 or top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        raise ValueError("hidden states and routes must be rank two")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert weights must be rank three")
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, _, intermediate_dim = down_proj.shape
    if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != num_tokens:
        raise ValueError("route shapes must agree with hidden states")
    top_k = top_k_index.shape[1]
    if num_experts < 1 or top_k < 1 or top_k > num_experts or hidden_dim < 1 or intermediate_dim < 1:
        raise ValueError("expert, hidden, intermediate and top-k dimensions must be positive; top-k <= experts")
    if gate_up_proj.shape != (num_experts, 2 * intermediate_dim, hidden_dim):
        raise ValueError("gate/up expert weight shape does not match down weights")
    if down_proj.shape[1] != hidden_dim:
        raise ValueError("down expert hidden dimension does not match hidden states")
    if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("grouped_moe requires CUDA fp16 or bf16 hidden states")
    tensors = (top_k_index, top_k_weights, gate_up_proj, down_proj)
    if any(t.device != hidden_states.device for t in tensors):
        raise ValueError("all grouped_moe tensors must be on the same CUDA device")
    if gate_up_proj.dtype != hidden_states.dtype or down_proj.dtype != hidden_states.dtype:
        raise ValueError("expert weights and hidden states must have the same dtype")
    if top_k_index.dtype not in (torch.int32, torch.int64) or not top_k_weights.is_floating_point():
        raise ValueError("routes must be integer indices with floating-point weights")
    return _grouped_moe_impl(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj)


def _grouped_moe_impl(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj):
    """Shared execution body; CPU tests invoke this with TRITON_INTERPRET=1.

    Public callers go through grouped_moe's CUDA/dtype/shape validation. The
    private interpreter entry executes the same routing and kernel launches.
    It does not provide CPU inference support in the public model interface.
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, _, intermediate_dim = down_proj.shape
    top_k = top_k_index.shape[1]
    if num_tokens == 0:
        return torch.empty_like(hidden_states)

    num_routes = num_tokens * top_k
    capacity = num_routes + num_experts * (BLOCK_M - 1)
    num_tiles = triton.cdiv(capacity, BLOCK_M)
    flat_experts = top_k_index.reshape(-1).to(torch.int64)
    flat_route_weights = top_k_weights.reshape(-1)
    sorted_order = torch.argsort(flat_experts, stable=True)
    counts = torch.zeros(num_experts, device=hidden_states.device, dtype=torch.int32)
    counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int32))
    ends = torch.cumsum(counts, dim=0, dtype=torch.int32)
    offsets = ends - counts
    padded_counts = torch.div(counts + BLOCK_M - 1, BLOCK_M, rounding_mode="floor") * BLOCK_M
    padded_ends = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    padded_offsets = padded_ends - padded_counts
    padded_route_ids = torch.full((capacity,), -1, device=hidden_states.device, dtype=torch.int32)
    tile_experts = torch.full((num_tiles,), -1, device=hidden_states.device, dtype=torch.int32)
    intermediate = torch.empty((capacity, 2 * intermediate_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    route_output = torch.empty((num_routes, hidden_dim), device=hidden_states.device, dtype=torch.float32)
    output = torch.empty((num_tokens, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)

    _scatter_sorted_routes[(triton.cdiv(num_routes, 256),)](
        sorted_order, flat_experts, offsets, padded_offsets, padded_route_ids, tile_experts,
        NUM_ROUTES=num_routes, TILE_M=BLOCK_M, BLOCK=256,
    )
    _grouped_gate_up[(num_tiles, triton.cdiv(2 * intermediate_dim, BLOCK_N))](
        hidden_states, gate_up_proj, padded_route_ids, tile_experts, padded_ends, intermediate,
        *hidden_states.stride(), *gate_up_proj.stride(),
        NUM_EXPERTS=num_experts, TOP_K=top_k, HIDDEN_DIM=hidden_dim,
        INTERMEDIATE_DIM=intermediate_dim, CAPACITY=capacity,
        TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=BLOCK_K, num_warps=4,
    )
    _grouped_down[(num_tiles, triton.cdiv(hidden_dim, BLOCK_N))](
        intermediate, down_proj, flat_route_weights, padded_route_ids, tile_experts, padded_ends, route_output,
        *down_proj.stride(), NUM_EXPERTS=num_experts, HIDDEN_DIM=hidden_dim,
        INTERMEDIATE_DIM=intermediate_dim, CAPACITY=capacity,
        TILE_M=BLOCK_M, TILE_N=BLOCK_N, TILE_K=BLOCK_K, num_warps=4,
    )
    _reduce_routes[(num_tokens, triton.cdiv(hidden_dim, 128))](
        route_output, output, HIDDEN_DIM=hidden_dim, TOP_K=top_k,
        BLOCK_H=128, BLOCK_TOP_K=triton.next_power_of_2(top_k), num_warps=4,
    )
    return output
