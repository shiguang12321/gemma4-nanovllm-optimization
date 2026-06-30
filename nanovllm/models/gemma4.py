import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.rotary_embedding import apply_rotary_emb
from nanovllm.utils.context import get_context

TRITON_MOE_GATE_BLOCK_M = 64
TRITON_MOE_GATE_BLOCK_K = 128
TRITON_MOE_DOWN_BLOCK_H = 32
TRITON_MOE_DOWN_BLOCK_I = 128
def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")
@triton.jit
def _gelu_tanh_kernel_expr(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    return 0.5 * x * (1.0 + tanh_inner)
@triton.jit
def gemma4_router_topk_kernel(
    logits_ptr,
    per_expert_scale_ptr,
    weights_ptr,
    index_ptr,
    logits_stride_t: tl.constexpr,
    logits_stride_e: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_k: tl.constexpr,
    index_stride_t: tl.constexpr,
    index_stride_k: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    logits = tl.load(
        logits_ptr + token_id * logits_stride_t + expert_offsets * logits_stride_e,
        mask=expert_offsets < NUM_EXPERTS,
        other=-float("inf"),
    ).to(tl.float32)
    logits = tl.where(logits == logits, logits, -float("inf"))
    top_offsets = tl.arange(0, TOP_K)
    top_values = tl.full((TOP_K,), -float("inf"), tl.float32)
    top_indices = tl.zeros((TOP_K,), tl.int32)
    for k in tl.static_range(0, TOP_K):
        value = tl.max(logits, axis=0)
        candidates = tl.where(logits == value, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidates, axis=0)
        top_values = tl.where(top_offsets == k, value, top_values)
        top_indices = tl.where(top_offsets == k, expert_id, top_indices)
        logits = tl.where(expert_offsets == expert_id, -float("inf"), logits)
    top_max = tl.max(top_values, axis=0)
    exp_values = tl.exp(top_values - top_max)
    denom = tl.sum(exp_values, axis=0)
    scales = tl.load(per_expert_scale_ptr + top_indices).to(tl.float32)
    weights = exp_values / denom * scales
    tl.store(
        weights_ptr + token_id * weights_stride_t + top_offsets * weights_stride_k,
        weights,
    )
    tl.store(
        index_ptr + token_id * index_stride_t + top_offsets * index_stride_k,
        top_indices,
    )
def gemma4_router_topk_triton(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
):
    num_tokens, num_experts = logits.shape
    top_k_weights = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.float32)
    top_k_index = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.int32)
    block_e = triton.next_power_of_2(num_experts)
    gemma4_router_topk_kernel[(num_tokens,)](
        logits,
        per_expert_scale,
        top_k_weights,
        top_k_index,
        logits.stride(0),
        logits.stride(1),
        top_k_weights.stride(0),
        top_k_weights.stride(1),
        top_k_index.stride(0),
        top_k_index.stride(1),
        num_experts,
        top_k,
        block_e,
        num_warps=4,
    )
    return top_k_weights, top_k_index
@triton.jit
def gemma4_moe_gate_up_kernel(
    hidden_ptr,
    topk_index_ptr,
    gate_up_ptr,
    tmp_ptr,
    hidden_stride_t: tl.constexpr,
    hidden_stride_h: tl.constexpr,
    topk_stride_t: tl.constexpr,
    topk_stride_k: tl.constexpr,
    gate_up_stride_e: tl.constexpr,
    gate_up_stride_m: tl.constexpr,
    gate_up_stride_h: tl.constexpr,
    tmp_stride_r: tl.constexpr,
    tmp_stride_m: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_id = tl.program_id(0)
    m_block = tl.program_id(1)
    token_id = route_id // TOP_K
    topk_pos = route_id - token_id * TOP_K
    expert_id = tl.load(topk_index_ptr + token_id * topk_stride_t + topk_pos * topk_stride_k)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), tl.float32)
    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + token_id * hidden_stride_t + k * hidden_stride_h,
            mask=k < HIDDEN_DIM,
            other=0.0,
        )
        weight = tl.load(
            gate_up_ptr
            + expert_id * gate_up_stride_e
            + offs_m[:, None] * gate_up_stride_m
            + k[None, :] * gate_up_stride_h,
            mask=(offs_m[:, None] < 2 * INTERMEDIATE_DIM) & (k[None, :] < HIDDEN_DIM),
            other=0.0,
        )
        acc += tl.sum(weight.to(tl.float32) * hidden[None, :].to(tl.float32), axis=1)
    tl.store(
        tmp_ptr + route_id * tmp_stride_r + offs_m * tmp_stride_m,
        acc,
        mask=offs_m < 2 * INTERMEDIATE_DIM,
    )
@triton.jit
def gemma4_moe_down_kernel(
    tmp_ptr,
    topk_index_ptr,
    topk_weights_ptr,
    down_ptr,
    out_ptr,
    tmp_stride_r: tl.constexpr,
    tmp_stride_m: tl.constexpr,
    topk_index_stride_t: tl.constexpr,
    topk_index_stride_k: tl.constexpr,
    topk_weights_stride_t: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    down_stride_e: tl.constexpr,
    down_stride_h: tl.constexpr,
    down_stride_i: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_block = tl.program_id(1)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = tl.arange(0, BLOCK_I)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for topk_pos in range(0, TOP_K):
        route_id = token_id * TOP_K + topk_pos
        expert_id = tl.load(topk_index_ptr + token_id * topk_index_stride_t + topk_pos * topk_index_stride_k)
        route_weight = tl.load(
            topk_weights_ptr + token_id * topk_weights_stride_t + topk_pos * topk_weights_stride_k
        ).to(tl.float32)
        for i_start in range(0, INTERMEDIATE_DIM, BLOCK_I):
            i = i_start + offs_i
            gate = tl.load(
                tmp_ptr + route_id * tmp_stride_r + i * tmp_stride_m,
                mask=i < INTERMEDIATE_DIM,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                tmp_ptr + route_id * tmp_stride_r + (INTERMEDIATE_DIM + i) * tmp_stride_m,
                mask=i < INTERMEDIATE_DIM,
                other=0.0,
            ).to(tl.float32)
            act = _gelu_tanh_kernel_expr(gate) * up
            weight = tl.load(
                down_ptr
                + expert_id * down_stride_e
                + offs_h[:, None] * down_stride_h
                + i[None, :] * down_stride_i,
                mask=(offs_h[:, None] < HIDDEN_DIM) & (i[None, :] < INTERMEDIATE_DIM),
                other=0.0,
            )
            acc += tl.sum(weight.to(tl.float32) * act[None, :], axis=1) * route_weight
    tl.store(
        out_ptr + token_id * out_stride_t + offs_h * out_stride_h,
        acc,
        mask=offs_h < HIDDEN_DIM,
    )
def gemma4_moe_decode_triton(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
):
    num_tokens, top_k = top_k_index.shape
    hidden_dim = hidden_states.size(1)
    intermediate_dim = down_proj.size(2)
    num_routes = num_tokens * top_k
    tmp = torch.empty(
        (num_routes, 2 * intermediate_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    out = torch.empty_like(hidden_states)
    gate_block_m = TRITON_MOE_GATE_BLOCK_M
    gate_block_k = TRITON_MOE_GATE_BLOCK_K
    down_block_h = TRITON_MOE_DOWN_BLOCK_H
    down_block_i = TRITON_MOE_DOWN_BLOCK_I
    gemma4_moe_gate_up_kernel[(num_routes, triton.cdiv(2 * intermediate_dim, gate_block_m))](
        hidden_states,
        top_k_index,
        gate_up_proj,
        tmp,
        hidden_states.stride(0),
        hidden_states.stride(1),
        top_k_index.stride(0),
        top_k_index.stride(1),
        gate_up_proj.stride(0),
        gate_up_proj.stride(1),
        gate_up_proj.stride(2),
        tmp.stride(0),
        tmp.stride(1),
        top_k,
        hidden_dim,
        intermediate_dim,
        gate_block_m,
        gate_block_k,
        num_warps=4,
    )
    gemma4_moe_down_kernel[(num_tokens, triton.cdiv(hidden_dim, down_block_h))](
        tmp,
        top_k_index,
        top_k_weights,
        down_proj,
        out,
        tmp.stride(0),
        tmp.stride(1),
        top_k_index.stride(0),
        top_k_index.stride(1),
        top_k_weights.stride(0),
        top_k_weights.stride(1),
        down_proj.stride(0),
        down_proj.stride(1),
        down_proj.stride(2),
        out.stride(0),
        out.stride(1),
        top_k,
        hidden_dim,
        intermediate_dim,
        down_block_h,
        down_block_i,
        num_warps=4,
    )
    return out
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.pow(y.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(x.dtype)
class Gemma4ScaledEmbedding(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__(num_embeddings, embedding_dim)
        self.padding_idx = padding_idx
        self.register_buffer("embed_scale", torch.tensor(embedding_dim ** 0.5), persistent=False)
    def forward(self, x: torch.Tensor):
        return super().forward(x) * self.embed_scale.to(self.weight.dtype)
class Gemma4RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_position = config.max_position_embeddings
        layer_types = set(config.layer_types)
        for layer_type in layer_types:
            head_dim = config.global_head_dim if layer_type == "full_attention" and config.global_head_dim else config.head_dim
            params = config.rope_parameters[layer_type]
            base = params["rope_theta"]
            if params["rope_type"] == "proportional":
                rope_angles = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
                )
                nope_angles = head_dim // 2 - rope_angles
                if nope_angles > 0:
                    inv_freq = torch.cat((inv_freq, torch.zeros(nope_angles, dtype=torch.float)), dim=0)
            else:
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
            positions = torch.arange(self.max_position, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            self.register_buffer(f"{layer_type}_cos", freqs.cos(), persistent=False)
            self.register_buffer(f"{layer_type}_sin", freqs.sin(), persistent=False)
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, layer_type: str):
        cos = getattr(self, f"{layer_type}_cos")[positions].unsqueeze(1)
        sin = getattr(self, f"{layer_type}_sin")[positions].unsqueeze(1)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
class Gemma4Attention(nn.Module):
    def __init__(self, config, layer_idx: int, rotary_emb: Gemma4RotaryEmbedding):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.num_heads = config.num_attention_heads
        self.use_alt = config.attention_k_eq_v and not self.is_sliding
        self.num_kv_heads = config.num_global_key_value_heads if self.use_alt else config.num_key_value_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = config.num_key_value_heads
        self.rotary_emb = rotary_emb
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = None if self.use_alt else nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        window = config.sliding_window if self.is_sliding else None
        self.attn = Attention(self.num_heads, self.head_dim, 1.0, self.num_kv_heads, window)
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = k if self.v_proj is None else self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k, self.layer_type)
        v = self.v_norm(v)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))
class Gemma4MLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))
class Gemma4Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size
        self.hidden_dim = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        context = get_context()
        if not context.is_prefill and hidden_states.size(0) <= 16:
            return self.forward_decode(hidden_states, top_k_index, top_k_weights)
        final_hidden_states = torch.zeros_like(hidden_states)
        top_k = top_k_index.size(1)
        with torch.no_grad():
            flat_experts = top_k_index.flatten()
            sorted_experts, order = torch.sort(flat_experts)
            expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
            token_idx = torch.div(order, top_k, rounding_mode="floor")
            top_k_pos = order - token_idx * top_k
        offset = 0
        for expert_idx, count in zip(expert_ids, counts):
            end = offset + count
            tokens = token_idx[offset:end]
            positions = top_k_pos[offset:end]
            offset = end
            expert_idx = expert_idx.item()
            current_state = hidden_states[tokens]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = gelu_tanh(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[tokens, positions, None]
            final_hidden_states.index_add_(0, tokens, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states
    def forward_decode(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        if hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16):
            return gemma4_moe_decode_triton(
                hidden_states,
                top_k_index,
                top_k_weights,
                self.gate_up_proj,
                self.down_proj,
            )
        num_tokens, top_k = top_k_index.shape
        final_hidden_states = torch.zeros_like(hidden_states)
        for pos in range(top_k):
            expert_ids = top_k_index[:, pos]
            gate_up_weight = torch.index_select(self.gate_up_proj, 0, expert_ids)
            gate_up = torch.bmm(gate_up_weight, hidden_states.unsqueeze(-1)).squeeze(-1)
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = gelu_tanh(gate) * up
            down_weight = torch.index_select(self.down_proj, 0, expert_ids)
            current_hidden_states = torch.bmm(down_weight, current_hidden_states.unsqueeze(-1)).squeeze(-1)
            current_hidden_states = current_hidden_states * top_k_weights[:, pos, None]
            final_hidden_states += current_hidden_states.to(final_hidden_states.dtype)
        return final_hidden_states
class Gemma4Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        self.scalar_root_size = config.hidden_size ** -0.5
    def forward(self, hidden_states: torch.Tensor):
        context = get_context()
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        logits = self.proj(hidden_states)
        if (
            not context.is_prefill
            and logits.is_cuda
            and logits.size(1) <= 256
            and self.config.top_k_experts <= 16
        ):
            return gemma4_router_topk_triton(logits, self.per_expert_scale, self.config.top_k_experts)
        router_probabilities = F.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index
class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, rotary_emb: Gemma4RotaryEmbedding):
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx, rotary_emb)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4Router(config)
            self.experts = Gemma4Experts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            top_k_weights, top_k_index = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar
class Gemma4Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Gemma4ScaledEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = Gemma4RotaryEmbedding(config)
        self.layers = nn.ModuleList([Gemma4DecoderLayer(config, i, self.rotary_emb) for i in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)
class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
        return logits
