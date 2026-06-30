import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


TRITON_PAGED_BLOCK_N_256 = 32
TRITON_PAGED_BLOCK_N_512 = 16
TRITON_PAGED_SPLIT_CHUNK = 128


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def repeat_kv(x: torch.Tensor, num_heads: int):
    if x.size(1) == num_heads:
        return x
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


@triton.jit
def paged_decode_attention_splitk_stage1_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    block_table_stride: tl.constexpr,
    partial_stride_b: tl.constexpr,
    partial_stride_h: tl.constexpr,
    partial_stride_c: tl.constexpr,
    partial_stride_d: tl.constexpr,
    partial_meta_stride_b: tl.constexpr,
    partial_meta_stride_h: tl.constexpr,
    partial_meta_stride_c: tl.constexpr,
    scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
    MAX_LEN: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    chunk_id = tl.program_id(2)
    kv_head = head_id // (NUM_HEADS // NUM_KV_HEADS)

    context_len = tl.load(context_lens_ptr + batch_id)
    if SLIDING_WINDOW > 0:
        attn_len = tl.minimum(context_len, SLIDING_WINDOW)
        start_pos = context_len - attn_len
    else:
        attn_len = context_len
        start_pos = 0

    chunk_start = chunk_id * CHUNK_N
    if chunk_start >= attn_len:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + batch_id * q_stride_b + head_id * q_stride_h + offs_d * q_stride_d).to(tl.float32)
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    for local_base in range(0, CHUNK_N, BLOCK_N):
        offs_n = chunk_start + local_base + tl.arange(0, BLOCK_N)
        abs_pos = start_pos + offs_n
        block_idx = abs_pos // BLOCK_SIZE
        block_off = abs_pos - block_idx * BLOCK_SIZE
        valid = (offs_n < attn_len) & (offs_n < MAX_LEN)
        block_ids = tl.load(
            block_tables_ptr + batch_id * block_table_stride + block_idx,
            mask=valid,
            other=0,
        )
        k = tl.load(
            k_cache_ptr
            + block_ids[:, None] * cache_stride_block
            + block_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1) * scale
        qk = tl.where(valid, qk, -float("inf"))
        m_chunk = tl.max(qk, axis=0)
        has_values = tl.sum(tl.where(valid, 1, 0), axis=0) > 0
        m_new = tl.maximum(m_i, m_chunk)
        m_new_safe = tl.where(has_values | (l_i > 0.0), m_new, 0.0)
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_new_safe), 0.0)
        p = tl.where(valid, tl.exp(qk - m_new_safe), 0.0)
        v = tl.load(
            v_cache_ptr
            + block_ids[:, None] * cache_stride_block
            + block_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * old_scale + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * old_scale + tl.sum(p, axis=0)
        m_i = tl.where(has_values | (l_i > 0.0), m_new, m_i)

    tl.store(
        partial_out_ptr
        + batch_id * partial_stride_b
        + head_id * partial_stride_h
        + chunk_id * partial_stride_c
        + offs_d * partial_stride_d,
        acc,
    )
    tl.store(
        partial_m_ptr
        + batch_id * partial_meta_stride_b
        + head_id * partial_meta_stride_h
        + chunk_id * partial_meta_stride_c,
        m_i,
    )
    tl.store(
        partial_l_ptr
        + batch_id * partial_meta_stride_b
        + head_id * partial_meta_stride_h
        + chunk_id * partial_meta_stride_c,
        l_i,
    )


@triton.jit
def paged_decode_attention_splitk_stage2_kernel(
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    context_lens_ptr,
    out_ptr,
    partial_stride_b: tl.constexpr,
    partial_stride_h: tl.constexpr,
    partial_stride_c: tl.constexpr,
    partial_stride_d: tl.constexpr,
    partial_meta_stride_b: tl.constexpr,
    partial_meta_stride_h: tl.constexpr,
    partial_meta_stride_c: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_N: tl.constexpr,
    MAX_LEN: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens_ptr + batch_id)
    if SLIDING_WINDOW > 0:
        attn_len = tl.minimum(context_len, SLIDING_WINDOW)
    else:
        attn_len = context_len
    attn_len = tl.minimum(attn_len, MAX_LEN)
    valid_chunks = (attn_len + CHUNK_N - 1) // CHUNK_N
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    for chunk_id in range(0, NUM_CHUNKS):
        valid_chunk = chunk_id < valid_chunks
        l_j = tl.load(
            partial_l_ptr
            + batch_id * partial_meta_stride_b
            + head_id * partial_meta_stride_h
            + chunk_id * partial_meta_stride_c,
            mask=valid_chunk,
            other=0.0,
        ).to(tl.float32)
        m_j_raw = tl.load(
            partial_m_ptr
            + batch_id * partial_meta_stride_b
            + head_id * partial_meta_stride_h
            + chunk_id * partial_meta_stride_c,
            mask=valid_chunk,
            other=-float("inf"),
        ).to(tl.float32)
        valid_j = valid_chunk & (l_j > 0.0)
        m_j = tl.where(valid_j, m_j_raw, m_i)
        m_new = tl.maximum(m_i, m_j)
        m_new_safe = tl.where((l_i > 0.0) | valid_j, m_new, 0.0)
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_new_safe), 0.0)
        new_scale = tl.where(valid_j, tl.exp(m_j_raw - m_new_safe), 0.0)
        acc_j = tl.load(
            partial_out_ptr
            + batch_id * partial_stride_b
            + head_id * partial_stride_h
            + chunk_id * partial_stride_c
            + offs_d * partial_stride_d,
            mask=valid_chunk,
            other=0.0,
        ).to(tl.float32)
        acc = acc * old_scale + acc_j * new_scale
        l_i = l_i * old_scale + l_j * new_scale
        m_i = tl.where((l_i > 0.0) | valid_j, m_new, m_i)

    acc = acc / l_i
    tl.store(
        out_ptr + batch_id * out_stride_b + head_id * out_stride_h + offs_d * out_stride_d,
        acc,
    )


def paged_decode_attention_splitk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    sliding_window: int | None,
):
    bs, num_heads, head_dim = q.shape
    block_size = k_cache.size(1)
    max_len = block_tables.size(1) * block_size
    if sliding_window is not None:
        max_len = min(max_len, sliding_window)
    chunk_n = min(TRITON_PAGED_SPLIT_CHUNK, max_len)
    num_chunks = triton.cdiv(max_len, chunk_n)
    block_n = TRITON_PAGED_BLOCK_N_512 if head_dim >= 512 else TRITON_PAGED_BLOCK_N_256
    partial_out = torch.empty((bs, num_heads, num_chunks, head_dim), device=q.device, dtype=q.dtype)
    partial_m = torch.empty((bs, num_heads, num_chunks), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((bs, num_heads, num_chunks), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    paged_decode_attention_splitk_stage1_kernel[(bs, num_heads, num_chunks)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_out,
        partial_m,
        partial_l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_tables.stride(0),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        float(scale),
        num_heads,
        k_cache.size(2),
        head_dim,
        block_size,
        block_n,
        chunk_n,
        triton.cdiv(max_len, block_n) * block_n,
        sliding_window or 0,
        num_warps=8,
    )
    paged_decode_attention_splitk_stage2_kernel[(bs, num_heads)](
        partial_out,
        partial_m,
        partial_l,
        context_lens,
        out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        head_dim,
        num_chunks,
        chunk_n,
        max_len,
        sliding_window or 0,
        num_warps=8,
    )
    return out


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.k_cache = self.v_cache = torch.tensor([])

    def gather_cache(self, cache: torch.Tensor, block_table: torch.Tensor, length: int):
        pieces = []
        remaining = length
        for block_id in block_table.tolist():
            if remaining <= 0:
                break
            take = min(remaining, cache.size(1))
            pieces.append(cache[block_id, :take])
            remaining -= take
        return torch.cat(pieces, dim=0)

    def sdpa_one(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, q_start: int = 0):
        attn_mask = None
        if self.sliding_window is not None:
            k_start = max(0, q_start - self.sliding_window + 1)
            k_end = min(k.size(0), q_start + q.size(0))
            k = k[k_start:k_end]
            v = v[k_start:k_end]
            q_pos = torch.arange(q_start, q_start + q.size(0), device=q.device)
            k_pos = torch.arange(k_start, k_start + k.size(0), device=q.device)
            attn_mask = (k_pos[None, :] <= q_pos[:, None]) & (
                k_pos[None, :] >= q_pos[:, None] - self.sliding_window + 1
            )
            causal = False
        elif causal and (q_start != 0 or q.size(0) != k.size(0)):
            q_pos = torch.arange(q_start, q_start + q.size(0), device=q.device)
            k_pos = torch.arange(k.size(0), device=q.device)
            attn_mask = k_pos[None, :] <= q_pos[:, None]
            causal = False
        k = repeat_kv(k, self.num_heads)
        v = repeat_kv(v, self.num_heads)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal and q.size(2) > 1, scale=self.scale
        )
        return out.squeeze(0).transpose(0, 1)

    def decode_batched_sdpa(self, q: torch.Tensor):
        context = get_context()
        if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim in (256, 512):
            return paged_decode_attention_splitk(
                q,
                self.k_cache,
                self.v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                self.sliding_window,
            )
        bs = q.size(0)
        context_lens = context.context_lens
        max_cache_len = context.block_tables.size(1) * self.k_cache.size(1)
        if self.sliding_window is not None:
            max_len = min(max_cache_len, self.sliding_window)
        else:
            max_len = max_cache_len
        block_size = self.k_cache.size(1)
        pos = torch.arange(max_len, device=q.device)
        if self.sliding_window is not None:
            first_pos = (context_lens - max_len).clamp_min(0)
            abs_pos = first_pos.unsqueeze(1) + pos.unsqueeze(0)
        else:
            abs_pos = pos.unsqueeze(0).expand(bs, max_len)
        block_idx = torch.div(abs_pos, block_size, rounding_mode="floor")
        block_off = abs_pos - block_idx * block_size
        block_ids = torch.gather(context.block_tables, 1, block_idx)
        valid = abs_pos < context_lens.unsqueeze(1)
        safe_block_ids = torch.where(valid, block_ids, torch.zeros_like(block_ids))
        k = self.k_cache[safe_block_ids, block_off]
        v = self.v_cache[safe_block_ids, block_off]
        if k.size(2) != self.num_heads:
            repeat = self.num_heads // k.size(2)
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.view(bs, 1, self.num_heads, self.head_dim).transpose(1, 2)
        mask = valid[:, None, None, :]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=self.scale
        )
        return out.transpose(1, 2).reshape(bs, self.num_heads, self.head_dim)

    def torch_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        outs = []
        if context.is_prefill:
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
            for i in range(len(cu_q) - 1):
                qs, qe = cu_q[i], cu_q[i + 1]
                k_start, k_end = cu_k[i], cu_k[i + 1]
                q_len = qe - qs
                k_len = k_end - k_start
                if context.block_tables is None:
                    kv = (k[k_start:k_end], v[k_start:k_end])
                else:
                    kv = (
                        self.gather_cache(self.k_cache, context.block_tables[i], k_len),
                        self.gather_cache(self.v_cache, context.block_tables[i], k_len),
                    )
                outs.append(self.sdpa_one(q[qs:qe], kv[0], kv[1], True, k_len - q_len))
            return torch.cat(outs, dim=0)
        if q.size(0) > 1 or (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim in (256, 512)):
            return self.decode_batched_sdpa(q)
        for i in range(q.size(0)):
            seqlen = int(context.context_lens[i])
            kv = (
                self.gather_cache(self.k_cache, context.block_tables[i], seqlen),
                self.gather_cache(self.v_cache, context.block_tables[i], seqlen),
            )
            outs.append(self.sdpa_one(q[i:i + 1], kv[0], kv[1], False, seqlen - 1))
        return torch.cat(outs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        return self.torch_forward(q, k, v)
