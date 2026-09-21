import torch
from torch import nn
import triton
import triton.language as tl
from functools import lru_cache

from nanovllm.utils.context import get_context
from nanovllm.layers.attention_tiles import (
    cascade_prefix_kernel,
    paged_decode_gqa_stage1_kernel,
)


TRITON_PAGED_BLOCK_N_256 = 32
TRITON_PAGED_BLOCK_N_512 = 16
TRITON_PAGED_SPLIT_CHUNK = 128


@lru_cache(maxsize=None)
def _device_sm_count(device_index: int) -> int:
    # Called during module construction, never during graph capture/replay.
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def select_decode_chunk_size(max_len: int, batch_size: int, program_heads: int, sm_count: int) -> int:
    """Choose from bounded tiles using only graph-static metadata.

    This is an occupancy-oriented heuristic, not a measured performance claim.
    Keep the established 128-token tile when device information is unavailable.
    """
    if sm_count <= 0:
        return TRITON_PAGED_SPLIT_CHUNK
    target_programs = 2 * sm_count
    for chunk in (512, 256, 128):
        programs = batch_size * program_heads * ((max_len + chunk - 1) // chunk)
        if programs >= target_programs:
            return chunk
    return 128


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
): #把key和value存到kv cache里
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


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor): #把key和value存到kv cache里
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def repeat_kv(x: torch.Tensor, num_heads: int): #如果key和value的head数量不等于Q的head数量，则重复key和value
    if x.size(1) == num_heads:
        return x
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


@triton.jit   #对“某条序列的某个 Q head 的某一个 KV chunk”计算局部 Attention，
              #并保存这个 chunk 的 acc、m、l，留给 Stage 2 合并。
def paged_decode_attention_splitk_stage1_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    prefix_len_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_d,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    block_table_stride,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    partial_stride_d,
    partial_meta_stride_b,
    partial_meta_stride_h,
    partial_meta_stride_c,
    scale,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
    MAX_LEN,
    SLIDING_WINDOW: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
):
    batch_id = tl.program_id(0) #batch的第几条序列
    head_id = tl.program_id(1) #第几个Q head
    chunk_id = tl.program_id(2) #第几个KV chunk
    kv_head = head_id // (NUM_HEADS // NUM_KV_HEADS) #第几个KV head

    context_len = tl.load(context_lens_ptr + batch_id) #历史长度
    if SLIDING_WINDOW > 0:
        attn_len = tl.minimum(context_len, SLIDING_WINDOW) #滑动窗口长度
        start_pos = context_len - attn_len
    else:
        attn_len = context_len  #如果没有滑动窗口，则使用整个历史
        start_pos = 0

    if HAS_PREFIX:
        start_pos = tl.maximum(start_pos, tl.load(prefix_len_ptr))
        attn_len = tl.maximum(context_len - start_pos, 0)

    chunk_start = chunk_id * CHUNK_N  #当前chunk的起始位置
    if chunk_start >= attn_len:
        return

    offs_d = tl.arange(0, HEAD_DIM) #当前Q head的第几个维度
    q = tl.load(q_ptr + batch_id * q_stride_b + head_id * q_stride_h + offs_d * q_stride_d).to(tl.float32) #当前Q head的第几个维度
    m_i = tl.full((), -float("inf"), tl.float32) #当前chunk内attention score 的最大值
    l_i = tl.full((), 0.0, tl.float32) #当前chunk的softmax归一化和
    acc = tl.zeros((HEAD_DIM,), tl.float32) #当前chunk的attention输出

    for local_base in range(0, CHUNK_N, BLOCK_N): #每个block多少token
        offs_n = chunk_start + local_base + tl.arange(0, BLOCK_N) #当前chunk的第几个token
        abs_pos = start_pos + offs_n #当前token的绝对位置
        block_idx = abs_pos // BLOCK_SIZE #当前token所在的block索引
        block_off = abs_pos - block_idx * BLOCK_SIZE #当前token在block内的偏移量
        valid = (offs_n < attn_len) & (offs_n < MAX_LEN) #当前token是否有效

        block_ids = tl.load(
            block_tables_ptr + batch_id * block_table_stride + block_idx,
            mask=valid,
            other=0,
        ) #通过block table找到真正的物理 kv block

        k = tl.load(
            k_cache_ptr
            + block_ids[:, None] * cache_stride_block
            + block_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)  #收集k,沿最后一维求和,得到分数,再和q相乘得到attention score

        qk = tl.sum(k * q[None, :], axis=1) * scale #Q和这批k做点积
        qk = tl.where(valid, qk, -float("inf")) #无效位置达成负无穷



        m_chunk = tl.max(qk, axis=0) #当前chunk内attention score 的最大值
        has_values = tl.sum(tl.where(valid, 1, 0), axis=0) > 0 #在这一批32个位置后，有几个是真的历史token，只要个数>0，说明有历史token

        m_new = tl.maximum(m_i, m_chunk) #更新最大值
        m_new_safe = tl.where(has_values | (l_i > 0.0), m_new, 0.0)#真正拿去减的那个max
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_new_safe), 0.0)#旧 acc/l 要乘的缩放。
        p = tl.where(valid, tl.exp(qk - m_new_safe), 0.0)#每个位置要乘的归一化因子

        v = tl.load(
            v_cache_ptr
            + block_ids[:, None] * cache_stride_block
            + block_off[:, None] * cache_stride_token
            + kv_head * cache_stride_head
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)  #收集v,沿最后一维求和,得到分数,再和q相乘得到attention score

        acc = acc * old_scale + tl.sum(p[:, None] * v, axis=0) #更新acc
        l_i = l_i * old_scale + tl.sum(p, axis=0) #更新l
        m_i = tl.where(has_values | (l_i > 0.0), m_new, m_i) #更新m

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
#把这个 (batch, head, chunk) 在 SRAM 里算完的三本账，写到显存里，供 stage2 来合。

@triton.jit  #合并 Stage 1 的多个 chunk 的结果，得到最终的 Attention 输出。
def paged_decode_attention_splitk_stage2_kernel(
    partial_out_ptr, #Stage 1 的多个 chunk 的结果
    partial_m_ptr, #Stage 1 的多个 chunk 的attention score 的最大值
    partial_l_ptr, #Stage 1 的多个 chunk 的softmax归一化和
    context_lens_ptr, #历史长度
    out_ptr, #最终的Attention 输出
    prefix_len_ptr,
    prefix_out_ptr,
    prefix_m_ptr,
    prefix_l_ptr,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    partial_stride_d,
    partial_meta_stride_b,
    partial_meta_stride_h,
    partial_meta_stride_c,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    num_heads,
    HEAD_DIM: tl.constexpr,   #head维度
    NUM_CHUNKS,   #Runtime count: bucket width need not specialize this loop.
    CHUNK_N: tl.constexpr,   #每个chunk多少token
    MAX_LEN,
    SLIDING_WINDOW: tl.constexpr,   #滑动窗口长度
    HAS_PREFIX: tl.constexpr,
):
    batch_id = tl.program_id(0)  #batch的第几条序列
    head_id = tl.program_id(1)  #第几个Q head
    offs_d = tl.arange(0, HEAD_DIM)  #第几个维度
    context_len = tl.load(context_lens_ptr + batch_id)  #历史长度
    if SLIDING_WINDOW > 0:
        attn_len = tl.minimum(context_len, SLIDING_WINDOW)  #滑动窗口长度
    else:
        attn_len = context_len  #如果没有滑动窗口，则使用整个历史
    if HAS_PREFIX:
        start_pos = tl.maximum(context_len - attn_len, tl.load(prefix_len_ptr))
        attn_len = tl.maximum(context_len - start_pos, 0)
    attn_len = tl.minimum(attn_len, MAX_LEN)  #滑动窗口长度和cache长度取最小
    valid_chunks = (attn_len + CHUNK_N - 1) // CHUNK_N  #多少个chunk有效
    m_i = tl.full((), -float("inf"), tl.float32)  #当前chunk内attention score 的最大值
    l_i = tl.full((), 0.0, tl.float32)  #当前chunk的softmax归一化和
    acc = tl.zeros((HEAD_DIM,), tl.float32)  #当前chunk的attention输出
    if HAS_PREFIX:
        prefix_row = batch_id * num_heads + head_id
        m_i = tl.load(prefix_m_ptr + prefix_row)
        l_i = tl.load(prefix_l_ptr + prefix_row)
        acc = tl.load(prefix_out_ptr + prefix_row * HEAD_DIM + offs_d)

    chunk_id = 0
    while chunk_id < NUM_CHUNKS: #Runtime loop; also supported by the CPU interpreter.
        valid_chunk = chunk_id < valid_chunks #当前chunk是否有效
        #对每个chunk_id, 读取这个 chunk 的 acc、m、l，作为 Stage 1 的合并对象。
        l_j = tl.load(
            partial_l_ptr
            + batch_id * partial_meta_stride_b
            + head_id * partial_meta_stride_h
            + chunk_id * partial_meta_stride_c,
            mask=valid_chunk,
            other=0.0,
        ).to(tl.float32)
        #load Stage 1 的多个 chunk 的attention score 的最大值
        m_j_raw = tl.load(
            partial_m_ptr
            + batch_id * partial_meta_stride_b
            + head_id * partial_meta_stride_h
            + chunk_id * partial_meta_stride_c,
            mask=valid_chunk,
            other=-float("inf"),
        ).to(tl.float32)
        #load Stage 1 的多个 chunk 的attention score 的最大值
        valid_j = valid_chunk & (l_j > 0.0)  #当前chunk是否有效且l_j>0
        m_j = tl.where(valid_j, m_j_raw, m_i)  #当前chunk的attention score 的最大值
        m_new = tl.maximum(m_i, m_j) #更新最大值
        m_new_safe = tl.where((l_i > 0.0) | valid_j, m_new, 0.0) #真正拿去减的那个max
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_new_safe), 0.0) #旧 acc/l 要乘的缩放。
        new_scale = tl.where(valid_j, tl.exp(m_j_raw - m_new_safe), 0.0) #每个位置要乘的归一化因子
        #load Stage 1 的多个 chunk 的attention输出
        acc_j = tl.load(
            partial_out_ptr
            + batch_id * partial_stride_b
            + head_id * partial_stride_h
            + chunk_id * partial_stride_c
            + offs_d * partial_stride_d,
            mask=valid_chunk,
            other=0.0,
        ).to(tl.float32)  #load Stage 1 的多个 chunk 的attention输出

        acc = acc * old_scale + acc_j * new_scale #更新acc
        l_i = l_i * old_scale + l_j * new_scale #更新l
        m_i = tl.where((l_i > 0.0) | valid_j, m_new, m_i) #更新m
        chunk_id += 1

    # Empty graph padding rows have l_i == 0 and must remain finite.
    acc = acc / tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(
        out_ptr + batch_id * out_stride_b + head_id * out_stride_h + offs_d * out_stride_d,
        acc,
    )


def paged_decode_attention_splitk(
    q: torch.Tensor, #每条序列1个Q
    k_cache: torch.Tensor, #KV cache
    v_cache: torch.Tensor, #KV cache
    block_tables: torch.Tensor, #block table
    context_lens: torch.Tensor, #历史长度
    scale: float, #缩放因子
    sliding_window: int | None, #滑动窗口
    *,
    impl: str = "splitk",
    cascade_prefix_len: torch.Tensor | None = None,
    sm_count: int = 0,
):
    """Paged decode, optionally sharing KV loads across heads / prefix queries.

    cascade_prefix_len is a device int32[1] containing a full-block prefix that
    all real rows share physically. Its contents may change between graph
    replays; its presence and all tensor shapes are graph-static.
    """
    bs, num_heads, head_dim = q.shape #batch size, head数量, head维度
    if impl not in ("splitk", "splitk_gqa"):
        raise ValueError(f"Unknown attention decode implementation: {impl}")
    if num_heads % k_cache.size(2):
        raise ValueError("Query head count must be divisible by KV head count")
    if k_cache.shape != v_cache.shape or k_cache.stride() != v_cache.stride():
        raise ValueError("K and V caches must have matching shapes and strides")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or block_tables.stride(-1) != 1:
        raise ValueError("Paged decode requires contiguous head dimensions and block-table columns")
    if head_dim not in (256, 512):
        raise ValueError("Paged decode currently supports head dimensions 256 and 512")
    if block_tables.size(1) < 1 or sliding_window is not None and sliding_window <= 0:
        raise ValueError("Cache width and sliding window must be positive")
    if cascade_prefix_len is not None:
        if cascade_prefix_len.shape != (1,) or cascade_prefix_len.dtype != torch.int32:
            raise ValueError("cascade_prefix_len must be an int32 tensor of shape [1]")
        if cascade_prefix_len.device != q.device:
            raise ValueError("cascade_prefix_len must be on the query device")
    block_size = k_cache.size(1) #kv cache一页存多少token
    max_len = block_tables.size(1) * block_size   #cache长度
    if sliding_window is not None: #如果有滑动窗口
        max_len = min(max_len, sliding_window) #滑动窗口长度
    group_size = num_heads // k_cache.size(2)
    gqa_program_heads = k_cache.size(2) * triton.cdiv(group_size, 16)
    program_heads = num_heads if impl == "splitk" else gqa_program_heads
    chunk_n = (TRITON_PAGED_SPLIT_CHUNK if impl == "splitk" else
               select_decode_chunk_size(max_len, bs, program_heads, sm_count))
    num_chunks = triton.cdiv(max_len, chunk_n) #多少个chunk
    block_n = TRITON_PAGED_BLOCK_N_512 if head_dim >= 512 else TRITON_PAGED_BLOCK_N_256 #Triton Kernel一次处理多少个kv token
    partial_out = torch.empty((bs, num_heads, num_chunks, head_dim), device=q.device, dtype=torch.float32)
    partial_m = torch.empty((bs, num_heads, num_chunks), device=q.device, dtype=torch.float32) # 每个chunk内attention score 的最大值，用于稳定的softmax合并
    partial_l = torch.empty((bs, num_heads, num_chunks), device=q.device, dtype=torch.float32) # 每个chunk的softmax归一化和
    out = torch.empty_like(q) #输出
    has_prefix = cascade_prefix_len is not None
    # Dummy pointers are compile-time dead when HAS_PREFIX=False.
    prefix_len = cascade_prefix_len if has_prefix else context_lens
    prefix_out, prefix_m, prefix_l = partial_out, partial_m, partial_l
    if has_prefix:
        prefix_out = torch.empty((bs, num_heads, head_dim), device=q.device, dtype=torch.float32)
        prefix_m = torch.empty((bs, num_heads), device=q.device, dtype=torch.float32)
        prefix_l = torch.empty_like(prefix_m)
        cascade_prefix_kernel[(triton.cdiv(bs * group_size, 16), k_cache.size(2))](
            q, k_cache, v_cache, block_tables, context_lens, prefix_len,
            prefix_out, prefix_m, prefix_l,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            bs, num_heads, float(scale), block_tables.size(1) * block_size,
            NUM_KV_HEADS=k_cache.size(2), HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size, BLOCK_Q=16, BLOCK_N=32,
            SLIDING_WINDOW=sliding_window or 0, num_warps=8,
        )

    stage1 = paged_decode_attention_splitk_stage1_kernel if impl == "splitk" else paged_decode_gqa_stage1_kernel
    stage1[(bs, program_heads, num_chunks)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_out,
        partial_m,
        partial_l,
        prefix_len,
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
        has_prefix,
        num_warps=8,
    )
    paged_decode_attention_splitk_stage2_kernel[(bs, num_heads)](
        partial_out,
        partial_m,
        partial_l,
        context_lens,
        out,
        prefix_len,
        prefix_out,
        prefix_m,
        prefix_l,
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
        num_heads,
        head_dim,
        num_chunks,
        chunk_n,
        max_len,
        sliding_window or 0,
        has_prefix,
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
        attn_decode_impl: str = "splitk",
        enable_cascade_decode: bool = False,
    ):
        super().__init__()
        if attn_decode_impl not in ("splitk", "splitk_gqa"):
            raise ValueError(f"Unknown attention decode implementation: {attn_decode_impl}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.attn_decode_impl = attn_decode_impl
        self.enable_cascade_decode = enable_cascade_decode
        self.decode_sm_count = _device_sm_count(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        self.k_cache = self.v_cache = torch.tensor([])

    def gather_cache(self, cache: torch.Tensor, block_table: torch.Tensor | list[int], length: int):
        pieces = []
        remaining = length
        # Normal prefill supplies the runner's CPU metadata, avoiding per-layer
        # device-to-host synchronization. Tensor fallback supports old callers.
        block_ids = block_table.tolist() if isinstance(block_table, torch.Tensor) else block_table
        for block_id in block_ids:
            if remaining <= 0:
                break
            take = min(remaining, cache.size(1))
            pieces.append(cache[block_id, :take])
            remaining -= take
        return torch.cat(pieces, dim=0) if pieces else cache.new_empty((0, *cache.shape[2:]))

    def sdpa_one(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, q_start: int = 0):  #单条序列的SDPA,带滑动窗口
        attn_mask = None #注意力掩码
        if self.sliding_window is not None: #如果有滑动窗口
            k_start = max(0, q_start - self.sliding_window + 1) #滑动窗口起始位置
            k_end = min(k.size(0), q_start + q.size(0)) #滑动窗口结束位置
            k = k[k_start:k_end]
            v = v[k_start:k_end] #截取滑动窗口内的k和v
            q_pos = torch.arange(q_start, q_start + q.size(0), device=q.device) #q的绝对位置
            k_pos = torch.arange(k_start, k_start + k.size(0), device=q.device) #k的绝对位置
            attn_mask = (k_pos[None, :] <= q_pos[:, None]) & ( #注意力掩码
                k_pos[None, :] >= q_pos[:, None] - self.sliding_window + 1
            )
            causal = False
        elif causal and (q_start != 0 or q.size(0) != k.size(0)): #如果没有滑动窗口，但是有因果关系
            q_pos = torch.arange(q_start, q_start + q.size(0), device=q.device)
            k_pos = torch.arange(k.size(0), device=q.device) #k的绝对位置
            attn_mask = k_pos[None, :] <= q_pos[:, None] #注意力掩码
            causal = False #因果关系
        k = repeat_kv(k, self.num_heads)
        v = repeat_kv(v, self.num_heads) #把kv head扩展成和Q head一样多
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0) #转置并添加batch维度
        v = v.transpose(0, 1).unsqueeze(0) #转置并添加batch维度
        out = torch.nn.functional.scaled_dot_product_attention( #调用SDPA
            q, k, v, attn_mask=attn_mask, is_causal=causal and q.size(2) > 1, scale=self.scale
        )
        return out.squeeze(0).transpose(0, 1)

    def decode_batched_sdpa(self, q: torch.Tensor): #decode的主路径 批量decode
        context = get_context()
        if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim in (256, 512):
            prefix_len = getattr(context, "cascade_prefix_len", None) if self.enable_cascade_decode else None
            if self.enable_cascade_decode and prefix_len is None:
                raise ValueError("Cascade decode requires Context.cascade_prefix_len (device int32[1])")
            return paged_decode_attention_splitk(
                q,
                self.k_cache,
                self.v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                self.sliding_window,
                impl=self.attn_decode_impl,
                cascade_prefix_len=prefix_len,
                sm_count=self.decode_sm_count,
            )   #优先走自己的 Triton split-K
        bs = q.size(0) #batch size
        context_lens = context.context_lens #历史长度
        max_cache_len = context.block_tables.size(1) * self.k_cache.size(1) #cache长度
        if self.sliding_window is not None: #如果有滑动窗口
            max_len = min(max_cache_len, self.sliding_window)
        else: #如果没有滑动窗口
            max_len = max_cache_len
        block_size = self.k_cache.size(1) #block大小
        pos = torch.arange(max_len, device=q.device)
        if self.sliding_window is not None: #如果有滑动窗口
            first_pos = (context_lens - max_len).clamp_min(0)
            abs_pos = first_pos.unsqueeze(1) + pos.unsqueeze(0)
        else: #如果没有滑动窗口
            abs_pos = pos.unsqueeze(0).expand(bs, max_len)
        block_idx = torch.div(abs_pos, block_size, rounding_mode="floor") #计算block索引
        block_off = abs_pos - block_idx * block_size
        block_ids = torch.gather(context.block_tables, 1, block_idx) #通过block table找到真正的物理 kv block
        valid = abs_pos < context_lens.unsqueeze(1) #有效性掩码
        safe_block_ids = torch.where(valid, block_ids, torch.zeros_like(block_ids)) #安全block索引
        k = self.k_cache[safe_block_ids, block_off] #收集k
        v = self.v_cache[safe_block_ids, block_off] #收集v
        if k.size(2) != self.num_heads: #如果k的head数量不等于Q的head数量
            repeat = self.num_heads // k.size(2)
            k = k.repeat_interleave(repeat, dim=2) #重复k
            v = v.repeat_interleave(repeat, dim=2) #重复v
        k = k.transpose(1, 2) #转置k
        v = v.transpose(1, 2) #转置v
        q = q.view(bs, 1, self.num_heads, self.head_dim).transpose(1, 2) #转置q
        mask = valid[:, None, None, :] #注意力掩码
        out = torch.nn.functional.scaled_dot_product_attention( #调用SDPA
            q, k, v, attn_mask=mask, is_causal=False, scale=self.scale
        )
        return out.transpose(1, 2).reshape(bs, self.num_heads, self.head_dim)

    def torch_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): #前向计算
        context = get_context()
        outs = []
        if context.is_prefill:  #预填
            cu_q = getattr(context, "cu_seqlens_q_cpu", None)
            cu_k = getattr(context, "cu_seqlens_k_cpu", None)
            if cu_q is None:
                cu_q = context.cu_seqlens_q.tolist()
            if cu_k is None:
                cu_k = context.cu_seqlens_k.tolist()
            cpu_tables = getattr(context, "block_tables_cpu", None)
            #如果现在是 Prefill，
            # 就取出 batch 中每条序列的 Q 和 K/V 在拼接 Tensor 中的起止边界，
            # 为后面逐条序列做 Attention 做准备。
            for i in range(len(cu_q) - 1):
                qs, qe = cu_q[i], cu_q[i + 1]
                k_start, k_end = cu_k[i], cu_k[i + 1]
                q_len = qe - qs
                k_len = k_end - k_start
                if context.block_tables is None: #如果block_tables为空
                    kv = (k[k_start:k_end], v[k_start:k_end])
                else:
                    table = cpu_tables[i] if cpu_tables is not None else context.block_tables[i].tolist()
                    kv = (
                        self.gather_cache(self.k_cache, table, k_len),
                        self.gather_cache(self.v_cache, table, k_len),
                    )
                outs.append(self.sdpa_one(q[qs:qe], kv[0], kv[1], True, k_len - q_len))
            return torch.cat(outs, dim=0)
        #没有 Prefix/Paged Cache  直接用当前 k/v

        #有 Prefix/Paged Cache 根据 block_table 把完整历史 k/v 从 cache 里 gather 出来

        if q.size(0) > 1 or (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim in (256, 512)): #   小 batch 用分页 decode；大 batch 用 batched SDPA
            return self.decode_batched_sdpa(q)
        for i in range(q.size(0)): #逐条序列做 Attention
            seqlen = int(context.context_lens[i])
            kv = (
                self.gather_cache(self.k_cache, context.block_tables[i], seqlen),
                self.gather_cache(self.v_cache, context.block_tables[i], seqlen),
            )
            outs.append(self.sdpa_one(q[i:i + 1], kv[0], kv[1], False, seqlen - 1))
        return torch.cat(outs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):#
        context = get_context()  #获取context
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): #如果k_cache和v_cache不为空
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) #把k和v写入cache
        return self.torch_forward(q, k, v) #调用torch_forward计算注意力
