"""Opt-in tiled decode kernels.

Each query keeps its own online-softmax state. The tiles share K/V loads, never
attention outputs, across GQA heads or sequences with an identical KV prefix.
Both paths write unnormalized FP32 (acc, m, l) for the common merge kernel.
"""

import triton
import triton.language as tl


@triton.jit
def cascade_prefix_kernel(
    q_ptr, k_ptr, v_ptr, block_tables_ptr, context_lens_ptr, prefix_len_ptr,
    out_ptr, m_ptr, l_ptr,
    q_stride_b, q_stride_h, q_stride_d,
    cache_stride_block, cache_stride_token, cache_stride_head,
    batch_size, num_heads, scale, max_cache_len,
    NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    """One KV tile serves BLOCK_Q queries from sequences and their GQA heads.

    The caller guarantees that [0, prefix_len) uses the same physical blocks in
    every real sequence. Row zero therefore supplies the shared block IDs.
    Sliding windows remain per query, including different sequence lengths.
    """
    flat_queries = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    kv_head = tl.program_id(1)
    group_size = num_heads // NUM_KV_HEADS
    rows = flat_queries // group_size
    heads = kv_head * group_size + flat_queries % group_size
    dims = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens_ptr + rows, mask=rows < batch_size, other=0)
    valid_q = (rows < batch_size) & (context_len > 0)
    prefix_len = tl.maximum(0, tl.minimum(tl.load(prefix_len_ptr), max_cache_len))
    visible_start = tl.full((BLOCK_Q,), 0, tl.int32)
    if SLIDING_WINDOW > 0:
        visible_start = tl.maximum(context_len - SLIDING_WINDOW, 0)
    first_visible = tl.minimum(visible_start, prefix_len)
    first_tile = tl.min(tl.where(valid_q, first_visible, prefix_len), 0)
    first_tile = first_tile // BLOCK_N * BLOCK_N
    q = tl.load(
        q_ptr + rows[:, None] * q_stride_b + heads[:, None] * q_stride_h + dims[None, :] * q_stride_d,
        mask=valid_q[:, None], other=0.0,
    )
    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    base = first_tile
    while base < prefix_len:
        positions = base + tl.arange(0, BLOCK_N)
        valid_n = positions < prefix_len
        physical = tl.load(block_tables_ptr + positions // BLOCK_SIZE, mask=valid_n, other=0)
        offsets = (
            physical[:, None] * cache_stride_block
            + (positions % BLOCK_SIZE)[:, None] * cache_stride_token
            + kv_head * cache_stride_head + dims[None, :]
        )
        # No query/batch axis appears in these loads: this KV tile is reused.
        k = tl.load(k_ptr + offsets, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptr + offsets, mask=valid_n[:, None], other=0.0)
        valid = (
            valid_q[:, None] & valid_n[None, :]
            & (positions[None, :] >= visible_start[:, None])
            & (positions[None, :] < context_len[:, None])
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32) * scale
        scores = tl.where(valid, scores, -float("inf"))
        has_values = tl.sum(valid.to(tl.int32), 1) > 0
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_safe = tl.where(has_values | (l_i > 0.0), m_new, 0.0)
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_safe), 0.0)
        p = tl.where(valid, tl.exp(scores - m_safe[:, None]), 0.0)
        # FP32 probabilities avoid an extra BF16/FP16 rounding before P @ V.
        acc = acc * old_scale[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
        l_i = l_i * old_scale + tl.sum(p, 1)
        m_i = tl.where(has_values | (l_i > 0.0), m_new, m_i)
        base += BLOCK_N
    row_ids = rows * num_heads + heads
    tl.store(out_ptr + row_ids[:, None] * HEAD_DIM + dims[None, :], acc, mask=(rows < batch_size)[:, None])
    tl.store(m_ptr + row_ids, m_i, mask=rows < batch_size)
    tl.store(l_ptr + row_ids, l_i, mask=rows < batch_size)


@triton.jit
def paged_decode_gqa_stage1_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_tables_ptr, context_lens_ptr,
    partial_out_ptr, partial_m_ptr, partial_l_ptr, prefix_len_ptr,
    q_stride_b, q_stride_h, q_stride_d,
    cache_stride_block, cache_stride_token, cache_stride_head, block_table_stride,
    partial_stride_b, partial_stride_h, partial_stride_c, partial_stride_d,
    partial_meta_stride_b, partial_meta_stride_h, partial_meta_stride_c,
    scale,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, CHUNK_N: tl.constexpr,
    MAX_LEN, SLIDING_WINDOW: tl.constexpr, HAS_PREFIX: tl.constexpr,
):
    """One program reuses each K/V load for up to 16 Q heads in one GQA group."""
    BLOCK_Q: tl.constexpr = 16
    GROUP_SIZE: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    GROUP_TILES: tl.constexpr = triton.cdiv(GROUP_SIZE, BLOCK_Q)
    batch = tl.program_id(0)
    head_tile = tl.program_id(1)
    chunk = tl.program_id(2)
    kv_head = head_tile // GROUP_TILES
    group_offset = head_tile % GROUP_TILES * BLOCK_Q + tl.arange(0, BLOCK_Q)
    heads = kv_head * GROUP_SIZE + group_offset
    valid_head = group_offset < GROUP_SIZE
    dims = tl.arange(0, HEAD_DIM)
    context_len = tl.load(context_lens_ptr + batch)
    start_pos = 0
    if SLIDING_WINDOW > 0:
        start_pos = tl.maximum(context_len - SLIDING_WINDOW, 0)
    if HAS_PREFIX:
        start_pos = tl.maximum(start_pos, tl.load(prefix_len_ptr))
    attn_len = tl.maximum(context_len - start_pos, 0)
    chunk_start = chunk * CHUNK_N
    if chunk_start >= attn_len:
        return
    q = tl.load(
        q_ptr + batch * q_stride_b + heads[:, None] * q_stride_h + dims[None, :] * q_stride_d,
        mask=valid_head[:, None], other=0.0,
    )
    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    for local_base in range(0, CHUNK_N, BLOCK_N):
        local_positions = chunk_start + local_base + tl.arange(0, BLOCK_N)
        positions = start_pos + local_positions
        valid_n = (local_positions < attn_len) & (local_positions < MAX_LEN)
        physical = tl.load(
            block_tables_ptr + batch * block_table_stride + positions // BLOCK_SIZE,
            mask=valid_n, other=0,
        )
        offsets = (
            physical[:, None] * cache_stride_block
            + (positions % BLOCK_SIZE)[:, None] * cache_stride_token
            + kv_head * cache_stride_head + dims[None, :]
        )
        k = tl.load(k_cache_ptr + offsets, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_cache_ptr + offsets, mask=valid_n[:, None], other=0.0)
        valid = valid_head[:, None] & valid_n[None, :]
        scores = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32) * scale
        scores = tl.where(valid, scores, -float("inf"))
        has_values = tl.sum(valid.to(tl.int32), 1) > 0
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_safe = tl.where(has_values | (l_i > 0.0), m_new, 0.0)
        old_scale = tl.where(l_i > 0.0, tl.exp(m_i - m_safe), 0.0)
        p = tl.where(valid, tl.exp(scores - m_safe[:, None]), 0.0)
        acc = acc * old_scale[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
        l_i = l_i * old_scale + tl.sum(p, 1)
        m_i = tl.where(has_values | (l_i > 0.0), m_new, m_i)
    out_offsets = (
        batch * partial_stride_b + heads[:, None] * partial_stride_h
        + chunk * partial_stride_c + dims[None, :] * partial_stride_d
    )
    meta_offsets = batch * partial_meta_stride_b + heads * partial_meta_stride_h + chunk * partial_meta_stride_c
    tl.store(partial_out_ptr + out_offsets, acc, mask=valid_head[:, None])
    tl.store(partial_m_ptr + meta_offsets, m_i, mask=valid_head)
    tl.store(partial_l_ptr + meta_offsets, l_i, mask=valid_head)
