"""Host-only decode planning; importing this module never initializes CUDA."""

from bisect import bisect_left
from collections.abc import Sequence


def parse_graph_block_buckets(max_num_blocks: int, buckets: list[int] | None = None) -> list[int]:
    if type(max_num_blocks) is not int or max_num_blocks < 1:
        raise ValueError("max_num_blocks must be a positive integer")
    if buckets is None:
        result = []
        value = 1
        while value <= max_num_blocks:
            result.append(value)
            value *= 2
    else:
        if not isinstance(buckets, (list, tuple)):
            raise ValueError("graph_block_buckets must be a list of positive integers")
        if any(type(value) is not int or value < 1 for value in buckets):
            raise ValueError("graph_block_buckets must contain positive integers")
        result = [value for value in buckets if value <= max_num_blocks]
    # Split-K grid depends on table WIDTH, not the live context length. CUDA
    # Graph freezes that grid; width buckets avoid launching maximum-length
    # work for short contexts. Explicit [4, 5, 6] retains the old short buckets.
    return sorted(set(result + [max_num_blocks]))


def graph_batch_limit(max_num_seqs: int, moe_impl: str, auto_limit: int) -> int:
    if max_num_seqs < 1 or auto_limit < 1:
        raise ValueError("batch limits must be positive")
    if moe_impl not in ("auto", "grouped"):
        raise ValueError("moe_impl must be 'auto' or 'grouped'")
    return max_num_seqs if moe_impl == "grouped" else min(max_num_seqs, auto_limit)


def select_graph_key(batch_size: int, num_blocks: int,
                     batch_buckets: Sequence[int], block_buckets: Sequence[int]):
    """Return a covering (batch, width) or None for the eager path."""
    if batch_size < 1 or num_blocks < 1:
        return None
    batch_index = bisect_left(batch_buckets, batch_size)
    block_index = bisect_left(block_buckets, num_blocks)
    if batch_index == len(batch_buckets) or block_index == len(block_buckets):
        return None
    return batch_buckets[batch_index], block_buckets[block_index]


def common_cached_prefix_tokens(block_tables: Sequence[Sequence[int]],
                                cached_lengths: Sequence[int], block_size: int) -> int:
    """Longest common physical prefix of complete, already cached blocks.

    The caller excludes the token being decoded from cached_lengths. Partial
    or mutable current blocks must never enter the shared-prefix calculation.
    """
    if block_size < 1 or len(block_tables) != len(cached_lengths):
        raise ValueError("invalid prefix metadata")
    if any(length < 0 for length in cached_lengths):
        raise ValueError("cached lengths cannot be negative")
    if len(block_tables) < 2:
        return 0
    limit = min(min(len(table), length // block_size)
                for table, length in zip(block_tables, cached_lengths))
    common = 0
    for index in range(limit):
        block_id = block_tables[0][index]
        if block_id < 0 or any(table[index] != block_id for table in block_tables[1:]):
            break
        common += 1
    return common * block_size
