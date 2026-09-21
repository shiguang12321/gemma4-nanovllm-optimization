from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    cascade_prefix_len: torch.Tensor | None = None
    cu_seqlens_q_cpu: tuple[int, ...] | None = None
    cu_seqlens_k_cpu: tuple[int, ...] | None = None
    block_tables_cpu: tuple[tuple[int, ...], ...] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, *, cascade_prefix_len=None, cu_seqlens_q_cpu=None, cu_seqlens_k_cpu=None, block_tables_cpu=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, cascade_prefix_len, cu_seqlens_q_cpu, cu_seqlens_k_cpu, block_tables_cpu)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
