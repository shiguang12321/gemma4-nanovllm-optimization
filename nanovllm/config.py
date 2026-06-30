import os
import json
from types import SimpleNamespace
from dataclasses import dataclass
import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 16
    max_model_len: int = 1792
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.tensor_parallel_size == 1
        hf_config = self.load_hf_config()
        if getattr(hf_config, "model_type", None) == "gemma4":
            hf_config = hf_config.text_config
        self.hf_config = hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.hf_config.max_position_embeddings = self.max_model_len

    def load_hf_config(self):
        try:
            return AutoConfig.from_pretrained(self.model)
        except ValueError:
            with open(os.path.join(self.model, "config.json")) as f:
                data = json.load(f)
            return self.to_namespace(data)

    def to_namespace(self, data):
        if isinstance(data, dict):
            values = {}
            for k, v in data.items():
                values[k] = v if k in ("rope_parameters", "id2label", "label2id") else self.to_namespace(v)
            return SimpleNamespace(**values)
        if isinstance(data, list):
            return [self.to_namespace(v) for v in data]
        if data == "bfloat16":
            return torch.bfloat16
        if data == "float16":
            return torch.float16
        if data == "float32":
            return torch.float32
        return data
