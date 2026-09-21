# 面向 Gemma4 MoE 的 nano-vLLM 推理优化

**Languages:** [English](README.md) | [中文](README.zh-CN.md)

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，完成 [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) 单卡推理适配，并优化 Decode 路径上的 Attention、MoE 与 CUDA Graph。

## 结果（单卡 NVIDIA H20，BF16）

| | |
|---|---|
| Decode 16 路 | **279 → 643 tok/s（2.3×）** |
| Decode 64 路 | **1190 tok/s** |
| 全局层 Attention | 16Q / 2KV、4K 上下文，kernel **2.46 → 0.53 ms** |
| GSM8K 全量 1319 题 | **76.19% → 75.66%**，耗时约减半（1042 s → 525 s） |

16 路负载为 unique prompt 512 → 128。其中 CUDA Graph 将 decode 从 279 提到 569 tok/s（2.0×），GQA 与 grouped MoE 再提到 643。共享前缀场景下，Cascade Decode 相对 Graph 再提升 18%。

## 主要工作

- 适配 Gemma4：滑窗 / 全局交替注意力、RoPE、MoE 路由与专家计算
- Triton 两阶段 Split-K Paged Decode Attention；GQA 按 KV Head 分组复用 K/V
- 分组式 MoE：Top-k 按专家聚合后做 grouped GEMM，支持 CUDA Graph 捕获
- Decode CUDA Graph 按 batch 与 KV block 分桶
- `lm_eval` 适配，用于 GSM8K 精度对比

## 使用

```text
Python 3.12 · PyTorch 2.8.0 · CUDA 12.8 · 单卡 NVIDIA H20
```

```bash
git clone https://github.com/shiguang12321/gemma4-nanovllm-optimization
cd gemma4-nanovllm-optimization
export PYTHONPATH=$(pwd):$PYTHONPATH
export MODEL=/path/to/gemma-4-26B-A4B
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    MODEL,
    attn_decode_impl="splitk_gqa",
    moe_impl="grouped",
    enable_cascade_decode=True,
)
try:
    outputs = llm.generate(
        ["Explain grouped-query attention in one sentence."],
        SamplingParams(temperature=0.0, max_tokens=128),
        use_tqdm=False,
    )
    print(outputs[0]["text"])
finally:
    llm.exit()
```

GSM8K：

```bash
lm_eval \
  --model nano_vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
  --tasks gsm8k \
  --batch_size 16
```

## 配置

| 参数 | 默认 | 说明 |
|---|---|---|
| `attn_decode_impl` | `"splitk"` | `"splitk_gqa"` 启用 GQA 分组 Attention |
| `moe_impl` | `"auto"` | `"grouped"` 用于更大 batch 的 Graph 捕获 |
| `enable_cascade_decode` | `False` | 共享前缀 Decode |
| `enforce_eager` | `False` | `True` 关闭 CUDA Graph |

## License

MIT。基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，上游版权见 [LICENSE](LICENSE)。
