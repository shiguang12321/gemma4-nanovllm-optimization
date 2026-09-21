# Gemma4 MoE Inference Optimization on nano-vLLM

**Languages:** [English](README.md) | [中文](README.zh-CN.md)

Single-GPU inference for [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B), built on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). This repository adapts Gemma4 and optimizes decode-path attention, MoE, and CUDA Graph.

## Results (1× NVIDIA H20, BF16)

| | |
|---|---|
| Decode, 16-way | **279 → 643 tok/s (2.3×)** |
| Decode, 64-way | **1190 tok/s** |
| Global-layer attention | 16Q / 2KV, 4K context, kernel **2.46 → 0.53 ms** |
| GSM8K, 1319 problems | **76.19% → 75.66%**, runtime about halved (1042 s → 525 s) |

The 16-way workload uses unique prompts, 512 → 128. CUDA Graph raises decode from 279 to 569 tok/s (2.0×); GQA and grouped MoE take it to 643. On a shared-prefix workload, cascade decode adds another 18% over Graph.

## Highlights

- Gemma4 adaptation: sliding / global attention, RoPE, MoE routing and expert compute
- Two-stage Triton Split-K paged decode attention, with GQA reusing K/V by KV head
- Grouped MoE: aggregate top-k routes by expert, then grouped GEMM, CUDA Graph compatible
- Decode CUDA Graph buckets by batch and KV block width
- `lm_eval` adapter for GSM8K

## Usage

```text
Python 3.12 · PyTorch 2.8.0 · CUDA 12.8 · 1× NVIDIA H20
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

GSM8K:

```bash
lm_eval \
  --model nano_vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
  --tasks gsm8k \
  --batch_size 16
```

## Configuration

| Option | Default | Notes |
|---|---|---|
| `attn_decode_impl` | `"splitk"` | `"splitk_gqa"` enables GQA grouped attention |
| `moe_impl` | `"auto"` | `"grouped"` for larger-batch Graph capture |
| `enable_cascade_decode` | `False` | shared-prefix decode |
| `enforce_eager` | `False` | `True` disables CUDA Graph |

## License

MIT. Based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). See [LICENSE](LICENSE).
