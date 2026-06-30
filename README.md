# Gemma4-MoE Inference Optimization Based on nano-vLLM

Languages: [English](#project-overview) | [中文](#中文版)

## Project Overview

This repository is a derivative engineering project based on
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). It
keeps the compact nano-vLLM runtime style while adapting the engine for
single-GPU Gemma4-26B-A4B inference.

The work focuses on Gemma4 model loading, prefill and decode execution paths,
MoE routing and expert execution, KV cache layout, and Triton kernels used by
latency-sensitive decode workloads. The project is intended as a focused
inference optimization codebase, not a from-scratch inference engine.

## Key Features

- Gemma4-26B-A4B text-model structure and weight mapping support.
- RoPE, RMSNorm, sliding-window attention, and Gemma4 attention variants.
- Prefix KV cache support for shared-prefix decode batches.
- Triton Split-K paged decode attention with sliding-window handling.
- Triton Router Top-k and MoE decode kernels for decode-stage expert routing.
- Batch/block two-dimensional CUDA Graph buckets for decode replay.
- Last-token logits and greedy sampler fast paths.
- `lm_eval` model registration through `sitecustomize.py` and
  `nanovllm/lm_eval_adapter.py`.

## Changes from Upstream nano-vLLM

The main project-specific changes are concentrated in the following files:

- `nanovllm/models/gemma4.py`
  - Gemma4 model structure adaptation.
  - Gemma4 RoPE, RMSNorm, attention, MLP, router, and expert modules.
  - Triton Router Top-k and MoE decode kernels.

- `nanovllm/layers/attention.py`
  - Split-K paged decode attention.
  - Sliding-window support in prefill and decode paths.
  - Batched decode handling for paged KV cache.

- `nanovllm/engine/model_runner.py`
  - Gemma4 model construction.
  - Gemma4 KV cache allocation and context preparation.
  - Batch/block CUDA Graph capture buckets for decode.
  - Shared-prefix decode metadata propagation.

- `nanovllm/layers/sampler.py`
  - Greedy decode fast path.
  - Mixed greedy/sampling path with temperature-aware sampling.

- `nanovllm/lm_eval_adapter.py`
  - `lm_eval` adapter registration for `nano_vllm`.
  - GSM8K prompt and output post-processing used for local accuracy checks.

- `sitecustomize.py`
  - Opportunistic `lm_eval` adapter registration when `lm_eval` is installed.

## Environment

The measured local environment used for the results below was:

```text
Python 3.12
CUDA 12.8
PyTorch 2.8.0
NVIDIA H20
```

Model weights are not included in this repository. Use a local model directory
and add the repository root to `PYTHONPATH` when running from source:

```bash
cd /workspace/gemma4-nanovllm
export PYTHONPATH=/workspace/gemma4-nanovllm:${PYTHONPATH}
```

## Accuracy Evaluation

Example GSM8K evaluation command:

```bash
lm_eval \
  --model nano_vllm \
  --model_args pretrained=/models/gemma-4-26B-A4B,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
  --tasks gsm8k \
  --batch_size 16
```

Local validation result:

```text
gsm8k flexible-extract exact_match = 0.7491
gsm8k strict-match exact_match    = 0.7475
```

## Performance Benchmark

The measured throughput result below used this fixed benchmark configuration:

```text
concurrency = 10
input_len = 1024
output_len = 100
variance = 0.5
shared_prefix_len = 256
num_requests = 300
```

Local benchmark result:

```text
elapsed_s = 163.75
generated_tokens = 29812
throughput_tok_s = 182.05
end_to_end_tok_s = 2018.55
```

## Project Structure

```text
README.md
LICENSE
sitecustomize.py
nanovllm/
  __init__.py
  config.py
  llm.py
  lm_eval_adapter.py
  sampling_params.py
  engine/
  layers/
  models/
  utils/
```

## Acknowledgements

This project is developed based on
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). The
upstream project is licensed under the MIT License.

The upstream copyright notice and license text are preserved in `LICENSE`.

---

## 中文版

## 项目概述

本仓库是在
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
基础上进行二次开发的推理优化项目。项目保留 nano-vLLM 轻量、紧凑的运行时风格，
并围绕 Gemma4-26B-A4B 的单卡推理进行了模型适配与性能路径优化。

本项目重点覆盖 Gemma4 模型加载、Prefill 与 Decode 执行路径、MoE 路由与专家执行、
KV Cache 布局，以及 Decode 阶段使用的 Triton kernel。该仓库定位为面向 Gemma4-MoE
的推理优化工程，而不是从零实现的完整推理引擎。

## 主要特性

- 支持 Gemma4-26B-A4B 文本模型结构与权重映射。
- 支持 RoPE、RMSNorm、滑动窗口注意力以及 Gemma4 注意力结构。
- 支持面向共享前缀 Decode batch 的 Prefix KV Cache。
- 实现带滑动窗口处理的 Triton Split-K Paged Decode Attention。
- 实现 Decode 阶段使用的 Triton Router Top-k 与 MoE Decode Kernel。
- 支持 Batch/Block 二维 CUDA Graph bucket，用于 Decode replay。
- 支持 Last-token logits 与 Greedy Sampler 快路径。
- 通过 `sitecustomize.py` 与 `nanovllm/lm_eval_adapter.py` 注册 `lm_eval` 模型适配器。

## 相对上游 nano-vLLM 的主要修改

主要改动集中在以下文件：

- `nanovllm/models/gemma4.py`
  - 适配 Gemma4 模型结构。
  - 实现 Gemma4 RoPE、RMSNorm、Attention、MLP、Router 与 Expert 模块。
  - 实现 Triton Router Top-k 与 MoE Decode Kernel。

- `nanovllm/layers/attention.py`
  - 实现 Split-K Paged Decode Attention。
  - 在 Prefill 与 Decode 路径中支持 Sliding Window。
  - 支持基于 Paged KV Cache 的 Batched Decode 处理。

- `nanovllm/engine/model_runner.py`
  - 构建 Gemma4 模型执行路径。
  - 适配 Gemma4 KV Cache 分配与上下文准备逻辑。
  - 为 Decode 阶段捕获 Batch/Block CUDA Graph bucket。
  - 传递共享前缀 Decode 所需的运行时元数据。

- `nanovllm/layers/sampler.py`
  - 实现 Greedy Decode 快路径。
  - 支持 Greedy 与温度采样混合场景。

- `nanovllm/lm_eval_adapter.py`
  - 注册 `nano_vllm` 的 `lm_eval` 适配器。
  - 提供 GSM8K 本地准确率验证使用的 prompt 与输出后处理逻辑。

- `sitecustomize.py`
  - 当环境中安装了 `lm_eval` 时，自动尝试注册适配器。

## 运行环境

以下结果对应的本地环境为：

```text
Python 3.12
CUDA 12.8
PyTorch 2.8.0
NVIDIA H20
```

本仓库不包含模型权重。运行源码时，请使用本地模型目录，并将仓库根目录加入
`PYTHONPATH`：

```bash
cd /workspace/gemma4-nanovllm
export PYTHONPATH=/workspace/gemma4-nanovllm:${PYTHONPATH}
```

## 准确率验证

GSM8K 验证命令示例：

```bash
lm_eval \
  --model nano_vllm \
  --model_args pretrained=/models/gemma-4-26B-A4B,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
  --tasks gsm8k \
  --batch_size 16
```

本地验证结果：

```text
gsm8k flexible-extract exact_match = 0.7491
gsm8k strict-match exact_match    = 0.7475
```

## 性能基准

以下吞吐结果使用固定基准配置获得：

```text
concurrency = 10
input_len = 1024
output_len = 100
variance = 0.5
shared_prefix_len = 256
num_requests = 300
```

本地基准结果：

```text
elapsed_s = 163.75
generated_tokens = 29812
throughput_tok_s = 182.05
end_to_end_tok_s = 2018.55
```

## 项目结构

```text
README.md
LICENSE
sitecustomize.py
nanovllm/
  __init__.py
  config.py
  llm.py
  lm_eval_adapter.py
  sampling_params.py
  engine/
  layers/
  models/
  utils/
```

## 致谢

本项目基于
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
开发。上游项目采用 MIT License。

上游项目的版权声明与许可证文本已保留在 `LICENSE` 中。
