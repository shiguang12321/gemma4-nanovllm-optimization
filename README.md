# Gemma4-MoE Inference Optimization Based on nano-vLLM

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
