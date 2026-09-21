from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # Keep host planning and CPU tests importable without the GPU runtime.
    if name == "LLM":
        from nanovllm.llm import LLM
        globals()[name] = LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
