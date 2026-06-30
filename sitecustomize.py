import importlib.util


if importlib.util.find_spec("lm_eval") is not None:
    try:
        import nanovllm.lm_eval_adapter  # noqa: F401
    except Exception:
        pass
