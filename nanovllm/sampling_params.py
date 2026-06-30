from dataclasses import dataclass


@dataclass(slots=True, init=False)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = -1
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    stop_regex: str | None = None

    def __init__(
        self,
        temperature: float = 1.0,
        max_tokens: int = 64,
        ignore_eos: bool = False,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        stop_regex: str | None = None,
        **kwargs,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.stop_regex = stop_regex
        self.__post_init__()

    def __post_init__(self):
        assert self.temperature >= 0.0
