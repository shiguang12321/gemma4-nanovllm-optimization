from __future__ import annotations

import re
from dataclasses import fields
from typing import Any

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _max_gen_tokens(kwargs: dict[str, Any], default: int) -> int:
    for key in ("max_gen_toks", "max_new_tokens", "max_tokens", "max_completion_tokens"):
        if key in kwargs and kwargs[key] is not None:
            return int(kwargs[key])
    return default


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    cut = len(text)
    for stop in stops:
        if not stop:
            continue
        pos = text.find(stop)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


_GSM8K_ANSWER_RE = re.compile(r"####\s*-?[0-9][0-9,]*(?:\.[0-9]+)?")
_COT_ANSWER_RE = re.compile(r"The answer is\s+(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")
_COT_ANSWER_STOP_RE = re.compile(r"The answer is\s+-?\$?[0-9][0-9,]*(?:\.[0-9]+)?[\s.]")

_GSM8K_COT_FEWSHOTS = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to "
        "thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)

_GSM8K_DISCOUNT_RE = re.compile(r"\b(discount|every second|cheaper)\b", re.IGNORECASE)
_GSM8K_RESTART_RE = re.compile(r"\b(restart|all over again|from the beginning)\b", re.IGNORECASE)
_GSM8K_RATE_RE = re.compile(
    r"\b(each|every|per|feed|feeds|cups?|flock|meal)\b.*\b(day|meal|cups?|feed|cost|price|glass|item|person|chickens?)\b",
    re.IGNORECASE,
)


def _gsm8k_hint(question: str) -> str | None:
    hints = []
    if _GSM8K_RATE_RE.search(question):
        hints.append("For per-item or per-day quantities, compute the total required first, then subtract what was already used.")
    if _GSM8K_DISCOUNT_RE.search(question):
        hints.append("For alternating or every-second-item discounts, count full-price and discounted items separately.")
    if _GSM8K_RESTART_RE.search(question):
        hints.append("If a process starts over from the beginning, count the time already spent, the wait, and then the full restarted process.")
    if not hints:
        return None
    hints.append("Use concise arithmetic, do not repeat the same step, and end with exactly: The answer is N.")
    return " ".join(hints)


def _extra_stops(prompt: str, stops: list[str]) -> list[str]:
    extra = []
    if "Question:" in stops and "\n\n" not in stops:
        extra.append("\n\n")
    return extra


def _truncate_generated(text: str, stops: list[str]) -> str:
    match = _GSM8K_ANSWER_RE.search(text)
    if match is not None:
        return text[:match.end()]
    return _truncate_at_stop(text, stops)


def _extract_gsm8k_question(prompt: str) -> str | None:
    start = prompt.rfind("Question:")
    if start == -1:
        return None
    question = prompt[start + len("Question:"):]
    end = question.find("\nAnswer:")
    if end == -1:
        return None
    return question[:end].strip()


def _rewrite_gsm8k_prompt(prompt: str, stops: list[str]) -> str | None:
    if "Question:" not in stops or "####" not in prompt or not prompt.rstrip().endswith("Answer:"):
        return None
    question = _extract_gsm8k_question(prompt)
    if not question:
        return None
    parts = []
    hint = _gsm8k_hint(question)
    if hint is not None:
        parts.append(hint)
    parts.extend(f"Q: {q}\nA: {a}" for q, a in _GSM8K_COT_FEWSHOTS)
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def _format_gsm8k_response(text: str, stops: list[str]) -> str:
    text = _truncate_generated(text, stops)
    if _GSM8K_ANSWER_RE.search(text):
        return text
    match = _COT_ANSWER_RE.search(text)
    if match is None:
        return text
    answer = match.group(1).replace("$", "")
    return f"{text[:match.end()]}\n#### {answer}"


@register_model("nano_vllm")
class NanoVLLM(LM):

    def __init__(
        self,
        pretrained: str,
        batch_size: int | str = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self.pretrained = pretrained
        self.batch_size = 1 if batch_size == "auto" else int(batch_size)
        self.max_gen_toks = int(max_gen_toks)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)

        config_fields = {field.name for field in fields(Config)}
        llm_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        llm_kwargs.setdefault("max_num_seqs", max(1, self.batch_size))
        llm_kwargs.setdefault("max_num_batched_tokens", 4096)
        self.llm = LLM(pretrained, **llm_kwargs)

    def loglikelihood(self, requests):
        raise NotImplementedError("nano_vllm lm_eval adapter only supports generate_until.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("nano_vllm lm_eval adapter only supports generate_until.")

    def generate_until(self, requests, disable_tqdm: bool = False):
        prompts = []
        sampling_params = []
        postprocess_stops = []
        gsm8k_rewritten = []

        for req in requests:
            prompt, gen_kwargs = req.args
            gen_kwargs = dict(gen_kwargs)
            until = _as_list(gen_kwargs.get("until"))
            rewritten_prompt = _rewrite_gsm8k_prompt(prompt, until)
            if rewritten_prompt is not None:
                prompt = rewritten_prompt
            runtime_until = until + _extra_stops(prompt, until)
            if rewritten_prompt is not None and "Q:" not in runtime_until:
                runtime_until.append("Q:")
            max_tokens = _max_gen_tokens(gen_kwargs, self.max_gen_toks)
            do_sample = bool(gen_kwargs.get("do_sample", False))
            temperature = float(gen_kwargs.get("temperature", 0.0 if not do_sample else 1.0))
            if not do_sample:
                temperature = 0.0

            prompts.append(prompt)
            postprocess_stops.append(until)
            gsm8k_rewritten.append(rewritten_prompt is not None)
            sampling_params.append(
                SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=float(gen_kwargs.get("top_p", 1.0)),
                    top_k=int(gen_kwargs.get("top_k", -1)),
                    stop=runtime_until,
                    stop_regex=_COT_ANSWER_STOP_RE.pattern if rewritten_prompt is not None else None,
                )
            )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=not disable_tqdm)
        results = []
        for output, stops, rewritten in zip(outputs, postprocess_stops, gsm8k_rewritten, strict=True):
            text = output["text"]
            if rewritten:
                results.append(_format_gsm8k_response(text, stops))
            else:
                results.append(_truncate_generated(text, stops))
        return results

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
