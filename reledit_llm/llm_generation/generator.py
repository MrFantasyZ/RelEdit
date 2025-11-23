"""
LLM generation helpers with retry for JSON outputs.
You can plug in any callable `llm_fn(prompt: str) -> str`.
"""
from typing import Callable, Any, List
from .json_utils import try_parse_json


class LLMGenerator:
    def __init__(self, llm_fn: Callable[[str], str], max_retries: int = 3):
        self.llm_fn = llm_fn
        self.max_retries = max_retries

    def generate_json(self, prompt: str) -> Any:
        last_err = None
        for _ in range(self.max_retries):
            try:
                return try_parse_json(self.llm_fn(prompt))
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to parse LLM JSON after {self.max_retries} tries: {last_err}")


def dummy_llm_fn(prompt: str) -> str:
    """
    Placeholder LLM function. Replace with actual model call.
    """
    return "{}"
