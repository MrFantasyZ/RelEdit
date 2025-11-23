"""
Simple Hugging Face generation wrapper for meta-llama/Llama-3.1-8B-Instruct (or compatible chat models).
Uses the model's chat template to format the prompt and returns the generated text.
"""
from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


@lru_cache(maxsize=1)
def load_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    )
    return tokenizer, model


class HFChatGenerator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.tokenizer, self.model = load_model(model_name, device=device)
        self.gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
        )

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs ONLY JSON when asked."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, generation_config=self.gen_config)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Heuristic: remove the original prompt portion
        try:
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            if text.startswith(prompt_text):
                text = text[len(prompt_text) :].strip()
        except Exception:
            pass
        return text.strip()


def get_hidden_size(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda") -> int:
    tokenizer, model = load_model(model_name, device=device)
    return model.config.hidden_size
