"""
FFN key extraction aligned with AlphaEdit/MEMIT: capture MLP input at a chosen layer.
Uses repr_tools.get_reprs_at_word_tokens to get token-level representations.
"""
from typing import List
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome import repr_tools

# Lazy globals
_MODEL = None
_TOKENIZER = None


def init_model(model_name: str, device: str = "cuda"):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    ).eval()


def extract_keys(
    sentences: List[str],
    model_name: str,
    device: str = "cuda",
    layer: int = 4,
    module_template: str = "model.layers.{}",
) -> np.ndarray:
    """
    Extract MLP input representations for the last token of each sentence at the specified layer.
    Returns array of shape (hidden_dim, num_sentences).
    """
    if _MODEL is None or _TOKENIZER is None:
        init_model(model_name, device=device)
    keys = []
    for sent in sentences:
        ctxs = [sent]
        idxs = [[-1]]  # last token index list
        reps = repr_tools.get_reprs_at_idxs(
            _MODEL,
            _TOKENIZER,
            contexts=ctxs,
            idxs=idxs,
            layer=layer,
            module_template=module_template,
            track="in",  # MLP input
        )
        rep = reps[0][0, 0, :].float().cpu().numpy()
        keys.append(rep)
    if not keys:
        return np.zeros((0, 0))
    return np.stack(keys, axis=1)
