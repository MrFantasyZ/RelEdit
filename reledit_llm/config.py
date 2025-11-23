"""
Lightweight configuration for the LLM-generated subspace pipeline.
Tune values per environment and experiments.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # Data
    counterfact_path: str = "data/counterfact.json"
    output_dir: str = "reledit_outputs"
    limit: Optional[int] = None
    offset: int = 0

    # Generation parameters
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512
    num_paths: int = 5
    num_questions: int = 4
    device: str = "cuda"

    # Subspace parameters
    hidden_dim: int = 4096
    svd_var_threshold: float = 0.9
    max_triples_for_subspace: int = 64

    # Misc
    seed: int = 42
    verbose: bool = True
    extra: dict = field(default_factory=dict)
