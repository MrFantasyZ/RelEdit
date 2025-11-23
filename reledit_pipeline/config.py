"""
Basic configuration dataclass for the relation-aware pipeline.
Tune values per environment; this file is intentionally lightweight.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # Paths
    counterfact_path: str = "data/counterfact.json"
    output_dir: str = "rgs_dataset"
    prompt_templates_path: str = "rgs_dataset/metadata/prompt_templates.json"

    # KG search / graph extraction
    max_path_len: int = 2  # k in the design doc
    expansion_hops: int = 1  # n in the design doc
    allowed_relations: Optional[List[str]] = None  # if None, allow all
    allowed_entity_types: Optional[List[str]] = None  # wikidata types (QIDs)

    # Subspace construction
    max_rel_vectors: int = 256  # cap per fact for K_rel sampling
    k0_sample_size: int = 100_000
    svd_var_threshold: float = 0.95

    # Generation
    num_edits: int = 2
    questions_per_edit: int = 2
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 512

    # Misc
    seed: int = 42
    device: str = "cuda"
    verbose: bool = True
    disable_network: bool = False  # set True if Wikidata access is blocked

    # RelEdit weights
    alpha_rel: float = 0.5  # weight for relation subspace vs null-space

    extra: dict = field(default_factory=dict)
