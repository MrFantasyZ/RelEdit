"""
RelEdit pipeline scaffolding for relation-aware subspace workflow.

This package provides small, composable modules for:
- Loading and splitting CounterFact records.
- Building Wikidata-backed subgraphs (interfaces only; implement per infra).
- Generating constrained question prompts.
- Preparing subspace inputs (K0, K_rel) for RelEdit editing.
"""

from .config import PipelineConfig
from .data_loader import CounterFactRecord, CounterFactDataset

__all__ = [
    "PipelineConfig",
    "CounterFactRecord",
    "CounterFactDataset",
]
