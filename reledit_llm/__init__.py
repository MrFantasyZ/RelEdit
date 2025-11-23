"""
RelEdit LLM-generated subspace pipeline.

This package implements the LLM-only workflow described in
RelEdit_LLM_Generated_Subspace_Pipeline.txt.
It focuses on:
- CounterFact loading and extension records.
- LLM prompt builders for type extraction, path generation, entity expansion, and question generation.
- Minimal subspace scaffolding (triple verbalization, vector extraction stubs).
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
