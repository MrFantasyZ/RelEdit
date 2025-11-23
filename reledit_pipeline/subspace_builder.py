"""
Subspace construction scaffolding for RelEdit.

Provides placeholders for:
- Building global K0 and its null-space projection P_null.
- Building per-fact relation subspace K_rel and projection P_rel(fact).

Implementations should plug in the actual model forward hooks to extract
layer activations (e.g., FFN keys) and perform SVD/PCA on the collected
vectors. This scaffolding only defines interfaces and simple stubs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SubspaceArtifacts:
    k0_path: Path
    p_null_path: Path
    k_rel_path: Optional[Path] = None
    p_rel_path: Optional[Path] = None


def save_projection(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def compute_null_space_projection(k0: np.ndarray, var_threshold: float = 0.95) -> np.ndarray:
    """
    Given K0 ∈ R^{d×N}, compute projection onto (approx) null-space.
    """
    u, s, _ = np.linalg.svd(k0, full_matrices=False)
    # Keep components with small singular values
    total = np.sum(s)
    keep = s <= (1 - var_threshold) * total / len(s)
    u_null = u[:, keep] if keep.any() else u[:, -1:]
    return u_null @ u_null.T


def compute_span_projection(k_rel: np.ndarray, var_threshold: float = 0.95) -> np.ndarray:
    """
    Projection onto the span (principal components) of relation-specific vectors.
    """
    u, s, _ = np.linalg.svd(k_rel, full_matrices=False)
    if var_threshold < 1.0:
        cum = np.cumsum(s) / np.sum(s)
        r = np.searchsorted(cum, var_threshold) + 1
        u = u[:, :r]
    return u @ u.T


def combine_projections(p_null: np.ndarray, p_rel: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    P'(fact) = P_null + alpha * P_rel; caller may orthogonalize if needed.
    """
    return p_null + alpha * p_rel


def placeholder_extract_keys(prompts: List[str]) -> np.ndarray:
    """
    Placeholder: replace with actual model forward pass to extract key vectors.
    """
    # Fake keys for scaffolding; shape (hidden_dim, num_prompts)
    d = 16
    keys = np.random.randn(d, len(prompts))
    return keys


def build_k_rel_for_fact(prompts: List[str], out_dir: Path, fact_id: str, var_threshold: float = 0.95) -> SubspaceArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    k_rel = placeholder_extract_keys(prompts)
    p_rel = compute_span_projection(k_rel, var_threshold=var_threshold)
    k_rel_path = out_dir / f"{fact_id}_k_rel.npy"
    p_rel_path = out_dir / f"{fact_id}_p_rel.npy"
    save_projection(k_rel, k_rel_path)
    save_projection(p_rel, p_rel_path)
    return SubspaceArtifacts(k0_path=Path(), p_null_path=Path(), k_rel_path=k_rel_path, p_rel_path=p_rel_path)
