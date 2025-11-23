"""
Subspace utilities: SVD/PCA to build relational subspaces from extracted vectors.
"""
from pathlib import Path
from typing import Tuple
import numpy as np


def svd_basis(matrix: np.ndarray, var_threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if var_threshold < 1.0:
        cum = np.cumsum(s) / np.sum(s)
        r = np.searchsorted(cum, var_threshold) + 1
        u = u[:, :r]
        s = s[:r]
    return u, s


def save_basis(basis: np.ndarray, variance: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, basis=basis, variance=variance)
