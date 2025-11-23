"""
AlphaEdit runner that wires official AlphaEdit code with P_null constructed
from precomputed covariance (mom2) statistics.

Key points:
- Uses AlphaEdit's closed-form update (no gradient training).
- Loads cov stats via AlphaEdit's get_cov; expects official stats under STATS_DIR.
- Builds per-layer P_null from small eigenvalues of the covariance matrix.
- Initializes cache_c as zeros and accumulates it across sequential edits.
"""
from typing import List, Tuple
import torch

from AlphaEdit.AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from AlphaEdit.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from util import nethook


def build_p_null_from_cov(
    cov: torch.Tensor, nullspace_threshold: float
) -> torch.Tensor:
    """
    Given covariance Σ, compute projection onto approximate null-space.
    Eigenvectors with eigenvalues <= threshold * max_eig are retained.
    """
    # cov is on cuda; move to cpu for eig if needed
    cov_cpu = cov.float().cpu()
    eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
    max_eig = eigvals.max()
    mask = eigvals <= (nullspace_threshold * max_eig)
    if mask.sum() == 0:
        # fallback: keep the smallest eigenvector
        min_idx = torch.argmin(eigvals)
        mask[min_idx] = True
    u_null = eigvecs[:, mask]  # [d, k]
    p_null = u_null @ u_null.T  # [d, d]
    return p_null


def load_p_null_mats(
    model, tok, hparams: AlphaEditHyperParams
) -> torch.Tensor:
    """
    For each layer in hparams.layers, load covariance stats and build P_null.
    Returns a tensor P of shape [num_layers, d, d].
    """
    p_list: List[torch.Tensor] = []
    for layer in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer)
        cov = get_cov(
            model,
            tok,
            layer_name=layer_name,
            mom2_dataset=hparams.mom2_dataset,
            mom2_n_samples=hparams.mom2_n_samples,
            mom2_dtype=hparams.mom2_dtype,
            inv=False,
            force_recompute=False,
        )
        p_null = build_p_null_from_cov(cov, hparams.nullspace_threshold)
        p_list.append(p_null)
    # pad to common shape if needed (should all match)
    return torch.stack(p_list, dim=0)


def init_cache_c(p_mats: torch.Tensor) -> torch.Tensor:
    """
    Initialize cache_c as zeros with same shapes as P_null per layer.
    """
    return torch.zeros_like(p_mats)


def run_alphaedit_once(
    model,
    tok,
    requests: List[dict],
    hparams: AlphaEditHyperParams,
    p_mats: torch.Tensor,
    cache_c: torch.Tensor,
) -> Tuple[any, torch.Tensor]:
    """
    Apply AlphaEdit closed-form update on the given requests.
    Returns updated model and updated cache_c.
    """
    # Ensure model is in eval, no grads
    model.eval()
    nethook.set_requires_grad(False, model)
    # AlphaEdit expects P and cache_c on CPU; will move to cuda internally
    model, cache_c = apply_AlphaEdit_to_model(
        model,
        tok,
        requests=requests,
        hparams=hparams,
        cache_template=None,
        cache_c=cache_c,
        P=p_mats,
    )
    return model, cache_c
