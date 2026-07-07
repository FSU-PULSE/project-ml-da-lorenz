"""
metrics_cupy.py — GPU Energy Score for CuPy particle-filter ensembles.

Mirrors :func:`metrics.energy_score` for ``(Nx, Ne)`` CuPy arrays. Total ES is
computed as ``accuracy - spread`` from the same decomposition (no scoringrules
on device).
"""

from __future__ import annotations

import cupy as cp


def energy_score_cupy(ensemble, truth):
    """Energy Score and accuracy/spread decomposition on GPU.

    Parameters
    ----------
    ensemble : cupy.ndarray, shape (Nx, Ne)
        Variable-first / member-second (matches ``Xf_k``, ``Xa_k``).
    truth : cupy.ndarray, shape (Nx,)
        Verifying state on the same device as ``ensemble``.

    Returns
    -------
    es, acc, spr : float
    """
    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble must be 2-D (Nx, Ne); got shape {ensemble.shape}"
        )
    if truth.ndim != 1:
        raise ValueError(f"truth must be 1-D (Nx,); got shape {truth.shape}")
    if truth.shape[0] != ensemble.shape[0]:
        raise ValueError(
            f"variable dimension mismatch: ensemble Nx={ensemble.shape[0]}, "
            f"truth Nx={truth.shape[0]}"
        )

    if not cp.all(cp.isfinite(ensemble)):
        return float('nan'), float('nan'), float('nan')

    ens = ensemble.T  # (Ne, Nx)
    Ne = ensemble.shape[1]
    diffs = ens - truth[None, :]
    acc = float(cp.mean(cp.linalg.norm(diffs, axis=1)))
    # Pairwise distances (Ne is modest in PF benchmarks)
    pdist = cp.linalg.norm(ens[:, None, :] - ens[None, :, :], axis=2)
    spr = float(cp.sum(pdist) / (2.0 * Ne * Ne))
    es = acc - spr
    return es, acc, spr
