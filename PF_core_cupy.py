"""
PF_core_cupy.py — Bootstrap Particle Filter with CuPy on-device state.

Parallel to :mod:`PF_core` for GPU-accelerated assimilation. Particle arrays and
PF linear algebra live on the default CUDA device; truth integration (DAPyr) and
observation draws use NumPy on CPU so ``obs_seed`` matches :mod:`PF_core` /
:mod:`EnKF_core` benchmarks.

Surrogate forecasts use :meth:`SurrogateModel.batch_rollout_tensor` and a
DLPack bridge to CuPy — no per-cycle host round-trip for rollout outputs.

Filter-internal stochasticity (subset draw, systematic resampling, jitter) uses
the same ``numpy.random.RandomState`` and helpers as :mod:`PF_core`, so
``filter_seed`` matches :func:`PF_core.run_pf` when the forecast callable
returns identical particle updates.
"""

from __future__ import annotations

import numpy as np
import torch
import cupy as cp
from tqdm import tqdm

from lorenz.lorenz_systems import LorenzSystems
from PF_core import (
    PFConfig,
    _jitter_ensemble,
    _raw_C12,
    _scott_bandwidth,
    systematic_resample,
)
from metrics_cupy import energy_score_cupy


def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """Zero-copy CuPy view of a contiguous CUDA tensor (DLPack; avoids H2D after surrogate rollout)."""
    if not t.is_cuda:
        raise ValueError("torch_to_cupy requires a CUDA tensor.")
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))


def _to_numpy_result(result: dict) -> dict:
    """Convert all CuPy arrays in a PF result dict to NumPy."""
    out = {}
    for key, val in result.items():
        if isinstance(val, cp.ndarray):
            out[key] = cp.asnumpy(val)
        else:
            out[key] = val
    return out


# ============================================================
# Forecast wrappers (GPU particle state)
# ============================================================
def make_surrogate_forecaster_gpu(surrogate, M):
    """Wrap SurrogateModel for CuPy ensemble layout ``(Nx, Ne)``.

    ``states`` columns are ensemble members; each gets a ``prev_time_steps=1`` history,
    ``M`` autoregressive steps on GPU via ``batch_rollout_tensor``. Returns full trajectories
    ``(Ne, M+1, Nx)`` with time 0 = analysis (current column).
    """
    def forecaster(states):
        if states.ndim != 2:
            raise ValueError(f"Expected (Nx, Ne), got shape {states.shape}")
        Ne = states.shape[1]
        Nx = states.shape[0]
        # PF_core uses (Nx, Ne); surrogate API expects batch-major (Ne, prev_steps, Nx).
        hist = states.T.reshape(Ne, 1, Nx)
        traj_t = surrogate.batch_rollout_tensor(hist, num_steps=M)
        sol = torch_to_cupy(traj_t)
        out = cp.empty((Ne, M + 1, Nx), dtype=sol.dtype)
        out[:, 0, :] = states.T
        out[:, 1:, :] = sol
        return out
    return forecaster


def make_lorenz_forecaster_gpu(dt, M, system_type='63'):
    """Lorenz integrator on CPU; returns CuPy trajectories ``(Ne, M+1, Nx)``.

    DAPyr/numba path is particle-serial here (accuracy baseline, not throughput-optimized).
    """
    def forecaster(states):
        states_np = cp.asnumpy(states)
        if states_np.ndim != 2:
            raise ValueError(f"Expected (Nx, Ne), got shape {states_np.shape}")
        trajs = [
            LorenzSystems.generate_trajectory_fast(
                system_type, states_np[:, e], dt, M + 1)
            for e in range(states_np.shape[1])
        ]
        return cp.asarray(np.stack(trajs, axis=0))
    return forecaster


# ============================================================
# Bootstrap PF runner (GPU)
# ============================================================
def run_pf_cupy(forecast_fn, xt_0, Xf_pool, config: PFConfig, model_name=None):
    """
    Run bootstrap PF with particle state on the default CuPy CUDA device.

    Parameters match :func:`PF_core.run_pf`. ``Xf_pool`` may be NumPy or CuPy
    ``(Ne_total, Nx)``. Returned dict values are **NumPy** on host for plotting.
    """
    cfg = config

    # Split streams so obs locations/noise reproducibly match PF_core; resampling/jitter matches PF_core helpers.
    obs_rng = np.random.RandomState(cfg.obs_seed)
    filt_rng = np.random.RandomState(cfg.filter_seed)

    Nx = xt_0.shape[0]
    I = np.eye(Nx)

    if isinstance(Xf_pool, cp.ndarray):
        pool = Xf_pool
    else:
        pool = cp.asarray(Xf_pool)

    # Subsample ``Ne`` columns from the pre-spinup GPU pool (same pattern as PF_core.run_pf).
    ind = filt_rng.permutation(pool.shape[0])[:cfg.Ne]
    Xf_k = pool[ind, :].T.copy()
    w = cp.full(cfg.Ne, 1.0 / cfg.Ne, dtype=cp.float64)
    xt_k = np.asarray(xt_0, dtype=np.float64).copy()

    Ny = int(round(cfg.p * Nx))
    R_inv_diag = 1.0 / (cfg.sig_obs ** 2)

    # Per-cycle diagnostics: GPU tensors for weighted moments vs truth; xt_traj etc. stay NumPy on CPU.
    errorf = cp.zeros(cfg.T)
    errora = cp.zeros(cfg.T)
    spread = cp.zeros(cfg.T)
    errorf_es = cp.zeros(cfg.T)
    errora_es = cp.zeros(cfg.T)
    errorf_es_acc = cp.zeros(cfg.T)
    errorf_es_spr = cp.zeros(cfg.T)
    errora_es_acc = cp.zeros(cfg.T)
    errora_es_spr = cp.zeros(cfg.T)
    xt_traj = np.zeros((cfg.T, Nx))
    xf_traj = np.zeros((cfg.T, Nx))
    xa_traj = np.zeros((cfg.T, Nx))
    Xf_all = cp.zeros((cfg.T, Nx, cfg.Ne))
    Xa_all = cp.zeros((cfg.T, Nx, cfg.Ne))
    weights_f = cp.zeros((cfg.T, cfg.Ne))
    weights_a = cp.zeros((cfg.T, cfg.Ne))
    n_eff = cp.zeros(cfg.T)
    resampled = cp.zeros(cfg.T, dtype=cp.bool_)
    # Stored forecast tubes: ensemble on device; truth trajectory from CPU integrator each cycle.
    ens_fcst_traj = cp.full((cfg.T, cfg.Ne, cfg.M + 1, Nx), cp.nan)
    truth_fcst_traj = np.full((cfg.T, cfg.M + 1, Nx), np.nan)

    # Duplicate truth on GPU for fused RMSE / energy_score without repeated small H2D in the inner loop.
    xt_k_gpu = cp.asarray(xt_k)

    diverged = False

    def _fill_nan_from(k):
        for arr in (errorf, errora, spread,
                    errorf_es, errora_es,
                    errorf_es_acc, errorf_es_spr,
                    errora_es_acc, errora_es_spr,
                    n_eff):
            arr[k:] = cp.nan

    desc = f"{model_name} | PF DA (CuPy)" if model_name else "PF DA cycles (CuPy)"
    for k in tqdm(range(cfg.T), desc=desc, leave=False):
        if cp.any(cp.isnan(Xf_k)) or cp.any(cp.abs(Xf_k) > 1e6):
            print(f"  WARNING: Particle ensemble diverged at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break

        xf_k = Xf_k @ w
        diffs = Xf_k - xf_k[:, None]
        var_w = (diffs ** 2) @ w
        # Per-state weighted variance averaged over dims (spread-skill bookkeeping).
        spread[k] = float(cp.mean(cp.sqrt(var_w)))

        errorf[k] = float(cp.linalg.norm(xf_k - xt_k_gpu) / cp.sqrt(Nx))

        es_f, acc_f, spr_f = energy_score_cupy(Xf_k, xt_k_gpu)
        errorf_es[k] = es_f
        errorf_es_acc[k] = acc_f
        errorf_es_spr[k] = spr_f

        xt_traj[k] = xt_k
        xf_traj[k] = cp.asnumpy(xf_k)
        Xf_all[k] = Xf_k.copy()
        weights_f[k] = w.copy()

        obs_comp = obs_rng.permutation(Nx)[:Ny]
        H_k = I[obs_comp, :]
        y_k = H_k @ xt_k + cfg.sig_obs * obs_rng.standard_normal(Ny)
        y_k_gpu = cp.asarray(y_k)

        H_gpu = cp.asarray(H_k)
        innov = y_k_gpu[:, None] - H_gpu @ Xf_k
        loglik = -0.5 * R_inv_diag * cp.sum(innov ** 2, axis=0)
        loglik = cp.where(cp.isfinite(loglik), loglik, -cp.inf)
        finite_mask = cp.isfinite(loglik)
        if not bool(cp.any(finite_mask)):
            print(f"  WARNING: All particles have zero likelihood at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break
        # Subtract max log-likelihood before exp() for numeric stability (same scaling on all weights).
        loglik_max = float(cp.max(loglik[finite_mask]))
        loglik = loglik - loglik_max
        w_new = w * cp.exp(loglik)
        total = float(cp.sum(w_new))
        if (not np.isfinite(total)) or total <= 0.0:
            print(f"  WARNING: Filter weight collapse at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break
        w = w_new / total

        n_eff_k = float(1.0 / cp.sum(w ** 2))
        n_eff[k] = n_eff_k

        Xa_k = Xf_k.copy()
        # Effective sample-size threshold triggers systematic resampling + optional kernel jitter (PF_core parity).
        if n_eff_k <= cfg.NER * cfg.Ne:
            Xa_np = cp.asnumpy(Xa_k)
            w_np = cp.asnumpy(w)
            C12 = cfg.reg * _scott_bandwidth(cfg.Ne, Nx) * _raw_C12(Xa_np, w_np)
            idx = systematic_resample(w_np, filt_rng)
            Xa_k = Xa_k[:, idx]
            w = cp.full(cfg.Ne, 1.0 / cfg.Ne, dtype=cp.float64)
            resampled[k] = True
            if cfg.jitter and cfg.reg > 0:
                Xa_k = cp.asarray(
                    _jitter_ensemble(cp.asnumpy(Xa_k), idx, C12, cfg.nuj, filt_rng)
                )

        xa_k = Xa_k @ w
        if cfg.use_inflation:
            DXa_k = Xa_k - xa_k[:, None]
            Xa_k = xa_k[:, None] + cfg.infl_factor * DXa_k
            xa_k = Xa_k @ w

        errora[k] = float(cp.linalg.norm(xa_k - xt_k_gpu) / cp.sqrt(Nx))
        xa_traj[k] = cp.asnumpy(xa_k)
        Xa_all[k] = Xa_k.copy()
        weights_a[k] = w.copy()

        es_a, acc_a, spr_a = energy_score_cupy(Xa_k, xt_k_gpu)
        errora_es[k] = es_a
        errora_es_acc[k] = acc_a
        errora_es_spr[k] = spr_a

        traj_e = forecast_fn(Xa_k)
        ens_fcst_traj[k, :, :, :] = traj_e
        # Last forecast time slice becomes forecast particles for the next cycle (bootstrap structure).
        Xf_k = traj_e[:, -1, :].T

        if cfg.sig_dyn > 0.0:
            # D shape (Nx, Ne) — matches PF_core_cupy column-per-particle layout
            D = cp.asarray(
                filt_rng.standard_normal((Nx, cfg.Ne))
            )
            # Add inflated noise: N(0, qroot * sig_dyn^2 * I) instead of N(0, sig_dyn^2 * I)
            Xf_k = Xf_k + np.sqrt(cfg.qroot) * cfg.sig_dyn * D

            if cfg.qroot != 1.0:
                # Importance weight correction: log p/q = -0.5 ||d||^2 (1 - 1/alpha)
                # Sum over state dimension (axis=0) to get one scalar per particle
                log_corr = (
                    -0.5 * cp.sum(D ** 2, axis=0) * (1.0 - 1.0 / cfg.qroot)
                )
                # Subtract max before exp() for numerical stability
                log_corr -= float(cp.max(log_corr))
                w = w * cp.exp(log_corr)
                w = w / cp.sum(w)

        xt_next = LorenzSystems.generate_trajectory_fast(
            '63', xt_k, cfg.dt, cfg.M + 1
        )
        truth_fcst_traj[k, :, :] = xt_next
        xt_k = xt_next[-1, :]
        xt_k_gpu = cp.asarray(xt_k)

    result = {
        'errorf': errorf,
        'errora': errora,
        'spread': spread,
        'errorf_es': errorf_es,
        'errora_es': errora_es,
        'errorf_es_acc': errorf_es_acc,
        'errorf_es_spr': errorf_es_spr,
        'errora_es_acc': errora_es_acc,
        'errora_es_spr': errora_es_spr,
        'xt_traj': xt_traj,
        'xf_traj': xf_traj,
        'xa_traj': xa_traj,
        'Xf_all': Xf_all,
        'Xa_all': Xa_all,
        'ens_fcst_traj': ens_fcst_traj,
        'truth_fcst_traj': truth_fcst_traj,
        'weights_f': weights_f,
        'weights_a': weights_a,
        'n_eff': n_eff,
        'resampled': resampled,
        'diverged': diverged,
    }
    return _to_numpy_result(result)
