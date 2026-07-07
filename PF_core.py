"""
PF_core.py — Bootstrap Particle Filter for ML-surrogate data assimilation.

Implements a Sequential Importance Resampling (SIR) bootstrap particle filter
with the same interface as :mod:`EnKF_core`, so the two filters can share the
benchmarking and plotting pipeline. The filter follows the standard
formulation in van Leeuwen (2009) and Chen (2003), with practical add-ons
common in geophysical PF implementations:

- **Systematic resampling** (Carpenter–Clifford–Fearnhead 1999) — lowest
  variance among unbiased resampling schemes.
- **Post-resample jittering** — additive Gaussian noise scaled by the
  empirical (weighted) ensemble covariance and a Silverman-style bandwidth,
  to combat sample impoverishment.
- **Multiplicative inflation** — anomaly scaling around the weighted mean,
  the direct analog of EnKF inflation.

The proposal is the prior (bootstrap PF): particles are propagated through
the forecast model with no additional importance-density correction. With
deterministic ML surrogates this means particle diversity comes only from
the initial spread plus jittering after resampling.

Returned dictionary mirrors :func:`EnKF_core.run_enkf` so existing plotting
and table-printing helpers work unchanged.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm
from lorenz.lorenz_systems import LorenzSystems
from metrics import energy_score


# ============================================================
# Particle filter configuration
# ============================================================
class PFConfig:
    """All tunable parameters for a single bootstrap PF experiment.

    Random number streams
    ---------------------
    Same convention as :class:`EnKF_core.EnKFConfig`:

    - ``obs_seed``    : observation realization. Share across filters
                        being compared head-to-head.
    - ``filter_seed`` : filter-internal stochasticity (ensemble subset
                        draw, resampling, jittering).

    The legacy ``seed`` argument is retained as an alias.
    """

    def __init__(
        self,
        T:int=30,                        # number of DA cycles
        M:int=10,                        # forecast steps between assimilation cycles
        Ne:int=20,                       # ensemble (particle) size
        dt:float=0.01,                   # time step
        p:float=1.0,                     # fraction of observed state variables
        sig_obs:float=1.1,               # observation error std
        NER:float=0.5,                   # resample when N_eff <= NER * Ne
        reg:float=0.1,                   # jitter bandwidth scale (0 disables jitter)
        jitter:bool=True,                # post-resample regularization on/off
        nuj:bool=True,                   # "no unique jitter": jitter duplicates only
        use_inflation:bool=False,        # multiplicative anomaly inflation
        infl_factor:float=1.02,          # multiplicative anomaly inflation factor
        seed:int|None=None,              # legacy alias: sets both seeds when provided
        obs_seed:int|None=None,          # observation realization
        filter_seed:int|None=None,       # filter-internal stochasticity
        use_localization:bool=False,     # Legacy field: unused in PF
        qroot:float=1.0,                 # Proposal noise inflation factor (alpha >= 1)
        sig_dyn:float=0.0,               # Std of additive isotropic process noise for qroot
    ):
        self.T = T
        self.M = M
        self.Ne = Ne
        self.dt = dt
        self.p = p
        self.sig_obs = sig_obs
        self.NER = NER
        self.reg = reg
        self.jitter = jitter
        self.nuj = nuj
        self.use_inflation = use_inflation
        self.infl_factor = infl_factor
        self.use_localization = use_localization
        if obs_seed is None:
            obs_seed = seed if seed is not None else 10
        if filter_seed is None:
            filter_seed = seed if seed is not None else 10
        self.obs_seed = obs_seed
        self.filter_seed = filter_seed
        self.seed = seed if seed is not None else obs_seed
        self.qroot = qroot
        self.sig_dyn = sig_dyn


# ============================================================
# Resampling
# ============================================================
def systematic_resample(weights, rng):
    """Systematic resampling — return parent indices for the new ensemble.

    Lowest-variance unbiased resampling scheme (Carpenter, Clifford &
    Fearnhead, 1999): a single uniform draw spawns ``N`` evenly spaced
    "teeth" along the weight CDF, eliminating most sampling noise from
    the weight-to-index mapping.

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Normalized particle weights, summing to 1.
    rng : np.random.RandomState
        Filter-side random stream.

    Returns
    -------
    indices : np.ndarray, shape (N,) of int
        Parent indices for the resampled ensemble.
    """
    N = len(weights)
    positions = (rng.random_sample() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against floating-point overshoot
    indices = np.zeros(N, dtype=np.intp)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


# ============================================================
# Jitter (post-resample regularization)
# ============================================================
def _scott_bandwidth(Ne, Nx):
    """Scott's rule-of-thumb KDE bandwidth (matches dapper.auto_bandw)."""
    return Ne ** (-1.0 / (Nx + 4.0))


def _raw_C12(E, w):
    """Weighted matrix square root of the ensemble covariance.

    Returns C12 of shape (Ne, Nx) such that C12.T @ C12 ≈ weighted cov of E,
    using a weighted mean, weighted anomalies, and the unbiased weight factor
    1 / (1 - Σw²). Mirrors dapper's raw_C12 with a column-particle layout.
    """
    _, Ne = E.shape
    sumw2 = float(np.sum(w * w))
    if (not np.isfinite(sumw2)) or sumw2 >= 1.0 - 1e-12:
        w = np.full(Ne, 1.0 / Ne)
        sumw2 = 1.0 / Ne
    mu = E @ w
    A = E - mu[:, None]
    ub = 1.0 / (1.0 - sumw2)
    return (np.sqrt(ub * w[None, :]) * A).T


def _mask_duplicates_sorted(idx):
    """True at positions whose value equals a neighbour in the sorted index."""
    dups = idx == np.roll(idx, 1)
    dups |= idx == np.roll(idx, -1)
    return dups


def _jitter_ensemble(E, idx, C12, nuj, rng):
    """Add Gaussian jitter using a pre-computed weighted-cov factor.

    E    : (Nx, Ne) post-resample ensemble (already indexed by idx)
    idx  : (Ne,)    resample indices (sorted, from systematic_resample)
    C12  : (Ne, Nx) bandwidth-scaled square root of weighted cov
    nuj  : bool     if True, jitter only duplicates; else jitter all
    """
    _, Ne = E.shape
    rank = C12.shape[0]
    if nuj:
        dups = _mask_duplicates_sorted(idx)
        n = int(dups.sum())
        if n == 0:
            return E
        D = rng.standard_normal((n, rank))
        noise = (D @ C12).T
        out = E.copy()
        out[:, dups] += noise
        return out
    D = rng.standard_normal((Ne, rank))
    noise = (D @ C12).T
    return E + noise


# ============================================================
# Forecast wrappers (kept here for parity with EnKF_core)
# ============================================================
# Surrogate model forecaster
def make_surrogate_forecaster(surrogate, M):
    """Wrap a SurrogateModel into a forecast callable.
    Accepts a single state (Nx,) or an ensemble (Nx, Ne). Returns trajectories
    (M+1, Nx) or (Ne, M+1, Nx), including the initial state.
    """
    def forecaster(states):
        states = np.asarray(states)
        if states.ndim == 1:
            sol = np.asarray(surrogate.rollout(states, num_steps=M))
            return np.vstack([states.reshape(1, -1), sol])
        # EnKF layout (Nx, Ne) -> batch_rollout (Ne, 1, Nx) via flat (Ne, Nx)
        sol = surrogate.batch_rollout(states.T, num_steps=M)  # (Ne, M, Nx)
        out = np.empty((states.shape[1], M + 1, states.shape[0]))
        out[:, 0, :] = states.T
        out[:, 1:, :] = sol
        return out
    return forecaster

# Lorenz system forecaster
def make_lorenz_forecaster(dt, M, system_type='63'):
    """Wrap the Lorenz solver into a forecast callable.

    Accepts a single state (Nx,) or an ensemble (Nx, Ne). Returns trajectories
    (M+1, Nx) or (Ne, M+1, Nx), including the initial state.
    """
    def forecaster(states):
        states = np.asarray(states)
        if states.ndim == 1:
            return LorenzSystems.generate_trajectory_fast(
                system_type, states, dt, M + 1)
        trajs = [
            LorenzSystems.generate_trajectory_fast(
                system_type, states[:, e], dt, M + 1)
            for e in range(states.shape[1])
        ]
        return np.stack(trajs, axis=0)  # (Ne, M+1, Nx)
    return forecaster

# ============================================================
# Bootstrap PF runner
# ============================================================
def run_pf(forecast_fn, xt_0, Xf_pool, config: PFConfig, model_name=None):
    """
    Run a full bootstrap Particle Filter assimilation experiment.

    Parameters
    ----------
    forecast_fn : callable
        ``forecast_fn(state) -> np.ndarray of shape (M+1, Nx)`` — full
        trajectory, including the initial state.
    xt_0 : np.ndarray, shape (Nx,)
        Initial true state (propagated with the real Lorenz solver).
    Xf_pool : np.ndarray, shape (Ne_total, Nx)
        Pre-propagated ensemble pool from which Ne particles are drawn.
    config : PFConfig
        Experiment configuration.

    Returns
    -------
    dict with keys:
        'errorf'         : np.ndarray (T,)  – forecast RMSE per cycle
        'errora'         : np.ndarray (T,)  – analysis RMSE per cycle
        'spread'         : np.ndarray (T,)  – ensemble spread per cycle
        'errorf_es'      : np.ndarray (T,)  – forecast Energy Score per cycle
        'errora_es'      : np.ndarray (T,)  – analysis Energy Score per cycle
        'errorf_es_acc'  : np.ndarray (T,)  – forecast ES accuracy term
        'errorf_es_spr'  : np.ndarray (T,)  – forecast ES spread term
        'errora_es_acc'  : np.ndarray (T,)  – analysis ES accuracy term
        'errora_es_spr'  : np.ndarray (T,)  – analysis ES spread term
        'xt_traj'        : np.ndarray (T, Nx) – true state at each cycle
        'xf_traj'        : np.ndarray (T, Nx) – weighted forecast mean
        'xa_traj'        : np.ndarray (T, Nx) – weighted analysis mean
        'Xf_all'         : np.ndarray (T, Nx, Ne) – forecast particles
        'Xa_all'         : np.ndarray (T, Nx, Ne) – analysis particles
        'ens_fcst_traj'  : np.ndarray (T, Ne, M+1, Nx) – inter-cycle traj
        'truth_fcst_traj': np.ndarray (T, M+1, Nx) – truth over each window
        'weights_f'      : np.ndarray (T, Ne) – pre-update particle weights
        'weights_a'      : np.ndarray (T, Ne) – post-update particle weights
        'n_eff'          : np.ndarray (T,)  – effective sample size per cycle
        'resampled'      : np.ndarray (T,)  of bool – resample fired this cycle
        'diverged'       : bool – whether the filter diverged
    """
    cfg = config

    # Split RNGs: shared obs realization across filters, independent
    # filter-internal stochasticity. See PFConfig docstring.
    obs_rng = np.random.RandomState(cfg.obs_seed)
    filt_rng = np.random.RandomState(cfg.filter_seed)

    Nx = xt_0.shape[0]
    I = np.eye(Nx)

    # Draw initial particle subset from pool
    ind = filt_rng.permutation(Xf_pool.shape[0])[:cfg.Ne]
    Xf_k = Xf_pool[ind, :].copy().T  # (Nx, Ne)
    w = np.full(cfg.Ne, 1.0 / cfg.Ne)  # uniform initial weights
    xt_k = xt_0.copy()

    # Observation setup
    Ny = int(round(cfg.p * Nx))
    R_k = (cfg.sig_obs ** 2) * np.eye(Ny)
    R_inv_diag = 1.0 / (cfg.sig_obs ** 2)  # R is diagonal in this codebase

    # Storage — shared schema with run_enkf
    errorf = np.zeros(cfg.T)
    errora = np.zeros(cfg.T)
    spread = np.zeros(cfg.T)
    errorf_es = np.zeros(cfg.T)
    errora_es = np.zeros(cfg.T)
    errorf_es_acc = np.zeros(cfg.T)
    errorf_es_spr = np.zeros(cfg.T)
    errora_es_acc = np.zeros(cfg.T)
    errora_es_spr = np.zeros(cfg.T)
    xt_traj = np.zeros((cfg.T, Nx))
    xf_traj = np.zeros((cfg.T, Nx))
    xa_traj = np.zeros((cfg.T, Nx))
    Xf_all = np.zeros((cfg.T, Nx, cfg.Ne))
    Xa_all = np.zeros((cfg.T, Nx, cfg.Ne))

    # PF-specific storage
    weights_f = np.zeros((cfg.T, cfg.Ne))
    weights_a = np.zeros((cfg.T, cfg.Ne))
    n_eff = np.zeros(cfg.T)
    resampled = np.zeros(cfg.T, dtype=bool)

    # Inter-cycle trajectories
    ens_fcst_traj = np.full((cfg.T, cfg.Ne, cfg.M + 1, Nx), np.nan)
    truth_fcst_traj = np.full((cfg.T, cfg.M + 1, Nx), np.nan)

    diverged = False

    def _fill_nan_from(k):
        """NaN out all per-cycle metrics from cycle k onward."""
        for arr in (errorf, errora, spread,
                    errorf_es, errora_es,
                    errorf_es_acc, errorf_es_spr,
                    errora_es_acc, errora_es_spr,
                    n_eff):
            arr[k:] = np.nan

    desc = f"{model_name} | PF DA" if model_name else "PF DA cycles"
    for k in tqdm(range(cfg.T), desc=desc, leave=False):
        # ----------------------------------------------------------
        # Hard divergence check (matches EnKF behaviour)
        # ----------------------------------------------------------
        if np.any(np.isnan(Xf_k)) or np.any(np.abs(Xf_k) > 1e6):
            print(f"  WARNING: Particle ensemble diverged at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break

        # ----------------------------------------------------------
        # Forecast statistics (weighted)
        # ----------------------------------------------------------
        xf_k = Xf_k @ w                                  # weighted mean
        # Weighted spread per variable, then averaged
        diffs = Xf_k - xf_k[:, None]                     # (Nx, Ne)
        var_w = (diffs ** 2) @ w                         # (Nx,)
        spread[k] = float(np.mean(np.sqrt(var_w)))

        # Forecast RMSE (mean estimator vs truth)
        errorf[k] = np.linalg.norm(xf_k - xt_k) / np.sqrt(Nx)

        # Forecast Energy Score (uniform-weight cloud, full state)
        es_f, acc_f, spr_f = energy_score(Xf_k, xt_k)
        errorf_es[k] = es_f
        errorf_es_acc[k] = acc_f
        errorf_es_spr[k] = spr_f

        xt_traj[k] = xt_k
        xf_traj[k] = xf_k
        Xf_all[k] = Xf_k.copy()
        weights_f[k] = w.copy()

        # ----------------------------------------------------------
        # Observation (obs-side stochasticity)
        # ----------------------------------------------------------
        obs_comp = obs_rng.permutation(Nx)[:Ny]
        H_k = I[obs_comp, :]
        y_k = H_k @ xt_k + cfg.sig_obs * obs_rng.standard_normal(Ny)

        # ----------------------------------------------------------
        # Reweight: bootstrap PF likelihood update
        #   w_new ∝ w * p(y | x_i),    p Gaussian with covariance R
        # NaN-safe: any non-finite particle gets log-likelihood = -inf.
        # ----------------------------------------------------------
        innov = y_k[:, None] - H_k @ Xf_k                # (Ny, Ne)
        # R diagonal here, so whitened-innov squared norm is straightforward.
        loglik = -0.5 * R_inv_diag * np.sum(innov ** 2, axis=0)  # (Ne,)
        loglik = np.where(np.isfinite(loglik), loglik, -np.inf)
        # Stabilize before exp
        finite_mask = np.isfinite(loglik)
        if not np.any(finite_mask):
            print(f"  WARNING: All particles have zero likelihood at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break
        loglik = loglik - np.max(loglik[finite_mask])
        w_new = w * np.exp(loglik)
        total = w_new.sum()
        if (not np.isfinite(total)) or total <= 0.0:
            print(f"  WARNING: Filter weight collapse at cycle {k}")
            _fill_nan_from(k)
            diverged = True
            break
        w = w_new / total

        # Effective sample size
        n_eff_k = 1.0 / np.sum(w ** 2)
        n_eff[k] = n_eff_k

        # ----------------------------------------------------------
        # Conditional resampling (systematic) + jittering
        # ----------------------------------------------------------
        Xa_k = Xf_k.copy()
        if n_eff_k <= cfg.NER * cfg.Ne:
            # Pre-resample weighted-cov factor (must be built before w/idx mutate)
            C12 = cfg.reg * _scott_bandwidth(cfg.Ne, Nx) * _raw_C12(Xa_k, w)
            idx = systematic_resample(w, filt_rng)
            Xa_k = Xa_k[:, idx]
            w = np.full(cfg.Ne, 1.0 / cfg.Ne)
            resampled[k] = True
            if cfg.jitter and cfg.reg > 0:
                Xa_k = _jitter_ensemble(Xa_k, idx, C12, cfg.nuj, filt_rng)

        # ----------------------------------------------------------
        # Multiplicative inflation around the (weighted) analysis mean
        # ----------------------------------------------------------
        xa_k = Xa_k @ w  # weighted mean (uniform after resample)
        if cfg.use_inflation:
            DXa_k = Xa_k - xa_k[:, None]
            Xa_k = xa_k[:, None] + cfg.infl_factor * DXa_k
            xa_k = Xa_k @ w  # recompute (mean is unchanged but cheap)

        # Analysis RMSE
        errora[k] = np.linalg.norm(xa_k - xt_k) / np.sqrt(Nx)
        xa_traj[k] = xa_k
        Xa_all[k] = Xa_k.copy()
        weights_a[k] = w.copy()

        # Analysis Energy Score (uniform-weight cloud, full state)
        es_a, acc_a, spr_a = energy_score(Xa_k, xt_k)
        errora_es[k] = es_a
        errora_es_acc[k] = acc_a
        errora_es_spr[k] = spr_a

        # ----------------------------------------------------------
        # Forecast to next cycle (store full per-particle trajectories)
        # ----------------------------------------------------------
        traj_e = forecast_fn(Xa_k)              # (Ne, M+1, Nx)
        ens_fcst_traj[k, :, :, :] = traj_e
        Xf_k = traj_e[:, -1, :].T               # (Nx, Ne)

        # Truth forward — always the real Lorenz solver
        xt_next = LorenzSystems.generate_trajectory_fast(
            '63', xt_k, cfg.dt, cfg.M + 1
        )
        truth_fcst_traj[k, :, :] = xt_next
        xt_k = xt_next[-1, :]

    return {
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
