# %%
"""
D_PF.py — Bootstrap Particle Filter benchmark driver.

Mirrors D_EnKF.py but uses run_pf instead of run_enkf. Designed so the
two drivers can be run with the same surrogate models, the same initial
ensemble pools, and the same observation realization (set obs_seed equal
across the two), enabling head-to-head EnKF-vs-PF comparisons.

The benchmark summary table includes both RMSE and Energy Score so the
distributional fidelity of each surrogate is visible at a glance — this
is the metric that detects variance collapse, which RMSE alone cannot.
"""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
from plotting_helpers import *
import pandas as pd

import torch
import numpy as np
import os
import pickle as pkl
from os.path import join

from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems
from PF_core import (
    PFConfig,
    make_surrogate_forecaster,
    make_lorenz_forecaster,
    run_pf,
)

torch.set_default_dtype(torch.float64)

os.makedirs("figures", exist_ok=True)

# Global figure counter for unique filenames
_fig_count = [0]

# Global configuration parameters
_T = 350
_M = 100
_Ne = 1000
_CORE_SEED = 10
_NE_MAX = 1000

pkl_filename_string = 'l63_pf_benchmark_results_S_{:03d}_T_{:04d}.pkl'.format(_CORE_SEED, _T)
benchmark_results = {}


def _print_benchmark_table(results, title=""):
    """Print a summary DataFrame including RMSE and Energy Score."""
    rows = []
    for name, res in results.items():
        n_valid = int(np.sum(~np.isnan(res['errorf'])))
        mf = np.nanmean(res['errorf'])
        ma = np.nanmean(res['errora'])
        ms = np.nanmean(res['spread'])
        mfes = np.nanmean(res['errorf_es'])
        maes = np.nanmean(res['errora_es'])
        # ES decomposition averages — fingerprint of variance collapse
        macc = np.nanmean(res['errora_es_acc'])
        mspr = np.nanmean(res['errora_es_spr'])
        ratio = ms / mf if mf > 0 else np.nan
        rows.append({
            'Model': name,
            'fRMSE': round(mf, 4),
            'aRMSE': round(ma, 4),
            'fES': round(mfes, 4),
            'aES': round(maes, 4),
            'aES_acc': round(macc, 4),
            'aES_spr': round(mspr, 4),
            'Spread': round(ms, 4),
            'Spread/fRMSE': round(ratio, 4),
            'Valid_cycles': n_valid,
            'Diverged': res['diverged'],
        })
    df = pd.DataFrame(rows)
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False))
    return df


def _savefig(label="fig"):
    """Save current figure to figures/ directory with a sequential counter prefix."""
    fname = f"figures/PF_{_fig_count[0]:03d}_{label}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    _fig_count[0] += 1
    print(f"  [saved: {fname}]")


# ============================================================
# Model initialization (identical to D_EnKF.py)
# ============================================================
model_dir = "models"


def init_models(n_steps):
    if n_steps == 1:
        print("Initializing single time step models")
        model_paths = {
            'DenseNN': join(model_dir, 'DenseNN_L63_trial1_1775267887_best_model.pth'),
            'ResDenseNN': join(model_dir, 'ResDenseNN_L63_trial1_1775267929_best_model.pth'),
            'LSTMNN': join(model_dir, 'LSTMNN_L63_trial1_1779133789_best_model.pth'),
            'RNN_tanh': join(model_dir, 'RNN_L63_trial1_1779133920_best_model.pth'),
            'RNN_relu': join(model_dir, 'RNN_L63_trial1_1779117546_best_model.pth'),
        }
    else:
        raise ValueError(f"Invalid number of steps: {n_steps}")

    surrogates_palette = {
        'Lorenz63': '#000000',
        'Truth': '#000000',
        'DenseNN': '#D85A30',
        'ResDenseNN': '#7F770A',
        'LSTMNN': '#7F77DD',
        'RNN_relu': '#1D9E75',
        'RNN_tanh': '#1DDE75',
    }

    surrogates = {
        'DenseNN': SurrogateModel(model_paths['DenseNN']),
        'ResDenseNN': SurrogateModel(model_paths['ResDenseNN']),
        'LSTMNN': SurrogateModel(model_paths['LSTMNN']),
        'RNN_relu': SurrogateModel(model_paths['RNN_relu']),
        'RNN_tanh': SurrogateModel(model_paths['RNN_tanh']),
    }

    return surrogates, surrogates_palette


surrogates, surrogates_palette = init_models(1)

# ============================================================
# Utility functions
# ============================================================
VAR_NAMES = ['x', 'y', 'z']

# ============================================================
# Spin-up: generate true initial condition on the attractor
# (Same procedure as D_EnKF.py, so head-to-head comparisons share IC.)
# ============================================================
np.random.seed(_CORE_SEED)
Nx = 3
spinup_steps = 2000
x0 = np.random.randn(Nx).astype(np.float64)
dt = 0.01
n_steps = len(np.arange(0, 10, dt))

traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps + 1)
xt_init = traj[-1, :]  # on the attractor after spin-up

# Create a perturbed initial condition and propagate both
d0 = 1.0
xp_init = xt_init + d0 * np.random.randn(Nx)

# Propagate both to build an initial ensemble pool
traj_true = LorenzSystems.generate_trajectory_fast('63', xt_init, dt, n_steps + 1)[1:, :]
traj_pert = LorenzSystems.generate_trajectory_fast('63', xp_init, dt, n_steps + 1)[1:, :]

xt = traj_true[-1, :]   # true state after propagation
xp = traj_pert[-1, :]   # perturbed state after propagation

# %% Build initial ensemble pool (propagated with each surrogate once)
Ne_total = _NE_MAX

# Ensemble centered on perturbed state
Xf_pool = np.outer(np.ones(Ne_total), xp) + d0 * np.random.randn(Ne_total, Nx)

# Propagate ensemble pool forward to decorrelate members
M_spinup = len(np.arange(0, 10, dt))

# We store per-surrogate propagated pools so each architecture
# starts from its own spun-up ensemble
Xf_pools = {}
for name, model in surrogates.items():
    pool = Xf_pool.copy()
    sol = model.batch_rollout(pool.reshape(Ne_total, 1, Nx), num_steps=M_spinup)
    Xf_pools[name] = sol[:, -1, :]                                       #[Ne_total, M_spinup+1, Nx]
    print(f"  Ensemble pool ready for {name}")

# Lorenz baseline pool (perfect model)
pool_lorenz = Xf_pool.copy()
for e in range(Ne_total):
    sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :], dt, M_spinup + 1)
    pool_lorenz[e, :] = sol[-1, :]
Xf_pools['Lorenz63'] = pool_lorenz
print(f"  Ensemble pool ready for Lorenz63 (baseline)")

# Propagate truth forward the same duration
traj_truth_spinup = LorenzSystems.generate_trajectory_fast('63', xt, dt, M_spinup + 1)
xt = traj_truth_spinup[-1, :]


# ============================================================
# Benchmark runner — runs PF for Lorenz baseline + every surrogate
# ============================================================
def run_all_models(cfg, surrogates, Xf_pools, xt_0, palette):
    """Run PF for Lorenz baseline + all surrogates under a given config."""
    results = {}

    # Lorenz63 baseline
    lorenz_fn = make_lorenz_forecaster(cfg.dt, cfg.M)
    res = run_pf(lorenz_fn, xt_0.copy(), Xf_pools['Lorenz63'], cfg, model_name='Lorenz63')
    results['Lorenz63'] = res
    print(f"  Lorenz63    | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  "
          f"fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  "
          f"N_eff={np.nanmean(res['n_eff']):.1f}  resampled={int(res['resampled'].sum())}  "
          f"div={res['diverged']}")

    # Surrogates
    for name, model in surrogates.items():
        fn = make_surrogate_forecaster(model, cfg.M)
        res = run_pf(fn, xt_0.copy(), Xf_pools[name], cfg, model_name=name)
        results[name] = res
        print(f"  {name:12s} | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  "
              f"fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  "
              f"N_eff={np.nanmean(res['n_eff']):.1f}  resampled={int(res['resampled'].sum())}  "
              f"div={res['diverged']}")

    return results


# %% Start experiments
# ============================================================
# Benchmark 1: Baseline (reference conditions)
# ============================================================
print("\n" + "=" * 60)
print("BENCHMARK 1 [PF]: Baseline (reference conditions)")
print("=" * 60)

cfg_base = PFConfig(
    T=_T, M=_M, Ne=_Ne, dt=0.01,
    p=1.0, sig_obs=1.0,
    NER=1.0, reg=1.0, jitter=True, nuj=True,
    use_inflation=False,
    obs_seed=10, filter_seed=10,
)

results_base = run_all_models(cfg_base, surrogates, Xf_pools, xt, surrogates_palette)
benchmark_results["b1"] = results_base
cfg_base.use_localization = False

_print_benchmark_table(
    results_base,
    f"BENCHMARK 1 [PF] — Baseline Summary "
    f"(T={_T}, M={_M}, Ne={_Ne}, p=1.0, σ=1.0, NER={cfg_base.NER}, reg={cfg_base.reg})",
)

# Plotting helpers from EnKF pipeline are reused — they consume the
# shared keys (errorf, errora, spread, Xf_all, Xa_all, ...).
#plot_rmse_comparison(results_base, surrogates_palette, cfg_base, " — PF Baseline")
plot_es_comparison(results_base, surrogates_palette, cfg_base, " — PF Baseline")
#plot_spread_comparison(results_base, surrogates_palette, cfg_base, " — PF Baseline")
#plot_spread_vs_rmse(results_base, surrogates_palette, cfg_base, " — PF Baseline")

# Spaghetti: compare all models for each variable
for v in range(Nx):
    plot_ensemble_spaghetti_multi(results_base, surrogates_palette, cfg_base,
                                  cycle_range=(0, 30), var_idx=v)

plot_spread_reduction_multi(results_base, surrogates_palette, cfg_base,
                            cycle_range=(0, 30), var_idx=0)

print("BENCHMARK DONE")

# %%
