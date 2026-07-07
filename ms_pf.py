# %%
import argparse
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
from plotting_helpers import *
from enkf_results_store import EnKFResultsStore, summarize_cycles

import torch
import numpy as np
import cupy as cp
from scipy.linalg import sqrtm
import os

from os.path import join
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems
from PF_core import PFConfig
from PF_core_cupy import (
    make_surrogate_forecaster_gpu,
    make_lorenz_forecaster_gpu,
    run_pf_cupy,
)

torch.set_default_dtype(torch.float64)


os.makedirs("figures", exist_ok=True)

# Global figure counter for unique filenames
_fig_count = [0]

# Global configuration parameters
_T = 350
_Ne = 1000
_CORE_SEED = 10
_NE_MAX = 1000
VAR_NAMES = ['x', 'y', 'z']

def _print_benchmark_table(results, title=""):
    """Print a summary DataFrame for a set of results."""
    rows = []
    for name, res in results.items():
        n_valid = int(np.sum(~np.isnan(res['errorf'])))
        mf = np.nanmean(res['errorf'])
        ma = np.nanmean(res['errora'])
        ms = np.nanmean(res['spread'])
        ratio = ms / mf if mf > 0 else np.nan
        rows.append({
            'Model': name,
            'fRMSE': round(mf, 4),
            'aRMSE': round(ma, 4),
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
    fname = f"figures/{_fig_count[0]:03d}_{label}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    _fig_count[0] += 1
    print(f"  [saved: {fname}]")

# ============================================================
# Benchmark runner
# ============================================================
def run_all_models(cfg, surrogates, Xf_pools, xt_0, palette):
    """Run PF for Lorenz baseline + all surrogates under a given config."""
    results = {}

    # Lorenz63 baseline
    lorenz_fn = make_lorenz_forecaster_gpu(cfg.dt, cfg.M)
    res = run_pf_cupy(lorenz_fn, xt_0.copy(), Xf_pools['Lorenz63'], cfg, model_name='Lorenz63')
    results['Lorenz63'] = res
    print(f"  Lorenz63    | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  div={res['diverged']}")

    # Surrogates
    for name, model in surrogates.items():
        fn = make_surrogate_forecaster_gpu(model, cfg.M)
        res = run_pf_cupy(fn, xt_0.copy(), Xf_pools[name], cfg, model_name=name)
        results[name] = res
        print(f"  {name:12s} | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  div={res['diverged']}")

    return results

# ============================================================
# Model initialization
# ============================================================
model_dir = "models"

def init_models(n_steps=1):
    if n_steps == 1:
        print("Initializing single time step models")
        model_paths = {
            'DenseNN': join(model_dir, 'DenseNN_L63_trial1_1775267887_best_model.pth'),
            'ResDenseNN': join(model_dir, 'ResDenseNN_L63_trial1_1775267929_best_model.pth'),
            'LSTMNN': join(model_dir, 'LSTMNN_L63_trial1_1779133789_best_model.pth'),
            'RNN_tanh': join(model_dir, 'RNN_L63_trial1_1779205626_best_model.pth'),
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

def spinup_models(models, 
                 initial_condition, 
                 dt=0.01,
                 spinup_steps=2000,
                 n_steps=1000,
                 Nx=3,
                 core_seed=10,
                 ):
    np.random.seed(core_seed)
    traj = LorenzSystems.generate_trajectory_fast('63', initial_condition, dt, spinup_steps + 1)
    xt_init = traj[-1, :]  # on the attractor after spin-up

    # Create a perturbed initial condition and propagate both
    d0 = 1.0
    xp_init = xt_init + d0 * np.random.randn(Nx)

    # Propagate both to build an initial ensemble pool
    traj_true = LorenzSystems.generate_trajectory_fast('63', xt_init, dt, n_steps + 1)[1:, :]
    traj_pert = LorenzSystems.generate_trajectory_fast('63', xp_init, dt, n_steps + 1)[1:, :]

    xt = traj_true[-1, :]   # true state after propagation
    xp = traj_pert[-1, :]   # perturbed state after propagation

    # Build initial ensemble pool (propagated with each surrogate once)
    Ne_total = _NE_MAX

    # Ensemble centered on perturbed state
    Xf_pool = np.outer(np.ones(Ne_total), xp) + d0 * np.random.randn(Ne_total, Nx)

    # Propagate ensemble pool forward to decorrelate members
    M_spinup = len(np.arange(0, 10, dt))

    # We store per-surrogate propagated pools so each architecture
    # starts from its own spun-up ensemble
    Xf_pools = {}

    pool_lorenz = cp.asarray(Xf_pool.copy())
    for e in range(Ne_total):
        sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :].get(), dt, M_spinup + 1)
        pool_lorenz[e, :] = cp.asarray(sol[-1, :])
    Xf_pools['Lorenz63'] = pool_lorenz.copy()

    for model_name in models.keys():
        Xf_pools[model_name] = pool_lorenz.copy()
    print(f"  Ensemble pool ready for Lorenz63 (baseline)")

    # Propagate truth forward the same duration
    traj_truth_spinup = LorenzSystems.generate_trajectory_fast('63', xt, dt, M_spinup + 1)
    xt = traj_truth_spinup[-1, :]

    return Xf_pools, xt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EnKF experiments')
    parser.add_argument('--seed', type=int, default=132, help='Master seed')
    parser.add_argument('--ic', type=int, default=100, help='Number of initial conditions')
    parser.add_argument('--m', type=int, default=50, help='Number of forecast steps between assimilation cycles')
    parser.add_argument('--ne', type=int, default=1000, help='Number of ensemble members')
    parser.add_argument('--reg', type=float, default=0.3, help='Regularization parameter')
    args = parser.parse_args()

    _MASTER_SEED = args.seed
    _IC_N = args.ic
    _M = args.m
    _Ne = args.ne
    _Reg = args.reg
    surrogates, surrogates_palette = init_models()
    np.random.default_rng(_MASTER_SEED)
    initial_conditions = np.random.randn(_IC_N, 3) + 1
    
    model_names = ['Lorenz63', *surrogates.keys()]
    results_store = EnKFResultsStore(_IC_N, _T, model_names)
    results_h5_path = f'results/ms_pf_L63_seed_{_MASTER_SEED}_ics_{_IC_N}_ne_{_Ne}_reg_{_Reg}.h5'

    for iic, ic in enumerate(initial_conditions):
        Xf_pools, xt = spinup_models(surrogates, ic)
        print(f"  Experiment {iic + 1} of {_IC_N}: Initial condition: {ic}")
        # Run a full EnKF experiment using the EnKFConfig class

        cfg = PFConfig(
            T=_T, M=_M, Ne=_Ne, dt=0.01,
            p=1.0, sig_obs=1.0,
            NER=0.5, reg=_Reg, jitter=True, nuj=True,
            use_inflation=False,
            obs_seed=10, filter_seed=10,
            qroot=1.0, sig_dyn=0.0,
        )
        results = run_all_models(cfg, surrogates, Xf_pools, xt, surrogates_palette)
        # plot_rmse_comparison(results, surrogates_palette, cfg, f" — Results for initial condition: {ic}")
        plot_es_comparison(results, surrogates_palette, cfg, f" — Results for initial condition: {ic}")

        results_store.record(iic, ic, _MASTER_SEED + iic + 1, results)

    results_store.to_hdf5(
        results_h5_path,
        attrs={
            'master_seed': _MASTER_SEED,
            'T': _T,
            'M': _M,
            'Ne': _Ne,
            'dt': 0.01,
            'p': 1.0,
            'sig_obs': 1.0,
            'enkf_type': 'deterministic',
        },
    )
    print(f"  [saved: {results_h5_path}]")
    # print(summarize_cycles(results_store).to_string(index=False))

# %%
