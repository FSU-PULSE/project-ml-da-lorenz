# %%
"""
D_PF_cupy.py — Bootstrap Particle Filter benchmark driver (GPU / CuPy).

Mirrors :mod:`D_PF` using :mod:`PF_core_cupy` and on-device surrogate rollouts.
Requires CUDA, CuPy (wheel matched to PyTorch CUDA), and SurrogateModel on GPU.

Observation realizations and filter resampling/jitter use the same NumPy RNG
and helpers as :mod:`D_PF` (``obs_seed``, ``filter_seed``). Surrogate rollouts
still run on GPU and may differ slightly from CPU ``batch_rollout``.
"""

import matplotlib
# Headless backend: this script saves figures to disk and does not need a display.
matplotlib.use('Agg')
from plotting_helpers import *
import pandas as pd

import torch
import numpy as np
import os
from os.path import join

try:
    import cupy as cp
except ImportError as e:
    raise ImportError(
        "D_PF_cupy requires CuPy. Install a CUDA-matched wheel, e.g. "
        "pip install -r requirements-gpu.txt"
    ) from e

from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems
from PF_core import PFConfig
from PF_core_cupy import (
    make_surrogate_forecaster_gpu,
    make_lorenz_forecaster_gpu,
    run_pf_cupy,
)

if not torch.cuda.is_available():
    raise RuntimeError("D_PF_cupy requires CUDA (torch.cuda.is_available() is False).")

# Match Lorenz / DAPyr double precision so surrogate rollouts align with the physics integrator.
torch.set_default_dtype(torch.float64)

os.makedirs("figures", exist_ok=True)

_fig_count = [0]

# PF experiment grid: T assimilation cycles, M forecast steps per cycle, Ne particles.
_T = 100
_M = 100
_Ne = 1000
# Fixed RNG for ICs and NumPy-side filter pieces (observations, resampling); see PFConfig obs_seed/filter_seed.
_CORE_SEED = 10
# Max particle count; individual pools are sized to this before spinup.
_NE_MAX = 1000

benchmark_results = {}


def _print_benchmark_table(results, title=""):
    """Aggregate per-cycle diagnostics from run_pf_cupy into a single summary row per model."""
    rows = []
    for name, res in results.items():
        n_valid = int(np.sum(~np.isnan(res['errorf'])))
        # RMSE: forecast (f) vs analysis (a). Spread: ensemble spread vs truth.
        mf = np.nanmean(res['errorf'])
        ma = np.nanmean(res['errora'])
        ms = np.nanmean(res['spread'])
        # ES = ensemble skill / error space metrics (decomposition into accuracy vs spread terms).
        mfes = np.nanmean(res['errorf_es'])
        maes = np.nanmean(res['errora_es'])
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
    # Serial counter keeps plot filenames ordered when multiple helpers call save.
    fname = f"figures/PF_GPU_{_fig_count[0]:03d}_{label}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    _fig_count[0] += 1
    print(f"  [saved: {fname}]")


model_dir = "models"


def init_models(n_steps):
    """Load L63 surrogate checkpoints on CUDA; paths are trial-specific (update when retraining)."""
    if n_steps == 1:
        print("Initializing single time step models (CUDA)")
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
        name: SurrogateModel(path, device='cuda')
        for name, path in {
            'DenseNN': model_paths['DenseNN'],
            'ResDenseNN': model_paths['ResDenseNN'],
            'LSTMNN': model_paths['LSTMNN'],
            'RNN_relu': model_paths['RNN_relu'],
            'RNN_tanh': model_paths['RNN_tanh'],
        }.items()
    }

    return surrogates, surrogates_palette


surrogates, surrogates_palette = init_models(1)

VAR_NAMES = ['x', 'y', 'z']
np.random.seed(_CORE_SEED)
# %% to avoid reseting the random seed
Nx = 3
# Burn-in from a random IC so the “truth” path is on the attractor before we branch trajectories.
spinup_steps = 2000
x0 = np.random.randn(Nx).astype(np.float64)
dt = 0.01
# Ten seconds of model time at dt (used for short reference trajectories below).
n_steps = len(np.arange(0, 10, dt))

traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps + 1)
xt_init = traj[-1, :]

d0 = 1.0
# Perturbed reference state: used only to build the initial particle cloud (not the assimilated truth).
xp_init = xt_init + d0 * np.random.randn(Nx)

traj_true = LorenzSystems.generate_trajectory_fast('63', xt_init, dt, n_steps + 1)[1:, :]
traj_pert = LorenzSystems.generate_trajectory_fast('63', xp_init, dt, n_steps + 1)[1:, :]

xt = traj_true[-1, :]
xp = traj_pert[-1, :]

Ne_total = _NE_MAX
# Common initial ensemble: all particles near xp, i.i.d. Gaussian perturbation (PF cold start).
Xf_pool = np.outer(np.ones(Ne_total), xp) + d0 * np.random.randn(Ne_total, Nx)

# Spin up each pool for the same wall-clock interval as n_steps so cycle-0 forecast starts from comparable dynamics.
M_spinup = len(np.arange(0, 10, dt))

# We store per-surrogate propagated pools so each architecture
# starts from its own spun-up ensemble
Xf_pools = {}

pool_lorenz = cp.asarray(Xf_pool.copy())
for e in range(Ne_total):
    sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :].get(), dt, M_spinup + 1)
    pool_lorenz[e, :] = cp.asarray(sol[-1, :])
Xf_pools['Lorenz63'] = pool_lorenz.copy()
print(f"  Ensemble pool ready for Lorenz63 (baseline)")

for model_name in surrogates.keys():
    Xf_pools[model_name] = pool_lorenz.copy()
print(f"  Ensemble pool ready for {model_name} (baseline)")

# Propagate truth forward the same duration
traj_truth_spinup = LorenzSystems.generate_trajectory_fast('63', xt, dt, M_spinup + 1)[1:, :]
xt = traj_truth_spinup[-1, :]

def run_all_models(cfg, surrogates, Xf_pools, xt_0, palette):
    """Run CuPy PF for Lorenz63 (gold standard) then each GPU surrogate; palette reserved for plotting helpers."""
    results = {}

    # Ground-truth forecast: DAPyr on CPU per call, state moved to GPU inside PF_core_cupy.
    lorenz_fn = make_lorenz_forecaster_gpu(cfg.dt, cfg.M)
    res = run_pf_cupy(lorenz_fn, xt_0.copy(), Xf_pools['Lorenz63'], cfg, model_name='Lorenz63')
    results['Lorenz63'] = res
    print(f"  Lorenz63    | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  "
          f"fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  "
          f"N_eff={np.nanmean(res['n_eff']):.1f}  resampled={int(res['resampled'].sum())}  "
          f"div={res['diverged']}")

    for name, model in surrogates.items():
        fn = make_surrogate_forecaster_gpu(model, cfg.M)
        res = run_pf_cupy(fn, xt_0.copy(), Xf_pools[name], cfg, model_name=name)
        results[name] = res
        print(f"  {name:12s} | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  "
              f"fES={np.nanmean(res['errorf_es']):.3f}  aES={np.nanmean(res['errora_es']):.3f}  "
              f"N_eff={np.nanmean(res['n_eff']):.1f}  resampled={int(res['resampled'].sum())}  "
              f"div={res['diverged']}")

    return results


print("\n" + "=" * 60)
print("BENCHMARK 1 [PF GPU]: Baseline (reference conditions)")
print("=" * 60)
reg = 0.8
cfg_base = PFConfig(
    T=_T, M=_M, Ne=_Ne, dt=0.01,
    p=1.0, sig_obs=1.0,
    # NER / reg / jitter / nuj: resampling noise and regularization knobs (see PF_core.PFConfig).
    NER=0.5, reg=reg, jitter=True, nuj=True,
    use_inflation=False,
    obs_seed=18, filter_seed=15,
    qroot=1.0, sig_dyn=0.0,
)

results_base = run_all_models(cfg_base, surrogates, Xf_pools, xt, surrogates_palette)
benchmark_results["b1"] = results_base

_print_benchmark_table(
    results_base,
    f"BENCHMARK 1 [PF GPU] — Baseline Summary "
    f"(T={_T}, M={_M}, Ne={_Ne}, p=1.0, σ=1.0, NER={cfg_base.NER}, reg={cfg_base.reg})",
)

plot_es_comparison(results_base, surrogates_palette, cfg_base, " — PF GPU Baseline")

for v in range(Nx):
    plot_ensemble_spaghetti_multi(results_base, surrogates_palette, cfg_base,
                                  cycle_range=(81, 100), var_idx=v)

#plot_spread_reduction_multi(results_base, surrogates_palette, cfg_base,
#                            cycle_range=(0, 100), var_idx=0)

# rank histogram
# plot_talagrand_grid(results_base, surrogates_palette, cfg_base, title_suffix=f" — PF reg = {reg:0.2f}")

print("BENCHMARK DONE")

# %%
