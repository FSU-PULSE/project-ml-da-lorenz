# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
import tqdm
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems

torch.set_default_dtype(torch.float64)

# ============================================================
# Model initialization
# ============================================================
model_dir = "models"

def init_models(n_steps):
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

surrogates, surrogates_palette = init_models(1)

VAR_NAMES = ['x', 'y', 'z']
_SEED = 132
np.random.seed(_SEED)

Ne = 1000
Nx = 3
d0 = 0.01
n_steps = 1_000
dt = 0.01
time = np.arange(0, n_steps * dt + dt, dt)

x0 = np.ones(3) + np.random.randn(3)
sol = LorenzSystems.generate_trajectory_fast('63',x0, dt, 345454)
x0 = sol[-1, :]

Xf_pool = np.outer(np.ones(Ne), x0) + d0 * np.random.randn(Ne, Nx)

surrogate_pools = {name: [] for name in surrogates.keys()}
surrogate_series = {name: [] for name in surrogates.keys()}

for name, model in surrogates.items():
    pool = Xf_pool.copy()
    sol = model.batch_rollout(pool.reshape(Ne, 1, Nx), num_steps=n_steps)
    surrogate_series[name] = sol
    tqdm.tqdm.write(f"Ensemble pool ready for {name}")

# Lorenz baseline pool (perfect model)
pool_lorenz = Xf_pool.copy()
lorenz_pools = []
for e in tqdm.tqdm(range(Ne), desc="Running Lorenz63", unit="member", leave=False):
    sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :], dt, n_steps + 1)
    lorenz_pools.append(sol)
lorenz_series = np.stack(lorenz_pools)
print(f"Ensemble pool ready for Lorenz63 (baseline)")

model_types = list(surrogates.keys())
n_models = len(model_types)
step_start = 0
step_limit = n_steps
time_slice = time[step_start + 1 : step_limit + 1]

for ii, var_name in enumerate(VAR_NAMES):
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for jj, model_type in enumerate(model_types):
        ax = axes[jj]
        ax.plot(
            time_slice,
            surrogate_series[model_type][:, step_start:step_limit, ii].T,
            color=surrogates_palette[model_type],
            alpha=0.4,
            linewidth=0.5,
            zorder=2,
        )
        ax.plot(
            time_slice,
            surrogate_series[model_type][:, step_start:step_limit, ii].T.mean(axis=1),
            color="red",
            linewidth=2,
            label="Surrogate Mean",
            linestyle="--",
            alpha=0.7,
            zorder=10,
        )
        ax.plot(
            time_slice,
            lorenz_series[:, step_start + 1 : step_limit + 1, ii].T,
            color=surrogates_palette["Lorenz63"],
            alpha=0.4,
            linewidth=0.5,
            zorder=1,
        )
        ax.plot(
            time_slice,
            lorenz_series[:, step_start + 1 : step_limit + 1, ii].T.mean(axis=1),
            color="blue",
            linewidth=2,
            label="Lorenz Mean",
            linestyle="--",
            alpha=0.7,
            zorder=9,
        )
        ax.legend()
        ax.set_ylabel(var_name)
        ax.set_title(model_type)

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"{var_name}, Ne = {Ne}, perturbation amplitude = {d0}, time steps = {step_limit}"
    )
    plt.tight_layout()
    out_path = f"outputs/spread_visualizations_{var_name}_pert_{d0}_nsteps_{step_limit}_4.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved {out_path}")
# %%
