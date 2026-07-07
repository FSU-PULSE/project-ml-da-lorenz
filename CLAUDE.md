# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML surrogate modeling framework for Lorenz dynamical systems (L63, L96, L05 Model III). The goal is to train neural networks as fast replacements for numerical ODE integrators in Ensemble Data Assimilation (DA) workflows. Trained surrogates propagate ensemble members through forecast cycles instead of the expensive numerical solver.

## Setup

DAPyr (not on PyPI) must be installed locally **before** installing other dependencies:

```bash
conda activate lorenzo   # project conda environment
pip install DAPyr
pip install -r requirements.txt
```

For GPU training, `torch` in `requirements.txt` is CPU-only by default — replace with the appropriate CUDA wheel.

For GPU particle-filter benchmarks (`D_PF_cupy`), also install CuPy matched to your CUDA build:

```bash
pip install -r requirements-gpu.txt   # e.g. cupy-cuda12x
```

## Commands

```bash
# Interactive Lorenz system simulator
python 0_LorenzDashboard.py       # http://localhost:5007

# Interactive ML training studio (dataset gen, training, live loss plots)
python 1_SingleMLTraining.py      # http://localhost:8050

# Batch training driven by config.yml (optional: pass a custom config path)
python Main_ML.py                     # uses config.yml
python Main_ML.py my_experiment.yml   # custom config

# SLURM (HPC cluster) — passes config file as first arg
sbatch train.slurm [config_file.yml]

# Comparative evaluation of saved models
python ML_Model_Comparison.py

# EnKF DA benchmarks (surrogate vs. truth); outputs go to figures/
python D_EnKF.py

# Particle Filter DA benchmarks (NumPy); outputs go to figures/
python D_PF.py

# Particle Filter DA benchmarks (CuPy / GPU surrogates); requires CUDA + cupy
python D_PF_cupy.py

# TensorBoard (logs written to runs/)
tensorboard --logdir runs/

# Tests
pytest tests/
pytest tests/test_lorenz_systems.py::test_generate_trajectory_l63   # single test
```

## Architecture

### Entry Points

| File | Role |
|------|------|
| `0_LorenzDashboard.py` | Dash app — Lorenz trajectory simulator |
| `1_SingleMLTraining.py` | Dash app — full interactive training UI |
| `Main_ML.py` | CLI — batch training from `config.yml` |
| `ML_Model_Comparison.py` | CLI — evaluate and compare saved models |
| `D_EnKF.py` | Script — full EnKF DA experiments with surrogate vs. truth |
| `D_PF.py` | Script — bootstrap PF benchmarks (NumPy); mirrors `D_EnKF` setup for head-to-head comparisons |
| `D_PF_cupy.py` | Script — same PF benchmarks with CuPy on-device state and GPU surrogate rollouts |

### Core Modules

**`MachineLearning.py`** — Four model architectures, all sharing the same interface:
- Input: `(batch, prev_time_steps × state_dim)` — flattened history window (oldest-first)
- Output: `(batch, state_dim)` — predicted next state

| Class | Notes |
|-------|-------|
| `DenseNN` | Standard MLP |
| `ResDenseNN` | Residual MLP; predicts Δ, returns `current + Δ` — more stable training |
| `LSTMNN` | LSTM; reshapes input to `(batch, prev_time_steps, state_dim)` internally; `forward` returns `(output, hidden)` |
| `RNN` | Vanilla RNN with same reshaping; `forward` returns `(output, hidden)`; `rnn_nonlinearity` is configurable (`'tanh'`/`'relu'`) |

**`Training.py`** — Training loop and rollout logic:
- `train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping, ..., max_rollout_steps, gamma, stateful_rollout, initial_lr, lr_phase_decay, lr_scheduler_patience, lr_scheduler_factor)` — gradient-descent loop; returns `(trained_model, history)`
- `recursive_rollout(model, initial_input, num_steps, prev_time_steps, device, stateful_rollout=False)` → `(batch, num_steps, nx)` — autoregressive rollout; when `stateful_rollout=True` and model has `stateful=True`, hidden state is threaded across steps (BPTT). Window shifts each step (drop oldest, append prediction).
- Multi-step loss: `Σ γ^s · loss_s`; γ defaults to 0.9 but is configurable via `training.rollout_gamma`
- `EarlyStopping` — patience-based; when patience is exhausted and `rollout_steps < max_rollout_steps`, the phase advances (`rollout_steps += 1`), optimizer state is cleared, LR is decayed by `lr_phase_decay`, and a fresh `ReduceLROnPlateau` scheduler is created. Training only stops when patience is exhausted at `max_rollout_steps`.
- Within each phase, `ReduceLROnPlateau` (patience=`lr_scheduler_patience`, factor=`lr_scheduler_factor`) reduces LR on plateau.
- Validation uses the **same `rollout_steps`** as the current training phase — there is no separate fixed validation depth.
- No gradient clipping is applied; stability for recurrent models is managed via the progressive phase schedule and `stateful_rollout`.

**`datasets/LorenzDataset.py`** — PyTorch Dataset:
- Generates trajectories via `LorenzSystems.generate_trajectory_fast()` (DAPyr + numbalsoda)
- Caches per-trajectory `.npz` files in `dataset_cache/` keyed by a hash of the full dataset config — any config change invalidates the cache
- Fits `StandardScaler` on training data; normalization stats saved with model checkpoint
- Sample: input = `data[i-prev_time_steps:i].flatten()`, target = next `_MAX_FUTURE=10` steps
- Valid index caching prevents samples from crossing trajectory boundaries (different ICs)

**`lorenz/lorenz_systems.py`** — DAPyr integration layer:
- `LorenzSystems.generate_trajectory_fast()` — preferred; single batch call to SUNDIALS via numbalsoda
- `LorenzSystems.generate_trajectory()` — legacy per-step loop (slower, avoid)
- Caches DAPyr RHS function pointers by `(system_type, params)` to avoid Numba recompilation

**`SurrogateModel.py`** — Inference wrapper:
- Loads any checkpoint, reconstructs architecture from metadata
- Handles normalization/denormalization internally; users always work in physical space
- `predict(history)` → next state `(N,)`; `rollout(history, n_steps)` → `(n_steps, N)`; calling the object (`surrogate(history)`) aliases `predict`
- `batch_rollout` / `batch_rollout_tensor` / `batch_rollout_cupy` — batched autoregressive rollout; tensor/cupy variants keep results on GPU (DLPack bridge for CuPy)
- Auto-discovers `.yml` sidecar next to `.pth` for full config reconstruction

**`PF_core.py`** / **`PF_core_cupy.py`** — Bootstrap particle filter (shared `PFConfig` from `PF_core`):
- `run_pf` / `run_pf_cupy` — full DA cycle; CuPy version returns NumPy dicts for plotting. Filter RNG, systematic resampling, and jitter use the same NumPy helpers as `run_pf` (`filter_seed` matches); particle linear algebra stays on GPU.
- `make_surrogate_forecaster(_gpu)` / `make_lorenz_forecaster(_gpu)` — forecast callables for ensemble layout `(Nx, Ne)`
- Truth integration always uses CPU DAPyr; `obs_seed` observations are CPU NumPy in both variants

**`EnKF_core.py`** — Core EnKF logic (imported by `D_EnKF.py`):
- `EnKFConfig` — dataclass for all tunable parameters
- `run_enkf(forecast_fn, xt_0, Xf_pool, config)` — runs full DA cycle; `forecast_fn` wraps either `SurrogateModel` or ground-truth Lorenz
- `make_surrogate_forecaster(surrogate, M)` / `make_lorenz_forecaster(dt, M)` — factory functions for `forecast_fn`; **note: `make_lorenz_forecaster` is currently hardcoded for L63 only**
- Supports stochastic EnKF and deterministic ETKF, multiplicative inflation, Gaspari-Cohn localization

**`D_EnKF.py`** — EnKF experiment orchestrator:
- Imports from `EnKF_core.py`, runs 5 benchmark sweeps: baseline, M, Ne, p (observed fraction), σ_obs
- Figures output to `figures/`

**`EnKFConfig`** key fields:
```python
T=30          # DA cycles
M=10          # forecast steps between assimilation
Ne=20         # ensemble size
dt=0.01
p=1.0         # fraction of observed state vars
sig_obs=1.1   # observation error std
enkf_type='stochastic'  # or 'deterministic' (ETKF)
use_localization=False; loc_radius=3.0
use_inflation=False;    infl_factor=1.02
```

**`plotting_helpers.py`** — Matplotlib/Plotly utilities for EnKF output (RMSE curves, rank histograms, spread-skill plots). Consumed by `D_EnKF.py`; not called from training code.

**`verify/`** — Standalone analysis scripts (not part of the main pipeline):
- `spread_visualizations.py` — ensemble spread plots for saved surrogates
- `model_surrogate_skill.py` — spectral/statistical skill metrics (power spectra, skewness, kurtosis) for surrogate vs truth
- `surrogate_ensemble_skill.py` — ensemble skill evaluation
- `dataset_validation.py` — validates cached `.npz` dataset files in `dataset_cache/`
- These scripts use hardcoded model paths (in `models/`) and must be updated manually when new checkpoints are trained.

### Data Flow

```
config.yml
  └─▶ LorenzDataset
        ├─ DAPyr RK45/LSODA integrator (numbalsoda)
        ├─ Cache .npz per trajectory (hash-keyed)
        └─ StandardScaler normalization
  └─▶ DataLoader (train/val/test 70/20/10)
  └─▶ Model (DenseNN | ResDenseNN | LSTMNN | RNN)
        ├─ recursive_rollout with progressive depth
        └─ Multi-step loss (γ decay, configurable)
  └─▶ Checkpoint .pth + sidecar .yml
  └─▶ SurrogateModel (inference)
  └─▶ EnKF assimilation cycles
```

### Model Checkpoint Format

`.pth` contains: `model_state_dict`, `train_mean`, `train_std`, `architecture` dict.  
`.yml` sidecar stores the full training config for reproducibility.  
Filenames: `{ModelType}_L{system}_trial{n}_{timestamp}.pth`.

## Configuration (`config.yml`)

Controls all batch training parameters. `1_SingleMLTraining.py` GUI exposes the same knobs.

```yaml
dataset:
  system_type: '63'          # '63' | '96' | '05'
  dt: 0.001                  # integration time step
  ns: 100000                 # samples per start location
  save_dt: 10                # subsampling factor (effective Δt = dt × save_dt)
  std: 1.0                   # observation noise std (0 = clean)
  prev_time_steps: 1         # history window size
  ds_noise: false            # whether to add noise to dataset
  num_start_locations: 20    # independent ICs
  system_params: {F: 15.0, K: 32, l05_c: 0.6, l05_b: 10.0}  # ignored for L63

model:
  type: 'RNN'                # DenseNN | ResDenseNN | LSTMNN | RNN
  hidden_layers: [64]
  hidden_activation: 'ReLU'  # 'ReLU' | 'Tanh' | 'Sigmoid'
  output_activation: null
  rnn_nonlinearity: 'tanh'   # RNN only: 'tanh' | 'relu'

training:
  num_epochs: 10000
  batch_size: 2048
  learning_rate: 0.001
  n_trials: 1                # independent runs (each saved with timestamp)
  early_stopping_patience: 30
  early_stopping_min_delta: 1.0e-6  # minimum loss improvement to reset patience counter
  loss_func: 'MSE'           # 'MSE' | 'Huber'
  split_train: 70
  split_val: 20
  split_test: 10
  max_rollout_steps: 20      # phases advance 1 → max_rollout_steps via patience
  rollout_gamma: 0.95        # geometric decay weight for multi-step loss
  stateful_rollout: true     # thread hidden state through rollout (RNN/LSTMNN only)
  lr_phase_decay: 0.5        # multiply LR by this at each phase advance
  lr_scheduler_patience: 10  # ReduceLROnPlateau patience within a phase
  lr_scheduler_factor: 0.5   # ReduceLROnPlateau reduction factor

paths:
  outputs: 'models'          # full checkpoints (.pth) + configs (.yml)
  models: 'models'
  runs: 'runs'               # TensorBoard logs
  dataset_cache: 'dataset_cache'
```

## Key Constraints

- **Fixed state dimensions**: L96 is always N=40, L05 is always N=480 — hardcoded in DAPyr's Numba kernels. L63 is N=3.
- **Normalization**: Always normalize inputs with saved `train_mean`/`train_std`; denormalize predictions. `SurrogateModel` does this automatically — raw models do not.
- **Thread management**: `1_SingleMLTraining.py` sets `OMP/MKL/NUMBA_NUM_THREADS=1` to prevent oversubscription with its `ThreadPoolExecutor`.
- **Validation rollout**: Uses the same `rollout_steps` as the current training phase — there is no separate fixed validation depth. Both train and val loss are computed with the current phase's depth.
- **Cache invalidation**: The dataset cache key covers the full dataset config dict. Changing any dataset parameter (dt, ns, system_params, etc.) will trigger regeneration of all trajectory files.
- **Recurrent model hidden state**: `LSTMNN` and `RNN` return `(output, hidden)` tuples. When `stateful_rollout=True`, `recursive_rollout` threads hidden state across autoregressive steps; when `False`, hidden resets per step. Hidden state always resets at the start of every sample/batch — consistent with DA usage where each forecast cycle starts fresh via `SurrogateModel.rollout` (which calls `reset_hidden()` internally).
