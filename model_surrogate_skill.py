# %%
from os.path import join
from SurrogateModel import SurrogateModel
from MachineLearning import LSTMNN, RNN
from lorenz.lorenz_systems import LorenzSystems
import numpy as np
import torch
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from glob import glob
import pandas as pd

torch.set_default_dtype(torch.float64)

model_dir = 'models'

def init_models(n_steps):
    if n_steps == 1:
        # Single time step models paths
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
        'Truth': '#000000',
        'DenseNN': '#D85A30',
        'ResDenseNN': '#7F770A',
        'LSTMNN': '#7F77DD',
        'RNN_relu': '#1D9E75',
        'RNN_tanh': '#1DDE75',
    }

    # Create surrogate models
    surrogates = {
        'DenseNN': SurrogateModel(model_paths['DenseNN']),
        'ResDenseNN': SurrogateModel(model_paths['ResDenseNN']),
        'LSTMNN': SurrogateModel(model_paths['LSTMNN']),
        'RNN_relu': SurrogateModel(model_paths['RNN_relu']),
        'RNN_tanh': SurrogateModel(model_paths['RNN_tanh']),
    }

    return surrogates, surrogates_palette, model_paths

VAR_NAMES = ['x', 'y', 'z']

# True Lorenz 63 system
init_steps = 1

# Initialize surrogate models
surrogates, surrogates_palette, model_paths = init_models(init_steps)

x0 = np.array([1.0, 1.0, 1.1])
dt = 0.01
n_steps = (200_000 + init_steps)

true_traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, n_steps)
history = true_traj[:init_steps,:]

print("Running surrogates...")
# Run surrogates
traj_surrogates = {
    'DenseNN': surrogates['DenseNN'].rollout(history, num_steps=n_steps),
    'ResDenseNN': surrogates['ResDenseNN'].rollout(history, num_steps=n_steps),
    'LSTMNN': surrogates['LSTMNN'].rollout(history, num_steps=n_steps),
    'RNN_relu': surrogates['RNN_relu'].rollout(history, num_steps=n_steps),
    'RNN_tanh': surrogates['RNN_tanh'].rollout(history, num_steps=n_steps),
}

# %% 3D plot comparison
cut_off = 4_000# 20000 steps = 200 time units = 181.2 Lyapunov times
fig = plt.figure(figsize=(30, 10))
ax = [fig.add_subplot(1, 1, 1, projection="3d")]
ax[0].scatter(history[:,0], history[:,1], history[:,2], color='red', label='initialization')
ax[0].plot(true_traj[:cut_off,0], true_traj[:cut_off,1], true_traj[:cut_off,2], color='black', label='True trajectory', linewidth=1)
for model_type in traj_surrogates:
    ax[0].plot(traj_surrogates[model_type][:cut_off,0], traj_surrogates[model_type][:cut_off,1], traj_surrogates[model_type][:cut_off,2], label=model_type, color=surrogates_palette[model_type], alpha=0.5, linewidth=0.9)
plt.legend()
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_zlabel('Z')
ax[0].set_title(f'Lorenz63 Attractor Comparison - {init_steps} time steps')
plt.savefig(f'outputs/lorenz63_attractor_comparison_{init_steps}_3d.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
print(f'Saved lorenz63_attractor_comparison_{init_steps}_3d.png')

# %% k-RMSE
@torch.no_grad()
def rollout(model, history0, K):
    """
    Autoregressively roll a windowed surrogate forward K steps.

    Uses ``SurrogateModel.batch_rollout`` so each batch member has an
    independent hidden state (required for RNN/LSTM).

    Parameters
    ----------
    model : SurrogateModel
    history0 : array-like
        Shape (prev_time_steps, N) for a single IC, or (B, prev_time_steps, N)
        for a batch of ICs.
    K : int
        Number of steps to roll out.

    Returns
    -------
    np.ndarray
        Shape (K, N) for single IC, or (K, B, N) for batched ICs.
    """
    p = int(getattr(model, "prev_time_steps", 1))
    history0 = np.asarray(history0, dtype=np.float64)

    if history0.ndim == 2:
        history0 = history0[None, ...]  # (1, p, N)
        squeeze_batch = True
    elif history0.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError(f"history0 must have ndim 2 or 3, got {history0.ndim}")

    B, p_in, N = history0.shape
    if p_in != p:
        raise ValueError(f"history0 has p={p_in} but model.prev_time_steps={p}")

    out = model.batch_rollout(history0, K)  # (B, K, N)
    preds = np.transpose(out, (1, 0, 2))      # (K, B, N)

    if squeeze_batch:
        return preds[:, 0, :]
    return preds


def kstep_rmse(model, traj, K, n_ics=500, stride=40):
    """
    Compute RMSE(k) for k = 1..K averaged over many initial conditions.
    
    Args:
        model:   surrogate model
        traj:    (T, 3) ground truth trajectory (use test_traj)
        K:       int, max rollout length. Start with K=50.
        n_ics:   number of initial conditions to average over
        stride:  spacing between ICs along the trajectory
                 (stride * dt should be >> 1 Lyapunov time to get
                  independent ICs: 1 LT ≈ 110 steps at dt=0.01)
    
    Returns:
        rmse_curve:  (K,) array of RMSE values, one per step
    """
    if hasattr(traj, "detach"):
        traj = traj.detach().cpu().numpy()
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"traj must have shape (T, N); got {traj.shape}")

    T, N = traj.shape
    p = int(getattr(model, "prev_time_steps", 1))
    n_in = getattr(model, "input_size", None)
    if n_in is not None and int(n_in) != N:
        raise ValueError(f"traj has N={N} but model.input_size={n_in}")

    # Define start index as the *last* index in the initial history window.
    # History window for start t is traj[t-p+1 : t+1] (oldest-first).
    min_start = p - 1
    max_start = T - K - 1
    if max_start < min_start:
        raise ValueError(f"Trajectory too short for p={p} and K={K}: T={T}")

    starts = np.arange(min_start, max_start + 1, stride, dtype=int)[:n_ics]
    B = len(starts)
    if B == 0:
        raise ValueError("No initial conditions sampled; decrease stride or n_ics.")

    history0 = np.stack([traj[t - p + 1 : t + 1] for t in starts], axis=0)  # (B, p, N)
    truth = np.stack([traj[starts + k] for k in range(1, K + 1)], axis=0)    # (K, B, N)

    preds = rollout(model, history0, K)  # (K, B, N)

    # RMSE(k): mean over variables, then mean over ICs
    sq_err = (preds - truth) ** 2
    rmse = np.sqrt(sq_err.mean(axis=2)).mean(axis=1)  # (K,)
    return rmse

## All variables
test_traj = torch.tensor(true_traj[1:])
K = 50  # 50 steps = 0.5 time units at dt=0.01 ≈ 0.45 Lyapunov times
K = 500
rmse_mlp  = kstep_rmse(surrogates['DenseNN'],  test_traj, K, stride=200)
rmse_lstm = kstep_rmse(surrogates['LSTMNN'], test_traj, K, stride=200)
rmse_rnn_tanh = kstep_rmse(surrogates['RNN_tanh'], test_traj, K, stride=200)
rmse_rnn  = kstep_rmse(surrogates['RNN_relu'],  test_traj, K, stride=200)
rmse_resdense = kstep_rmse(surrogates['ResDenseNN'], test_traj, K, stride=200)

# Climatological RMSE: error you'd get by always predicting the mean
clim_rmse = test_traj[1:,:].std(0).norm().item()  # scalar ≈ 15–16 for L63 (excluding initial condition)
steps = np.arange(1, K + 1)
time_units = steps * 0.01   # convert to L63 time units
lyapunov_times = time_units * 0.906  # convert to Lyapunov times

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(lyapunov_times, rmse_mlp,  label='DenseNN',  color=surrogates_palette['DenseNN'])
ax.semilogy(lyapunov_times, rmse_resdense, label='ResDenseNN', color=surrogates_palette['ResDenseNN'])
ax.semilogy(lyapunov_times, rmse_lstm, label='LSTMNN', color=surrogates_palette['LSTMNN'])
ax.semilogy(lyapunov_times, rmse_rnn,  label='RNN_relu',  color=surrogates_palette['RNN_relu'])
ax.semilogy(lyapunov_times, rmse_rnn_tanh,  label='RNN_tanh',  color=surrogates_palette['RNN_tanh'])

ax.axhline(clim_rmse, ls='--', color='gray', lw=0.8, label=f'Climatology {clim_rmse:.2f}')

# Training rollout interval (every 8 steps = 0.08 tu = 0.073 LT)
ax.axvline(5 * 0.01 * 0.906, ls=':', color='black', lw=0.8, 
           label='Rollout interval (5 steps)')
ax.axvline(20 * 0.01 * 0.906, ls=':', color='black', lw=0.8, 
           label='Rollout interval (20 steps)')

ax.set_xlabel('Lyapunov times')
ax.set_ylabel('RMSE (log scale)')
ax.set_title('k-step RMSE — autoregressive rollout')
ax.legend()
plt.tight_layout()
plt.savefig(f'kstep_rmse_nsteps_{init_steps}.png', dpi=300)

# %% Per variable k-step RMSE
def kstep_rmse_per_variable(model, traj, K, n_ics=500, stride=200):
    """Returns (K, N) — separate RMSE curve per variable (same ICs as kstep_rmse)."""
    if hasattr(traj, "detach"):
        traj = traj.detach().cpu().numpy()
    traj = np.asarray(traj, dtype=np.float64)
    T, N = traj.shape
    p = int(getattr(model, "prev_time_steps", 1))
    n_in = getattr(model, "input_size", None)
    if n_in is not None and int(n_in) != N:
        raise ValueError(f"traj has N={N} but model.input_size={n_in}")
    min_start = p - 1
    max_start = T - K - 1
    if max_start < min_start:
        raise ValueError(f"Trajectory too short for p={p} and K={K}: T={T}")
    starts = np.arange(min_start, max_start + 1, stride, dtype=int)[:n_ics]
    if len(starts) == 0:
        raise ValueError("No initial conditions sampled; decrease stride or n_ics.")
    history0 = np.stack([traj[t - p + 1 : t + 1] for t in starts], axis=0)
    truth = np.stack([traj[starts + k] for k in range(1, K + 1)], axis=0)
    preds = rollout(model, history0, K)
    rmse_xyz = np.sqrt(((preds - truth) ** 2).mean(axis=1))
    return rmse_xyz


rmse_mlp_x = kstep_rmse_per_variable(surrogates['DenseNN'],  test_traj, K, stride=200)
rmse_lstm_x = kstep_rmse_per_variable(surrogates['LSTMNN'], test_traj, K, stride=200)
rmse_rnn_tanh_x = kstep_rmse_per_variable(surrogates['RNN_tanh'], test_traj, K, stride=200)
rmse_rnn_x = kstep_rmse_per_variable(surrogates['RNN_relu'],  test_traj, K, stride=200)
rmse_resdense_x = kstep_rmse_per_variable(surrogates['ResDenseNN'], test_traj, K, stride=200)

_ts = test_traj[1:, :]
clim_rmse_x = np.asarray(_ts.std(0).cpu() if hasattr(_ts, 'cpu') else _ts.std(0)).ravel()
n_vars = rmse_mlp_x.shape[1]
fig, axes = plt.subplots(1, n_vars, figsize=(13, 4), sharey=True)
axes = np.atleast_1d(axes)
for i, ax in enumerate(axes):
    ax.semilogy(lyapunov_times, rmse_mlp_x[:, i], label='DenseNN', color=surrogates_palette['DenseNN'])
    ax.semilogy(lyapunov_times, rmse_resdense_x[:, i], label='ResDenseNN', color=surrogates_palette['ResDenseNN'])
    ax.semilogy(lyapunov_times, rmse_lstm_x[:, i], label='LSTMNN', color=surrogates_palette['LSTMNN'])
    ax.semilogy(lyapunov_times, rmse_rnn_x[:, i], label='RNN_relu', color=surrogates_palette['RNN_relu'])
    ax.semilogy(lyapunov_times, rmse_rnn_tanh_x[:, i], label='RNN_tanh', color=surrogates_palette['RNN_tanh'])
    ax.axhline(clim_rmse_x[i], ls='--', color='gray', lw=0.8, label=f'Climatology {clim_rmse_x[i]:.2f}')

    ax.set_xlabel('Lyapunov times')
    ax.set_ylabel('RMSE (log scale)')
    ax.set_title(f'Variable {VAR_NAMES[i]}')
    ax.legend()
plt.tight_layout()
plt.savefig(f'kstep_rmse_nsteps_{init_steps}_per_variable.png', dpi=300)


# %% PSD analysis
# implement PSD function
def compute_psd(trajectory, dt=0.01, nperseg=8196):
    """
    Compute PSD for each variable using Welch's method.
    trajectory: (T, 3) numpy array
    Returns:
        freqs:   (F,) frequency array in 1/time_units
        psds:    (3, F) power at each frequency, per variable
    """
    psds = []
    for i in range(trajectory.shape[1]):
        f, p = welch(trajectory[:, i], fs=1.0 / dt, 
                     nperseg=nperseg, window='hann', detrend='constant')
        p = p / (p.sum() * (f[1] - f[0]))
        psds.append(p)
    return f, np.stack(psds, axis=0)

print("Running PSD analysis...")
f, psd_true = compute_psd(true_traj[10_000:])
_, psd_dense = compute_psd(traj_surrogates['DenseNN'][10_000:])
_, psd_resdense = compute_psd(traj_surrogates['ResDenseNN'][10_000:])
_, psd_lstm = compute_psd(traj_surrogates['LSTMNN'][10_000:])
_, psd_rnn = compute_psd(traj_surrogates['RNN_relu'][10_000:])
_, psd_rnn_tanh = compute_psd(traj_surrogates['RNN_tanh'][10_000:])

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for i, ax in enumerate(axes):
    ax.loglog(f, psd_true[i], color=surrogates_palette['Truth'],
                lw=1.5, label='Truth', zorder=5)
    ax.loglog(f, psd_lstm[i], color=surrogates_palette['LSTMNN'],
                lw=1.0, label='LSTM', alpha=0.85)
    ax.loglog(f, psd_rnn[i],  color=surrogates_palette['RNN_relu'],
                lw=1.0, label='RNN_relu',  alpha=0.85)
    ax.loglog(f, psd_rnn_tanh[i],  color=surrogates_palette['RNN_tanh'],
                lw=1.0, label='RNN_tanh',  alpha=0.85)
    ax.loglog(f, psd_dense[i],  color=surrogates_palette['DenseNN'],
                lw=1.0, label='Dense',  alpha=0.85, ls='--')
    ax.loglog(f, psd_resdense[i],  color=surrogates_palette['ResDenseNN'],
                lw=1.0, label='ResDense',  alpha=0.85, ls='--')

    ax.set_xlim(0, 0.5)       # L63 has negligible energy above 0.5
    ax.set_ylim(1e-5, 1e1)
    ax.set_xlabel('Frequency (1 / time unit)')
    ax.set_title(f'Variable {VAR_NAMES[i]}')
    ax.axvline(0.065, ls=':', lw=0.8, color='gray')  # dominant L63 peak


axes[0].set_ylabel('Power spectral density (log scale)')
axes[0].legend(fontsize=9)
plt.suptitle(f'Power spectral density - free-run comparison - {init_steps} steps initialisation', y=1.02)
plt.tight_layout()
plt.savefig(f'psd_comparison_nsteps_{init_steps}.png', dpi=300, bbox_inches='tight')

# %% Spectral divergence
def log_spectral_distance(psd_model, psd_truth, freq_max=0.5, f=f):
    """
    Mean squared log-ratio of power spectra, averaged over variables
    and frequency range. Lower is better; 0 is perfect.
    """
    mask = f <= freq_max
    log_ratio = np.log(psd_model[:, mask] / psd_truth[:, mask])  # (3, F_masked)
    return float(np.mean(log_ratio ** 2))

lsd_dense     = log_spectral_distance(psd_dense,     psd_true, f=f)
lsd_resdense  = log_spectral_distance(psd_resdense,  psd_true, f=f)
lsd_lstm      = log_spectral_distance(psd_lstm,      psd_true, f=f)
lsd_rnn       = log_spectral_distance(psd_rnn,       psd_true, f=f)
lsd_rnn_tanh  = log_spectral_distance(psd_rnn_tanh,  psd_true, f=f)

print(f"LSD  Dense:    {lsd_dense:.3f}")
print(f"LSD ResDense:  {lsd_resdense:.3f}")
print(f"LSD LSTM:      {lsd_lstm:.3f}")
print(f"LSD  RNN_relu: {lsd_rnn:.3f}")
print(f"LSD  RNN_tanh: {lsd_rnn_tanh:.3f}")

def hf_power_fraction(psd, f, f_split=0.2):
    """Fraction of total power above f_split, averaged over variables."""
    hf    = psd[:, f >= f_split].sum(axis=1)
    total = psd.sum(axis=1)
    return float((hf / total).mean())

hf_true      = hf_power_fraction(psd_true,     f)
hf_dense     = hf_power_fraction(psd_dense,    f)
hf_resdense  = hf_power_fraction(psd_resdense, f)
hf_lstm      = hf_power_fraction(psd_lstm,     f)
hf_rnn       = hf_power_fraction(psd_rnn,      f)
hf_rnn_tanh  = hf_power_fraction(psd_rnn_tanh, f)

print(f"HF Power Fraction True:     {hf_true:.3f}")
print(f"HF Power Fraction Dense:    {hf_dense:.3f}")
print(f"HF Power Fraction ResDense: {hf_resdense:.3f}")
print(f"HF Power Fraction LSTM:     {hf_lstm:.3f}")
print(f"HF Power Fraction RNN_relu: {hf_rnn:.3f}")
print(f"HF Power Fraction RNN_tanh: {hf_rnn_tanh:.3f}")


# %%  Climatological behaviour
print("Running climatological behaviour analysis...")
n_steps = 500_000
x0 = np.array([1.0, 1.0, 1.1])
true_traj = LorenzSystems.generate_trajectory_fast('63', x0, 0.01, n_steps)
history = true_traj[:init_steps,:]
traj_surrogates = {
    'Truth': true_traj[1:,:],
    'DenseNN': surrogates['DenseNN'].rollout(history, num_steps=n_steps),
    'ResDenseNN': surrogates['ResDenseNN'].rollout(history, num_steps=n_steps),
    'LSTMNN': surrogates['LSTMNN'].rollout(history, num_steps=n_steps),
    'RNN_relu': surrogates['RNN_relu'].rollout(history, num_steps=n_steps),
    'RNN_tanh': surrogates['RNN_tanh'].rollout(history, num_steps=n_steps),
}

def marginal_stats(traj, name):
    """Print and return per-variable mean, std, skewness, kurtosis."""
    stats = {}
    print(f"\n{name}")
    print(f"{'':6s}  {'mean':>8s}  {'std':>8s}  {'skew':>8s}  {'kurt':>8s}")
    for i, v in enumerate(VAR_NAMES):
        col = traj[:, i]
        m  = col.mean()
        s  = col.std()
        sk = skew(col)
        ku = kurtosis(col)   # excess kurtosis; Gaussian = 0
        print(f"  {v}:   {m:8.3f}  {s:8.3f}  {sk:8.3f}  {ku:8.3f}")
        stats[v] = dict(mean=m, std=s, skew=sk, kurt=ku)
    return stats

all_stats = {name: marginal_stats(t, name) for name, t in traj_surrogates.items()}

def covariance_analysis(trajs_dict):
    """
    Compute and compare full 3x3 covariance matrices.
    Returns dict of covariance matrices and a Frobenius-norm difference table.
    """
    covs = {}
    for name, traj in trajs_dict.items():
        # Remove mean, compute full covariance
        c = np.cov(traj.T)    # (3, 3)
        covs[name] = c

    truth_cov = covs['Truth']
    print("\nCovariance matrix — Truth:")
    print(np.round(truth_cov, 2))

    print("\nFrobenius norm ||C_model - C_truth|| / ||C_truth||:")
    for name, c in covs.items():
        if name == 'Truth':
            continue
        rel_err = np.linalg.norm(c - truth_cov, 'fro') / \
                  np.linalg.norm(truth_cov, 'fro')
        print(f"  {name:6s}: {rel_err:.4f}")

    return covs

covs = covariance_analysis(traj_surrogates)

fig, ax = plt.subplots(1, len(traj_surrogates), figsize=(14, 3.5))
for ax, (name, c) in zip(ax, covs.items()):
    im = ax.imshow(c, vmin=-80, vmax=100, cmap='RdBu_r')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(VAR_NAMES)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(VAR_NAMES)
    ax.set_title(name)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{c[i,j]:.1f}', ha='center',
                    va='center', fontsize=8, color='k')
#plt.colorbar(im, ax=ax[-1])
plt.suptitle('Climatological covariance matrices')
plt.tight_layout()
plt.savefig(f'covariance_matrices_nsteps_{init_steps}.png', dpi=300)

def marginal_stats(traj, name):
    """Print and return per-variable mean, std, skewness, kurtosis."""
    stats = {}
    print(f"\n{name}")
    print(f"{'':6s}  {'mean':>8s}  {'std':>8s}  {'skew':>8s}  {'kurt':>8s}")
    for i, v in enumerate(VAR_NAMES):
        col = traj[:, i]
        m  = col.mean()
        s  = col.std()
        sk = skew(col)
        ku = kurtosis(col)   # excess kurtosis; Gaussian = 0
        print(f"  {v}:   {m:8.3f}  {s:8.3f}  {sk:8.3f}  {ku:8.3f}")
        stats[v] = dict(mean=m, std=s, skew=sk, kurt=ku)
    return stats

all_stats = {name: marginal_stats(t, name) for name, t in traj_surrogates.items()}

def plot_attractor_comparison(trajs_dict, subsample=5):
    """
    2D projections of the attractor for all models.
    subsample: plot every Nth point to reduce overplotting.
    """
    projections = [('x', 'z', 0, 2), ('x', 'y', 0, 1), ('y', 'z', 1, 2)]
    n_models = len(trajs_dict)
    fig, axes = plt.subplots(len(projections), n_models,
                             figsize=(3.5 * n_models, 9))

    for col, (name, traj) in enumerate(trajs_dict.items()):
        t = traj[::subsample]
        for row, (xlab, ylab, xi, yi) in enumerate(projections):
            ax = axes[row, col]
            ax.scatter(t[:, xi], t[:, yi], s=0.2, alpha=0.3,
                       c=surrogates_palette[name] if name in surrogates_palette else '#888780',
                       rasterized=True)
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(name)

    plt.suptitle('Attractor geometry — 2D projections')
    plt.tight_layout()
    plt.savefig(f'attractor_geometry_nsteps_{init_steps}.png', dpi=300)

plot_attractor_comparison(traj_surrogates)

def wing_balance(traj, name):
    """
    Fraction of time spent with x > 0. Should be ≈ 0.5 for L63.
    Significant deviation → attractor asymmetry.
    """
    frac = (traj[:, 0] > 0).mean()
    print(f"{name:6s}: x > 0  {frac*100:.1f}%  |  x < 0  {(1-frac)*100:.1f}%")
    return frac

for name, t in traj_surrogates.items():
    wing_balance(t, name)

def autocorrelation(signal, max_lag):
    """
    Normalized autocorrelation R(τ) = Cov(x_t, x_{t+τ}) / Var(x).
    Returns (max_lag,) array.
    """
    signal = signal - signal.mean()
    var    = (signal ** 2).mean()
    acf    = []
    for lag in range(1, max_lag + 1):
        c = (signal[:-lag] * signal[lag:]).mean() / var
        acf.append(c)
    return np.array(acf)


def plot_acf_comparison(trajs_dict, max_lag=500):
    """
    Autocorrelation functions for all three variables and all models.
    max_lag=500 steps = 5 time units ≈ 4.5 Lyapunov times.
    """
    lags = np.arange(1, max_lag + 1) * dt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, (vname, ax) in enumerate(zip(VAR_NAMES, axes)):
        for name, traj in trajs_dict.items():
            acf = autocorrelation(traj[:, i], max_lag)
            ls  = '--' if name == 'MLP' else '-'
            ax.plot(lags, acf, color=surrogates_palette[name],
                    label=name, lw=1.0, ls=ls)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('Lag (time units)')
        ax.set_title(f'ACF — variable {vname}')

    axes[0].set_ylabel('Autocorrelation')
    axes[0].legend(fontsize=9)
    plt.suptitle('Autocorrelation functions')
    plt.tight_layout()
    plt.savefig(f'acf_comparison_nsteps_{init_steps}.png', dpi=300)

plot_acf_comparison(traj_surrogates)

def regime_switching_rate(traj, name, dt=0.01):
    """
    Mean number of wing switches per unit time.
    True L63: roughly 0.08-0.12 switches per time unit.
    """
    signs = np.sign(traj[:, 0])
    # A switch occurs when consecutive signs differ
    switches = np.sum(np.diff(signs) != 0)
    rate = switches / (len(traj) * dt)
    mean_residence = 1.0 / rate if rate > 0 else np.inf
    print(f"{name:6s}: {rate:.4f} switches/tu  "
          f"(mean residence {mean_residence:.2f} tu)")
    return rate

for name, t in traj_surrogates.items():
    regime_switching_rate(t, name)

def attractor_summary_table(trajs_dict):
    rows = []
    truth = trajs_dict['Truth']
    truth_cov = np.cov(truth.T)

    for name, traj in trajs_dict.items():
        cov = np.cov(traj.T)
        row = {
            'Model'      : name,
            'z mean'     : f"{traj[:,2].mean():.2f}",
            'x std'      : f"{traj[:,0].std():.2f}",
            'Corr(x,y)'  : f"{cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):.3f}",
            'Wing bal.'  : f"{(traj[:,0]>0).mean():.3f}",
            'Switch rate': f"{regime_switching_rate(traj, name):.4f}",
            'Cov err'    : 'ref' if name == 'Truth' else
                           f"{np.linalg.norm(cov-truth_cov,'fro')/np.linalg.norm(truth_cov,'fro'):.3f}",
        }
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    print(df.to_string())
    return df

df = attractor_summary_table(traj_surrogates)
# df.to_csv(f'attractor_summary_nsteps_{init_steps}.csv')


# %% Lyapunov exponents

# Lorenz-63 reference (continuous-time, same time units as dt)
THEORETICAL_LLE = 0.9056

def compute_lle(
    forward_ref,
    x0,
    forward_pert=None,
    dt=0.01,
    n_steps=100_000,
    n_spinup=1_000,
    d0=1e-8,
    renorm_interval=1,
):
    """
    Compute the Largest Lyapunov Exponent using Benettin's method.
 
    Parameters
    ----------
    forward_ref : callable
        Black-box model: state(t) -> state(t + dt).
        Signature: forward_ref(state) -> np.ndarray
    x0 : np.ndarray
        Initial condition (e.g. shape (3,) for Lorenz-63).
    forward_pert : callable, optional
        Forward map for the perturbed trajectory. Defaults to ``forward_ref``.
        For stateful RNN/LSTM surrogates, pass a second ``SurrogateModel``
        instance so hidden state is not shared between branches.
    dt : float
        Time step used by forward_model.
    n_steps : int
        Total number of forward steps for the LLE computation.
    n_spinup : int
        Transient steps to let the trajectory settle onto the attractor
        BEFORE starting to accumulate Lyapunov sums.
    d0 : float
        Initial perturbation magnitude (keep small — linear regime).
    renorm_interval : int
        Renormalize the perturbation every this many steps.
        1 is safest for strongly chaotic systems.
 
    Returns
    -------
    lle : float
        Estimated largest Lyapunov exponent (in units of 1/time).
    lle_running : np.ndarray
        Running estimate of LLE at each renormalization event
        (useful for convergence diagnostics).
    """
    if forward_pert is None:
        forward_pert = forward_ref

    state_dim = x0.shape[0]
 
    # --- spin-up: both branches reach the attractor with aligned hidden state ---
    x_ref = x0.copy()
    x_pert = x0.copy()
    for _ in range(n_spinup):
        x_ref = forward_ref(x_ref)
        x_pert = forward_pert(x_pert)
 
    # --- initial perturbation (random direction, magnitude d0) ---
    delta = np.random.randn(state_dim)
    delta = delta / np.linalg.norm(delta) * d0
 
    x_pert = x_ref + delta
 
    # --- main loop ---
    log_growth_sum = 0.0
    n_renorm = 0
    lle_running = []
 
    for step in range(1, n_steps + 1):
        # propagate both trajectories
        x_ref = forward_ref(x_ref)
        x_pert = forward_pert(x_pert)
 
        if step % renorm_interval == 0:
            # measure perturbation growth
            delta = x_pert - x_ref
            d1 = np.linalg.norm(delta)
 
            if d1 == 0.0:
                # model has collapsed — perturbation killed
                print(f"⚠ Perturbation collapsed to zero at step {step}.")
                break
 
            log_growth_sum += np.log(d1 / d0)
            n_renorm += 1
 
            # running estimate
            elapsed_time = n_renorm * renorm_interval * dt
            lle_running.append(log_growth_sum / elapsed_time)
 
            # renormalize: keep direction, reset magnitude to d0
            delta = delta / d1 * d0
            x_pert = x_ref + delta
 
    if n_renorm == 0:
        raise RuntimeError("No renormalization events occurred.")
 
    total_time = n_renorm * renorm_interval * dt
    lle = log_growth_sum / total_time
 
    return lle, np.array(lle_running)


def _lle_forwarders(name):
    """Reference and perturbed forward callables for Benettin LLE."""
    ref = surrogates[name]
    if isinstance(ref.model, (LSTMNN, RNN)):
        pert = SurrogateModel(model_paths[name])
        return ref, pert
    return ref, None


lle_ic = true_traj[2000, :]
lle_dense = compute_lle(surrogates['DenseNN'], lle_ic)
lle_resdense = compute_lle(surrogates['ResDenseNN'], lle_ic)
_ref, _pert = _lle_forwarders('LSTMNN')
lle_lstm = compute_lle(_ref, lle_ic, forward_pert=_pert)
_ref, _pert = _lle_forwarders('RNN_relu')
lle_rnn = compute_lle(_ref, lle_ic, forward_pert=_pert)
_ref, _pert = _lle_forwarders('RNN_tanh')
lle_rnn_tanh = compute_lle(_ref, lle_ic, forward_pert=_pert)

# LLE is in units of 1/time (elapsed_time includes dt)
print(f"LLE Dense:    {lle_dense[0]}  (theory ~ {THEORETICAL_LLE:.4f})")
print(f"LLE ResDense: {lle_resdense[0]}")
print(f"LLE LSTM:     {lle_lstm[0]}")
print(f"LLE RNN_relu: {lle_rnn[0]}")
print(f"LLE RNN_tanh: {lle_rnn_tanh[0]}")

# Add LLE to attractor summary table
df['LLE'] = [THEORETICAL_LLE, lle_dense[0], lle_resdense[0], lle_lstm[0], lle_rnn[0], lle_rnn_tanh[0]]

# %%
