import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
# Global figure counter for unique filenames
_fig_count = [0]
VAR_NAMES = ['x', 'y', 'z']  # state variable names

def _savefig(label="fig"):
    """Save current figure to figures/ directory with a sequential counter prefix."""
    fname = f"figures/{_fig_count[0]:03d}_{label}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    _fig_count[0] += 1
    print(f"  [saved: {fname}]")

# %%
# ============================================================
# Plotting functions
# ============================================================
def plot_rmse_comparison(results, palette, cfg, title_suffix=""):
    """Forecast & Analysis RMSE side by side for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    cycles = np.arange(cfg.T)
    for name, res in results.items():
        color = palette.get(name, '#888888')
        ls = '--' if name == 'Lorenz63' else '-'
        lw = 2.0 if name == 'Lorenz63' else 1.5
        axes[0].plot(cycles, res['errorf'], color=color, label=name, lw=lw, ls=ls)
        axes[1].plot(cycles, res['errora'], color=color, label=name, lw=lw, ls=ls)
    axes[0].set_title('Forecast RMSE')
    axes[0].set_xlabel('DA Cycle'); axes[0].set_ylabel('RMSE')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Analysis RMSE')
    axes[1].set_xlabel('DA Cycle')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    loc_str = f"Loc r={cfg.loc_radius}" if cfg.use_localization else "No Loc"
    infl_str = f"Infl λ={cfg.infl_factor}" if cfg.use_inflation else "No Infl"
    fig.suptitle(f'EnKF Comparison  |  {loc_str}  |  {infl_str}  |  Ne={cfg.Ne}  |  M={cfg.M}  |  p={cfg.p}  |  σ_obs={cfg.sig_obs}{title_suffix}', fontsize=11)
    plt.tight_layout()
    _savefig(f"rmse_comparison")

def plot_es_comparison(results, palette, cfg, title_suffix=""):
    """Forecast & Analysis Energy Score and RMSE side by side for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    cycles = np.arange(cfg.T)
    for name, res in results.items():
        color = palette.get(name, '#888888')
        ls = '--' if name == 'Lorenz63' else '-'
        lw = 2.0 if name == 'Lorenz63' else 1.5
        axes[0].plot(cycles, res['errorf_es'], color=color, label=name, lw=lw, ls=ls)
        axes[1].plot(cycles, res['errora_es'], color=color, label=name, lw=lw, ls=ls)
        axes[2].plot(cycles, res['errorf'], color=color, label=name, lw=lw, ls=ls)
        axes[3].plot(cycles, res['errora'], color=color, label=name, lw=lw, ls=ls)
    axes[0].set_title('Forecast Energy Score')
    axes[0].set_xlabel('DA Cycle'); axes[0].set_ylabel('Energy Score')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Analysis Energy Score')
    axes[1].set_xlabel('DA Cycle')
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title('Forecast RMSE')
    axes[2].set_xlabel('DA Cycle'); axes[2].set_ylabel('RMSE')
    axes[2].grid(True, alpha=0.3)
    axes[3].set_title('Analysis RMSE')
    axes[3].set_xlabel('DA Cycle')
    axes[3].grid(True, alpha=0.3)
    loc_str = f"Loc r={cfg.loc_radius}" if cfg.use_localization else "No Loc"
    infl_str = f"Infl λ={cfg.infl_factor}" if cfg.use_inflation else "No Infl"
    fig.suptitle(f'EnKF Comparison  |  {loc_str}  |  {infl_str}  |  Ne={cfg.Ne}  |  M={cfg.M}  |  p={cfg.p}  |  σ_obs={cfg.sig_obs}{title_suffix}', fontsize=11)
    plt.tight_layout()
    _savefig("es_comparison")

def plot_spread_comparison(results, palette, cfg, title_suffix=""):
    """Ensemble spread time series."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cycles = np.arange(cfg.T)
    for name, res in results.items():
        color = palette.get(name, '#888888')
        ls = '--' if name == 'Lorenz63' else '-'
        lw = 2.0 if name == 'Lorenz63' else 1.5
        ax.plot(cycles, res['spread'], color=color, label=name, lw=lw, ls=ls)
    ax.set_title(f'Ensemble Spread per Cycle{title_suffix}')
    ax.set_xlabel('DA Cycle'); ax.set_ylabel('Mean Std')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig("spread_comparison")

def plot_spread_vs_rmse(results, palette, cfg, title_suffix=""):
    """Scatter: mean spread vs mean forecast RMSE — reveals calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        ms = np.nanmean(res['spread'])
        mf = np.nanmean(res['errorf'])
        color = palette.get(name, '#888888')
        marker = 's' if name == 'Lorenz63' else 'o'
        ax.scatter(ms, mf, color=color, s=120, marker=marker, edgecolors='k', zorder=5)
        ax.annotate(name, (ms, mf), textcoords="offset points", xytext=(8, 5), fontsize=9)
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Spread = RMSE (ideal)')
    ax.set_xlabel('Mean Ensemble Spread')
    ax.set_ylabel('Mean Forecast RMSE')
    ax.set_title(f'Spread vs RMSE {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig("spread_vs_rmse")

def compute_rank_histogram(Xf_all, xt_traj):
    """
    Compute rank histograms (Talagrand diagrams) from stored ensembles.

    Parameters
    ----------
    Xf_all : np.ndarray, shape (T, Nx, Ne)
        Forecast ensemble at each DA cycle.
    xt_traj : np.ndarray, shape (T, Nx)
        True state at each DA cycle.

    Returns
    -------
    ranks : np.ndarray, shape (T * Nx,)
        Rank of the truth within the ensemble (0 to Ne inclusive).
    """
    T, Nx, Ne = Xf_all.shape
    ranks = np.zeros(T * Nx, dtype=int)
    idx = 0
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            ranks[idx:idx + Nx] = -1  # mark invalid
            idx += Nx
            continue
        for v in range(Nx):
            sorted_ens = np.sort(Xf_all[k, v, :])
            rank = np.searchsorted(sorted_ens, xt_traj[k, v])
            ranks[idx] = rank
            idx += 1
    # Remove invalid entries
    ranks = ranks[ranks >= 0]
    return ranks

def compute_rank_histogram_per_var(Xf_all, xt_traj):
    """Compute rank histograms separately for each state variable."""
    T, Nx, Ne = Xf_all.shape
    ranks_per_var = {v: [] for v in range(Nx)}
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            continue
        for v in range(Nx):
            sorted_ens = np.sort(Xf_all[k, v, :])
            rank = np.searchsorted(sorted_ens, xt_traj[k, v])
            ranks_per_var[v].append(rank)
    return {v: np.array(r) for v, r in ranks_per_var.items()}

def plot_talagrand_grid(results, palette, cfg, ensemble_key='Xf_all', title_suffix=""):
    """
    Talagrand rank histograms: one row per model, one column per state variable,
    plus one column for the aggregated histogram.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    Nx = results[model_names[0]]['xt_traj'].shape[1]
    Ne = cfg.Ne

    fig, axes = plt.subplots(n_models, Nx + 1, figsize=(4 * (Nx + 1), 3 * n_models),
                              squeeze=False)
    bins = np.arange(Ne + 2) - 0.5  # bin edges for ranks 0..Ne

    for i, name in enumerate(model_names):
        res = results[name]
        color = palette.get(name, '#888888')

        # Per-variable ranks
        ranks_pv = compute_rank_histogram_per_var(res[ensemble_key], res['xt_traj'])
        for v in range(Nx):
            ax = axes[i, v]
            if len(ranks_pv[v]) > 0:
                ax.hist(ranks_pv[v], bins=bins, color=color, edgecolor='black', alpha=0.8, density=True)
                ax.axhline(1.0 / (Ne + 1), color='k', ls='--', alpha=0.5, label='Uniform')
            ax.set_xlim(-0.5, Ne + 0.5)
            if i == 0:
                ax.set_title(f'{VAR_NAMES[v]}', fontsize=11)
            if v == 0:
                ax.set_ylabel(name, fontsize=10, fontweight='bold')

        # Aggregated ranks
        ax_agg = axes[i, Nx]
        ranks_all = compute_rank_histogram(res[ensemble_key], res['xt_traj'])
        if len(ranks_all) > 0:
            ax_agg.hist(ranks_all, bins=bins, color=color, edgecolor='black', alpha=0.8, density=True)
            ax_agg.axhline(1.0 / (Ne + 1), color='k', ls='--', alpha=0.5)
        ax_agg.set_xlim(-0.5, Ne + 0.5)
        if i == 0:
            ax_agg.set_title('All variables', fontsize=11)

    ens_label = "Forecast" if ensemble_key == 'Xf_all' else "Analysis"
    fig.suptitle(f'Talagrand Diagrams ({ens_label} Ensemble)  |  Ne={Ne}  |  M={cfg.M}  |  p={cfg.p}{title_suffix}', fontsize=13)
    plt.tight_layout()
    _savefig("talagrand")

def plot_ensemble_spaghetti(res, name, palette, cfg, cycle_range=None, var_indices=None):
    """
    Spaghetti plot: full trajectory of every ensemble member between DA cycles,
    plus ensemble mean and truth. Analysis corrections are visible as jumps.

    Parameters
    ----------
    res : dict — single model result from run_enkf
    name : str — model name (for title and color)
    palette : dict
    cfg : EnKFConfig
    cycle_range : tuple (start, end) — which cycles to plot (default: all)
    var_indices : list of int — which state variables to plot (default: all)
    """
    ens_traj = res['ens_fcst_traj']    # (T, Ne, M+1, Nx)
    truth_traj = res['truth_fcst_traj']  # (T, M+1, Nx)
    T, Ne, Mp1, Nx = ens_traj.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    if var_indices is None:
        var_indices = list(range(Nx))

    n_vars = len(var_indices)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 4 * n_vars), sharex=True,
                              squeeze=False, dpi=600)
    color = palette.get(name, '#888888')

    for row, v in enumerate(var_indices):
        ax = axes[row, 0]

        for k in range(k_start, k_end):
            if np.any(np.isnan(ens_traj[k])):
                break
            # Time axis: each cycle spans M+1 points starting at k*M
            t_offset = k * M
            t_local = np.arange(Mp1) + t_offset

            # Individual members — thin, transparent
            for e in range(Ne):
                ax.plot(t_local, ens_traj[k, e, :, v],
                        color=color, alpha=0.6, linewidth=0.5)

            # Ensemble mean trajectory
            ens_mean = np.mean(ens_traj[k, :, :, v], axis=0)
            ax.plot(t_local, ens_mean, color='red', linewidth=2.0, alpha=0.8)

            # Truth
            ax.plot(t_local, truth_traj[k, :, v],
                    color='blue', linewidth=2.0, alpha=0.9)

            # Mark analysis time (start of each forecast window)
            ax.axvline(t_offset, color='gray', linewidth=0.4, alpha=0.4, linestyle=':')

        ax.set_ylabel(f'{VAR_NAMES[v]}', fontsize=12)
        ax.grid(True, alpha=0.2)

    # Legend (only on first axis)
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.2, label='Truth'),
        Line2D([0], [0], color='red', linewidth=1.5, label=f'{name} (ens. mean)'),
        Line2D([0], [0], color=color, linewidth=0.5, alpha=0.4, label=f'{name} members'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right')

    axes[-1, 0].set_xlabel('Forecast step (cumulative)')
    fig.suptitle(f'Ensemble — {name}  |  Ne={cfg.Ne}  |  M={cfg.M}  |  cycles {k_start}–{k_end-1}',
                 fontsize=13)
    plt.tight_layout()
    _savefig(f"spaghetti_{name}")

def plot_ensemble_spaghetti_multi(results, palette, cfg, cycle_range=None, var_idx=0):
    """
    Compare spaghetti plots across all architectures for a single state variable.
    One subplot row per model.
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    T = cfg.T
    M = cfg.M
    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3 * n_models), sharex=True,
                              squeeze=False, dpi=600)

    for i, name in enumerate(model_names):
        ax = axes[i, 0]
        res = results[name]
        ens_traj = res['ens_fcst_traj']
        truth_traj = res['truth_fcst_traj']
        Ne = ens_traj.shape[1]
        Mp1 = ens_traj.shape[2]
        color = palette.get(name, '#888888')

        for k in range(k_start, k_end):
            if np.any(np.isnan(ens_traj[k])):
                break
            t_offset = k * M
            t_local = np.arange(Mp1) + t_offset

            for e in range(Ne):
                ax.plot(t_local, ens_traj[k, e, :, var_idx],
                        color=color, alpha=0.6, linewidth=0.5)
            ens_mean = np.mean(ens_traj[k, :, :, var_idx], axis=0)
            ax.plot(t_local, ens_mean, color="red", linewidth=2.0, alpha=0.8)
            ax.plot(t_local, truth_traj[k, :, var_idx],
                    color='blue', linewidth=2.0, alpha=0.9)
            ax.axvline(t_offset, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')

        ax.set_ylabel(name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel('Forecast step (cumulative)')
    fig.suptitle(f'Ensemble Comparison — {VAR_NAMES[var_idx]}  |  Ne={cfg.Ne}  |  M={cfg.M}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(f"spaghetti_multi_{VAR_NAMES[var_idx]}")

def plot_spread_reduction(res, name, palette, cfg, cycle_range=None, var_idx=0):
    """
    For each cycle, show how the ensemble std evolves within the forecast window.
    This reveals whether spread grows (healthy divergence) or collapses (variance death).
    Also marks the analysis correction as a vertical jump.
    """
    ens_traj = res['ens_fcst_traj']    # (T, Ne, M+1, Nx)
    T, Ne, Mp1, Nx = ens_traj.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    color = palette.get(name, '#888888')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False, dpi=600)

    # Compute per-cycle spreads for the full run (bars should not depend on the selected window)
    analysis_spread_all = []
    forecast_spread_all = []
    valid_T = 0
    for k in range(T):
        if np.any(np.isnan(ens_traj[k])):
            break
        std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
        analysis_spread_all.append(std_over_time[0])
        forecast_spread_all.append(std_over_time[-1])
        valid_T += 1

    # Top: spread over time within each selected forecast window (x-axis is cumulative model time steps)
    last_t = None
    for k in range(k_start, min(k_end, valid_T)):
        t_offset = k * M
        t_local = np.arange(Mp1) + t_offset
        std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
        axes[0].plot(t_local, std_over_time, color=color, linewidth=1.0, alpha=0.6)
        axes[0].axvline(t_offset, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')
        last_t = t_local[-1]

    axes[0].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[0].set_title(f'Intra-cycle spread evolution — {name}')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel('Model time step (cumulative)')
    if last_t is not None:
        axes[0].set_xlim(k_start * M, last_t)

    # Bottom: bar chart of analysis vs forecast spread per cycle
    cycles_plot = np.arange(valid_T)
    width = 0.35
    axes[1].bar(cycles_plot - width/2, analysis_spread_all, width,
                label='After analysis', color=color, alpha=0.5, edgecolor='black')
    axes[1].bar(cycles_plot + width/2, forecast_spread_all, width,
                label='Before next analysis', color=color, alpha=1.0, edgecolor='black')
    axes[1].set_xlabel('DA Cycle')
    axes[1].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[1].set_title('Spread: post-analysis vs end-of-forecast')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2, axis='y')

    # Visually indicate which cycles are shown in the top panel
    if valid_T > 0:
        sel_start = max(0, min(k_start, valid_T - 1))
        sel_end = max(0, min(k_end, valid_T))
        if sel_end > sel_start:
            axes[1].axvspan(sel_start - 0.5, sel_end - 0.5, color='gray', alpha=0.12, zorder=0)

    fig.suptitle(f'Spread Reduction Diagnostic — {name}  |  Ne={cfg.Ne}  |  M={cfg.M}', fontsize=13)
    plt.tight_layout()
    _savefig(f"spread_reduction_{name}")


def plot_spread_reduction_multi(results, palette, cfg, cycle_range=None, var_idx=0):
    """
    Multi-model version of `plot_spread_reduction`.

    - Top: for each model, overlay the intra-cycle spread evolution for the selected cycle window
           (x-axis in cumulative model time steps).
    - Bottom: for each model, show full-run per-cycle spread after analysis vs end-of-forecast
              (x-axis is DA cycle index, not timesteps).
    """
    if not results:
        return

    # Infer global T, M from cfg/first entry, but compute per-model valid_T independently.
    any_res = next(iter(results.values()))
    ens0 = any_res['ens_fcst_traj']
    T0, _, Mp1, _ = ens0.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T0)
    k_start, k_end = cycle_range
    k_end = min(k_end, T0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False, dpi=600)

    max_valid_T = 0
    for name, res in results.items():
        ens_traj = res['ens_fcst_traj']  # (T, Ne, M+1, Nx)
        T, _, Mp1_here, _ = ens_traj.shape
        M_here = Mp1_here - 1
        if M_here != M:
            # If different M per model, the cumulative-time overlay becomes ambiguous.
            # In that case, fall back to each model's own time axis (still decoupled from cycles below).
            M_used = M_here
        else:
            M_used = M

        color = palette.get(name, '#888888')

        # Determine valid cycles until NaNs (divergence)
        valid_T = 0
        analysis_spread_all = []
        forecast_spread_all = []
        for k in range(T):
            if np.any(np.isnan(ens_traj[k])):
                break
            std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
            analysis_spread_all.append(std_over_time[0])
            forecast_spread_all.append(std_over_time[-1])
            valid_T += 1
        max_valid_T = max(max_valid_T, valid_T)

        # Top: selected window only
        for k in range(k_start, min(k_end, valid_T)):
            t_offset = k * M_used
            t_local = np.arange(Mp1_here) + t_offset
            std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
            axes[0].plot(t_local, std_over_time, color=color, linewidth=0.8, alpha=0.60)

        # Bottom: full-run per-cycle series (use lines to avoid unreadable grouped bars)
        cycles = np.arange(valid_T)
        axes[1].plot(cycles, analysis_spread_all, color=color, alpha=0.6, lw=1.0, ls='--',
                     label=f'{name} (post)')
        axes[1].plot(cycles, forecast_spread_all, color=color, alpha=0.6, lw=1.0, ls='-',
                     label=f'{name} (end)')

    # Top styling
    axes[0].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[0].set_title('Intra-cycle spread evolution (selected window)')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel('Model time step (cumulative)')
    axes[0].axvline(k_start * M, color='gray', linewidth=0.6, alpha=0.5, linestyle=':')
    axes[0].axvline(k_end * M, color='gray', linewidth=0.6, alpha=0.5, linestyle=':')
    axes[0].set_xlim(k_start * M, k_end * M)

    # Bottom styling
    axes[1].set_xlabel('DA Cycle')
    axes[1].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[1].set_title('Spread per cycle (full run): post-analysis vs end-of-forecast')
    axes[1].grid(True, alpha=0.2, axis='y')
    if max_valid_T > 0:
        sel_start = max(0, min(k_start, max_valid_T - 1))
        sel_end = max(0, min(k_end, max_valid_T))
        if sel_end > sel_start:
            axes[1].axvspan(sel_start - 0.5, sel_end - 0.5, color='gray', alpha=0.10, zorder=0)
        axes[1].set_xlim(-0.5, max_valid_T - 0.5)

    # Legend (keep it compact)
    axes[1].legend(loc='upper right', fontsize=7, ncol=2, frameon=True)

    fig.suptitle(f'Spread Reduction Diagnostic (multi)  |  Ne={cfg.Ne}  |  M={cfg.M}', fontsize=13)
    plt.tight_layout()
    _savefig("spread_reduction_multi")


def plot_trajectory_comparison(results, palette, cfg, spinup=0, title_suffix=""):
    """
    Compare ensemble forecast mean trajectories against truth for each
    state variable (x, y, z), one subplot per variable.

    Parameters
    ----------
    results      : dict from run_all_models
    palette      : colour dict
    cfg          : EnKFConfig
    spinup       : cycles to skip at the start of the plot (not trimmed
                   from data, just shifts x-axis start for clarity)
    title_suffix : str appended to figure title
    """
    cycles = np.arange(cfg.T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for var_idx, (ax, var_name) in enumerate(zip(axes, VAR_NAMES)):

        # --- truth (same for all models) ---
        xt = results['Lorenz63']['xt_traj'][spinup:, var_idx]
        ax.plot(cycles[spinup:], xt,
                color='black', lw=1.5, ls='--',
                label='truth', zorder=10)

        # --- each model's forecast mean ---
        for name, res in results.items():
            if res['diverged']:
                # plot only up to divergence point, then stop
                valid = ~np.isnan(res['xf_traj'][:, var_idx])
                c = cycles[spinup:][valid[spinup:]]
                v = res['xf_traj'][spinup:, var_idx][valid[spinup:]]
            else:
                c = cycles[spinup:]
                v = res['xf_traj'][spinup:, var_idx]

            color = palette.get(name, '#888888')
            ls    = '--' if name == 'Lorenz63' else '-'
            lw    = 2.0  if name == 'Lorenz63' else 1.2
            alpha = 0.6  if name == 'Lorenz63' else 0.85

            ax.plot(c, v, color=color, lw=lw, ls=ls,
                    alpha=alpha, label=name if var_idx == 0 else '_nolegend_')

        ax.set_ylabel(f'${var_name}$', fontsize=11)
        ax.grid(True, alpha=0.2)

        # mark spinup boundary
        if spinup > 0:
            ax.axvline(spinup, color='gray', lw=0.8,
                       ls=':', alpha=0.6, label='spinup end' if var_idx == 0 else '_nolegend_')

    axes[0].legend(loc='upper right', fontsize=8, ncol=4)
    axes[-1].set_xlabel('DA cycle', fontsize=11)

    fig.suptitle(
        f'Forecast ensemble mean vs truth  |  '
        f'Ne={cfg.Ne}  |  M={cfg.M}  |  σ_obs={cfg.sig_obs}'
        f'{title_suffix}',
        fontsize=11
    )
    plt.tight_layout()
    _savefig("trajectory_comparison")