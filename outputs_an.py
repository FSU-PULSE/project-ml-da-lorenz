# %%
import pandas as pd
import matplotlib.pyplot as plt
from enkf_results_store import (
    load_enkf_results_hdf5,
    summarize_cycles,
    SUMMARY_TABLE_METRICS,
    write_summary_table,
)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

surrogates_palette = {
        'Lorenz63': '#000000',
        'Truth': '#000000',
        'DenseNN': '#D85A30',
        'ResDenseNN': '#7F770A',
        'LSTMNN': '#7F77DD',
        'RNN_relu': '#1D9E75',
        'RNN_tanh': '#1DDE75',
    }

CYCLE_START = 50
EXCLUDE_DIVERGED = False
TABLE_OUTPUT_FORMAT = 'csv'  # 'csv' or 'md'
_MODEL_ORDER = ['Lorenz63', 'DenseNN', 'ResDenseNN', 'LSTMNN', 'RNN_relu', 'RNN_tanh']


def _ic_mask(results, model, exclude_diverged=EXCLUDE_DIVERGED):
    d = results['models'][model]['diverged']
    return ~d if exclude_diverged else np.ones(len(d), dtype=bool)


def metric_ST(results, model, metric, cycle_start=CYCLE_START, exclude_diverged=EXCLUDE_DIVERGED):
    """Return (S, T) array for one model/metric after IC mask and cycle slice."""
    m = results['models'][model][metric]
    mask = _ic_mask(results, model, exclude_diverged)
    return m[mask, cycle_start:]


def plot_metric_evolution(metric_ST, label, ax, color, cycle_start=CYCLE_START):
    mu = np.nanmean(metric_ST, axis=0)
    q25, q75 = np.nanpercentile(metric_ST, [25, 75], axis=0)
    cycles = np.arange(metric_ST.shape[1]) + cycle_start
    ax.plot(cycles, mu, color=color, lw=1.5, label=label)
    ax.fill_between(cycles, q25, q75, color=color, alpha=0.2)


def plot_temporal_evolution(results, model, cycle_start=CYCLE_START, exclude_diverged=EXCLUDE_DIVERGED):
    """2x2 temporal metrics with IQR bands across ICs for one model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rmse_f = metric_ST(results, model, 'errorf', cycle_start, exclude_diverged)
    rmse_a = metric_ST(results, model, 'errora', cycle_start, exclude_diverged)
    es_f = metric_ST(results, model, 'errorf_es', cycle_start, exclude_diverged)
    es_a = metric_ST(results, model, 'errora_es', cycle_start, exclude_diverged)
    acc_f = metric_ST(results, model, 'errorf_es_acc', cycle_start, exclude_diverged)
    spr_f = metric_ST(results, model, 'errorf_es_spr', cycle_start, exclude_diverged)
    spread = metric_ST(results, model, 'spread', cycle_start, exclude_diverged)

    plot_metric_evolution(rmse_f, 'Forecast RMSE', axes[0, 0], 'steelblue', cycle_start)
    plot_metric_evolution(rmse_a, 'Analysis RMSE', axes[0, 0], 'tomato', cycle_start)
    plot_metric_evolution(es_f, 'Forecast ES', axes[0, 1], 'steelblue', cycle_start)
    plot_metric_evolution(es_a, 'Analysis ES', axes[0, 1], 'tomato', cycle_start)
    plot_metric_evolution(acc_f, 'Accuracy term', axes[1, 0], 'purple', cycle_start)
    plot_metric_evolution(spr_f, 'Spread term', axes[1, 0], 'orange', cycle_start)
    plot_metric_evolution(spread, 'Marginal spread', axes[1, 1], 'green', cycle_start)
    plot_metric_evolution(rmse_a, 'Analysis RMSE', axes[1, 1], 'tomato', cycle_start)

    panel_titles = [
        'RMSE',
        'Energy Score',
        'Forecast ES decomposition',
        'Spread vs RMSE (calibration)',
    ]
    for ax, title in zip(axes.ravel(), panel_titles):
        ax.set_title(title)
        ax.set_xlabel('DA cycle')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    n_ic = rmse_f.shape[0]
    fig.suptitle(
        f'{model} — temporal evolution (n_IC={n_ic}, cycles ≥ {cycle_start}, seed={seed})'
        f'diverged excluded={exclude_diverged})',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout()
    return fig


def plot_es_decomposition_boxplots(results, models, cycle_start=CYCLE_START, exclude_diverged=EXCLUDE_DIVERGED):
    """Boxplots of cycle-mean forecast ES accuracy vs spread terms, one panel per model."""
    n = len(models)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes, models):
        acc_f = np.nanmean(metric_ST(results, model, 'errorf_es_acc', cycle_start, exclude_diverged), axis=1)
        spr_f = np.nanmean(metric_ST(results, model, 'errorf_es_spr', cycle_start, exclude_diverged), axis=1)
        bp = ax.boxplot(
            [acc_f, spr_f],
            tick_labels=['Accuracy', 'Spread'],
            patch_artist=True,
            showmeans=True,
        )
        color = surrogates_palette.get(model, '#888888')
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor(color)
        bp['boxes'][1].set_alpha(0.35)
        plt.setp(bp['medians'][0], color='black', linewidth=2)
        plt.setp(bp['medians'][1], color='black', linewidth=2)
        plt.setp(
            bp['means'][0],
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=5,
        )
        plt.setp(
            bp['means'][1],
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=5,
        )
        ax.set_title(model, fontweight='bold')
        ax.set_ylabel('ES component (cycle mean)')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Forecast ES decomposition across ICs (cycles ≥ {cycle_start}, '
        f'diverged excluded={exclude_diverged})',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout()
    return fig


#results_h5_path = f'results/ms_denkf_L63_seed_31_ics_1000.h5'
results_h5_path = f'results/ms_denkf_L63_seed_99_ics_1000.h5'
experiment_name = results_h5_path.split('_')[1]
seed = results_h5_path.split('_')[4]
results = load_enkf_results_hdf5(results_h5_path)
results_df = summarize_cycles(
    results_h5_path, cycle_start=CYCLE_START, metrics=SUMMARY_TABLE_METRICS
)
plot_df = results_df[~results_df['diverged']].copy() if EXCLUDE_DIVERGED else results_df.copy()
plot_models = [m for m in _MODEL_ORDER if m in plot_df['model'].values]
print(results_df.head())
print(f"Plotting {len(plot_df)} / {len(results_df)} rows (cycle_start={CYCLE_START}, exclude_diverged={EXCLUDE_DIVERGED})")


# %% Temporal evolution (2x2 IQR across ICs) — one figure per model
for model in plot_models:
    plot_temporal_evolution(results, model)
    out = f'results/temporal_evolution_{experiment_name}_{model}_ic_start{CYCLE_START}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    print(f'Saved {out}')


# %% Forecast ES decomposition — accuracy vs spread boxplots per model
plot_es_decomposition_boxplots(results, plot_models)
es_out = f'results/es_decomposition_forecast_{experiment_name}_seed_{seed}_ic_start{CYCLE_START}.png'
plt.savefig(es_out, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()
print(f'Saved {es_out}')

print(f'\nForecast ES decomposition medians (cycles ≥ {CYCLE_START}):')
for model in plot_models:
    acc_f = np.nanmean(metric_ST(results, model, 'errorf_es_acc'), axis=1)
    spr_f = np.nanmean(metric_ST(results, model, 'errorf_es_spr'), axis=1)
    es_f = np.nanmean(metric_ST(results, model, 'errorf_es'), axis=1)
    print(
        f'  {model:12s}  acc_med={np.nanmedian(acc_f):.4f}  spr_med={np.nanmedian(spr_f):.4f}  '
        f'es_med={np.nanmedian(es_f):.4f}  (acc−spr)={np.nanmedian(acc_f - spr_f):.4f}'
    )


# Create 3d scatter plot of all the initial conditions for each model
ics = results_df[results_df['model'] == 'Lorenz63'][['x', 'y', 'z']]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ics['x'], ics['y'], ics['z'], c='b', marker='.', s=20, alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Initial Conditions')
plt.tight_layout()
plt.show()


# %% Boxplot by model for errorf_mean, errora_mean, errorf_es_mean, errora_es_mean
targets = ['errorf_mean', 'errora_mean', 'errorf_es_mean', 'errora_es_mean']
titles = [
    f'Forecast RMSE (cycles ≥ {CYCLE_START})',
    f'Analysis RMSE (cycles ≥ {CYCLE_START})',
    f'Forecast ES (cycles ≥ {CYCLE_START})',
    f'Analysis ES (cycles ≥ {CYCLE_START})',
]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, target, title in zip(axes, targets, titles):
    data = [plot_df.loc[plot_df['model'] == m, target].values for m in plot_models]
    bp = ax.boxplot(data, tick_labels=plot_models, patch_artist=True, showmeans=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(target.replace('_mean', '').replace('_', ' '), fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for i, model in enumerate(plot_models):
        color = surrogates_palette.get(model, '#888888')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.6)
        plt.setp(bp['medians'][i], color='black', linewidth=2)
        plt.setp(
            bp['means'][i],
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=5,
        )
    ax.tick_params(axis='x', rotation=30)

fig.suptitle(
    f'{experiment_name} metrics across ICs (cycle_start={CYCLE_START}, diverged excluded={EXCLUDE_DIVERGED})',
    fontsize=13,
    fontweight='bold',
)
plt.tight_layout()
out_name = f'results/rmse_es_comparison_boxplots_{experiment_name}_seed_{seed}_ic_start{CYCLE_START}.png'
plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()

# %% Summary scalar table (μ ± σ across ICs)
table_path = f'results/summary_table_{experiment_name}_seed_{seed}_ic_start{CYCLE_START}'
table_notes = [
    f'- Source: `{results_h5_path}`',
    f'- Cycles: mean over indices ≥ {CYCLE_START}',
    f'- ICs: {len(plot_df) // len(plot_models)} per model'
    f' ({len(plot_df)} rows total; diverged excluded={EXCLUDE_DIVERGED})',
    '- ΔES%: `(ESᶠ − ESᵃ) / ESᶠ × 100` per IC, then μ ± σ across ICs',
]
write_summary_table(
    plot_df,
    table_path,
    fmt=TABLE_OUTPUT_FORMAT,
    model_order=plot_models,
    decimals=4,
    notes=table_notes if TABLE_OUTPUT_FORMAT == 'md' else None,
)
print(f'Wrote results/summary_table_{experiment_name}_seed_{seed}_ic_start{CYCLE_START}.{TABLE_OUTPUT_FORMAT}')
# %%
