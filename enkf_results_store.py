"""HDF5 storage for multi-IC EnKF benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Per-cycle scalars returned by EnKF_core.run_enkf (excludes trajectories).
CYCLE_METRICS = (
    'errorf',
    'errora',
    'spread',
    'errorf_es',
    'errora_es',
    'errorf_es_acc',
    'errorf_es_spr',
    'errora_es_acc',
    'errora_es_spr',
)


class EnKFResultsStore:
    """Pre-allocated in-memory buffers; write once to HDF5 at the end."""

    def __init__(self, n_ic: int, n_cycles: int, model_names: list[str]):
        self.n_ic = n_ic
        self.n_cycles = n_cycles
        self.model_names = list(model_names)
        self.seed = np.zeros(n_ic, dtype=np.int64)
        self.ic_index = np.arange(n_ic, dtype=np.int64)
        self.initial_conditions = np.full((n_ic, 3), np.nan, dtype=np.float64)
        self._filled = 0
        self.models: dict[str, dict[str, np.ndarray]] = {}
        for name in self.model_names:
            self.models[name] = {
                metric: np.full((n_ic, n_cycles), np.nan, dtype=np.float64)
                for metric in CYCLE_METRICS
            }
            self.models[name]['diverged'] = np.zeros(n_ic, dtype=bool)

    def record(self, iic: int, ic: np.ndarray, seed: int, results: dict[str, dict]) -> None:
        """Store one initial-condition experiment."""
        self.seed[iic] = seed
        self.initial_conditions[iic] = ic
        for model_name, res in results.items():
            if model_name not in self.models:
                raise KeyError(
                    f"Unknown model {model_name!r}; expected one of {self.model_names}"
                )
            buf = self.models[model_name]
            for metric in CYCLE_METRICS:
                buf[metric][iic, :] = res[metric]
            buf['diverged'][iic] = bool(res['diverged'])
        self._filled += 1

    def to_hdf5(
        self,
        path: str | Path,
        *,
        attrs: dict[str, Any] | None = None,
        compression: str = 'gzip',
    ) -> Path:
        """Write all buffers to HDF5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as f:
            if attrs:
                for key, val in attrs.items():
                    if isinstance(val, (dict, list)):
                        f.attrs[key] = json.dumps(val)
                    else:
                        f.attrs[key] = val
            f.attrs['n_ic'] = self.n_ic
            f.attrs['n_cycles'] = self.n_cycles
            f.attrs['model_names'] = json.dumps(self.model_names)
            f.attrs['cycle_metrics'] = json.dumps(list(CYCLE_METRICS))

            f.create_dataset('ic_index', data=self.ic_index)
            f.create_dataset('seed', data=self.seed)
            f.create_dataset('initial_conditions', data=self.initial_conditions)

            models_grp = f.create_group('models')
            for model_name, arrays in self.models.items():
                mg = models_grp.create_group(model_name)
                for key, arr in arrays.items():
                    mg.create_dataset(key, data=arr, compression=compression)

        return path


def load_enkf_results_hdf5(path: str | Path) -> dict[str, Any]:
    """Load HDF5 file into nested dicts (mirrors EnKFResultsStore layout)."""
    path = Path(path)
    out: dict[str, Any] = {'path': path, 'attrs': {}, 'models': {}}
    with h5py.File(path, 'r') as f:
        out['attrs'] = dict(f.attrs)
        out['ic_index'] = f['ic_index'][:]
        out['seed'] = f['seed'][:]
        out['initial_conditions'] = f['initial_conditions'][:]
        for model_name in f['models']:
            out['models'][model_name] = {
                key: f[f'models/{model_name}/{key}'][:]
                for key in f[f'models/{model_name}']
            }
    if 'model_names' in out['attrs']:
        out['model_names'] = json.loads(out['attrs']['model_names'])
    return out


def summarize_cycles(
    store_or_path: EnKFResultsStore | str | Path,
    metrics: tuple[str, ...] = ('errorf', 'errora', 'errorf_es', 'errora_es'),
    cycle_start: int = 0,
    cycle_stop: int | None = None,
) -> 'pd.DataFrame':
    """Long-format summary: one row per (ic_index, model) with mean/std per metric.

    Means and stds are computed over cycles ``cycle_start:cycle_stop`` only
    (default: all cycles).
    """
    import pandas as pd

    cycle_slice = slice(cycle_start, cycle_stop)

    if isinstance(store_or_path, EnKFResultsStore):
        ic_index = store_or_path.ic_index
        seed = store_or_path.seed
        ics = store_or_path.initial_conditions
        models = store_or_path.models
        model_names = store_or_path.model_names
    else:
        data = load_enkf_results_hdf5(store_or_path)
        ic_index = data['ic_index']
        seed = data['seed']
        ics = data['initial_conditions']
        models = data['models']
        model_names = data.get('model_names', list(models.keys()))

    rows = []
    for iic in range(len(ic_index)):
        for model_name in model_names:
            if model_name not in models:
                continue
            m = models[model_name]
            row = {
                'ic_index': int(ic_index[iic]),
                'seed': int(seed[iic]),
                'x': ics[iic, 0],
                'y': ics[iic, 1],
                'z': ics[iic, 2],
                'model': model_name,
                'diverged': bool(m['diverged'][iic]),
            }
            for metric in metrics:
                vals = m[metric][iic][cycle_slice]
                row[f'{metric}_mean'] = float(np.nanmean(vals))
                row[f'{metric}_std'] = float(np.nanstd(vals))
            rows.append(row)
    return pd.DataFrame(rows)


SUMMARY_TABLE_METRICS = ('errorf', 'errora', 'errorf_es', 'errora_es', 'spread')


def _mu_sigma_str(values: np.ndarray, decimals: int = 4) -> str:
    """Format nan-aware mean ± std for markdown table cells."""
    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    fmt = f'{{:.{decimals}f}}'
    return f'{fmt.format(mu)} ± {fmt.format(sigma)}'


def per_ic_delta_es_pct(es_f: np.ndarray, es_a: np.ndarray) -> np.ndarray:
    """Relative ES improvement from assimilation (%), per IC."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return (es_f - es_a) / es_f * 100.0


SUMMARY_TABLE_FORMATS = ('md', 'csv')


def build_summary_table(
    df: 'pd.DataFrame',
    *,
    model_order: list[str] | None = None,
) -> 'pd.DataFrame':
    """Aggregate per-model μ and σ across ICs (numeric columns)."""
    import pandas as pd

    required = {
        'model',
        'errorf_mean',
        'errora_mean',
        'errorf_es_mean',
        'errora_es_mean',
        'spread_mean',
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'summary table missing columns: {sorted(missing)}')

    models = model_order or sorted(df['model'].unique())
    models = [m for m in models if m in df['model'].values]

    rows = []
    for model in models:
        sub = df.loc[df['model'] == model]
        delta_es = per_ic_delta_es_pct(
            sub['errorf_es_mean'].to_numpy(),
            sub['errora_es_mean'].to_numpy(),
        )
        rows.append({
            'Method': model,
            'RMSE_f_mean': float(np.nanmean(sub['errorf_mean'])),
            'RMSE_f_std': float(np.nanstd(sub['errorf_mean'])),
            'RMSE_a_mean': float(np.nanmean(sub['errora_mean'])),
            'RMSE_a_std': float(np.nanstd(sub['errora_mean'])),
            'ES_f_mean': float(np.nanmean(sub['errorf_es_mean'])),
            'ES_f_std': float(np.nanstd(sub['errorf_es_mean'])),
            'ES_a_mean': float(np.nanmean(sub['errora_es_mean'])),
            'ES_a_std': float(np.nanstd(sub['errora_es_mean'])),
            'Spread_mean': float(np.nanmean(sub['spread_mean'])),
            'Spread_std': float(np.nanstd(sub['spread_mean'])),
            'delta_ES_pct_mean': float(np.nanmean(delta_es)),
            'delta_ES_pct_std': float(np.nanstd(delta_es)),
        })
    return pd.DataFrame(rows)


def _format_agg_mu_sigma(mu: float, sigma: float, decimals: int) -> str:
    fmt = f'{{:.{decimals}f}}'
    return f'{fmt.format(mu)} ± {fmt.format(sigma)}'


def write_summary_table(
    df: 'pd.DataFrame',
    path: str | Path,
    *,
    fmt: str = 'csv',
    model_order: list[str] | None = None,
    decimals: int = 4,
    title: str = 'EnKF summary scalars',
    notes: list[str] | None = None,
) -> Path:
    """Write μ ± σ summary table (one row per model) to markdown or CSV."""
    import pandas as pd

    fmt = fmt.lower()
    if fmt not in SUMMARY_TABLE_FORMATS:
        raise ValueError(f"fmt must be one of {SUMMARY_TABLE_FORMATS}, got {fmt!r}")

    path = Path(path)
    if path.suffix.lower() not in ('.md', '.csv'):
        path = path.with_suffix(f'.{fmt}')

    numeric_df = build_summary_table(df, model_order=model_order)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == 'csv':
        numeric_df.to_csv(path, index=False, float_format=f'%.{decimals}f')
        return path

    pairs = [
        ('RMSEᶠ', 'RMSE_f_mean', 'RMSE_f_std'),
        ('RMSEᵃ', 'RMSE_a_mean', 'RMSE_a_std'),
        ('ESᶠ', 'ES_f_mean', 'ES_f_std'),
        ('ESᵃ', 'ES_a_mean', 'ES_a_std'),
        ('Spread', 'Spread_mean', 'Spread_std'),
        ('ΔES%', 'delta_ES_pct_mean', 'delta_ES_pct_std'),
    ]
    display_df = pd.DataFrame({'Method': numeric_df['Method']})
    for label, mu_col, sig_col in pairs:
        display_df[label] = [
            _format_agg_mu_sigma(m, s, decimals)
            for m, s in zip(numeric_df[mu_col], numeric_df[sig_col])
        ]

    lines = [
        f'# {title}',
        '',
        '| Method | RMSEᶠ | RMSEᵃ | ESᶠ | ESᵃ | Spread | ΔES% |',
        '|--------|-------|-------|-----|-----|--------|------|',
    ]
    for _, row in display_df.iterrows():
        lines.append(
            '| {Method} | {RMSEᶠ} | {RMSEᵃ} | {ESᶠ} | {ESᵃ} | {Spread} | {ΔES%} |'.format(
                **row.to_dict()
            )
        )
    if notes:
        lines.extend(['', *notes])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def write_summary_table_md(
    df: 'pd.DataFrame',
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Backward-compatible alias for ``write_summary_table(..., fmt='md')``."""
    return write_summary_table(df, path, fmt='md', **kwargs)
