"""Tests for HDF5 EnKF results storage."""

import numpy as np

from enkf_results_store import (
    CYCLE_METRICS,
    EnKFResultsStore,
    load_enkf_results_hdf5,
    per_ic_delta_es_pct,
    summarize_cycles,
    write_summary_table,
    write_summary_table_md,
)


def _fake_run_result(T: int, diverged: bool = False) -> dict:
    t = np.arange(T, dtype=float)
    return {
        'errorf': t * 0.1,
        'errora': t * 0.2,
        'spread': np.ones(T),
        'errorf_es': t * 0.05,
        'errora_es': t * 0.06,
        'errorf_es_acc': t * 0.03,
        'errorf_es_spr': t * 0.02,
        'errora_es_acc': t * 0.04,
        'errora_es_spr': t * 0.01,
        'diverged': diverged,
    }


def test_hdf5_roundtrip(tmp_path):
    T, n_ic = 4, 2
    models = ['Lorenz63', 'DenseNN']
    store = EnKFResultsStore(n_ic, T, models)
    for iic in range(n_ic):
        ic = np.array([float(iic), 0.0, 1.0])
        store.record(
            iic,
            ic,
            seed=100 + iic,
            results={m: _fake_run_result(T, diverged=(iic == 1)) for m in models},
        )

    path = store.to_hdf5(tmp_path / 'test.h5', attrs={'T': T})
    loaded = load_enkf_results_hdf5(path)

    assert loaded['attrs']['n_ic'] == n_ic
    assert loaded['attrs']['n_cycles'] == T
    np.testing.assert_array_equal(loaded['initial_conditions'], store.initial_conditions)
    for model in models:
        for metric in CYCLE_METRICS:
            np.testing.assert_array_equal(
                loaded['models'][model][metric],
                store.models[model][metric],
            )


def test_summarize_cycles_long_format(tmp_path):
    store = EnKFResultsStore(1, 3, ['M0'])
    store.record(0, np.zeros(3), 7, {'M0': _fake_run_result(3)})
    df = summarize_cycles(store)
    assert len(df) == 1
    assert df.loc[0, 'model'] == 'M0'
    assert df.loc[0, 'errorf_mean'] == np.nanmean([0.0, 0.1, 0.2])


def test_summarize_cycles_cycle_burn_in(tmp_path):
    store = EnKFResultsStore(1, 3, ['M0'])
    store.record(0, np.zeros(3), 7, {'M0': _fake_run_result(3)})
    df = summarize_cycles(store, cycle_start=1)
    assert df.loc[0, 'errorf_mean'] == np.nanmean([0.1, 0.2])


def test_per_ic_delta_es_pct():
    es_f = np.array([2.0, 4.0])
    es_a = np.array([1.0, 2.0])
    np.testing.assert_array_almost_equal(
        per_ic_delta_es_pct(es_f, es_a), [50.0, 50.0]
    )


def test_write_summary_table_md(tmp_path):
    store = EnKFResultsStore(2, 3, ['M0'])
    for iic in range(2):
        store.record(iic, np.zeros(3), 7 + iic, {'M0': _fake_run_result(3)})
    df = summarize_cycles(
        store, metrics=('errorf', 'errora', 'errorf_es', 'errora_es', 'spread')
    )
    path = write_summary_table_md(df, tmp_path / 'table.md', model_order=['M0'], decimals=2)
    text = path.read_text(encoding='utf-8')
    assert '| M0 |' in text
    assert 'RMSEᶠ' in text
    assert 'ΔES%' in text


def test_write_summary_table_csv(tmp_path):
    store = EnKFResultsStore(2, 3, ['M0'])
    for iic in range(2):
        store.record(iic, np.zeros(3), 7 + iic, {'M0': _fake_run_result(3)})
    df = summarize_cycles(
        store, metrics=('errorf', 'errora', 'errorf_es', 'errora_es', 'spread')
    )
    path = write_summary_table(df, tmp_path / 'table', fmt='csv', model_order=['M0'])
    text = path.read_text(encoding='utf-8')
    assert path.suffix == '.csv'
    assert 'RMSE_f_mean' in text.splitlines()[0]
    assert 'M0' in text
