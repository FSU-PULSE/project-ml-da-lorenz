"""Tests for CuPy particle filter and GPU surrogate rollouts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("cupy")
import cupy as cp

from PF_core import PFConfig, make_lorenz_forecaster, run_pf, systematic_resample
from PF_core_cupy import make_lorenz_forecaster_gpu, run_pf_cupy
from metrics_cupy import energy_score_cupy


cuda_available = torch.cuda.is_available()
pytestmark_gpu = pytest.mark.skipif(
    not cuda_available, reason="CUDA required for CuPy PF tests"
)


def test_systematic_resample_valid_indices():
    rng = np.random.RandomState(0)
    w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    idx = systematic_resample(w, rng)
    assert idx.shape == (4,)
    assert idx.min() >= 0
    assert idx.max() < 4


@pytestmark_gpu
def test_energy_score_cupy_finite():
    rng = cp.random.RandomState(1)
    ens = rng.standard_normal((3, 20))
    truth = rng.standard_normal(3)
    es, acc, spr = energy_score_cupy(ens, truth)
    assert np.isfinite(es)
    assert np.isfinite(acc)
    assert np.isfinite(spr)
    assert abs(es - (acc - spr)) < 1e-10


@pytestmark_gpu
def test_run_pf_cupy_identity_forecaster():
    """Smoke test: GPU PF loop with trivial forecast (no surrogate)."""
    Nx, Ne, M, T = 3, 8, 2, 2

    def forecaster(states):
        Ne_loc = states.shape[1]
        out = cp.empty((Ne_loc, M + 1, Nx), dtype=cp.float64)
        out[:, 0, :] = states.T
        for m in range(1, M + 1):
            out[:, m, :] = states.T + 0.01 * m
        return out

    pool = cp.random.RandomState(0).standard_normal((20, Nx))
    xt_0 = np.zeros(Nx)
    cfg = PFConfig(
        T=T, M=M, Ne=Ne, dt=0.01, p=1.0, sig_obs=0.5,
        NER=0.5, reg=0.1, obs_seed=42, filter_seed=43,
    )
    res = run_pf_cupy(forecaster, xt_0, pool, cfg, model_name='mock')
    assert 'errorf' in res
    assert res['errorf'].shape == (T,)
    assert res['Xf_all'].shape == (T, Nx, Ne)
    assert isinstance(res['errorf'], np.ndarray)


@pytestmark_gpu
def test_batch_rollout_tensor_matches_numpy():
    """SurrogateModel GPU tensor rollout vs NumPy batch_rollout."""
    from os.path import join
    from SurrogateModel import SurrogateModel

    path = join(
        "models", "DenseNN_L63_trial1_1775267887_best_model.pth"
    )
    if not __import__("os").path.isfile(path):
        pytest.skip("DenseNN checkpoint not present")

    model = SurrogateModel(path, device="cuda")
    rng = np.random.RandomState(0)
    B = 16
    hist = rng.randn(B, 1, 3)
    steps = 5
    np_out = model.batch_rollout(hist, steps)
    t_out = model.batch_rollout_tensor(hist, steps).detach().cpu().numpy()
    np.testing.assert_allclose(np_out, t_out, rtol=1e-10, atol=1e-12)


@pytestmark_gpu
def test_run_pf_cupy_matches_numpy_pf_lorenz():
    """Filter path (RNG, resample, jitter) matches PF_core when forecast does."""
    Nx, Ne, M, T = 3, 12, 3, 4
    dt = 0.01
    rng = np.random.RandomState(7)
    pool = rng.standard_normal((30, Nx))
    xt_0 = rng.standard_normal(Nx)
    cfg = PFConfig(
        T=T, M=M, Ne=Ne, dt=dt, p=1.0, sig_obs=0.8,
        NER=0.8, reg=0.5, jitter=True, nuj=True,
        obs_seed=11, filter_seed=22,
    )
    fn_np = make_lorenz_forecaster(dt, M)
    fn_gpu = make_lorenz_forecaster_gpu(dt, M)
    res_np = run_pf(fn_np, xt_0.copy(), pool.copy(), cfg, model_name='L63')
    res_gpu = run_pf_cupy(fn_gpu, xt_0.copy(), pool.copy(), cfg, model_name='L63')

    for key in (
        'errorf', 'errora', 'spread', 'n_eff', 'resampled',
        'weights_f', 'weights_a', 'Xf_all', 'Xa_all',
        'xf_traj', 'xa_traj', 'xt_traj',
    ):
        np.testing.assert_allclose(
            res_np[key], res_gpu[key], rtol=0, atol=1e-12, err_msg=key,
        )
    assert res_np['diverged'] == res_gpu['diverged']
