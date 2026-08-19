"""
pytest suite for _engine_pca.py. Synthetic data only, no network.

Run: python -m pytest bist30_strategy_lab/test_engine_pca.py -v
(from the afistaj/ project root, with the venv active)
"""

import numpy as np
import pandas as pd
import pytest

import _engine_pca as pca
import _common as c


def _hourly_index(n_sessions, bars_per_session=9):
    idx = []
    d = pd.Timestamp("2024-01-01")
    added = 0
    while added < n_sessions:
        if d.weekday() < 5:
            for h in range(bars_per_session):
                idx.append(d + pd.Timedelta(hours=10 + h))
            added += 1
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(idx)


def _synthetic_universe(n_sessions=300, n_tickers=10, seed=0):
    """
    Every ticker = a shared common (market) factor * its own loading, plus
    an independent, mean-reverting idiosyncratic component -- exactly the
    structure PCA stat-arb is meant to exploit: factor 1 should dominate,
    and the leftover residual per ticker should be the traded signal.
    """
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n_sessions)
    n = len(idx)

    market = rng.normal(0, 0.01, n)
    loadings_true = rng.uniform(0.5, 1.5, n_tickers)

    prices = {}
    for k in range(n_tickers):
        idio = np.zeros(n)
        for t in range(1, n):
            idio[t] = idio[t - 1] + 0.08 * (0 - idio[t - 1]) + rng.normal(0, 0.003)
        log_ret = loadings_true[k] * market + np.diff(idio, prepend=0.0)
        prices[f"T{k}.IS"] = np.exp(4.0 + np.cumsum(log_ret))
    return pd.DataFrame(prices, index=idx)


def test_compute_pca_factors_recovers_dominant_common_factor():
    prices = _synthetic_universe(n_sessions=300, n_tickers=10)
    log_ret = np.log(prices).diff().dropna().to_numpy()
    loadings, eigvals = pca.compute_pca_factors(log_ret, n_factors=3)
    assert loadings.shape == (10, 3)
    # first eigenvalue should dominate (the shared market factor)
    assert eigvals[0] > 3 * eigvals[1]
    # all tickers should load onto factor 1 with the same sign (common factor)
    signs = np.sign(loadings[:, 0])
    assert (signs == signs[0]).all() or (signs == -signs[0]).all()


def test_residualize_removes_common_factor():
    prices = _synthetic_universe(n_sessions=300, n_tickers=10)
    log_ret = np.log(prices).diff().dropna()
    fit = log_ret.iloc[:200].to_numpy()
    loadings, _ = pca.compute_pca_factors(fit, n_factors=1)

    raw_corr = np.corrcoef(log_ret.iloc[200:].to_numpy(), rowvar=False)
    residuals = np.array([pca.residualize(row, loadings) for row in log_ret.iloc[200:].to_numpy()])
    resid_corr = np.corrcoef(residuals, rowvar=False)

    off_diag_raw = np.abs(raw_corr[np.triu_indices(10, k=1)]).mean()
    off_diag_resid = np.abs(resid_corr[np.triu_indices(10, k=1)]).mean()
    assert off_diag_resid < off_diag_raw


def test_pca_residuals_uncorrelated_with_top_factor():
    prices = _synthetic_universe(n_sessions=300, n_tickers=10)
    log_ret = np.log(prices).diff().dropna()
    fit = log_ret.iloc[:250].to_numpy()
    loadings, _ = pca.compute_pca_factors(fit, n_factors=1)

    oos = log_ret.iloc[250:].to_numpy()
    factor_scores = oos @ loadings[:, 0]
    residuals = np.array([pca.residualize(row, loadings) for row in oos])
    corr_with_factor = np.array([
        np.corrcoef(residuals[:, i], factor_scores)[0, 1] for i in range(residuals.shape[1])
    ])
    assert np.all(np.abs(corr_with_factor) < 0.15)


def test_walk_forward_pca_end_to_end_synthetic():
    prices = _synthetic_universe(n_sessions=300, n_tickers=10)
    params = {**pca.DEFAULTS, "fit_window": 150, "rebalance": 30, "n_factors": 2, "z_window": 40}
    pnl, n_active, trades, split_idx = pca.walk_forward_pca(prices, params)
    assert len(pnl) == len(prices)
    assert n_active.max() <= 10


def test_walk_forward_pca_skips_refit_when_insufficient_names():
    prices = _synthetic_universe(n_sessions=150, n_tickers=10)
    # cripple all but 3 tickers so the fit-name minimum can't be met
    prices = prices.copy()
    for col in prices.columns[3:]:
        prices.loc[prices.index[50:], col] = np.nan
    params = {**pca.DEFAULTS, "fit_window": 100, "rebalance": 20, "n_factors": 2,
              "min_fit_names": 8, "in_sample_fraction": 0.3}
    pnl, n_active, trades, split_idx = pca.walk_forward_pca(prices, params)
    assert n_active.max() == 0  # never enough names to fit -> never trades


def test_run_full_backtest_pca_is_deterministic():
    prices = _synthetic_universe(n_sessions=250, n_tickers=8)
    params = {"fit_window": 120, "rebalance": 25, "n_factors": 2}
    r1 = pca.run_full_backtest_pca(prices, params=params, bars_per_year=9 * 252)
    r2 = pca.run_full_backtest_pca(prices, params=params, bars_per_year=9 * 252)
    np.testing.assert_array_equal(r1["results"]["pnl"].to_numpy(), r2["results"]["pnl"].to_numpy())


def test_add_capital_pnl_never_implies_leverage_for_pca():
    prices = _synthetic_universe(n_sessions=250, n_tickers=8)
    params = {"fit_window": 120, "rebalance": 25, "n_factors": 2}
    result = pca.run_full_backtest_pca(prices, params=params, bars_per_year=9 * 252)
    out = c.add_capital_pnl(result, starting_capital=50_000)
    assert out["dollar_per_unit"] == pytest.approx(50_000)
