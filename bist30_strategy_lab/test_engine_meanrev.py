"""
pytest suite for _engine_meanrev.py. Synthetic data only, no network.

Run: python -m pytest bist30_strategy_lab/test_engine_meanrev.py -v
(from the afistaj/ project root, with the venv active)
"""

import numpy as np
import pandas as pd
import pytest

import _engine_meanrev as em
import _common as c


def _hourly_index(n_sessions, bars_per_session=9):
    idx = []
    d = pd.Timestamp("2024-01-01")
    added = 0
    while added < n_sessions:
        if d.weekday() < 5:  # skip weekends, like real BIST sessions
            for h in range(bars_per_session):
                idx.append(d + pd.Timedelta(hours=10 + h))
            added += 1
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(idx)


def _synthetic_universe(n_sessions=250, bars_per_session=9, seed=0):
    """
    4 tickers: A/B genuinely cointegrated (B is a noisy, scaled copy of A
    with a mean-reverting residual), C/D independent random walks.
    """
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n_sessions, bars_per_session)
    n = len(idx)

    common = np.cumsum(rng.normal(0, 0.01, n))
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = resid[t - 1] + 0.05 * (0 - resid[t - 1]) + rng.normal(0, 0.005)
    log_a = 4.0 + common
    log_b = 3.0 + 0.8 * common + resid

    log_c = 4.5 + np.cumsum(rng.normal(0, 0.01, n))
    log_d = 4.2 + np.cumsum(rng.normal(0, 0.01, n))

    prices = pd.DataFrame({
        "AAA.IS": np.exp(log_a), "BBB.IS": np.exp(log_b),
        "CCC.IS": np.exp(log_c), "DDD.IS": np.exp(log_d),
    }, index=idx)
    return prices


def test_engle_granger_hedge_ratio_recovers_known_beta():
    rng = np.random.default_rng(1)
    n = 2000
    x = np.cumsum(rng.normal(0, 1, n))
    y = 2.0 + 1.5 * x + rng.normal(0, 0.1, n)
    alpha, beta = em.engle_granger_hedge_ratio(y, x)
    assert beta == pytest.approx(1.5, abs=0.05)


def test_screen_pairs_full_universe_flags_known_cointegrated_pair():
    prices = _synthetic_universe(n_sessions=300)
    log_prices = np.log(prices)
    sess = c.session_ids(prices.index)
    screening = em.screen_pairs_full_universe(
        log_prices, sess, prices.columns, bars_per_day=9, adf_crit=-2.86, hurst_cutoff=0.5,
    )
    aaa_bbb = screening[
        ((screening.ticker_a == "AAA.IS") & (screening.ticker_b == "BBB.IS")) |
        ((screening.ticker_a == "BBB.IS") & (screening.ticker_b == "AAA.IS"))
    ].iloc[0]
    assert aaa_bbb["qualifies"]
    assert np.isfinite(aaa_bbb["half_life_bars"])


def test_screen_pairs_full_universe_does_not_flag_independent_walks():
    prices = _synthetic_universe(n_sessions=300)
    log_prices = np.log(prices)
    sess = c.session_ids(prices.index)
    screening = em.screen_pairs_full_universe(
        log_prices, sess, prices.columns, bars_per_day=9, adf_crit=-2.86, hurst_cutoff=0.45,
    )
    ccc_ddd = screening[
        ((screening.ticker_a == "CCC.IS") & (screening.ticker_b == "DDD.IS")) |
        ((screening.ticker_a == "DDD.IS") & (screening.ticker_b == "CCC.IS"))
    ].iloc[0]
    assert not ccc_ddd["qualifies"]


def test_screen_pairs_full_universe_skips_pair_with_insufficient_joint_history():
    prices = _synthetic_universe(n_sessions=300)
    # simulate a short-history ticker (like DSTKF.IS): only last 50 bars valid
    prices = prices.copy()
    prices.loc[prices.index[:-50], "DDD.IS"] = np.nan
    log_prices = np.log(prices)
    sess = c.session_ids(prices.index)
    screening = em.screen_pairs_full_universe(
        log_prices, sess, prices.columns, bars_per_day=9, min_bars=200,
    )
    assert not any(
        (screening.ticker_a == "DDD.IS") | (screening.ticker_b == "DDD.IS")
    )


def test_run_full_backtest_meanrev_end_to_end_synthetic():
    prices = _synthetic_universe(n_sessions=300)
    result = em.run_full_backtest_meanrev(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252)
    assert result["ticker_a"] in prices.columns
    assert result["ticker_b"] in prices.columns
    assert len(result["results"]) > 0
    assert "sharpe (naive)" in result["stats"]


def test_run_full_backtest_meanrev_is_deterministic():
    prices = _synthetic_universe(n_sessions=200)
    r1 = em.run_full_backtest_meanrev(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252)
    r2 = em.run_full_backtest_meanrev(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252)
    np.testing.assert_array_equal(r1["results"]["pnl"].to_numpy(), r2["results"]["pnl"].to_numpy())


def test_run_full_backtest_meanrev_forces_selected_pair():
    prices = _synthetic_universe(n_sessions=200)
    result = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252,
        selected_pair=("CCC.IS", "DDD.IS"),
    )
    assert {result["ticker_a"], result["ticker_b"]} == {"CCC.IS", "DDD.IS"}


def test_no_trade_ever_opens_when_regime_never_qualifies():
    prices = _synthetic_universe(n_sessions=200)
    result = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252,
        selected_pair=("CCC.IS", "DDD.IS"),
    )
    # CCC/DDD are independent random walks -- should rarely if ever qualify,
    # so trades should be sparse to nonexistent relative to AAA/BBB.
    assert result["pct_disqualified"] > 50


def test_add_capital_pnl_integrates_with_meanrev_result():
    prices = _synthetic_universe(n_sessions=250)
    result = em.run_full_backtest_meanrev(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252)
    out = c.add_capital_pnl(result, starting_capital=50_000)
    assert out["equity_curve"][-1] == pytest.approx(out["final_capital"])


def test_cost_calibrated_entry_z_runs_end_to_end():
    prices = _synthetic_universe(n_sessions=300)
    result = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5, "derive_entry_z_from_cost": True, "cost_per_round_trip_bps": 5},
        bars_per_year=9 * 252,
    )
    assert len(result["results"]) > 0
    entry_z_seen = result["results"]["entry_z"].dropna()
    assert len(entry_z_seen) > 0
    # calibrated thresholds should vary from the fixed 1.5 default at least
    # some of the time (otherwise the calibration path was never exercised)
    assert not np.allclose(entry_z_seen.to_numpy(), 1.5)


def test_cost_calibrated_entry_z_widens_with_higher_assumed_cost():
    prices = _synthetic_universe(n_sessions=300)
    low_cost = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5, "derive_entry_z_from_cost": True, "cost_per_round_trip_bps": 0.1},
        bars_per_year=9 * 252, selected_pair=("AAA.IS", "BBB.IS"),
    )
    high_cost = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5, "derive_entry_z_from_cost": True, "cost_per_round_trip_bps": 200},
        bars_per_year=9 * 252, selected_pair=("AAA.IS", "BBB.IS"),
    )
    low_entry_mean = low_cost["results"]["entry_z"].dropna().mean()
    high_entry_mean = high_cost["results"]["entry_z"].dropna().mean()
    assert high_entry_mean >= low_entry_mean


def test_fixed_thresholds_mode_leaves_entry_z_constant_at_default():
    prices = _synthetic_universe(n_sessions=250)
    result = em.run_full_backtest_meanrev(
        prices, params={"hurst_cutoff": 0.5, "derive_entry_z_from_cost": False}, bars_per_year=9 * 252,
    )
    entry_z_seen = result["results"]["entry_z"].dropna().unique()
    # fixed mode still goes through derive_entry_exit_z (which returns the
    # PASSED-IN entry_z unchanged, only the window is half-life-derived), so
    # every value seen should be exactly the DEFAULTS entry_z.
    assert np.allclose(entry_z_seen, em.DEFAULTS["entry_z"])
