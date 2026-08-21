"""
pytest suite for _common.py. Synthetic data only, no network -- same
convention as intraday_pairs_trading/test_engine.py.

Run: python -m pytest bist30_strategy_lab/test_common.py -v
(from the afistaj/ project root, with the venv active)
"""

import numpy as np
import pandas as pd
import pytest

import _common as c


def _hourly_index(n_sessions, bars_per_session):
    idx = []
    for d in range(n_sessions):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for h in range(bars_per_session):
            idx.append(day + pd.Timedelta(hours=10 + h))
    return pd.DatetimeIndex(idx)


def test_session_ids_counts_sessions_correctly():
    idx = _hourly_index(4, 5)
    sess = c.session_ids(idx)
    assert sess[-1] == 3
    assert sess[0] == 0


def test_session_ids_monotonic_nondecreasing():
    idx = _hourly_index(5, 6)
    sess = c.session_ids(idx)
    assert np.all(np.diff(sess) >= 0)


def test_mask_overnight_gaps_excludes_session_boundary():
    idx = _hourly_index(3, 4)
    sess = c.session_ids(idx)
    diffs = np.diff(np.arange(len(idx)))
    valid = c.mask_overnight_gaps(diffs, sess)
    # bar 4 is the first bar of session 1 -- the diff INTO it (idx 3) must be masked out
    assert valid[3] == False  # noqa: E712
    assert valid[0] == True   # noqa: E712


def test_split_by_session_lands_on_a_session_boundary():
    idx = _hourly_index(10, 5)
    sess = c.session_ids(idx)
    split_idx, sess2 = c.split_by_session(idx, 0.6)
    assert split_idx == 0 or sess2[split_idx] != sess2[split_idx - 1]


def test_hurst_distinguishes_mean_reverting_from_random_walk():
    rng = np.random.default_rng(0)
    n = 3000
    mean_rev = np.zeros(n)
    for t in range(1, n):
        mean_rev[t] = mean_rev[t - 1] + 0.1 * (0 - mean_rev[t - 1]) + rng.normal(0, 1)
    random_walk = np.cumsum(rng.normal(0, 1, n))

    h_mr = c.hurst_on_levels(mean_rev)
    h_rw = c.hurst_on_levels(random_walk)
    assert h_mr < 0.4
    assert h_rw > 0.4


def test_adf_session_aware_rejects_random_walk_null_for_mean_reverting_spread():
    rng = np.random.default_rng(1)
    n = 3000
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] + 0.1 * (0 - spread[t - 1]) + rng.normal(0, 0.5)
    sess = np.zeros(n, dtype=int)  # one continuous session -- no gaps to mask
    df_tau, gamma_hat = c.adf_session_aware(spread, sess)
    assert df_tau < -2.86  # comfortably beats the ~5% critical value
    assert gamma_hat < 0


def test_adf_session_aware_does_not_reject_for_pure_random_walk():
    rng = np.random.default_rng(2)
    n = 3000
    spread = np.cumsum(rng.normal(0, 1, n))
    sess = np.zeros(n, dtype=int)
    df_tau, gamma_hat = c.adf_session_aware(spread, sess)
    assert df_tau > -2.86


def test_ou_half_life_matches_known_kappa_on_synthetic_ou_series():
    rng = np.random.default_rng(3)
    n = 5000
    kappa_true = 0.05
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] + kappa_true * (0 - spread[t - 1]) + rng.normal(0, 1)
    sess = np.zeros(n, dtype=int)
    _, gamma_hat = c.adf_session_aware(spread, sess)
    half_life = c.ou_half_life(gamma_hat)
    expected = np.log(2) / kappa_true
    assert half_life == pytest.approx(expected, rel=0.3)


def test_ou_half_life_returns_inf_for_non_mean_reverting_gamma():
    assert c.ou_half_life(0.0) == np.inf
    assert c.ou_half_life(0.01) == np.inf  # positive gamma -> explosive, not reverting


def test_ou_half_life_with_bars_per_day_returns_both_units():
    half_life_bars, half_life_days = c.ou_half_life(-0.1, bars_per_day=9)
    assert half_life_days == pytest.approx(half_life_bars / 9)


def test_derive_entry_exit_z_scales_with_half_life():
    z_short, _, _ = c.derive_entry_exit_z(10)
    z_long, _, _ = c.derive_entry_exit_z(100)
    assert z_long > z_short


def test_derive_entry_exit_z_clips_to_bounds():
    z_window, _, _ = c.derive_entry_exit_z(np.inf, min_window=10, max_window=500)
    assert z_window == 500
    z_window, _, _ = c.derive_entry_exit_z(0.001, min_window=10, max_window=500)
    assert z_window == 10


def test_min_history_guard():
    assert c.min_history_guard(20, 10) is True
    assert c.min_history_guard(5, 10) is False


def test_dollar_per_unit_never_implies_leverage():
    dpu = c.dollar_per_unit_for_trade(100_000, scale_max=3.0)
    # worst case: position=1, scale=scale_max -> notional = scale_max * dpu
    worst_case_notional = 3.0 * dpu
    assert worst_case_notional == pytest.approx(100_000)
    assert worst_case_notional <= 100_000 + 1e-6


def test_add_capital_pnl_equity_curve_consistent_with_final_capital():
    pnl = np.array([0.0, 1.0, -0.5, 2.0])
    result = {
        "results": pd.DataFrame({"pnl": pnl}),
        "trades": [{"pnl": 1.0}, {"pnl": -0.5}],
    }
    out = c.add_capital_pnl(result, starting_capital=10_000)
    assert out["equity_curve"][-1] == pytest.approx(out["final_capital"])
    assert out["equity_curve"][0] == pytest.approx(10_000 + pnl[0] * out["dollar_per_unit"])
    assert all("pnl_currency" in t for t in out["trades"])


def test_performance_stats_zero_pnl_gives_zero_return():
    pnl = np.zeros(100)
    stats = c.performance_stats(pnl, 0, bars_per_year=2000)
    assert stats["annualized return"] == 0.0
    assert stats["max drawdown"] == 0.0


def test_cost_sensitivity_is_non_increasing_in_cost():
    rng = np.random.default_rng(4)
    n = 200
    position = np.sign(rng.normal(size=n))
    pnl = rng.normal(0.01, 1.0, n)
    table = c.cost_sensitivity_table(pnl, position, 0, bars_per_year=2000)
    rets = table["annualized return"].to_numpy()
    assert np.all(np.diff(rets) <= 1e-9)


def test_base_unit_multiplier_reads_vol_scale_for_spread_engine():
    results = pd.DataFrame({"scale": [0.2, 1.0, 3.0], "position": [1.0, -1.0, 1.0]})
    assert c.base_unit_multiplier(results, 0) == pytest.approx(0.2)
    assert c.base_unit_multiplier(results, 2) == pytest.approx(3.0)


def test_base_unit_multiplier_splits_the_book_across_basket_names():
    """top_n=5 puts a fifth of the unit into each name, not the whole unit."""
    results = pd.DataFrame({"n_held": [0, 1, 5]})
    assert c.base_unit_multiplier(results, 1) == pytest.approx(1.0)
    assert c.base_unit_multiplier(results, 2) == pytest.approx(0.2)
    assert c.base_unit_multiplier(pd.DataFrame({"n_active": [4]}), 0) == pytest.approx(0.25)


def test_base_unit_multiplier_is_one_when_nothing_is_held_or_recorded():
    assert c.base_unit_multiplier(pd.DataFrame({"n_held": [0]}), 0) == pytest.approx(1.0)
    assert c.base_unit_multiplier(pd.DataFrame({"pnl": [0.0]}), 0) == pytest.approx(1.0)


def test_traded_amount_never_exceeds_capital_for_a_basket_name():
    """Each name's amount is unit/n, so the whole book is one unit -- never levered."""
    capital = 100_000
    unit = c.dollar_per_unit_for_trade(capital, scale_max=1.0)
    for n_held in (1, 3, 5, 30):
        results = pd.DataFrame({"n_held": [n_held]})
        per_name = unit * c.base_unit_multiplier(results, 0)
        assert per_name * n_held == pytest.approx(capital)


def test_spread_unit_amount_stays_within_capital_at_max_scale():
    capital = 100_000
    unit = c.dollar_per_unit_for_trade(capital)
    results = pd.DataFrame({"scale": [c.SCALE_MAX]})
    assert unit * c.base_unit_multiplier(results, 0) == pytest.approx(capital)
