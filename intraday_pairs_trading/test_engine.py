"""
Test suite for _engine.py -- the shared core logic behind 01, 02, 03, and
04_interactive_dashboard.py. No network access needed: everything here
runs on small synthetic price series built in-process, specifically
designed to exercise the session-boundary edge cases this module cares
about (overnight gaps, hedge-ratio changes between sessions, thin
same-session windows).

Run with: python -m pytest intraday_pairs_trading/test_engine.py -v
(from the afistaj/ project root, with the venv active)

This suite exists because two real bugs were found by manual inspection
of backtest output during development (an overnight-gap z-score spike,
and a near-zero-sample-size z-score blowup at session open) -- both would
have been caught immediately by test_walk_forward_zscore_bounded_* below
if it had existed first. That's the point of this file.
"""

import numpy as np
import pandas as pd
import pytest

import _engine as eng


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def make_session_index(n_sessions, bars_per_session, freq_minutes=5):
    """A DatetimeIndex that looks like real intraday data: N business days,
    each with `bars_per_session` bars starting at 07:00, so consecutive
    sessions have a large time gap between them (mimicking the overnight gap)."""
    days = pd.bdate_range("2026-01-05", periods=n_sessions, tz="UTC")
    all_ts = []
    for day in days:
        start = day + pd.Timedelta(hours=7)
        all_ts.extend(pd.date_range(start, periods=bars_per_session, freq=f"{freq_minutes}min"))
    return pd.DatetimeIndex(all_ts)


def make_cointegrated_prices(n_sessions=10, bars_per_session=40, beta_true=0.8,
                              overnight_jump_scale=0.0, seed=0):
    """
    Two tickers, A and B, with A = beta_true * B + mean-reverting noise
    (i.e. genuinely cointegrated by construction). `overnight_jump_scale`
    optionally injects a large deterministic jump into B's log price at
    the start of every session (beyond session 0), to stress-test
    session-boundary handling -- 0.0 means no injected jump.
    """
    rng = np.random.default_rng(seed)
    index = make_session_index(n_sessions, bars_per_session)
    n = len(index)

    log_b = np.empty(n)
    log_b[0] = np.log(100.0)
    ou_noise = np.empty(n)
    ou_noise[0] = 0.0
    theta, sigma = 0.1, 0.002

    sess = eng.session_ids(index)
    for t in range(1, n):
        jump = overnight_jump_scale if sess[t] != sess[t - 1] else 0.0
        log_b[t] = log_b[t - 1] + rng.normal(scale=0.001) + jump
        ou_noise[t] = ou_noise[t - 1] + theta * (0 - ou_noise[t - 1]) + rng.normal(scale=sigma)

    log_a = beta_true * log_b + 0.5 + ou_noise
    prices = pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=index)
    return prices, sess


def make_random_walk_prices(n_sessions=10, bars_per_session=40, seed=1):
    """Two INDEPENDENT random walks -- should not screen as cointegrated."""
    rng = np.random.default_rng(seed)
    index = make_session_index(n_sessions, bars_per_session)
    n = len(index)
    log_a = np.log(100.0) + np.cumsum(rng.normal(scale=0.002, size=n))
    log_b = np.log(50.0) + np.cumsum(rng.normal(scale=0.002, size=n))
    prices = pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=index)
    return prices, eng.session_ids(index)


# ---------------------------------------------------------------------------
# session_ids / mask_overnight_gaps
# ---------------------------------------------------------------------------

def test_session_ids_counts_sessions_correctly():
    index = make_session_index(n_sessions=5, bars_per_session=12)
    sess = eng.session_ids(index)
    assert sess[0] == 0
    assert sess[-1] == 4
    assert len(sess) == 5 * 12
    # each session should have exactly bars_per_session bars
    counts = pd.Series(sess).value_counts().sort_index()
    assert (counts == 12).all()


def test_session_ids_monotonic_nondecreasing():
    index = make_session_index(n_sessions=6, bars_per_session=8)
    sess = eng.session_ids(index)
    assert np.all(np.diff(sess) >= 0)


def test_mask_overnight_gaps_excludes_first_bar_of_each_session():
    index = make_session_index(n_sessions=4, bars_per_session=10)
    sess = eng.session_ids(index)
    dummy = np.arange(len(index), dtype=float)
    diffs = np.diff(dummy)
    valid = eng.mask_overnight_gaps(diffs, sess)

    # diffs[i] corresponds to the transition INTO bar i+1. It should be
    # invalid exactly when bar i+1 is the first bar of a new session.
    session_starts = np.concatenate([[True], sess[1:] != sess[:-1]])
    expected_valid = ~session_starts[1:]
    assert np.array_equal(valid, expected_valid)
    # 4 sessions -> 3 internal boundaries excluded
    assert (~valid).sum() == 3


# ---------------------------------------------------------------------------
# hurst_on_levels / adf_session_aware -- sanity against known regimes
# ---------------------------------------------------------------------------

def test_hurst_distinguishes_mean_reverting_from_random_walk():
    rng = np.random.default_rng(42)
    n = 2000

    # Mean-reverting (OU-like)
    mr = np.empty(n)
    mr[0] = 0.0
    for t in range(1, n):
        mr[t] = mr[t - 1] + 0.05 * (0 - mr[t - 1]) + rng.normal(scale=1.0)
    mr_positive = mr - mr.min() + 10  # hurst_on_levels doesn't need log-positivity, but keep realistic

    # Random walk
    rw = 10 + np.cumsum(rng.normal(scale=1.0, size=n))

    H_mr = eng.hurst_on_levels(mr_positive)
    H_rw = eng.hurst_on_levels(rw)

    assert H_mr < 0.4, f"expected mean-reverting H well below 0.5, got {H_mr}"
    assert 0.35 < H_rw < 0.65, f"expected random-walk H near 0.5, got {H_rw}"
    assert H_mr < H_rw


def test_adf_session_aware_rejects_random_walk_null_for_mean_reverting_spread():
    rng = np.random.default_rng(7)
    n = 1500
    spread = np.empty(n)
    spread[0] = 0.0
    for t in range(1, n):
        spread[t] = spread[t - 1] + 0.08 * (0 - spread[t - 1]) + rng.normal(scale=0.5)
    sess = np.zeros(n, dtype=int)  # single session -- no masking needed here

    df_tau, gamma_hat = eng.adf_session_aware(spread, sess)
    assert gamma_hat < 0, "mean-reverting series should have a negative gamma_hat"
    assert df_tau < eng.DEFAULTS["adf_crit"], "should clearly reject the random-walk null"


def test_adf_session_aware_does_not_reject_for_pure_random_walk():
    rng = np.random.default_rng(7)
    n = 1500
    spread = np.cumsum(rng.normal(scale=0.5, size=n))
    sess = np.zeros(n, dtype=int)

    df_tau, _ = eng.adf_session_aware(spread, sess)
    assert df_tau > eng.DEFAULTS["adf_crit"], "a pure random walk should NOT reject the null"


def test_adf_session_aware_masks_overnight_gap_influence():
    """
    A single huge overnight jump inserted into an otherwise mean-reverting
    spread should NOT be able to flip the ADF conclusion, because
    adf_session_aware excludes that one row from the regression entirely.
    """
    # Both sessions revert to the SAME target (0) -- only the single first
    # bar of session 2 gets a one-off shock baked directly into the
    # recursion (a genuine gap, not a permanent equilibrium shift). Session
    # 2 then spends the rest of its bars decaying back toward 0 just like
    # session 1 always does, so pooling them (minus the one masked bar)
    # should still show strong mean reversion.
    rng = np.random.default_rng(3)
    n = 800
    theta = 0.08
    spread = np.empty(n)
    spread[0] = 0.0
    gap_bar = n // 2
    for t in range(1, n):
        shock = 500.0 if t == gap_bar else 0.0
        spread[t] = spread[t - 1] + shock + theta * (0 - spread[t - 1]) + rng.normal(scale=0.3)

    sess = np.concatenate([np.zeros(gap_bar, dtype=int), np.ones(n - gap_bar, dtype=int)])

    df_tau_masked, _ = eng.adf_session_aware(spread, sess)
    df_tau_naive, _ = eng.adf_session_aware(spread, np.zeros(n, dtype=int))  # no masking (single fake "session")

    # Masking the one contaminated row should give a comfortably significant
    # result; the naive (unmasked) version, corrupted by one huge outlier
    # row, should be noticeably weaker (less negative) than the masked one.
    assert df_tau_masked < eng.DEFAULTS["adf_crit"]
    assert df_tau_masked < df_tau_naive


# ---------------------------------------------------------------------------
# split_by_session
# ---------------------------------------------------------------------------

def test_split_by_session_lands_on_a_session_boundary():
    prices, sess = make_random_walk_prices(n_sessions=10, bars_per_session=20)
    split_idx, sess_out = eng.split_by_session(prices, in_sample_fraction=0.6)
    assert np.array_equal(sess, sess_out)
    # split_idx must be the first bar of some session (or the very end)
    assert split_idx == 0 or sess_out[split_idx] != sess_out[split_idx - 1]


def test_split_by_session_fraction_roughly_respected():
    prices, sess = make_random_walk_prices(n_sessions=10, bars_per_session=20)
    split_idx, _ = eng.split_by_session(prices, in_sample_fraction=0.6)
    n_sessions = sess[-1] + 1
    session_at_split = sess[split_idx]
    assert abs(session_at_split - round(n_sessions * 0.6)) <= 1


# ---------------------------------------------------------------------------
# screen_pairs
# ---------------------------------------------------------------------------

def test_screen_pairs_flags_known_cointegrated_pair():
    prices, sess = make_cointegrated_prices(n_sessions=15, bars_per_session=50, beta_true=0.8)
    log_prices = np.log(prices)
    screening = eng.screen_pairs(log_prices, sess, prices.columns,
                                  adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"])
    row = screening.iloc[0]
    assert row["pair"] == "A/B"
    assert row["qualifies"]
    assert abs(row["beta"] - 0.8) < 0.1


def test_screen_pairs_does_not_flag_independent_random_walks():
    prices, sess = make_random_walk_prices(n_sessions=15, bars_per_session=50)
    log_prices = np.log(prices)
    screening = eng.screen_pairs(log_prices, sess, prices.columns,
                                  adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"])
    assert not screening.iloc[0]["qualifies"]


# ---------------------------------------------------------------------------
# walk_forward_intraday -- the two bugs found during manual testing
# ---------------------------------------------------------------------------

def test_walk_forward_hedge_ratio_constant_within_a_session():
    prices, sess = make_cointegrated_prices(n_sessions=12, bars_per_session=40, beta_true=0.8)
    log_prices = np.log(prices)
    log_a, log_b = log_prices["A"].to_numpy(), log_prices["B"].to_numpy()
    split_idx = int(np.searchsorted(sess, 6))  # first half in-sample

    beta_path, *_ = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        session_lookback=3, z_window=10,
        adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"],
    )
    oos_beta = pd.Series(beta_path[split_idx:])
    oos_sess = pd.Series(sess[split_idx:])
    per_session_unique_counts = oos_beta.groupby(oos_sess).nunique()
    assert (per_session_unique_counts == 1).all(), "beta must be constant within each session"


def test_walk_forward_disqualifies_session_with_negative_beta():
    """
    REGRESSION TEST for a third bug found on real data: with
    session_lookback=5, the daily hedge-ratio re-estimation occasionally
    produced a NEGATIVE beta for GARAN.IS/ISCTR.IS (two same-sector BIST
    banks) on 2026-08-10 -- economically nonsensical (implies "when one
    rises, the other should fall"), a symptom of too little data for a
    stable regression, not a real relationship. Once beta goes negative,
    the "spread" stops representing genuine relative value, yet the
    strategy traded it anyway and generated a real-money-looking P&L that
    didn't reconcile against the two stocks' actual price moves.

    This constructs a window where ticker_b is forced to move OPPOSITE
    ticker_a (guaranteeing a negative OLS beta for that specific lookback)
    and confirms the affected session is disqualified -- BETA_SANITY_BOUNDS
    is the fix.
    """
    n_sessions, bars_per_session = 10, 30
    index = make_session_index(n_sessions, bars_per_session)
    sess = eng.session_ids(index)
    n = len(index)
    rng = np.random.default_rng(5)

    log_a = np.log(100.0) + np.cumsum(rng.normal(scale=0.001, size=n))
    noise_b = rng.normal(scale=0.0005, size=n)  # independent noise so no regression here is ever perfectly singular
    log_b = 0.8 * log_a + noise_b  # positively related baseline everywhere...

    # ...except sessions 2-4, the exact lookback window session 5 will use
    # (session_lookback=3 -> session 5 looks at sessions >= 2 and < 5).
    # Force log_b to move as the (noisy) mirror image of log_a there,
    # guaranteeing a negative OLS slope for that window.
    invert_mask = (sess >= 2) & (sess < 5)
    log_b[invert_mask] = np.log(50.0) - (log_a[invert_mask] - log_a[invert_mask][0]) + noise_b[invert_mask]

    split_idx = int(np.searchsorted(sess, 5))
    beta_path, _, qualified_path, *_ = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        session_lookback=3, z_window=10,
        adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"],
    )
    session5_mask = sess[split_idx:] == 5
    beta_session5 = beta_path[split_idx:][session5_mask][0]
    assert beta_session5 < 0, f"expected the constructed scenario to produce negative beta, got {beta_session5}"
    assert not qualified_path[split_idx:][session5_mask].any(), \
        "a session whose re-estimated beta is negative must be disqualified, not traded"


def test_walk_forward_zscore_bounded_despite_large_overnight_jump():
    """
    REGRESSION TEST for the first bug found manually: a large overnight
    price jump combined with a session-to-session hedge-ratio change used
    to produce z-scores in the hundreds at session open. With the
    session-boundary-aware rolling window, |z| should stay in a sane range.
    """
    prices, sess = make_cointegrated_prices(
        n_sessions=12, bars_per_session=40, beta_true=0.8, overnight_jump_scale=0.05,
    )
    log_prices = np.log(prices)
    log_a, log_b = log_prices["A"].to_numpy(), log_prices["B"].to_numpy()
    split_idx = int(np.searchsorted(sess, 6))

    _, z_path, *_ = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        session_lookback=3, z_window=15,
        adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"],
    )
    oos_z = z_path[split_idx:]
    assert np.nanmax(np.abs(oos_z)) < 20, f"z-score should never blow up; max |z| = {np.nanmax(np.abs(oos_z))}"


def test_walk_forward_zscore_zero_during_min_window_warmup():
    """
    REGRESSION TEST for the second bug: with fewer than MIN_WINDOW_BARS
    same-session observations, the z-score must be exactly 0 (no signal),
    not a wild value from a near-zero standard deviation estimate.
    """
    prices, sess = make_cointegrated_prices(n_sessions=8, bars_per_session=30, beta_true=0.8)
    log_prices = np.log(prices)
    log_a, log_b = log_prices["A"].to_numpy(), log_prices["B"].to_numpy()
    split_idx = int(np.searchsorted(sess, 4))

    _, z_path, *_ = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        session_lookback=2, z_window=15,
        adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"],
    )
    # For every out-of-sample session, the first MIN_WINDOW_BARS bars should have z == 0
    oos_sess = sess[split_idx:]
    oos_z = z_path[split_idx:]
    for s in np.unique(oos_sess):
        bars_of_session = np.where(oos_sess == s)[0]
        warmup = bars_of_session[:eng.MIN_WINDOW_BARS]
        assert np.all(oos_z[warmup] == 0.0), f"session {s}: expected z==0 during warmup"


def test_walk_forward_qualified_defaults_false_before_oos_start():
    prices, sess = make_cointegrated_prices(n_sessions=8, bars_per_session=20, beta_true=0.8)
    log_prices = np.log(prices)
    log_a, log_b = log_prices["A"].to_numpy(), log_prices["B"].to_numpy()
    split_idx = int(np.searchsorted(sess, 4))

    _, _, qualified_path, *_ = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        session_lookback=2, z_window=10,
        adf_crit=eng.DEFAULTS["adf_crit"], hurst_cutoff=eng.DEFAULTS["hurst_cutoff"],
    )
    assert not qualified_path[:split_idx].any(), "pre-OOS bars are unfilled placeholders, must read as False"


# ---------------------------------------------------------------------------
# run_signal_pnl_and_trades -- entry/exit/stop/EOD logic
# ---------------------------------------------------------------------------

def _make_flat_backtest_inputs(n, oos_start, session_len):
    sess = np.arange(n) // session_len
    is_last_bar = np.zeros(n, dtype=bool)
    for s in np.unique(sess):
        idx = np.where(sess == s)[0]
        is_last_bar[idx[-1]] = True
    timestamps = np.arange(n)  # plain ints as stand-in timestamps, used only as dict keys
    qualified = np.ones(n, dtype=bool)
    sigma = np.ones(n)
    return sess, is_last_bar, timestamps, qualified, sigma


def test_entry_and_reversion_exit():
    n, session_len = 20, 20
    sess, is_last_bar, timestamps, qualified, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    z = np.zeros(n)
    z[5:10] = 2.0  # entry at 5 (short spread, z above entry_z), sustained so it doesn't self-revert
    z[10] = -0.1   # crosses back through zero -> exit
    spread = np.cumsum(np.full(n, 0.01))  # arbitrary monotonic spread for pnl computation

    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, spread, is_last_bar, timestamps, oos_start_idx=0, n=n,
        entry_z=1.5, stop_z=3.5,
    )
    assert n_trades == 1
    assert len(trades) == 1
    trade = trades[0]
    assert trade["direction"] == "short spread"
    assert trade["entry_time"] == 5
    assert trade["exit_time"] == 10
    assert trade["exit_reason"] == "reversion"


def test_hard_stop_triggers_and_counts():
    n, session_len = 20, 20
    sess, is_last_bar, timestamps, qualified, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    z = np.zeros(n)
    z[3:7] = -2.0  # entry at 3 (long spread), sustained so it doesn't self-revert
    z[7] = -4.0    # beyond stop_z=3.5 -> hard stop
    spread = np.cumsum(np.full(n, 0.01))

    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, spread, is_last_bar, timestamps, oos_start_idx=0, n=n,
        entry_z=1.5, stop_z=3.5,
    )
    assert n_stops == 1
    assert trades[0]["exit_reason"] == "hard stop"
    assert trades[0]["exit_time"] == 7


def test_forced_end_of_day_flatten_when_no_exit_signal():
    n, session_len = 10, 10
    sess, is_last_bar, timestamps, qualified, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    z = np.zeros(n)
    z[2:] = 2.0  # entry at 2, sustained (never reverts or stops out) through session end
    spread = np.cumsum(np.full(n, 0.01))

    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, spread, is_last_bar, timestamps, oos_start_idx=0, n=n,
        entry_z=1.5, stop_z=3.5,
    )
    assert n_eod_flat == 1
    assert trades[0]["exit_reason"] == "end of day"
    assert trades[0]["exit_time"] == n - 1
    assert position[-1] == 0.0, "must never carry a position past the last bar of a session"


def test_no_position_ever_survives_past_its_session():
    """Broader invariant, checked across many sessions with random z paths."""
    rng = np.random.default_rng(11)
    n_sessions, session_len = 6, 15
    n = n_sessions * session_len
    sess, is_last_bar, timestamps, qualified, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    z = rng.normal(scale=2.0, size=n)
    spread = np.cumsum(rng.normal(scale=0.01, size=n))

    _, position, trades, *_ = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, spread, is_last_bar, timestamps, oos_start_idx=0, n=n,
        entry_z=1.5, stop_z=3.5,
    )
    for s in range(n_sessions):
        last_bar_of_s = (s + 1) * session_len - 1
        assert position[last_bar_of_s] == 0.0, f"position leaked past end of session {s}"


def test_regime_disqualification_forces_flat_and_blocks_new_entries():
    n, session_len = 20, 20
    sess, is_last_bar, timestamps, _, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    qualified = np.ones(n, dtype=bool)
    qualified[5:] = False  # disqualified from bar 5 onward
    z = np.zeros(n)
    z[2:5] = 2.0  # entry at 2 (sustained so it doesn't self-revert), still qualified through bar 4
    z[10] = -2.0  # would be a new entry signal, but disqualified -- must be ignored

    _, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, np.cumsum(np.full(n, 0.01)), is_last_bar, timestamps,
        oos_start_idx=0, n=n, entry_z=1.5, stop_z=3.5,
    )
    assert n_regime_flat == 1
    assert trades[0]["exit_reason"] == "regime break"
    assert trades[0]["exit_time"] == 5
    assert n_trades == 1, "the disqualified period's z=-2.0 must NOT open a new trade"
    assert np.all(position[5:] == 0.0)


# ---------------------------------------------------------------------------
# attach_trade_pnl / performance_stats / cost_sensitivity_table
# ---------------------------------------------------------------------------

def test_attach_trade_pnl_matches_manual_sum():
    n, session_len = 20, 20
    sess, is_last_bar, timestamps, qualified, sigma = _make_flat_backtest_inputs(n, 0, session_len)
    z = np.zeros(n)
    z[5] = 2.0
    z[10] = -0.1
    spread = np.cumsum(np.full(n, 0.01))

    pnl, position, trades, *_ = eng.run_signal_pnl_and_trades(
        z, sigma, qualified, spread, is_last_bar, timestamps, oos_start_idx=0, n=n,
        entry_z=1.5, stop_z=3.5,
    )
    trades = eng.attach_trade_pnl(trades, pnl, timestamps)
    i0, i1 = 5, 10
    expected = float(np.nansum(pnl[i0 + 1:i1 + 1]))
    assert abs(trades[0]["pnl"] - expected) < 1e-12


def test_performance_stats_zero_pnl_gives_zero_return():
    pnl = np.zeros(100)
    stats = eng.performance_stats(pnl, oos_start_idx=0)
    assert stats["annualized return"] == 0.0
    assert stats["max drawdown"] == 0.0


def test_cost_sensitivity_is_non_increasing_in_cost():
    rng = np.random.default_rng(5)
    n = 200
    position = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    spread = np.cumsum(rng.normal(scale=0.01, size=n))
    dspread = np.diff(spread, prepend=spread[0])
    pnl = np.roll(position, 1) * dspread
    pnl[0] = 0.0

    table = eng.cost_sensitivity_table(pnl, position, oos_start_idx=0, cost_bps_grid=(0, 5, 10, 20, 50))
    returns = table["annualized return"].to_numpy()
    assert np.all(np.diff(returns) <= 1e-9), "higher cost must never IMPROVE annualized return"


# ---------------------------------------------------------------------------
# dollar_per_unit_for_trade / add_capital_pnl
#
# REGRESSION CONTEXT: two earlier designs were rejected. (1) A global
# factor calibrated to single-bar GARCH volatility oversized multi-bar
# trades ~5x. (2) Sizing to each trade's distance-to-stop implied
# leverage up to 900x capital. The user wants no manual dial AND no
# leverage: a fixed base unit (capital/SCALE_MAX) that the strategy's
# own existing GARCH scale (already in `pnl`) then adjusts bar by bar,
# capped so exposure can never exceed starting_capital.
# ---------------------------------------------------------------------------

def test_dollar_per_unit_for_trade_scales_with_capital_and_never_exceeds_it():
    dpu_1 = eng.dollar_per_unit_for_trade(starting_capital=100_000)
    dpu_2 = eng.dollar_per_unit_for_trade(starting_capital=200_000)
    assert dpu_2 == pytest.approx(2 * dpu_1)
    assert dpu_1 == pytest.approx(100_000 / eng.SCALE_MAX)
    assert dpu_1 * eng.SCALE_MAX <= 100_000 + 1e-9  # max possible exposure never exceeds capital


def test_add_capital_pnl_equity_curve_consistent_with_final_capital():
    prices, _ = make_cointegrated_prices(n_sessions=15, bars_per_session=30, beta_true=0.8, seed=17)
    params = dict(eng.DEFAULTS)
    params.update(session_lookback=3, z_window=10)
    result = eng.run_full_backtest(prices, params)

    starting_capital = 100_000
    result = eng.add_capital_pnl(result, starting_capital=starting_capital)

    assert result["equity_curve"][-1] == pytest.approx(result["final_capital"])
    expected_return_pct = (result["final_capital"] / starting_capital - 1) * 100
    assert result["total_return_pct"] == pytest.approx(expected_return_pct)
    for t in result["trades"]:
        assert t["dollar_per_unit"] == pytest.approx(result["dollar_per_unit"])
        assert t["pnl_currency"] == pytest.approx(t["pnl"] * t["dollar_per_unit"])


# ---------------------------------------------------------------------------
# Integration: run_full_backtest end to end on synthetic data (no network)
# ---------------------------------------------------------------------------

def test_run_full_backtest_end_to_end_synthetic():
    prices, _ = make_cointegrated_prices(n_sessions=20, bars_per_session=40, beta_true=0.8, seed=99)
    params = dict(eng.DEFAULTS)
    params.update(session_lookback=3, z_window=15)

    result = eng.run_full_backtest(prices, params)

    assert result["ticker_a"] in ("A", "B")
    assert result["ticker_b"] in ("A", "B")
    assert isinstance(result["stats"]["sharpe (naive)"], float) or np.isnan(result["stats"]["sharpe (naive)"])
    assert len(result["cost_table"]) == 6  # default cost_bps_grid has 6 entries

    results_df = result["results"]
    split_idx = result["split_idx"]
    # No trade in the log should straddle a session boundary
    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        entry_sess = results_df.set_index("timestamp").loc[trades_df["entry_time"], "session"].to_numpy()
        exit_sess = results_df.set_index("timestamp").loc[trades_df["exit_time"], "session"].to_numpy()
        assert np.array_equal(entry_sess, exit_sess), "a trade must never span two sessions"

    # Position must be flat at the very end of the dataset (last bar is always a session end)
    assert results_df["position"].to_numpy()[-1] == 0.0


def test_run_full_backtest_is_deterministic():
    """Same inputs, same params -> identical results (no hidden randomness / global RNG state leakage)."""
    prices, _ = make_cointegrated_prices(n_sessions=15, bars_per_session=30, beta_true=0.8, seed=21)
    params = dict(eng.DEFAULTS)
    params.update(session_lookback=3, z_window=10)

    r1 = eng.run_full_backtest(prices, params)
    r2 = eng.run_full_backtest(prices, params)

    assert r1["stats"] == r2["stats"]
    pd.testing.assert_frame_equal(r1["results"], r2["results"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
