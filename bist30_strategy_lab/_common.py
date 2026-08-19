"""
Shared math and sizing utilities for this mission's three strategy engines
(_engine_meanrev.py, _engine_momentum.py, _engine_pca.py).

This is mission-LOCAL sharing only -- the same pattern intraday_pairs_
trading/_engine_vol.py uses when it does `import _engine as eng`. It is not
imported by, and does not import from, examples/ or intraday_pairs_trading/;
those are separate missions per the "don't mix them" rule from earlier in
this project.

Even at hourly resolution, BIST30 prices have real session boundaries (the
market is closed overnight and on weekends), so the session-aware ADF/gap-
masking machinery from intraday_pairs_trading/_engine.py is still needed
here -- a naive rolling window would otherwise treat every Monday-morning
open (or every day's open) as if it were a real intra-session move.
"""

import numpy as np
import pandas as pd

# Bounds on any GARCH/vol-targeting-style scale factor used by an engine.
# dollar_per_unit_for_trade uses SCALE_MAX as its own ceiling so that even
# the most aggressive (calmest-market) sizing never exceeds capital -- same
# no-leverage guarantee as intraday_pairs_trading/_engine.py.
SCALE_MIN, SCALE_MAX = 0.2, 3.0


def session_ids(index):
    """Integer session id per calendar day (same definition across all three engines)."""
    dates = index.date
    is_new = np.concatenate([[True], dates[1:] != dates[:-1]])
    return np.cumsum(is_new) - 1


def mask_overnight_gaps(diffs, sess_ids):
    """Boolean mask selecting only same-session bar-to-bar diffs."""
    sess_of_diff = sess_ids[1:]
    sess_of_prev = sess_ids[:-1]
    return sess_of_diff == sess_of_prev


def split_by_session(index, in_sample_fraction):
    """In/out-of-sample split index, landing on a session boundary."""
    sess = session_ids(index)
    n_sessions = sess[-1] + 1
    split_session = int(n_sessions * in_sample_fraction)
    split_idx = int(np.searchsorted(sess, split_session))
    return split_idx, sess


def hurst_on_levels(x, lags=None):
    """Variance-scaling Hurst exponent -- identical calculation used throughout this repo."""
    x = np.asarray(x, dtype=float)
    if lags is None:
        lags = range(2, max(20, len(x) // 10))
    tau_values = np.array(list(lags))
    var_values = np.array([np.var(x[tau:] - x[:-tau]) for tau in tau_values])
    poly = np.polyfit(np.log(tau_values), np.log(var_values), 1)
    return poly[0] / 2.0


def adf_session_aware(spread, sess_ids):
    """
    From-scratch ADF DF_tau statistic (p=1, constant only), skipping
    overnight/weekend-gap rows -- adapted from intraday_pairs_trading/
    _engine.py's adf_session_aware, unchanged in logic (it was already
    generic to any session-boundary definition, not 5-minute-specific).
    Returns (df_tau, gamma_hat); gamma_hat is reused directly by
    ou_half_life below instead of re-fitting the same regression twice.
    """
    y = np.asarray(spread, dtype=float)
    dy = np.diff(y)
    valid = mask_overnight_gaps(dy, sess_ids)
    y_lag = y[:-1][valid]
    dy_valid = dy[valid]

    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta_hat, *_ = np.linalg.lstsq(X, dy_valid, rcond=None)
    residuals = dy_valid - X @ beta_hat
    dof = len(dy_valid) - 2
    sigma2 = (residuals @ residuals) / dof
    se_gamma = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    gamma_hat = beta_hat[1]
    df_tau = gamma_hat / se_gamma
    return df_tau, gamma_hat


def ou_residual_sigma(spread, sess_ids):
    """
    Residual standard deviation of the SAME AR(1) regression adf_session_
    aware fits (Δy_t = alpha + gamma*y_(t-1) + residual) -- the per-bar
    noise scale an OU process's kappa/sigma calibration needs (see
    _ou_calibration.py). Kept as its own function, recomputing the cheap
    2-column OLS, rather than changing adf_session_aware's return
    signature -- that function is called from several existing sites/tests
    that only expect (df_tau, gamma_hat) back.
    """
    y = np.asarray(spread, dtype=float)
    dy = np.diff(y)
    valid = mask_overnight_gaps(dy, sess_ids)
    y_lag = y[:-1][valid]
    dy_valid = dy[valid]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta_hat, *_ = np.linalg.lstsq(X, dy_valid, rcond=None)
    residuals = dy_valid - X @ beta_hat
    return float(np.std(residuals))


def ou_half_life(gamma_hat, bars_per_day=None):
    """
    Ornstein-Uhlenbeck half-life derived from the SAME AR(1) regression
    adf_session_aware already fits: discretizing dy = kappa*(theta-y)dt +
    sigma*dW as Δy_t = alpha + gamma*y_(t-1) + noise gives gamma = -kappa
    (one-bar dt), so kappa = -gamma_hat and half_life = ln(2)/kappa, in bars.

    This replaces the ad hoc fixed z-score thresholds (ENTRY_Z=1.5,
    STOP_Z=3.5) both prior missions used with a principled, per-pair/per-
    residual-derived value: how long a real deviation from the mean is
    expected to take to halve, given THIS spread's own estimated mean-
    reversion speed -- not a constant borrowed from a different pair.

    Returns np.inf (no finite half-life -- not mean-reverting) if gamma_hat
    is >= 0. If bars_per_day is given, also returns the half-life in
    calendar days for readability (dashboard/WORKFLOW.md use).
    """
    kappa = -gamma_hat
    if kappa <= 0:
        return (np.inf, np.inf) if bars_per_day else np.inf
    half_life_bars = np.log(2) / kappa
    if bars_per_day:
        return half_life_bars, half_life_bars / bars_per_day
    return half_life_bars


def derive_entry_exit_z(half_life_bars, min_window=10, max_window=500,
                         entry_z=1.5, stop_z=3.5):
    """
    Turns a fitted half-life into a z-score rolling window: a spread that
    reverts quickly should be judged against a short recent window, one
    that reverts slowly needs a longer window to get a stable mean/std
    estimate. entry_z/stop_z keep the same interpretable meaning (in
    standard deviations) as both prior missions used -- only the WINDOW
    that z is computed over is now principled instead of a fixed constant.
    """
    if not np.isfinite(half_life_bars):
        z_window = max_window
    else:
        z_window = int(np.clip(round(3 * half_life_bars), min_window, max_window))
    return z_window, entry_z, stop_z


def min_history_guard(n_bars, min_bars):
    """
    True if there's enough trailing history to trust a fit on this window,
    False otherwise. Used by every engine's walk-forward loop so a short-
    history ticker (e.g. universe.DSTKF.IS) is automatically skipped/NaN'd
    for any window it doesn't yet have enough bars for, rather than needing
    a hardcoded per-ticker exclusion list.
    """
    return n_bars >= min_bars


def dollar_per_unit_for_trade(starting_capital, scale_max=SCALE_MAX):
    """
    No manual risk-% dial -- same no-leverage guarantee and reasoning as
    intraday_pairs_trading/_engine.py's dollar_per_unit_for_trade: an
    engine's own situational vol-targeting scale (bounded by scale_max)
    multiplies this unit, so worst case notional = 1 * scale_max *
    (capital/scale_max) = capital. Never leveraged, no dial to tune.
    """
    return starting_capital / scale_max


def add_capital_pnl(result, starting_capital):
    """
    Adds currency-denominated fields to a run_full_backtest_*() result:
    equity curve, final capital, total return %, each trade's PnL in
    currency terms. Generic across all three engines in this mission as
    long as `result` has a `results` DataFrame with a `pnl` column and a
    `trades` list of dicts with a `pnl` key -- the shared contract every
    engine's run_full_backtest_* is written to honor.
    """
    pnl = result["results"]["pnl"].to_numpy()
    dpu = dollar_per_unit_for_trade(starting_capital, result.get("scale_max", SCALE_MAX))

    for t in result["trades"]:
        t["dollar_per_unit"] = dpu
        t["starting_capital"] = starting_capital
        t["pnl_currency"] = t["pnl"] * dpu

    dollar_pnl = np.nan_to_num(pnl) * dpu
    equity = starting_capital + np.cumsum(dollar_pnl)
    final_capital = float(equity[-1]) if len(equity) else float(starting_capital)
    total_return_pct = (final_capital / starting_capital - 1) * 100 if starting_capital > 0 else 0.0

    result["dollar_per_unit"] = dpu
    result["equity_curve"] = equity
    result["starting_capital"] = starting_capital
    result["final_capital"] = final_capital
    result["total_return_pct"] = total_return_pct
    return result


def performance_stats(pnl, oos_start_idx, bars_per_year):
    """
    Same shape as both prior missions' performance_stats, but bars_per_year
    is a required argument here (not defaulted/guessed) -- this mission's
    real hourly bar density should be MEASURED from actual downloaded data
    (see universe.bars_per_year), not assumed the way the daily (252) and
    5-minute (95*250) missions did.
    """
    pnl = pnl[oos_start_idx:]
    ann_ret = np.nanmean(pnl) * bars_per_year
    ann_vol = np.nanstd(pnl) * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = np.nancumsum(pnl)
    max_drawdown = np.max(np.maximum.accumulate(cum) - cum) if len(cum) else 0.0
    return {
        "annualized return": round(float(ann_ret), 4),
        "annualized vol": round(float(ann_vol), 4),
        "sharpe (naive)": round(float(sharpe), 3) if np.isfinite(sharpe) else float("nan"),
        "max drawdown": round(float(max_drawdown), 4),
    }


def cost_sensitivity_table(pnl, position, oos_start_idx, bars_per_year,
                            cost_bps_grid=(0, 1, 2, 5, 10, 20)):
    """Sharpe/return as a function of a per-switch transaction cost, same diagnostic as both prior missions."""
    position_prev = np.roll(position, 1)
    position_prev[0] = 0.0
    turnover = np.abs(position - position_prev)

    rows = []
    for cost_bps in cost_bps_grid:
        cost = turnover * (cost_bps / 10000)
        pnl_after_cost = pnl - cost
        pnl_after_cost[oos_start_idx] = 0.0
        stats = performance_stats(pnl_after_cost, oos_start_idx, bars_per_year)
        stats["cost (bps/switch)"] = cost_bps
        rows.append(stats)
    return pd.DataFrame(rows).set_index("cost (bps/switch)")
