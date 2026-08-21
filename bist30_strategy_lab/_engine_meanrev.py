"""
OU-half-life pairs mean reversion, screened across the FULL BIST30 universe
(all C(30,2)=435 pairs), not an ad hoc subset.

Upgrades over both prior missions' pairs-trading approach in one respect:
entry/exit z-score windows are DERIVED per pair from that pair's own fitted
OU half-life (_common.ou_half_life / derive_entry_exit_z) instead of a
single fixed constant (examples/06's LOOKBACK=252 / z_window, intraday's
z_window=60) borrowed across every pair regardless of how fast or slowly it
actually reverts.

Position sizing uses a REALIZED-volatility target (rolling std of the
spread's own bar-to-bar change), not a full GARCH refit per pair -- with up
to 435 candidate pairs to screen and dozens of rebalances each in a full-
universe walk-forward, refitting arch_model that many times is needless
cost for a signal that ISN'T what drives entries/exits (GARCH forecasts
volatility, not direction). The same no-leverage bound (_common.SCALE_MIN/
SCALE_MAX) still applies -- only the volatility ESTIMATE feeding the scale
is simpler.
"""

import numpy as np
import pandas as pd
from itertools import combinations

import _common as c
import _ou_calibration as ou_calib

DEFAULTS = dict(
    in_sample_fraction=0.6,
    session_lookback=20,     # trailing sessions used to refit hedge ratio / regime checks
    rebalance_sessions=5,    # re-validate + re-derive z_window every N sessions
    adf_crit=-2.86,
    hurst_cutoff=0.45,
    entry_z=1.5,
    stop_z=3.5,
    min_window=10,
    max_window=500,
    vol_window=30,           # bars, for the realized-vol sizing scale
    # Cost-calibrated entry_z (see _ou_calibration.py): off by default so the
    # fixed 1.5/3.5 thresholds remain the baseline every existing test/result
    # was built against. When on, entry_z/stop_z are re-derived at every
    # rebalance from THIS pair's fitted (kappa, sigma) via Monte Carlo
    # simulation against an assumed round-trip cost, instead of using the
    # fixed entry_z/stop_z above.
    derive_entry_z_from_cost=False,
    cost_per_round_trip_bps=5,   # assumed round-trip cost, in bps of log-spread units -- same convention as cost_sensitivity_table
)


def engle_granger_hedge_ratio(log_a, log_b):
    X = np.column_stack([np.ones_like(log_b), log_b])
    (alpha, beta), *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return alpha, beta


def screen_pairs_full_universe(log_prices, sess_ids, tickers, bars_per_day,
                                adf_crit=-2.86, hurst_cutoff=0.45, min_bars=200):
    """
    JOB 1: screen every pair in the universe (up to 435 for the full
    BIST30 list). Each pair is restricted to its own jointly-valid (non-
    NaN) date range first -- see universe.download_universe's docstring on
    why short-history tickers like DSTKF.IS/TRALT.IS are left with NaN
    cells rather than truncating the whole panel.
    """
    rows = []
    for a, b in combinations(tickers, 2):
        pair_df = log_prices[[a, b]].dropna()
        if len(pair_df) < min_bars:
            continue  # not enough joint history for this pair -- skip, don't force-fit
        la, lb = pair_df[a].to_numpy(), pair_df[b].to_numpy()
        pair_sess = c.session_ids(pair_df.index)
        alpha, beta = engle_granger_hedge_ratio(la, lb)
        spread = la - beta * lb - alpha
        df_tau, gamma_hat = c.adf_session_aware(spread, pair_sess)
        H = c.hurst_on_levels(spread)
        hl_bars, hl_days = c.ou_half_life(gamma_hat, bars_per_day=bars_per_day)
        rows.append({
            "pair": f"{a}/{b}", "ticker_a": a, "ticker_b": b, "beta": beta,
            "adf_stat": df_tau, "H": H, "half_life_bars": hl_bars, "half_life_days": hl_days,
            "n_bars": len(pair_df),
            "qualifies": (df_tau < adf_crit) and (H < hurst_cutoff) and np.isfinite(hl_bars),
        })
    return pd.DataFrame(rows).sort_values("adf_stat").reset_index(drop=True)


def derive_entry_exit_z(half_life_bars, min_window, max_window, entry_z, stop_z):
    return c.derive_entry_exit_z(half_life_bars, min_window, max_window, entry_z, stop_z)


def walk_forward_meanrev(log_a, log_b, sess_ids, oos_start_idx, params):
    """
    Per-bar walk-forward loop for one already-selected pair. Every
    `rebalance_sessions` sessions: refit the hedge ratio on the trailing
    `session_lookback` sessions, re-check ADF/Hurst regime qualification,
    and re-derive the z-score window from the freshly fitted half-life.
    The z-score itself is a simple rolling window over raw spread LEVELS
    (like examples/06's daily walk-forward, not session-capped like
    intraday's 5-minute version) -- half-life-derived windows here often
    span many sessions, and an overnight/weekend price gap is real
    information the spread level should reflect, not an artifact to hide
    from the trading signal (unlike the ADF/Hurst REGIME check, which does
    use session-aware gap masking, since that regression genuinely is
    biased by being fed a same-index massive overnight diff as if it were
    one ordinary bar-to-bar move).
    """
    n = len(log_a)
    beta_path = np.full(n, np.nan)
    z_path = np.full(n, np.nan)
    qualified_path = np.zeros(n, dtype=bool)
    spread_path = np.full(n, np.nan)
    z_window_path = np.full(n, np.nan)
    half_life_path = np.full(n, np.nan)
    entry_z_path = np.full(n, np.nan)
    stop_z_path = np.full(n, np.nan)

    lookback = params["session_lookback"]
    rebalance = params["rebalance_sessions"]
    adf_crit = params["adf_crit"]
    hurst_cutoff = params["hurst_cutoff"]
    derive_from_cost = params.get("derive_entry_z_from_cost", False)
    cost_per_round_trip = params.get("cost_per_round_trip_bps", 5) / 10000

    beta_s, alpha_s, qualified_s, z_window_s, hl_s = None, None, True, params["max_window"], np.inf
    entry_z_s, stop_z_s = params["entry_z"], params["stop_z"]
    calib_rng = np.random.default_rng(0)
    last_rebalance_session = None

    for t in range(oos_start_idx, n):
        s = sess_ids[t]
        do_rebalance = (last_rebalance_session is None) or (s - last_rebalance_session >= rebalance)

        if do_rebalance:
            last_rebalance_session = s
            win_mask = (sess_ids >= s - lookback) & (sess_ids < s)
            if win_mask.sum() >= 50:
                win_a, win_b = log_a[win_mask], log_b[win_mask]
                win_sess = sess_ids[win_mask]
                alpha_s, beta_s = engle_granger_hedge_ratio(win_a, win_b)
                spread_hist = win_a - beta_s * win_b - alpha_s
                df_tau, gamma_hat = c.adf_session_aware(spread_hist, win_sess)
                H = c.hurst_on_levels(spread_hist)
                hl_s = c.ou_half_life(gamma_hat)
                qualified_s = (df_tau < adf_crit) and (H < hurst_cutoff + 0.05) and np.isfinite(hl_s)
                z_window_s, entry_z_s, stop_z_s = derive_entry_exit_z(
                    hl_s, params["min_window"], params["max_window"], params["entry_z"], params["stop_z"]
                )

                if derive_from_cost and qualified_s:
                    kappa = -gamma_hat
                    sigma = c.ou_residual_sigma(spread_hist, win_sess)
                    calib = ou_calib.calibrate_entry_z(
                        kappa, sigma, cost_per_round_trip, rng=calib_rng
                    )
                    if calib is not None:
                        entry_z_s, stop_z_s = calib["entry_z"], calib["stop_z"]
            elif beta_s is None:
                qualified_s = False

        if beta_s is not None:
            spread_t = log_a[t] - beta_s * log_b[t] - alpha_s
            window_start = max(0, t - z_window_s)
            bars_available = t - window_start
            if not c.min_history_guard(bars_available, params["min_window"]):
                z_t = 0.0
            else:
                hist_a, hist_b = log_a[window_start:t], log_b[window_start:t]
                spread_window = hist_a - beta_s * hist_b - alpha_s
                mu, sd = spread_window.mean(), spread_window.std()
                z_t = (spread_t - mu) / sd if sd > 0 else 0.0

            beta_path[t] = beta_s
            z_path[t] = z_t
            qualified_path[t] = qualified_s
            spread_path[t] = spread_t
            z_window_path[t] = z_window_s
            half_life_path[t] = hl_s
            entry_z_path[t] = entry_z_s
            stop_z_path[t] = stop_z_s

    return dict(beta_path=beta_path, z_path=z_path, qualified_path=qualified_path,
                spread_path=spread_path, z_window_path=z_window_path, half_life_path=half_life_path,
                entry_z_path=entry_z_path, stop_z_path=stop_z_path)


def run_signal_pnl_and_trades(paths, timestamps, oos_start_idx, n, vol_window):
    """
    Trading loop mirroring both prior missions' shape (entry/exit/hard-stop,
    regime-break flattening, trade log), but position size is scaled by a
    REALIZED-vol target (rolling std of the spread's own diff over
    vol_window bars) instead of a GARCH forecast -- see module docstring.
    No forced end-of-session flatten: half-life-derived windows routinely
    span multiple sessions, so multi-day holds are expected here, same as
    examples/06's daily mission (not intraday's day-trading mission).

    entry_z/stop_z are read PER BAR from paths["entry_z_path"]/["stop_z_path"]
    (constant across a rebalance period in the fixed-threshold mode, but
    varying period-to-period when derive_entry_z_from_cost is on -- see
    walk_forward_meanrev), not passed as one scalar for the whole backtest.
    """
    z_path, qualified_path, spread_path = paths["z_path"], paths["qualified_path"], paths["spread_path"]
    entry_z_path, stop_z_path = paths["entry_z_path"], paths["stop_z_path"]

    pos = 0.0
    position = np.zeros(n)
    n_stops = 0
    n_regime_flat = 0
    n_trades = 0
    trades = []
    entry_t = None
    entry_z_value = None

    def close_trade(exit_t, reason):
        nonlocal entry_t, entry_z_value
        if entry_t is None:
            return
        trades.append({
            "entry_time": timestamps[entry_t], "exit_time": timestamps[exit_t],
            "direction": "long spread" if position[entry_t] > 0 else "short spread",
            "entry_z": entry_z_value, "exit_z": z_path[exit_t],
            "hold_bars": exit_t - entry_t, "exit_reason": reason,
        })
        entry_t, entry_z_value = None, None

    for t in range(oos_start_idx, n):
        if not qualified_path[t]:
            if pos != 0:
                close_trade(t, "regime break")
                pos = 0.0
                n_regime_flat += 1
        else:
            zt = z_path[t]
            if np.isnan(zt):
                continue
            entry_z_t, stop_z_t = entry_z_path[t], stop_z_path[t]
            if pos == 0.0:
                if zt > entry_z_t:
                    pos = -1.0
                    n_trades += 1
                    entry_t, entry_z_value = t, zt
                elif zt < -entry_z_t:
                    pos = 1.0
                    n_trades += 1
                    entry_t, entry_z_value = t, zt
            elif pos == 1.0 and zt >= 0:
                close_trade(t, "reversion")
                pos = 0.0
            elif pos == -1.0 and zt <= 0:
                close_trade(t, "reversion")
                pos = 0.0
            elif abs(zt) > stop_z_t:
                close_trade(t, "hard stop")
                pos = 0.0
                n_stops += 1
        position[t] = pos

    dspread = np.diff(spread_path, prepend=spread_path[0])
    realized_vol = pd.Series(dspread).rolling(vol_window, min_periods=5).std().to_numpy()
    target_vol = np.nanmedian(realized_vol[oos_start_idx:])
    scale = np.clip(target_vol / realized_vol, c.SCALE_MIN, c.SCALE_MAX)
    scale = np.nan_to_num(scale, nan=1.0)

    pnl = np.roll(position, 1) * dspread * scale
    pnl[oos_start_idx] = 0.0
    # scale is returned, not just consumed: it IS the position size this bar
    # (a +-1-unit spread position scaled by the vol target), so it's what
    # turns an abstract unit into a real traded amount downstream -- see
    # _common.base_unit_multiplier.
    return pnl, position, trades, n_stops, n_regime_flat, n_trades, scale


def attach_trade_pnl(trades, pnl, timestamps):
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    for trade in trades:
        i0, i1 = ts_to_idx[trade["entry_time"]], ts_to_idx[trade["exit_time"]]
        trade["pnl"] = float(np.nansum(pnl[i0 + 1:i1 + 1]))
    return trades


def attach_trade_prices(trades, prices, ticker_a, ticker_b):
    for t in trades:
        t["entry_price_a"] = float(prices.loc[t["entry_time"], ticker_a])
        t["entry_price_b"] = float(prices.loc[t["entry_time"], ticker_b])
        t["exit_price_a"] = float(prices.loc[t["exit_time"], ticker_a])
        t["exit_price_b"] = float(prices.loc[t["exit_time"], ticker_b])
    return trades


def screen_only(prices, params, bars_per_day):
    log_prices = np.log(prices)
    full_sess = c.session_ids(prices.index)
    split_idx, _ = c.split_by_session(prices.index, params["in_sample_fraction"])
    insample_log = log_prices.iloc[:split_idx]
    screening = screen_pairs_full_universe(
        insample_log, full_sess[:split_idx], prices.columns, bars_per_day,
        params["adf_crit"], params["hurst_cutoff"],
    )
    return screening, split_idx, full_sess, log_prices


def run_full_backtest_meanrev(prices, params=None, bars_per_year=None, screening_result=None, selected_pair=None):
    """
    One call, all stages: screen full universe -> select pair -> walk-
    forward -> trades. Mirrors intraday_pairs_trading/_engine.py's
    run_full_backtest shape/contract (a `results` DataFrame with a `pnl`
    column, a `trades` list of pnl-keyed dicts) so _common.add_capital_pnl
    and validation.py's generic driver work unmodified across all 3 engines.
    """
    params = {**DEFAULTS, **(params or {})}
    bars_per_day = (bars_per_year / 252) if bars_per_year else 9
    if bars_per_year is None:
        from universe import bars_per_year as _bpy
        bars_per_year = _bpy(prices.index)
        bars_per_day = bars_per_year / 252

    if screening_result is None:
        screening, split_idx, sess, log_prices = screen_only(prices, params, bars_per_day)
    else:
        screening, split_idx, sess, log_prices = screening_result

    qualifying = screening[screening["qualifies"]]
    if selected_pair is not None:
        ta, tb = selected_pair
        match = screening[
            ((screening.ticker_a == ta) & (screening.ticker_b == tb)) |
            ((screening.ticker_a == tb) & (screening.ticker_b == ta))
        ]
        if len(match) == 0:
            raise ValueError(f"pair {ta}/{tb} not found among this universe's screened pairs")
        best = match.iloc[0]
    elif len(qualifying) > 0:
        best = qualifying.iloc[0]
    else:
        best = screening.iloc[0]

    ticker_a, ticker_b = best["ticker_a"], best["ticker_b"]
    pair_log = log_prices[[ticker_a, ticker_b]].dropna()
    pair_sess = c.session_ids(pair_log.index)
    pair_split_idx, _ = c.split_by_session(pair_log.index, params["in_sample_fraction"])
    log_a, log_b = pair_log[ticker_a].to_numpy(), pair_log[ticker_b].to_numpy()

    paths = walk_forward_meanrev(log_a, log_b, pair_sess, pair_split_idx, params)
    pnl, position, trades, n_stops, n_regime_flat, n_trades, scale = run_signal_pnl_and_trades(
        paths, pair_log.index.to_numpy(), pair_split_idx, len(pair_log), params["vol_window"],
    )
    trades = attach_trade_pnl(trades, pnl, pair_log.index.to_numpy())
    trades = attach_trade_prices(trades, prices.reindex(pair_log.index), ticker_a, ticker_b)
    stats = c.performance_stats(pnl, pair_split_idx, bars_per_year)
    cost_table = c.cost_sensitivity_table(pnl, position, pair_split_idx, bars_per_year)

    results = pd.DataFrame({
        "timestamp": pair_log.index, "beta": paths["beta_path"], "z": paths["z_path"],
        "qualified": paths["qualified_path"], "spread": paths["spread_path"],
        "half_life_bars": paths["half_life_path"], "entry_z": paths["entry_z_path"],
        "stop_z": paths["stop_z_path"], "position": position, "scale": scale, "pnl": pnl,
    })

    return dict(
        ticker_a=ticker_a, ticker_b=ticker_b, screening=screening, results=results,
        stats=stats, cost_table=cost_table, trades=trades, split_idx=pair_split_idx,
        n_stops=n_stops, n_regime_flat=n_regime_flat, n_trades=n_trades,
        pct_disqualified=round(float((~paths["qualified_path"][pair_split_idx:]).mean() * 100), 1),
        scale_max=c.SCALE_MAX, bars_per_year=bars_per_year,
    )
