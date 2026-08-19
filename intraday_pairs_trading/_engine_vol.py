"""
Volatility mean-reversion engine -- pairs trading and single-asset price
mean reversion were both tried and rejected: individual BIST stock PRICES
are essentially random walks (confirmed: 0/16 stocks pass ADF+Hurst at
5m, 15m, or 60m resolution). Their REALIZED VOLATILITY, however, passes
overwhelmingly (16/16, ADF stats -9 to -14 vs. a -2.86 threshold, Hurst
0.04-0.08) -- volatility clustering-and-reverting is one of the most
robust empirical facts in finance (it's the whole premise GARCH is built
on), unlike price-level mean reversion.

STRATEGY: when a stock's realized volatility spikes well above its own
recent norm (statistically confirmed via the same ADF/Hurst walk-forward
machinery used elsewhere in this project) AND that spike is itself
confirmed to be a genuine mean-reverting regime for this stock, take a
CONTRARIAN position in the stock -- direction set by recent price
momentum (a sharp drop tends to see some bounce; a sharp rise tends to
see some pullback), sized down while the storm is on. Exit once realized
volatility has normalized (the reversion played out); hard-stop if
volatility keeps expanding instead (the thesis failing).

Reuses generic utilities directly from _engine.py (session handling,
ADF/Hurst, GARCH-scale no-leverage capital sizing, download, trade-card
CSS) rather than duplicating them -- see that file for the shared pieces.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import _engine as eng

VOL_WINDOW = 12       # bars of returns used for the realized-vol signal (~1h at 5m bars)
MOMENTUM_BARS = 3     # trailing bars used as a fallback if DI+/DI- can't be computed (e.g. missing OHLC)
DI_PERIOD = 14         # Wilder's standard smoothing period for +DI/-DI
EXIT_RV_Z = 0.5        # realized-vol z-score has to fall back below this to count as "normalized"
RV_Z_CLIP = 20.0       # sanity cap -- see walk_forward_vol: RV's own rolling std can be tiny in calm
                        # periods, so a real but modest RV move can blow the z-score up numerically


def compute_realized_vol(log_price, window=VOL_WINDOW):
    """Rolling std of log-returns -- the volatility SIGNAL this whole strategy trades. NaN for the first `window` bars."""
    ret = np.diff(log_price, prepend=log_price[0])
    ret[0] = np.nan
    rv = pd.Series(ret).rolling(window).std().to_numpy()
    return rv


def download_intraday_ohlc(tickers, interval=eng.BIST_INTERVAL, period=eng.BIST_PERIOD):
    """
    Full OHLC (not just Close) -- needed for +DI/-DI, which is defined on
    the high/low RANGE each bar moved through, not just where it closed.
    Same Europe/Istanbul conversion as _engine.download_intraday_universe.
    """
    import yfinance as yf
    data = yf.download(tickers, interval=interval, period=period, progress=False, auto_adjust=True)
    data = data.dropna()
    data.index = data.index.tz_convert("Europe/Istanbul")
    return data


def compute_directional_indicators(high, low, close, period=DI_PERIOD):
    """
    Wilder's +DI / -DI (the directional-movement half of ADX), computed
    from scratch. Standard definition:
      +DM_t = max(high_t - high_{t-1}, 0) if that exceeds the down-move, else 0
      -DM_t = max(low_{t-1} - low_t, 0) if that exceeds the up-move, else 0
      TR_t  = max(high_t-low_t, |high_t-prev_close|, |low_t-prev_close|)
      +DI_t = 100 * Wilder_smooth(+DM, period) / Wilder_smooth(TR, period)
      -DI_t = 100 * Wilder_smooth(-DM, period) / Wilder_smooth(TR, period)
    +DI > -DI means upward directional movement has been dominant over
    the window; -DI > +DI means downward movement has dominated. Used
    here as a more standard, smoothed replacement for a raw N-bar price
    difference when deciding which way to take the contrarian trade on a
    volatility spike.
    """
    n = len(close)
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = minus_dm[0] = 0.0

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]

    # Wilder's smoothing: an EWM with alpha = 1/period (not a plain SMA).
    smooth_plus_dm = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    smooth_minus_dm = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    smooth_tr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(smooth_tr > 0, 100 * smooth_plus_dm / smooth_tr, 0.0)
        minus_di = np.where(smooth_tr > 0, 100 * smooth_minus_dm / smooth_tr, 0.0)

    # First `period` bars are an unreliable warm-up for the EWM smoothing.
    plus_di[:period] = np.nan
    minus_di[:period] = np.nan
    return plus_di, minus_di


def screen_vol_mean_reversion(log_prices, sess_ids, tickers, adf_crit, hurst_cutoff, vol_window=VOL_WINDOW):
    """
    JOB 1 for this strategy: for each ticker, test whether ITS OWN
    realized volatility (not price) is mean-reverting, in-sample only.
    """
    rows = []
    for ticker in tickers:
        log_price = log_prices[ticker].to_numpy()
        rv = compute_realized_vol(log_price, vol_window)
        valid = ~np.isnan(rv)
        rv_valid, sess_valid = rv[valid], sess_ids[valid]
        df_tau, _ = eng.adf_session_aware(rv_valid, sess_valid)
        H = eng.hurst_on_levels(rv_valid)
        rows.append({
            "ticker": ticker, "adf_stat": df_tau, "H": H,
            "qualifies": (df_tau < adf_crit) and (H < hurst_cutoff),
        })
    return pd.DataFrame(rows).sort_values("adf_stat").reset_index(drop=True)


def walk_forward_vol(log_price, sess_ids, oos_start_idx, session_lookback, z_window,
                      adf_crit, hurst_cutoff, vol_window=VOL_WINDOW, max_bars_per_session=130,
                      high=None, low=None, di_period=DI_PERIOD):
    """
    Session-based walk-forward, same architecture as the pairs engine's
    walk_forward_intraday, but the thing being screened/z-scored is the
    REALIZED VOLATILITY series, and the GARCH fit (for position-sizing
    only, unrelated to the entry signal) runs on the stock's own RETURNS.

    high/low (optional): if given, +DI/-DI are computed once up front
    (a standard indicator, not something that needs session-by-session
    recalibration the way the regime check or GARCH fit do) and returned
    for use as the entry-direction signal in run_signal_pnl_and_trades_vol.
    """
    n = len(log_price)
    rv = compute_realized_vol(log_price, vol_window)
    close = np.exp(log_price)
    if high is not None and low is not None:
        plus_di, minus_di = compute_directional_indicators(high, low, close, di_period)
    else:
        plus_di = minus_di = np.full(n, np.nan)

    rv_z_path = np.full(n, np.nan)
    qualified_path = np.zeros(n, dtype=bool)
    sigma_path = np.full(n, np.nan)
    is_last_bar_of_session = np.zeros(n, dtype=bool)

    qualified_s = True
    sigma_forecast_s = None
    regime_hurst_cutoff = hurst_cutoff + 0.05
    current_session_start_idx = oos_start_idx

    for t in range(oos_start_idx, n):
        s = sess_ids[t]
        session_start_bar = t == 0 or sess_ids[t - 1] != s
        session_end_bar = (t == n - 1) or (sess_ids[t + 1] != s)
        is_last_bar_of_session[t] = session_end_bar

        if session_start_bar:
            current_session_start_idx = t
            win_mask = (sess_ids >= s - session_lookback) & (sess_ids < s)
            win_sess = sess_ids[win_mask]
            rv_win = rv[win_mask]
            valid = ~np.isnan(rv_win)

            df_tau, _ = eng.adf_session_aware(rv_win[valid], win_sess[valid])
            H = eng.hurst_on_levels(rv_win[valid])
            qualified_s = (df_tau < adf_crit) and (H < regime_hurst_cutoff)

            # GARCH on the stock's own RETURNS -- for position-sizing only, separate from the RV entry signal.
            ret_win = np.diff(log_price[win_mask])
            ret_valid = eng.mask_overnight_gaps(ret_win, win_sess)
            am = arch_model(pd.Series(ret_win[ret_valid]) * 1000, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            fc = res.forecast(horizon=max_bars_per_session, reindex=False)
            sigma_forecast_s = np.sqrt(fc.variance.values[0]) / 1000
            bar_in_session = 0
        else:
            bar_in_session += 1

        # Same session-boundary-aware, thin-sample-guarded rolling window as
        # the pairs engine's z-score -- now normalizing the RV series instead of price/spread.
        window_start = max(current_session_start_idx, t - z_window)
        bars_available = t - window_start
        rv_hist = rv[window_start:t]
        rv_hist = rv_hist[~np.isnan(rv_hist)]
        if bars_available < eng.MIN_WINDOW_BARS or len(rv_hist) < eng.MIN_WINDOW_BARS or np.isnan(rv[t]):
            rv_z_t = 0.0
        else:
            mu, sd = rv_hist.mean(), rv_hist.std()
            rv_z_t = (rv[t] - mu) / sd if sd > 0 else 0.0
            # RV is usually tightly clustered in calm periods, so its own
            # rolling std can be tiny -- a perfectly real, modest drop in
            # RV can then divide through to an enormous z (confirmed on
            # real data: -38 from RV moving 0.0076 -> 0.0033 while its
            # recent std happened to be very small). That's a genuine
            # effect, not a data error, but a spike that large adds no
            # extra trading information beyond "far past the stop" -- clip
            # it so it can't distort the chart or read as a more extreme
            # signal than entry_z/stop_z ever need to see.
            rv_z_t = float(np.clip(rv_z_t, -RV_Z_CLIP, RV_Z_CLIP))

        sigma_t = sigma_forecast_s[min(bar_in_session, len(sigma_forecast_s) - 1)]

        rv_z_path[t] = rv_z_t
        qualified_path[t] = qualified_s
        sigma_path[t] = sigma_t

    return rv_z_path, qualified_path, sigma_path, is_last_bar_of_session, plus_di, minus_di


def run_signal_pnl_and_trades_vol(rv_z_path, sigma_path, qualified_path, log_price,
                                   is_last_bar_of_session, timestamps, oos_start_idx, n,
                                   entry_z, stop_z, exit_z=EXIT_RV_Z, momentum_bars=MOMENTUM_BARS,
                                   plus_di=None, minus_di=None):
    """
    Trading loop for the volatility strategy. Structurally similar to the
    pairs engine's run_signal_pnl_and_trades, but:
      - the SIGNAL (rv_z_path) is one-sided (a vol spike is always a
        positive z-score -- there's no "negative vol spike"), so entry
        DIRECTION comes from Wilder's +DI/-DI (a smoothed, standard
        directional-movement indicator -- see compute_directional_indicators)
        when available, falling back to a raw N-bar price difference
        (momentum_bars) only for the warm-up period +DI/-DI hasn't
        stabilized yet.
      - EXIT and STOP are both on the vol signal reverting/expanding
        further, not on the price signal crossing zero.
      - PnL still accrues from the stock's own price changes -- the
        thing being TRADED is the stock, only the ENTRY/EXIT/STOP
        TRIGGER is the volatility signal.
    """
    if plus_di is None or minus_di is None:
        plus_di = minus_di = np.full(n, np.nan)

    pos = 0.0
    position = np.zeros(n)
    n_stops = 0
    n_regime_flat = 0
    n_eod_flat = 0
    n_trades = 0
    trades = []
    entry_t = None
    entry_rv_z_value = None

    def close_trade(exit_t, reason):
        nonlocal entry_t, entry_rv_z_value
        if entry_t is None:
            return
        trades.append({
            "entry_time": timestamps[entry_t],
            "exit_time": timestamps[exit_t],
            "direction": "long" if position[entry_t] > 0 else "short",
            "entry_rv_z": entry_rv_z_value,
            "exit_rv_z": rv_z_path[exit_t],
            "entry_plus_di": plus_di[entry_t],
            "entry_minus_di": minus_di[entry_t],
            "hold_bars": exit_t - entry_t,
            "exit_reason": reason,
        })
        entry_t = None
        entry_rv_z_value = None

    for t in range(oos_start_idx, n):
        if not qualified_path[t]:
            if pos != 0:
                close_trade(t, "regime break")
                pos = 0.0
                n_regime_flat += 1
        else:
            rv_zt = rv_z_path[t]
            if pos == 0.0:
                # Don't open a brand-new position on the session's last bar --
                # it would be force-flattened one line below in the same
                # iteration, a meaningless 0-bar, 0-pnl "trade" in the log.
                if rv_zt > entry_z and not is_last_bar_of_session[t]:
                    if not np.isnan(plus_di[t]) and not np.isnan(minus_di[t]):
                        # -DI dominant (downward movement has led) -> contrarian long, betting on a bounce.
                        # +DI dominant (upward movement has led) -> contrarian short, betting on a pullback.
                        if minus_di[t] > plus_di[t]:
                            pos = 1.0
                        elif plus_di[t] > minus_di[t]:
                            pos = -1.0
                    elif t - momentum_bars >= 0:
                        momentum = log_price[t] - log_price[t - momentum_bars]
                        if momentum < 0:
                            pos = 1.0
                        elif momentum > 0:
                            pos = -1.0
                    if pos != 0.0:
                        n_trades += 1
                        entry_t, entry_rv_z_value = t, rv_zt
            else:
                if rv_zt <= exit_z:
                    close_trade(t, "vol normalized")
                    pos = 0.0
                elif rv_zt > stop_z:
                    close_trade(t, "vol kept expanding")
                    pos = 0.0
                    n_stops += 1

        if is_last_bar_of_session[t] and pos != 0.0:
            close_trade(t, "end of day")
            pos = 0.0
            n_eod_flat += 1

        position[t] = pos

    dprice = np.diff(log_price, prepend=log_price[0])
    target_vol = np.nanmedian(sigma_path[oos_start_idx:])
    scale = np.clip(target_vol / sigma_path, eng.SCALE_MIN, eng.SCALE_MAX)
    scale = np.nan_to_num(scale, nan=1.0)

    pnl = np.roll(position, 1) * dprice * scale
    pnl[oos_start_idx] = 0.0
    return pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades


def attach_trade_price_single(trades, prices, ticker):
    """Single-instrument version of _engine.attach_trade_prices -- one ticker, not two legs."""
    for t in trades:
        t["entry_price"] = float(prices.loc[t["entry_time"], ticker])
        t["exit_price"] = float(prices.loc[t["exit_time"], ticker])
    return trades


def screen_only(ohlc, params):
    """ohlc: the full OHLC frame from download_intraday_ohlc (MultiIndex columns: Close/High/Low/.../ticker)."""
    prices = ohlc["Close"]
    log_prices = np.log(prices)
    split_idx, sess = eng.split_by_session(prices, params["in_sample_fraction"])
    insample_log = log_prices.iloc[:split_idx]
    insample_sess = sess[:split_idx]
    screening = screen_vol_mean_reversion(
        insample_log, insample_sess, prices.columns,
        params["adf_crit"], params["hurst_cutoff"], params.get("vol_window", VOL_WINDOW),
    )
    return screening, split_idx, sess, log_prices


def run_full_backtest(ohlc, params, screening_result=None, selected_ticker=None):
    if screening_result is None:
        screening, split_idx, sess, log_prices = screen_only(ohlc, params)
    else:
        screening, split_idx, sess, log_prices = screening_result

    if selected_ticker is not None:
        match = screening[screening.ticker == selected_ticker]
        if len(match) == 0:
            raise ValueError(f"ticker {selected_ticker} not found among this universe's screened tickers")
        best = match.iloc[0]
    else:
        best = screening.iloc[0]

    ticker = best["ticker"]
    log_price = log_prices[ticker].to_numpy()
    high = ohlc["High"][ticker].to_numpy()
    low = ohlc["Low"][ticker].to_numpy()
    prices = ohlc["Close"]

    rv_z_path, qualified_path, sigma_path, is_last_bar, plus_di, minus_di = walk_forward_vol(
        log_price, sess, split_idx,
        params["session_lookback"], params["z_window"], params["adf_crit"], params["hurst_cutoff"],
        params.get("vol_window", VOL_WINDOW), high=high, low=low,
        di_period=params.get("di_period", DI_PERIOD),
    )
    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = run_signal_pnl_and_trades_vol(
        rv_z_path, sigma_path, qualified_path, log_price, is_last_bar, prices.index.to_numpy(),
        split_idx, len(prices), params["entry_z"], params["stop_z"],
        exit_z=params.get("exit_z", EXIT_RV_Z), momentum_bars=params.get("momentum_bars", MOMENTUM_BARS),
        plus_di=plus_di, minus_di=minus_di,
    )
    trades = eng.attach_trade_pnl(trades, pnl, prices.index.to_numpy())
    trades = attach_trade_price_single(trades, prices, ticker)
    stats = eng.performance_stats(pnl, split_idx)
    cost_table = eng.cost_sensitivity_table(pnl, position, split_idx)

    rv = compute_realized_vol(log_price, params.get("vol_window", VOL_WINDOW))
    results = pd.DataFrame({
        "timestamp": prices.index,
        "session": sess,
        "log_price": log_price,
        "rv": rv,
        "rv_z": rv_z_path,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "qualified": qualified_path,
        "sigma": sigma_path,
        "position": position,
        "pnl": pnl,
        "is_last_bar_of_session": is_last_bar,
    })

    return dict(
        ticker=ticker, screening=screening, results=results,
        stats=stats, cost_table=cost_table, trades=trades, split_idx=split_idx,
        n_stops=n_stops, n_regime_flat=n_regime_flat, n_eod_flat=n_eod_flat, n_trades=n_trades,
        pct_disqualified=round(float((~qualified_path[split_idx:]).mean() * 100), 1),
    )


def run_full_backtest_basket(ohlc, params, n_tickers=5):
    """
    Trade the top `n_tickers` qualifying tickers SIMULTANEOUSLY, each
    independently signaled off its own realized-volatility regime, so the
    portfolio's return doesn't hinge on which single ticker happened to
    be picked -- directly answers the overfitting problem confirmed
    earlier (a parameter change that helped one ticker badly hurt others).
    Each of the per-ticker backtests reuses run_full_backtest exactly as
    the single-ticker dashboard does; this just runs it once per ticker
    and hands back all of them for add_capital_pnl_basket to combine.
    """
    screening_result = screen_only(ohlc, params)
    screening = screening_result[0]
    qualifying = screening[screening["qualifies"]]
    tickers = qualifying["ticker"].head(n_tickers).tolist()
    if not tickers:
        raise ValueError("no tickers qualify at these thresholds -- widen adf_crit/hurst_cutoff, or lower n_tickers isn't the issue here")

    per_ticker_results = {
        ticker: run_full_backtest(ohlc, params, screening_result=screening_result, selected_ticker=ticker)
        for ticker in tickers
    }
    return dict(tickers=tickers, screening=screening, per_ticker_results=per_ticker_results, split_idx=screening_result[1])


def add_capital_pnl_basket(basket_result, starting_capital):
    """
    Splits starting_capital EVENLY across the traded tickers (each gets
    its own no-leverage budget via _engine.add_capital_pnl), then sums
    their currency P&L into one portfolio equity curve. The sum across
    all tickers' maximum simultaneous exposure still never exceeds your
    total starting_capital, even if every position happens to be open
    at once -- same no-leverage guarantee as the single-ticker version,
    just spread across a basket instead of concentrated in one name.
    """
    tickers = basket_result["tickers"]
    per_capital = starting_capital / len(tickers)
    split_idx = basket_result["split_idx"]

    all_trades = []
    combined_dollar_pnl = None
    for ticker in tickers:
        r = eng.add_capital_pnl(basket_result["per_ticker_results"][ticker], per_capital)
        basket_result["per_ticker_results"][ticker] = r
        for t in r["trades"]:
            t["ticker"] = ticker
        all_trades.extend(r["trades"])

        pnl = r["results"]["pnl"].to_numpy()
        dollar_pnl = np.nan_to_num(pnl) * r["dollar_per_unit"]
        combined_dollar_pnl = dollar_pnl if combined_dollar_pnl is None else combined_dollar_pnl + dollar_pnl

    equity = starting_capital + np.cumsum(combined_dollar_pnl)
    all_trades.sort(key=lambda t: t["entry_time"])
    combined_pnl_frac = combined_dollar_pnl / starting_capital  # currency pnl expressed as a return series, for performance_stats
    stats = eng.performance_stats(combined_pnl_frac, split_idx)

    basket_result.update(
        starting_capital=starting_capital, per_ticker_capital=per_capital,
        equity_curve=equity, final_capital=float(equity[-1]),
        total_return_pct=(float(equity[-1]) / starting_capital - 1) * 100,
        trades=all_trades, stats=stats, n_trades=len(all_trades),
        n_stops=sum(basket_result["per_ticker_results"][tk]["n_stops"] for tk in tickers),
        pct_disqualified=round(float(np.mean([basket_result["per_ticker_results"][tk]["pct_disqualified"] for tk in tickers])), 1),
        cost_table=cost_sensitivity_table_basket(basket_result, split_idx, starting_capital),
    )
    return basket_result


def cost_sensitivity_table_basket(basket_result, split_idx, starting_capital, cost_bps_grid=(0, 1, 2, 5, 10, 20)):
    """Same diagnostic as the single-ticker cost table, but costs are charged per ticker's own turnover, then summed."""
    tickers = basket_result["tickers"]
    rows = []
    for cost_bps in cost_bps_grid:
        combined = None
        for ticker in tickers:
            r = basket_result["per_ticker_results"][ticker]
            pnl = r["results"]["pnl"].to_numpy()
            position = r["results"]["position"].to_numpy()
            dpu = r["dollar_per_unit"]
            position_prev = np.roll(position, 1)
            position_prev[0] = 0.0
            turnover = np.abs(position - position_prev)
            dollar_pnl = np.nan_to_num(pnl) * dpu - turnover * (cost_bps / 10000) * dpu
            combined = dollar_pnl if combined is None else combined + dollar_pnl
        stats = eng.performance_stats(combined / starting_capital, split_idx)
        stats["cost (bps/switch)"] = cost_bps
        rows.append(stats)
    return pd.DataFrame(rows).set_index("cost (bps/switch)")
