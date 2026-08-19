"""
Shared, parameterized core logic for the intraday pairs-trading mission.

01/02 (the step-by-step CLI scripts) and 04 (the interactive dashboard)
all call into this module, so a parameter changed in the interactive
dashboard runs through the EXACT same code path as the batch scripts --
no risk of the two drifting out of sync or disagreeing with each other.

This file has no leading digit so it can be `import`ed normally (unlike
01_*.py / 02_*.py, which need importlib -- see those files' own docstrings).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller  # noqa: F401  (kept for reference/parity with examples/02)
from arch import arch_model
from itertools import combinations

BIST_UNIVERSE = [
    "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS",
    "KCHOL.IS", "SAHOL.IS", "EREGL.IS", "SISE.IS",
    "TCELL.IS", "TTKOM.IS", "THYAO.IS", "PGSUS.IS",
    "FROTO.IS", "TOASO.IS", "BIMAS.IS",
]
DAILY_SELECTED_PAIR = ("SAHOL.IS", "YKBNK.IS")

BIST_INTERVAL = "5m"
BIST_PERIOD = "60d"

MIN_WINDOW_BARS = 10  # see walk_forward_intraday: minimum same-session bars before trusting the z-score

# Bounds on the GARCH-vol-targeting scale factor (run_signal_pnl_and_trades):
# smaller position when the spread is more volatile right now, larger when
# calm. dollar_per_unit_for_trade uses SCALE_MAX as its own ceiling so that
# even the most aggressive (calmest-market) sizing never exceeds capital.
SCALE_MIN, SCALE_MAX = 0.2, 3.0

# Sane range for a session's re-estimated hedge ratio. Confirmed on real
# data: with only session_lookback=5 sessions of history, the daily OLS
# refit can occasionally produce a NEGATIVE beta for two same-sector BIST
# banks (GARAN.IS/ISCTR.IS) -- economically nonsensical (it implies "when
# one rises, the other should fall", contradicting the entire reason the
# pair was selected). Once beta goes degenerate like this, the "spread"
# stops representing real relative value and a trade against it is really
# just noise -- see walk_forward_intraday's qualified_s computation.
BETA_SANITY_BOUNDS = (0.0, 3.0)

# Defaults -- these are what 01/02/03 use unmodified. The interactive
# dashboard (04) lets every one of these be overridden per run.
DEFAULTS = dict(
    in_sample_fraction=0.6,
    adf_crit=-2.86,       # ~5% Dickey-Fuller critical value, constant-only regression, large n
    hurst_cutoff=0.45,    # used as-is for screening; walk-forward regime checks loosen it by +0.05 internally
    session_lookback=5,
    z_window=60,
    entry_z=1.5,
    stop_z=3.5,
)


def download_intraday_universe(tickers, interval=BIST_INTERVAL, period=BIST_PERIOD):
    import yfinance as yf
    prices = yf.download(tickers, interval=interval, period=period, progress=False, auto_adjust=True)["Close"]
    prices = prices.dropna()
    # yfinance returns UTC timestamps -- BIST trades 10:00-18:00 Istanbul
    # time (UTC+3 year-round, Turkey has no DST), so left as UTC every bar
    # looks like it happened ~3 hours earlier than it really did (e.g. the
    # session open prints as ~06:55 UTC instead of ~09:55 local). Convert
    # once here so every downstream consumer (screening, walk-forward,
    # both dashboards, the trade log) shows the time a BIST trader would
    # actually recognize. Session boundaries are unaffected: the trading
    # session doesn't span UTC midnight either way, so which calendar day
    # each bar falls in doesn't change.
    prices.index = prices.index.tz_convert("Europe/Istanbul")
    return prices


def session_ids(index):
    dates = index.date
    is_new = np.concatenate([[True], dates[1:] != dates[:-1]])
    return np.cumsum(is_new) - 1


def mask_overnight_gaps(diffs, sess_ids):
    sess_of_diff = sess_ids[1:]
    sess_of_prev = sess_ids[:-1]
    return sess_of_diff == sess_of_prev


def hurst_on_levels(x, lags=None):
    x = np.asarray(x, dtype=float)
    if lags is None:
        lags = range(2, max(20, len(x) // 10))
    tau_values = np.array(list(lags))
    var_values = np.array([np.var(x[tau:] - x[:-tau]) for tau in tau_values])
    poly = np.polyfit(np.log(tau_values), np.log(var_values), 1)
    return poly[0] / 2.0


def adf_session_aware(spread, sess_ids):
    """From-scratch ADF DF_tau statistic (p=1, constant only), skipping overnight-gap rows."""
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


def split_by_session(prices, in_sample_fraction):
    sess = session_ids(prices.index)
    n_sessions = sess[-1] + 1
    split_session = int(n_sessions * in_sample_fraction)
    split_idx = int(np.searchsorted(sess, split_session))
    return split_idx, sess


def screen_pairs(log_prices, sess_ids, tickers, adf_crit, hurst_cutoff):
    rows = []
    for a, b in combinations(tickers, 2):
        la, lb = log_prices[a].to_numpy(), log_prices[b].to_numpy()
        X = np.column_stack([np.ones_like(lb), lb])
        (alpha, beta), *_ = np.linalg.lstsq(X, la, rcond=None)
        spread = la - beta * lb - alpha
        df_tau, gamma_hat = adf_session_aware(spread, sess_ids)
        H = hurst_on_levels(spread)
        rows.append({
            "pair": f"{a}/{b}", "ticker_a": a, "ticker_b": b,
            "beta": beta, "adf_stat": df_tau, "H": H,
            "qualifies": (df_tau < adf_crit) and (H < hurst_cutoff),
        })
    return pd.DataFrame(rows).sort_values("adf_stat").reset_index(drop=True)


def walk_forward_intraday(log_a, log_b, sess_ids, oos_start_idx,
                           session_lookback, z_window, adf_crit, hurst_cutoff,
                           max_bars_per_session=130):
    """
    Same walk-forward loop as 02_intraday_walk_forward.py, parameterized.
    hurst_cutoff is loosened by +0.05 for the periodic in-loop regime
    check (matching 02's original 0.45 screening / 0.5 regime-check split)
    -- screening should be strict since it's choosing among many pairs,
    the regime re-check should be a bit more forgiving since it's asking
    "has this SPECIFIC, already-vetted pair broken down," not "is this the
    best pair available."
    """
    n = len(log_a)
    beta_path = np.full(n, np.nan)
    z_path = np.full(n, np.nan)
    qualified_path = np.zeros(n, dtype=bool)
    sigma_path = np.full(n, np.nan)
    spread_path = np.full(n, np.nan)
    sd_path = np.full(n, np.nan)  # the z-score's own rolling std, in raw spread units -- see dollar_per_unit_for_trade
    is_last_bar_of_session = np.zeros(n, dtype=bool)

    beta_s, alpha_s, qualified_s, sigma_forecast_s = None, None, True, None
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
            win_a, win_b = log_a[win_mask], log_b[win_mask]
            win_sess = sess_ids[win_mask]

            X = np.column_stack([np.ones_like(win_b), win_b])
            (alpha_s, beta_s), *_ = np.linalg.lstsq(X, win_a, rcond=None)
            spread_hist = win_a - beta_s * win_b - alpha_s

            df_tau, _ = adf_session_aware(spread_hist, win_sess)
            H = hurst_on_levels(spread_hist)
            beta_sane = BETA_SANITY_BOUNDS[0] < beta_s < BETA_SANITY_BOUNDS[1]
            qualified_s = (df_tau < adf_crit) and (H < regime_hurst_cutoff) and beta_sane

            dspread_hist = np.diff(spread_hist)
            valid = mask_overnight_gaps(dspread_hist, win_sess)
            am = arch_model(pd.Series(dspread_hist[valid]) * 1000, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            fc = res.forecast(horizon=max_bars_per_session, reindex=False)
            sigma_forecast_s = np.sqrt(fc.variance.values[0]) / 1000
            bar_in_session = 0
        else:
            bar_in_session += 1

        spread_t = log_a[t] - beta_s * log_b[t] - alpha_s

        # The rolling z-score window is capped at the CURRENT session's
        # start -- never let it span a session boundary. A real overnight
        # price gap, viewed through a freshly re-estimated (session-to-
        # session) hedge ratio, can otherwise look like an enormous
        # artificial jump the instant a new session opens, even though
        # nothing anomalous happened intraday. Early in a session this
        # means a short, noisier window; it grows to the full z_window
        # size as the session progresses.
        #
        # A short window isn't just noisier, though -- with only 1-2
        # points its STANDARD DEVIATION is itself unstable (can be tiny
        # by chance), which blows the z-score up to absurd values (seen
        # empirically: z > 400 on the 2nd bar of a session) even though
        # nothing unusual happened. MIN_WINDOW_BARS treats those first
        # few minutes of each session as "not enough same-session history
        # to trade on yet" -- z_t = 0 (no signal) until it's cleared,
        # which also means no new position can open in that window.
        window_start = max(current_session_start_idx, t - z_window)
        bars_available = t - window_start
        if bars_available < MIN_WINDOW_BARS:
            z_t = 0.0
            sd = np.nan
        else:
            hist_a, hist_b = log_a[window_start:t], log_b[window_start:t]
            spread_window = hist_a - beta_s * hist_b - alpha_s
            mu, sd = spread_window.mean(), spread_window.std()
            z_t = (spread_t - mu) / sd if sd > 0 else 0.0

        sigma_t = sigma_forecast_s[min(bar_in_session, len(sigma_forecast_s) - 1)]

        beta_path[t] = beta_s
        z_path[t] = z_t
        qualified_path[t] = qualified_s
        sigma_path[t] = sigma_t
        spread_path[t] = spread_t
        sd_path[t] = sd

    return beta_path, z_path, qualified_path, sigma_path, spread_path, is_last_bar_of_session, sd_path


def run_signal_pnl_and_trades(z_path, sigma_path, qualified_path, spread_path,
                               is_last_bar_of_session, timestamps, oos_start_idx, n,
                               entry_z, stop_z):
    """
    Same trading loop as 02's run_signal_and_pnl, PLUS a trade log: one row
    per completed position (entry to exit), with WHY it closed. This is
    what powers the dashboard's positions table.
    """
    pos = 0.0
    position = np.zeros(n)
    n_stops = 0
    n_regime_flat = 0
    n_eod_flat = 0
    n_trades = 0
    trades = []
    entry_t = None
    entry_z_value = None

    def close_trade(exit_t, reason):
        nonlocal entry_t, entry_z_value
        if entry_t is None:
            return
        trades.append({
            "entry_time": timestamps[entry_t],
            "exit_time": timestamps[exit_t],
            "direction": "long spread" if position[entry_t] > 0 else "short spread",
            "entry_z": entry_z_value,
            "exit_z": z_path[exit_t],
            "hold_bars": exit_t - entry_t,
            "exit_reason": reason,
        })
        entry_t = None
        entry_z_value = None

    for t in range(oos_start_idx, n):
        if not qualified_path[t]:
            if pos != 0:
                close_trade(t, "regime break")
                pos = 0.0
                n_regime_flat += 1
        else:
            zt = z_path[t]
            if pos == 0.0:
                if zt > entry_z:
                    pos = -1.0
                    n_trades += 1
                    entry_t, entry_z_value = t, zt
                elif zt < -entry_z:
                    pos = 1.0
                    n_trades += 1
                    entry_t, entry_z_value = t, zt
            elif pos == 1.0 and zt >= 0:
                close_trade(t, "reversion")
                pos = 0.0
            elif pos == -1.0 and zt <= 0:
                close_trade(t, "reversion")
                pos = 0.0
            elif abs(zt) > stop_z:
                close_trade(t, "hard stop")
                pos = 0.0
                n_stops += 1

        if is_last_bar_of_session[t] and pos != 0.0:
            close_trade(t, "end of day")
            pos = 0.0
            n_eod_flat += 1

        position[t] = pos

    dspread = np.diff(spread_path, prepend=spread_path[0])
    target_vol = np.nanmedian(sigma_path[oos_start_idx:])
    scale = np.clip(target_vol / sigma_path, SCALE_MIN, SCALE_MAX)
    scale = np.nan_to_num(scale, nan=1.0)

    pnl = np.roll(position, 1) * dspread * scale
    pnl[oos_start_idx] = 0.0

    # Trade PnL is filled in by attach_trade_pnl() once this pnl array exists.
    return pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades


def attach_trade_pnl(trades, pnl, timestamps):
    """Second pass: sum each trade's bar-by-bar pnl now that both are available."""
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    for trade in trades:
        i0 = ts_to_idx[trade["entry_time"]]
        i1 = ts_to_idx[trade["exit_time"]]
        trade["pnl"] = float(np.nansum(pnl[i0 + 1:i1 + 1]))
    return trades


def attach_trade_prices(trades, prices, ticker_a, ticker_b):
    """
    Adds the actual quoted close price of BOTH legs at entry and exit --
    the real, checkable "buy at X, sell at Y" numbers behind each trade,
    as opposed to the abstract spread/z-score units everything else here
    is computed in. Lets a trade be verified against real market data
    (e.g. a BIST quote history) bar by bar.
    """
    for t in trades:
        t["entry_price_a"] = float(prices.loc[t["entry_time"], ticker_a])
        t["entry_price_b"] = float(prices.loc[t["entry_time"], ticker_b])
        t["exit_price_a"] = float(prices.loc[t["exit_time"], ticker_a])
        t["exit_price_b"] = float(prices.loc[t["exit_time"], ticker_b])
    return trades


TRADE_CARD_CSS = """
  .trade-cards { display: flex; flex-direction: column; gap: 4px; }
  .trade-card { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); }
  .trade-card summary {
    cursor: pointer; padding: 7px 10px 7px 22px; font-size: 0.78rem; list-style: none;
    display: grid; grid-template-columns: 1.7fr 1.9fr 0.8fr 0.8fr 0.6fr 1fr 1.1fr;
    gap: 8px; align-items: center; position: relative;
  }
  .trade-card summary::-webkit-details-marker { display: none; }
  /* position:absolute takes this OUT of grid flow -- without it, the
     ::before pseudo-element becomes an actual grid item (per spec) and
     shifts every real column over by one, since it's the first child. */
  .trade-card summary::before {
    content: '\\25B8'; color: #8a8a86; position: absolute; left: 8px; top: 9px;
  }
  .trade-card[open] summary::before { content: '\\25BE'; }
  .trade-cards-header {
    display: grid; grid-template-columns: 1.7fr 1.9fr 0.8fr 0.8fr 0.6fr 1fr 1.1fr;
    gap: 8px; padding: 4px 10px 4px 26px; font-size: 0.72rem; color: #8a8a86;
  }
  .trade-detail { padding: 4px 14px 10px 26px; border-top: 1px solid var(--border); font-size: 0.78rem; }
  .trade-detail table { width: auto; min-width: 320px; }
"""


def build_trade_cards_html(trades, ticker_a, ticker_b):
    """
    One collapsible card per trade instead of one wide table row -- adding
    the actual entry/exit price of both legs to every row (what a trade
    was verifiably bought/sold at) would otherwise push the table past a
    readable width. Collapsed, a card shows the same summary a table row
    would; expanded, it shows both legs' real quoted prices and % moves,
    so a specific trade can be checked against real market data without
    every other trade's row also carrying that width.

    Works with or without currency conversion having been run: if a
    trade has "pnl_currency" (i.e. add_capital_pnl already ran, as in
    04's interactive dashboard), PnL is shown in TRY; otherwise it falls
    back to the raw spread-pnl unit (as in 03's static dashboard, which
    doesn't model capital at all).
    """
    has_currency = bool(trades) and "pnl_currency" in trades[0]

    header = f"""<div class="trade-cards-header">
      <span>Entry &rarr; Exit</span><span>Direction</span><span>Entry z</span>
      <span>Exit z</span><span>Hold</span><span>Reason</span><span>PnL</span>
    </div>"""

    cards = []
    for t in trades:
        if has_currency:
            pnl_value = t["pnl_currency"]
            pnl_text = f"{pnl_value:+,.0f} TRY"
        else:
            pnl_value = t["pnl"]
            pnl_text = f"{pnl_value:+.5f} (raw units)"
        pnl_class = "pos" if pnl_value > 0 else "neg"
        entry_pct_a = (t["exit_price_a"] / t["entry_price_a"] - 1) * 100
        entry_pct_b = (t["exit_price_b"] / t["entry_price_b"] - 1) * 100

        sizing_note = (
            f"""<p class="hint">Base unit: <strong>{t['dollar_per_unit']:,.0f} TRY</strong> out of
            {t['starting_capital']:,.0f} TRY starting capital &mdash; the strategy's own GARCH-based
            volatility scale (smaller when the spread is choppy, larger when calm) then adjusts the
            actual exposure bar by bar, automatically, capped so it never exceeds your capital.
            This trade's {pnl_text} reflects that full path, not just this one number.</p>"""
            if has_currency else ""
        )
        cards.append(f"""<details class="trade-card">
  <summary>
    <span>{t['entry_time']} &rarr; {t['exit_time']}</span>
    <span>{direction_label(t['direction'], ticker_a, ticker_b)}</span>
    <span>{t['entry_z']:.2f}</span>
    <span>{t['exit_z']:.2f}</span>
    <span>{t['hold_bars']}</span>
    <span>{t['exit_reason']}</span>
    <span class="{pnl_class}">{pnl_text}</span>
  </summary>
  <div class="trade-detail">
    <table>
      <tr><th></th><th>Entry price</th><th>Exit price</th><th>Change</th></tr>
      <tr><td>{ticker_a}</td><td>{t['entry_price_a']:.2f}</td><td>{t['exit_price_a']:.2f}</td>
          <td class="{'pos' if entry_pct_a >= 0 else 'neg'}">{entry_pct_a:+.2f}%</td></tr>
      <tr><td>{ticker_b}</td><td>{t['entry_price_b']:.2f}</td><td>{t['exit_price_b']:.2f}</td>
          <td class="{'pos' if entry_pct_b >= 0 else 'neg'}">{entry_pct_b:+.2f}%</td></tr>
    </table>
    {sizing_note}
  </div>
</details>""")

    return header + f'<div class="trade-cards">{"".join(cards)}</div>'


def direction_label(direction, ticker_a, ticker_b):
    """
    "short spread" / "long spread" says which way the ABSTRACT spread was
    traded, but names neither of the two actual stocks -- easy to misread
    as if only one instrument were involved. spread = log(ticker_a) -
    beta*log(ticker_b), so:
      - "short spread" (bet the spread falls) = sell ticker_a, buy ticker_b
      - "long spread" (bet the spread rises)  = buy ticker_a, sell ticker_b
    """
    if direction == "short spread":
        return f"sell {ticker_a} / buy {ticker_b}"
    elif direction == "long spread":
        return f"buy {ticker_a} / sell {ticker_b}"
    return direction


def performance_stats(pnl, oos_start_idx, bars_per_year=None):
    pnl = pnl[oos_start_idx:]
    if bars_per_year is None:
        bars_per_year = 95 * 250
    ann_ret = np.nanmean(pnl) * bars_per_year
    ann_vol = np.nanstd(pnl) * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = np.nancumsum(pnl)
    max_drawdown = np.max(np.maximum.accumulate(cum) - cum)
    return {
        "annualized return": round(float(ann_ret), 4),
        "annualized vol": round(float(ann_vol), 4),
        "sharpe (naive)": round(float(sharpe), 3),
        "max drawdown": round(float(max_drawdown), 4),
    }


def cost_sensitivity_table(pnl, position, oos_start_idx, cost_bps_grid=(0, 1, 2, 5, 10, 20)):
    position_prev = np.roll(position, 1)
    position_prev[0] = 0.0
    turnover = np.abs(position - position_prev)

    rows = []
    for cost_bps in cost_bps_grid:
        cost = turnover * (cost_bps / 10000)
        pnl_after_cost = pnl - cost
        pnl_after_cost[oos_start_idx] = 0.0
        stats = performance_stats(pnl_after_cost, oos_start_idx)
        stats["cost (bps/switch)"] = cost_bps
        rows.append(stats)
    return pd.DataFrame(rows).set_index("cost (bps/switch)")


def dollar_per_unit_for_trade(starting_capital):
    """
    No manual risk-% dial: the strategy already decides how aggressively
    to size EACH BAR on its own, via the GARCH-based volatility scale
    already baked into the raw pnl before this function ever runs (see
    run_signal_pnl_and_trades: scale = clip(target_vol/sigma_t, SCALE_MIN,
    SCALE_MAX) -- smaller effective exposure when the spread is more
    volatile right now, larger when it's calm). This function only fixes
    the CURRENCY UNIT that scale multiplies against, set so that even at
    the maximum possible scale (SCALE_MAX), the notional traded never
    exceeds starting_capital: position(+-1) * scale(<=SCALE_MAX) *
    dollar_per_unit <= 1 * SCALE_MAX * (capital/SCALE_MAX) = capital.
    No leverage, ever, and no dial to tune -- the situational sizing is
    already happening upstream.

    PREVIOUS APPROACHES (kept here as lessons, not code):
    (1) a single global conversion factor calibrated to the median
        SINGLE-BAR GARCH-forecast volatility -- silently oversized any
        multi-bar trade (confirmed: a 52-bar trade lost 18,838 TRY
        against an intended 3,750 TRY budget, ~5x oversized).
    (2) sizing to each trade's distance from its entry z-score to the
        hard stop -- mathematically standard risk-based sizing, but
        implies LEVERAGE whenever volatility is small relative to the
        risk budget (confirmed: routinely 5-900x starting_capital).
    (3) a fixed user-chosen risk_per_trade_pct% of capital -- removes
        leverage but also removes any situational awareness, requiring
        a manual dial the user explicitly asked not to have to set.
    """
    return starting_capital / SCALE_MAX


def add_capital_pnl(result, starting_capital):
    """
    Adds currency-denominated fields to a run_full_backtest() result: an
    equity curve, final capital, total return %, and each trade's PnL in
    currency terms. Every trade uses the SAME dollar_per_unit (see
    dollar_per_unit_for_trade) -- the bar-by-bar GARCH scale already in
    `pnl` is what makes the actual currency exposure vary with volatility.
    """
    results_df = result["results"]
    pnl = results_df["pnl"].to_numpy()
    n = len(pnl)

    dpu = dollar_per_unit_for_trade(starting_capital)

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


def screen_only(prices, params):
    """
    Just the screening stage (Step 1): in-sample split + screen_pairs.
    Split out from run_full_backtest so a caller (e.g. the interactive
    dashboard) can screen once to show the user which pairs qualified,
    let them pick one, and reuse that same screening result for the
    actual walk-forward run rather than re-screening twice.
    """
    log_prices = np.log(prices)
    split_idx, sess = split_by_session(prices, params["in_sample_fraction"])
    insample_log = log_prices.iloc[:split_idx]
    insample_sess = sess[:split_idx]
    screening = screen_pairs(insample_log, insample_sess, prices.columns,
                              params["adf_crit"], params["hurst_cutoff"])
    return screening, split_idx, sess, log_prices


def run_full_backtest(prices, params, screening_result=None, selected_pair=None):
    """
    One call, all four stages (screen -> select -> walk-forward -> trades),
    with every tunable parameter passed explicitly. This is what both the
    interactive dashboard and (indirectly, via defaults) the batch scripts
    call.

    selected_pair: optional (ticker_a, ticker_b) tuple to force a SPECIFIC
    pair instead of automatically taking the top-ranked one by ADF stat --
    lets a caller offer a choice among the pairs that passed screening
    (or even one that didn't) rather than always silently trading whichever
    pair happened to rank first. Matches either ticker order.
    screening_result: optional pre-computed screen_only() output, to avoid
    re-running the screen when the caller already has one (e.g. to
    populate a pair-selection dropdown before the actual backtest runs).
    """
    if screening_result is None:
        screening, split_idx, sess, log_prices = screen_only(prices, params)
    else:
        screening, split_idx, sess, log_prices = screening_result

    if selected_pair is not None:
        ta, tb = selected_pair
        match = screening[
            ((screening.ticker_a == ta) & (screening.ticker_b == tb)) |
            ((screening.ticker_a == tb) & (screening.ticker_b == ta))
        ]
        if len(match) == 0:
            raise ValueError(f"pair {ta}/{tb} not found among this universe's screened pairs")
        best = match.iloc[0]
    else:
        best = screening.iloc[0]

    ticker_a, ticker_b = best["ticker_a"], best["ticker_b"]
    log_a = log_prices[ticker_a].to_numpy()
    log_b = log_prices[ticker_b].to_numpy()

    beta_path, z_path, qualified_path, sigma_path, spread_path, is_last_bar, sd_path = walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        params["session_lookback"], params["z_window"], params["adf_crit"], params["hurst_cutoff"],
    )
    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = run_signal_pnl_and_trades(
        z_path, sigma_path, qualified_path, spread_path, is_last_bar, prices.index.to_numpy(),
        split_idx, len(prices), params["entry_z"], params["stop_z"],
    )
    trades = attach_trade_pnl(trades, pnl, prices.index.to_numpy())
    trades = attach_trade_prices(trades, prices, ticker_a, ticker_b)
    stats = performance_stats(pnl, split_idx)
    cost_table = cost_sensitivity_table(pnl, position, split_idx)

    results = pd.DataFrame({
        "timestamp": prices.index,
        "session": sess,
        "beta": beta_path,
        "z": z_path,
        "qualified": qualified_path,
        "sigma": sigma_path,
        "spread": spread_path,
        "sd": sd_path,
        "position": position,
        "pnl": pnl,
        "is_last_bar_of_session": is_last_bar,
    })

    return dict(
        ticker_a=ticker_a, ticker_b=ticker_b, screening=screening,
        results=results, stats=stats, cost_table=cost_table, trades=trades,
        split_idx=split_idx, n_stops=n_stops, n_regime_flat=n_regime_flat,
        n_eod_flat=n_eod_flat, n_trades=n_trades,
        pct_disqualified=round(float((~qualified_path[split_idx:]).mean() * 100), 1),
    )
