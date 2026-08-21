"""
BIST30 cross-sectional momentum -- STANDALONE, single-file backtest.

This is the momentum strategy from this mission (`_engine_momentum.py`),
extracted verbatim together with the pieces of `_common.py` and
`universe.py` it depends on, so it can be copied to another machine and run
on its own. Nothing here was re-tuned, simplified or "improved" during the
extraction: the trading logic, the parameters, the session-aware split, the
sizing and the statistics are byte-for-byte the same calculations the
mission's own engine performs, and the numbers this file prints have been
checked to match `_engine_momentum.run_full_backtest_momentum` exactly on
the same price panel.

Requirements: python 3.9+, numpy, pandas, and yfinance (only if you let it
download; not needed when reading a saved CSV).

    pip install numpy pandas yfinance

Run it:

    python momentum_standalone.py                       # download + backtest
    python momentum_standalone.py --top-n 3 --rebalance 90
    python momentum_standalone.py --save-prices bist30_hourly.csv
    python momentum_standalone.py --prices bist30_hourly.csv       # no network
    python momentum_standalone.py --trades-csv trades.csv

YOUR OWN DATA. --prices takes whatever shape the data is already in, so a
platform export doesn't have to be reformatted by hand:

    --prices panel.csv        wide: first column dates, one column per ticker
    --prices bars.csv         long: one row per bar per ticker (ticker + close columns)
    --prices C:\\data\\bist     a FOLDER of one file per ticker; the file name is the ticker

Any extension works (.csv, .txt, .vrd, ...) as long as the CONTENT is
delimited text -- the separator (`;` `,` tab `|`), a decimal comma, and
day-first dates are all detected. Column names are matched in English or
Turkish (tarih, hisse, kapanis, fiyat, ...); where a file is unusual, name
the columns with --date-col / --ticker-col / --close-col. A vendor's own
BINARY format has to be exported to text from the platform first.

Check what was parsed before trusting a number:

    python momentum_standalone.py --prices C:\\data\\bist --inspect

Data expectations: CLOSE prices, adjusted for splits and dividends (an
unadjusted series turns every split into a fake -50% momentum signal), one
row per bar per ticker, and enough history for the lookback windows. The
defaults are in HOURLY bars -- feed daily data and the script says so and
suggests daily-equivalent windows rather than silently backtesting a
nine-month lookback.

REPRODUCIBILITY WARNING. yfinance serves a ROLLING window of hourly bars
(~730 days), so downloading tomorrow gives a slightly different panel than
downloading today, and the backtest's numbers will drift accordingly. If
you need a specific run to be reproducible -- e.g. to show the same figures
to someone else -- save the panel once with --save-prices and re-run from
that CSV with --prices-csv.

WHAT THIS IS NOT. No transaction costs are charged in the headline result
(a cost-sensitivity table is printed separately, so you can see what the
edge survives), no slippage, no market impact, no borrow, no lot rounding,
no dividends/taxes beyond yfinance's auto-adjusted close, and no live
trading connection. It is a research backtest for evaluating an idea.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# From universe.py -- the real BIST30 constituents and the hourly data layer.
# --------------------------------------------------------------------------

BIST30_INDEX = "XU030.IS"

BIST30_TICKERS = [
    "AEFES.IS", "AKBNK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS",
    "DSTKF.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GARAN.IS", "GUBRF.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS",
    "MGROS.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TRALT.IS", "TTKOM.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS",
]

# Tickers worth knowing about before trusting a lookback-heavy fit on them.
# TRALT.IS was renamed from KOZAL.IS and yfinance only carries it from
# 2025-11-24; DSTKF.IS IPO'd in Jan 2025. Neither is excluded -- a ticker
# without enough trailing history simply can't be ranked at a given
# rebalance (its ROC/MA are NaN), which is the honest handling.
SHORT_HISTORY_TICKERS = {"DSTKF.IS", "TRALT.IS"}

# yfinance caps the 60-minute interval at ~730 days of history (vs. 60 days
# for 5m/15m bars, and effectively unlimited for daily bars). ~2 years of
# hourly bars is the deliberate middle ground: finer than a daily study,
# far more usable history than 5-minute intraday.
BIST30_INTERVAL = "60m"
BIST30_PERIOD = "730d"
BIST_TZ = "Europe/Istanbul"


def download_universe(tickers=None, period=BIST30_PERIOD, interval=BIST30_INTERVAL):
    """Hourly auto-adjusted close panel for the given tickers (default: the full BIST30)."""
    import yfinance as yf
    if tickers is None:
        tickers = BIST30_TICKERS
    prices = yf.download(tickers, period=period, interval=interval,
                         progress=False, auto_adjust=True)["Close"]
    # Only drop rows where EVERY ticker is missing (e.g. a holiday nothing
    # traded on). Do NOT forward-fill or drop rows for a single short-
    # history ticker's sake -- that would silently truncate the ENTIRE
    # 30-ticker panel down to whichever ticker listed most recently.
    prices = prices.dropna(how="all")
    # yfinance returns UTC timestamps; convert once so every downstream
    # consumer shows the hour a BIST trader would actually recognize.
    prices.index = prices.index.tz_convert(BIST_TZ)
    return prices


def bars_per_year(index):
    """
    Measures the real average bar density from an actual downloaded index,
    rather than assuming a bar count per day the way a daily (252) or
    5-minute (95*250) study does -- BIST's hourly bar count per session
    isn't something to guess at when it's cheap to measure directly.
    """
    if len(index) < 2:
        return np.nan
    n_days = pd.Series(index).dt.date.nunique()
    if n_days == 0:
        return np.nan
    bars_per_day = len(index) / n_days
    return bars_per_day * 252


# --------------------------------------------------------------------------
# From _common.py -- session handling, sizing and performance statistics.
# --------------------------------------------------------------------------

# Bounds on any vol-targeting-style scale factor. Momentum never uses them
# (it declares scale_max=1.0 -- see run_full_backtest_momentum), but
# add_capital_pnl keeps the same default the shared module has.
SCALE_MIN, SCALE_MAX = 0.2, 3.0


def session_ids(index):
    """Integer session id per calendar day."""
    dates = index.date
    is_new = np.concatenate([[True], dates[1:] != dates[:-1]])
    return np.cumsum(is_new) - 1


def split_by_session(index, in_sample_fraction):
    """In/out-of-sample split index, landing on a session boundary."""
    sess = session_ids(index)
    n_sessions = sess[-1] + 1
    split_session = int(n_sessions * in_sample_fraction)
    split_idx = int(np.searchsorted(sess, split_session))
    return split_idx, sess


def dollar_per_unit_for_trade(starting_capital, scale_max=SCALE_MAX):
    """
    No manual risk-% dial: an engine's own situational scale (bounded by
    scale_max) multiplies this unit, so worst case notional =
    1 * scale_max * (capital/scale_max) = capital. Never leveraged.
    """
    return starting_capital / scale_max


def base_unit_multiplier(results, bar_idx):
    """
    How many base units a position opened at `bar_idx` actually put to
    work, per name. Multiply by dollar_per_unit_for_trade's unit to get the
    real traded amount in currency.

    For this equal-weighted basket engine each held name gets 1/n of the
    book, so the per-name multiplier shrinks as the basket grows: top_n=5
    puts a fifth of the unit into each name, not the whole unit.
    """
    for column in ("n_held", "n_active"):
        if column in results:
            n_positions = float(results[column].iat[bar_idx])
            return 1.0 / n_positions if n_positions > 0 else 1.0
    return 1.0


def add_capital_pnl(result, starting_capital):
    """
    Adds currency-denominated fields to a backtest result: equity curve,
    final capital, total return %, each trade's PnL in currency terms.

    Note that sizing is off the STARTING capital and never compounds --
    equity can run to 150k and the next trade is still sized off 100k. That
    keeps the Sharpe honest (under compounding the late trades dominate the
    statistics purely by being bigger), but it understates both what a
    reinvesting account would earn and the drawdown it would take.
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
    """Out-of-sample only. bars_per_year is required, not guessed -- measure it from the real index."""
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


# --------------------------------------------------------------------------
# From _engine_momentum.py -- the strategy itself.
#
# Cross-sectional momentum / trend-following across the BIST30 basket.
# Long-only, equal-capital top-N rotation: no shorting, consistent with the
# no-leverage philosophy and avoiding BIST margin/short-sale mechanics.
#
# At each rebalance, rank all tickers with enough trailing history by their
# rate-of-change (ROC) over a lookback window, keep only those a simple
# trend filter (fast MA above slow MA) confirms are actually trending
# rather than just noisy, and hold the top N equally weighted until the
# next rebalance.
#
# Sizing note: "pnl" here is directly the basket's fractional bar-to-bar
# return (e.g. 0.002 = 0.2%), and this engine is never more than 100%
# invested (no leverage, nothing borrowed) -- so it declares scale_max=1.0,
# which makes add_capital_pnl's dollar_pnl = pnl * (capital / 1.0) =
# pnl * capital, exactly the right currency conversion for a
# fractional-return series.
# --------------------------------------------------------------------------

DEFAULTS = dict(
    in_sample_fraction=0.6,
    lookback_mom=180,   # bars (~20 sessions at 9 bars/day) trailing ROC-ranking window
    ma_fast=45,          # bars (~5 sessions)
    ma_slow=180,         # bars (~20 sessions)
    rebalance=45,        # bars (~5 sessions) between re-ranking
    top_n=5,
    min_history_bars=200,
)


def compute_roc(prices, lookback):
    """Rate of change over `lookback` bars, per ticker."""
    return prices / prices.shift(lookback) - 1.0


def trend_filter(prices, fast, slow):
    """True where the fast trailing MA is above the slow trailing MA -- a whipsaw guard, not the ranking signal itself."""
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    return ma_fast > ma_slow


def rank_universe(roc_row, trend_row, top_n):
    """Given one rebalance date's ROC values and trend-filter flags, return the top-N eligible tickers."""
    eligible = roc_row[trend_row.fillna(False)].dropna()
    return eligible.sort_values(ascending=False).head(top_n).index.tolist()


def walk_forward_momentum(prices, params):
    """
    Rebalance every `rebalance` bars using ONLY data available as of the
    PREVIOUS bar's close (roc.iloc[t-1] / trend.iloc[t-1], never t itself)
    -- no look-ahead into the bar a position would actually be sized on.
    A ticker missing from the panel that bar (e.g. DSTKF.IS/TRALT.IS before
    their listing/rename date) simply can't be ranked (ROC/MA are NaN)
    until it has enough real history.
    """
    log_ret = np.log(prices).diff()
    roc = compute_roc(prices, params["lookback_mom"])
    trend = trend_filter(prices, params["ma_fast"], params["ma_slow"])
    n = len(prices)
    tickers = list(prices.columns)
    split_idx, _ = split_by_session(prices.index, params["in_sample_fraction"])

    pnl = np.zeros(n)
    n_held_path = np.zeros(n, dtype=int)
    turnover_path = np.zeros(n)
    contrib = pd.DataFrame(0.0, index=prices.index, columns=tickers)

    current_holdings = {}   # ticker -> entry bar index
    trades = []
    rebalance = params["rebalance"]
    top_n = params["top_n"]
    last_rebal_t = None

    for t in range(split_idx, n):
        do_rebal = (last_rebal_t is None) or (t - last_rebal_t >= rebalance)
        if do_rebal:
            last_rebal_t = t
            roc_row = roc.iloc[t - 1]
            trend_row = trend.iloc[t - 1]
            new_selected = rank_universe(roc_row, trend_row, top_n)

            dropped = [tk for tk in current_holdings if tk not in new_selected]
            added = [tk for tk in new_selected if tk not in current_holdings]
            turnover_path[t] = (len(dropped) + len(added)) / max(top_n, 1)

            for tk in dropped:
                entry_t = current_holdings.pop(tk)
                trades.append({"ticker": tk, "entry_time": prices.index[entry_t],
                               "exit_time": prices.index[t], "exit_reason": "rebalance drop",
                               "_entry_t": entry_t, "_exit_t": t})
            for tk in added:
                current_holdings[tk] = t

        if current_holdings:
            w = 1.0 / len(current_holdings)
            for tk in current_holdings:
                r = log_ret[tk].iloc[t]
                if np.isfinite(r):
                    contrib.iat[t, tickers.index(tk)] = w * r
                    pnl[t] += w * r
        n_held_path[t] = len(current_holdings)

    for tk, entry_t in current_holdings.items():
        trades.append({"ticker": tk, "entry_time": prices.index[entry_t],
                       "exit_time": prices.index[n - 1], "exit_reason": "end of backtest",
                       "_entry_t": entry_t, "_exit_t": n - 1})

    for tr in trades:
        seg = contrib[tr["ticker"]].iloc[tr["_entry_t"] + 1: tr["_exit_t"] + 1]
        tr["pnl"] = float(seg.sum())
        tr["hold_bars"] = tr["_exit_t"] - tr["_entry_t"]
        del tr["_entry_t"]
        del tr["_exit_t"]

    return pnl, n_held_path, turnover_path, trades, split_idx


def cost_sensitivity_table_momentum(pnl, turnover_path, split_idx, bars_per_year,
                                    cost_bps_grid=(0, 1, 2, 5, 10, 20)):
    """
    Basket-turnover version of a cost sweep: turnover_path is already a
    fraction-of-portfolio-turned-over series, so a cost of `bps` per unit
    of turnover is charged on the bars a rebalance actually traded.
    """
    rows = []
    for cost_bps in cost_bps_grid:
        cost = turnover_path * (cost_bps / 10000)
        pnl_after_cost = pnl - cost
        pnl_after_cost[split_idx] = 0.0
        stats = performance_stats(pnl_after_cost, split_idx, bars_per_year)
        stats["cost (bps/switch)"] = cost_bps
        rows.append(stats)
    return pd.DataFrame(rows).set_index("cost (bps/switch)")


def run_full_backtest_momentum(prices, params=None, bars_per_year_value=None):
    """One call: rank -> rotate -> trade log -> stats."""
    params = {**DEFAULTS, **(params or {})}
    if bars_per_year_value is None:
        bars_per_year_value = bars_per_year(prices.index)

    pnl, n_held_path, turnover_path, trades, split_idx = walk_forward_momentum(prices, params)
    stats = performance_stats(pnl, split_idx, bars_per_year_value)
    cost_table = cost_sensitivity_table_momentum(pnl, turnover_path, split_idx, bars_per_year_value)

    results = pd.DataFrame({"timestamp": prices.index, "n_held": n_held_path, "pnl": pnl})

    return dict(
        results=results, stats=stats, cost_table=cost_table, trades=trades,
        split_idx=split_idx, n_trades=len(trades), scale_max=1.0,
        bars_per_year=bars_per_year_value,
        avg_n_held=round(float(n_held_path[split_idx:].mean()), 2),
    )


# --------------------------------------------------------------------------
# Traded amounts -- what each position actually put to work, in lira.
# --------------------------------------------------------------------------

def attach_trade_amounts(result, prices):
    """
    Adds to every trade the quoted entry/exit price of its ticker, the
    amount that position put to work at entry, and the share count that
    amount buys.

    The amount is the base unit (starting capital / scale_max) times the
    engine's own sizing at that bar: with top_n=5 each name gets a fifth of
    the book (20,000 TRY out of 100,000), not the whole unit. It is
    measured AT ENTRY -- if the basket's size changes mid-hold, the weight
    drifts with it, which is why per-trade PnL is attributed bar by bar
    rather than from one fixed stake.

    Share counts are indicative: the amount divided by the quoted price.
    The backtest sizes in continuous units and never rounds to a lot.
    """
    results = result["results"]
    dpu = result.get("dollar_per_unit")
    positions = {pd.Timestamp(ts).value: i for i, ts in enumerate(results["timestamp"].to_numpy())}

    for trade in result["trades"]:
        ticker = trade.get("ticker")
        entry_i = positions.get(pd.Timestamp(trade["entry_time"]).value)
        if ticker is not None and ticker in prices.columns:
            try:
                trade["entry_price"] = float(prices.at[trade["entry_time"], ticker])
                trade["exit_price"] = float(prices.at[trade["exit_time"], ticker])
            except (KeyError, ValueError, TypeError):
                pass
        if dpu is None or entry_i is None:
            continue
        trade["amount"] = dpu * base_unit_multiplier(results, entry_i)
        entry_price = trade.get("entry_price")
        if entry_price and np.isfinite(entry_price):
            trade["shares"] = int(trade["amount"] // entry_price)
    return result["trades"]


def trades_frame(result):
    """The trade log as a DataFrame, chronological, ready to print or save."""
    rows = []
    for n, trade in enumerate(sorted(result["trades"], key=lambda t: t["entry_time"]), start=1):
        rows.append({
            "trade": n,
            "ticker": trade["ticker"],
            "entry_time": pd.Timestamp(trade["entry_time"]).strftime("%Y-%m-%d %H:%M"),
            "exit_time": pd.Timestamp(trade["exit_time"]).strftime("%Y-%m-%d %H:%M"),
            "hold_bars": trade["hold_bars"],
            "exit_reason": trade["exit_reason"],
            "amount_try": round(trade.get("amount", float("nan")), 0),
            "shares": trade.get("shares"),
            "entry_price": round(trade.get("entry_price", float("nan")), 2),
            "exit_price": round(trade.get("exit_price", float("nan")), 2),
            "move_pct": round((trade["exit_price"] / trade["entry_price"] - 1) * 100, 2)
                        if trade.get("entry_price") and trade.get("exit_price") else float("nan"),
            "pnl_try": round(trade.get("pnl_currency", trade["pnl"]), 0),
        })
    return pd.DataFrame(rows)


def per_ticker_frame(result):
    """Which stocks the strategy actually traded, and how each did."""
    frame = trades_frame(result)
    if frame.empty:
        return frame
    grouped = frame.groupby("ticker").agg(
        trades=("trade", "count"),
        avg_amount_try=("amount_try", "mean"),
        avg_move_pct=("move_pct", "mean"),
        total_pnl_try=("pnl_try", "sum"),
    ).sort_values("total_pnl_try", ascending=False)
    return grouped.round(2).reset_index()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# Column names this importer recognises, lowercased. Turkish exports from
# BIST data vendors label things in Turkish, so both languages are listed.
DATE_COLUMNS = ("datetime", "date", "timestamp", "time", "tarih", "tarih_saat", "gun")
TIME_COLUMNS = ("time", "saat", "hour")
TICKER_COLUMNS = ("ticker", "symbol", "sembol", "hisse", "kod", "code", "name", "isim")
CLOSE_COLUMNS = ("close", "adj close", "adjclose", "kapanis", "kapanış", "kapanis_fiyat",
                 "son", "son_fiyat", "price", "fiyat", "c")


def _pick_column(columns, candidates):
    """First column whose lowercased, stripped name matches one of `candidates`."""
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_table(path):
    """
    Read one delimited text file without demanding a particular dialect.

    Data exported from a Turkish trading platform is routinely semicolon-
    separated with a decimal comma (1.234,56) and day-first dates, none of
    which pandas assumes by default -- so try the plausible combinations
    rather than making the user reformat the file by hand. Any extension is
    accepted (.csv, .txt, .vrd, ...): what matters is whether the CONTENT is
    delimited text. A vendor's binary format has to be exported to text from
    the platform first; this reader will say so rather than guess.
    """
    last_error = None
    for sep in (None, ";", ",", "\t", "|"):
        for decimal in (".", ","):
            try:
                frame = pd.read_csv(path, sep=sep, decimal=decimal, engine="python"
                                    if sep is None else "c", float_precision="round_trip",
                                    skipinitialspace=True)
            except Exception as error:      # unreadable with this dialect -- try the next
                last_error = error
                continue
            if frame.shape[1] < 2:
                continue
            # A decimal-comma file read with decimal="." leaves price columns
            # as strings, so require at least one column that really parsed
            # as a number before accepting this combination.
            if frame.select_dtypes("number").shape[1] >= 1 or decimal == ",":
                return frame
    raise ValueError(f"could not read {path} as delimited text ({last_error}). If it is a "
                     f"platform's own binary format, export it to CSV first.")


def _to_index(values, times=None):
    """Parse a date column (optionally plus a separate time column) into a BIST-local index."""
    text = values.astype(str).str.strip()
    if times is not None:
        text = text + " " + times.astype(str).str.strip()
    # An ISO stamp (2026-08-19 13:30, what --save-prices writes) is
    # unambiguous and parses fastest as-is. Anything else is read day-first,
    # because 03.09.2026 in a Turkish export is 3 September, not 9 March.
    sample = next((s for s in text if s and s.lower() != "nan"), "")
    iso_like = len(sample) >= 10 and sample[:4].isdigit() and sample[4] in "-/"
    index = pd.to_datetime(text, dayfirst=not iso_like, errors="coerce")
    if index.isna().all():
        raise ValueError("no parseable dates in the date column")
    index = pd.DatetimeIndex(index)
    return index.tz_convert(BIST_TZ) if index.tz is not None else index.tz_localize(BIST_TZ)


def _wide_panel(frame):
    """A table that is already dates x tickers: first column dates, the rest prices."""
    date_col = _pick_column(frame.columns, DATE_COLUMNS) or frame.columns[0]
    index = _to_index(frame[date_col])
    panel = frame.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    panel.index = index
    return panel


def _long_panel(frame, date_col, ticker_col, close_col):
    """A table with one row per (bar, ticker) -- pivoted into dates x tickers."""
    time_col = _pick_column([c for c in frame.columns if c != date_col], TIME_COLUMNS)
    index = _to_index(frame[date_col], frame[time_col] if time_col is not None else None)
    tidy = pd.DataFrame({
        "timestamp": index,
        "ticker": frame[ticker_col].astype(str).str.strip(),
        "close": pd.to_numeric(frame[close_col], errors="coerce"),
    }).dropna(subset=["close"])
    # Keep the last quote if a file repeats a (bar, ticker) pair.
    return tidy.pivot_table(index="timestamp", columns="ticker", values="close", aggfunc="last")


def load_price_panel(path, date_col=None, ticker_col=None, close_col=None):
    """
    Build the dates x tickers close-price panel the backtest needs, from
    whatever shape the data is in:

    - a WIDE file: first column dates, one column per ticker (what
      --save-prices writes, and what most "export all symbols" dumps give);
    - a LONG file: one row per bar per ticker, with a ticker column and a
      close/price column (auto-detected, or named with --date-col /
      --ticker-col / --close-col);
    - a DIRECTORY of one file per ticker, each with a date column and a
      close column -- the file name (minus extension) is used as the ticker,
      which is how most platforms export a watchlist.

    Column names are matched in English or Turkish (kapanis, tarih, hisse,
    fiyat, ...); everything else about the dialect is sniffed by read_table.
    """
    path = pathlib.Path(path)

    if path.is_dir():
        panels = {}
        problems = []
        for file in sorted(path.iterdir()):
            if file.is_dir() or file.suffix.lower() in (".xlsx", ".xls", ".pkl", ".zip"):
                continue
            try:
                frame = read_table(file)
                file_date_col = date_col or _pick_column(frame.columns, DATE_COLUMNS) or frame.columns[0]
                file_close_col = close_col or _pick_column(frame.columns, CLOSE_COLUMNS)
                if file_close_col is None:
                    # Single price column beside the date, unlabelled: take it.
                    others = [c for c in frame.columns if c != file_date_col]
                    file_close_col = others[-1] if len(others) == 1 else None
                if file_close_col is None:
                    problems.append(f"{file.name}: no close column in {list(frame.columns)}")
                    continue
                series = pd.Series(pd.to_numeric(frame[file_close_col], errors="coerce").to_numpy(),
                                   index=_to_index(frame[file_date_col]))
                panels[file.stem.upper()] = series[~series.index.duplicated(keep="last")]
            except Exception as error:
                problems.append(f"{file.name}: {error}")
        if not panels:
            raise ValueError(f"no readable price files in {path}. " + "; ".join(problems[:5]))
        if problems:
            print(f"note: skipped {len(problems)} file(s) in {path}: {problems[0]}", file=sys.stderr)
        panel = pd.DataFrame(panels)
    else:
        frame = read_table(path)
        found_date = date_col or _pick_column(frame.columns, DATE_COLUMNS)
        found_ticker = ticker_col or _pick_column(frame.columns, TICKER_COLUMNS)
        found_close = close_col or _pick_column(frame.columns, CLOSE_COLUMNS)
        if found_ticker is not None and found_close is not None:
            panel = _long_panel(frame, found_date or frame.columns[0], found_ticker, found_close)
        else:
            panel = _wide_panel(frame)

    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    panel = panel.dropna(how="all").dropna(axis=1, how="all")
    if panel.empty:
        raise ValueError(f"{path} parsed, but no usable price data came out of it")
    return panel.astype(float)


# Kept under its original name: this is what --prices-csv used to call, and
# a wide CSV still takes the same path through load_price_panel.
load_prices_csv = load_price_panel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone BIST30 cross-sectional momentum backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--period", default=BIST30_PERIOD, help="yfinance history window")
    data.add_argument("--interval", default=BIST30_INTERVAL, help="yfinance bar interval")
    data.add_argument("--prices-csv", "--prices", dest="prices_csv", default=None,
                      help="read prices from this file or folder instead of downloading "
                           "(wide CSV, long CSV, or a folder of one file per ticker)")
    data.add_argument("--date-col", default=None,
                      help="name of the date column, if auto-detection misses it")
    data.add_argument("--ticker-col", default=None,
                      help="name of the ticker column in a long-format file")
    data.add_argument("--close-col", default=None, help="name of the close/price column")
    data.add_argument("--inspect", action="store_true",
                      help="parse the price data, report what was found, and stop without backtesting")
    data.add_argument("--save-prices", default=None,
                      help="save the downloaded panel to this CSV (for a reproducible re-run)")
    data.add_argument("--tickers", default=None,
                      help="comma-separated ticker list (default: the full BIST30)")

    strat = parser.add_argument_group("strategy")
    strat.add_argument("--top-n", type=int, default=DEFAULTS["top_n"],
                       help="names held AT ONCE (the basket rotates, so it trades more names than this)")
    strat.add_argument("--lookback", type=int, default=DEFAULTS["lookback_mom"], help="ROC ranking window, bars")
    strat.add_argument("--ma-fast", type=int, default=DEFAULTS["ma_fast"], help="fast MA, bars")
    strat.add_argument("--ma-slow", type=int, default=DEFAULTS["ma_slow"], help="slow MA, bars")
    strat.add_argument("--rebalance", type=int, default=DEFAULTS["rebalance"], help="bars between re-ranking")
    strat.add_argument("--in-sample-fraction", type=float, default=DEFAULTS["in_sample_fraction"],
                       help="fraction of sessions used as the untraded in-sample window")
    strat.add_argument("--capital", type=float, default=100_000.0, help="starting capital, TRY")

    out = parser.add_argument_group("output")
    out.add_argument("--trades-csv", default=None, help="write the full trade log here")
    out.add_argument("--equity-csv", default=None, help="write the bar-by-bar equity curve here")
    out.add_argument("--max-trades", type=int, default=25, help="trades to print (0 = all)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)

    if args.prices_csv:
        try:
            prices = load_price_panel(args.prices_csv, date_col=args.date_col,
                                      ticker_col=args.ticker_col, close_col=args.close_col)
        except FileNotFoundError:
            print(f"No such price file or folder: {args.prices_csv}\nRun once without --prices-csv "
                  f"and pass --save-prices {args.prices_csv} to create one.", file=sys.stderr)
            return 2
        except ValueError as error:
            print(f"Could not read prices from {args.prices_csv}: {error}\nPoint --date-col / "
                  f"--ticker-col / --close-col at the right columns, or add --inspect to see what "
                  f"was parsed.", file=sys.stderr)
            return 2
        source = f"file {args.prices_csv}"
    else:
        tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
        try:
            prices = download_universe(tickers, period=args.period, interval=args.interval)
        except ImportError:
            print("yfinance is not installed. Either `pip install yfinance` or pass "
                  "--prices-csv with a saved panel.", file=sys.stderr)
            return 2
        source = f"yfinance {args.period} @ {args.interval}"
        if args.save_prices:
            prices.to_csv(args.save_prices)

    bpy = bars_per_year(prices.index)
    n_days = pd.Series(prices.index).dt.date.nunique()
    per_day = len(prices) / max(n_days, 1)

    if args.inspect:
        print(f"parsed {len(prices):,} bars x {prices.shape[1]} tickers from {source}")
        print(f"span   : {prices.index[0]} -> {prices.index[-1]} ({BIST_TZ})")
        print(f"density: {per_day:.1f} bars per trading day, {n_days:,} days, {bpy:,.0f} bars/year")
        print(f"tickers: {', '.join(map(str, prices.columns))}")
        print("\nbars per ticker (fewest first):")
        print(prices.notna().sum().sort_values().to_string())
        print("\nfirst 3 rows:")
        print(prices.head(3).to_string())
        return 0

    if prices.empty or len(prices) < 300:
        print(f"Only {len(prices)} bars came back -- not enough history to backtest. "
              f"Run with --inspect to see what was parsed.", file=sys.stderr)
        return 2

    # The bar-count defaults (lookback 180, rebalance 45) are HOURLY bars.
    # Feeding daily data leaves them running, but 180 daily bars is a
    # nine-month lookback, not a one-month one -- so say so rather than
    # silently backtesting a different strategy than the one on paper.
    using_defaults = (args.lookback == DEFAULTS["lookback_mom"]
                      and args.rebalance == DEFAULTS["rebalance"])
    if per_day < 2 and using_defaults:
        print(f"WARNING: this data has ~{per_day:.1f} bar(s) per trading day, so it looks daily, but the "
              f"default windows (lookback={args.lookback}, rebalance={args.rebalance}) are in HOURLY "
              f"bars. On daily data that is a {args.lookback}-day lookback rebalanced every "
              f"{args.rebalance} days. For the same calendar behaviour as the hourly study, try roughly "
              f"--lookback 20 --ma-fast 5 --ma-slow 20 --rebalance 5.\n", file=sys.stderr)

    params = dict(in_sample_fraction=args.in_sample_fraction, lookback_mom=args.lookback,
                  ma_fast=args.ma_fast, ma_slow=args.ma_slow, rebalance=args.rebalance,
                  top_n=args.top_n)

    result = run_full_backtest_momentum(prices, params=params, bars_per_year_value=bpy)
    result = add_capital_pnl(result, args.capital)
    attach_trade_amounts(result, prices)

    split_time = prices.index[result["split_idx"]]
    print("=" * 96)
    print("BIST30 cross-sectional momentum -- standalone backtest")
    print("=" * 96)
    print(f"data          : {source}")
    print(f"panel         : {len(prices):,} bars x {prices.shape[1]} tickers, "
          f"{prices.index[0]:%Y-%m-%d %H:%M} -> {prices.index[-1]:%Y-%m-%d %H:%M} ({BIST_TZ})")
    print(f"bar density   : {bpy:,.0f} bars/year, measured from this panel (not assumed)")
    print(f"in-sample     : up to {split_time:%Y-%m-%d} ({args.in_sample_fraction:.0%} of sessions, NOT traded)")
    print(f"out-of-sample : {split_time:%Y-%m-%d} onward, {len(prices) - result['split_idx']:,} bars -- "
          f"every number below is from this window only")
    print(f"parameters    : top_n={args.top_n} (names held at once), lookback={args.lookback}, "
          f"ma_fast={args.ma_fast}, ma_slow={args.ma_slow}, rebalance={args.rebalance} bars")

    print("\n--- Result (no trading costs) ---")
    print(f"starting capital : {result['starting_capital']:>14,.0f} TRY")
    print(f"final capital    : {result['final_capital']:>14,.0f} TRY")
    print(f"total return     : {result['total_return_pct']:>14.2f} %")
    for key, value in result["stats"].items():
        print(f"{key:<17}: {value:>14}")
    print(f"{'trades':<17}: {result['n_trades']:>14}")
    print(f"{'avg names held':<17}: {result['avg_n_held']:>14}")
    print(f"{'amount per name':<17}: {result['dollar_per_unit'] / max(args.top_n, 1):>14,.0f} TRY "
          f"(= {result['dollar_per_unit']:,.0f} base unit / {args.top_n} names)")

    print("\n--- Cost sensitivity (the headline above is the 0 bps row) ---")
    print(result["cost_table"].to_string())

    per_ticker = per_ticker_frame(result)
    print(f"\n--- Per stock ({len(per_ticker)} traded) ---")
    print(per_ticker.to_string(index=False))

    frame = trades_frame(result)
    shown = frame if args.max_trades <= 0 else frame.head(args.max_trades)
    print(f"\n--- Trade log ({len(frame)} trades"
          f"{f', first {len(shown)} shown' if len(shown) < len(frame) else ''}) ---")
    print(shown.to_string(index=False))

    if args.trades_csv:
        frame.to_csv(args.trades_csv, index=False)
        print(f"\nwrote {args.trades_csv}")
    if args.equity_csv:
        pd.DataFrame({"timestamp": prices.index, "equity_try": result["equity_curve"]}).to_csv(
            args.equity_csv, index=False)
        print(f"wrote {args.equity_csv}")

    print("\nReminders: no transaction costs in the headline (see the table), sizing is off the STARTING "
          "capital so it never compounds, and yfinance's hourly window rolls -- save the panel with "
          "--save-prices if this run needs to be reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
