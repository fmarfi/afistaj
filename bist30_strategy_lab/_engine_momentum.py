"""
Cross-sectional momentum / trend-following across the BIST30 basket --
genuinely new to this repo, not a variant of the pairs/mean-reversion idea
tried (and found wanting) twice already this session. Long-only, equal-
capital top-N rotation: no shorting, consistent with the no-leverage
philosophy and avoiding BIST margin/short-sale mechanics.

At each rebalance, rank all tickers with enough trailing history by their
rate-of-change (ROC) over a lookback window, keep only those a simple trend
filter (fast MA above slow MA) confirms are actually trending rather than
just noisy, and hold the top N equally weighted until the next rebalance.

Sizing note: unlike the mean-reversion engine's +-1-unit spread positions,
"pnl" here is directly the basket's fractional bar-to-bar return (e.g.
0.002 = 0.2%), and this engine is never more than 100% invested (no
leverage, nothing borrowed) -- so it declares scale_max=1.0, which makes
_common.add_capital_pnl's dollar_pnl = pnl * (capital / 1.0) = pnl * capital,
exactly the right currency conversion for a fractional-return series,
without needing its own bespoke capital-sizing function.
"""

import numpy as np
import pandas as pd

import _common as c

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
    their listing/rename date -- see universe.SHORT_HISTORY_TICKERS) simply
    can't be ranked (ROC/MA are NaN) until it has enough real history.
    """
    log_ret = np.log(prices).diff()
    roc = compute_roc(prices, params["lookback_mom"])
    trend = trend_filter(prices, params["ma_fast"], params["ma_slow"])
    n = len(prices)
    tickers = list(prices.columns)
    split_idx, _ = c.split_by_session(prices.index, params["in_sample_fraction"])

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


def cost_sensitivity_table_momentum(pnl, turnover_path, split_idx, bars_per_year, cost_bps_grid=(0, 1, 2, 5, 10, 20)):
    """
    Basket-turnover version of _common.cost_sensitivity_table: turnover_path
    is already a fraction-of-portfolio-turned-over series (see
    walk_forward_momentum), not a +-1 position diff, so this applies cost
    directly rather than going through _common's position-diff derivation.
    """
    rows = []
    for cost_bps in cost_bps_grid:
        cost = turnover_path * (cost_bps / 10000)
        pnl_after_cost = pnl - cost
        pnl_after_cost[split_idx] = 0.0
        stats = c.performance_stats(pnl_after_cost, split_idx, bars_per_year)
        stats["cost (bps/switch)"] = cost_bps
        rows.append(stats)
    return pd.DataFrame(rows).set_index("cost (bps/switch)")


def run_full_backtest_momentum(prices, params=None, bars_per_year=None):
    """
    One call: rank -> rotate -> trade log -> stats. Same `results`/`trades`
    contract as the other two engines (results has a `pnl` column, trades
    are pnl-keyed dicts) so _common.add_capital_pnl works unmodified.
    scale_max=1.0 declares "never more than fully invested" -- see module
    docstring for why that makes dollar sizing come out correct for a
    fractional-return pnl series.
    """
    params = {**DEFAULTS, **(params or {})}
    if bars_per_year is None:
        from universe import bars_per_year as _bpy
        bars_per_year = _bpy(prices.index)

    pnl, n_held_path, turnover_path, trades, split_idx = walk_forward_momentum(prices, params)
    stats = c.performance_stats(pnl, split_idx, bars_per_year)
    cost_table = cost_sensitivity_table_momentum(pnl, turnover_path, split_idx, bars_per_year)

    results = pd.DataFrame({"timestamp": prices.index, "n_held": n_held_path, "pnl": pnl})

    return dict(
        results=results, stats=stats, cost_table=cost_table, trades=trades,
        split_idx=split_idx, n_trades=len(trades), scale_max=1.0, bars_per_year=bars_per_year,
        avg_n_held=round(float(n_held_path[split_idx:].mean()), 2),
    )
