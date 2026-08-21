"""
The momentum engine, with the selection rule swappable.

_engine_momentum.py hard-codes one rule: rank by ROC, keep what the fast/slow
MA filter confirms. The upgrade bench (momentum_upgrades.py) tests ten other
rules -- DMI direction, ADX strength, Hurst persistence, volatility-adjusted
ranking -- but only ever as pnl paths, because that is all a Monte Carlo
needs. This module runs any of those rules through a FULL walk-forward, with
a trade log, so a variant can be explored bar by bar in the dashboard exactly
like the built-in families.

Two rules it follows so a variant can be compared to the original honestly:

  1. The selection rule is imported from momentum_upgrades.make_selector,
     not reimplemented. The dashboard and the bench therefore cannot drift
     into testing two slightly different versions of "the same" variant.

  2. Everything else -- the session-boundary split, rebalancing every N bars
     off the previous bar's data, equal weighting, per-trade attribution from
     the contribution matrix -- is the same code path _engine_momentum uses.
     `variant="baseline"` reproduces run_full_backtest_momentum's pnl exactly
     (asserted by test_engine_momentum_variants.py), so any difference you
     see between a variant and the original is the rule, not the plumbing.

Needs OHLC, not just closes: DMI/ADX are built from highs and lows.
"""

import numpy as np
import pandas as pd

import _common as c
import _engine_momentum as mo
import momentum_monte_carlo as mc
import momentum_upgrades as up

# Same knobs as _engine_momentum plus the ones the new rules need. Bar counts
# throughout, matching the dashboard's other momentum fields.
DEFAULTS = dict(
    mo.DEFAULTS,
    variant="baseline",
    di_window=45,       # DMI/ADX lookback, bars (~5 sessions hourly)
    adx_min=20.0,       # trend-strength floor for the ADX rules
    hurst_window=250,   # trailing window for the Hurst exponent, bars
    hurst_min=0.5,      # H > 0.5 = the name has been trending, not reverting
)

VARIANT_KINDS = {name: kind for name, kind, _ in up.VARIANTS}
VARIANT_LABELS = {name: label for name, _, label in up.VARIANTS}

# Hurst is the one genuinely slow signal (a rolling regression per name), and
# the dashboard re-runs on every parameter change. Cache it per (window, panel
# identity) so only a change that actually affects it pays the cost.
_hurst_cache = {}


def _hurst_matrix(close, window):
    key = (window, len(close), close.index[-1], close.shape[1])
    if key not in _hurst_cache:
        if len(_hurst_cache) > 4:
            _hurst_cache.clear()
        _hurst_cache[key] = up.rolling_hurst(close, window=window).to_numpy()
    return _hurst_cache[key]


def build_context(panel, params):
    """Signals for every rule, on the shared bar timeline."""
    close = panel["Close"]
    prepared = mc.prepare(close, params)
    needs_hurst = params["variant"] in up.HURST_VARIANTS
    extra = up.build_extra(panel, close, params, params["di_window"], params["ma_fast"],
                           _hurst_matrix(close, params["hurst_window"]) if needs_hurst else None)
    selector = up.make_selector(VARIANT_KINDS[params["variant"]], extra,
                                {"adx_min": params["adx_min"], "hurst_min": params["hurst_min"]})
    return prepared, selector


def walk_forward_variant(panel, params, selector, prepared):
    """
    _engine_momentum.walk_forward_momentum with the ranking call replaced.

    Deliberately kept line-for-line recognisable against the original rather
    than refactored into something shared: the two have to stay comparable,
    and a clever abstraction over both would make it hard to see that the
    only difference is which names the rule returns.
    """
    close = panel["Close"]
    log_ret = np.log(close).diff()
    n = len(close)
    tickers = list(close.columns)
    split_idx, _ = c.split_by_session(close.index, params["in_sample_fraction"])

    pnl = np.zeros(n)
    n_held_path = np.zeros(n, dtype=int)
    turnover_path = np.zeros(n)
    contrib = pd.DataFrame(0.0, index=close.index, columns=tickers)

    current_holdings = {}
    trades = []
    rebalance = params["rebalance"]
    top_n = params["top_n"]
    last_rebal_t = None

    for t in range(split_idx, n):
        do_rebal = (last_rebal_t is None) or (t - last_rebal_t >= rebalance)
        if do_rebal:
            last_rebal_t = t
            # t-1, never t: the bar being traded on has not closed yet when
            # the decision is made.
            new_selected = selector(prepared, t - 1, top_n, None)

            dropped = [tk for tk in current_holdings if tk not in new_selected]
            added = [tk for tk in new_selected if tk not in current_holdings]
            turnover_path[t] = (len(dropped) + len(added)) / max(top_n, 1)

            for tk in dropped:
                entry_t = current_holdings.pop(tk)
                trades.append({"ticker": tk, "entry_time": close.index[entry_t],
                               "exit_time": close.index[t], "exit_reason": "rebalance drop",
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
        trades.append({"ticker": tk, "entry_time": close.index[entry_t],
                       "exit_time": close.index[n - 1], "exit_reason": "end of backtest",
                       "_entry_t": entry_t, "_exit_t": n - 1})

    for tr in trades:
        seg = contrib[tr["ticker"]].iloc[tr["_entry_t"] + 1: tr["_exit_t"] + 1]
        tr["pnl"] = float(seg.sum())
        tr["hold_bars"] = tr["_exit_t"] - tr["_entry_t"]
        del tr["_entry_t"]
        del tr["_exit_t"]

    return pnl, n_held_path, turnover_path, trades, split_idx


def run_full_backtest_variant(panel, params=None, bars_per_year=None):
    """
    Same result contract as the three built-in engines -- a `results` frame
    with a pnl column, pnl-keyed trade dicts, scale_max -- so _common.
    add_capital_pnl, the trade cards and the per-stock charts all work on a
    variant without a special case anywhere.
    """
    params = {**DEFAULTS, **(params or {})}
    if params["variant"] not in VARIANT_KINDS:
        raise ValueError("unknown variant %r (known: %s)"
                         % (params["variant"], ", ".join(VARIANT_KINDS)))

    close = panel["Close"]
    if bars_per_year is None:
        from universe import bars_per_year as _bpy
        bars_per_year = _bpy(close.index)

    prepared, selector = build_context(panel, params)
    pnl, n_held_path, turnover_path, trades, split_idx = walk_forward_variant(
        panel, params, selector, prepared)

    stats = c.performance_stats(pnl, split_idx, bars_per_year)
    cost_table = mo.cost_sensitivity_table_momentum(pnl, turnover_path, split_idx, bars_per_year)
    results = pd.DataFrame({"timestamp": close.index, "n_held": n_held_path, "pnl": pnl})

    return dict(
        results=results, stats=stats, cost_table=cost_table, trades=trades,
        split_idx=split_idx, n_trades=len(trades), scale_max=1.0, bars_per_year=bars_per_year,
        avg_n_held=round(float(n_held_path[split_idx:].mean()), 2),
        variant=params["variant"], variant_label=VARIANT_LABELS[params["variant"]],
        pct_idle=round(float((n_held_path[split_idx:] == 0).mean() * 100), 1),
    )
