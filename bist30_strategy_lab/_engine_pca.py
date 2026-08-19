"""
PCA-based basket statistical arbitrage across the whole BIST30 universe at
once -- addresses the "small ad hoc pair-candidate set" problem the two
prior missions' pairs trading had directly, by not picking a pair at all:
every ticker's return is decomposed into a common-factor part (explained by
the top N principal components of the universe's return covariance) and an
idiosyncratic residual part, and the residual -- not the raw price -- is
what's traded as mean-reverting.

This is a from-scratch numpy eigendecomposition (np.linalg.eigh on the
covariance matrix), not scikit-learn's PCA -- keeps this mission dependency-
free the same way GARCH/ADF/Hurst were hand-rolled in prior missions.

Positions are long/short (a name can be shorted against the basket, same as
the pairs engine already shorts one leg) but always market/factor-neutral
BY CONSTRUCTION -- the traded quantity is a residual return that's already
had the common factors removed. Sizing: equal weight across currently-open
residual positions, capped so aggregate exposure never exceeds 100% of
capital (scale_max=1.0, same reasoning as _engine_momentum.py's basket
sizing) -- no leverage.
"""

import numpy as np
import pandas as pd

import _common as c

DEFAULTS = dict(
    in_sample_fraction=0.6,
    fit_window=500,      # bars of trailing returns used to fit PCA loadings each refit
    rebalance=45,         # bars between refits
    n_factors=3,
    z_window=90,          # bars, rolling window for the residual z-score
    entry_z=1.5,
    stop_z=3.5,
    min_fit_names=8,      # need at least this many tickers with full history in the fit window
)


def compute_pca_factors(returns_window, n_factors):
    """
    returns_window: T x K numpy array, no NaN. Returns (loadings: K x n_factors,
    eigenvalues: n_factors,) -- the top n_factors principal components of the
    return covariance matrix, sorted by explained variance descending.
    """
    R = returns_window - returns_window.mean(axis=0, keepdims=True)
    cov = np.cov(R, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_factors = min(n_factors, eigvecs.shape[1])
    return eigvecs[:, :n_factors], eigvals[:n_factors]


def residualize(returns_row, loadings):
    """
    returns_row: K-vector, this bar's returns for the K tickers the loadings
    were fit on. Returns the K-vector of residuals after projecting out the
    factor-explained part: explained = loadings @ (loadings.T @ returns_row).
    """
    factor_scores = loadings.T @ returns_row
    explained = loadings @ factor_scores
    return returns_row - explained


def walk_forward_pca(prices, params):
    log_ret = np.log(prices).diff()
    n = len(prices)
    split_idx, _ = c.split_by_session(prices.index, params["in_sample_fraction"])

    fit_window, rebalance = params["fit_window"], params["rebalance"]
    n_factors, z_window = params["n_factors"], params["z_window"]
    entry_z, stop_z = params["entry_z"], params["stop_z"]
    min_fit_names = params["min_fit_names"]

    pnl = np.zeros(n)
    n_active_path = np.zeros(n, dtype=int)
    trades = []
    position = {}        # ticker -> +1.0 / -1.0
    entry_info = {}       # ticker -> (entry_t, entry_z)
    cum_resid = {}
    resid_buffer = {}     # ticker -> list, bounded to z_window
    # per-bar, per-ticker currency-contribution-equivalent (position * residual
    # * weight), so each closed trade's own 'pnl' can be attributed after the
    # fact -- same shape/contract as _engine_momentum.py's `contrib` matrix,
    # needed so _common.add_capital_pnl (which expects every trade dict to
    # carry a 'pnl' key) works unmodified across all three engines.
    contrib = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    loadings, fit_tickers = None, []
    last_rebal_t = None

    for t in range(split_idx, n):
        do_refit = (last_rebal_t is None) or (t - last_rebal_t >= rebalance)
        if do_refit:
            last_rebal_t = t
            window = log_ret.iloc[max(0, t - fit_window):t]
            valid_cols = window.columns[window.notna().all()]
            if len(valid_cols) >= max(n_factors + 1, min_fit_names):
                fit_tickers = list(valid_cols)
                loadings, _ = compute_pca_factors(window[fit_tickers].to_numpy(), n_factors)
                for tk in fit_tickers:
                    cum_resid.setdefault(tk, 0.0)
                    resid_buffer.setdefault(tk, [])
            else:
                fit_tickers = []

        if fit_tickers:
            r_t = log_ret.iloc[t][fit_tickers]
            if r_t.notna().all():
                resid_t = residualize(r_t.to_numpy(), loadings)

                # Pass 1: update residual state, compute this bar's z-score per ticker.
                z_by_ticker = {}
                for i, tk in enumerate(fit_tickers):
                    cum_resid[tk] += resid_t[i]
                    buf = resid_buffer[tk]
                    buf.append(cum_resid[tk])
                    if len(buf) > z_window:
                        buf.pop(0)
                    if len(buf) >= 10:
                        arr = np.array(buf)
                        mu, sd = arr.mean(), arr.std()
                        z_by_ticker[tk] = (cum_resid[tk] - mu) / sd if sd > 0 else 0.0
                    else:
                        z_by_ticker[tk] = 0.0

                # Pass 2: pnl for ALREADY-open positions, weighted by the
                # number of positions open at the START of this bar -- fixed
                # once per bar so the result doesn't depend on ticker
                # iteration order (unlike computing it inline per ticker).
                weight = 1.0 / max(len(position), 1)
                for i, tk in enumerate(fit_tickers):
                    if tk in position:
                        bar_contrib = position[tk] * resid_t[i] * weight
                        pnl[t] += bar_contrib
                        contrib.at[prices.index[t], tk] = bar_contrib

                # Pass 3: entries/exits based on this bar's z-score.
                for tk in fit_tickers:
                    z_t = z_by_ticker[tk]
                    if tk not in position:
                        if z_t > entry_z:
                            position[tk] = -1.0
                            entry_info[tk] = (t, z_t)
                        elif z_t < -entry_z:
                            position[tk] = 1.0
                            entry_info[tk] = (t, z_t)
                    else:
                        pos = position[tk]
                        reason = None
                        if (pos == 1.0 and z_t >= 0) or (pos == -1.0 and z_t <= 0):
                            reason = "reversion"
                        elif abs(z_t) > stop_z:
                            reason = "hard stop"
                        if reason:
                            entry_t, entry_zv = entry_info.pop(tk)
                            trades.append({
                                "ticker": tk, "entry_time": prices.index[entry_t],
                                "exit_time": prices.index[t], "entry_z": entry_zv,
                                "exit_z": z_t, "hold_bars": t - entry_t, "exit_reason": reason,
                                "_entry_t": entry_t, "_exit_t": t,
                            })
                            del position[tk]

        n_active_path[t] = len(position)

    for tk, (entry_t, entry_zv) in entry_info.items():
        trades.append({
            "ticker": tk, "entry_time": prices.index[entry_t], "exit_time": prices.index[n - 1],
            "entry_z": entry_zv, "exit_z": float("nan"), "hold_bars": n - 1 - entry_t,
            "exit_reason": "end of backtest", "_entry_t": entry_t, "_exit_t": n - 1,
        })

    # Per-trade pnl attribution: sum this ticker's bar-by-bar contribution
    # (see the `contrib` matrix above) over exactly its own holding period --
    # same technique as _engine_momentum.py's trade-pnl attribution.
    for tr in trades:
        seg = contrib[tr["ticker"]].iloc[tr["_entry_t"] + 1: tr["_exit_t"] + 1]
        tr["pnl"] = float(seg.sum())
        del tr["_entry_t"]
        del tr["_exit_t"]

    return pnl, n_active_path, trades, split_idx


def run_full_backtest_pca(prices, params=None, bars_per_year=None):
    """
    One call: refit PCA -> residualize -> trade -> stats. Same `results`/
    `trades` contract as the other two engines: `results` has a `pnl`
    column, and every trade dict carries its own attributed `pnl` (summed
    from the per-bar contribution matrix over exactly its holding period --
    see walk_forward_pca), so _common.add_capital_pnl works unmodified.
    scale_max=1.0, same no-leverage reasoning as _engine_momentum.py:
    equal-weighted residual positions never sum to more than 100% of
    capital in aggregate.
    """
    params = {**DEFAULTS, **(params or {})}
    if bars_per_year is None:
        from universe import bars_per_year as _bpy
        bars_per_year = _bpy(prices.index)

    pnl, n_active_path, trades, split_idx = walk_forward_pca(prices, params)
    stats = c.performance_stats(pnl, split_idx, bars_per_year)
    position_proxy = n_active_path.astype(float)
    cost_table = c.cost_sensitivity_table(pnl, position_proxy, split_idx, bars_per_year)

    results = pd.DataFrame({"timestamp": prices.index, "n_active": n_active_path, "pnl": pnl})

    return dict(
        results=results, stats=stats, cost_table=cost_table, trades=trades,
        split_idx=split_idx, n_trades=len(trades), scale_max=1.0, bars_per_year=bars_per_year,
        avg_n_active=round(float(n_active_path[split_idx:].mean()), 2),
    )
