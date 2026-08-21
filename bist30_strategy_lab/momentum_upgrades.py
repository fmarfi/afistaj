"""
Candidate upgrades to the BIST30 momentum strategy, each measured the same
way: not by a single backtest, but by the whole rebalance-phase distribution
against the buy-and-hold benchmark it has to beat.

WHY THIS FILE EXISTS. momentum_monte_carlo.py showed the plain strategy's
46.3% headline was an 84th-percentile draw and that its phase-neutral result
(28.5%, Sharpe 1.01) does not beat equal-weight buy-and-hold (29.0%, Sharpe
1.25). So an "upgrade" here means one thing only: does the variant clear
that benchmark across the phase distribution? A variant that merely raises
the single headline has not been shown to do anything.

WHAT IS TESTED

  1. DIRECTIONAL MOVEMENT (+DI / -DI / ADX, Wilder). The idea that these are
     dependable trend signals is a hypothesis, so it is tested three ways --
     as a gate (+DI > -DI), as a strength filter (ADX above a threshold),
     and as the ranking score itself -- at several window lengths, because
     "does DI work" has no answer independent of the window.

     NOTE: DMI needs HIGH and LOW. The mission's own panel is close-only
     (universe.download_universe takes ["Close"]), so this module downloads
     full OHLC itself. Nothing that ran before is affected.

  2. STATISTICAL STATE. Two per-stock measures, both from the mission's
     existing machinery rather than new inventions:
       - Hurst exponent on trailing log prices (_common.hurst_on_levels):
         is this name currently trending (H > 0.5) or mean-reverting?
         Momentum bought into a mean-reverting name is a bet against the
         name's own measured behaviour.
       - realized volatility, used for risk-adjusted ranking (ROC / vol),
         which is the standard answer to "momentum ranks pick up the most
         volatile names rather than the strongest trends".

     "Predicted state" is deliberately read as MEASURED PERSISTENCE, not as
     a price forecast. Hurst says how strongly this series has been
     continuing its moves; that is a defensible statistical statement about
     the next move. A model that claims to predict the next return on two
     years of hourly bars would not be.

Run:
    python momentum_upgrades.py                      # download OHLC, test everything
    python momentum_upgrades.py --ohlc panel.pkl     # reuse a cached OHLC panel
    python momentum_upgrades.py --skip-hurst         # faster, drops the Hurst variants
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd

import _common as c
import _engine_momentum as mo
import momentum_monte_carlo as mc
import universe as uni


# --------------------------------------------------------------------------
# Data: OHLC, which the close-only panel cannot provide
# --------------------------------------------------------------------------

def download_ohlc(tickers=None, period=uni.BIST30_PERIOD, interval=uni.BIST30_INTERVAL):
    """Open/High/Low/Close panel -- same request universe.download_universe makes, keeping all four fields."""
    import yfinance as yf
    if tickers is None:
        tickers = uni.BIST30_TICKERS
    raw = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
    panel = {}
    for field in ("Open", "High", "Low", "Close"):
        frame = raw[field].dropna(how="all")
        frame.index = frame.index.tz_convert("Europe/Istanbul")
        panel[field] = frame
    return panel


def load_ohlc(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------
# Wilder's directional movement -- the user's +-DI idea
# --------------------------------------------------------------------------

def wilder_dmi(high, low, close, window):
    """
    +DI, -DI and ADX, Wilder's smoothing, computed per column across the
    whole panel at once. Same formulation as dashboard/bist30_webapp.py's
    compute_dmi (ewm with alpha = 1/window), so the numbers agree with what
    the chart in this repo already draws.

    Reading, for the ranking rules below:
      +DI > -DI   : upward directional movement dominates
      ADX         : how STRONG the trend is, regardless of direction
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    previous_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(),
                            (high - previous_close).abs(),
                            (low - previous_close).abs()]).groupby(level=0).max()
    true_range = true_range.reindex(close.index)

    alpha = 1.0 / window
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return plus_di, minus_di, adx


# --------------------------------------------------------------------------
# Statistical state of each name
# --------------------------------------------------------------------------

def rolling_hurst(close, window=250, step=9):
    """
    Trailing Hurst exponent per name -- H > 0.5 means this series has been
    CONTINUING its moves (a trend the momentum rank can ride), H < 0.5 means
    it has been reverting (a trend the rank will keep buying at the top of).

    Uses _common.hurst_on_levels, the same estimator the rest of the mission
    uses. Recomputed every `step` bars and held constant in between: it is a
    slow-moving regime statistic, and computing it on every bar would cost
    thirty times more for a number that barely moves between sessions.
    """
    values = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    log_prices = np.log(close)
    for ticker in close.columns:
        series = log_prices[ticker].to_numpy()
        for end in range(window, len(series) + 1, step):
            segment = series[end - window:end]
            if np.isnan(segment).any():
                continue
            try:
                values.iloc[end - 1, values.columns.get_loc(ticker)] = c.hurst_on_levels(segment)
            except (ValueError, np.linalg.LinAlgError):
                continue
    return values.ffill()


def realized_volatility(close, window):
    """Trailing standard deviation of log returns -- the denominator for risk-adjusted momentum."""
    return np.log(close).diff().rolling(window).std()


# --------------------------------------------------------------------------
# Selectors: each is one candidate upgrade
# --------------------------------------------------------------------------

def make_selector(kind, extra, thresholds):
    """
    Build a selector for mc.run. Every variant keeps the SAME universe, the
    same basket size and the same rebalance clock -- only the rule that
    picks the names changes, so any difference in the result is attributable
    to the rule and not to a second thing that moved at the same time.
    """
    adx_min = thresholds.get("adx_min", 20.0)
    hurst_min = thresholds.get("hurst_min", 0.5)

    def eligible_mask(prepared, bar):
        """Baseline eligibility: finite ROC and the fast MA above the slow MA."""
        finite = np.isfinite(prepared["roc_values"][bar])
        return finite & prepared["trend_values"][bar]

    def select(prepared, bar, top_n, rng):
        roc = prepared["roc_values"][bar]
        finite = np.isfinite(roc)
        mask = eligible_mask(prepared, bar)
        score = roc.copy()

        if kind == "baseline":
            pass
        elif kind == "di_gate":
            # Trend filter replaced by Wilder's direction test.
            mask = finite & (extra["plus_di"][bar] > extra["minus_di"][bar])
        elif kind == "ma_and_di":
            mask = mask & (extra["plus_di"][bar] > extra["minus_di"][bar])
        elif kind == "ma_and_adx":
            mask = mask & (extra["adx"][bar] >= adx_min)
        elif kind == "ma_di_adx":
            mask = (mask & (extra["plus_di"][bar] > extra["minus_di"][bar])
                    & (extra["adx"][bar] >= adx_min))
        elif kind == "rank_by_di":
            # DI as the ranking score itself, not just a gate.
            mask = mask & (extra["plus_di"][bar] > extra["minus_di"][bar])
            score = extra["plus_di"][bar] - extra["minus_di"][bar]
        elif kind == "rank_by_adx_weighted":
            mask = mask & (extra["adx"][bar] >= adx_min)
            score = roc * (extra["adx"][bar] / 100.0)
        elif kind == "risk_adjusted":
            volatility = extra["vol"][bar]
            score = np.where(volatility > 0, roc / volatility, np.nan)
        elif kind == "ma_and_hurst":
            mask = mask & (extra["hurst"][bar] >= hurst_min)
        elif kind == "hurst_and_di":
            mask = (finite & (extra["hurst"][bar] >= hurst_min)
                    & (extra["plus_di"][bar] > extra["minus_di"][bar]))
        elif kind == "risk_adjusted_di":
            volatility = extra["vol"][bar]
            mask = mask & (extra["plus_di"][bar] > extra["minus_di"][bar])
            score = np.where(volatility > 0, roc / volatility, np.nan)
        else:
            raise ValueError("unknown selector: " + kind)

        mask = mask & np.isfinite(score)
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            return []
        # Stable descending sort, so ties resolve by universe order exactly
        # as _engine_momentum.rank_universe resolves them.
        order = candidates[np.argsort(-score[candidates], kind="stable")]
        return [prepared["tickers"][i] for i in order[:top_n]]

    return select


# --------------------------------------------------------------------------
# Evaluation: the phase distribution, not one backtest
# --------------------------------------------------------------------------

# Every candidate, by name. `--variant <name>` backtests one on its own;
# with no name they are all compared side by side. One registry so the
# comparison table and the standalone backtest can never describe different
# strategies.
VARIANTS = [
    ("baseline",             "baseline",             "ROC + MA filter (current strategy)"),
    ("di",                   "di_gate",              "+DI > -DI instead of the MA filter"),
    ("ma_di",                "ma_and_di",            "MA filter AND +DI > -DI"),
    ("ma_adx",               "ma_and_adx",           "MA filter AND ADX >= threshold"),
    ("ma_di_adx",            "ma_di_adx",            "MA + DI + ADX (full DMI gate)"),
    ("rank_di",              "rank_by_di",           "rank by (+DI - -DI) instead of ROC"),
    ("rank_adx",             "rank_by_adx_weighted", "rank by ROC x ADX strength"),
    ("riskadj",              "risk_adjusted",        "rank by ROC / volatility"),
    ("riskadj_di",           "risk_adjusted_di",     "ROC / volatility, DI-gated"),
    ("ma_hurst",             "ma_and_hurst",         "MA filter AND Hurst >= 0.5"),
    ("hurst_di",             "hurst_and_di",         "Hurst >= 0.5 AND +DI > -DI"),
]
HURST_VARIANTS = {"ma_hurst", "hurst_di"}
# Which knobs each variant actually depends on -- used to re-test it around
# its own settings instead of at one cherry-picked point.
ADX_VARIANTS = {"ma_adx", "ma_di_adx", "rank_adx"}
DI_VARIANTS = {"di", "ma_di", "ma_di_adx", "rank_di", "riskadj_di", "hurst_di"} | ADX_VARIANTS


def evaluate(prepared, split_idx, bars_per_year, selector, label, cost_bps=5):
    """
    One variant, judged over every rebalance phase plus the staggered
    all-tranche run (the phase-luck-free number).
    """
    params = prepared["params"]
    runs = [mc.run(prepared, split_idx + offset, selector=selector)
            for offset in range(params["rebalance"])]
    stats = [mc.stats_for(r, bars_per_year) for r in runs]
    returns = np.array([s["annualized return"] for s in stats])
    sharpes = np.array([s["sharpe (naive)"] for s in stats])

    blended = dict(pnl=np.mean([r["pnl"] for r in runs], axis=0),
                   turnover=np.mean([r["turnover"] for r in runs], axis=0),
                   start_bar=split_idx + params["rebalance"])
    tranched = mc.stats_for(blended, bars_per_year)
    costed = mc.stats_for(mc.apply_cost(blended, cost_bps), bars_per_year)

    # How concentrated was it, really? A filter that leaves one name standing
    # turns the "strategy" into a single position, and its return says more
    # about that name than about the rule.
    traded = runs[0]["n_held"][split_idx:]
    held = traded[traded > 0]
    years = max(1e-9, (len(prepared["log_returns"]) - split_idx) / bars_per_year)

    return {
        "variant": label,
        "median return": float(np.median(returns)),
        "median sharpe": float(np.median(sharpes)),
        "worst phase": float(np.min(returns)),
        "best phase": float(np.max(returns)),
        "tranched return": tranched["annualized return"],
        "tranched sharpe": tranched["sharpe (naive)"],
        "after cost return": costed["annualized return"],
        "after cost sharpe": costed["sharpe (naive)"],
        "avg held": float(held.mean()) if len(held) else 0.0,
        "bars idle %": float((traded == 0).mean() * 100),
        "turnover/yr": float(np.sum(runs[0]["turnover"]) / years),
    }


def build_extra(panel, close, params, di_window, vol_window, hurst=None):
    """Indicator matrices for one DMI window -- rebuilt when the window changes."""
    plus_di, minus_di, adx = wilder_dmi(panel["High"], panel["Low"], close, di_window)
    extra = {"plus_di": plus_di.to_numpy(), "minus_di": minus_di.to_numpy(),
             "adx": adx.to_numpy(),
             "vol": realized_volatility(close, vol_window).to_numpy()}
    if hurst is not None:
        extra["hurst"] = hurst
    return extra


def neighborhood_stability(panel, close, prepared, split_idx, bars_per_year, name, kind,
                           base_di_window, base_adx_min, hurst=None):
    """
    The same variant re-measured around its own settings, not at one point.

    A rule that works only at ADX >= 30 with a 45-bar window, and inverts at
    a 126-bar window, has not found a market property -- it has found a cell
    in a table. Reporting the WORST and MEDIAN of the neighbourhood makes
    that visible before anyone trades it. Variants with no knobs return their
    single value for all three columns, which is the honest answer for them.
    """
    di_windows = (14, 45, 126) if name in DI_VARIANTS else (base_di_window,)
    adx_levels = (15.0, 20.0, 25.0, 30.0) if name in ADX_VARIANTS else (base_adx_min,)
    # The Hurst variants have a knob too, and a neighbourhood of ONE would
    # print their single best cell as if it were stable. The Hurst WINDOW is
    # not swept here (it would mean recomputing the estimator per setting);
    # sweep it with --hurst-window if a variant gets this far.
    hurst_levels = (0.45, 0.50, 0.55, 0.60) if name in HURST_VARIANTS else (0.5,)

    sharpes = []
    for di_window in di_windows:
        extra = build_extra(panel, close, prepared["params"], di_window,
                            prepared["params"]["ma_fast"], hurst)
        for adx_min, hurst_min in [(a, h) for a in adx_levels for h in hurst_levels]:
            selector = make_selector(kind, extra, {"adx_min": adx_min, "hurst_min": hurst_min})
            runs = [mc.run(prepared, split_idx + offset, selector=selector)
                    for offset in range(prepared["params"]["rebalance"])]
            blended = dict(pnl=np.mean([r["pnl"] for r in runs], axis=0),
                           turnover=np.mean([r["turnover"] for r in runs], axis=0),
                           start_bar=split_idx + prepared["params"]["rebalance"])
            sharpes.append(mc.stats_for(blended, bars_per_year)["sharpe (naive)"])

    sharpes = np.array([v for v in sharpes if np.isfinite(v)])
    if not len(sharpes):
        return {"nbhd median sharpe": np.nan, "nbhd worst sharpe": np.nan, "nbhd settings": 0}
    return {"nbhd median sharpe": float(np.median(sharpes)),
            "nbhd worst sharpe": float(np.min(sharpes)),
            "nbhd settings": int(len(sharpes))}


def backtest_one(panel, close, prepared, split_idx, bars_per_year, name, kind, label,
                 extra, args):
    """
    One variant on its own, in depth: the phase distribution, the phase-neutral
    tranched run, costs, concentration, the benchmark, and whether its ranking
    beats a random draw from the same eligible set.
    """
    params = prepared["params"]
    selector = make_selector(kind, extra, {"adx_min": args.adx_min, "hurst_min": 0.5})
    runs = [mc.run(prepared, split_idx + offset, selector=selector)
            for offset in range(params["rebalance"])]
    stats = [mc.stats_for(r, bars_per_year) for r in runs]
    returns = np.array([s["annualized return"] for s in stats])
    sharpes = np.array([s["sharpe (naive)"] for s in stats])

    blended = dict(pnl=np.mean([r["pnl"] for r in runs], axis=0),
                   turnover=np.mean([r["turnover"] for r in runs], axis=0),
                   start_bar=split_idx + params["rebalance"])
    tranched = mc.stats_for(blended, bars_per_year)
    benchmark = mc.stats_for(mc.buy_and_hold(prepared, split_idx), bars_per_year)

    print("=" * 96)
    print("BACKTEST: %s   [--variant %s]" % (label, name))
    print("=" * 96)
    print("panel          : %s bars x %d tickers, %s -> %s"
          % (format(len(close), ","), close.shape[1],
             close.index[0].strftime("%Y-%m-%d"), close.index[-1].strftime("%Y-%m-%d")))
    print("out-of-sample  : from %s, %s bars"
          % (close.index[split_idx].strftime("%Y-%m-%d"), format(len(close) - split_idx, ",")))
    print("settings       : top_n=%d, rebalance=%d bars, lookback=%d bars, DMI window=%d, ADX>=%.0f"
          % (params["top_n"], params["rebalance"], params["lookback_mom"],
             args.di_window, args.adx_min))

    print("\n--- Phase distribution (the same strategy started on each of %d bars) ---"
          % params["rebalance"])
    print("  annualized return  min %6.1f%%   median %6.1f%%   max %6.1f%%"
          % (returns.min() * 100, np.median(returns) * 100, returns.max() * 100))
    print("  sharpe             min %6.2f    median %6.2f    max %6.2f"
          % (sharpes.min(), np.median(sharpes), sharpes.max()))

    print("\n--- Phase-neutral result (book staggered across all phases) ---")
    for cost_bps in (0, 5, 10, 20):
        costed = mc.stats_for(mc.apply_cost(blended, cost_bps), bars_per_year)
        print("  %2d bps: %6.1f%% annualized, Sharpe %5.2f, max drawdown %.2f"
              % (cost_bps, costed["annualized return"] * 100, costed["sharpe (naive)"],
                 costed["max drawdown"]))
    print("  benchmark (buy and hold, no costs): %.1f%% / Sharpe %.2f"
          % (benchmark["annualized return"] * 100, benchmark["sharpe (naive)"]))

    traded = runs[0]["n_held"][split_idx:]
    held = traded[traded > 0]
    years = max(1e-9, (len(close) - split_idx) / bars_per_year)
    print("\n--- What it actually did ---")
    print("  average names held while invested : %.2f of %d" % (held.mean() if len(held) else 0, params["top_n"]))
    print("  bars holding nothing              : %.0f%%" % ((traded == 0).mean() * 100))
    print("  turnover per year                 : %.1f x the book" % (np.sum(runs[0]["turnover"]) / years))

    print("\n--- Is the rule better than picking at random from the same names? ---")
    random_returns, _ = mc.permutation_test(prepared, split_idx, bars_per_year,
                                            mc.random_eligible_selector, args.sims)
    p_value = float((random_returns >= np.median(returns)).mean())
    print("  P(random basket >= this variant's median phase) = %.3f  ->  %s"
          % (p_value, "beats chance" if p_value < 0.05 else
             "not distinguishable from chance" if p_value > 0.20 else "borderline"))

    stability = neighborhood_stability(panel, close, prepared, split_idx, bars_per_year,
                                       name, kind, args.di_window, args.adx_min,
                                       extra.get("hurst"))
    print("\n--- Robustness across its own settings (%d combinations) ---" % stability["nbhd settings"])
    print("  phase-neutral Sharpe: median %.2f, worst %.2f"
          % (stability["nbhd median sharpe"], stability["nbhd worst sharpe"]))
    if np.isfinite(stability["nbhd worst sharpe"]) and stability["nbhd worst sharpe"] < 0:
        print("  WARNING: this rule turns negative somewhere in its own parameter neighbourhood.")
        print("  A setting that only works at one threshold is a fitted cell, not an edge.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test candidate upgrades to BIST30 momentum.")
    parser.add_argument("--ohlc", default=None, help="cached OHLC panel (pickle) instead of downloading")
    parser.add_argument("--save-ohlc", default=None, help="save the downloaded OHLC panel here")
    parser.add_argument("--di-window", type=int, default=45, help="DMI/ADX window in bars")
    parser.add_argument("--adx-min", type=float, default=20.0, help="ADX threshold for the strength filters")
    parser.add_argument("--hurst-window", type=int, default=250, help="trailing window for the Hurst exponent")
    parser.add_argument("--skip-hurst", action="store_true", help="skip the Hurst variants (much faster)")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--variant", default=None,
                        help="backtest ONE variant in depth instead of comparing all (see --list)")
    parser.add_argument("--list", action="store_true", help="list the variant names and exit")
    parser.add_argument("--sims", type=int, default=200,
                        help="permutation simulations used by --variant")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)

    if args.list:
        print("Variants (use with --variant <name>):\n")
        for name, _, label in VARIANTS:
            print("  %-12s %s" % (name, label))
        return 0

    panel = load_ohlc(args.ohlc) if args.ohlc else download_ohlc()
    if args.save_ohlc:
        with open(args.save_ohlc, "wb") as handle:
            pickle.dump(panel, handle)

    close = panel["Close"]
    params = {**mo.DEFAULTS}
    bars_per_year = uni.bars_per_year(close.index)

    # The walk-forward used below is the mission's own, verified bar for bar
    # before anything is measured against it.
    worst, split_idx = mc.verify_runner_matches_engine(close, params)
    prepared = mc.prepare(close, params)

    print("=" * 108)
    print("BIST30 momentum -- candidate upgrades, judged on the phase distribution")
    print("=" * 108)
    print("panel        : %s bars x %d tickers, %s -> %s (OHLC)"
          % (format(len(close), ","), close.shape[1],
             close.index[0].strftime("%Y-%m-%d"), close.index[-1].strftime("%Y-%m-%d")))
    print("runner check : max |fast - engine| = %.3g per bar" % worst)
    print("DMI window   : %d bars (~%.1f sessions)   ADX threshold: %.0f"
          % (args.di_window, args.di_window / (bars_per_year / 252), args.adx_min))

    started = time.time()
    plus_di, minus_di, adx = wilder_dmi(panel["High"], panel["Low"], close, args.di_window)
    extra = {
        "plus_di": plus_di.to_numpy(),
        "minus_di": minus_di.to_numpy(),
        "adx": adx.to_numpy(),
        "vol": realized_volatility(close, params["ma_fast"]).to_numpy(),
    }
    print("indicators   : DMI/ADX built in %.1fs" % (time.time() - started))

    wanted = [(name, kind, label) for name, kind, label in VARIANTS
              if args.skip_hurst is False or name not in HURST_VARIANTS]
    needs_hurst = any(name in HURST_VARIANTS for name, _, _ in wanted)
    if args.variant:
        wanted = [row for row in wanted if row[0] == args.variant]
        if not wanted:
            print("Unknown variant %r. Available: %s"
                  % (args.variant, ", ".join(name for name, _, _ in VARIANTS)))
            return 2
        needs_hurst = wanted[0][0] in HURST_VARIANTS

    if needs_hurst:
        started = time.time()
        extra["hurst"] = rolling_hurst(close, window=args.hurst_window).to_numpy()
        print("             : rolling Hurst built in %.1fs" % (time.time() - started))

    # Backtest one variant on its own, in depth.
    if args.variant:
        name, kind, label = wanted[0]
        return backtest_one(panel, close, prepared, split_idx, bars_per_year,
                            name, kind, label, extra, args)

    thresholds = {"adx_min": args.adx_min, "hurst_min": 0.5}
    rows = []
    for name, kind, label in wanted:
        selector = make_selector(kind, extra, thresholds)
        row = evaluate(prepared, split_idx, bars_per_year, selector, label,
                       cost_bps=args.cost_bps)
        row["variant"] = "%-11s %s" % (name, label)
        # Re-measure around the variant's own settings. A rule that only
        # works at one threshold is a fitted cell, and the table should say
        # so in the same row that shows its flattering headline.
        row.update(neighborhood_stability(panel, close, prepared, split_idx, bars_per_year,
                                          name, kind, args.di_window, args.adx_min,
                                          extra.get("hurst")))
        rows.append(row)

    benchmark = mc.stats_for(mc.buy_and_hold(prepared, split_idx), bars_per_year)
    rows.append({"variant": "BENCHMARK: buy and hold all 30",
                 "median return": benchmark["annualized return"],
                 "median sharpe": benchmark["sharpe (naive)"],
                 "worst phase": benchmark["annualized return"],
                 "best phase": benchmark["annualized return"],
                 "tranched return": benchmark["annualized return"],
                 "tranched sharpe": benchmark["sharpe (naive)"],
                 "after cost return": benchmark["annualized return"],
                 "after cost sharpe": benchmark["sharpe (naive)"],
                 "avg held": float(close.shape[1]), "bars idle %": 0.0, "turnover/yr": 0.0,
                 "nbhd median sharpe": benchmark["sharpe (naive)"],
                 "nbhd worst sharpe": benchmark["sharpe (naive)"], "nbhd settings": 1})

    table = pd.DataFrame(rows)
    for column in table.columns:
        if "return" in column or "phase" in column:
            table[column] = (table[column] * 100).round(1)
        elif "sharpe" in column:
            table[column] = table[column].round(2)

    print("\nAll figures out-of-sample. 'tranched' = book staggered across every rebalance phase,")
    print("which is the number that does not change when you re-run it. Costs charged on turnover;")
    print("the benchmark trades once and pays none.\n")
    print(table.to_string(index=False))

    beat = table[(table["tranched sharpe"] > benchmark["sharpe (naive)"]) &
                 (table["variant"] != "BENCHMARK: buy and hold all 30")]
    print("\n" + "-" * 108)
    if len(beat):
        print("Variants that beat buy-and-hold on phase-neutral Sharpe:")
        print(beat[["variant", "tranched return", "tranched sharpe", "after cost sharpe"]].to_string(index=False))
        print("\nBefore trusting any of these: they were all measured on the SAME sample, so the best")
        print("of nine is flattered by selection. Re-check the winner with momentum_monte_carlo.py's")
        print("permutation test and on a different start date before putting it anywhere near money.")
    else:
        print("No variant beat buy-and-hold on phase-neutral Sharpe. On this sample, +-DI, ADX,")
        print("Hurst and risk-adjusted ranking do not rescue the strategy -- the problem is not the")
        print("filter, it is that a long-only rotation inside a rising basket mostly reproduces the")
        print("basket. That points at market-relative ranking or a cash rule, not another indicator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
