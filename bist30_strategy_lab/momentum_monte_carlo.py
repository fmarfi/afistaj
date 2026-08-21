"""
Monte Carlo robustness testing for the BIST30 momentum strategy.

WHY THIS EXISTS. A single backtest prints one number. We measured that the
same code, same parameters, same universe produces 19% to 76% annualized
depending on nothing but the day it was run -- because the rebalance grid
is anchored to whichever bar the out-of-sample window happens to start on,
and shifting that by one session reshuffles every pick that follows. One
number out of that range is not a result; it is a draw from a distribution
nobody had looked at.

WHAT MONTE CARLO DOES AND DOES NOT DO HERE. It adds no return and finds no
new signal -- it is a truth-teller, not an alpha source. What it answers:

  1. HOW LUCKY was the headline? (rebalance-phase sweep, start-date sweep)
  2. Is the RANKING doing any work, or would five names picked at random
     from the same universe have done as well? (random-basket permutation)
  3. Would simply owning all 30 have done better? (buy-and-hold benchmark)
  4. How much of the Sharpe survives resampling the return path?
     (moving block bootstrap -- reused from validation.py, not re-written)

Question 2 matters most for this particular strategy. BIST30 rose hard over
the sample and the result is carried by a few names; "long 5 of 30 rising
stocks" can look like an edge when it is really just market exposure plus
concentration. A permutation test is the cheapest way to tell those apart.

This module is READ-ONLY with respect to the rest of the mission: it reuses
_engine_momentum's own signal functions rather than copying them, so it
cannot drift from the strategy it is testing. Its fast runner is verified
bar-for-bar against _engine_momentum.walk_forward_momentum before any
simulation is run (see verify_runner_matches_engine).

Run:
    python momentum_monte_carlo.py                     # full report
    python momentum_monte_carlo.py --sims 1000         # more permutations
    python momentum_monte_carlo.py --prices panel.csv  # offline, saved panel
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

import _common as c
import _engine_momentum as mo
import universe as uni
import validation as val


# --------------------------------------------------------------------------
# A fast, exactly-equivalent walk-forward
# --------------------------------------------------------------------------

def prepare(prices, params):
    """
    Everything a run needs, computed once: log returns, and the ROC / trend
    signals from the engine's OWN functions so a simulation can never test
    slightly different signals than the strategy uses.
    """
    params = {**mo.DEFAULTS, **(params or {})}
    log_returns = np.log(prices).diff().to_numpy()
    roc = mo.compute_roc(prices, params["lookback_mom"])
    trend = mo.trend_filter(prices, params["ma_fast"], params["ma_slow"])
    return dict(
        params=params,
        tickers=list(prices.columns),
        log_returns=log_returns,
        roc=roc, trend=trend,
        roc_values=roc.to_numpy(),
        trend_values=trend.fillna(False).to_numpy(),
        n_bars=len(prices),
    )


def rank_selector(prepared, signal_bar, top_n, rng):
    """The real strategy: the engine's own ranking, on the previous bar's data."""
    return mo.rank_universe(prepared["roc"].iloc[signal_bar],
                            prepared["trend"].iloc[signal_bar], top_n)


def random_eligible_selector(prepared, signal_bar, top_n, rng):
    """
    The null hypothesis for the RANKING: same universe, same trend filter,
    same number of names, but chosen at random among those that qualify.
    If the real strategy cannot beat this, the ROC ordering is decoration.
    """
    eligible = [i for i in range(len(prepared["tickers"]))
                if prepared["trend_values"][signal_bar, i]
                and np.isfinite(prepared["roc_values"][signal_bar, i])]
    if not eligible:
        return []
    chosen = rng.choice(eligible, size=min(top_n, len(eligible)), replace=False)
    return [prepared["tickers"][i] for i in chosen]


def random_any_selector(prepared, signal_bar, top_n, rng):
    """
    The null hypothesis for the WHOLE signal, filter included: any names
    that had a price that bar. Separates "we picked well" from "we were
    long a rising market".
    """
    available = [i for i in range(len(prepared["tickers"]))
                 if np.isfinite(prepared["roc_values"][signal_bar, i])]
    if not available:
        return []
    chosen = rng.choice(available, size=min(top_n, len(available)), replace=False)
    return [prepared["tickers"][i] for i in chosen]


def run(prepared, start_bar, selector=rank_selector, rng=None, top_n=None):
    """
    The walk-forward, vectorized per rebalance block.

    Equivalent to _engine_momentum.walk_forward_momentum by construction:
    holdings only change at a rebalance, so the equal weight 1/n is constant
    across each block and the block's pnl is the held names' summed returns
    over n. Non-finite returns are skipped but still divide by n -- exactly
    what the engine does when a held name has no bar.

    start_bar is a parameter rather than being derived from a fraction: it
    is the thing the phase/start-date sweeps need to vary.
    """
    params = prepared["params"]
    rebalance = params["rebalance"]
    top_n = top_n if top_n is not None else params["top_n"]
    n_bars = prepared["n_bars"]
    tickers = prepared["tickers"]
    index_of = {t: i for i, t in enumerate(tickers)}

    pnl = np.zeros(n_bars)
    n_held = np.zeros(n_bars, dtype=int)
    turnover = np.zeros(n_bars)
    held = []

    for t0 in range(start_bar, n_bars, rebalance):
        selected = selector(prepared, t0 - 1, top_n, rng)
        dropped = [t for t in held if t not in selected]
        added = [t for t in selected if t not in held]
        turnover[t0] = (len(dropped) + len(added)) / max(top_n, 1)
        held = selected

        t1 = min(t0 + rebalance, n_bars)
        if held:
            columns = [index_of[t] for t in held]
            block = prepared["log_returns"][t0:t1][:, columns]
            pnl[t0:t1] = np.nansum(block, axis=1) / len(held)
            n_held[t0:t1] = len(held)

    return dict(pnl=pnl, n_held=n_held, turnover=turnover, start_bar=start_bar)


def tranche_run(prepared, base_start, n_tranches):
    """
    Split the book into `n_tranches` equal sub-portfolios, each rebalancing
    on a different offset within the cycle.

    This is what the phase sweep argues for. If the outcome depends on which
    bar you happen to rebalance on, do not bet the whole book on one bar --
    stagger it. The tranches hold the same strategy, so this adds no signal
    and cannot manufacture edge; what it removes is the luck. Whether the
    strategy is worth running is then a question about the tranched number,
    which is the same every time you run it.
    """
    step = max(1, prepared["params"]["rebalance"] // n_tranches)
    offsets = [base_start + j * step for j in range(n_tranches)]
    blended = np.mean([run(prepared, offset)["pnl"] for offset in offsets], axis=0)
    turnover = np.mean([run(prepared, offset)["turnover"] for offset in offsets], axis=0)
    return dict(pnl=blended, turnover=turnover, start_bar=max(offsets))


def buy_and_hold(prepared, start_bar):
    """Own all 30 equally -- the benchmark the strategy has to beat to be worth its turnover."""
    log_returns = prepared["log_returns"]
    n_bars = prepared["n_bars"]
    pnl = np.zeros(n_bars)
    for t in range(start_bar, n_bars):
        row = log_returns[t]
        finite = np.isfinite(row)
        if finite.any():
            pnl[t] = np.nansum(row[finite]) / finite.sum()
    return dict(pnl=pnl, start_bar=start_bar)


def stats_for(result, bars_per_year):
    """Annualized return / Sharpe over the traded window only."""
    return c.performance_stats(result["pnl"], result["start_bar"], bars_per_year)


# --------------------------------------------------------------------------
# Verification: the fast runner must equal the engine before it is trusted
# --------------------------------------------------------------------------

def verify_runner_matches_engine(prices, params=None, tolerance=1e-12):
    """
    Runs the real engine and this module's fast runner on the same data and
    compares the bar-by-bar pnl. A Monte Carlo built on a subtly different
    walk-forward would produce a confident distribution around the wrong
    strategy, so this runs before any simulation and raises if it fails.
    """
    params = {**mo.DEFAULTS, **(params or {})}
    engine_pnl, _, _, _, split_idx = mo.walk_forward_momentum(prices, params)
    prepared = prepare(prices, params)
    fast = run(prepared, split_idx)
    worst = float(np.max(np.abs(engine_pnl - fast["pnl"])))
    if worst > tolerance:
        raise AssertionError(
            "fast runner disagrees with _engine_momentum by %.3g -- refusing to "
            "simulate against a strategy that is not the strategy" % worst)
    return worst, split_idx


# --------------------------------------------------------------------------
# The three sweeps
# --------------------------------------------------------------------------

def apply_cost(result, cost_bps):
    """
    Charge a per-unit-of-turnover cost on the bars a rebalance actually
    traded -- same convention as _engine_momentum's cost table. A rotation
    strategy pays this; a buy-and-hold benchmark does not, which is exactly
    why the two have to be compared after costs, not before.
    """
    pnl = result["pnl"] - result["turnover"] * (cost_bps / 10000.0)
    pnl[result["start_bar"]] = 0.0
    return dict(pnl=pnl, start_bar=result["start_bar"])


def phase_sweep(prepared, base_start, bars_per_year):
    """
    The same strategy started one bar later, and later again, through a full
    rebalance cycle. Nothing about the strategy changes -- only which bar the
    rebalance grid lands on. A wide spread here means the headline number is
    a property of the calendar, not of the edge.
    """
    rows = []
    for offset in range(prepared["params"]["rebalance"]):
        result = run(prepared, base_start + offset)
        stats = stats_for(result, bars_per_year)
        rows.append({"offset_bars": offset,
                     "annualized return": stats["annualized return"],
                     "sharpe": stats["sharpe (naive)"],
                     "max drawdown": stats["max drawdown"]})
    return pd.DataFrame(rows)


def start_date_sweep(prepared, timestamps, bars_per_year, n_starts=24, step_bars=45):
    """
    The same strategy evaluated from a range of out-of-sample start dates --
    what re-running the backtest on different days actually does.
    """
    rows = []
    first = prepared["params"]["lookback_mom"] + 5
    last = prepared["n_bars"] - 400
    if last <= first:
        return pd.DataFrame(rows)
    starts = np.linspace(first, last, num=min(n_starts, max(2, (last - first) // step_bars)))
    for start in starts.astype(int):
        result = run(prepared, int(start))
        stats = stats_for(result, bars_per_year)
        rows.append({"start": pd.Timestamp(timestamps[int(start)]).strftime("%Y-%m-%d"),
                     "oos_bars": prepared["n_bars"] - int(start),
                     "annualized return": stats["annualized return"],
                     "sharpe": stats["sharpe (naive)"]})
    return pd.DataFrame(rows)


def permutation_test(prepared, start_bar, bars_per_year, selector, n_sims, seed=0):
    """
    Re-run the strategy many times with the ranking replaced by a random
    draw, and ask where the real result falls in that distribution.

    The p-value is the fraction of random runs that did AT LEAST as well as
    the real one: 0.02 means only 2% of coin-flip baskets matched it, 0.45
    means the ranking is indistinguishable from chance on this sample.
    """
    rng = np.random.default_rng(seed)
    returns = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    for i in range(n_sims):
        result = run(prepared, start_bar, selector=selector, rng=rng)
        stats = stats_for(result, bars_per_year)
        returns[i] = stats["annualized return"]
        sharpes[i] = stats["sharpe (naive)"]
    return returns, sharpes


def percentile_row(label, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"what": label}
    return {"what": label, "min": values.min(), "p05": np.percentile(values, 5),
            "median": np.median(values), "p95": np.percentile(values, 95),
            "max": values.max()}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Monte Carlo robustness report for BIST30 momentum.")
    parser.add_argument("--prices", default=None, help="saved price panel (CSV) instead of downloading")
    parser.add_argument("--sims", type=int, default=400, help="permutation simulations per null")
    parser.add_argument("--top-n", type=int, default=mo.DEFAULTS["top_n"])
    parser.add_argument("--rebalance", type=int, default=mo.DEFAULTS["rebalance"])
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args(argv)

    pd.set_option("display.width", 140)

    if args.prices:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        import momentum_standalone as standalone
        prices = standalone.load_price_panel(args.prices)
    else:
        prices = uni.download_universe()

    params = {**mo.DEFAULTS, "top_n": args.top_n, "rebalance": args.rebalance}
    bars_per_year = uni.bars_per_year(prices.index)
    timestamps = prices.index

    worst, split_idx = verify_runner_matches_engine(prices, params)
    prepared = prepare(prices, params)

    print("=" * 92)
    print("BIST30 momentum -- Monte Carlo robustness report")
    print("=" * 92)
    print("panel            : %s bars x %d tickers, %s -> %s"
          % (format(len(prices), ","), prices.shape[1],
             timestamps[0].strftime("%Y-%m-%d"), timestamps[-1].strftime("%Y-%m-%d")))
    print("parameters       : top_n=%d, rebalance=%d bars, lookback=%d bars"
          % (params["top_n"], params["rebalance"], params["lookback_mom"]))
    print("runner check     : max |fast - engine| = %.3g per bar (must be ~0)" % worst)

    headline = run(prepared, split_idx)
    headline_stats = stats_for(headline, bars_per_year)
    print("headline         : %.1f%% annualized, Sharpe %.2f  <- the single number a backtest prints"
          % (headline_stats["annualized return"] * 100, headline_stats["sharpe (naive)"]))

    # 1. how much of that is the calendar?
    phases = phase_sweep(prepared, split_idx, bars_per_year)
    print("\n--- 1. Rebalance-phase sweep (same strategy, grid shifted 0..%d bars) ---"
          % (params["rebalance"] - 1))
    print("Nothing about the strategy changes; only which bar it happens to rebalance on.")
    print(pd.DataFrame([
        percentile_row("annualized return", phases["annualized return"]),
        percentile_row("sharpe", phases["sharpe"]),
    ]).to_string(index=False, float_format=lambda v: "%.3f" % v))
    rank_in_phases = float((phases["annualized return"] <= headline_stats["annualized return"]).mean())
    print("The headline sits at the %.0fth percentile of its own phase distribution." % (rank_in_phases * 100))

    # 2. and how much is the start date?
    starts = start_date_sweep(prepared, timestamps, bars_per_year)
    if len(starts):
        print("\n--- 2. Out-of-sample start-date sweep ---")
        print(pd.DataFrame([
            percentile_row("annualized return", starts["annualized return"]),
            percentile_row("sharpe", starts["sharpe"]),
        ]).to_string(index=False, float_format=lambda v: "%.3f" % v))

    # 3. is the ranking doing anything?
    print("\n--- 3. Permutation tests: %d simulations each ---" % args.sims)
    print("Same universe, same trend filter, same basket size -- ranking replaced by a random draw.")
    rows = []
    p_values = {}
    for label, selector in (("random from trend-eligible", random_eligible_selector),
                            ("random from all available", random_any_selector)):
        returns, sharpes = permutation_test(prepared, split_idx, bars_per_year,
                                            selector, args.sims)
        rows.append(percentile_row(label + " (ann. return)", returns))
        p_values[label] = float((returns >= headline_stats["annualized return"]).mean())
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.3f" % v))
    for label, p in p_values.items():
        verdict = ("the ranking beat chance" if p < 0.05 else
                   "NOT distinguishable from chance" if p > 0.20 else "borderline")
        print("  P(random >= real) vs %-28s = %.3f  -> %s" % (label, p, verdict))

    # 4. and would just owning everything have done better?
    bh = buy_and_hold(prepared, split_idx)
    bh_stats = stats_for(bh, bars_per_year)
    print("\n--- 4. Benchmark: equal-weight all %d names, no trading ---" % prices.shape[1])
    print("  buy and hold          : %6.1f%% annualized, Sharpe %.2f  (no turnover, no costs)"
          % (bh_stats["annualized return"] * 100, bh_stats["sharpe (naive)"]))
    print("  strategy, headline    : %6.1f%% annualized, Sharpe %.2f  <- one lucky phase"
          % (headline_stats["annualized return"] * 100, headline_stats["sharpe (naive)"]))
    print("  strategy, median phase: %6.1f%% annualized, Sharpe %.2f  <- what to expect"
          % (phases["annualized return"].median() * 100, phases["sharpe"].median()))

    # The decision-relevant number: how often does the rotation actually beat
    # simply owning the basket, once you stop cherry-picking the start bar?
    beat_return = float((phases["annualized return"] > bh_stats["annualized return"]).mean())
    beat_sharpe = float((phases["sharpe"] > bh_stats["sharpe (naive)"]).mean())
    print("  across the %d phases it beat buy-and-hold on return %.0f%% of the time, on Sharpe %.0f%%."
          % (len(phases), beat_return * 100, beat_sharpe * 100))

    print("\n  After costs (charged on turnover; buy-and-hold pays none):")
    for cost_bps in (0, 5, 10, 20):
        costed = [stats_for(apply_cost(run(prepared, split_idx + offset), cost_bps), bars_per_year)
                  for offset in range(prepared["params"]["rebalance"])]
        median_return = float(np.median([s["annualized return"] for s in costed]))
        median_sharpe = float(np.median([s["sharpe (naive)"] for s in costed]))
        print("    %2d bps: strategy median %5.1f%% / Sharpe %.2f   vs buy-and-hold %5.1f%% / Sharpe %.2f"
              % (cost_bps, median_return * 100, median_sharpe,
                 bh_stats["annualized return"] * 100, bh_stats["sharpe (naive)"]))

    # 4b. the fix the phase sweep argues for
    print("\n--- 4b. Staggered tranches: the book split across rebalance offsets ---")
    print("Removes the phase luck rather than hoping for a good draw. Adds no signal.")
    for n_tranches in (1, 3, 5, 9):
        tranched = tranche_run(prepared, split_idx, n_tranches)
        stats = stats_for(tranched, bars_per_year)
        costed = stats_for(apply_cost(tranched, 5), bars_per_year)
        print("  %2d tranche(s): %5.1f%% / Sharpe %4.2f   (after 5 bps: %5.1f%% / Sharpe %4.2f)"
              % (n_tranches, stats["annualized return"] * 100, stats["sharpe (naive)"],
                 costed["annualized return"] * 100, costed["sharpe (naive)"]))
    print("  benchmark    : %5.1f%% / Sharpe %4.2f   (buy and hold, no costs)"
          % (bh_stats["annualized return"] * 100, bh_stats["sharpe (naive)"]))

    # 5. resample the path itself (reuses validation.py, not re-implemented)
    print("\n--- 5. Moving block bootstrap of the out-of-sample path ---")
    boot = val.bootstrap_significance(headline["pnl"][split_idx:], bars_per_year,
                                      block_length=int(max(10, params["rebalance"])))
    print("  median Sharpe %.2f, 90%% CI [%.2f, %.2f], P(Sharpe > 0) = %.3f"
          % (boot["sharpe_median"], boot["sharpe_ci_low"], boot["sharpe_ci_high"],
             boot["p_sharpe_gt_0"]))

    print("\n" + "-" * 92)
    print("Report the MEDIAN and the range, not the headline. The headline is one draw from")
    print("the phase distribution above and there is no reason to prefer it to any other draw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
