"""
Batch: run every candidate momentum upgrade through the robustness bench and
cache the result for the dashboard's Robustness tab.

Same shape as 02/03: the expensive work happens here, once, and writes a
pickle; the dashboard only renders. A full bench is several minutes (each
variant is 45 phase runs, and the parameter grids are 15 cells of 45 runs
each), which is far too slow to sit inside a web request.

What lands in the cache, per variant:
  - the whole rebalance-phase distribution, not a single backtest
  - the phase-neutral ("tranched") equity curve, at 0 and 5 bps
  - concentration and turnover, so a variant that "wins" by holding one
    name is visibly a bigger bet rather than a better rule
  - a PARAMETER GRID: the same rule re-measured across its own knobs. This
    is the one that matters. Every variant that beat the benchmark on this
    sample did so in a single cell whose neighbours collapse, and a grid
    shows that at a glance where a summary number hides it.

Run:  python 05_run_upgrade_bench.py [--ohlc cached_panel.pkl]
Then: python 04_interactive_dashboard.py  ->  Robustness tab
"""

import argparse
import pathlib
import pickle
import time

import numpy as np

import _engine_momentum as mo
import momentum_monte_carlo as mc
import momentum_upgrades as up
import universe as uni

CACHE_PATH = pathlib.Path(__file__).parent / "_upgrade_bench.pkl"

# Each variant's own knobs, as (row label, row values, column label, column
# values). A variant with no knobs gets no grid -- there is nothing to be
# fragile about.
DI_WINDOWS = (14, 45, 126)
ADX_LEVELS = (10.0, 15.0, 20.0, 25.0, 30.0)
HURST_WINDOWS = (150, 250, 400)
HURST_LEVELS = (0.45, 0.50, 0.55, 0.60)


def grid_axes(name):
    if name in up.ADX_VARIANTS:
        return ("DMI window (bars)", DI_WINDOWS, "ADX threshold", ADX_LEVELS)
    if name in up.HURST_VARIANTS:
        return ("Hurst window (bars)", HURST_WINDOWS, "Hurst threshold", HURST_LEVELS)
    if name in up.DI_VARIANTS:
        return ("DMI window (bars)", DI_WINDOWS, "(no second knob)", (None,))
    return (None, (), None, ())


def phase_runs(prepared, split_idx, selector):
    return [mc.run(prepared, split_idx + offset, selector=selector)
            for offset in range(prepared["params"]["rebalance"])]


def blend(runs, split_idx, rebalance):
    """The phase-neutral book: every rebalance offset held in equal size."""
    return dict(pnl=np.mean([r["pnl"] for r in runs], axis=0),
                turnover=np.mean([r["turnover"] for r in runs], axis=0),
                start_bar=split_idx + rebalance)


def equity_curve(result, capital=100_000.0):
    """Currency path, sized off starting capital -- same no-compounding convention as _common.add_capital_pnl."""
    return capital + np.cumsum(np.nan_to_num(result["pnl"]) * capital)


def measure(prepared, split_idx, bars_per_year, selector, capital):
    runs = phase_runs(prepared, split_idx, selector)
    stats = [mc.stats_for(r, bars_per_year) for r in runs]
    blended = blend(runs, split_idx, prepared["params"]["rebalance"])
    tranched = mc.stats_for(blended, bars_per_year)
    costed = mc.stats_for(mc.apply_cost(blended, 5), bars_per_year)

    traded = runs[0]["n_held"][split_idx:]
    held = traded[traded > 0]
    years = max(1e-9, len(traded) / bars_per_year)

    return {
        "phase_returns": np.array([s["annualized return"] for s in stats]),
        "phase_sharpes": np.array([s["sharpe (naive)"] for s in stats]),
        "tranched": tranched,
        "costed": costed,
        "equity": equity_curve(blended, capital),
        "equity_after_cost": equity_curve(mc.apply_cost(blended, 5), capital),
        "avg_held": float(held.mean()) if len(held) else 0.0,
        "idle_pct": float((traded == 0).mean() * 100),
        "turnover_per_year": float(np.sum(runs[0]["turnover"]) / years),
    }


def build_grid(panel, close, prepared, split_idx, bars_per_year, name, kind, hurst_cache):
    """
    The same rule across its own parameter neighbourhood. Returns the
    phase-neutral Sharpe and annualized return per cell, so the dashboard can
    show whether a variant sits on a plateau or on a single lucky cell.
    """
    row_label, row_values, column_label, column_values = grid_axes(name)
    if not len(row_values):
        return None

    sharpe = np.full((len(row_values), len(column_values)), np.nan)
    annual = np.full((len(row_values), len(column_values)), np.nan)
    params = prepared["params"]

    for i, row_value in enumerate(row_values):
        if name in up.HURST_VARIANTS:
            if row_value not in hurst_cache:
                hurst_cache[row_value] = up.rolling_hurst(close, window=row_value).to_numpy()
            extra = up.build_extra(panel, close, params, 45, params["ma_fast"], hurst_cache[row_value])
        else:
            extra = up.build_extra(panel, close, params, int(row_value), params["ma_fast"],
                                   hurst_cache.get(250))

        for j, column_value in enumerate(column_values):
            thresholds = {"adx_min": 20.0, "hurst_min": 0.5}
            if name in up.ADX_VARIANTS and column_value is not None:
                thresholds["adx_min"] = float(column_value)
            if name in up.HURST_VARIANTS and column_value is not None:
                thresholds["hurst_min"] = float(column_value)

            selector = up.make_selector(kind, extra, thresholds)
            blended = blend(phase_runs(prepared, split_idx, selector), split_idx, params["rebalance"])
            stats = mc.stats_for(blended, bars_per_year)
            sharpe[i, j] = stats["sharpe (naive)"]
            annual[i, j] = stats["annualized return"]

    return {"row_label": row_label, "rows": list(row_values),
            "col_label": column_label, "cols": list(column_values),
            "sharpe": sharpe, "annualized": annual}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the upgrade-bench cache for the dashboard.")
    parser.add_argument("--ohlc", default=None, help="cached OHLC panel instead of downloading")
    parser.add_argument("--save-ohlc", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--skip-grids", action="store_true",
                        help="summary only -- much faster, but the fragility view will be empty")
    args = parser.parse_args(argv)

    started = time.time()
    panel = up.load_ohlc(args.ohlc) if args.ohlc else up.download_ohlc()
    if args.save_ohlc:
        with open(args.save_ohlc, "wb") as handle:
            pickle.dump(panel, handle)

    close = panel["Close"]
    params = {**mo.DEFAULTS}
    bars_per_year = uni.bars_per_year(close.index)
    worst, split_idx = mc.verify_runner_matches_engine(close, params)
    prepared = mc.prepare(close, params)

    print("panel  : %s bars x %d tickers, %s -> %s"
          % (format(len(close), ","), close.shape[1],
             close.index[0].strftime("%Y-%m-%d"), close.index[-1].strftime("%Y-%m-%d")))
    print("runner : max |fast - engine| = %.3g" % worst)

    hurst_cache = {250: up.rolling_hurst(close, window=250).to_numpy()}
    extra = up.build_extra(panel, close, params, 45, params["ma_fast"], hurst_cache[250])

    variants = []
    for name, kind, label in up.VARIANTS:
        step = time.time()
        selector = up.make_selector(kind, extra, {"adx_min": 20.0, "hurst_min": 0.5})
        record = {"name": name, "kind": kind, "label": label}
        record.update(measure(prepared, split_idx, bars_per_year, selector, args.capital))
        record["grid"] = (None if args.skip_grids else
                          build_grid(panel, close, prepared, split_idx, bars_per_year,
                                     name, kind, hurst_cache))
        variants.append(record)
        print("  %-11s tranched %6.1f%% / Sharpe %5.2f   (%.0fs)"
              % (name, record["tranched"]["annualized return"] * 100,
                 record["tranched"]["sharpe (naive)"], time.time() - step))

    benchmark_run = mc.buy_and_hold(prepared, split_idx)
    benchmark = mc.stats_for(benchmark_run, bars_per_year)
    benchmark_equity = equity_curve(benchmark_run, args.capital)

    cache = {
        "generated_at": time.time(),
        "timestamps": close.index,
        "split_idx": split_idx,
        "params": params,
        "bars_per_year": bars_per_year,
        "capital": args.capital,
        "panel_shape": (len(close), close.shape[1]),
        "panel_span": (close.index[0], close.index[-1]),
        "variants": variants,
        "benchmark": {"stats": benchmark, "equity": benchmark_equity},
    }
    with open(CACHE_PATH, "wb") as handle:
        pickle.dump(cache, handle)

    print("\nwrote %s in %.0fs" % (CACHE_PATH.name, time.time() - started))
    print("Open the dashboard's Robustness tab: python 04_interactive_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
