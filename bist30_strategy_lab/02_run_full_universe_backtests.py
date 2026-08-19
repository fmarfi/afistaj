"""
Batch step 2: download the BIST30 universe once, run all 3 strategy
families' full-universe evaluation, and cache the results to disk.

Numbering mirrors examples/'s convention (script "01" doesn't exist in this
mission -- there's no separate screening-only step since each engine's
run_full_backtest_* already screens internally). Caching mirrors
intraday_pairs_trading/run_all.py's pattern: the interactive dashboard
(04_interactive_dashboard.py) reads this script's output instead of
recomputing an expensive full-universe walk-forward on every page load.

Usage: python bist30_strategy_lab/02_run_full_universe_backtests.py
(from the afistaj/ project root, with the venv active -- requires network
access to download from yfinance)
"""

import pickle
import time
import pathlib

from universe import download_universe, bars_per_year, BIST30_TICKERS
import _engine_meanrev as em
import _engine_momentum as mo
import _engine_pca as pca
import validation as v

CACHE_PATH = pathlib.Path(__file__).parent / "_universe_backtest_cache.pkl"


def main():
    print("=" * 70)
    print(f"Downloading BIST30 universe ({len(BIST30_TICKERS)} tickers, hourly bars)")
    print("=" * 70)
    t0 = time.time()
    prices = download_universe()
    bpy = bars_per_year(prices.index)
    print(f"{prices.shape[0]} bars x {prices.shape[1]} tickers "
          f"({prices.index.min().date()} to {prices.index.max().date()}), "
          f"~{bpy:.0f} bars/year -- {time.time()-t0:.1f}s\n")

    print("=" * 70)
    print("Evaluating: mean-reversion (OU half-life pairs, full-universe screen)")
    print("=" * 70)
    t0 = time.time()
    ev_meanrev = v.evaluate_meanrev_universe(prices, bars_per_year=bpy, top_k_pairs=15)
    print(f"{ev_meanrev['n_units']} qualifying pairs tested, "
          f"{ev_meanrev['pct_units_profitable']:.1f}% OOS-profitable -- {time.time()-t0:.1f}s\n")

    print("=" * 70)
    print("Evaluating: mean-reversion, COST-CALIBRATED entry_z (see _ou_calibration.py)")
    print("=" * 70)
    t0 = time.time()
    ev_meanrev_calibrated = v.evaluate_meanrev_universe(
        prices, params={"derive_entry_z_from_cost": True, "cost_per_round_trip_bps": 5},
        bars_per_year=bpy, top_k_pairs=15, family="mean_reversion_ou_cost_calibrated",
    )
    print(f"{ev_meanrev_calibrated['n_units']} qualifying pairs tested, "
          f"{ev_meanrev_calibrated['pct_units_profitable']:.1f}% OOS-profitable -- {time.time()-t0:.1f}s "
          "(slower: simulates each pair's own OU process at every rebalance)\n")

    print("=" * 70)
    print("Evaluating: momentum (cross-sectional rotation, whole basket)")
    print("=" * 70)
    t0 = time.time()
    ev_momentum = v.evaluate_basket_universe("momentum", mo.run_full_backtest_momentum, prices, bars_per_year=bpy)
    print(f"{ev_momentum['n_units']} names traded, "
          f"{ev_momentum['pct_units_profitable']:.1f}% net-profitable -- {time.time()-t0:.1f}s\n")

    print("=" * 70)
    print("Evaluating: PCA basket stat-arb (whole universe)")
    print("=" * 70)
    t0 = time.time()
    ev_pca = v.evaluate_basket_universe("pca_stat_arb", pca.run_full_backtest_pca, prices, bars_per_year=bpy)
    print(f"{ev_pca['n_units']} names traded, "
          f"{ev_pca['pct_units_profitable']:.1f}% net-profitable -- {time.time()-t0:.1f}s\n")

    cache = {
        "prices": prices, "bars_per_year": bpy, "downloaded_at": time.time(),
        "evaluations": {
            "mean_reversion_ou": ev_meanrev,
            "mean_reversion_ou_cost_calibrated": ev_meanrev_calibrated,
            "momentum": ev_momentum, "pca_stat_arb": ev_pca,
        },
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cached to {CACHE_PATH.name} -- run 03_run_validation_leaderboard.py next.")


if __name__ == "__main__":
    main()
