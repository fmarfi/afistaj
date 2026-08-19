"""
Batch step 3: load 02's cached full-universe results, build the honest,
DSR-corrected leaderboard, run the bootstrap significance check and a
parameter-perturbation sensitivity check on the top-ranked family, and
save everything the dashboard's leaderboard tab needs.

Usage: python bist30_strategy_lab/03_run_validation_leaderboard.py
(from the afistaj/ project root, with the venv active -- run
02_run_full_universe_backtests.py first)
"""

import pickle
import pathlib

import pandas as pd

import validation as v
import _engine_meanrev as em
import _engine_momentum as mo
import _engine_pca as pca

CACHE_PATH = pathlib.Path(__file__).parent / "_universe_backtest_cache.pkl"
LEADERBOARD_CSV = pathlib.Path(__file__).parent / "_leaderboard.csv"
LEADERBOARD_PKL = pathlib.Path(__file__).parent / "_leaderboard_detail.pkl"

# Total number of strategy/parameter combinations actually tried on the way
# to these 4 evaluated variants (3 families + 1 cost-calibrated variant of
# mean-reversion) -- see WORKFLOW.md's "Validation methodology" section for
# the accounting. This is a deliberately conservative (i.e. generous to the
# strategies) count: it does NOT include the many parameter values explored
# earlier this session on the two abandoned missions (pairs, then volatility
# mean-reversion), only the tuning that happened within this mission's own
# build. Bumped 12 -> 14 when the cost-calibrated entry_z variant was added:
# +1 for trying the calibration idea itself, +1 for the specific
# cost_per_round_trip_bps=5 assumption used.
N_TRIALS = 14

RUN_FNS = {"momentum": mo.run_full_backtest_momentum, "pca_stat_arb": pca.run_full_backtest_pca}


def main():
    if not CACHE_PATH.exists():
        print(f"{CACHE_PATH.name} not found -- run 02_run_full_universe_backtests.py first.")
        return

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    prices, bpy, evaluations = cache["prices"], cache["bars_per_year"], cache["evaluations"]

    print("=" * 70)
    print(f"LEADERBOARD (n_trials={N_TRIALS} -- see WORKFLOW.md for the accounting)")
    print("=" * 70)
    board = v.leaderboard(list(evaluations.values()), n_trials=N_TRIALS, starting_capital=100_000)
    print(board.to_string(index=False))

    print("\n" + "=" * 70)
    print("Bootstrap significance of the TOP-RANKED family's OOS pnl")
    print("=" * 70)
    top_family = board.iloc[0]["family"]
    top_result = evaluations[top_family]["best_result"]
    pnl_oos = top_result["results"]["pnl"].to_numpy()[top_result["split_idx"]:]
    block_length = {"mean_reversion_ou": 45, "momentum": 45, "pca_stat_arb": 45}.get(top_family, 45)
    sig = v.bootstrap_significance(pnl_oos, bpy, block_length=block_length, n_boot=2000)
    print(f"{top_family}: median Sharpe {sig['sharpe_median']:.3f} "
          f"(90% CI {sig['sharpe_ci_low']:.3f} to {sig['sharpe_ci_high']:.3f}), "
          f"P(Sharpe > 0) = {sig['p_sharpe_gt_0']:.3f}")

    print("\n" + "=" * 70)
    print(f"Parameter-perturbation sensitivity for {top_family}")
    print("=" * 70)
    sensitivity = None
    if top_family in RUN_FNS:
        run_fn = RUN_FNS[top_family]
        base_params = mo.DEFAULTS if top_family == "momentum" else pca.DEFAULTS
        perturb_keys = ["rebalance"] if top_family == "momentum" else ["rebalance", "z_window"]
        sensitivity = v.perturbation_sensitivity(run_fn, prices, base_params, perturb_keys, bpy)
        print(f"base Sharpe: {sensitivity['base_sharpe']:.3f}")
        print(sensitivity["perturbations"].to_string(index=False))
    else:
        print("(mean-reversion's parameters are per-pair-derived, not global constants -- "
              "see WORKFLOW.md for why a global perturbation sweep doesn't apply the same way here.)")

    board.to_csv(LEADERBOARD_CSV, index=False)
    with open(LEADERBOARD_PKL, "wb") as f:
        pickle.dump({"board": board, "evaluations": evaluations, "bars_per_year": bpy,
                     "top_family": top_family, "bootstrap": sig, "sensitivity": sensitivity,
                     "n_trials": N_TRIALS}, f)
    print(f"\nSaved {LEADERBOARD_CSV.name} and {LEADERBOARD_PKL.name} -- the dashboard reads the latter.")


if __name__ == "__main__":
    main()
