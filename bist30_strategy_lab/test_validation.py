"""
pytest suite for validation.py. Synthetic data only, no network.

Run: python -m pytest bist30_strategy_lab/test_validation.py -v
(from the afistaj/ project root, with the venv active)
"""

import numpy as np
import pandas as pd
import pytest

import validation as v


def _synthetic_meanrev_universe(n_sessions=250, bars_per_session=9, seed=0):
    """Minimal cointegrated-pair panel, same construction as test_engine_meanrev.py's helper."""
    rng = np.random.default_rng(seed)
    idx = []
    d = pd.Timestamp("2024-01-01")
    added = 0
    while added < n_sessions:
        if d.weekday() < 5:
            for h in range(bars_per_session):
                idx.append(d + pd.Timedelta(hours=10 + h))
            added += 1
        d += pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(idx)
    n = len(idx)

    common = np.cumsum(rng.normal(0, 0.01, n))
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = resid[t - 1] + 0.05 * (0 - resid[t - 1]) + rng.normal(0, 0.005)
    log_a = 4.0 + common
    log_b = 3.0 + 0.8 * common + resid
    return pd.DataFrame({"AAA.IS": np.exp(log_a), "BBB.IS": np.exp(log_b)}, index=idx)


def test_evaluate_meanrev_universe_respects_custom_family_label():
    """
    Regression test: evaluate_meanrev_universe used to hardcode "family":
    "mean_reversion_ou" regardless of what params/label were passed, so a
    second call with a different label (e.g. for a cost-calibrated variant)
    silently collided with the first under the SAME dict key in
    02_run_full_universe_backtests.py's cache, causing the leaderboard/
    bootstrap-check logic to pick the wrong evaluation's result.
    """
    prices = _synthetic_meanrev_universe()
    ev_default = v.evaluate_meanrev_universe(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252)
    ev_custom = v.evaluate_meanrev_universe(prices, params={"hurst_cutoff": 0.5}, bars_per_year=9 * 252,
                                             family="mean_reversion_ou_cost_calibrated")
    assert ev_default["family"] == "mean_reversion_ou"
    assert ev_custom["family"] == "mean_reversion_ou_cost_calibrated"
    assert ev_default["family"] != ev_custom["family"]


def test_deflated_sharpe_ratio_decreases_as_n_trials_increases():
    dsr_1 = v.deflated_sharpe_ratio(1.5, n_trials=1, n_obs=1000, bars_per_year=2000)["dsr"]
    dsr_10 = v.deflated_sharpe_ratio(1.5, n_trials=10, n_obs=1000, bars_per_year=2000)["dsr"]
    dsr_100 = v.deflated_sharpe_ratio(1.5, n_trials=100, n_obs=1000, bars_per_year=2000)["dsr"]
    assert dsr_1 > dsr_10 > dsr_100


def test_deflated_sharpe_ratio_increases_with_more_observations():
    dsr_short = v.deflated_sharpe_ratio(1.5, n_trials=10, n_obs=200, bars_per_year=2000)["dsr"]
    dsr_long = v.deflated_sharpe_ratio(1.5, n_trials=10, n_obs=5000, bars_per_year=2000)["dsr"]
    assert dsr_long > dsr_short


def test_deflated_sharpe_ratio_zero_sharpe_gives_low_dsr():
    dsr = v.deflated_sharpe_ratio(0.0, n_trials=5, n_obs=1000, bars_per_year=2000)["dsr"]
    assert dsr < 0.5


def test_expected_max_sharpe_increases_with_n_trials():
    small = v.expected_max_sharpe(2, sr_std_per_period=0.05)
    large = v.expected_max_sharpe(200, sr_std_per_period=0.05)
    assert large > small


def test_expected_max_sharpe_zero_for_single_trial():
    assert v.expected_max_sharpe(1, sr_std_per_period=0.05) == 0.0


def test_moving_block_bootstrap_indices_preserve_local_order():
    rng = np.random.default_rng(0)
    idx = v.moving_block_bootstrap_indices(20, block_length=5, rng=rng)
    assert len(idx) == 20
    # every drawn block of 5 consecutive indices should itself be consecutive
    diffs_within_first_block = np.diff(idx[:5])
    assert np.all(diffs_within_first_block == 1)


def test_bootstrap_significance_high_p_sharpe_for_genuinely_profitable_series():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.02, 0.05, 500)  # strong, consistent positive drift
    result = v.bootstrap_significance(pnl, bars_per_year=2000, block_length=10, n_boot=500,
                                       rng=np.random.default_rng(2))
    assert result["p_sharpe_gt_0"] > 0.9


def test_bootstrap_significance_lower_confidence_for_zero_mean_than_strong_drift():
    """
    A zero-mean noise series' bootstrap P(Sharpe>0) should sit meaningfully
    below a strongly, consistently profitable series' -- not pinned to an
    absolute band, since a single finite noise sample can have a nonzero
    sample mean by chance (that's real sampling variation, not a bug).
    """
    rng_profitable = np.random.default_rng(1)
    pnl_profitable = rng_profitable.normal(0.02, 0.05, 500)
    result_profitable = v.bootstrap_significance(pnl_profitable, bars_per_year=2000, block_length=10,
                                                   n_boot=500, rng=np.random.default_rng(2))

    rng_noise = np.random.default_rng(3)
    pnl_noise = rng_noise.normal(0.0, 0.05, 500)
    result_noise = v.bootstrap_significance(pnl_noise, bars_per_year=2000, block_length=10,
                                              n_boot=500, rng=np.random.default_rng(4))

    assert result_noise["p_sharpe_gt_0"] < result_profitable["p_sharpe_gt_0"]
    assert result_noise["sharpe_ci_low"] < 0 < result_noise["sharpe_ci_high"]  # CI straddles zero


def test_leaderboard_ranks_by_dsr_not_raw_sharpe():
    """
    Family A: modest Sharpe but tested on many units, most profitable.
    Family B: a higher raw Sharpe but from a single lucky-looking unit.
    With the SAME n_trials applied to both, DSR should not simply follow
    raw Sharpe -- and the leaderboard's primary sort key is DSR.
    """
    def _fake_result(sharpe, n_bars=2000):
        pnl = np.random.default_rng(0).normal(sharpe * 0.001, 0.02, n_bars)
        results = pd.DataFrame({"pnl": pnl})
        return {"stats": {"sharpe (naive)": sharpe, "max drawdown": 0.05},
                "results": results, "trades": []}

    eval_a = {"family": "A", "n_units": 20, "pct_units_profitable": 80.0,
              "best_result": _fake_result(1.0), "bars_per_year": 2000}
    eval_b = {"family": "B", "n_units": 1, "pct_units_profitable": 100.0,
              "best_result": _fake_result(1.4), "bars_per_year": 2000}

    board = v.leaderboard([eval_a, eval_b], n_trials=50)
    assert set(board["family"]) == {"A", "B"}
    assert list(board.columns).count("dsr") == 1
    # ranking is by DSR then consistency -- just confirm it's sorted correctly by its own stated key
    assert board["dsr"].is_monotonic_decreasing


def test_bootstrap_significance_tolerates_leading_nan_warmup_bars():
    """
    Regression test: mean-reversion's walk-forward loop leaves pnl as NaN
    until it has enough trailing sessions to fit a hedge ratio at all
    (see _engine_meanrev.walk_forward_meanrev). A resample block drawn
    from that warm-up region must not turn the whole bootstrap's Sharpe
    into NaN.
    """
    rng = np.random.default_rng(6)
    pnl = np.concatenate([np.full(50, np.nan), rng.normal(0.02, 0.05, 450)])
    result = v.bootstrap_significance(pnl, bars_per_year=2000, block_length=10, n_boot=300,
                                       rng=np.random.default_rng(7))
    assert np.isfinite(result["sharpe_median"])
    assert np.isfinite(result["p_sharpe_gt_0"])


def test_leaderboard_skips_evaluations_with_no_result():
    ev = {"family": "empty", "n_units": 0, "pct_units_profitable": float("nan"),
          "best_result": None, "bars_per_year": 2000}
    board = v.leaderboard([ev], n_trials=5)
    assert len(board) == 0
