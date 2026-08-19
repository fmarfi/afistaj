"""
pytest suite for _ou_calibration.py. Synthetic simulation only, no network.

Run: python -m pytest bist30_strategy_lab/test_ou_calibration.py -v
(from the afistaj/ project root, with the venv active)
"""

import numpy as np
import pytest

import _ou_calibration as oc


def test_simulate_ou_paths_shape():
    rng = np.random.default_rng(0)
    paths = oc.simulate_ou_paths(kappa=0.05, sigma=1.0, n_bars=200, n_sims=50, rng=rng)
    assert paths.shape == (50, 200)
    assert np.all(paths[:, 0] == 0.0)


def test_simulate_ou_paths_variance_approaches_stationary_variance():
    rng = np.random.default_rng(1)
    kappa, sigma = 0.1, 1.0
    paths = oc.simulate_ou_paths(kappa, sigma, n_bars=2000, n_sims=2000, rng=rng)
    late_var = paths[:, -1].var()
    theoretical_var = sigma ** 2 / (2 * kappa)
    assert late_var == pytest.approx(theoretical_var, rel=0.25)


def test_simulate_trading_pnl_zero_cost_favors_moderate_over_very_wide_entry():
    """
    Not a claim that tighter is always better (see the interior-optimum
    test below -- the tightest grid entry loses to its own too-tight
    stop) -- just that a moderate entry clearly beats a very wide,
    rarely-triggered one at zero cost, since the wide one captures almost
    no reversions at all.
    """
    rng = np.random.default_rng(2)
    kappa, sigma = 0.05, 1.0
    stationary_std = sigma / np.sqrt(2 * kappa)
    paths = oc.simulate_ou_paths(kappa, sigma, n_bars=1000, n_sims=500, rng=rng)
    tight_pnl, _ = oc.simulate_trading_pnl(paths, entry_z=0.25, stop_z=1.0, stationary_std=stationary_std,
                                            cost_per_round_trip=0.0)
    wide_pnl, _ = oc.simulate_trading_pnl(paths, entry_z=2.5, stop_z=6.0, stationary_std=stationary_std,
                                           cost_per_round_trip=0.0)
    assert tight_pnl >= wide_pnl - 1e-9


def test_calibrate_entry_z_widens_as_cost_increases():
    rng_seed = 42
    kappa, sigma = 0.08, 1.0
    low_cost = oc.calibrate_entry_z(kappa, sigma, cost_per_round_trip=0.0001,
                                     rng=np.random.default_rng(rng_seed))
    high_cost = oc.calibrate_entry_z(kappa, sigma, cost_per_round_trip=0.05,
                                      rng=np.random.default_rng(rng_seed))
    assert high_cost["entry_z"] >= low_cost["entry_z"]


def test_calibrated_entry_z_has_interior_optimum_even_at_zero_cost():
    """
    Because stop_z scales proportionally with entry_z (same 1.5:3.5 ratio
    as the fixed-threshold default), the tightest grid entry doesn't win
    even at zero transaction cost -- its equally-tight stop triggers on
    ordinary noise before a position can revert. The calibrator should find
    a genuine interior optimum, not just walk to the grid's edge.
    """
    kappa, sigma = 0.08, 1.0
    result = oc.calibrate_entry_z(kappa, sigma, cost_per_round_trip=0.0,
                                   rng=np.random.default_rng(3))
    assert min(oc.DEFAULT_ENTRY_Z_GRID) < result["entry_z"] < max(oc.DEFAULT_ENTRY_Z_GRID)


def test_calibrate_entry_z_returns_none_for_non_mean_reverting_kappa():
    assert oc.calibrate_entry_z(kappa=0.0, sigma=1.0, cost_per_round_trip=0.001) is None
    assert oc.calibrate_entry_z(kappa=-0.1, sigma=1.0, cost_per_round_trip=0.001) is None


def test_calibrate_entry_z_returns_none_for_invalid_sigma():
    assert oc.calibrate_entry_z(kappa=0.05, sigma=0.0, cost_per_round_trip=0.001) is None


def test_calibrate_entry_z_stop_ratio_is_preserved():
    result = oc.calibrate_entry_z(kappa=0.05, sigma=1.0, cost_per_round_trip=0.01,
                                   stop_z_ratio=2.0, rng=np.random.default_rng(4))
    assert result["stop_z"] == pytest.approx(result["entry_z"] * 2.0)
